from __future__ import annotations

import json
from typing import Any

import pytest

from video_demo import outline, suggest


def _cand(i: int, start: float, tags: list[str], dur: float = 8.0, **kw: Any):
    return suggest.Candidate(
        video=kw.pop("video", "v"),
        index=i,
        shot=i,
        start=start,
        end=start + dur,
        tags={
            "description": kw.pop("description", f"d{i}"),
            "shot_size": "中景",
            "time_of_day": kw.pop("time_of_day", "白天"),
            "tags": tags,
            "highlight": 4,
            "issues": ["无明显问题"],
        },
        flags=[],
        sharpness=500.0,
        sharpness_rel=1.0,
        quality=kw.pop("quality", 0.5),
        tech_raw=0.0,
        aes_raw=0.0,
    )


def _forest_then_cooking() -> list[suggest.Candidate]:
    forest = [_cand(i, 10.0 * i, ["森林", "小路", "树木"]) for i in range(4)]
    cooking = [
        _cand(4 + i, 40.0 + 10.0 * i, ["野炊", "火", "锅"], quality=0.4 + 0.1 * i)
        for i in range(4)
    ]
    return forest + cooking


def _assess(cands: list[suggest.Candidate]) -> list[suggest.Assessment]:
    return [suggest.assess(c) for c in cands]


def _summary(highlights: list[int] | None = None, text: str = "s") -> dict:
    return {
        "summary": text,
        "places": ["森林"],
        "subjects": [],
        "activities": ["徒步"],
        "highlights": [{"segment": n, "why": "w"} for n in highlights or []],
    }


def _chunks(
    cands: list[suggest.Candidate],
    groups: list[list[int]],
    highlights: list[list[int]] | None = None,
) -> list[outline.Chunk]:
    return [
        outline.Chunk(
            number=k,
            videos=[cands[g[0]].video],
            members=g,
            start=cands[g[0]].start,
            end=cands[g[-1]].end,
            summary=_summary((highlights or [[]] * len(groups))[k - 1]),
        )
        for k, g in enumerate(groups, start=1)
    ]


def _plan(
    parts: list[tuple[int, int, int, list[str]]], seconds: list[int] | None = None
) -> dict[str, Any]:
    """Parts as (first chunk, last chunk, weight, scenes); ``seconds`` sets
    duration_sec directly, as validation would, for the fill tests."""
    plan = {
        "title": "t",
        "parts": [
            {
                "name": f"p{k}",
                "summary": "s",
                "first_chunk": a,
                "last_chunk": b,
                "weight": d,
                "scenes": sc,
            }
            for k, (a, b, d, sc) in enumerate(parts)
        ],
    }
    for part, sec in zip(plan["parts"], seconds or [], strict=False):
        part["duration_sec"] = sec
    return plan


def test_split_chunks_by_time_and_video() -> None:
    walk = [_cand(i, 10.0 * i, ["街道"]) for i in range(70)]
    groups = outline.split_chunks(walk)
    # 0-290 s, 300-590 s, 600-690 s.
    assert [len(g) for g in groups] == [30, 30, 10]
    other = [*walk[:3], _cand(9, 0.0, ["街道"], video="w")]
    assert outline.split_chunks(other) == [[0, 1, 2], [3]]
    assert outline.split_chunks(walk[:5], chunk_sec=20.0) == [[0, 1], [2, 3], [4]]


def test_segment_and_chunk_lines() -> None:
    cands = _forest_then_cooking()
    text = outline.segment_lines([4, 5], cands, _assess(cands))
    assert text.splitlines()[0].startswith("[1] 0:40 | 白天 |")
    assert text.splitlines()[1].endswith("d5 | 野炊、火、锅")
    lines = outline.chunk_lines(_chunks(cands, [[0, 1, 2, 3], [4, 5, 6, 7]]))
    assert lines.splitlines()[1] == "[2] 0:40–1:18 | s | 森林 | 徒步"


@pytest.mark.parametrize(
    ("summary", "message"),
    [
        ({"summary": " "}, "non-empty"),
        (_summary([0]), "outside"),
        (_summary([5]), "outside"),
        (_summary([2, 2]), "twice"),
    ],
)
def test_validate_summary_rejects(summary: dict, message: str) -> None:
    with pytest.raises(outline.OutlineError, match=message):
        outline.validate_summary(summary, 4)


def test_summarize_chunk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    cands = _forest_then_cooking()
    replies = iter([json.dumps(_summary([9])), json.dumps(_summary([2]))])
    prompts: list[str] = []

    def fake(prompt: str, *_: Any) -> tuple[str, float]:
        prompts.append(prompt)
        return next(replies), 1.0

    monkeypatch.setattr(outline, "_request", fake)
    summary, _, seconds, errors = outline.summarize_chunk(
        [4, 5, 6, 7], cands, _assess(cands)
    )
    assert summary["highlights"][0]["segment"] == 2 and seconds == 2.0
    assert "outside 1-4" in errors[0]
    assert "0:40–1:18" in prompts[0] and "[4]" in prompts[0]


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        ([(1, 1, 5, ["a"])], "need"),
        ([(1, 2, 5, ["a"]), (2, 2, 5, ["a"])], "overlap"),
        ([(2, 2, 5, ["a"]), (1, 1, 5, ["a"])], "overlap"),
        ([(1, 1, 5, ["a"]), (2, 3, 5, ["a"])], "outside"),
        ([(1, 1, 0, ["a"]), (2, 2, 5, ["a"])], "weight must"),
        ([(1, 1, 11, ["a"]), (2, 2, 5, ["a"])], "weight"),
        ([(1, 1, 5, []), (2, 2, 5, ["a"])], "needs 1-4 scenes"),
        ([(1, 1, 5, list("abcde")), (2, 2, 5, ["a"])], "needs 1-4 scenes"),
        ([(1, 1, 5, ["a", "a"]), (2, 2, 5, ["a"])], "listed twice"),
        ([(1, 1, 5, [" "]), (2, 2, 5, ["a"])], "empty scene"),
    ],
)
def test_validate_outline_rejects(
    parts: list[tuple[int, int, int, list[str]]], message: str
) -> None:
    with pytest.raises(outline.OutlineError, match=message):
        outline.validate_outline(_plan(parts), 2, 40.0)


def test_plan_outline_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cands = _forest_then_cooking()
    chunks = _chunks(cands, [[0, 1, 2, 3], [4, 5, 6, 7]])
    good = json.dumps(_plan([(1, 1, 1, ["森林"]), (2, 2, 3, ["野炊"])]))
    replies = iter(["not json", good])
    monkeypatch.setattr(outline, "_request", lambda *_: (next(replies), 1.0))
    plan, raw, seconds, errors = outline.plan_outline(chunks, 40.0)
    assert plan["parts"][1]["scenes"] == ["野炊"] and seconds == 2.0
    # 40 s: 10 s each, the other 20 s split 1:3.
    assert [p["duration_sec"] for p in plan["parts"]] == [15, 25]
    assert len(errors) == 1 and "not JSON" in errors[0]

    monkeypatch.setattr(outline, "_request", lambda *_: ("{}", 1.0))
    with pytest.raises(outline.OutlineError, match="temperature=0.8"):
        outline.plan_outline(chunks, 40.0)


def test_part_limits_follow_the_target() -> None:
    assert outline.part_limits(600.0) == (10.0, 8)
    assert outline.part_limits(30.0) == (10.0, 3)
    # A 17 s video at 1/6 still allows two parts of half the target.
    assert outline.part_limits(2.8) == (1.4, 2)
    parts = outline.outline_schema(1, 3)["properties"]["parts"]
    assert (parts["minItems"], parts["maxItems"]) == (1, 3)
    # No more parts than chunks; one chunk makes a one-part outline.
    assert outline.parts_range(600.0, 40) == (2, 8)
    assert outline.parts_range(600.0, 5) == (2, 5)
    assert outline.parts_range(2.8, 1) == (1, 1)


def test_allocate_meets_target_and_minimum() -> None:
    assert outline.allocate([1, 1, 1], 30.0) == [10, 10, 10]
    secs = outline.allocate([1, 5, 10, 2], 600.0)
    assert sum(secs) == 600 and min(secs) >= 10 and secs[2] > secs[1] > secs[3]
    assert sum(outline.allocate([3, 7], 2.8)) == 3
    with pytest.raises(ValueError, match="do not fit"):
        outline.allocate([1, 1, 1, 1], 30.0)


def _distinct(n: int, **kw: Any) -> list[suggest.Candidate]:
    """Segments 10 s apart whose tags never repeat."""
    return [_cand(i, 10.0 * i, [f"t{i}", f"u{i}"], **kw) for i in range(n)]


def _scenes(groups: list[list[int]]) -> list[outline.Scene]:
    return [outline.Scene(n, g) for n, g in enumerate(groups, start=1)]


def _flat(members: range, values: dict[int, float] | None = None) -> dict:
    return {i: (values or {}).get(i, 0.0) for i in members}


def test_part_scenes_and_share() -> None:
    cands = _distinct(4)
    chunks = _chunks(cands, [[0, 1], [2, 3]])
    scenes = _scenes([[0], [1], [2, 3]])
    part = _plan([(2, 2, 5, ["a"])])["parts"][0]
    assert [s.number for s in outline.part_scenes(part, chunks, scenes)] == [3]
    assert outline.share(30.0, [100.0, 100.0]) == [15.0, 15.0]
    assert outline.share(30.0, [8.0, 100.0]) == [8.0, 22.0]
    assert outline.share(30.0, [5.0, 5.0]) == [5.0, 5.0]
    # Weighted: 45:17:13 of 75 s, the first part holds only 6 s; the other
    # two take the rest in their own proportion.
    out = outline.share(75.0, [6.0, 100.0, 100.0], [45.0, 17.0, 13.0])
    assert out == pytest.approx([6.0, 69.0 * 17 / 30, 69.0 * 13 / 30])
    assert outline.share(30.0, [100.0, 100.0], [2.0, 1.0]) == [20.0, 10.0]
    # 15 clips of 6 s for a 75 s cut: each is cut to 5 s.
    assert outline.level([6.0] * 15, 75.0, 1.5) == pytest.approx(5.0)
    # Short clips stay whole; the long ones share what is left.
    assert outline.level([2.0, 10.0, 10.0], 12.0, 1.5) == pytest.approx(5.0)
    assert outline.level([6.0] * 10, 5.0, 1.5) == 1.5
    assert outline.rescale({1: 0.9, 2: 1.0, 3: 0.95}) == pytest.approx(
        {1: 0.0, 2: 1.0, 3: 0.5}
    )
    assert outline.rescale({1: 0.7, 2: 0.7}) == {1: 1.0, 2: 1.0}


def test_fill_lets_relevance_lead_over_quality() -> None:
    # RRF values close together (rank 1 vs rank 3) still decide over a
    # better-looking scene.
    cands = [
        *_distinct(2, quality=0.2),
        *[_cand(i, 10.0 * i, [f"t{i}"], quality=0.9) for i in (2, 3)],
        *[_cand(i, 10.0 * i, [f"t{i}"], quality=0.5) for i in (4, 5)],
    ]
    scenes = _scenes([[0, 1], [2, 3], [4, 5]])
    clips, _, slots = outline.fill(
        _plan([(1, 1, 5, ["a"])], seconds=[8]),
        _chunks(cands, [list(range(6))]),
        scenes,
        cands,
        _assess(cands),
        [[{1: 1.0, 2: 0.968, 3: 0.9}]],
        [[_flat(range(6))]],
    )
    assert slots[0][0]["scene"] == 1 and [c.index for c in clips] == [0]


def test_fill_gives_each_slot_its_own_scene() -> None:
    cands = _distinct(8)
    assess = _assess(cands)
    chunks = _chunks(cands, [[0, 1, 2, 3], [4, 5, 6, 7]])
    scenes = _scenes([[0, 1], [2, 3], [4, 5], [6, 7]])
    plan = _plan([(1, 1, 5, ["城堡", "街道"]), (2, 2, 5, ["教堂"])], seconds=[16, 8])
    # Both slots like scene 2 best; the better match takes it.
    scene_rel = [[{1: 0.5, 2: 1.0}, {1: 0.9, 2: 0.95}], [{3: 0.2, 4: 1.0}]]
    seg_rel = [[_flat(range(4))] * 2, [_flat(range(4, 8))]]
    clips, notes, slots = outline.fill(
        plan, chunks, scenes, cands, assess, scene_rel, seg_rel
    )
    assert [(c.role, c.index) for c in clips] == [("p0", 0), ("p0", 2), ("p1", 6)]
    assert [c.order for c in clips] == [0, 1, 2]
    assert "场景「城堡」→ 场景 2（相关度 100%）" in clips[1].reason
    assert [(s["scene"], s["seconds"]) for s in slots[0]] == [(2, 8.0), (1, 8.0)]
    assert slots[0][0]["top"] == [2, 1]
    assert notes == []


def test_fill_moves_seconds_a_small_scene_cannot_hold() -> None:
    cands = _distinct(4)
    assess = _assess(cands)
    chunks = _chunks(cands, [[0, 1, 2, 3]])
    scenes = _scenes([[0], [1, 2, 3]])
    plan = _plan([(1, 1, 5, ["a", "b"])], seconds=[32])
    scene_rel = [[{1: 1.0, 2: 0.0}, {1: 0.0, 2: 1.0}]]
    seg_rel = [[_flat(range(4))] * 2]
    clips, notes, slots = outline.fill(
        plan, chunks, scenes, cands, assess, scene_rel, seg_rel
    )
    assert [s["seconds"] for s in slots[0]] == [8.0, 24.0]
    assert [c.index for c in clips] == [0, 1, 2, 3] and notes == []


def test_fill_ranks_segments_in_the_scene() -> None:
    cands = _distinct(4)
    cands[3] = _cand(3, 30.0, ["t3"], quality=0.45)
    assess = _assess(cands)
    scenes = _scenes([[0, 1, 2, 3]])
    plan = _plan([(1, 1, 5, ["a"])], seconds=[8])
    scene_rel = [[{1: 1.0}]]
    plain = _chunks(cands, [[0, 1, 2, 3]])
    clips, _, _ = outline.fill(
        plan, plain, scenes, cands, assess, scene_rel, [[_flat(range(4))]]
    )
    assert [c.index for c in clips] == [0]
    # Relevance to the slot picks a segment among equal scores.
    rel = [[_flat(range(4), {2: 1.0})]]
    clips, _, _ = outline.fill(plan, plain, scenes, cands, assess, scene_rel, rel)
    assert [c.index for c in clips] == [2] and "片段相关度 100%" in clips[0].reason
    # The summary's highlight bonus outweighs a slightly lower score.
    named = _chunks(cands, [[0, 1, 2, 3]], highlights=[[4]])
    clips, _, _ = outline.fill(
        plan, named, scenes, cands, assess, scene_rel, [[_flat(range(4))]]
    )
    assert [c.index for c in clips] == [3]
    assert "分段总结点名的亮点" in clips[0].reason


def test_fill_reports_missing_scenes_and_short_parts() -> None:
    cands = [*_distinct(4), _cand(4, 40.0, ["x"], dur=1.0)]
    assess = _assess(cands)
    chunks = _chunks(cands, [[0, 1, 2, 3, 4]])
    # Scene 2 has only a segment too short to use.
    scenes = _scenes([[0, 1, 2, 3], [4]])
    plan = _plan([(1, 1, 5, ["a", "b"])], seconds=[300])
    scene_rel = [[{1: 1.0, 2: 1.0}, {1: 0.5, 2: 1.0}]]
    seg_rel = [[_flat(range(5))] * 2]
    clips, notes, slots = outline.fill(
        plan, chunks, scenes, cands, assess, scene_rel, seg_rel
    )
    assert [s["scene"] for s in slots[0]] == [1, None]
    assert "「p0」场景「b」没有可用的场景" in notes[0]
    assert "「p0」素材不足" in notes[1]
    assert [c.index for c in clips] == [0, 1, 2, 3]


# --- groups of short videos -----------------------------------------------------


@pytest.mark.parametrize(
    ("lengths", "groups"),
    [
        # Short videos fill a group until it holds CHUNK_SEC.
        ([181.0, 153.0, 128.0, 17.0, 31.0, 300.0], [[0, 1], [2, 3, 4], [5]]),
        # A long video closes the open group and stands alone.
        ([100.0, 4020.0, 50.0], [[0], [1], [2]]),
        ([300.0], [[0]]),
        ([10.0] * 3, [[0, 1, 2]]),
        ([], []),
    ],
)
def test_group_videos_never_splits_a_video(
    lengths: list[float], groups: list[list[int]]
) -> None:
    assert outline.group_videos(lengths) == groups


def test_group_highlights_renumber_along_the_group() -> None:
    videos = [(3, _summary([2])), (2, _summary([1, 2]))]
    assert [h["segment"] for h in outline.group_highlights(videos)] == [2, 4, 5]


def test_summarize_group_reads_the_video_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, dict]] = []
    reply = {
        "summary": "先逛寺庙再看湖",
        "places": [],
        "subjects": [],
        "activities": [],
    }

    def fake(prompt: str, _t: float, schema: dict, *_: Any) -> tuple[str, float]:
        seen.append((prompt, schema))
        return json.dumps(reply), 1.0

    monkeypatch.setattr(outline, "_request", fake)
    summary, _, _, _ = outline.summarize_group(
        [(181.0, _summary(text="寺庙")), (153.0, _summary(text="湖"))]
    )
    assert summary == reply
    prompt, schema = seen[0]
    assert "连续拍摄的 2 段视频（共 5.6 分钟）" in prompt
    assert "[1] 3:01 | 寺庙 | 森林 | 徒步\n[2] 2:33 | 湖" in prompt
    assert "highlights" not in schema["properties"]
    with pytest.raises(outline.OutlineError, match="non-empty"):
        outline.validate_group({"summary": " "})


def test_clip_span_keeps_the_middle() -> None:
    c = _cand(0, 10.0, ["a"], dur=8.0)
    assert outline.clip_span(c, None) == (10.0, 18.0)
    assert outline.clip_span(c, 3.0) == (12.5, 15.5)
    assert outline.clip_span(c, 10.0) == (10.0, 18.0)


def test_clip_span_avoids_cutting_sentences() -> None:
    c = _cand(0, 10.0, ["a"], dur=8.0)
    # The middle window 12.5-15.5 would cut the sentence 11-14 in half.
    assert outline.clip_span(c, 3.0, [(11.0, 14.0)]) == (11.0, 14.0)
    # A sentence longer than the window: cut it where least is lost.
    assert outline.clip_span(c, 3.0, [(10.5, 17.5)]) == (10.0, 13.0)
    # Speech outside the segment changes nothing.
    assert outline.clip_span(c, 3.0, [(0.0, 9.0)]) == (12.5, 15.5)


def test_pick_segments_counts_trimmed_lengths() -> None:
    cands = [_cand(i, 10.0 * i, [f"t{i}"]) for i in range(4)]
    rel = {i: 1.0 for i in range(4)}
    # 12 s from 8 s segments: two whole ones, or four 3 s middles.
    assert len(outline.pick_segments(rel, 12.0, cands, _assess(cands), set())) == 2
    picked = outline.pick_segments(rel, 12.0, cands, _assess(cands), set(), 3.0)
    assert len(picked) == 4
