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


def _plan(
    parts: list[tuple[int, int, int, list[str]]], seconds: list[int] | None = None
) -> dict[str, Any]:
    """Parts as (first scene, last scene, weight, keywords); ``seconds`` sets
    duration_sec directly, as validation would, for the fill tests."""
    plan = {
        "title": "t",
        "parts": [
            {
                "name": f"p{k}",
                "summary": "s",
                "first_scene": a,
                "last_scene": b,
                "weight": d,
                "keywords": kw,
            }
            for k, (a, b, d, kw) in enumerate(parts)
        ],
    }
    for part, sec in zip(plan["parts"], seconds or [], strict=False):
        part["duration_sec"] = sec
    return plan


def test_scenes_split_on_tag_change_video_and_length() -> None:
    cands = _forest_then_cooking()
    scenes = outline.group_scenes(cands, _assess(cands))
    assert [s.members for s in scenes] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert scenes[1].start == 40.0 and scenes[1].end == 78.0
    # Another video always starts a new scene.
    other = [*cands[:2], _cand(9, 0.0, ["森林", "小路", "树木"], video="w")]
    assert len(outline.group_scenes(other, _assess(other))) == 2
    # A long take with constant tags is cut every SCENE_MAX_SEC.
    walk = [_cand(i, 10.0 * i, ["街道"]) for i in range(20)]
    lengths = [s.end - s.start for s in outline.group_scenes(walk, _assess(walk))]
    assert max(lengths) <= outline.SCENE_MAX_SEC


def test_scene_lines_show_time_score_and_tags() -> None:
    cands = _forest_then_cooking()
    text = outline.scene_lines(outline.group_scenes(cands, _assess(cands)))
    assert text.splitlines()[1].startswith("[2] 0:40–1:18 | 白天 |")
    assert "野炊、火、锅" in text


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        ([(1, 1, 5, ["a"])], "need"),
        ([(1, 2, 5, ["a"]), (2, 2, 5, ["a"])], "overlap"),
        ([(1, 1, 5, ["a"]), (2, 3, 5, ["a"])], "outside"),
        ([(1, 1, 0, ["a"]), (2, 2, 5, ["a"])], "weight must"),
        ([(1, 1, 11, ["a"]), (2, 2, 5, ["a"])], "weight"),
        ([(1, 1, 5, []), (2, 2, 5, ["a"])], "keywords"),
    ],
)
def test_validate_outline_rejects(
    parts: list[tuple[int, int, int, list[str]]], message: str
) -> None:
    cands = _forest_then_cooking()
    scenes = outline.group_scenes(cands, _assess(cands))
    with pytest.raises(outline.OutlineError, match=message):
        outline.validate_outline(_plan(parts), scenes, 40.0)


def test_plan_outline_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cands = _forest_then_cooking()
    scenes = outline.group_scenes(cands, _assess(cands))
    good = json.dumps(_plan([(1, 1, 1, ["森林"]), (2, 2, 3, ["野炊"])]))
    replies = iter(["not json", good])
    monkeypatch.setattr(outline, "_request", lambda *_: (next(replies), 1.0))
    plan, raw, seconds, errors = outline.plan_outline(scenes, 40.0)
    assert plan["parts"][1]["keywords"] == ["野炊"] and seconds == 2.0
    # 40 s: 10 s each, the other 20 s split 1:3.
    assert [p["duration_sec"] for p in plan["parts"]] == [15, 25]
    assert len(errors) == 1 and "not JSON" in errors[0]

    monkeypatch.setattr(outline, "_request", lambda *_: ("{}", 1.0))
    with pytest.raises(outline.OutlineError, match="temperature=0.8"):
        outline.plan_outline(scenes, 40.0)


def test_part_limits_follow_the_target() -> None:
    assert outline.part_limits(600.0) == (10.0, 8)
    assert outline.part_limits(30.0) == (10.0, 3)
    # A 17 s video at 1/6 still allows two parts of half the target.
    assert outline.part_limits(2.8) == (1.4, 2)
    parts = outline.outline_schema(1, 3)["properties"]["parts"]
    assert (parts["minItems"], parts["maxItems"]) == (1, 3)
    # No more parts than scenes; one scene makes a one-part outline.
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


def test_fill_stays_in_each_part_and_prefers_keyword_matches() -> None:
    # Forest segments share two of four tags: one scene, but not repeats.
    forest = [_cand(i, 10.0 * i, ["森林", "小路", f"a{i}", f"b{i}"]) for i in range(4)]
    cooking = [
        _cand(4, 40.0, ["野炊", "火", "锅"], quality=0.3, description="点火"),
        *[_cand(i, 10.0 * i, ["野炊", "火", "锅"], quality=0.5) for i in (5, 6, 7)],
    ]
    cands = forest + cooking
    assess = _assess(cands)
    scenes = outline.group_scenes(cands, assess)
    assert [s.members for s in scenes] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    plan = _plan([(1, 1, 5, ["森林"]), (2, 2, 5, ["点火"])], seconds=[16, 8])
    clips, notes = outline.fill(plan, scenes, cands, assess)
    # The lower-scoring segment wins its part on the keyword.
    assert [(c.role, c.index) for c in clips] == [("p0", 0), ("p0", 1), ("p1", 4)]
    assert all(c.duration == pytest.approx(8.0) for c in clips)
    assert notes == []


def test_fill_reports_a_short_part() -> None:
    cands = _forest_then_cooking()
    assess = _assess(cands)
    scenes = outline.group_scenes(cands, assess)
    clips, notes = outline.fill(
        _plan([(1, 1, 5, ["森林"]), (2, 2, 5, ["野炊"])], seconds=[10, 300]),
        scenes,
        cands,
        assess,
    )
    assert "「p1」素材不足" in notes[0]
    assert [c.order for c in clips] == list(range(len(clips)))
