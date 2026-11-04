from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from video_demo import cli, instruct, layout, outline, report, suggest


def _cand(i: int, start: float, dur: float = 8.0, **kw: Any) -> suggest.Candidate:
    return suggest.Candidate(
        video="v",
        index=i,
        shot=i,
        start=start,
        end=start + dur,
        tags={
            "description": f"d{i}",
            "shot_size": "中景",
            "time_of_day": kw.pop("time_of_day", "白天"),
            "tags": [f"t{i}", f"u{i}"],
            "highlight": 4,
            "issues": ["无明显问题"],
        },
        highlight=4.0,
        flags=kw.pop("flags", []),
        sharpness=500.0,
        sharpness_rel=1.0,
        quality=kw.pop("quality", 0.5),
        tech_raw=0.0,
        aes_raw=0.0,
    )


def _assess(cands: list[suggest.Candidate]) -> list[suggest.Assessment]:
    return [suggest.assess(c) for c in cands]


def _scenes(groups: list[list[int]]) -> list[outline.Scene]:
    return [outline.Scene(n, g) for n, g in enumerate(groups, start=1)]


def _slot(query: str, sources: list[int] | None = None) -> dict[str, Any]:
    return {"query": query, "sources": sources or [1]}


def _plan(parts: list[tuple[int, list[dict]]], seconds: list[int] | None = None):
    """Parts as (weight, slots); ``seconds`` sets duration_sec directly."""
    plan: dict[str, Any] = {
        "title": "t",
        "materials": [{"scene": 1, "content": "教堂正面"}],
        "design": "先外观后内部",
        "pace": "整段",
        "unmet": [],
        "parts": [
            {"name": f"p{k}", "summary": "s", "weight": w, "scenes": slots}
            for k, (w, slots) in enumerate(parts)
        ],
    }
    for part, sec in zip(plan["parts"], seconds or [], strict=False):
        part["duration_sec"] = sec
    return plan


def _flat(n: int) -> dict[int, float]:
    return {i: 0.0 for i in range(n)}


# --- duration ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parsed", "expected"),
    [
        ({"target_sec": 60, "quote": "60 秒"}, {"target_sec": 60, "quote": "60 秒"}),
        ({"target_sec": 90, "quote": "一分半"}, {"target_sec": 90, "quote": "一分半"}),
        ({"target_sec": 0, "quote": " "}, {"target_sec": 0, "quote": ""}),
    ],
)
def test_validate_duration_accepts(parsed: dict, expected: dict) -> None:
    text = "剪 60 秒的教堂，或者一分半也行"
    assert instruct.validate_duration(parsed, text) == expected


@pytest.mark.parametrize(
    ("parsed", "message"),
    [
        ({"target_sec": 60, "quote": ""}, "without the words"),
        ({"target_sec": 60, "quote": "1 分钟"}, "not in the instruction"),
        # "3 段" is a count of clips, not a length.
        ({"target_sec": 3, "quote": "3 段"}, "not a length"),
        ({"target_sec": 0, "quote": "3 段"}, "no length but quote"),
        ({"target_sec": -1, "quote": ""}, "target_sec must be"),
        ({"target_sec": "60", "quote": "60 秒"}, "target_sec must be"),
    ],
)
def test_validate_duration_rejects(parsed: dict, message: str) -> None:
    with pytest.raises(outline.OutlineError, match=message):
        instruct.validate_duration(parsed, "剪 3 段夜景，60 秒")


def test_parse_duration_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(
        [json.dumps({"target_sec": 30, "quote": "30 秒"}), '{"target_sec": 0}']
    )
    monkeypatch.setattr(outline, "_request", lambda *_: (next(replies), 1.0))
    parsed, _, seconds, errors = instruct.parse_duration("只要夜景")
    assert parsed == {"target_sec": 0, "quote": ""} and seconds == 2.0
    assert "not in the instruction" in errors[0]
    monkeypatch.setattr(
        outline,
        "_request",
        lambda *_: (json.dumps({"target_sec": 30, "quote": "30 秒"}), 1.0),
    )
    with pytest.raises(outline.OutlineError, match="temperature=0.8"):
        instruct.parse_duration("只要夜景")


# --- outline -------------------------------------------------------------------


NUMBERS = {1, 2}


def test_validate_request_allocates_seconds() -> None:
    plan = instruct.validate_request(_plan([(5, [_slot("教堂")])]), 15.0, NUMBERS)
    assert plan["parts"][0]["duration_sec"] == 15
    # part_limits(30) allows three parts of 10 s.
    plan = _plan([(1, [_slot("a")]), (1, [_slot("b")]), (2, [_slot("c")])])
    instruct.validate_request(plan, 30.0, NUMBERS)
    assert sum(p["duration_sec"] for p in plan["parts"]) == 30


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (_plan([(1, [_slot(x)]) for x in "abcd"]), "4 parts, need 1-3"),
        (_plan([]), "0 parts"),
        (_plan([(5, [_slot("a"), _slot("a")])]), "listed twice"),
        (_plan([(5, [_slot("a")]), (5, [_slot("a")])]), "listed twice"),
        (_plan([(5, [_slot(" ")])]), "empty scene"),
        (_plan([(0, [_slot("a")])]), "weight"),
        (_plan([(5, [])]), "needs 1-4 scenes"),
        ({**_plan([(5, [_slot("a")])]), "design": " "}, "design"),
        ({**_plan([(5, [_slot("a")])]), "materials": []}, "materials"),
        (
            {**_plan([(5, [_slot("a")])]), "materials": [{"scene": 3, "content": "c"}]},
            "not in the catalog",
        ),
        (
            {**_plan([(5, [_slot("a")])]), "materials": [{"scene": 1, "content": ""}]},
            "empty content",
        ),
        # A slot's sources come from the materials it was designed from.
        (_plan([(5, [_slot("a", [2])])]), "not in materials"),
        (_plan([(5, [{**_slot("a"), "sources": []}])]), "needs 1-3 sources"),
        ({**_plan([(5, [_slot("a")])]), "unmet": [""]}, "unmet"),
        ({**_plan([(5, [_slot("a")])]), "pace": "慢"}, "pace"),
    ],
)
def test_validate_request_rejects(plan: dict, message: str) -> None:
    with pytest.raises(outline.OutlineError, match=message):
        instruct.validate_request(plan, 30.0, NUMBERS)


def test_plan_request_prompt_and_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    # Scene 2 kept no footage, so it is not in the catalog.
    catalog = [
        {"number": 1, "start": 0.0, "end": 38.0, "videos": [1], "title": "教堂"},
        {"number": 3, "start": 50.0, "end": 70.0, "videos": [1], "title": "街道"},
    ]
    catalog[0]["summary"], catalog[1]["summary"] = "正面", "车"
    seen: list[tuple[str, dict]] = []

    def fake(prompt: str, _t: float, schema: dict, *_: Any) -> tuple[str, float]:
        seen.append((prompt, schema))
        return json.dumps(_plan([(5, [_slot("教堂外观")])])), 1.0

    monkeypatch.setattr(outline, "_request", fake)
    plan, _, _, errors = instruct.plan_request("剪 30 秒教堂", catalog, 70.0, 30.0)
    assert plan["parts"][0]["duration_sec"] == 30 and errors == []
    prompt, schema = seen[0]
    assert "「剪 30 秒教堂」" in prompt and "切成的 2 个场景" in prompt
    # One video: no video numbers in the lines.
    assert "[1] 0:00–0:38 | 教堂 | 正面\n[3] 0:50–1:10 | 街道 | 车" in prompt
    assert "时段" not in prompt
    assert schema["properties"]["parts"]["maxItems"] == 3
    # Fields are generated in this order: material and design come first.
    order = ["title", "materials", "design", "pace", "unmet", "parts"]
    assert list(schema["properties"]) == order
    assert schema["properties"]["pace"]["enum"] == ["整段", "适中", "快"]
    # The shared schema is not changed by the per-call limit.
    assert instruct.REQUEST_SCHEMA["properties"]["parts"]["maxItems"] == 8


def test_catalog_lines_name_the_videos() -> None:
    catalog = [
        {"number": 1, "start": 0.0, "end": 20.0, "videos": [1], "title": "a"},
        {"number": 2, "start": 20.0, "end": 45.0, "videos": [1, 2], "title": "b"},
    ]
    for s in catalog:
        s["summary"] = "s"
    assert instruct.catalog_lines(catalog) == (
        "[1] 视频 1 0:00–0:20 | a | s\n[2] 视频 1–2 0:20–0:45 | b | s"
    )


def test_shortlist_keeps_the_most_relevant_in_timeline_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(instruct, "MAX_CATALOG_SCENES", 2)
    catalog = [{"number": n} for n in (1, 2, 3, 5)]
    relevance = {1: 0.2, 2: 0.9, 3: 0.1, 5: 1.0}
    assert [s["number"] for s in instruct.shortlist(catalog, relevance)] == [2, 5]
    assert instruct.shortlist(catalog[:2], relevance) == catalog[:2]


def test_scene_catalog_per_video_and_across(monkeypatch: pytest.MonkeyPatch) -> None:
    cands = [_cand(i, 10.0 * i) for i in range(3)]
    cands += [replace(_cand(i, 10.0 * i), video="w") for i in range(2)]
    # Scene 2 runs from the end of video v into video w.
    scenes = _scenes([[0, 1], [2, 3], [4]])
    seen: list[tuple] = []

    def fake(vid: str, groups: list[list[int]], cal: list, assessments: list):
        seen.append((vid, groups, len(cal)))
        return {n: {"title": f"{vid}{n}", "summary": "s"} for n in range(1, 2)}

    def fake_multi(path: Path, label: str, groups: list, cal: list, a: list, key):
        seen.append((label, groups, key(cal[groups[0][-1]])))
        return {1: {"title": "vw", "summary": "s"}}

    monkeypatch.setattr(cli, "_scene_summaries", fake)
    monkeypatch.setattr(cli, "_cached_scene_summaries", fake_multi)
    catalog = cli._scene_catalog(scenes, cands, {"v": 0.0, "w": 100.0})
    # Each video is summarised on its own, with indices into its own
    # segments; the scene across videos is cached by video and index.
    assert seen == [
        ("v", [[0, 1]], 3),
        ("w", [[1]], 2),
        ("跨视频", [[2, 3]], "w:0"),
    ]
    assert [(s["number"], s["start"], s["end"], s["videos"]) for s in catalog] == [
        (1, 0.0, 18.0, [1]),
        (2, 20.0, 108.0, [1, 2]),
        (3, 110.0, 118.0, [2]),
    ]
    assert [s["title"] for s in catalog] == ["v1", "vw", "w1"]


# --- fill ----------------------------------------------------------------------


def test_fill_request_follows_the_outline_not_shooting_time() -> None:
    cands = [_cand(i, 10.0 * i) for i in range(4)]
    scenes = _scenes([[0, 1], [2, 3]])
    # Part 1 asks for what was shot last, part 2 for what was shot first.
    plan = _plan([(5, [_slot("内部")]), (5, [_slot("外观")])], seconds=[8, 8])
    scene_rel = [[{1: 0.2, 2: 1.0}], [{1: 1.0, 2: 0.9}]]
    seg_rel = [[_flat(4)], [_flat(4)]]
    clips, notes, slots = instruct.fill_request(
        plan, scenes, cands, _assess(cands), scene_rel, seg_rel, set()
    )
    assert [(c.role, c.index) for c in clips] == [("p0", 2), ("p1", 0)]
    assert [c.order for c in clips] == [0, 1] and notes == []
    assert [s[0]["scene"] for s in slots] == [2, 1]
    assert "场景「内部」→ 场景 2（相关度 100%）" in clips[0].reason


def test_fill_request_fast_pace_takes_the_middle_of_each_segment() -> None:
    cands = [_cand(i, 10.0 * i) for i in range(4)]
    plan = {**_plan([(5, [_slot("a")])], seconds=[12]), "pace": "快"}
    clips, notes, slots = instruct.fill_request(
        plan,
        _scenes([[0, 1, 2, 3]]),
        cands,
        _assess(cands),
        [[{1: 1.0}]],
        [[_flat(4)]],
        set(),
    )
    # 12 s as four 3 s middles of the 8 s segments, in shooting order.
    assert [(c.src_start, c.src_end) for c in clips] == [
        (2.5, 5.5),
        (12.5, 15.5),
        (22.5, 25.5),
        (32.5, 35.5),
    ]
    assert notes == [] and "处的 3.0s" in clips[0].reason


def test_fill_request_lets_relevance_lead_inside_the_scene() -> None:
    # Segment 0 looks best but fits the slot least (a facade for an
    # interior); segment 1 is darker and fits best.
    cands = [_cand(0, 0.0, quality=1.0), _cand(1, 10.0, quality=0.2)]
    assess = _assess(cands)
    assert assess[0].score > assess[1].score
    seg_rel = {0: 0.0, 1: 1.0}
    plan = _plan([(5, [_slot("内部")])], seconds=[8])
    clips, _, _ = instruct.fill_request(
        plan, _scenes([[0, 1]]), cands, assess, [[{1: 1.0}]], [[seg_rel]], set()
    )
    assert [c.index for c in clips] == [1]
    # The highlights' share keeps quality ahead.
    assert outline.pick_segments(seg_rel, 8.0, cands, assess, set()) == [0]


def test_fill_request_uses_a_scene_once() -> None:
    cands = [_cand(i, 10.0 * i) for i in range(4)]
    scenes = _scenes([[0, 1], [2, 3]])
    plan = _plan([(5, [_slot("a")]), (5, [_slot("b")])], seconds=[8, 8])
    # Both slots like scene 1 best; the better match takes it.
    scene_rel = [[{1: 1.0, 2: 0.0}], [{1: 0.9, 2: 0.95}]]
    clips, _, slots = instruct.fill_request(
        plan, scenes, cands, _assess(cands), scene_rel, [[_flat(4)]] * 2, set()
    )
    assert [s[0]["scene"] for s in slots] == [1, 2]
    assert len({(c.video, c.index) for c in clips}) == len(clips) == 2


def test_fill_request_notes_taken_scenes_and_skips_rejects() -> None:
    cands = [
        _cand(0, 0.0),
        _cand(1, 10.0, flags=["blurry", "shaky"], quality=0.0),
        _cand(2, 20.0, flags=["blurry", "shaky"], quality=0.0),
    ]
    assess = _assess(cands)
    assert [a.keep for a in assess] == [True, False, False]
    # Scene 2 has only rejected segments, so one scene serves two slots.
    scenes = _scenes([[0, 1], [2]])
    plan = _plan([(5, [_slot("a"), _slot("b")])], seconds=[30])
    scene_rel = [[{1: 1.0, 2: 1.0}, {1: 0.5, 2: 1.0}]]
    clips, notes, slots = instruct.fill_request(
        plan, scenes, cands, assess, scene_rel, [[_flat(3)] * 2], set()
    )
    assert [c.index for c in clips] == [0]
    assert [s["scene"] for s in slots[0]] == [1, None]
    assert "「p0」场景「b」符合的场景都已分给其他槽位" in notes
    assert "可用素材只够 8.0s，目标 30s" in notes


def test_fill_request_fills_a_target_equal_to_all_usable_footage() -> None:
    cands = [_cand(i, 10.0 * i) for i in range(4)]
    plan = instruct.validate_request(
        _plan([(5, [_slot("a"), _slot("b")])]), 32.0, NUMBERS
    )
    scene_rel = [[{1: 1.0, 2: 0.0}, {1: 0.0, 2: 1.0}]]
    clips, notes, _ = instruct.fill_request(
        plan,
        _scenes([[0, 1], [2, 3]]),
        cands,
        _assess(cands),
        scene_rel,
        [[_flat(4)] * 2],
        set(),
    )
    assert sum(c.duration for c in clips) == 32.0 and notes == []


def test_fill_request_passes_a_short_part_on_to_the_others() -> None:
    # Part p0 plans 24 s but its scene holds one 8 s segment; p1 takes the
    # other 16 s from its larger scene, so the cut still reaches 32 s.
    cands = [_cand(i, 10.0 * i) for i in range(5)]
    plan = _plan([(3, [_slot("a")]), (1, [_slot("b")])], seconds=[24, 8])
    clips, notes, _ = instruct.fill_request(
        plan,
        _scenes([[0], [1, 2, 3, 4]]),
        cands,
        _assess(cands),
        [[{1: 1.0, 2: 0.0}], [{1: 0.0, 2: 1.0}]],
        [[_flat(5)], [_flat(5)]],
        set(),
    )
    assert sum(c.duration for c in clips) == 32.0
    assert [c.index for c in clips][0] == 0
    assert notes == ["「p0」素材只够 8.0s（计划 24s），其余分给其他部分"]


# --- report --------------------------------------------------------------------


def test_write_edit_shows_instruction_slots_and_misses(tmp_path: Path) -> None:
    work = tmp_path / "data" / "work"
    layout.signals_path(work, "v").parent.mkdir(parents=True)
    layout.signals_path(work, "v").write_text(
        json.dumps(
            [{"index": i, "keyframes": [f"seg{i:03d}_0.jpg"]} for i in range(2)]
        ),
        encoding="utf-8",
    )
    layout.tags_path(work, "v").parent.mkdir(parents=True)
    layout.tags_path(work, "v").write_text(
        "\n".join(
            json.dumps({"index": i, "tags": {"description": f"教堂{i}"}})
            for i in range(2)
        ),
        encoding="utf-8",
    )
    plan = _plan([(5, [_slot("教堂"), _slot("灯光", [2])])], seconds=[20])
    plan["unmet"] = ["烟花"]
    timeline = {
        "instruction": "剪 20 秒<教堂>",
        "created": "20261028-120000",
        "videos": ["v"],
        "target_sec": 20.0,
        "target_source": {"kind": "instruction", "quote": "20 秒"},
        "total_sec": 8.0,
        "notes": ["「p0」场景「灯光」符合的场景都已分给其他槽位"],
        "clips": [
            {
                "order": 0,
                "role": "p0",
                "video": "v",
                "index": 1,
                "src_start": 10.0,
                "src_end": 18.0,
                "duration": 8.0,
                "description": "教堂1",
                "reason": "r",
            }
        ],
        "assessments": [{"video": "v", "index": i, "keep": True} for i in range(2)],
        "slots": [
            [
                {
                    "query": "教堂",
                    "sources": [1],
                    "scene": 1,
                    "relevance": 1.0,
                    "seconds": 20.0,
                    "top": [1],
                },
                {
                    "query": "灯光",
                    "sources": [2],
                    "scene": None,
                    "relevance": None,
                    "seconds": 0.0,
                    "top": [],
                },
            ]
        ],
        "outline": plan,
        "outline_raw": '{"title": "<t>"}',
        "outline_seconds": 3.0,
        "outline_failed_attempts": [],
        "duration_failed_attempts": ["x"],
        "chunks": [
            {
                "number": 1,
                "videos": ["v"],
                "start": 0.0,
                "end": 18.0,
                "segments": [["v", 0], ["v", 1]],
            }
        ],
        "catalog": [
            {
                "number": 1,
                "start": 0.0,
                "end": 18.0,
                "videos": [1],
                "title": "<教堂>",
                "summary": "s",
            }
        ],
        "catalog_total": 1,
        "order": "文件名",
        "scenes": [
            {
                "number": 1,
                "video": "v",
                "start": 0.0,
                "end": 18.0,
                "segments": [["v", 0], ["v", 1]],
            }
        ],
    }
    video = layout.Video(id="v", sha256="s", file="long/walk.mp4", size=1, mtime_ns=0)
    folder = layout.edit_dir(
        tmp_path / "out", "20261028-120000", timeline["instruction"]
    )
    path = report.write_edit(tmp_path, folder, [video], timeline)
    html = path.read_text("utf-8")
    assert path == folder / "report.html"
    assert "剪 20 秒&lt;教堂&gt;" in html and "素材里找不到：烟花" in html
    assert "指令里的「20 秒」" in html and "重试 1 次" in html
    assert "<li>场景 1（0:00.0–0:18.0）「&lt;教堂&gt;」：教堂正面</li>" in html
    assert "成片设计</b><p>先外观后内部" in html
    assert (
        "设计取自场景 1</span> → 场景 1" in html
        and "是设计的场景" in html
        and "walk.mp4 0:00.0–0:18.0" in html
    )
    assert '<img class="taken" src="../../../data/work/keyframes/' in html
    assert "「灯光」</b><span class='muted'> 设计取自场景 2" in html
    assert "没有可用的场景" in html
    assert "<span class='miss'>选到 8s</span>" in html
    assert 'src="preview.mp4"' in html
    assert '<pre class="raw">{&quot;title&quot;: &quot;&lt;t&gt;&quot;}</pre>' in html


# --- review --------------------------------------------------------------------


def _review_setup() -> (
    tuple[list[suggest.Candidate], list[suggest.Assessment], dict, list[suggest.Clip]]
):
    cands = [_cand(i, i * 10.0) for i in range(6)]
    plan = _plan([(5, [_slot("a")]), (5, [_slot("b")])], seconds=[10, 10])
    filled = [
        suggest.Clip(0, "p0", "v", 0, 0.0, 8.0, "d0", "r"),
        suggest.Clip(1, "p1", "v", 1, 10.0, 18.0, "d1", "r"),
    ]
    return cands, _assess(cands), plan, filled


def _relevance() -> list[list[dict[int, float]]]:
    # Part p0 finds c2 and c3, part p1 finds c4; c5 is relevant to nothing.
    return [[{2: 0.9, 3: 0.5, 0: 0.4}], [{4: 0.8, 1: 0.7}]]


def _answer(clips: list[tuple[str, str, float]], on_beat: bool = False) -> str:
    return json.dumps(
        {
            "notes": "换了开场",
            "on_beat": on_beat,
            "clips": [
                {"segment": s, "part": p, "seconds": sec, "why": "好"}
                for s, p, sec in clips
            ],
        },
        ensure_ascii=False,
    )


def test_review_request_builds_the_planner_cut(monkeypatch) -> None:
    cands, assessments, plan, filled = _review_setup()
    asked: list[str] = []

    def planner(system, prompt, schema, name, temperature):
        asked.append(prompt)
        return _answer([("c2", "p0", 7.0), ("c0", "p0", 5.0), ("c4", "p1", 8.0)]), 1.0

    monkeypatch.setattr(outline, "PLANNER", planner)
    clips, review, _, _, errors = instruct.review_request(
        "剪 20 秒", 20.0, plan, filled, cands, assessments, _relevance()
    )
    assert [c.index for c in clips] == [2, 0, 4]
    assert [c.role for c in clips] == ["p0", "p0", "p1"]
    assert [round(c.duration, 1) for c in clips] == [7.0, 5.0, 8.0]
    assert clips[0].reason.startswith("审片：好") and not errors
    # The cut and the unused segments relevant to each part are offered;
    # c5 is relevant to no part and c1 is already in the cut.
    assert "c2 |" in asked[0] and "c4 |" in asked[0] and "c5 |" not in asked[0]
    assert asked[0].count("c1 |") == 1


def test_review_request_levels_a_cut_that_runs_over(monkeypatch) -> None:
    cands, assessments, plan, filled = _review_setup()
    answer = _answer([("c2", "p0", 8.0), ("c0", "p0", 8.0), ("c4", "p1", 8.0)])
    monkeypatch.setattr(outline, "PLANNER", lambda *a: (answer, 0.0))
    clips, *_ = instruct.review_request(
        "剪 20 秒", 20.0, plan, filled, cands, assessments, _relevance()
    )
    assert sum(c.duration for c in clips) == pytest.approx(20.0, abs=0.01)


@pytest.mark.parametrize(
    ("clips", "error"),
    [
        ([("c5", "p0", 10.0), ("c0", "p1", 10.0)], "not offered"),
        ([("c0", "p0", 8.0), ("c0", "p1", 8.0), ("c2", "p1", 4.0)], "twice"),
        ([("c0", "p9", 8.0), ("c2", "p1", 8.0), ("c4", "p1", 4.0)], "part"),
        ([("c0", "p0", 9.0), ("c2", "p1", 8.0), ("c4", "p1", 4.0)], "outside"),
        ([("c0", "p0", 1.0), ("c2", "p1", 8.0), ("c4", "p1", 8.0)], "outside"),
        ([("c0", "p0", 8.0), ("c2", "p1", 8.0)], "add up to 16.0s"),
        ([("x2", "p0", 8.0)], "bad segment"),
    ],
)
def test_validate_review_rejects(clips, error) -> None:
    cands, _, _, _ = _review_setup()
    with pytest.raises(outline.OutlineError, match=error):
        instruct.validate_review(
            json.loads(_answer(clips)), {0, 1, 2, 3, 4}, ["p0", "p1"], cands, 20.0
        )


def test_validate_review_needs_notes() -> None:
    cands, _, _, _ = _review_setup()
    parsed = json.loads(_answer([("c0", "p0", 8.0), ("c2", "p1", 8.0)]))
    parsed["notes"] = " "
    with pytest.raises(outline.OutlineError, match="notes"):
        instruct.validate_review(parsed, {0, 2}, ["p0", "p1"], cands, 16.0)


def test_whole_sentences_widens_to_the_sentence_inside_the_segment() -> None:
    speech = [(2.0, 4.0), (6.5, 9.5), (11.0, 14.0)]
    # Cuts into the first and second sentences: both taken in, padded.
    assert outline.whole_sentences(3.0, 7.0, 0.0, 10.0, speech) == (1.8, 9.7)
    # The third runs past the segment's end (10 s): it stays cut there.
    assert outline.whole_sentences(5.0, 9.8, 0.0, 10.0, speech) == (5.0, 9.8)
    # Padding stops at the segment.
    assert outline.whole_sentences(2.5, 3.0, 2.0, 10.0, speech) == (2.0, 4.2)


def test_review_keeps_spoken_clips_whole_and_shortens_the_others(
    monkeypatch,
) -> None:
    cands, assessments, plan, filled = _review_setup()
    # c2 (20-28 s) holds a sentence at 21-27 s; the planner asks for 4 s.
    lines = {"v": [(21.0, 27.0, "我们明天去冲浪")]}
    asked: list[str] = []

    def planner(system, prompt, schema, name, temperature):
        asked.append(prompt)
        return _answer([("c2", "p0", 4.0), ("c0", "p0", 8.0), ("c4", "p1", 8.0)]), 0

    monkeypatch.setattr(outline, "PLANNER", planner)
    clips, review, *_ = instruct.review_request(
        "剪 20 秒", 20.0, plan, filled, cands, assessments, _relevance(), lines
    )
    assert "对白：1.0–7.0s「我们明天去冲浪」" in asked[0]
    spoken = clips[0]
    assert spoken.src_start <= 20.8 and spoken.src_end >= 27.2  # whole, padded
    # The clips without speech gave up the difference.
    assert sum(c.duration for c in clips) == pytest.approx(20.0, abs=0.01)
    assert review["fit"] == ""


def test_review_notes_when_the_speech_alone_runs_over(monkeypatch) -> None:
    cands, assessments, plan, filled = _review_setup()
    lines = {"v": [(0.5, 7.5, "第一句"), (20.5, 27.5, "第二句"), (40.5, 47.5, "三")]}
    answer = _answer([("c0", "p0", 7.0), ("c2", "p0", 7.0), ("c4", "p1", 7.0)])
    monkeypatch.setattr(outline, "PLANNER", lambda *a: (answer, 0.0))
    clips, review, *_ = instruct.review_request(
        "剪 20 秒", 20.0, plan, filled, cands, assessments, _relevance(), lines
    )
    assert all(c.duration >= 7.0 for c in clips)
    assert "保留整句" in review["fit"]


def test_review_on_beats_starts_clips_on_beats(monkeypatch) -> None:
    cands, assessments, plan, filled = _review_setup()
    answer = _answer([("c2", "p0", 4.0), ("c0", "p0", 6.0), ("c4", "p1", 6.0)], True)
    monkeypatch.setattr(outline, "PLANNER", lambda *a: (answer, 0.0))

    def grid(video: str, lo: float, hi: float) -> tuple[float, list[float]]:
        return 0.5, [0.1 + 0.5 * k for k in range(200) if lo <= 0.1 + 0.5 * k <= hi]

    clips, review, *_ = instruct.review_request(
        "剪 15 秒，卡在节拍上",
        15.0,
        plan,
        filled,
        cands,
        assessments,
        _relevance(),
        beat_grid=grid,
    )
    for c in clips:
        assert round((c.src_start - 0.1) / 0.5, 6) % 1 == 0  # on a beat
        assert round(c.duration / 0.5, 6) % 1 == 0  # whole beats
    assert sum(c.duration for c in clips) <= 15.5 and review["fit"] == ""


def test_review_on_beats_needs_beats(monkeypatch) -> None:
    cands, assessments, plan, filled = _review_setup()
    answer = _answer([("c0", "p0", 8.0), ("c2", "p1", 8.0), ("c4", "p1", 4.0)], True)
    monkeypatch.setattr(outline, "PLANNER", lambda *a: (answer, 0.0))
    with pytest.raises(outline.OutlineError, match="beat_grid"):
        instruct.review_request(
            "剪 20 秒", 20.0, plan, filled, cands, assessments, _relevance()
        )


@pytest.mark.parametrize(
    ("brightness", "words"),
    [(8.0, "画面很暗"), (44.9, "画面很暗"), (52.0, "画面偏暗"), (90.0, "画面明亮")],
)
def test_lightness_reads_measured_brightness(brightness: float, words: str) -> None:
    assert instruct.lightness(brightness) == words


def test_review_line_shows_time_light_and_measured_brightness() -> None:
    c = replace(_cand(3, 10.0, time_of_day="黄昏"), brightness=20.0)
    c.tags["light"] = "逆光"
    a = suggest.assess(c)
    line = instruct._segment_line(3, c, a, {"v": 1}, {})
    assert "黄昏，逆光，画面很暗" in line
    # Without signals the line still reads, with no brightness.
    plain = instruct._segment_line(3, _cand(3, 10.0), a, {"v": 1}, {})
    assert "白天，—" in plain and "画面" not in plain.split("|")[4]
