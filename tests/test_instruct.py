from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from video_demo import instruct, layout, outline, report, suggest


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


def _slot(query: str, time_of_day: str = instruct.ANY_TIME) -> dict[str, str]:
    return {"query": query, "time_of_day": time_of_day}


def _plan(parts: list[tuple[int, list[dict]]], seconds: list[int] | None = None):
    """Parts as (weight, slots); ``seconds`` sets duration_sec directly."""
    plan: dict[str, Any] = {
        "title": "t",
        "understanding": "u",
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


def test_validate_request_allocates_seconds() -> None:
    plan = instruct.validate_request(_plan([(5, [_slot("教堂")])]), 15.0)
    assert plan["parts"][0]["duration_sec"] == 15
    # part_limits(30) allows three parts of 10 s.
    plan = _plan([(1, [_slot("a")]), (1, [_slot("b")]), (2, [_slot("c")])])
    assert [p["duration_sec"] for p in instruct.validate_request(plan, 30.0)["parts"]]
    assert sum(p["duration_sec"] for p in plan["parts"]) == 30


@pytest.mark.parametrize("when", instruct.SLOT_TIMES)
def test_validate_request_takes_every_slot_time(when: str) -> None:
    plan = instruct.validate_request(_plan([(5, [_slot("a", when)])]), 30.0)
    assert plan["parts"][0]["scenes"][0]["time_of_day"] == when


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (_plan([(1, [_slot(x)]) for x in "abcd"]), "4 parts, need 1-3"),
        (_plan([]), "0 parts"),
        (_plan([(5, [_slot("a"), _slot("a")])]), "listed twice"),
        (_plan([(5, [_slot("a")]), (5, [_slot("a")])]), "listed twice"),
        (_plan([(5, [_slot(" ")])]), "empty scene"),
        (_plan([(5, [_slot("a", "凌晨")])]), "time_of_day"),
        (_plan([(5, [_slot("a", "不确定")])]), "time_of_day"),
        (_plan([(0, [_slot("a")])]), "weight"),
        (_plan([(5, [])]), "needs 1-4 scenes"),
        ({**_plan([(5, [_slot("a")])]), "understanding": ""}, "understanding"),
        ({**_plan([(5, [_slot("a")])]), "unmet": [""]}, "unmet"),
    ],
)
def test_validate_request_rejects(plan: dict, message: str) -> None:
    with pytest.raises(outline.OutlineError, match=message):
        instruct.validate_request(plan, 30.0)


def test_plan_request_prompt_and_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [
        outline.Chunk(
            1,
            "v",
            [0, 1, 2, 3],
            0.0,
            38.0,
            {
                "summary": "教堂",
                "places": ["教堂"],
                "subjects": [],
                "activities": ["参观"],
                "highlights": [],
            },
        )
    ]
    seen: list[tuple[str, dict]] = []

    def fake(prompt: str, _t: float, schema: dict, *_: Any) -> tuple[str, float]:
        seen.append((prompt, schema))
        return json.dumps(_plan([(5, [_slot("教堂外观")])])), 1.0

    monkeypatch.setattr(outline, "_request", fake)
    plan, _, _, errors = instruct.plan_request("剪 30 秒教堂", chunks, 30.0)
    assert plan["parts"][0]["duration_sec"] == 30 and errors == []
    prompt, schema = seen[0]
    assert "「剪 30 秒教堂」" in prompt and "[1] 0:00–0:38 | 教堂" in prompt
    assert schema["properties"]["parts"]["maxItems"] == 3
    # The shared schema is not changed by the per-call limit.
    assert instruct.REQUEST_SCHEMA["properties"]["parts"]["maxItems"] == 8


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


def test_fill_request_time_of_day_never_takes_another_time() -> None:
    cands = [
        _cand(0, 0.0),
        _cand(1, 10.0),
        _cand(2, 20.0, time_of_day="夜晚"),
        _cand(3, 30.0, time_of_day="夜晚"),
    ]
    scenes = _scenes([[0, 1], [2, 3]])
    plan = _plan([(5, [_slot("灯光", "夜晚"), _slot("海边", "黄昏")])], seconds=[16])
    # Scene 1 is the better text match, but it is daytime.
    scene_rel = [[{1: 1.0, 2: 0.5}, {1: 1.0, 2: 1.0}]]
    clips, notes, slots = instruct.fill_request(
        plan, scenes, cands, _assess(cands), scene_rel, [[_flat(4)] * 2], set()
    )
    assert [s["scene"] for s in slots[0]] == [2, None]
    assert slots[0][1]["top"] == []
    assert "「p0」场景「海边」没有时段为「黄昏」的可用场景" in notes
    assert {c.index for c in clips} <= {2, 3} and "，夜晚→" in clips[0].reason


def test_fill_request_mixed_scene_has_no_time() -> None:
    cands = [_cand(0, 0.0), _cand(1, 10.0, time_of_day="夜晚")]
    assert instruct.scene_time(cands, [0, 1]) is None
    assert instruct.scene_time(cands, [1]) == "夜晚"


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
    assert "「p0」素材不足：选到 8.0s，目标 30s" in notes


def test_fill_request_fills_a_target_equal_to_all_usable_footage() -> None:
    cands = [_cand(i, 10.0 * i) for i in range(4)]
    plan = instruct.validate_request(_plan([(5, [_slot("a"), _slot("b")])]), 32.0)
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
    plan = _plan([(5, [_slot("教堂"), _slot("灯光", "夜晚")])], seconds=[20])
    plan["unmet"] = ["烟花"]
    timeline = {
        "instruction": "剪 20 秒<教堂>",
        "created": "20261028-120000",
        "videos": ["v"],
        "target_sec": 20.0,
        "target_source": {"kind": "instruction", "quote": "20 秒"},
        "total_sec": 8.0,
        "notes": ["「p0」场景「灯光」没有时段为「夜晚」的可用场景"],
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
                    "time_of_day": "不限",
                    "scene": 1,
                    "relevance": 1.0,
                    "seconds": 20.0,
                    "top": [1],
                },
                {
                    "query": "灯光",
                    "time_of_day": "夜晚",
                    "scene": None,
                    "relevance": None,
                    "seconds": 0.0,
                    "top": [],
                },
            ]
        ],
        "outline": plan,
        "outline_seconds": 3.0,
        "outline_failed_attempts": [],
        "duration_failed_attempts": ["x"],
        "chunks": [{"start": 0.0, "end": 18.0}],
        "scenes": [
            {"number": 1, "video": "v", "start": 0.0, "end": 18.0, "segments": [0, 1]}
        ],
    }
    video = layout.Video(id="v", sha256="s", file="long/walk.mp4", size=1, mtime_ns=0)
    folder = layout.edit_dir(tmp_path / "out", "20261028-120000")
    path = report.write_edit(tmp_path, folder, [video], timeline)
    html = path.read_text("utf-8")
    assert path == folder / "report.html"
    assert "剪 20 秒&lt;教堂&gt;" in html and "素材里找不到：烟花" in html
    assert "指令里的「20 秒」" in html and "重试 1 次" in html
    assert "「教堂」</b> → 场景 1" in html and "walk.mp4 0:00.0–0:18.0" in html
    assert '<img class="taken" src="../../../data/work/keyframes/' in html
    assert '「灯光」（夜晚）</b><p class="miss">没有可用的场景' in html
    assert "<span class='miss'>选到 8s</span>" in html
    assert 'src="preview.mp4"' in html
