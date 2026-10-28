from __future__ import annotations

from typing import Any

import pytest

from video_demo import outline, scenes, suggest


def _cand(i: int, dur: float = 5.0, **tags: Any) -> suggest.Candidate:
    base = {
        "description": f"片段{i}",
        "shot_size": "中景",
        "time_of_day": "白天",
        "tags": ["街道", "行人", f"t{i}"],
        "highlight": 4,
        "issues": ["无明显问题"],
    }
    return suggest.Candidate(
        video="v",
        index=i,
        shot=0,
        start=i * 5.0,
        end=i * 5.0 + dur,
        tags={**base, **tags},
        flags=[],
        sharpness=500.0,
        sharpness_rel=1.0,
        quality=0.5,
        tech_raw=0.0,
        aes_raw=0.0,
    )


def _pool(
    rejected: set[int], n: int = 6
) -> tuple[list[suggest.Candidate], list[suggest.Assessment]]:
    return [_cand(i) for i in range(n)], [
        suggest.Assessment("v", i, 1.5 if i in rejected else 3.4, i not in rejected)
        for i in range(n)
    ]


def test_rejected_segments_split_a_scene_into_clips() -> None:
    cands, assessments = _pool({0, 2, 3})
    assert scenes.kept_runs(list(range(6)), cands, assessments) == [[1], [4, 5]]


def test_a_fully_rejected_scene_keeps_nothing() -> None:
    cands, assessments = _pool({0, 1})
    assert scenes.kept_runs([0, 1], cands, assessments) == []


def test_too_short_segments_are_removed_like_rejected_ones() -> None:
    cands = [_cand(0), _cand(1, dur=1.0), _cand(2)]
    assessments = [suggest.Assessment("v", i, 3.4, True) for i in range(3)]
    assert scenes.kept_runs([0, 1, 2], cands, assessments) == [[0], [2]]
    assert "短于" in scenes.removal_reason(cands[1], assessments[1])


def test_one_clip_per_scene_records_where_each_part_comes_from() -> None:
    cands, assessments = _pool({2})
    runs = scenes.kept_runs([1, 2, 3, 4], cands, assessments)
    assert scenes.kept_spans(runs, cands) == [(5.0, 10.0), (15.0, 25.0)]
    entry = scenes.scene_entry(
        3,
        [1, 2, 3, 4],
        cands,
        assessments,
        {"title": "街头", "summary": "行人穿过街道。"},
        "scene_03.mp4",
    )
    assert entry["title"] == "街头" and entry["start"] == 5.0 and entry["end"] == 25.0
    assert entry["file"] == "scene_03.mp4" and entry["duration"] == 15.0
    # The second part starts in the clip where the first one ends.
    assert [(p["start"], p["end"], p["at"]) for p in entry["parts"]] == [
        (5.0, 10.0, 0.0),
        (15.0, 25.0, 5.0),
    ]
    assert [[s["index"] for s in p["segments"]] for p in entry["parts"]] == [
        [1],
        [3, 4],
    ]
    assert [r["index"] for r in entry["removed"]] == [2]
    assert entry["removed"][0]["why"].startswith("得分 1.5")
    assert entry["kept_sec"] == 15.0
    assert entry["tags"][:2] == ["街道", "行人"]


def test_a_fully_rejected_scene_has_no_clip_and_no_summary() -> None:
    cands, assessments = _pool({0, 1})
    entry = scenes.scene_entry(1, [0, 1], cands, assessments, None, None)
    assert entry["file"] is None and entry["parts"] == [] and "title" not in entry
    with pytest.raises(ValueError, match="summary"):
        scenes.scene_entry(1, [0, 1], cands, assessments, {"title": "x"}, None)
    with pytest.raises(ValueError, match="clip_file"):
        scenes.scene_entry(1, [0, 1, 2, 3], cands, assessments, {"title": "x"}, None)


@pytest.mark.parametrize(
    ("parsed", "match"),
    [
        ({"title": " ", "summary": "有内容"}, "title is empty"),
        ({"title": "标题", "summary": ""}, "summary is empty"),
        ({"title": "一" * 13, "summary": "有内容"}, "title over 12"),
        ("not an object", "must be an object"),
    ],
)
def test_scene_summary_is_rejected_when_unusable(parsed: Any, match: str) -> None:
    with pytest.raises(outline.OutlineError, match=match):
        scenes.validate_scene_summary(parsed)


def test_scene_summary_is_trimmed() -> None:
    got = scenes.validate_scene_summary({"title": " 街头 ", "summary": "行人。 "})
    assert got == {"title": "街头", "summary": "行人。"}
