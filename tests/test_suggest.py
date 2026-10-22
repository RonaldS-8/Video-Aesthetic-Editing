from __future__ import annotations

from typing import Any

import pytest

from video_demo import suggest


def _tags(**kw: Any) -> dict[str, Any]:
    base = {
        "description": "d",
        "shot_size": "中景",
        "time_of_day": "白天",
        "tags": ["a", "b", "c"],
        "highlight": 4,
        "issues": ["无明显问题"],
    }
    return {**base, **kw}


def _cand(
    i: int, shot: int | None = None, dur: float = 5.0, **kw: Any
) -> suggest.Candidate:
    flags = kw.pop("flags", [])
    rel = kw.pop("sharpness_rel", 1.0)
    return suggest.Candidate(
        video="v.webm",
        index=i,
        shot=i if shot is None else shot,
        start=i * 10.0,
        end=i * 10.0 + dur,
        tags=_tags(**kw),
        flags=flags,
        sharpness=500.0,
        sharpness_rel=rel,
    )


def test_assess_penalises_measured_defects() -> None:
    a = suggest.assess(_cand(0, flags=["blurry", "shaky"], highlight=4))
    assert a.score == 1.5 and not a.keep
    assert any("偏糊" in r for r in a.reasons) and any("抖动" in r for r in a.reasons)


def test_assess_does_not_penalise_dark_night_scene() -> None:
    night = suggest.assess(_cand(0, flags=["low_light"], time_of_day="夜晚"))
    day = suggest.assess(_cand(1, flags=["low_light"], time_of_day="白天"))
    assert night.score == 4.0 and day.score == 3.0


def test_timeline_opens_wide_dedupes_and_validates() -> None:
    cands = [
        _cand(0, highlight=5, shot_size="远景", tags=["城市", "夜景", "灯光"]),
        _cand(1, shot=0, highlight=5, tags=["x", "y", "z"]),  # next window, shot 0
        _cand(2, highlight=5, tags=["城市", "夜景", "灯光"]),  # duplicate tags
        _cand(3, highlight=4, tags=["船", "商场", "水"]),
        _cand(4, highlight=3, shot_size="全景", tags=["极光", "夜空", "湖"]),
        _cand(5, highlight=2, tags=["p", "q", "r"]),  # dropped (score 2)
    ]
    assessments = [suggest.assess(c) for c in cands]
    clips, notes = suggest.build_timeline(cands, assessments, target_sec=12.0)
    assert clips[0].role == "开场" and clips[0].index == 0
    assert clips[-1].role == "收尾" and clips[-1].index == 4
    used = {c.index for c in clips}
    assert used == {0, 3, 4}
    assert all(c.duration <= suggest.MAX_CLIP_SEC for c in clips)
    assert notes == []


def test_distant_windows_of_one_long_take_are_both_usable() -> None:
    cands = [
        _cand(0, shot=0, highlight=5, shot_size="远景", tags=["a1", "a2", "a3"]),
        _cand(1, shot=0, highlight=5, tags=["b1", "b2", "b3"]),
        _cand(2, shot=0, highlight=5, tags=["c1", "c2", "c3"]),
    ]
    clips, _ = suggest.build_timeline(cands, [suggest.assess(c) for c in cands], 30.0)
    assert {c.index for c in clips} == {0, 2}


def _two_video_pool(names: tuple[str, str]) -> list[suggest.Candidate]:
    # Same VLM highlight everywhere; only content differs between segments.
    rel = {(0, 0): 1.9, (0, 1): 0.8, (0, 2): 1.0, (1, 0): 1.6, (1, 1): 1.2}
    return [
        suggest.Candidate(
            video=names[v],
            index=i,
            shot=i * 2,
            start=i * 10.0,
            end=i * 10.0 + 5.0,
            tags=_tags(
                description=f"内容{v}{i}",
                shot_size="远景",
                tags=[f"t{v}{i}a", f"t{v}{i}b", f"t{v}{i}c"],
            ),
            flags=[],
            sharpness=500.0,
            sharpness_rel=r,
        )
        for (v, i), r in rel.items()
    ]


def test_selection_ignores_file_names_and_input_order() -> None:
    def picked(cands: list[suggest.Candidate]) -> list[str]:
        clips, _ = suggest.build_timeline(
            cands, [suggest.assess(c) for c in cands], 12.0
        )
        return [c.description for c in clips]

    base = picked(_two_video_pool(("01_a.webm", "02_b.webm")))
    renamed = picked(_two_video_pool(("99_a.webm", "00_b.webm")))
    reversed_input = picked(list(reversed(_two_video_pool(("01_a.webm", "02_b.webm")))))
    assert base == renamed == reversed_input
    # The sharpest segments win the tie on highlight, whichever file they are in.
    assert set(base) == {"内容00", "内容10", "内容11"}


def test_sharpness_term_is_bounded() -> None:
    sharp = suggest.assess(_cand(0, sharpness_rel=8.0))
    soft = suggest.assess(_cand(1, sharpness_rel=0.1))
    black = suggest.assess(_cand(2, sharpness_rel=0.0))
    assert sharp.score == 4.5 and soft.score == 3.5 and black.score == 3.5
    with pytest.raises(ValueError):
        suggest.assess(_cand(3, sharpness_rel=-1.0))


def test_timeline_reports_shortfall_instead_of_padding() -> None:
    cands = [_cand(0, highlight=5, shot_size="远景")]
    clips, notes = suggest.build_timeline(cands, [suggest.assess(cands[0])], 30.0)
    assert len(clips) == 1 and notes and "不足" in notes[0]


def test_validate_timeline_rejects_out_of_segment_clip() -> None:
    cand = _cand(0)
    bad = suggest.Clip(0, "主体", "v.webm", 0, 3.0, 6.0, "d", "r")
    with pytest.raises(ValueError, match="exceeds"):
        suggest.validate_timeline([bad], [cand])
