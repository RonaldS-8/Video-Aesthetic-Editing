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
    quality = kw.pop("quality", 0.5)
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
        quality=quality,
        tech_raw=0.0,
        aes_raw=0.0,
    )


def test_score_endpoints_and_weights() -> None:
    assert suggest.assess(_cand(0, quality=1.0, highlight=5)).score == 5.0
    assert suggest.assess(_cand(1, quality=0.0, highlight=1)).score == 1.0
    # The VLM highlight moves the score by at most 4 x 20% = 0.8.
    low = suggest.assess(_cand(2, quality=0.5, highlight=1)).score
    high = suggest.assess(_cand(3, quality=0.5, highlight=5)).score
    assert high - low == pytest.approx(0.8)


def test_quality_model_outweighs_vlm_highlight() -> None:
    good_shot = suggest.assess(_cand(0, quality=0.9, highlight=1))
    vlm_favourite = suggest.assess(_cand(1, quality=0.3, highlight=5))
    assert good_shot.score > vlm_favourite.score


@pytest.mark.parametrize("quality", [-0.01, 1.01])
def test_quality_out_of_range_is_rejected(quality: float) -> None:
    with pytest.raises(ValueError, match="quality"):
        suggest.assess(_cand(0, quality=quality))


def test_vlm_issues_no_longer_affect_the_score() -> None:
    clean = suggest.assess(_cand(0, issues=["无明显问题"]))
    flagged = suggest.assess(_cand(1, issues=["模糊", "抖动"]))
    assert clean.score == flagged.score


def test_assess_penalises_measured_defects() -> None:
    # 1 + 4 * (0.8 * 0.5 + 0.2 * 0.75) = 3.2, then -1.5 blurry, -1.0 shaky.
    a = suggest.assess(_cand(0, flags=["blurry", "shaky"], highlight=4))
    assert a.score == pytest.approx(0.7) and not a.keep
    assert any("偏糊" in r for r in a.reasons) and any("抖动" in r for r in a.reasons)


def test_assess_does_not_penalise_dark_night_scene() -> None:
    night = suggest.assess(_cand(0, flags=["low_light"], time_of_day="夜晚"))
    day = suggest.assess(_cand(1, flags=["low_light"], time_of_day="白天"))
    assert night.score == pytest.approx(3.2) and day.score == pytest.approx(2.2)


def test_timeline_opens_wide_dedupes_and_validates() -> None:
    cands = [
        _cand(0, highlight=5, shot_size="远景", tags=["城市", "夜景", "灯光"]),
        _cand(1, shot=0, highlight=5, tags=["x", "y", "z"]),  # next window, shot 0
        _cand(2, highlight=5, tags=["城市", "夜景", "灯光"]),  # duplicate tags
        _cand(3, highlight=4, tags=["船", "商场", "水"]),
        _cand(4, highlight=3, shot_size="全景", tags=["极光", "夜空", "湖"]),
        _cand(5, quality=0.1, highlight=2, tags=["p", "q", "r"]),  # dropped (1.52)
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
            quality=0.5,
            tech_raw=0.0,
            aes_raw=0.0,
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
    # Equal scores: the sharpest segments win the tie, whichever file they are in.
    assert set(base) == {"内容00", "内容10", "内容11"}


def test_negative_sharpness_is_rejected() -> None:
    with pytest.raises(ValueError, match="sharpness_rel"):
        suggest.assess(_cand(0, sharpness_rel=-1.0))


def test_timeline_reports_shortfall_instead_of_padding() -> None:
    cands = [_cand(0, highlight=5, shot_size="远景")]
    clips, notes = suggest.build_timeline(cands, [suggest.assess(cands[0])], 30.0)
    assert len(clips) == 1 and notes and "不足" in notes[0]


def test_validate_timeline_rejects_out_of_segment_clip() -> None:
    cand = _cand(0)
    bad = suggest.Clip(0, "主体", "v.webm", 0, 3.0, 6.0, "d", "r")
    with pytest.raises(ValueError, match="exceeds"):
        suggest.validate_timeline([bad], [cand])


def _hl(cands: list[suggest.Candidate], target: float) -> list[suggest.Clip]:
    clips, _ = suggest.build_highlights(
        cands, [suggest.assess(c) for c in cands], target
    )
    return clips


def test_highlights_keep_whole_segments_in_shooting_order() -> None:
    # 10 s apart, 8 s long, best quality last; distinct tags, no repeats.
    cands = [
        _cand(i, dur=8.0, quality=0.3 + 0.05 * i, tags=[f"t{i}", f"u{i}", f"v{i}"])
        for i in range(10)
    ]
    clips = _hl(cands, 24.0)
    assert [c.index for c in clips] == sorted(c.index for c in clips)
    assert all(c.duration == pytest.approx(8.0) for c in clips)
    assert sum(c.duration for c in clips) >= 24.0
    # Best three by score, since all fall into one 5-minute window plus
    # the coverage pick (the window's best, which is also the best overall).
    assert {c.index for c in clips} == {7, 8, 9}


def test_highlights_cover_every_window() -> None:
    # One strong segment at the start; weak ones every 5 minutes later.
    def at(i: int, start: float, quality: float) -> suggest.Candidate:
        c = _cand(i, dur=6.0, quality=quality, tags=[f"t{i}", f"u{i}", f"v{i}"])
        return suggest.Candidate(**{**c.__dict__, "start": start, "end": start + 6})

    cands = [at(0, 0.0, 0.9), at(1, 10.0, 0.9), at(2, 320.0, 0.2), at(3, 650.0, 0.2)]
    clips = _hl(cands, 6.0)
    assert [c.index for c in clips] == [0, 2, 3]
    assert [c.role for c in clips] == ["覆盖", "覆盖", "覆盖"]


def test_highlights_skip_near_repeats_but_not_distant_ones() -> None:
    same = ["a", "b", "c"]
    near = [_cand(0, dur=6.0, tags=same, quality=0.9), _cand(1, dur=6.0, tags=same)]
    assert [c.index for c in _hl(near, 100.0)] == [0]
    far = near[:1] + [
        suggest.Candidate(**{**near[1].__dict__, "start": 200.0, "end": 206.0})
    ]
    assert [c.index for c in _hl(far, 100.0)] == [0, 1]


def test_highlights_report_a_shortfall() -> None:
    clips, notes = suggest.build_highlights(
        [_cand(0, dur=5.0)], [suggest.assess(_cand(0, dur=5.0))], 60.0
    )
    assert len(clips) == 1 and "可用素材不足" in notes[0]


def test_calibration_ranks_night_among_night() -> None:
    cands = [
        _cand(0, quality=0.8),
        _cand(1, quality=0.6),
        _cand(2, quality=0.3, time_of_day="夜晚"),
        _cand(3, quality=0.2, time_of_day="夜晚"),
    ]
    got = [c.quality for c in suggest.calibrate_by_time(cands)]
    assert got == [0.75, 0.25, 0.75, 0.25]
    assert suggest.calibrate_by_time([_cand(0, quality=0.1)])[0].quality == 0.5
