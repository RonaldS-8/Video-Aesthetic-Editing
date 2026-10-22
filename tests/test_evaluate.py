from __future__ import annotations

import pytest

from video_demo.evaluate import score_boundaries


def test_perfect_match() -> None:
    s = score_boundaries([5.0, 10.0], [5.0, 10.0])
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


def test_tolerance_is_inclusive_and_one_to_one() -> None:
    s = score_boundaries([5.5, 5.4], [5.0], tolerance=0.5)
    # Only the closer prediction may claim the single truth cut.
    assert s.matched == [(5.4, 5.0)]
    assert s.false_alarms == [5.5]
    assert s.precision == 0.5 and s.recall == 1.0


def test_miss_and_empty_prediction() -> None:
    s = score_boundaries([], [3.0])
    assert s.f1 == 0.0 and s.missed == [3.0]


def test_negative_tolerance_rejected() -> None:
    with pytest.raises(ValueError):
        score_boundaries([1.0], [1.0], tolerance=-0.1)
