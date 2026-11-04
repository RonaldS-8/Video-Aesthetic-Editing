from __future__ import annotations

import numpy as np
import pytest

from video_demo import beats


def _clicks(bpm: float, seconds: float, offset: float, rate: int = beats.RATE):
    """A click every beat from ``offset``, over quiet noise."""
    rng = np.random.default_rng(3)
    out = rng.normal(0, 0.01, int(seconds * rate)).astype(np.float32)
    click = np.hanning(200).astype(np.float32) * np.sin(np.arange(200) * 0.9)
    for t in np.arange(offset, seconds - 0.05, 60 / bpm):
        at = int(t * rate)
        out[at : at + 200] += click
    return out


@pytest.mark.parametrize("bpm", [96.0, 128.0, 150.0])
def test_find_beats_follows_a_click_track(bpm: float) -> None:
    period, times = beats.find_beats(_clicks(bpm, 12.0, 0.31), beats.RATE)
    assert period == pytest.approx(60 / bpm, rel=0.02)
    clicks = np.arange(0.31, 12.0, 60 / bpm)
    # Every found beat is on a click (within 1.5 hops of 12 ms).
    for t in times[1:-1]:
        assert np.min(np.abs(clicks - t)) < 0.03


def test_find_beats_needs_sound() -> None:
    with pytest.raises(ValueError, match="sound"):
        beats.find_beats(np.zeros(beats.RATE * 4, np.float32), beats.RATE)


def test_snap_starts_on_a_beat_and_lasts_whole_beats() -> None:
    times = [0.25 + 0.5 * k for k in range(40)]
    assert beats.snap(3.1, 2.2, 0.0, 20.0, 0.5, times) == (3.25, 5.25)
    # Not enough room before the segment ends: fewer beats.
    assert beats.snap(9.0, 3.0, 8.0, 10.6, 0.5, times) == (8.75, 10.25)
    # No beat fits inside: the window stays as it was.
    assert beats.snap(1.0, 0.3, 1.0, 1.3, 0.5, times) == (1.0, 1.3)
