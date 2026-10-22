from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from video_demo import signals
from video_demo.media import write_video
from video_demo.segment import Segment


def test_flags_thresholds_and_relative_blur() -> None:
    assert signals.flags_for(100.0, 500.0, 1.0, median_sharpness=600.0) == []
    assert signals.flags_for(39.9, 500.0, 1.0, 600.0) == ["low_light"]
    # Sharp in absolute terms but far below this video's own median.
    assert signals.flags_for(100.0, 100.0, 1.0, 600.0) == ["blurry"]
    assert signals.flags_for(100.0, 29.0, 1.0, 50.0) == ["blurry"]
    assert signals.flags_for(100.0, 500.0, 12.1, 600.0) == ["shaky"]


def test_compute_exports_keyframes_and_flags_dark_flat_segment(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    textured = rng.integers(0, 255, (480, 854, 3), np.uint8)
    dark_flat = np.full((480, 854, 3), 10, np.uint8)
    video = tmp_path / "v.mp4"
    write_video([textured] * 60 + [dark_flat] * 60, video)
    segs = [Segment(0, 0.0, 2.0, 0), Segment(1, 2.0, 4.0, 1)]
    out = signals.compute(video, segs, tmp_path / "kf")
    assert out[0].flags == []
    assert set(out[1].flags) == {"low_light", "blurry"}
    for r in out:
        assert len(r.keyframes) == 2
        assert all((tmp_path / "kf" / k).exists() for k in r.keyframes)


def test_compute_rejects_segment_without_frames(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    write_video([np.zeros((480, 854, 3), np.uint8)] * 30, video)
    with pytest.raises(ValueError, match="No frames"):
        signals.compute(video, [Segment(0, 5.0, 6.0, 0)], tmp_path / "kf")
