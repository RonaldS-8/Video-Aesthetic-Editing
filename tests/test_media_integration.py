"""End-to-end check on a tiny generated video: encode, sample, detect cuts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from video_demo.media import (
    OUT_FPS,
    OUT_SIZE,
    probe,
    read_clip,
    sample_frames,
    write_video,
)
from video_demo.segment import detect_cuts


def _block_video(path: Path) -> None:
    rng = np.random.default_rng(7)
    frames = []
    for _ in range(3):
        img = rng.integers(0, 255, (OUT_SIZE[1] // 40, OUT_SIZE[0] // 40, 3), np.uint8)
        img = np.kron(img, np.ones((40, 40, 1), np.uint8))
        canvas = np.zeros((OUT_SIZE[1], OUT_SIZE[0], 3), np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        frames += [canvas] * (2 * OUT_FPS)
    write_video(frames, path)


def test_hard_cuts_found_on_generated_video(tmp_path: Path) -> None:
    video = tmp_path / "blocks.mp4"
    _block_video(video)
    info = probe(video)
    assert abs(info.duration - 6.0) < 0.2 and not info.has_audio
    cuts = detect_cuts(video)
    assert [round(c.time, 1) for c in cuts] == [2.0, 4.0]
    assert all(c.kind == "hard" for c in cuts)


def test_sampling_rate_and_clip_reading(tmp_path: Path) -> None:
    video = tmp_path / "blocks.mp4"
    _block_video(video)
    times = [t for t, _ in sample_frames(video, fps=2, max_side=96)]
    assert 11 <= len(times) <= 13
    assert all(b - a >= 0.49 for a, b in zip(times, times[1:], strict=False))
    clip = read_clip(video, 1.0, 2.0)
    assert len(clip) == 2 * OUT_FPS and clip[0].shape == (OUT_SIZE[1], OUT_SIZE[0], 3)
