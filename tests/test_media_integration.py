"""End-to-end check on a tiny generated video: encode, sample, detect cuts."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from video_demo.media import (
    OUT_FPS,
    OUT_SIZE,
    count_frames,
    cut_clip,
    probe,
    read_clip,
    sample_frames,
    segment_frames_every,
    segment_frames_picked,
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


def test_frames_every_second_per_span(tmp_path: Path) -> None:
    video = tmp_path / "blocks.mp4"
    _block_video(video)
    spans = [(0.0, 2.5), (2.5, 3.0), (4.2, 6.0)]
    got = dict(segment_frames_every(video, spans, 1.0, (64, 36)))
    # ceil(duration) frames: 0, 1, 2 s / 2.5 s / 4.2, 5.2 s.
    assert [len(got[i]) for i in range(3)] == [3, 1, 2]
    assert got[0][0].shape == (36, 64, 3)
    # Frames come from the right block: 0-2 s, 2-4 s and 4-6 s differ.
    assert not np.array_equal(got[0][0], got[2][0])
    assert np.abs(got[0][0].astype(int) - got[0][1].astype(int)).mean() < 2


def test_picked_frames_match_counts_and_are_bounded(tmp_path: Path) -> None:
    video = tmp_path / "blocks.mp4"
    _block_video(video)
    spans = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    counts = count_frames(video, spans)
    assert counts == [2 * OUT_FPS] * 3
    picks = [{0, 5}, set(), {59}]
    got = list(segment_frames_picked(video, spans, picks, max_side=96))
    assert [(i, sorted(f), seen) for i, f, seen in got] == [
        (0, [0, 5], 60),
        (1, [], 60),
        (2, [59], 60),
    ]
    # 854x480 scaled to a 96-pixel long side, even dimensions.
    assert got[0][1][0].shape == (52, 96, 3)


def test_clip_running_to_the_end_of_the_video(tmp_path: Path) -> None:
    video = tmp_path / "blocks.mp4"
    _block_video(video)
    duration = probe(video).duration
    clip = read_clip(video, 4.0, duration - 4.0)
    assert len(clip) == round((duration - 4.0) * OUT_FPS)
    with pytest.raises(ValueError, match="needed"):
        read_clip(video, 4.0, duration - 4.0 + 1.0)


def _grey_steps_with_audio(path: Path, fps: int = 25, seconds: int = 4) -> None:
    """640x360 video whose grey level steps up every second, with a tone."""
    with av.open(str(path), mode="w") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = 640, 360, "yuv420p"
        audio = container.add_stream("aac", rate=48000)
        audio.layout = "stereo"
        for n in range(fps * seconds):
            arr = np.full((360, 640, 3), 40 * (n // fps + 1), np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame.pts = n
            container.mux(video.encode(frame))
        t = np.arange(48000 * seconds) / 48000
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for i in range(0, len(tone), 1024):
            chunk = tone[i : i + 1024]
            frame = av.AudioFrame.from_ndarray(
                np.stack([chunk, chunk]), format="fltp", layout="stereo"
            )
            frame.rate, frame.pts = 48000, i
            container.mux(audio.encode(frame))
        container.mux(video.encode())
        container.mux(audio.encode())


def test_cut_clip_joins_spans_at_source_size_rate_and_audio(tmp_path: Path) -> None:
    src, out = tmp_path / "steps.mp4", tmp_path / "cut" / "clip.mp4"
    _grey_steps_with_audio(src)
    before = src.read_bytes()
    # Seconds 0 and 2 (grey 40 and 120), skipping second 1 (grey 80).
    cut_clip(src, [(0.0, 1.0), (2.0, 3.0)], out)
    assert src.read_bytes() == before
    info = probe(out)
    assert (info.width, info.height, round(info.fps)) == (640, 360, 25)
    assert info.has_audio and abs(info.duration - 2.0) < 0.05
    frames = read_clip(out, 0.0, 2.0)
    greys = [round(f.mean()) for f in frames]
    assert abs(greys[0] - 40) < 3 and abs(greys[-1] - 120) < 3
    assert all(abs(g - 80) > 20 for g in greys)
    assert not list(out.parent.glob("*.partial.mp4"))
    with pytest.raises(ValueError, match="Empty span"):
        cut_clip(src, [(2.0, 2.0)], out)
    with pytest.raises(ValueError, match="overlapping"):
        cut_clip(src, [(2.0, 3.0), (1.0, 2.0)], out)
