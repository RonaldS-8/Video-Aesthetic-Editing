"""Deterministic per-segment signals and keyframe export (no model involved)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from video_demo.media import sample_frames
from video_demo.segment import Segment

SIGNAL_FPS = 4
KEYFRAME_SIDE = 448
# One keyframe per window of this length for tagging (2026-10-26; was two
# fixed frames at 30% and 70%, which missed content that changes within a
# segment, e.g. hand-held walks and whip pans). Within each window the
# sharpest sample is kept, so motion-blurred frames are avoided.
KEYFRAME_STEP_SEC = 2.0

# Heuristic thresholds, eyeballed on the six demo videos (not tuned on labels).
# Night footage is legitimately dark, so this is a hint, not a defect.
LOW_LIGHT_LUMA = 40.0
# Laplacian variance depends on content (night sky is smooth), so blur is
# judged against the video's own median as well as an absolute floor.
BLUR_ABS = 30.0
BLUR_REL = 0.3
# Std of frame-to-frame global shift change; walking footage sits around 5-10.
SHAKE_JITTER = 12.0


@dataclass(frozen=True)
class SegmentSignals:
    brightness: float  # mean luma 0-255
    sharpness: float  # median variance of Laplacian
    motion: float  # mean |frame diff| between samples
    pan_speed: float  # mean global shift (px per sample, 448-px frame)
    shake: float  # std of global shift changes
    flags: list[str]
    keyframes: list[str]
    keyframe_times: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def flags_for(
    brightness: float, sharpness: float, shake: float, median_sharpness: float
) -> list[str]:
    flags: list[str] = []
    if brightness < LOW_LIGHT_LUMA:
        flags.append("low_light")
    if sharpness < BLUR_ABS or sharpness < BLUR_REL * median_sharpness:
        flags.append("blurry")
    if shake > SHAKE_JITTER:
        flags.append("shaky")
    return flags


def compute(
    path: Path,
    segments: list[Segment],
    keyframe_dir: Path,
    end_time: float | None = None,
) -> list[SegmentSignals]:
    """One decode pass: bucket samples by segment, then score each bucket."""
    buckets: list[list[tuple[float, np.ndarray]]] = [[] for _ in segments]
    seg_idx = 0
    for t, rgb in sample_frames(path, SIGNAL_FPS, KEYFRAME_SIDE, end_time):
        while seg_idx < len(segments) - 1 and t >= segments[seg_idx].end:
            seg_idx += 1
        if segments[seg_idx].start <= t < segments[seg_idx].end + 1e-3:
            buckets[seg_idx].append((t, rgb))

    keyframe_dir.mkdir(parents=True, exist_ok=True)
    results = [
        _score_segment(seg, samples, keyframe_dir)
        for seg, samples in zip(segments, buckets, strict=True)
    ]
    median_sharpness = float(np.median([r.sharpness for r in results]))
    return [
        replace(
            r, flags=flags_for(r.brightness, r.sharpness, r.shake, median_sharpness)
        )
        for r in results
    ]


def _score_segment(
    seg: Segment, samples: list[tuple[float, np.ndarray]], keyframe_dir: Path
) -> SegmentSignals:
    if not samples:
        raise ValueError(f"No frames sampled for segment {seg.index} {seg}")
    gray = [cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for _, rgb in samples]
    grayf = [g.astype(np.float32) for g in gray]
    brightness = float(np.mean([g.mean() for g in gray]))
    per_frame_sharpness = [cv2.Laplacian(g, cv2.CV_64F).var() for g in gray]
    sharpness = float(np.median(per_frame_sharpness))
    motion = pan = shake = 0.0
    if len(grayf) > 1:
        pairs = list(zip(grayf, grayf[1:], strict=False))
        motion = float(np.mean([np.abs(b - a).mean() for a, b in pairs]))
        shifts = np.array([cv2.phaseCorrelate(a, b)[0] for a, b in pairs])
        pan = float(np.mean(np.linalg.norm(shifts, axis=1)))
        if len(shifts) > 1:
            shake = float(np.mean(np.std(np.diff(shifts, axis=0), axis=0)))

    names: list[str] = []
    times: list[float] = []
    for k, i in enumerate(pick_keyframes(seg, samples, per_frame_sharpness)):
        t, rgb = samples[i]
        name = f"seg{seg.index:03d}_{k}.jpg"
        Image.fromarray(rgb).save(keyframe_dir / name, quality=88)
        names.append(name)
        times.append(round(t, 3))

    return SegmentSignals(
        brightness=round(brightness, 1),
        sharpness=round(sharpness, 1),
        motion=round(motion, 2),
        pan_speed=round(pan, 2),
        shake=round(shake, 2),
        flags=[],
        keyframes=names,
        keyframe_times=times,
    )


def pick_keyframes(
    seg: Segment,
    samples: list[tuple[float, np.ndarray]],
    sharpness: list[float],
    step: float = KEYFRAME_STEP_SEC,
) -> list[int]:
    """Indices of the sharpest sample in each ``step``-second window.

    Windows split the segment evenly into ceil(duration / step) parts, so a
    10 s segment gets 5 keyframes and anything up to ``step`` gets one. A
    window without samples (very short segments) is skipped; the segment
    always gets at least one keyframe.
    """
    n = max(1, math.ceil(seg.duration / step - 1e-9))
    width = seg.duration / n
    picked: list[int] = []
    for w in range(n):
        lo, hi = seg.start + w * width, seg.start + (w + 1) * width
        inside = [i for i, (t, _) in enumerate(samples) if lo <= t < hi]
        if inside:
            picked.append(max(inside, key=lambda i: (sharpness[i], -i)))
    if not picked:
        picked.append(max(range(len(samples)), key=lambda i: (sharpness[i], -i)))
    return picked
