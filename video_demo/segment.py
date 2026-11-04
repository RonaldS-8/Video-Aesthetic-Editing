"""Two-level segmentation: shot boundaries, then windows of about a fixed
length whose splits fall between sentences where speech allows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scenedetect import AdaptiveDetector, detect

from video_demo.media import sample_frames

# Raised from 8 s for hour-long raw footage (2026-10-26); videos segmented
# before keep their 8 s windows.
MAX_SEGMENT_SEC = 10.0
MIN_SEGMENT_SEC = 1.0
# A split inside a long shot moves at most this far from the equal-window
# point to fall between sentences (2026-11-02); videos segmented before keep
# their equal windows.
SPLIT_SHIFT_SEC = 2.0

# Dissolve detector: grayscale frames at 10 fps, compared 0.5 s either side.
DISSOLVE_FPS = 10
DISSOLVE_HALF = 5
DISSOLVE_MIN_DIFF = 25.0
DISSOLVE_MAX_RATIO = 0.45
DISSOLVE_ALPHA = (0.35, 0.65)
# A dissolve this close to a hard cut is treated as the same transition.
CUT_MERGE_SEC = 1.0


Span = tuple[float, float]


@dataclass(frozen=True)
class Cut:
    time: float
    kind: str  # "hard" (scenedetect) or "dissolve" (blend test)


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    end: float
    shot: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def detect_hard_cuts(path: Path, end_time: float | None = None) -> list[float]:
    scenes = detect(str(path), AdaptiveDetector(), end_time=end_time, backend="pyav")
    return [round(s.seconds, 3) for s, _ in scenes[1:]]


def find_dissolves(times: np.ndarray, gray: np.ndarray) -> list[float]:
    """Find mid-points where a frame is a linear blend of its neighbours.

    For a cross-dissolve, ``F[t] ~ a*F[t-h] + (1-a)*F[t+h]`` with ``a`` near
    0.5. Camera pans also change content over ``2h`` but cannot be explained
    by such a blend, which keeps the residual ratio high.
    """
    h = DISSOLVE_HALF
    hits: list[tuple[float, float]] = []
    for i in range(h, len(gray) - h):
        before, after, mid = gray[i - h], gray[i + h], gray[i]
        diff = float(np.mean(np.abs(after - before)))
        if diff < DISSOLVE_MIN_DIFF:
            continue
        y = before - after
        x = mid - after
        alpha = float(np.clip(np.sum(x * y) / np.sum(y * y), 0.0, 1.0))
        ratio = float(np.mean(np.abs(x - alpha * y))) / (diff / 2)
        if (
            ratio <= DISSOLVE_MAX_RATIO
            and DISSOLVE_ALPHA[0] <= alpha <= DISSOLVE_ALPHA[1]
        ):
            hits.append((float(times[i]), diff))
    # Keep the strongest hit of each run of neighbouring frames.
    peaks: list[tuple[float, float]] = []
    for t, diff in hits:
        if peaks and t - peaks[-1][0] <= 2 * h / DISSOLVE_FPS:
            if diff > peaks[-1][1]:
                peaks[-1] = (t, diff)
        else:
            peaks.append((t, diff))
    return [round(t, 3) for t, _ in peaks]


def detect_dissolves(path: Path, end_time: float | None = None) -> list[float]:
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    times: list[float] = []
    frames: list[np.ndarray] = []
    for t, rgb in sample_frames(path, DISSOLVE_FPS, max_side=96, end_time=end_time):
        times.append(t)
        frames.append(rgb.astype(np.float32) @ weights)
    if len(frames) <= 2 * DISSOLVE_HALF:
        return []
    return find_dissolves(np.array(times), np.stack(frames))


def combine_cuts(hard: list[float], dissolves: list[float]) -> list[Cut]:
    cuts = [Cut(t, "hard") for t in hard]
    cuts += [
        Cut(t, "dissolve")
        for t in dissolves
        if all(abs(t - h) > CUT_MERGE_SEC for h in hard)
    ]
    return sorted(cuts, key=lambda c: c.time)


def detect_cuts(path: Path, end_time: float | None = None) -> list[Cut]:
    return combine_cuts(
        detect_hard_cuts(path, end_time), detect_dissolves(path, end_time)
    )


def spans_from_cuts(cuts: list[Cut], limit: float) -> list[tuple[float, float]]:
    """Tile [0, limit] with shot spans; one span covering the clip if no cuts."""
    edges = [0.0] + [c.time for c in cuts if 0.0 < c.time < limit] + [limit]
    return [(a, b) for a, b in zip(edges, edges[1:], strict=False) if b > a]


def detect_shots(
    path: Path, duration: float, end_time: float | None = None
) -> list[tuple[float, float]]:
    limit = duration if end_time is None else min(duration, end_time)
    return spans_from_cuts(detect_cuts(path, end_time), limit)


def merge_short(
    spans: list[tuple[float, float]], min_len: float = MIN_SEGMENT_SEC
) -> list[tuple[float, float]]:
    """Fold spans shorter than ``min_len`` into their previous neighbour."""
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and end - start < min_len:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_len:
        first, second = merged[0], merged[1]
        merged[0:2] = [(first[0], second[1])]
    return merged


def split_long(
    shots: list[tuple[float, float]],
    max_len: float = MAX_SEGMENT_SEC,
    sentences: Sequence[Span] = (),
    clauses: Sequence[Span] = (),
) -> list[Segment]:
    """Cut each shot into windows of about ``max_len``.

    A shot longer than ``max_len`` gets as many windows as equal ones would
    need; each split starts at the equal-window point and moves at most
    ``SPLIT_SHIFT_SEC`` to the nearest point between ``sentences``, else
    between ``clauses`` (spans of speech), else stays put. Without speech
    the windows are equal.
    """
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    segments: list[Segment] = []
    for shot_idx, (start, end) in enumerate(shots):
        if end <= start:
            raise ValueError(f"Empty shot span: {(start, end)}")
        parts = max(1, math.ceil((end - start) / max_len - 1e-9))
        edges = [start]
        for left in range(parts, 1, -1):
            target = edges[-1] + (end - edges[-1]) / left
            lo = max(edges[-1] + MIN_SEGMENT_SEC, target - SPLIT_SHIFT_SEC)
            hi = min(end - MIN_SEGMENT_SEC, target + SPLIT_SHIFT_SEC)
            point = _between(sentences, lo, hi, target)
            if point is None:
                point = _between(clauses, lo, hi, target)
            edges.append(target if point is None else point)
        edges.append(end)
        for seg_start, seg_end in zip(edges, edges[1:], strict=False):
            segments.append(
                Segment(len(segments), round(seg_start, 3), round(seg_end, 3), shot_idx)
            )
    return segments


def _between(
    spans: Sequence[Span], lo: float, hi: float, target: float
) -> float | None:
    """The point in [lo, hi] nearest ``target`` that is inside none of
    ``spans``; None when every point there is."""
    if lo > hi:
        return None
    points = [target] + [t for a, b in spans for t in (a, b)]
    free = [t for t in points if lo <= t <= hi and not strictly_inside(t, spans)]
    return min(free, key=lambda t: (abs(t - target), t), default=None)


def strictly_inside(t: float, spans: Sequence[Span]) -> bool:
    return any(a < t < b for a, b in spans)
