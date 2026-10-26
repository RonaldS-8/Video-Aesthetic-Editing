"""Video probing and frame sampling built on PyAV."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

OUT_FPS = 30
OUT_SIZE = (854, 480)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if container.duration is None:
            raise ValueError(f"Container reports no duration: {path}")
        return VideoInfo(
            path=path,
            duration=float(container.duration / av.time_base),
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=float(stream.average_rate or 0.0),
            has_audio=len(container.streams.audio) > 0,
        )


def sample_frames(
    path: Path,
    fps: float,
    max_side: int = 448,
    end_time: float | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, RGB frame) at roughly ``fps`` in one sequential pass."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    step = 1.0 / fps
    next_t = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = float(frame.time)
            if end_time is not None and t > end_time:
                break
            if t + 1e-6 < next_t:
                continue
            next_t = t + step
            scale = max_side / max(frame.width, frame.height)
            if scale < 1.0:
                frame = frame.reformat(
                    width=int(frame.width * scale) // 2 * 2,
                    height=int(frame.height * scale) // 2 * 2,
                )
            yield t, frame.to_ndarray(format="rgb24")


def count_frames(
    path: Path,
    spans: list[tuple[float, float]],
    end_time: float | None = None,
) -> list[int]:
    """Frames per [start, end) span, from packet timestamps (no decoding)."""
    counts = [0] * len(spans)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.pts is None or packet.size == 0:
                continue
            t = float(packet.pts * stream.time_base)
            if end_time is not None and t > end_time:
                continue
            for i, (start, end) in enumerate(spans):
                if start <= t < end:
                    counts[i] += 1
                    break
    return counts


def segment_frames_picked(
    path: Path,
    spans: list[tuple[float, float]],
    picks: list[set[int]],
    max_side: int | None = None,
    end_time: float | None = None,
) -> Iterator[tuple[int, dict[int, np.ndarray], int]]:
    """Yield (span index, {frame index: RGB frame}, frames seen) in one pass.

    Only the frames listed in ``picks[i]`` (indices within span i) are
    converted and kept, so memory is bounded by the picks, not the span
    length. Frames larger than ``max_side`` on their long side are scaled
    down. ``frames seen`` lets the caller check the count it picked from.
    """
    if len(picks) != len(spans):
        raise ValueError("one pick set per span")
    idx, seen = 0, 0
    kept: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = float(frame.time)
            if end_time is not None and t > end_time:
                break
            while idx < len(spans) and t >= spans[idx][1]:
                yield idx, kept, seen
                kept, seen, idx = {}, 0, idx + 1
            if idx >= len(spans):
                break
            if t < spans[idx][0]:
                continue
            if seen in picks[idx]:
                scale = (
                    1.0
                    if max_side is None
                    else max_side / max(frame.width, frame.height)
                )
                if scale < 1.0:
                    frame = frame.reformat(
                        width=int(frame.width * scale) // 2 * 2,
                        height=int(frame.height * scale) // 2 * 2,
                    )
                kept[seen] = frame.to_ndarray(format="rgb24")
            seen += 1
    while idx < len(spans):
        yield idx, kept, seen
        kept, seen, idx = {}, 0, idx + 1


def segment_frames_every(
    path: Path,
    spans: list[tuple[float, float]],
    step: float,
    size: tuple[int, int],
    end_time: float | None = None,
) -> Iterator[tuple[int, list[np.ndarray]]]:
    """Yield (span index, RGB frames at start, start + step, ...) in one pass.

    Each span gets the first frame at or after every target time before its
    end, scaled to ``size`` (width, height) with bicubic interpolation, so a
    span of d seconds has ceil(d / step) frames. ``spans`` are sorted and do
    not overlap.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    idx, k = 0, 0
    buf: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = float(frame.time)
            if end_time is not None and t > end_time:
                break
            while idx < len(spans) and (
                t >= spans[idx][1] or spans[idx][0] + k * step >= spans[idx][1]
            ):
                yield idx, buf
                buf, idx, k = [], idx + 1, 0
            if idx >= len(spans):
                break
            if t + 1e-6 >= spans[idx][0] + k * step:
                buf.append(
                    frame.reformat(
                        width=size[0],
                        height=size[1],
                        format="rgb24",
                        interpolation="BICUBIC",
                    ).to_ndarray()
                )
                k += 1
    while idx < len(spans):
        yield idx, buf
        buf, idx = [], idx + 1


def read_clip(path: Path, start: float, duration: float) -> list[np.ndarray]:
    """Resample [start, start+duration) of ``path`` to OUT_FPS RGB frames."""
    n_out = round(duration * OUT_FPS)
    targets = [start + i / OUT_FPS for i in range(n_out)]
    frames: list[np.ndarray] = []
    last: np.ndarray | None = None
    ended = False
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        video_end = (
            float(container.duration / av.time_base) if container.duration else 0.0
        )
        container.seek(int(max(0.0, start - 1.0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = float(frame.time)
            # Sample-and-hold: every target before this frame shows the last one.
            while last is not None and len(frames) < n_out and targets[len(frames)] < t:
                frames.append(last)
            if len(frames) >= n_out:
                break
            last = frame.reformat(
                width=OUT_SIZE[0], height=OUT_SIZE[1], format="rgb24"
            ).to_ndarray()
        else:
            ended = True
    # A clip that runs to the end of the video: the last frame stays on screen
    # until the video ends (there is no later frame to end its hold).
    while (
        ended
        and last is not None
        and len(frames) < n_out
        and targets[len(frames)] < video_end + 1e-3
    ):
        frames.append(last)
    if len(frames) < n_out:
        raise ValueError(
            f"{path.name}: needed {n_out} frames from {start}s, got {len(frames)}"
        )
    return frames


def write_video(frames: Iterable[np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(OUT_FPS))
        stream.width, stream.height = OUT_SIZE
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        for arr in frames:
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
