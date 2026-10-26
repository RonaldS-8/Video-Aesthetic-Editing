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


def segment_frames(
    path: Path,
    spans: list[tuple[float, float]],
    end_time: float | None = None,
) -> Iterator[tuple[int, list[np.ndarray]]]:
    """Yield (span index, native-resolution RGB frames) in one sequential pass.

    ``spans`` are sorted, non-overlapping [start, end) ranges; frames outside
    them are skipped and each span's frames are released once yielded.
    """
    idx = 0
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
            while idx < len(spans) and t >= spans[idx][1]:
                yield idx, buf
                buf = []
                idx += 1
            if idx >= len(spans):
                break
            if t >= spans[idx][0]:
                buf.append(frame.to_ndarray(format="rgb24"))
    # The last span usually ends where the video (or analysis) ends.
    while idx < len(spans):
        yield idx, buf
        buf = []
        idx += 1


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
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
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
