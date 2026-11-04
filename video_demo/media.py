"""Video probing and frame sampling built on PyAV."""

from __future__ import annotations

import functools
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_FPS = 30
OUT_SIZE = (854, 480)
# Cuts made from an instruction are rendered at 1080p, for showing; frames
# are streamed into the encoder (iter_clip), so memory holds about one.
EDIT_SIZE = (1920, 1080)
# x264 quality: previews are for looking at, finished cuts for showing.
PREVIEW_CRF, EDIT_CRF = 23, 18
# Preview sound: stereo AAC; every clip fades in and out this fast so the
# joins do not click.
AUDIO_RATE = 48_000
AUDIO_BITRATE = 128_000
AUDIO_FADE_SEC = 0.025
AAC_FRAME = 1024
# Narration over the original sound: each line is brought to this RMS
# level; the original is turned down to DUCK_GAIN (about -12 dB) while a
# line plays, easing in and out over DUCK_RAMP_SEC.
NARRATION_RMS = 0.1
NARRATION_PEAK = 0.95
DUCK_GAIN = 0.25
DUCK_RAMP_SEC = 0.15
# Burned subtitles, sized for OUT_SIZE frames and scaled with the frame
# height: white, dark outline, near the bottom.
SUBTITLE_SIZE = 26
SUBTITLE_STROKE = 2
SUBTITLE_MARGIN = 24
# A translucent dark box behind the text keeps it readable on any picture
# (and over captions already burned into the source).
SUBTITLE_PAD = 6
SUBTITLE_BOX_ALPHA = 150

# Title card of a cut (sized for OUT_SIZE frames too): a large title and
# a smaller line under it, centred, fading in and out.
TITLE_SIZE = 44
TITLE_SUB_SIZE = 22
TITLE_FADE_SEC = 0.5
# A cut opens from black and closes to black; its sound fades with it.
FADE_IN_SEC = 0.5
FADE_OUT_SEC = 1.2
# Dissolve between two clips: the first runs on this long past its end
# (a handle from the source) under the start of the next.
DISSOLVE_SEC = 0.6
# At every other cut of a finished cut the sound crossfades over this.
AUDIO_XFADE_SEC = 0.15

# Changes frame number n (counted in the output) before it is encoded.
FrameDraw = Callable[[int, np.ndarray], np.ndarray]


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


def capture_time(path: Path) -> datetime | None:
    """When the video was shot: the container's (or its video stream's)
    ``creation_time`` tag as phones and cameras write it, None if absent."""
    with av.open(str(path)) as container:
        tags = {**container.streams.video[0].metadata, **container.metadata}
    value = next((v for k, v in tags.items() if k.lower() == "creation_time"), None)
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


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


# PQ (SMPTE ST 2084) pictures are shown as SDR BT.709: linear light, P3 to
# BT.709 primaries, then a curve that keeps light up to PQ_KNEE as it is and
# rolls off what is brighter, then the BT.709 transfer. Diffuse white at
# 100 nits (as Meridian is graded; 203 nits left faces too dark).
_PQ_M1, _PQ_M2 = 2610 / 16384, 2523 / 4096 * 128
_PQ_C1, _PQ_C2, _PQ_C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
PQ_WHITE_NITS = 100.0
PQ_KNEE = 0.6
P3_TO_709 = np.array(
    [[1.2249, -0.2247, 0.0], [-0.0420, 1.0419, 0.0], [-0.0197, -0.0786, 1.0979]],
    np.float32,
)
# Linear light (white = 1) is looked up on a square-root scale up to this,
# so shadows keep fine steps.
_PQ_TOP, _PQ_STEPS = 16.0, 4096
# A stream that does not say its transfer is PQ when hardly any frame
# reaches near white, which SDR footage does in most frames: share of
# PQ_PROBE_FRAMES frames whose 99.9th percentile reaches PQ_BRIGHT_LEVEL.
# Measured: Meridian (PQ, unflagged) 0 %; SDR walks, vlogs and a night
# timelapse 78-100 %.
PQ_PROBE_FRAMES = 40
PQ_BRIGHT_LEVEL = 217
PQ_BRIGHT_SHARE = 0.1
_TRC_UNSPECIFIED, _TRC_PQ, _TRC_HLG = 2, 16, 18


def _pq_tables() -> tuple[np.ndarray, np.ndarray]:
    """(8-bit PQ code -> linear light with white 1, square-root-scaled
    linear light -> 8-bit SDR code)."""
    e = (np.arange(256) / 255) ** (1 / _PQ_M2)
    nits = 10000 * (np.maximum(e - _PQ_C1, 0) / (_PQ_C2 - _PQ_C3 * e)) ** (1 / _PQ_M1)
    x = (np.arange(_PQ_STEPS) / (_PQ_STEPS - 1)) ** 2 * _PQ_TOP
    over = np.maximum(x - PQ_KNEE, 0)
    rolled = PQ_KNEE + (1 - PQ_KNEE) * (1 - np.exp(-over / (1 - PQ_KNEE)))
    y = np.where(x <= PQ_KNEE, x, rolled)
    sdr = np.where(y < 0.018, 4.5 * y, 1.099 * y**0.45 - 0.099)
    return (nits / PQ_WHITE_NITS).astype(np.float32), (
        np.clip(sdr, 0, 1) * 255 + 0.5
    ).astype(np.uint8)


_PQ_LINEAR, _PQ_SDR = _pq_tables()


def _pq_mapped(rgb: np.ndarray) -> np.ndarray:
    """pq_to_sdr worked out from the formula (slow: for building the cube)."""
    linear = _PQ_LINEAR[rgb] @ P3_TO_709.T
    idx = np.sqrt(np.clip(linear, 0, _PQ_TOP) / _PQ_TOP) * (_PQ_STEPS - 1)
    return _PQ_SDR[idx.astype(np.int32)]


@functools.cache
def _pq_cube() -> np.ndarray:
    """_pq_mapped of every 8-bit RGB colour, indexed by r << 16 | g << 8 | b
    (48 MB; a frame is then one lookup instead of float maths per pixel)."""
    levels = np.arange(256, dtype=np.uint8)
    g, b = np.meshgrid(levels, levels, indexing="ij")
    plane = np.stack([np.zeros_like(g), g, b], axis=-1).reshape(-1, 3)
    cube = np.empty((256, 256 * 256, 3), dtype=np.uint8)
    for r in range(256):
        plane[:, 0] = r
        cube[r] = _pq_mapped(plane)
    return cube.reshape(-1, 3)


def pq_to_sdr(rgb: np.ndarray) -> np.ndarray:
    """An 8-bit RGB picture coded in PQ with P3 primaries, as SDR BT.709."""
    code = rgb.astype(np.int32)
    return _pq_cube()[code[..., 0] << 16 | code[..., 1] << 8 | code[..., 2]]


@functools.cache
def is_pq(path: Path) -> bool:
    """Whether ``path``'s picture is coded in PQ: as its stream says, or,
    when it says nothing, by its highlights (PQ_BRIGHT_SHARE). HLG is not
    supported and raises."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        trc = stream.codec_context.color_trc
        if trc == _TRC_HLG:
            raise ValueError(f"{path.name}: HLG pictures are not supported")
        if trc != _TRC_UNSPECIFIED:
            return trc == _TRC_PQ
        duration = float(container.duration / av.time_base)
        # The key frame at or before each probe time: no frames in between
        # are decoded.
        stream.codec_context.skip_frame = "NONKEY"
        bright = 0
        for k in range(PQ_PROBE_FRAMES):
            t = duration * (k + 0.5) / PQ_PROBE_FRAMES
            container.seek(int(t / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                if frame.time is not None:
                    rgb = frame.reformat(width=320, height=180, format="rgb24")
                    top = np.percentile(rgb.to_ndarray().max(axis=2), 99.9)
                    bright += top >= PQ_BRIGHT_LEVEL
                    break
    return bright < PQ_BRIGHT_SHARE * PQ_PROBE_FRAMES


# Analysis of footage larger than this reads a proxy of this size and rate
# (make_proxy): 4K is decoded once instead of once per analysis pass.
PROXY_SIZE = (1280, 720)
PROXY_FPS = 30


def make_proxy(src: Path, dst: Path) -> None:
    """``src``'s picture at most PROXY_SIZE (its own aspect) and PROXY_FPS,
    no sound, with the same timestamps; written to a temporary file and
    moved into place, so a stopped run leaves no half proxy."""
    tmp = dst.with_suffix(".tmp.mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(src)) as inp, av.open(str(tmp), mode="w") as out:
        stream = inp.streams.video[0]
        stream.thread_type = "AUTO"
        ctx = stream.codec_context
        w, h = fit_size(ctx.width, ctx.height, PROXY_SIZE)
        video = out.add_stream("libx264", rate=PROXY_FPS)
        video.width, video.height = w, h
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "23", "preset": "veryfast"}
        video.codec_context.time_base = Fraction(1, PROXY_FPS)
        last = -1
        for frame in inp.decode(stream):
            if frame.time is None:
                continue
            pts = round(float(frame.time) * PROXY_FPS)
            if pts <= last:
                continue
            last = pts
            small = frame.reformat(width=w, height=h, format="yuv420p")
            small.pts, small.time_base = pts, Fraction(1, PROXY_FPS)
            out.mux(video.encode(small))
        out.mux(video.encode())
    tmp.replace(dst)


def fit_size(width: int, height: int, frame: tuple[int, int]) -> tuple[int, int]:
    """The largest even size with the aspect of ``width`` x ``height`` that
    fits in ``frame``."""
    scale = min(frame[0] / width, frame[1] / height)
    return (
        max(2, round(width * scale / 2) * 2),
        max(2, round(height * scale / 2) * 2),
    )


def edit_size(sources: list[tuple[int, int, float]]) -> tuple[int, int]:
    """EDIT_SIZE, or the same turned upright when portrait sources fill more
    of the cut; ``sources`` are (width, height, seconds in the cut)."""
    upright = sum(s for w, h, s in sources if h > w)
    wide = sum(s for w, h, s in sources if h <= w)
    return (EDIT_SIZE[1], EDIT_SIZE[0]) if upright > wide else EDIT_SIZE


def _placed(
    frame: av.VideoFrame, size: tuple[int, int], pq: bool = False
) -> np.ndarray:
    """``frame`` scaled to fit ``size`` with its own aspect, centred on
    black; a ``pq`` picture is shown as SDR (pq_to_sdr)."""
    w, h = fit_size(frame.width, frame.height, size)
    picture = frame.reformat(width=w, height=h, format="rgb24").to_ndarray()
    if pq:
        picture = pq_to_sdr(picture)
    if (w, h) == size:
        return picture
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    x, y = (size[0] - w) // 2, (size[1] - h) // 2
    canvas[y : y + h, x : x + w] = picture
    return canvas


def iter_clip(
    path: Path,
    start: float,
    duration: float,
    size: tuple[int, int] = OUT_SIZE,
    pq: bool = False,
) -> Iterator[np.ndarray]:
    """[start, start+duration) of ``path`` as OUT_FPS RGB frames of
    ``size``, one at a time (a picture of another aspect is fitted inside,
    on black; ``pq`` pictures are shown as SDR); raises at the end when the
    video ran out of frames."""
    n_out = round(duration * OUT_FPS)
    targets = [start + i / OUT_FPS for i in range(n_out)]
    sent = 0
    # The decoded frame on hold and its picture, placed only once a target
    # shows it: frames before the start or between targets are skipped.
    held: av.VideoFrame | None = None
    last: np.ndarray | None = None
    ended = False
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        video_end = (
            float(container.duration / av.time_base) if container.duration else 0.0
        )
        container.seek(int(max(0.0, start - 1.0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = float(frame.time)
            # Sample-and-hold: every target before this frame shows the last one.
            while held is not None and sent < n_out and targets[sent] < t:
                if last is None:
                    last = _placed(held, size, pq)
                yield last
                sent += 1
            if sent >= n_out:
                break
            held, last = frame, None
        else:
            ended = True
    # A clip that runs to the end of the video: the last frame stays on screen
    # until the video ends (there is no later frame to end its hold).
    while (
        ended and held is not None and sent < n_out and targets[sent] < video_end + 1e-3
    ):
        if last is None:
            last = _placed(held, size, pq)
        yield last
        sent += 1
    if sent < n_out:
        raise ValueError(
            f"{path.name}: needed {n_out} frames from {start}s, got {sent}"
        )


def read_clip(
    path: Path,
    start: float,
    duration: float,
    size: tuple[int, int] = OUT_SIZE,
    pq: bool = False,
) -> list[np.ndarray]:
    """iter_clip, all frames at once."""
    return list(iter_clip(path, start, duration, size, pq))


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


def read_audio_clip(
    path: Path,
    start: float,
    duration: float,
    fade_in: bool = True,
    fade_out: bool = True,
) -> np.ndarray:
    """[start, start+duration) of ``path``'s first audio stream as stereo
    float32 at AUDIO_RATE, shape (2, n); silence where the file has no sound
    (no audio stream, or it ends early). Each end fades over AUDIO_FADE_SEC
    unless told not to (an end that is crossfaded instead).
    Samples are placed by the first decoded frame's time and counted from
    there, so the clip is off by at most the resampler's delay."""
    n = round(duration * AUDIO_RATE)
    out = np.zeros((2, n), np.float32)
    base = round(start * AUDIO_RATE)
    with av.open(str(path)) as container:
        if container.streams.audio:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(
                format="fltp", layout="stereo", rate=AUDIO_RATE
            )
            container.seek(int(max(0.0, start - 1.0) / stream.time_base), stream=stream)
            at: int | None = None
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                if float(frame.time) >= start + duration:
                    break
                if at is None:
                    at = round(float(frame.time) * AUDIO_RATE) - base
                for chunk in resampler.resample(frame):
                    at = _place(out, chunk.to_ndarray(), at)
            if at is not None:
                for chunk in resampler.resample(None):
                    at = _place(out, chunk.to_ndarray(), at)
    fade = min(round(AUDIO_FADE_SEC * AUDIO_RATE), n // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if fade_in:
            out[:, :fade] *= ramp
        if fade_out:
            out[:, n - fade :] *= ramp[::-1]
    return out


def crossfade(carry: np.ndarray, sound: np.ndarray) -> np.ndarray:
    """``sound`` with ``carry`` (what the clip before it sounds like just
    past its end) fading out over its start while it fades in, at equal
    power, so a cut does not dip or click."""
    out = sound.copy()
    m = min(carry.shape[1], out.shape[1])
    t = np.linspace(0.0, 1.0, m, dtype=np.float32) * (np.pi / 2)
    out[:, :m] = carry[:, :m] * np.cos(t) + out[:, :m] * np.sin(t)
    return out


def _place(out: np.ndarray, samples: np.ndarray, at: int) -> int:
    """Copy ``samples`` into ``out`` from index ``at`` (may lie outside it);
    returns the index after them."""
    lo, hi = max(at, 0), min(at + samples.shape[1], out.shape[1])
    if lo < hi:
        out[:, lo:hi] = samples[:, lo - at : hi - at]
    return at + samples.shape[1]


def voice_track(mono: np.ndarray, rate: int) -> np.ndarray:
    """A narration line (mono at ``rate``) as stereo at AUDIO_RATE, brought
    to NARRATION_RMS without peaks above NARRATION_PEAK, edges faded."""
    if mono.size == 0 or not np.any(mono):
        raise ValueError("Narration line is silent")
    n = round(mono.size * AUDIO_RATE / rate)
    x = np.interp(np.arange(n) * rate / AUDIO_RATE, np.arange(mono.size), mono).astype(
        np.float32
    )
    x *= NARRATION_RMS / float(np.sqrt(np.mean(x**2)))
    peak = float(np.abs(x).max())
    if peak > NARRATION_PEAK:
        x *= NARRATION_PEAK / peak
    fade = min(round(AUDIO_FADE_SEC * AUDIO_RATE), n // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        x[:fade] *= ramp
        x[n - fade :] *= ramp[::-1]
    return np.stack([x, x])


def duck_and_mix(
    samples: np.ndarray, at: int, narration: list[tuple[float, np.ndarray]]
) -> np.ndarray:
    """``samples`` (stereo, starting at sample ``at`` of the output) with
    the original turned down under each narration line and the lines added.
    ``narration`` is (start seconds, stereo samples from voice_track)."""
    n = samples.shape[1]
    idx = np.arange(at, at + n)
    gain = np.ones(n, np.float32)
    voices = np.zeros_like(samples)
    ramp = max(round(DUCK_RAMP_SEC * AUDIO_RATE), 1)
    for start, voice in narration:
        s0 = round(start * AUDIO_RATE)
        s1 = s0 + voice.shape[1]
        if s1 + ramp <= at or s0 - ramp >= at + n:
            continue
        down = np.clip((s0 - idx) / ramp, 0.0, 1.0)  # 1 before, 0 from s0
        up = np.clip((idx - s1) / ramp, 0.0, 1.0)  # 0 until s1, 1 after
        env = DUCK_GAIN + (1 - DUCK_GAIN) * np.maximum(down, up)
        gain = np.minimum(gain, env.astype(np.float32))
        lo, hi = max(s0, at), min(s1, at + n)
        if lo < hi:
            voices[:, lo - at : hi - at] += voice[:, lo - s0 : hi - s0]
    return samples * gain + voices


def dissolve(
    tail: tuple[list[np.ndarray], np.ndarray],
    head: tuple[Iterable[np.ndarray], np.ndarray],
) -> tuple[Iterator[np.ndarray], np.ndarray]:
    """``head`` (frames and sound of a clip) with ``tail`` (what the clip
    before it shows and sounds like just past its end) laid over its start
    and fading out: the two dissolve into each other without changing the
    clip's length. The head's frames are blended as they are read."""
    sound = head[1].copy()
    n = len(tail[0])

    def frames() -> Iterator[np.ndarray]:
        for k, frame in enumerate(head[0]):
            if k < n:
                a = (k + 1) / (n + 1)
                mixed = tail[0][k].astype(np.float32) * (1 - a) + frame * a
                frame = mixed.astype(np.uint8)
            yield frame

    m = min(tail[1].shape[1], sound.shape[1])
    ramp = np.linspace(0.0, 1.0, m, dtype=np.float32)
    sound[:, :m] = tail[1][:, :m] * (1 - ramp) + sound[:, :m] * ramp
    return frames(), sound


def edge_gain(at: int, n: int, total: int, fade_in: int, fade_out: int) -> np.ndarray:
    """Gain for samples (or frames) at..at+n of ``total``: rising from 0
    over the first ``fade_in``, falling to 0 over the last ``fade_out``."""
    idx = np.arange(at, at + n, dtype=np.float32)
    rise = np.clip((idx + 1) / max(fade_in, 1), 0.0, 1.0)
    fall = np.clip((total - idx) / max(fade_out, 1), 0.0, 1.0)
    return np.minimum(rise, fall)


def write_av(
    clips: Iterable[tuple[Iterable[np.ndarray], np.ndarray]],
    outputs: list[tuple[Path, FrameDraw | None]],
    narration: list[tuple[float, np.ndarray]] | None = None,
    size: tuple[int, int] = OUT_SIZE,
    total_sec: float | None = None,
    crf: int = PREVIEW_CRF,
) -> None:
    """Clips (OUT_FPS RGB frames of ``size``, and their sound from
    read_audio_clip) back to back into each output mp4, H.264 + AAC, in
    one pass over the clips; an output's draw changes each frame before it
    is encoded. ``narration`` lines are mixed over the sound
    (duck_and_mix). With ``total_sec`` (the cut's length) picture and sound
    fade in from black and out to it (FADE_IN_SEC, FADE_OUT_SEC)."""
    with ExitStack() as stack:
        sinks = []
        for path, draw in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            container = stack.enter_context(av.open(str(path), mode="w"))
            video = container.add_stream("libx264", rate=Fraction(OUT_FPS))
            video.width, video.height = size
            video.pix_fmt = "yuv420p"
            video.options = {"crf": str(crf), "preset": "veryfast"}
            audio = container.add_stream("aac", rate=AUDIO_RATE)
            audio.layout = "stereo"
            audio.bit_rate = AUDIO_BITRATE
            sinks.append((container, video, audio, draw))
        n_frames = n_samples = 0
        pending = np.zeros((2, 0), np.float32)

        def encode_audio(samples: np.ndarray) -> None:
            nonlocal n_samples
            for container, _, audio, _ in sinks:
                frame = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(samples), format="fltp", layout="stereo"
                )
                frame.sample_rate = AUDIO_RATE
                frame.pts, frame.time_base = n_samples, Fraction(1, AUDIO_RATE)
                container.mux(audio.encode(frame))
            n_samples += samples.shape[1]

        last_frame = round(total_sec * OUT_FPS) if total_sec else 0
        last_sample = round(total_sec * AUDIO_RATE) if total_sec else 0
        for frames, samples in clips:
            for arr in frames:
                if total_sec:
                    gain = edge_gain(
                        n_frames,
                        1,
                        last_frame,
                        round(FADE_IN_SEC * OUT_FPS),
                        round(FADE_OUT_SEC * OUT_FPS),
                    )[0]
                    if gain < 1:
                        arr = (arr * gain).astype(np.uint8)
                for container, video, _, draw in sinks:
                    shown = arr if draw is None else draw(n_frames, arr)
                    frame = av.VideoFrame.from_ndarray(shown, format="rgb24")
                    container.mux(video.encode(frame))
                n_frames += 1
            at = n_samples + pending.shape[1]
            if narration:
                samples = duck_and_mix(samples, at, narration)
            if total_sec:
                samples = samples * edge_gain(
                    at,
                    samples.shape[1],
                    last_sample,
                    round(FADE_IN_SEC * AUDIO_RATE),
                    round(FADE_OUT_SEC * AUDIO_RATE),
                )
            pending = np.concatenate([pending, samples], axis=1)
            whole = pending.shape[1] - pending.shape[1] % AAC_FRAME
            for k in range(0, whole, AAC_FRAME):
                encode_audio(pending[:, k : k + AAC_FRAME])
            pending = pending[:, whole:]
        if pending.shape[1]:
            encode_audio(pending)
        for container, video, audio, _ in sinks:
            container.mux(video.encode())
            container.mux(audio.encode())


class SubtitlePainter:
    """Draws timed lines of text near the bottom of ``frame``-sized frames,
    white with a dark outline on a translucent dark box, sized for OUT_SIZE
    and scaled with the frame height. ``cues`` are (start, end, text) in
    seconds of the output, in time order; where two overlap the later one
    shows."""

    def __init__(
        self,
        cues: list[tuple[float, float, str]],
        font: Path,
        size: int = SUBTITLE_SIZE,
        frame: tuple[int, int] = OUT_SIZE,
    ) -> None:
        self._cues = cues
        self._starts = [c[0] for c in cues]
        self._frame = frame
        # Sized by the shorter side, so text keeps its size on upright frames.
        self._scale = min(frame) / OUT_SIZE[1]
        self._font = ImageFont.truetype(str(font), round(size * self._scale))
        self._drawn: dict[int, tuple[np.ndarray, np.ndarray, int, int]] = {}

    def __call__(self, n: int, frame: np.ndarray) -> np.ndarray:
        t = n / OUT_FPS
        k = bisect_right(self._starts, t) - 1
        if k < 0 or t >= self._cues[k][1]:
            return frame
        rgb, alpha, x, y = self._text(k)
        out = frame.copy()
        h, w = alpha.shape[:2]
        region = out[y : y + h, x : x + w].astype(np.float32)
        blended = region * (1 - alpha) + rgb * alpha
        out[y : y + h, x : x + w] = blended.astype(np.uint8)
        return out

    def _text(self, k: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        if k not in self._drawn:
            text = self._cues[k][2]
            stroke = round(SUBTITLE_STROKE * self._scale)
            box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox(
                (0, 0), text, font=self._font, stroke_width=stroke
            )
            pad = round(SUBTITLE_PAD * self._scale)
            w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
            if w > self._frame[0]:
                raise ValueError(f"Subtitle wider than the frame: {text!r}")
            image = Image.new("RGBA", (w, h), (0, 0, 0, SUBTITLE_BOX_ALPHA))
            ImageDraw.Draw(image).text(
                (pad - box[0], pad - box[1]),
                text,
                font=self._font,
                fill=(255, 255, 255, 255),
                stroke_width=stroke,
                stroke_fill=(0, 0, 0, 255),
            )
            arr = np.asarray(image, dtype=np.float32)
            x = (self._frame[0] - w) // 2
            y = self._frame[1] - round(SUBTITLE_MARGIN * self._scale) - h
            self._drawn[k] = (arr[..., :3], arr[..., 3:] / 255.0, x, y)
        return self._drawn[k]


class TitlePainter:
    """A title card over the opening of a cut: ``title`` large and ``line``
    smaller under it, centred on ``frame``-sized frames, shown from
    ``start`` to ``end`` seconds and fading in and out over TITLE_FADE_SEC.
    A title too wide for the frame is drawn smaller."""

    def __init__(
        self,
        title: str,
        line: str,
        font: Path,
        start: float,
        end: float,
        frame: tuple[int, int] = OUT_SIZE,
    ) -> None:
        self._start, self._end = start, end
        scale = min(frame) / OUT_SIZE[1]
        size = round(TITLE_SIZE * scale)
        big = ImageFont.truetype(str(font), size)
        while big.getlength(title) > 0.9 * frame[0] and size > 12:
            size -= 2
            big = ImageFont.truetype(str(font), size)
        small = ImageFont.truetype(str(font), round(TITLE_SUB_SIZE * scale))
        image = Image.new("RGBA", frame, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        stroke = max(1, round(SUBTITLE_STROKE * scale))
        gap = round(12 * scale)
        h_big = big.getbbox(title)[3]
        h_small = small.getbbox(line)[3] if line else 0
        y = (frame[1] - h_big - (gap + h_small if line else 0)) // 2
        for text, font_, top in ((title, big, y), (line, small, y + h_big + gap)):
            if text:
                draw.text(
                    ((frame[0] - font_.getlength(text)) / 2, top),
                    text,
                    font=font_,
                    fill=(255, 255, 255, 255),
                    stroke_width=stroke,
                    stroke_fill=(0, 0, 0, 200),
                )
        arr = np.asarray(image, dtype=np.float32)
        rows = np.flatnonzero(arr[..., 3].any(axis=1))
        self._top, self._bottom = int(rows[0]), int(rows[-1]) + 1
        self._rgb = arr[self._top : self._bottom, :, :3]
        self._alpha = arr[self._top : self._bottom, :, 3:] / 255.0

    def __call__(self, n: int, frame: np.ndarray) -> np.ndarray:
        t = n / OUT_FPS
        if not self._start <= t < self._end:
            return frame
        fade = min(t - self._start, self._end - t) / TITLE_FADE_SEC
        alpha = self._alpha * min(1.0, fade)
        out = frame.copy()
        region = out[self._top : self._bottom].astype(np.float32)
        out[self._top : self._bottom] = (
            region * (1 - alpha) + self._rgb * alpha
        ).astype(np.uint8)
        return out


def compose(*draws: FrameDraw | None) -> FrameDraw | None:
    """One FrameDraw applying ``draws`` in order (None ones skipped)."""
    used = [d for d in draws if d is not None]
    if not used:
        return None

    def draw(n: int, frame: np.ndarray) -> np.ndarray:
        for d in used:
            frame = d(n, frame)
        return frame

    return draw


# Scene material keeps the source's look: near-lossless H.264 and AAC audio.
CUT_CRF = "18"
CUT_AUDIO_BITRATE = 192_000


def cut_clip(path: Path, spans: list[tuple[float, float]], out_path: Path) -> None:
    """Re-encode the ``spans`` ([start, end) seconds, in order) of ``path``
    back to back into one H.264 + AAC mp4, at the source's own size and frame
    rate, with its first audio track. The source file is only read.

    Re-encoding makes every cut frame-accurate (copying packets could only cut
    at keyframes). Audio frames are kept whole and realigned at the start of
    each span, so audio edges are off by at most one frame (about 20 ms) and
    the error does not add up over spans. Written to a ``.partial`` file
    first, so an interrupted run never leaves a clip that looks finished.
    """
    if not spans:
        raise ValueError(f"No spans to cut from {path.name}")
    for (start, end), nxt in zip(spans, [*spans[1:], None], strict=True):
        if end <= start:
            raise ValueError(f"Empty span {start}-{end} of {path.name}")
        if nxt is not None and nxt[0] < end:
            raise ValueError(f"Spans out of order or overlapping: {spans}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.stem + ".partial.mp4")
    with av.open(str(path)) as src, av.open(str(partial), mode="w") as dst:
        vin = src.streams.video[0]
        rate = vin.average_rate or vin.guessed_rate
        if not rate:
            raise ValueError(f"{path.name}: no frame rate")
        vout = dst.add_stream("libx264", rate=rate)
        vout.width = vin.codec_context.width
        vout.height = vin.codec_context.height
        vout.pix_fmt = "yuv420p"
        vout.options = {"crf": CUT_CRF, "preset": "veryfast"}
        tick = Fraction(1) / Fraction(rate)
        ain = src.streams.audio[0] if src.streams.audio else None
        streams = [vin]
        if ain is not None:
            aout = dst.add_stream("aac", rate=ain.rate)
            aout.layout = ain.layout.name
            aout.bit_rate = CUT_AUDIO_BITRATE
            resampler = av.AudioResampler(
                format="fltp", layout=ain.layout.name, rate=ain.rate
            )
            streams.append(ain)
        last_pts, samples, offset = -1, 0, 0.0
        for start, end in spans:
            first_pts = last_pts
            if ain is not None:
                samples = max(samples, round(offset * ain.rate))
            src.seek(int(max(0.0, start - 1.0) / vin.time_base), stream=vin)
            done = {s.index: False for s in streams}
            for packet in src.demux(*streams):
                if all(done.values()):
                    break
                for frame in packet.decode():
                    if frame.time is None:
                        continue
                    t = float(frame.time)
                    if t >= end:
                        done[packet.stream.index] = True
                        continue
                    if t < start:
                        continue
                    if packet.stream is vin:
                        pts = round((offset + t - start) / tick)
                        # Variable frame rate: two source frames may land on
                        # one output tick; keep the first.
                        if pts <= last_pts:
                            continue
                        out = frame.reformat(format="yuv420p")
                        out.pts, out.time_base, last_pts = pts, tick, pts
                        dst.mux(vout.encode(out))
                    else:
                        for af in resampler.resample(frame):
                            af.pts, af.time_base = samples, Fraction(1, ain.rate)
                            samples += af.samples
                            dst.mux(aout.encode(af))
            if last_pts == first_pts:
                raise ValueError(f"{path.name}: no video frames in {start}-{end}")
            offset += end - start
        dst.mux(vout.encode())
        if ain is not None:
            dst.mux(aout.encode())
    partial.replace(out_path)
