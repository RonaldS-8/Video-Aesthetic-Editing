"""End-to-end check on a tiny generated video: encode, sample, detect cuts."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from video_demo import media
from video_demo.media import (
    AUDIO_RATE,
    EDIT_SIZE,
    FADE_IN_SEC,
    FADE_OUT_SEC,
    OUT_FPS,
    OUT_SIZE,
    SubtitlePainter,
    TitlePainter,
    compose,
    count_frames,
    crossfade,
    cut_clip,
    dissolve,
    edge_gain,
    edit_size,
    fit_size,
    is_pq,
    make_proxy,
    pq_to_sdr,
    probe,
    read_audio_clip,
    read_clip,
    sample_frames,
    segment_frames_every,
    segment_frames_picked,
    write_av,
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


def _tone_video(path: Path, seconds: int = 2) -> None:
    frames = [np.full((OUT_SIZE[1], OUT_SIZE[0], 3), 40, np.uint8)] * (
        seconds * OUT_FPS
    )
    t = np.arange(seconds * AUDIO_RATE) / AUDIO_RATE
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    write_av([(frames, np.stack([tone, tone]))], [(path, None)])


def test_previews_carry_sound_and_clips_fade(tmp_path: Path) -> None:
    video = tmp_path / "tone.mp4"
    _tone_video(video)
    info = probe(video)
    assert info.has_audio and abs(info.duration - 2.0) < 0.1
    sound = read_audio_clip(video, 0.5, 1.0)
    assert sound.shape == (2, AUDIO_RATE)
    assert sound[:, 0].max() == 0.0 and sound[:, -1].max() == 0.0
    assert 0.3 < np.abs(sound[:, AUDIO_RATE // 2 :][:, :2000]).max() < 0.6


def test_audio_of_a_silent_video_is_silence(tmp_path: Path) -> None:
    video = tmp_path / "blocks.mp4"
    _block_video(video)
    assert not read_audio_clip(video, 1.0, 0.5).any()


FONT = Path("C:/Windows/Fonts/msyh.ttc")


@pytest.mark.skipif(not FONT.exists(), reason="needs a CJK font")
def test_burned_subtitles_only_while_the_cue_lasts(tmp_path: Path) -> None:
    painter = SubtitlePainter([(1.0, 2.0, "你好 world")], FONT)
    frame = np.zeros((OUT_SIZE[1], OUT_SIZE[0], 3), np.uint8)
    assert painter(0, frame) is frame
    drawn = painter(OUT_FPS + 1, frame)
    assert drawn.any() and not frame.any()
    assert not drawn[: OUT_SIZE[1] // 2].any()  # near the bottom only
    assert painter(2 * OUT_FPS, frame) is frame
    with pytest.raises(ValueError, match="wider"):
        SubtitlePainter([(0.0, 1.0, "长" * 60)], FONT)(0, frame)


def test_dissolve_blends_the_tail_into_the_head_keeping_its_length() -> None:
    tail = ([np.full((2, 2, 3), 200, np.uint8)] * 3, np.full((2, 4), 1.0, np.float32))
    head = ([np.zeros((2, 2, 3), np.uint8)] * 5, np.zeros((2, 6), np.float32))
    blended, sound = dissolve(tail, head)
    frames = list(blended)
    assert len(frames) == 5 and sound.shape == (2, 6)
    levels = [int(f[0, 0, 0]) for f in frames]
    assert levels == [150, 100, 50, 0, 0]  # tail fades out over 3 frames
    assert sound[0].tolist() == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0, 0.0, 0.0])


def test_edge_gain_fades_both_ends() -> None:
    g = edge_gain(0, 10, 10, 2, 4)
    assert g.tolist() == pytest.approx([0.5, 1, 1, 1, 1, 1, 1, 0.75, 0.5, 0.25])
    assert edge_gain(4, 2, 10, 2, 4).tolist() == [1.0, 1.0]


def test_finished_cut_fades_from_and_to_black(tmp_path: Path) -> None:
    n = 2 * OUT_FPS
    frames = [np.full((EDIT_SIZE[1], EDIT_SIZE[0], 3), 200, np.uint8)] * n
    tone = np.full(2 * AUDIO_RATE, 0.5, np.float32)
    path = tmp_path / "cut.mp4"
    write_av([(frames, np.stack([tone, tone]))], [(path, None)], None, EDIT_SIZE, 2.0)
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        assert (v.width, v.height) == EDIT_SIZE
        shot = [f.to_ndarray(format="rgb24").mean() for f in c.decode(v)]
    # Black at both ends; full between the fade in (0.5 s) and out (from 0.8 s).
    assert shot[0] < 30 and shot[-1] < 30
    full = round(FADE_IN_SEC * OUT_FPS), n - round(FADE_OUT_SEC * OUT_FPS)
    assert all(x > 190 for x in shot[full[0] : full[1]])


@pytest.mark.skipif(not FONT.exists(), reason="needs a CJK font")
def test_subtitles_and_title_scale_with_the_frame() -> None:
    small = SubtitlePainter([(0.0, 1.0, "你好")], FONT)
    large = SubtitlePainter([(0.0, 1.0, "你好")], FONT, frame=EDIT_SIZE)
    a = small(0, np.zeros((OUT_SIZE[1], OUT_SIZE[0], 3), np.uint8))
    b = large(0, np.zeros((EDIT_SIZE[1], EDIT_SIZE[0], 3), np.uint8))
    rows_a = np.flatnonzero(a.any(axis=(1, 2)))
    rows_b = np.flatnonzero(b.any(axis=(1, 2)))
    assert (rows_b[-1] - rows_b[0]) / (rows_a[-1] - rows_a[0]) == pytest.approx(
        EDIT_SIZE[1] / OUT_SIZE[1], rel=0.1
    )
    card = TitlePainter("爱丁堡城市漫游", "城市漫步", FONT, 0.5, 3.5, EDIT_SIZE)
    frame = np.zeros((EDIT_SIZE[1], EDIT_SIZE[0], 3), np.uint8)
    assert card(0, frame) is frame  # before it starts
    mid = card(2 * OUT_FPS, frame)
    rows = np.flatnonzero(mid.any(axis=(1, 2)))
    assert EDIT_SIZE[1] * 0.3 < rows.mean() < EDIT_SIZE[1] * 0.7  # centred
    faded = card(round(0.6 * OUT_FPS), frame)  # 0.1 s into its fade-in
    assert faded.max() < mid.max()
    both = compose(card, large)
    assert both is not None and both(2 * OUT_FPS, frame).any()
    assert compose(None, None) is None


def test_fit_size_keeps_the_aspect_inside_the_frame() -> None:
    assert fit_size(1920, 1080, EDIT_SIZE) == EDIT_SIZE
    assert fit_size(1080, 1920, (1280, 720)) == (404, 720)
    assert fit_size(3840, 1600, (1280, 720)) == (1280, 534)
    assert fit_size(3840, 2160, (1920, 1080)) == (1920, 1080)


def test_edit_size_turns_upright_when_portrait_fills_more() -> None:
    assert edit_size([(1920, 1080, 30.0), (1080, 1920, 20.0)]) == EDIT_SIZE
    upright = (EDIT_SIZE[1], EDIT_SIZE[0])
    assert edit_size([(1920, 1080, 10.0), (1080, 1920, 20.0)]) == upright


def test_read_clip_puts_a_portrait_picture_on_black(tmp_path: Path) -> None:
    path = tmp_path / "upright.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=OUT_FPS)
        stream.width, stream.height = 240, 480
        stream.pix_fmt = "yuv420p"
        for _ in range(OUT_FPS):
            grey = np.full((480, 240, 3), 200, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(grey)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frame = read_clip(path, 0.0, 0.5, OUT_SIZE)[0]
    assert frame.shape == (OUT_SIZE[1], OUT_SIZE[0], 3)
    middle = frame[:, OUT_SIZE[0] // 2]
    assert middle.mean() > 150  # the picture, scaled to the full height
    assert frame[:, :100].max() == 0 and frame[:, -100:].max() == 0  # black sides


def test_pq_to_sdr_keeps_black_and_rolls_off_highlights() -> None:
    grey = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], 3, axis=2)
    out = pq_to_sdr(grey)[:, 0, 0].astype(int)
    assert out[0] == 0 and out[-1] == 255
    assert (np.diff(out) >= 0).all()  # a grey ramp stays a ramp
    # PQ code 130 is about 100 nits, the white the cut is graded to: near
    # SDR white, not the 51 % grey it shows as when read as SDR.
    assert 225 <= out[130] <= 245


def test_pq_to_sdr_lookup_matches_the_formula() -> None:
    colours = np.random.default_rng(0).integers(0, 256, (64, 64, 3), np.uint8)
    assert np.array_equal(pq_to_sdr(colours), media._pq_mapped(colours))


def _flat_video(path: Path, level: int) -> None:
    frames = [np.full((OUT_SIZE[1], OUT_SIZE[0], 3), level, np.uint8)] * OUT_FPS * 2
    write_video(frames, path)


def test_is_pq_reads_unflagged_highlights(tmp_path: Path) -> None:
    dim, bright = tmp_path / "dim.mp4", tmp_path / "bright.mp4"
    _flat_video(dim, 130)  # never near white: PQ-coded picture
    _flat_video(bright, 235)
    assert is_pq(dim) and not is_pq(bright)


def test_crossfade_keeps_a_steady_sound_steady() -> None:
    carry = np.full((2, 100), 0.5, np.float32)
    sound = np.full((2, 300), 0.5, np.float32)
    out = crossfade(carry, sound)
    assert out.shape == sound.shape
    # Equal power: two uncorrelated halves keep the level; equal ones
    # (the same tone) rise at most by sqrt(2) in the middle, never dip.
    assert out[0, 0] == pytest.approx(0.5) and out[0, -1] == pytest.approx(0.5)
    assert out.min() >= 0.5 - 1e-6 and out.max() <= 0.5 * np.sqrt(2) + 1e-6


def test_proxy_is_smaller_with_the_same_times(tmp_path: Path) -> None:
    src, proxy = tmp_path / "big.mp4", tmp_path / "proxy" / "p.mp4"
    with av.open(str(src), mode="w") as container:
        stream = container.add_stream("libx264", rate=50)
        stream.width, stream.height = 1920, 1080
        stream.pix_fmt = "yuv420p"
        for k in range(100):  # 2 s at 50 fps, a new grey level each 0.5 s
            grey = np.full((1080, 1920, 3), 40 + 40 * (k // 25), np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(grey)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    make_proxy(src, proxy)
    info = probe(proxy)
    assert (info.width, info.height) == (1280, 720) and not info.has_audio
    assert info.duration == pytest.approx(2.0, abs=0.05)
    # The same moment shows the same picture in both.
    late = read_clip(proxy, 1.6, 0.1, (64, 36))[0].mean()
    assert late == pytest.approx(read_clip(src, 1.6, 0.1, (64, 36))[0].mean(), abs=3)
    assert not proxy.with_suffix(".tmp.mp4").exists()
