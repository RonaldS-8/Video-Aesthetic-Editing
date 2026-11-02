"""Narration: free slots between speech, line checks, voicing, ducking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from video_demo import category, cli, media, narrate
from video_demo.narrate import Slot
from video_demo.outline import OutlineError

CLIPS = [
    {"video": "v", "index": 0, "src_start": 0.0, "duration": 6.0, "role": "开场"},
    {"video": "v", "index": 1, "src_start": 20.0, "duration": 5.0, "role": "街巷"},
    {"video": "w", "index": 0, "src_start": 0.0, "duration": 8.0, "role": "收尾"},
]


def test_speech_is_placed_on_the_cut_clock() -> None:
    speech = {"v": [(1.0, 2.0), (19.0, 21.5), (40.0, 41.0)], "w": [(7.5, 9.0)]}
    assert narrate.speech_on_timeline(CLIPS, speech) == [
        (1.0, 2.0),
        (6.0, 7.5),
        (18.5, 19.0),
    ]


def test_slots_skip_speech_and_stop_before_it() -> None:
    spoken = [(1.0, 2.0), (6.0, 7.5), (18.5, 19.0)]
    slots = narrate.free_slots(CLIPS, spoken)
    # Clip 1 starts at 0.3 s with speech at 1.0: 0.5 s free, too short.
    # Clip 2 starts inside speech (6.3 s). Clip 3 at 11.3 s, speech at 18.5.
    assert slots == [Slot(3, 11.3, pytest.approx(7.0))]
    assert narrate.free_slots(CLIPS, []) == [
        Slot(1, 0.3, 18.5),
        Slot(2, 6.3, 12.5),
        Slot(3, 11.3, 7.5),
    ]


SLOTS = [Slot(1, 0.3, 18.5), Slot(2, 6.3, 12.5), Slot(3, 11.3, 7.5)]
TOTAL = sum(c["duration"] for c in CLIPS)


def test_lines_are_sized_to_the_time_before_the_next_line() -> None:
    lines = narrate.validate_lines(
        {"lines": [{"clip": 1, "text": "古城的清晨"}, {"clip": 3, "text": "再见。"}]},
        SLOTS,
        1,
        3,
    )
    assert lines[0] == {"clip": 1, "start": 0.3, "window": 10.8, "text": "古城的清晨"}
    assert lines[1]["window"] == 7.5
    # Clip 3's own 7.5 s holds 28 characters; 29 are too many.
    with pytest.raises(OutlineError, match="29 characters, 28 fit"):
        narrate.validate_lines(
            {"lines": [{"clip": 1, "text": "好"}, {"clip": 3, "text": "长" * 29}]},
            SLOTS,
            1,
            3,
        )
    # 25 characters with 5.8 s before the next line pass here: the window
    # goes on to voice_lines, which shortens the line if its voice runs over.
    long = narrate.validate_lines(
        {"lines": [{"clip": 1, "text": "长" * 25}, {"clip": 2, "text": "好"}]},
        SLOTS,
        1,
        3,
    )
    assert long[0]["window"] == pytest.approx(5.8)


@pytest.mark.parametrize(
    "lines, message",
    [
        ([{"clip": 2, "text": "a"}, {"clip": 1, "text": "b"}], "increase"),
        ([{"clip": 4, "text": "好"}], "no room"),
        ([{"clip": 1, "text": "Edinburgh 到了"}], "Latin"),
        ([{"clip": 1, "text": " "}], "empty"),
        ([{"clip": n, "text": "好"} for n in (1, 2, 3)], "3 lines, need 1-2"),
    ],
)
def test_bad_lines_are_rejected(lines: list[dict], message: str) -> None:
    with pytest.raises(OutlineError, match=message):
        narrate.validate_lines({"lines": lines}, SLOTS, 1, 2)


def test_long_silences_are_filled_one_line_at_a_time() -> None:
    clips = [
        {"video": "v", "index": n, "src_start": 0.0, "duration": 6.0, "role": "x"}
        for n in range(8)
    ]
    tags = {("v", n): {"description": f"画面{n}"} for n in range(8)}
    slots = [Slot(n, 0.3 + 6.0 * (n - 1), 5.0) for n in range(1, 9)]
    assert narrate.silences([1, 3, 5, 7], slots, 48.0) == []
    # Clip 1 then clip 6: 0.3-30.3 s unspoken with clips 2-5 free, and
    # 30.3 s to the end (48 s) with clips 7-8 free.
    assert narrate.silences([1, 6], slots, 48.0) == [
        (0.3, 30.3, [2, 3, 4, 5]),
        (30.3, 48.0, [7, 8]),
    ]
    asked: list[str] = []

    def write_one(prompt: str, limit: int) -> tuple[str, float]:
        asked.append(prompt)
        return f"补{len(asked)}", 1.5

    lines = narrate.validate_lines(
        {"lines": [{"clip": 1, "text": "开场"}, {"clip": 6, "text": "收尾"}]},
        slots,
        1,
        8,
    )
    profile = category.load_profile("city-walk")
    out, seconds = narrate.fill_silences(lines, clips, tags, slots, profile, write_one)
    # The free clip nearest the middle of 0.3-30.3 s is 3 (12.3 s); that
    # still leaves 12.3-30.3 s (18 s), so clip 4 too; then 7 for the end.
    assert [(x["clip"], x["text"]) for x in out] == [
        (1, "开场"),
        (3, "补1"),
        (4, "补2"),
        (6, "收尾"),
        (7, "补3"),
    ]
    assert out[1]["filled"] is True and "filled" not in out[0]
    assert seconds == 4.5 and narrate.silences([1, 3, 4, 6, 7], slots, 48.0) == []
    assert "「开场」" in asked[0] and "「收尾」" in asked[0] and "画面2" in asked[0]


def test_line_range_follows_the_cut_length() -> None:
    assert narrate.line_range(75.0, 20) == (5, 10)
    assert narrate.line_range(75.0, 4) == (4, 4)  # few slots: use them all
    assert narrate.line_range(5.0, 3) == (1, 2)


def test_shot_list_shows_landmarks_and_room() -> None:
    tags = {
        ("v", 0): {
            "description": "仰拍尖塔",
            "landmark": "圣吉尔斯大教堂",
            "time_of_day": "白天",
            "light": "顺光",
        },
        ("v", 1): {"description": "石板路"},
        ("w", 0): {"description": "黄昏远景", "landmark": ""},
    }
    rows = narrate.clip_rows(CLIPS, tags, [Slot(1, 0.3, 5.0)]).splitlines()
    assert rows == [
        "[1] 0.0s | 开场 | 仰拍尖塔（圣吉尔斯大教堂）（白天）（顺光） | 20 字",
        "[2] 6.0s | 街巷 | 石板路 | 不能放",
        "[3] 11.0s | 收尾 | 黄昏远景 | 不能放",
    ]


def test_the_prompt_carries_the_category_style(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = category.load_profile("city-walk")
    seen: dict = {}

    def fake_ask(prompt, schema, name, validate, server_url):  # type: ignore[no-untyped-def]
        seen.update(prompt=prompt, schema=schema)
        lines = [{"clip": 1, "text": "走进老城"}, {"clip": 3, "text": "再见"}]
        return validate({"lines": lines}), "{}", 1.0, []

    monkeypatch.setattr(narrate.outline, "_ask", fake_ask)
    tags = {(c["video"], c["index"]): {"description": "街景"} for c in CLIPS}
    lines, *_ = narrate.write_lines(CLIPS, tags, SLOTS, profile)
    assert lines[0]["text"] == "走进老城"
    assert profile.narration in seen["prompt"] and "「城市漫步」" in seen["prompt"]
    assert seen["schema"]["properties"]["lines"]["maxItems"] == 3
    with pytest.raises(OutlineError, match="no clip has room"):
        narrate.write_lines(CLIPS, tags, [], profile)


def test_long_voices_are_shortened_then_rejected() -> None:
    rate = 10
    line = {"clip": 1, "start": 0.3, "window": 2.0, "text": "一二三四五六七八"}
    asked: list[tuple[str, int]] = []

    def speak(text: str) -> np.ndarray:
        return np.ones(len(text) * 4, np.float32)  # 0.4 s per character

    def short(text: str, limit: int) -> str:
        asked.append((text, limit))
        return text[:limit]

    [(done, voice)] = narrate.voice_lines([line], speak, rate, short)
    assert asked == [("一二三四五六七八", 4)] and done["text"] == "一二三四"
    assert done["seconds"] == 1.6 and len(voice) == 16

    with pytest.raises(OutlineError, match="still takes"):
        narrate.voice_lines([line], speak, rate, lambda text, limit: text)


def test_ducking_turns_the_original_down_under_a_line() -> None:
    rate = media.AUDIO_RATE
    samples = np.ones((2, 4 * rate), np.float32)
    voice = np.full((2, rate), 0.1, np.float32)
    out = media.duck_and_mix(samples, 0, [(2.0, voice)])
    assert out[0, rate] == pytest.approx(1.0)  # 1 s: before the ramp
    assert out[0, int(2.5 * rate)] == pytest.approx(media.DUCK_GAIN + 0.1)
    assert out[0, int(3.075 * rate)] == pytest.approx(
        media.DUCK_GAIN + (1 - media.DUCK_GAIN) * 0.5, abs=1e-3
    )
    # A chunk that starts mid-line (sample offset) gets the same envelope.
    tail = media.duck_and_mix(samples[:, :rate], int(2.5 * rate), [(2.0, voice)])
    assert tail[0, 0] == pytest.approx(out[0, int(2.5 * rate)])


def test_voice_track_is_levelled_and_stereo() -> None:
    mono = 0.5 * np.sin(np.linspace(0, 200, 24_000)).astype(np.float32)
    track = media.voice_track(mono, 24_000)
    assert track.shape == (2, media.AUDIO_RATE)
    rms = float(np.sqrt(np.mean(track[0, 2000:-2000] ** 2)))
    assert rms == pytest.approx(media.NARRATION_RMS, rel=0.05)
    with pytest.raises(ValueError, match="silent"):
        media.voice_track(np.zeros(100, np.float32), 24_000)


def test_narration_text_gives_cut_and_source_times() -> None:
    played = [
        {"video": "a", "src_start": 100.0, "duration": 5.0},
        {"video": "b", "src_start": 62.0, "duration": 4.0},
    ]
    record = {
        "voice": "zf_088",
        "lines": [
            {"start": 0.3, "seconds": 2.0, "clip": 1, "text": "清晨的老城。"},
            {"start": 5.3, "seconds": 1.5, "clip": 2, "text": "钟声响起。"},
        ],
    }
    text = cli._narration_text(record, played, {"a": "walk.mp4", "b": "bell.mp4"})
    rows = text.splitlines()
    assert rows[0] == "解说词：2 句，音色 zf_088，成片 9.0 秒"
    assert rows[3] == "1\t00:00.3–00:02.3\t1\twalk.mp4\t01:40.3\t清晨的老城。"
    assert rows[4] == "2\t00:05.3–00:06.8\t2\tbell.mp4\t01:02.3\t钟声响起。"


def test_check_answers_are_validated() -> None:
    fix = {"verdict": "fix", "problems": [" 画面是白天 "]}
    assert narrate.validate_check(fix) == ["画面是白天"]
    assert narrate.validate_check({"verdict": "ok", "problems": ["x"]}) == []
    for bad, message in [
        ({"verdict": "fix", "problems": []}, "name its problems"),
        ({"verdict": "fix", "problems": "x"}, "list of strings"),
        ({"problems": []}, "verdict"),
    ]:
        with pytest.raises(OutlineError, match=message):
            narrate.validate_check(bad)
    assert narrate.validate_fix({"text": "白天的城堡。"}, 10) == "白天的城堡。"
    for bad, message in [({"text": "长" * 11}, "1-10"), ({"text": "Castle"}, "Latin")]:
        with pytest.raises(OutlineError, match=message):
            narrate.validate_fix(bad, 10)


def test_a_line_is_rewritten_only_when_the_check_finds_problems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def fake_ask(prompt, schema, name, validate, server_url):  # type: ignore[no-untyped-def]
        asked.append(name)
        if name == "narration_check":
            verdict = "fix" if "「黄昏" in prompt[0]["text"] else "ok"
            return (
                validate({"verdict": verdict, "problems": ["画面是白天"]}),
                "",
                1.0,
                [],
            )
        return validate({"text": "白天的城堡。"}), "", 2.0, []

    monkeypatch.setattr(narrate.outline, "_ask", fake_ask)
    monkeypatch.setattr(narrate.tag, "image_part", lambda p: {"image": str(p)})
    line = {"clip": 1, "start": 0.3, "window": 5.0, "text": "黄昏的城堡。"}
    # Check, fix, then the fixed line checks out: 1 + 2 + 1 s.
    assert narrate.check_line(line, [Path("a.jpg")], "城堡（白天）") == (
        {"problems": ["画面是白天"], "text": "白天的城堡。"},
        4.0,
    )
    assert asked == ["narration_check", "narration_fix", "narration_check"]
    asked.clear()
    plain = {**line, "text": "走进老街。"}
    assert narrate.check_line(plain, [Path("a.jpg")], "老街") == (
        {"problems": [], "text": "走进老街。"},
        1.0,
    )
    assert asked == ["narration_check"]


def test_lines_are_checked_against_their_clips_frames() -> None:
    frames = {
        (c["video"], c["index"]): [
            Path(f"{c['video']}{c['index']}_{k}.jpg") for k in range(5)
        ]
        for c in CLIPS
    }
    tags = {
        (c["video"], c["index"]): {"description": "街景", "time_of_day": "白天"}
        for c in CLIPS
    }
    seen: list = []

    def fake_check(line, picked, text):  # type: ignore[no-untyped-def]
        seen.append((picked, text))
        fixed = line["text"].replace("黄昏", "白天")
        return {
            "problems": [] if fixed == line["text"] else ["画面是白天"],
            "text": fixed,
        }, 2.0

    lines = [
        {"clip": 1, "start": 0.3, "window": 5.0, "text": "黄昏的城堡。"},
        {"clip": 3, "start": 11.3, "window": 5.0, "text": "走进老街。"},
    ]
    out, records, seconds = narrate.check_lines(lines, CLIPS, tags, frames, fake_check)
    assert [x["text"] for x in out] == ["白天的城堡。", "走进老街。"]
    assert records[0] == {
        "clip": 1,
        "written": "黄昏的城堡。",
        "checked": "白天的城堡。",
        "problems": ["画面是白天"],
    }
    assert seconds == 4.0
    # Three frames, first and last kept, and the clip's description.
    assert [p.name for p in seen[0][0]] == ["v0_0.jpg", "v0_2.jpg", "v0_4.jpg"]
    assert seen[0][1] == "街景（白天）"


def test_the_writer_sees_one_frame_per_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = category.load_profile("city-walk")
    seen: dict = {}

    def fake_ask(prompt, schema, name, validate, server_url):  # type: ignore[no-untyped-def]
        seen["prompt"] = prompt
        lines = [{"clip": 1, "text": "走进老城"}, {"clip": 3, "text": "再见"}]
        return validate({"lines": lines}), "{}", 1.0, []

    monkeypatch.setattr(narrate.outline, "_ask", fake_ask)
    monkeypatch.setattr(narrate.tag, "image_part", lambda p: {"image": p.name})
    tags = {(c["video"], c["index"]): {"description": "街景"} for c in CLIPS}
    frames = {
        (c["video"], c["index"]): [
            Path(f"{c['video']}{c['index']}_{k}.jpg") for k in range(3)
        ]
        for c in CLIPS
    }
    narrate.write_lines(CLIPS, tags, SLOTS, profile, frames=frames)
    parts = seen["prompt"]
    assert "以截图为准" in parts[0]["text"]
    assert parts[1:] == [
        {"type": "text", "text": "[1]"},
        {"image": "v0_1.jpg"},
        {"type": "text", "text": "[2]"},
        {"image": "v1_1.jpg"},
        {"type": "text", "text": "[3]"},
        {"image": "w0_1.jpg"},
    ]


def test_rewritten_lines_lose_the_prompts_quote_marks() -> None:
    fixed = narrate.validate_fix({"text": "「走到大教堂前。」"}, 20)
    assert fixed == "走到大教堂前。"


def test_a_rewrite_that_keeps_the_line_overrules_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def fake_ask(prompt, schema, name, validate, server_url):  # type: ignore[no-untyped-def]
        asked.append(name)
        if name == "narration_check":
            return validate({"verdict": "fix", "problems": ["可能是黄昏"]}), "", 1.0, []
        return validate({"text": "白天的城堡。"}), "", 2.0, []

    monkeypatch.setattr(narrate.outline, "_ask", fake_ask)
    monkeypatch.setattr(narrate.tag, "image_part", lambda p: {"image": str(p)})
    line = {"clip": 1, "start": 0.3, "window": 5.0, "text": "白天的城堡。"}
    assert narrate.check_line(line, [Path("a.jpg")], "城堡") == (
        {"problems": ["可能是黄昏"], "text": "白天的城堡。"},
        3.0,
    )
    assert asked == ["narration_check", "narration_fix"]
