"""Subtitles: token times, refinement checks, alignment, cues, mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from video_demo import outline, subtitle
from video_demo.subtitle import Cue, Sentence


def _sentence(sid: int, text: str, start: float = 0.0, step: float = 0.2) -> Sentence:
    times = [(start + k * step, start + (k + 1) * step) for k in range(len(text))]
    return Sentence(sid, text, times)


# --- recognition output -------------------------------------------------------


def test_token_times_spread_tokens_over_characters() -> None:
    text, times = subtitle.token_times([" look", "你"], [1.0, 2.0], 2.5)
    assert text == " look你"
    assert times[0] == pytest.approx((1.0, 1.2))
    assert times[4] == pytest.approx((1.8, 2.0))
    assert times[5] == pytest.approx((2.0, 2.5))


def test_token_times_rejects_mismatch() -> None:
    with pytest.raises(subtitle.SubtitleError, match="timestamps"):
        subtitle.token_times(["a", "b"], [0.0], 1.0)


def test_split_sentences_at_final_punctuation() -> None:
    text = " 你好。Ok. 3.5 km! ."
    times = [(float(k), k + 1.0) for k in range(len(text))]
    pieces = subtitle.split_sentences(text, times)
    assert [p for p, _ in pieces] == ["你好。", "Ok.", "3.5 km!"]
    assert pieces[1][1][0] == (4.0, 5.0)


# --- refinement -----------------------------------------------------------------


def test_validate_refinement_accepts_small_edits() -> None:
    batch = [_sentence(1, "我在上海外摊"), _sentence(2, "No.")]
    answer = {
        "sentences": [{"id": 1, "text": "我在上海外滩"}, {"id": 2, "text": "No."}]
    }
    assert subtitle.validate_refinement(answer, batch) == {
        1: "我在上海外滩",
        2: "No.",
    }


@pytest.mark.parametrize(
    "rows, match",
    [
        ([{"id": 1, "text": "a"}], "ids changed"),  # dropped a sentence
        ([{"id": 2, "text": "x"}, {"id": 1, "text": "y"}], "ids changed"),  # moved
        ([{"id": 1, "text": " "}, {"id": 2, "text": "No."}], "empty"),
        (
            [{"id": 1, "text": "今天天气真好我们出去玩"}, {"id": 2, "text": "No."}],
            "too much",
        ),
    ],
)
def test_validate_refinement_rejects_bad_answers(
    rows: list[dict[str, Any]], match: str
) -> None:
    batch = [_sentence(1, "我在上海外摊"), _sentence(2, "No.")]
    with pytest.raises(outline.OutlineError, match=match):
        subtitle.validate_refinement({"sentences": rows}, batch)


def test_validate_refinement_ignores_punctuation_and_case() -> None:
    batch = [_sentence(1, "嗯行行行。"), _sentence(2, "welcome to shanghai")]
    rows = [
        {"id": 1, "text": "嗯，行，行，行。"},
        {"id": 2, "text": "Welcome to Shanghai!"},
    ]
    assert (
        subtitle.validate_refinement({"sentences": rows}, batch)[1]
        == "嗯，行，行，行。"
    )


def test_wording_counts_words_not_letters() -> None:
    assert subtitle.wording("Two000 years, 上海!") == ["two000", "years", "上", "海"]
    old, new = subtitle.wording("Two000 years later."), subtitle.wording(
        "Two thousand years later."
    )
    assert subtitle.edit_distance(old, new) == 2  # one replaced, one inserted


def test_validate_refinement_rejects_malformed() -> None:
    with pytest.raises(outline.OutlineError, match="malformed"):
        subtitle.validate_refinement({"lines": []}, [_sentence(1, "a")])


def test_refine_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = [_sentence(1, "我在上海外摊")]
    answers = iter(
        [
            "not json",
            json.dumps({"sentences": [{"id": 1, "text": "完全不同的另一句话呀"}]}),
            json.dumps({"sentences": [{"id": 1, "text": "我在上海外滩"}]}),
        ]
    )
    monkeypatch.setattr(subtitle, "_request", lambda *a, **k: (next(answers), 1.0))
    refined, seconds, failed = subtitle.refine(batch, "vlog", "城市")
    assert refined == {1: "我在上海外滩"} and seconds == 3.0 and len(failed) == 2

    monkeypatch.setattr(subtitle, "_request", lambda *a, **k: ("{}", 1.0))
    with pytest.raises(subtitle.SubtitleError, match="malformed"):
        subtitle.refine(batch, "vlog", "城市")


def test_refine_batches_keep_every_id(monkeypatch: pytest.MonkeyPatch) -> None:
    sentences = [_sentence(i, f"第{i}句") for i in range(1, 30)]
    calls: list[list[int]] = []

    def fake(prompt: str, t: float, schema: dict, mt: int, url: str) -> tuple:
        ids = schema["properties"]["sentences"]["items"]["properties"]["id"]["enum"]
        calls.append(ids)
        rows = [{"id": i, "text": f"第{i}句"} for i in ids]
        return json.dumps({"sentences": rows}), 0.5

    monkeypatch.setattr(subtitle, "_request", fake)
    refined, _, _ = subtitle.refine(sentences, "vlog", "城市")
    assert len(calls) == 2 and len(calls[0]) == subtitle.REFINE_BATCH
    assert sorted(refined) == list(range(1, 30))


def test_video_context_counts_scenes_and_subjects() -> None:
    rows = [
        {"tags": {"scene": "外滩", "subjects": ["女士", "建筑"]}},
        {"tags": {"scene": "外滩", "subjects": ["女士"]}},
    ]
    assert subtitle.video_context(rows, top=2) == "外滩、女士"
    assert subtitle.video_context([]) == "（无标签）"


# --- alignment -------------------------------------------------------------------


def test_align_keeps_times_for_replacements() -> None:
    s = _sentence(1, "上海外摊", start=10.0)
    times = subtitle.align_times(s.text, s.times, "上海外滩")
    assert times == s.times


def test_align_insertion_takes_neighbour_gap() -> None:
    text = "ab"
    times = [(0.0, 1.0), (2.0, 3.0)]
    out = subtitle.align_times(text, times, "a,b")
    assert out == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    # Inserted at the very start / end: a point at the neighbour.
    assert subtitle.align_times(text, times, "xab")[0] == (0.0, 0.0)
    assert subtitle.align_times(text, times, "abx")[-1] == (3.0, 3.0)


def test_align_deletion_drops_times() -> None:
    s = _sentence(1, "嗯我们走", start=5.0)
    out = subtitle.align_times(s.text, s.times, "我们走")
    assert out == s.times[1:]


def test_align_rejects_bad_input() -> None:
    with pytest.raises(subtitle.SubtitleError):
        subtitle.align_times("ab", [(0.0, 1.0)], "ab")


def test_refined_sentences_requires_every_id() -> None:
    raw = [_sentence(1, "ab"), _sentence(2, "cd")]
    with pytest.raises(subtitle.SubtitleError, match="exactly"):
        subtitle.refined_sentences(raw, {1: "ab"})
    out = subtitle.refined_sentences(raw, {1: "ab", 2: "ce"})
    assert out[1].text == "ce" and out[1].times == raw[1].times


# --- cues ------------------------------------------------------------------------


def test_cues_break_at_punctuation_within_width() -> None:
    text = "今天我们去外滩看看风景，然后去城隍庙吃小笼包好不好"
    cues = subtitle.sentence_cues(_sentence(1, text, step=0.15))
    assert [c.text for c in cues] == [
        "今天我们去外滩看看风景",
        "然后去城隍庙吃小笼包好不好",
    ]
    for c in cues:
        assert sum(subtitle.char_width(ch) for ch in c.text) <= subtitle.MAX_LINE_WIDTH


def test_cues_never_split_latin_words() -> None:
    text = "Welcome to my channel I am in Shanghai right now"
    cues = subtitle.sentence_cues(_sentence(1, text, step=0.05))
    assert len(cues) == 2
    assert " ".join(c.text for c in cues) == text


def test_cues_respect_max_duration_and_pauses() -> None:
    times = [(k * 1.0, k * 1.0 + 0.9) for k in range(10)]
    cues = subtitle.sentence_cues(Sentence(1, "一二三四五六七八九十", times))
    assert all(c.end - c.start <= subtitle.MAX_CUE_SEC for c in cues)
    assert "".join(c.text for c in cues) == "一二三四五六七八九十"


def test_cues_avoid_breaks_shorter_than_min() -> None:
    # Fast speech: the comma comes after 0.35 s, the space after 1.5 s.
    text = "Hi, the weather is so sunny and windy today"
    cues = subtitle.sentence_cues(_sentence(1, text, step=0.05))
    assert cues[0].text != "Hi,"
    assert all(c.end - c.start >= subtitle.MIN_CUE_SEC for c in cues[:-1])


def test_make_cues_min_length_without_overlap() -> None:
    a = _sentence(1, "好。", start=0.0, step=0.1)
    b = _sentence(2, "走", start=0.5, step=0.1)
    c = _sentence(3, "嗯", start=5.0, step=0.1)
    cues = subtitle.make_cues([a, b, c])
    assert cues[0].end == pytest.approx(0.5)  # stops where the next starts
    assert cues[2].end == pytest.approx(6.0)  # stretched to MIN_CUE_SEC
    assert cues[0].text == "好。"


# --- timeline mapping ------------------------------------------------------------


def _cue(start: float, text: str, step: float) -> Cue:
    times = tuple((start + k * step, start + (k + 1) * step) for k in range(len(text)))
    return Cue(start, start + len(text) * step, text, times)


def test_map_cues_keeps_only_the_characters_heard() -> None:
    cues = {"v1": [_cue(1.0, "你好", 0.5), _cue(9.0, "一二三四五", 0.2)]}
    clips = [
        {"video": "v1", "src_start": 0.0, "duration": 9.7},  # 一二三 heard
        {"video": "v1", "src_start": 9.7, "duration": 2.0},  # 四五: 0.3 s
    ]
    out, records = subtitle.map_cues(cues, clips)
    assert out == [Cue(1.0, 2.0, "你好"), Cue(9.0, pytest.approx(9.6), "一二三…")]
    assert [(r["clip"], r["status"]) for r in records] == [
        (1, "kept"),
        (1, "trimmed"),
        (2, "dropped"),
    ]
    assert records[1]["kept"] == "一二三…" and records[1]["text"] == "一二三四五"
    assert records[2]["reason"] == "切点内只剩 0.30s"


def test_map_cues_never_keeps_part_of_a_latin_word() -> None:
    cues = {"v": [_cue(0.0, "Hi there", 0.3)]}
    out, _ = subtitle.map_cues(
        cues, [{"video": "v", "src_start": 0.0, "duration": 1.6}]
    )
    assert [c.text for c in out] == ["Hi…"]
    assert out[0].end == pytest.approx(0.6)
    out, records = subtitle.map_cues(
        cues, [{"video": "v", "src_start": 0.0, "duration": 0.4}]  # half of "Hi"
    )
    assert out == [] and records[0]["status"] == "dropped"


def test_map_cues_offsets_later_clips_and_needs_times() -> None:
    cues = {"v1": [_cue(5.0, "后面", 0.5)], "v2": [_cue(0.0, "前面", 0.5)]}
    clips = [
        {"video": "v2", "src_start": 0.0, "duration": 2.0},
        {"video": "v1", "src_start": 4.0, "duration": 3.0},
    ]
    out, _ = subtitle.map_cues(cues, clips)
    assert out == [Cue(0.0, 1.0, "前面"), Cue(3.0, 4.0, "后面")]
    clip = [{"video": "v", "src_start": 1.0, "duration": 2.0}]
    # A translation has no times per character: cut, it is kept whole when
    # at least half of it plays, else dropped.
    out, records = subtitle.map_cues({"v": [Cue(0.0, 2.0, "译文")]}, clip)
    assert out == [Cue(0.0, 1.0, "…译文")] and records[0]["status"] == "trimmed"
    out, records = subtitle.map_cues({"v": [Cue(0.0, 1.8, "译文")]}, clip)
    assert out == [] and records[0]["status"] == "dropped"
    out, _ = subtitle.map_cues({"v": [Cue(2.0, 3.6, "译文")]}, clip)
    assert out == [Cue(1.0, 2.0, "译文…")]
    assert subtitle.map_cues({"v": [Cue(1.0, 2.0, "译文")]}, clip)[0] == [
        Cue(0.0, 1.0, "译文")
    ]
    with pytest.raises(subtitle.SubtitleError, match="per-character"):
        subtitle.map_cues({"v": [Cue(0.0, 2.0, "错位", ((0.0, 1.0),))]}, clip)


def test_map_cues_empty_clips() -> None:
    assert subtitle.map_cues({"v": [Cue(0.0, 1.0, "a")]}, []) == ([], [])


def test_cues_carry_their_characters_times() -> None:
    cues = subtitle.make_cues([_sentence(1, " 你好，世界。", step=0.5)])
    assert all(len(c.times) == len(c.text) for c in cues)


def test_speech_spans_by_sentence_and_clause() -> None:
    spoken, clauses = subtitle.speech_spans(
        [_sentence(1, "你好，世界。", step=1.0), _sentence(2, "再见", start=8.0)]
    )
    assert spoken == [(0.0, 6.0), (8.0, pytest.approx(8.4))]
    assert clauses == [(0.0, 3.0), (3.0, 6.0), (8.0, pytest.approx(8.4))]


def test_srt_format() -> None:
    srt = subtitle.to_srt([Cue(0.0, 1.5, "你好"), Cue(3661.2, 3662.0, "Hi")])
    assert srt == (
        "1\n00:00:00,000 --> 00:00:01,500\n你好\n\n"
        "2\n01:01:01,200 --> 01:01:02,000\nHi\n\n"
    )


# --- caches and retrieval text ----------------------------------------------------


def test_asr_and_subtitle_round_trip(tmp_path: Path) -> None:
    raw = [_sentence(1, "上海外摊"), _sentence(2, "你好", start=3.0)]
    subtitle.write_asr(tmp_path / "asr.json", "vid", "m", raw)
    assert subtitle.read_asr(tmp_path / "asr.json", "vid")[1].text == "你好"
    with pytest.raises(subtitle.SubtitleError, match="stale"):
        subtitle.read_asr(tmp_path / "asr.json", "other")
    with pytest.raises(subtitle.SubtitleError, match="not found"):
        subtitle.read_asr(tmp_path / "none.json", "vid")
    refined = subtitle.refined_sentences(raw, {1: "上海外滩", 2: "你好"})
    path = tmp_path / "sub.json"
    cues = subtitle.write_subtitles(path, "vid", "m", raw, refined, 1.0, [])
    assert subtitle.read_cues(path, "vid") == cues


def test_speech_text_by_segment() -> None:
    sentences = [_sentence(1, "我们去外滩", start=0.0, step=1.0)]
    assert subtitle.speech_text(sentences, 0.0, 2.0) == "我们"
    assert subtitle.speech_text(sentences, 2.0, 10.0) == "去外滩"
    assert subtitle.speech_text(sentences, 10.0, 12.0) == ""
    assert subtitle.speech_digest(["a"]) != subtitle.speech_digest(["b"])
