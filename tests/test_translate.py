"""Translation: validation, glossary, translators, cues, storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from video_demo import outline, subtitle, translate
from video_demo.subtitle import Cue, Sentence
from video_demo.translate import Item


def _sentence(sid: int, text: str, start: float = 0.0, step: float = 0.1) -> Sentence:
    times = [(start + k * step, start + (k + 1) * step) for k in range(len(text))]
    return Sentence(sid, text, times)


# --- validation ------------------------------------------------------------------


def test_validate_translation_happy() -> None:
    item = Item(1, "Good morning, how are you?")
    assert translate.validate_translation(item, " 早上好，你们好吗？ ", "zh") == (
        "早上好，你们好吗？"
    )


@pytest.mark.parametrize(
    "text, match",
    [
        ("  ", "empty"),
        ("Good morning, how are you?", "copied"),
        ("Bonjour, comment ça va", "not Chinese"),
        ("早" * 200, "length"),
        ("好", "length"),
    ],
)
def test_validate_translation_rejects(text: str, match: str) -> None:
    item = Item(1, "Good morning, how are you?")
    with pytest.raises(outline.OutlineError, match=match):
        translate.validate_translation(item, text, "zh")


def test_validate_translation_short_source_may_stay_latin() -> None:
    assert translate.validate_translation(Item(1, "TED."), "TED。", "zh") == "TED。"


def test_already_target_skips_chinese() -> None:
    assert translate.already_target("这个 25 块", "zh")
    assert not translate.already_target("beef or pork 好香", "zh")


def test_source_size_counts_words_and_cjk() -> None:
    assert translate.source_size("I'm in Shanghai 2026") == 4
    assert translate.source_size("上海 vlog") == 3


# --- glossary --------------------------------------------------------------------


def test_validate_glossary_keeps_terms_in_text() -> None:
    answer = {
        "terms": [
            {"source": "Ken Robinson", "target": "肯·罗宾逊"},
            {"source": "Gillian Lynne", "target": "吉莉安·林恩"},  # not in text
            {"source": "TED", "target": "TED"},  # same as source: allowed
            {"source": "Stratford", "target": "Stratford-x"},  # not Chinese
            {"source": "ken robinson", "target": "罗宾逊"},  # duplicate
        ]
    }
    text = "I'm Ken Robinson at TED, from Stratford."
    assert translate.validate_glossary(answer, text, "zh") == {
        "Ken Robinson": "肯·罗宾逊",
        "TED": "TED",
    }
    with pytest.raises(outline.OutlineError, match="malformed"):
        translate.validate_glossary({"words": []}, text, "zh")


def test_terms_match_whole_words_only() -> None:
    glossary = {"arts": "艺术", "M": "学士", "Ken Robinson": "肯·罗宾逊"}
    assert translate.terms_for("It starts with KEN ROBINSON.", glossary) == {
        "Ken Robinson": "肯·罗宾逊"
    }
    answer = {"terms": [{"source": "M", "target": "学士"}]}
    assert translate.validate_glossary(answer, "an M A degree", "zh") == {}


def test_merge_terms_user_wins() -> None:
    model = {"Shakespeare": "莎士比亚", "TED": "TED大会"}
    assert translate.merge_terms(model, {"ted": "TED"}) == {
        "Shakespeare": "莎士比亚",
        "ted": "TED",
    }


def test_build_glossary_chunks_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    sentences = [_sentence(1, "Ken Robinson spoke."), _sentence(2, "我们去外滩")]
    prompts: list[str] = []

    def fake(prompt: str, t: float, *a: Any, **k: Any) -> tuple[str, float]:
        prompts.append(prompt)
        rows = [{"source": "Ken Robinson", "target": "肯·罗宾逊"}]
        return json.dumps({"terms": rows}), 2.0

    monkeypatch.setattr(translate, "_chat", fake)
    terms, seconds = translate.build_glossary(sentences, "TED", "zh")
    assert terms == {"Ken Robinson": "肯·罗宾逊"} and seconds == 2.0
    assert len(prompts) == 1 and "外滩" not in prompts[0]  # Chinese skipped

    monkeypatch.setattr(translate, "_chat", lambda *a, **k: ("oops", 1.0))
    with pytest.raises(translate.TranslateError, match="not JSON"):
        translate.build_glossary(sentences, "TED", "zh")


def test_chunks_split_by_size() -> None:
    assert translate.chunks(["aaaa", "bbbb", "cc"], 9) == ["aaaa\nbbbb", "cc"]
    assert translate.chunks([], 9) == []


def test_read_user_terms(tmp_path: Path) -> None:
    good = tmp_path / "terms.json"
    good.write_text(json.dumps({"TED": "TED"}), encoding="utf-8")
    assert translate.read_user_terms(good) == {"TED": "TED"}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["TED"]), encoding="utf-8")
    with pytest.raises(translate.TranslateError, match="map"):
        translate.read_user_terms(bad)


# --- translators -----------------------------------------------------------------


def test_llm_adapter_retries_bad_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = [Item(1, "Good morning, how are you?"), Item(2, "Thank you so much.")]
    good = {
        "sentences": [
            {"id": 1, "text": "早上好，你们好吗？"},
            {"id": 2, "text": "非常感谢。"},
        ]
    }
    bad = {
        "sentences": [{"id": 1, "text": "早上好，你们好吗？"}, {"id": 2, "text": ""}]
    }
    answers = iter([json.dumps(bad), json.dumps(good)])
    seen: list[str] = []

    def fake(prompt: str, *a: Any, **k: Any) -> tuple[str, float]:
        seen.append(prompt)
        return next(answers), 1.0

    monkeypatch.setattr(translate, "_chat", fake)
    adapter = translate.LlmTranslatorAdapter("TED")
    out, seconds, errors = adapter.translate(
        batch, ["Earlier line."], {"morning": "早上"}, "zh"
    )
    assert out == {1: "早上好，你们好吗？", 2: "非常感谢。"}
    assert seconds == 2.0 and len(errors) == 1 and "empty" in errors[0]
    assert "morning → 早上" in seen[0] and "Earlier line." in seen[0]


def test_llm_adapter_rejects_changed_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = json.dumps({"sentences": [{"id": 9, "text": "谢谢。"}]})
    monkeypatch.setattr(translate, "_chat", lambda *a, **k: (rows, 1.0))
    adapter = translate.LlmTranslatorAdapter("TED")
    with pytest.raises(translate.TranslateError, match="ids changed"):
        adapter.translate([Item(1, "Thank you.")], [], {}, "zh")


def test_hymt_prompt_uses_official_templates() -> None:
    adapter = translate.HyMtAdapter("http://x")
    one = adapter.prompt(Item(1, "Thank you."), [], {}, "zh")
    assert one.startswith("将以下文本翻译为简体中文") and one.endswith("Thank you.")
    two = adapter.prompt(
        Item(2, "Ken Robinson, thank you."),
        ["Before."],
        {"Ken Robinson": "肯·罗宾逊", "Other": "其他"},
        "zh",
    )
    assert two.startswith("参考下面的翻译：\nKen Robinson 翻译成 肯·罗宾逊")
    assert "Other" not in two and "Before.\n参考上面的信息" in two


def test_hymt_adapter_carries_context(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    def fake(prompt: str, t: float, n: int, url: str) -> tuple[str, float]:
        prompts.append(prompt)
        return ("你好。" if prompt.endswith("Hello there.") else "再见了。"), 0.5

    monkeypatch.setattr(translate, "_plain", fake)
    out, seconds, _ = translate.HyMtAdapter("http://x").translate(
        [Item(1, "Hello there."), Item(2, "Goodbye now.")], [], {}, "zh"
    )
    assert out == {1: "你好。", 2: "再见了。"} and seconds == 1.0
    assert prompts[1].startswith("Hello there.\n参考上面的信息")


def test_translate_sentences_batches_and_skips_target() -> None:
    sentences = [
        _sentence(1, "Hello there."),
        _sentence(2, "我们去外滩。"),
        _sentence(3, "Goodbye now."),
    ]

    class Fake:
        name = "fake"
        calls: list[tuple[list[int], list[str]]] = []

        def translate(
            self, batch: list[Item], context: list[str], glossary: dict, lang: str
        ) -> tuple[dict[int, str], float, list[str]]:
            self.calls.append(([it.id for it in batch], context))
            return {it.id: "译文。" for it in batch}, 1.0, []

    fake = Fake()
    out, seconds, _ = translate.translate_sentences(sentences, fake, {}, "zh", 1)
    assert [ids for ids, _ in fake.calls] == [[1], [3]] and seconds == 2.0
    assert fake.calls[1][1] == ["Hello there.", "我们去外滩。"]
    assert out[2] == "我们去外滩。"


# --- cues ------------------------------------------------------------------------


def test_cut_points_snap_to_marks_and_words() -> None:
    text = "欢迎来到我的频道，我现在在上海"
    assert translate.cut_points(text, [0.4]) == [9]  # after the comma
    assert translate.cut_points("一二三四五六七八九十", [0.5]) == [5]
    assert translate.cut_points("ab Shanghai cd", [0.3]) == [3]  # not inside a word
    assert translate.cut_points("一二三四", [0.6, 0.1]) == [2, 2]  # never backwards


def test_translated_cues_follow_source_cues() -> None:
    text = "Welcome to my channel I am in Shanghai right now"
    s = _sentence(1, text, start=10.0, step=0.05)
    src = subtitle.sentence_cues(s)
    assert len(src) == 2
    # Fits one line: kept whole over both source cues, never cut mid-word.
    short = translate.translated_cues([s], {1: "我对这一切感到非常震撼"}, True)
    assert len(short) == 1 and short[0].text.startswith("我对这一切感到非常震撼\n")
    assert short[0].start == pytest.approx(src[0].start)
    assert short[0].end == pytest.approx(src[1].end)
    # Two lines' worth: cut at the comma, each piece on its source cue.
    zh = {1: "欢迎大家来到我的这个频道，我现在正在中国的上海这座城市"}
    cues = translate.translated_cues([s], zh, False)
    assert [c.text for c in cues] == [
        "欢迎大家来到我的这个频道",
        "我现在正在中国的上海这座城市",
    ]
    assert cues[0].start == pytest.approx(src[0].start)
    assert cues[1].start == pytest.approx(src[1].start)
    bi = translate.translated_cues([s], zh, True)
    assert bi[0].text == "欢迎大家来到我的这个频道\n" + src[0].text


def test_untranslated_sentence_keeps_source_cues() -> None:
    s = _sentence(1, "这个二十五块。")
    cues = translate.translated_cues([s], {1: "这个二十五块。"}, True)
    assert [c.text for c in cues] == ["这个二十五块。"]  # one line, no repeat


def test_group_cues_keeps_whole_cues() -> None:
    src = [Cue(0.0, 1.0, "aaaa"), Cue(1.0, 2.0, "bb"), Cue(2.0, 3.0, "cccc")]
    two = translate.group_cues(src, 2)
    # Both boundaries are 0.1 from the middle; the first one wins.
    assert two == [Cue(0.0, 1.0, "aaaa"), Cue(1.0, 3.0, "bb cccc")]
    assert translate.group_cues(src, 1) == [Cue(0.0, 3.0, "aaaa bb cccc")]
    assert translate.group_cues(src, 3) == src


def test_split_translation_merges_empty_pieces() -> None:
    src = [Cue(0.0, 1.0, "I've been blown away by"), Cue(1.0, 2.0, "it.")]
    out = translate.split_translation("惊呆了", src)
    assert [c.text for c in out] == ["惊呆了"]
    assert out[0].start == 0.0 and out[-1].end == 2.0
    # An empty first piece hands its time to the next one.
    lead = [Cue(0.0, 1.0, "So"), Cue(1.0, 2.0, "we found this restaurant")]
    assert translate.split_translation("好", lead) == [Cue(0.0, 2.0, "好")]


def test_translated_cues_require_every_sentence() -> None:
    with pytest.raises(translate.TranslateError, match="no translation"):
        translate.translated_cues([_sentence(1, "Hello.")], {}, False)


def test_split_wide_shares_time() -> None:
    cue = Cue(0.0, 4.0, "这是一句非常非常长的翻译，需要分成两行才能放得下的字幕")
    out = translate.split_wide(cue)
    assert len(out) == 2 and out[0].text.endswith("翻译")
    assert out[0].end == pytest.approx(out[1].start)
    assert all(sum(subtitle.char_width(c) for c in x.text) <= 16 for x in out)
    no_marks = translate.split_wide(Cue(0.0, 2.0, "一" * 20))
    assert [len(x.text) for x in no_marks] == [10, 10]


def test_too_fast_flags_dense_cues() -> None:
    fast, ok = Cue(0.0, 1.0, "一二三四五六七八九十"), Cue(0.0, 2.0, "一二三\nabc")
    assert translate.too_fast([fast, ok]) == [fast]


# --- storage ---------------------------------------------------------------------


def test_translation_and_glossary_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    src = [_sentence(1, "hello there")]
    translate.write_translation(path, "v", "zh", "fake", {}, {1: "你好"}, 1.0, [], src)
    assert translate.read_translation(path, "v", "zh", src) == {1: "你好"}
    with pytest.raises(translate.TranslateError, match="stale"):
        translate.read_translation(path, "v", "ja", src)
    # Refined again: the cached translation no longer matches the sentences.
    with pytest.raises(translate.TranslateError, match="stale"):
        translate.read_translation(path, "v", "zh", [_sentence(1, "hello here")])
    g = tmp_path / "g.json"
    translate.write_glossary(g, "v", {"TED": "TED"}, 1.0)
    assert translate.read_glossary(g, "v") == {"TED": "TED"}
    with pytest.raises(translate.TranslateError, match="stale"):
        translate.read_glossary(g, "w")


def test_translated_count_ignores_sentences_already_in_target() -> None:
    src = [_sentence(1, "hello there"), _sentence(2, "你好")]
    assert translate.translated_count(src, {1: "你好啊", 2: "你好"}) == 1
    assert translate.translated_count(src[1:], {2: "你好"}) == 0


def test_unreachable_translator_is_a_translate_error() -> None:
    with pytest.raises(translate.TranslateError, match="连不上翻译服务"):
        translate._post({}, "http://127.0.0.1:9")


def test_numbers_alone_need_no_translation() -> None:
    assert translate.already_target("114 1726.", "zh")
    assert not translate.already_target("Platform 11.", "zh")
