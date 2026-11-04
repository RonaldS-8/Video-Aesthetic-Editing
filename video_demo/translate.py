"""Subtitle translation: glossary, sentence translation, bilingual cues.

Input is the refined subtitle sentences (``subtitle.py``). Translators get
whole sentences; times are never guessed across languages. Instead each
translation is cut onto the sentence's source cues: a cue's share of the
source characters sets where its piece of the translation ends, moved to
the nearest punctuation or space, and the piece takes the cue's time.

Before translating, the 4B reads the transcript and lists names and terms
with one translation each (only terms that occur in the transcript are
kept); the user's own terms override them. Every request carries the
glossary and the preceding sentences as context.

Two translators share one interface: ``LlmTranslatorAdapter`` sends
batches of numbered sentences to a chat model with a JSON schema, and
``HyMtAdapter`` sends one sentence at a time to Tencent's Hy-MT2 translation
model with its own prompt templates. Answers that drop sentences, copy the
source, are not in the target language or have an implausible length are
retried and finally raised; the source text is never passed off as a
translation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from video_demo import outline, subtitle
from video_demo.subtitle import Cue, Sentence

TRANSLATION_VERSION = 2  # 2: records the source sentences' digest
GLOSSARY_VERSION = 1
CONTEXT_SENTENCES = 5  # preceding source sentences sent as context
LLM_BATCH = 20  # sentences per chat-model request
GLOSSARY_CHUNK_CHARS = 6000  # transcript characters per glossary request
MAX_TERMS = 30  # per glossary request
MIN_TERM_CHARS = 2  # a single letter would match almost every sentence
# Characters of translation per source word (or source character for CJK).
MIN_LENGTH_RATIO = 0.3
MAX_LENGTH_RATIO = 8.0
SNAP_SHARE = 0.25  # a cut may move this share of the translation to a mark
BREAK_MARKS = frozenset("，、；：,;:。！？!? ")
MAX_READING_CPS = 9.0  # Chinese characters per second a viewer can read
LANGUAGES = {"zh": "简体中文"}
HYMT_SAMPLING = {"top_k": 20, "top_p": 0.6, "repeat_penalty": 1.05}


class TranslateError(RuntimeError):
    """A sentence or the glossary could not be translated acceptably."""


@dataclass(frozen=True)
class Item:
    id: int
    text: str


def is_cjk(ch: str) -> bool:
    return 0x3400 <= ord(ch) <= 0x9FFF or 0xF900 <= ord(ch) <= 0xFAFF


def cjk_share(text: str) -> float:
    letters = [c for c in text if c.isalnum()]
    return sum(map(is_cjk, letters)) / len(letters) if letters else 0.0


def source_size(text: str) -> int:
    """Words for spaced scripts, characters for CJK."""
    return sum(map(is_cjk, text)) + len(re.findall(r"[a-zA-ZÀ-ɏ']+|\d+", text))


def already_target(text: str, lang: str) -> bool:
    """Sentences already in the target language, or without any letters
    (numbers, punctuation), need no translation."""
    if not any(c.isalpha() for c in text):
        return True
    return lang == "zh" and cjk_share(text) >= 0.5


def validate_translation(item: Item, text: str, lang: str) -> str:
    """``text`` stripped; raises (OutlineError, for the retry loop) unless
    it is a plausible translation of ``item``."""
    text = text.strip()
    if not text:
        raise outline.OutlineError(f"sentence {item.id}: empty translation")
    words = source_size(item.text)
    if words >= 2 and text.casefold() == item.text.casefold():
        raise outline.OutlineError(f"sentence {item.id}: copied the source")
    if lang == "zh" and words >= 2 and not any(map(is_cjk, text)):
        raise outline.OutlineError(f"sentence {item.id}: not Chinese: {text!r}")
    size = len(re.sub(r"\s", "", text))
    if words and not MIN_LENGTH_RATIO <= size / words <= MAX_LENGTH_RATIO:
        raise outline.OutlineError(
            f"sentence {item.id}: implausible length {size} for {words} words"
        )
    return text


# --- glossary -------------------------------------------------------------------


GLOSSARY_PROMPT = """下面是视频《{title}》的字幕原文。请列出其中需要统一译法的
人名、地名、机构名和专业术语（最多 {limit} 个），每个给出一个{language}译法。
原文必须是字幕里出现过的原样写法；普通词语不要列。

{text}"""


def glossary_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "maxItems": MAX_TERMS,
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "minLength": 1},
                        "target": {"type": "string", "minLength": 1},
                    },
                    "required": ["source", "target"],
                },
            }
        },
        "required": ["terms"],
    }


def occurs(term: str, text: str) -> bool:
    """``term`` in ``text`` as whole words, ignoring case."""
    pattern = r"(?<![a-z0-9])" + re.escape(term.casefold()) + r"(?![a-z0-9])"
    return re.search(pattern, text.casefold()) is not None


def validate_glossary(answer: Any, text: str, lang: str) -> dict[str, str]:
    """Terms that occur in ``text`` (case-insensitive, first spelling wins)
    with a translation in the target language (or kept as is); raises when
    the answer is malformed."""
    try:
        rows = [
            (str(r["source"]).strip(), str(r["target"]).strip())
            for r in answer["terms"]
        ]
    except (KeyError, TypeError) as exc:
        raise outline.OutlineError(f"malformed glossary: {exc}") from exc
    out: dict[str, str] = {}
    for source, target in rows:
        if len(source) < MIN_TERM_CHARS or not target or not occurs(source, text):
            continue
        if lang == "zh" and not any(map(is_cjk, target)) and target != source:
            continue
        if source.casefold() not in {k.casefold() for k in out}:
            out[source] = target
    return out


def chunks(texts: list[str], limit: int) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    size = 0
    for t in texts:
        if cur and size + len(t) > limit:
            out.append("\n".join(cur))
            cur, size = [], 0
        cur.append(t)
        size += len(t) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def build_glossary(
    sentences: list[Sentence],
    title: str,
    lang: str,
    server_url: str = outline.SERVER_URL,
) -> tuple[dict[str, str], float]:
    """Model-extracted terms for the whole transcript, earliest chunk wins."""
    terms: dict[str, str] = {}
    seconds = 0.0
    source = [s.text for s in sentences if not already_target(s.text, lang)]
    for text in chunks(source, GLOSSARY_CHUNK_CHARS):
        prompt = GLOSSARY_PROMPT.format(
            title=title, limit=MAX_TERMS, language=LANGUAGES[lang], text=text
        )
        found, elapsed = _with_retries(
            lambda t, p=prompt: _chat(p, t, glossary_schema(), 1500, server_url),
            lambda raw, x=text: validate_glossary(_json(raw), x, lang),
        )
        seconds += elapsed
        for k, v in found.items():
            if k.casefold() not in {t.casefold() for t in terms}:
                terms[k] = v
    return terms, seconds


def merge_terms(model: dict[str, str], user: dict[str, str]) -> dict[str, str]:
    """User terms override model terms for the same source."""
    folded = {k.casefold() for k in user}
    return {**{k: v for k, v in model.items() if k.casefold() not in folded}, **user}


def terms_for(text: str, glossary: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in glossary.items() if occurs(k, text)}


# --- translators ----------------------------------------------------------------


class Translator(Protocol):
    name: str

    def translate(
        self,
        batch: list[Item],
        context: list[str],
        glossary: dict[str, str],
        lang: str,
    ) -> tuple[dict[int, str], float, list[str]]:
        """Translation per item id, model seconds, failed attempts."""
        ...


LLM_SYSTEM = "你是专业字幕译者，只按要求输出 JSON。"

LLM_PROMPT = """下面是视频《{title}》的字幕，每行一句，前面是编号。
请逐句译成{language}。
{terms}{context}
要求：译文准确、口语、简洁，适合做字幕；人名和术语按参考译法。
每个编号都要返回，顺序不变，text 是这一句的完整译文。

{lines}"""


class LlmTranslatorAdapter:
    """A chat model (Qwen3.5-4B) translating numbered batches as JSON."""

    def __init__(self, title: str, server_url: str = outline.SERVER_URL) -> None:
        self.name = f"llm@{server_url}"
        self.title = title
        self.server_url = server_url

    def translate(
        self,
        batch: list[Item],
        context: list[str],
        glossary: dict[str, str],
        lang: str,
    ) -> tuple[dict[int, str], float, list[str]]:
        lines = "\n".join(f"{it.id}. {it.text}" for it in batch)
        used = terms_for(lines, glossary)
        terms = (
            "参考译法：" + "；".join(f"{k} → {v}" for k, v in used.items()) + "\n"
            if used
            else ""
        )
        ctx = (
            "前文（只供参考，不要翻译）：\n" + "\n".join(context) + "\n"
            if context
            else ""
        )
        prompt = LLM_PROMPT.format(
            title=self.title,
            language=LANGUAGES[lang],
            terms=terms,
            context=ctx,
            lines=lines,
        )
        ids = [it.id for it in batch]
        schema = subtitle.refine_schema(ids)
        max_tokens = 400 + 4 * sum(len(it.text) for it in batch)

        def check(raw: str) -> dict[int, str]:
            answer = _json(raw)
            try:
                rows = [(int(r["id"]), str(r["text"])) for r in answer["sentences"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise outline.OutlineError(f"malformed translation: {exc}") from exc
            if [i for i, _ in rows] != ids:
                raise outline.OutlineError(f"ids changed: {[i for i, _ in rows]}")
            return {
                it.id: validate_translation(it, text, lang)
                for it, (_, text) in zip(batch, rows, strict=True)
            }

        errors: list[str] = []
        out, seconds = _with_retries(
            lambda t: _chat(prompt, t, schema, max_tokens, self.server_url),
            check,
            errors,
        )
        return out, seconds, errors


HYMT_TERMS = "参考下面的翻译：\n{pairs}"
HYMT_CONTEXT = (
    "{context}\n参考上面的信息，把下面的文本翻译成{language}，"
    "注意不需要翻译上文，也不要额外解释："
)
HYMT_DEFAULT = "将以下文本翻译为{language}，注意只需要输出翻译后的结果，不要额外解释："


class HyMtAdapter:
    """Tencent Hy-MT2 (llama-server), one sentence per request, with its
    official terminology and context prompt templates."""

    def __init__(self, server_url: str) -> None:
        self.name = f"hy-mt2@{server_url}"
        self.server_url = server_url

    def prompt(
        self, item: Item, context: list[str], glossary: dict[str, str], lang: str
    ) -> str:
        blocks = []
        used = terms_for(item.text, glossary)
        if used:
            pairs = "\n".join(f"{k} 翻译成 {v}" for k, v in used.items())
            blocks.append(HYMT_TERMS.format(pairs=pairs))
        if context:
            rule = HYMT_CONTEXT.format(
                context="\n".join(context), language=LANGUAGES[lang]
            )
        else:
            rule = HYMT_DEFAULT.format(language=LANGUAGES[lang])
        blocks.append(rule)
        return "\n\n".join(blocks) + "\n\n" + item.text

    def translate(
        self,
        batch: list[Item],
        context: list[str],
        glossary: dict[str, str],
        lang: str,
    ) -> tuple[dict[int, str], float, list[str]]:
        out: dict[int, str] = {}
        seconds = 0.0
        errors: list[str] = []
        ctx = list(context)
        for it in batch:
            prompt = self.prompt(it, ctx[-CONTEXT_SENTENCES:], glossary, lang)
            budget = 64 + 4 * len(it.text)
            text, elapsed = _with_retries(
                lambda t, p=prompt, n=budget: _plain(p, t, n, self.server_url),
                lambda raw, x=it: validate_translation(x, raw, lang),
                errors,
            )
            out[it.id] = text
            seconds += elapsed
            ctx.append(it.text)
        return out, seconds, errors


def translate_sentences(
    sentences: list[Sentence],
    translator: Translator,
    glossary: dict[str, str],
    lang: str,
    batch_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[int, str], float, list[str]]:
    """Translation of every sentence (sentences already in ``lang`` stay as
    they are), model seconds, failed attempts."""
    items = [Item(s.id, s.text) for s in sentences]
    out = {it.id: it.text for it in items if already_target(it.text, lang)}
    todo = [it for it in items if it.id not in out]
    position = {it.id: k for k, it in enumerate(items)}
    seconds = 0.0
    failed: list[str] = []
    for k in range(0, len(todo), batch_size):
        batch = todo[k : k + batch_size]
        first = position[batch[0].id]
        context = [it.text for it in items[max(0, first - CONTEXT_SENTENCES) : first]]
        got, elapsed, errors = translator.translate(batch, context, glossary, lang)
        out.update(got)
        seconds += elapsed
        failed.extend(errors)
        if progress:
            progress(min(k + batch_size, len(todo)), len(todo))
    return out, seconds, failed


# --- cues ----------------------------------------------------------------------


def cut_points(text: str, shares: list[float]) -> list[int]:
    """Where to cut ``text`` so its pieces follow ``shares`` (cumulative,
    ending below 1): each cut moves to the nearest break mark within
    ``SNAP_SHARE`` of the text, never inside a Latin word, and cuts never
    go backwards."""
    n = len(text)
    marks = [k for k in range(1, n) if text[k - 1] in BREAK_MARKS]
    cuts: list[int] = []
    for share in shares:
        target = round(n * share)
        near = [k for k in marks if abs(k - target) <= SNAP_SHARE * n]
        k = min(near, key=lambda m: abs(m - target)) if near else target
        while (
            0 < k < n
            and text[k - 1].isascii()
            and text[k - 1].isalnum()
            and text[k].isascii()
            and text[k].isalnum()
        ):
            k += 1
        cuts.append(max(k, cuts[-1] if cuts else 0))
    return cuts


def group_cues(src: list[Cue], n: int) -> list[Cue]:
    """``src`` merged into ``n`` runs of whole cues of about equal source
    length (texts joined by a space)."""
    sizes = [max(len(c.text), 1) for c in src]
    total = sum(sizes)
    ends = [sum(sizes[: k + 1]) / total for k in range(len(src) - 1)]
    cuts: list[int] = []
    for j in range(1, n):
        free = [k for k in range(1, len(src)) if k not in cuts]
        cuts.append(min(free, key=lambda k: abs(ends[k - 1] - j / n)))
    bounds = [0, *sorted(cuts), len(src)]
    return [
        Cue(src[a].start, src[b - 1].end, " ".join(c.text for c in src[a:b]))
        for a, b in zip(bounds, bounds[1:], strict=False)
    ]


def split_translation(text: str, src: list[Cue]) -> list[Cue]:
    """``text`` (a sentence's translation) cut onto its source cues: only
    into as many pieces as it needs lines (a cut without punctuation may
    split a Chinese word), the cues grouped to match; each piece takes its
    group's time, and groups that get no text merge into a neighbour."""
    width = sum(subtitle.char_width(c) for c in text)
    lines = max(1, math.ceil(width / subtitle.MAX_LINE_WIDTH))
    src = group_cues(src, min(lines, len(src)))
    sizes = [max(len(c.text), 1) for c in src]
    total = sum(sizes)
    shares = [sum(sizes[: k + 1]) / total for k in range(len(src) - 1)]
    bounds = [0, *cut_points(text, shares), len(text)]
    out: list[Cue] = []
    start = src[0].start
    for k, cue in enumerate(src):
        piece = text[bounds[k] : bounds[k + 1]].strip().rstrip("，、；：,;:")
        if piece:
            out.append(Cue(start, cue.end, piece))
        elif out:  # an empty piece's time goes to the piece before it
            out[-1] = Cue(out[-1].start, cue.end, out[-1].text)
        else:  # ... or, at the start, to the piece after it
            continue
        if k + 1 < len(src):
            start = src[k + 1].start
    return out


def split_wide(cue: Cue) -> list[Cue]:
    """A translated cue wider than one line, split at punctuation (else
    evenly) with its time shared by character count."""
    text = cue.text
    width = sum(subtitle.char_width(c) for c in text)
    if width <= subtitle.MAX_LINE_WIDTH:
        return [cue]
    n = int(width // subtitle.MAX_LINE_WIDTH) + 1
    bounds = [
        0,
        *sorted(set(cut_points(text, [j / n for j in range(1, n)]))),
        len(text),
    ]
    out = []
    span = cue.end - cue.start
    for a, b in zip(bounds, bounds[1:], strict=False):
        piece = text[a:b].strip().rstrip("，、；：,;:")
        if piece:
            out.append(
                Cue(
                    cue.start + span * a / len(text),
                    cue.start + span * b / len(text),
                    piece,
                )
            )
    return out


def translated_cues(
    sentences: list[Sentence], translations: dict[int, str], bilingual: bool
) -> list[Cue]:
    """Cues of the translation on the source cues' times; with
    ``bilingual`` the matching source text goes on a second line."""
    raw: list[Cue] = []
    for s in sentences:
        if s.id not in translations:
            raise TranslateError(f"sentence {s.id} has no translation")
        src = subtitle.sentence_cues(s)
        if not src:
            continue
        if translations[s.id] == s.text:  # already in the target language
            raw.extend(src)
            continue
        pieces = split_translation(translations[s.id], src)
        if bilingual:
            for piece in pieces:
                under = " ".join(
                    c.text for c in src if piece.start <= c.start < piece.end
                )
                raw.append(Cue(piece.start, piece.end, f"{piece.text}\n{under}"))
        else:
            raw.extend(c for piece in pieces for c in split_wide(piece))
    return subtitle.settle(raw)


def translated_count(sentences: list[Sentence], translations: dict[int, str]) -> int:
    """Sentences whose translation differs from the source (the rest were
    already in the target language)."""
    return sum(translations.get(s.id, s.text) != s.text for s in sentences)


def too_fast(cues: list[Cue]) -> list[Cue]:
    """Cues whose first line needs more than ``MAX_READING_CPS``."""
    return [
        c
        for c in cues
        if len(c.text.split("\n")[0]) / max(c.end - c.start, 1e-3) > MAX_READING_CPS
    ]


# --- storage -------------------------------------------------------------------


def write_translation(
    path: Path,
    vid: str,
    lang: str,
    translator: str,
    glossary: dict[str, str],
    translations: dict[int, str],
    seconds: float,
    failed: list[str],
    sentences: list[Sentence],
) -> None:
    data = {
        "version": TRANSLATION_VERSION,
        "video_id": vid,
        "lang": lang,
        "source": source_digest(sentences),
        "translator": translator,
        "glossary": glossary,
        "seconds": round(seconds, 1),
        "failed_attempts": failed,
        "translations": {str(k): v for k, v in sorted(translations.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def source_digest(sentences: list[Sentence]) -> str:
    """Fingerprint of the sentences a translation was made from."""
    text = "\n".join(f"{s.id}\t{s.text}" for s in sentences)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def is_current(path: Path, vid: str, lang: str, sentences: list[Sentence]) -> bool:
    """Whether ``path`` holds this version's translation of ``sentences``."""
    if not path.exists():
        return False
    data = json.loads(path.read_text("utf-8"))
    return (
        data.get("version") == TRANSLATION_VERSION
        and data.get("video_id") == vid
        and data.get("lang") == lang
        and data.get("source") == source_digest(sentences)
    )


def read_translation(
    path: Path, vid: str, lang: str, sentences: list[Sentence]
) -> dict[int, str]:
    """The cached translation of ``sentences``; raises when it was made
    from other sentences (the subtitles were refined again) or by another
    version."""
    if not is_current(path, vid, lang, sentences):
        raise TranslateError(f"{path} is stale; rerun subtitle")
    data = json.loads(path.read_text("utf-8"))
    return {int(k): v for k, v in data["translations"].items()}


def write_glossary(path: Path, vid: str, terms: dict[str, str], seconds: float) -> None:
    data = {
        "version": GLOSSARY_VERSION,
        "video_id": vid,
        "seconds": round(seconds, 1),
        "terms": terms,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def read_glossary(path: Path, vid: str) -> dict[str, str]:
    data = json.loads(path.read_text("utf-8"))
    if data["version"] != GLOSSARY_VERSION or data["video_id"] != vid:
        raise TranslateError(f"{path} is stale; rerun with --force")
    return dict(data["terms"])


def read_user_terms(path: Path) -> dict[str, str]:
    """A JSON object mapping source terms to their translations."""
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and k and v for k, v in data.items()
    ):
        raise TranslateError(f"{path} must map each term to its translation")
    return data


# --- requests -------------------------------------------------------------------


def _json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise outline.OutlineError(f"Model output is not JSON: {raw[:200]!r}") from exc


def _with_retries(
    call: Callable[[float], tuple[str, float]],
    check: Callable[[str], Any],
    errors: list[str] | None = None,
) -> tuple[Any, float]:
    """``check(call(t))`` for each temperature until it passes; failed
    attempts go to ``errors``. Raises TranslateError when all fail."""
    tries: list[str] = []
    seconds = 0.0
    for temperature in outline.TEMPERATURES:
        raw, elapsed = call(temperature)
        seconds += elapsed
        try:
            result = check(raw)
        except outline.OutlineError as exc:
            tries.append(f"t={temperature}: {exc}")
            continue
        if errors is not None:
            errors.extend(tries)
        return result, seconds
    raise TranslateError("；".join(tries))


def _post(body: dict[str, Any], server_url: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{server_url}/v1/chat/completions", json=body, timeout=outline.TIMEOUT_SEC
        )
    except requests.ConnectionError as exc:
        raise TranslateError(
            f"连不上翻译服务 {server_url}，先启动它（见 README 的“字幕翻译”）"
        ) from exc
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"], elapsed


def _chat(
    prompt: str,
    temperature: float,
    schema: dict[str, Any],
    max_tokens: int,
    server_url: str,
) -> tuple[str, float]:
    body = {
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        },
    }
    return _post(body, server_url)


def _plain(
    prompt: str, temperature: float, max_tokens: int, server_url: str
) -> tuple[str, float]:
    """A translation model's request: no system prompt, plain text out."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        **HYMT_SAMPLING,
    }
    return _post(body, server_url)
