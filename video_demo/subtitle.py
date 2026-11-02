"""AI subtitles: speech recognition, model refinement, aligned cues.

Qwen3.5-4B cannot hear, so speech goes through SenseVoice-Small (int8, via
sherpa-onnx on the CPU) after silero VAD cuts it into utterances. SenseVoice
gives every token a start time; each character inherits its token's span.
Recognition runs as its own stage and releases its model before the 4B is
used, so the two never share the edge box's memory.

The 4B then corrects recognition errors sentence by sentence, with what the
video shows (places, subjects from the tags) as context. It may only edit
text: every sentence id comes back, none is merged or moved, and each edit
stays small. Answers that break these rules are retried and finally raised.

Refined text is aligned back to the recognised characters by edit distance
(kept and replaced characters keep their times, inserted ones take their
neighbours'), then broken into cues by fixed rules and written as SRT, per
video or mapped onto an edit timeline.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import requests

from video_demo import outline

SAMPLE_RATE = 16000
VAD_WINDOW = 512  # samples per silero VAD step
VAD_MIN_SILENCE_SEC = 0.3
VAD_MAX_SPEECH_SEC = 15.0
VAD_BUFFER_SEC = 60.0
ASR_THREADS = 4
SENSEVOICE_DIR = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
VAD_FILE = "silero_vad.onnx"
ASR_VERSION = 1
SUBTITLE_VERSION = 1

SENTENCE_END = frozenset("。！？!?.")
SOFT_BREAK = frozenset("，、；：,;:")
PAUSE_SEC = 0.3  # a gap this long between characters is a natural break

MAX_EDIT_RATIO = 0.3
MIN_EDIT_ALLOWANCE = 2  # short sentences may always change this many chars
REFINE_BATCH = 25  # sentences per model request

MAX_LINE_WIDTH = 16.0  # CJK characters; other characters count half
MIN_CUE_SEC = 1.0
MAX_CUE_SEC = 6.0
MIN_KEPT_SEC = 0.5  # a cue cut by a clip boundary must keep this much
MIN_KEPT_SHARE = 0.5  # ... and this share of its length


class SubtitleError(RuntimeError):
    """Recognition, refinement or alignment could not produce subtitles."""


@dataclass(frozen=True)
class Sentence:
    id: int
    text: str
    # (start, end) seconds for every character of ``text``.
    times: list[tuple[float, float]]

    @property
    def start(self) -> float:
        return self.times[0][0]

    @property
    def end(self) -> float:
        return self.times[-1][1]


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


# --- recognition ---------------------------------------------------------------


def read_audio(path: Path, chunk_sec: float = 30.0) -> Iterator[np.ndarray]:
    """The first audio stream as 16 kHz mono float32, in chunks of about
    ``chunk_sec`` (a long video is never held in memory at once)."""
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise SubtitleError(f"{path} has no audio stream")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        pending: list[np.ndarray] = []
        size = 0
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                pending.append(out.to_ndarray().reshape(-1))
                size += pending[-1].size
            if size >= chunk_sec * SAMPLE_RATE:
                yield np.concatenate(pending)
                pending, size = [], 0
        for out in resampler.resample(None):
            pending.append(out.to_ndarray().reshape(-1))
        if pending:
            yield np.concatenate(pending)


def token_times(
    tokens: list[str], starts: list[float], end: float
) -> tuple[str, list[tuple[float, float]]]:
    """Text of ``tokens`` with a span per character: a token lasts until the
    next one starts (the last until ``end``); its characters share that span
    evenly."""
    if len(tokens) != len(starts):
        raise SubtitleError(f"{len(tokens)} tokens but {len(starts)} timestamps")
    text: list[str] = []
    times: list[tuple[float, float]] = []
    for k, token in enumerate(tokens):
        t0 = starts[k]
        t1 = max(starts[k + 1] if k + 1 < len(starts) else end, t0)
        step = (t1 - t0) / max(len(token), 1)
        for j, ch in enumerate(token):
            text.append(ch)
            times.append((t0 + j * step, t0 + (j + 1) * step))
    return "".join(text), times


def split_sentences(
    text: str, times: list[tuple[float, float]]
) -> list[tuple[str, list[tuple[float, float]]]]:
    """An utterance split after sentence-final punctuation; surrounding
    spaces trimmed, pieces without a letter or digit dropped."""
    pieces: list[tuple[str, list[tuple[float, float]]]] = []
    begin = 0
    for i, ch in enumerate(text):
        decimal = ch == "." and 0 < i < len(text) - 1 and text[i + 1].isdigit()
        if ch in SENTENCE_END and not decimal:
            pieces.append((text[begin : i + 1], times[begin : i + 1]))
            begin = i + 1
    pieces.append((text[begin:], times[begin:]))
    out = []
    for piece, span in pieces:
        lead = len(piece) - len(piece.lstrip())
        trail = len(piece.rstrip())
        piece, span = piece[lead:trail], span[lead:trail]
        if any(c.isalnum() for c in piece):
            out.append((piece, span))
    return out


class SenseVoiceAdapter:
    """SenseVoice-Small int8 behind silero VAD, both on sherpa-onnx (CPU)."""

    model = SENSEVOICE_DIR

    def __init__(self, models_dir: Path, threads: int = ASR_THREADS) -> None:
        import sherpa_onnx

        model_dir = models_dir / SENSEVOICE_DIR
        vad_file = models_dir / VAD_FILE
        for need in (model_dir / "model.int8.onnx", model_dir / "tokens.txt", vad_file):
            if not need.exists():
                raise SubtitleError(f"{need} not found; see README (AI 字幕)")
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=threads,
            use_itn=True,
        )
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(vad_file)
        config.silero_vad.min_silence_duration = VAD_MIN_SILENCE_SEC
        config.silero_vad.max_speech_duration = VAD_MAX_SPEECH_SEC
        config.sample_rate = SAMPLE_RATE
        self._vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=VAD_BUFFER_SEC
        )

    def transcribe(self, path: Path) -> list[Sentence]:
        """Sentences of the video's speech, numbered from 1 in time order."""
        pieces: list[tuple[str, list[tuple[float, float]]]] = []
        rest = np.zeros(0, dtype=np.float32)
        for chunk in read_audio(path):
            rest = np.concatenate([rest, chunk])
            usable = rest.size - rest.size % VAD_WINDOW
            for i in range(0, usable, VAD_WINDOW):
                self._vad.accept_waveform(rest[i : i + VAD_WINDOW])
            rest = rest[usable:]
            pieces.extend(self._drain())
        if rest.size:
            self._vad.accept_waveform(rest)
        self._vad.flush()
        pieces.extend(self._drain())
        self._vad.reset()
        return [Sentence(n, text, times) for n, (text, times) in enumerate(pieces, 1)]

    def _drain(self) -> list[tuple[str, list[tuple[float, float]]]]:
        out = []
        while not self._vad.empty():
            seg = self._vad.front
            start = seg.start / SAMPLE_RATE
            samples = np.asarray(seg.samples, dtype=np.float32)
            self._vad.pop()
            stream = self._recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            self._recognizer.decode_stream(stream)
            result = stream.result
            text, times = token_times(
                list(result.tokens),
                [start + t for t in result.timestamps],
                start + samples.size / SAMPLE_RATE,
            )
            out.extend(split_sentences(text, times))
        return out

    def close(self) -> None:
        """Free the models before the 4B server needs the memory."""
        del self._recognizer, self._vad
        gc.collect()


def sentences_to_json(sentences: list[Sentence]) -> list[dict[str, Any]]:
    return [
        {
            "id": s.id,
            "text": s.text,
            "times": [[round(a, 3), round(b, 3)] for a, b in s.times],
        }
        for s in sentences
    ]


def sentences_from_json(rows: list[dict[str, Any]]) -> list[Sentence]:
    return [Sentence(r["id"], r["text"], [(a, b) for a, b in r["times"]]) for r in rows]


def write_asr(path: Path, vid: str, model: str, sentences: list[Sentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": ASR_VERSION,
        "video_id": vid,
        "model": model,
        "sentences": sentences_to_json(sentences),
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def read_asr(path: Path, vid: str) -> list[Sentence]:
    if not path.exists():
        raise SubtitleError(f"{path} not found; run `python -m video_demo subtitle`")
    data = json.loads(path.read_text("utf-8"))
    if data["version"] != ASR_VERSION or data["video_id"] != vid:
        raise SubtitleError(f"{path} is stale; rerun subtitle with --force")
    return sentences_from_json(data["sentences"])


# --- refinement by the 4B --------------------------------------------------------


REFINE_SYSTEM = "你是字幕校对员，只按要求输出 JSON。"

REFINE_PROMPT = """下面是视频《{title}》语音识别出的字幕，每行一句，前面是编号。
画面里出现过：{context}。

请逐句改正明显的识别错误：同音错字、专有名词、标点和大小写。
不要改写、润色、翻译、合并或拆分句子；拿不准就保持原样。
每个编号都要返回，顺序不变，text 是改正后的整句。

{lines}"""


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wording(text: str) -> list[str]:
    """``text`` as the units the edit limit counts: Latin words and numbers
    whole, other characters one by one; punctuation and case are ignored,
    since fixing them, or a word's spelling, moves no word."""
    plain = "".join(
        c for c in text.casefold() if not unicodedata.category(c).startswith("P")
    )
    return re.findall(r"[a-z0-9À-ɏ']+|[^\sa-z0-9À-ɏ']", plain)


def edit_allowance(units: list[str]) -> int:
    return max(MIN_EDIT_ALLOWANCE, int(MAX_EDIT_RATIO * len(units)))


def refine_schema(ids: list[int]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "enum": ids},
                        "text": {"type": "string", "minLength": 1},
                    },
                    "required": ["id", "text"],
                },
            }
        },
        "required": ["sentences"],
    }


def validate_refinement(answer: Any, batch: list[Sentence]) -> dict[int, str]:
    """Refined text by id; raises unless every id comes back once, in order,
    with a non-empty text no further than ``edit_allowance`` from the
    original."""
    try:
        rows = answer["sentences"]
        got = [(int(r["id"]), str(r["text"]).strip()) for r in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise outline.OutlineError(f"malformed refinement: {exc}") from exc
    want = [s.id for s in batch]
    if [i for i, _ in got] != want:
        raise outline.OutlineError(
            f"ids changed: expected {want}, got {[i for i, _ in got]}"
        )
    out: dict[int, str] = {}
    for s, (i, text) in zip(batch, got, strict=True):
        if not text:
            raise outline.OutlineError(f"sentence {i} came back empty")
        dist = edit_distance(wording(s.text), wording(text))
        if dist > edit_allowance(wording(s.text)):
            raise outline.OutlineError(
                f"sentence {i} changed too much ({dist} edits): "
                f"{s.text!r} -> {text!r}"
            )
        out[i] = text
    return out


def video_context(tag_rows: list[dict[str, Any]], top: int = 15) -> str:
    """Most frequent scenes and subjects the tagger saw in the video."""
    counts: Counter[str] = Counter()
    for r in tag_rows:
        counts[r["tags"]["scene"]] += 1
        counts.update(r["tags"]["subjects"])
    return "、".join(word for word, _ in counts.most_common(top)) or "（无标签）"


def refine(
    sentences: list[Sentence],
    title: str,
    context: str,
    server_url: str = outline.SERVER_URL,
) -> tuple[dict[int, str], float, list[str]]:
    """Corrected text for every sentence, model seconds, failed attempts.

    Raises SubtitleError when a batch fails at every temperature."""
    refined: dict[int, str] = {}
    seconds = 0.0
    failed: list[str] = []
    for k in range(0, len(sentences), REFINE_BATCH):
        batch = sentences[k : k + REFINE_BATCH]
        lines = "\n".join(f"{s.id}. {s.text}" for s in batch)
        prompt = REFINE_PROMPT.format(title=title, context=context, lines=lines)
        ids = [s.id for s in batch]
        max_tokens = 400 + 3 * sum(len(s.text) for s in batch)
        errors: list[str] = []
        for temperature in outline.TEMPERATURES:
            raw, elapsed = _request(
                prompt, temperature, refine_schema(ids), max_tokens, server_url
            )
            seconds += elapsed
            try:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise outline.OutlineError(
                        f"Model output is not JSON: {raw[:200]!r}"
                    ) from exc
                refined.update(validate_refinement(parsed, batch))
                break
            except outline.OutlineError as exc:
                errors.append(f"sentences {ids[0]}-{ids[-1]} t={temperature}: {exc}")
        else:
            raise SubtitleError("；".join(errors))
        failed.extend(errors)
    return refined, seconds, failed


def _request(
    prompt: str,
    temperature: float,
    schema: dict[str, Any],
    max_tokens: int,
    server_url: str,
) -> tuple[str, float]:
    body = {
        "messages": [
            {"role": "system", "content": REFINE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "subtitles", "schema": schema},
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(
        f"{server_url}/v1/chat/completions", json=body, timeout=outline.TIMEOUT_SEC
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"], elapsed


# --- alignment ------------------------------------------------------------------


def align_times(
    original: str, times: list[tuple[float, float]], refined: str
) -> list[tuple[float, float]]:
    """A span for every character of ``refined``: characters kept or replaced
    keep the original character's span; inserted ones take the gap between
    their neighbours (a point when there is none)."""
    if len(original) != len(times) or not original:
        raise SubtitleError("original text and times do not match")
    n, m = len(original), len(refined)
    cost = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(
                cost[i - 1, j] + 1,
                cost[i, j - 1] + 1,
                cost[i - 1, j - 1] + (original[i - 1] != refined[j - 1]),
            )
    # Walk back: source[j] is the original index refined[j] stands on, or -1.
    source = [-1] * m
    i, j = n, m
    while i > 0 and j > 0:
        diff = original[i - 1] != refined[j - 1]
        if cost[i, j] == cost[i - 1, j - 1] + diff:
            source[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif cost[i, j] == cost[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    out: list[tuple[float, float]] = []
    for j in range(m):
        if source[j] >= 0:
            out.append(times[source[j]])
            continue
        before = out[-1][1] if out else None
        later = (times[source[k]][0] for k in range(j + 1, m) if source[k] >= 0)
        after = next(later, None)
        if before is None and after is None:
            before, after = times[0][0], times[-1][1]
        a = before if before is not None else after
        b = after if after is not None else before
        assert a is not None and b is not None
        out.append((a, max(a, b)))
    return out


def refined_sentences(
    sentences: list[Sentence], refined: dict[int, str]
) -> list[Sentence]:
    if set(refined) != {s.id for s in sentences}:
        raise SubtitleError("refinement does not cover exactly the sentences")
    return [
        Sentence(s.id, refined[s.id], align_times(s.text, s.times, refined[s.id]))
        for s in sentences
    ]


# --- cues ----------------------------------------------------------------------


def char_width(ch: str) -> float:
    return 1.0 if ord(ch) >= 0x2E80 else 0.5


def _break_score(text: str, times: list[tuple[float, float]], k: int) -> float:
    """How good it is to end a cue before character ``k`` (0 = not allowed:
    inside a Latin word)."""
    prev, nxt = text[k - 1], text[k]
    if prev in SENTENCE_END or prev in SOFT_BREAK:
        return 3.0
    if times[k][0] - times[k - 1][1] >= PAUSE_SEC:
        return 2.0
    if nxt == " " or prev == " ":
        return 1.0
    if prev.isascii() and prev.isalnum() and nxt.isascii() and nxt.isalnum():
        return 0.0
    return 0.5


def sentence_cues(sentence: Sentence) -> list[Cue]:
    """One sentence as cues of at most ``MAX_LINE_WIDTH`` and
    ``MAX_CUE_SEC``, breaking at punctuation, then pauses, then spaces."""
    text, times = sentence.text, sentence.times
    cues: list[Cue] = []
    begin = 0
    while begin < len(text):
        width, end = 0.0, begin
        while end < len(text):
            width += char_width(text[end])
            too_long = times[end][1] - times[begin][0] > MAX_CUE_SEC
            if (width > MAX_LINE_WIDTH or too_long) and end > begin:
                break
            end += 1
        if end < len(text):
            # Best break within the fitting range, later wins ties; breaks
            # leaving the cue shorter than MIN_CUE_SEC only when no other.
            scored = [
                (times[k - 1][1] - times[begin][0] >= MIN_CUE_SEC, score, k)
                for k in range(begin + 1, end + 1)
                if (score := _break_score(text, times, k)) > 0
            ]
            if scored:
                end = max(scored)[2]
        piece = text[begin:end].strip().rstrip("，、；：")
        if piece:
            cues.append(Cue(times[begin][0], times[end - 1][1], piece))
        begin = end
    return cues


def make_cues(sentences: list[Sentence]) -> list[Cue]:
    """Cues for all sentences: each at least ``MIN_CUE_SEC`` long where the
    next cue leaves room, at most ``MAX_CUE_SEC``, never overlapping."""
    raw = [c for s in sentences for c in sentence_cues(s)]
    raw.sort(key=lambda c: c.start)
    out: list[Cue] = []
    for k, c in enumerate(raw):
        limit = raw[k + 1].start if k + 1 < len(raw) else float("inf")
        end = min(max(c.end, c.start + MIN_CUE_SEC), limit, c.start + MAX_CUE_SEC)
        out.append(Cue(c.start, max(end, c.start), c.text))
    return out


def map_cues(
    cues_by_video: dict[str, list[Cue]], clips: list[dict[str, Any]]
) -> list[Cue]:
    """Cues on an edit timeline: ``clips`` (``video``, ``src_start``,
    ``duration``) play back to back. A cue is trimmed to the clip it falls
    in and dropped when less than ``MIN_KEPT_SEC`` or ``MIN_KEPT_SHARE`` of
    it remains."""
    out: list[Cue] = []
    offset = 0.0
    for clip in clips:
        a = clip["src_start"]
        b = a + clip["duration"]
        for c in cues_by_video.get(clip["video"], []):
            start, end = max(c.start, a), min(c.end, b)
            kept = end - start
            if kept < MIN_KEPT_SEC or kept < MIN_KEPT_SHARE * (c.end - c.start):
                continue
            out.append(Cue(start - a + offset, end - a + offset, c.text))
        offset += clip["duration"]
    return out


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    return "".join(
        f"{n}\n{_srt_time(c.start)} --> {_srt_time(c.end)}\n{c.text}\n\n"
        for n, c in enumerate(cues, start=1)
    )


def write_subtitles(
    path: Path,
    vid: str,
    asr_model: str,
    raw: list[Sentence],
    refined: list[Sentence],
    seconds: float,
    failed: list[str],
) -> list[Cue]:
    cues = make_cues(refined)
    data = {
        "version": SUBTITLE_VERSION,
        "video_id": vid,
        "asr_model": asr_model,
        "refine_seconds": round(seconds, 1),
        "failed_attempts": failed,
        "raw": {s.id: s.text for s in raw},
        "sentences": sentences_to_json(refined),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return cues


def read_subtitles(path: Path, vid: str) -> dict[str, Any]:
    data = json.loads(path.read_text("utf-8"))
    if data["version"] != SUBTITLE_VERSION or data["video_id"] != vid:
        raise SubtitleError(f"{path} is stale; rerun subtitle with --force")
    return data


def read_cues(path: Path, vid: str) -> list[Cue]:
    """Cues rebuilt from the stored sentences, so changing the breaking
    rules needs no new refinement."""
    data = read_subtitles(path, vid)
    return make_cues(sentences_from_json(data["sentences"]))


# --- speech for retrieval ------------------------------------------------------


def speech_text(sentences: list[Sentence], start: float, end: float) -> str:
    """The characters spoken within [start, end), sentences joined by a
    space."""
    parts = []
    for s in sentences:
        if s.end <= start or s.start >= end:
            continue
        spoken = zip(s.text, s.times, strict=True)
        piece = "".join(ch for ch, (a, b) in spoken if start <= (a + b) / 2 < end)
        piece = piece.strip()
        if piece:
            parts.append(piece)
    return " ".join(parts)


def speech_digest(texts: list[str]) -> str:
    """Fingerprint of the speech an index was built from."""
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
