"""Narration for a cut. The 4B writes short lines for the clips actually
chosen, each placed where nobody speaks and sized to the time it has
there; a TTS voices them and the preview mixes them over the original
sound, which is turned down while a line plays."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from video_demo import outline, tag
from video_demo.outline import OutlineError

if TYPE_CHECKING:
    from video_demo.category import Profile

# Mandarin narration pace the character budgets assume.
CHARS_PER_SEC = 4.0
# A line starts this long after its clip does, and ends this long before the
# next speech or line.
LINE_LEAD_SEC = 0.3
LINE_GAP_SEC = 0.2
# Free time shorter than this gets no line; a line never runs longer than
# MAX_LINE_SEC.
MIN_WINDOW_SEC = 1.5
MAX_LINE_SEC = 7.0
MAX_LINE_CHARS = 28
# About one line per this many seconds of cut.
SEC_PER_LINE = 8.0
# Lines may not leave a stretch longer than this unspoken while a clip in
# it has room for one.
MAX_SILENCE_SEC = 2 * SEC_PER_LINE
# Frames of its clip a line is checked against; a fixed line is checked
# again, at most this many rounds in all.
CHECK_FRAMES = 3
CHECK_ROUNDS = 2
# Lines may be shortened this many times when the voice runs over.
SHORTEN_TRIES = 2
_LATIN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class Slot:
    """Where a line may start: ``clip`` (from 1), at ``start`` seconds of
    the cut, with ``window`` seconds before speech or the end."""

    clip: int
    start: float
    window: float


def budget(seconds: float) -> int:
    """Characters a line may have for ``seconds`` of free time."""
    return int(min(seconds, MAX_LINE_SEC) * CHARS_PER_SEC)


def speech_on_timeline(
    clips: list[dict[str, Any]], speech: dict[str, list[tuple[float, float]]]
) -> list[tuple[float, float]]:
    """Spans of the cut where someone speaks: each video's sentence spans
    (``speech``) inside the clips that play it, on the cut's clock."""
    out: list[tuple[float, float]] = []
    offset = 0.0
    for clip in clips:
        a, b = clip["src_start"], clip["src_start"] + clip["duration"]
        for s, e in speech.get(clip["video"], []):
            lo, hi = max(s, a), min(e, b)
            if lo < hi:
                out.append((lo - a + offset, hi - a + offset))
        offset += clip["duration"]
    return sorted(out)


def free_slots(
    clips: list[dict[str, Any]], spoken: list[tuple[float, float]]
) -> list[Slot]:
    """A slot at each clip whose start (plus LINE_LEAD_SEC) is outside
    speech and has at least MIN_WINDOW_SEC before the next speech or the
    end of the cut."""
    total = sum(c["duration"] for c in clips)
    slots = []
    offset = 0.0
    for number, clip in enumerate(clips, start=1):
        start = offset + LINE_LEAD_SEC
        offset += clip["duration"]
        if any(s <= start < e for s, e in spoken):
            continue
        end = min([s for s, _ in spoken if s > start] + [total])
        window = end - start - LINE_GAP_SEC
        if window >= MIN_WINDOW_SEC:
            slots.append(Slot(number, round(start, 3), round(window, 3)))
    return slots


def line_range(total_sec: float, n_slots: int) -> tuple[int, int]:
    """(fewest, most) lines for a cut of ``total_sec`` with ``n_slots``: at
    most about one per SEC_PER_LINE; at least as many as it takes to leave
    no silence longer than MAX_SILENCE_SEC (which validate_lines checks
    where it counts)."""
    most = max(1, min(n_slots, round(total_sec / SEC_PER_LINE) + 1))
    fewest = math.ceil(total_sec / MAX_SILENCE_SEC)
    return max(1, min(fewest, most)), most


NARRATION_PROMPT = """下面是一段 {total:.0f} 秒成片的镜头表，按播放顺序排列，
素材类型是「{name}」。
每行：[镜头号] 开始时间 | 所在部分 | 画面 | 这里最多能说的字数
标“不能放”的镜头那里有人说话或时间太短，不能挂解说。

{rows}
{pictures}
请为这段成片写解说词。风格：{style}

要求：
1. 写 {fewest}–{most} 句，每句挂在一个镜头上（clip 写镜头号），镜头号从小到大、
   不重复，不能挂在标“不能放”的镜头上；相邻两句、开头到第一句、最后一句到
   结尾都不要空出 {silence:.0f} 秒以上（中间的镜头都标“不能放”时除外）；
2. 每句的字数（含标点）不超过那个镜头标的字数，越短越好，宁可少说；
   一句挂上去以后，它后面的镜头要留够时间给它说完；
3. 只用中文，不要英文字母；地名、建筑名只能用镜头表里出现过的；
   时段和光线照镜头表说，不要编画面里没有的事；
4. 第一句是开场，要一下抓住人；最后一句收尾，留一点余味；
5. 整段连起来像带人走过一条路线，前后句有衔接，不要写成互不相干的图片说明。
只输出 JSON。"""


def narration_schema(most: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "minItems": 1,
                "maxItems": most,
                "items": {
                    "type": "object",
                    "properties": {
                        "clip": {"type": "integer", "minimum": 1},
                        "text": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_LINE_CHARS,
                        },
                    },
                    "required": ["clip", "text"],
                },
            }
        },
        "required": ["lines"],
    }


def clip_rows(
    clips: list[dict[str, Any]], tags: dict[tuple[str, int], dict], slots: list[Slot]
) -> str:
    """The shot list the model reads: what each clip shows (with its
    landmark, time of day and light when tags have them) and how many
    characters fit there."""
    by_clip = {s.clip: s for s in slots}
    rows = []
    offset = 0.0
    for number, clip in enumerate(clips, start=1):
        t = tags[(clip["video"], clip["index"])]
        seen = _seen(t)
        room = f"{budget(by_clip[number].window)} 字" if number in by_clip else "不能放"
        rows.append(f"[{number}] {offset:.1f}s | {clip['role']} | {seen} | {room}")
        offset += clip["duration"]
    return "\n".join(rows)


def silences(
    numbers: list[int], slots: list[Slot], total: float
) -> list[tuple[float, float, list[int]]]:
    """Stretches longer than MAX_SILENCE_SEC that the lines on clips
    ``numbers`` leave unspoken (from the start, between lines, to the end
    of a ``total`` second cut) while clips inside them have room: (from,
    to, those clips)."""
    by_clip = {s.clip: s for s in slots}
    starts = [0.0, *(by_clip[n].start for n in numbers), total]
    out = []
    for a, b in zip(starts, starts[1:], strict=False):
        free = [s.clip for s in slots if a < s.start < b and s.clip not in numbers]
        if b - a > MAX_SILENCE_SEC and free:
            out.append((a, b, free))
    return out


def _seen(tags: dict[str, Any]) -> str:
    """A clip as the model reads it: its description, with its landmark,
    time of day and light when the tags have them."""
    notes = [tags.get(k) for k in ("landmark", "time_of_day", "light")]
    return tags["description"] + "".join(f"（{x}）" for x in notes if x)


def validate_lines(
    answer: Any, slots: list[Slot], fewest: int, most: int
) -> list[dict[str, Any]]:
    """Lines with their start and window (until the next line or speech);
    raises OutlineError when any rule is broken. Silences are filled
    afterwards (fill_silences)."""
    if not isinstance(answer, dict) or not isinstance(answer.get("lines"), list):
        raise OutlineError("Expected an object with a lines list")
    lines = answer["lines"]
    if not fewest <= len(lines) <= most:
        raise OutlineError(f"{len(lines)} lines, need {fewest}-{most}")
    by_clip = {s.clip: s for s in slots}
    numbers = [line.get("clip") for line in lines]
    if any(n not in by_clip for n in numbers):
        bad = [n for n in numbers if n not in by_clip]
        raise OutlineError(f"clips {bad} have no room for narration")
    if numbers != sorted(set(numbers)):
        raise OutlineError(f"clips must increase without repeats: {numbers}")
    out = []
    for k, line in enumerate(lines):
        slot = by_clip[line["clip"]]
        text = str(line.get("text", "")).strip()
        if not text:
            raise OutlineError(f"line on clip {slot.clip} is empty")
        if _LATIN.search(text):
            raise OutlineError(f"line on clip {slot.clip} has Latin letters: {text}")
        window = slot.window
        if k + 1 < len(lines):
            nxt = by_clip[lines[k + 1]["clip"]]
            window = min(window, nxt.start - slot.start - LINE_GAP_SEC)
        # The shot list shows each clip's own budget; running into the next
        # line is settled by the voice's measured length (voice_lines).
        if len(text) > budget(slot.window):
            raise OutlineError(
                f"line on clip {slot.clip} has {len(text)} characters, "
                f"{budget(slot.window)} fit before speech or the end"
            )
        out.append(
            {
                **{k: v for k, v in line.items() if k == "filled"},
                "clip": slot.clip,
                "start": slot.start,
                "window": round(window, 3),
                "text": text,
            }
        )
    return out


PICTURES_NOTE = """
每个镜头后面附了一张它的截图（按镜头号）。镜头表的文字是自动打的标签，
可能有错（比如把阴天标成黄昏）；和截图不一致时以截图为准。
"""


def _shown(
    clips: list[dict[str, Any]], frames: dict[tuple[str, int], list[Path]] | None
) -> list[Path] | None:
    """One frame per clip (the middle keyframe of its segment), or None."""
    if frames is None:
        return None
    return [
        frames[(c["video"], c["index"])][len(frames[(c["video"], c["index"])]) // 2]
        for c in clips
    ]


def _with_pictures(text: str, labelled: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    """Message content: ``text``, then each picture after its label."""
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for label, path in labelled:
        parts += [{"type": "text", "text": label}, tag.image_part(path)]
    return parts


def write_lines(
    clips: list[dict[str, Any]],
    tags: dict[tuple[str, int], dict],
    slots: list[Slot],
    profile: Profile,
    server_url: str = outline.SERVER_URL,
    write_one: Any = None,
    frames: dict[tuple[str, int], list[Path]] | None = None,
) -> tuple[list[dict[str, Any]], str, float, list[str]]:
    """(lines, raw answer, seconds, failed tries); raises OutlineError when
    no attempt passes validate_lines. With ``frames`` (keyframes by video
    and segment index) the model also sees a frame of every clip. Silences
    the model left are then filled one line at a time (fill_silences, with
    ``write_one``)."""
    if not slots:
        raise OutlineError("no clip has room for narration")
    total = sum(c["duration"] for c in clips)
    fewest, most = line_range(total, len(slots))
    prompt = NARRATION_PROMPT.format(
        total=total,
        name=profile.name,
        rows=clip_rows(clips, tags, slots),
        style=profile.narration,
        fewest=fewest,
        most=most,
        silence=MAX_SILENCE_SEC,
        pictures=PICTURES_NOTE if frames else "",
    )
    shown = _shown(clips, frames)
    content = (
        _with_pictures(prompt, [(f"[{k}]", p) for k, p in enumerate(shown, start=1)])
        if shown
        else prompt
    )
    lines, raw, seconds, errors = outline._ask(
        content,
        narration_schema(most),
        "narration",
        lambda parsed: validate_lines(parsed, slots, fewest, most),
        server_url,
    )
    lines, more = fill_silences(
        lines, clips, tags, slots, profile, write_one or write_one_line, shown
    )
    return lines, raw, seconds + more, errors


FILL_PROMPT = """一段 {total:.0f} 秒成片的解说在 {a:.0f}–{b:.0f} 秒之间太久没说话，
请为第 {clip} 个镜头（{at:.1f} 秒开始）补一句解说。
这个镜头：{seen}
前一句：{before}
后一句：{after}
风格：{style}
要求：和前后两句接得上，只说这个镜头里能确认的事；不超过 {limit} 个字
（含标点），只用中文。只输出 JSON。"""


def write_one_line(
    prompt: str | list[dict[str, Any]], limit: int, server_url: str = outline.SERVER_URL
) -> tuple[str, float]:
    """One line of at most ``limit`` Chinese characters for FILL_PROMPT
    (text, or text and the clip's picture)."""
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1, "maxLength": limit}},
        "required": ["text"],
    }

    def check(parsed: Any) -> str:
        text = _answer_text(parsed)
        if not text or len(text) > limit or _LATIN.search(text):
            raise OutlineError(f"not a usable line of at most {limit}: {text!r}")
        return text

    text, _, seconds, _ = outline._ask(prompt, schema, "fill_line", check, server_url)
    return text, seconds


def fill_silences(
    lines: list[dict[str, Any]],
    clips: list[dict[str, Any]],
    tags: dict[tuple[str, int], dict],
    slots: list[Slot],
    profile: Profile,
    write_one: Any = write_one_line,
    shown: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """``lines`` with a line added in every stretch longer than
    MAX_SILENCE_SEC that has a clip with room: on the free clip nearest its
    middle, written to follow the lines around it (and looking at its
    picture, ``shown`` holding one per clip). Returns the lines (added ones
    marked ``filled``) and the model seconds."""
    total = sum(c["duration"] for c in clips)
    by_clip = {s.clip: s for s in slots}
    seconds = 0.0
    while gaps := silences([x["clip"] for x in lines], slots, total):
        a, b, free = gaps[0]
        slot = by_clip[min(free, key=lambda n: abs(by_clip[n].start - (a + b) / 2))]
        before = [x for x in lines if x["clip"] < slot.clip]
        after = [x for x in lines if x["clip"] > slot.clip]
        window = slot.window
        if after:
            window = min(window, after[0]["start"] - slot.start - LINE_GAP_SEC)
        clip = clips[slot.clip - 1]
        prompt = FILL_PROMPT.format(
            total=total,
            a=a,
            b=b,
            clip=slot.clip,
            at=slot.start,
            seen=_seen(tags[(clip["video"], clip["index"])]),
            before=f"「{before[-1]['text']}」" if before else "无（这是开头）",
            after=f"「{after[0]['text']}」" if after else "无（这是结尾）",
            style=profile.narration,
            limit=budget(window),
        )
        content = (
            _with_pictures(
                prompt,
                [
                    (
                        "这个镜头的截图（和文字不一致时以截图为准）：",
                        shown[slot.clip - 1],
                    )
                ],
            )
            if shown
            else prompt
        )
        text, elapsed = write_one(content, budget(window))
        seconds += elapsed
        added = {"clip": slot.clip, "text": text, "filled": True}
        lines = validate_lines(
            {"lines": [*before, added, *after]}, slots, 1, len(slots)
        )
    return lines, seconds


CHECK_PROMPT = """这几张图按时间顺序取自成片里的一个镜头。镜头表对它的描述：
{seen}

给这个镜头写的解说词：「{text}」

请对照画面核对这句解说，只找两类问题：
A. 和画面矛盾的说法：比如画面是白天却说黄昏，画面里没有的东西说成有；
B. 画面和描述都确认不了的事实：编造的经历、数字、地名或建筑名。
下面这些不算问题：解说没提到画面里的某样东西；比喻、感受和想象（“好像”
“仿佛”）；和画面里确认的地标直接相关的一句历史常识。

没有问题：verdict 写 ok，problems 为空列表。
有问题：verdict 写 fix，problems 逐条写出问题。只输出 JSON。"""

CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "fix"]},
        "problems": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "minLength": 1, "maxLength": 40},
        },
    },
    "required": ["verdict", "problems"],
}


def validate_check(answer: Any) -> list[str]:
    """The problems the model found in a line ([] when its verdict is ok);
    a fix must name at least one."""
    if not isinstance(answer, dict) or answer.get("verdict") not in ("ok", "fix"):
        raise OutlineError("Expected an object with verdict ok or fix")
    if answer["verdict"] == "ok":
        return []
    problems = answer.get("problems")
    if not isinstance(problems, list) or not all(
        isinstance(p, str) and p.strip() for p in problems
    ):
        raise OutlineError("problems must be a list of strings")
    if not problems:
        raise OutlineError("a fix must name its problems")
    return [p.strip() for p in problems]


FIX_PROMPT = """这几张图取自成片里的一个镜头。镜头表对它的描述（自动打的标签，
可能有错，和图不一致时以图为准）：
{seen}

给它写的解说词：「{text}」
核对画面时发现的问题：
{problems}

请只改有问题的部分，其余措辞和语气保留，改成画面里确实有的内容；
确认不了的地名、建筑名直接删掉或换成描述里的说法，不要改成别的名字；
不超过 {limit} 个字（含标点），只用中文。只输出 JSON。"""


def _answer_text(answer: Any) -> str:
    """The ``text`` of a JSON answer, without spaces or the quote marks
    the prompts put around a line."""
    text = str(answer.get("text", "")) if isinstance(answer, dict) else ""
    return text.strip().strip("「」“”\"'").strip()


def validate_fix(answer: Any, limit: int) -> str:
    """The rewritten line: 1..limit Chinese characters. It may equal the
    line it rewrites: looking at the frames, the rewrite can find nothing
    wrong (check_line keeps the line then)."""
    text = _answer_text(answer)
    if not text or len(text) > limit:
        raise OutlineError(f"fixed line must have 1-{limit} characters: {text!r}")
    if _LATIN.search(text):
        raise OutlineError(f"fixed line has Latin letters: {text}")
    return text


def check_line(
    line: dict[str, Any],
    frames: list[Path],
    seen: str,
    server_url: str = outline.SERVER_URL,
) -> tuple[dict[str, Any], float]:
    """The 4B looks at the clip's frames and lists what the line says that
    the picture does not show; when it finds something, a second call
    rewrites just that, and the rewrite is checked again (CHECK_ROUNDS in
    all; what the last round finds is fixed but not checked again). A
    rewrite that, with the frames in view, leaves the line as it was
    overrules the check. Returns ({problems of every round, text},
    seconds)."""
    limit = budget(line["window"])
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1, "maxLength": limit}},
        "required": ["text"],
    }
    text, found, seconds = line["text"], [], 0.0
    for _ in range(CHECK_ROUNDS):
        prompt = [
            {"type": "text", "text": CHECK_PROMPT.format(seen=seen, text=text)},
            *(tag.image_part(p) for p in frames),
        ]
        problems, _, elapsed, _ = outline._ask(
            prompt, CHECK_SCHEMA, "narration_check", validate_check, server_url
        )
        seconds += elapsed
        if not problems:
            break
        found += problems
        current = text
        fix = FIX_PROMPT.format(
            seen=seen,
            text=current,
            problems="\n".join(f"- {p}" for p in problems),
            limit=limit,
        )
        text, _, elapsed, _ = outline._ask(
            [{"type": "text", "text": fix}, *(tag.image_part(p) for p in frames)],
            schema,
            "narration_fix",
            lambda parsed: validate_fix(parsed, limit),
            server_url,
        )
        seconds += elapsed
        if text == current:
            break
    return {"problems": found, "text": text}, seconds


def check_lines(
    lines: list[dict[str, Any]],
    clips: list[dict[str, Any]],
    tags: dict[tuple[str, int], dict],
    frames: dict[tuple[str, int], list[Path]],
    check: Any = check_line,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Every line checked against its clip's frames (``frames`` by video
    and segment index). Returns the lines as checked, a record per line
    (clip, written, checked, problems) and the model seconds."""
    out, records = [], []
    seconds = 0.0
    for line in lines:
        clip = clips[line["clip"] - 1]
        key = (clip["video"], clip["index"])
        seen = _seen(tags[key])
        picked = _spread(frames[key], CHECK_FRAMES)
        checked, elapsed = check(line, picked, seen)
        seconds += elapsed
        out.append({**line, "text": checked["text"]})
        records.append(
            {
                "clip": line["clip"],
                "written": line["text"],
                "checked": checked["text"],
                "problems": checked["problems"],
            }
        )
    return out, records, seconds


def _spread(items: list[Path], n: int) -> list[Path]:
    """At most ``n`` of ``items``, evenly spaced, first and last kept."""
    if len(items) <= n:
        return list(items)
    return [items[round(k * (len(items) - 1) / (n - 1))] for k in range(n)]


SHORTEN_PROMPT = """把下面这句解说缩短到不超过 {limit} 个字（含标点），意思和语气不变，
只用中文，不要英文字母：
「{text}」
只输出 JSON。"""


def shorten(text: str, limit: int, server_url: str = outline.SERVER_URL) -> str:
    """The line rewritten to at most ``limit`` characters."""
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1, "maxLength": limit}},
        "required": ["text"],
    }

    def check(parsed: Any) -> str:
        short = _answer_text(parsed)
        if not short or len(short) > limit or _LATIN.search(short):
            raise OutlineError(f"not a usable line of at most {limit}: {short!r}")
        return short

    return outline._ask(
        SHORTEN_PROMPT.format(limit=limit, text=text),
        schema,
        "short_line",
        check,
        server_url,
    )[0]


def voice_lines(
    lines: list[dict[str, Any]],
    synthesize: Any,
    rate: int,
    shorten_line: Any = shorten,
) -> list[tuple[dict[str, Any], np.ndarray]]:
    """Each line with its voice (mono at ``rate``). A line whose voice runs
    past its window is shortened (at most SHORTEN_TRIES times, aiming at the
    length that would fit) and voiced again; still too long is an error."""
    out = []
    for line in lines:
        text = line["text"]
        for attempt in range(SHORTEN_TRIES + 1):
            voice = synthesize(text)
            seconds = len(voice) / rate
            if seconds <= line["window"]:
                break
            if attempt == SHORTEN_TRIES:
                raise OutlineError(
                    f"line on clip {line['clip']} still takes {seconds:.1f}s, "
                    f"{line['window']:.1f}s free: {text}"
                )
            limit = max(2, int(len(text) * line["window"] / seconds * 0.9))
            text = shorten_line(text, limit)
        out.append(({**line, "text": text, "seconds": round(seconds, 2)}, voice))
    return out
