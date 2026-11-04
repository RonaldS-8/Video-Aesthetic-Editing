"""Highlights outline: summarise chunks, plan from the summaries, fill by rules.

An hour of footage has hundreds of segments, too many for the model to read
at once (the scene list of a 67-minute walk was ~20k tokens and the 4B model
lost the order of its parts). So the work is split, map-reduce style:

1. Chunks: a long video is cut every CHUNK_SEC; shorter videos (clips
   shot on a trip) are summarised whole, then consecutive ones are grouped
   into chunks of at least CHUNK_SEC, never splitting a video.
2. Chunk summaries: Qwen3.5-4B (llama-server) reads one chunk's (or short
   video's) segment descriptions and returns a short summary, places,
   subjects, activities and up to three highlight segments. A group of
   videos gets a summary written from its videos' summaries. Chunks are
   summarised independently, so nothing earlier is forgotten; summaries
   are cached.
3. Outline: the model reads all chunk summaries (a few hundred tokens) and
   returns parts in shooting order, each covering a range of chunks, with a
   weight and 1-4 scene slots (what each scene should show). Seconds per
   part are allocated from the weights.
4. Fill (rules): each slot is matched to one scene (consecutive segments
   that show the same place, see retrieve.split_scenes) inside its part's
   chunks, by retrieval relevance; the slot's seconds are then taken from
   that scene as whole segments ranked by score, relevance and whether the
   summary named them as highlights.

Every model answer is checked deterministically; a failing answer is retried
and finally raised, never replaced by a default.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import requests

from video_demo.suggest import (
    DUP_JACCARD,
    DUP_NEAR_SEC,
    MIN_CLIP_SEC,
    Assessment,
    Candidate,
    Clip,
    _jaccard,
    validate_timeline,
)

SERVER_URL = "http://127.0.0.1:8092"
TIMEOUT_SEC = 900
TEMPERATURES = (0.2, 0.5, 0.8)

# The calls a planner other than the local model may answer (ab_eval.cloud):
# with PLANNER set to (system, prompt, schema, name, temperature) -> (answer,
# seconds), these go to it instead. Prompts with pictures stay local.
PLANNER_CALLS = frozenset({"edit_outline", "edit_review"})
Planner = Callable[[str, str, dict[str, Any], str, float], tuple[str, float]]
PLANNER: Planner | None = None
# Told each PLANNER_CALLS prompt (system, prompt, schema, name, temperature)
# whoever answers it, so a run can keep what was asked.
RECORDER: Callable[[str, str, dict[str, Any], str, float], None] | None = None


def planner_name() -> str:
    """Where PLANNER_CALLS are answered, for reports."""
    return "Qwen3.5-4B（本地）" if PLANNER is None else "云端"


CHUNK_SEC = 300.0
# Bump when the summary prompt or schema changes, so cached summaries are
# regenerated instead of mixed with new ones.
SUMMARY_VERSION = 2
MAX_HIGHLIGHTS = 3
# 40 cut reasons off mid-sentence in about half the Edinburgh chunks.
MAX_WHY_CHARS = 60

MIN_PARTS, MAX_PARTS = 2, 8
MIN_PART_SEC = 10.0
# The model weighs each part 1-10; seconds are computed from the weights,
# since a 4B model does not add up lengths to a target reliably.
MIN_WEIGHT, MAX_WEIGHT = 1, 10
MAX_SLOTS = 4  # scene slots per part
# Relevance is rescaled among the candidates compared (best 1, worst 0)
# before it is weighed: fused RRF values sit close together (rank 1 in all
# three rankings is 1.0, rank 3 still 0.97), so on Edinburgh quality chose
# the cathedral front over its interior for 彩色玻璃窗.
# Choosing a scene for a slot: rescaled relevance + this x the mean quality
# of its usable segments ((score - 1) / 4); relevance leads.
SCENE_QUALITY_WEIGHT = 0.2
# Ranking inside the chosen scene, on one 0-1 scale: quality ((score - 1)
# / 4) x (1 - share) + relevance to the slot (rescaled among the scene's
# segments) x share, + HIGHLIGHT_BONUS when the chunk summary named the
# segment. Highlights keep the footage most worth keeping, so quality
# leads; 0.2 is the order the earlier score (1-5) + relevance + 0.5 bonus
# gave. An edit by instruction lets relevance lead (instruct.py).
RELEVANCE_SHARE = 0.2
# Checked on Edinburgh: most named segments already rank high without it;
# 0.1 (0.5 on the 1-5 score) brings in the distinct ones (a landmark among
# plain street walks), 0.2 also forces in ones the text-only summary
# overrated.
HIGHLIGHT_BONUS = 0.1


class OutlineError(RuntimeError):
    """The model output is not a usable summary or outline."""


@dataclass(frozen=True)
class Chunk:
    number: int  # 1-based along the timeline, as shown to the model
    videos: list[str]  # one, or a group of whole short videos
    members: list[int]  # indices into the candidate list
    start: float  # on the shared timeline
    end: float
    summary: dict[str, Any]


@dataclass(frozen=True)
class Scene:
    number: int  # 1-based along the timeline
    members: list[int]  # consecutive candidate indices, inside one chunk


def _fmt(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def split_chunks(
    candidates: list[Candidate], chunk_sec: float = CHUNK_SEC
) -> list[list[int]]:
    """Consecutive segments of one video, a new chunk every ``chunk_sec``."""
    groups: list[list[int]] = []
    for i, c in enumerate(candidates):
        if groups:
            first = candidates[groups[-1][0]]
            if c.video == first.video and c.start - first.start < chunk_sec:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups


# --- chunk summaries -------------------------------------------------------

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "minLength": 1, "maxLength": 150},
        "places": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "subjects": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "activities": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "highlights": {
            "type": "array",
            "maxItems": MAX_HIGHLIGHTS,
            "items": {
                "type": "object",
                "properties": {
                    "segment": {"type": "integer", "minimum": 1},
                    "why": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_WHY_CHARS,
                    },
                },
                "required": ["segment", "why"],
            },
        },
    },
    "required": ["summary", "places", "subjects", "activities", "highlights"],
}

SYSTEM_PROMPT = (
    "你是视频剪辑师，负责从原始素材里整理精彩瞬间。"
    "只根据给出的描述，不要编造素材里没有的东西。所有文字用简体中文。"
)

SUMMARY_PROMPT = """下面是一段视频素材中连续 {minutes:.1f} 分钟（{start}–{end}）的片段，
按时间排列。
每行：[片段编号] 时间 | 时段 | 得分（1–5，越高画面越好、越精彩） | 描述 | 标签

{lines}

请总结这一段：
- summary 用 2–4 句话说明这段里发生了什么、在什么地方；
- places、subjects、activities 分别列出主要地点、人物或主体、活动，各不超过 5 个；
- highlights 选出最值得保留的 0–3 个片段（segment 写上面的片段编号），
  why 用一句话说明理由，不超过 {max_why} 个字。
只输出 JSON。"""


def segment_lines(
    members: list[int], candidates: list[Candidate], assessments: list[Assessment]
) -> str:
    """One line per segment, numbered 1.. within the chunk."""
    return "\n".join(
        f"[{n}] {_fmt(candidates[i].start)} | {candidates[i].tags['time_of_day']} | "
        f"{assessments[i].score:.1f} | {_described(candidates[i].tags)} | "
        f"{'、'.join(candidates[i].tags['tags'])}"
        for n, i in enumerate(members, start=1)
    )


def _described(tags: dict[str, Any]) -> str:
    """The description, with the landmark when category tags name one."""
    landmark = tags.get("landmark")
    return f"{tags['description']}（{landmark}）" if landmark else tags["description"]


def validate_summary(summary: Any, n_segments: int) -> dict[str, Any]:
    if not isinstance(summary, dict) or not str(summary.get("summary", "")).strip():
        raise OutlineError("summary must be a non-empty string")
    seen: set[int] = set()
    for h in summary.get("highlights", []):
        n = h.get("segment")
        if not isinstance(n, int) or not 1 <= n <= n_segments:
            raise OutlineError(f"highlight segment {n} outside 1-{n_segments}")
        if n in seen:
            raise OutlineError(f"highlight segment {n} listed twice")
        seen.add(n)
    return summary


def summarize_chunk(
    members: list[int],
    candidates: list[Candidate],
    assessments: list[Assessment],
    server_url: str = SERVER_URL,
) -> tuple[dict[str, Any], str, float, list[str]]:
    """(summary, raw, seconds, failed tries) for one chunk."""
    first, last = candidates[members[0]], candidates[members[-1]]
    prompt = SUMMARY_PROMPT.format(
        minutes=(last.end - first.start) / 60,
        start=_fmt(first.start),
        end=_fmt(last.end),
        lines=segment_lines(members, candidates, assessments),
        max_why=MAX_WHY_CHARS,
    )
    return _ask(
        prompt,
        SUMMARY_SCHEMA,
        "chunk_summary",
        lambda parsed: validate_summary(parsed, len(members)),
        server_url,
    )


# --- groups of short videos ----------------------------------------------------


def group_videos(lengths: list[float], chunk_sec: float = CHUNK_SEC) -> list[list[int]]:
    """Positions of the videos (in timeline order) that form each chunk.

    A video of ``chunk_sec`` or more stands alone (it is cut into chunks of
    its own); shorter ones are grouped with the next until the group holds
    ``chunk_sec``. A long video closes the group before it; what is left at
    the end is a group too. A video is never split between groups.
    """
    groups: list[list[int]] = []
    open_group: list[int] = []
    held = 0.0
    for k, length in enumerate(lengths):
        if length >= chunk_sec:
            if open_group:
                groups.append(open_group)
            groups.append([k])
            open_group, held = [], 0.0
            continue
        open_group.append(k)
        held += length
        if held >= chunk_sec:
            groups.append(open_group)
            open_group, held = [], 0.0
    if open_group:
        groups.append(open_group)
    return groups


GROUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        k: v for k, v in SUMMARY_SCHEMA["properties"].items() if k != "highlights"
    },
    "required": ["summary", "places", "subjects", "activities"],
}

GROUP_PROMPT = """下面是连续拍摄的 {n} 段视频（共 {minutes:.1f} 分钟），按时间排列，
每段已有总结。
每行：[视频序号] 时长 | 总结 | 地点 | 活动

{lines}

请把这几段合起来总结：
- summary 用 2–4 句话说明这些视频里依次发生了什么、在什么地方；
- places、subjects、activities 分别列出主要地点、人物或主体、活动，各不超过 5 个。
只输出 JSON。"""


def validate_group(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict) or not str(summary.get("summary", "")).strip():
        raise OutlineError("summary must be a non-empty string")
    return summary


def summarize_group(
    videos: list[tuple[float, dict[str, Any]]], server_url: str = SERVER_URL
) -> tuple[dict[str, Any], str, float, list[str]]:
    """(summary, raw, seconds, failed tries) for a group of short videos,
    each given as (length in seconds, its summary)."""
    lines = "\n".join(
        f"[{n}] {_fmt(length)} | {s['summary']} | "
        f"{'、'.join(s['places']) or '—'} | {'、'.join(s['activities']) or '—'}"
        for n, (length, s) in enumerate(videos, start=1)
    )
    prompt = GROUP_PROMPT.format(
        n=len(videos), minutes=sum(x for x, _ in videos) / 60, lines=lines
    )
    return _ask(prompt, GROUP_SCHEMA, "group_summary", validate_group, server_url)


def group_highlights(
    videos: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """The videos' highlights renumbered along the group's segments; each
    video is given as (its number of segments, its summary)."""
    out: list[dict[str, Any]] = []
    before = 0
    for n_segments, summary in videos:
        out += [{**h, "segment": before + h["segment"]} for h in summary["highlights"]]
        before += n_segments
    return out


# --- outline -----------------------------------------------------------------

OUTLINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 30},
        "parts": {
            "type": "array",
            "minItems": MIN_PARTS,
            "maxItems": MAX_PARTS,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 20},
                    "summary": {"type": "string", "minLength": 1, "maxLength": 80},
                    "first_chunk": {"type": "integer", "minimum": 1},
                    "last_chunk": {"type": "integer", "minimum": 1},
                    "weight": {
                        "type": "integer",
                        "minimum": MIN_WEIGHT,
                        "maximum": MAX_WEIGHT,
                    },
                    "scenes": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 20},
                        "minItems": 1,
                        "maxItems": MAX_SLOTS,
                    },
                },
                "required": [
                    "name",
                    "summary",
                    "first_chunk",
                    "last_chunk",
                    "weight",
                    "scenes",
                ],
            },
        },
    },
    "required": ["title", "parts"],
}

OUTLINE_PROMPT = """下面是一段 {total_min:.1f} 分钟原始素材按时间分成的 {n} 块，
每块已有总结。
每行：[块编号] 起止时间 | 总结 | 地点 | 活动

{chunks}

请把它整理成约 {target_min:.1f} 分钟（{target_sec:.0f} 秒）的精彩瞬间大纲，
供用户进一步剪辑：
- 分成 {min_parts}–{max_parts} 个部分（不能超过 {max_parts} 个），按时间顺序排列，
  每部分覆盖连续的几块（first_chunk 到 last_chunk），部分之间不重叠，
  没意思的块可以跳过；
- name 是这一部分的标题，summary 用一句话说明这部分讲什么；
- weight 是这一部分的分量（1–10 的整数），内容丰富、精彩的部分给高分；
  程序会按分量把 {target_sec:.0f} 秒分给各部分，每部分至少 {min_part:.0f} 秒；
- scenes 是这一部分要的 1–{max_slots} 个场景，每个用一个短语说明画面
  （如“远处的城堡”“教堂里的彩色玻璃窗”），从这几块的总结、地点和活动里选，
  互不重复；程序会为每个场景从素材里找一段连续的场景。
只输出 JSON。"""


def part_limits(target_sec: float) -> tuple[float, int]:
    """(shortest part, most parts) for a target length.

    The rule is the long-footage one: parts of at least MIN_PART_SEC, 2-8 of
    them; it applies unchanged from an 80 s target up (1 h at 1/6 -> 10 min).
    Below that, as a stopgap so the few-minute test vlogs can be outlined,
    the number of parts is capped by how many shortest parts fit, and under
    a 20 s target the shortest part becomes half the target.
    """
    min_part = min(MIN_PART_SEC, target_sec / MIN_PARTS)
    return min_part, max(MIN_PARTS, min(MAX_PARTS, int(target_sec // min_part)))


def parts_range(target_sec: float, n_chunks: int) -> tuple[int, int]:
    """(fewest, most) parts: part_limits, and no more parts than chunks
    (a short video is a single chunk, which makes a one-part outline)."""
    most = min(part_limits(target_sec)[1], n_chunks)
    return min(MIN_PARTS, most), most


def outline_schema(min_parts: int, max_parts: int) -> dict[str, Any]:
    """The schema with this outline's part limits, so decoding enforces them."""
    schema = json.loads(json.dumps(OUTLINE_SCHEMA))
    schema["properties"]["parts"]["minItems"] = min_parts
    schema["properties"]["parts"]["maxItems"] = max_parts
    return schema


def allocate(weights: list[int], target_sec: float) -> list[int]:
    """Whole seconds per part: the shortest part each, the rest by weight.

    Sums to round(target_sec) exactly (largest remainder); every part gets
    at least part_limits(target_sec)[0], rounded down.
    """
    min_part, _ = part_limits(target_sec)
    total = round(target_sec)
    base = [math.floor(min_part)] * len(weights)
    spare = total - sum(base)
    if spare < 0:
        raise ValueError(f"{len(weights)} parts do not fit in {target_sec:.0f} s")
    shares = [spare * w / sum(weights) for w in weights]
    extra = [math.floor(x) for x in shares]
    order = sorted(range(len(weights)), key=lambda i: -(shares[i] - extra[i]))
    for i in order[: spare - sum(extra)]:
        extra[i] += 1
    return [b + e for b, e in zip(base, extra, strict=True)]


def chunk_lines(chunks: list[Chunk]) -> str:
    return "\n".join(
        f"[{c.number}] {_fmt(c.start)}–{_fmt(c.end)} | {c.summary['summary']} | "
        f"{'、'.join(c.summary['places']) or '—'} | "
        f"{'、'.join(c.summary['activities']) or '—'}"
        for c in chunks
    )


def validate_outline(outline: Any, n_chunks: int, target_sec: float) -> dict[str, Any]:
    """Deterministic checks on the model's outline; raises OutlineError.

    Adds each part's ``duration_sec``, allocated from the weights.
    """
    if not isinstance(outline, dict) or not isinstance(outline.get("parts"), list):
        raise OutlineError("Expected an object with a parts list")
    parts = outline["parts"]
    min_parts, max_parts = parts_range(target_sec, n_chunks)
    if not min_parts <= len(parts) <= max_parts:
        raise OutlineError(f"{len(parts)} parts, need {min_parts}-{max_parts}")
    last = 0
    for k, p in enumerate(parts, start=1):
        first_c, last_c = p.get("first_chunk"), p.get("last_chunk")
        if not isinstance(first_c, int) or not isinstance(last_c, int):
            raise OutlineError(f"part {k}: chunk numbers must be integers")
        if not 1 <= first_c <= last_c <= n_chunks:
            raise OutlineError(
                f"part {k}: chunks {first_c}-{last_c} outside 1-{n_chunks}"
            )
        if first_c <= last:
            raise OutlineError(f"part {k}: chunks overlap or are out of order")
        last = last_c
        if not str(p.get("name", "")).strip() or not str(p.get("summary", "")).strip():
            raise OutlineError(f"part {k}: name and summary must not be empty")
        scenes = p.get("scenes")
        if not isinstance(scenes, list) or not 1 <= len(scenes) <= MAX_SLOTS:
            raise OutlineError(f"part {k}: needs 1-{MAX_SLOTS} scenes")
        if any(not str(q).strip() for q in scenes):
            raise OutlineError(f"part {k}: empty scene")
        if len(set(scenes)) < len(scenes):
            raise OutlineError(f"part {k}: scene listed twice")
        weight = p.get("weight")
        if not isinstance(weight, int) or not MIN_WEIGHT <= weight <= MAX_WEIGHT:
            raise OutlineError(f"part {k}: weight must be {MIN_WEIGHT}-{MAX_WEIGHT}")
    seconds = allocate([p["weight"] for p in parts], target_sec)
    for p, sec in zip(parts, seconds, strict=True):
        p["duration_sec"] = sec
    return outline


def plan_outline(
    chunks: list[Chunk],
    target_sec: float,
    server_url: str = SERVER_URL,
) -> tuple[dict[str, Any], str, float, list[str]]:
    """(outline, raw, seconds, failed tries) from the chunk summaries."""
    min_part, _ = part_limits(target_sec)
    min_parts, max_parts = parts_range(target_sec, len(chunks))
    prompt = OUTLINE_PROMPT.format(
        total_min=(chunks[-1].end - chunks[0].start) / 60,
        n=len(chunks),
        chunks=chunk_lines(chunks),
        target_min=target_sec / 60,
        target_sec=target_sec,
        min_parts=min_parts,
        max_parts=max_parts,
        min_part=min_part,
        max_slots=MAX_SLOTS,
    )
    return _ask(
        prompt,
        outline_schema(min_parts, max_parts),
        "highlights_outline",
        lambda parsed: validate_outline(parsed, len(chunks), target_sec),
        server_url,
    )


# --- model calls ---------------------------------------------------------------


def _ask(
    prompt: str | list[dict[str, Any]],
    schema: dict[str, Any],
    name: str,
    validate: Any,
    server_url: str,
) -> tuple[dict[str, Any], str, float, list[str]]:
    """Ask with each temperature until the answer validates; a retry is told
    why the answer before it failed.

    ``prompt`` is text, or text and image parts (OpenAI message content).
    Raises OutlineError listing every attempt when all fail.
    """
    errors: list[str] = []
    seconds = 0.0
    asked = prompt
    for temperature in TEMPERATURES:
        raw, elapsed = _request(asked, temperature, schema, name, server_url)
        seconds += elapsed
        try:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise OutlineError(f"Model output is not JSON: {raw[:200]!r}") from exc
            return validate(parsed), raw, seconds, errors
        except OutlineError as exc:
            errors.append(f"temperature={temperature}: {exc}")
            asked = _with_error(prompt, str(exc))
    raise OutlineError("；".join(errors))


RETRY_NOTE = (
    "\n\n上一次的回答没有通过检查：{error}\n请改正这个问题，重新输出完整的 JSON。"
)


def _with_error(
    prompt: str | list[dict[str, Any]], error: str
) -> str | list[dict[str, Any]]:
    """``prompt`` with the reason the last answer failed added at its end."""
    note = RETRY_NOTE.format(error=error)
    if isinstance(prompt, str):
        return prompt + note
    return [*prompt, {"type": "text", "text": note.strip()}]


def _request(
    prompt: str | list[dict[str, Any]],
    temperature: float,
    schema: dict[str, Any],
    name: str,
    server_url: str,
) -> tuple[str, float]:
    if name in PLANNER_CALLS and isinstance(prompt, str):
        if RECORDER is not None:
            RECORDER(SYSTEM_PROMPT, prompt, schema, name, temperature)
        if PLANNER is not None:
            return PLANNER(SYSTEM_PROMPT, prompt, schema, name, temperature)
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2000,
        # Qwen3.5 thinks before answering by default; these answers need no
        # chain of thought and the JSON schema already constrains them.
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema},
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(
        f"{server_url}/v1/chat/completions", json=body, timeout=TIMEOUT_SEC
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"], elapsed


# --- fill ----------------------------------------------------------------------


def highlight_members(chunk: Chunk) -> set[int]:
    """Candidate indices the chunk summary named as highlights."""
    return {chunk.members[h["segment"] - 1] for h in chunk.summary["highlights"]}


def part_scenes(
    part: dict[str, Any], chunks: list[Chunk], scenes: list[Scene]
) -> list[Scene]:
    """The scenes inside a part's chunk range."""
    span = chunks[part["first_chunk"] - 1 : part["last_chunk"]]
    members = {i for c in span for i in c.members}
    return [s for s in scenes if s.members[0] in members]


def rescale(values: dict[int, float]) -> dict[int, float]:
    """Best 1, worst 0 (all 1 when they are equal)."""
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {i: 1.0 for i in values}
    return {i: (v - low) / (high - low) for i, v in values.items()}


def share(
    total: float, capacity: list[float], weights: list[float] | None = None
) -> list[float]:
    """Split ``total`` in proportion to ``weights`` (evenly by default), no
    share above its capacity; what a small one cannot hold goes to the
    others in the same proportions."""
    w = weights or [1.0] * len(capacity)
    out = [0.0] * len(capacity)
    left = total
    open_ = set(range(len(capacity)))
    while open_:
        unit = left / sum(w[j] for j in open_)
        full = [j for j in open_ if capacity[j] < w[j] * unit]
        if not full:
            for j in open_:
                out[j] = w[j] * unit
            break
        for j in full:
            out[j] = capacity[j]
            left -= capacity[j]
            open_.remove(j)
    return out


def level(lengths: list[float], total: float, floor: float) -> float:
    """The cap L (at least ``floor``) for which the lengths, each cut to at
    most L, add up to ``total``; ``floor`` when even that runs over."""
    ordered = sorted(lengths)
    for n, cap in enumerate(ordered):
        # Lengths below ``cap`` stay whole, the rest are cut to the level.
        candidate = (total - sum(ordered[:n])) / (len(ordered) - n)
        if candidate <= cap:
            return max(floor, candidate)
    return ordered[-1]


def clip_length(c: Candidate, max_clip_sec: float | None) -> float:
    """How long a segment runs in the cut: whole, or at most max_clip_sec."""
    return c.duration if max_clip_sec is None else min(c.duration, max_clip_sec)


def clip_span(
    c: Candidate,
    max_clip_sec: float | None,
    speech: Sequence[tuple[float, float]] = (),
) -> tuple[float, float]:
    """The part of a segment that goes into the cut: all of it, or a window
    of ``max_clip_sec``. The window cuts as little of ``speech`` (sentence
    spans) as it can, each cut sentence costing its shorter side; among
    equal windows the one nearest the middle wins (the start and end of a
    handheld segment are where the camera is still moving into or out of
    the shot)."""
    length = clip_length(c, max_clip_sec)
    middle = (c.start + c.end) / 2
    lo, hi = c.start, c.end - length
    if hi - lo < 1e-9:
        return c.start, c.end
    edges = [t for span in speech for t in span if c.start < t < c.end]
    starts = {middle - length / 2} | {
        min(max(x, lo), hi) for t in edges for x in (t, t - length)
    }

    def cut(x: float) -> float:
        return sum(
            min(e - a, b - e) for e in (x, x + length) for a, b in speech if a < e < b
        )

    best = min(
        starts, key=lambda x: (round(cut(x), 2), abs(x + length / 2 - middle), x)
    )
    return best, best + length


# Kept before and after a sentence a window is widened to take in, so its
# first and last syllables are not clipped.
SPEECH_PAD_SEC = 0.2


def whole_sentences(
    start: float,
    end: float,
    lo: float,
    hi: float,
    speech: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """[start, end] widened to take in every sentence it cuts, with
    SPEECH_PAD_SEC around it, as far as [lo, hi] (the segment) allows; a
    sentence that runs past the segment stays cut."""
    for a, b in sorted(speech):
        if a < start < b and a >= lo:
            start = max(lo, a - SPEECH_PAD_SEC)
        if a < end < b and b <= hi:
            end = min(hi, b + SPEECH_PAD_SEC)
    return start, end


def pick_segments(
    rel: dict[int, float],
    seconds: float,
    candidates: list[Candidate],
    assessments: list[Assessment],
    named: set[int],
    max_clip_sec: float | None = None,
    relevance_share: float = RELEVANCE_SHARE,
) -> list[int]:
    """Segments of one scene for a slot of ``seconds``.

    ``rel`` maps the scene's usable candidates to their rescaled relevance.
    They are ranked by quality and relevance, both 0-1, weighed by
    ``relevance_share``, + highlight bonus, and taken while a segment
    brings the total closer to ``seconds``, skipping near repeats. A
    segment counts whole, or as at most ``max_clip_sec`` (clip_span).
    """

    def value(i: int) -> float:
        quality = (assessments[i].score - 1) / 4
        return (
            (1 - relevance_share) * quality
            + relevance_share * rel[i]
            + (HIGHLIGHT_BONUS if i in named else 0.0)
        )

    # Ties in timeline order (candidate index), also across videos.
    members = sorted(rel, key=lambda i: (-value(i), i))
    picked: list[int] = []
    total = 0.0
    for i in members:
        c = candidates[i]
        length = clip_length(c, max_clip_sec)
        if picked and total + length / 2 > seconds:
            continue
        repeat = any(
            candidates[p].video == c.video
            and abs(candidates[p].start - c.start) <= DUP_NEAR_SEC
            and _jaccard(candidates[p].tags["tags"], c.tags["tags"]) >= DUP_JACCARD
            for p in picked
        )
        if repeat:
            continue
        picked.append(i)
        total += length
    return picked


def fill(
    outline: dict[str, Any],
    chunks: list[Chunk],
    scenes: list[Scene],
    candidates: list[Candidate],
    assessments: list[Assessment],
    scene_relevance: list[list[dict[int, float]]],
    segment_relevance: list[list[dict[int, float]]],
) -> tuple[list[Clip], list[str], list[list[dict[str, Any]]]]:
    """Fill each part's scene slots, one scene per slot.

    ``scene_relevance[part][slot]`` maps the scene numbers inside the part
    to retrieval relevance (0-1); ``segment_relevance[part][slot]`` maps the
    part's candidates. All (slot, scene) pairs of a part are taken best
    first (relevance + quality), so each slot gets a scene of its own. The
    part's seconds are shared among its filled slots, capped by what each
    scene holds; a slot then takes whole segments of its scene ranked by
    score, relevance and highlight bonus, skipping near repeats, while a
    segment brings its total closer to the slot's seconds. Clips keep
    shooting order; each clip's role is its part's name.

    Returns (clips, notes, slots): per part, each slot's query, scene,
    relevance, seconds and its five best scenes.
    """
    notes: list[str] = []
    chosen: list[tuple[int, int, int]] = []  # (part, slot, candidate)
    slots_out: list[list[dict[str, Any]]] = []
    named = set().union(*(highlight_members(c) for c in chunks))
    by_number = {s.number: s for s in scenes}

    def usable(i: int) -> bool:
        return assessments[i].keep and candidates[i].duration >= MIN_CLIP_SEC

    for k, part in enumerate(outline["parts"]):
        pool = [
            s for s in part_scenes(part, chunks, scenes) if any(map(usable, s.members))
        ]
        quality = {
            s.number: statistics.mean(
                (assessments[i].score - 1) / 4 for i in s.members if usable(i)
            )
            for s in pool
        }
        value = {
            (j, n): r + SCENE_QUALITY_WEIGHT * quality[n]
            for j in range(len(part["scenes"]))
            for n, r in rescale(
                {s.number: scene_relevance[k][j][s.number] for s in pool}
            ).items()
        }
        assigned: dict[int, int] = {}  # slot -> scene number
        for j, n in sorted(value, key=lambda x: (-value[x], x)):
            if j not in assigned and n not in assigned.values():
                assigned[j] = n
        capacity = {
            j: sum(candidates[i].duration for i in by_number[n].members if usable(i))
            for j, n in assigned.items()
        }
        seconds = dict(
            zip(
                capacity,
                share(part["duration_sec"], list(capacity.values())),
                strict=True,
            )
        )
        part_total = 0.0
        slots: list[dict[str, Any]] = []
        for j, query in enumerate(part["scenes"]):
            ranked = sorted((s.number for s in pool), key=lambda n: -value[(j, n)])
            slot: dict[str, Any] = {
                "query": query,
                "scene": assigned.get(j),
                "relevance": None,
                "seconds": 0.0,
                "top": ranked[:5],
            }
            slots.append(slot)
            if j not in assigned:
                notes.append(f"「{part['name']}」场景「{query}」没有可用的场景")
                continue
            scene = by_number[assigned[j]]
            rel = rescale(
                {i: segment_relevance[k][j][i] for i in scene.members if usable(i)}
            )
            slot["relevance"] = round(scene_relevance[k][j][scene.number], 3)
            slot["seconds"] = round(seconds[j], 1)
            picked = pick_segments(rel, seconds[j], candidates, assessments, named)
            chosen += [(k, j, i) for i in picked]
            part_total += sum(candidates[i].duration for i in picked)
        slots_out.append(slots)
        if part_total < 0.8 * part["duration_sec"]:
            notes.append(
                f"「{part['name']}」素材不足：选到 {part_total:.1f}s，"
                f"目标 {part['duration_sec']}s"
            )
    # Candidates are in timeline order, also across videos.
    chosen.sort(key=lambda x: (x[0], x[2]))
    clips = []
    for order, (k, j, i) in enumerate(chosen):
        c = candidates[i]
        slot = slots_out[k][j]
        mark = "，分段总结点名的亮点" if i in named else ""
        clips.append(
            Clip(
                order=order,
                role=outline["parts"][k]["name"],
                video=c.video,
                index=c.index,
                src_start=round(c.start, 3),
                src_end=round(c.end, 3),
                description=c.tags["description"],
                reason=f"场景「{slot['query']}」→ 场景 {slot['scene']}"
                f"（相关度 {slot['relevance']:.0%}），得分 "
                f"{assessments[i].score:.1f}，片段相关度 "
                f"{segment_relevance[k][j][i]:.0%}{mark}",
            )
        )
    validate_timeline(clips, candidates)
    return clips, notes, slots_out
