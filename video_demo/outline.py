"""Highlights outline: a language model plans the parts, rules fill them.

The segments of a recording are grouped into scenes (consecutive segments
with similar tags) so that an hour of footage fits in the model's context.
Qwen3.5-4B (llama-server) reads the scene list and returns an outline: a few
parts in shooting order, each with a title, a one-line summary, the scene
range it covers, a target length and keywords. The outline is checked
deterministically; a failing outline is retried and finally raised, never
replaced by a default.

Filling the parts (choosing segments) is rule-based, see ``fill``.
"""

from __future__ import annotations

import json
import math
import time
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
TIMEOUT_SEC = 600
TEMPERATURES = (0.2, 0.5, 0.8)

# A new scene starts at a new video, when the tags change (overlap with the
# previous segment below this), or when the scene would grow longer than this.
SCENE_SPLIT_JACCARD = 0.3
SCENE_MAX_SEC = 60.0

MIN_PARTS, MAX_PARTS = 2, 8
MIN_PART_SEC = 10.0
# The model weighs each part 1-10; seconds are computed from the weights,
# since a 4B model does not add up lengths to a target reliably.
MIN_WEIGHT, MAX_WEIGHT = 1, 10


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


def parts_range(target_sec: float, n_scenes: int) -> tuple[int, int]:
    """(fewest, most) parts: part_limits, and no more parts than scenes
    (a 17 s clip can be a single scene, which makes a one-part outline)."""
    most = min(part_limits(target_sec)[1], n_scenes)
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


# Weight of the keyword match (0-1) next to the segment score (1-5).
KEYWORD_WEIGHT = 1.0

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
                    "name": {"type": "string", "maxLength": 20},
                    "summary": {"type": "string", "maxLength": 80},
                    "first_scene": {"type": "integer", "minimum": 1},
                    "last_scene": {"type": "integer", "minimum": 1},
                    "weight": {
                        "type": "integer",
                        "minimum": MIN_WEIGHT,
                        "maximum": MAX_WEIGHT,
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 12},
                        "minItems": 1,
                        "maxItems": 8,
                    },
                },
                "required": [
                    "name",
                    "summary",
                    "first_scene",
                    "last_scene",
                    "weight",
                    "keywords",
                ],
            },
        },
    },
    "required": ["title", "parts"],
}

SYSTEM_PROMPT = (
    "你是视频剪辑师，负责从原始素材里整理精彩瞬间的大纲。"
    "只根据给出的场景描述安排内容，不要编造素材里没有的东西。所有文字用简体中文。"
)

USER_PROMPT = """下面是一段 {total_min:.1f} 分钟原始素材的场景列表，按拍摄时间排列。
每行：[场景编号] 起止时间 | 时段 | 平均得分（1–5，越高画面越好、越精彩） | 内容 | 标签

{scenes}

请把它整理成约 {target_min:.1f} 分钟（{target_sec:.0f} 秒）的精彩瞬间大纲，
供用户进一步剪辑：
- 分成 {min_parts}–{max_parts} 个部分（不能超过 {max_parts} 个），按时间顺序排列，
  每部分覆盖连续的一段场景（first_scene 到 last_scene），部分之间不重叠，
  没意思的场景可以跳过；
- name 是这一部分的标题，summary 用一句话说明这部分讲什么；
- weight 是这一部分的分量（1–10 的整数），内容丰富、得分高的部分给高分；
  程序会按分量把 {target_sec:.0f} 秒分给各部分，每部分至少 {min_part:.0f} 秒；
- keywords 是 1–8 个用来挑选画面的关键词，从这部分场景的内容和标签里选。
只输出 JSON。"""


class OutlineError(RuntimeError):
    """The model output is not a usable outline."""


@dataclass(frozen=True)
class Scene:
    number: int  # 1-based, as shown to the model
    members: list[int]  # indices into the candidate list
    video: str
    start: float
    end: float
    time_of_day: str
    score: float
    descriptions: list[str]
    tags: list[str]


def _fmt(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def group_scenes(
    candidates: list[Candidate],
    assessments: list[Assessment],
    offsets: dict[str, float] | None = None,
) -> list[Scene]:
    """Consecutive segments with similar tags, in shooting order.

    ``offsets`` places each video on one timeline (as in highlights mode);
    candidates must already be in that order.
    """
    if len(candidates) != len(assessments):
        raise ValueError("candidates and assessments must align")
    if not candidates:
        raise ValueError("no segments to outline")
    offsets = offsets or {}
    groups: list[list[int]] = []
    for i, c in enumerate(candidates):
        if groups:
            first = candidates[groups[-1][0]]
            prev = candidates[groups[-1][-1]]
            # Compared with the previous segment, not the whole scene, so a
            # scene that slowly changes stays together until SCENE_MAX_SEC.
            same = (
                c.video == first.video
                and _jaccard(prev.tags["tags"], c.tags["tags"]) >= SCENE_SPLIT_JACCARD
                and c.end - first.start <= SCENE_MAX_SEC
            )
            if same:
                groups[-1].append(i)
                continue
        groups.append([i])

    scenes = []
    for n, members in enumerate(groups, start=1):
        cs = [candidates[i] for i in members]
        descs = list(dict.fromkeys(c.tags["description"] for c in cs))
        tags = list(dict.fromkeys(t for c in cs for t in c.tags["tags"]))
        times = [c.tags["time_of_day"] for c in cs]
        offset = offsets.get(cs[0].video, 0.0)
        scenes.append(
            Scene(
                number=n,
                members=members,
                video=cs[0].video,
                start=offset + cs[0].start,
                end=offset + cs[-1].end,
                time_of_day=max(set(times), key=times.count),
                score=round(sum(assessments[i].score for i in members) / len(cs), 1),
                descriptions=descs[:1] + descs[-1:] if len(descs) > 1 else descs,
                tags=tags[:6],
            )
        )
    return scenes


def scene_lines(scenes: list[Scene]) -> str:
    return "\n".join(
        f"[{s.number}] {_fmt(s.start)}–{_fmt(s.end)} | {s.time_of_day} | "
        f"{s.score:.1f} | {'；'.join(s.descriptions)} | {'、'.join(s.tags)}"
        for s in scenes
    )


def available_sec(scenes: list[Scene], candidates: list[Candidate]) -> list[float]:
    """Length of usable segments per scene (what a part could fill)."""
    return [
        sum(
            candidates[i].duration
            for i in s.members
            if candidates[i].duration >= MIN_CLIP_SEC
        )
        for s in scenes
    ]


def validate_outline(
    outline: Any, scenes: list[Scene], target_sec: float
) -> dict[str, Any]:
    """Deterministic checks on the model's outline; raises OutlineError.

    Adds each part's ``duration_sec``, allocated from the weights.
    """
    if not isinstance(outline, dict) or not isinstance(outline.get("parts"), list):
        raise OutlineError("Expected an object with a parts list")
    parts = outline["parts"]
    min_parts, max_parts = parts_range(target_sec, len(scenes))
    if not min_parts <= len(parts) <= max_parts:
        raise OutlineError(f"{len(parts)} parts, need {min_parts}-{max_parts}")
    last = 0
    for k, p in enumerate(parts, start=1):
        first_s, last_s = p.get("first_scene"), p.get("last_scene")
        if not isinstance(first_s, int) or not isinstance(last_s, int):
            raise OutlineError(f"part {k}: scene numbers must be integers")
        if not 1 <= first_s <= last_s <= len(scenes):
            raise OutlineError(
                f"part {k}: scenes {first_s}-{last_s} outside 1-{len(scenes)}"
            )
        if first_s <= last:
            raise OutlineError(f"part {k}: scenes overlap or are out of order")
        last = last_s
        if not str(p.get("name", "")).strip() or not str(p.get("summary", "")).strip():
            raise OutlineError(f"part {k}: name and summary must not be empty")
        if not p.get("keywords"):
            raise OutlineError(f"part {k}: no keywords")
        weight = p.get("weight")
        if not isinstance(weight, int) or not MIN_WEIGHT <= weight <= MAX_WEIGHT:
            raise OutlineError(f"part {k}: weight must be {MIN_WEIGHT}-{MAX_WEIGHT}")
    seconds = allocate([p["weight"] for p in parts], target_sec)
    for p, sec in zip(parts, seconds, strict=True):
        p["duration_sec"] = sec
    return outline


def plan_outline(
    scenes: list[Scene],
    target_sec: float,
    server_url: str = SERVER_URL,
) -> tuple[dict[str, Any], str, float, list[str]]:
    """Ask the model for an outline; (outline, raw, seconds, failed tries).

    Raises OutlineError listing every attempt when all fail validation.
    """
    total = scenes[-1].end - scenes[0].start
    min_part, _ = part_limits(target_sec)
    min_parts, max_parts = parts_range(target_sec, len(scenes))
    prompt = USER_PROMPT.format(
        total_min=total / 60,
        scenes=scene_lines(scenes),
        target_min=target_sec / 60,
        target_sec=target_sec,
        min_parts=min_parts,
        max_parts=max_parts,
        min_part=min_part,
    )
    errors: list[str] = []
    seconds = 0.0
    schema = outline_schema(min_parts, max_parts)
    for temperature in TEMPERATURES:
        raw, elapsed = _request(prompt, temperature, schema, server_url)
        seconds += elapsed
        try:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise OutlineError(f"Model output is not JSON: {raw[:200]!r}") from exc
            return validate_outline(parsed, scenes, target_sec), raw, seconds, errors
        except OutlineError as exc:
            errors.append(f"temperature={temperature}: {exc}")
    raise OutlineError("；".join(errors))


def _request(
    prompt: str, temperature: float, schema: dict[str, Any], server_url: str
) -> tuple[str, float]:
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2000,
        # Qwen3.5 thinks before answering by default; the outline needs no
        # chain of thought and the JSON schema already constrains the answer.
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "highlights_outline", "schema": schema},
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(
        f"{server_url}/v1/chat/completions", json=body, timeout=TIMEOUT_SEC
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"], elapsed


def _keyword_match(c: Candidate, keywords: list[str]) -> float:
    """Share of the part's keywords found in the segment's tags or text."""
    text = " ".join(
        [
            c.tags["description"],
            c.tags.get("scene", ""),
            c.tags.get("action", ""),
            *c.tags["tags"],
            *c.tags.get("subjects", []),
        ]
    )
    return sum(k in text for k in keywords) / len(keywords)


def fill(
    outline: dict[str, Any],
    scenes: list[Scene],
    candidates: list[Candidate],
    assessments: list[Assessment],
) -> tuple[list[Clip], list[str]]:
    """Choose whole segments for each part inside its scene range.

    Within a part, segments are ranked by score + keyword match and taken
    until the part's length is reached; near repeats (similar tags within
    60 s) are skipped. Clips keep shooting order; each clip's role is its
    part's name.
    """
    notes: list[str] = []
    chosen: list[tuple[int, int]] = []  # (part, candidate)
    for k, part in enumerate(outline["parts"]):
        members = [
            i
            for s in scenes[part["first_scene"] - 1 : part["last_scene"]]
            for i in s.members
        ]
        pool = sorted(
            (
                (
                    assessments[i].score
                    + KEYWORD_WEIGHT * _keyword_match(candidates[i], part["keywords"]),
                    i,
                )
                for i in members
                if assessments[i].keep and candidates[i].duration >= MIN_CLIP_SEC
            ),
            key=lambda x: (-x[0], candidates[x[1]].start),
        )
        picked: list[int] = []
        total = 0.0
        for _, i in pool:
            if total >= part["duration_sec"]:
                break
            c = candidates[i]
            repeat = any(
                candidates[j].video == c.video
                and abs(candidates[j].start - c.start) <= DUP_NEAR_SEC
                and _jaccard(candidates[j].tags["tags"], c.tags["tags"]) >= DUP_JACCARD
                for j in picked
            )
            if repeat:
                continue
            picked.append(i)
            chosen.append((k, i))
            total += c.duration
        if total < 0.8 * part["duration_sec"]:
            notes.append(
                f"「{part['name']}」素材不足：选到 {total:.1f}s，"
                f"目标 {part['duration_sec']}s"
            )
    chosen.sort(key=lambda x: (x[0], candidates[x[1]].start))
    clips = []
    for order, (k, i) in enumerate(chosen):
        c = candidates[i]
        clips.append(
            Clip(
                order=order,
                role=outline["parts"][k]["name"],
                video=c.video,
                index=c.index,
                src_start=round(c.start, 3),
                src_end=round(c.end, 3),
                description=c.tags["description"],
                reason=f"得分 {assessments[i].score:.1f}，关键词匹配 "
                f"{_keyword_match(c, outline['parts'][k]['keywords']):.0%}",
            )
        )
    validate_timeline(clips, candidates, max_clip_sec=None)
    return clips, notes
