"""Edit by instruction: the user says what cut they want, in one sentence.

The highlights outline follows shooting order and fills each part from its
own chunks. Here the user's instruction decides what goes in and in which
order (e.g. "60 秒的爱丁堡教堂，先外观后内部", "只要夜景"):

1. Duration: Qwen3.5-4B reads the instruction and returns the length it
   asks for with the words that say so; the words must appear in the
   instruction, so a length the user never gave is rejected. No length in
   the instruction: the caller uses its ratio of the input.
2. Outline: the model reads the chunk summaries (outline.py) and the
   instruction and returns parts in the order of the cut, each with a
   weight and 1-4 scene slots; a slot may require a time of day. It also
   lists what the instruction asks for that the summaries do not have.
3. Fill (rules): every scene of the footage is a candidate for every slot.
   All (slot, scene) pairs are taken best first, so a scene is used once;
   a slot with a time of day only takes scenes whose usable segments are
   mostly of that time, and stays empty rather than take another. Inside
   the scene, segments are picked as in the highlights (pick_segments).
   Clips follow the outline: part, slot, then shooting time.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from typing import Any

from video_demo import outline
from video_demo.outline import (
    MAX_SLOTS,
    MAX_WEIGHT,
    MIN_WEIGHT,
    SCENE_QUALITY_WEIGHT,
    Chunk,
    OutlineError,
    Scene,
)
from video_demo.suggest import (
    MIN_CLIP_SEC,
    Assessment,
    Candidate,
    Clip,
    validate_timeline,
)

ANY_TIME = "不限"
# Times of day a slot may require; the tag values minus 不确定. Whether the
# instruction calls for one is left to the model (instructions can be
# abstract), the prompt only tells it not to add one unasked.
SLOT_TIMES = [ANY_TIME, "白天", "黄昏", "夜晚", "室内"]
MAX_UNMET = 5
MAX_QUOTE_CHARS = 20
MAX_TARGET_SEC = 36000
# The words quoted for a length must hold a number and a unit of time, so
# "剪 3 段夜景" is not read as 3 seconds.
NUMERALS = set("0123456789０１２３４５６７８９零一二两三四五六七八九十百半")
TIME_UNITS = ("秒", "分", "小时", "s", "S", "min")

# --- duration ----------------------------------------------------------------

DURATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_sec": {"type": "integer", "minimum": 0, "maximum": MAX_TARGET_SEC},
        "quote": {"type": "string", "maxLength": MAX_QUOTE_CHARS},
    },
    "required": ["target_sec", "quote"],
}

DURATION_PROMPT = """用户的剪辑要求：
「{instruction}」

成片要多长？
- 要求里写了成片时长（如“60 秒”“一分半”“2 分钟”）：target_sec 写换算成的秒数，
  quote 照抄要求里说时长的那几个字；
- 没写：target_sec 写 0，quote 写空字符串。
只输出 JSON。"""


def validate_duration(parsed: Any, instruction: str) -> dict[str, Any]:
    """The length must come with the instruction's own words for it."""
    if not isinstance(parsed, dict):
        raise OutlineError(f"duration must be an object: {parsed!r}")
    target, quote = parsed.get("target_sec"), str(parsed.get("quote", "")).strip()
    if not isinstance(target, int) or not 0 <= target <= MAX_TARGET_SEC:
        raise OutlineError(f"target_sec must be 0-{MAX_TARGET_SEC}: {target!r}")
    if target == 0:
        if quote:
            raise OutlineError(f"no length but quote {quote!r}")
        return {"target_sec": 0, "quote": ""}
    if not quote:
        raise OutlineError(f"length {target}s without the words that give it")
    if quote not in instruction:
        raise OutlineError(f"quote {quote!r} is not in the instruction")
    if not NUMERALS & set(quote) or not any(u in quote for u in TIME_UNITS):
        raise OutlineError(f"quote {quote!r} is not a length")
    return {"target_sec": target, "quote": quote}


def parse_duration(
    instruction: str, server_url: str = outline.SERVER_URL
) -> tuple[dict[str, Any], str, float, list[str]]:
    """({target_sec, quote}, raw, seconds, failed tries); 0 means the
    instruction gives no length."""
    return outline._ask(
        DURATION_PROMPT.format(instruction=instruction),
        DURATION_SCHEMA,
        "edit_duration",
        lambda parsed: validate_duration(parsed, instruction),
        server_url,
    )


# --- outline -------------------------------------------------------------------

REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 30},
        "understanding": {"type": "string", "minLength": 1, "maxLength": 80},
        "unmet": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 30},
            "maxItems": MAX_UNMET,
        },
        "parts": {
            "type": "array",
            "minItems": 1,
            "maxItems": outline.MAX_PARTS,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 20},
                    "summary": {"type": "string", "minLength": 1, "maxLength": 80},
                    "weight": {
                        "type": "integer",
                        "minimum": MIN_WEIGHT,
                        "maximum": MAX_WEIGHT,
                    },
                    "scenes": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_SLOTS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 20,
                                },
                                "time_of_day": {"type": "string", "enum": SLOT_TIMES},
                            },
                            "required": ["query", "time_of_day"],
                        },
                    },
                },
                "required": ["name", "summary", "weight", "scenes"],
            },
        },
    },
    "required": ["title", "understanding", "unmet", "parts"],
}

REQUEST_PROMPT = """用户的剪辑要求：
「{instruction}」

下面是 {total_min:.1f} 分钟原始素材按时间分成的 {n} 块，每块已有总结。
每行：[块编号] 起止时间 | 总结 | 地点 | 活动

{chunks}

请按用户的要求设计一个约 {target_sec:.0f} 秒的成片大纲：
- understanding 用一句话说明你怎么理解这个要求；
- unmet 列出要求里提到、但上面的总结里找不到的内容，都找得到就留空；
- 分成 1–{max_parts} 个部分（不能超过 {max_parts} 个），按成片里的先后排列，
  先后由用户的要求决定，不必按拍摄时间；
- name 是这一部分的标题，summary 用一句话说明这部分讲什么；
- weight 是这一部分的分量（1–10 的整数），程序会按分量把 {target_sec:.0f} 秒
  分给各部分，每部分至少 {min_part:.0f} 秒；
- scenes 是这一部分要的 1–{max_slots} 个场景：query 用一个短语说明画面
  （如“远处的城堡”“教堂里的彩色玻璃窗”），从上面的总结、地点和活动里选，
  所有部分的场景互不重复；time_of_day 是这个场景要求的时段
  （{times}），只有用户的要求里明确说了时段才填，否则一律填“{any_time}”。
程序会为每个场景从全部素材里找一段连续的场景，一个场景只用一次。
只输出 JSON。"""


def request_schema(max_parts: int) -> dict[str, Any]:
    schema = json.loads(json.dumps(REQUEST_SCHEMA))
    schema["properties"]["parts"]["maxItems"] = max_parts
    return schema


def validate_request(plan: Any, target_sec: float) -> dict[str, Any]:
    """Deterministic checks on the model's outline; raises OutlineError.

    Adds each part's ``duration_sec``, allocated from the weights.
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("parts"), list):
        raise OutlineError("Expected an object with a parts list")
    if not str(plan.get("understanding", "")).strip():
        raise OutlineError("understanding must not be empty")
    unmet = plan.get("unmet", [])
    if not isinstance(unmet, list) or any(not str(u).strip() for u in unmet):
        raise OutlineError("unmet must be a list of non-empty strings")
    parts = plan["parts"]
    max_parts = outline.part_limits(target_sec)[1]
    if not 1 <= len(parts) <= max_parts:
        raise OutlineError(f"{len(parts)} parts, need 1-{max_parts}")
    queries: set[str] = set()
    for k, p in enumerate(parts, start=1):
        if not str(p.get("name", "")).strip() or not str(p.get("summary", "")).strip():
            raise OutlineError(f"part {k}: name and summary must not be empty")
        weight = p.get("weight")
        if not isinstance(weight, int) or not MIN_WEIGHT <= weight <= MAX_WEIGHT:
            raise OutlineError(f"part {k}: weight must be {MIN_WEIGHT}-{MAX_WEIGHT}")
        slots = p.get("scenes")
        if not isinstance(slots, list) or not 1 <= len(slots) <= MAX_SLOTS:
            raise OutlineError(f"part {k}: needs 1-{MAX_SLOTS} scenes")
        for slot in slots:
            query = str(slot.get("query", "")).strip()
            if not query:
                raise OutlineError(f"part {k}: empty scene")
            if query in queries:
                raise OutlineError(f"part {k}: scene {query!r} listed twice")
            queries.add(query)
            when = slot.get("time_of_day")
            if when not in SLOT_TIMES:
                raise OutlineError(
                    f"part {k}: time_of_day {when!r} not in {SLOT_TIMES}"
                )
    seconds = outline.allocate([p["weight"] for p in parts], target_sec)
    for p, sec in zip(parts, seconds, strict=True):
        p["duration_sec"] = sec
    return plan


def plan_request(
    instruction: str,
    chunks: list[Chunk],
    target_sec: float,
    server_url: str = outline.SERVER_URL,
) -> tuple[dict[str, Any], str, float, list[str]]:
    """(outline, raw, seconds, failed tries) for the instruction."""
    min_part, max_parts = outline.part_limits(target_sec)
    prompt = REQUEST_PROMPT.format(
        instruction=instruction,
        total_min=sum(c.end - c.start for c in chunks) / 60,
        n=len(chunks),
        chunks=outline.chunk_lines(chunks),
        target_sec=target_sec,
        max_parts=max_parts,
        min_part=min_part,
        max_slots=MAX_SLOTS,
        times="、".join(SLOT_TIMES[1:]),
        any_time=ANY_TIME,
    )
    return outline._ask(
        prompt,
        request_schema(max_parts),
        "edit_outline",
        lambda parsed: validate_request(parsed, target_sec),
        server_url,
    )


# --- fill ----------------------------------------------------------------------


def scene_time(candidates: list[Candidate], usable: list[int]) -> str | None:
    """The time of day of more than half of the scene's usable segments."""
    counts = Counter(candidates[i].tags["time_of_day"] for i in usable)
    time, n = counts.most_common(1)[0]
    return time if 2 * n > len(usable) else None


def fill_request(
    plan: dict[str, Any],
    scenes: list[Scene],
    candidates: list[Candidate],
    assessments: list[Assessment],
    scene_relevance: list[list[dict[int, float]]],
    segment_relevance: list[list[dict[int, float]]],
    named: set[int],
) -> tuple[list[Clip], list[str], list[list[dict[str, Any]]]]:
    """Fill every slot of the outline from all scenes, one scene per slot.

    ``scene_relevance[part][slot]`` maps every scene number to retrieval
    relevance (0-1), ``segment_relevance[part][slot]`` every candidate;
    ``named`` are the candidates the chunk summaries called highlights.
    A slot's pool is the scenes with usable segments and, when the slot
    asks for a time of day, mostly of that time. All (slot, scene) pairs of
    the outline are taken best first (relevance rescaled within the slot's
    pool + quality), so a scene goes to one slot. A part's seconds are
    shared among its filled slots as in outline.fill.

    Returns (clips, notes, slots): per part, each slot's query, time of
    day, scene, relevance, seconds and its five best scenes.
    """

    def usable(i: int) -> bool:
        return assessments[i].keep and candidates[i].duration >= MIN_CLIP_SEC

    kept = {s.number: [i for i in s.members if usable(i)] for s in scenes}
    by_number = {s.number: s for s in scenes if kept[s.number]}
    times = {n: scene_time(candidates, kept[n]) for n in by_number}
    quality = {
        n: statistics.mean((assessments[i].score - 1) / 4 for i in kept[n])
        for n in by_number
    }
    pools: dict[tuple[int, int], list[int]] = {}
    value: dict[tuple[int, int, int], float] = {}
    for k, part in enumerate(plan["parts"]):
        for j, slot in enumerate(part["scenes"]):
            want = slot["time_of_day"]
            pool = [n for n in by_number if want == ANY_TIME or times[n] == want]
            pools[(k, j)] = pool
            if pool:
                rel = outline.rescale({n: scene_relevance[k][j][n] for n in pool})
                for n, r in rel.items():
                    value[(k, j, n)] = r + SCENE_QUALITY_WEIGHT * quality[n]
    assigned: dict[tuple[int, int], int] = {}
    for k, j, n in sorted(value, key=lambda x: (-value[x], x)):
        if (k, j) not in assigned and n not in assigned.values():
            assigned[(k, j)] = n

    notes: list[str] = []
    chosen: list[tuple[int, int, int]] = []  # (part, slot, candidate)
    slots_out: list[list[dict[str, Any]]] = []
    for k, part in enumerate(plan["parts"]):
        filled = [j for j in range(len(part["scenes"])) if (k, j) in assigned]
        capacity = [
            sum(candidates[i].duration for i in kept[assigned[(k, j)]]) for j in filled
        ]
        seconds = dict(
            zip(filled, outline.share(part["duration_sec"], capacity), strict=True)
        )
        part_total = 0.0
        slots: list[dict[str, Any]] = []
        for j, query in enumerate(part["scenes"]):
            pool = pools[(k, j)]
            ranked = sorted(pool, key=lambda n: (-value[(k, j, n)], n))
            slot: dict[str, Any] = {
                "query": query["query"],
                "time_of_day": query["time_of_day"],
                "scene": assigned.get((k, j)),
                "relevance": None,
                "seconds": 0.0,
                "top": ranked[:5],
            }
            slots.append(slot)
            if (k, j) not in assigned:
                why = (
                    f"没有时段为「{query['time_of_day']}」的可用场景"
                    if not pool
                    else "符合的场景都已分给其他槽位"
                )
                notes.append(f"「{part['name']}」场景「{query['query']}」{why}")
                continue
            n = assigned[(k, j)]
            rel = outline.rescale({i: segment_relevance[k][j][i] for i in kept[n]})
            slot["relevance"] = round(scene_relevance[k][j][n], 3)
            slot["seconds"] = round(seconds[j], 1)
            picked = outline.pick_segments(
                rel, seconds[j], candidates, assessments, named
            )
            chosen += [(k, j, i) for i in picked]
            part_total += sum(candidates[i].duration for i in picked)
        slots_out.append(slots)
        if part_total < 0.8 * part["duration_sec"]:
            notes.append(
                f"「{part['name']}」素材不足：选到 {part_total:.1f}s，"
                f"目标 {part['duration_sec']}s"
            )
    chosen.sort(key=lambda x: (x[0], x[1], candidates[x[2]].start))
    clips = []
    for order, (k, j, i) in enumerate(chosen):
        c = candidates[i]
        slot = slots_out[k][j]
        when = "" if slot["time_of_day"] == ANY_TIME else f"，{slot['time_of_day']}"
        mark = "，分段总结点名的亮点" if i in named else ""
        clips.append(
            Clip(
                order=order,
                role=plan["parts"][k]["name"],
                video=c.video,
                index=c.index,
                src_start=round(c.start, 3),
                src_end=round(c.end, 3),
                description=c.tags["description"],
                reason=f"场景「{slot['query']}」{when}→ 场景 {slot['scene']}"
                f"（相关度 {slot['relevance']:.0%}），得分 "
                f"{assessments[i].score:.1f}，片段相关度 "
                f"{segment_relevance[k][j][i]:.0%}{mark}",
            )
        )
    validate_timeline(clips, candidates)
    return clips, notes, slots_out
