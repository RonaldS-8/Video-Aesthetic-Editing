"""Edit by instruction: the user says what cut they want, in one sentence.

The highlights outline follows shooting order and fills each part from its
own chunks. Here the user's instruction decides what goes in and in which
order (e.g. "60 秒的爱丁堡教堂，先外观后内部", "只要夜景"):

1. Duration: Qwen3.5-4B reads the instruction and returns the length it
   asks for with the words that say so; the words must appear in the
   instruction, so a length the user never gave is rejected. No length in
   the instruction: the caller uses its ratio of the input.
2. Outline: the model reads the scene catalog (each scene's title and
   summary, see scenes.py) and the instruction. Scenes are finer than the
   chunk summaries, which turn one or two mistagged segments into a whole
   chunk "白天至黄昏". The answer's fields are written in order, so they
   double as its working: first the scenes that fit the instruction (scene
   and content), then a design of the cut (what opens, what follows, why),
   what the instruction asks for that the catalog does not have, and last
   the parts in the order of the cut, each with a weight and 1-4 scene
   slots. A slot names the catalog scenes it was designed from (shown in
   the report; retrieval still searches every scene). Times of day are
   not a filter: what the instruction asks of the picture goes into the
   slot's query and acts through retrieval.
3. Fill (rules): every scene of the footage is a candidate for every slot.
   All (slot, scene) pairs are taken best first, so a scene is used once.
   Inside the scene, segments are picked as in the highlights
   (pick_segments). Clips follow the outline: part, slot, then shooting
   time.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from video_demo import beats, outline

if TYPE_CHECKING:
    from video_demo.category import Profile
from video_demo.outline import (
    MAX_SLOTS,
    MAX_WEIGHT,
    MIN_WEIGHT,
    SCENE_QUALITY_WEIGHT,
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

MAX_UNMET = 5
MAX_MATERIALS = 8
MAX_MATERIAL_CHARS = 40
MAX_DESIGN_CHARS = 150
MAX_SLOT_SOURCES = 3
# Pace of the cut, chosen by the model from the instruction: the longest a
# segment (about 10 s) may run in the cut; shorter ones keep their middle.
PACE_CLIP_SEC: dict[str, float | None] = {"整段": None, "适中": 6.0, "快": 3.0}
# Inside a scene the instruction decides what fits, so relevance leads
# (outline.pick_segments; highlights use 0.2). On Edinburgh the cathedral
# front (quality 0.95, last in all three rankings for 教堂内部穹顶) came
# before the darker interior (0.46-0.58) at 0.5; from about 0.6 the
# interior leads.
RELEVANCE_SHARE = 0.7
# The scene catalog in the prompt: Edinburgh's 59 scenes take ~3.4k tokens,
# so 60 keeps the outline call inside a 6k context; longer footage gives
# the model the scenes most relevant to the instruction.
MAX_CATALOG_SCENES = 60
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
        "materials": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_MATERIALS,
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "integer", "minimum": 1},
                    "content": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_MATERIAL_CHARS,
                    },
                },
                "required": ["scene", "content"],
            },
        },
        "design": {"type": "string", "minLength": 1, "maxLength": MAX_DESIGN_CHARS},
        "pace": {"type": "string", "enum": list(PACE_CLIP_SEC)},
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
                                "sources": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": MAX_SLOT_SOURCES,
                                    "items": {"type": "integer", "minimum": 1},
                                },
                            },
                            "required": ["query", "sources"],
                        },
                    },
                },
                "required": ["name", "summary", "weight", "scenes"],
            },
        },
    },
    "required": ["title", "materials", "design", "pace", "unmet", "parts"],
}

REQUEST_PROMPT = """用户的剪辑要求：
「{instruction}」

下面是 {total_min:.1f} 分钟原始素材切成的 {n} 个场景，按拍摄时间排列，
每个场景有标题和概况。
每行：[场景编号] 起止时间 | 标题 | 概况

{catalog}
{structure}
请按用户的要求，从这些场景里设计一个约 {target_sec:.0f} 秒的成片，按顺序写：
0. title：成片的片名，会作为片头标题出现在画面上，像短片标题那样简短
   （不超过 12 个字），不写时长；
1. materials：通读上面的场景，挑出符合要求的场景，每条写场景编号（scene）和
   这个场景里能用的具体内容（content：地点、主体、画面，不超过 {max_material} 个字），
   不超过 {max_materials} 条；
2. design：用 2–3 句话写成片的设计：开头、中间、结尾分别用 materials 里的
   哪些场景，为什么这样安排；要求里有时间先后（如从白天到夜晚、春夏秋冬）
   时，按这个先后安排；写完对照用户的要求检查内容和先后顺序；
3. pace：成片的剪辑节奏，按用户的要求判断：整段（每段素材完整保留，约 10 秒，
   适合慢慢看的风景、过程）、适中（每段最多 6 秒）、快（每段最多 3 秒，
   适合节奏轻快的混剪）；
4. unmet：要求里提到、但场景里找不到的内容，都找得到就留空；找不到的
   内容不要拿别的场景冒充；
5. parts：照 design 分成 1–{max_parts} 个部分（不能超过 {max_parts} 个），按成片里的
   先后排列，先后由 design 决定，不必按拍摄时间：
   - name 是这一部分的标题，summary 用一句话说明这部分讲什么；
   - weight 是这一部分的分量（1–10 的整数），程序会按分量把 {target_sec:.0f} 秒
     分给各部分，每部分至少 {min_part:.0f} 秒；
   - scenes 是这一部分要的 1–{max_slots} 个镜头：query 用一个短语说明要的画面
     （如“远处的城堡”“教堂里的彩色玻璃窗”），写看得见的主体、动作和光线，
     不写地名（场景标题里的地名可能认错），所有部分的 query 互不重复；
     sources 写这个画面取自哪几个场景（1–{max_sources} 个编号），只能写
     materials 里已经列出的编号，要用的场景先加进 materials。
程序会按每个 query 从全部场景里检索一个最符合的场景，一个场景只用一次。
只输出 JSON。"""


# The category's usual structure, offered as a reference (the user's
# instruction still decides).
STRUCTURE_PROMPT = """
这批素材属于「{name}」，这类成片常用的结构是：{structure}
设计时可以参考，但以用户的要求为准。
"""


def request_schema(max_parts: int) -> dict[str, Any]:
    schema = json.loads(json.dumps(REQUEST_SCHEMA))
    schema["properties"]["parts"]["maxItems"] = max_parts
    return schema


def catalog_lines(catalog: list[dict[str, Any]]) -> str:
    """One line per scene; with more than one video, which video(s) the
    scene is in (numbered in timeline order)."""
    many = len({v for s in catalog for v in s["videos"]}) > 1

    def where(s: dict[str, Any]) -> str:
        if not many:
            return ""
        first, last = s["videos"][0], s["videos"][-1]
        return f"视频 {first} " if first == last else f"视频 {first}–{last} "

    return "\n".join(
        f"[{s['number']}] {where(s)}{outline._fmt(s['start'])}–"
        f"{outline._fmt(s['end'])} | {s['title']} | {s['summary']}"
        for s in catalog
    )


def shortlist(
    catalog: list[dict[str, Any]], relevance: dict[int, float]
) -> list[dict[str, Any]]:
    """At most MAX_CATALOG_SCENES scenes for the prompt: the most relevant
    to the instruction (``relevance`` by scene number), in timeline order."""
    if len(catalog) <= MAX_CATALOG_SCENES:
        return catalog
    best = sorted(catalog, key=lambda s: (-relevance[s["number"]], s["number"]))
    keep = {s["number"] for s in best[:MAX_CATALOG_SCENES]}
    return [s for s in catalog if s["number"] in keep]


def validate_request(plan: Any, target_sec: float, numbers: set[int]) -> dict[str, Any]:
    """Deterministic checks on the model's outline; raises OutlineError.

    Material scenes must be in the catalog (``numbers``), and a slot's
    sources must be among the materials it was designed from. Adds each
    part's ``duration_sec``, allocated from the weights.
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("parts"), list):
        raise OutlineError("Expected an object with a parts list")
    materials = plan.get("materials")
    if not isinstance(materials, list) or not materials:
        raise OutlineError("materials must be a non-empty list")
    picked: set[int] = set()
    for m in materials:
        n = m.get("scene")
        if not isinstance(n, int) or n not in numbers:
            raise OutlineError(f"material scene {n!r} is not in the catalog")
        if not str(m.get("content", "")).strip():
            raise OutlineError(f"material of scene {n}: empty content")
        picked.add(n)
    if not str(plan.get("design", "")).strip():
        raise OutlineError("design must not be empty")
    if plan.get("pace") not in PACE_CLIP_SEC:
        raise OutlineError(f"pace {plan.get('pace')!r} not in {list(PACE_CLIP_SEC)}")
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
            sources = slot.get("sources")
            if (
                not isinstance(sources, list)
                or not 1 <= len(sources) <= MAX_SLOT_SOURCES
                or any(not isinstance(n, int) for n in sources)
            ):
                raise OutlineError(
                    f"part {k}: scene {query!r} needs 1-{MAX_SLOT_SOURCES} sources"
                )
            if not set(sources) <= picked:
                raise OutlineError(
                    f"part {k}: scene {query!r} sources {sources} not in materials"
                )
    seconds = outline.allocate([p["weight"] for p in parts], target_sec)
    for p, sec in zip(parts, seconds, strict=True):
        p["duration_sec"] = sec
    return plan


def plan_request(
    instruction: str,
    catalog: list[dict[str, Any]],
    total_sec: float,
    target_sec: float,
    server_url: str = outline.SERVER_URL,
    profile: Profile | None = None,
) -> tuple[dict[str, Any], str, float, list[str]]:
    """(outline, raw, seconds, failed tries) for the instruction.

    ``catalog`` is the scenes that keep footage, each with its number,
    start, end, title and summary; ``total_sec`` the footage's length;
    ``profile`` the footage's category, whose usual structure is offered.
    """
    min_part, max_parts = outline.part_limits(target_sec)
    prompt = REQUEST_PROMPT.format(
        instruction=instruction,
        total_min=total_sec / 60,
        n=len(catalog),
        catalog=catalog_lines(catalog),
        structure=(
            STRUCTURE_PROMPT.format(name=profile.name, structure=profile.structure)
            if profile
            else ""
        ),
        target_sec=target_sec,
        max_parts=max_parts,
        min_part=min_part,
        max_slots=MAX_SLOTS,
        max_material=MAX_MATERIAL_CHARS,
        max_materials=MAX_MATERIALS,
        max_sources=MAX_SLOT_SOURCES,
    )
    numbers = {s["number"] for s in catalog}
    return outline._ask(
        prompt,
        request_schema(max_parts),
        "edit_outline",
        lambda parsed: validate_request(parsed, target_sec, numbers),
        server_url,
    )


# --- fill ----------------------------------------------------------------------


def fill_request(
    plan: dict[str, Any],
    scenes: list[Scene],
    candidates: list[Candidate],
    assessments: list[Assessment],
    scene_relevance: list[list[dict[int, float]]],
    segment_relevance: list[list[dict[int, float]]],
    named: set[int],
    speech: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[Clip], list[str], list[list[dict[str, Any]]]]:
    """Fill every slot of the outline from all scenes, one scene per slot.

    ``scene_relevance[part][slot]`` maps every scene number to retrieval
    relevance (0-1), ``segment_relevance[part][slot]`` every candidate;
    ``named`` are the candidates the chunk summaries called highlights;
    ``speech`` maps a video to its sentence spans, which trimmed clips
    avoid cutting (outline.clip_span).
    Every scene with usable segments is in every slot's pool. All (slot,
    scene) pairs of the outline are taken best first (relevance rescaled
    within the pool + quality), so a scene goes to one slot. A part's
    seconds are shared among its filled slots as in outline.fill.

    Returns (clips, notes, slots): per part, each slot's query, sources,
    scene, relevance, seconds and its five best scenes.
    """

    def usable(i: int) -> bool:
        return assessments[i].keep and candidates[i].duration >= MIN_CLIP_SEC

    kept = {s.number: [i for i in s.members if usable(i)] for s in scenes}
    pool = [s.number for s in scenes if kept[s.number]]
    quality = {
        n: statistics.mean((assessments[i].score - 1) / 4 for i in kept[n])
        for n in pool
    }
    value: dict[tuple[int, int, int], float] = {}
    for k, part in enumerate(plan["parts"]):
        for j in range(len(part["scenes"])):
            if pool:
                rel = outline.rescale({n: scene_relevance[k][j][n] for n in pool})
                for n, r in rel.items():
                    value[(k, j, n)] = r + SCENE_QUALITY_WEIGHT * quality[n]
    assigned: dict[tuple[int, int], int] = {}
    for k, j, n in sorted(value, key=lambda x: (-value[x], x)):
        if (k, j) not in assigned and n not in assigned.values():
            assigned[(k, j)] = n

    max_clip = PACE_CLIP_SEC[plan["pace"]]

    def length(i: int) -> float:
        return outline.clip_length(candidates[i], max_clip)

    last = len(plan["parts"]) - 1

    def fits(k: int, i: int) -> bool:
        """Category tags call the segment a good opening (first part) or
        ending (last part) shot."""
        tags = candidates[i].tags
        return (k == 0 and tags.get("opening") is True) or (
            k == last and tags.get("ending") is True
        )

    notes: list[str] = []
    chosen: list[tuple[int, int, int]] = []  # (part, slot, candidate)
    slots_out: list[list[dict[str, Any]]] = []
    # A part whose scenes cannot hold its seconds passes the rest on to the
    # other parts, in proportion to the outline's own durations.
    planned = [float(part["duration_sec"]) for part in plan["parts"]]
    room = [
        sum(
            sum(map(length, kept[assigned[(k, j)]]))
            for j in range(len(part["scenes"]))
            if (k, j) in assigned
        )
        for k, part in enumerate(plan["parts"])
    ]
    budget = outline.share(sum(planned), room, planned)
    passed_on = sum(budget) > sum(planned) - 0.5
    if not passed_on:
        notes.append(f"可用素材只够 {sum(budget):.1f}s，目标 {sum(planned):.0f}s")
    for k, part in enumerate(plan["parts"]):
        if passed_on and budget[k] < planned[k] - 0.5:
            notes.append(
                f"「{part['name']}」素材只够 {budget[k]:.1f}s（计划 "
                f"{planned[k]:.0f}s），其余分给其他部分"
            )
        filled = [j for j in range(len(part["scenes"])) if (k, j) in assigned]
        capacity = [sum(map(length, kept[assigned[(k, j)]])) for j in filled]
        seconds = dict(zip(filled, outline.share(budget[k], capacity), strict=True))
        part_total = 0.0
        slots: list[dict[str, Any]] = []
        for j, query in enumerate(part["scenes"]):
            ranked = sorted(pool, key=lambda n: (-value[(k, j, n)], n))
            slot: dict[str, Any] = {
                "query": query["query"],
                "sources": query["sources"],
                "scene": assigned.get((k, j)),
                "relevance": None,
                "seconds": 0.0,
                "top": ranked[:5],
            }
            slots.append(slot)
            if (k, j) not in assigned:
                why = "符合的场景都已分给其他槽位" if pool else "没有可用的场景"
                notes.append(f"「{part['name']}」场景「{query['query']}」{why}")
                continue
            n = assigned[(k, j)]
            rel = outline.rescale({i: segment_relevance[k][j][i] for i in kept[n]})
            slot["relevance"] = round(scene_relevance[k][j][n], 3)
            slot["seconds"] = round(seconds[j], 1)
            picked = outline.pick_segments(
                rel,
                seconds[j],
                candidates,
                assessments,
                named | {i for i in kept[n] if fits(k, i)},
                max_clip,
                RELEVANCE_SHARE,
            )
            chosen += [(k, j, i) for i in picked]
            part_total += sum(map(length, picked))
        slots_out.append(slots)
        if part_total < 0.8 * budget[k]:
            notes.append(
                f"「{part['name']}」素材不足：选到 {part_total:.1f}s，"
                f"目标 {budget[k]:.1f}s"
            )

    def place(x: tuple[int, int, int]) -> tuple:
        """Part order; within the first part a good opening shot leads, within
        the last a good ending shot closes; else slot, then shooting order."""
        k, j, i = x
        tags = candidates[i].tags
        first = k == 0 and tags.get("opening") is True
        final = k == last and tags.get("ending") is True and not first
        return (k, not first, final, j, i)

    chosen.sort(key=place)
    # Slots take whole clips, so their sum can run past the target: then
    # every clip is cut to one common length that brings it back.
    lengths = [length(i) for _, _, i in chosen]
    if sum(lengths) > sum(planned) + 0.5:
        cap = outline.level(lengths, sum(planned), MIN_CLIP_SEC)
        notes.append(
            f"选中的镜头共 {sum(lengths):.1f}s，超过目标，每段最多取 {cap:.1f}s"
        )
        max_clip = cap if max_clip is None else min(max_clip, cap)
    clips = []
    for order, (k, j, i) in enumerate(chosen):
        c = candidates[i]
        slot = slots_out[k][j]
        start, end = outline.clip_span(c, max_clip, (speech or {}).get(c.video, []))
        mark = "，分段总结点名的亮点" if i in named else ""
        if k == 0 and c.tags.get("opening") is True:
            mark += "，适合开场"
        if k == last and c.tags.get("ending") is True:
            mark += "，适合收尾"
        if end - start < c.duration:
            mark += (
                f"，取 {start - c.start:.1f}–{end - c.start:.1f}s "
                f"处的 {end - start:.1f}s"
            )
        clips.append(
            Clip(
                order=order,
                role=plan["parts"][k]["name"],
                video=c.video,
                index=c.index,
                src_start=round(start, 3),
                src_end=round(end, 3),
                description=c.tags["description"],
                reason=f"场景「{slot['query']}」→ 场景 {slot['scene']}"
                f"（相关度 {slot['relevance']:.0%}），得分 "
                f"{assessments[i].score:.1f}，片段相关度 "
                f"{segment_relevance[k][j][i]:.0%}{mark}",
            )
        )
    validate_timeline(clips, candidates)
    return clips, notes, slots_out


# --- review --------------------------------------------------------------------

# Unused segments offered per part besides the cut, by retrieval relevance.
REVIEW_POOL = 12
# The reviewed cut may run this much under or over the target; a cut that
# runs over is then levelled down to it (outline.level).
REVIEW_UNDER_SEC = 1.0
REVIEW_OVER_SEC = 6.0
MAX_WHY_CHARS = 40
MAX_REVIEW_NOTES = 200
# What is said in a segment, as shown to the reviewer.
MAX_SAID_CHARS = 160

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "notes": {"type": "string", "minLength": 1, "maxLength": MAX_REVIEW_NOTES},
        "on_beat": {"type": "boolean"},
        "clips": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "segment": {"type": "string", "pattern": "^c[0-9]+$"},
                    "part": {"type": "string", "minLength": 1},
                    "seconds": {"type": "number", "minimum": MIN_CLIP_SEC},
                    "why": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_WHY_CHARS,
                    },
                },
                "required": ["segment", "part", "seconds", "why"],
            },
        },
    },
    "required": ["notes", "on_beat", "clips"],
}

REVIEW_PROMPT = """你是这条成片的剪辑总监。程序已经按大纲选出一版粗剪，\
请审片并定出最终剪辑。

用户的剪辑要求：「{instruction}」
成片时长：{target:.0f} 秒，最终各镜头秒数合计在 {low:.0f}–{high:.0f} 秒之间。
片名：{title}
大纲（按播放顺序）：
{parts}
{structure}
现在的粗剪，按播放顺序：
{current}

还没用上的候选镜头，按所属部分和检索相关度排列：
{pool}

镜头每行：编号 | 视频与原片时间、时长 | 连续镜头 | 画面描述 | 地标 | \
时段，光线，明暗 | 精彩度（1–5）| 画质分（1–5）| 景别{speech_note}
“连续镜头”相同（如都是 s12）的几段出自原片同一个不间断的镜头。
“时段”和“光线”是看图写的，常把白天的逆光写成黄昏；“明暗”是程序量出来的
画面亮度，更可靠：夜景画面很暗，白天和黄昏画面明亮。

审片时：
- 先看整体：开场第一个镜头要能抓人、交代地点或主角；结尾镜头要有收束感；
  中间按大纲推进；
- 先后：要求里有时间先后（如从白天到夜晚、春夏秋冬、一天的行程）时，镜头
  要按这个先后排；白天和夜晚先看“明暗”，再看画面描述，不要只看“时段”；
  放错时段的镜头换掉；
- 衔接：相邻两个镜头要在景别、角度或场景上有明显变化，不要连着几个几乎
  一样的画面；同一连续镜头的两段不要前后挨着放，那会像跳切；换场景时用
  远景、空镜或一个动作的开始和结束来过渡；同一场景里景别可以由远到近推进，
  运动方向前后一致；
- 节奏：每个镜头的秒数按内容定，动作、人物和快切段落短，风景、建筑和
  需要看清的画面长；不要每段都一样长；
- 画面描述是依据，不要用描述里没有的内容做判断；画质分低或有晃动的镜头
  只在内容不可替代时使用；电影和夜景本来就暗，暗不算毛病；
- 不要用片头片尾字幕、黑场或纯文字画面；画面描述里提到字幕、名单、
  文字或黑屏的镜头都不要用；
- 有对白的镜头按台词取舍：要用就让秒数盖住完整的句子（对白前的数字是句子在
  镜头里的起止秒数），不要停在半句话上；程序也会把切到的句子补全；
- 可以替换、删除、增加或调整顺序，只能用上面列出的编号，每个编号最多用一次，
  秒数不超过这个镜头的时长。

输出 JSON：
- notes：一两句话说明你对粗剪改了什么、为什么；
- on_beat：剪辑要求让切点卡在音乐节拍上时写 true（程序会把每个镜头的起点
  挪到节拍上、秒数取整拍），否则写 false；
- clips：最终的镜头，按播放顺序，每项 segment 是编号（如 c12），part 是所属部分的
  名称（大纲里的名称之一），seconds 是这个镜头在成片里放多少秒，why 是一句话理由。
只输出 JSON。"""


# Mean luma (signals.py) below which a segment reads as very dark (night,
# lit by lamps) and as dim (lights coming on, deep shade); measured on day,
# dusk and night aerials: night 8-36, lights on about 50, day and dusk 80+.
VERY_DARK_LUMA = 45.0
DIM_LUMA = 75.0


def lightness(brightness: float) -> str:
    """The measured brightness of a segment, in words for the review."""
    if brightness < VERY_DARK_LUMA:
        return "画面很暗"
    if brightness < DIM_LUMA:
        return "画面偏暗"
    return "画面明亮"


def _segment_line(
    i: int,
    c: Candidate,
    a: Assessment,
    videos: dict[str, int],
    lines: dict[str, list[tuple[float, float, str]]],
) -> str:
    """One segment for the review: where it is, what is seen and, when
    someone speaks in it, what is said (sentences with their seconds into
    the segment)."""
    tags = c.tags
    where = f"视频{videos[c.video]} " if len(videos) > 1 else ""
    light = f"{tags['time_of_day']}，{tags.get('light') or '—'}"
    if c.brightness is not None:
        light += f"，{lightness(c.brightness)}"
    said = [
        f"{max(0.0, s - c.start):.1f}–{min(c.duration, e - c.start):.1f}s「{text}」"
        for s, e, text in lines.get(c.video, [])
        if s < c.end and e > c.start
    ]
    words = " ".join(said)
    if len(words) > MAX_SAID_CHARS:
        words = words[:MAX_SAID_CHARS] + "…"
    return (
        f"c{i} | {where}{outline._fmt(c.start)}–{outline._fmt(c.end)} "
        f"{c.duration:.1f}s | s{c.shot} | {tags['description']} | "
        f"{tags.get('landmark') or '—'} | {light} | "
        f"{c.highlight:.1f} | {a.score:.1f} | {tags['shot_size']}"
        + (f" | 对白：{words}" if said else "")
    )


def _fit(
    spans: list[tuple[float, float]],
    picked: list[int],
    candidates: list[Candidate],
    speech: dict[str, list[tuple[float, float]]],
    target_sec: float,
) -> tuple[list[tuple[float, float]], str]:
    """Bring a cut that runs over the target back to it by shortening the
    clips without speech (outline.level), so no sentence is cut; returns
    the spans and a note when the speech alone keeps it over."""
    total = sum(e - s for s, e in spans)
    if total <= target_sec + 0.5:
        return spans, ""

    def spoken(k: int) -> bool:
        s, e = spans[k]
        video = candidates[picked[k]].video
        return any(a < e and b > s for a, b in speech.get(video, []))

    free = [k for k in range(len(spans)) if not spoken(k)]
    fixed = sum(e - s for k, (s, e) in enumerate(spans) if k not in free)
    if free:
        cap = outline.level(
            [spans[k][1] - spans[k][0] for k in free], target_sec - fixed, MIN_CLIP_SEC
        )
        for k in free:
            c = candidates[picked[k]]
            length = min(spans[k][1] - spans[k][0], cap)
            spans[k] = outline.clip_span(c, length, speech.get(c.video, []))
    total = sum(e - s for s, e in spans)
    if total > target_sec + 0.5:
        return spans, f"对白镜头保留整句，成片 {total:.1f}s，超出目标"
    return spans, ""


# Sound looked at around a segment to find its beats.
BEAT_MARGIN_SEC = 4.0
BeatGrid = Callable[[str, float, float], tuple[float, list[float]]]


def _on_beats(
    items: list[dict[str, Any]],
    picked: list[int],
    candidates: list[Candidate],
    speech: dict[str, list[tuple[float, float]]],
    target_sec: float,
    beat_grid: BeatGrid,
) -> tuple[list[tuple[float, float]], str]:
    """Windows that start on a beat and last whole beats (beats.snap), then
    whole beats taken off the longest clips while the cut runs over."""
    spans, periods = [], []
    for item, i in zip(items, picked, strict=True):
        c = candidates[i]
        period, times = beat_grid(
            c.video, max(0.0, c.start - BEAT_MARGIN_SEC), c.end + BEAT_MARGIN_SEC
        )
        start, _ = outline.clip_span(c, float(item["seconds"]), speech.get(c.video, []))
        spans.append(
            beats.snap(start, float(item["seconds"]), c.start, c.end, period, times)
        )
        periods.append(period)
    while sum(e - s for s, e in spans) > target_sec + 0.5:
        k = max(range(len(spans)), key=lambda k: spans[k][1] - spans[k][0])
        s, e = spans[k]
        if e - s - periods[k] < MIN_CLIP_SEC:
            total = sum(e - s for s, e in spans)
            return spans, f"按整拍剪，成片 {total:.1f}s，超出目标"
        spans[k] = (s, e - periods[k])
    return spans, ""


def validate_review(
    parsed: Any,
    offered: set[int],
    parts: list[str],
    candidates: list[Candidate],
    target_sec: float,
) -> dict[str, Any]:
    """The reviewed cut: offered segments once each, known parts, seconds
    within each segment and a total near the target; raises OutlineError."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("clips"), list):
        raise OutlineError("Expected an object with a clips list")
    if not isinstance(parsed.get("on_beat"), bool):
        raise OutlineError("on_beat must be true or false")
    if not str(parsed.get("notes", "")).strip():
        raise OutlineError("notes must not be empty")
    if not parsed["clips"]:
        raise OutlineError("clips must not be empty")
    used: set[int] = set()
    total = 0.0
    for k, item in enumerate(parsed["clips"], start=1):
        ref = str(item.get("segment", ""))
        if not (ref.startswith("c") and ref[1:].isdigit()):
            raise OutlineError(f"clip {k}: bad segment {ref!r}")
        i = int(ref[1:])
        if i not in offered:
            raise OutlineError(f"clip {k}: {ref} was not offered")
        if i in used:
            raise OutlineError(f"clip {k}: {ref} used twice")
        used.add(i)
        if item.get("part") not in parts:
            raise OutlineError(f"clip {k}: part {item.get('part')!r} not in {parts}")
        sec = item.get("seconds")
        if isinstance(sec, bool) or not isinstance(sec, (int, float)):
            raise OutlineError(f"clip {k}: seconds must be a number")
        if not MIN_CLIP_SEC <= sec <= candidates[i].duration + 0.05:
            raise OutlineError(
                f"clip {k}: {sec}s outside {MIN_CLIP_SEC}-"
                f"{candidates[i].duration:.1f}s of {ref}"
            )
        if not str(item.get("why", "")).strip():
            raise OutlineError(f"clip {k}: why must not be empty")
        total += sec
    if not target_sec - REVIEW_UNDER_SEC <= total <= target_sec + REVIEW_OVER_SEC:
        raise OutlineError(
            f"clips add up to {total:.1f}s, need "
            f"{target_sec - REVIEW_UNDER_SEC:.0f}-{target_sec + REVIEW_OVER_SEC:.0f}s"
        )
    return parsed


def review_request(
    instruction: str,
    target_sec: float,
    plan: dict[str, Any],
    clips: list[Clip],
    candidates: list[Candidate],
    assessments: list[Assessment],
    segment_relevance: list[list[dict[int, float]]],
    lines: dict[str, list[tuple[float, float, str]]] | None = None,
    server_url: str = outline.SERVER_URL,
    profile: Profile | None = None,
    beat_grid: BeatGrid | None = None,
) -> tuple[list[Clip], dict[str, Any], str, float, list[str]]:
    """The planner reviews the filled cut against the instruction and
    returns the final one: (clips, review, raw, seconds, failed tries).

    It sees the cut and, per part, the REVIEW_POOL unused segments most
    relevant to the part's slots, with what is said in them (``lines``:
    sentences per video), and may replace, drop, add or reorder them and
    set each clip's seconds. Each clip takes the window of its segment that
    cuts the least speech (outline.clip_span), widened to whole sentences
    (outline.whole_sentences); a cut that then runs over the target is
    brought back by shortening the clips without speech (``review["fit"]``
    notes when the speech alone runs over). When the planner answers
    ``on_beat``, each window instead starts on a beat of its sound and lasts
    whole beats (``beat_grid``: a video and a time range -> beat period and
    times), and a cut that runs over loses whole beats from its longest
    clips."""
    lines = lines or {}
    speech = {v: [(s, e) for s, e, _ in ls] for v, ls in lines.items()}
    position = {(c.video, c.index): i for i, c in enumerate(candidates)}
    videos: dict[str, int] = {}
    for c in candidates:
        videos.setdefault(c.video, len(videos) + 1)
    current = [position[(c.video, c.index)] for c in clips]
    offered = set(current)
    pool_lines = []
    for k, part in enumerate(plan["parts"]):
        relevance: dict[int, float] = {}
        for slot in segment_relevance[k]:
            for i, r in slot.items():
                relevance[i] = max(r, relevance.get(i, 0.0))
        best = sorted(
            (
                i
                for i in relevance
                if i not in offered
                and assessments[i].keep
                and candidates[i].duration >= MIN_CLIP_SEC
            ),
            key=lambda i: (-relevance[i], i),
        )[:REVIEW_POOL]
        offered |= set(best)
        pool_lines.append(f"【{part['name']}】")
        pool_lines += [
            _segment_line(i, candidates[i], assessments[i], videos, lines) for i in best
        ]
    current_lines = [
        f"{n}. [{clip.role}] 用 {clip.duration:.1f}s ← "
        + _segment_line(i, candidates[i], assessments[i], videos, lines)
        for n, (clip, i) in enumerate(zip(clips, current, strict=True), start=1)
    ]
    names = [p["name"] for p in plan["parts"]]
    prompt = REVIEW_PROMPT.format(
        instruction=instruction,
        target=target_sec,
        low=target_sec - REVIEW_UNDER_SEC,
        high=target_sec + REVIEW_OVER_SEC,
        title=plan["title"],
        parts="\n".join(
            f"- {p['name']}（约 {p['duration_sec']:.0f} 秒）：{p['summary']}"
            for p in plan["parts"]
        ),
        structure=(
            STRUCTURE_PROMPT.format(name=profile.name, structure=profile.structure)
            if profile
            else ""
        ),
        current="\n".join(current_lines),
        pool="\n".join(pool_lines),
        speech_note="；有人说话的镜头另列对白" if lines else "",
    )
    review, raw, seconds, errors = outline._ask(
        prompt,
        REVIEW_SCHEMA,
        "edit_review",
        lambda parsed: validate_review(parsed, offered, names, candidates, target_sec),
        server_url,
    )
    picked = [int(item["segment"][1:]) for item in review["clips"]]
    if review["on_beat"]:
        if beat_grid is None:
            raise OutlineError("on_beat needs the sound's beats (beat_grid)")
        spans, review["fit"] = _on_beats(
            review["clips"], picked, candidates, speech, target_sec, beat_grid
        )
    else:
        spans = []
        for item, i in zip(review["clips"], picked, strict=True):
            c = candidates[i]
            said = speech.get(c.video, [])
            start, end = outline.clip_span(c, float(item["seconds"]), said)
            spans.append(outline.whole_sentences(start, end, c.start, c.end, said))
        spans, review["fit"] = _fit(spans, picked, candidates, speech, target_sec)
    out = []
    for order, (item, i, (start, end)) in enumerate(
        zip(review["clips"], picked, spans, strict=True)
    ):
        c = candidates[i]
        out.append(
            Clip(
                order=order,
                role=item["part"],
                video=c.video,
                index=c.index,
                src_start=round(start, 3),
                src_end=round(end, 3),
                description=c.tags["description"],
                reason=f"审片：{item['why']}（得分 {assessments[i].score:.1f}）",
            )
        )
    validate_timeline(out, candidates)
    return out, review, raw, seconds, errors
