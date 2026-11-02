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
from typing import TYPE_CHECKING, Any

from video_demo import outline

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
   哪些场景，为什么这样安排；写完对照用户的要求检查内容和先后顺序；
3. pace：成片的剪辑节奏，按用户的要求判断：整段（每段素材完整保留，约 10 秒，
   适合慢慢看的风景、过程）、适中（每段最多 6 秒）、快（每段最多 3 秒，
   适合节奏轻快的混剪）；
4. unmet：要求里提到、但场景里找不到的内容，都找得到就留空；
5. parts：照 design 分成 1–{max_parts} 个部分（不能超过 {max_parts} 个），按成片里的
   先后排列，先后由 design 决定，不必按拍摄时间：
   - name 是这一部分的标题，summary 用一句话说明这部分讲什么；
   - weight 是这一部分的分量（1–10 的整数），程序会按分量把 {target_sec:.0f} 秒
     分给各部分，每部分至少 {min_part:.0f} 秒；
   - scenes 是这一部分要的 1–{max_slots} 个镜头：query 用一个短语说明要的画面
     （如“远处的城堡”“教堂里的彩色玻璃窗”），所有部分的 query 互不重复；
     sources 写这个画面取自 materials 里的哪几个场景（1–{max_sources} 个编号）。
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
