"""Scene material: every scene of a video with its rejected segments cut out.

Besides the suggested timelines, users get the footage itself to compose on
their own. Scenes are the same ones the highlights fill uses (consecutive
segments grouped by vector similarity, see retrieve.split_scenes). Inside a
scene, segments the edit suggestions reject (score <= DROP_SCORE, e.g. blurry,
shaky or too dark, or shorter than MIN_CLIP_SEC) are cut out, so this is also
a pass that removes unusable footage; the kept stretches are joined into one
clip per scene. The source video is never modified. Qwen3.5-4B writes a short
title and summary of what a scene keeps; everything else in the description
comes from the tags.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from video_demo import outline
from video_demo.suggest import DROP_SCORE, MIN_CLIP_SEC, Assessment, Candidate

# Bump when the summary prompt or schema changes; cached summaries of an
# older version are written again.
SCENES_VERSION = 1
MAX_TITLE_CHARS = 12
MAX_SUMMARY_CHARS = 80
TOP_TAGS = 8

SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1, "maxLength": MAX_TITLE_CHARS},
        "summary": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_SUMMARY_CHARS,
        },
    },
    "required": ["title", "summary"],
}

SCENE_PROMPT = """下面是视频素材里一个连续场景（{start}–{end}）保留下来的 {n} 个片段，
按时间排列。
每行：[片段编号] 时间 | 时段 | 得分（1–5，越高画面越好、越精彩） | 描述 | 标签

{lines}

请为这个场景写：
- title：不超过 {max_title} 个字的短标题，说明地点或主要内容；
- summary：1–2 句话的概况，说明画面里有什么、发生了什么，不超过 {max_summary} 个字。
只输出 JSON。"""


def usable(c: Candidate, a: Assessment) -> bool:
    """The same rule the highlights use to pick segments."""
    return a.keep and c.duration >= MIN_CLIP_SEC


def kept_runs(
    members: list[int], candidates: list[Candidate], assessments: list[Assessment]
) -> list[list[int]]:
    """Stretches of consecutive usable segments; rejected ones split them."""
    runs: list[list[int]] = [[]]
    for i in members:
        if usable(candidates[i], assessments[i]):
            runs[-1].append(i)
        elif runs[-1]:
            runs.append([])
    return [r for r in runs if r]


def validate_scene_summary(parsed: Any) -> dict[str, str]:
    if not isinstance(parsed, dict):
        raise outline.OutlineError(f"scene summary must be an object: {parsed!r}")
    out: dict[str, str] = {}
    for key, limit in (("title", MAX_TITLE_CHARS), ("summary", MAX_SUMMARY_CHARS)):
        text = str(parsed.get(key, "")).strip()
        if not text:
            raise outline.OutlineError(f"scene {key} is empty")
        if len(text) > limit:
            raise outline.OutlineError(f"scene {key} over {limit} chars: {text}")
        out[key] = text
    return out


def describe_scene(
    kept: list[int],
    candidates: list[Candidate],
    assessments: list[Assessment],
    server_url: str = outline.SERVER_URL,
) -> tuple[dict[str, str], str, float, list[str]]:
    """(title and summary, raw, seconds, failed tries) for one scene's kept
    segments. Raises OutlineError when every try fails."""
    first, last = candidates[kept[0]], candidates[kept[-1]]
    prompt = SCENE_PROMPT.format(
        start=outline._fmt(first.start),
        end=outline._fmt(last.end),
        n=len(kept),
        lines=outline.segment_lines(kept, candidates, assessments),
        max_title=MAX_TITLE_CHARS,
        max_summary=MAX_SUMMARY_CHARS,
    )
    return outline._ask(
        prompt, SCENE_SCHEMA, "scene_summary", validate_scene_summary, server_url
    )


def _segment(c: Candidate, a: Assessment) -> dict[str, Any]:
    return {
        "index": c.index,
        "start": round(c.start, 3),
        "end": round(c.end, 3),
        "score": a.score,
        "description": c.tags["description"],
    }


def removal_reason(c: Candidate, a: Assessment) -> str:
    if not a.keep:
        return f"得分 {a.score:.1f} ≤ {DROP_SCORE:.0f}"
    return f"时长 {c.duration:.1f}s 短于 {MIN_CLIP_SEC}s"


def kept_spans(
    runs: list[list[int]], candidates: list[Candidate]
) -> list[tuple[float, float]]:
    """Source [start, end) of each kept stretch, in order: what the scene's
    clip is cut from."""
    return [
        (round(candidates[r[0]].start, 3), round(candidates[r[-1]].end, 3))
        for r in runs
    ]


def scene_entry(
    number: int,
    members: list[int],
    candidates: list[Candidate],
    assessments: list[Assessment],
    summary: dict[str, str] | None,
    clip_file: str | None,
) -> dict[str, Any]:
    """The scenes.json record of one scene. ``summary`` and ``clip_file`` are
    None exactly when every segment was rejected (nothing to cut or
    summarise)."""
    runs = kept_runs(members, candidates, assessments)
    for name, value in (("summary", summary), ("clip_file", clip_file)):
        if (value is None) != (not runs):
            raise ValueError(f"scene {number}: {name} must exist exactly when kept")
    kept = [i for r in runs for i in r]
    spans = kept_spans(runs, candidates)
    at = [0.0]
    for start, end in spans:
        at.append(at[-1] + end - start)
    tags = [c.tags for c in (candidates[i] for i in kept)]
    first, last = candidates[members[0]], candidates[members[-1]]
    return {
        "number": number,
        "start": round(first.start, 3),
        "end": round(last.end, 3),
        **(summary or {}),
        "kept_sec": round(sum(candidates[i].duration for i in kept), 2),
        "time_of_day": (
            Counter(t["time_of_day"] for t in tags).most_common(1)[0][0]
            if tags
            else None
        ),
        "shot_sizes": list(dict.fromkeys(t["shot_size"] for t in tags)),
        "tags": [
            tag
            for tag, _ in Counter(x for t in tags for x in t["tags"]).most_common(
                TOP_TAGS
            )
        ],
        "file": clip_file,
        "duration": round(at[-1], 3),
        # Where each kept stretch comes from in the source, and where it
        # starts in the scene's clip.
        "parts": [
            {
                "start": start,
                "end": end,
                "at": round(offset, 3),
                "segments": [_segment(candidates[i], assessments[i]) for i in r],
            }
            for r, (start, end), offset in zip(runs, spans, at, strict=False)
        ],
        "removed": [
            {
                **_segment(candidates[i], assessments[i]),
                "why": removal_reason(candidates[i], assessments[i]),
                "reasons": assessments[i].reasons,
            }
            for i in members
            if not usable(candidates[i], assessments[i])
        ],
    }
