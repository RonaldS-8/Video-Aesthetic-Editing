"""Rule-based edit suggestions from segment signals and VLM tags.

Scoring and slot filling are deterministic so every suggestion can be traced
back to a signal or a tag; the VLM only supplies the tags.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

WIDE_SHOTS = {"远景", "全景"}
CLOSING_TIMES = {"黄昏", "夜晚"}
# Body order for a multi-video reel: day -> dusk -> night, never by file name.
TIME_ORDER = {"白天": 0, "室内": 1, "不确定": 1, "黄昏": 2, "夜晚": 3}
MAX_CLIP_SEC = 4.0
MIN_CLIP_SEC = 1.5
DROP_SCORE = 2.0
DUP_JACCARD = 0.5
# The VLM highlight is a 1-5 integer and most segments get 3 or 4, so a
# continuous sharpness term (relative to the segment's own video) separates
# ties. +/-0.5 at 2x / 0.5x the video's median sharpness.
SHARPNESS_WEIGHT = 0.5


@dataclass(frozen=True)
class Candidate:
    video: str
    index: int
    shot: int
    start: float
    end: float
    tags: dict[str, Any]
    flags: list[str]
    sharpness: float
    # sharpness / median sharpness of the same video
    sharpness_rel: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Assessment:
    video: str
    index: int
    score: float
    keep: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Clip:
    order: int
    role: str
    video: str
    index: int
    src_start: float
    src_end: float
    description: str
    reason: str

    @property
    def duration(self) -> float:
        return self.src_end - self.src_start


def assess(c: Candidate) -> Assessment:
    """Score = VLM highlight (1-5) + sharpness term - penalties for defects."""
    tags = c.tags
    if c.sharpness_rel < 0:
        raise ValueError(f"sharpness_rel must be non-negative: {c}")
    # Saturates at 0.5x and 2x the median; a flat black frame (0) gets -0.5.
    rel = min(2.0, max(0.5, c.sharpness_rel))
    bonus = SHARPNESS_WEIGHT * math.log2(rel)
    score = float(tags["highlight"]) + bonus
    reasons = [
        f"精彩度 {tags['highlight']}/5",
        f"清晰度为本片中位数的 {c.sharpness_rel:.2f} 倍（{bonus:+.2f}）",
    ]
    if "blurry" in c.flags:
        score -= 1.5
        reasons.append(f"画面偏糊（清晰度 {c.sharpness:.0f}）")
    if "shaky" in c.flags:
        score -= 1.0
        reasons.append("抖动明显")
    if "low_light" in c.flags and tags["time_of_day"] != "夜晚":
        score -= 1.0
        reasons.append("非夜景但亮度过低")
    vlm_issues = [i for i in tags["issues"] if i not in ("无明显问题", "过暗")]
    if vlm_issues:
        score -= 0.5 * len(vlm_issues)
        reasons.append("模型指出：" + "、".join(vlm_issues))
    if c.duration < MIN_CLIP_SEC:
        score -= 0.5
        reasons.append(f"时长仅 {c.duration:.1f}s")
    keep = score > DROP_SCORE
    return Assessment(c.video, c.index, round(score, 3), keep, reasons)


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def _trim(c: Candidate) -> tuple[float, float]:
    """Centre a clip of at most MAX_CLIP_SEC inside the segment."""
    length = min(c.duration, MAX_CLIP_SEC)
    mid = (c.start + c.end) / 2
    return round(mid - length / 2, 3), round(mid + length / 2, 3)


def build_timeline(
    candidates: list[Candidate],
    assessments: list[Assessment],
    target_sec: float,
) -> tuple[list[Clip], list[str]]:
    """Fill opening / body / closing slots up to ``target_sec``.

    Selection depends only on segment content (scores, tags, signals), never
    on file names or input order.
    Returns the clips and notes describing any shortfall.
    """
    if len(candidates) != len(assessments):
        raise ValueError("candidates and assessments must align")
    pool = sorted(
        (
            (a.score, c)
            for c, a in zip(candidates, assessments, strict=True)
            if a.keep and c.duration >= MIN_CLIP_SEC
        ),
        # Exact score ties fall back to content, not to the file name.
        key=lambda x: (-x[0], -x[1].sharpness_rel, x[1].tags["description"]),
    )
    chosen: list[tuple[str, float, Candidate]] = []

    def usable(c: Candidate) -> bool:
        for _, _, d in chosen:
            # Neighbouring windows of one shot look alike; distant windows of a
            # long single take are distinct moments and may both be used.
            if d.video == c.video and d.shot == c.shot and abs(d.index - c.index) <= 1:
                return False
            if _jaccard(d.tags["tags"], c.tags["tags"]) >= DUP_JACCARD:
                return False
        return True

    def pick(role: str, pred: Any) -> None:
        for score, c in pool:
            if pred(c) and usable(c):
                chosen.append((role, score, c))
                return

    pick("开场", lambda c: c.tags["shot_size"] in WIDE_SHOTS)
    pick(
        "收尾",
        lambda c: c.tags["shot_size"] in WIDE_SHOTS
        or c.tags["time_of_day"] in CLOSING_TIMES,
    )
    total = sum(_trim(c)[1] - _trim(c)[0] for _, _, c in chosen)
    for score, c in pool:
        if total >= target_sec:
            break
        if usable(c):
            chosen.append(("主体", score, c))
            total += _trim(c)[1] - _trim(c)[0]

    notes: list[str] = []
    if total < target_sec * 0.8:
        notes.append(f"可用素材不足：去重后只凑到 {total:.1f}s，目标 {target_sec:.0f}s")
    opening = [x for x in chosen if x[0] == "开场"]
    closing = [x for x in chosen if x[0] == "收尾"]
    body = [x for x in chosen if x[0] == "主体"]
    if len({c.video for _, _, c in chosen}) == 1:
        body.sort(key=lambda x: x[2].start)  # one video: keep its own chronology
    else:
        body.sort(key=lambda x: (TIME_ORDER[x[2].tags["time_of_day"]], -x[1]))
    clips: list[Clip] = []
    for order, (role, score, c) in enumerate(opening + body + closing):
        s, e = _trim(c)
        clips.append(
            Clip(
                order=order,
                role=role,
                video=c.video,
                index=c.index,
                src_start=s,
                src_end=e,
                description=c.tags["description"],
                reason=f"得分 {score:.1f}，{c.tags['shot_size']}，"
                + "、".join(c.tags["tags"][:3]),
            )
        )
    validate_timeline(clips, candidates)
    return clips, notes


def validate_timeline(clips: list[Clip], candidates: list[Candidate]) -> None:
    """Deterministic checks; raise instead of emitting a broken timeline."""
    by_key = {(c.video, c.index): c for c in candidates}
    seen: set[tuple[str, int]] = set()
    for clip in clips:
        key = (clip.video, clip.index)
        if key in seen:
            raise ValueError(f"Segment used twice: {key}")
        seen.add(key)
        seg = by_key[key]
        if not (seg.start - 1e-3 <= clip.src_start < clip.src_end <= seg.end + 1e-3):
            raise ValueError(f"Clip {clip} exceeds its segment {seg}")
        if clip.duration > MAX_CLIP_SEC + 1e-3:
            raise ValueError(f"Clip longer than {MAX_CLIP_SEC}s: {clip}")


def clip_to_dict(clip: Clip) -> dict[str, Any]:
    return {**asdict(clip), "duration": round(clip.duration, 3)}
