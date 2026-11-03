"""Rule-based edit suggestions from segment signals, quality scores and tags.

Scoring and slot filling are deterministic so every suggestion can be traced
back to a signal, a DOVER quality score or a VLM tag.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

# Segments shorter than this are penalised and never selected.
MIN_CLIP_SEC = 1.5
DROP_SCORE = 2.0
DUP_JACCARD = 0.5
# "Is this shot good" comes mainly from the DOVER quality model (0-1); the VLM
# highlight (1-5 integer, mostly 3 or 4 in practice) keeps a small weight.
AESTHETIC_WEIGHT = 0.8
VLM_WEIGHT = 0.2


@dataclass(frozen=True)
class Candidate:
    video: str
    index: int
    shot: int
    start: float
    end: float
    tags: dict[str, Any]
    # VLM highlight on 1-5: the probability-weighted mean of the digits the
    # model could write (tag.highlight_score), not just the one it wrote
    highlight: float
    flags: list[str]
    sharpness: float
    # sharpness / median sharpness of the same video
    sharpness_rel: float
    # DOVER fused quality in [0, 1] and its raw technical / aesthetic outputs
    quality: float
    tech_raw: float
    aes_raw: float

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
    """Score on a 1-5 scale from DOVER quality (80%) and VLM highlight (20%),
    minus penalties for measured defects."""
    tags = c.tags
    if c.sharpness_rel < 0:
        raise ValueError(f"sharpness_rel must be non-negative: {c}")
    if not 0.0 <= c.quality <= 1.0:
        raise ValueError(f"quality must be in [0, 1]: {c}")
    if not 1.0 <= c.highlight <= 5.0:
        raise ValueError(f"highlight must be in [1, 5]: {c}")
    highlight = (c.highlight - 1) / 4
    combined = AESTHETIC_WEIGHT * c.quality + VLM_WEIGHT * highlight
    score = 1 + 4 * combined
    reasons = [
        f"美学模型 {c.quality:.2f}（技术 {c.tech_raw:+.3f} / 美学 {c.aes_raw:+.3f}）",
        f"VLM 精彩度 {c.highlight:.2f}/5（权重 {VLM_WEIGHT:.0%}）",
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
    if c.duration < MIN_CLIP_SEC:
        score -= 0.5
        reasons.append(f"时长仅 {c.duration:.1f}s")
    keep = score > DROP_SCORE
    return Assessment(c.video, c.index, round(score, 3), keep, reasons)


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


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


def clip_to_dict(clip: Clip) -> dict[str, Any]:
    return {**asdict(clip), "duration": round(clip.duration, 3)}


# Highlights mode: about 1/6 of the input (1 h -> 10 min) as whole segments,
# for the user to edit further rather than a finished cut.
HIGHLIGHT_RATIO = 1 / 6
# At least one segment from every window of this length, so the selection
# covers the whole recording instead of clustering in one part.
COVERAGE_WINDOW_SEC = 300.0
# Tag overlap only counts as a repeat between segments this close in time;
# a long walk shares tags all the way through.
DUP_NEAR_SEC = 60.0


def _percentiles(values: list[float]) -> list[float]:
    """Mid-rank percentile in (0, 1); a single value gets 0.5."""
    order = sorted(values)
    n = len(values)
    return [
        (sum(v < x for v in order) + 0.5 * sum(v == x for v in order)) / n
        for x in values
    ]


def calibrate_by_time(candidates: list[Candidate]) -> list[Candidate]:
    """Replace DOVER quality with its percentile among segments of the same
    time of day, so night footage is not ranked against daylight."""
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(candidates):
        groups.setdefault(c.tags["time_of_day"], []).append(i)
    quality = [0.0] * len(candidates)
    for members in groups.values():
        pct = _percentiles([candidates[i].quality for i in members])
        for i, p in zip(members, pct, strict=True):
            quality[i] = p
    return [replace(c, quality=q) for c, q in zip(candidates, quality, strict=True)]


def build_highlights(
    candidates: list[Candidate],
    assessments: list[Assessment],
    target_sec: float,
    offsets: dict[str, float] | None = None,
) -> tuple[list[Clip], list[str]]:
    """Pick whole segments up to ``target_sec``, covering every window.

    ``offsets`` places each video on one timeline (the sum of the analysed
    lengths before it) for coverage and ordering; videos default to 0.
    Clips keep shooting order. Selection depends only on segment content.
    """
    if len(candidates) != len(assessments):
        raise ValueError("candidates and assessments must align")
    offsets = offsets or {}

    def at(c: Candidate) -> float:
        return offsets.get(c.video, 0.0) + c.start

    pool = sorted(
        (
            (a.score, c)
            for c, a in zip(candidates, assessments, strict=True)
            if a.keep and c.duration >= MIN_CLIP_SEC
        ),
        key=lambda x: (-x[0], -x[1].sharpness_rel, x[1].tags["description"]),
    )
    chosen: list[tuple[str, float, Candidate]] = []

    def usable(c: Candidate) -> bool:
        for _, _, d in chosen:
            if d.video == c.video and d.index == c.index:
                return False
            near = d.video == c.video and abs(d.start - c.start) <= DUP_NEAR_SEC
            if near and _jaccard(d.tags["tags"], c.tags["tags"]) >= DUP_JACCARD:
                return False
        return True

    total = 0.0
    windows: dict[int, list[tuple[float, Candidate]]] = {}
    for score, c in pool:
        windows.setdefault(int(at(c) // COVERAGE_WINDOW_SEC), []).append((score, c))
    for _, members in sorted(windows.items()):
        for score, c in members:
            if usable(c):
                chosen.append(("覆盖", score, c))
                total += c.duration
                break
    for score, c in pool:
        if total >= target_sec:
            break
        if usable(c):
            chosen.append(("精彩", score, c))
            total += c.duration

    notes: list[str] = []
    if total < target_sec * 0.8:
        notes.append(f"可用素材不足：去重后只选到 {total:.1f}s，目标 {target_sec:.0f}s")
    chosen.sort(key=lambda x: at(x[2]))
    clips = [
        Clip(
            order=order,
            role=role,
            video=c.video,
            index=c.index,
            src_start=round(c.start, 3),
            src_end=round(c.end, 3),
            description=c.tags["description"],
            reason=f"得分 {score:.1f}，{c.tags['shot_size']}，"
            + "、".join(c.tags["tags"][:3]),
        )
        for order, (role, score, c) in enumerate(chosen)
    ]
    validate_timeline(clips, candidates)
    return clips, notes
