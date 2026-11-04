"""Shortlist segments before the 4B tags them: tagging is the slow step on
the CPU (about a minute a segment), so only the segments a cut could use
are tagged and scored.

No model is used here, only what ingest already measured: the video is cut
into equal stretches of time and the best segment of each is kept, so the
whole video stays covered; blurry, shaky and very short segments lose,
dark ones lose a little (dark can be the look of the film), and segments
with speech gain a little. A long shot split into windows keeps at most
MAX_PER_SHOT of them.

The segments file is rewritten with the kept segments under their own
numbers; the full list is kept beside it (``*.full.json``) and is what a
second run reads, so shortlisting again gives the same answer. Signals,
key frames and tags are looked up by number and need no change.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from video_demo import layout, subtitle

# A clip in our cuts is about this long, and each clip of the cut gets this
# many candidates to choose from.
CLIP_SEC = 3.0
PICKS_PER_CLIP = 5
MIN_PICKS = 40
MAX_PER_SHOT = 2
SHORT_SEC = 1.5
FLAG_COST = {"blurry": 1.0, "shaky": 1.0, "low_light": 0.5}
SHORT_COST = 0.5
SPEECH_GAIN = 0.3


@dataclass(frozen=True)
class Piece:
    index: int
    start: float
    end: float
    shot: int
    score: float


def quota(target_seconds: float) -> int:
    """How many segments to keep for a cut of ``target_seconds``."""
    return max(MIN_PICKS, math.ceil(target_seconds / CLIP_SEC * PICKS_PER_CLIP))


def _overlap(start: float, end: float, spans: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in spans)


def score(
    start: float, end: float, flags: list[str], spoken: list[tuple[float, float]]
) -> float:
    """Higher is more worth tagging."""
    length = end - start
    value = -sum(FLAG_COST.get(f, 0.0) for f in flags)
    if length < SHORT_SEC:
        value -= SHORT_COST
    return value + SPEECH_GAIN * _overlap(start, end, spoken) / length


def pick(pieces: list[Piece], n: int, start: float, end: float) -> list[Piece]:
    """At most ``n`` of ``pieces`` (in time order), the best of each of
    ``n`` equal stretches of [start, end); stretches left empty are filled
    with the best of the rest. At most MAX_PER_SHOT per shot."""
    if n >= len(pieces):
        return list(pieces)
    width = (end - start) / n
    bins: list[list[Piece]] = [[] for _ in range(n)]
    for p in pieces:
        mid = (p.start + p.end) / 2
        bins[min(n - 1, max(0, int((mid - start) / width)))].append(p)
    kept: dict[int, Piece] = {}
    per_shot: dict[int, int] = {}

    def take(p: Piece) -> bool:
        if p.index in kept or per_shot.get(p.shot, 0) >= MAX_PER_SHOT:
            return False
        kept[p.index] = p
        per_shot[p.shot] = per_shot.get(p.shot, 0) + 1
        return True

    for b in bins:
        for p in sorted(b, key=lambda p: -p.score):
            if take(p):
                break
    for p in sorted(pieces, key=lambda p: -p.score):
        if len(kept) >= n:
            break
        take(p)
    return sorted(kept.values(), key=lambda p: p.index)


def shares(durations: dict[str, float], n: int) -> dict[str, int]:
    """``n`` split over videos by their length, at least one each."""
    total = sum(durations.values())
    return {v: max(1, round(n * d / total)) for v, d in durations.items()}


def shortlist(work: Path, ids: list[str], target_seconds: float) -> dict[str, int]:
    """Shortlist the segments of videos ``ids`` in the data/work folder
    ``work`` for a cut of ``target_seconds``; returns kept counts by id."""
    full: dict[str, dict] = {}
    for vid in ids:
        path = layout.segments_path(work, vid)
        kept_all = path.with_name(path.stem + ".full.json")
        if not kept_all.exists():
            kept_all.write_text(path.read_text("utf-8"), "utf-8")
        full[vid] = json.loads(kept_all.read_text("utf-8"))
    durations = {v: d["analysed_until"] for v, d in full.items()}
    counts = {}
    for vid, n in shares(durations, quota(target_seconds)).items():
        data = full[vid]
        signals = json.loads(layout.signals_path(work, vid).read_text("utf-8"))
        asr = layout.asr_path(work, vid)
        spoken = (
            subtitle.speech_spans(subtitle.read_asr(asr, vid))[0]
            if asr.exists()
            else []
        )
        pieces = [
            Piece(
                s["index"],
                s["start"],
                s["end"],
                s["shot"],
                score(s["start"], s["end"], signals[s["index"]]["flags"], spoken),
            )
            for s in data["segments"]
        ]
        kept = {p.index for p in pick(pieces, n, 0.0, durations[vid])}
        trimmed = dict(data)
        trimmed["segments"] = [s for s in data["segments"] if s["index"] in kept]
        trimmed["shortlist"] = {"kept": len(kept), "of": len(data["segments"])}
        layout.segments_path(work, vid).write_text(
            json.dumps(trimmed, ensure_ascii=False, indent=2), "utf-8"
        )
        counts[vid] = len(kept)
    return counts
