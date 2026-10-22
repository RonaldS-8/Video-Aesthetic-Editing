"""Boundary-level precision / recall / F1 for shot detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryScore:
    precision: float
    recall: float
    f1: float
    matched: list[tuple[float, float]]
    missed: list[float]
    false_alarms: list[float]


def score_boundaries(
    predicted: list[float], truth: list[float], tolerance: float = 0.5
) -> BoundaryScore:
    """Greedy one-to-one matching of cut times within ``tolerance`` seconds."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    pairs = sorted(
        (abs(p - t), p, t) for p in predicted for t in truth if abs(p - t) <= tolerance
    )
    used_p: set[float] = set()
    used_t: set[float] = set()
    matched: list[tuple[float, float]] = []
    for _, p, t in pairs:
        if p in used_p or t in used_t:
            continue
        used_p.add(p)
        used_t.add(t)
        matched.append((p, t))
    tp = len(matched)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return BoundaryScore(
        precision=precision,
        recall=recall,
        f1=f1,
        matched=sorted(matched, key=lambda x: x[1]),
        missed=sorted(t for t in truth if t not in used_t),
        false_alarms=sorted(p for p in predicted if p not in used_p),
    )
