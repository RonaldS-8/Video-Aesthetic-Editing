"""Boundary-level precision / recall / F1 for shot detection, and rank
correlation between scorers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their positions)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    for value in np.unique(x):
        tied = x == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation with tie handling (Pearson on average ranks)."""
    if len(a) != len(b):
        raise ValueError("inputs must have the same length")
    if len(a) < 2:
        raise ValueError("need at least two pairs")
    ra, rb = _ranks(np.asarray(a, float)), _ranks(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        raise ValueError("an input is constant; correlation is undefined")
    return float(np.corrcoef(ra, rb)[0, 1])
