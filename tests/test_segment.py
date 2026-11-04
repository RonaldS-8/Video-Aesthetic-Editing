from __future__ import annotations

import numpy as np
import pytest

from video_demo.segment import (
    Cut,
    combine_cuts,
    find_dissolves,
    merge_short,
    spans_from_cuts,
    split_long,
)


def test_split_long_keeps_short_shots_whole() -> None:
    segs = split_long([(0.0, 5.0), (5.0, 9.0)], max_len=8.0)
    assert [(s.start, s.end, s.shot) for s in segs] == [(0.0, 5.0, 0), (5.0, 9.0, 1)]


def test_split_long_equal_windows_at_boundary() -> None:
    # Exactly 2 * max_len must give two windows, not three.
    segs = split_long([(0.0, 16.0)], max_len=8.0)
    assert [(s.start, s.end) for s in segs] == [(0.0, 8.0), (8.0, 16.0)]
    segs = split_long([(0.0, 16.1)], max_len=8.0)
    assert len(segs) == 3
    assert segs[-1].end == 16.1
    assert [s.index for s in segs] == [0, 1, 2]


def test_split_long_moves_splits_between_sentences() -> None:
    sentences = [(8.5, 10.5), (10.8, 14.0)]
    segs = split_long([(0.0, 20.0)], max_len=10.0, sentences=sentences)
    assert [(s.start, s.end) for s in segs] == [(0.0, 10.5), (10.5, 20.0)]


def test_split_long_then_between_clauses_then_stays() -> None:
    sentences = [(7.0, 13.0)]
    clauses = [(7.0, 9.6), (9.6, 13.0)]
    segs = split_long([(0.0, 20.0)], 10.0, sentences, clauses)
    assert segs[0].end == 9.6
    # Speech all around the split point: it stays at the equal window.
    segs = split_long([(0.0, 20.0)], 10.0, sentences)
    assert segs[0].end == 10.0


def test_split_long_shift_is_bounded() -> None:
    # The only gap is 3 s away, beyond SPLIT_SHIFT_SEC.
    segs = split_long([(0.0, 20.0)], 10.0, [(5.0, 13.0), (13.0, 19.0)])
    assert segs[0].end == 10.0
    # Later splits share what is left: 30 s, first split moved to 9.5.
    segs = split_long([(0.0, 30.0)], 10.0, [(9.5, 11.0)])
    assert [s.end for s in segs] == [9.5, 19.75, 30.0]


def test_split_long_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        split_long([(3.0, 3.0)])
    with pytest.raises(ValueError):
        split_long([(0.0, 1.0)], max_len=0)


def test_merge_short_folds_into_previous_and_first_into_next() -> None:
    spans = [(0.0, 0.4), (0.4, 5.0), (5.0, 5.5), (5.5, 9.0)]
    assert merge_short(spans, min_len=1.0) == [(0.0, 5.5), (5.5, 9.0)]


def test_spans_from_cuts_tiles_range() -> None:
    cuts = [Cut(2.0, "hard"), Cut(5.0, "dissolve"), Cut(12.0, "hard")]
    assert spans_from_cuts(cuts, 10.0) == [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0)]
    assert spans_from_cuts([], 7.5) == [(0.0, 7.5)]


def test_combine_cuts_drops_dissolve_next_to_hard_cut() -> None:
    cuts = combine_cuts([3.0, 10.0], [3.6, 7.0])
    assert [(c.time, c.kind) for c in cuts] == [
        (3.0, "hard"),
        (7.0, "dissolve"),
        (10.0, "hard"),
    ]


def _textures(seed: int, n: int = 2) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(0, 255, (54, 96)).astype(np.float32) for _ in range(n)]


def test_find_dissolves_detects_linear_blend() -> None:
    a, b = _textures(0)
    frames = [a] * 20
    frames += [a * (1 - k / 11) + b * (k / 11) for k in range(1, 11)]
    frames += [b] * 20
    times = np.arange(len(frames)) / 10
    hits = find_dissolves(times, np.stack(frames))
    assert len(hits) == 1
    assert abs(hits[0] - 2.5) <= 0.2


def test_find_dissolves_ignores_camera_pan() -> None:
    # Negative case: a pan changes content a lot but is not a blend.
    (wide,) = _textures(1, 1)
    wide = np.concatenate([wide] * 4, axis=1)
    frames = [wide[:, 4 * k : 4 * k + 96] for k in range(50)]
    times = np.arange(len(frames)) / 10
    assert find_dissolves(times, np.stack(frames)) == []
