from __future__ import annotations

import pytest

from video_demo import quality


def test_percentiles_rank_each_value_within_all() -> None:
    assert quality.percentiles([3.0, 1.0, 2.0, 2.0]) == [1.0, 0.25, 0.75, 0.75]


def test_night_bias_compares_medians_and_counts_night_in_top() -> None:
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.6]
    night = [False, False, False, True, True, True]
    bias = quality.night_bias(scores, night, top_n=3)
    assert bias.night_in_top == 0
    assert bias.night_median < bias.day_median
    assert quality.night_bias(scores, night, top_n=4).night_in_top == 1


def test_night_bias_needs_both_groups_and_matching_lengths() -> None:
    with pytest.raises(ValueError, match="both"):
        quality.night_bias([1.0, 2.0], [True, True])
    with pytest.raises(ValueError, match="same length"):
        quality.night_bias([1.0, 2.0], [True])


def test_worst_rank_counts_from_the_lowest_score() -> None:
    scores = [0.5, 0.1, 0.9, 0.1]
    assert quality.worst_rank(scores, 1) == 1
    assert quality.worst_rank(scores, 3) == 1
    assert quality.worst_rank(scores, 0) == 3
    assert quality.worst_rank(scores, 2) == 4


def test_rank_matrix_is_symmetric_with_unit_diagonal() -> None:
    m = quality.rank_matrix({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    assert m["a"]["a"] == 1.0 and m["a"]["b"] == m["b"]["a"] == -1.0


def test_cost_sums_decode_and_inference_per_footage_second() -> None:
    records = [
        {"seconds": 2.0, "decode_seconds": 1.0, "peak_mib": 900, "threads": 2},
        {"seconds": 4.0, "decode_seconds": 1.0, "peak_mib": 1200, "threads": 2},
    ]
    c = quality.cost(records, [4.0, 4.0])
    assert c.cpu_hours_per_footage_hour == 1.0
    assert c.mean_infer_seconds == 3.0
    assert c.peak_memory_mib == 1200 and c.threads == 2


def test_cost_without_decode_or_memory_fields() -> None:
    c = quality.cost([{"seconds": 3.0}], [6.0])
    assert c.cpu_hours_per_footage_hour == 0.5
    assert c.peak_memory_mib is None and c.threads is None
    with pytest.raises(ValueError, match="duration"):
        quality.cost([{"seconds": 3.0}], [])


def test_peak_memory_is_positive() -> None:
    assert quality.peak_memory_mib() > 1.0
