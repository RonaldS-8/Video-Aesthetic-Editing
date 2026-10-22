from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from video_demo import aesthetic

# The val-l1080p sample types from DOVER's dover.yml.
SAMPLE_TYPES = {
    "technical": {
        "fragments_h": 7,
        "fragments_w": 7,
        "fsize_h": 32,
        "fsize_w": 32,
        "aligned": 32,
        "clip_len": 32,
        "frame_interval": 2,
        "num_clips": 3,
    },
    "aesthetic": {
        "size_h": 224,
        "size_w": 224,
        "clip_len": 32,
        "frame_interval": 2,
        "t_frag": 32,
        "num_clips": 1,
    },
}


def _mean_model(views: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Stand-in for DOVER: depends on exactly which pixels were sampled."""
    return [views["technical"].mean(), views["aesthetic"].mean()]


def _frames(n: int = 40, seed: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (480, 640, 3), np.uint8) for _ in range(n)]


def _adapter(model: aesthetic.ScoringModel = _mean_model) -> aesthetic.DoverAdapter:
    return aesthetic.DoverAdapter(SAMPLE_TYPES, model, "sha")


def test_fuse_matches_official_midpoint_and_is_monotonic() -> None:
    assert aesthetic.fuse(0.1107, -0.08285) == pytest.approx(0.5)
    assert aesthetic.fuse(0.2, 0.0) > aesthetic.fuse(0.1, 0.0)
    assert aesthetic.fuse(0.1, 0.05) > aesthetic.fuse(0.1, 0.0)
    assert 0.0 < aesthetic.fuse(-0.3, -0.2) < aesthetic.fuse(0.3, 0.1) < 1.0
    # Far outside DOVER's output range: saturates instead of overflowing.
    assert aesthetic.fuse(-50.0, -50.0) == 0.0 and aesthetic.fuse(50.0, 50.0) == 1.0


def test_views_have_dover_shapes_even_for_short_segments() -> None:
    # 40 frames is fewer than the 96 the technical view samples; DOVER wraps.
    views = _adapter().views(_frames(40))
    assert tuple(views["technical"].shape) == (3, 3, 32, 224, 224)
    assert tuple(views["aesthetic"].shape) == (1, 3, 32, 224, 224)


def test_score_is_reproducible() -> None:
    frames = _frames(70)
    first = _adapter().score(frames)
    second = _adapter().score(frames)
    assert first == second
    assert first.fused == pytest.approx(
        aesthetic.fuse(first.technical, first.aesthetic)
    )


def test_score_rejects_empty_segment() -> None:
    with pytest.raises(ValueError, match="No frames"):
        _adapter().score([])


def test_score_rejects_non_finite_model_output() -> None:
    def nan_model(views: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        return [torch.tensor(math.nan), torch.tensor(0.0)]

    with pytest.raises(ValueError, match="non-finite"):
        _adapter(nan_model).score(_frames(8))


def test_load_reports_missing_config_and_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config"):
        aesthetic.DoverAdapter.load(tmp_path)
    (tmp_path / "dover.yml").write_text(
        "test_load_path: ./pretrained_weights/DOVER.pth\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="weights"):
        aesthetic.DoverAdapter.load(tmp_path)
