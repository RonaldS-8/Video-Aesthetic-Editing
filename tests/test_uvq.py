from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from video_demo import uvq


def _frame(value: int) -> np.ndarray:
    return np.full((uvq.SIZE[1], uvq.SIZE[0], 3), value, np.uint8)


def _mean_model(x: torch.Tensor) -> torch.Tensor:
    """Stand-in for UVQ 1.5: the mean of its normalised input."""
    assert tuple(x.shape) == (1, 1, 3, uvq.SIZE[1], uvq.SIZE[0])
    return x.mean().reshape(1, 1)


def test_to_input_maps_pixels_to_minus_one_one() -> None:
    frame = _frame(0)
    frame[0, 0] = (255, 0, 0)
    x = uvq.to_input(frame)
    assert tuple(x.shape) == (1, 1, 3, 1080, 1920)
    assert x[0, 0, 0, 0, 0] == pytest.approx(1.0)
    assert x[0, 0, 1, 0, 0] == pytest.approx(-1.0)
    assert x.min() == pytest.approx(-1.0)


def test_to_input_rejects_frames_not_scaled_to_1080p() -> None:
    with pytest.raises(ValueError, match="1920x1080"):
        uvq.to_input(np.zeros((480, 854, 3), np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        uvq.to_input(np.zeros((1080, 1920, 3), np.float32))


def test_score_averages_frames_one_by_one() -> None:
    score = uvq.UvqAdapter(_mean_model, "sha").score([_frame(0), _frame(255)])
    assert score.per_frame == pytest.approx([-1.0, 1.0])
    assert score.mean == pytest.approx(0.0)


def test_score_rejects_empty_segment_and_non_finite_output() -> None:
    with pytest.raises(ValueError, match="No frames"):
        uvq.UvqAdapter(_mean_model, "sha").score([])

    def nan_model(_: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[math.nan]])

    with pytest.raises(ValueError, match="non-finite"):
        uvq.UvqAdapter(nan_model, "sha").score([_frame(9)])


def test_load_names_missing_checkpoints(tmp_path: Path) -> None:
    ckpt = uvq.checkpoint_dir(tmp_path)
    ckpt.mkdir(parents=True)
    (ckpt / "content_net.pth").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="distortion_net.pth"):
        uvq.UvqAdapter.load(tmp_path)
