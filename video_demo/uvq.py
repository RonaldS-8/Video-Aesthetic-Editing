"""Segment quality scores from Google UVQ 1.5 (Apache-2.0).

UVQ 1.5 (https://github.com/google/uvq, PyTorch port in ``uvq1p5_pytorch``)
scores one frame per second, each scaled to 1920x1080, and averages the frame
scores. The official reader decodes with an ffmpeg pipe
(``fps=1:start_time=0:round=up``, bicubic scaling); here PyAV supplies the
same frames (``media.segment_frames_every``) and each frame is run alone, as
the official CPU path does (batch size 1).
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Frame size and sampling step the official UVQ 1.5 reader uses.
SIZE = (1920, 1080)
STEP_SECONDS = 1.0
CHECKPOINTS = ("content_net.pth", "distortion_net.pth", "aggregation_net.pth")

# Takes (1, 1, 3, H, W) in [-1, 1] and returns the frame's quality score.
FrameModel = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class UvqScore:
    mean: float
    per_frame: list[float]


def to_input(frame: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB -> (1, 1, 3, H, W) float in [-1, 1], as the reader yields."""
    if frame.shape != (SIZE[1], SIZE[0], 3) or frame.dtype != np.uint8:
        raise ValueError(
            f"UVQ 1.5 needs {SIZE[0]}x{SIZE[1]} uint8 RGB frames, "
            f"got {frame.shape} {frame.dtype}"
        )
    x = torch.from_numpy(frame).float() * (2.0 / 255.0) - 1.0
    return x.permute(2, 0, 1).reshape(1, 1, 3, SIZE[1], SIZE[0])


class UvqAdapter:
    """Scores one segment (RGB frames one second apart) with UVQ 1.5."""

    def __init__(self, model: FrameModel, weights_sha256: str) -> None:
        self.model = model
        self.weights_sha256 = weights_sha256

    @classmethod
    def load(cls, uvq_root: Path, weights_dir: Path) -> UvqAdapter:
        missing = [name for name in CHECKPOINTS if not (weights_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"UVQ 1.5 weights not found in {weights_dir}: {missing}"
            )
        # The official package imports its siblings by path, not as a package.
        sys.path.insert(0, str(uvq_root.resolve()))
        from uvq1p5_pytorch.utils import uvq1p5

        # The official constructor reads weights next to its own code; build
        # without them and load each network from weights_dir instead.
        model = uvq1p5.UVQ1p5(pretrained=False)
        nets = (model.content_net, model.distortion_net, model.aggregation_net)
        for net, name in zip(nets, CHECKPOINTS, strict=True):
            net.load_state_dict(str(weights_dir / name))
        model.uvq1p5_core.to(memory_format=torch.channels_last)
        digest = hashlib.sha256()
        for name in CHECKPOINTS:
            digest.update((weights_dir / name).read_bytes())
        return cls(model.uvq1p5_core, digest.hexdigest())

    def score(self, frames: list[np.ndarray]) -> UvqScore:
        if not frames:
            raise ValueError("No frames to score")
        with torch.inference_mode():
            per_frame = [float(self.model(to_input(f)).item()) for f in frames]
        if not all(math.isfinite(s) for s in per_frame):
            raise ValueError(f"UVQ 1.5 returned non-finite scores: {per_frame}")
        return UvqScore(sum(per_frame) / len(per_frame), per_frame)
