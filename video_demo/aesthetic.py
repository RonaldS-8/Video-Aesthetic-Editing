"""Segment quality scores from DOVER (technical + aesthetic video quality).

DOVER (ICCV 2023, https://github.com/VQAssessment/DOVER) is licensed under the
S-Lab License for non-commercial use only; this demo uses it for internal
evaluation. View sampling and normalisation follow the official
``evaluate_one_video.py``; only frame decoding differs (PyAV, per segment).
DOVER-Mobile loads through the same adapter from ``dover-mobile.yml``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import yaml

# DOVER targets torch 1.13 / older timm; these deprecation notices are
# harmless and would otherwise interleave with the progress bars.
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad")

from dover.datasets import UnifiedFrameSampler, get_single_view  # noqa: E402
from dover.models import DOVER  # noqa: E402

# Same sampling settings the official one-video script reads from dover.yml.
CONFIG_SET = "val-l1080p"
# DOVER samples frames and fragment offsets at random; a fixed seed per
# segment makes scores reproducible across runs.
SEED = 0
# ImageNet mean / std on 0-255 RGB, as in evaluate_one_video.py.
MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
STD = torch.FloatTensor([58.395, 57.12, 57.375])


class ScoringModel(Protocol):
    def __call__(self, views: dict[str, torch.Tensor]) -> list[torch.Tensor]: ...


@dataclass(frozen=True)
class DoverScore:
    technical: float
    aesthetic: float
    fused: float


def fuse(technical: float, aesthetic: float) -> float:
    """Official score-level fusion (``fuse_results``), mapped to [0, 1]."""
    x = (technical - 0.1107) / 0.07355 * 0.6104 + (
        aesthetic + 0.08285
    ) / 0.03774 * 0.3896
    # Numerically stable sigmoid (no overflow for large |x|).
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def build_samplers(sample_types: dict[str, Any]) -> dict[str, UnifiedFrameSampler]:
    samplers: dict[str, UnifiedFrameSampler] = {}
    for stype, sopt in sample_types.items():
        if "t_frag" not in sopt:
            samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )
    return samplers


def weights_path(dover_root: Path, config_name: str = "dover.yml") -> Path:
    config = yaml.safe_load((dover_root / config_name).read_text("utf-8"))
    return (dover_root / config["test_load_path"]).resolve()


class DoverAdapter:
    """Scores one segment (a list of RGB frames) with a DOVER model."""

    def __init__(
        self,
        sample_types: dict[str, Any],
        model: ScoringModel,
        weights_sha256: str,
        device: str = "cpu",
    ) -> None:
        self.sample_types = sample_types
        self.samplers = build_samplers(sample_types)
        self.model = model
        self.weights_sha256 = weights_sha256
        self.device = device
        self.seed = SEED

    @classmethod
    def load(
        cls, dover_root: Path, device: str = "cpu", config_name: str = "dover.yml"
    ) -> DoverAdapter:
        config_path = dover_root / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"DOVER config not found: {config_path}")
        config = yaml.safe_load(config_path.read_text("utf-8"))
        weights = weights_path(dover_root, config_name)
        if not weights.exists():
            raise FileNotFoundError(f"DOVER weights not found: {weights}")
        # DOVER prints its backbone set-up ("Setting backbone: ...") on build.
        with contextlib.redirect_stdout(io.StringIO()):
            model = DOVER(**config["model"]["args"]).to(device)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.eval()
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        sample_types = config["data"][CONFIG_SET]["args"]["sample_types"]
        return cls(sample_types, model, digest, device)

    def sample_indices(self, n_frames: int) -> dict[str, np.ndarray]:
        """Frame indices each view samples from a segment of ``n_frames``.

        The samplers draw only from Python's and NumPy's generators and the
        views only from torch's, so drawing all indices first gives the same
        result as the official interleaved order.
        """
        if n_frames < 1:
            raise ValueError("No frames to score")
        random.seed(SEED)
        np.random.seed(SEED)
        return {stype: self.samplers[stype](n_frames) for stype in self.sample_types}

    def needed_frames(self, n_frames: int) -> set[int]:
        """Every frame index any view uses; only these need decoding."""
        return {int(i) for inds in self.sample_indices(n_frames).values() for i in inds}

    def views(self, frames: list[np.ndarray]) -> dict[str, torch.Tensor]:
        """Technical fragments and aesthetic resize views, normalised."""
        return self.views_from(dict(enumerate(frames)), len(frames))

    def views_from(
        self, frames: dict[int, np.ndarray], n_frames: int
    ) -> dict[str, torch.Tensor]:
        """Views from the sampled frames only, keyed by index in the segment."""
        indices = self.sample_indices(n_frames)
        needed = sorted({int(i) for inds in indices.values() for i in inds})
        missing = [i for i in needed if i not in frames]
        if missing:
            raise ValueError(f"Frames not decoded: {missing[:5]}")
        position = {i: p for p, i in enumerate(needed)}
        # T, H, W, C uint8 -> C, T, H, W, matching decord's torch bridge.
        video = torch.from_numpy(np.stack([frames[i] for i in needed])).permute(
            3, 0, 1, 2
        )
        torch.manual_seed(SEED)
        views: dict[str, torch.Tensor] = {}
        for stype, sopt in self.sample_types.items():
            inds = torch.tensor([position[int(i)] for i in indices[stype]]).long()
            v = get_single_view(video[:, inds], stype, **sopt)
            num_clips = sopt.get("num_clips", 1)
            views[stype] = (
                ((v.permute(1, 2, 3, 0) - MEAN) / STD)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(self.device)
            )
        return views

    def score(self, frames: list[np.ndarray]) -> DoverScore:
        return self.score_from(dict(enumerate(frames)), len(frames))

    def score_from(self, frames: dict[int, np.ndarray], n_frames: int) -> DoverScore:
        """Score a segment of ``n_frames`` from just its sampled frames."""
        views = self.views_from(frames, n_frames)
        with torch.no_grad():
            technical, aesthetic = (r.mean().item() for r in self.model(views))
        if not (math.isfinite(technical) and math.isfinite(aesthetic)):
            raise ValueError(
                f"DOVER returned non-finite scores: {technical}, {aesthetic}"
            )
        return DoverScore(technical, aesthetic, fuse(technical, aesthetic))
