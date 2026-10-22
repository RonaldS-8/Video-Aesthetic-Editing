"""Build a synthetic test video whose cut points are known exactly."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from video_demo.media import OUT_FPS, read_clip, write_video


@dataclass(frozen=True)
class ClipSpec:
    source: str
    start: float
    duration: float
    # Cross-dissolve length (s) blending the tail of the previous clip in.
    dissolve_in: float = 0.0


def build(raw_dir: Path, clips: list[ClipSpec], out_path: Path) -> dict:
    """Write the synthetic video and return its ground truth."""
    timeline: list[np.ndarray] = []
    cuts: list[dict[str, float | str]] = []
    for i, clip in enumerate(clips):
        frames = read_clip(raw_dir / clip.source, clip.start, clip.duration)
        n_mix = round(clip.dissolve_in * OUT_FPS)
        if i > 0:
            if n_mix:
                tail = timeline[-n_mix:]
                for k in range(n_mix):
                    alpha = (k + 1) / (n_mix + 1)
                    tail[k] = (
                        tail[k].astype(np.float32) * (1 - alpha)
                        + frames[k].astype(np.float32) * alpha
                    ).astype(np.uint8)
                timeline[-n_mix:] = tail
                frames = frames[n_mix:]
                # Ground truth for a dissolve is its midpoint.
                cut_t = (len(timeline) - n_mix / 2) / OUT_FPS
                cuts.append({"time": round(cut_t, 3), "type": "dissolve"})
            else:
                # A cut between two parts of the same take is a jump cut.
                kind = "jump" if clip.source == clips[i - 1].source else "hard"
                cuts.append({"time": round(len(timeline) / OUT_FPS, 3), "type": kind})
        timeline.extend(frames)

    write_video(timeline, out_path)
    truth = {
        "video": out_path.name,
        "duration": round(len(timeline) / OUT_FPS, 3),
        "clips": [asdict(c) for c in clips],
        "cuts": cuts,
    }
    out_path.with_suffix(".truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return truth
