"""Run per-segment stages (tagging, quality scoring) side by side.

Both stages append one JSON line per finished segment, so an interrupted run
resumes where it stopped. A failing stage does not stop the other one; its
error is collected and reported at the end.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

# fn(advance) does the work and calls advance(note) after each segment.
StageFn = Callable[[Callable[[str], None]], None]


@dataclass(frozen=True)
class Stage:
    name: str
    total: int
    done: int
    fn: StageFn


@dataclass(frozen=True)
class StageResult:
    name: str
    seconds: float
    processed: int
    error: BaseException | None


class StageError(RuntimeError):
    """Names the stage, video and segment where processing failed."""

    def __init__(self, stage: str, video: str, index: int, cause: BaseException):
        super().__init__(f"{stage} 失败：{video} 片段 #{index}：{cause}")
        self.stage = stage
        self.video = video
        self.index = index


def done_indices(jsonl_path: Path) -> set[int]:
    if not jsonl_path.exists():
        return set()
    return {
        json.loads(line)["index"]
        for line in jsonl_path.read_text("utf-8").splitlines()
        if line.strip()
    }


def append_record(jsonl_path: Path, record: dict) -> None:
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_parallel(stages: list[Stage], show_progress: bool = True) -> list[StageResult]:
    """Run every stage in its own thread with one progress bar each."""
    bars = [
        tqdm(
            total=s.total,
            initial=s.done,
            desc=s.name,
            position=i,
            unit="段",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for i, s in enumerate(stages)
    ]

    def run(stage: Stage, bar: tqdm) -> StageResult:
        processed = 0

        def advance(note: str) -> None:
            nonlocal processed
            processed += 1
            bar.set_postfix_str(note[:30], refresh=False)
            bar.update(1)

        t0 = time.perf_counter()
        try:
            stage.fn(advance)
            error = None
        except Exception as exc:  # reported after the other stages finish
            error = exc
        return StageResult(stage.name, time.perf_counter() - t0, processed, error)

    try:
        with ThreadPoolExecutor(max_workers=len(stages)) as pool:
            futures = [
                pool.submit(run, s, b) for s, b in zip(stages, bars, strict=True)
            ]
            return [f.result() for f in futures]
    finally:
        for bar in bars:
            bar.close()
