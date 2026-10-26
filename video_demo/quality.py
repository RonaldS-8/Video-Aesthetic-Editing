"""Compare segment quality scorers (DOVER, DOVER-Mobile, UVQ 1.5).

Measures what the NAS proposal needs to know about each scorer: CPU time and
peak memory per hour of footage, rank agreement with the others, and how much
it favours daytime over night segments.
"""

from __future__ import annotations

import ctypes
import statistics
import sys
from dataclasses import dataclass

from video_demo.evaluate import spearman

# A scorer's record counts as night when the VLM tagged it so.
NIGHT = "夜晚"
TOP_N = 30


def peak_memory_mib() -> float:
    """Peak resident memory of this process so far, in MiB."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / 2**20
    import resource

    # Linux reports KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def percentiles(values: list[float]) -> list[float]:
    """Share of all values at or below each value (1.0 for the maximum)."""
    ordered = sorted(values)
    n = len(ordered)
    return [sum(v <= x for v in ordered) / n for x in values]


@dataclass(frozen=True)
class NightBias:
    night_median: float
    day_median: float
    night_in_top: int


def night_bias(scores: list[float], night: list[bool], top_n: int = TOP_N) -> NightBias:
    """Median percentile of night vs day segments, and night segments in the top."""
    if len(scores) != len(night):
        raise ValueError("scores and night flags must have the same length")
    if all(night) or not any(night):
        raise ValueError("need both night and day segments")
    pct = percentiles(scores)
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    return NightBias(
        night_median=statistics.median(p for p, n in zip(pct, night, strict=True) if n),
        day_median=statistics.median(
            p for p, n in zip(pct, night, strict=True) if not n
        ),
        night_in_top=sum(night[i] for i in top),
    )


def worst_rank(scores: list[float], index: int) -> int:
    """1 for the lowest score; ties share the best (lowest) rank."""
    return 1 + sum(s < scores[index] for s in scores)


def rank_matrix(columns: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        a: {b: round(spearman(columns[a], columns[b]), 2) for b in columns}
        for a in columns
    }


@dataclass(frozen=True)
class Cost:
    segments: int
    footage_seconds: float
    decode_seconds: float
    infer_seconds: float
    mean_infer_seconds: float
    cpu_hours_per_footage_hour: float
    peak_memory_mib: float | None
    threads: int | None


def cost(records: list[dict], durations: list[float]) -> Cost:
    """CPU time and peak memory over one scorer's per-segment records.

    ``decode_seconds`` is optional (older DOVER records only time inference);
    peak memory is the largest ``peak_mib`` any record reports.
    """
    if not records or len(records) != len(durations):
        raise ValueError("need one duration per record")
    footage = sum(durations)
    decode = sum(r.get("decode_seconds", 0.0) for r in records)
    infer = sum(r["seconds"] for r in records)
    peaks = [r["peak_mib"] for r in records if "peak_mib" in r]
    threads = {r.get("threads") for r in records}
    return Cost(
        segments=len(records),
        footage_seconds=round(footage, 1),
        decode_seconds=round(decode, 1),
        infer_seconds=round(infer, 1),
        mean_infer_seconds=round(infer / len(records), 2),
        cpu_hours_per_footage_hour=round((decode + infer) / footage, 2),
        peak_memory_mib=round(max(peaks)) if peaks else None,
        threads=threads.pop() if len(threads) == 1 else None,
    )
