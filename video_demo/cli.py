"""Command line entry: python -m video_demo <stage>."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from video_demo import (
    evaluate,
    layout,
    outline,
    quality,
    report,
    runner,
    signals,
    suggest,
    synth,
    tag,
)
from video_demo.media import (
    count_frames,
    probe,
    read_clip,
    segment_frames_every,
    segment_frames_picked,
    write_video,
)
from video_demo.segment import (
    MAX_SEGMENT_SEC,
    Segment,
    detect_cuts,
    merge_short,
    spans_from_cuts,
    split_long,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
WORK_DIR = ROOT / "data" / "work"
OUT_DIR = ROOT / "out"
# DOVER-scored results; the VLM-only baseline stays in OUT_DIR (from `main`).
RESULT_DIR = OUT_DIR / "dover"
BASELINE_REPORT = OUT_DIR / "report.html"
SYNTH_PATH = ROOT / "data" / "synth" / "synth_cuts.mp4"
DOVER_ROOT = ROOT / "third_party" / "DOVER"
UVQ_ROOT = ROOT / "third_party" / "uvq"

# Where a per-segment stage keeps its records for one video.
PathFn = Callable[[Path, str], Path]
# Scorers compared against DOVER by `quality` / `compare-quality`; their
# scores are for that comparison only and do not feed `suggest`.
COMPARE_MODELS = ("dover-mobile", "uvq")
ONLY_HELP = "video id or file prefix (path under data/raw), e.g. 330ff4"
COLLECTION_HELP = "first folder under data/raw, e.g. vlog, tvsum, summe, long"
COMPARE_FILE = "quality_compare.json"
UVQ_COMMIT = "c204ecc04f093e90c941cb126c7ac3d5beca0414"
SETUP_HINTS = {
    "dover-mobile": (
        "把 https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/"
        "DOVER-Mobile.pth 下载到 third_party/DOVER/pretrained_weights/"
    ),
    "uvq": (
        f"从 https://github.com/google/uvq/tree/{UVQ_COMMIT} 取 uvq1p5_pytorch/、"
        "utils/ 与 LICENSE，放到 third_party/uvq/（见 README）"
    ),
}
LLAMA_SERVER_CMD = (
    "llama-server -m Qwen3VL-2B-Instruct-Q4_K_M.gguf "
    "--mmproj mmproj-Qwen3VL-2B-Instruct-F16.gguf "
    "--host 127.0.0.1 --port 8091 -ngl 99 -c 8192 -np 1"
)
DOVER_SETUP_CMD = (
    "git clone https://github.com/VQAssessment/DOVER.git third_party/DOVER，"
    "再把 https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/"
    "DOVER.pth 下载到 third_party/DOVER/pretrained_weights/"
)
NEXT_STEPS = "下一步：python -m video_demo suggest → render → report"

# Seconds to analyse per video id; only the first 5 minutes of the long vlog
# are analysed for this demo.
END_TIME = {"2c36abc81b285d57": 300.0}  # 06_shanghai_24h_vlog.webm

SYNTH_CLIPS = [
    synth.ClipSpec("vlog/04_mbs_skypark.webm", 0.5, 5.0),
    synth.ClipSpec("vlog/04_mbs_skypark.webm", 10.0, 5.0),
    synth.ClipSpec("vlog/05_mbs_gondola.webm", 1.0, 6.0),
    synth.ClipSpec("vlog/03_aurora_timelapse.webm", 3.0, 5.0),
    synth.ClipSpec("vlog/05_mbs_gondola.webm", 16.0, 5.0),
    synth.ClipSpec("vlog/03_aurora_timelapse.webm", 50.0, 5.0),
    synth.ClipSpec("vlog/03_aurora_timelapse.webm", 80.0, 6.0, dissolve_in=1.0),
    synth.ClipSpec("vlog/05_mbs_gondola.webm", 24.0, 5.5, dissolve_in=1.0),
]
# Held-out set from other videos, built after the dissolve thresholds were
# fixed, with dissolves of 0.5-1.5 s to check the detector does not overfit.
HOLDOUT_CLIPS = [
    synth.ClipSpec("vlog/06_shanghai_24h_vlog.webm", 55.5, 5.5),
    synth.ClipSpec("vlog/02_bled_bohinj_vlog.webm", 61.5, 5.5),
    synth.ClipSpec("vlog/06_shanghai_24h_vlog.webm", 103.5, 5.5, dissolve_in=1.0),
    synth.ClipSpec("vlog/01_lion_city_singapore.webm", 132.5, 5.5, dissolve_in=0.5),
    synth.ClipSpec("vlog/06_shanghai_24h_vlog.webm", 149.5, 5.5, dissolve_in=1.5),
    synth.ClipSpec("vlog/02_bled_bohinj_vlog.webm", 123.0, 5.0),
    synth.ClipSpec("vlog/06_shanghai_24h_vlog.webm", 196.5, 5.0, dissolve_in=1.0),
]
SYNTH_SETS = {
    "main": (SYNTH_CLIPS, SYNTH_PATH),
    "holdout": (HOLDOUT_CLIPS, SYNTH_PATH.with_name("synth_holdout.mp4")),
}


def _videos() -> list[layout.Video]:
    return layout.load_videos(RAW_DIR)


def _raw(video: layout.Video) -> Path:
    return RAW_DIR / video.file


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


def _update_manifest() -> list[layout.Video]:
    """Register new videos in data/raw; print what changed."""
    manifest, changes = layout.update_manifest(RAW_DIR)
    path = layout.write_manifest(RAW_DIR, manifest)
    for v in changes.added:
        print(f"新视频：{v.id}  {v.file}")
    for d in changes.duplicates:
        print(f"重复，跳过：{d.file} 与 {d.same_as} 内容相同（可删除以节约空间）")
    for old, new in changes.moved:
        print(f"位置变了：{old.file} → {new.file}（同一 id {new.id}，结果沿用）")
    for old, new in changes.changed:
        print(f"内容变了：{new.file} {old.id} → {new.id}（旧 id 的结果保留未动）")
    for v in changes.removed:
        print(f"已移除：{v.id}  {v.file}（它的结果保留未动）")
    print(
        f"清单 {path}：{len(manifest.videos)} 条视频，"
        f"{len(manifest.duplicates)} 个重复文件，本次计算哈希 {changes.hashed} 个"
    )
    return manifest.videos


def _pending(
    videos: list[layout.Video], out: PathFn, force: bool
) -> list[layout.Video]:
    """Videos still without ``out``; all of them with ``force``."""
    return [v for v in videos if force or not out(WORK_DIR, v.id).exists()]


def cmd_ingest(args: argparse.Namespace) -> None:
    """Register new videos, then segment them and compute their signals.

    Raw videos stay where they are under their own names; only the manifest
    and the derived files use the content hash. Videos that already have
    results are left alone.
    """
    videos = _update_manifest()
    todo = _pending(videos, layout.segments_path, force=False)
    if not todo:
        print("没有需要切分的新视频。")
        return
    _segment(todo)
    _signals(_pending(todo, layout.signals_path, force=False))
    print(
        f"已切分 {len(todo)} 条新视频。下一步：python -m video_demo analyze"
        "（只处理未完成的片段）→ suggest → render → report"
    )


def cmd_segment(args: argparse.Namespace) -> None:
    videos = _pending(_selected(args), layout.segments_path, args.force)
    if args.force:
        print("重新切分：该视频已有的信号、关键帧、标签、美学分需要手动删除后重跑")
    _segment(videos)


def _segment(videos: list[layout.Video]) -> None:
    for video in videos:
        layout.verify(RAW_DIR, video)
        path = _raw(video)
        info = probe(path)
        end_time = END_TIME.get(video.id)
        analysed = info.duration if end_time is None else end_time
        t0 = time.perf_counter()
        cuts = detect_cuts(path, end_time)
        raw_shots = spans_from_cuts(cuts, analysed)
        shots = merge_short(raw_shots)
        segments = split_long(shots)
        elapsed = time.perf_counter() - t0
        payload = {
            "video_id": video.id,
            "sha256": video.sha256,
            "video": video.file,
            "duration": info.duration,
            "analysed_until": analysed,
            "fps": info.fps,
            "has_audio": info.has_audio,
            "detect_seconds": round(elapsed, 2),
            "max_segment_sec": MAX_SEGMENT_SEC,
            "cuts": [{"time": c.time, "kind": c.kind} for c in cuts],
            "raw_shot_count": len(raw_shots),
            "shots": shots,
            "segments": [s.to_dict() for s in segments],
        }
        _write_json(layout.segments_path(WORK_DIR, video.id), payload)
        n_dissolve = sum(c.kind == "dissolve" for c in cuts)
        print(
            f"{video.id} {video.file}: {len(cuts)} cuts ({n_dissolve} dissolve) -> "
            f"{len(shots)} shots, {len(segments)} segments "
            f"({elapsed:.1f}s for {analysed:.0f}s video)"
        )


def _load_segments(vid: str) -> tuple[dict, list[Segment]]:
    path = layout.segments_path(WORK_DIR, vid)
    data = json.loads(path.read_text("utf-8"))
    if data["video_id"] != vid:
        raise ValueError(f"{path} 属于 {data['video_id']}，不是 {vid}")
    return data, [Segment(**s) for s in data["segments"]]


def _load_signals(vid: str) -> list[dict]:
    return json.loads(layout.signals_path(WORK_DIR, vid).read_text("utf-8"))


def cmd_signals(args: argparse.Namespace) -> None:
    _signals(_pending(_selected(args), layout.signals_path, args.force))


def _signals(videos: list[layout.Video]) -> None:
    for video in videos:
        _, segments = _load_segments(video.id)
        t0 = time.perf_counter()
        results = signals.compute(
            _raw(video),
            segments,
            layout.keyframes_dir(WORK_DIR, video.id),
            END_TIME.get(video.id),
        )
        payload = [
            {"index": s.index, **r.to_dict()}
            for s, r in zip(segments, results, strict=True)
        ]
        _write_json(layout.signals_path(WORK_DIR, video.id), payload)
        flagged = sum(bool(r.flags) for r in results)
        print(
            f"{video.id} {video.file}: {len(results)} segments, {flagged} flagged "
            f"({time.perf_counter() - t0:.1f}s)"
        )


def _selected(args: argparse.Namespace) -> list[layout.Video]:
    """Videos in ``--collection`` whose id or file starts with ``--only``."""
    collection = getattr(args, "collection", "")
    only = getattr(args, "only", "")
    videos = [
        v
        for v in _videos()
        if (not collection or v.collection == collection)
        and (not only or v.id.startswith(only) or v.file.startswith(only))
    ]
    if not videos:
        raise SystemExit(
            f"没有符合条件的视频（collection={collection!r}，only={only!r}）"
        )
    return videos


def _out(args: argparse.Namespace) -> Path:
    """``--out``, else out/<collection> (vlog keeps out/dover); highlights
    and ranked go to a folder of that name inside it, since every mode
    writes one timeline per video under the same names."""
    if args.out is not None:
        return args.out
    base = RESULT_DIR if args.collection == "vlog" else OUT_DIR / args.collection
    mode = getattr(args, "mode", "reel")
    return base if mode == "reel" else base / mode


def _summary_name(mode: str) -> str:
    return REEL_NAME if mode == "reel" else HIGHLIGHTS_NAME


def _todo(vid: str, out: PathFn, limit: int) -> list[Segment]:
    """Segments of ``vid`` not yet recorded in ``out`` (resume support)."""
    _, segments = _load_segments(vid)
    done = runner.done_indices(out(WORK_DIR, vid))
    todo = [s for s in segments if s.index not in done]
    return todo[:limit] if limit else todo


def _counts(videos: list[layout.Video], out: PathFn, limit: int) -> tuple[int, int]:
    """(segments already done, segments this run will process)."""
    done = todo = 0
    for video in videos:
        _, segments = _load_segments(video.id)
        remaining = _todo(video.id, out, 0)
        done += len(segments) - len(remaining)
        todo += min(limit, len(remaining)) if limit else len(remaining)
    return done, todo


def _tag_stage(videos: list[layout.Video], limit: int) -> runner.StageFn:
    def run(advance: Callable[[str], None]) -> None:
        for video in videos:
            sig = _load_signals(video.id)
            out_path = layout.tags_path(WORK_DIR, video.id)
            keyframes = layout.keyframes_dir(WORK_DIR, video.id)
            for seg in _todo(video.id, layout.tags_path, limit):
                frames = [keyframes / k for k in sig[seg.index]["keyframes"]]
                try:
                    tags, raw, elapsed, errors = tag.tag_segment(frames, seg.duration)
                except Exception as exc:
                    raise runner.StageError(
                        "打标签", video.file, seg.index, exc
                    ) from exc
                runner.append_record(
                    out_path,
                    {
                        "index": seg.index,
                        "seconds": round(elapsed, 2),
                        "tags": tags,
                        "raw": raw,
                        "failed_attempts": errors,
                    },
                )
                retry = f" (retries {len(errors)})" if errors else ""
                advance(
                    f"{video.file} seg{seg.index:03d} {elapsed:5.1f}s{retry} "
                    f"h={tags['highlight']} {tags['description']}"
                )

    return run


# DOVER's technical view samples fragments at native resolution; larger
# frames are scaled to this long side first so a 4K segment fits in memory.
DOVER_MAX_SIDE = 1920


def _dover_frames(
    adapter: Any, video: layout.Video, spans: list[tuple[float, float]]
) -> Iterator[tuple[int, dict[int, np.ndarray], int]]:
    """(span index, the frames DOVER samples, frames in the span).

    Frames are counted from packet timestamps first so only the sampled ones
    are decoded into memory; a decode that sees a different count is an
    error rather than a silently different sample.
    """
    end = END_TIME.get(video.id)
    counts = count_frames(_raw(video), spans, end)
    picks = [adapter.needed_frames(n) if n else set() for n in counts]
    for i, frames, seen in segment_frames_picked(
        _raw(video), spans, picks, DOVER_MAX_SIDE, end
    ):
        if seen != counts[i]:
            raise ValueError(
                f"{video.file} {spans[i]}: "
                f"按时间戳数出 {counts[i]} 帧，解码得到 {seen} 帧"
            )
        yield i, frames, seen


def _quality_stage(
    videos: list[layout.Video], limit: int, adapter: Any
) -> runner.StageFn:
    def run(advance: Callable[[str], None]) -> None:
        for video in videos:
            todo = _todo(video.id, layout.aesthetic_path, limit)
            out_path = layout.aesthetic_path(WORK_DIR, video.id)
            spans = [(s.start, s.end) for s in todo]
            for i, frames, n_frames in _dover_frames(adapter, video, spans):
                seg = todo[i]
                t0 = time.perf_counter()
                try:
                    score = adapter.score_from(frames, n_frames)
                except Exception as exc:
                    raise runner.StageError(
                        "美学打分", video.file, seg.index, exc
                    ) from exc
                elapsed = time.perf_counter() - t0
                runner.append_record(
                    out_path,
                    {
                        "index": seg.index,
                        "technical": round(score.technical, 5),
                        "aesthetic": round(score.aesthetic, 5),
                        "fused": round(score.fused, 4),
                        "n_frames": n_frames,
                        "decoded_frames": len(frames),
                        "seconds": round(elapsed, 2),
                        "model": "DOVER",
                        "weights_sha256": adapter.weights_sha256,
                        "seed": adapter.seed,
                    },
                )
                advance(
                    f"{video.file} seg{seg.index:03d} {elapsed:4.1f}s "
                    f"q={score.fused:.2f}"
                )

    return run


def _load_quality_adapter() -> Any:
    # Imported lazily: torch + DOVER take seconds to import and only these
    # stages need them.
    from video_demo.aesthetic import DoverAdapter

    return DoverAdapter.load(DOVER_ROOT)


def _print_line(note: str) -> None:
    print(note, flush=True)


def cmd_tag(args: argparse.Namespace) -> None:
    _tag_stage(_selected(args), args.limit)(_print_line)


def cmd_aesthetic(args: argparse.Namespace) -> None:
    videos = _selected(args)
    _quality_stage(videos, args.limit, _load_quality_adapter())(_print_line)


def _load_compare_adapter(model: str) -> Any:
    try:
        if model == "dover-mobile":
            from video_demo.aesthetic import DoverAdapter

            return DoverAdapter.load(DOVER_ROOT, config_name="dover-mobile.yml")
        from video_demo.uvq import UvqAdapter

        return UvqAdapter.load(UVQ_ROOT)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\n    {SETUP_HINTS[model]}") from exc


def _compare_frames(
    model: str, adapter: Any, video: layout.Video, spans: list[tuple[float, float]]
) -> Iterator[tuple[int, dict[int, np.ndarray], int]]:
    """(span index, frames by index in the span, frames in the span)."""
    if model == "dover-mobile":
        yield from _dover_frames(adapter, video, spans)
        return
    from video_demo import uvq

    for i, frames in segment_frames_every(
        _raw(video), spans, uvq.STEP_SECONDS, uvq.SIZE, END_TIME.get(video.id)
    ):
        yield i, dict(enumerate(frames)), len(frames)


def _compare_out(model: str) -> PathFn:
    return lambda work, vid: layout.compare_path(work, model, vid)


def _compare_scores(
    model: str, adapter: Any, frames: dict[int, np.ndarray], n_frames: int
) -> dict:
    if model == "dover-mobile":
        score = adapter.score_from(frames, n_frames)
        return {
            "technical": round(score.technical, 5),
            "aesthetic": round(score.aesthetic, 5),
            "fused": round(score.fused, 4),
            "seed": adapter.seed,
        }
    score = adapter.score([frames[i] for i in sorted(frames)])
    return {
        "uvq": round(score.mean, 4),
        "per_frame": [round(s, 4) for s in score.per_frame],
    }


def cmd_quality(args: argparse.Namespace) -> None:
    """Score segments with DOVER-Mobile or UVQ 1.5, timing decode and inference
    separately and recording this process's peak memory."""
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    threads = torch.get_num_threads()
    adapter = _load_compare_adapter(args.model)
    out = _compare_out(args.model)
    for video in _selected(args):
        todo = _todo(video.id, out, args.limit)
        spans = [(s.start, s.end) for s in todo]
        frames_iter = _compare_frames(args.model, adapter, video, spans)
        while True:
            t0 = time.perf_counter()
            try:
                i, frames, n_frames = next(frames_iter)
            except StopIteration:
                break
            decode = time.perf_counter() - t0
            seg = todo[i]
            t0 = time.perf_counter()
            try:
                scores = _compare_scores(args.model, adapter, frames, n_frames)
            except Exception as exc:
                raise runner.StageError(args.model, video.file, seg.index, exc) from exc
            elapsed = time.perf_counter() - t0
            runner.append_record(
                out(WORK_DIR, video.id),
                {
                    "index": seg.index,
                    **scores,
                    "n_frames": n_frames,
                    "decode_seconds": round(decode, 2),
                    "seconds": round(elapsed, 2),
                    "peak_mib": round(quality.peak_memory_mib()),
                    "threads": threads,
                    "model": args.model,
                    "weights_sha256": adapter.weights_sha256,
                },
            )
            print(
                f"{video.file} seg{seg.index:03d} {n_frames}f "
                f"decode {decode:4.1f}s infer {elapsed:4.1f}s",
                flush=True,
            )


def _read_records(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise RuntimeError(f"缺少 {path}：先运行对应的打分阶段")
    return {
        r["index"]: r for r in map(json.loads, path.read_text("utf-8").splitlines())
    }


def cmd_compare_quality(args: argparse.Namespace) -> None:
    """Cost, rank agreement and night bias of DOVER, DOVER-Mobile and UVQ 1.5."""
    sources: dict[str, PathFn] = {
        "DOVER": layout.aesthetic_path,
        **{m: _compare_out(m) for m in COMPARE_MODELS},
    }
    rows: list[dict] = []
    for video in _selected(args):
        _, segments = _load_segments(video.id)
        tag_records = _read_records(layout.tags_path(WORK_DIR, video.id))
        tags = {i: r["tags"] for i, r in tag_records.items()}
        sig = _load_signals(video.id)
        records = {
            name: _read_records(out(WORK_DIR, video.id))
            for name, out in sources.items()
        }
        for name, recs in records.items():
            missing = [s.index for s in segments if s.index not in recs]
            if missing:
                raise RuntimeError(f"{video.file}: {name} 缺少片段 {missing[:5]}")
        for s in segments:
            rows.append(
                {
                    "segment": f"{video.id}#{s.index}",
                    "file": video.file,
                    "duration": s.duration,
                    "night": tags[s.index]["time_of_day"] == quality.NIGHT,
                    "description": tags[s.index]["description"],
                    "flags": sig[s.index]["flags"],
                    "highlight": tags[s.index]["highlight"],
                    "brightness": sig[s.index]["brightness"],
                    **{name: recs[s.index] for name, recs in records.items()},
                }
            )
    columns = {
        "DOVER technical": [r["DOVER"]["technical"] for r in rows],
        "DOVER aesthetic": [r["DOVER"]["aesthetic"] for r in rows],
        "DOVER fused": [r["DOVER"]["fused"] for r in rows],
        "DOVER-Mobile technical": [r["dover-mobile"]["technical"] for r in rows],
        "DOVER-Mobile aesthetic": [r["dover-mobile"]["aesthetic"] for r in rows],
        "DOVER-Mobile fused": [r["dover-mobile"]["fused"] for r in rows],
        "UVQ 1.5": [r["uvq"]["uvq"] for r in rows],
    }
    night = [r["night"] for r in rows]
    durations = [r["duration"] for r in rows]
    overall = ("DOVER fused", "DOVER-Mobile fused", "UVQ 1.5")
    bottom = {
        col: sorted(range(len(rows)), key=lambda i, c=col: columns[c][i])[:5]
        for col in overall
    }
    spread = [
        max(r["uvq"]["per_frame"]) - min(r["uvq"]["per_frame"])
        for r in rows
        if len(r["uvq"]["per_frame"]) >= 3
    ]
    result = {
        "segments": len(rows),
        "night_segments": sum(night),
        "cost": {
            name: asdict(quality.cost([r[name] for r in rows], durations))
            for name in sources
        },
        "spearman": quality.rank_matrix(
            {
                **columns,
                "VLM highlight": [float(r["highlight"]) for r in rows],
                "brightness": [r["brightness"] for r in rows],
            }
        ),
        "night_bias": {
            col: asdict(quality.night_bias(values, night))
            for col, values in columns.items()
        },
        "worst_ranks": [
            {
                "segment": rows[i]["segment"],
                "file": rows[i]["file"],
                "description": rows[i]["description"],
                "flags": rows[i]["flags"],
                **{col: quality.worst_rank(columns[col], i) for col in overall},
            }
            for i in sorted({i for idx in bottom.values() for i in idx})
        ],
        "uvq_frame_spread": {
            "segments_with_3_frames": len(spread),
            "median_max_minus_min": round(statistics.median(spread), 3),
            "sd_across_segments": round(statistics.pstdev(columns["UVQ 1.5"]), 3),
        },
    }
    out = _out(args) / COMPARE_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, c in result["cost"].items():
        print(
            f"{name}: {c['mean_infer_seconds']}s/段，"
            f"每小时素材 {c['cpu_hours_per_footage_hour']} h CPU，"
            f"峰值内存 {c['peak_memory_mib']} MiB，线程 {c['threads']}"
        )
    for col in overall:
        b = result["night_bias"][col]
        print(
            f"{col}: 夜晚中位 {b['night_median']:.2f} / 白天 {b['day_median']:.2f}，"
            f"前 {quality.TOP_N} 中夜景 {b['night_in_top']}，"
            f"与 DOVER 融合 ρ={result['spearman'][col]['DOVER fused']}"
        )
    print(f"Wrote {out}")


def missing_inputs(
    videos: list[layout.Video], need_tag: bool, need_quality: bool
) -> list[str]:
    """Preflight problems for ``analyze``, each with the command that fixes it."""
    problems: list[str] = []
    for video in videos:
        for out, stage in (
            (layout.segments_path, "segment"),
            (layout.signals_path, "signals"),
        ):
            path = out(WORK_DIR, video.id)
            if not path.exists():
                problems.append(
                    f"{video.file} 缺少 {path.name}："
                    f"先运行 python -m video_demo {stage}"
                )
    if problems:
        return problems
    if need_tag:
        try:
            tag.check_server()
        except ConnectionError as exc:
            problems.append(f"{exc}\n    启动命令：{LLAMA_SERVER_CMD}")
    if need_quality:
        config = DOVER_ROOT / "dover.yml"
        weights = DOVER_ROOT / "pretrained_weights" / "DOVER.pth"
        if not config.exists() or not weights.exists():
            problems.append(
                f"未找到 DOVER 代码或权重（{weights}）\n    {DOVER_SETUP_CMD}"
            )
    return problems


def cmd_analyze(args: argparse.Namespace) -> None:
    """Tag segments (llama-server on the iGPU) and score quality (DOVER on
    the CPU) at the same time; both resume from what is already on disk."""
    videos = _selected(args)
    problems = missing_inputs(videos, need_tag=False, need_quality=False)
    if problems:
        raise SystemExit("开工检查未通过：\n- " + "\n- ".join(problems))
    tag_done, tag_todo = _counts(videos, layout.tags_path, args.limit)
    q_done, q_todo = _counts(videos, layout.aesthetic_path, args.limit)
    if not tag_todo and not q_todo:
        print(f"全部片段已完成打标签和美学打分，无需重跑。{NEXT_STEPS}")
        return
    problems = missing_inputs(videos, need_tag=tag_todo > 0, need_quality=q_todo > 0)
    if problems:
        raise SystemExit("开工检查未通过：\n- " + "\n- ".join(problems))

    stages: list[runner.Stage] = []
    if tag_todo:
        stages.append(
            runner.Stage(
                "打标签", tag_done + tag_todo, tag_done, _tag_stage(videos, args.limit)
            )
        )
    if q_todo:
        print("加载 DOVER 模型……", flush=True)
        adapter = _load_quality_adapter()
        stages.append(
            runner.Stage(
                "美学打分",
                q_done + q_todo,
                q_done,
                _quality_stage(videos, args.limit, adapter),
            )
        )
    print(
        f"开始：打标签 {tag_todo} 段、美学打分 {q_todo} 段"
        f"（已完成的 {tag_done} / {q_done} 段跳过）",
        flush=True,
    )
    t0 = time.perf_counter()
    results = runner.run_parallel(stages)
    wall = time.perf_counter() - t0

    print(f"\n完成，总用时 {wall:.0f}s（并发）：")
    for r in results:
        status = "失败" if r.error else "完成"
        print(f"- {r.name}：{status}，处理 {r.processed} 段，用时 {r.seconds:.0f}s")
    failed = [r for r in results if r.error]
    if failed:
        details = "\n".join(f"- {r.error}" for r in failed)
        raise SystemExit(
            f"有任务失败（已写入的结果已保存，修复后重跑会接着做）：\n{details}"
        )
    print(NEXT_STEPS)


PER_VIDEO_TARGET_SEC = 30.0
REEL_TARGET_SEC = 45.0
REEL_NAME = "all_videos_reel"
# One timeline across the whole collection in highlights / ranked mode.
HIGHLIGHTS_NAME = "highlights"
MODES = ("reel", "highlights", "ranked")
OUTLINE_SERVER_CMD = (
    "llama-server -m Qwen3.5-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8092 "
    "-ngl 99 -c 16384 -np 1（模型在 OCR_test/models/gguf）"
)


def _candidates(video: layout.Video) -> list[suggest.Candidate]:
    _, segments = _load_segments(video.id)
    sig = _load_signals(video.id)
    tags_path = layout.tags_path(WORK_DIR, video.id)
    tags = {
        r["index"]: r["tags"]
        for r in map(json.loads, tags_path.read_text("utf-8").splitlines())
    }
    missing = [s.index for s in segments if s.index not in tags]
    if missing:
        raise RuntimeError(
            f"{video.file}: {len(missing)} segments not tagged: {missing[:5]}"
        )
    quality_path = layout.aesthetic_path(WORK_DIR, video.id)
    if not quality_path.exists():
        raise RuntimeError(f"{video.file}: no {quality_path.name}; run `analyze` first")
    quality = {
        r["index"]: r
        for r in map(json.loads, quality_path.read_text("utf-8").splitlines())
    }
    missing = [s.index for s in segments if s.index not in quality]
    if missing:
        raise RuntimeError(
            f"{video.file}: {len(missing)} segments not scored: {missing[:5]}"
        )
    median = statistics.median(sig[s.index]["sharpness"] for s in segments)
    return [
        suggest.Candidate(
            video=video.id,
            index=s.index,
            shot=s.shot,
            start=s.start,
            end=s.end,
            tags=tags[s.index],
            flags=sig[s.index]["flags"],
            sharpness=sig[s.index]["sharpness"],
            sharpness_rel=sig[s.index]["sharpness"] / median,
            quality=quality[s.index]["fused"],
            tech_raw=quality[s.index]["technical"],
            aes_raw=quality[s.index]["aesthetic"],
        )
        for s in segments
    ]


def _timeline_payload(
    name: str,
    cands: list[suggest.Candidate],
    target: float,
) -> dict:
    assessments = [suggest.assess(c) for c in cands]
    clips, notes = suggest.build_timeline(cands, assessments, target)
    return {
        "name": name,
        "target_sec": target,
        "total_sec": round(sum(c.duration for c in clips), 2),
        "notes": notes,
        "clips": [suggest.clip_to_dict(c) for c in clips],
        "assessments": [vars(a) for a in assessments],
    }


def _highlights_payload(
    name: str,
    cands: list[suggest.Candidate],
    target: float,
    offsets: dict[str, float],
    mode: str,
) -> dict:
    """Highlights: a model outline filled by rules. Ranked: score only (the
    baseline highlights mode kept for comparison)."""
    cands = suggest.calibrate_by_time(cands)
    assessments = [suggest.assess(c) for c in cands]
    extra: dict[str, Any] = {}
    if mode == "ranked":
        clips, notes = suggest.build_highlights(cands, assessments, target, offsets)
    else:
        scenes = outline.group_scenes(cands, assessments, offsets)
        try:
            plan, raw, seconds, errors = outline.plan_outline(scenes, target)
        except outline.OutlineError as exc:
            raise SystemExit(
                f"{name}：大纲生成失败（3 次都没通过校验）：{exc}"
            ) from exc
        clips, notes = outline.fill(plan, scenes, cands, assessments)
        extra = {
            "outline": plan,
            "outline_raw": raw,
            "outline_seconds": round(seconds, 1),
            "outline_failed_attempts": errors,
            "scenes": [
                {
                    "number": sc.number,
                    "start": round(sc.start, 2),
                    "end": round(sc.end, 2),
                    "segments": [[cands[i].video, cands[i].index] for i in sc.members],
                }
                for sc in scenes
            ],
        }
    return {
        "name": name,
        "mode": mode,
        "target_sec": round(target, 1),
        "total_sec": round(sum(c.duration for c in clips), 2),
        "notes": notes,
        "clips": [suggest.clip_to_dict(c) for c in clips],
        "assessments": [vars(a) for a in assessments],
        **extra,
    }


def cmd_suggest(args: argparse.Namespace) -> None:
    """Reel mode: a 30 s cut per video and a 45 s reel. Highlights mode:
    ``--ratio`` of each video, and of the whole collection, as whole segments."""
    out = _out(args)
    if args.mode == "highlights":
        try:
            tag.check_server(outline.SERVER_URL)
        except ConnectionError as exc:
            raise SystemExit(f"{exc}\n    启动命令：{OUTLINE_SERVER_CMD}") from exc
    every: list[suggest.Candidate] = []
    offsets: dict[str, float] = {}
    total = 0.0
    for video in _selected(args):
        cands = _candidates(video)
        every += cands
        analysed = _load_segments(video.id)[0]["analysed_until"]
        offsets[video.id] = total
        total += analysed
        if args.mode == "reel":
            payload = _timeline_payload(video.id, cands, PER_VIDEO_TARGET_SEC)
        else:
            payload = _highlights_payload(
                video.id, cands, args.ratio * analysed, {video.id: 0.0}, args.mode
            )
        _write_json(layout.timeline_path(out, video.id), payload)
        kept = sum(a["keep"] for a in payload["assessments"])
        print(
            f"{video.id} {video.file}: keep {kept}/{len(cands)}, "
            f"{len(payload['clips'])} clips {payload['total_sec']}s {payload['notes']}"
        )
    name = _summary_name(args.mode)
    if args.mode == "reel":
        summary = _timeline_payload(name, every, REEL_TARGET_SEC)
    else:
        summary = _highlights_payload(
            name, every, args.ratio * total, offsets, args.mode
        )
    _write_json(layout.timeline_path(out, name), summary)
    print(f"{name}: {len(summary['clips'])} clips {summary['total_sec']}s")


def cmd_render(args: argparse.Namespace) -> None:
    out = _out(args)
    videos = _selected(args)
    files = {v.id: v.file for v in videos}
    for name in [v.id for v in videos] + [_summary_name(args.mode)]:
        timeline = json.loads(layout.timeline_path(out, name).read_text("utf-8"))

        def frames(tl: dict = timeline):
            for clip in tl["clips"]:
                yield from read_clip(
                    RAW_DIR / files[clip["video"]], clip["src_start"], clip["duration"]
                )

        target = layout.preview_path(out, name)
        t0 = time.perf_counter()
        write_video(frames(), target)
        print(f"{target.name}: {time.perf_counter() - t0:.1f}s")


def cmd_report(args: argparse.Namespace) -> None:
    path = report.write(
        ROOT,
        _out(args),
        _selected(args),
        _summary_name(args.mode),
        BASELINE_REPORT,
        args.mode,
    )
    print(f"Wrote {path}")


def cmd_synth(args: argparse.Namespace) -> None:
    clips, path = SYNTH_SETS[args.set]
    truth = synth.build(RAW_DIR, clips, path)
    print(f"Wrote {path} ({truth['duration']}s, {len(truth['cuts'])} cuts)")


def cmd_eval(args: argparse.Namespace) -> None:
    _, path = SYNTH_SETS[args.set]
    truth = json.loads(path.with_suffix(".truth.json").read_text("utf-8"))
    cut_times = [c["time"] for c in truth["cuts"]]
    cuts = detect_cuts(path)
    predicted = [c.time for c in cuts]
    score = evaluate.score_boundaries(predicted, cut_times, args.tolerance)
    result = {
        "set": args.set,
        "video": truth["video"],
        "duration": truth["duration"],
        "clips": truth["clips"],
        "tolerance": args.tolerance,
        "truth": truth["cuts"],
        "predicted": [{"time": c.time, "kind": c.kind} for c in cuts],
        "precision": round(score.precision, 3),
        "recall": round(score.recall, 3),
        "f1": round(score.f1, 3),
        "matched": score.matched,
        "missed": score.missed,
        "false_alarms": score.false_alarms,
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"synth_eval_{args.set}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="video_demo")
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser(
        "ingest", help="register new videos in data/raw, segment them, signals"
    ).set_defaults(func=cmd_ingest)
    for name, func in (("segment", cmd_segment), ("signals", cmd_signals)):
        p = sub.add_parser(name)
        p.add_argument(
            "--force", action="store_true", help="redo videos that already have it"
        )
        p.add_argument("--collection", default="", help=COLLECTION_HELP)
        p.add_argument("--only", default="", help=ONLY_HELP)
        p.set_defaults(func=func)
    p_tag = sub.add_parser("tag")
    p_tag.add_argument("--collection", default="", help=COLLECTION_HELP)
    p_tag.add_argument("--only", default="", help=ONLY_HELP)
    p_tag.add_argument("--limit", type=int, default=0, help="max segments per video")
    p_tag.set_defaults(func=cmd_tag)
    for name, func, help_text in (
        ("aesthetic", cmd_aesthetic, "DOVER quality scores only"),
        ("analyze", cmd_analyze, "tag + DOVER quality scores in parallel"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--collection", default="", help=COLLECTION_HELP)
        p.add_argument("--only", default="", help=ONLY_HELP)
        p.add_argument("--limit", type=int, default=0, help="max segments per video")
        p.set_defaults(func=func)
    p_quality = sub.add_parser(
        "quality", help="DOVER-Mobile or UVQ 1.5 scores, compared against DOVER"
    )
    p_quality.add_argument("--model", choices=COMPARE_MODELS, required=True)
    p_quality.add_argument(
        "--threads", type=int, default=0, help="torch CPU threads (0: torch default)"
    )
    p_quality.add_argument("--collection", default="", help=COLLECTION_HELP)
    p_quality.add_argument("--only", default="", help=ONLY_HELP)
    p_quality.add_argument(
        "--limit", type=int, default=0, help="max segments per video"
    )
    p_quality.set_defaults(func=cmd_quality)
    for name, func in (
        ("suggest", cmd_suggest),
        ("compare-quality", cmd_compare_quality),
        ("render", cmd_render),
        ("report", cmd_report),
    ):
        p = sub.add_parser(name)
        p.add_argument("--collection", default="vlog", help=COLLECTION_HELP)
        p.add_argument("--only", default="", help=ONLY_HELP)
        p.add_argument(
            "--out",
            type=Path,
            default=None,
            help="result dir (default out/<collection>; vlog uses out/dover)",
        )
        if func is not cmd_compare_quality:
            p.add_argument(
                "--mode",
                choices=MODES,
                default="reel",
                help="reel: 30 s per video + 45 s reel; highlights: model outline "
                "filled with --ratio of input; ranked: --ratio by score only",
            )
        if func is cmd_suggest:
            p.add_argument(
                "--ratio",
                type=float,
                default=suggest.HIGHLIGHT_RATIO,
                help="highlights length / input length (default 1/6)",
            )
        p.set_defaults(func=func)
    p_synth = sub.add_parser("synth")
    p_synth.add_argument("--set", choices=sorted(SYNTH_SETS), default="main")
    p_synth.set_defaults(func=cmd_synth)
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--set", choices=sorted(SYNTH_SETS), default="main")
    p_eval.add_argument("--tolerance", type=float, default=0.5)
    p_eval.set_defaults(func=cmd_eval)
    args = parser.parse_args()
    args.func(args)
