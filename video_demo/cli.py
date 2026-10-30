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
    instruct,
    layout,
    outline,
    quality,
    report,
    retrieve,
    runner,
    signals,
    subtitle,
    suggest,
    synth,
    tag,
)
from video_demo import scenes as scene_material
from video_demo.media import (
    count_frames,
    cut_clip,
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
# Third-party model code stays in third_party/; every weight file lives in
# models/ (see models/README.md for sources, versions and SHA-256).
DOVER_ROOT = ROOT / "third_party" / "DOVER"
UVQ_ROOT = ROOT / "third_party" / "uvq"
MODELS_DIR = ROOT / "models"
DOVER_WEIGHTS = MODELS_DIR / "dover"
UVQ_WEIGHTS = MODELS_DIR / "uvq"

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
        "DOVER-Mobile.pth 下载到 models/dover/"
    ),
    "uvq": (
        f"从 https://github.com/google/uvq/tree/{UVQ_COMMIT} 取 uvq1p5_pytorch/、"
        "utils/ 与 LICENSE 放到 third_party/uvq/，三个 .pth 权重放到 models/uvq/"
        "（见 models/README.md）"
    ),
}
LLAMA_SERVER_CMD = (
    "llama-server -m models/gguf/Qwen3VL-2B-Instruct-Q4_K_M.gguf "
    "--mmproj models/gguf/mmproj-Qwen3VL-2B-Instruct-F16.gguf "
    "--host 127.0.0.1 --port 8091 -ngl 99 -c 8192 -np 1"
)
DOVER_SETUP_CMD = (
    "git clone https://github.com/VQAssessment/DOVER.git third_party/DOVER，"
    "再把 https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/"
    "DOVER.pth 下载到 models/dover/"
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
    """``--out``, else out/<collection> (vlog keeps out/dover)."""
    if args.out is not None:
        return args.out
    return RESULT_DIR if args.collection == "vlog" else OUT_DIR / args.collection


def _result_dirs(args: argparse.Namespace, out: Path) -> dict[str, Path]:
    """Result folder per video id, checked against every video of the
    collection (not only ``--only``) so two videos never share a folder."""
    collection = getattr(args, "collection", "")
    return layout.result_dirs(
        out, [v for v in _videos() if not collection or v.collection == collection]
    )


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

    return DoverAdapter.load(DOVER_ROOT, DOVER_WEIGHTS)


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

            return DoverAdapter.load(
                DOVER_ROOT, DOVER_WEIGHTS, config_name="dover-mobile.yml"
            )
        from video_demo.uvq import UvqAdapter

        return UvqAdapter.load(UVQ_ROOT, UVQ_WEIGHTS)
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
        weights = DOVER_WEIGHTS / "DOVER.pth"
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


# Name of the timeline across the whole collection.
HIGHLIGHTS_NAME = "highlights"
# highlights: model outline filled by retrieval, for short and long videos
# alike; ranked: the same length by score only, kept as a baseline to judge
# the outline against.
MODES = ("highlights", "ranked")
OUTLINE_SERVER_CMD = (
    "llama-server -m models/gguf/Qwen3.5-4B-Q4_K_M.gguf --host 127.0.0.1 "
    "--port 8092 -ngl 99 -c 16384 -np 1"
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


def _video_summaries(
    vid: str, cands: list[suggest.Candidate]
) -> list[tuple[list[int], dict]]:
    """(chunk members as indices into ``cands``, summary) for one video.

    ``cands`` are all of the video's segments in order. Summaries are cached
    in summary_<ID>.json and reused while the chunk's segments, CHUNK_SEC
    and SUMMARY_VERSION are unchanged; new ones are written chunk by chunk,
    so an interrupted run resumes.
    """
    cal = suggest.calibrate_by_time(cands)
    assessments = [suggest.assess(c) for c in cal]
    groups = outline.split_chunks(cal)
    path = layout.summary_path(WORK_DIR, vid)
    cached: dict[tuple[int, ...], dict] = {}
    if path.exists():
        data = json.loads(path.read_text("utf-8"))
        if (
            data["version"] == outline.SUMMARY_VERSION
            and data["chunk_sec"] == outline.CHUNK_SEC
        ):
            cached = {tuple(c["segments"]): c for c in data["chunks"]}
    entries: list[dict] = []
    out: list[tuple[list[int], dict]] = []
    for k, members in enumerate(groups, start=1):
        key = tuple(cal[i].index for i in members)
        entry = cached.get(key)
        if entry is None:
            try:
                summary, raw, seconds, errors = outline.summarize_chunk(
                    members, cal, assessments
                )
            except outline.OutlineError as exc:
                raise SystemExit(
                    f"{vid} 第 {k} 块（片段 {key[0]}–{key[-1]}）总结失败：{exc}"
                ) from exc
            entry = {
                "segments": list(key),
                "start": round(cal[members[0]].start, 2),
                "end": round(cal[members[-1]].end, 2),
                "summary": summary,
                "raw": raw,
                "seconds": round(seconds, 1),
                "failed_attempts": errors,
            }
            print(
                f"{vid} 第 {k}/{len(groups)} 块总结 {seconds:.0f}s："
                f"{summary['summary'][:40]}",
                flush=True,
            )
        entries.append(entry)
        out.append((members, entry["summary"]))
        _write_json(
            path,
            {
                "version": outline.SUMMARY_VERSION,
                "chunk_sec": outline.CHUNK_SEC,
                "video_id": vid,
                "chunks": entries,
            },
        )
    return out


def _chunks(
    cands: list[suggest.Candidate], offsets: dict[str, float]
) -> list[outline.Chunk]:
    """Chunks along the timeline, each video's from its own summaries."""
    chunks: list[outline.Chunk] = []
    start = 0
    while start < len(cands):
        vid = cands[start].video
        end = start
        while end < len(cands) and cands[end].video == vid:
            end += 1
        for members, summary in _video_summaries(vid, cands[start:end]):
            first, last = cands[start + members[0]], cands[start + members[-1]]
            chunks.append(
                outline.Chunk(
                    number=len(chunks) + 1,
                    video=vid,
                    members=[start + i for i in members],
                    start=offsets.get(vid, 0.0) + first.start,
                    end=offsets.get(vid, 0.0) + last.end,
                    summary=summary,
                )
            )
        start = end
    return chunks


Vectors = dict[tuple[str, int], np.ndarray]


def _scene_groups(
    chunk_members: list[list[int]],
    cands: list[suggest.Candidate],
    text: Vectors,
    image: Vectors,
) -> list[list[int]]:
    """Candidate indices of each scene along the timeline; a scene never
    crosses a chunk (``chunk_members``) or a shot cut."""
    out: list[list[int]] = []
    for ms in chunk_members:
        keys = [(cands[i].video, cands[i].index) for i in ms]
        sim = retrieve.boundary_similarity(
            np.stack([text[k] for k in keys]), np.stack([image[k] for k in keys])
        )
        shots = [cands[i].shot for i in ms]
        hard = {p for p in range(len(ms) - 1) if shots[p] != shots[p + 1]}
        for g in retrieve.split_scenes(
            [cands[i].start for i in ms], [cands[i].end for i in ms], sim, hard
        ):
            out.append([ms[p] for p in g])
    return out


class _Retrieval:
    """Query embedder and the segment vectors of the selected videos."""

    def __init__(self, videos: list[layout.Video]):
        from video_demo import embed

        try:
            self.embedder = embed.EmbeddingGemmaAdapter()
        except embed.EmbedError as exc:
            raise SystemExit(f"向量模型不可用：{exc}") from exc
        self.text: dict[tuple[str, int], np.ndarray] = {}
        self.image: dict[tuple[str, int], np.ndarray] = {}
        self.docs: dict[tuple[str, int], str] = {}
        for video in videos:
            try:
                rows = _tag_rows(video.id)
                said, digest = _segment_speech(video.id)
                idx = retrieve.load_index(
                    layout.index_path(WORK_DIR, video.id),
                    self.embedder.model,
                    check_complete=len(rows),
                    speech=digest,
                )
            except (SystemExit, retrieve.RetrievalError) as exc:
                # MediaPipe's own cleanup fails once the interpreter is
                # shutting down, so release it before exiting.
                self.embedder.close()
                raise SystemExit(f"{video.id} {video.file}：{exc}") from exc
            for i, t, m in zip(idx["index"], idx["text"], idx["image"], strict=True):
                self.text[(video.id, int(i))] = t
                self.image[(video.id, int(i))] = m
            for r in rows:
                self.docs[(video.id, r["index"])] = retrieve.segment_text(
                    r["tags"], said.get(r["index"], "")
                )

    def _key(self, c: suggest.Candidate) -> tuple[str, int]:
        return (c.video, c.index)

    def scenes(
        self, chunks: list[outline.Chunk], cands: list[suggest.Candidate]
    ) -> list[outline.Scene]:
        """Scenes along the timeline; a scene never crosses a chunk or a
        shot cut."""
        groups = _scene_groups(
            [ch.members for ch in chunks], cands, self.text, self.image
        )
        return [outline.Scene(n, g) for n, g in enumerate(groups, start=1)]

    def rank(
        self,
        queries: list[str],
        pool: list[outline.Scene],
        cands: list[suggest.Candidate],
    ) -> tuple[list[dict[int, float]], list[dict[int, float]], list[dict]]:
        """Per query: relevance of the scenes in ``pool`` and of their
        segments, and the top five of each scene ranking."""
        members = [i for s in pool for i in s.members]
        keys = {i: self._key(cands[i]) for i in members}
        docs = {i: self.docs[keys[i]] for i in members}
        text = {i: self.text[keys[i]] for i in members}
        image = {i: self.image[keys[i]] for i in members}
        scene_docs = {s.number: "；".join(docs[i] for i in s.members) for s in pool}
        scene_text = {
            s.number: retrieve.pooled([text[i] for i in s.members]) for s in pool
        }
        scene_image = {
            s.number: retrieve.pooled([image[i] for i in s.members]) for s in pool
        }
        numbers = [s.number for s in pool]
        per_scene, per_segment, tops = [], [], []
        for query in queries:
            vec = self.embedder.embed_query(query)
            rel, rankings = retrieve.relevance(
                query, vec, numbers, scene_docs, scene_text, scene_image
            )
            per_scene.append(rel)
            per_segment.append(
                retrieve.relevance(query, vec, members, docs, text, image)[0]
            )
            tops.append(
                {
                    "query": query,
                    "top": {name: ranking[:5] for name, ranking in rankings.items()},
                }
            )
        return per_scene, per_segment, tops

    def relevance(
        self,
        plan: dict,
        chunks: list[outline.Chunk],
        scenes: list[outline.Scene],
        cands: list[suggest.Candidate],
    ) -> tuple[list[list[dict[int, float]]], list[list[dict[int, float]]], list[dict]]:
        """Per part and scene slot: relevance of the scenes in the part's
        chunks and of their segments, and the top five of each scene
        ranking for the timeline file."""
        scene_rel: list[list[dict[int, float]]] = []
        segment_rel: list[list[dict[int, float]]] = []
        report: list[dict] = []
        for part in plan["parts"]:
            per_scene, per_segment, tops = self.rank(
                part["scenes"], outline.part_scenes(part, chunks, scenes), cands
            )
            scene_rel.append(per_scene)
            segment_rel.append(per_segment)
            report.append({"part": part["name"], "queries": tops})
        return scene_rel, segment_rel, report


def _tag_rows(vid: str) -> list[dict]:
    path = layout.tags_path(WORK_DIR, vid)
    if not path.exists():
        raise SystemExit(f"{vid}：还没有打标签（{path}）")
    return [json.loads(x) for x in path.read_text("utf-8").splitlines()]


def _segment_speech(vid: str) -> tuple[dict[int, str], str]:
    """What is said in each segment (from refined subtitles, none when the
    video has no subtitles) and the fingerprint the index records."""
    path = layout.subtitle_path(WORK_DIR, vid)
    if not path.exists():
        return {}, ""
    sentences = subtitle.sentences_from_json(
        subtitle.read_subtitles(path, vid)["sentences"]
    )
    _, segments = _load_segments(vid)
    said = {s.index: subtitle.speech_text(sentences, s.start, s.end) for s in segments}
    return said, subtitle.speech_digest([said[k] for k in sorted(said)])


def cmd_index(args: argparse.Namespace) -> None:
    """Text and keyframe image vectors per segment, for retrieval fill."""
    from video_demo import embed

    videos = _selected(args)
    try:
        embedder = embed.EmbeddingGemmaAdapter()
    except embed.EmbedError as exc:
        raise SystemExit(f"向量模型不可用：{exc}") from exc
    print(f"{embedder.model}, {embedder.dimensions} dimensions")
    for video in videos:
        path = layout.index_path(WORK_DIR, video.id)
        if args.force:
            path.unlink(missing_ok=True)
        rows = _tag_rows(video.id)
        sig = {s["index"]: s for s in _load_signals(video.id)}
        kdir = layout.keyframes_dir(WORK_DIR, video.id)
        said, digest = _segment_speech(video.id)
        segments = [
            (
                r["index"],
                r["tags"],
                [kdir / k for k in sig[r["index"]]["keyframes"]],
                said.get(r["index"], ""),
            )
            for r in rows
        ]
        t0 = time.perf_counter()

        def progress(n: int, total: int, vid: str = video.id, t0: float = t0) -> None:
            if n % 50 == 0 or n == total:
                print(f"  {vid} {n}/{total} ({time.perf_counter() - t0:.0f}s)")

        try:
            retrieve.build_index(
                path, embedder.model, segments, embedder, digest, progress
            )
        except retrieve.RetrievalError as exc:
            raise SystemExit(f"{video.id} {video.file}：{exc}") from exc
        print(f"{video.id} {video.file}: {len(segments)} segments -> {path}")
    embedder.close()


def _chunk_rows(
    chunks: list[outline.Chunk], cands: list[suggest.Candidate]
) -> list[dict]:
    return [
        {
            "number": ch.number,
            "video": ch.video,
            "start": round(ch.start, 2),
            "end": round(ch.end, 2),
            "segments": [cands[i].index for i in ch.members],
            "summary": ch.summary,
        }
        for ch in chunks
    ]


def _scene_rows(
    scenes: list[outline.Scene],
    cands: list[suggest.Candidate],
    offsets: dict[str, float],
) -> list[dict]:
    """Scenes with start and end on the timeline that ``offsets`` sets
    (empty: within their own video)."""
    return [
        {
            "number": sc.number,
            "video": cands[sc.members[0]].video,
            "start": round(
                offsets.get(cands[sc.members[0]].video, 0.0)
                + cands[sc.members[0]].start,
                2,
            ),
            "end": round(
                offsets.get(cands[sc.members[-1]].video, 0.0)
                + cands[sc.members[-1]].end,
                2,
            ),
            "segments": [cands[i].index for i in sc.members],
        }
        for sc in scenes
    ]


def _highlights_payload(
    name: str,
    cands: list[suggest.Candidate],
    target: float,
    offsets: dict[str, float],
    mode: str,
    retrieval: _Retrieval | None = None,
) -> dict:
    """Highlights: a model outline whose scene slots are filled from
    retrieved scenes (needs ``retrieval``). Ranked: score only (the
    baseline highlights mode kept for comparison)."""
    chunks = _chunks(cands, offsets) if mode == "highlights" else []
    cands = suggest.calibrate_by_time(cands)
    assessments = [suggest.assess(c) for c in cands]
    extra: dict[str, Any] = {}
    if mode == "ranked":
        clips, notes = suggest.build_highlights(cands, assessments, target, offsets)
    else:
        if retrieval is None:
            raise ValueError("highlights mode needs retrieval")
        scenes = retrieval.scenes(chunks, cands)
        try:
            plan, raw, seconds, errors = outline.plan_outline(chunks, target)
        except outline.OutlineError as exc:
            raise SystemExit(
                f"{name}：大纲生成失败（3 次都没通过校验）：{exc}"
            ) from exc
        scene_rel, segment_rel, retrieved = retrieval.relevance(
            plan, chunks, scenes, cands
        )
        clips, notes, slots = outline.fill(
            plan, chunks, scenes, cands, assessments, scene_rel, segment_rel
        )
        extra = {
            "slots": slots,
            "retrieval": retrieved,
            "outline": plan,
            "outline_raw": raw,
            "outline_seconds": round(seconds, 1),
            "outline_failed_attempts": errors,
            "chunks": _chunk_rows(chunks, cands),
            "scenes": _scene_rows(scenes, cands, offsets),
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
    """``--ratio`` of each video, and of the whole collection, as whole
    segments; one timeline per video in its result folder and one across the
    collection at its root."""
    out = _out(args)
    dirs = _result_dirs(args, out)
    if args.mode == "highlights":
        try:
            tag.check_server(outline.SERVER_URL)
        except ConnectionError as exc:
            raise SystemExit(f"{exc}\n    启动命令：{OUTLINE_SERVER_CMD}") from exc
    retrieval = _Retrieval(_selected(args)) if args.mode == "highlights" else None
    every: list[suggest.Candidate] = []
    offsets: dict[str, float] = {}
    total = 0.0
    for video in _selected(args):
        cands = _candidates(video)
        every += cands
        analysed = _load_segments(video.id)[0]["analysed_until"]
        offsets[video.id] = total
        total += analysed
        payload = _highlights_payload(
            video.id,
            cands,
            args.ratio * analysed,
            {video.id: 0.0},
            args.mode,
            retrieval,
        )
        _write_json(layout.timeline_path(dirs[video.id], args.mode), payload)
        kept = sum(a["keep"] for a in payload["assessments"])
        print(
            f"{video.id} {video.file}: keep {kept}/{len(cands)}, "
            f"{len(payload['clips'])} clips {payload['total_sec']}s {payload['notes']}"
        )
    name = HIGHLIGHTS_NAME
    if len(offsets) == 1:
        # One video: the collection timeline is that video's; asking the
        # model for a second outline would only give a different one.
        summary = {**payload, "name": name}
    else:
        summary = _highlights_payload(
            name, every, args.ratio * total, offsets, args.mode, retrieval
        )
    _write_json(layout.timeline_path(out, args.mode), summary)
    print(f"{name}: {len(summary['clips'])} clips {summary['total_sec']}s")
    if retrieval:
        retrieval.embedder.close()


def _scene_catalog(
    scenes: list[outline.Scene],
    cands: list[suggest.Candidate],
    offsets: dict[str, float],
) -> list[dict]:
    """Number, start, end (on the timeline ``offsets`` sets), title and
    summary of every scene that keeps footage.

    ``cands`` are not calibrated: each video is calibrated and assessed on
    its own, as ``scenes`` does, so the summaries cached by ``scenes`` are
    reused and missing ones are written to the same cache.
    """
    catalog: list[dict] = []
    start = 0
    while start < len(cands):
        vid = cands[start].video
        end = start
        while end < len(cands) and cands[end].video == vid:
            end += 1
        own = [s for s in scenes if start <= s.members[0] < end]
        cal = suggest.calibrate_by_time(cands[start:end])
        summaries = _scene_summaries(
            vid,
            [[i - start for i in s.members] for s in own],
            cal,
            [suggest.assess(c) for c in cal],
        )
        for n, sc in enumerate(own, start=1):
            if n in summaries:
                catalog.append(
                    {
                        "number": sc.number,
                        "start": round(
                            offsets.get(vid, 0.0) + cands[sc.members[0]].start, 2
                        ),
                        "end": round(
                            offsets.get(vid, 0.0) + cands[sc.members[-1]].end, 2
                        ),
                        **summaries[n],
                    }
                )
        start = end
    return catalog


def cmd_edit(args: argparse.Namespace) -> None:
    """A cut for the user's instruction across the selected videos: the
    outline model reads the instruction and the scene catalog, every scene
    is searched for its slots. Timeline, preview and report go to a folder
    of their own under out/<collection>/edits/."""
    instruction = args.instruction.strip()
    if not instruction:
        raise SystemExit("剪辑要求不能为空")
    out = _out(args)
    try:
        tag.check_server(outline.SERVER_URL)
    except ConnectionError as exc:
        raise SystemExit(f"{exc}\n    启动命令：{OUTLINE_SERVER_CMD}") from exc
    videos = _selected(args)
    cands: list[suggest.Candidate] = []
    offsets: dict[str, float] = {}
    total = 0.0
    for video in videos:
        cands += _candidates(video)
        offsets[video.id] = total
        total += _load_segments(video.id)[0]["analysed_until"]
    chunks = _chunks(cands, offsets)
    try:
        duration, duration_raw, _, duration_errors = instruct.parse_duration(
            instruction
        )
    except outline.OutlineError as exc:
        raise SystemExit(f"读取成片时长失败（3 次都没通过校验）：{exc}") from exc
    raw = cands
    cands = suggest.calibrate_by_time(cands)
    assessments = [suggest.assess(c) for c in cands]
    usable = sum(
        c.duration
        for c, a in zip(cands, assessments, strict=True)
        if a.keep and c.duration >= suggest.MIN_CLIP_SEC
    )
    if duration["target_sec"]:
        target = float(duration["target_sec"])
        source = {"kind": "instruction", "quote": duration["quote"]}
    else:
        target = args.ratio * total
        source = {"kind": "ratio", "ratio": args.ratio}
    if target > usable:
        raise SystemExit(f"要求成片 {target:.0f}s，可用素材只有 {usable:.0f}s")
    retrieval = _Retrieval(videos)
    try:
        scenes = retrieval.scenes(chunks, cands)
        catalog = _scene_catalog(scenes, raw, offsets)
        try:
            plan, outline_raw, seconds, errors = instruct.plan_request(
                instruction, catalog, total, target
            )
        except outline.OutlineError as exc:
            raise SystemExit(f"大纲生成失败（3 次都没通过校验）：{exc}") from exc
        ranked = [
            retrieval.rank([s["query"] for s in part["scenes"]], scenes, cands)
            for part in plan["parts"]
        ]
    finally:
        retrieval.embedder.close()
    named = set().union(*(outline.highlight_members(ch) for ch in chunks))
    clips, notes, slots = instruct.fill_request(
        plan,
        scenes,
        cands,
        assessments,
        [r[0] for r in ranked],
        [r[1] for r in ranked],
        named,
    )
    if not clips:
        raise SystemExit(f"没有选到任何片段：{'；'.join(notes)}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = layout.edit_dir(out, stamp)
    timeline = {
        "name": "edit",
        "mode": "edit",
        "instruction": instruction,
        "created": stamp,
        "videos": [v.id for v in videos],
        "target_sec": round(target, 1),
        "target_source": source,
        "duration_raw": duration_raw,
        "duration_failed_attempts": duration_errors,
        "total_sec": round(sum(c.duration for c in clips), 2),
        "notes": notes,
        "clips": [suggest.clip_to_dict(c) for c in clips],
        "assessments": [vars(a) for a in assessments],
        "slots": slots,
        "retrieval": [
            {"part": part["name"], "queries": r[2]}
            for part, r in zip(plan["parts"], ranked, strict=True)
        ],
        "outline": plan,
        "outline_raw": outline_raw,
        "outline_seconds": round(seconds, 1),
        "outline_failed_attempts": errors,
        "chunks": _chunk_rows(chunks, cands),
        "catalog": catalog,
        # Within their own video: the cut is not in shooting order.
        "scenes": _scene_rows(scenes, cands, {}),
    }
    _write_json(layout.edit_timeline_path(folder), timeline)
    print(
        f"{plan['title']}：{len(clips)} clips {timeline['total_sec']}s / "
        f"目标 {target:.0f}s {notes}"
    )
    _write_preview(
        timeline, {v.id: v.file for v in videos}, layout.edit_preview_path(folder)
    )
    print(f"Wrote {report.write_edit(ROOT, folder, videos, timeline)}")


def cmd_render(args: argparse.Namespace) -> None:
    out = _out(args)
    dirs = _result_dirs(args, out)
    videos = _selected(args)
    files = {v.id: v.file for v in videos}
    for folder in [dirs[v.id] for v in videos] + [out]:
        path = layout.timeline_path(folder, args.mode)
        timeline = json.loads(path.read_text("utf-8"))
        _write_preview(timeline, files, layout.preview_path(folder, args.mode))


def _write_preview(timeline: dict, files: dict[str, str], target: Path) -> None:
    """The timeline's clips joined into one preview video; ``files`` maps
    video ids to raw file names."""

    def frames() -> Iterator:
        for clip in timeline["clips"]:
            yield from read_clip(
                RAW_DIR / files[clip["video"]], clip["src_start"], clip["duration"]
            )

    t0 = time.perf_counter()
    write_video(frames(), target)
    print(f"{target}: {time.perf_counter() - t0:.1f}s")


def _index_vectors(video: layout.Video) -> tuple[Vectors, Vectors]:
    """A video's stored text and image vectors. Scenes only compare vectors
    within one video, so any complete index will do; no embedder is loaded."""
    path = layout.index_path(WORK_DIR, video.id)
    if not path.exists():
        raise SystemExit(
            f"{video.id} {video.file}：缺少 {path.name}，"
            "先运行 python -m video_demo index"
        )
    try:
        idx = retrieve.load_index(
            path, retrieve.read_model(path), check_complete=len(_tag_rows(video.id))
        )
    except retrieve.RetrievalError as exc:
        raise SystemExit(f"{video.id} {video.file}：{exc}") from exc
    text: Vectors = {}
    image: Vectors = {}
    for i, t, m in zip(idx["index"], idx["text"], idx["image"], strict=True):
        text[(video.id, int(i))] = t
        image[(video.id, int(i))] = m
    return text, image


def _scene_summaries(
    vid: str,
    groups: list[list[int]],
    cands: list[suggest.Candidate],
    assessments: list[suggest.Assessment],
) -> dict[int, dict[str, str]]:
    """Title and summary per scene number, for scenes that keep footage.

    Cached in scene_<ID>.json by the kept segments and SCENES_VERSION, and
    saved after every new answer, so an interrupted run resumes.
    """
    path = layout.scene_summary_path(WORK_DIR, vid)
    cached: dict[tuple[int, ...], dict] = {}
    if path.exists():
        data = json.loads(path.read_text("utf-8"))
        if data["version"] == scene_material.SCENES_VERSION:
            cached = {tuple(e["segments"]): e for e in data["scenes"]}
    used: list[dict] = []
    out: dict[int, dict[str, str]] = {}
    for n, members in enumerate(groups, start=1):
        runs = scene_material.kept_runs(members, cands, assessments)
        kept = [i for r in runs for i in r]
        if not kept:
            continue
        key = tuple(cands[i].index for i in kept)
        entry = cached.get(key)
        if entry is None:
            try:
                summary, raw, seconds, errors = scene_material.describe_scene(
                    kept, cands, assessments
                )
            except outline.OutlineError as exc:
                raise SystemExit(
                    f"{vid} 场景 {n}：概况生成失败（3 次都没通过校验）：{exc}"
                ) from exc
            entry = {
                "segments": list(key),
                **summary,
                "raw": raw,
                "seconds": round(seconds, 1),
                "failed_attempts": errors,
            }
            print(f"{vid} 场景 {n}/{len(groups)}：{summary['title']}（{seconds:.0f}s）")
        used.append(entry)
        out[n] = {"title": entry["title"], "summary": entry["summary"]}
        if key not in cached:
            cached[key] = entry
            _write_json(
                path, {"version": scene_material.SCENES_VERSION, "scenes": used}
            )
    _write_json(path, {"version": scene_material.SCENES_VERSION, "scenes": used})
    return out


def _scene_listing(
    video: layout.Video, n_scenes: int, entries: list[dict], complete: bool
) -> dict:
    """scenes.json: ``complete`` is False while clips are still being cut."""
    removed = sum(r["end"] - r["start"] for e in entries for r in e["removed"])
    return {
        "video": video.id,
        "file": video.file,
        "rule": f"片段得分 ≤ {suggest.DROP_SCORE:.0f} 或短于 "
        f"{suggest.MIN_CLIP_SEC}s 的视为废片，从场景中切除；"
        "每个场景的保留片段按时间顺序接成一个视频，原片不改动",
        "complete": complete,
        "scenes_total": n_scenes,
        "kept_sec": round(sum(e["kept_sec"] for e in entries), 2),
        "removed_sec": round(removed, 2),
        "scenes": entries,
    }


def cmd_scenes(args: argparse.Namespace) -> None:
    """Scene material besides the timelines: one clip per scene, its kept
    segments back to back with the rejected ones cut out, at the source's
    size, frame rate and audio, and scenes.json describing every clip."""
    out = _out(args)
    dirs = _result_dirs(args, out)
    videos = _selected(args)
    vectors = {v.id: _index_vectors(v) for v in videos}
    try:
        tag.check_server(outline.SERVER_URL)
    except ConnectionError as exc:
        raise SystemExit(f"{exc}\n    启动命令：{OUTLINE_SERVER_CMD}") from exc
    for video in videos:
        cands = suggest.calibrate_by_time(_candidates(video))
        assessments = [suggest.assess(c) for c in cands]
        groups = _scene_groups(outline.split_chunks(cands), cands, *vectors[video.id])
        summaries = _scene_summaries(video.id, groups, cands, assessments)
        folder = dirs[video.id]
        listing = layout.scenes_dir(folder) / "scenes.json"
        # A clip on disk is reused only if the last listing cut it from the
        # same source spans.
        done: dict[str, list[list[float]]] = {}
        if listing.exists() and not args.force:
            for s in json.loads(listing.read_text("utf-8"))["scenes"]:
                if s["file"]:
                    done[s["file"]] = [[p["start"], p["end"]] for p in s["parts"]]
        entries: list[dict] = []
        for n, members in enumerate(groups, start=1):
            runs = scene_material.kept_runs(members, cands, assessments)
            path = layout.scene_clip_path(folder, n) if runs else None
            if path is not None:
                spans = scene_material.kept_spans(runs, cands)
                if not path.exists() or done.get(path.name) != [list(s) for s in spans]:
                    t0 = time.perf_counter()
                    cut_clip(RAW_DIR / video.file, spans, path)
                    print(
                        f"  {path.name} {sum(e - s for s, e in spans):.1f}s "
                        f"from {len(spans)} part(s) ({time.perf_counter() - t0:.1f}s)"
                    )
            entries.append(
                scene_material.scene_entry(
                    n,
                    members,
                    cands,
                    assessments,
                    summaries.get(n),
                    path.name if path else None,
                )
            )
            _write_json(listing, _scene_listing(video, len(groups), entries, False))
        _write_json(listing, _scene_listing(video, len(groups), entries, True))
        expected = {e["file"] for e in entries if e["file"]}
        stale = sorted(
            p.name
            for p in layout.scenes_dir(folder).glob("scene_*.mp4")
            if p.name not in expected and not p.name.endswith(".partial.mp4")
        )
        print(
            f"{video.id} {video.file}: {len(groups)} scenes, {len(expected)} clips, "
            f"kept {sum(e['kept_sec'] for e in entries):.0f}s -> {listing}"
        )
        if stale:
            print(f"  以下旧文件不再对应任何场景（未删除）：{', '.join(stale)}")


def cmd_report(args: argparse.Namespace) -> None:
    out = _out(args)
    path = report.write(
        ROOT,
        out,
        _selected(args),
        _result_dirs(args, out),
        BASELINE_REPORT,
        args.mode,
    )
    print(f"Wrote {path}")


def cmd_subtitle(args: argparse.Namespace) -> None:
    """Speech recognition, then 4B refinement, then SRT per video; with
    ``--mode`` also SRT for that mode's timelines."""
    out = _out(args)
    dirs = _result_dirs(args, out)
    videos = [v for v in _selected(args) if _load_segments(v.id)[0]["has_audio"]]
    todo = [
        v for v in videos if args.force or not layout.asr_path(WORK_DIR, v.id).exists()
    ]
    if todo:
        # Recognition runs alone and frees its models before the 4B is used.
        try:
            asr = subtitle.SenseVoiceAdapter(ROOT / "models" / "asr")
        except subtitle.SubtitleError as exc:
            raise SystemExit(f"语音识别模型不可用：{exc}") from exc
        for v in todo:
            t0 = time.perf_counter()
            sentences = asr.transcribe(RAW_DIR / v.file)
            path = layout.asr_path(WORK_DIR, v.id)
            subtitle.write_asr(path, v.id, asr.model, sentences)
            print(
                f"{v.id} {v.file}: {len(sentences)} sentences, "
                f"{time.perf_counter() - t0:.1f}s -> {path}"
            )
        asr.close()
    cues_by_video: dict[str, list[subtitle.Cue]] = {}
    for v in videos:
        path = layout.subtitle_path(WORK_DIR, v.id)
        if args.force or v in todo or not path.exists():
            raw = subtitle.read_asr(layout.asr_path(WORK_DIR, v.id), v.id)
            context = subtitle.video_context(_tag_rows(v.id))
            try:
                refined, seconds, failed = subtitle.refine(
                    raw, Path(v.file).stem, context
                )
            except subtitle.SubtitleError as exc:
                raise SystemExit(f"{v.id} {v.file}：字幕校对失败：{exc}") from exc
            sentences = subtitle.refined_sentences(raw, refined)
            asr_model = json.loads(layout.asr_path(WORK_DIR, v.id).read_text("utf-8"))[
                "model"
            ]
            subtitle.write_subtitles(
                path, v.id, asr_model, raw, sentences, seconds, failed
            )
            changed = sum(refined[s.id] != s.text for s in raw)
            print(
                f"{v.id} {v.file}: refined {changed}/{len(raw)} sentences "
                f"in {seconds:.1f}s ({len(failed)} retries)"
            )
        cues_by_video[v.id] = subtitle.read_cues(path, v.id)
        srt = layout.srt_path(dirs[v.id])
        srt.parent.mkdir(parents=True, exist_ok=True)
        srt.write_text(subtitle.to_srt(cues_by_video[v.id]), encoding="utf-8")
        print(f"  {len(cues_by_video[v.id])} cues -> {srt}")
    if args.mode:
        for folder in [dirs[v.id] for v in videos] + [out]:
            path = layout.timeline_path(folder, args.mode)
            if not path.exists():
                continue
            clips = json.loads(path.read_text("utf-8"))["clips"]
            cues = subtitle.map_cues(cues_by_video, clips)
            srt = layout.srt_path(folder, args.mode)
            srt.write_text(subtitle.to_srt(cues), encoding="utf-8")
            print(f"{len(cues)} cues on {path.name} -> {srt}")


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
    p_index = sub.add_parser(
        "index", help="EmbeddingGemma 2 text and image vectors per segment"
    )
    p_index.add_argument("--force", action="store_true", help="rebuild the index")
    p_index.add_argument("--collection", default="", help=COLLECTION_HELP)
    p_index.add_argument("--only", default="", help=ONLY_HELP)
    p_index.set_defaults(func=cmd_index)
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
        ("scenes", cmd_scenes),
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
        if func is cmd_scenes:
            p.add_argument(
                "--force", action="store_true", help="cut every scene clip again"
            )
        elif func is not cmd_compare_quality:
            p.add_argument(
                "--mode",
                choices=MODES,
                default="highlights",
                help="highlights: model outline filled with --ratio of input; "
                "ranked: --ratio by score only (baseline)",
            )
        if func is cmd_suggest:
            p.add_argument(
                "--ratio",
                type=float,
                default=suggest.HIGHLIGHT_RATIO,
                help="highlights length / input length (default 1/6)",
            )
        p.set_defaults(func=func)
    p_sub = sub.add_parser(
        "subtitle", help="speech recognition + 4B refinement -> SRT per video"
    )
    p_sub.add_argument("--collection", default="vlog", help=COLLECTION_HELP)
    p_sub.add_argument("--only", default="", help=ONLY_HELP)
    p_sub.add_argument("--out", type=Path, default=None, help="result dir")
    p_sub.add_argument(
        "--mode",
        choices=["", *MODES],
        default="",
        help="also map subtitles onto this mode's timelines",
    )
    p_sub.add_argument(
        "--force", action="store_true", help="recognise and refine again"
    )
    p_sub.set_defaults(func=cmd_subtitle)
    p_edit = sub.add_parser(
        "edit", help="a cut for one instruction: outline, retrieval fill, report"
    )
    p_edit.add_argument(
        "instruction", help="剪辑要求，如：剪 60 秒的教堂，先外观后内部"
    )
    p_edit.add_argument("--collection", default="vlog", help=COLLECTION_HELP)
    p_edit.add_argument("--only", default="", help=ONLY_HELP)
    p_edit.add_argument(
        "--out",
        type=Path,
        default=None,
        help="result dir (default out/<collection>; vlog uses out/dover)",
    )
    p_edit.add_argument(
        "--ratio",
        type=float,
        default=suggest.HIGHLIGHT_RATIO,
        help="cut length / input length when the instruction gives none",
    )
    p_edit.set_defaults(func=cmd_edit)
    p_synth = sub.add_parser("synth")
    p_synth.add_argument("--set", choices=sorted(SYNTH_SETS), default="main")
    p_synth.set_defaults(func=cmd_synth)
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--set", choices=sorted(SYNTH_SETS), default="main")
    p_eval.add_argument("--tolerance", type=float, default=0.5)
    p_eval.set_defaults(func=cmd_eval)
    args = parser.parse_args()
    args.func(args)
