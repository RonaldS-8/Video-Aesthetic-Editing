"""Command line entry: python -m video_demo <stage>."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from video_demo import (
    beats,
    category,
    evaluate,
    instruct,
    layout,
    narrate,
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
    translate,
    tts,
)
from video_demo import scenes as scene_material
from video_demo.media import (
    AUDIO_RATE,
    AUDIO_XFADE_SEC,
    DISSOLVE_SEC,
    DUCK_GAIN,
    EDIT_CRF,
    OUT_FPS,
    OUT_SIZE,
    PREVIEW_CRF,
    PROXY_SIZE,
    FrameDraw,
    SubtitlePainter,
    TitlePainter,
    capture_time,
    compose,
    count_frames,
    crossfade,
    cut_clip,
    dissolve,
    edit_size,
    is_pq,
    iter_clip,
    make_proxy,
    probe,
    read_audio_clip,
    read_clip,
    segment_frames_every,
    segment_frames_picked,
    voice_track,
    write_av,
)
from video_demo.segment import (
    MAX_SEGMENT_SEC,
    SPLIT_SHIFT_SEC,
    Segment,
    detect_cuts,
    merge_short,
    spans_from_cuts,
    split_long,
    strictly_inside,
)

ROOT = Path(__file__).resolve().parent.parent
# VIDEO_DEMO_DATA (layout.data_dir) and VIDEO_DEMO_OUT move the data and
# result folders, e.g. to scratch space for one case of an evaluation.
DATA_DIR = layout.data_dir(ROOT)
RAW_DIR = DATA_DIR / "raw"
WORK_DIR = DATA_DIR / "work"
OUT_DIR = Path(os.environ.get("VIDEO_DEMO_OUT", ROOT / "out"))
# Burned subtitles are drawn with this font unless VIDEO_DEMO_FONT is set.
# Language of the subtitles burned into previews: speech in another
# language is shown translated (``subtitle --translate``).
SUBTITLE_LANG = "zh"
SUBTITLE_FONT = (
    Path("C:/Windows/Fonts/msyh.ttc")
    if sys.platform == "win32"
    else Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
)
# The frozen VLM-only report from `main`, linked from every report.
BASELINE_REPORT = layout.baseline_report_path(OUT_DIR)
SYNTH_PATH = ROOT / "data" / "synth" / "synth_cuts.mp4"
# Third-party model code stays in third_party/; every weight file lives in
# models/ (see models/README.md for sources, versions and SHA-256).
DOVER_ROOT = ROOT / "third_party" / "DOVER"
UVQ_ROOT = ROOT / "third_party" / "uvq"
MODELS_DIR = layout.models_dir(ROOT)
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
# One Qwen3.5-4B server tags segments (with the vision projector) and
# writes summaries and outlines.
LLAMA_SERVER_CMD = (
    "llama-server -m models/gguf/Qwen3.5-4B-Q4_K_M.gguf "
    "--mmproj models/gguf/mmproj-Qwen3.5-4B-Q8_0.gguf "
    "--host 127.0.0.1 --port 8092 -ngl 99 -c 16384 -np 1"
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


def _picture(video: layout.Video) -> Path:
    """The file whose picture analysis reads: the proxy when the video has
    one (make_proxy), else the video itself."""
    proxy = layout.proxy_path(WORK_DIR, video.id)
    return proxy if proxy.exists() else _raw(video)


def _make_proxies(videos: list[layout.Video]) -> None:
    """A proxy for every video larger than PROXY_SIZE that has none yet."""
    for video in videos:
        info = probe(_raw(video))
        proxy = layout.proxy_path(WORK_DIR, video.id)
        if info.width * info.height <= PROXY_SIZE[0] * PROXY_SIZE[1] or proxy.exists():
            continue
        t0 = time.perf_counter()
        make_proxy(_raw(video), proxy)
        print(
            f"{video.id} {video.file}: proxy {info.width}x{info.height} -> "
            f"{PROXY_SIZE[1]}p in {time.perf_counter() - t0:.0f}s",
            flush=True,
        )


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
    """Shots, then windows split between sentences: speech is recognised
    first for every video with audio (cached in data/work/asr/)."""
    infos = {}
    for video in videos:
        layout.verify(RAW_DIR, video)
        infos[video.id] = probe(_raw(video))
    _make_proxies(videos)
    _recognise([v for v in videos if infos[v.id].has_audio], force=False)
    for video in videos:
        path = _picture(video)
        info = infos[video.id]
        end_time = END_TIME.get(video.id)
        analysed = info.duration if end_time is None else end_time
        sentences = (
            subtitle.read_asr(layout.asr_path(WORK_DIR, video.id), video.id)
            if info.has_audio
            else []
        )
        spoken, clauses = subtitle.speech_spans(sentences)
        t0 = time.perf_counter()
        cuts = detect_cuts(path, end_time)
        raw_shots = spans_from_cuts(cuts, analysed)
        shots = merge_short(raw_shots)
        segments = split_long(shots, sentences=spoken, clauses=clauses)
        elapsed = time.perf_counter() - t0
        boundaries = [
            {
                "time": a.end,
                "kind": "shot" if a.shot != b.shot else "split",
                "speech": (
                    "mid_clause"
                    if strictly_inside(a.end, clauses)
                    else "clause_break" if strictly_inside(a.end, spoken) else "quiet"
                ),
            }
            for a, b in zip(segments, segments[1:], strict=False)
        ]
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
            "split_rule": "speech",
            "split_shift_sec": SPLIT_SHIFT_SEC,
            "cuts": [{"time": c.time, "kind": c.kind} for c in cuts],
            "raw_shot_count": len(raw_shots),
            "shots": shots,
            "segments": [s.to_dict() for s in segments],
            "boundaries": boundaries,
        }
        _write_json(layout.segments_path(WORK_DIR, video.id), payload)
        n_dissolve = sum(c.kind == "dissolve" for c in cuts)
        print(
            f"{video.id} {video.file}: {len(cuts)} cuts ({n_dissolve} dissolve) -> "
            f"{len(shots)} shots, {len(segments)} segments "
            f"({elapsed:.1f}s for {analysed:.0f}s video)"
        )
        for kind in ("split", "shot"):
            here = [b for b in boundaries if b["kind"] == kind]
            if here:
                cut = sum(b["speech"] == "mid_clause" for b in here)
                clause = sum(b["speech"] == "clause_break" for b in here)
                print(
                    f"  {kind}: {len(here)} boundaries, {clause} between clauses, "
                    f"{cut} inside a clause"
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
            _picture(video),
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


def _ordered(videos: list[layout.Video]) -> tuple[list[layout.Video], str]:
    """Videos in shooting order when every one carries its capture time,
    else in the manifest's (file name) order; and which order it is. One
    missing time keeps the file order rather than guess where it goes."""
    times = [capture_time(RAW_DIR / v.file) for v in videos]
    if all(t is not None for t in times):
        order = sorted(range(len(videos)), key=lambda k: (times[k], k))
        return [videos[k] for k in order], "拍摄时间"
    return videos, "文件名"


def _out(args: argparse.Namespace) -> Path:
    """``--out``, else out/<collection>."""
    if args.out is not None:
        return args.out
    return OUT_DIR / args.collection


def _copy_tags(video: layout.Video) -> None:
    """Put the video's tags in its default result folder under its name."""
    out = OUT_DIR / video.collection if video.collection else OUT_DIR
    dirs = layout.result_dirs(
        out, [v for v in _videos() if v.collection == video.collection]
    )
    path = layout.tags_copy_path(dirs[video.id])
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(layout.tags_path(WORK_DIR, video.id), path)


def _write_index() -> None:
    print(f"Wrote {report.write_index(OUT_DIR)}")


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


def _tag_stage(
    videos: list[layout.Video], limit: int, profiles: dict[str, category.Profile]
) -> runner.StageFn:
    """Tag each video's remaining segments with its category's prompt
    (``profiles``, from _prepare_tagging)."""

    def run(advance: Callable[[str], None]) -> None:
        for video in videos:
            if video.id not in profiles:
                continue
            profile = profiles[video.id]
            sig = _load_signals(video.id)
            out_path = layout.tags_path(WORK_DIR, video.id)
            keyframes = layout.keyframes_dir(WORK_DIR, video.id)
            for seg in _todo(video.id, layout.tags_path, limit):
                frames = [keyframes / k for k in sig[seg.index]["keyframes"]]
                try:
                    tags, raw, elapsed, errors, score = tag.tag_segment(
                        frames, seg.duration, profile=profile
                    )
                except Exception as exc:
                    raise runner.StageError(
                        "打标签", video.file, seg.index, exc
                    ) from exc
                runner.append_record(
                    out_path,
                    {
                        "index": seg.index,
                        "category": profile.key,
                        "prompt": profile.digest,
                        "seconds": round(elapsed, 2),
                        "tags": tags,
                        **score,
                        "raw": raw,
                        "failed_attempts": errors,
                    },
                )
                retry = f" (retries {len(errors)})" if errors else ""
                advance(
                    f"{video.file} seg{seg.index:03d} {elapsed:5.1f}s{retry} "
                    f"h={score['highlight_p']:.2f} {tags['description']}"
                )
            if out_path.exists():
                _copy_tags(video)

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
    counts = count_frames(_picture(video), spans, end)
    picks = [adapter.needed_frames(n) if n else set() for n in counts]
    for i, frames, seen in segment_frames_picked(
        _picture(video), spans, picks, DOVER_MAX_SIDE, end
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
    videos = _selected(args)
    if args.rescore:
        _rescore(videos)
        return
    profiles = _prepare_tagging(videos, args.category, args.retag)
    _tag_stage(videos, args.limit, profiles)(_print_line)
    _write_index()


def _rescore(videos: list[layout.Video]) -> None:
    """Add the probability-weighted highlight score to tag rows made
    before it was recorded (tag.rescore), rewriting each video's tags file
    after every row so an interrupted run resumes where it stopped."""
    for video in videos:
        path = layout.tags_path(WORK_DIR, video.id)
        if not path.exists():
            continue
        rows = _tag_rows(video.id)
        todo = [k for k, r in enumerate(rows) if "highlight_p" not in r]
        if not todo:
            continue
        _, segments = _load_segments(video.id)
        durations = {s.index: s.duration for s in segments}
        sig = _load_signals(video.id)
        keyframes = layout.keyframes_dir(WORK_DIR, video.id)
        for n, k in enumerate(todo, start=1):
            row = rows[k]
            key = row.get("category")
            profile = category.load_profile(key) if key else None
            frames = [keyframes / f for f in sig[row["index"]]["keyframes"]]
            try:
                score, seconds = tag.rescore(
                    row, frames, durations[row["index"]], profile
                )
            except Exception as exc:
                raise runner.StageError(
                    "补精彩度", video.file, row["index"], exc
                ) from exc
            rows[k] = {**row, **score}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )
            tmp.replace(path)
            print(
                f"{video.file} {n}/{len(todo)} seg{row['index']:03d} "
                f"{seconds:4.1f}s h={row['tags']['highlight']} → "
                f"{score['highlight_p']:.2f}",
                flush=True,
            )
        _copy_tags(video)


def _prepare_tagging(
    videos: list[layout.Video], choice: str, retag: bool
) -> dict[str, category.Profile]:
    """The category profile of each video that has segments to tag (all of
    them with ``retag``). Tags made with another category or prompt are an
    error unless ``retag``: then they are moved aside, kept for comparison,
    and the video is tagged again from the start."""
    profiles: dict[str, category.Profile] = {}
    for video in videos:
        path = layout.tags_path(WORK_DIR, video.id)
        if not retag and not _todo(video.id, layout.tags_path, 0):
            continue
        profile = _profile(video, choice)
        profiles[video.id] = profile
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            first = json.loads(f.readline())
        made = (first.get("category"), first.get("prompt"))
        if made == (profile.key, profile.digest):
            continue
        label = f"{made[0]}-{made[1]}" if made[0] else "generic"
        if not retag:
            raise SystemExit(
                f"{video.file}：已有标签是按 {label} 提示词打的，与「{profile.name}」"
                f"当前的提示词不一致；加 --retag 重新打（旧标签改名保留）"
            )
        old = layout.retired_tags_path(WORK_DIR, video.id, label)
        if old.exists():
            raise SystemExit(f"{old} 已存在，先处理它再重打")
        path.replace(old)
        print(f"{video.file}：旧标签移到 {old.name}，按「{profile.name}」重新打")
    return profiles


def _title_card(title: str, videos: list[layout.Video]) -> tuple[str, str]:
    """The title card of an edit: the outline's title up to its first colon
    (the rest is the model's tagline, often longer than a card holds), and
    the material's category under it."""
    head = re.split("[：:]", title, maxsplit=1)[0].strip() or title
    profile = _main_profile(videos)
    return head, profile.name if profile else ""


def _main_profile(videos: list[layout.Video]) -> category.Profile | None:
    """The category covering most of the analysed footage among videos that
    have one; None when none has been categorised."""
    seconds: dict[str, float] = {}
    for v in videos:
        path = layout.category_path(WORK_DIR, v.id)
        if path.exists():
            key = json.loads(path.read_text("utf-8"))["category"]
            length = _load_segments(v.id)[0]["analysed_until"]
            seconds[key] = seconds.get(key, 0.0) + length
    if not seconds:
        return None
    return category.load_profile(max(seconds, key=lambda k: (seconds[k], k)))


def _profile(video: layout.Video, choice: str = "") -> category.Profile:
    """The video's category: ``choice`` (a key) when given, else the cached
    one, else the 4B picks it now from keyframes spread over the video."""
    profiles = category.load_profiles()
    path = layout.category_path(WORK_DIR, video.id)
    if choice:
        if not path.exists() or json.loads(path.read_text("utf-8"))["category"] != (
            choice
        ):
            _write_json(
                path, {"video_id": video.id, "category": choice, "source": "user"}
            )
        return profiles[choice]
    if path.exists():
        return profiles[json.loads(path.read_text("utf-8"))["category"]]
    sig = _load_signals(video.id)
    keyframes = layout.keyframes_dir(WORK_DIR, video.id)
    n = min(category.CLASSIFY_FRAMES, len(sig))
    picks = sorted({round(k * (len(sig) - 1) / max(n - 1, 1)) for k in range(n)})
    frames = [keyframes / sig[i]["keyframes"][0] for i in picks]
    minutes = _load_segments(video.id)[0]["analysed_until"] / 60
    tag.check_server()
    try:
        choice_, raw, seconds, errors = category.classify(frames, minutes, profiles)
    except category.CategoryError as exc:
        raise SystemExit(f"{video.file}：判断素材类型失败：{exc}") from exc
    _write_json(
        path,
        {
            "video_id": video.id,
            **choice_,
            "source": "model",
            "raw": raw,
            "seconds": round(seconds, 1),
            "failed_attempts": errors,
        },
    )
    profile = profiles[choice_["category"]]
    print(f"{video.file}：素材类型 {profile.name}（{choice_['reason']}）")
    return profile


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
        _picture(video), spans, uvq.STEP_SECONDS, uvq.SIZE, END_TIME.get(video.id)
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
    if not tag_todo and not q_todo and not args.retag:
        print(f"全部片段已完成打标签和美学打分，无需重跑。{NEXT_STEPS}")
        return
    problems = missing_inputs(
        videos, need_tag=tag_todo > 0 or args.retag, need_quality=q_todo > 0
    )
    if problems:
        raise SystemExit("开工检查未通过：\n- " + "\n- ".join(problems))
    # Retagging moves old tags aside, so count again afterwards.
    profiles = _prepare_tagging(videos, args.category, args.retag)
    tag_done, tag_todo = _counts(videos, layout.tags_path, args.limit)

    stages: list[runner.Stage] = []
    if tag_todo:
        stages.append(
            runner.Stage(
                "打标签",
                tag_done + tag_todo,
                tag_done,
                _tag_stage(videos, args.limit, profiles),
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


def _candidates(video: layout.Video) -> list[suggest.Candidate]:
    _, segments = _load_segments(video.id)
    sig = _load_signals(video.id)
    tags_path = layout.tags_path(WORK_DIR, video.id)
    rows = {
        r["index"]: r
        for r in map(json.loads, tags_path.read_text("utf-8").splitlines())
    }
    tags = {i: r["tags"] for i, r in rows.items()}
    missing = [s.index for s in segments if s.index not in tags]
    if missing:
        raise RuntimeError(
            f"{video.file}: {len(missing)} segments not tagged: {missing[:5]}"
        )
    unscored = [i for i, r in rows.items() if "highlight_p" not in r]
    if unscored:
        raise RuntimeError(
            f"{video.file}: {len(unscored)} segments have no probability-weighted "
            f"highlight score; run `tag --rescore`"
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
            highlight=rows[s.index]["highlight_p"],
            flags=sig[s.index]["flags"],
            sharpness=sig[s.index]["sharpness"],
            sharpness_rel=sig[s.index]["sharpness"] / median,
            quality=quality[s.index]["fused"],
            tech_raw=quality[s.index]["technical"],
            aes_raw=quality[s.index]["aesthetic"],
            category=rows[s.index].get("category"),
            brightness=sig[s.index]["brightness"],
        )
        for s in segments
    ]


def _video_summaries(
    vid: str, cands: list[suggest.Candidate]
) -> list[tuple[list[int], dict]]:
    """(chunk members as indices into ``cands``, summary) for one video.

    ``cands`` are all of the video's segments in order. Summaries are cached
    in summary_<ID>.json and reused while the chunk's segments, the text the
    model reads (tags and scores, by hash), CHUNK_SEC and SUMMARY_VERSION
    are unchanged; new ones are written chunk by chunk, so an interrupted
    run resumes.
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
        digest = _digest(outline.segment_lines(members, cal, assessments))
        entry = cached.get(key)
        if entry is None or entry.get("input") != digest:
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
                "input": digest,
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


def _video_spans(cands: list[suggest.Candidate]) -> list[tuple[str, int, int]]:
    """(video, first, end) of each video's run of candidates, in order."""
    spans: list[tuple[str, int, int]] = []
    for i, c in enumerate(cands):
        if spans and spans[-1][0] == c.video:
            spans[-1] = (c.video, spans[-1][1], i + 1)
        else:
            spans.append((c.video, i, i + 1))
    return spans


def _digest(text: str) -> str:
    """Short hash of a model input, to tell when a cached answer is stale."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: layout.ID_LENGTH]


def _group_summary(videos: list[tuple[str, float, dict]]) -> dict:
    """Summary of a group of short videos, each (id, length, its summary).

    Cached in group_<key>.json, the key a hash of the videos' summaries, so
    a group is summarised again when any of its videos is.
    """
    key = _digest(json.dumps([outline.SUMMARY_VERSION, videos], ensure_ascii=False))
    path = layout.group_summary_path(WORK_DIR, key)
    if path.exists():
        return json.loads(path.read_text("utf-8"))["summary"]
    ids = "、".join(v for v, _, _ in videos)
    try:
        summary, raw, seconds, errors = outline.summarize_group(
            [(length, s) for _, length, s in videos]
        )
    except outline.OutlineError as exc:
        raise SystemExit(f"视频组（{ids}）总结失败：{exc}") from exc
    _write_json(
        path,
        {
            "version": outline.SUMMARY_VERSION,
            "videos": [v for v, _, _ in videos],
            "summary": summary,
            "raw": raw,
            "seconds": round(seconds, 1),
            "failed_attempts": errors,
        },
    )
    print(f"视频组 {ids} 总结 {seconds:.0f}s：{summary['summary'][:40]}", flush=True)
    return summary


def _chunks(
    cands: list[suggest.Candidate], offsets: dict[str, float]
) -> list[outline.Chunk]:
    """Chunks along the timeline: a long video's own, and groups of whole
    short videos (outline.group_videos) summarised from the videos'
    summaries."""
    spans = _video_spans(cands)
    summaries = [_video_summaries(vid, cands[a:b]) for vid, a, b in spans]
    lengths = [cands[b - 1].end for _, _, b in spans]
    chunks: list[outline.Chunk] = []

    def add(videos: list[str], members: list[int], summary: dict) -> None:
        first, last = cands[members[0]], cands[members[-1]]
        chunks.append(
            outline.Chunk(
                number=len(chunks) + 1,
                videos=videos,
                members=members,
                start=offsets.get(first.video, 0.0) + first.start,
                end=offsets.get(last.video, 0.0) + last.end,
                summary=summary,
            )
        )

    for group in outline.group_videos(lengths):
        if len(group) == 1:
            vid, a, _ = spans[group[0]]
            for members, summary in summaries[group[0]]:
                add([vid], [a + i for i in members], summary)
            continue
        # A short video is a single chunk of its own summaries.
        whole = [summaries[k][0] for k in group]
        summary = _group_summary(
            [
                (spans[k][0], lengths[k], s)
                for k, (_, s) in zip(group, whole, strict=True)
            ]
        )
        add(
            [spans[k][0] for k in group],
            [
                spans[k][1] + i
                for k, (ms, _) in zip(group, whole, strict=True)
                for i in ms
            ],
            {
                **summary,
                "highlights": outline.group_highlights(
                    [(len(ms), s) for ms, s in whole]
                ),
            },
        )
    return chunks


Vectors = dict[tuple[str, int], np.ndarray]


def _scene_groups(
    chunk_members: list[list[int]],
    cands: list[suggest.Candidate],
    text: Vectors,
    image: Vectors,
    offsets: dict[str, float] | None = None,
) -> list[list[int]]:
    """Candidate indices of each scene along the timeline; a scene never
    crosses a chunk (``chunk_members``). Shot cuts and video boundaries
    inside a chunk are joinable where the content either side is similar
    (retrieve.FRAGMENT_JOIN_SIM). ``offsets`` place each video on the
    chunk's timeline (needed when a chunk spans videos)."""
    offsets = offsets or {}
    out: list[list[int]] = []
    for ms in chunk_members:
        keys = [(cands[i].video, cands[i].index) for i in ms]
        sim = retrieve.boundary_similarity(
            np.stack([text[k] for k in keys]), np.stack([image[k] for k in keys])
        )
        cut = [
            p
            for p in range(len(ms) - 1)
            if keys[p][0] != keys[p + 1][0]
            or cands[ms[p]].shot != cands[ms[p + 1]].shot
        ]
        soft = {p for p in cut if sim[p] >= retrieve.FRAGMENT_JOIN_SIM}
        at = [offsets.get(cands[i].video, 0.0) for i in ms]
        for g in retrieve.split_scenes(
            [a + cands[i].start for a, i in zip(at, ms, strict=True)],
            [a + cands[i].end for a, i in zip(at, ms, strict=True)],
            sim,
            set(cut) - soft,
            soft,
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
        self,
        chunks: list[outline.Chunk],
        cands: list[suggest.Candidate],
        offsets: dict[str, float],
    ) -> list[outline.Scene]:
        """Scenes along the timeline; a scene never crosses a chunk."""
        groups = _scene_groups(
            [ch.members for ch in chunks], cands, self.text, self.image, offsets
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
            "videos": ch.videos,
            "start": round(ch.start, 2),
            "end": round(ch.end, 2),
            "segments": [[cands[i].video, cands[i].index] for i in ch.members],
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
    (empty: within their own videos; a scene may run from one video into
    the next, ``video`` is where it starts)."""
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
            "segments": [[cands[i].video, cands[i].index] for i in sc.members],
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
        scenes = retrieval.scenes(chunks, cands, offsets)
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
            raise SystemExit(f"{exc}\n    启动命令：{LLAMA_SERVER_CMD}") from exc
    retrieval = _Retrieval(_selected(args)) if args.mode == "highlights" else None
    every: list[suggest.Candidate] = []
    offsets: dict[str, float] = {}
    total = 0.0
    videos, order = _ordered(_selected(args))
    for video in videos:
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
        summary = {
            **_highlights_payload(
                name, every, args.ratio * total, offsets, args.mode, retrieval
            ),
            "order": order,
        }
    _write_json(layout.timeline_path(out, args.mode), summary)
    print(f"{name}: {len(summary['clips'])} clips {summary['total_sec']}s")
    if retrieval:
        retrieval.embedder.close()


def _scene_catalog(
    scenes: list[outline.Scene],
    cands: list[suggest.Candidate],
    offsets: dict[str, float],
) -> list[dict]:
    """Number, start, end (on the timeline ``offsets`` sets), the videos it
    is in (1-based, in timeline order), title and summary of every scene
    that keeps footage.

    ``cands`` are not calibrated: each video is calibrated and assessed on
    its own, as ``scenes`` does, so the summaries cached by ``scenes`` are
    reused and missing ones are written to the same cache. Scenes that run
    across videos are cached in scene_multi.json.
    """
    spans = _video_spans(cands)
    cal: list[suggest.Candidate] = []
    for _, a, b in spans:
        cal += suggest.calibrate_by_time(cands[a:b])
    assessments = [suggest.assess(c) for c in cal]
    summaries: dict[int, dict[str, str]] = {}
    for vid, a, b in spans:
        own = [s for s in scenes if a <= s.members[0] and s.members[-1] < b]
        got = _scene_summaries(
            vid, [[i - a for i in s.members] for s in own], cal[a:b], assessments[a:b]
        )
        summaries.update({own[n - 1].number: x for n, x in got.items()})
    multi = [s for s in scenes if cal[s.members[0]].video != cal[s.members[-1]].video]
    if multi:
        got = _cached_scene_summaries(
            layout.multi_scene_path(WORK_DIR),
            "跨视频",
            [s.members for s in multi],
            cal,
            assessments,
            lambda c: f"{c.video}:{c.index}",
        )
        summaries.update({multi[n - 1].number: x for n, x in got.items()})
    order = {vid: k for k, (vid, _, _) in enumerate(spans, start=1)}
    catalog: list[dict] = []
    for sc in scenes:
        if sc.number not in summaries:
            continue
        first, last = cands[sc.members[0]], cands[sc.members[-1]]
        catalog.append(
            {
                "number": sc.number,
                "start": round(offsets.get(first.video, 0.0) + first.start, 2),
                "end": round(offsets.get(last.video, 0.0) + last.end, 2),
                "videos": sorted({order[cands[i].video] for i in sc.members}),
                **summaries[sc.number],
            }
        )
    return catalog


def _collection_candidates(
    videos: list[layout.Video],
) -> tuple[list[suggest.Candidate], dict[str, float], float]:
    """Candidates of ``videos`` one after another, each video's start on
    that timeline, and the timeline's length."""
    cands: list[suggest.Candidate] = []
    offsets: dict[str, float] = {}
    total = 0.0
    for video in videos:
        cands += _candidates(video)
        offsets[video.id] = total
        total += _load_segments(video.id)[0]["analysed_until"]
    return cands, offsets, total


def cmd_catalog(args: argparse.Namespace) -> None:
    """Summarise the chunks and scenes ``edit`` reads, ahead of it: they
    depend on the footage only, so they belong to preparing it, and edit
    then finds them cached."""
    try:
        tag.check_server(outline.SERVER_URL)
    except ConnectionError as exc:
        raise SystemExit(f"{exc}\n    启动命令：{LLAMA_SERVER_CMD}") from exc
    videos, _ = _ordered(_selected(args))
    cands, offsets, _ = _collection_candidates(videos)
    chunks = _chunks(cands, offsets)
    retrieval = _Retrieval(videos)
    try:
        scenes = retrieval.scenes(chunks, suggest.calibrate_by_time(cands), offsets)
        catalog = _scene_catalog(scenes, cands, offsets)
    finally:
        retrieval.embedder.close()
    print(f"{len(chunks)} 块、{len(catalog)} 个场景的概况已缓存")


def cmd_edit(args: argparse.Namespace) -> Path:
    """A cut for the user's instruction across the selected videos: the
    outline model reads the instruction and the scene catalog, every scene
    is searched for its slots. Timeline, preview and report go to a folder
    of their own under out/<collection>/edits/, which is returned."""
    instruction = args.instruction.strip()
    if not instruction:
        raise SystemExit("剪辑要求不能为空")
    out = _out(args)
    try:
        tag.check_server(outline.SERVER_URL)
    except ConnectionError as exc:
        raise SystemExit(f"{exc}\n    启动命令：{LLAMA_SERVER_CMD}") from exc
    videos, order = _ordered(_selected(args))
    cands, offsets, total = _collection_candidates(videos)
    chunks = _chunks(cands, offsets)
    if args.target is not None:
        # The request gives the length in seconds: nothing to read.
        if args.target <= 0:
            raise SystemExit(f"--target 必须大于 0：{args.target}")
        duration = {"target_sec": args.target, "quote": ""}
        duration_raw, duration_errors = "", []
    else:
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
    if args.target is not None:
        target = float(args.target)
        source = {"kind": "request"}
    elif duration["target_sec"]:
        target = float(duration["target_sec"])
        source = {"kind": "instruction", "quote": duration["quote"]}
    else:
        target = args.ratio * total
        source = {"kind": "ratio", "ratio": args.ratio}
    if target > usable:
        raise SystemExit(f"要求成片 {target:.0f}s，可用素材只有 {usable:.0f}s")
    retrieval = _Retrieval(videos)
    try:
        scenes = retrieval.scenes(chunks, cands, offsets)
        catalog = _scene_catalog(scenes, raw, offsets)
        shown = catalog
        if len(catalog) > instruct.MAX_CATALOG_SCENES:
            listed = {s["number"] for s in catalog}
            relevance = retrieval.rank(
                [instruction], [s for s in scenes if s.number in listed], cands
            )[0][0]
            shown = instruct.shortlist(catalog, relevance)
        try:
            plan, outline_raw, seconds, errors = instruct.plan_request(
                instruction, shown, total, target, profile=_main_profile(videos)
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
    speech = _speech_spans(videos)
    clips, notes, slots = instruct.fill_request(
        plan,
        scenes,
        cands,
        assessments,
        [r[0] for r in ranked],
        [r[1] for r in ranked],
        named,
        speech,
    )
    if not clips:
        raise SystemExit(f"没有选到任何片段：{'；'.join(notes)}")
    reviewed: dict | None = None
    if args.review:
        filled = clips
        try:
            clips, review, review_raw, review_seconds, review_errors = (
                instruct.review_request(
                    instruction,
                    target,
                    plan,
                    clips,
                    cands,
                    assessments,
                    [r[1] for r in ranked],
                    _speech_lines(videos),
                    profile=_main_profile(videos),
                    beat_grid=_beat_grid(videos),
                )
            )
        except outline.OutlineError as exc:
            raise SystemExit(f"审片失败（3 次都没通过校验）：{exc}") from exc
        reviewed = {
            "notes": review["notes"],
            "fit": review["fit"],
            "raw": review_raw,
            "seconds": round(review_seconds, 1),
            "failed_attempts": review_errors,
            "filled": [suggest.clip_to_dict(c) for c in filled],
        }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = layout.edit_dir(out, stamp, instruction)
    timeline = {
        "name": "edit",
        "mode": "edit",
        "instruction": instruction,
        "created": stamp,
        "videos": [v.id for v in videos],
        "order": order,
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
        "planner": outline.planner_name(),
        "outline_raw": outline_raw,
        "outline_seconds": round(seconds, 1),
        "outline_failed_attempts": errors,
        # With --review: the planner's review and the cut it replaced.
        "review": reviewed,
        "chunks": _chunk_rows(chunks, cands),
        # The scenes the model read; catalog_total counts all that keep
        # footage (more than MAX_CATALOG_SCENES are shortlisted).
        "catalog": shown,
        "catalog_total": len(catalog),
        # Within their own video: the cut is not in shooting order.
        "scenes": _scene_rows(scenes, cands, {}),
    }
    print(
        f"{plan['title']}：{len(clips)} clips {timeline['total_sec']}s / "
        f"目标 {target:.0f}s {notes}"
    )
    voices: list[tuple[float, np.ndarray]] = []
    if args.narrate:
        timeline["narration"], voices = _narrate(timeline, videos, args.voice)
    _write_json(layout.edit_timeline_path(folder), timeline)
    # A caller that renders the timeline itself skips the preview.
    if getattr(args, "preview", True):
        _write_preview(
            timeline,
            {v.id: v.file for v in videos},
            layout.edit_preview_path(folder),
            voices,
            _title_card(plan["title"], videos),
        )
    print(f"Wrote {report.write_edit(ROOT, folder, videos, timeline)}")
    _write_index()
    return folder


def cmd_render(args: argparse.Namespace) -> None:
    out = _out(args)
    dirs = _result_dirs(args, out)
    videos = _selected(args)
    files = {v.id: v.file for v in videos}
    for folder in [dirs[v.id] for v in videos] + [out]:
        path = layout.timeline_path(folder, args.mode)
        timeline = json.loads(path.read_text("utf-8"))
        _write_preview(timeline, files, layout.preview_path(folder, args.mode))


def _speech_spans(videos: list[layout.Video]) -> dict[str, list[tuple[float, float]]]:
    """Sentence spans of each video whose speech has been recognised."""
    out = {}
    for v in videos:
        path = layout.asr_path(WORK_DIR, v.id)
        if path.exists():
            out[v.id] = subtitle.speech_spans(subtitle.read_asr(path, v.id))[0]
    return out


def _beat_grid(
    videos: list[layout.Video],
) -> Callable[[str, float, float], tuple[float, list[float]]]:
    """Beats of a video's own sound between two times (beats.find_beats):
    (period, times in the video's seconds)."""
    files = {v.id: _raw(v) for v in videos}

    def grid(vid: str, lo: float, hi: float) -> tuple[float, list[float]]:
        sound = read_audio_clip(files[vid], lo, hi - lo, fade_in=False, fade_out=False)
        period, times = beats.find_beats(sound.mean(axis=0), AUDIO_RATE)
        return period, [lo + t for t in times]

    return grid


def _speech_lines(
    videos: list[layout.Video],
) -> dict[str, list[tuple[float, float, str]]]:
    """(start, end, text) of each sentence said in each video: the refined
    subtitles where they exist, else what was recognised."""
    out = {}
    for v in videos:
        refined = layout.subtitle_path(WORK_DIR, v.id)
        if refined.exists():
            data = subtitle.read_subtitles(refined, v.id)
            sentences = subtitle.sentences_from_json(data["sentences"])
        elif layout.asr_path(WORK_DIR, v.id).exists():
            sentences = subtitle.read_asr(layout.asr_path(WORK_DIR, v.id), v.id)
        else:
            continue
        out[v.id] = [(s.start, s.end, s.text) for s in sentences if s.times]
    return out


def _played(clips: list[dict]) -> list[dict]:
    """``clips`` as the preview plays them: whole frames, so the preview's
    clock (its subtitles, its narration) follows frame counts, not the
    timeline's exact seconds."""
    return [
        {**clip, "duration": round(clip["duration"] * OUT_FPS) / OUT_FPS}
        for clip in clips
    ]


# The title card of a finished cut stays this long (from just after the
# fade in).
TITLE_SEC = (0.4, 3.6)


def _write_preview(
    timeline: dict,
    files: dict[str, str],
    target: Path,
    voices: list[tuple[float, np.ndarray]] | None = None,
    title: tuple[str, str] | None = None,
    finished: bool | None = None,
    burn: bool = True,
) -> None:
    """The timeline's clips joined into one preview video with their sound;
    ``files`` maps video ids to raw file names; ``voices`` are narration
    lines (start, stereo samples) mixed over the sound. Subtitles of what is
    said, and the narration's, go to the subtitles folder next to it, with a
    second preview there that has them drawn into the picture (unless
    ``burn`` is False).

    A ``finished`` cut is made for showing: EDIT_SIZE (upright when portrait
    sources fill more of it, media.edit_size), a dissolve where the outline
    moves to its next part, and fades from and to black at both ends; it is
    finished by default when it has a ``title`` (a title and a line under
    it), which is drawn as a title card."""
    finished = title is not None if finished is None else finished
    played = _played(timeline["clips"])
    if finished:
        shapes = {v: probe(RAW_DIR / f) for v, f in files.items()}
        size = edit_size(
            [
                (shapes[c["video"]].width, shapes[c["video"]].height, c["duration"])
                for c in played
            ]
        )
    else:
        size = OUT_SIZE
    cues = _preview_subtitles({**timeline, "clips": played}, target) or []
    said = [
        (line["start"], line["start"] + line["seconds"], line["text"])
        for line in timeline.get("narration", {}).get("lines", [])
    ]
    if said:
        _write_text(
            layout.narration_srt_path(target),
            subtitle.to_srt([subtitle.Cue(*c) for c in said]),
        )
        _write_text(
            layout.narration_text_path(target),
            _narration_text(timeline["narration"], played, files),
        )
    card = (
        TitlePainter(*title, _subtitle_font(), *TITLE_SEC, frame=size)
        if title
        else None
    )
    outputs: list[tuple[Path, FrameDraw | None]] = [(target, card)]
    if burn and (cues or said):
        shown = sorted([(c.start, c.end, c.text) for c in cues] + said)
        painter = SubtitlePainter(shown, _subtitle_font(), frame=size)
        outputs.append((layout.burned_preview_path(target), compose(card, painter)))

    # HDR (PQ) sources are shown as SDR in a finished cut.
    pq = {v: finished and is_pq(RAW_DIR / f) for v, f in files.items()}

    def clips() -> Iterator[tuple[Iterator[np.ndarray], np.ndarray]]:
        # What the clip before runs on with past its end: its picture where
        # the outline moves to its next part (dissolve), and its sound at
        # every cut of a finished cut (crossfade).
        tail: list[np.ndarray] | None = None
        carry: np.ndarray | None = None
        for k, clip in enumerate(played):
            path = RAW_DIR / files[clip["video"]]
            start, duration = clip["src_start"], clip["duration"]
            nxt = played[k + 1] if k + 1 < len(played) else None
            dissolving = finished and nxt is not None and nxt["role"] != clip["role"]
            # The handle runs on past the clip's end, as far as the source goes.
            wanted = DISSOLVE_SEC if dissolving else AUDIO_XFADE_SEC
            handle = (
                min(wanted, probe(path).duration - start - duration)
                if finished and nxt is not None
                else 0.0
            )
            if handle < 2 / OUT_FPS:
                handle, dissolving = 0.0, False
            n = round(duration * AUDIO_RATE)
            heard = read_audio_clip(
                path,
                start,
                duration + handle,
                fade_in=carry is None,
                fade_out=handle == 0.0,
            )
            sound, next_carry = heard[:, :n], heard[:, n:]
            # Read as encoded: one frame at a time in memory, not the clip.
            frames = iter_clip(path, start, duration, size, pq[clip["video"]])
            if tail is not None and carry is not None:
                frames, sound = dissolve((tail, carry), (frames, sound))
            elif carry is not None:
                sound = crossfade(carry, sound)
            tail = (
                read_clip(path, start + duration, handle, size, pq[clip["video"]])
                if dissolving
                else None
            )
            carry = next_carry if handle else None
            yield frames, sound

    t0 = time.perf_counter()
    total = sum(c["duration"] for c in played) if finished else None
    write_av(
        clips(), outputs, voices, size, total, EDIT_CRF if finished else PREVIEW_CRF
    )
    print(f"{target}: {time.perf_counter() - t0:.1f}s")


def _narration_text(record: dict, played: list[dict], files: dict[str, str]) -> str:
    """The narration as a table to read: each line's time on the cut, its
    clip and where that is in the source video, and the words."""
    starts = [0.0]
    for clip in played:
        starts.append(starts[-1] + clip["duration"])
    rows = [
        f"解说词：{len(record['lines'])} 句，音色 {record['voice']}，"
        f"成片 {starts[-1]:.1f} 秒",
        "",
        "序号\t成片时间\t镜头\t原片\t原片时间\t解说",
    ]
    for k, line in enumerate(record["lines"], start=1):
        clip = played[line["clip"] - 1]
        src = clip["src_start"] + line["start"] - starts[line["clip"] - 1]
        end = line["start"] + line["seconds"]
        rows.append(
            f"{k}\t{_clock(line['start'])}–{_clock(end)}\t{line['clip']}\t"
            f"{files[clip['video']]}\t{_clock(src)}\t{line['text']}"
        )
    return "\n".join(rows) + "\n"


def _clock(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def _narrate(
    timeline: dict, videos: list[layout.Video], voice: str
) -> tuple[dict, list[tuple[float, np.ndarray]]]:
    """Narration for an edit: the 4B writes lines for the chosen clips in
    the gaps between speech, Kokoro voices them. Returns the record kept in
    the timeline and the lines' audio (start, stereo samples)."""
    profile = _main_profile(videos)
    if profile is None:
        raise SystemExit("解说要按素材类型写：先运行 tag 或 analyze（可加 --category）")
    used = {c["video"] for c in timeline["clips"]}
    unheard = [
        v.file
        for v in videos
        if v.id in used
        and _load_segments(v.id)[0]["has_audio"]
        and not layout.asr_path(WORK_DIR, v.id).exists()
    ]
    if unheard:
        raise SystemExit(
            f"这些视频有声音但还没做语音识别，解说会压到对白：{unheard}；"
            "先运行 subtitle"
        )
    played = _played(timeline["clips"])
    spoken = narrate.speech_on_timeline(played, _speech_spans(videos))
    slots = narrate.free_slots(played, spoken)
    tags = {
        (v.id, r["index"]): r["tags"]
        for v in videos
        if v.id in used
        for r in _tag_rows(v.id)
    }
    frames = {
        (v.id, row["index"]): [
            layout.keyframes_dir(WORK_DIR, v.id) / k for k in row["keyframes"]
        ]
        for v in videos
        if v.id in used
        for row in _load_signals(v.id)
    }
    try:
        lines, raw, seconds, errors = narrate.write_lines(
            played, tags, slots, profile, frames=frames
        )
    except outline.OutlineError as exc:
        raise SystemExit(f"解说词生成失败（3 次都没通过校验）：{exc}") from exc
    try:
        lines, checks, check_seconds = narrate.check_lines(lines, played, tags, frames)
    except outline.OutlineError as exc:
        raise SystemExit(f"解说词核对失败（3 次都没通过校验）：{exc}") from exc
    for c in checks:
        if c["checked"] != c["written"]:
            print(f"  核对 镜头 {c['clip']}：{c['written']} → {c['checked']}")
    try:
        speaker = tts.KokoroAdapter(MODELS_DIR, voice)
        t0 = time.perf_counter()
        voiced = narrate.voice_lines(lines, speaker.synthesize, speaker.rate)
    except (tts.TTSError, outline.OutlineError) as exc:
        raise SystemExit(f"解说配音失败：{exc}") from exc
    tts_seconds = time.perf_counter() - t0
    record = {
        "category": profile.key,
        "voice": voice,
        "tts_model": tts.KOKORO_REPO,
        "write_seconds": round(seconds, 1),
        "check_seconds": round(check_seconds, 1),
        "checks": checks,
        "tts_seconds": round(tts_seconds, 1),
        "raw": raw,
        "failed_attempts": errors,
        "duck_gain": DUCK_GAIN,
        "lines": [line for line, _ in voiced],
    }
    for line, _ in voiced:
        added = "（补空档）" if line.get("filled") else ""
        print(
            f"  解说 {line['start']:5.1f}s 镜头 {line['clip']}：{line['text']}{added}"
        )
    return record, [(line["start"], voice_track(v, speaker.rate)) for line, v in voiced]


def _subtitle_font() -> Path:
    """The font burned subtitles are drawn with: VIDEO_DEMO_FONT, else the
    platform's usual CJK font. Missing is an error, not a silent swap."""
    font = Path(os.environ.get("VIDEO_DEMO_FONT", SUBTITLE_FONT))
    if not font.exists():
        raise SystemExit(
            f"烧录字幕的字体 {font} 不存在；装上中文字体，或用 VIDEO_DEMO_FONT 指定"
        )
    return font


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

    Cached in scene_<ID>.json by the kept segments and SCENES_VERSION.
    """
    return _cached_scene_summaries(
        layout.scene_summary_path(WORK_DIR, vid),
        vid,
        groups,
        cands,
        assessments,
        lambda c: c.index,
    )


def _cached_scene_summaries(
    path: Path,
    label: str,
    groups: list[list[int]],
    cands: list[suggest.Candidate],
    assessments: list[suggest.Assessment],
    key_of: Callable[[suggest.Candidate], int | str],
) -> dict[int, dict[str, str]]:
    """Scene summaries cached in ``path``, keyed by ``key_of`` of each kept
    segment and checked against the text the model reads (by hash), so new
    tags or scores are summarised again. Saved after every new answer, so
    an interrupted run resumes; entries other runs wrote are kept (``edit``
    and ``scenes`` share it).
    """
    cached: dict[tuple, dict] = {}
    if path.exists():
        data = json.loads(path.read_text("utf-8"))
        if data["version"] == scene_material.SCENES_VERSION:
            cached = {tuple(e["segments"]): e for e in data["scenes"]}

    def save() -> None:
        _write_json(
            path,
            {
                "version": scene_material.SCENES_VERSION,
                "scenes": list(cached.values()),
            },
        )

    out: dict[int, dict[str, str]] = {}
    for n, members in enumerate(groups, start=1):
        runs = scene_material.kept_runs(members, cands, assessments)
        kept = [i for r in runs for i in r]
        if not kept:
            continue
        key = tuple(key_of(cands[i]) for i in kept)
        digest = _digest(outline.segment_lines(kept, cands, assessments))
        entry = cached.get(key)
        if entry is None or entry.get("input") != digest:
            try:
                summary, raw, seconds, errors = scene_material.describe_scene(
                    kept, cands, assessments
                )
            except outline.OutlineError as exc:
                raise SystemExit(
                    f"{label} 场景 {n}：概况生成失败（3 次都没通过校验）：{exc}"
                ) from exc
            entry = {
                "segments": list(key),
                "input": digest,
                **summary,
                "raw": raw,
                "seconds": round(seconds, 1),
                "failed_attempts": errors,
            }
            print(
                f"{label} 场景 {n}/{len(groups)}：{summary['title']}（{seconds:.0f}s）"
            )
            cached[key] = entry
            save()
        out[n] = {"title": entry["title"], "summary": entry["summary"]}
    save()
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
        raise SystemExit(f"{exc}\n    启动命令：{LLAMA_SERVER_CMD}") from exc
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
    _write_index()


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
    _write_index()


def _recognise(
    videos: list[layout.Video], force: bool, language: str = ""
) -> list[layout.Video]:
    """Speech recognition for ``videos`` without a cached result (all with
    ``force``); returns those recognised now. ``language`` as in
    SenseVoiceAdapter ("" detects it per sentence)."""
    todo = [v for v in videos if force or not layout.asr_path(WORK_DIR, v.id).exists()]
    if not todo:
        return []
    # Recognition runs alone and frees its models before the 4B is used.
    try:
        asr = subtitle.SenseVoiceAdapter(MODELS_DIR / "asr", language)
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
    return todo


def cmd_subtitle(args: argparse.Namespace) -> None:
    """Speech recognition, then 4B refinement, then translation of what is
    not in Chinese (unless ``--translate none``), then each video's
    subtitles folder; with ``--mode`` also subtitles for that mode's
    previews."""
    out = _out(args)
    dirs = _result_dirs(args, out)
    videos = [v for v in _selected(args) if _load_segments(v.id)[0]["has_audio"]]
    todo = _recognise(videos, args.force, args.language)
    for v in videos:
        path = layout.subtitle_path(WORK_DIR, v.id)
        if args.force or v in todo or not path.exists():
            raw = subtitle.read_asr(layout.asr_path(WORK_DIR, v.id), v.id)
            # Untagged videos (e.g. talks added only for subtitles) are
            # refined with their title as the only context.
            tagged = layout.tags_path(WORK_DIR, v.id).exists()
            context = subtitle.video_context(_tag_rows(v.id) if tagged else [])
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
        _write_video_subtitles(v, dirs[v.id])
        if args.translate != "none":
            _translate_video(args, v, dirs[v.id])
    if args.mode:
        for folder in [dirs[v.id] for v in videos] + [out]:
            path = layout.timeline_path(folder, args.mode)
            if path.exists():
                timeline = json.loads(path.read_text("utf-8"))
                _preview_subtitles(timeline, layout.preview_path(folder, args.mode))


def _write_video_subtitles(video: layout.Video, folder: Path) -> None:
    """A video's subtitles folder: recognised and refined SRT, and what the
    4B changed in each sentence."""
    raw = subtitle.read_asr(layout.asr_path(WORK_DIR, video.id), video.id)
    data = subtitle.read_subtitles(layout.subtitle_path(WORK_DIR, video.id), video.id)
    refined = subtitle.sentences_from_json(data["sentences"])
    _write_text(
        layout.recognized_srt_path(folder), subtitle.to_srt(subtitle.make_cues(raw))
    )
    cues = subtitle.make_cues(refined)
    _write_text(layout.refined_srt_path(folder), subtitle.to_srt(cues))
    by_id = {s.id: s for s in refined}
    changed = [
        {
            "id": s.id,
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "recognized": s.text,
            "refined": by_id[s.id].text,
        }
        for s in raw
        if by_id[s.id].text != s.text
    ]
    _write_json(
        layout.subtitle_changes_path(folder),
        {
            "asr_model": data["asr_model"],
            "refine_seconds": data["refine_seconds"],
            "sentences": len(raw),
            "changed": changed,
            "failed_attempts": data["failed_attempts"],
        },
    )
    print(
        f"  {len(cues)} cues, {len(changed)} sentences changed -> "
        f"{layout.subtitles_dir(folder)}"
    )


def _translate_video(
    args: argparse.Namespace, video: layout.Video, folder: Path
) -> None:
    """Translate what ``video`` says that is not in ``args.translate``
    (cached), and write the translated and bilingual subtitles to its
    subtitles folder when anything needed translating."""
    lang = args.translate
    title = Path(video.file).stem
    data = subtitle.read_subtitles(layout.subtitle_path(WORK_DIR, video.id), video.id)
    sentences = subtitle.sentences_from_json(data["sentences"])
    path = layout.translation_path(WORK_DIR, video.id, lang)
    try:
        if args.force or not translate.is_current(path, video.id, lang, sentences):
            glossary = _glossary(args, video, sentences)
            translator: translate.Translator
            if args.translator == "hymt":
                translator = translate.HyMtAdapter(args.translate_url)
                batch = 1
            else:
                translator = translate.LlmTranslatorAdapter(title)
                batch = translate.LLM_BATCH
            parts, seconds, failed = translate.translate_sentences(
                sentences, translator, glossary, lang, batch
            )
            translate.write_translation(
                path,
                video.id,
                lang,
                translator.name,
                glossary,
                parts,
                seconds,
                failed,
                sentences,
            )
            print(
                f"{video.id} {video.file}: translated "
                f"{translate.translated_count(sentences, parts)}/{len(sentences)} "
                f"sentences in {seconds:.1f}s ({len(failed)} retries)"
            )
        parts = translate.read_translation(path, video.id, lang, sentences)
        if not translate.translated_count(sentences, parts):
            return
        for track, bilingual in ((lang, False), (f"{lang}.bilingual", True)):
            cues = translate.translated_cues(sentences, parts, bilingual)
            _write_text(
                layout.translated_srt_path(folder, track), subtitle.to_srt(cues)
            )
            print(f"  {len(cues)} cues -> {layout.translated_srt_path(folder, track)}")
    except translate.TranslateError as exc:
        raise SystemExit(f"{video.id} {video.file}：字幕翻译失败：{exc}") from exc
    fast = translate.too_fast(translate.translated_cues(sentences, parts, False))
    if fast:
        print(
            f"  {len(fast)} cues read faster than "
            f"{translate.MAX_READING_CPS:.0f} chars/s"
        )


def _glossary(
    args: argparse.Namespace, video: layout.Video, sentences: list[subtitle.Sentence]
) -> dict[str, str]:
    """The user's ``--terms``, over the 4B's own list with ``--glossary``."""
    lang = args.translate
    user_terms = translate.read_user_terms(args.terms) if args.terms else {}
    if not args.glossary:
        return user_terms
    path = layout.glossary_path(WORK_DIR, video.id, lang)
    if args.force or not path.exists():
        terms, seconds = translate.build_glossary(
            sentences, Path(video.file).stem, lang
        )
        translate.write_glossary(path, video.id, terms, seconds)
        print(f"{video.id} {video.file}: {len(terms)} terms in {seconds:.1f}s")
    return translate.merge_terms(translate.read_glossary(path, video.id), user_terms)


def _preview_subtitles(timeline: dict, preview: Path) -> list[subtitle.Cue] | None:
    """Subtitles mapped onto the timeline that ``preview`` plays, written as
    SRT plus a record of every cut; when a video in it was translated, also
    the translated and bilingual tracks. Returns the cues to burn in (the
    translated track when there is one); None when no video in it has
    subtitles. Videos with audio but no refined subtitles yet are named, not
    hidden."""
    tracks: dict[str, dict[str, list[subtitle.Cue]]] = {"": {}}
    missing = []
    for vid in dict.fromkeys(c["video"] for c in timeline["clips"]):
        path = layout.subtitle_path(WORK_DIR, vid)
        if path.exists():
            tracks[""][vid] = subtitle.read_cues(path, vid)
        elif _load_segments(vid)[0]["has_audio"]:
            missing.append(vid)
    if not tracks[""]:
        return None
    for vid in list(tracks[""]):
        for track, translated in _translated_tracks(vid).items():
            tracks.setdefault(track, {})
            tracks[track][vid] = translated
    for track in tracks:
        for vid, cues in tracks[""].items():
            tracks[track].setdefault(vid, cues)
    mapped = {
        track: _write_preview_track(timeline, preview, track, by_video, missing)
        for track, by_video in tracks.items()
    }
    return mapped[SUBTITLE_LANG] if SUBTITLE_LANG in mapped else mapped[""]


def _translated_tracks(vid: str) -> dict[str, list[subtitle.Cue]]:
    """A video's translated and bilingual cues by track name; empty when it
    has no translation or nothing in it needed translating."""
    path = layout.translation_path(WORK_DIR, vid, SUBTITLE_LANG)
    if not path.exists():
        return {}
    data = subtitle.read_subtitles(layout.subtitle_path(WORK_DIR, vid), vid)
    sentences = subtitle.sentences_from_json(data["sentences"])
    try:
        parts = translate.read_translation(path, vid, SUBTITLE_LANG, sentences)
    except translate.TranslateError as exc:
        raise SystemExit(f"{vid}：{exc}") from exc
    if not translate.translated_count(sentences, parts):
        return {}
    return {
        SUBTITLE_LANG: translate.translated_cues(sentences, parts, False),
        f"{SUBTITLE_LANG}.bilingual": translate.translated_cues(sentences, parts, True),
    }


def _write_preview_track(
    timeline: dict,
    preview: Path,
    track: str,
    cues_by_video: dict[str, list[subtitle.Cue]],
    missing: list[str],
) -> list[subtitle.Cue]:
    """One subtitle track on ``preview``: SRT and the record of every cut."""
    cues, records = subtitle.map_cues(cues_by_video, timeline["clips"])
    _write_text(layout.preview_srt_path(preview, track), subtitle.to_srt(cues))
    counts = Counter(r["status"] for r in records)
    _write_json(
        layout.preview_cue_map_path(preview, track),
        {
            "preview": preview.name,
            "track": track or "原文",
            "rule": (
                f"被切点截断的字幕只留切点内听得到的字（不拆英文单词），"
                f"剩不到 {subtitle.MIN_KEPT_SEC}s 或没有完整的字就丢弃；"
                f"截断的译文在片段内不足一半就丢弃，否则整句显示并加“…”"
            ),
            "counts": dict(counts),
            "videos_without_subtitles": missing,
            "cues": records,
        },
    )
    note = f"，{len(missing)} 条有声视频还没有字幕" if missing else ""
    print(
        f"{len(cues)} {track or 'source'} cues on {preview.name}（完整 "
        f"{counts['kept']}，截断 {counts['trimmed']}，丢弃 {counts['dropped']}{note}）"
    )
    return cues


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    _write_json(layout.synth_eval_path(OUT_DIR, args.set), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _category_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--category",
        choices=["", *category.CATEGORIES],
        default="",
        help="material category (default: the model picks from keyframes)",
    )
    parser.add_argument(
        "--retag",
        action="store_true",
        help="tag again with the category prompt; old tags are kept aside",
    )


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
    p_catalog = sub.add_parser(
        "catalog", help="summarise the chunks and scenes edit reads (cached)"
    )
    p_catalog.add_argument("--collection", default="", help=COLLECTION_HELP)
    p_catalog.add_argument("--only", default="", help=ONLY_HELP)
    p_catalog.set_defaults(func=cmd_catalog)
    p_tag = sub.add_parser("tag")
    p_tag.add_argument("--collection", default="", help=COLLECTION_HELP)
    p_tag.add_argument("--only", default="", help=ONLY_HELP)
    p_tag.add_argument("--limit", type=int, default=0, help="max segments per video")
    _category_args(p_tag)
    p_tag.add_argument(
        "--rescore",
        action="store_true",
        help="add the probability-weighted highlight score to old tags",
    )
    p_tag.set_defaults(func=cmd_tag)
    for name, func, help_text in (
        ("aesthetic", cmd_aesthetic, "DOVER quality scores only"),
        ("analyze", cmd_analyze, "tag + DOVER quality scores in parallel"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--collection", default="", help=COLLECTION_HELP)
        p.add_argument("--only", default="", help=ONLY_HELP)
        p.add_argument("--limit", type=int, default=0, help="max segments per video")
        if name == "analyze":
            _category_args(p)
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
            help="result dir (default out/<collection>)",
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
        "--force", action="store_true", help="recognise, refine, translate again"
    )
    p_sub.add_argument(
        "--language",
        default="",
        help="speech language for SenseVoice (zh, en, ja, ko, yue); "
        "default: detect per utterance (mixed speech)",
    )
    p_sub.add_argument(
        "--translate",
        choices=[*sorted(translate.LANGUAGES), "none"],
        default=SUBTITLE_LANG,
        help="translate speech not in this language (default zh; none: skip)",
    )
    p_sub.add_argument(
        "--translator",
        choices=["llm", "hymt"],
        default="hymt",
        help="hymt: Hy-MT2 on --translate-url; llm: the 4B outline server",
    )
    p_sub.add_argument(
        "--translate-url",
        default="http://127.0.0.1:8093",
        help="llama-server running Hy-MT2",
    )
    p_sub.add_argument(
        "--terms", type=Path, default=None, help="JSON object: term -> translation"
    )
    p_sub.add_argument(
        "--glossary",
        action="store_true",
        help="let the 4B extract names and terms first (slow; no gain measured)",
    )
    p_sub.set_defaults(func=cmd_subtitle)
    p_edit = sub.add_parser(
        "edit", help="a cut for one instruction: outline, retrieval fill, report"
    )
    p_edit.add_argument(
        "instruction", help="剪辑要求，如：剪 60 秒的教堂，先外观后内部"
    )
    p_edit.add_argument("--collection", default="vlog", help=COLLECTION_HELP)
    p_edit.add_argument(
        "--narrate",
        action="store_true",
        help="add a narration voice over the original sound",
    )
    p_edit.add_argument(
        "--voice", default=tts.DEFAULT_VOICE, help="Kokoro voice for narration"
    )
    p_edit.add_argument("--only", default="", help=ONLY_HELP)
    p_edit.add_argument(
        "--out",
        type=Path,
        default=None,
        help="result dir (default out/<collection>)",
    )
    p_edit.add_argument(
        "--ratio",
        type=float,
        default=suggest.HIGHLIGHT_RATIO,
        help="cut length / input length when the instruction gives none",
    )
    p_edit.add_argument(
        "--target",
        type=float,
        default=None,
        help="cut length in seconds; the instruction is then not read for one",
    )
    p_edit.add_argument(
        "--review",
        action="store_true",
        help="let the planner review the filled cut and set the final clips",
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
