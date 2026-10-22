"""Command line entry: python -m video_demo <stage>."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from video_demo import evaluate, report, runner, signals, suggest, synth, tag
from video_demo.media import probe, read_clip, segment_frames, write_video
from video_demo.segment import (
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

TAGS_FILE = "tags.jsonl"
QUALITY_FILE = "aesthetic.jsonl"
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

# Only the first 5 minutes of the long vlog are analysed for this demo.
END_TIME = {"06_shanghai_24h_vlog": 300.0}

SYNTH_CLIPS = [
    synth.ClipSpec("04_mbs_skypark.webm", 0.5, 5.0),
    synth.ClipSpec("04_mbs_skypark.webm", 10.0, 5.0),
    synth.ClipSpec("05_mbs_gondola.webm", 1.0, 6.0),
    synth.ClipSpec("03_aurora_timelapse.webm", 3.0, 5.0),
    synth.ClipSpec("05_mbs_gondola.webm", 16.0, 5.0),
    synth.ClipSpec("03_aurora_timelapse.webm", 50.0, 5.0),
    synth.ClipSpec("03_aurora_timelapse.webm", 80.0, 6.0, dissolve_in=1.0),
    synth.ClipSpec("05_mbs_gondola.webm", 24.0, 5.5, dissolve_in=1.0),
]
# Held-out set from other videos, built after the dissolve thresholds were
# fixed, with dissolves of 0.5-1.5 s to check the detector does not overfit.
HOLDOUT_CLIPS = [
    synth.ClipSpec("06_shanghai_24h_vlog.webm", 55.5, 5.5),
    synth.ClipSpec("02_bled_bohinj_vlog.webm", 61.5, 5.5),
    synth.ClipSpec("06_shanghai_24h_vlog.webm", 103.5, 5.5, dissolve_in=1.0),
    synth.ClipSpec("01_lion_city_singapore.webm", 132.5, 5.5, dissolve_in=0.5),
    synth.ClipSpec("06_shanghai_24h_vlog.webm", 149.5, 5.5, dissolve_in=1.5),
    synth.ClipSpec("02_bled_bohinj_vlog.webm", 123.0, 5.0),
    synth.ClipSpec("06_shanghai_24h_vlog.webm", 196.5, 5.0, dissolve_in=1.0),
]
SYNTH_SETS = {
    "main": (SYNTH_CLIPS, SYNTH_PATH),
    "holdout": (HOLDOUT_CLIPS, SYNTH_PATH.with_name("synth_holdout.mp4")),
}


def _videos() -> list[Path]:
    videos = sorted(RAW_DIR.glob("*.webm"))
    if not videos:
        raise FileNotFoundError(f"No .webm files in {RAW_DIR}")
    return videos


def cmd_segment(_: argparse.Namespace) -> None:
    for path in _videos():
        info = probe(path)
        end_time = END_TIME.get(path.stem)
        analysed = info.duration if end_time is None else end_time
        t0 = time.perf_counter()
        cuts = detect_cuts(path, end_time)
        raw_shots = spans_from_cuts(cuts, analysed)
        shots = merge_short(raw_shots)
        segments = split_long(shots)
        elapsed = time.perf_counter() - t0
        out = WORK_DIR / path.stem
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "video": path.name,
            "duration": info.duration,
            "analysed_until": analysed,
            "fps": info.fps,
            "has_audio": info.has_audio,
            "detect_seconds": round(elapsed, 2),
            "cuts": [{"time": c.time, "kind": c.kind} for c in cuts],
            "raw_shot_count": len(raw_shots),
            "shots": shots,
            "segments": [s.to_dict() for s in segments],
        }
        (out / "segments.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_dissolve = sum(c.kind == "dissolve" for c in cuts)
        print(
            f"{path.name}: {len(cuts)} cuts ({n_dissolve} dissolve) -> "
            f"{len(shots)} shots, {len(segments)} segments "
            f"({elapsed:.1f}s for {analysed:.0f}s video)"
        )


def _load_segments(stem: str) -> tuple[dict, list[Segment]]:
    data = json.loads((WORK_DIR / stem / "segments.json").read_text("utf-8"))
    return data, [Segment(**s) for s in data["segments"]]


def cmd_signals(_: argparse.Namespace) -> None:
    for path in _videos():
        _, segments = _load_segments(path.stem)
        t0 = time.perf_counter()
        results = signals.compute(
            path, segments, WORK_DIR / path.stem / "keyframes", END_TIME.get(path.stem)
        )
        payload = [
            {"index": s.index, **r.to_dict()}
            for s, r in zip(segments, results, strict=True)
        ]
        (WORK_DIR / path.stem / "signals.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        flagged = sum(bool(r.flags) for r in results)
        print(
            f"{path.name}: {len(results)} segments, {flagged} flagged "
            f"({time.perf_counter() - t0:.1f}s)"
        )


def _selected(args: argparse.Namespace) -> list[Path]:
    return [p for p in _videos() if not args.only or p.stem.startswith(args.only)]


def _todo(stem: str, out_name: str, limit: int) -> list[Segment]:
    """Segments of ``stem`` not yet recorded in ``out_name`` (resume support)."""
    _, segments = _load_segments(stem)
    done = runner.done_indices(WORK_DIR / stem / out_name)
    todo = [s for s in segments if s.index not in done]
    return todo[:limit] if limit else todo


def _counts(paths: list[Path], out_name: str, limit: int) -> tuple[int, int]:
    """(segments already done, segments this run will process)."""
    done = todo = 0
    for path in paths:
        _, segments = _load_segments(path.stem)
        remaining = _todo(path.stem, out_name, 0)
        done += len(segments) - len(remaining)
        todo += min(limit, len(remaining)) if limit else len(remaining)
    return done, todo


def _tag_stage(paths: list[Path], limit: int) -> runner.StageFn:
    def run(advance: Callable[[str], None]) -> None:
        for path in paths:
            sig = json.loads((WORK_DIR / path.stem / "signals.json").read_text("utf-8"))
            out_path = WORK_DIR / path.stem / TAGS_FILE
            for seg in _todo(path.stem, TAGS_FILE, limit):
                frames = [
                    WORK_DIR / path.stem / "keyframes" / k
                    for k in sig[seg.index]["keyframes"]
                ]
                try:
                    tags, raw, elapsed, errors = tag.tag_segment(frames, seg.duration)
                except Exception as exc:
                    raise runner.StageError(
                        "打标签", path.stem, seg.index, exc
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
                    f"{path.stem} seg{seg.index:03d} {elapsed:5.1f}s{retry} "
                    f"h={tags['highlight']} {tags['description']}"
                )

    return run


def _quality_stage(paths: list[Path], limit: int, adapter: Any) -> runner.StageFn:
    def run(advance: Callable[[str], None]) -> None:
        for path in paths:
            todo = _todo(path.stem, QUALITY_FILE, limit)
            out_path = WORK_DIR / path.stem / QUALITY_FILE
            spans = [(s.start, s.end) for s in todo]
            for i, frames in segment_frames(path, spans, END_TIME.get(path.stem)):
                seg = todo[i]
                t0 = time.perf_counter()
                try:
                    score = adapter.score(frames)
                except Exception as exc:
                    raise runner.StageError(
                        "美学打分", path.stem, seg.index, exc
                    ) from exc
                elapsed = time.perf_counter() - t0
                runner.append_record(
                    out_path,
                    {
                        "index": seg.index,
                        "technical": round(score.technical, 5),
                        "aesthetic": round(score.aesthetic, 5),
                        "fused": round(score.fused, 4),
                        "n_frames": len(frames),
                        "seconds": round(elapsed, 2),
                        "model": "DOVER",
                        "weights_sha256": adapter.weights_sha256,
                        "seed": adapter.seed,
                    },
                )
                advance(
                    f"{path.stem} seg{seg.index:03d} {elapsed:4.1f}s "
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
    paths = _selected(args)
    _quality_stage(paths, args.limit, _load_quality_adapter())(_print_line)


def missing_inputs(paths: list[Path], need_tag: bool, need_quality: bool) -> list[str]:
    """Preflight problems for ``analyze``, each with the command that fixes it."""
    problems: list[str] = []
    for path in paths:
        for name, stage in (("segments.json", "segment"), ("signals.json", "signals")):
            if not (WORK_DIR / path.stem / name).exists():
                problems.append(
                    f"{path.stem} 缺少 {name}：先运行 python -m video_demo {stage}"
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
    paths = _selected(args)
    problems = missing_inputs(paths, need_tag=False, need_quality=False)
    if problems:
        raise SystemExit("开工检查未通过：\n- " + "\n- ".join(problems))
    tag_done, tag_todo = _counts(paths, TAGS_FILE, args.limit)
    q_done, q_todo = _counts(paths, QUALITY_FILE, args.limit)
    if not tag_todo and not q_todo:
        print(f"全部片段已完成打标签和美学打分，无需重跑。{NEXT_STEPS}")
        return
    problems = missing_inputs(paths, need_tag=tag_todo > 0, need_quality=q_todo > 0)
    if problems:
        raise SystemExit("开工检查未通过：\n- " + "\n- ".join(problems))

    stages: list[runner.Stage] = []
    if tag_todo:
        stages.append(
            runner.Stage(
                "打标签", tag_done + tag_todo, tag_done, _tag_stage(paths, args.limit)
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
                _quality_stage(paths, args.limit, adapter),
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


def _candidates(stem: str) -> list[suggest.Candidate]:
    data, segments = _load_segments(stem)
    sig = json.loads((WORK_DIR / stem / "signals.json").read_text("utf-8"))
    tags_path = WORK_DIR / stem / TAGS_FILE
    tags = {
        r["index"]: r["tags"]
        for r in map(json.loads, tags_path.read_text("utf-8").splitlines())
    }
    missing = [s.index for s in segments if s.index not in tags]
    if missing:
        raise RuntimeError(f"{stem}: {len(missing)} segments not tagged: {missing[:5]}")
    quality_path = WORK_DIR / stem / QUALITY_FILE
    if not quality_path.exists():
        raise RuntimeError(f"{stem}: no {QUALITY_FILE}; run `analyze` first")
    quality = {
        r["index"]: r
        for r in map(json.loads, quality_path.read_text("utf-8").splitlines())
    }
    missing = [s.index for s in segments if s.index not in quality]
    if missing:
        raise RuntimeError(f"{stem}: {len(missing)} segments not scored: {missing[:5]}")
    median = statistics.median(sig[s.index]["sharpness"] for s in segments)
    return [
        suggest.Candidate(
            video=data["video"],
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


def cmd_suggest(args: argparse.Namespace) -> None:
    out = args.out / "timelines"
    out.mkdir(parents=True, exist_ok=True)
    every: list[suggest.Candidate] = []
    for path in _videos():
        cands = _candidates(path.stem)
        every += cands
        payload = _timeline_payload(path.stem, cands, PER_VIDEO_TARGET_SEC)
        (out / f"{path.stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        kept = sum(a["keep"] for a in payload["assessments"])
        print(
            f"{path.stem}: keep {kept}/{len(cands)}, "
            f"{len(payload['clips'])} clips {payload['total_sec']}s {payload['notes']}"
        )
    reel = _timeline_payload(REEL_NAME, every, REEL_TARGET_SEC)
    (out / f"{REEL_NAME}.json").write_text(
        json.dumps(reel, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{REEL_NAME}: {len(reel['clips'])} clips {reel['total_sec']}s")


def cmd_render(args: argparse.Namespace) -> None:
    for tl_path in sorted((args.out / "timelines").glob("*.json")):
        timeline = json.loads(tl_path.read_text("utf-8"))

        def frames(tl: dict = timeline):
            for clip in tl["clips"]:
                yield from read_clip(
                    RAW_DIR / clip["video"], clip["src_start"], clip["duration"]
                )

        target = args.out / "previews" / f"{tl_path.stem}.mp4"
        t0 = time.perf_counter()
        write_video(frames(), target)
        print(f"{target.name}: {time.perf_counter() - t0:.1f}s")


def cmd_report(args: argparse.Namespace) -> None:
    stems = [p.stem for p in _videos()]
    path = report.write(ROOT, args.out, stems, REEL_NAME, BASELINE_REPORT)
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
    sub.add_parser("segment").set_defaults(func=cmd_segment)
    sub.add_parser("signals").set_defaults(func=cmd_signals)
    p_tag = sub.add_parser("tag")
    p_tag.add_argument("--only", default="", help="video stem prefix, e.g. 04")
    p_tag.add_argument("--limit", type=int, default=0, help="max segments per video")
    p_tag.set_defaults(func=cmd_tag)
    for name, func, help_text in (
        ("aesthetic", cmd_aesthetic, "DOVER quality scores only"),
        ("analyze", cmd_analyze, "tag + DOVER quality scores in parallel"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--only", default="", help="video stem prefix, e.g. 04")
        p.add_argument("--limit", type=int, default=0, help="max segments per video")
        p.set_defaults(func=func)
    for name, func in (
        ("suggest", cmd_suggest),
        ("render", cmd_render),
        ("report", cmd_report),
    ):
        p = sub.add_parser(name)
        p.add_argument(
            "--out",
            type=Path,
            default=RESULT_DIR,
            help="result dir (default out/dover)",
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
