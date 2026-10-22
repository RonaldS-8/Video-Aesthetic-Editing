"""Command line entry: python -m video_demo <stage>."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from video_demo import evaluate, report, signals, suggest, synth, tag
from video_demo.media import probe, read_clip, write_video
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
SYNTH_PATH = ROOT / "data" / "synth" / "synth_cuts.mp4"

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


def cmd_tag(args: argparse.Namespace) -> None:
    videos = [p for p in _videos() if not args.only or p.stem.startswith(args.only)]
    for path in videos:
        _, segments = _load_segments(path.stem)
        sig = json.loads((WORK_DIR / path.stem / "signals.json").read_text("utf-8"))
        out_path = WORK_DIR / path.stem / "tags.jsonl"
        done: set[int] = set()
        if out_path.exists():
            for line in out_path.read_text("utf-8").splitlines():
                done.add(json.loads(line)["index"])
        todo = [s for s in segments if s.index not in done]
        if args.limit:
            todo = todo[: args.limit]
        for seg in todo:
            frames = [
                WORK_DIR / path.stem / "keyframes" / k
                for k in sig[seg.index]["keyframes"]
            ]
            tags, raw, elapsed, errors = tag.tag_segment(frames, seg.duration)
            record = {
                "index": seg.index,
                "seconds": round(elapsed, 2),
                "tags": tags,
                "raw": raw,
                "failed_attempts": errors,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            retry = f" (retries {len(errors)})" if errors else ""
            print(
                f"{path.stem} seg{seg.index:03d} {elapsed:5.1f}s{retry} "
                f"h={tags['highlight']} {tags['description']}",
                flush=True,
            )


PER_VIDEO_TARGET_SEC = 30.0
REEL_TARGET_SEC = 45.0
REEL_NAME = "all_videos_reel"


def _candidates(stem: str) -> list[suggest.Candidate]:
    data, segments = _load_segments(stem)
    sig = json.loads((WORK_DIR / stem / "signals.json").read_text("utf-8"))
    tags_path = WORK_DIR / stem / "tags.jsonl"
    tags = {
        r["index"]: r["tags"]
        for r in map(json.loads, tags_path.read_text("utf-8").splitlines())
    }
    missing = [s.index for s in segments if s.index not in tags]
    if missing:
        raise RuntimeError(f"{stem}: {len(missing)} segments not tagged: {missing[:5]}")
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


def cmd_suggest(_: argparse.Namespace) -> None:
    out = OUT_DIR / "timelines"
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


def cmd_render(_: argparse.Namespace) -> None:
    for tl_path in sorted((OUT_DIR / "timelines").glob("*.json")):
        timeline = json.loads(tl_path.read_text("utf-8"))

        def frames(tl: dict = timeline):
            for clip in tl["clips"]:
                yield from read_clip(
                    RAW_DIR / clip["video"], clip["src_start"], clip["duration"]
                )

        target = OUT_DIR / "previews" / f"{tl_path.stem}.mp4"
        t0 = time.perf_counter()
        write_video(frames(), target)
        print(f"{target.name}: {time.perf_counter() - t0:.1f}s")


def cmd_report(_: argparse.Namespace) -> None:
    path = report.write(ROOT, [p.stem for p in _videos()], REEL_NAME)
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
    sub.add_parser("suggest").set_defaults(func=cmd_suggest)
    sub.add_parser("render").set_defaults(func=cmd_render)
    sub.add_parser("report").set_defaults(func=cmd_report)
    p_synth = sub.add_parser("synth")
    p_synth.add_argument("--set", choices=sorted(SYNTH_SETS), default="main")
    p_synth.set_defaults(func=cmd_synth)
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--set", choices=sorted(SYNTH_SETS), default="main")
    p_eval.add_argument("--tolerance", type=float, default=0.5)
    p_eval.set_defaults(func=cmd_eval)
    args = parser.parse_args()
    args.func(args)
