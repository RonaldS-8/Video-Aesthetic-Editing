"""One case of the video edit comparison, from case.json to a finished cut.

    python -m ab_eval.run <case.json> <out> [--planner api|local]
        [--no-narrated] [--no-subtitles] [--all-segments]
        [--work DIR] [--seed DIR] [--narrate-cut]

The case's sources are linked into a data folder of their own (``--work``,
default ``<out>/work``) and go through the video_demo stages, each of which
resumes from what is on disk: ingest (speech recognition, segments,
signals), a shortlist of the segments (ab_eval.shortlist; ``--all-segments``
keeps them all), analyze (tags and DOVER), subtitle (only for the narrated
cut), index, catalog (chunk and scene summaries) and edit. ``--seed`` links
finished stage files of the same videos (by content id) from another
data/work folder, to improve on earlier results without analysing again; a
formal run starts empty.

Writes to ``<out>``:
- ``output.mp4``: the cut the case asks for, with its original sound and
  no music, subtitles or title;
- ``narrated/``: the same cut with narration, title and burned Chinese
  subtitles, and the subtitle files (skipped with ``--no-narrated``);
- ``timeline.json``, ``report.html``, ``trace.json`` and ``steps.json``;
- ``prompts/``: the outline and review prompts as they were asked.

``--narrate-cut`` makes only ``narrated/`` from the cut already in ``<out>``
(its timeline as it is), with its own trace there.

The outline and the review of the chosen clips are written in the cloud
(``--planner api``, ab_eval.cloud; the default when ANTHROPIC_API_KEY is
set) or by the local Qwen3.5-4B.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ab_eval.cloud import ClaudeApiPlanner, connect
from ab_eval.shortlist import shortlist
from ab_eval.trace import Trace
from video_demo import layout

ROOT = Path(__file__).resolve().parent.parent
CASE_ID = re.compile(r"^case-\d{2}$")
SOURCE_FILE = re.compile(r"^case-\d{2}-\d+\.[A-Za-z0-9]+$")
# The two phases timed apart: analysing the footage, and cutting it for
# the instruction.
PREPARE, CUT = "prepare", "cut"
# The narrated version made afterwards from a finished cut (--narrate-cut);
# not part of the timed phases.
NARRATED = "narrated"


class CaseError(ValueError):
    pass


@dataclass(frozen=True)
class Case:
    id: str
    folder: Path
    sources: list[tuple[str, str]]  # (file, sha256)
    target_seconds: float
    format: dict[str, Any]
    instruction: str


def load_case(path: Path) -> Case:
    """The case as #342 defines it; raises CaseError on anything else."""
    data = json.loads(path.read_text("utf-8"))
    need = {"id", "provider", "sources", "request", "license", "origin"}
    if set(data) != need:
        raise CaseError(f"{path}: fields {sorted(data)} are not {sorted(need)}")
    if not CASE_ID.match(data["id"]):
        raise CaseError(f"{path}: bad id {data['id']!r}")
    if not data["sources"]:
        raise CaseError(f"{path}: no sources")
    sources = []
    for s in data["sources"]:
        if set(s) != {"file", "sha256"} or not SOURCE_FILE.match(s["file"]):
            raise CaseError(f"{path}: bad source {s}")
        if not re.fullmatch(r"[0-9a-f]{64}", s["sha256"]):
            raise CaseError(f"{path}: bad sha256 for {s['file']}")
        sources.append((s["file"], s["sha256"]))
    request = data["request"]
    if set(request) != {"target_seconds", "format", "instruction"}:
        raise CaseError(f"{path}: request fields {sorted(request)}")
    target = request["target_seconds"]
    if not isinstance(target, (int, float)) or target <= 0:
        raise CaseError(f"{path}: target_seconds must be > 0, got {target!r}")
    if not isinstance(request["format"], dict):
        raise CaseError(f"{path}: format must be an object")
    instruction = request["instruction"].strip()
    if not instruction:
        raise CaseError(f"{path}: empty instruction")
    return Case(
        data["id"], path.parent, sources, float(target), request["format"], instruction
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def verify(case: Case) -> None:
    for file, digest in case.sources:
        got = sha256(case.folder / file)
        if got != digest:
            raise CaseError(f"{file}: sha256 {got} is not {digest}")


def stage(case: Case, raw: Path) -> list[str]:
    """Link the sources into ``raw``/<case id>/ (hard links on Windows,
    symbolic links elsewhere); returns their video ids."""
    folder = raw / case.id
    folder.mkdir(parents=True, exist_ok=True)
    for file, _ in case.sources:
        src, dst = (case.folder / file).resolve(), folder / file
        if dst.exists():
            continue
        if sys.platform == "win32":
            os.link(src, dst)
        else:
            dst.symlink_to(src)
    return [digest[:16] for _, digest in case.sources]


def seed(source: Path, work: Path, ids: list[str]) -> int:
    """Hard-link the stage files of ``ids`` from ``source`` (a data/work
    folder) into ``work``; returns how many files and folders were linked."""
    linked = 0
    for kind in sorted(p for p in source.iterdir() if p.is_dir()):
        for item in kind.iterdir():
            if not any(vid in item.name for vid in ids):
                continue
            dst = work / kind.name / item.name
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if item.is_dir():
                shutil.copytree(item, dst, copy_function=os.link)
            else:
                os.link(item, dst)
            linked += 1
    return linked


def video_demo(*args: str) -> None:
    """Run one video_demo command in its own process, so each stage's
    models are freed when it ends."""
    subprocess.run(
        [sys.executable, "-m", "video_demo", *args],
        cwd=ROOT,
        env=os.environ,
        check=True,
    )


# The two llama-server models the stages ask for: the 4B (tags, summaries,
# subtitles, outline, review) and Hy-MT2 (translation). Paths are relative
# to the models folder (VIDEO_DEMO_MODELS).
SERVERS = {
    "qwen": (
        8092,
        [
            "-m",
            "gguf/Qwen3.5-4B-Q4_K_M.gguf",
            "--mmproj",
            "gguf/mmproj-Qwen3.5-4B-Q8_0.gguf",
            "-c",
            "16384",
        ],
    ),
    "hymt": (8093, ["-m", "gguf/Hy-MT2-1.8B-Q4_K_M.gguf", "-c", "4096"]),
}
SERVER_START_SEC = 600


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
            return True
    except OSError:
        return False


@contextlib.contextmanager
def serve(name: str) -> Iterator[None]:
    """Run the named llama-server for the steps inside; one already answering
    on its port is used as it is and left running. The binary is
    LLAMA_SERVER, or llama-server on PATH."""
    port, model_args = SERVERS[name]
    if _healthy(port):
        yield
        return
    models = layout.models_dir(ROOT)
    args = [
        a if a.startswith("-") or a.isdigit() else str(models / a) for a in model_args
    ]
    process = subprocess.Popen(
        [
            os.environ.get("LLAMA_SERVER", "llama-server"),
            *args,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-np",
            "1",
        ]
    )
    try:
        deadline = time.monotonic() + SERVER_START_SEC
        while not _healthy(port):
            if process.poll() is not None:
                raise RuntimeError(
                    f"llama-server（{name}）启动失败，退出码 {process.returncode}"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"llama-server（{name}）{SERVER_START_SEC} 秒内没有就绪"
                )
            time.sleep(2)
        yield
    finally:
        process.terminate()
        process.wait()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m ab_eval.run")
    p.add_argument("case", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--planner",
        choices=("api", "local"),
        default="api" if os.environ.get("ANTHROPIC_API_KEY") else "local",
        help="who writes the outline and reviews the clips (default: api when "
        "ANTHROPIC_API_KEY is set, else the local model)",
    )
    p.add_argument("--no-narrated", action="store_true")
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="stop after the prepare phase (analysis), before the cut",
    )
    p.add_argument(
        "--narrate-cut",
        action="store_true",
        help="make only the narrated version of the cut already in <out> "
        "(subtitles, translation, narration); its trace goes to <out>/narrated",
    )
    p.add_argument(
        "--no-subtitles",
        action="store_true",
        help="cut without subtitles even when the case asks for them",
    )
    p.add_argument(
        "--all-segments",
        action="store_true",
        help="tag every segment instead of a shortlist (ab_eval.shortlist)",
    )
    p.add_argument("--work", type=Path, help="data folder (default <out>/work)")
    p.add_argument("--seed", type=Path, help="data/work folder to reuse")
    args = p.parse_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.out.resolve()
    work = (args.work or out / "work").resolve()
    narrated = not args.no_narrated
    # video_demo.cli reads these when it is imported (below).
    os.environ["VIDEO_DEMO_DATA"] = str(work / "data")
    os.environ["VIDEO_DEMO_OUT"] = str(work / "out")
    (work / "out").mkdir(parents=True, exist_ok=True)
    trace = Trace()
    planner = None
    try:
        with trace.step("check", "检查", phase=PREPARE):
            case = load_case(args.case.resolve())
            verify(case)
        subtitles = bool(case.format.get("subtitles")) and not args.no_subtitles
        with trace.step("stage", "读取与解码", phase=PREPARE) as span:
            ids = stage(case, work / "data" / "raw")
            if args.seed is not None:
                (work / "data" / "work").mkdir(parents=True, exist_ok=True)
                span.attributes["seeded"] = seed(
                    args.seed.resolve(), work / "data" / "work", ids
                )
        if args.narrate_cut:
            with serve("qwen"), trace.step("subtitle", "理解", phase=NARRATED):
                video_demo("subtitle", "--collection", case.id, "--translate", "none")
            with serve("hymt"), trace.step("translate", "理解", phase=NARRATED):
                video_demo("subtitle", "--collection", case.id)
            narrate_cut(case, out, work, trace)
            return 0
        with trace.step("prepare", "切分", phase=PREPARE):
            video_demo("ingest")
        if not args.all_segments:
            with trace.step("shortlist", "选择与编排", phase=PREPARE) as span:
                kept = shortlist(work / "data" / "work", ids, case.target_seconds)
                span.attributes["segments"] = sum(kept.values())
        with serve("qwen"):
            with trace.step("analyze", "理解", phase=PREPARE):
                video_demo("analyze", "--collection", case.id)
            if narrated or subtitles:
                # The 4B refines what was recognised, then Hy-MT2 translates
                # it: two steps, so the two models are never loaded together.
                with trace.step("subtitle", "理解", phase=PREPARE):
                    video_demo(
                        "subtitle", "--collection", case.id, "--translate", "none"
                    )
        if narrated or subtitles:
            with serve("hymt"), trace.step("translate", "理解", phase=PREPARE):
                video_demo("subtitle", "--collection", case.id)
        with trace.step("index", "理解", phase=PREPARE):
            video_demo("index", "--collection", case.id)
        with serve("qwen"), trace.step("catalog", "理解", phase=PREPARE):
            video_demo("catalog", "--collection", case.id)
        if args.prepare_only:
            return 0

        from video_demo import cli, outline

        planner = connect() if args.planner == "api" else None
        outline.PLANNER = planner
        outline.RECORDER = keep_prompts(out / "prompts")
        where = "local" if args.planner == "local" else "cloud"
        with serve("qwen"), trace.step("edit", "选择与编排", phase=CUT, planner=where):
            folder = cli.cmd_edit(
                argparse.Namespace(
                    instruction=case.instruction,
                    collection=case.id,
                    narrate=narrated,
                    voice=cli.tts.DEFAULT_VOICE,
                    only="",
                    out=work,
                    ratio=0.0,
                    target=case.target_seconds,
                    review=True,
                    preview=narrated,
                )
            )
        with trace.step("render", "渲染", phase=CUT, subtitles=subtitles):
            timeline = json.loads((folder / "timeline.json").read_text("utf-8"))
            videos = cli._selected(argparse.Namespace(collection=case.id, only=""))
            rendered = work / "render" / "output.mp4"
            cli._write_preview(
                timeline,
                {v.id: v.file for v in videos},
                rendered,
                finished=True,
                burn=subtitles,
            )
            out.mkdir(parents=True, exist_ok=True)
            burned = rendered.with_name("output_subtitled.mp4")
            shutil.copy2(burned if burned.exists() else rendered, out / "output.mp4")
            if subtitles:
                _copy_dir(rendered.parent / "subtitles", out / "subtitles")
        _collect(folder, out, narrated)
    finally:
        if planner is not None:
            _record_calls(trace, planner)
        trace.write(out / NARRATED if args.narrate_cut else out)
    return 0


# The prompt file each planner call is kept in (the last asked of a call
# that was retried: the one whose answer was used).
PROMPT_FILES = {"edit_outline": "outline.prompt.md", "edit_review": "review.prompt.md"}


Recorder = Callable[[str, str, dict[str, Any], str, float], None]


def keep_prompts(folder: Path) -> Recorder:
    """An outline.RECORDER that writes each prompt to ``folder``."""

    def record(
        system: str, prompt: str, schema: dict[str, Any], name: str, temperature: float
    ) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / PROMPT_FILES[name]).write_text(
            f"# {name}（temperature {temperature}）\n\n## system\n\n{system}\n\n"
            f"## prompt\n\n{prompt}\n\n## schema\n\n```json\n"
            f"{json.dumps(schema, ensure_ascii=False, indent=1)}\n```\n",
            encoding="utf-8",
        )

    return record


def _record_calls(trace: Trace, planner: ClaudeApiPlanner) -> None:
    """One span per planner call, inside the edit step; API calls carry what
    left the host (#343: anas.external.*)."""
    for call in planner.calls:
        attributes: dict[str, Any] = {"planner": "cloud"}
        if call.transport == "api":
            attributes |= {
                "anas.external.target": "cloud",
                "anas.external.service": planner.model,
                "anas.external.upload_bytes": call.upload_bytes,
                "anas.external.input_tokens": call.input_tokens,
                "anas.external.output_tokens": call.output_tokens,
                "anas.external.cost": round(call.cost_usd, 4),
                "anas.external.currency": "USD",
            }
        trace.call(call.name, call.start_ns, call.seconds, attributes)


def _copy_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def _collect(folder: Path, out: Path, narrated: bool) -> None:
    """Copy the edit's timeline, report and (narrated) previews to ``out``."""
    shutil.copy2(folder / "timeline.json", out / "timeline.json")
    video = out / "narrated" / "output.mp4" if narrated else out / "output.mp4"
    report = (folder / "report.html").read_text("utf-8")
    (out / "report.html").write_text(_relink(report, folder, out, video), "utf-8")
    if narrated:
        _collect_narrated(folder, out / "narrated")


def _collect_narrated(folder: Path, dst: Path) -> None:
    """Copy the narrated preview in ``folder`` (with its subtitles burned in
    when it has any), its subtitle files and narration text to ``dst``."""
    dst.mkdir(parents=True, exist_ok=True)
    burned = folder / "preview_subtitled.mp4"
    shutil.copy2(
        burned if burned.exists() else folder / "preview.mp4", dst / "output.mp4"
    )
    _copy_dir(folder / "subtitles", dst / "subtitles")
    for text in folder.glob("*.narration.txt"):
        shutil.copy2(text, dst / text.name)


def narrate_cut(case: Case, out: Path, work: Path, trace: Trace) -> None:
    """The narrated version of the cut already in ``out``: the same timeline
    with narration, a title card and burned subtitles of what is said and
    narrated, in ``out``/narrated/. The cut itself is not touched."""
    from video_demo import cli

    timeline = json.loads((out / "timeline.json").read_text("utf-8"))
    videos = cli._selected(argparse.Namespace(collection=case.id, only=""))
    with serve("qwen"), trace.step("narrate", "其他", phase=NARRATED):
        timeline["narration"], voices = cli._narrate(
            timeline, videos, cli.tts.DEFAULT_VOICE
        )
    folder = work / "narrated"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=1), "utf-8"
    )
    with trace.step("render", "渲染", phase=NARRATED):
        cli._write_preview(
            timeline,
            {v.id: v.file for v in videos},
            folder / "preview.mp4",
            voices,
            cli._title_card(timeline["outline"]["title"], videos),
        )
    _collect_narrated(folder, out / "narrated")
    shutil.copy2(folder / "timeline.json", out / "narrated" / "timeline.json")


_LINK = re.compile(r'\b(src|href)="([^"#:]+)"')


def _relink(report: str, folder: Path, out: Path, video: Path) -> str:
    """The report links files relative to the edit folder; re-point them at
    ``out`` so the copy opens, and play ``video`` (the delivered cut) instead
    of the edit's preview, which is not copied."""

    def fix(match: re.Match[str]) -> str:
        target = folder / html.unescape(match.group(2))
        if target.suffix == ".mp4" and target.parent == folder:
            target = video
        link = Path(os.path.relpath(target, out)).as_posix()
        return f'{match.group(1)}="{html.escape(link)}"'

    return _LINK.sub(fix, report)


if __name__ == "__main__":
    sys.exit(main())
