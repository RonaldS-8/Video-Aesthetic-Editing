"""Video identity and where every stage keeps its files.

A video is identified by the first ``ID_LENGTH`` hex digits of its SHA-256,
not by its file name. Raw videos keep their own names in ``data/raw`` (no
copies, to save space); the manifest ``data/raw/manifest.json`` maps each id
to its file and full hash. Files with identical content get one id and are
processed once. Updating the manifest only hashes files that are new or
whose size or modification time changed.

Videos may sit in sub-folders of ``data/raw``; the first folder level is the
video's *collection* (e.g. ``tvsum/``), which stages use to pick a data set.
``file`` in the manifest is the path relative to ``data/raw``.

Every derived file is named ``<kind>_<id>`` inside a directory per kind, e.g.
``data/work/tags/tag_330ff4c78eabf568.jsonl``. Results under ``out`` are the
exception: they sit in a folder named after the video (``result_dirs``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ID_LENGTH = 16


def data_dir(root: Path) -> Path:
    """``root``/data, or VIDEO_DEMO_DATA when set (e.g. scratch space for
    one case of an evaluation)."""
    return Path(os.environ.get("VIDEO_DEMO_DATA", root / "data"))


def models_dir(root: Path) -> Path:
    """``root``/models, or VIDEO_DEMO_MODELS when set (e.g. the models baked
    into a container image)."""
    return Path(os.environ.get("VIDEO_DEMO_MODELS", root / "models"))


MANIFEST_NAME = "manifest.json"
# Containers the photo library accepts as video (A-NAS photo-video spec).
VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".m4v", ".3gp", ".mkv", ".webm", ".avi", ".mts", ".m2ts"}
)


@dataclass(frozen=True)
class Video:
    id: str
    sha256: str
    file: str
    size: int
    # Only used to decide whether the file must be hashed again.
    mtime_ns: int

    @property
    def collection(self) -> str:
        """First folder under data/raw, or "" for files directly in it."""
        parts = Path(self.file).parts
        return parts[0] if len(parts) > 1 else ""


@dataclass(frozen=True)
class Duplicate:
    file: str
    same_as: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class Manifest:
    videos: list[Video]
    duplicates: list[Duplicate]


@dataclass(frozen=True)
class ManifestChanges:
    added: list[Video]
    duplicates: list[Duplicate]
    # Files no longer in data/raw, and listed files whose content changed
    # (they get a new id; results under the old id are left untouched).
    removed: list[Video]
    changed: list[tuple[Video, Video]]
    # Same content under a new path: the id and its results carry over.
    moved: list[tuple[Video, Video]]
    hashed: int


def scan(raw_dir: Path) -> list[Path]:
    """Video files anywhere under ``raw_dir``, by suffix (any case)."""
    return sorted(
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )


def relative(raw_dir: Path, path: Path) -> str:
    return path.relative_to(raw_dir).as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(raw_dir: Path) -> Manifest | None:
    path = raw_dir / MANIFEST_NAME
    if not path.exists():
        return None
    data = json.loads(path.read_text("utf-8"))
    if data["id_length"] != ID_LENGTH:
        raise ValueError(f"{path} uses ids of {data['id_length']} digits")
    return Manifest(
        [Video(**v) for v in data["videos"]],
        [Duplicate(**d) for d in data["duplicates"]],
    )


def update_manifest(raw_dir: Path) -> tuple[Manifest, ManifestChanges]:
    """Bring the manifest in line with ``data/raw``.

    Files listed with the same size and modification time keep their entry
    without being read; every other file is hashed. Existing videos keep
    their order, new ones follow by file name, and a file whose content is
    already listed is recorded as a duplicate instead of a new video.
    """
    previous = read_manifest(raw_dir) or Manifest([], [])
    known: dict[str, Video | Duplicate] = {
        **{v.file: v for v in previous.videos},
        **{d.file: d for d in previous.duplicates},
    }
    # Listed videos first, in their previous order; new files by name.
    order = {v.file: i for i, v in enumerate(previous.videos)}
    paths = sorted(
        scan(raw_dir),
        key=lambda p: (order.get(relative(raw_dir, p), len(order)), p.as_posix()),
    )
    videos: dict[str, Video] = {}
    duplicates: list[Duplicate] = []
    hashed = 0
    for path in paths:
        name = relative(raw_dir, path)
        stat = path.stat()
        entry = known.get(name)
        unchanged = (
            entry is not None
            and entry.size == stat.st_size
            and entry.mtime_ns == stat.st_mtime_ns
        )
        if unchanged and isinstance(entry, Duplicate):
            duplicates.append(entry)
            continue
        if unchanged and isinstance(entry, Video):
            sha = entry.sha256
        else:
            sha = file_sha256(path)
            hashed += 1
        vid = sha[:ID_LENGTH]
        if vid in videos:
            if videos[vid].sha256 != sha:
                raise ValueError(
                    f"{name} and {videos[vid].file} share the id {vid} "
                    "but differ in content; lengthen ID_LENGTH"
                )
            duplicates.append(Duplicate(name, vid, stat.st_size, stat.st_mtime_ns))
            continue
        videos[vid] = Video(vid, sha, name, stat.st_size, stat.st_mtime_ns)
    if not videos:
        raise FileNotFoundError(f"No video files under {raw_dir}")

    # A duplicate whose original left can no longer point at it.
    orphans = [d for d in duplicates if d.same_as not in videos]
    if orphans:
        raise ValueError(
            "重复文件的原件已不在 data/raw："
            + "、".join(d.file for d in orphans)
            + "。删除清单 data/raw/manifest.json 后重新运行"
        )

    old_ids = {v.id for v in previous.videos}
    old_by_file = {v.file: v for v in previous.videos}
    new_files = {v.file for v in videos.values()} | {d.file for d in duplicates}
    old_by_id = {v.id: v for v in previous.videos}
    moved = [
        (old_by_id[v.id], v)
        for v in videos.values()
        if v.id in old_by_id and old_by_id[v.id].file != v.file
    ]
    changed = [
        (old_by_file[v.file], v)
        for v in videos.values()
        if v.file in old_by_file and old_by_file[v.file].id != v.id
    ]
    changed_files = {old.file for old, _ in changed}
    manifest = Manifest(list(videos.values()), duplicates)
    changes = ManifestChanges(
        added=[
            v
            for v in videos.values()
            if v.id not in old_ids and v.file not in changed_files
        ],
        duplicates=[d for d in duplicates if d.file not in known],
        removed=[
            v for v in previous.videos if v.file not in new_files and v.id not in videos
        ],
        changed=changed,
        moved=moved,
        hashed=hashed,
    )
    return manifest, changes


def write_manifest(raw_dir: Path, manifest: Manifest) -> Path:
    path = raw_dir / MANIFEST_NAME
    payload = {
        "hash": "sha256",
        "id_length": ID_LENGTH,
        "videos": [asdict(v) for v in manifest.videos],
        "duplicates": [asdict(d) for d in manifest.duplicates],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    return path


def load_videos(raw_dir: Path) -> list[Video]:
    """Videos in the manifest, checked against what is on disk.

    Raises when the manifest is missing, a listed file is gone or changed
    size, or a raw video is not listed: run the ``ingest`` stage.
    """
    manifest = read_manifest(raw_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"缺少 {raw_dir / MANIFEST_NAME}：先运行 python -m video_demo ingest"
        )
    videos = manifest.videos
    listed = {v.file for v in videos} | {d.file for d in manifest.duplicates}
    problems = [f"{v.file} 已不存在" for v in videos if not (raw_dir / v.file).exists()]
    problems += [
        f"{v.file} 大小变了"
        for v in videos
        if (raw_dir / v.file).exists() and (raw_dir / v.file).stat().st_size != v.size
    ]
    problems += [
        f"{relative(raw_dir, p)} 不在清单里"
        for p in scan(raw_dir)
        if relative(raw_dir, p) not in listed
    ]
    if problems:
        raise ValueError(
            "清单与 data/raw 不一致（"
            + "；".join(problems)
            + "）：先运行 python -m video_demo ingest"
        )
    return videos


def verify(raw_dir: Path, video: Video) -> None:
    """Full-content check, for stages that create results from the raw file."""
    sha = file_sha256(raw_dir / video.file)
    if sha != video.sha256:
        raise ValueError(
            f"{video.file} 的内容与清单不符（{sha[:ID_LENGTH]} ≠ {video.id}）："
            "先运行 python -m video_demo ingest"
        )


# Derived files: one directory per kind, each file named <kind>_<id>.
def segments_path(work: Path, vid: str) -> Path:
    return work / "segments" / f"segment_{vid}.json"


def signals_path(work: Path, vid: str) -> Path:
    return work / "signals" / f"signal_{vid}.json"


def keyframes_dir(work: Path, vid: str) -> Path:
    return work / "keyframes" / f"keyframe_{vid}"


def tags_path(work: Path, vid: str) -> Path:
    return work / "tags" / f"tag_{vid}.jsonl"


def retired_tags_path(work: Path, vid: str, label: str) -> Path:
    """Tags set aside when the video is tagged again with another prompt;
    ``label`` names the prompt they were made with."""
    return work / "tags" / f"tag_{vid}.{label}.jsonl"


def category_path(work: Path, vid: str) -> Path:
    """The video's material category, chosen by the model or the user."""
    return work / "categories" / f"category_{vid}.json"


def aesthetic_path(work: Path, vid: str) -> Path:
    return work / "aesthetic" / f"aesthetic_{vid}.jsonl"


def index_path(work: Path, vid: str) -> Path:
    """Text and image vectors per segment (EmbeddingGemma 2)."""
    return work / "index" / f"index_{vid}.npz"


def summary_path(work: Path, vid: str) -> Path:
    """Chunk summaries for the highlights outline (cached per video)."""
    return work / "summaries" / f"summary_{vid}.json"


def group_summary_path(work: Path, key: str) -> Path:
    """The summary of a group of short videos, by a hash of its videos'
    summaries (cached; see outline.group_videos)."""
    return work / "summaries" / f"group_{key}.json"


def scene_summary_path(work: Path, vid: str) -> Path:
    """Model-written scene titles and summaries (cached per video)."""
    return work / "scenes" / f"scene_{vid}.json"


def multi_scene_path(work: Path) -> Path:
    """Summaries of scenes that run across videos (cached)."""
    return work / "scenes" / "scene_multi.json"


def proxy_path(work: Path, vid: str) -> Path:
    """A smaller copy of a large video's picture that analysis reads."""
    return work / "proxy" / f"proxy_{vid}.mp4"


def asr_path(work: Path, vid: str) -> Path:
    """Raw speech recognition sentences with per-character times."""
    return work / "asr" / f"asr_{vid}.json"


def subtitle_path(work: Path, vid: str) -> Path:
    """Model-refined sentences, aligned times and subtitle cues."""
    return work / "subtitles" / f"subtitle_{vid}.json"


def translation_path(work: Path, vid: str, lang: str) -> Path:
    """Translated subtitle sentences, one part per source cue."""
    return work / "translations" / f"translation_{lang}_{vid}.json"


def glossary_path(work: Path, vid: str, lang: str) -> Path:
    """Names and terms with their translations (model-extracted)."""
    return work / "glossaries" / f"glossary_{lang}_{vid}.json"


def compare_path(work: Path, model: str, vid: str) -> Path:
    """Scores from a model compared against DOVER (``quality`` stage)."""
    kind = model.replace("-", "_")
    return work / "quality" / f"quality_{kind}_{vid}.jsonl"


# Results: one folder per video under the collection's output directory, named
# after the video's file (looked up in the manifest), so people can find a
# video's results by name. Files inside are named by mode; the same names at
# the collection root hold the timeline across all of its videos.
def result_dirs(out: Path, videos: list[Video]) -> dict[str, Path]:
    """Each video's result folder: its path under the collection folder of
    ``data/raw`` without the suffix (``long/a.mp4`` -> ``out/a``).

    Raises if two videos would share a folder (same name, other suffix).
    """
    dirs: dict[str, Path] = {}
    owner: dict[Path, str] = {}
    for v in videos:
        parts = Path(v.file).with_suffix("").parts
        path = out.joinpath(*(parts[1:] if v.collection else parts))
        if path in owner:
            raise ValueError(
                f"{owner[path]} 和 {v.file} 的结果目录同名（{path}），请重命名其一"
            )
        owner[path] = v.file
        dirs[v.id] = path
    return dirs


def tags_copy_path(folder: Path) -> Path:
    """A copy of the video's tags in its result folder, so they can be found
    by file name; ``data/work/tags`` stays the source the stages read."""
    return folder / "tags.jsonl"


def timeline_path(folder: Path, mode: str) -> Path:
    """``folder`` is a video's result folder, or the collection's for the
    timeline across all of its videos."""
    return folder / f"timeline_{mode}.json"


def preview_path(folder: Path, mode: str) -> Path:
    return folder / f"preview_{mode}.mp4"


def subtitles_dir(folder: Path) -> Path:
    """A result folder's subtitles: the video's own, recognised and refined,
    and those of each preview in the folder."""
    return folder / "subtitles"


def recognized_srt_path(folder: Path) -> Path:
    return subtitles_dir(folder) / "recognized.srt"


def refined_srt_path(folder: Path) -> Path:
    return subtitles_dir(folder) / "refined.srt"


def translated_srt_path(folder: Path, track: str) -> Path:
    """The refined subtitles translated: ``track`` is a language (``zh``)
    or ``zh.bilingual`` (the source line under the translation)."""
    return subtitles_dir(folder) / f"refined.{track}.srt"


def subtitle_changes_path(folder: Path) -> Path:
    """What the 4B changed in each recognised sentence."""
    return subtitles_dir(folder) / "changes.json"


def preview_srt_path(preview: Path, track: str = "") -> Path:
    """Subtitles for ``preview``, named after it in the subtitles folder
    next to it (players such as VLC look there); ``track`` names a
    translation as in translated_srt_path."""
    name = f"{preview.stem}.{track}" if track else preview.stem
    return subtitles_dir(preview.parent) / f"{name}.srt"


def preview_cue_map_path(preview: Path, track: str = "") -> Path:
    """Which cue went where on ``preview``, and what was cut."""
    name = f"{preview.stem}.{track}" if track else preview.stem
    return subtitles_dir(preview.parent) / f"{name}.mapping.json"


def narration_srt_path(preview: Path) -> Path:
    """The narration lines of ``preview`` as subtitles."""
    return subtitles_dir(preview.parent) / f"{preview.stem}.narration.srt"


def narration_text_path(preview: Path) -> Path:
    """The narration of ``preview`` to read: each line with its time on the
    cut and in the source video."""
    return preview.parent / f"{preview.stem}.narration.txt"


def burned_preview_path(preview: Path) -> Path:
    """``preview`` with its subtitles drawn into the picture, next to it."""
    return preview.parent / f"{preview.stem}_subtitled.mp4"


def report_path(out: Path, mode: str) -> Path:
    return out / f"report_{mode}.html"


# Scene material: one clip per scene (its kept segments back to back, the
# rejected ones cut out) and scenes.json describing them.
def scenes_dir(folder: Path) -> Path:
    return folder / "scenes"


def scene_clip_path(folder: Path, scene: int) -> Path:
    return scenes_dir(folder) / f"scene_{scene:02d}.mp4"


# Edits by instruction: one folder per instruction under the collection's
# output directory, named by when it was made and the start of the
# instruction, so earlier edits are kept and can be told apart.
EDIT_NAME_CHARS = 20
# Characters Windows does not allow in a file name, and whitespace.
_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]+')


def edit_dir(out: Path, stamp: str, instruction: str) -> Path:
    name = _UNSAFE_NAME.sub("_", instruction.strip())[:EDIT_NAME_CHARS]
    name = name.strip("._")
    return out / "edits" / (f"{stamp}_{name}" if name else stamp)


def edit_timeline_path(folder: Path) -> Path:
    return folder / "timeline.json"


def edit_preview_path(folder: Path) -> Path:
    return folder / "preview.mp4"


def edit_report_path(folder: Path) -> Path:
    return folder / "report.html"


# Outside the collections: the page linking every result, the frozen
# VLM-only baseline, and the cut-detection evaluation on synthetic videos.
def index_page_path(out: Path) -> Path:
    return out / "index.html"


def baseline_report_path(out: Path) -> Path:
    return out / "baseline" / "report.html"


def synth_eval_path(out: Path, name: str) -> Path:
    return out / "synth" / f"eval_{name}.json"
