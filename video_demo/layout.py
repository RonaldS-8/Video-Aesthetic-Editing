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
``data/work/tags/tag_330ff4c78eabf568.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ID_LENGTH = 16
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


def aesthetic_path(work: Path, vid: str) -> Path:
    return work / "aesthetic" / f"aesthetic_{vid}.jsonl"


def compare_path(work: Path, model: str, vid: str) -> Path:
    """Scores from a model compared against DOVER (``quality`` stage)."""
    kind = model.replace("-", "_")
    return work / "quality" / f"quality_{kind}_{vid}.jsonl"


def timeline_path(out: Path, name: str) -> Path:
    """``name`` is a video id or the name of a multi-video reel."""
    return out / "timelines" / f"timeline_{name}.json"


def preview_path(out: Path, name: str) -> Path:
    return out / "previews" / f"preview_{name}.mp4"
