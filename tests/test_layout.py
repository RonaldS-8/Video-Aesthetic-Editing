from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from video_demo import layout


def _video(raw: Path, name: str, content: bytes, mtime: int | None = None) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / name
    path.write_bytes(content)
    if mtime is not None:
        os.utime(path, ns=(mtime, mtime))
    return path


def _id(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[: layout.ID_LENGTH]


def test_ids_are_content_hashes_and_duplicates_are_kept_once(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _video(raw, "b.webm", b"beta")
    _video(raw, "a.webm", b"alpha")
    _video(raw, "a_copy.webm", b"alpha")
    manifest, changes = layout.update_manifest(raw)
    assert [(v.file, v.id) for v in manifest.videos] == [
        ("a.webm", _id(b"alpha")),
        ("b.webm", _id(b"beta")),
    ]
    assert manifest.videos[0].sha256 == hashlib.sha256(b"alpha").hexdigest()
    assert [(d.file, d.same_as) for d in manifest.duplicates] == [
        ("a_copy.webm", _id(b"alpha"))
    ]
    assert len(changes.added) == 2 and changes.hashed == 3


def test_update_only_hashes_new_or_changed_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _video(raw, "b.webm", b"beta", mtime=1_000_000_000)
    _video(raw, "c.webm", b"gamma", mtime=1_000_000_000)
    layout.write_manifest(raw, layout.update_manifest(raw)[0])

    _video(raw, "a.webm", b"alpha")  # new, sorts first by name
    manifest, changes = layout.update_manifest(raw)
    assert changes.hashed == 1
    assert [v.id for v in changes.added] == [_id(b"alpha")]
    # Existing videos keep their place; new ones follow.
    assert [v.file for v in manifest.videos] == ["b.webm", "c.webm", "a.webm"]
    layout.write_manifest(raw, manifest)

    unchanged, changes = layout.update_manifest(raw)
    assert changes.hashed == 0 and not changes.added
    assert unchanged == manifest


def test_update_reports_changed_and_removed_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _video(raw, "a.webm", b"alpha", mtime=1_000_000_000)
    _video(raw, "b.webm", b"beta", mtime=1_000_000_000)
    layout.write_manifest(raw, layout.update_manifest(raw)[0])
    _video(raw, "a.webm", b"ALPHA!", mtime=2_000_000_000)
    (raw / "b.webm").unlink()
    manifest, changes = layout.update_manifest(raw)
    assert [(old.id, new.id) for old, new in changes.changed] == [
        (_id(b"alpha"), _id(b"ALPHA!"))
    ]
    assert not changes.added
    assert [v.file for v in changes.removed] == ["b.webm"]
    assert [v.id for v in manifest.videos] == [_id(b"ALPHA!")]


def test_id_prefix_collision_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    _video(raw, "a.webm", b"alpha")
    _video(raw, "b.webm", b"beta")
    fake = {"a.webm": "0" * 16 + "a" * 48, "b.webm": "0" * 16 + "b" * 48}
    monkeypatch.setattr(layout, "file_sha256", lambda path: fake[path.name])
    with pytest.raises(ValueError, match="share the id"):
        layout.update_manifest(raw)


def test_empty_raw_dir_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    with pytest.raises(FileNotFoundError, match="No video files"):
        layout.update_manifest(tmp_path / "raw")


def test_load_videos_rejects_a_stale_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _video(raw, "a.webm", b"alpha")
    with pytest.raises(FileNotFoundError, match="ingest"):
        layout.load_videos(raw)
    layout.write_manifest(raw, layout.update_manifest(raw)[0])
    assert [v.file for v in layout.load_videos(raw)] == ["a.webm"]

    _video(raw, "new.webm", b"new")
    with pytest.raises(ValueError, match="new.webm 不在清单里"):
        layout.load_videos(raw)
    (raw / "new.webm").unlink()
    _video(raw, "a.webm", b"alpha, longer")
    with pytest.raises(ValueError, match="a.webm 大小变了"):
        layout.load_videos(raw)


def test_verify_catches_same_size_content_change(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _video(raw, "a.webm", b"alpha")
    video = layout.update_manifest(raw)[0].videos[0]
    layout.verify(raw, video)
    _video(raw, "a.webm", b"ALPHA")
    with pytest.raises(ValueError, match="与清单不符"):
        layout.verify(raw, video)


def test_derived_files_are_named_kind_underscore_id(tmp_path: Path) -> None:
    vid = "330ff4c78eabf568"
    work = tmp_path / "work"
    names = [
        layout.segments_path(work, vid),
        layout.signals_path(work, vid),
        layout.keyframes_dir(work, vid),
        layout.tags_path(work, vid),
        layout.aesthetic_path(work, vid),
        layout.compare_path(work, "dover-mobile", vid),
    ]
    assert [p.relative_to(tmp_path).as_posix() for p in names] == [
        f"work/segments/segment_{vid}.json",
        f"work/signals/signal_{vid}.json",
        f"work/keyframes/keyframe_{vid}",
        f"work/tags/tag_{vid}.jsonl",
        f"work/aesthetic/aesthetic_{vid}.jsonl",
        f"work/quality/quality_dover_mobile_{vid}.jsonl",
    ]


def _listed(vid: str, file: str) -> layout.Video:
    return layout.Video(vid, vid * 4, file, 1, 0)


def test_results_go_to_a_folder_named_after_the_video(tmp_path: Path) -> None:
    out = tmp_path / "out"
    videos = [
        _listed("a" * 16, "long/edinburgh_walk.webm"),
        _listed("b" * 16, "long/trip/day1.MP4"),
        _listed("c" * 16, "loose.mov"),
    ]
    dirs = layout.result_dirs(out, videos)
    assert {k: p.relative_to(out).as_posix() for k, p in dirs.items()} == {
        "a" * 16: "edinburgh_walk",
        "b" * 16: "trip/day1",
        "c" * 16: "loose",
    }
    folder = dirs["a" * 16]
    names = [
        layout.timeline_path(folder, "highlights"),
        layout.preview_path(folder, "ranked"),
        layout.timeline_path(out, "highlights"),
        layout.report_path(out, "ranked"),
        layout.scene_clip_path(folder, 3),
    ]
    assert [p.relative_to(out).as_posix() for p in names] == [
        "edinburgh_walk/timeline_highlights.json",
        "edinburgh_walk/preview_ranked.mp4",
        "timeline_highlights.json",
        "report_ranked.html",
        "edinburgh_walk/scenes/scene_03.mp4",
    ]


def test_edit_folders_carry_the_start_of_the_instruction(tmp_path: Path) -> None:
    def name(instruction: str) -> str:
        return layout.edit_dir(tmp_path, "20261029-120000", instruction).name

    assert name("剪 60 秒的爱丁堡教堂，先外观后内部") == (
        "20261029-120000_剪_60_秒的爱丁堡教堂，先外观后内部"
    )
    assert name('a/b\\c:"d"?*<e>|\n') == "20261029-120000_a_b_c_d_e"
    assert name("  ...  ") == "20261029-120000"
    assert layout.edit_dir(tmp_path, "s", "x").parent == tmp_path / "edits"


def test_index_baseline_and_synth_sit_outside_the_collections(
    tmp_path: Path,
) -> None:
    names = [
        layout.index_page_path(tmp_path),
        layout.baseline_report_path(tmp_path),
        layout.synth_eval_path(tmp_path, "main"),
        layout.tags_copy_path(tmp_path / "vlog" / "01"),
        layout.refined_srt_path(tmp_path / "vlog" / "01"),
        layout.preview_srt_path(tmp_path / "vlog" / "preview_highlights.mp4"),
        layout.burned_preview_path(tmp_path / "vlog" / "preview_highlights.mp4"),
        layout.translated_srt_path(tmp_path / "vlog" / "01", "zh.bilingual"),
        layout.preview_srt_path(tmp_path / "vlog" / "preview_highlights.mp4", "zh"),
        layout.narration_text_path(tmp_path / "vlog" / "edits" / "e" / "preview.mp4"),
    ]
    assert [p.relative_to(tmp_path).as_posix() for p in names] == [
        "index.html",
        "baseline/report.html",
        "synth/eval_main.json",
        "vlog/01/tags.jsonl",
        "vlog/01/subtitles/refined.srt",
        "vlog/subtitles/preview_highlights.srt",
        "vlog/preview_highlights_subtitled.mp4",
        "vlog/01/subtitles/refined.zh.bilingual.srt",
        "vlog/subtitles/preview_highlights.zh.srt",
        "vlog/edits/e/preview.narration.txt",
    ]


def test_result_folders_must_not_collide(tmp_path: Path) -> None:
    videos = [_listed("a" * 16, "long/walk.mp4"), _listed("b" * 16, "long/walk.mov")]
    with pytest.raises(ValueError, match="结果目录同名"):
        layout.result_dirs(tmp_path, videos)


def test_videos_in_collection_folders_any_container(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _video(raw / "tvsum", "a.MP4", b"alpha")
    _video(raw / "long", "b.mov", b"beta")
    _video(raw, "c.webm", b"gamma")
    _video(raw, "notes.txt", b"not a video")
    manifest, _ = layout.update_manifest(raw)
    got = {v.file: v.collection for v in manifest.videos}
    assert got == {"tvsum/a.MP4": "tvsum", "long/b.mov": "long", "c.webm": ""}


def test_moving_a_file_keeps_its_id(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _video(raw, "a.webm", b"alpha")
    layout.write_manifest(raw, layout.update_manifest(raw)[0])
    (raw / "vlog").mkdir()
    (raw / "a.webm").rename(raw / "vlog" / "a.webm")
    manifest, changes = layout.update_manifest(raw)
    assert [(old.file, new.file) for old, new in changes.moved] == [
        ("a.webm", "vlog/a.webm")
    ]
    assert not changes.added and not changes.removed
    assert [v.id for v in manifest.videos] == [_id(b"alpha")]
