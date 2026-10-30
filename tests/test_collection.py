"""Collections of short videos: order, chunk groups, scenes across videos."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from video_demo import cli, layout, outline, scenes, suggest


def _cand(video: str, i: int, start: float, dur: float = 8.0) -> suggest.Candidate:
    return suggest.Candidate(
        video=video,
        index=i,
        shot=i,
        start=start,
        end=start + dur,
        tags={
            "description": f"{video}{i}",
            "shot_size": "中景",
            "time_of_day": "白天",
            "tags": [f"t{i}"],
            "highlight": 4,
            "issues": ["无明显问题"],
        },
        flags=[],
        sharpness=500.0,
        sharpness_rel=1.0,
        quality=0.5,
        tech_raw=0.0,
        aes_raw=0.0,
    )


def _video(vid: str) -> layout.Video:
    return layout.Video(id=vid, sha256=vid, file=f"vlog/{vid}.mp4", size=1, mtime_ns=0)


def _summary(text: str, highlights: list[int]) -> dict:
    return {
        "summary": text,
        "places": [text],
        "subjects": [],
        "activities": [],
        "highlights": [{"segment": n, "why": "w"} for n in highlights],
    }


def test_ordered_by_capture_time_only_when_every_video_has_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos = [_video("a"), _video("b"), _video("c")]
    times = {"a": datetime(2026, 5, 2), "b": datetime(2026, 5, 1), "c": None}
    monkeypatch.setattr(cli, "capture_time", lambda p: times[p.stem])
    assert cli._ordered(videos) == (videos, "文件名")
    times["c"] = datetime(2026, 5, 3)
    ordered, order = cli._ordered(videos)
    assert [v.id for v in ordered] == ["b", "a", "c"] and order == "拍摄时间"


def test_chunks_group_whole_short_videos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # a and b are 160 s each (one group of 320 s), c is 400 s on its own.
    cands = [_cand("a", i, 80.0 * i, 80.0) for i in range(2)]
    cands += [_cand("b", i, 80.0 * i, 80.0) for i in range(2)]
    cands += [_cand("c", i, 200.0 * i, 200.0) for i in range(2)]
    own = {
        "a": [([0, 1], _summary("寺庙", [2]))],
        "b": [([0, 1], _summary("湖", [1]))],
        "c": [([0], _summary("城1", [])), ([1], _summary("城2", [1]))],
    }
    monkeypatch.setattr(cli, "_video_summaries", lambda vid, _c: own[vid])
    monkeypatch.setattr(cli, "WORK_DIR", tmp_path)
    calls: list[list] = []

    def group(videos: list) -> tuple[dict, str, float, list[str]]:
        calls.append(videos)
        reply = {"summary": "寺庙和湖", "places": [], "subjects": [], "activities": []}
        return reply, "{}", 1.0, []

    monkeypatch.setattr(outline, "summarize_group", group)
    offsets = {"a": 0.0, "b": 160.0, "c": 320.0}
    for _ in range(2):
        chunks = cli._chunks(cands, offsets)
        assert [(c.videos, c.members, c.start, c.end) for c in chunks] == [
            (["a", "b"], [0, 1, 2, 3], 0.0, 320.0),
            (["c"], [4], 320.0, 520.0),
            (["c"], [5], 520.0, 720.0),
        ]
        # The videos' highlights, numbered along the group.
        assert chunks[0].summary["summary"] == "寺庙和湖"
        assert outline.highlight_members(chunks[0]) == {1, 2}
    # The second run reads the cached group summary.
    assert len(calls) == 1 and [length for length, _ in calls[0]] == [160.0, 160.0]


def test_scene_groups_join_similar_videos_on_the_shared_timeline() -> None:
    # One shot per video, as a phone clip is.
    cands = [replace(_cand("a", i, 8.0 * i), shot=0) for i in range(2)]
    cands += [replace(_cand("b", i, 8.0 * i), shot=0) for i in range(2)]
    e1, e2 = np.eye(2, dtype=np.float32)
    same = {(c.video, c.index): e1 for c in cands}
    # 16 s per video: too short alone, joined across the video boundary.
    groups = cli._scene_groups([[0, 1, 2, 3]], cands, same, same, {"b": 16.0})
    assert groups == [[0, 1, 2, 3]]
    # Different content in b: the boundary stays, short scenes or not.
    other = {**same, ("b", 0): e2, ("b", 1): e2}
    groups = cli._scene_groups([[0, 1, 2, 3]], cands, other, other, {"b": 16.0})
    assert groups == [[0, 1], [2, 3]]


def test_cached_scene_summaries_keep_other_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "scene_multi.json"
    older = {"segments": ["x:9"], "title": "旧", "summary": "s"}
    path.write_text(
        json.dumps({"version": scenes.SCENES_VERSION, "scenes": [older]}), "utf-8"
    )
    cands = [_cand("a", 0, 0.0), _cand("b", 0, 0.0)]
    monkeypatch.setattr(
        scenes,
        "describe_scene",
        lambda *_: ({"title": "新", "summary": "s"}, "{}", 1.0, []),
    )
    got = cli._cached_scene_summaries(
        path,
        "跨视频",
        [[0, 1]],
        cands,
        [suggest.assess(c) for c in cands],
        lambda c: f"{c.video}:{c.index}",
    )
    assert got == {1: {"title": "新", "summary": "s"}}
    saved: list[dict[str, Any]] = json.loads(path.read_text("utf-8"))["scenes"]
    assert [e["segments"] for e in saved] == [["x:9"], ["a:0", "b:0"]]


def test_cached_scene_summaries_redo_when_tags_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[int] = []

    def describe(*_: Any) -> tuple[dict, str, float, list[str]]:
        calls.append(1)
        return {"title": "t", "summary": "s"}, "{}", 1.0, []

    monkeypatch.setattr(scenes, "describe_scene", describe)
    path = tmp_path / "scene_a.json"
    cands = [_cand("a", 0, 0.0)]

    def run(cs: list[suggest.Candidate]) -> None:
        cli._cached_scene_summaries(
            path, "a", [[0]], cs, [suggest.assess(c) for c in cs], lambda c: c.index
        )

    run(cands)
    run(cands)
    assert len(calls) == 1
    # Same segment, a new description (e.g. tagged by another model).
    retagged = replace(cands[0], tags={**cands[0].tags, "description": "新"})
    run([retagged])
    assert len(calls) == 2


def test_tags_are_copied_to_the_result_folder_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos = [
        layout.Video("a" * 16, "s", "vlog/01_trip.webm", 1, 0),
        layout.Video("b" * 16, "s", "long/walk.mp4", 1, 0),
    ]
    monkeypatch.setattr(cli, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(cli, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(cli, "_videos", lambda: videos)
    source = layout.tags_path(tmp_path / "work", "a" * 16)
    source.parent.mkdir(parents=True)
    source.write_text('{"index": 0}\n', "utf-8")
    cli._copy_tags(videos[0])
    copy = tmp_path / "out" / "vlog" / "01_trip" / "tags.jsonl"
    assert copy.read_text("utf-8") == '{"index": 0}\n'
