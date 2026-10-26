from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from video_demo import cli, layout, runner


def _stage(name: str, fn: runner.StageFn, total: int = 2) -> runner.Stage:
    return runner.Stage(name, total, 0, fn)


def test_stages_run_concurrently() -> None:
    a_started, b_started = threading.Event(), threading.Event()

    def a(advance: Callable[[str], None]) -> None:
        a_started.set()
        # Only returns if stage b is running at the same time.
        assert b_started.wait(timeout=5)
        advance("a")

    def b(advance: Callable[[str], None]) -> None:
        b_started.set()
        assert a_started.wait(timeout=5)
        advance("b")

    results = runner.run_parallel([_stage("a", a), _stage("b", b)], show_progress=False)
    assert [(r.name, r.processed, r.error) for r in results] == [
        ("a", 1, None),
        ("b", 1, None),
    ]


def test_a_failing_stage_does_not_stop_the_other() -> None:
    def broken(advance: Callable[[str], None]) -> None:
        advance("first")
        raise runner.StageError("打标签", "01_x", 3, ValueError("bad json"))

    def fine(advance: Callable[[str], None]) -> None:
        for i in range(3):
            advance(str(i))

    results = runner.run_parallel(
        [_stage("broken", broken), _stage("fine", fine, 3)], show_progress=False
    )
    assert results[0].processed == 1 and "片段 #3" in str(results[0].error)
    assert results[1].processed == 3 and results[1].error is None


def test_done_indices_resume(tmp_path: Path) -> None:
    path = tmp_path / "aesthetic.jsonl"
    assert runner.done_indices(path) == set()
    runner.append_record(path, {"index": 0, "fused": 0.5})
    runner.append_record(path, {"index": 2, "fused": 0.6})
    assert runner.done_indices(path) == {0, 2}


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signals: bool) -> str:
    """One registered video with 3 segments; returns its id."""
    raw, work = tmp_path / "raw", tmp_path / "work"
    raw.mkdir()
    (raw / "01_demo.webm").write_bytes(b"demo")
    manifest, _ = layout.update_manifest(raw)
    layout.write_manifest(raw, manifest)
    vid = manifest.videos[0].id
    segments = [
        {"index": i, "start": i * 4.0, "end": i * 4.0 + 4.0, "shot": 0}
        for i in range(3)
    ]
    path = layout.segments_path(work, vid)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"video_id": vid, "segments": segments}), encoding="utf-8"
    )
    if signals:
        path = layout.signals_path(work, vid)
        path.parent.mkdir(parents=True)
        path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(cli, "RAW_DIR", raw)
    monkeypatch.setattr(cli, "WORK_DIR", work)
    monkeypatch.setattr(cli, "DOVER_ROOT", tmp_path / "no_dover")

    def must_not_run(*_: object, **__: object) -> None:
        raise AssertionError("stages must not start when preflight fails")

    monkeypatch.setattr(runner, "run_parallel", must_not_run)
    return vid


def _args() -> object:
    class Args:
        only = ""
        limit = 0

    return Args()


def test_analyze_stops_before_starting_when_signals_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, monkeypatch, signals=False)
    with pytest.raises(SystemExit, match="signals"):
        cli.cmd_analyze(_args())


def test_analyze_reports_server_and_weights_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, monkeypatch, signals=True)

    def down(*_: object, **__: object) -> None:
        raise ConnectionError("无法连接 llama-server")

    monkeypatch.setattr(cli.tag, "check_server", down)
    with pytest.raises(SystemExit) as info:
        cli.cmd_analyze(_args())
    message = str(info.value)
    assert "llama-server" in message and "启动命令" in message
    assert "DOVER" in message


def test_analyze_skips_when_everything_is_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    vid = _workspace(tmp_path, monkeypatch, signals=True)
    for out in (layout.tags_path, layout.aesthetic_path):
        for i in range(3):
            runner.append_record(out(cli.WORK_DIR, vid), {"index": i})
    cli.cmd_analyze(_args())
    assert "无需重跑" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("model", "hint"), [("dover-mobile", "DOVER-Mobile.pth"), ("uvq", "google/uvq")]
)
def test_quality_explains_missing_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, hint: str
) -> None:
    _workspace(tmp_path, monkeypatch, signals=True)
    monkeypatch.setattr(cli, "UVQ_ROOT", tmp_path / "no_uvq")

    class Args:
        only = ""
        limit = 0
        threads = 0

    Args.model = model
    with pytest.raises(SystemExit, match=hint):
        cli.cmd_quality(Args())


def test_compare_quality_names_the_missing_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = _workspace(tmp_path, monkeypatch, signals=True)
    for out in (layout.tags_path, layout.aesthetic_path):
        for i in range(3):
            runner.append_record(out(cli.WORK_DIR, vid), {"index": i, "tags": {}})

    class Args:
        out = tmp_path / "out"

    with pytest.raises(RuntimeError, match=f"quality_dover_mobile_{vid}.jsonl"):
        cli.cmd_compare_quality(Args())


def test_ingest_registers_and_segments_only_new_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    old = _workspace(tmp_path, monkeypatch, signals=True)
    (cli.RAW_DIR / "02_new.webm").write_bytes(b"new video")
    (cli.RAW_DIR / "02_new_copy.webm").write_bytes(b"new video")
    segmented: list[str] = []
    monkeypatch.setattr(cli, "_segment", lambda vs: segmented.extend(v.id for v in vs))
    monkeypatch.setattr(cli, "_signals", lambda vs: None)
    cli.cmd_ingest(_args())
    videos = layout.load_videos(cli.RAW_DIR)
    assert [v.file for v in videos] == ["01_demo.webm", "02_new.webm"]
    assert segmented == [videos[1].id] and old not in segmented
    out = capsys.readouterr().out
    assert "新视频" in out and "02_new_copy.webm" in out and "重复" in out


def test_ingest_without_new_videos_does_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _workspace(tmp_path, monkeypatch, signals=True)

    def must_not_segment(*_: object) -> None:
        raise AssertionError("nothing new to segment")

    monkeypatch.setattr(cli, "_segment", must_not_segment)
    cli.cmd_ingest(_args())
    assert "没有需要切分的新视频" in capsys.readouterr().out


def test_collection_selects_videos_and_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, monkeypatch, signals=True)
    (cli.RAW_DIR / "tvsum").mkdir()
    (cli.RAW_DIR / "tvsum" / "a.mp4").write_bytes(b"tvsum video")
    layout.write_manifest(cli.RAW_DIR, layout.update_manifest(cli.RAW_DIR)[0])

    class Args:
        collection = "tvsum"
        only = ""
        out = None

    assert [v.file for v in cli._selected(Args())] == ["tvsum/a.mp4"]
    assert cli._out(Args()) == cli.OUT_DIR / "tvsum"
    Args.mode = "highlights"
    assert cli._out(Args()) == cli.OUT_DIR / "tvsum" / "highlights"
    Args.mode = "reel"
    Args.collection = "vlog"
    assert cli._out(Args()) == cli.RESULT_DIR
    with pytest.raises(SystemExit, match="没有符合条件的视频"):
        cli._selected(Args())
