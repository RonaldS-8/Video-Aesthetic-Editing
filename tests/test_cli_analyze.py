from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from video_demo import cli, runner


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


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signals: bool) -> None:
    raw, work = tmp_path / "raw", tmp_path / "work"
    raw.mkdir()
    (raw / "01_demo.webm").write_bytes(b"")
    stem_dir = work / "01_demo"
    stem_dir.mkdir(parents=True)
    segments = [
        {"index": i, "start": i * 4.0, "end": i * 4.0 + 4.0, "shot": 0}
        for i in range(3)
    ]
    (stem_dir / "segments.json").write_text(
        json.dumps({"segments": segments}), encoding="utf-8"
    )
    if signals:
        (stem_dir / "signals.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(cli, "RAW_DIR", raw)
    monkeypatch.setattr(cli, "WORK_DIR", work)
    monkeypatch.setattr(cli, "DOVER_ROOT", tmp_path / "no_dover")

    def must_not_run(*_: object, **__: object) -> None:
        raise AssertionError("stages must not start when preflight fails")

    monkeypatch.setattr(runner, "run_parallel", must_not_run)


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
    _workspace(tmp_path, monkeypatch, signals=True)
    for name in (cli.TAGS_FILE, cli.QUALITY_FILE):
        for i in range(3):
            runner.append_record(cli.WORK_DIR / "01_demo" / name, {"index": i})
    cli.cmd_analyze(_args())
    assert "无需重跑" in capsys.readouterr().out
