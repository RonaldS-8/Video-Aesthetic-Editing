from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ab_eval import cloud, run, shortlist, trace
from video_demo import layout, outline


def _write_case(folder: Path, **request: Any) -> Path:
    media = folder / "case-07-1.mp4"
    media.write_bytes(b"picture")
    case = {
        "id": "case-07",
        "provider": "B",
        "sources": [
            {"file": media.name, "sha256": hashlib.sha256(b"picture").hexdigest()}
        ],
        "request": {
            "target_seconds": 60,
            "format": {"music": False, "subtitles": False},
            "instruction": " 剪成 60 秒的冰雪大世界 ",
            **request,
        },
        "license": "CC BY 4.0",
        "origin": "https://example.org",
    }
    path = folder / "case-07.json"
    path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    return path


# --- case ----------------------------------------------------------------------


def test_load_case_reads_the_request(tmp_path: Path) -> None:
    case = run.load_case(_write_case(tmp_path))
    assert case.id == "case-07" and case.target_seconds == 60.0
    assert case.instruction == "剪成 60 秒的冰雪大世界"
    assert case.sources[0][0] == "case-07-1.mp4"
    run.verify(case)


@pytest.mark.parametrize(
    ("request_fields", "error"),
    [
        ({"target_seconds": 0}, "target_seconds"),
        ({"target_seconds": "60"}, "target_seconds"),
        ({"instruction": "  "}, "empty instruction"),
        ({"format": []}, "format"),
        ({"extra": 1}, "request fields"),
    ],
)
def test_load_case_rejects(tmp_path: Path, request_fields, error) -> None:
    with pytest.raises(run.CaseError, match=error):
        run.load_case(_write_case(tmp_path, **request_fields))


def test_load_case_rejects_bad_ids_and_files(tmp_path: Path) -> None:
    path = _write_case(tmp_path)
    data = json.loads(path.read_text("utf-8"))
    for key, value, error in [
        ("id", "case-7", "bad id"),
        ("sources", [{"file": "video.mp4", "sha256": "0" * 64}], "bad source"),
        ("sources", [], "no sources"),
    ]:
        path.write_text(json.dumps({**data, key: value}), encoding="utf-8")
        with pytest.raises(run.CaseError, match=error):
            run.load_case(path)


def test_verify_rejects_a_changed_source(tmp_path: Path) -> None:
    case = run.load_case(_write_case(tmp_path))
    (tmp_path / "case-07-1.mp4").write_bytes(b"other picture")
    with pytest.raises(run.CaseError, match="sha256"):
        run.verify(case)


def test_seed_links_only_the_case_videos(tmp_path: Path) -> None:
    source, work = tmp_path / "src", tmp_path / "work"
    (source / "tags").mkdir(parents=True)
    (source / "tags" / "tag_aaaa.jsonl").write_text("{}", encoding="utf-8")
    (source / "tags" / "tag_bbbb.jsonl").write_text("{}", encoding="utf-8")
    (source / "keyframes" / "keyframe_aaaa").mkdir(parents=True)
    (source / "keyframes" / "keyframe_aaaa" / "seg000_0.jpg").write_bytes(b"j")
    assert run.seed(source, work, ["aaaa"]) == 2
    assert (work / "tags" / "tag_aaaa.jsonl").exists()
    assert not (work / "tags" / "tag_bbbb.jsonl").exists()
    assert (work / "keyframes" / "keyframe_aaaa" / "seg000_0.jpg").exists()
    # A second run finds them there.
    assert run.seed(source, work, ["aaaa"]) == 0


# --- planners ------------------------------------------------------------------


def test_api_schema_drops_unsupported_constraints() -> None:
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 5},
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {"s": {"type": "string", "maxLength": 9}},
                },
            },
        },
    }
    assert cloud.api_schema(schema) == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "n": {"type": "integer"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"s": {"type": "string"}},
                },
            },
        },
    }


class _Stream:
    def __init__(self, message: Any) -> None:
        self.message = message

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def get_final_message(self) -> Any:
        return self.message


class _Client:
    def __init__(self, message: Any) -> None:
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self.message = message

    def _stream(self, **request: Any) -> _Stream:
        self.requests.append(request)
        return _Stream(self.message)


def _message(stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking"),
            SimpleNamespace(type="text", text='{"notes": "n"}'),
        ],
        usage=SimpleNamespace(input_tokens=10_000, output_tokens=2_000),
    )


def test_api_planner_sends_text_with_a_schema_and_records_cost() -> None:
    client = _Client(_message())
    planner = cloud.ClaudeApiPlanner(client)
    text, _ = planner(
        "sys", "审片", {"type": "object", "minProperties": 1}, "edit_review", 0.5
    )
    assert text == '{"notes": "n"}'
    sent = client.requests[0]
    assert sent["model"] == "claude-opus-5-5" and "temperature" not in sent
    assert sent["messages"] == [{"role": "user", "content": "审片"}]
    assert sent["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert sent["fallbacks"] == "default"
    call = planner.calls[0]
    assert (call.input_tokens, call.output_tokens) == (10_000, 2_000)
    # Opus 5.5: $4 / $20 per million tokens.
    assert call.cost_usd == pytest.approx(0.08)
    assert call.upload_bytes > len("审片".encode())


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_api_planner_raises_on_a_cut_off_answer(stop: str) -> None:
    planner = cloud.ClaudeApiPlanner(_Client(_message(stop)))
    with pytest.raises(cloud.PlannerError):
        planner("sys", "审片", {"type": "object"}, "edit_review", 0.2)


def test_only_text_outline_and_review_calls_go_to_the_planner(monkeypatch) -> None:
    sent: list[str] = []
    monkeypatch.setattr(
        outline, "PLANNER", lambda s, p, schema, name, t: (sent.append(name), 0.0)
    )
    local: list[str] = []

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "{}"}}]}

    def post(url: str, json: dict, timeout: float) -> _Resp:
        local.append(json["response_format"]["json_schema"]["name"])
        return _Resp()

    monkeypatch.setattr(outline.requests, "post", post)
    picture = [{"type": "image_url", "image_url": {"url": "data:"}}]
    outline._request("大纲", 0.2, {}, "edit_outline", "http://x")
    outline._request("审片", 0.2, {}, "edit_review", "http://x")
    outline._request(picture, 0.2, {}, "edit_review", "http://x")
    outline._request("场景", 0.2, {}, "scene_summary", "http://x")
    assert sent == ["edit_outline", "edit_review"]
    assert local == ["edit_review", "scene_summary"]


def test_planner_prompts_are_kept_whoever_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(outline, "RECORDER", run.keep_prompts(tmp_path))
    monkeypatch.setattr(outline, "PLANNER", None)

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(outline.requests, "post", lambda url, json, timeout: _Resp())
    outline._request("第一次大纲", 0.2, {}, "edit_outline", "http://x")
    outline._request("重试的大纲", 0.5, {}, "edit_outline", "http://x")
    monkeypatch.setattr(outline, "PLANNER", lambda s, p, schema, name, t: ("{}", 0.0))
    outline._request("审片", 0.2, {"type": "object"}, "edit_review", "http://x")
    outline._request("场景", 0.2, {}, "scene_summary", "http://x")

    assert sorted(f.name for f in tmp_path.iterdir()) == [
        "outline.prompt.md",
        "review.prompt.md",
    ]
    kept = (tmp_path / "outline.prompt.md").read_text("utf-8")
    assert "重试的大纲" in kept and "temperature 0.5" in kept
    assert outline.SYSTEM_PROMPT in kept
    assert '"type": "object"' in (tmp_path / "review.prompt.md").read_text("utf-8")


# --- trace ---------------------------------------------------------------------


def test_trace_writes_one_span_per_step(tmp_path: Path) -> None:
    t = trace.Trace()
    with t.step("check", "check", phase="prepare"):
        pass
    with (
        pytest.raises(RuntimeError),
        t.step("edit", "arrange", phase="cut", planner="api"),
    ):
        raise RuntimeError("boom")
    t.call("edit_review", 1_000_000_000, 2.5, {"anas.external.target": "cloud"})
    t.write(tmp_path)
    spans = json.loads((tmp_path / "trace.json").read_text("utf-8"))["resourceSpans"][
        0
    ]["scopeSpans"][0]["spans"]
    assert [s["name"] for s in spans] == ["check", "edit", "edit_review"]
    assert [s["status"]["code"] for s in spans] == [1, 2, 1]
    review = spans[2]
    assert int(review["endTimeUnixNano"]) - int(review["startTimeUnixNano"]) == (
        2_500_000_000
    )
    attrs = {a["key"]: a["value"] for a in spans[1]["attributes"]}
    assert attrs["anas.category"] == {"stringValue": "arrange"}
    summary = json.loads((tmp_path / "steps.json").read_text("utf-8"))
    steps = summary["steps"]
    assert steps[1]["ok"] is False and steps[1]["planner"] == "api"
    # Planner calls are inside the edit step, not counted again.
    assert set(summary["phases"]) == {"prepare", "cut"}
    # Each run is kept as well, for runs resumed after a pause.
    (kept,) = (tmp_path / "runs").glob("steps-*.json")
    assert json.loads(kept.read_text("utf-8")) == summary


# --- shortlist -------------------------------------------------------------------


def test_quota_grows_with_the_target_from_a_floor() -> None:
    assert shortlist.quota(20) == shortlist.MIN_PICKS
    assert shortlist.quota(60) == 100
    assert shortlist.quota(90) > shortlist.quota(60)


def test_score_prefers_clean_long_spoken_segments() -> None:
    clean = shortlist.score(0, 3, [], [])
    assert shortlist.score(0, 3, ["blurry"], []) < clean
    assert shortlist.score(0, 3, ["shaky"], []) < clean
    # Dark costs less: it can be the look of the film.
    assert shortlist.score(0, 3, ["blurry"], []) < shortlist.score(
        0, 3, ["low_light"], []
    )
    assert shortlist.score(0, 1, [], []) < clean
    assert shortlist.score(0, 3, [], [(1, 4)]) > clean


def _pieces(n: int, shot_of=lambda k: k, score_of=lambda k: 0.0):
    return [
        shortlist.Piece(k, float(k), float(k + 1), shot_of(k), score_of(k))
        for k in range(n)
    ]


def test_pick_covers_the_whole_video() -> None:
    # Later segments score higher, but each stretch of time keeps one.
    kept = shortlist.pick(_pieces(40, score_of=float), 4, 0.0, 40.0)
    assert [p.index for p in kept] == [9, 19, 29, 39]


def test_pick_keeps_at_most_two_windows_of_a_shot() -> None:
    kept = shortlist.pick(_pieces(20, shot_of=lambda k: 0), 10, 0.0, 20.0)
    assert len(kept) == shortlist.MAX_PER_SHOT


def test_pick_fills_empty_stretches_and_keeps_all_when_few() -> None:
    pieces = [p for p in _pieces(10) if p.index < 5]  # second half is empty
    assert len(shortlist.pick(pieces, 4, 0.0, 10.0)) == 4
    assert shortlist.pick(pieces, 9, 0.0, 10.0) == pieces


def test_shares_split_by_length() -> None:
    assert shortlist.shares({"a": 300.0, "b": 100.0}, 40) == {"a": 30, "b": 10}
    assert shortlist.shares({"a": 1000.0, "b": 1.0}, 40)["b"] == 1


def test_shortlist_rewrites_segments_and_is_repeatable(tmp_path: Path) -> None:
    segments = [
        {"index": k, "start": float(k), "end": float(k + 2), "shot": k}
        for k in range(0, 400, 2)
    ]
    data = {"video_id": "v1", "analysed_until": 400.0, "segments": segments}
    path = layout.segments_path(tmp_path, "v1")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), "utf-8")
    sig = layout.signals_path(tmp_path, "v1")
    sig.parent.mkdir(parents=True)
    sig.write_text(json.dumps([{"flags": []}] * 400), "utf-8")

    assert shortlist.shortlist(tmp_path, ["v1"], 20) == {"v1": shortlist.MIN_PICKS}
    first = json.loads(path.read_text("utf-8"))
    assert first["shortlist"] == {"kept": shortlist.MIN_PICKS, "of": 200}
    assert {s["index"] for s in first["segments"]} <= {s["index"] for s in segments}
    # A second run reads the full list again, not the shortlist.
    shortlist.shortlist(tmp_path, ["v1"], 60)
    assert len(json.loads(path.read_text("utf-8"))["segments"]) == 100


def test_relink_points_report_copy_at_delivered_files(tmp_path: Path) -> None:
    folder = tmp_path / "work" / "edits" / "e1"
    out = tmp_path / "out"
    page = (
        '<video src="preview.mp4"></video>'
        '<img src="../../data/k/a&amp;b.jpg"><a href="#top">x</a>'
    )
    fixed = run._relink(page, folder, out, out / "output.mp4")
    assert '<video src="output.mp4">' in fixed
    assert 'src="../work/data/k/a&amp;b.jpg"' in fixed
    assert 'href="#top"' in fixed


class _Process:
    def __init__(self, exit_code: int | None = None) -> None:
        self.returncode = exit_code
        self.stopped = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.stopped = True

    def wait(self) -> int:
        return 0


def test_serve_uses_a_running_server_as_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "_healthy", lambda port: True)

    def start(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must not start a second server")

    monkeypatch.setattr(run.subprocess, "Popen", start)
    with run.serve("qwen"):
        pass


def test_serve_starts_and_stops_its_own_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VIDEO_DEMO_MODELS", str(tmp_path))
    monkeypatch.setenv("LLAMA_SERVER", "my-llama-server")
    answers = iter([False, False, True])
    monkeypatch.setattr(run, "_healthy", lambda port: next(answers))
    monkeypatch.setattr(run.time, "sleep", lambda sec: None)
    started: dict[str, Any] = {}

    def start(cmd: list[str]) -> _Process:
        started["cmd"] = cmd
        started["process"] = _Process()
        return started["process"]

    monkeypatch.setattr(run.subprocess, "Popen", start)
    with run.serve("hymt"):
        assert not started["process"].stopped
    assert started["process"].stopped
    cmd = started["cmd"]
    assert cmd[0] == "my-llama-server"
    assert str(tmp_path / "gguf/Hy-MT2-1.8B-Q4_K_M.gguf") in cmd
    assert cmd[cmd.index("--port") + 1] == "8093"


def test_serve_raises_when_the_server_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "_healthy", lambda port: False)
    monkeypatch.setattr(run.subprocess, "Popen", lambda cmd: _Process(exit_code=1))
    with pytest.raises(RuntimeError, match="启动失败"):
        with run.serve("qwen"):
            pass


def test_models_dir_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("VIDEO_DEMO_MODELS", raising=False)
    assert layout.models_dir(tmp_path) == tmp_path / "models"
    monkeypatch.setenv("VIDEO_DEMO_MODELS", str(tmp_path / "m"))
    assert layout.models_dir(tmp_path) == tmp_path / "m"


def test_narrate_cut_keeps_the_cut_and_writes_narrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video_demo import cli

    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    cut = {"clips": [{"video": "v1"}], "outline": {"title": "冰城"}}
    (out / "timeline.json").write_text(json.dumps(cut), "utf-8")
    (out / "output.mp4").write_bytes(b"cut")
    video = SimpleNamespace(id="v1", file="case-07-1.webm")
    rendered: list[Any] = []

    def write_preview(timeline, files, target, voices, title) -> None:
        rendered.append((timeline["narration"], files, voices, title))
        target.write_bytes(b"plain")
        (target.parent / "preview_subtitled.mp4").write_bytes(b"burned")
        (target.parent / "subtitles").mkdir()
        (target.parent / "subtitles" / "preview.narration.srt").write_text("1")
        (target.parent / "preview.narration.txt").write_text("lines", "utf-8")

    monkeypatch.setattr(run, "serve", lambda name: contextlib.nullcontext())
    monkeypatch.setattr(cli, "_selected", lambda args: [video])
    monkeypatch.setattr(cli, "_narrate", lambda t, v, voice: ({"lines": []}, ["a"]))
    monkeypatch.setattr(cli, "_title_card", lambda title, videos: (title, "城市漫步"))
    monkeypatch.setattr(cli, "_write_preview", write_preview)
    case = SimpleNamespace(id="case-07")
    run.narrate_cut(case, out, work, trace.Trace())  # type: ignore[arg-type]

    assert rendered == [
        ({"lines": []}, {"v1": "case-07-1.webm"}, ["a"], ("冰城", "城市漫步"))
    ]
    assert json.loads((out / "timeline.json").read_text("utf-8")) == cut
    assert (out / "output.mp4").read_bytes() == b"cut"
    narrated = out / "narrated"
    assert (narrated / "output.mp4").read_bytes() == b"burned"
    assert (narrated / "subtitles" / "preview.narration.srt").exists()
    assert (narrated / "preview.narration.txt").exists()
    assert "narration" in json.loads((narrated / "timeline.json").read_text("utf-8"))
