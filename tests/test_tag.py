from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from video_demo import tag

GOOD: dict[str, Any] = {
    "description": "夜晚的城市天际线",
    "scene": "城市夜景",
    "subjects": ["高楼"],
    "action": "镜头固定",
    "shot_size": "远景",
    "time_of_day": "夜晚",
    "mood": "繁华",
    "tags": ["夜景", "高楼", "灯光"],
    "highlight": 4,
    "issues": ["无明显问题"],
}


def test_validate_accepts_good_tags() -> None:
    assert tag.validate(dict(GOOD)) == GOOD


@pytest.mark.parametrize(
    "patch",
    [
        {"shot_size": "大远景"},
        {"time_of_day": "凌晨"},
        {"highlight": 6},
        {"highlight": 0},
        {"highlight": True},
        {"highlight": 3.5},
        {"tags": ["只有", "两个"]},
        {"issues": ["曝光不足"]},
        {"description": "  "},
        {"subjects": "高楼"},
    ],
)
def test_validate_rejects_bad_values(patch: dict[str, Any]) -> None:
    with pytest.raises(tag.TagError):
        tag.validate({**GOOD, **patch})


def test_validate_rejects_missing_key_and_non_object() -> None:
    broken = dict(GOOD)
    del broken["highlight"]
    with pytest.raises(tag.TagError, match="Missing"):
        tag.validate(broken)
    with pytest.raises(tag.TagError):
        tag.validate(["not", "an", "object"])


class _Resp:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": self._content}}]}


def _frame(tmp_path: Path) -> Path:
    p = tmp_path / "f.jpg"
    p.write_bytes(b"\xff\xd8\xff\xd9")
    return p


def test_tag_segment_sends_images_and_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _Resp:
        seen.update(url=url, body=json)
        return _Resp(__import__("json").dumps(GOOD, ensure_ascii=False))

    monkeypatch.setattr(tag.requests, "post", fake_post)
    tags, raw, _, errors = tag.tag_segment([_frame(tmp_path)] * 2, 4.0)
    assert tags == GOOD and json.loads(raw) == GOOD and errors == []
    content = seen["body"]["messages"][1]["content"]
    assert sum(part["type"] == "image_url" for part in content) == 2
    assert seen["body"]["response_format"]["json_schema"]["schema"] is tag.TAG_SCHEMA


def test_tag_segment_raises_after_all_attempts_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[float] = []

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _Resp:
        calls.append(json["repeat_penalty"])
        return _Resp("这是一段描述")

    monkeypatch.setattr(tag.requests, "post", fake_post)
    with pytest.raises(tag.TagError, match="not JSON"):
        tag.tag_segment([_frame(tmp_path)], 2.0)
    assert calls == list(tag.REPEAT_PENALTIES)


def test_tag_segment_retries_a_repetition_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    looped = {**GOOD, "description": "白天的" * 8}
    replies = iter([looped, GOOD])

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _Resp:
        return _Resp(__import__("json").dumps(next(replies), ensure_ascii=False))

    monkeypatch.setattr(tag.requests, "post", fake_post)
    tags, _, _, errors = tag.tag_segment([_frame(tmp_path)], 2.0)
    assert tags == GOOD
    assert len(errors) == 1 and "repetition" in errors[0]


def test_schema_forbids_empty_text_fields() -> None:
    # Constrained decoding then cannot emit "" (a black credits segment did).
    for key in ("description", "scene", "action", "mood"):
        assert tag.TAG_SCHEMA["properties"][key]["minLength"] == 1
