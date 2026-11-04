from __future__ import annotations

import json
import math
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
        tokens = [
            {
                "token": ch,
                "logprob": 0.0,
                "top_logprobs": [{"token": ch, "logprob": 0.0}],
            }
            for ch in self._content
        ]
        return {
            "choices": [
                {"message": {"content": self._content}, "logprobs": {"content": tokens}}
            ]
        }


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
    tags, raw, _, errors, score = tag.tag_segment([_frame(tmp_path)] * 2, 4.0)
    assert tags == GOOD and json.loads(raw) == GOOD and errors == []
    assert score["highlight_p"] == GOOD["highlight"]
    assert seen["body"]["logprobs"] is True
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
    tags, _, _, errors, _ = tag.tag_segment([_frame(tmp_path)], 2.0)
    assert tags == GOOD
    assert len(errors) == 1 and "repetition" in errors[0]


def test_schema_forbids_empty_text_fields() -> None:
    # Constrained decoding then cannot emit "" (a black credits segment did).
    for key in ("description", "scene", "action", "mood"):
        assert tag.TAG_SCHEMA["properties"][key]["minLength"] == 1


def _top(**probs: float) -> list[dict[str, Any]]:
    return [{"token": t, "logprob": math.log(p)} for t, p in probs.items()]


def test_highlight_score_is_the_probability_weighted_digit() -> None:
    # The model wrote "3" but nearly split three ways; " " is not a digit.
    top = [*_top(**{"3": 0.36, "4": 0.27, "5": 0.31, " ": 0.06})]
    score = tag.highlight_score(top)
    assert score["highlight_p"] == pytest.approx(
        (3 * 36 + 4 * 27 + 5 * 31) / 94, abs=1e-3
    )
    assert sum(score["highlight_probs"].values()) == pytest.approx(1.0)
    assert score["highlight_probs"]["1"] == 0.0
    with pytest.raises(tag.TagError, match="no highlight digit"):
        tag.highlight_score(_top(**{" ": 0.9, "x": 0.1}))


def test_the_digit_after_highlight_is_found_in_the_token_stream() -> None:
    tokens = [
        {"token": t, "top_logprobs": _top(**{t: 1.0})}
        for t in [
            '{"',
            "description",
            '":',
            ' "',
            "3",
            "个人",
            '",',
            ' "',
            "highlight",
            '":',
            " ",
        ]
    ]
    tokens.append({"token": "4", "top_logprobs": _top(**{"4": 0.5, "3": 0.5})})
    # The "3" in the description is skipped; the digit after "highlight": is read.
    assert tag.highlight_score(tag._highlight_top(tokens))["highlight_p"] == 3.5
    with pytest.raises(tag.TagError, match="no highlight digit to read"):
        tag._highlight_top(tokens[:6])


def test_rescore_replays_the_answer_up_to_the_highlight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    class _Next:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            top = _top(**{"4": 0.6, "5": 0.4})
            return {
                "choices": [
                    {"logprobs": {"content": [{"token": "4", "top_logprobs": top}]}}
                ]
            }

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _Next:
        seen.update(body=json)
        return _Next()

    monkeypatch.setattr(tag.requests, "post", fake_post)
    raw = '{\n  "description": "城堡",\n  "highlight": 4,\n  "issues": []\n}'
    row = {"index": 7, "raw": raw, "failed_attempts": ["loop"]}
    score, _ = tag.rescore(row, [_frame(tmp_path)], 10.0, None)
    assert score["highlight_p"] == pytest.approx(4.4)
    messages = seen["body"]["messages"]
    assert messages[-1] == {"role": "assistant", "content": raw[: raw.index("4,")]}
    assert seen["body"]["max_tokens"] == 1 and "response_format" not in seen["body"]
    # The row's answer came on the second attempt: the same penalty again.
    assert seen["body"]["repeat_penalty"] == tag.REPEAT_PENALTIES[1]
    with pytest.raises(tag.TagError, match="no highlight"):
        tag.rescore({**row, "raw": "{}"}, [_frame(tmp_path)], 10.0, None)
