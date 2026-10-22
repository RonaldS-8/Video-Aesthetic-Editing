"""Segment tagging with a local Qwen3-VL-2B served by llama-server."""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

SERVER_URL = "http://127.0.0.1:8091"
TIMEOUT_SEC = 600
# Re-ask with a stronger repeat penalty when the 2B model loops; every attempt
# is recorded and the last error is raised if none pass validation.
REPEAT_PENALTIES = (1.1, 1.2, 1.3)
# A 2-6 character unit repeated 4+ times in a row ("白天的白天的白天的...").
_LOOP = re.compile(r"(.{2,6})\1{3,}")

SHOT_SIZES = ["远景", "全景", "中景", "近景", "特写"]
TIMES_OF_DAY = ["白天", "黄昏", "夜晚", "室内", "不确定"]
ISSUES = ["模糊", "过暗", "过曝", "抖动", "遮挡", "无明显问题"]

TAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "maxLength": 60},
        "scene": {"type": "string", "maxLength": 20},
        "subjects": {
            "type": "array",
            "items": {"type": "string", "maxLength": 12},
            "maxItems": 5,
        },
        "action": {"type": "string", "maxLength": 20},
        "shot_size": {"type": "string", "enum": SHOT_SIZES},
        "time_of_day": {"type": "string", "enum": TIMES_OF_DAY},
        "mood": {"type": "string", "maxLength": 20},
        "tags": {
            "type": "array",
            "items": {"type": "string", "maxLength": 12},
            "minItems": 3,
            "maxItems": 8,
        },
        "highlight": {"type": "integer", "minimum": 1, "maximum": 5},
        "issues": {"type": "array", "items": {"type": "string", "enum": ISSUES}},
    },
    "required": [
        "description",
        "scene",
        "subjects",
        "action",
        "shot_size",
        "time_of_day",
        "mood",
        "tags",
        "highlight",
        "issues",
    ],
}

SYSTEM_PROMPT = (
    "你是视频剪辑助理，负责给素材片段打标签。只根据画面内容回答，"
    "不要猜测画面里看不到的信息。所有文字用简体中文。"
)

USER_PROMPT = (
    "下面两张图是同一个视频片段（{duration:.1f} 秒）中先后两个时刻的画面。"
    "请输出 JSON：description 用一句话（不超过 40 字）描述片段内容；"
    "scene 是场景；subjects 是主要主体；action 是主体动作或镜头运动；"
    "shot_size 是景别；time_of_day 是时段；mood 是氛围；tags 是 3-8 个检索标签；"
    "highlight 是作为短视频素材的精彩程度 1-5（5 最精彩）；"
    "issues 是画面问题，没有问题填 无明显问题。"
)


class TagError(RuntimeError):
    """Raised when the model output does not satisfy the tag schema."""


def validate(tags: Any) -> dict[str, Any]:
    if not isinstance(tags, dict):
        raise TagError(f"Expected a JSON object, got {type(tags).__name__}")
    missing = [k for k in TAG_SCHEMA["required"] if k not in tags]
    if missing:
        raise TagError(f"Missing keys: {missing}")
    for key in ("description", "scene", "action", "mood"):
        if not isinstance(tags[key], str) or not tags[key].strip():
            raise TagError(f"{key} must be a non-empty string")
        if _LOOP.search(tags[key]):
            raise TagError(f"{key} is a repetition loop: {tags[key][:40]!r}")
    for key in ("subjects", "tags", "issues"):
        if not isinstance(tags[key], list) or not all(
            isinstance(x, str) for x in tags[key]
        ):
            raise TagError(f"{key} must be a list of strings")
    if not 3 <= len(tags["tags"]) <= 8:
        raise TagError(f"tags must have 3-8 items, got {len(tags['tags'])}")
    if tags["shot_size"] not in SHOT_SIZES:
        raise TagError(f"Invalid shot_size: {tags['shot_size']}")
    if tags["time_of_day"] not in TIMES_OF_DAY:
        raise TagError(f"Invalid time_of_day: {tags['time_of_day']}")
    bad_issues = [x for x in tags["issues"] if x not in ISSUES]
    if bad_issues:
        raise TagError(f"Invalid issues: {bad_issues}")
    highlight = tags["highlight"]
    if isinstance(highlight, bool) or not isinstance(highlight, int):
        raise TagError("highlight must be an integer")
    if not 1 <= highlight <= 5:
        raise TagError(f"highlight out of range: {highlight}")
    return tags


def check_server(server_url: str = SERVER_URL) -> None:
    """Raise if llama-server is not up and ready (``GET /health`` == 200)."""
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
    except requests.RequestException as exc:
        raise ConnectionError(f"无法连接 llama-server（{server_url}）：{exc}") from exc
    if resp.status_code != 200:
        raise ConnectionError(
            f"llama-server 未就绪（{server_url}/health 返回 {resp.status_code}），"
            "模型可能还在加载"
        )


def _image_part(path: Path) -> dict[str, Any]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def tag_segment(
    keyframes: list[Path], duration: float, server_url: str = SERVER_URL
) -> tuple[dict[str, Any], str, float, list[str]]:
    """Return (validated tags, raw text, total seconds, errors of failed tries).

    Raises the last TagError when every attempt fails validation.
    """
    images = [_image_part(p) for p in keyframes]
    errors: list[str] = []
    total = 0.0
    for penalty in REPEAT_PENALTIES:
        raw, elapsed = _request(images, duration, penalty, server_url)
        total += elapsed
        try:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TagError(f"Model output is not JSON: {raw[:200]!r}") from exc
            return validate(parsed), raw, total, errors
        except TagError as exc:
            errors.append(f"repeat_penalty={penalty}: {exc}")
            last = exc
    raise last


def _request(
    images: list[dict[str, Any]], duration: float, penalty: float, server_url: str
) -> tuple[str, float]:
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT.format(duration=duration)},
                    *images,
                ],
            },
        ],
        "temperature": 0.2,
        "repeat_penalty": penalty,
        "max_tokens": 400,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "segment_tags", "schema": TAG_SCHEMA},
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(
        f"{server_url}/v1/chat/completions", json=body, timeout=TIMEOUT_SEC
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"], elapsed
