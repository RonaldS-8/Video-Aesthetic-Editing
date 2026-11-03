"""Segment tagging with Qwen3.5-4B (with its vision projector) served by
llama-server; the same server writes the summaries and outlines."""

from __future__ import annotations

import base64
import json
import math
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from video_demo.category import Profile

SERVER_URL = "http://127.0.0.1:8092"
TIMEOUT_SEC = 600
# The highlight score is read as the probability-weighted mean of the
# digits the model could write there (G-Eval): the integer it writes hides
# how sure it was, and most segments get the same middle integer.
HIGHLIGHT_DIGITS = (1, 2, 3, 4, 5)
TOP_LOGPROBS = 10
_HIGHLIGHT_KEY = re.compile(r'"highlight"\s*:\s*$')

# Re-ask with a stronger repeat penalty when the model loops; every attempt
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
        "description": {"type": "string", "minLength": 1, "maxLength": 60},
        "scene": {"type": "string", "minLength": 1, "maxLength": 20},
        "subjects": {
            "type": "array",
            "items": {"type": "string", "maxLength": 12},
            "maxItems": 5,
        },
        "action": {"type": "string", "minLength": 1, "maxLength": 20},
        "shot_size": {"type": "string", "enum": SHOT_SIZES},
        "time_of_day": {"type": "string", "enum": TIMES_OF_DAY},
        "mood": {"type": "string", "minLength": 1, "maxLength": 20},
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
    "下面 {count} 张图是同一个视频片段（{duration:.1f} 秒）中按时间先后、"
    "约每 2 秒一张的画面。画面内容如果有变化（换了场景、主体或镜头方向），"
    "description 要概括全部内容，不要只写其中一张。"
    "请输出 JSON：description 用一句话（不超过 40 字）描述片段内容；"
    "scene 是场景；subjects 是主要主体；action 是主体动作或镜头运动；"
    "shot_size 是景别；time_of_day 是时段；mood 是氛围；tags 是 3-8 个检索标签；"
    "highlight 是作为短视频素材的精彩程度 1-5（5 最精彩）；"
    "issues 是画面问题，没有问题填 无明显问题。"
)


# Asked on top of USER_PROMPT when the video's category is known: its focus
# and highlight rubric, and the fields a cut's opening and narration use.
CATEGORY_PROMPT = (
    "\n这类素材是「{name}」。看片时注意：\n{focus}\n"
    "另外输出：landmark 是画面里能确认的地标或地名，认不出就填空字符串；"
    "light 是光线；composition 是构图特点；opening 表示这个片段适不适合"
    "做成片的开场镜头；ending 表示适不适合做收尾镜头。\n"
    "最后按下面的标准给 highlight 打分：\n{rubric}\n"
    "一段素材里多数片段是 3 分，4 分是少数好镜头，5 分很少，"
    "平淡或有瑕疵的给 2 分；对照标准逐档判断，不要都给中间分。"
)
CATEGORY_FIELDS: dict[str, Any] = {
    "landmark": {"type": "string", "maxLength": 20},
    "light": {"type": "string", "minLength": 1, "maxLength": 12},
    "composition": {"type": "string", "minLength": 1, "maxLength": 16},
    "opening": {"type": "boolean"},
    "ending": {"type": "boolean"},
}


def tag_schema(profile: Profile | None = None) -> dict[str, Any]:
    """TAG_SCHEMA, plus CATEGORY_FIELDS when tagging with a category."""
    if profile is None:
        return TAG_SCHEMA
    # The model writes fields in schema order: what it noticed (landmark,
    # light, composition, opening, ending) comes before the score.
    base = TAG_SCHEMA["properties"]
    first = {k: v for k, v in base.items() if k not in ("highlight", "issues")}
    properties = {
        **first,
        **CATEGORY_FIELDS,
        "highlight": base["highlight"],
        "issues": base["issues"],
    }
    return {
        **TAG_SCHEMA,
        "properties": json.loads(json.dumps(properties)),
        "required": [*TAG_SCHEMA["required"], *CATEGORY_FIELDS],
    }


def user_prompt(count: int, duration: float, profile: Profile | None = None) -> str:
    text = USER_PROMPT.format(count=count, duration=duration)
    if profile is not None:
        text += CATEGORY_PROMPT.format(
            name=profile.name, focus=profile.focus, rubric=profile.rubric
        )
    return text


class TagError(RuntimeError):
    """Raised when the model output does not satisfy the tag schema."""


def validate(tags: Any, profile: Profile | None = None) -> dict[str, Any]:
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
    if profile is not None:
        missing = [k for k in CATEGORY_FIELDS if k not in tags]
        if missing:
            raise TagError(f"Missing keys: {missing}")
        for key in ("landmark", "light", "composition"):
            if not isinstance(tags[key], str):
                raise TagError(f"{key} must be a string")
            tags[key] = tags[key].strip()
        for key in ("light", "composition"):
            if not tags[key]:
                raise TagError(f"{key} must be a non-empty string")
        for key in ("opening", "ending"):
            if not isinstance(tags[key], bool):
                raise TagError(f"{key} must be true or false")
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


def image_part(path: Path) -> dict[str, Any]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def highlight_score(top: list[dict[str, Any]]) -> dict[str, Any]:
    """{"highlight_p": mean, "highlight_probs": {digit: p}} from the top
    log-probabilities at the position where the highlight digit is written,
    renormalised over HIGHLIGHT_DIGITS."""
    probs = dict.fromkeys(HIGHLIGHT_DIGITS, 0.0)
    for alt in top:
        token = alt["token"].strip()
        if token.isdigit() and int(token) in probs:
            probs[int(token)] += math.exp(alt["logprob"])
    total = sum(probs.values())
    if total <= 0:
        raise TagError(f"no highlight digit among the top tokens: {top}")
    shares = {d: p / total for d, p in probs.items()}
    return {
        "highlight_p": round(sum(d * p for d, p in shares.items()), 3),
        "highlight_probs": {str(d): round(p, 4) for d, p in shares.items()},
    }


def _highlight_top(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The top log-probabilities of the digit written after "highlight":
    in a tagging answer's token stream."""
    text = ""
    for t in tokens:
        if t["token"].strip().isdigit() and _HIGHLIGHT_KEY.search(text):
            return t["top_logprobs"]
        text += t["token"]
    raise TagError("the answer has no highlight digit to read probabilities from")


def tag_segment(
    keyframes: list[Path],
    duration: float,
    server_url: str = SERVER_URL,
    profile: Profile | None = None,
) -> tuple[dict[str, Any], str, float, list[str], dict[str, Any]]:
    """Return (validated tags, raw text, total seconds, errors of failed
    tries, highlight score from highlight_score); with ``profile`` the
    category's focus and rubric are asked too.

    Raises the last TagError when every attempt fails validation.
    """
    images = [image_part(p) for p in keyframes]
    errors: list[str] = []
    total = 0.0
    for penalty in REPEAT_PENALTIES:
        raw, elapsed, tokens = _request(images, duration, penalty, server_url, profile)
        total += elapsed
        try:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TagError(f"Model output is not JSON: {raw[:200]!r}") from exc
            tags = validate(parsed, profile)
            score = highlight_score(_highlight_top(tokens))
            return tags, raw, total, errors, score
        except TagError as exc:
            errors.append(f"repeat_penalty={penalty}: {exc}")
            last = exc
    raise last


def _request(
    images: list[dict[str, Any]],
    duration: float,
    penalty: float,
    server_url: str,
    profile: Profile | None,
) -> tuple[str, float, list[dict[str, Any]]]:
    """(answer, seconds, its tokens with their top log-probabilities)."""
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt(len(images), duration, profile),
                    },
                    *images,
                ],
            },
        ],
        "temperature": 0.2,
        "repeat_penalty": penalty,
        "max_tokens": 500,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        # Qwen3.5 thinks before answering by default and would spend
        # max_tokens on it, leaving the answer empty.
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "segment_tags", "schema": tag_schema(profile)},
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(
        f"{server_url}/v1/chat/completions", json=body, timeout=TIMEOUT_SEC
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    if not choice.get("logprobs"):
        raise TagError("the server returned no log-probabilities")
    return choice["message"]["content"], elapsed, choice["logprobs"]["content"]


def rescore(
    row: dict[str, Any],
    keyframes: list[Path],
    duration: float,
    profile: Profile | None,
    server_url: str = SERVER_URL,
) -> tuple[dict[str, Any], float]:
    """The highlight score of a tag row made before scores were recorded:
    the tagging request is sent again with the row's own answer written up
    to "highlight": and one more token asked for, so the model is in the
    same place as when it chose the digit. Returns (highlight_score result,
    seconds)."""
    match = re.search(r'"highlight"\s*:\s*', row["raw"])
    if match is None:
        raise TagError(f"segment {row['index']}: its answer has no highlight")
    images = [image_part(p) for p in keyframes]
    penalty = REPEAT_PENALTIES[len(row.get("failed_attempts", []))]
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt(len(images), duration, profile),
                    },
                    *images,
                ],
            },
            {"role": "assistant", "content": row["raw"][: match.end()]},
        ],
        "temperature": 0.2,
        "repeat_penalty": penalty,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    resp = requests.post(
        f"{server_url}/v1/chat/completions", json=body, timeout=TIMEOUT_SEC
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    if not choice.get("logprobs"):
        raise TagError("the server returned no log-probabilities")
    return highlight_score(choice["logprobs"]["content"][0]["top_logprobs"]), elapsed
