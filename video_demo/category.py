"""Material categories. Each has a prompt file in prompts/categories/ with
what to look at, the 1-5 highlight rubric, the usual structure of a cut and
the narration style; the 4B picks a video's category from its keyframes."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from video_demo import tag

CATEGORY_DIR = Path(__file__).resolve().parent.parent / "prompts" / "categories"
CATEGORIES = ("city-walk", "nature", "food", "travel-vlog", "people")
SECTIONS = {
    "简介": "intro",
    "看片重点": "focus",
    "精彩度标准": "rubric",
    "成片结构": "structure",
    "解说风格": "narration",
}
# Keyframes the model sees to pick a category, spread over the video.
CLASSIFY_FRAMES = 8
TEMPERATURES = (0.2, 0.5, 0.8)


class CategoryError(RuntimeError):
    """A prompt file or the model's choice of category is unusable."""


@dataclass(frozen=True)
class Profile:
    key: str
    name: str
    intro: str
    focus: str
    rubric: str
    structure: str
    narration: str
    # Changes whenever the prompt file is edited; tags record it.
    digest: str


def load_profile(key: str) -> Profile:
    path = CATEGORY_DIR / f"{key}.prompt.md"
    text = path.read_text("utf-8")
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise CategoryError(f"{path.name}: first line must be '# <name>'")
    found: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines[1:]:
        if line.startswith("## "):
            title = line[3:].strip()
            if title not in SECTIONS:
                raise CategoryError(f"{path.name}: unknown section {title!r}")
            current = found.setdefault(SECTIONS[title], [])
        elif current is not None:
            current.append(line)
    body = {k: "\n".join(v).strip() for k, v in found.items()}
    missing = [t for t, k in SECTIONS.items() if not body.get(k)]
    if missing:
        raise CategoryError(f"{path.name}: missing or empty sections {missing}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return Profile(key, lines[0][2:].strip(), digest=digest, **body)


def load_profiles() -> dict[str, Profile]:
    return {key: load_profile(key) for key in CATEGORIES}


CLASSIFY_PROMPT = """下面 {n} 张图按时间顺序取自同一条视频（{minutes:.1f} 分钟）。
判断这条视频主要属于哪一类素材：
{choices}
输出 JSON：category 是类别名，reason 用一句话说明依据（不超过 40 字）。"""


def classify_schema(profiles: dict[str, Profile]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [p.name for p in profiles.values()],
            },
            "reason": {"type": "string", "minLength": 1, "maxLength": 60},
        },
        "required": ["category", "reason"],
    }


def validate_choice(answer: Any, profiles: dict[str, Profile]) -> dict[str, str]:
    """{"category": key, "reason": ...} from the model's answer."""
    if not isinstance(answer, dict):
        raise CategoryError(f"Expected a JSON object, got {type(answer).__name__}")
    by_name = {p.name: p.key for p in profiles.values()}
    if answer.get("category") not in by_name:
        raise CategoryError(f"Unknown category: {answer.get('category')!r}")
    reason = answer.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise CategoryError("reason must be a non-empty string")
    return {"category": by_name[answer["category"]], "reason": reason.strip()}


def classify(
    keyframes: list[Path],
    minutes: float,
    profiles: dict[str, Profile],
    server_url: str = tag.SERVER_URL,
) -> tuple[dict[str, str], str, float, list[str]]:
    """(category key and reason, raw answer, seconds, failed attempts).

    Raises CategoryError listing every attempt when none validates.
    """
    choices = "\n".join(f"- {p.name}：{p.intro}" for p in profiles.values())
    prompt = CLASSIFY_PROMPT.format(n=len(keyframes), minutes=minutes, choices=choices)
    images = [tag.image_part(p) for p in keyframes]
    errors: list[str] = []
    seconds = 0.0
    for temperature in TEMPERATURES:
        body = {
            "messages": [
                {"role": "system", "content": tag.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, *images],
                },
            ],
            "temperature": temperature,
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "video_category",
                    "schema": classify_schema(profiles),
                },
            },
        }
        t0 = time.perf_counter()
        resp = requests.post(
            f"{server_url}/v1/chat/completions", json=body, timeout=tag.TIMEOUT_SEC
        )
        seconds += time.perf_counter() - t0
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        try:
            return validate_choice(json.loads(raw), profiles), raw, seconds, errors
        except (json.JSONDecodeError, CategoryError) as exc:
            errors.append(f"temperature={temperature}: {exc}")
    raise CategoryError("；".join(errors))
