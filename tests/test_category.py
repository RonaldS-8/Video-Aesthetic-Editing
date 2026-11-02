"""Material categories: prompt files, the model's choice, category tags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_tag import GOOD, _frame, _Resp

from video_demo import category, tag
from video_demo.category import CategoryError


def test_every_category_has_a_complete_prompt_file() -> None:
    profiles = category.load_profiles()
    assert list(profiles) == list(category.CATEGORIES)
    city = profiles["city-walk"]
    assert city.name == "城市漫步"
    assert "5：" in city.rubric and "1：" in city.rubric
    assert "定场" in city.structure and city.narration
    assert len({p.digest for p in profiles.values()}) == len(profiles)


def test_a_prompt_file_missing_a_section_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bad.prompt.md").write_text(
        "# 坏的\n\n## 简介\n\nx\n\n## 看片重点\n\ny\n", "utf-8"
    )
    monkeypatch.setattr(category, "CATEGORY_DIR", tmp_path)
    with pytest.raises(CategoryError, match="missing or empty"):
        category.load_profile("bad")
    (tmp_path / "odd.prompt.md").write_text("# 怪\n## 别的\nx\n", "utf-8")
    with pytest.raises(CategoryError, match="unknown section"):
        category.load_profile("odd")


@pytest.mark.parametrize(
    "answer, message",
    [
        ({"category": "宠物", "reason": "猫"}, "Unknown category"),
        ({"category": "城市漫步", "reason": " "}, "reason"),
        (["城市漫步"], "JSON object"),
    ],
)
def test_validate_choice_rejects(answer: Any, message: str) -> None:
    with pytest.raises(CategoryError, match=message):
        category.validate_choice(answer, category.load_profiles())


def test_classify_retries_then_maps_the_name_to_its_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answers = iter(
        [
            json.dumps({"category": "宠物", "reason": "x"}, ensure_ascii=False),
            json.dumps({"category": "城市漫步", "reason": "街道"}, ensure_ascii=False),
        ]
    )
    seen: list[dict] = []

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _Resp:
        seen.append(json)
        return _Resp(next(answers))

    monkeypatch.setattr(category.requests, "post", fake_post)
    choice, _, _, errors = category.classify(
        [_frame(tmp_path)] * 3, 67.0, category.load_profiles()
    )
    assert choice == {"category": "city-walk", "reason": "街道"}
    assert len(errors) == 1 and "Unknown category" in errors[0]
    prompt = seen[0]["messages"][1]["content"][0]["text"]
    assert "- 城市漫步：" in prompt and "67.0 分钟" in prompt


CITY = {
    **GOOD,
    "landmark": " 圣吉尔斯大教堂 ",
    "light": "顺光",
    "composition": "仰拍尖塔",
    "opening": True,
    "ending": False,
}


def test_category_tags_add_fields_and_the_rubric(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = category.load_profile("city-walk")
    seen: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _Resp:
        seen.update(body=json)
        return _Resp(__import__("json").dumps(CITY, ensure_ascii=False))

    monkeypatch.setattr(tag.requests, "post", fake_post)
    tags, _, _, _ = tag.tag_segment([_frame(tmp_path)], 10.0, profile=profile)
    assert tags["landmark"] == "圣吉尔斯大教堂" and tags["opening"] is True
    text = seen["body"]["messages"][1]["content"][0]["text"]
    assert "「城市漫步」" in text and profile.rubric in text
    schema = seen["body"]["response_format"]["json_schema"]["schema"]
    assert {"landmark", "opening", "ending"} <= set(schema["required"])
    assert "landmark" not in tag.TAG_SCHEMA["properties"]


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"opening": "是"}, "true or false"),
        ({"light": " "}, "light must be"),
        ({"landmark": None}, "landmark must be"),
    ],
)
def test_category_tags_are_checked(patch: dict[str, Any], message: str) -> None:
    profile = category.load_profile("city-walk")
    with pytest.raises(tag.TagError, match=message):
        tag.validate({**CITY, **patch}, profile)
    missing = {k: v for k, v in CITY.items() if k != "ending"}
    with pytest.raises(tag.TagError, match="Missing keys"):
        tag.validate(missing, profile)
    # Without a category the extra fields are not required.
    assert tag.validate(dict(GOOD)) == GOOD
