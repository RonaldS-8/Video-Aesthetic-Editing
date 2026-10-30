from __future__ import annotations

import json
from pathlib import Path

from video_demo import layout, report


def _timeline() -> dict:
    return {
        "scenes": [
            {
                "number": 1,
                "video": "v",
                "start": 0.0,
                "end": 20.0,
                "segments": [["v", 0], ["v", 1]],
            },
            {
                "number": 2,
                "video": "v",
                "start": 20.0,
                "end": 40.0,
                "segments": [["v", 2], ["v", 3]],
            },
        ],
        "slots": [
            [
                {
                    "query": "城堡",
                    "scene": 2,
                    "relevance": 1.0,
                    "seconds": 10.0,
                    "top": [2, 1],
                },
                {
                    "query": "下雨",
                    "scene": None,
                    "relevance": None,
                    "seconds": 0.0,
                    "top": [],
                },
            ]
        ],
        "outline": {
            "parts": [
                {
                    "name": "p",
                    "summary": "s",
                    "first_chunk": 1,
                    "last_chunk": 1,
                    "weight": 5,
                    "duration_sec": 20,
                }
            ]
        },
        "assessments": [
            {"video": "v", "index": i, "keep": i != 3, "score": 3.0} for i in range(4)
        ],
        "clips": [{"video": "v", "index": 2, "role": "p", "duration": 10.0}],
    }


def test_scene_slots_shows_each_slot_and_its_scene() -> None:
    frames = {("v", i): f"k{i}.jpg" for i in range(4)}
    html = report.scene_slots(_timeline(), frames, {("v", 2): "城堡远景"})
    assert "2 个场景，1 个部分，2 个槽位" in html
    assert "「城堡」</b> → 场景 2" in html and "取了 1 段" in html
    assert '<img class="taken" src="k2.jpg" title="#2 城堡远景"' in html
    assert '<img class="dropped" src="k3.jpg"' in html
    assert "其他候选：场景 1（0:00.0）" in html
    assert '「下雨」</b><p class="miss">没有可用的场景' in html
    # 10 s of a 20 s target is reported short.
    assert "<span class='miss'>选到 10s</span>" in html
    assert 'title="场景 2 → 「城堡」 0:20.0–0:40.0"' in html


def test_index_links_every_collection_result(tmp_path: Path) -> None:
    out = tmp_path / "out"
    vlog = out / "vlog"
    (vlog / "01_trip" / "scenes").mkdir(parents=True)
    (vlog / "01_trip" / "scenes" / "scenes.json").write_text(
        json.dumps({"scenes": [{}, {}, {}]}), "utf-8"
    )
    layout.tags_copy_path(vlog / "01_trip").write_text("{}", "utf-8")
    layout.report_path(vlog, "highlights").write_text("", "utf-8")
    for stamp, instruction in (("20261028-1", "旧<一>"), ("20261029-1", "只要夜景")):
        folder = layout.edit_dir(vlog, stamp, instruction)
        folder.mkdir(parents=True)
        layout.edit_report_path(folder).write_text("", "utf-8")
        layout.edit_timeline_path(folder).write_text(
            json.dumps(
                {
                    "created": stamp,
                    "instruction": instruction,
                    "outline": {"title": "夜色"},
                    "total_sec": 41.25,
                    "target_sec": 45.0,
                }
            ),
            "utf-8",
        )
    (out / "long").mkdir()
    layout.baseline_report_path(out).parent.mkdir()
    layout.baseline_report_path(out).write_text("", "utf-8")

    html = report.write_index(out).read_text("utf-8")
    assert "<h2>vlog</h2>" in html
    assert "<h2>long</h2>" not in html and "<h2>baseline</h2>" not in html
    assert '<a href="vlog/report_highlights.html">报告</a>' in html
    assert "按指令剪辑（2 次）" in html
    assert html.index("只要夜景") < html.index("旧&lt;一&gt;")
    assert "41.2s / 45s" in html or "41.3s / 45s" in html
    assert '<a href="vlog/edits/20261029-1_只要夜景/report.html">报告</a>' in html
    assert '<a href="vlog/01_trip/scenes">目录</a>（3 个）' in html
    assert '<a href="vlog/01_trip/tags.jsonl">tags.jsonl</a>' in html
    assert '<a href="baseline/report.html">报告</a>' in html
