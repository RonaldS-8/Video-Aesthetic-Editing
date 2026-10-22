"""Static HTML report: segmentation eval, per-segment tags, edit suggestions."""

from __future__ import annotations

import json
import os
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

from video_demo.evaluate import spearman

FLAG_LABELS = {"low_light": "低照度", "blurry": "偏糊", "shaky": "抖动"}
CUT_LABELS = {"hard": "硬切", "jump": "跳切", "dissolve": "叠化"}

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dc;
--keep:#2f7d4f;--drop:#b4442f;--sel:#2b5fd9;--chip:#efeee8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161614;
--card:#1f1f1c;--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;--keep:#5fbf86;
--drop:#e0765f;--sel:#7fa3ff;--chip:#2a2a26}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 12px}
h3{font-size:16px;margin:20px 0 8px}.muted{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;
margin:16px 0}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:12px}.card b{display:block;font-size:20px}
table{width:100%;border-collapse:collapse;background:var(--card);font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;
vertical-align:top}th{color:var(--muted);font-weight:600}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
.bar{position:relative;height:34px;background:var(--chip);border-radius:6px;
overflow:hidden;margin:8px 0 4px}.bar span{position:absolute;top:0;bottom:0;
border-right:1px solid var(--bg)}.bar span.sel{box-shadow:inset 0 0 0 3px var(--sel)}
.thumbs img{width:120px;border-radius:4px;display:block;margin-bottom:3px}
.chip{display:inline-block;background:var(--chip);border-radius:10px;
padding:0 7px;margin:1px 2px 1px 0;font-size:12px}
.keep{color:var(--keep);font-weight:600}.drop{color:var(--drop);font-weight:600}
.clips{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.clip{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:8px;font-size:12px}.clip img{width:100%;border-radius:4px}
video{width:100%;max-width:640px;border-radius:8px;background:#000}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:4px 12px;margin:12px 0}summary{cursor:pointer;font-weight:600;padding:8px 0}
.ok{color:var(--keep)}.miss{color:var(--drop)}
"""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _score_color(score: float, keep: bool) -> str:
    if not keep:
        return "var(--drop)"
    # 2..5 -> light to dark green via opacity.
    alpha = 0.35 + 0.65 * max(0.0, min(1.0, (score - 2) / 3))
    return f"rgba(47,125,79,{alpha:.2f})"


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


def _eval_section(ev: dict) -> str:
    preds = {round(p["time"], 3): p["kind"] for p in ev["predicted"]}
    matched = {round(t, 3): round(p, 3) for p, t in ev["matched"]}
    rows = []
    for cut in ev["truth"]:
        t = round(cut["time"], 3)
        if t in matched:
            p = matched[t]
            status = (
                f'<span class="ok">✓ 检出 {p:.2f}s'
                f"（{CUT_LABELS.get(preds[p], preds[p])}）</span>"
            )
        else:
            status = '<span class="miss">✗ 漏检</span>'
        rows.append(
            f"<tr><td>{t:.2f}s</td><td>{CUT_LABELS[cut['type']]}</td>"
            f"<td>{status}</td></tr>"
        )
    fa = ", ".join(f"{x:.2f}s" for x in ev["false_alarms"]) or "无"
    sources = sorted({c["source"][:2] for c in ev["clips"]})
    return f"""
<h3>{escape(SET_TITLES[ev['set']])}</h3>
<p class="muted">素材 {'/'.join(sources)} · {ev['duration']:.1f}s ·
切点 {len(ev['truth'])} · 容差 ±{ev['tolerance']}s</p>
<div class="cards">
<div class="card"><span class="muted">准确率</span><b>{ev['precision']:.2f}</b></div>
<div class="card"><span class="muted">召回率</span><b>{ev['recall']:.2f}</b></div>
<div class="card"><span class="muted">F1</span><b>{ev['f1']:.2f}</b></div>
<div class="card"><span class="muted">误报</span><b>{escape(fa)}</b></div></div>
<div class="scroll"><table><tr><th>真实切点</th><th>类型</th><th>结果</th></tr>
{''.join(rows)}</table></div>
"""


SET_TITLES = {
    "main": "调参集：03/04/05 一镜到底片段拼接（叠化阈值在此集上确定）",
    "holdout": "留出集：01/02/06 片段拼接，叠化 0.5–1.5 秒（阈值确定后才构建）",
}

EVAL_NOTE = """
<p>结论：硬切（含调参集里的跳切）在两组上全部检出、无误报；叠化只在调参集上检出，
留出集 4 个叠化全部漏检。原因是"中间帧≈前后帧线性混合"的检验假设两段画面基本静止，
手持行走素材里两段都在动，混合残差与普通运镜无法区分。渐变转场需要换学习型检测器
（如 TransNetV2）。漏检的后果是一个片段跨两个场景；8 秒窗口上限限制了影响范围。</p>
"""


def _clip_cards(clips: list[dict], seg_frames: dict[tuple[str, int], str]) -> str:
    cards = []
    for c in clips:
        img = seg_frames.get((c["video"], c["index"]), "")
        cards.append(
            f'<div class="clip"><img src="{escape(img)}" alt="">'
            f"<b>{c['order'] + 1}. {escape(c['role'])}</b> "
            f"<span class='muted'>{c['duration']:.1f}s</span><br>"
            f"<span class='muted'>{escape(c['video'][:24])} "
            f"{_fmt(c['src_start'])}–{_fmt(c['src_end'])}</span><br>"
            f"{escape(c['description'])}<br>"
            f"<span class='muted'>{escape(c['reason'])}</span></div>"
        )
    return f'<div class="clips">{"".join(cards)}</div>'


def _video_section(
    root: Path, out_dir: Path, stem: str, n: int
) -> tuple[str, dict, dict]:
    work = root / "data" / "work" / stem
    seg = _read_json(work / "segments.json")
    sig = _read_json(work / "signals.json")
    tag_rows = [
        json.loads(x) for x in (work / "tags.jsonl").read_text("utf-8").splitlines()
    ]
    tags = {r["index"]: r for r in tag_rows}
    quality = {
        r["index"]: r
        for r in map(
            json.loads, (work / "aesthetic.jsonl").read_text("utf-8").splitlines()
        )
    }
    tl = _read_json(out_dir / "timelines" / f"{stem}.json")
    assess = {a["index"]: a for a in tl["assessments"]}
    selected = {c["index"] for c in tl["clips"]}
    analysed = seg["analysed_until"]
    rel = Path(os.path.relpath(work / "keyframes", out_dir)).as_posix() + "/"
    frames = {
        (seg["video"], s["index"]): rel + sig[s["index"]]["keyframes"][0]
        for s in seg["segments"]
    }

    spans = []
    for s in seg["segments"]:
        a = assess[s["index"]]
        left = 100 * s["start"] / analysed
        width = 100 * (s["end"] - s["start"]) / analysed
        cls = ' class="sel"' if s["index"] in selected else ""
        desc = tags[s["index"]]["tags"]["description"]
        title = f"#{s['index']} {_fmt(s['start'])} {desc}"
        color = _score_color(a["score"], a["keep"])
        spans.append(
            f'<span{cls} title="{escape(title)}" style="left:{left:.3f}%;'
            f'width:{width:.3f}%;background:{color}"></span>'
        )

    rows = []
    for s in seg["segments"]:
        i = s["index"]
        t = tags[i]["tags"]
        a = assess[i]
        q = quality[i]
        thumbs = "".join(f'<img src="{rel}{k}" alt="">' for k in sig[i]["keyframes"])
        chips = "".join(f'<span class="chip">{escape(x)}</span>' for x in t["tags"])
        flags = "、".join(FLAG_LABELS[f] for f in sig[i]["flags"]) or "—"
        verdict = (
            '<span class="keep">保留</span>'
            if a["keep"]
            else '<span class="drop">删除</span>'
        )
        star = " ★入选" if i in selected else ""
        rows.append(
            f"<tr><td>#{i}<br><span class='muted'>镜头 {s['shot']}</span></td>"
            f"<td>{_fmt(s['start'])}–{_fmt(s['end'])}<br>"
            f"<span class='muted'>{s['end'] - s['start']:.1f}s</span></td>"
            f"<td class='thumbs'>{thumbs}</td>"
            f"<td>{escape(t['description'])}<br>{chips}<br><span class='muted'>"
            f"{escape(t['shot_size'])} · {escape(t['time_of_day'])} · "
            f"{escape(t['mood'])} · {escape(t['action'])}</span></td>"
            f"<td><b>{q['fused']:.2f}</b><br><span class='muted'>"
            f"技术 {q['technical']:+.3f}<br>美学 {q['aesthetic']:+.3f}<br>"
            f"VLM {t['highlight']}/5</span></td>"
            f"<td>{escape(flags)}<br><span class='muted'>"
            f"亮度 {sig[i]['brightness']:.0f} 清晰 {sig[i]['sharpness']:.0f} "
            f"抖动 {sig[i]['shake']:.1f}</span></td>"
            f"<td>{verdict}{star}<br>{a['score']:.1f}<br><span class='muted'>"
            f"{escape('；'.join(a['reasons']))}</span></td></tr>"
        )

    kinds = [c["kind"] for c in seg["cuts"]]
    kept = sum(a["keep"] for a in tl["assessments"])
    note = "".join(f"<p class='miss'>{escape(x)}</p>" for x in tl["notes"])
    partial = (
        f"（只分析前 {analysed:.0f} 秒，全长 {_fmt(seg['duration'])}）"
        if analysed < seg["duration"] - 1
        else ""
    )
    audio = "有音轨" if seg["has_audio"] else "无音轨"
    title = escape(seg["video"] + partial)
    html = f"""
<details{' open' if n == 0 else ''}><summary>{title}</summary>
<p class="muted">{_fmt(analysed)} · {seg['fps']:.2f} fps · {audio} ·
切点 {len(kinds)}（硬切 {kinds.count('hard')}，叠化 {kinds.count('dissolve')}）·
镜头 {len(seg['shots'])} · 片段 {len(seg['segments'])} · 保留 {kept} ·
切分耗时 {seg['detect_seconds']}s</p>
<div class="bar">{''.join(spans)}</div>
<p class="muted">色条：绿色越深得分越高，红色为建议删除，
蓝框为入选建议时间线的片段。</p>
<h3>建议时间线（{tl['total_sec']}s / 目标 {tl['target_sec']:.0f}s）</h3>{note}
<video src="previews/{escape(stem)}.mp4" controls preload="metadata"></video>
{_clip_cards(tl['clips'], frames)}
<h3>逐片段标签</h3>
<div class="scroll"><table><tr><th>片段</th><th>时间</th><th>关键帧</th>
<th>模型标签</th><th>美学分 / VLM 精彩度</th><th>画质信号</th><th>建议</th></tr>
{''.join(rows)}</table></div></details>
"""
    stats = {
        "segments": len(seg["segments"]),
        "analysed": analysed,
        "tag_seconds": [r["seconds"] for r in tag_rows],
        "quality_seconds": [r["seconds"] for r in quality.values()],
        "highlight_vs_quality": [
            (tags[i]["tags"]["highlight"], quality[i]["fused"]) for i in sorted(quality)
        ],
        "detect_seconds": seg["detect_seconds"],
        "kept": kept,
    }
    return html, stats, frames


def write(
    root: Path, out_dir: Path, stems: list[str], reel_name: str, baseline: Path
) -> Path:
    """Write ``out_dir/report.html`` from the timelines in ``out_dir``.

    ``baseline`` is the frozen VLM-only report kept for side-by-side review.
    """
    evals = [_read_json(root / "out" / f"synth_eval_{n}.json") for n in SET_TITLES]
    sections: list[str] = []
    all_frames: dict[tuple[str, int], str] = {}
    totals = {
        "segments": 0,
        "analysed": 0.0,
        "tag": [],
        "quality": [],
        "pairs": [],
        "detect": 0.0,
        "kept": 0,
    }
    for n, stem in enumerate(stems):
        html, stats, frames = _video_section(root, out_dir, stem, n)
        sections.append(html)
        all_frames.update(frames)
        totals["segments"] += stats["segments"]
        totals["analysed"] += stats["analysed"]
        totals["tag"] += stats["tag_seconds"]
        totals["quality"] += stats["quality_seconds"]
        totals["pairs"] += stats["highlight_vs_quality"]
        totals["detect"] += stats["detect_seconds"]
        totals["kept"] += stats["kept"]
    reel = _read_json(out_dir / "timelines" / f"{reel_name}.json")
    tag_times = totals["tag"]
    avg_tag = sum(tag_times) / len(tag_times)
    q_times = totals["quality"]
    avg_q = sum(q_times) / len(q_times)
    rho = spearman([h for h, _ in totals["pairs"]], [q for _, q in totals["pairs"]])
    baseline_link = Path(os.path.relpath(baseline, out_dir)).as_posix()
    cards = [
        ("视频", f"{len(stems)}", ""),
        ("分析时长", f"{totals['analysed'] / 60:.1f} 分钟", ""),
        ("片段", f"{totals['segments']}", ""),
        ("建议保留", f"{totals['kept']}", ""),
        ("切分总耗时", f"{totals['detect']:.0f}s", ""),
        ("打标签", f"{avg_tag:.1f}s/段", f"合计 {sum(tag_times) / 60:.0f} 分钟"),
        ("美学打分", f"{avg_q:.1f}s/段", f"合计 {sum(q_times) / 60:.1f} 分钟"),
        ("VLM 精彩度 vs 美学分", f"ρ = {rho:.2f}", "Spearman 秩相关"),
    ]
    card_html = "".join(
        f'<div class="card"><span class="muted">{k}</span><b>{v}</b>'
        f'<span class="muted">{extra}</span></div>'
        for k, v, extra in cards
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>视频切分与剪辑建议（DOVER 美学打分版）</title><style>{CSS}</style></head>
<body><main>
<h1>视频切分与剪辑建议 Demo · DOVER 美学打分版</h1>
<p class="muted">对照：<a href="{escape(baseline_link)}">基线版报告
（VLM 精彩度打分）</a>。两版的切分、标签和画质信号相同，
只有片段得分和由此产生的选片不同。</p>
<p class="muted">{date.today().isoformat()} · 本机 Core Ultra 5 225U / 15.4 GB，无独显 ·
切分 PySceneDetect AdaptiveDetector + 叠化检测 · 打标签 Qwen3-VL-2B Q4_K_M（llama.cpp
Vulkan）· 美学打分 DOVER（CPU，S-Lab 许可，仅内部验证）·
剪辑建议为规则打分 + 槽位填充</p>
<div class="cards">{card_html}</div>
<h2>1. 切分评测（合成视频，切点已知）</h2>
<p class="muted">两层切分：PySceneDetect AdaptiveDetector 找硬切 + 线性混合检验找叠化，
镜头内再按 ≤8 秒切窗口，短于 1 秒的碎片并入相邻片段。</p>
{''.join(_eval_section(ev) for ev in evals)}{EVAL_NOTE}
<h2>2. 跨视频精华合集（{reel['total_sec']}s / 目标 {reel['target_sec']:.0f}s）</h2>
<p class="muted">从全部片段里按得分取素材（1–5 分：DOVER 美学分占 80%、
VLM 精彩度占 20%，再减画质扣分；与文件名和输入顺序无关），
去掉同一镜头相邻窗口和标签高度重合的片段；
按“开场全景 → 主体（白天 → 黄昏 → 夜晚）→ 收尾”排列。</p>
<video src="previews/{escape(reel_name)}.mp4" controls preload="metadata"></video>
{_clip_cards(reel['clips'], all_frames)}
<h2>3. 逐视频结果</h2>
{''.join(sections)}
<p class="muted">素材来自 Wikimedia Commons（CC BY / CC BY-SA 3.0），
署名见 data/raw/SOURCES.md。</p>
</main></body></html>"""
    path = out_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
