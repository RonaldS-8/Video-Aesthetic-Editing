"""Static HTML report: segmentation eval, per-segment tags, edit suggestions."""

from __future__ import annotations

import json
import os
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

from video_demo import layout
from video_demo.evaluate import spearman
from video_demo.segment import MAX_SEGMENT_SEC
from video_demo.signals import KEYFRAME_STEP_SEC

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
.part{border-left:4px solid;padding:2px 0 2px 12px;margin:18px 0}
.part h3{margin:4px 0}.slot{background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;margin:8px 0}
.strip{display:flex;gap:4px;overflow-x:auto;margin:6px 0}
.strip img{width:112px;flex:none;border-radius:4px;border:3px solid transparent}
.strip img.taken{border-color:var(--sel)}.strip img.dropped{opacity:.35}
pre.raw{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:10px 12px;overflow-x:auto;font-size:12px;white-space:pre-wrap}
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


def _clip_cards(
    clips: list[dict],
    seg_frames: dict[tuple[str, int], str],
    files: dict[str, str],
) -> str:
    """``files`` maps video ids to raw file names for the captions."""
    cards = []
    for c in clips:
        img = seg_frames.get((c["video"], c["index"]), "")
        cards.append(
            f'<div class="clip"><img src="{escape(img)}" alt="">'
            f"<b>{c['order'] + 1}. {escape(c['role'])}</b> "
            f"<span class='muted'>{c['duration']:.1f}s</span><br>"
            f"<span class='muted'>{escape(files[c['video']][:24])} "
            f"{_fmt(c['src_start'])}–{_fmt(c['src_end'])}</span><br>"
            f"{escape(c['description'])}<br>"
            f"<span class='muted'>{escape(c['reason'])}</span></div>"
        )
    return f'<div class="clips">{"".join(cards)}</div>'


def keyframe_paths(
    work: Path, vid: str, sig: list[dict], out_dir: Path
) -> dict[tuple[str, int], str]:
    """Each segment's first keyframe, relative to ``out_dir``; ``sig`` is
    the video's signal records."""
    rel = _rel(layout.keyframes_dir(work, vid), out_dir) + "/"
    return {(vid, s["index"]): rel + s["keyframes"][0] for s in sig}


def _video_section(
    root: Path, out_dir: Path, result_dir: Path, mode: str, video: layout.Video, n: int
) -> tuple[str, dict, dict]:
    work = root / "data" / "work"
    vid = video.id
    seg = _read_json(layout.segments_path(work, vid))
    sig = _read_json(layout.signals_path(work, vid))
    tag_rows = [
        json.loads(x)
        for x in layout.tags_path(work, vid).read_text("utf-8").splitlines()
    ]
    tags = {r["index"]: r for r in tag_rows}
    quality = {
        r["index"]: r
        for r in map(
            json.loads,
            layout.aesthetic_path(work, vid).read_text("utf-8").splitlines(),
        )
    }
    tl = _read_json(layout.timeline_path(result_dir, mode))
    assess = {a["index"]: a for a in tl["assessments"]}
    selected = {c["index"] for c in tl["clips"]}
    analysed = seg["analysed_until"]
    rel = _rel(layout.keyframes_dir(work, vid), out_dir) + "/"
    frames = keyframe_paths(work, vid, sig, out_dir)

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
    title = escape(f"{video.file}（{vid}）{partial}")
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
{outline_table(tl)}
<video src="{escape(_rel(layout.preview_path(result_dir, mode), out_dir))}" controls
preload="metadata"></video>
{_clip_cards(tl['clips'], frames, {vid: video.file})}
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
        "descriptions": {(vid, i): r["tags"]["description"] for i, r in tags.items()},
    }
    return html, stats, frames


def summary_heading(mode: str, timeline: dict) -> str:
    """Section 2 title and rules for the collection-wide timeline."""
    total, target = timeline["total_sec"], timeline["target_sec"]
    minutes = f"全集合 {total / 60:.1f} 分钟 / 目标 {target / 60:.1f} 分钟"
    if mode == "highlights":
        return f"""<h2>2. 精彩瞬间（{minutes}）</h2>
<p class="muted">按时间每 5 分钟一块（短视频各自总结后，整条凑满 5 分钟成一组），
Qwen3.5-4B 先逐块写总结（发生了什么、地点、活动、值得留的片段），再读全部总结
生成大纲：每部分的标题、内容、覆盖的块和 1–4 个场景槽位。视频按
{escape(timeline.get('order', '文件名'))}排列。相邻片段按向量相似度连成场景
（20–120 秒，内容相近时跨镜头、跨视频）；每个槽位在本部分的
块里检索（文本向量、图像向量、BM25 融合），选一个场景，相关度为主、画质为辅，
一个场景只给一个槽位。槽位的时长从这个场景里取：按片段得分（DOVER 美学分占
80%、VLM 精彩度占 20%，美学分在同一时段内取百分位）加片段相关度、再给总结
点名的亮点加分，整段选片，60 秒内标签高度重合的只留一段；按拍摄时间排列。</p>
{outline_table(timeline)}"""
    if mode == "ranked":
        return f"""<h2>2. 精彩瞬间 · 只按得分（{minutes}）</h2>
<p class="muted">对照基线，不用大纲：以切分片段为单位整段选入（1–5 分：DOVER 美学分占
80%、VLM 精彩度占 20%，再减画质扣分；美学分先在同一时段内取百分位）；
每 5 分钟至少选一段，60 秒内标签高度重合的只留一段；按拍摄时间排列。</p>"""
    raise ValueError(f"Unknown mode: {mode}")


def outline_table(timeline: dict) -> str:
    """The model's outline (parts, chunks, target and filled length) and the
    chunk summaries it was written from."""
    plan = timeline.get("outline")
    if not plan:
        return ""
    chunks = {ch["number"]: ch for ch in timeline["chunks"]}
    scenes = {sc["number"]: sc for sc in timeline["scenes"]}

    def slot_text(slot: dict) -> str:
        if slot["scene"] is None:
            return f"{escape(slot['query'])} → 未找到"
        sc = scenes[slot["scene"]]
        return (
            f"{escape(slot['query'])} → 场景 {sc['number']}<span class='muted'>"
            f"（{_fmt(sc['start'])}–{_fmt(sc['end'])}，{slot['relevance']:.0%}，"
            f"{slot['seconds']:.0f}s）</span>"
        )

    def chunk_span(ch: dict) -> str:
        many = f"<br>{len(ch['videos'])} 条视频" if len(ch["videos"]) > 1 else ""
        return f"{_fmt(ch['start'])}–{_fmt(ch['end'])}{many}"

    filled: dict[str, float] = {}
    for c in timeline["clips"]:
        filled[c["role"]] = filled.get(c["role"], 0.0) + c["duration"]
    rows = "".join(
        f"<tr><td>{k}. {escape(p['name'])}</td><td>{escape(p['summary'])}</td>"
        f"<td>块 {p['first_chunk']}–{p['last_chunk']}<br><span class='muted'>"
        f"{_fmt(chunks[p['first_chunk']]['start'])}–"
        f"{_fmt(chunks[p['last_chunk']]['end'])}</span></td>"
        f"<td>{p['duration_sec']}s</td><td>{filled.get(p['name'], 0.0):.1f}s</td>"
        f"<td>{'<br>'.join(map(slot_text, slots))}</td></tr>"
        for k, (p, slots) in enumerate(
            zip(plan["parts"], timeline["slots"], strict=True), start=1
        )
    )
    summary_rows = "".join(
        f"<tr><td>{ch['number']}</td><td>{chunk_span(ch)}</td>"
        f"<td>{escape(ch['summary']['summary'])}<br><span class='muted'>"
        f"{escape('、'.join(ch['summary']['places']))} · "
        f"{escape('、'.join(ch['summary']['activities']))}</span></td>"
        f"<td>{escape('；'.join(h['why'] for h in ch['summary']['highlights'])) or '—'}"
        "</td></tr>"
        for ch in timeline["chunks"]
    )
    tries = len(timeline.get("outline_failed_attempts", []))
    retry = f"，重试 {tries} 次" if tries else ""
    return f"""<p><b>大纲：{escape(plan.get('title', ''))}</b>
<span class="muted">（由 {len(timeline['chunks'])} 块的总结生成，用时
{timeline.get('outline_seconds', 0)}s{retry}）</span></p>
<div class="scroll"><table><tr><th>部分</th><th>内容</th><th>覆盖</th>
<th>目标</th><th>选到</th><th>场景槽位 → 检索到的场景</th></tr>{rows}</table></div>
<details><summary>分段总结（{len(timeline['chunks'])} 块）</summary>
<div class="scroll"><table><tr><th>块</th><th>时间</th><th>总结 · 地点 · 活动</th>
<th>亮点</th></tr>{summary_rows}</table></div></details>"""


PART_COLORS = ("#2b5fd9", "#d9822b", "#2f9d8f", "#b04aa8", "#8a9a2b", "#c4473a")


def _scene_strip(
    sc: dict,
    frames: dict[tuple[str, int], str],
    descriptions: dict[tuple[str, int], str],
    taken: set[tuple[str, int]],
    keep: dict[tuple[str, int], bool],
) -> str:
    """Every segment keyframe of a scene; taken ones framed, rejected faded."""
    imgs = []
    for vid, i in sc["segments"]:
        key = (vid, i)
        cls = "taken" if key in taken else ("" if keep[key] else "dropped")
        title = f"#{i} {descriptions.get(key, '')}"
        imgs.append(
            f'<img class="{cls}" src="{escape(frames[key])}" '
            f'title="{escape(title)}" alt="">'
        )
    return f'<div class="strip">{"".join(imgs)}</div>'


def scene_slots(
    timeline: dict,
    frames: dict[tuple[str, int], str],
    descriptions: dict[tuple[str, int], str],
) -> str:
    """The outline by scene: a map of all scenes, then per part each slot,
    the scene retrieved for it (every keyframe, the clips taken marked)
    and the runner-up scenes."""
    scenes = {sc["number"]: sc for sc in timeline["scenes"]}
    keep = {(a["video"], a["index"]): a["keep"] for a in timeline["assessments"]}
    taken = {(c["video"], c["index"]) for c in timeline["clips"]}
    first, last = timeline["scenes"][0]["start"], timeline["scenes"][-1]["end"]
    owner: dict[int, tuple[int, str]] = {}  # scene -> (part, slot query)
    for k, slots in enumerate(timeline["slots"]):
        for slot in slots:
            if slot["scene"] is not None:
                owner[slot["scene"]] = (k, slot["query"])

    spans = []
    for sc in timeline["scenes"]:
        left = 100 * (sc["start"] - first) / (last - first)
        width = 100 * (sc["end"] - sc["start"]) / (last - first)
        if sc["number"] in owner:
            k, query = owner[sc["number"]]
            color = PART_COLORS[k % len(PART_COLORS)]
            title = f"场景 {sc['number']} → 「{query}」"
        else:
            color = "var(--chip)" if sc["number"] % 2 else "var(--line)"
            title = f"场景 {sc['number']}"
        title += f" {_fmt(sc['start'])}–{_fmt(sc['end'])}"
        spans.append(
            f'<span title="{escape(title)}" style="left:{left:.3f}%;'
            f'width:{width:.3f}%;background:{color}"></span>'
        )

    def strip(sc: dict) -> str:
        return _scene_strip(sc, frames, descriptions, taken, keep)

    parts = []
    for k, (part, slots) in enumerate(
        zip(timeline["outline"]["parts"], timeline["slots"], strict=True)
    ):
        color = PART_COLORS[k % len(PART_COLORS)]
        got = sum(c["duration"] for c in timeline["clips"] if c["role"] == part["name"])
        short = got < 0.8 * part["duration_sec"]
        cards = []
        for slot in slots:
            head = f"<b>「{escape(slot['query'])}」</b>"
            if slot["scene"] is None:
                cards.append(
                    f'<div class="slot">{head}<p class="miss">没有可用的场景</p></div>'
                )
                continue
            sc = scenes[slot["scene"]]
            n_taken = sum(tuple(k) in taken for k in sc["segments"])
            others = "、".join(
                f"场景 {n}（{_fmt(scenes[n]['start'])}）"
                for n in slot["top"]
                if n != slot["scene"]
            )
            cards.append(
                f'<div class="slot">{head} → 场景 {sc["number"]}'
                f"<span class='muted'>（{_fmt(sc['start'])}–{_fmt(sc['end'])}，"
                f"{sc['end'] - sc['start']:.0f}s，{len(sc['segments'])} 段）</span><br>"
                f"<span class='muted'>检索相关度 {slot['relevance']:.0%} · "
                f"分到 {slot['seconds']:.0f}s · 取了 {n_taken} 段</span>"
                f"{strip(sc)}"
                f"<span class='muted'>其他候选：{others or '—'}</span></div>"
            )
        parts.append(
            f'<div class="part" style="border-left-color:{color}">'
            f"<h3>{k + 1}. {escape(part['name'])} "
            f"<span class='muted'>块 {part['first_chunk']}–{part['last_chunk']} · "
            f"分量 {part['weight']} · 目标 {part['duration_sec']}s · "
            f"<span class='{'miss' if short else 'ok'}'>选到 {got:.0f}s</span>"
            f"</span></h3><p class='muted'>{escape(part['summary'])}</p>"
            f"{''.join(cards)}</div>"
        )
    n_slots = sum(len(s) for s in timeline["slots"])
    counts = f"{len(timeline['scenes'])} 个场景，{len(parts)} 个部分，{n_slots} 个槽位"
    return f"""<h3>按场景的大纲：{counts}</h3>
<div class="bar">{''.join(spans)}</div>
<p class="muted">色条是整段素材的场景划分（相邻片段按向量相似度连成场景）；
着色的是被槽位选中的场景，颜色对应下面的部分，悬停看场景编号和对应槽位。
每个槽位下是所选场景的全部片段关键帧：蓝框为取用的片段，变淡的是画质不合格、
建议删除的片段。</p>
{''.join(parts)}"""


def _rel(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def write(
    root: Path,
    out_dir: Path,
    videos: list[layout.Video],
    result_dirs: dict[str, Path],
    baseline: Path,
    mode: str,
) -> Path:
    """Write ``out_dir/report_<mode>.html`` from the mode's timelines: one per
    video in its folder (``result_dirs``) and one across all at ``out_dir``.

    ``baseline`` is the frozen VLM-only report kept for side-by-side review.
    """
    evals = [_read_json(layout.synth_eval_path(root / "out", n)) for n in SET_TITLES]
    sections: list[str] = []
    all_frames: dict[tuple[str, int], str] = {}
    descriptions: dict[tuple[str, int], str] = {}
    totals = {
        "segments": 0,
        "analysed": 0.0,
        "tag": [],
        "quality": [],
        "pairs": [],
        "detect": 0.0,
        "kept": 0,
    }
    for n, video in enumerate(videos):
        html, stats, frames = _video_section(
            root, out_dir, result_dirs[video.id], mode, video, n
        )
        sections.append(html)
        all_frames.update(frames)
        descriptions.update(stats["descriptions"])
        totals["segments"] += stats["segments"]
        totals["analysed"] += stats["analysed"]
        totals["tag"] += stats["tag_seconds"]
        totals["quality"] += stats["quality_seconds"]
        totals["pairs"] += stats["highlight_vs_quality"]
        totals["detect"] += stats["detect_seconds"]
        totals["kept"] += stats["kept"]
    summary = _read_json(layout.timeline_path(out_dir, mode))
    tag_times = totals["tag"]
    avg_tag = sum(tag_times) / len(tag_times)
    q_times = totals["quality"]
    avg_q = sum(q_times) / len(q_times)
    rho = spearman([h for h, _ in totals["pairs"]], [q for _, q in totals["pairs"]])
    baseline_link = Path(os.path.relpath(baseline, out_dir)).as_posix()
    cards = [
        ("视频", f"{len(videos)}", ""),
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
（VLM 精彩度打分）</a>。基线版用 ≤8 秒切分、每段 2 张关键帧，按 VLM 精彩度
选片；本版切分 ≤{MAX_SEGMENT_SEC:.0f} 秒（2026-10-26 前切分的视频仍是 ≤8 秒）、
关键帧每 {KEYFRAME_STEP_SEC:.0f} 秒一张，片段得分以 DOVER 美学分为主。</p>
<p class="muted">{date.today().isoformat()} · 本机 Core Ultra 5 225U / 15.4 GB，无独显 ·
切分 PySceneDetect AdaptiveDetector + 叠化检测 · 打标签 Qwen3.5-4B Q4_K_M（llama.cpp
Vulkan）· 美学打分 DOVER（CPU，S-Lab 许可，仅内部验证）·
剪辑建议为规则打分 + 槽位填充</p>
<div class="cards">{card_html}</div>
<h2>1. 切分评测（合成视频，切点已知）</h2>
<p class="muted">两层切分：PySceneDetect AdaptiveDetector 找硬切 + 线性混合检验找叠化，
镜头内再按 ≤{MAX_SEGMENT_SEC:.0f} 秒切窗口，短于 1 秒的碎片并入相邻片段。</p>
{''.join(_eval_section(ev) for ev in evals)}{EVAL_NOTE}
{summary_heading(mode, summary)}
{scene_slots(summary, all_frames, descriptions) if mode == "highlights" else ""}
<video src="{escape(_rel(layout.preview_path(out_dir, mode), out_dir))}" controls
preload="metadata"></video>
{_clip_cards(summary['clips'], all_frames, {v.id: v.file for v in videos})}
<h2>3. 逐视频结果</h2>
{''.join(sections)}
<p class="muted">素材来自 Wikimedia Commons（CC BY / CC BY-SA 3.0），
署名见 data/raw/SOURCES.md。</p>
</main></body></html>"""
    path = layout.report_path(out_dir, mode)
    path.write_text(page, encoding="utf-8")
    return path


def write_edit(
    root: Path, folder: Path, videos: list[layout.Video], timeline: dict
) -> Path:
    """Write ``folder/report.html`` for an edit by instruction: the
    instruction and how the model read it, each part's slots with the scene
    retrieved for it, the preview and the clips."""
    work = root / "data" / "work"
    files = {v.id: v.file for v in videos}
    frames: dict[tuple[str, int], str] = {}
    descriptions: dict[tuple[str, int], str] = {}
    for vid in timeline["videos"]:
        frames.update(
            keyframe_paths(
                work, vid, _read_json(layout.signals_path(work, vid)), folder
            )
        )
        for line in layout.tags_path(work, vid).read_text("utf-8").splitlines():
            r = json.loads(line)
            descriptions[(vid, r["index"])] = r["tags"]["description"]
    plan = timeline["outline"]
    scenes = {sc["number"]: sc for sc in timeline["scenes"]}
    keep = {(a["video"], a["index"]): a["keep"] for a in timeline["assessments"]}
    taken = {(c["video"], c["index"]) for c in timeline["clips"]}
    catalog = {s["number"]: s for s in timeline["catalog"]}

    def catalog_entry(n: int) -> str:
        s = catalog[n]
        return (
            f"场景 {n}（{_fmt(s['start'])}–{_fmt(s['end'])}）"
            f"「{escape(s['title'])}」"
        )

    def where(sc: dict) -> str:
        first = files[sc["video"]].split("/")[-1]
        last = files[sc["segments"][-1][0]].split("/")[-1]
        if first == last:
            return f"{escape(first)} {_fmt(sc['start'])}–{_fmt(sc['end'])}"
        return (
            f"{escape(first)} {_fmt(sc['start'])} → "
            f"{escape(last)} {_fmt(sc['end'])}"
        )

    parts = []
    for k, (part, slots) in enumerate(
        zip(plan["parts"], timeline["slots"], strict=True)
    ):
        color = PART_COLORS[k % len(PART_COLORS)]
        got = sum(c["duration"] for c in timeline["clips"] if c["role"] == part["name"])
        short = got < 0.8 * part["duration_sec"]
        cards = []
        for slot in slots:
            head = (
                f"<b>「{escape(slot['query'])}」</b>"
                f"<span class='muted'> 设计取自场景 "
                f"{'、'.join(map(str, slot['sources']))}</span>"
            )
            if slot["scene"] is None:
                cards.append(
                    f'<div class="slot">{head}<p class="miss">没有可用的场景</p></div>'
                )
                continue
            sc = scenes[slot["scene"]]
            found = (
                "<span class='ok'>是设计的场景</span>"
                if sc["number"] in slot["sources"]
                else "<span class='miss'>不是设计的场景</span>"
            )
            others = "、".join(
                f"场景 {n}（{where(scenes[n])}）"
                for n in slot["top"]
                if n != sc["number"]
            )
            cards.append(
                f'<div class="slot">{head} → 场景 {sc["number"]}'
                f"<span class='muted'>（{where(sc)}，"
                f"{len(sc['segments'])} 段）</span><br>"
                f"<span class='muted'>检索相关度 {slot['relevance']:.0%} · "
                f"分到 {slot['seconds']:.0f}s · </span>{found}"
                f"{_scene_strip(sc, frames, descriptions, taken, keep)}"
                f"<span class='muted'>其他候选：{others or '—'}</span></div>"
            )
        parts.append(
            f'<div class="part" style="border-left-color:{color}">'
            f"<h3>{k + 1}. {escape(part['name'])} "
            f"<span class='muted'>分量 {part['weight']} · "
            f"目标 {part['duration_sec']}s · "
            f"<span class='{'miss' if short else 'ok'}'>选到 {got:.0f}s</span>"
            f"</span></h3><p class='muted'>{escape(part['summary'])}</p>"
            f"{''.join(cards)}</div>"
        )
    pace_note = {
        "整段": "每段素材完整保留",
        "适中": "每段最多 6 秒，取中间",
        "快": "每段最多 3 秒，取中间",
    }[plan["pace"]]
    total = timeline["catalog_total"]
    catalog_note = (
        f"{total} 个有可用片段"
        if len(catalog) == total
        else f"{total} 个有可用片段，按要求检索出 {len(catalog)} 个给模型"
    )
    order_note = f"视频按{escape(timeline['order'])}排列"
    source = timeline["target_source"]
    if source["kind"] == "instruction":
        target_note = f"指令里的「{escape(source['quote'])}」"
    else:
        target_note = f"指令没说时长，按素材的 {source['ratio']:.0%}"
    analysed = sum(ch["end"] - ch["start"] for ch in timeline["chunks"])
    cards_html = "".join(
        f'<div class="card"><span class="muted">{k}</span><b>{v}</b>'
        f'<span class="muted">{extra}</span></div>'
        for k, v, extra in (
            ("目标", f"{timeline['target_sec']:.0f}s", target_note),
            ("选到", f"{timeline['total_sec']:.1f}s", f"{len(timeline['clips'])} 段"),
            (
                "素材",
                f"{analysed / 60:.1f} 分钟",
                f"{len(timeline['videos'])} 条视频，{order_note}",
            ),
            ("场景", f"{len(timeline['scenes'])}", catalog_note),
        )
    )
    unmet = "、".join(escape(u) for u in plan["unmet"])
    materials = "".join(
        f"<li>{catalog_entry(m['scene'])}：{escape(m['content'])}</li>"
        for m in plan["materials"]
    )
    notes = "".join(f"<li>{escape(n)}</li>" for n in timeline["notes"])
    tries = len(timeline["outline_failed_attempts"]) + len(
        timeline["duration_failed_attempts"]
    )
    title = plan.get("title") or "按要求剪辑"
    failed = timeline["outline_failed_attempts"]
    raw_failed = (
        f"<details><summary>未通过校验的 {len(failed)} 次</summary><ul>"
        + "".join(f"<li>{escape(e)}</li>" for e in failed)
        + "</ul></details>"
        if failed
        else ""
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{CSS}</style></head>
<body><main>
<h1>{escape(title)}</h1>
<p class="muted">{escape(timeline['created'])} · 按用户要求剪辑 · 大纲 Qwen3.5-4B
（用时 {timeline['outline_seconds']}s{f"，重试 {tries} 次" if tries else ""}）·
检索 EmbeddingGemma 2 文本、图像向量 + BM25</p>
<div class="slot"><b>剪辑要求</b><p>{escape(timeline['instruction'])}</p>
<b>选用的素材</b><ul>{materials}</ul>
<b>成片设计</b><p>{escape(plan['design'])}</p>
<p class="muted">节奏：{escape(plan['pace'])}（{pace_note}）</p>
{f'<p class="miss">素材里找不到：{unmet}</p>' if unmet else ""}</div>
<div class="cards">{cards_html}</div>
{f'<ul class="miss">{notes}</ul>' if notes else ""}
<h2>大纲与检索到的场景</h2>
<p class="muted">大纲由模型读要求和场景目录（每个场景的标题和概况）写成：先挑场景，
再写设计，最后按设计分部分、定场景槽位，并标出每个槽位取自哪几个场景。
每个槽位在全部素材的场景里检索（文本向量、图像向量、BM25 融合），相关度为主、
画质为辅，一个场景只给一个槽位。场景内选片时相关度占 70%、画质占 30%（都缩放到
0–1），分段总结点名的亮点另有加分；节奏为适中或快时只取片段中间。
成片按大纲顺序排列。关键帧蓝框为取用的片段，变淡的是画质不合格的片段。</p>
{''.join(parts)}
<h2>预览</h2>
<video src="{escape(_rel(layout.edit_preview_path(folder), folder))}" controls
preload="metadata"></video>
{_clip_cards(timeline['clips'], frames, files)}
<h2>大纲原文</h2>
<p class="muted">模型输出的 JSON，未经改动；各部分的秒数由程序按 weight 另算。</p>
<pre class="raw">{escape(timeline['outline_raw'])}</pre>
{raw_failed}
</main></body></html>"""
    path = layout.edit_report_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


# Folders directly under out/ that are not collections.
NOT_COLLECTIONS = frozenset({"baseline", "synth"})
MODE_TITLES = {"highlights": "精彩瞬间", "ranked": "只按得分的对照"}


def _link(path: Path, start: Path, text: str) -> str:
    if not path.exists():
        return '<span class="muted">—</span>'
    return f'<a href="{escape(_rel(path, start))}">{escape(text)}</a>'


def _edit_rows(collection: Path, out: Path) -> list[str]:
    """One row per edit by instruction, newest first."""
    rows = []
    for path in sorted(
        collection.glob("edits/*/timeline.json"), key=lambda p: p.parent.name
    ):
        t = _read_json(path)
        folder = path.parent
        rows.append(
            f"<tr><td>{escape(t['created'])}</td>"
            f"<td>{escape(t['instruction'])}</td>"
            f"<td>{escape(t['outline'].get('title') or '')}</td>"
            f"<td>{t['total_sec']:.1f}s / {t['target_sec']:.0f}s</td>"
            f"<td>{_link(layout.edit_report_path(folder), out, '报告')} · "
            f"{_link(layout.edit_preview_path(folder), out, '预览')}</td></tr>"
        )
    return rows[::-1]


def _video_rows(collection: Path, out: Path) -> list[str]:
    rows = []
    for folder in sorted(p for p in collection.iterdir() if p.is_dir()):
        listing = layout.scenes_dir(folder) / "scenes.json"
        tags = layout.tags_copy_path(folder)
        timelines = [layout.timeline_path(folder, m) for m in MODE_TITLES]
        if not any(p.exists() for p in (listing, tags, *timelines)):
            continue
        scenes = (
            f"{_link(layout.scenes_dir(folder), out, '目录')}"
            f"（{len(_read_json(listing)['scenes'])} 个）"
            if listing.exists()
            else _link(listing, out, "")
        )
        rows.append(
            f"<tr><td>{escape(folder.name)}</td><td>{scenes}</td>"
            f"<td>{_link(tags, out, 'tags.jsonl')}</td>"
            f"<td>{' · '.join(_link(t, out, t.name) for t in timelines)}</td></tr>"
        )
    return rows


def write_index(out: Path) -> Path:
    """Write ``out/index.html``: every collection's reports, previews, edits
    by instruction and per-video results, with links, rebuilt from what is
    on disk."""
    sections = []
    for collection in sorted(
        p for p in out.iterdir() if p.is_dir() and p.name not in NOT_COLLECTIONS
    ):
        reports = [
            f"<li>{MODE_TITLES[m]}："
            f"{_link(layout.report_path(collection, m), out, '报告')} · "
            f"{_link(layout.preview_path(collection, m), out, '预览')}</li>"
            for m in MODE_TITLES
            if layout.report_path(collection, m).exists()
        ]
        edits = _edit_rows(collection, out)
        videos = _video_rows(collection, out)
        if not (reports or edits or videos):
            continue
        edit_table = (
            '<div class="scroll"><table><tr><th>时间</th><th>指令</th>'
            "<th>标题</th><th>选到 / 目标</th><th></th></tr>"
            f"{''.join(edits)}</table></div>"
            if edits
            else '<p class="muted">还没有</p>'
        )
        sections.append(
            f"<h2>{escape(collection.name)}</h2><ul>{''.join(reports)}</ul>"
            f"<h3>按指令剪辑（{len(edits)} 次）</h3>{edit_table}"
            '<h3>逐条视频</h3><div class="scroll"><table><tr><th>视频</th>'
            "<th>场景素材</th><th>标签</th><th>时间线</th></tr>"
            f"{''.join(videos)}</table></div>"
        )
    evals = [
        _link(layout.synth_eval_path(out, n), out, f"{n}（{SET_TITLES[n][:3]}）")
        for n in SET_TITLES
    ]
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>结果索引</title><style>{CSS}</style></head>
<body><main>
<h1>结果索引</h1>
<p class="muted">每个集合（data/raw 下的第一层文件夹）一节；运行 tag、scenes、
report、edit 后自动更新。中间结果在 data/work，按视频 ID 命名，ID 与文件名的
对应见 data/raw/manifest.json。</p>
{''.join(sections)}
<h2>其他</h2>
<ul><li>早期基线：{_link(layout.baseline_report_path(out), out, '报告')}
（只用 VLM 精彩度选片）</li>
<li>切分评测（合成视频）：{' · '.join(evals)}</li></ul>
</main></body></html>"""
    path = layout.index_page_path(out)
    path.write_text(page, encoding="utf-8")
    return path
