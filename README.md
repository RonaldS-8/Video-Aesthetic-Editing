# 视频切分、片段打标与剪辑建议 Demo

本地验证"打标签 + 剪辑建议"链路的第一步：把视频切成片段，
逐段打标签和美学打分，再按规则给出剪辑建议。全部在本机离线运行。

- 结果报告（浏览器直接打开）：
  - 短 vlog 精彩瞬间：[out/dover/report_highlights.html](out/dover/report_highlights.html)，
    只按得分的对照：[out/dover/report_ranked.html](out/dover/report_ranked.html)
  - 长素材（爱丁堡 67 分钟）精彩瞬间：
    [out/long/report_highlights.html](out/long/report_highlights.html)，
    对照：[out/long/report_ranked.html](out/long/report_ranked.html)
  - 基线版（VLM 精彩度打分，来自 `main`，冻结不再生成）：
    [out/report.html](out/report.html)
- 测试素材与署名：[data/raw/SOURCES.md](data/raw/SOURCES.md)
- 在基线上做的改进（逐项记录）：[docs/改进记录.md](docs/改进记录.md)
- 前期调研：[docs/research/AI视频剪辑与片段检索-调研.md](docs/research/AI视频剪辑与片段检索-调研.md)
- 向量检索原理笔记：[docs/research/向量检索与Embedding原理-讨论笔记.md](docs/research/向量检索与Embedding原理-讨论笔记.md)
- 目标方案（已定稿，暂停）：[docs/plans/目标方案-结构设计与检索填槽.md](docs/plans/目标方案-结构设计与检索填槽.md)
- 美学模型提案（issue 原文）：[docs/issues/片段美学模型.md](docs/issues/片段美学模型.md)
- 质量模型实测对比（DOVER、DOVER-Mobile、UVQ 1.5，含运行方法）：
  [docs/research/质量模型对比-DOVER-UVQ.md](docs/research/质量模型对比-DOVER-UVQ.md)

## 目录结构

```text
Video/
├── video_demo/          代码（python -m video_demo <阶段>）
│   ├── cli.py           命令行入口与各阶段
│   ├── media.py         解码、抽帧、写视频（PyAV）
│   ├── segment.py       镜头切分与窗口切片
│   ├── signals.py       亮度、清晰度、抖动等确定性信号
│   ├── layout.py        视频 ID（内容哈希）、清单、各类文件路径
│   ├── tag.py           Qwen3-VL-2B 打标签（llama-server）
│   ├── aesthetic.py     DOVER / DOVER-Mobile 美学打分
│   ├── uvq.py           UVQ 1.5 质量打分（仅模型对比用）
│   ├── quality.py       质量模型对比统计（仅模型对比用）
│   ├── runner.py        并发运行打标签与美学打分
│   ├── suggest.py       片段打分与时间线填槽
│   ├── outline.py       分块总结、大纲生成与按部分填素材（Qwen3.5-4B）
│   ├── embed.py         EmbeddingGemma 2 文本与图像向量（MediaPipe）
│   ├── retrieve.py      向量索引、BM25 与 RRF 融合检索
│   ├── evaluate.py      切点评测、秩相关
│   ├── report.py        HTML 报告
│   └── synth.py         合成评测视频
├── tests/               pytest 测试
├── docs/
│   ├── 改进记录.md      在基线上做的改进，逐项记录
│   ├── research/        调研与原理笔记
│   ├── plans/           方案设计
│   └── issues/          提给 NAS 仓库的 issue 原文
├── data/
│   ├── raw/             原始视频（原名，不入库）、清单 manifest.json、SOURCES.md
│   ├── synth/           合成评测视频（mp4 不入库）与真值
│   └── work/            按类型分目录：切分、信号、关键帧、标签、美学分（见"流水线"）
├── out/                 基线版报告与时间线（来自 main）、切分评测结果
│   ├── dover/           vlog 集合的精彩瞬间：每条视频一个原名目录 + 报告
│   └── long/            长素材集合，结构同上（预览视频都不入库）
├── third_party/DOVER/   DOVER 代码（不入库）
├── third_party/uvq/     UVQ 1.5 代码（不入库，仅模型对比用）
├── models/              全部模型权重（不入库）；来源、版本、SHA-256 见
│   │                    models/README.md
│   ├── dover/           DOVER、DOVER-Mobile
│   ├── uvq/             UVQ 1.5 三个网络
│   ├── embeddinggemma/  EmbeddingGemma 2 与清单（清单入库，与 A-NAS 一致）
│   └── gguf/            打标签用 Qwen3-VL-2B、大纲用 Qwen3.5-4B
├── requirements.txt
└── pyproject.toml
```

## 结果（2026-10-22，本机实测）

| 项 | 数值 |
|---|---|
| 分析时长 | 13.5 分钟（6 条视频，上海 vlog 只取前 5 分钟） |
| 片段数 | 177（镜头 130，镜头内按 ≤8 s 再切） |
| 切分耗时 | 172 s，约为视频时长的 1/5 |
| 打标签 | 平均 10.2 s/段（最长 15.3 s），合计 30 分钟 |
| 打标签失败重试 | 1 次（03 片尾黑屏字幕，复读后重试通过） |
| 美学打分（DOVER，CPU） | 平均 12.8 s/段（最长 24.9 s），合计 38 分钟 |
| 打标签与美学打分并发 | 04 的 3 段：总用时 64 s，两步分开跑共 98 s |

切分评测（合成视频，容差 ±0.5 s）：

| 评测集 | 切点 | 准确率 | 召回率 | F1 |
|---|---|---|---|---|
| 调参集（03/04/05，硬切 4、跳切 1、叠化 2） | 7 | 1.00 | 1.00 | 1.00 |
| 留出集（01/02/06，硬切 2、叠化 4） | 6 | 1.00 | 0.33 | 0.50 |

## 结论

1. **硬切可靠，渐变转场是短板。**
   - 硬切和跳切在两组上全部检出，没有误报。
   - PySceneDetect 默认检测器漏掉全部叠化。自写的"线性混合"检验
     只在调参集上有效，留出集 4 个叠化全漏：两段画面都在动时，
     混合残差和普通运镜分不开。
   - 下一步换学习型检测器（如 TransNetV2），需要另装模型。
2. **vlog 的"镜头边界"不等于"内容边界"。**
   - 02 片头用甩镜转场，17–26 s 内换了 6 个场景却没有切点。
   - 8 s 窗口上限兜住了一部分；更好的做法是按相邻窗口的
     图像向量相似度再切一刀。
3. **2B 模型的描述和标签可用，打分和问题判断不可用。**
   - 描述能认出鱼尾狮、滨海湾金沙等地标，没有 GPS 也有地点线索。
   - 精彩度 1–5 分里 156/177 是 3 或 4 分，区分度弱。
   - "画面问题"177 段全部填"无明显问题"，模糊、过暗等只能靠
     确定性信号判断。
   - 偶发复读（"白天的白天的……"）：现在用长度约束、重复惩罚、
     复读检测和最多 3 次重试处理，仍失败就报错，不填默认值。
4. **剪辑建议的排序与文件名无关。**
   - 早期版本按文件名打破同分，合集 14 段里 10 段来自 01。
   - 现在得分 = 精彩度 + 相对清晰度（±0.5）− 画质扣分，
     177 段有 117 种不同得分；剩余同分按内容描述排序。
   - 已验证：把视频随机改名并打乱输入顺序 5 次，选片完全一致。
   - 合集里 01 仍占 6/13，原因是它是剪辑成片，精彩度和清晰度都高。
   - 这版"按精彩度选片"只作为基线保留，不再继续调；
     目标方案见下一步。
5. **一镜到底素材凑不满 30 s。** 04（17 s）、05（31 s）是同一
   场景的连续横摇，相邻窗口标签高度重合，被去重后只剩 1–2 段。
   报告里如实标注"可用素材不足"，没有补凑。

## 美学打分（分支 `feature/dover-aesthetic`）

片段得分改为 1 + 4 ×（0.8 × DOVER 融合分 + 0.2 × VLM 精彩度归一化）
− 画质扣分。VLM 的 `issues` 不再扣分，清晰度加分也去掉了
（DOVER 的技术分支已经覆盖清晰度）。阈值 `DROP_SCORE` 没有改。
基线结果在 `main` 分支。

| 项 | 基线（VLM 精彩度） | DOVER 为主 |
|---|---|---|
| 得分取值 | 精彩度 156/177 为 3 或 4 | 融合分 0.11–0.86，177 段有 173 个不同值 |
| 建议保留 | 153/177 | 147/177 |
| 合集 13 段的来源 | 01×6、03×3、06×2、02×1、04×1 | 01×6、06×5、02×2 |

1. **DOVER 能识别出 VLM 看不出的废片。**
   最低两段是 02 片尾的"订阅按钮"画面（0.11，VLM 精彩度给了 4）
   和 01 的全黑片段（0.12）。最高几段是清晰的特写：蜘蛛、冰淇淋、花朵。
2. **两种分数基本不相关。** VLM 精彩度和 DOVER 融合分的
   Spearman ρ = 0.03，说明两者衡量的不是同一件事。
3. **DOVER 给夜景的分数偏低，合集里极光全部落选。**
   夜晚片段中位数 0.46，白天 0.55；极光 03 的中位数 0.43，
   其中 VLM 给 5 分的极光片段只得了 0.24。DOVER 是按网络 UGC 的
   画质标注训练的，暗、细节少的画面会被判为低质量。
   要不要按时段分别归一化（或者给夜景加补偿），需要决定。
4. **CPU 上比作者给的数字慢。** 作者环境里是 3.6 s/视频；
   本机 Core Ultra 5 225U 每段平均 12.8 s，177 段共 38 分钟。
   并发时，打标签会占用部分 CPU，DOVER 每段会再慢一些。
5. **结果可复现。** 固定了随机种子，重跑 04 得到的分数逐字节一致。
   和官方 `evaluate_one_video.py` 比（同一个 H.264 文件；
   官方脚本用 decord 读不了 VP9 的 webm）：官方两次为 0.412 和 0.410，
   本适配器为 0.404。

## 下一步

目标方案：**按用户要求先设计结构，再检索素材填槽。**

```text
用户要求（"剪一个 30 秒的新加坡夜游，节奏轻快"）
  → 模型设计结构：段落、每段时长、要什么画面、景别、节奏
  → 每个槽位生成检索条件（文本 + 时段/地点/人物等过滤）
  → 在片段索引里检索 Top-K（标签、描述、向量）
  → 槽位内按匹配度和画质排序，去重
  → 输出时间线，确定性校验（时长、不重复、不越界）
  → 用户逐槽替换候选
```

- 本次已有的切分、逐段标签、画质信号、时间线校验和预览渲染
  可以直接复用；要补的是结构设计、片段向量索引和检索。
- 精彩度降为槽位内排序的一个因素，不再决定选什么。
- 渐变转场：接入 TransNetV2，在两组合成视频上复测。
- 内容边界：相邻窗口的图像向量相似度骤降处补切点。
- 06 的语音转写（SenseVoice）还没做。

## 运行环境

- 机器：Core Ultra 5 225U（12 核）、15.4 GB 内存、Intel 核显，无独显。
- Python 3.11 虚拟环境 `.venv`，依赖见 `requirements.txt`。
  PyAV 自带 FFmpeg 库，不需要单独安装 ffmpeg。
- 打标签模型：Qwen3-VL-2B-Instruct Q4_K_M（GGUF，约 1.1 GB）
  加 F16 视觉投影（约 0.8 GB），放在 `models/gguf/`，由 llama.cpp
  Vulkan 版 `llama-server` 提供 OpenAI 兼容接口（端口 8091）。
- 美学打分模型：DOVER，代码放在 `third_party/DOVER/`，权重放在
  `models/dover/`，都不入库。
- 所有权重的存放路径、来源和版本见 [models/README.md](models/README.md)。

## 流水线：切分 → 描述 → 打分 → 剪辑

### 视频 ID 与命名

- **视频 ID = 视频文件内容 SHA-256 的前 16 位**（如 `330ff4c78eabf568`），
  与文件名无关。改名不影响 ID；内容相同的两个文件得到同一个 ID，
  只处理一次。
- **原始视频保留原名、原位置，不复制、不改名**，以节约空间。
  ID 与原文件名的对应关系记在清单 `data/raw/manifest.json` 里。
- 每个派生文件放在按类型分的目录里，文件名统一为 `<类型>_<ID>`，
  如 `data/work/tags/tag_330ff4c78eabf568.jsonl`。单个文件拿出来也能
  认出属于哪条视频。
- **集合**：`data/raw/` 下的第一层文件夹就是一个数据集合，如 `vlog/`
  （现有 6 条 vlog）、`tvsum/`、`summe/`、`long/`。中间结果按 ID 共用，
  剪辑建议、预览和报告按集合分别输出。

### 数据流

```text
data/raw/<集合>/*（原名）
  │
  ├─ ⓪ ingest     登记新视频、去重 → data/raw/manifest.json
  │                并对新视频执行 ① ②
  ├─ ① segment    切分            → data/work/segments/segment_<ID>.json
  ├─ ② signals    画质信号与关键帧 → data/work/signals/signal_<ID>.json
  │                                  data/work/keyframes/keyframe_<ID>/
  ├─ ③ tag        画面描述与标签   → data/work/tags/tag_<ID>.jsonl
  ├─ ④ aesthetic  美学打分         → data/work/aesthetic/aesthetic_<ID>.jsonl
  │      ③④ 可用 analyze 一起跑：打标签用核显，DOVER 用 CPU
  ├─ ④' index     文本与图像向量   → data/work/index/index_<ID>.npz
  ├─ ⑤ suggest    片段评分与选片   → out/<集合>/<原名>/timeline_<模式>.json
  │                                  out/<集合>/timeline_<模式>.json（全集合）
  │                highlights 模式另写分块总结
  │                                → data/work/summaries/summary_<ID>.json
  ├─ ⑥ render     预览视频         → out/<集合>/<原名>/preview_<模式>.mp4
  └─ ⑦ report     汇总报告         → out/<集合>/report_<模式>.html
```

| 阶段 | 命令 | 做什么 | 读 | 写 |
|---|---|---|---|---|
| ⓪ 导入 | `ingest` | 扫描 `data/raw/` 及其子文件夹里的视频文件，只对新增或大小、修改时间变了的文件算哈希；新内容登记为新视频，与已有视频内容相同的记为重复并跳过；然后对还没有切分结果的视频执行 ①②，已处理的视频不动 | 原视频、清单 | 清单、①② 的产出 |
| ① 切分 | `segment` | 先核对文件哈希与清单一致；AdaptiveDetector 找硬切，线性混合检验找叠化；<1 s 的碎镜头并入相邻镜头，镜头内再按 ≤10 s 切成片段（2026-10-26 前切分的视频是 ≤8 s） | 原视频 | `segment_<ID>.json` |
| ② 画质信号 | `signals` | 4 fps 采样（长边 448 px），算亮度、清晰度（Laplacian 方差）、运动、全局位移抖动并给出问题标记；关键帧按每 2 s 一个时间窗、每窗选清晰度最高的一帧（10 s 片段 5 张，≤2 s 的片段 1 张） | 原视频、`segment_<ID>.json` | `signal_<ID>.json`、`keyframe_<ID>/` |
| ③ 描述 | `tag` | 片段的全部关键帧按时间顺序送 Qwen3-VL-2B，要求描述覆盖片段内出现的全部内容，按 JSON Schema 约束输出描述、场景、主体、动作、景别、时段、氛围、标签、精彩度、画面问题；复读或格式不符时重试，最多尝试 3 次，仍失败就报错 | `segment_<ID>.json`、`signal_<ID>.json`、`keyframe_<ID>/` | `tag_<ID>.jsonl` |
| ④ 打分 | `aesthetic` | 先按时间戳数出片段帧数，按 DOVER 的采样规则选出要用的帧（每段最多约 130 帧），只解码这些帧送 DOVER（CPU），长边超过 1920 的先缩小；得到技术分、美学分和官方融合分（0–1）；固定随机种子，可复现，与改造前逐项一致 | 原视频、`segment_<ID>.json` | `aesthetic_<ID>.jsonl` |
| ③+④ | `analyze` | 先做开工检查（输入文件、llama-server、DOVER 权重），再并发运行 ③ 和 ④，双进度条 | 同上 | 同上 |
| ④' 向量 | `index` | EmbeddingGemma 2 740M（与 A-NAS 客户端同一文件、同一调用方式）给每段算两个 768 维向量：标签文本（描述、场景、动作、主体、标签）一个，关键帧图像向量的平均一个；图片先按客户相册的缩略图处理（长边 512、JPEG 82）。每 20 段存一次，可续跑；换模型后要 `--force` 重建。CPU 上每张图约 1.35 s，67 分钟素材约 30 分钟 | `tag_<ID>.jsonl`、`signal_<ID>.json`、`keyframe_<ID>/` | `index_<ID>.npz` |
| ⑤ 剪辑 | `suggest` | 片段分 = 1 + 4 ×（0.8 × 融合分 + 0.2 × 精彩度归一化）− 画质扣分。两种模式见下 | ①②③④ 的产出 | `timeline_<模式>.json` |
| ⑥ 预览 | `render` | 按时间线从原视频截取并拼接（854×480、30 fps、无音轨） | 时间线、清单、原视频 | `preview_<模式>.mp4` |
| ⑦ 报告 | `report` | 汇总切分评测、逐片段标签与分数、剪辑建议 | 以上全部 | `report_<模式>.html` |

⑤⑥⑦ 用 `--collection <集合>` 选数据（默认 `vlog`），输出到
`out/<集合>/`；`vlog` 集合沿用 `out/dover/`。`--out <目录>` 可以改。
每条视频的结果放在以原文件名（去掉扩展名，按清单从 ID 查出）命名的
子目录里；同一集合里两个文件只差扩展名时会报错，需要先改名。
`segment`、`signals` 默认跳过已有结果的视频，`--force` 才会重做。

剪辑建议的两种模式（`suggest --mode`，`render`、`report` 也要给同样的
`--mode`）。长短视频都用同一套精彩瞬间流程，不再单独做固定时长的短片：

| 模式 | 用途 | 规则 |
|---|---|---|
| `highlights`（默认） | 精彩瞬间：从素材里挑出约 1/6（1 小时 → 10 分钟）供用户再剪 | ① 每条视频按时间每 5 分钟分一块，Qwen3.5-4B 逐块读片段描述写总结（发生了什么、地点、主体、活动、最多 3 个值得留的片段），按视频缓存在 `data/work/summaries/summary_<ID>.json`，中断后接着跑；② 4B 读全部块总结生成大纲：2–8 个部分（不超过块数），每部分有标题、内容说明、覆盖的块范围、分量（1–10）和 1–4 个场景槽位（每个用短语说明要什么画面），程序校验（块有效、按时间不重叠、槽位不重复），按分量分配各部分秒数（每部分至少 10 s，总和等于目标），不合格重试，全部失败报错；③ 按场景填槽（需要先跑 `index`）：每块内相邻片段按向量相似度连成场景（前后各 2 段平均向量的相似度为局部最低且低于 0.89 处切开，场景 20–120 s，不跨镜头），每个槽位只在本部分的块里检索场景：场景向量取所含片段的平均，按文本向量、图像向量、标签 BM25（按字二元组）三路排序后用 RRF 融合；相关度在候选场景间拉开（最高 1、最低 0）后加 0.2 × 场景画质，一个场景只给一个槽位。部分的秒数在槽位间平分，场景装不下的让给其他槽位；槽位在自己的场景里按片段得分 + 片段相关度（场景内拉开）+ 总结点名的亮点加 0.5 分整段选片，60 s 内标签高度重合的只留一段；按拍摄时间排序。美学分先在同一时段内取百分位。需要 4B 服务（端口 8092）。每条视频一条时间线，另有全集合的一条（集合里只有一条视频时直接复用它的大纲）；`--ratio` 改比例 |
| `ranked` | 精彩瞬间的对照基线，不用大纲 | 只按得分整段选片；每 5 分钟的时间窗口至少选一段；其余同 `highlights` |

### 存放路径

```text
data/
├── raw/                                   输入
│   ├── <集合>/<原文件名>                   原始视频（原名，不入库）
│   ├── manifest.json                       清单：ID ↔ 原文件名（入库）
│   └── SOURCES.md                          每条视频的来源、作者、许可
└── work/                                  中间结果（入库）
    ├── segments/segment_<ID>.json          ① 切分
    ├── signals/signal_<ID>.json            ② 画质信号
    ├── keyframes/keyframe_<ID>/seg<NNN>_<k>.jpg   ② 关键帧，k 从 0 起按时间顺序
    ├── tags/tag_<ID>.jsonl                 ③ 描述与标签
    └── aesthetic/aesthetic_<ID>.jsonl      ④ 美学分
out/<集合>/                                输出（vlog 集合为 out/dover/）
├── <原名>/                                 每条视频一个目录，按原文件名
│   ├── timeline_highlights.json            ⑤ 精彩瞬间时间线
│   ├── timeline_ranked.json                ⑤ 只按得分的对照
│   └── preview_<模式>.mp4                  ⑥ 预览（不入库）
├── timeline_<模式>.json                    ⑤ 全集合的时间线
├── preview_<模式>.mp4                      ⑥ 全集合预览（不入库）
├── report_<模式>.html                      ⑦ 报告
└── quality_compare.json                    质量模型对比结果
```

`data/work/quality/` 下是模型对比用的分数（DOVER-Mobile、UVQ 1.5），
不参与剪辑建议，见评估文档。

### 输入要求

- 视频放进 `data/raw/<集合>/`。支持的扩展名（不分大小写）：
  mp4、mov、m4v、3gp、mkv、webm、avi、mts、m2ts，与 A-NAS 相册一致；
  编码不限，PyAV 能解码即可；容器要能报出时长。
- 文件名随意，建议用英文、不含空格，方便人看；程序只认内容哈希。
- 放进去后运行 `ingest`。内容与已有视频相同的文件会被标为重复并
  提示，可以删掉以节约空间；程序不会自动删除任何文件。
- 新视频要在 `data/raw/SOURCES.md` 里登记来源、作者和许可。
- 只分析前一段时长时，在 `video_demo/cli.py` 的 `END_TIME` 里按
  视频 ID 设秒数（上海 vlog 只取前 300 s）。
- 音轨目前不使用，只在切分结果里记录有没有音轨。

### 清单 `data/raw/manifest.json`

| 字段 | 含义 |
|---|---|
| `hash`、`id_length` | 固定为 `sha256`、`16` |
| `videos` | 每条视频：`id`、`sha256`（完整 64 位）、`file`（相对 `data/raw` 的路径，第一层文件夹即集合）、`size`（字节）、`mtime_ns`（修改时间，只用来判断要不要重算哈希） |
| `duplicates` | 重复文件：`file`、`same_as`（与之内容相同的视频 ID）、`size`、`mtime_ns` |

各阶段运行前都会核对清单：清单里的文件不见了或大小变了，或
`data/raw` 里有没登记的视频，就报错并提示先运行 `ingest`。
原文件内容变了会得到新 ID，旧 ID 的结果保留不动；文件只是移动或
改名时 ID 不变，结果沿用。

### 文件格式

通用约定：

- 文件之间靠**视频 ID** 关联；同一视频的所有文件名里都是同一个 ID。
- 时间一律是**相对原视频开头的秒数**。
- 片段 `index` 从 0 开始连续编号，是同一视频各文件之间关联片段的键。
- `.jsonl` 每行一个片段，处理完一段追加一行；中断后重跑会按
  `index` 跳过已完成的片段。
- 任何一步失败都直接报错并停在那一段，不写默认值。
- **用 `--force` 重新切分后，必须先删掉该视频的 `signal_<ID>.json`、
  `keyframe_<ID>/`、`tag_<ID>.jsonl`、`aesthetic_<ID>.jsonl` 再往下跑**：
  续跑只认 `index`，不会发现片段边界已经变了。

`segment_<ID>.json`（一个对象）：

| 字段 | 含义 |
|---|---|
| `video_id`、`sha256` | 视频 ID、完整哈希 |
| `video` | 原视频文件名（仅供人看） |
| `duration` / `analysed_until` | 视频时长 / 实际分析到的秒数 |
| `fps`、`has_audio` | 帧率、有没有音轨 |
| `detect_seconds` | 切分耗时 |
| `cuts` | 切点列表：`time`（秒）、`kind`（`hard` 硬切 / `dissolve` 叠化） |
| `raw_shot_count`、`shots` | 合并前的镜头数；合并后的镜头 `[起, 止]` 列表 |
| `segments` | 片段列表：`index`、`start`、`end`、`shot`（所属镜头序号） |

`signal_<ID>.json`（数组，每个片段一项）：

| 字段 | 含义 |
|---|---|
| `index` | 片段序号 |
| `brightness` | 平均亮度，0–255 |
| `sharpness` | 清晰度，Laplacian 方差的中位数 |
| `motion`、`pan_speed`、`shake` | 相邻采样帧差、平均全局位移、位移变化的标准差 |
| `flags` | 问题标记，取值 `low_light`（亮度 < 40）、`blurry`（清晰度 < 30 或低于本视频中位数的 30%）、`shaky`（抖动 > 12） |
| `keyframes`、`keyframe_times` | 关键帧文件名（在 `keyframe_<ID>/` 下）和对应时间（秒）；2026-10-26 前生成的只有 2 张、没有时间 |

`tag_<ID>.jsonl`（每行一个片段）：

| 字段 | 含义 |
|---|---|
| `index`、`seconds` | 片段序号、打标签耗时 |
| `tags.description` | 一句话描述，不超过 40 字 |
| `tags.scene`、`tags.action`、`tags.mood` | 场景、动作或镜头运动、氛围 |
| `tags.subjects` | 主要主体，最多 5 个 |
| `tags.shot_size` | 景别：远景 / 全景 / 中景 / 近景 / 特写 |
| `tags.time_of_day` | 时段：白天 / 黄昏 / 夜晚 / 室内 / 不确定 |
| `tags.tags` | 检索标签，3–8 个 |
| `tags.highlight` | 精彩度，1–5 的整数 |
| `tags.issues` | 画面问题：模糊 / 过暗 / 过曝 / 抖动 / 遮挡 / 无明显问题 |
| `raw` | 模型原始输出 |
| `failed_attempts` | 被拒绝的重试记录（较早的数据没有这个字段） |

`aesthetic_<ID>.jsonl`（每行一个片段）：

| 字段 | 含义 |
|---|---|
| `index` | 片段序号 |
| `technical`、`aesthetic` | DOVER 技术分支、美学分支的原始输出 |
| `fused` | 官方融合分，0–1，剪辑评分用的就是它 |
| `n_frames`、`decoded_frames`、`seconds` | 片段帧数、实际解码的采样帧数（改造后才有）、推理耗时 |
| `model`、`weights_sha256`、`seed` | 模型名、权重 sha256、随机种子 |

`timeline_<模式>.json`：

| 字段 | 含义 |
|---|---|
| `name`、`target_sec`、`total_sec` | 时间线名（视频 ID，全集合为 `highlights`）、目标时长（输入时长 × 比例）、实际时长 |
| `mode` | `highlights` 或 `ranked` |
| `outline`、`outline_raw`、`outline_seconds`、`outline_failed_attempts`、`chunks` | 仅 highlights：模型给出的大纲（`title`，`parts` 里每部分 `name`、`summary`、`first_chunk`、`last_chunk`、`weight`、`scenes`（场景槽位），以及程序算出的 `duration_sec`）、原始输出、用时、被拒绝的尝试，以及每块的编号、视频、起止、所含片段编号和总结（`summary`、`places`、`subjects`、`activities`、`highlights`） |
| `scenes`、`slots`、`retrieval` | 仅 highlights：场景列表（编号、视频、起止、所含片段编号）；每部分每个槽位的查询、分到的场景、相关度、秒数和最合适的 5 个场景；每个槽位查询的文本、图像、BM25 三路排序前 5 个场景 |
| `notes` | 说明，如"可用素材不足" |
| `clips` | 按播放顺序的片段：`order`、`role`（highlights：所属大纲部分的标题；ranked：覆盖 / 精彩）、`video`（视频 ID）、`index`、`src_start`、`src_end`、`duration`（整段）、`description`、`reason` |
| `assessments` | 每个片段的评估：`video`（视频 ID）、`index`、`score`（1–5）、`keep`（`score` > 2 才可选）、`reasons` |

## 运行

### 安装（只需一次）

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

权重都放在 `models/` 下，每个文件的路径、下载地址、版本和 SHA-256 见
[models/README.md](models/README.md)；下面只列代码和依赖的安装。

DOVER（torch CPU 版和代码；权重 `models/dover/DOVER.pth`，约 240 MB）：

```bash
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
git clone https://github.com/VQAssessment/DOVER.git third_party/DOVER
```

```bash
curl -L -o models/dover/DOVER.pth --create-dirs https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth
```

```bash
.venv/Scripts/python -m pip install --no-deps -e third_party/DOVER
```

本次使用的 DOVER 提交为 `f1ddc96`（2024-08-12），权重 sha256 前缀
`f4a42c0bbc94`。DOVER 采用 S-Lab 许可，仅限非商用，这里只做内部验证。

EmbeddingGemma 2 740M（检索用，485 MB，Apache-2.0）。版本和 SHA-256 记在
`models/embeddinggemma/embeddinggemma-2-740m.json`（抄自 A-NAS 的
`deploy/models/`），加载时会核对：

```bash
curl -L -o models/embeddinggemma/embeddinggemma-2-740m.litertlm --create-dirs https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/resolve/24d962e906c7d332c6428e71c9676855024569e2/embeddinggemma-2-740m.litertlm
```

MediaPipe 要不带依赖装：它会拉 opencv-contrib-python，和项目的
opencv-python-headless 抢同一个 `cv2`（导入时需要的 absl-py、flatbuffers、
matplotlib 已在 requirements.txt 里）：

```bash
.venv/Scripts/python -m pip install --no-deps mediapipe==1.1.1
```

### 依次运行

都在项目根目录执行。

⓪ 导入：把新视频放进 `data/raw/<集合>/` 后运行。它会更新清单、标出重复
文件，并只对新视频做 ① 切分和 ② 画质信号（不需要模型）：

```bash
.venv/Scripts/python -m video_demo ingest
```

需要重做某一步时，可以单独运行 `segment` 或 `signals`，加 `--force`
重做已有结果的视频。

③ 描述 + ④ 打分。先在另一个终端启动 llama-server，并保持运行：

```bash
llama-server -m models/gguf/Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj models/gguf/mmproj-Qwen3VL-2B-Instruct-F16.gguf --host 127.0.0.1 --port 8091 -ngl 99 -c 8192 -np 1
```

然后并发跑打标签和美学打分。可中断，重跑会跳过已完成的片段，
所以新导入视频后直接再跑一次即可；标签全部完成后它只做美学打分，
这时不需要 llama-server：

```bash
.venv/Scripts/python -m video_demo analyze
```

也可以分开跑：`tag` 只打标签，`aesthetic` 只做美学打分。

⑤ 剪辑建议、⑥ 预览、⑦ 报告（每次都覆盖该集合的全部视频重新生成）。
长短视频的步骤相同，下面以 `long` 集合为例（不带 `--collection` 时是
默认的 vlog 集合；三步都要带相同的参数，`--mode ranked` 出对照）。先在
另一个终端启动大纲用的 Qwen3.5-4B 服务，并保持运行：

```bash
llama-server -m models/gguf/Qwen3.5-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8092 -ngl 99 -c 16384 -np 1
```

先建向量索引（不需要 llama-server，可中断续跑）：

```bash
.venv/Scripts/python -m video_demo index --collection long
```

然后：

```bash
.venv/Scripts/python -m video_demo suggest --collection long
```

```bash
.venv/Scripts/python -m video_demo render --collection long
```

```bash
.venv/Scripts/python -m video_demo report --collection long
```

`tag`、`aesthetic`、`analyze` 支持 `--collection <集合>`、`--only <前缀>`
（视频 ID 或文件路径的前缀，如 `--only 330ff4` 或 `--only vlog/04`）和
`--limit <段数>`（每条视频最多处理几段），调试时用。

### 切分评测

合成已知切点的视频，再算切点的准确率、召回率、F1；
`--set` 可选 `main`（调参集）或 `holdout`（留出集）：

```bash
.venv/Scripts/python -m video_demo synth --set holdout
```

```bash
.venv/Scripts/python -m video_demo eval --set holdout
```

### 测试与代码检查

```bash
.venv/Scripts/python -m pytest
```

```bash
.venv/Scripts/ruff check video_demo tests
```

```bash
.venv/Scripts/black --check .
```
