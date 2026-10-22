# 视频切分、片段打标与剪辑建议 Demo

本地验证"打标签 + 剪辑建议"链路的第一步：把视频切成片段，
逐段打标签和美学打分，再按规则给出剪辑建议。全部在本机离线运行。

- 结果报告（浏览器直接打开，两版格式相同，对照着看）：
  - DOVER 美学打分版：[out/dover/report.html](out/dover/report.html)
  - 基线版（VLM 精彩度打分，来自 `main`，冻结不再生成）：
    [out/report.html](out/report.html)
- 测试素材与署名：[data/raw/SOURCES.md](data/raw/SOURCES.md)
- 前期调研：[docs/research/AI视频剪辑与片段检索-调研.md](docs/research/AI视频剪辑与片段检索-调研.md)
- 向量检索原理笔记：[docs/research/向量检索与Embedding原理-讨论笔记.md](docs/research/向量检索与Embedding原理-讨论笔记.md)
- 目标方案（已定稿，暂停）：[docs/plans/目标方案-结构设计与检索填槽.md](docs/plans/目标方案-结构设计与检索填槽.md)
- 美学模型提案（issue 原文）：[docs/issues/片段美学模型.md](docs/issues/片段美学模型.md)

## 目录结构

```text
Video/
├── video_demo/          代码（python -m video_demo <阶段>）
│   ├── cli.py           命令行入口与各阶段
│   ├── media.py         解码、抽帧、写视频（PyAV）
│   ├── segment.py       镜头切分与窗口切片
│   ├── signals.py       亮度、清晰度、抖动等确定性信号
│   ├── tag.py           Qwen3-VL-2B 打标签（llama-server）
│   ├── aesthetic.py     DOVER 美学打分
│   ├── runner.py        并发运行打标签与美学打分
│   ├── suggest.py       片段打分与时间线填槽
│   ├── evaluate.py      切点评测、秩相关
│   ├── report.py        HTML 报告
│   └── synth.py         合成评测视频
├── tests/               pytest 测试
├── docs/
│   ├── research/        调研与原理笔记
│   ├── plans/           方案设计
│   └── issues/          提给 NAS 仓库的 issue 原文
├── data/
│   ├── raw/             原始视频（不入库，见 SOURCES.md）
│   ├── synth/           合成评测视频（mp4 不入库）与真值
│   └── work/<视频>/     切分、信号、标签、美学分、关键帧
├── out/                 基线版报告与时间线、切分评测结果
│   └── dover/           DOVER 版报告与时间线（suggest/render/report 默认写这里）
│                        （两处的 previews/ 都不入库）
├── third_party/DOVER/   DOVER 代码与权重（不入库）
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
- 打标签模型：Qwen3-VL-2B-Instruct Q4_K_M（GGUF，约 1 GB）
  加 F16 视觉投影，复用 JoyAI 目录里的文件，由 llama.cpp
  Vulkan 版 `llama-server` 提供 OpenAI 兼容接口（端口 8091）。

## 流水线

| 阶段 | 命令 | 做什么 | 产出 |
|---|---|---|---|
| 切分 | `segment` | AdaptiveDetector 找硬切，线性混合检验找叠化；镜头内按 ≤8 s 切窗口，<1 s 碎片并入相邻片段 | `data/work/<视频>/segments.json` |
| 画质信号 | `signals` | 4 fps 采样，算亮度、清晰度（Laplacian 方差）、运动、全局位移抖动；导出 2 张关键帧 | `signals.json`、`keyframes/` |
| 打标签 | `tag` | 2 张关键帧送 Qwen3-VL-2B，按 JSON Schema 约束输出描述、场景、主体、动作、景别、时段、氛围、标签、精彩度、画面问题 | `tags.jsonl` |
| 美学打分 | `aesthetic` | 片段原分辨率帧送 DOVER（CPU），得到技术分、美学分和官方融合分（0–1）；固定随机种子，可复现 | `aesthetic.jsonl` |
| 打标签 + 美学打分 | `analyze` | 先做开工检查，再并发运行上面两步（打标签用核显，DOVER 用 CPU），双进度条；可中断，重跑接着做 | 同上 |
| 剪辑建议 | `suggest` | 片段分 = 1 + 4 ×（0.8 × 美学分 + 0.2 × VLM 精彩度归一化）− 画质扣分；按"开场全景 → 主体 → 收尾"填槽，同镜头和标签高度重合的去重 | `out/dover/timelines/*.json` |
| 预览 | `render` | 按时间线拼出预览视频（无音轨） | `out/dover/previews/*.mp4` |
| 报告 | `report` | 汇总评测、逐片段标签和建议 | `out/dover/report.html`（`--out out` 可改输出目录） |
| 评测 | `synth` / `eval` | 拼接已知切点的合成视频，算切点准确率、召回率、F1 | `out/synth_eval_*.json` |

上海 vlog 只分析前 300 秒（`cli.py` 里的 `END_TIME`）。

## 复现

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

安装 DOVER（torch CPU 版、代码和权重，权重约 240 MB）：

```bash
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
git clone https://github.com/VQAssessment/DOVER.git third_party/DOVER
```

```bash
curl -L -o third_party/DOVER/pretrained_weights/DOVER.pth --create-dirs https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth
```

```bash
.venv/Scripts/python -m pip install --no-deps -e third_party/DOVER
```

本次使用的 DOVER 提交为 `f1ddc96`（2024-08-12），权重 sha256 前缀
`f4a42c0bbc94`。DOVER 采用 S-Lab 许可，仅限非商用，这里只做内部验证。

先启动 llama-server（路径按本机 JoyAI 目录）：

```bash
llama-server -m Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj mmproj-Qwen3VL-2B-Instruct-F16.gguf --host 127.0.0.1 --port 8091 -ngl 99 -c 8192 -np 1
```

然后依次运行各阶段。`analyze` 会同时打标签和做美学打分，
可中断，重跑会跳过已完成的片段；如果标签已经全部完成，
它就只做美学打分，这时不需要启动 llama-server：

```bash
.venv/Scripts/python -m video_demo segment
```

```bash
.venv/Scripts/python -m video_demo signals
```

```bash
.venv/Scripts/python -m video_demo analyze
```

```bash
.venv/Scripts/python -m video_demo suggest
```

```bash
.venv/Scripts/python -m video_demo render
```

```bash
.venv/Scripts/python -m video_demo report
```

评测：

```bash
.venv/Scripts/python -m video_demo synth --set holdout
```

```bash
.venv/Scripts/python -m video_demo eval --set holdout
```

测试与代码检查：

```bash
.venv/Scripts/python -m pytest
```

```bash
.venv/Scripts/ruff check video_demo tests
```
