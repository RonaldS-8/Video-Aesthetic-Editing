# Video Aesthetic Editing

在本地设备上为原始视频自动生成**精彩瞬间剪辑建议**和**按场景整理的素材**：
切分镜头、逐段描述、美学打分、规划大纲、检索填槽，全程离线运行，
适合部署在 NAS 等边缘设备上。

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Offline](https://img.shields.io/badge/runs-offline-lightgrey)
![CPU](https://img.shields.io/badge/GPU-not%20required-orange)

![精彩瞬间演示](assets/highlights_demo.webp)

<sub>6 条旅行 vlog（共 13.5 分钟）自动生成的跨视频精彩瞬间，节选前 51 秒。
素材来自 Wikimedia Commons，作者与许可见
[data/raw/SOURCES.md](data/raw/SOURCES.md)；本动图按 CC BY-SA 3.0 发布。</sub>

## 简介

一段几十分钟的旅行或生活素材，真正值得留下的往往只有几分钟。本项目
把这件事拆成可验证的几步，每一步都给出可追溯的理由：

- **切分**：PySceneDetect 找硬切，线性混合检验找叠化，镜头内再按
  ≤10 秒切成片段。
- **描述**：Qwen3-VL-2B 读片段内全部关键帧（每 2 秒一张），按 JSON
  Schema 输出描述、场景、景别、时段、标签等。
- **美学打分**：[DOVER](https://github.com/VQAssessment/DOVER) 给出技术
  分、美学分和融合分，作为"这个镜头好不好"的主要依据。
- **精彩瞬间**：Qwen3.5-4B 先逐块（每 5 分钟）总结素材，再写出 2–8
  部分的大纲；每部分的场景槽位用 EmbeddingGemma 2 的文本、图像向量和
  BM25 检索填充，最终取输入时长的约 1/6。出游时随手拍的多段短视频
  按整条成组总结，场景可以跨镜头、跨视频。
- **按指令剪辑**：用户用一句话提要求（如“剪 60 秒的教堂，先外观后
  内部”），Qwen3.5-4B 读要求和场景目录，先挑场景、写成片设计，再分部分
  写大纲；槽位在全部场景里检索，成片按大纲顺序排列。
- **场景素材**：同样的场景划分之外，剔除废片，每个场景输出一段原画质
  视频和一份描述，方便自行组合。

长短视频走同一套流程；每个决定（选了哪段、为什么、剔除了哪段）都写在
输出的 JSON 里。

## 流水线

```text
data/raw/<集合>/*
  │
  ├─ ⓪ ingest     登记视频、按内容去重          → data/raw/manifest.json
  ├─ ① segment    镜头切分 + ≤10 s 片段         → data/work/segments/
  ├─ ② signals    亮度、清晰度、抖动、关键帧     → data/work/signals/、keyframes/
  ├─ ③ tag        Qwen3-VL-2B 描述与标签        → data/work/tags/
  ├─ ④ aesthetic  DOVER 美学打分                → data/work/aesthetic/
  │      ③④ 可用 analyze 并发：打标签用核显，DOVER 用 CPU
  ├─ ④' index     EmbeddingGemma 2 文本与图像向量 → data/work/index/
  ├─ ⑤ suggest    大纲 + 检索填槽               → out/<集合>/<原名>/timeline_<模式>.json
  ├─ ⑥ render     预览视频                      → out/<集合>/<原名>/preview_<模式>.mp4
  ├─ ⑦ report     HTML 报告                     → out/<集合>/report_<模式>.html
  ├─ ⑧ scenes     场景素材（剔除废片）           → out/<集合>/<原名>/scenes/
  └─ ⑨ edit       按指令：大纲 + 全素材检索       → out/<集合>/edits/<时间>/
```

片段得分 = 1 + 4 ×（0.8 × DOVER 融合分 + 0.2 × VLM 精彩度）− 画质扣分；
DOVER 分先在同一时段（白天、黄昏、夜晚）内取百分位，避免夜景整体落选。
得分 ≤2 或短于 1.5 秒的片段视为废片，不进入任何输出。

剪辑建议有两种模式：

| 模式 | 做法 |
|---|---|
| `highlights`（默认） | 分块总结 → 大纲 → 每个槽位检索一个场景，场景内按得分、相关度和总结点名的亮点整段选片 |
| `ranked` | 对照基线：不用大模型，只按得分整段选片，每 5 分钟至少一段 |

## 效果

本机实测：Core Ultra 5 225U、15.4 GB 内存、Intel 核显，无独显。

**精彩瞬间**（目标为输入时长的 1/6）：

| 素材 | 时长 | 片段 | 目标 | highlights | ranked |
|---|---|---|---|---|---|
| 6 条旅行 vlog（854×480） | 13.5 分钟 | 162 | 134.9 s | 106.7 s，21 段 | 136.5 s，29 段 |
| 爱丁堡街头漫步（1080p60） | 67 分钟 | 403 | 670.1 s | 646.5 s，65 段 | 679.0 s，68 段 |

**场景素材**（剔除废片后，每个场景一个原画质文件）：

| 素材 | 场景 | 输出视频 | 由多段接成 | 保留 | 剔除 |
|---|---|---|---|---|---|
| 6 条旅行 vlog | 33 | 31 | 9 | 567 s | 243 s |
| 爱丁堡街头漫步 | 61 | 59 | 19 | 3314 s | 707 s |

**耗时**（爱丁堡 67 分钟，403 段）：切分 40 分钟，画质信号 7 分钟，
打标签 18.8 秒/段，DOVER 21.4 秒/段（两者并发），向量索引 31 分钟，
13 个分块总结约 12 分钟，场景概况 11 分钟，1080p60 原画质切片约每秒
素材 0.6 秒。

**切分**（合成视频，切点已知，容差 ±0.5 秒）：

| 评测集 | 切点 | 准确率 | 召回率 | F1 |
|---|---|---|---|---|
| 调参集（硬切 4、跳切 1、叠化 2） | 7 | 1.00 | 1.00 | 1.00 |
| 留出集（硬切 2、叠化 4） | 6 | 1.00 | 0.33 | 0.50 |

**质量模型对比**（177 段，2 线程，均与 DOVER 计秩相关）：

| 模型 | 每小时素材 CPU 时间 | 峰值内存 | 与 DOVER 融合分 ρ | 许可 |
|---|---|---|---|---|
| DOVER | 5–8 h | 5.4 GB | — | S-Lab，非商用 |
| DOVER-Mobile | 0.7–0.9 h | 2.0 GB | 0.76 | S-Lab，非商用 |
| UVQ 1.5 | 1.0–1.1 h | 1.1–1.4 GB | 0.64 | Apache-2.0 |

三个模型都会压低夜景，UVQ 1.5 最明显，且对纯黑画面给出中等分数；
所以质量分只用于候选内的重排，不单独用来过滤。复现：
`quality --model {dover-mobile,uvq}` 后运行 `compare-quality`。

## 安装

需要 Python 3.11。PyAV 自带 FFmpeg 库，无需另装 ffmpeg。

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

DOVER 代码放在 `third_party/`（不入库）：

```bash
git clone https://github.com/VQAssessment/DOVER.git third_party/DOVER
```

```bash
.venv/Scripts/python -m pip install --no-deps -e third_party/DOVER
```

MediaPipe 需要不带依赖安装，否则会引入 opencv-contrib-python，与项目的
opencv-python-headless 冲突：

```bash
.venv/Scripts/python -m pip install --no-deps mediapipe==1.1.1
```

**模型权重**统一放在 `models/` 下（约 5.5 GB，不入库）。每个文件的
存放路径、下载地址、版本和 SHA-256 见 [models/README.md](models/README.md)。

| 模型 | 用途 |
|---|---|
| Qwen3-VL-2B-Instruct Q4_K_M + 视觉投影 | 片段描述与标签 |
| Qwen3.5-4B Q4_K_M | 分块总结、大纲、场景概况 |
| DOVER | 美学打分 |
| EmbeddingGemma 2 740M | 检索与场景聚类的向量 |
| DOVER-Mobile、UVQ 1.5 | 仅用于质量模型对比 |

两个语言模型由 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的
`llama-server` 提供 OpenAI 兼容接口。

## 快速开始

所有命令在项目根目录执行。把视频放进 `data/raw/<集合>/`（例如
`data/raw/long/`），然后：

**1. 导入**：登记新视频、标出重复文件，并完成切分和画质信号。

```bash
.venv/Scripts/python -m video_demo ingest
```

**2. 描述与打分**：先在另一个终端启动打标签服务。

```bash
llama-server -m models/gguf/Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj models/gguf/mmproj-Qwen3VL-2B-Instruct-F16.gguf --host 127.0.0.1 --port 8091 -ngl 99 -c 8192 -np 1
```

```bash
.venv/Scripts/python -m video_demo analyze --collection long
```

**3. 向量索引**：

```bash
.venv/Scripts/python -m video_demo index --collection long
```

**4. 精彩瞬间与场景素材**：启动大纲服务，再依次运行。

```bash
llama-server -m models/gguf/Qwen3.5-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8092 -ngl 99 -c 16384 -np 1
```

```bash
.venv/Scripts/python -m video_demo suggest --collection long
```

```bash
.venv/Scripts/python -m video_demo render --collection long
```

```bash
.venv/Scripts/python -m video_demo report --collection long
```

```bash
.venv/Scripts/python -m video_demo scenes --collection long
```

每一步都可以中断，重跑会从已完成的位置继续。`suggest`、`render`、
`report` 加 `--mode ranked` 生成对照基线；`--only <前缀>` 只处理某条
视频（视频 ID 或文件路径的前缀）；`--ratio` 调整精彩瞬间占输入时长的
比例。模型输出未通过校验时最多重试 3 次，仍失败就报错，不会填默认值。

## 按指令剪辑

完成第 1–3 步并启动大纲服务后，用一句话说明想要的成片：

```bash
.venv/Scripts/python -m video_demo edit "剪 60 秒的爱丁堡教堂，先外观后内部" --collection long
```

一条命令生成时间线、预览和报告，放在 `out/<集合>/edits/<时间>/` 下，
每条指令一个目录，互不覆盖。做法：

1. **时长**：模型从要求里读出成片时长，并照抄说时长的原话（如
   “60 秒”）；原话必须出现在要求里且带数字和时间单位，否则视为编造、
   重试。要求里没写时长就按 `--ratio`（默认输入时长的 1/6）。
2. **大纲**：模型读要求和场景目录（每个场景的短标题和概况，多条视频时
   注明在第几条；与场景素材共用缓存，缺的当场生成），依次写出：挑中的
   场景和可用内容、2–3 句成片设计（开头、中间、结尾各用什么）、节奏、
   要求里提到但素材里找不到的内容，最后照设计按成片先后分 1–8 个部分，
   每部分 1–4 个场景槽位，标明取自哪几个场景。场景目录比 5 分钟一块的
   总结细，个别片段的时段误标不会扩散成整块的描述；时段等画面要求写进
   槽位的检索短语，只影响相关度，不做硬过滤。场景超过 60 个时（约
   1 小时以上的素材），先用整句要求检索出最相关的 60 个交给模型，
   保证大纲调用在 6k 上下文内。
3. **节奏**：模型按要求选「整段」（每段约 10 秒完整保留）、「适中」
   （每段最多 6 秒）或「快」（每段最多 3 秒，适合混剪）；截短时取片段
   中间，避开起止时的晃动。
4. **填槽**：每个槽位在全部场景里检索，所有（槽位, 场景）对按相关度和
   画质从高到低分配，一个场景只用一次。场景内选片时画质（片段得分）和
   与槽位的相关度都缩放到 0–1：精彩瞬间画质占 80%，按指令剪辑相关度
   占 70%，免得画质高但不对题的镜头（如找内景时选到明亮的外立面）
   排到昏暗但对题的镜头前面。

报告列出要求原文、挑中的场景、成片设计、时长来源、每个槽位选中的场景
（全部关键帧，取用的片段加框，并标出是否为设计里的场景）和其他候选、
预览、逐段理由，以及模型输出的大纲原文。

## 多条短视频

出游时在多个地方随手拍的短视频放进同一个集合（如 `data/raw/vlog/`），
`suggest` 和 `edit` 把它们当作一条时间线处理：

- **顺序**：每条视频都带拍摄时间（手机、相机写入的 `creation_time`）
  时按拍摄时间排，缺任何一条就按文件名排，不去猜；采用哪种写进时间线
  和报告。
- **分块**：5 分钟以上的视频照旧每 5 分钟一块；更短的视频先各自总结，
  再按顺序凑满 5 分钟成一组，由模型读组内各条视频的总结写组总结，
  一条视频不会被拆到两组里。组总结按组内视频的总结缓存，任何一条
  重新总结时组总结随之重写。
- **场景**：镜头切换和视频交界处，两侧内容相似（文本、图像向量相似度
  ≥0.80）就可以连成一个场景，不相似的保持断开、哪怕不满 20 秒。
  6 条 vlog（145 个镜头）从 130 个场景、中位 4.5 秒变为 33 个、中位
  25 秒，每个场景仍是同一个地方（唐人街灯笼、森林步道、日夜的鱼尾狮）；
  爱丁堡只有 3 处镜头切换，61 个场景变为 60 个。场景素材（`scenes`）
  仍按单条视频切出原画质片段。

## 输出

```text
out/<集合>/
├── <原名>/                       每条视频一个目录，按原文件名
│   ├── timeline_highlights.json  精彩瞬间时间线
│   ├── timeline_ranked.json      只按得分的对照
│   ├── preview_<模式>.mp4        预览（854×480）
│   └── scenes/
│       ├── scene_<NN>.mp4        场景 NN 剔除废片后的原画质视频
│       └── scenes.json           每个场景的描述与来源
├── edits/<时间>/                 按指令剪辑，每条指令一个目录
│   ├── timeline.json             时间线，另含指令、时长来源、场景目录、大纲
│   ├── preview.mp4               预览
│   └── report.html               报告
├── timeline_<模式>.json          整个集合的时间线
├── preview_<模式>.mp4            整个集合的预览
└── report_<模式>.html            报告：切分评测、逐段标签与分数、剪辑建议
```

- **时间线**记录每个入选片段的原片起止、所属大纲部分和入选理由，
  以及模型给出的大纲、分块总结、场景划分和每个槽位的检索结果。
- **场景素材**保持原分辨率、原帧率和音轨（H.264 + AAC，切点精确到帧），
  原片只读不改。`scenes.json` 给出每个场景的短标题、概况、时段、景别和
  高频标签，场景视频由原片哪几段接成，以及被剔除的片段和理由。
- 视频文件不入库；报告是静态 HTML，克隆后用浏览器直接打开。

<details>
<summary>文件字段说明</summary>

所有时间都是相对原视频开头的秒数；片段 `index` 从 0 开始，是同一视频
各文件之间关联片段的键。

`segment_<ID>.json`：`duration`、`analysed_until`、`fps`、`has_audio`、
`cuts`（`time`、`kind`：`hard` / `dissolve`）、`shots`、`segments`
（`index`、`start`、`end`、`shot`）。

`signal_<ID>.json`（每段一项）：`brightness`、`sharpness`、`motion`、
`pan_speed`、`shake`、`flags`（`low_light`、`blurry`、`shaky`）、
`keyframes`、`keyframe_times`。

`tag_<ID>.jsonl`（每行一段）：`tags.description`、`scene`、`subjects`、
`action`、`shot_size`、`time_of_day`、`mood`、`tags`、`highlight`（1–5）、
`issues`，以及模型原始输出 `raw` 和被拒绝的重试 `failed_attempts`。

`aesthetic_<ID>.jsonl`（每行一段）：DOVER 的 `technical`、`aesthetic`、
`fused`（0–1），解码帧数、耗时、模型名、权重 SHA-256 和随机种子。

`timeline_<模式>.json`：`target_sec`、`total_sec`、`notes`、`clips`
（`order`、`role`、`video`、`index`、`src_start`、`src_end`、`description`、
`reason`）、`assessments`（每段得分、是否保留、理由）；highlights 另有
`outline`、`chunks`、`scenes`、`slots`、`retrieval`。

`scenes/scenes.json`：`rule`、`complete`、`kept_sec`、`removed_sec`，
`scenes` 里每个场景有 `title`、`summary`、`file`、`duration`、
`time_of_day`、`shot_sizes`、`tags`、`parts`（原片 `start`、`end`，在场景
视频中的起点 `at`，所含片段）和 `removed`（片段、`why`、`reasons`）。

</details>

## 数据约定

- **视频 ID** 是文件内容 SHA-256 的前 16 位，与文件名无关：改名、移动
  不影响结果，内容相同的文件只处理一次。对应关系记在
  `data/raw/manifest.json`。
- 原始视频保留原名和原位置，不复制、不改动。中间结果按
  `<类型>_<ID>` 命名放在 `data/work/` 下；输出按原文件名放在 `out/` 下，
  同一集合里两个文件只差扩展名时会报错。
- `data/raw/` 下第一层文件夹是一个**集合**（如 `vlog/`、`long/`），
  剪辑建议和报告按集合输出。
- 支持 mp4、mov、m4v、3gp、mkv、webm、avi、mts、m2ts，编码不限，PyAV
  能解码即可。

## 目录结构

```text
video_demo/      代码，python -m video_demo <阶段>
tests/           pytest 测试
models/          模型权重（不入库）与索引 models/README.md
third_party/     DOVER、UVQ 1.5 的代码（不入库）
data/raw/        原始视频（不入库）、清单与来源说明
data/work/       中间结果：切分、信号、关键帧、标签、美学分、向量、总结
out/             剪辑建议、预览、报告与场景素材；out/report.html 为
                 只用 VLM 精彩度选片的早期基线
assets/          README 用图
```

## 已知局限

- **渐变转场**：叠化检测只在调参集上有效，留出集 4 个叠化全部漏检；
  计划接入学习型检测器（如 TransNetV2）。
- **场景填槽欠时长**：一个槽位只取一个场景，场景装不满时如实标注，
  不从别处补。
- **大纲先后不稳**：4B 写按指令剪辑的设计时偶尔违背要求里的顺序，
  如“先外观后内部”被写成外、内、外的“闭环”；同一条要求在爱丁堡上
  6 次里 2 次如此。报告列出设计和大纲原文，便于发现后重跑。
- **极短视频**：目标时长比一个片段还短时会整段入选，17 秒的视频选出
  8.3 秒。
- **夜景**：质量模型普遍压低夜景，目前靠按时段取百分位校准，效果仍需
  人工看片确认。
- **语音**：尚未做语音转写，以对白为主的片段识别不了。

## 开发

```bash
.venv/Scripts/python -m pytest
```

```bash
.venv/Scripts/ruff check .
```

```bash
.venv/Scripts/black --check .
```

切分评测可用合成视频复现：`python -m video_demo synth --set holdout`，
再运行 `python -m video_demo eval --set holdout`。

## 许可与致谢

本项目代码按 [Apache-2.0](LICENSE) 发布。第三方模型和素材各自适用其
许可：

- DOVER、DOVER-Mobile：S-Lab License，**仅限非商业用途**。
- UVQ 1.5、Qwen3-VL、Qwen3.5：Apache-2.0。
- EmbeddingGemma 2：仓库元数据标为 Apache-2.0，模型卡指向 Gemma 许可，
  商用前请自行确认。
- 示例视频：CC BY 3.0 / CC BY-SA 3.0，见
  [data/raw/SOURCES.md](data/raw/SOURCES.md)。

感谢 [DOVER](https://github.com/VQAssessment/DOVER)、
[UVQ](https://github.com/google/uvq)、
[PySceneDetect](https://github.com/Breakthrough/PySceneDetect)、
[llama.cpp](https://github.com/ggml-org/llama.cpp)、
[Qwen](https://github.com/QwenLM)、
[MediaPipe](https://github.com/google-ai-edge/mediapipe) 和
[PyAV](https://github.com/PyAV-Org/PyAV)。
