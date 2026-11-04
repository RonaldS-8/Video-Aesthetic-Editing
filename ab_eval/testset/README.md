# 方案 B 交付说明：一条命令出片

本文说明如何在一台新机器上装好环境，用一条命令把一个用例从
`case.json` 剪成成片。默认全程在本地，用 Qwen3.5-4B，不联网。

## 1. 整体流程

```text
case.json + 素材
  │
  ├─ 打标预处理（prepare）
  │    check     校验 case.json，按 sha256 核对素材
  │    stage     把素材链接进本用例的数据目录
  │    ingest    语音识别、避开说话处切分、画面信号
  │    shortlist 筛候选片段
  │    analyze   4B 给候选打标签，DOVER 打画质分
  │    subtitle  字幕（只在出字幕版时）
  │    index     画面与台词向量
  │    catalog   片段与场景摘要
  │
  └─ 用户指令剪辑（cut）
       edit      按指令和目标时长写大纲、选片、审片
       render    原声渲染，句子完整，1080p
```

两个阶段的耗时分别记在 `trace.json` 里。

## 2. 要打包的文件

从仓库根目录打包下面这些，目录结构保持不变：

| 路径 | 内容 |
|---|---|
| `ab_eval/` | 入口和封装（不含 `results/` 和 `testset/` 的素材） |
| `video_demo/` | 已有的剪辑流程 |
| `prompts/` | 各素材类型的提示词 |
| `third_party/DOVER/` | DOVER 画质模型的代码 |
| `requirements.txt` | Python 依赖 |
| `pyproject.toml` | 包与工具配置 |
| `models/README.md` | 权重清单：路径、来源、大小、SHA-256、下载命令 |
| `models/embeddinggemma/*.json` | 向量模型清单，加载时核对 |
| `models/Kokoro-82M-v1.1-zh/download.json` | 配音模型清单 |

**权重文件不打包**，用户按下表自己下载，放到对应路径（约 5.5 GB）。
来源、SHA-256 和下载命令见 `models/README.md`，下载后请核对 SHA-256。

| 放到 | 用途 | 来源 |
|---|---|---|
| `gguf/Qwen3.5-4B-Q4_K_M.gguf` | 打标签、摘要、大纲、审片 | `Qwen/Qwen3.5-4B` 转 GGUF |
| `gguf/mmproj-Qwen3.5-4B-Q8_0.gguf` | 4B 的视觉投影 | 同上 |
| `gguf/Hy-MT2-1.8B-Q4_K_M.gguf` | 字幕翻译 | `tencent/Hy-MT2-1.8B-GGUF` |
| `asr/silero_vad.onnx` | 人声检测 | sherpa-onnx 官方模型 |
| `asr/sherpa-onnx-sense-voice-…/` | 语音识别 | sherpa-onnx 官方模型 |
| `dover/DOVER.pth` | 画质评分 | `QualityAssessment/DOVER` v0.1.0 |
| `embeddinggemma/*.litertlm` | 画面与文字向量 | Hugging Face `litert-community` |
| `Kokoro-82M-v1.1-zh/` | 中文解说配音 | ModelScope `hexgrad/Kokoro-82M-v1.1-zh` |

路径都相对于模型目录（默认 `models/`，可用 `VIDEO_DEMO_MODELS` 改）。
Hy-MT2 只在出字幕版时用到，Kokoro 只在出解说版时用到。

不要打包：任何权重文件、`data/`、`out/`、`docs/`、`ab_eval/results/`。

另外需要 llama.cpp 的 `llama-server`，单独下载或编译，放进 `PATH`，
或用环境变量 `LLAMA_SERVER` 指到它。入口会按步骤自动启停模型服务。

## 3. 环境

1. Python 3.11 或以上，新建虚拟环境：

   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```

2. 先装 CPU 版 torch，再装依赖：

   ```bash
   pip install torch torchvision \
       --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   pip install --no-deps -e third_party/DOVER
   pip install --no-deps mediapipe==1.1.1
   ```

3. 中文字幕字体：Windows 默认用微软雅黑；Linux 默认用
   `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
   （Debian 的 `fonts-noto-cjk` 包）。

## 4. 路径配置

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `VIDEO_DEMO_MODELS` | 模型目录 | 仓库下的 `models/` |
| `LLAMA_SERVER` | llama-server 程序 | `PATH` 里的 `llama-server` |
| `VIDEO_DEMO_DATA` | 中间数据目录 | 由入口设为 `<out>/work/data` |
| `VIDEO_DEMO_OUT` | 剪辑工作目录 | 由入口设为 `<out>/work/out` |
| `VIDEO_DEMO_FONT` | 字幕字体 | 见上 |
| `ANTHROPIC_API_KEY` | 可选，设了就用云端写大纲和审片 | 不设，走本地 4B |

`--work DIR` 可以把中间数据放到别处，例如 `/tmp`。

## 5. 运行

模型服务由入口按步骤自动启停：4B 用 8092 端口，Hy-MT2 用 8093 端口，
两者不同时运行。端口上已经有服务在跑时，直接用它，用完也不关。

跑一个用例：

```bash
python -m ab_eval.run /case/case-01.json /out --no-narrated --no-subtitles
```

常用选项：

| 选项 | 作用 |
|---|---|
| `--planner local` | 用本地 4B 写大纲和审片（默认） |
| `--planner api` | 用云端，需要 `ANTHROPIC_API_KEY` |
| `--no-narrated` | 不出解说版 |
| `--no-subtitles` | 不出字幕 |
| `--prepare-only` | 只做打标预处理 |
| `--narrate-cut` | 只在已有成片上出解说版，成片不动 |
| `--work DIR` | 中间数据目录 |

## 6. 输出

| 文件 | 内容 |
|---|---|
| `output.mp4` | 按用例要求的成片：原声，无配乐、字幕和片头 |
| `narrated/output.mp4` | 特色版：AI 解说和中文字幕 |
| `narrated/subtitles/` | 字幕文件 |
| `timeline.json` | 每个镜头的来源和起止时间 |
| `report.html` | 可视化报告，浏览器打开 |
| `prompts/` | 写大纲和审片时发给模型的提示词 |
| `trace.json` | 各步骤耗时，OTLP 格式 |
| `steps.json` | 打标预处理和用户指令剪辑两段的耗时汇总 |

## 7. 测评资源

- 8 个 CPU，内存尽量控制在 3.4 GiB，最多 4.5 GiB；
- 同一时刻只运行一个大模型；
- 按每个用例 3 小时以内设计。
