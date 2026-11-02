# 模型权重索引

所有权重放在本目录下，按模型分子目录。权重文件不入库，只入库本说明
和清单；拿到代码后按下表把文件放到对应路径。第三方模型的代码放在
`third_party/`（见项目 README 的"安装"）。

路径都相对于项目根目录。SHA-256 是本机实际使用的文件算出的值，
下载后请核对：

```bash
sha256sum models/*/*.pth models/*/*.litertlm models/*/*.gguf
```

## 总表

| 路径 | 用途 | 大小（字节） | 许可 |
|---|---|---|---|
| `models/gguf/Qwen3.5-4B-Q4_K_M.gguf` | 打标签、分块摘要、大纲、场景概况 | 2783446912 | Apache-2.0 |
| `models/gguf/mmproj-Qwen3.5-4B-Q8_0.gguf` | 打标签的视觉投影 | 366894656 | Apache-2.0 |
| `models/gguf/Hy-MT2-1.8B-Q4_K_M.gguf` | 字幕翻译 | 1133080448 | Apache-2.0 |
| `models/dover/DOVER.pth` | 美学打分（参与剪辑建议） | 239729097 | S-Lab，非商用 |
| `models/dover/DOVER-Mobile.pth` | 质量模型对比 | 42794791 | S-Lab，非商用 |
| `models/uvq/content_net.pth` | 质量模型对比（UVQ 1.5） | 15289209 | Apache-2.0 |
| `models/uvq/distortion_net.pth` | 质量模型对比（UVQ 1.5） | 15290142 | Apache-2.0 |
| `models/uvq/aggregation_net.pth` | 质量模型对比（UVQ 1.5） | 300021 | Apache-2.0 |
| `models/embeddinggemma/embeddinggemma-2-740m.litertlm` | 检索和场景聚类用的文本与图像向量 | 484622336 | 见下文 |
| `models/Kokoro-82M-v1.1-zh/kokoro-v1_1-zh.pth` | 解说配音（另有 55 个女声 `voices/zf_*.pt`，共 28783205） | 327247856 | Apache-2.0 |

合计约 5.4 GB。剪辑建议（精彩瞬间）需要两个 gguf、`DOVER.pth` 和
EmbeddingGemma 2；只按得分的对照（`--mode ranked`）只用 Qwen3.5-4B
打标签，不需要 EmbeddingGemma 2。DOVER-Mobile 与 UVQ 1.5 只在 `quality` /
`compare-quality` 阶段用到。Hy-MT2 只在 `subtitle` 遇到外语时用到；Kokoro 只在 `edit --narrate` 时用到。

## 打标签与大纲：Qwen3.5-4B（GGUF，Q4_K_M）

- 来源：官方权重 `Qwen/Qwen3.5-4B`（Transformers 格式），在本机用
  llama.cpp（提交 `e36df60`，2026-10-15）转为 GGUF 后量化为 Q4_K_M。
  转换时的命令没有留下记录，以下面的 SHA-256 为准。
- 视觉投影 `mmproj-Qwen3.5-4B-Q8_0.gguf` 与上面的权重配套，同样没有留下
  转换记录，以 SHA-256 为准。
- 运行：llama.cpp `llama-server`（Vulkan 版），端口 8092，带 `--mmproj`；
  看图打标签和写摘要、大纲共用这一个服务，命令见项目 README 的
  "快速开始"。

| 文件 | SHA-256 |
|---|---|
| `Qwen3.5-4B-Q4_K_M.gguf` | `28f8d724a519e33e7b8b394dafde357c02f14ae44855fb224296e4318f2861cd` |
| `mmproj-Qwen3.5-4B-Q8_0.gguf` | `40a4f07d7bbdbb43011d6cf35ef751e4b1829ff47ee8aa4964c6296f571725ad` |

## 美学打分：DOVER / DOVER-Mobile

- 代码：`third_party/DOVER/`，`VQAssessment/DOVER` 提交 `f1ddc96`
  （2024-08-12）。配置文件 `dover.yml` / `dover-mobile.yml` 里的
  `test_load_path` 只取文件名，到 `models/dover/` 下查找。
- `DOVER.pth`：`QualityAssessment/DOVER` release v0.1.0。
- `DOVER-Mobile.pth`：`QualityAssessment/DOVER` release v0.5.0。

| 文件 | SHA-256 |
|---|---|
| `DOVER.pth` | `f4a42c0bbc94c94dd7409e7f40887d44c5c30314d1d09e7edf03cc35813b4838` |
| `DOVER-Mobile.pth` | `81b487be2aa4b3dd6920afa2e92294ed8fdd46a306911f75ecc8e6938a670884` |

```bash
curl -L --create-dirs -o models/dover/DOVER.pth https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth
```

```bash
curl -L --create-dirs -o models/dover/DOVER-Mobile.pth https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth
```

## 质量对比：UVQ 1.5

- 代码：`third_party/uvq/`，`google/uvq` 提交
  `c204ecc04f093e90c941cb126c7ac3d5beca0414`。
- 权重：同一提交的 `uvq1p5_pytorch/checkpoints/` 下三个文件，放到
  `models/uvq/`。

| 文件 | SHA-256 |
|---|---|
| `content_net.pth` | `a4ec16d43924d19da964f981ec69fb451c0bf9e24a8482b39f53833804847fbb` |
| `distortion_net.pth` | `e2fcabaa5526894ec5bc5b39ead2b15bfac9bb296f337394d29fc335fe772742` |
| `aggregation_net.pth` | `0f214c76adfbc3d06b48211837cfc46847ef6fe0102804d4d097ba170e769fc6` |

```bash
for f in aggregation_net.pth content_net.pth distortion_net.pth; do curl -sSfL --create-dirs -o "models/uvq/$f" "https://raw.githubusercontent.com/google/uvq/c204ecc04f093e90c941cb126c7ac3d5beca0414/uvq1p5_pytorch/checkpoints/$f"; done
```

## 检索向量：EmbeddingGemma 2 740M（LiteRT-LM）

- 清单：`models/embeddinggemma/embeddinggemma-2-740m.json`（入库，与端侧
  相册客户端使用同一份），记录来源、修订号、大小和 SHA-256；
  加载时会核对，不一致就报错。
- 来源：Hugging Face `litert-community/embeddinggemma-2-740m-litert-lm`，
  修订 `24d962e906c7d332c6428e71c9676855024569e2`。
- 许可：仓库元数据标为 Apache-2.0，模型卡链接的是 Gemma 许可，
  对外发布前需要确认。
- 运行：MediaPipe Universal Embedder。首次加载会在
  `models/embeddinggemma/cache/` 生成约 370 MB 的 XNNPack 缓存，
  可以删除，下次加载时重建。

| 文件 | SHA-256 |
|---|---|
| `embeddinggemma-2-740m.litertlm` | `e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c` |

```bash
curl -L --create-dirs -o models/embeddinggemma/embeddinggemma-2-740m.litertlm https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/resolve/24d962e906c7d332c6428e71c9676855024569e2/embeddinggemma-2-740m.litertlm
```

## 字幕翻译：Hy-MT2-1.8B（GGUF，Q4_K_M）

- 来源：Hugging Face `tencent/Hy-MT2-1.8B-GGUF` 的
  `Hy-MT2-1.8B-Q4_K_M.gguf`（翻译分支下载时未记修订号，以
  SHA-256 为准）。
- 运行：llama.cpp `llama-server`，端口 8093，命令见项目 README 的
  "字幕翻译"。

| 文件 | SHA-256 |
|---|---|
| `Hy-MT2-1.8B-Q4_K_M.gguf` | `dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699` |

## 解说配音：Kokoro-82M-v1.1-zh（PyTorch）

- 来源：ModelScope `hexgrad/Kokoro-82M-v1.1-zh`，修订
  `3bb92ed837a15179e8ccae96912cf71b196279ee`。只下了权重、`config.json`、
  `README.md` 和 55 个普通话女声 `voices/zf_*.pt`（男声与英文音色没下）。
- 清单：`models/Kokoro-82M-v1.1-zh/download.json`（入库）记录每个文件的
  路径、大小和 SHA-256，下载时已逐个与 ModelScope 给出的值核对。
- 运行：官方 `kokoro` 包加 `misaki[zh]`（见 `requirements.txt`），只用
  CPU，只读本地文件，不联网。只合成普通话，遇到英文直接报错。
- 默认音色 `zf_088`，可用 `edit --voice` 换。

| 文件 | SHA-256 |
|---|---|
| `kokoro-v1_1-zh.pth` | `b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b` |
| `config.json` | `bc333efa5ce4ceff433c8c8e5d027a1eca0166001e4e4a62bea2d26ff7a46890` |

按清单下载（音色的 SHA-256 见 `download.json`）：

```bash
for f in $(python -c "import json; print(' '.join(x['path'] for x in json.load(open('models/Kokoro-82M-v1.1-zh/download.json'))['files']))"); do curl -sSfL --create-dirs -o "models/Kokoro-82M-v1.1-zh/$f" "https://www.modelscope.cn/api/v1/models/hexgrad/Kokoro-82M-v1.1-zh/repo?Revision=3bb92ed837a15179e8ccae96912cf71b196279ee&FilePath=$f"; done
```
