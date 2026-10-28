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
| `models/gguf/Qwen3VL-2B-Instruct-Q4_K_M.gguf` | 打标签（VLM） | 1107409952 | Apache-2.0 |
| `models/gguf/mmproj-Qwen3VL-2B-Instruct-F16.gguf` | 打标签的视觉投影 | 819394848 | Apache-2.0 |
| `models/gguf/Qwen3.5-4B-Q4_K_M.gguf` | 精彩瞬间的分块摘要与大纲、场景素材的概况 | 2783446912 | Apache-2.0 |
| `models/dover/DOVER.pth` | 美学打分（参与剪辑建议） | 239729097 | S-Lab，非商用 |
| `models/dover/DOVER-Mobile.pth` | 质量模型对比 | 42794791 | S-Lab，非商用 |
| `models/uvq/content_net.pth` | 质量模型对比（UVQ 1.5） | 15289209 | Apache-2.0 |
| `models/uvq/distortion_net.pth` | 质量模型对比（UVQ 1.5） | 15290142 | Apache-2.0 |
| `models/uvq/aggregation_net.pth` | 质量模型对比（UVQ 1.5） | 300021 | Apache-2.0 |
| `models/embeddinggemma/embeddinggemma-2-740m.litertlm` | 检索和场景聚类用的文本与图像向量 | 484622336 | 见下文 |

合计约 5.5 GB。剪辑建议（精彩瞬间）需要三个 gguf、`DOVER.pth` 和
EmbeddingGemma 2；只按得分的对照（`--mode ranked`）不需要 Qwen3.5-4B
和 EmbeddingGemma 2。DOVER-Mobile 与 UVQ 1.5 只在 `quality` /
`compare-quality` 阶段用到。

## 打标签：Qwen3-VL-2B-Instruct（GGUF）

- 来源：ModelScope `Qwen/Qwen3-VL-2B-Instruct-GGUF`，文件修订
  `1d49f1a70c0c1ea52f82fd46ea44b388183aa16c`。
- 运行：llama.cpp `llama-server`（Vulkan 版），端口 8091，命令见项目
  README 的"依次运行"。

| 文件 | SHA-256 |
|---|---|
| `Qwen3VL-2B-Instruct-Q4_K_M.gguf` | `089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae` |
| `mmproj-Qwen3VL-2B-Instruct-F16.gguf` | `c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5` |

两个哈希与 ModelScope 文件列表中的值一致。

```bash
for f in Qwen3VL-2B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-2B-Instruct-F16.gguf; do curl -L --create-dirs -o "models/gguf/$f" "https://www.modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/master/$f"; done
```

## 大纲：Qwen3.5-4B（GGUF，Q4_K_M）

- 来源：官方权重 `Qwen/Qwen3.5-4B`（Transformers 格式），在本机用
  llama.cpp（提交 `e36df60`，2026-10-15）转为 GGUF 后量化为 Q4_K_M。
  转换时的命令没有留下记录，以下面的 SHA-256 为准。
- 运行：llama.cpp `llama-server`，端口 8092，不需要视觉投影。

| 文件 | SHA-256 |
|---|---|
| `Qwen3.5-4B-Q4_K_M.gguf` | `28f8d724a519e33e7b8b394dafde357c02f14ae44855fb224296e4318f2861cd` |

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

- 清单：`models/embeddinggemma/embeddinggemma-2-740m.json`（入库，抄自
  A-NAS 的 `deploy/models/`），记录来源、修订号、大小和 SHA-256；
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
