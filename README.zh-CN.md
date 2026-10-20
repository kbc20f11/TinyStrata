# TinyStrata

[English](README.md) · **简体中文** · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

**TinyStrata 是一款正在开发中的高速推理引擎，面向 30B 级 MoE 模型，仅支持 Windows，且仅提供 API。** 主要面向配备主流显卡、显存为 **8–24 GB** 且系统内存**不超过 32 GB** 的 PC。

## 为什么开发 TinyStrata？

随着 PC 配件价格上涨，组装一台高端 PC 对许多人来说正变得越来越难。

因此，我们开发了 TinyStrata，让 AI 能在配备 RTX 3060 级别或更高性能主流显卡的 PC 上，以本地方式高速运行。我们专注于让 30B 级的较小型 MoE 模型尽可能快地运行。

我们的主要目标是配备 **8–24 GB 显存、系统内存不超过 32 GB** 的 PC。我们认为，这一范围涵盖了配备主流显卡的 PC 中最常见的配置。

TinyStrata 基于 Strata，致力于高效利用 GPU 内存，让 30B 级 MoE 模型即使在显存有限的情况下也能高速运行。我们的目标是让你能在自己的 PC 上处理长对话和文档。

开发工作专注于 Windows 上的本地推理及其 API 服务。

TinyStrata 会根据你的硬件自动选择推理和内存设置，因此无需复杂配置或手动调优即可开始使用 API。

## 为什么不附带 UI？

目前已有许多支持 LLM API 的 UI 和智能体运行框架。TinyStrata 也支持常见的 API 格式，因此可以连接你喜欢的软件来使用。如果连接配置让你觉得繁琐，也可以请 AI 智能体帮你完成。将 UI 交给现有软件，让我们能够专注于推理引擎和 API 的开发。

## 重点支持的模型

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 仍是默认模型，适合在配备主流显卡的 PC 上兼顾速度、内存效率和回答质量。TinyStrata 专注于较小的 MoE 模型，并支持 **Gemma 4 26B-A4B-it 和 LFM2.5-24B-A2B**。

**Qwen3.6-35B-A3B UD-IQ3_XXS** 使用固定版本的 [Unsloth MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf)。两个文件共约 **15.0 GB**：包含 MTP 权重的语言模型 **14.1 GB**，视觉投影器（`mmproj-F16.gguf`）**0.9 GB**。

**Gemma 4 26B-A4B-it QAT Q4_0** 使用 [Google 官方模型及视觉投影器](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)，以及[匹配的 Unsloth QAT MTP 助手](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/mtp-gemma-4-26B-A4B-it.gguf)。三个文件共 **15.886 GB**：模型 **14.439 GB**、投影器 **1.195 GB**、助手 **0.252 GB**。

**LFM2.5-24B-A2B Q4_K_M** 使用官方 [LiquidAI GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF)。仅需下载一个语言模型文件，大小为 **14.53 GB**。

三个引擎均支持根据空闲显存自动调整的专家缓存、利用系统 RAM 的 CPU 专家计算、输入阶段权重流式传输及 CUDA Graph。它们使用相同的 API，支持运行中切换模型，并分别保存各模型的设置。不省略模型选中的专家。还支持按实际序列长度自动扩展 KV 物理内存，最大上下文容量保持不变。

| 功能 | [Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | [Gemma 4](https://huggingface.co/google/gemma-4-26B-A4B-it) | [LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) |
| --- | --- | --- | --- |
| 最大上下文 | 256K | 256K | 128K |
| 图像输入 | 支持；默认编码器常驻 GPU | 支持；默认编码器常驻 GPU | 不支持 |
| MTP | 3 | 3 | 不支持 |
| KV 格式 | FP32 / INT8 / 原生 FP8 | FP16 / INT8 / 原生 FP8（实验性） | FP16 / INT8 / 原生 FP8 |
| KV 自动选择（默认） | FP8 → INT8 | INT8 | FP8 → INT8 |

表头中的模型名称链接到官方模型卡。**FP8 → INT8** 表示 GPU 与已编译内核支持时使用 FP8 KV 和原生 FP8 Tensor Core Attention，否则使用 INT8。KV 精度与 GGUF 权重量化相互独立；低精度 KV 可能产生与 FP32（Qwen）或 FP16（Gemma/LFM）不同的输出。

Gemma 默认使用 INT8。原生 FP8 为实验性功能，可通过 `run-api.bat --model-family gemma4 --kv-type fp8-mma` 显式选择。

**上下文容量自动选择**依据 GPU 总显存，如下表所示；可用显存决定专家缓存的大小。显式指定 `--max-context N` 时，以该值为准。

| GPU 总显存（示例） | Qwen3.6 / Gemma 4 | LFM2.5 |
| --- | ---: | ---: |
| 8 GB | 64K | 64K |
| 12 GB | 128K | 128K |
| 16 GB 及以上 | 256K | 128K |

容量档位容许驱动报告值的差异，因此 16 GB 显卡即使报告为 15.x GiB，Qwen 和 Gemma 仍会选择 256K。

额外模型的可执行文件缺失时，请参见[手动构建说明](README.md#model-focus)。

## 安装方法

在 64 位 PowerShell 中运行以下命令，即可安装 TinyStrata 并在 `http://127.0.0.1:18080/v1` 启动 API：

```powershell
irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install.ps1' | iex
```

重启 Windows 后，要再次启动，请运行以下命令，或双击同一个 `run-api.bat`：

```powershell
& "$env:LOCALAPPDATA\TinyStrata\run-api.bat" --port 18080
```

或者，你也可以将 [TinyStrata 的 GitHub 仓库](https://github.com/kbc20f11/TinyStrata)链接交给 AI 智能体，让它帮你完成安装。

### 无需重启即可切换模型

安装器默认启动 Qwen。要在同一个 API 服务中使用 Gemma，请先下载并查看进度：

```powershell
$api = 'http://127.0.0.1:18080'
Invoke-RestMethod "$api/v1/model/download" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/download?model=gemma-4-26b-a4b-it"
```

重复查询进度，直到 `state` 为 `ready`，再明确请求切换。下载未启用的模型只会完成准备，不会自动切换：

```powershell
Invoke-RestMethod "$api/v1/model/switch" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/switch"
```

通过 `run-api.bat --model-family gemma4` 可直接启动 Gemma，默认 API 地址为 `http://127.0.0.1:18082/v1`。使用 LFM 时，将上述下载、进度查询和切换请求中的 `gemma-4-26b-a4b-it` 替换为 `lfm2.5-24b-a2b`；也可通过 `run-api.bat --model-family lfm2` 直接启动，默认地址为 `http://127.0.0.1:18081/v1`。

轮询 `GET /v1/model/switch`，直到 `state` 为 `ready`。切回 Qwen 时，向相同切换接口发送 `{"model":"qwen3.6-35b-a3b"}`。下载或切换开始时返回 `202`；生成期间的切换请求返回 `409`，请在生成结束后重试。切换期间的推理请求返回 `503`。端口和身份验证保持不变。模型选择只在当前服务会话中有效，重启后使用启动器选择的模型。

## 实测速度 — Qwen3.6

于 **2026-10-21** 在 Windows 上测量，使用 RTX PRO 4000 Blackwell SFF Edition（24 GB）、Ryzen 9 7945HX3D、64 GB 系统内存和固定的 Unsloth MTP UD-IQ3_XXS 模型。图像编码器常驻 GPU，通过 API 输入文本；启用思考、模型默认采样设置（温度 0.7）及 MTP 3。输入分块和专家分配自动选择，上下文容量按各显存档位设置。混合 CUDA Graph 保持默认关闭，全部专家常驻 GPU 时的常规 CUDA Graph 保持启用。

| 可用内存条件 | 上下文容量 | INT8 生成 | 原生 FP8 生成 | 备注 |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 117.80 tok/s | 128.48 tok/s | 所有权重常驻显存 |
| 约 16 GB | 256K | 98.90 tok/s | 98.11 tok/s | GPU 与 CPU 共同计算专家 |
| 约 12 GB | 128K | 87.99 tok/s | 89.44 tok/s | GPU 与 CPU 共同计算专家 |
| 约 8 GB | 64K | 60.69 tok/s | 61.60 tok/s | GPU 与 CPU 共同计算专家 |

每个 FP8/INT8 配置均使用相同的三段文本各运行两次，采用模型默认采样设置，每次生成包含思考的 128 个 token。速度为总生成 token 数除以总解码时间，不含加载与输入处理；上下文值表示缓存容量。内存受限行是在同一块 24 GB GPU 上另行预留 8、12 或 16 GiB，并设置相应显存档位的上下文容量，并非使用实际的 16/12/8 GB GPU。

## 实测速度 — Gemma 4

在同一台 Windows PC（64 GB 系统内存）上使用 Gemma 4 26B-A4B-it QAT Q4_0 及匹配的 QAT MTP 助手测量。图像编码器常驻 GPU，通过 API 输入文本，启用 MTP 3、CUDA Graph 及模型自身的默认思考和采样设置；上下文容量按各显存档位设置，输入分块及专家分配自动选择。

原生 FP8 和 INT8 于 **2026-10-21** 测量。

| 可用显存 | 上下文容量 | INT8（默认） | 原生 FP8（实验性） | 备注 |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 113.20 tok/s | 108.14 tok/s | INT8 / FP8: GPU 计算专家 |
| 约 16 GB | 256K | 100.08 tok/s | 94.53 tok/s | INT8 / FP8: GPU 和 CPU 计算专家 |
| 约 12 GB | 128K | 81.66 tok/s | 85.34 tok/s | INT8 / FP8: GPU 和 CPU 计算专家 |
| 约 8 GB | 64K | 51.54 tok/s | 50.52 tok/s | INT8 / FP8: GPU 和 CPU 计算专家 |

每个 FP8/INT8 配置均使用相同的三段文本各运行两次，采用模型默认采样设置，每次生成包含思考的 128 个 token。速度为总生成 token 数除以总解码时间，不含加载与输入处理；上下文值表示缓存容量。内存受限行是在同一块 24 GB GPU 上另行预留 8、12 或 16 GiB，并设置相应显存档位的上下文容量，并非使用实际的 16/12/8 GB GPU。

## 实测速度 — LFM2.5

在同一台 Windows PC（64 GB 系统内存）上使用官方 LFM2.5-24B-A2B Q4_K_M 测量，不使用 MTP 或 Vision。启用 CUDA Graph 及模型自身的默认思考和采样设置；上下文容量按各显存档位设置，输入分块及专家分配自动选择。

原生 FP8 和 INT8 于 **2026-10-21** 测量。

| 可用显存 | 上下文容量 | INT8 | 原生 FP8（默认） | 备注 |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 128K | 140.58 tok/s | 141.80 tok/s | INT8 / FP8: GPU 计算专家 |
| 约 16 GB | 128K | 135.63 tok/s | 138.21 tok/s | INT8 / FP8: GPU 和 CPU 计算专家 |
| 约 12 GB | 128K | 110.11 tok/s | 114.72 tok/s | INT8 / FP8: GPU 和 CPU 计算专家 |
| 约 8 GB | 64K | 71.10 tok/s | 73.92 tok/s | INT8 / FP8: GPU 和 CPU 计算专家 |

每个 FP8/INT8 配置均使用相同的三段文本各运行两次，采用模型默认采样设置，每次生成包含思考的 128 个 token。速度为总生成 token 数除以总解码时间，不含加载与输入处理；上下文值表示缓存容量。内存受限行是在同一块 24 GB GPU 上另行预留 8、12 或 16 GiB，并设置相应显存档位的上下文容量，并非使用实际的 16/12/8 GB GPU。

## 我应该使用哪个项目？

| 你的目标和硬件 | 项目 |
| --- | --- |
| 你希望在 **Windows** 上使用仅通过 API 提供的 Qwen3.6 / Gemma 4 / LFM2.5 推理，设备是配备主流显卡的 PC，配置为 **8–24 GB 显存、系统内存不超过 32 GB**。 | TinyStrata（**仅支持 Windows**，开发中） |
| 你同时拥有 **RTX 3060 级别或更高性能的 GPU** 和 **64 GB 或更多系统内存**。 | 推荐使用 [Strata（上游项目）](https://github.com/Niko1221/Strata)。 |
| 你希望使用**附带聊天 UI** 的项目。 | 推荐使用 [Strata（上游项目）](https://github.com/Niko1221/Strata)。 |

## 内存如何分工 — Qwen3.6

MoE 模型包含许多**专家（Experts）**，它们是针对每个 token 选择的小型神经网络。TinyStrata 会在 GPU 内存（显存）中保留尽可能多的专家。在生成过程中，选中但未缓存的专家可以使用系统内存在 CPU 上计算。处理较大的输入批次时，也可以改为将专家权重从系统内存流式传输到临时 GPU 缓冲区。模型选中的专家都会保留在计算中。

下载的模型文件总计约 **15.0 GB**：语言模型（包括 MTP 权重）为 **14.1 GB**，视觉投影器（`mmproj-F16.gguf`）为 **0.9 GB**。语言模型无需全部装入显存。GPU 内存还需要为视觉编码器和图像处理缓冲区、对话的 **KV 缓存**、模型的其他权重以及临时计算缓冲区预留空间：

![TinyStrata 如何在 GPU 显存与系统内存之间分配内存](assets/memory-sharing.png)

这是默认 GPU 图像模式下的布局示意图，并不表示各部分大小固定。每次加载模型时，TinyStrata 都会先启动并预热 GPU 视觉编码器，再加载语言模型；编码器的权重和预热后的工作缓冲区保持常驻。随后分配 KV 的初始页面、其他权重和工作缓冲区，并根据剩余空闲显存确定专家缓存容量。在 Windows 上，若能同时取得 CUDA 的空闲内存和 NVIDIA 驱动的空闲专用显存读数，则使用较小值。

缓存内容随使用的专家而变化。默认的 KV 自动扩展使短对话能为专家保留更多显存；序列变长时，专家缓存向 KV 归还内存。新的较短请求可以把不再使用的 KV 页面交还专家缓存。`run-api.bat --kv-whole` 改为预先分配全部 KV 容量，`run-api.bat --kv-grow` 恢复自动扩展。其他应用仍可能占用显存。

专家放置通常以保留 768 MiB 余量为目标。在 Qwen 的 `--kv-whole` 模式下，与 Strata 一样，当主模型的所有专家只差少量空间即可装入显存时，可以释放用于加速的 Q8 权重副本、缩小已有的 Attention 分数缓冲区，并将余量目标降至 256 MiB；MTP 专家可以在 CPU 上运行。`--vram-reserve-mib N` 可显式固定预留量。这是放置时的目标，不保证此后始终保有相同的空闲内存。

通常，常驻 GPU 的专家越多，CPU 的计算量就越少。在混合模式下，系统内存也会保留已缓存在显存中的专家的后备副本，因此增加显存并不会逐字节等量减少系统内存占用。也请为操作系统和其他应用留出内存。更大的**系统内存容量**可以容纳模型数据，但速度还取决于 CPU、内存带宽和 GPU。Windows 的“共享 GPU 内存”与 TinyStrata 主动在 CPU 上计算专家的机制不同。

KV 缓存始终保留在显存中；在这一卸载机制中，系统内存用于存放专家数据。因此，长对话仍然需要足够的 GPU 内存来容纳 KV 缓存。

## 自动设置 — Qwen3.6

新配置会自动选择 KV/Attention 和上下文容量，无需选择适用于 8 GB、12 GB 或 16 GB 的配置方案。

Qwen3.6 在 RTX 3060 上会选择 INT8。受支持的 RTX 40/50 系列环境可以选择原生 FP8；引擎会检查已编译的内核和 GPU 功能，以及硬件代际。如果原生 FP8 不可用，自动模式会使用 INT8。这一过程会选择受支持的路径，而无需在启动时运行速度或质量基准测试。

原生引擎在启动时，根据空闲显存、上下文容量以及专家能否装入 GPU，从 512、1024、2048 或 4096 token 中估算并选择输入分块大小。较大的分块可以减少输入处理期间重复从 RAM 向 GPU 传输权重的次数。`run-api.bat --chunk N` 可固定大小；`run-api.bat --chunk auto` 可恢复原生自动选择。

上下文容量是输入与生成输出合计的逻辑上限。默认按对话长度逐步分配 KV 物理内存，而不是为短对话预先分配全部容量。长文档仍需要更多计算，最终也需要更多 KV 内存。

达到完整容量时，Qwen 的 INT8 KV（包含 MTP 层）约需要 **64K 时 0.71 GiB、128K 时 1.42 GiB、256K 时 2.84 GiB**，其他权重和工作缓冲区还需额外空间。在此实现中，原生 FP8 的 KV 存储占用与 INT8 大致相同；其优势来自不同的计算路径，而非显著缩小缓存。

上下文容量按 GPU 总显存选择，专家缓存随后按加载时剩余的可用显存分配。其他应用使用显存越多，GPU 能缓存的专家就越少。输入超出上下文容量时，API 会拒绝请求，不会静默截断。

GGUF 权重量化与 KV 精度是两回事。选用的 UD-IQ3_XXS 模型使用压缩权重；INT8 或 FP8 描述的是对话缓存及其 Attention 路径。较低精度的 KV 可能改变回答，因此这些模式不保证与 FP32 KV 输出完全一致。该实现会保留所有可见的上下文位置，以及模型选中的每个专家。

新的启动器配置使用 MTP 3，它会提前提议若干 token，并由主模型验证。它可能提升生成速度，但收益取决于有多少提议被接受。验证旨在保留主模型的解码行为；不同的数值执行路径仍可能改变 token，因此启用 MTP 并不保证与禁用时输出完全一致。

MTP 同时支持贪心生成和基于温度的采样，包括重复惩罚、频率惩罚和存在惩罚。对于采样请求，主模型会对每个验证行进行采样，只接受与之匹配的草稿前缀。

显式设置会覆盖自动选择，现有的手动设置也会继续生效。例如，`run-api.bat --kv-type int8` 会固定使用 INT8；`run-api.bat --kv-type auto --max-context auto --expert-vram auto` 会恢复自动选择 KV、上下文和专家放置。

## API 服务器

**TinyStrata 仅提供 API。** 它提供兼容 OpenAI 的 Chat Completions（`/v1/chat/completions`）和 Responses（`/v1/responses`）API，以及兼容 Anthropic 的 Messages API（`/v1/messages`），并支持流式生成。请连接 API 客户端或外部前端；TinyStrata 不附带聊天 UI。

新配置采用各模型自己的采样默认值。每次请求指定的值优先；已保存的自定义设置保持不变。

| 设置 | Qwen3.6 | Gemma 4 | LFM2.5 |
| --- | ---: | ---: | ---: |
| `temperature` | 0.7 | 1.0 | 0.1 |
| `top_p` | 0.95 | 0.95 | 1.0 |
| `top_k` | 20 | 64 | 50 |
| `min_p` | 0.0 | 0.0 | 0.0 |
| `presence_penalty` | 1.5 | 0.0 | 0.0 |
| `repetition_penalty` | 1.0 | 1.0 | 1.05 |
| `enable_thinking` | true | true | true（始终启用） |
| `max_tokens`（包含思考） | 32768 | 32768 | 32768 |

API 默认监听 `http://127.0.0.1:18080`。要使用其他端口，请在已安装的 TinyStrata 文件夹中运行 `run-api.bat --port 18081`。

对于 Qwen，`run-api.bat --vision gpu` 使用 GPU 处理图像（新配置的默认值）；`--vision cpu` 使用 CPU，`--vision no` 禁用图像。`--vision-tokens N` 设置图像 token 上限（默认 4096）。这些选择会保存，后续启动时省略参数也会保留。

模型文件尚未安装时，API 也可以启动。选择 Qwen 时，调用 `POST /v1/model/download` 并使用空 JSON 请求体（`{}`），会下载固定的 [Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS 语言模型](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf)（14.1 GB，包含 MTP 权重）及其[视觉投影器](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf)（0.9 GB），总计约 **15.0 GB**。它会验证两个文件的 SHA-256 校验和，提取分词器并加载模型文件。`GET /v1/model/download` 可报告进度；中断的下载可以续传。重复的下载请求会复用当前任务，`POST /v1/model/download/cancel` 可以取消传输或校验。下载完成并不意味着推理已就绪：请等待下载状态变为 `ready`，且 `/health` 报告 `loaded: true`（启用图像时还需 `images: true`）。

安装程序会准备引擎可执行文件和 GPU 运行时库，并默认下载 Qwen 及其视觉投影器。现有配置会继续使用已选择的本地模型，直到你明确更改它。

## 致谢

TinyStrata 是 [Strata](https://github.com/Niko1221/Strata) 的独立分支，大量借鉴了其实现。感谢 [Niko](https://github.com/Niko1221) 和 Strata 社区提供引擎，以及让 TinyStrata 得以实现的工作。

同时感谢 [Aaron Flin](https://github.com/aflin) 和 [MoMoE 项目](https://github.com/aflin/momoe)，TinyStrata 所适配的 KV 缓存、attention 和量化专家代码来自他们的工作。

## 许可证

MIT。参见 [LICENSE](LICENSE)。

Copyright (c) 2026 Niko1221 and the Strata contributors.

上游署名信息：[NOTICE](NOTICE)。GGML：[MIT 许可证](third_party/ggml/LICENSE)。
