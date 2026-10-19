# TinyStrata

[English](README.md) · **简体中文** · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

**TinyStrata 是一款正在开发中的高速推理引擎，面向 30B 级 MoE 模型，仅支持 Windows，且仅提供 API。** 主要面向配备主流显卡、显存为 **8–24 GB** 且系统内存**不超过 32 GB** 的 PC。

## 为什么开发 TinyStrata？

随着 PC 配件价格上涨，组装一台高端 PC 对许多人来说正变得越来越难。

因此，我们开发了 TinyStrata，让 AI 能在配备 RTX 3060 级别或更高性能主流显卡的 PC 上，以本地方式高速运行。我们专注于让 30B 级的较小型 MoE 模型尽可能快地运行。

我们的主要目标是配备 **8–24 GB 显存、系统内存不超过 32 GB** 的 PC。我们认为，这一范围涵盖了配备主流显卡的 PC 中最常见的配置。

TinyStrata 基于 Strata，旨在提升显存有限环境下 Qwen3.6-35B-A3B 的推理速度。我们希望高效利用 GPU 内存，让你能在自己的 PC 上处理长对话和文档。

TinyStrata 将开发精力集中在 Windows 上的高速本地推理及其 API 服务上。这一方针指导着我们的实现和测试。

我们也希望尽可能简化上手过程。TinyStrata 会根据你的硬件自动选择推理和内存设置，因此无需复杂配置或手动调优即可使用 API。

## 为什么不附带 UI？

目前已有许多支持 LLM API 的 UI 和智能体运行框架。TinyStrata 也支持常见的 API 格式，因此可以连接你喜欢的软件来使用。如果连接配置让你觉得繁琐，也可以请 AI 智能体帮你完成。将 UI 交给现有软件，让我们能够专注于推理引擎和 API 的开发。

## 重点支持的模型

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 仍是默认模型，适合在配备主流显卡的 PC 上兼顾速度、内存效率和回答质量。TinyStrata 专注于较小的 MoE 模型，并新增了对 **LFM2.5-24B-A2B 的实验性支持**。

LFM 使用官方 [LiquidAI Q4_K_M GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF)（**14.53 GB**）、**FP16 KV**，最大上下文为 **128K**，不支持 MTP 或 Vision。它利用自动 VRAM 专家缓存、CPU 专家计算、输入阶段权重流式传输和 CUDA Graph，不省略模型选中的专家。默认值为 `temperature=0.1`、`top_k=50`、`repetition_penalty=1.05`，思考始终启用。Qwen 与 LFM 的设置分别保存。LFM 引擎仍在验证中，不保证与 llama.cpp 或不同 CPU/GPU 分配下的输出 token 完全一致。

LFM 可执行文件缺失时，请参见[手动构建说明](README.md#model-focus)。

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

安装器默认启动 Qwen。要在同一个 API 服务中使用 LFM，请先下载并查看进度：

```powershell
$api = 'http://127.0.0.1:18080'
Invoke-RestMethod "$api/v1/model/download" -Method Post -ContentType 'application/json' -Body '{"model":"lfm2.5-24b-a2b"}'
Invoke-RestMethod "$api/v1/model/download?model=lfm2.5-24b-a2b"
```

重复查询进度，直到 `state` 为 `ready`，再明确请求切换。下载未启用的模型只会完成准备，不会自动切换：

```powershell
Invoke-RestMethod "$api/v1/model/switch" -Method Post -ContentType 'application/json' -Body '{"model":"lfm2.5-24b-a2b"}'
Invoke-RestMethod "$api/v1/model/switch"
```

轮询 `GET /v1/model/switch`，直到 `state` 为 `ready`。切回 Qwen 时，向相同切换接口发送 `{"model":"qwen3.6-35b-a3b"}`。下载或切换开始时返回 `202`；生成期间的切换请求返回 `409`，请在生成结束后重试。切换期间的推理请求返回 `503`。端口和身份验证保持不变。模型选择只在当前服务会话中有效，重启后使用启动器选择的模型。

## 实测速度 — Qwen3.6

以下测量于 2026-10-19 在 Windows 上完成，使用 RTX PRO 4000 Blackwell SFF Edition（24 GB）、Ryzen 9 7945HX3D，以及固定的 Unsloth MTP UD-IQ3_XXS 模型。通过正常启动的 API 测量纯文本生成速度，视觉编码器始终常驻 GPU。

采用默认的思考和采样设置（温度 0.7）、MTP 3、512 token 的提示词分块，并自动选择上下文容量和专家放置。混合 CUDA Graph 保持默认关闭。FP8 列使用本 GPU 自动选择的 KV 路径；INT8 列仅将 KV 类型改为 INT8，以便比较。

**我们计划今后在搭载 8 GB、12 GB、16 GB 显存 GPU 的实机上进行测量。** 下表中减少可用内存的行，是在同一块 24 GB GPU 上限制空闲容量后测得的结果，并非物理 8/12/16 GB GPU 的基准测试。

| 可用内存条件 | 上下文容量 | INT8 生成 | 原生 FP8 生成 | 备注 |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 118.40 tok/s | 131.82 tok/s | 所有权重常驻显存 |
| 约 16 GB | 128K | 95.09 tok/s | 99.39 tok/s | GPU 与 CPU 共同计算专家 |
| 约 12 GB | 64K | 85.03 tok/s | 85.59 tok/s | GPU 与 CPU 共同计算专家 |
| 约 8 GB | 64K | 66.30 tok/s | 63.85 tok/s | GPU 与 CPU 共同计算专家 |

## 我应该使用哪个项目？

| 你的目标和硬件 | 项目 |
| --- | --- |
| 你希望在 **Windows** 上使用仅通过 API 提供的 Qwen3.6-35B-A3B 推理，设备是配备主流显卡的 PC，配置为 **8–24 GB 显存、系统内存不超过 32 GB**。 | TinyStrata（**仅支持 Windows**，开发中） |
| 你同时拥有 **RTX 3060 级别或更高性能的 GPU** 和 **64 GB 或更多系统内存**。 | 推荐使用 [Strata（上游项目）](https://github.com/Niko1221/Strata)。 |
| 你希望使用**附带聊天 UI** 的项目。 | 推荐使用 [Strata（上游项目）](https://github.com/Niko1221/Strata)。 |

## 内存如何分工 — Qwen3.6

MoE 模型包含许多**专家（Experts）**，它们是针对每个 token 选择的小型神经网络。TinyStrata 会在 GPU 内存（显存）中保留尽可能多的专家。在生成过程中，选中但未缓存的专家可以使用系统内存在 CPU 上计算。处理较大的输入批次时，也可以改为将专家权重从系统内存流式传输到临时 GPU 缓冲区。模型选中的专家都会保留在计算中。

下载的模型文件总计约 **15.0 GB**：语言模型（包括 MTP 权重）为 **14.1 GB**，视觉投影器（`mmproj-F16.gguf`）为 **0.9 GB**。语言模型无需全部装入显存。GPU 内存还需要为视觉编码器和图像处理缓冲区、对话的 **KV 缓存**、模型的其他权重以及临时计算缓冲区预留空间：

![TinyStrata 如何在 GPU 显存与系统内存之间分配内存](assets/memory-sharing.png)

这是默认 GPU 图像模式下的布局示意图，并不表示各部分大小固定。每次加载模型时，TinyStrata 都会先启动并预热 GPU 视觉编码器，再加载语言模型；编码器的权重和预热后的工作缓冲区保持常驻。随后分配 KV 缓存、其他权重和工作缓冲区，并根据剩余空闲显存确定专家缓存容量。在 Windows 上，若能同时取得 CUDA 的空闲内存和 NVIDIA 驱动的空闲专用显存读数，则使用较小值。

随着使用的专家发生变化，缓存内容也会改变；缓存容量则会在重新加载模型时重新计算。之后打开的其他应用仍可能占用显存。

专家放置通常以保留 768 MiB 余量为目标。与 Strata 一样，当主模型的所有专家只差少量空间即可装入显存时，可以释放用于加速的 Q8 权重副本、缩小已有的 Attention 分数缓冲区，并将余量目标降至 256 MiB；MTP 专家可以在 CPU 上运行。`--vram-reserve-mib N` 可显式固定预留量。这是放置时的目标，不保证此后始终保有相同的空闲内存。

通常，常驻 GPU 的专家越多，CPU 的计算量就越少。在混合模式下，系统内存也会保留已缓存在显存中的专家的后备副本，因此增加显存并不会逐字节等量减少系统内存占用。也请为操作系统和其他应用留出内存。更大的**系统内存容量**可以容纳模型数据，但速度还取决于 CPU、内存带宽和 GPU。Windows 的“共享 GPU 内存”与 TinyStrata 主动在 CPU 上计算专家的机制不同。

KV 缓存始终保留在显存中；在这一卸载机制中，系统内存用于存放专家数据。因此，长对话仍然需要足够的 GPU 内存来容纳 KV 缓存。

## 自动设置 — Qwen3.6

新配置会自动选择 KV/Attention 和上下文容量，无需选择适用于 8 GB、12 GB 或 16 GB 的配置方案。

RTX 3060 会选择 INT8。受支持的 RTX 40/50 系列环境可以选择原生 FP8；引擎会检查已编译的内核和 GPU 功能，以及硬件代际。如果原生 FP8 不可用，自动模式会使用 INT8。这一过程会选择受支持的路径，而无需在启动时运行速度或质量基准测试。**原生 FP8 Attention 仍属于实验性功能。**

原生引擎在启动时，根据空闲显存、上下文容量以及专家能否装入 GPU，从 512、1024、2048 或 4096 token 中估算并选择输入分块大小。较大的分块可以减少输入处理期间重复从 RAM 向 GPU 传输权重的次数。`run-api.bat --chunk N` 可固定大小；`run-api.bat --chunk auto` 可恢复原生自动选择。

上下文容量是为输入和生成输出合计预留的最大空间。即使对话很短，更大的容量也会占用更多 KV 内存，从而减少可供专家使用的显存。容量也不意味着不同对话长度下的速度相同：实际处理长文档需要更多计算。作为规模参考，仅 INT8 KV（包含 MTP 层）就需要约 **64K 时 0.71 GiB、128K 时 1.42 GiB、256K 时 2.84 GiB**，其他权重和工作缓冲区还需额外空间。在此实现中，原生 FP8 的 KV 存储占用与 INT8 大致相同；其优势来自不同的计算路径，而非显著缩小缓存。

空闲显存指加载时可用的内存，而不是 GPU 包装盒上标注的容量。在 Windows 上，自动选择上下文时会同时查询 CUDA 和 NVIDIA 驱动报告的整个设备的空闲内存；两次查询均成功时使用较小值。因此，其他应用可能导致系统选择较小的上下文。如果重新加载后容量减小，API 会拒绝超出新容量的输入，而不会静默截断。

GGUF 权重量化与 KV 精度是两回事。选用的 UD-IQ3_XXS 模型使用压缩权重；INT8 或 FP8 描述的是对话缓存及其 Attention 路径。较低精度的 KV 可能改变回答，因此这些模式不保证与 FP32 KV 输出完全一致。该实现会保留所有可见的上下文位置，以及模型选中的每个专家。

新的启动器配置使用 MTP 3，它会提前提议若干 token，并由主模型验证。它可能提升生成速度，但收益取决于有多少提议被接受。验证旨在保留主模型的解码行为；不同的数值执行路径仍可能改变 token，因此启用 MTP 并不保证与禁用时输出完全一致。

MTP 同时支持贪心生成和基于温度的采样，包括重复惩罚、频率惩罚和存在惩罚。对于采样请求，主模型会对每个验证行进行采样，只接受与之匹配的草稿前缀。

显式设置会覆盖自动选择，现有的手动设置也会继续生效。例如，`run-api.bat --kv-type int8` 会固定使用 INT8；`run-api.bat --kv-type auto --max-context auto --expert-vram auto` 会恢复自动选择 KV、上下文和专家放置。

## API 服务器

**TinyStrata 仅提供 API。** 它提供兼容 OpenAI 的 Chat Completions（`/v1/chat/completions`）和 Responses（`/v1/responses`）API，以及兼容 Anthropic 的 Messages API（`/v1/messages`），并支持流式生成。请连接 API 客户端或外部前端；TinyStrata 不附带聊天 UI。

Qwen 默认启用思考，设置为 `temperature=0.7`、`top_p=0.95`、`top_k=20`、`min_p=0.0`、`presence_penalty=1.5`、`repetition_penalty=1.0`，包含思考在内最多输出 **32,768 个 token**。每个请求中的设置优先。

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
