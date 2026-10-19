# TinyStrata

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

**TinyStrata is a fast, Windows-only, API-only inference engine for 30B-class MoE models.** It primarily targets PCs with mainstream graphics cards (**8–24 GB of VRAM**) and **32 GB of system RAM or less**.

## Why we are building TinyStrata

With PC component prices rising, putting together a high-end PC is increasingly out of reach for many people.

To address this, TinyStrata was created to run AI locally at high speed on PCs with an RTX 3060-class graphics card or better. Our focus is on making relatively small, 30B-class MoE models run as fast as possible.

Our main target is PCs with **8–24 GB of VRAM and 32 GB of system RAM or less**. We believe this range represents the most common configurations for PCs equipped with graphics cards.

Built on Strata, TinyStrata aims to improve inference speed for Qwen3.6-35B-A3B in environments with limited VRAM. We want to use GPU memory efficiently so you can work with long conversations and documents on your own PC.

TinyStrata concentrates development on running fast local inference on Windows and providing it through an API. This focus guides our implementation and testing.

We also aim to make getting started as simple as possible. TinyStrata automatically chooses inference and memory settings for your hardware, so you can use the API without complicated configuration or manual tuning.

## Why no bundled UI?

Many API-compatible UIs and agent harnesses for LLMs already exist. TinyStrata supports common API formats, so you can connect whichever you prefer. If configuring the connection feels tedious, you can ask an AI agent to set it up for you. Leaving the UI to existing software lets us focus development on the inference engine and API.

## Model focus

TinyStrata currently focuses exclusively on [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). We will consider supporting another model when a MoE model offers a better fit than Qwen3.6-35B-A3B for fast local inference on PCs with mainstream graphics cards.

We see too few options for 30B-class MoE models that work well on PCs with mainstream graphics cards, even as larger models continue to be developed and released. [Qwen3.6-35B-A3B, released on April 16, 2026](https://github.com/QwenLM/Qwen3.8#news), still suits our needs for a balance of speed, memory efficiency and answer quality. We have yet to find a smaller MoE model that clearly improves on this balance in our target environments.

## Installation

Run in 64-bit PowerShell to install TinyStrata and start the API at `http://127.0.0.1:18080/v1`:

```powershell
irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install.ps1' | iex
```

To start again after restarting Windows, run this command or double-click the same `run-api.bat`:

```powershell
& "$env:LOCALAPPDATA\TinyStrata\run-api.bat" --port 18080
```

Alternatively, give your AI agent the link to the [TinyStrata GitHub repository](https://github.com/kbc20f11/TinyStrata) and ask it to handle the installation.

## Measured speed

Measurements were taken on 2026-10-19 using Windows, an RTX PRO 4000 Blackwell SFF Edition (24 GB), a Ryzen 9 7945HX3D and the fixed Unsloth MTP UD-IQ3_XXS model. The GPU vision encoder was resident throughout these text-only API requests.

Default thinking and sampling settings (temperature 0.7), MTP 3, 512-token prompt chunks and automatic context and Expert placement were used. The hybrid CUDA Graph option stayed at its default off. The FP8 column uses automatic KV selection on this GPU; the INT8 column changes only the KV mode.

**We plan to benchmark physical GPUs with 8 GB, 12 GB and 16 GB of VRAM.** The reduced-memory rows below were measured by limiting free memory on the same 24 GB GPU; they are not benchmarks of physical 8/12/16 GB GPUs.

| Available VRAM | Context capacity | INT8 generation | Native FP8 generation | Notes |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 118.40 tok/s | 131.82 tok/s | All weights resident in VRAM |
| Approximately 16 GB | 128K | 95.09 tok/s | 99.39 tok/s | GPU and CPU Expert computation |
| Approximately 12 GB | 64K | 85.03 tok/s | 85.59 tok/s | GPU and CPU Expert computation |
| Approximately 8 GB | 64K | 66.30 tok/s | 63.85 tok/s | GPU and CPU Expert computation |

## Which project should I use?

| Your goal and hardware                                                                                                                       | Project                                                               |
| -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| You want API-only Qwen3.6-35B-A3B inference on a **Windows** PC with a mainstream graphics card, **8–24 GB of VRAM and 32 GB of system RAM or less**. | TinyStrata (**Windows only**, under development)                      |
| You have both **an RTX 3060-class GPU or better** and **64 GB of system RAM or more**.                                                        | We recommend [Strata (upstream)](https://github.com/Niko1221/Strata). |
| You want a **bundled chat UI**.                                                                                                              | We recommend [Strata (upstream)](https://github.com/Niko1221/Strata). |

## How memory is shared

A MoE model contains many **Experts**: smaller neural networks selected for each token. TinyStrata keeps as many Experts as it can in GPU memory (VRAM). During generation, it can compute selected uncached Experts on the CPU using system RAM. When processing larger input batches, it can instead stream Expert weights from RAM through temporary GPU buffers. Model-selected Experts are kept in the computation.

The downloaded model files total about **15.0 GB**: **14.1 GB** for the language model (including MTP weights) and **0.9 GB** for the vision projector (`mmproj-F16.gguf`). The language model does not all have to fit in VRAM. GPU memory also needs room for the vision encoder and image-processing buffers, the conversation's **KV cache**, the model's other weights and temporary computation buffers:

![How TinyStrata shares memory between GPU VRAM and system RAM](assets/memory-sharing.png)

This is a layout diagram, not a fixed-size partition. At each model load, TinyStrata starts and warms up the GPU vision encoder before loading the language model; its weights and warmed work buffers stay resident. It then allocates the KV cache, other weights and work buffers, and sizes the Expert cache from the remaining free VRAM. On Windows, it uses the smaller of CUDA's free-memory reading and the NVIDIA driver's free dedicated VRAM when both are available.

Cache contents change as different Experts are used; cache capacity is recalculated when the model reloads. Other applications opening later can still consume VRAM.

Expert placement normally targets 768 MiB of headroom. Following Strata, near-fit placement can release packed Q8 weight copies, shrink an Attention score buffer where present and reduce the target to 256 MiB so all main-model Experts fit in VRAM, while MTP Experts can run on the CPU. `--vram-reserve-mib N` fixes the reserve explicitly. This is a placement target, not a permanent guarantee of free memory.

More GPU-resident Experts usually means less CPU work. In hybrid mode, system RAM also retains backing copies of Experts that are cached in VRAM, so adding VRAM does not reduce RAM usage byte for byte. Leave RAM for the operating system and other applications too. More **RAM capacity** lets you hold model data, but speed also depends on the CPU, RAM bandwidth and GPU. Windows' "shared GPU memory" is different from TinyStrata deliberately computing Experts on the CPU.

The KV cache stays in VRAM; system RAM holds Expert data for this offload mechanism. A long conversation therefore still needs enough GPU memory for its KV cache.

## Automatic settings

New configurations choose KV/Attention and context capacity automatically. You do not need to select an 8 GB, 12 GB or 16 GB recipe.

An RTX 3060 selects INT8. Supported RTX 40/50-series environments can select native FP8; the engine checks the compiled kernels and GPU capabilities as well as the hardware generation. If native FP8 is unavailable, automatic mode uses INT8. This chooses a supported path without running a speed or quality benchmark at startup. **Native FP8 Attention remains experimental.**

The native engine estimates the prompt chunk size at startup, choosing 512, 1024, 2048 or 4096 tokens from available VRAM, context capacity and whether the Experts fit on the GPU. Larger chunks can reduce repeated RAM-to-GPU weight transfers during input processing. `run-api.bat --chunk N` fixes the size; `run-api.bat --chunk auto` restores native automatic selection.

Context capacity is the maximum space reserved for the input and generated output together. A longer capacity uses more KV memory even with a short conversation, leaving less VRAM for Experts. Capacity also does not imply equal speed at every conversation length: actually processing a long document takes more work. For scale, INT8 KV alone takes about **0.71 GiB at 64K, 1.42 GiB at 128K and 2.84 GiB at 256K**, including the MTP layer; other weights and work buffers are additional. Native FP8 uses approximately the same KV storage as INT8 in this implementation; its benefit is a different computation path, rather than a substantially smaller cache.

Free VRAM means memory available at load time, not the number printed on the GPU box. Other applications can therefore cause a smaller context to be selected. If a reload reduces capacity, the API rejects an input that no longer fits instead of silently cutting it short. On Windows, automatic context selection checks both CUDA and the NVIDIA driver's device-wide free memory and uses the smaller value when both queries succeed.

GGUF weight quantization and KV precision are separate. The selected UD-IQ3_XXS model has compressed weights; INT8 or FP8 describes the conversation cache and its Attention path. Lower-precision KV can change answers, so these modes do not promise identical output to FP32 KV. The implementation keeps all visible context positions and every model-selected Expert.

New launcher configurations use MTP 3, which proposes several tokens ahead and verifies them with the main model. It can improve generation speed, but the gain depends on how many proposals are accepted. Verification is intended to preserve the main model's decoding behavior; differences in numerical execution paths can still change tokens, so enabling MTP does not promise identical output to disabling it.

MTP supports both greedy generation and temperature-based sampling, including repetition, frequency and presence penalties. For sampled requests, the main model samples each verification row, and only the matching draft prefix is accepted.

Explicit settings override automatic choices, and existing manually selected settings remain in effect. For example, `run-api.bat --kv-type int8` fixes INT8; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` restores automatic KV, context and Expert placement.

## API server

**TinyStrata is API-only.** It provides OpenAI-compatible Chat Completions (`/v1/chat/completions`) and Responses (`/v1/responses`) APIs and an Anthropic-compatible Messages API (`/v1/messages`), including streamed generation. Connect an API client or an external frontend; TinyStrata does not include a bundled chat UI.

By default, thinking is enabled, with `temperature=0.7`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`, and a maximum of **32,768 output tokens** including thinking. Per-request settings take precedence.

The API listens on `http://127.0.0.1:18080` by default. To use another port, run `run-api.bat --port 18081` from the installed TinyStrata folder.

The API can start before the model files are installed. `POST /v1/model/download` with an empty JSON body (`{}`) downloads the fixed [Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS language model](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf) (14.1 GB, including MTP weights) and its [vision projector](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf) (0.9 GB), about **15.0 GB in total**. It verifies both files' SHA-256 checksums, extracts the tokenizer and loads them. `GET /v1/model/download` reports progress; interrupted downloads can resume. A repeated download request reuses the active job, and `POST /v1/model/download/cancel` can cancel transfer or verification. Download completion is not inference readiness: wait for the download state to become `ready` and `/health` to report `loaded: true` and `images: true`.

The installer prepares the engine and GPU runtime libraries and downloads both model files automatically. Existing configurations keep their selected local model until you explicitly change it.

## Acknowledgments

TinyStrata is an independent fork of [Strata](https://github.com/Niko1221/Strata) and draws heavily on its implementation. Thanks to [Niko](https://github.com/Niko1221) and the Strata community for the engine and the work that made TinyStrata possible.

Thanks also to [Aaron Flin](https://github.com/aflin) and the [MoMoE project](https://github.com/aflin/momoe) for the KV cache, attention and quantized Expert code adapted in TinyStrata.

## License

MIT. See [LICENSE](LICENSE).

Copyright (c) 2026 Niko1221 and the Strata contributors.

Upstream attribution: [NOTICE](NOTICE). GGML: [MIT license](third_party/ggml/LICENSE).
