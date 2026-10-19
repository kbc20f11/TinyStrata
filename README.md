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

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) remains the default model, with a balance of speed, memory efficiency and answer quality for PCs with mainstream graphics cards. TinyStrata focuses on compact MoE models and also supports **Gemma 4 26B-A4B-it and LFM2.5-24B-A2B**.

**Gemma 4 26B-A4B-it QAT Q4_0** uses the [Google-official target and vision projector](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) with a [matching Unsloth QAT Q4_0 MTP assistant](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/mtp-gemma-4-26B-A4B-it.gguf). The three downloads total **15.886 GB**: target 14.439 GB, projector 1.195 GB and assistant 0.252 GB. Its engine supports a GPU-resident image encoder, an adaptive GPU Expert cache with CPU computation from system RAM, and MTP 3 through the same API. Selected Experts are not omitted. Gemma supports **FP16 and INT8 KV**; its current automatic choice is **FP16**. Image input, MTP 3, RAM-assisted inference and live model switching have passed real-model tests.

LFM uses the official [LiquidAI Q4_K_M GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) (**14.53 GB**), **FP16 KV** and up to **128K context**, without MTP or Vision. It uses an automatic VRAM Expert cache, CPU Expert computation, prompt weight streaming and CUDA Graphs; selected Experts are not omitted. Its defaults are `temperature=0.1`, `top_k=50` and `repetition_penalty=1.05`, with reasoning always enabled. Qwen, Gemma and LFM settings are kept separately. This engine is still under validation; identical tokens to llama.cpp or across CPU/GPU placements are not guaranteed.

<details>
<summary>Build additional model engines manually if an executable is missing</summary>

From this checkout, use an x64 Visual Studio developer terminal with CUDA installed:

```powershell
cmake -S . -B ..\TinyStrata-work\build -DTINYSTRATA_BACKEND=CUDA
cmake --build ..\TinyStrata-work\build --config Release --target tinystrata-gemma4 tinystrata-lfm2 --parallel 4
```

Start Gemma with `run-api.bat --model-family gemma4` (default port 18082), or LFM with `run-api.bat --model-family lfm2` (default port 18081).

</details>

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

### Switch models without restarting

The installer starts with Qwen by default. To prepare Gemma on the same running API server, download it and check its progress:

```powershell
$api = 'http://127.0.0.1:18080'
Invoke-RestMethod "$api/v1/model/download" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/download?model=gemma-4-26b-a4b-it"
```

Repeat the progress request until `state` is `ready`, then switch explicitly. Downloading an inactive model does not activate it:

```powershell
Invoke-RestMethod "$api/v1/model/switch" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/switch"
```

To start directly with Gemma, run `run-api.bat --model-family gemma4`; its default API address is `http://127.0.0.1:18082/v1`. For LFM, replace `gemma-4-26b-a4b-it` in the download, progress and switch requests above with `lfm2.5-24b-a2b`, or start it with `run-api.bat --model-family lfm2` (default API address `http://127.0.0.1:18081/v1`).

Poll `GET /v1/model/switch` until `state` is `ready`. To return to Qwen, send the same switch request with `{"model":"qwen3.6-35b-a3b"}`. Starting a download or switch returns `202`; a switch during generation returns `409`, so retry after generation finishes. Inference during switching returns `503`. The port and authentication stay unchanged. Selection lasts for the current server session; restarting uses the model selected by the launcher.

## Measured speed — Qwen3.6

Measurements were taken on 2026-10-19 using Windows, an RTX PRO 4000 Blackwell SFF Edition (24 GB), a Ryzen 9 7945HX3D and the fixed Unsloth MTP UD-IQ3_XXS model. The GPU vision encoder was resident throughout these text-only API requests.

Default thinking and sampling settings (temperature 0.7), MTP 3, 512-token prompt chunks and automatic context and Expert placement were used. The hybrid CUDA Graph option stayed at its default off. The FP8 column uses automatic KV selection on this GPU; the INT8 column changes only the KV mode.

**We plan to benchmark physical GPUs with 8 GB, 12 GB and 16 GB of VRAM.** The reduced-memory rows below were measured by limiting free memory on the same 24 GB GPU; they are not benchmarks of physical 8/12/16 GB GPUs.

| Available VRAM | Context capacity | INT8 generation | Native FP8 generation | Notes |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 118.40 tok/s | 131.82 tok/s | All weights resident in VRAM |
| Approximately 16 GB | 128K | 95.09 tok/s | 99.39 tok/s | GPU and CPU Expert computation |
| Approximately 12 GB | 64K | 85.03 tok/s | 85.59 tok/s | GPU and CPU Expert computation |
| Approximately 8 GB | 64K | 66.30 tok/s | 63.85 tok/s | GPU and CPU Expert computation |

## Measured speed — Gemma 4

Measured on 2026-10-19 on the same Windows PC with 64 GB of system RAM, using Gemma 4 26B-A4B-it QAT Q4_0 and its matching QAT MTP assistant. The image encoder stayed on the GPU during text-only API requests. MTP 3, CUDA Graphs and default thinking/sampling (`temperature=1.0`, `top_p=0.95`, `top_k=64`) were enabled, with automatic context, 1,120-token prompt chunks and Expert placement.

| Available VRAM | Context capacity | FP16 generation | INT8 generation | Notes |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 105.56 tok/s | — | GPU and CPU Expert computation |
| Approximately 8 GB | 64K | 48.22 tok/s | 48.97 tok/s | Experts computed on the CPU during generation |

Each of the same three input texts was run once, generating 128 tokens including thinking per request. Speeds are total generated tokens divided by total decode time, excluding loading and prompt processing; 64K/256K are cache capacities, not input lengths. The 8 GB-equivalent row reserves 16 GiB separately on the 24 GB GPU; it is not a physical 8 GB GPU benchmark. The dash means unmeasured.

## Measured speed — LFM2.5 (experimental)

Measured on the same Windows PC with 64 GB of system RAM, using the official LFM2.5-24B-A2B Q4_K_M model, FP16 KV, no MTP or Vision, and CUDA Graphs enabled. Current API defaults were used: thinking enabled, `temperature=0.1`, `top_k=50` and `repetition_penalty=1.05`. Context capacity, prompt chunks (256 tokens in all cases) and Expert placement were selected automatically.

| Available VRAM | Context capacity | FP16 generation | Notes |
| --- | ---: | ---: | --- |
| 24 GB | 128K | 151.71 tok/s | All weights resident in VRAM |
| Approximately 16 GB | 128K | 111.71 tok/s | GPU and CPU Expert computation |
| Approximately 12 GB | 128K | 70.80 tok/s | GPU and CPU Expert computation |
| Approximately 8 GB | 64K | 51.20 tok/s | GPU and CPU Expert computation |

The same three input texts as the Qwen benchmark were run twice each, generating 128 tokens including thinking per request. Decode speed is total generated tokens divided by total decode time; loading and prompt processing are excluded. Reduced-memory rows limit available VRAM on the same 24 GB GPU and are not physical 8/12/16 GB GPU benchmarks. Model and sampling settings differ from the Qwen table.

## Which project should I use?

| Your goal and hardware                                                                                                                       | Project                                                               |
| -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| You want API-only Qwen3.6 / Gemma 4 / LFM2.5 inference on a **Windows** PC with a mainstream graphics card, **8–24 GB of VRAM and 32 GB of system RAM or less**. | TinyStrata (**Windows only**, under development)                      |
| You have both **an RTX 3060-class GPU or better** and **64 GB of system RAM or more**.                                                        | We recommend [Strata (upstream)](https://github.com/Niko1221/Strata). |
| You want a **bundled chat UI**.                                                                                                              | We recommend [Strata (upstream)](https://github.com/Niko1221/Strata). |

## How memory is shared — Qwen3.6

A MoE model contains many **Experts**: smaller neural networks selected for each token. TinyStrata keeps as many Experts as it can in GPU memory (VRAM). During generation, it can compute selected uncached Experts on the CPU using system RAM. When processing larger input batches, it can instead stream Expert weights from RAM through temporary GPU buffers. Model-selected Experts are kept in the computation.

The downloaded model files total about **15.0 GB**: **14.1 GB** for the language model (including MTP weights) and **0.9 GB** for the vision projector (`mmproj-F16.gguf`). The language model does not all have to fit in VRAM. GPU memory also needs room for the vision encoder and image-processing buffers, the conversation's **KV cache**, the model's other weights and temporary computation buffers:

![How TinyStrata shares memory between GPU VRAM and system RAM](assets/memory-sharing.png)

This is a layout diagram for the default GPU image mode, not a fixed-size partition. At each model load, TinyStrata starts and warms up the GPU vision encoder before loading the language model; its weights and warmed work buffers stay resident. It then allocates the KV cache, other weights and work buffers, and sizes the Expert cache from the remaining free VRAM. On Windows, it uses the smaller of CUDA's free-memory reading and the NVIDIA driver's free dedicated VRAM when both are available.

Cache contents change as different Experts are used; cache capacity is recalculated when the model reloads. Other applications opening later can still consume VRAM.

Expert placement normally targets 768 MiB of headroom. Following Strata, near-fit placement can release packed Q8 weight copies, shrink an Attention score buffer where present and reduce the target to 256 MiB so all main-model Experts fit in VRAM, while MTP Experts can run on the CPU. `--vram-reserve-mib N` fixes the reserve explicitly. This is a placement target, not a permanent guarantee of free memory.

More GPU-resident Experts usually means less CPU work. In hybrid mode, system RAM also retains backing copies of Experts that are cached in VRAM, so adding VRAM does not reduce RAM usage byte for byte. Leave RAM for the operating system and other applications too. More **RAM capacity** lets you hold model data, but speed also depends on the CPU, RAM bandwidth and GPU. Windows' "shared GPU memory" is different from TinyStrata deliberately computing Experts on the CPU.

The KV cache stays in VRAM; system RAM holds Expert data for this offload mechanism. A long conversation therefore still needs enough GPU memory for its KV cache.

## Automatic settings — Qwen3.6

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

For Qwen, thinking is enabled by default, with `temperature=0.7`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`, and a maximum of **32,768 output tokens** including thinking. Per-request settings take precedence.

The API listens on `http://127.0.0.1:18080` by default. To use another port, run `run-api.bat --port 18081` from the installed TinyStrata folder.

For Qwen, `run-api.bat --vision gpu` uses GPU image processing (the default for new setups); `--vision cpu` uses the CPU and `--vision no` disables images. `--vision-tokens N` sets the image-token limit (default: 4096). These choices are saved and reused when omitted on later starts.

The API can start before the model files are installed. With Qwen selected, `POST /v1/model/download` with an empty JSON body (`{}`) downloads the fixed [Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS language model](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf) (14.1 GB, including MTP weights) and its [vision projector](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf) (0.9 GB), about **15.0 GB in total**. It verifies both files' SHA-256 checksums, extracts the tokenizer and loads them. `GET /v1/model/download` reports progress; interrupted downloads can resume. A repeated download request reuses the active job, and `POST /v1/model/download/cancel` can cancel transfer or verification. Download completion is not inference readiness: wait for the download state to become `ready` and `/health` to report `loaded: true`, plus `images: true` when images are enabled.

The installer prepares the engine binaries and GPU runtime libraries, and downloads Qwen and its vision projector by default. Existing configurations keep their selected local model until you explicitly change it.

## Acknowledgments

TinyStrata is an independent fork of [Strata](https://github.com/Niko1221/Strata) and draws heavily on its implementation. Thanks to [Niko](https://github.com/Niko1221) and the Strata community for the engine and the work that made TinyStrata possible.

Thanks also to [Aaron Flin](https://github.com/aflin) and the [MoMoE project](https://github.com/aflin/momoe) for the KV cache, attention and quantized Expert code adapted in TinyStrata.

## License

MIT. See [LICENSE](LICENSE).

Copyright (c) 2026 Niko1221 and the Strata contributors.

Upstream attribution: [NOTICE](NOTICE). GGML: [MIT license](third_party/ggml/LICENSE).
