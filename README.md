# TinyStrata

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

**TinyStrata is a fast, Windows-only, API-only inference engine for 30B-class MoE models.** It primarily targets PCs with mainstream graphics cards (**8–24 GB of VRAM**) and **32 GB of system RAM or less**.

**NVIDIA GPUs (CUDA) are supported. The experimental AMD HIP/ROCm path uses INT8 KV and has been tested locally on a Radeon RX 9060 XT.**

## Why we are building TinyStrata

With PC component prices rising, putting together a high-end PC is increasingly out of reach for many people.

To address this, TinyStrata was created to run AI locally at high speed on PCs with an RTX 3060-class graphics card or better. Our focus is on making relatively small, 30B-class MoE models run as fast as possible.

Our main target is PCs with **8–24 GB of VRAM and 32 GB of system RAM or less**. We believe this range represents the most common configurations for PCs equipped with graphics cards.

TinyStrata builds on Strata to use GPU memory efficiently and run 30B-class MoE models quickly with limited VRAM. Our goal is to make long conversations and documents practical on your own PC.

We focus on local inference on Windows and serving it through an API.

TinyStrata automatically selects inference and memory settings for your hardware, so you can start using the API without complex configuration or manual tuning.

## Why no bundled UI?

Many API-compatible UIs and agent harnesses for LLMs already exist. TinyStrata supports common API formats, so you can connect whichever you prefer. If configuring the connection feels tedious, you can ask an AI agent to set it up for you. Leaving the UI to existing software lets us focus development on the inference engine and API.

## Model focus

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) remains the default model, with a balance of speed, memory efficiency and answer quality for PCs with mainstream graphics cards. TinyStrata focuses on compact MoE models and also supports **Gemma 4 26B-A4B-it and LFM2.5-24B-A2B**.

**Qwen3.6-35B-A3B UD-IQ3_XXS** uses the fixed [Unsloth MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf). The two downloads total about **15.0 GB**: language model including MTP weights **14.1 GB**, and vision projector (`mmproj-F16.gguf`) **0.9 GB**.

**Gemma 4 26B-A4B-it QAT Q4_0** uses the [Google-official model and vision projector](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf) with the [matching Unsloth QAT MTP assistant](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/mtp-gemma-4-26B-A4B-it.gguf). The three downloads total **15.886 GB**: model **14.439 GB**, projector **1.195 GB** and assistant **0.252 GB**.

**LFM2.5-24B-A2B Q4_K_M** uses the official [LiquidAI GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF). Its single language-model download is **14.53 GB**.

All three engines use an automatic VRAM Expert cache, CPU Expert computation from system RAM, prompt weight streaming and CUDA Graphs. They share the same API and support live model switching, with settings saved separately for each model. Selected Experts are not omitted. They also support automatic physical KV growth without changing the maximum context capacity.

| Feature | [Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | [Gemma 4](https://huggingface.co/google/gemma-4-26B-A4B-it) | [LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) |
| --- | --- | --- | --- |
| Maximum context | 256K | 256K | 128K |
| Image input | Yes; GPU-resident encoder by default | Yes; GPU-resident encoder by default | No |
| MTP | 3 | 3 | No |
| KV formats | FP32 / INT8 / native FP8 | FP16 / INT8 / native FP8 (experimental) | FP16 / INT8 / native FP8 |
| Automatic KV (default) | FP8 → INT8 | INT8 | FP8 → INT8 |

The column headings link to the official model cards. **FP8 → INT8** means native FP8 KV and FP8 Tensor Core Attention when the GPU and compiled kernels support it, otherwise INT8. KV precision is separate from GGUF weight quantization; lower-precision KV can change outputs relative to FP32 (Qwen) or FP16 (Gemma/LFM).

Gemma uses INT8 by default. Native FP8 is experimental and can be selected explicitly with `run-api.bat --model-family gemma4 --kv-type fp8-mma`.

**Automatic context capacity** uses the GPU’s total VRAM as shown below; available free VRAM determines the Expert cache size. An explicit `--max-context N` takes priority.

| Total VRAM (examples) | Qwen3.6 / Gemma 4 | LFM2.5 |
| --- | ---: | ---: |
| 8 GB | 64K | 64K |
| 12 GB | 128K | 128K |
| 16 GB or more | 256K | 128K |

Capacity classes allow for driver reporting differences, so a 16 GB card reported as 15.x GiB still selects 256K for Qwen and Gemma.

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

Measured on **2026-10-27** using Windows, an RTX PRO 4000 Blackwell SFF Edition (24 GB), a Ryzen 9 7945HX3D, 64 GB of system RAM and the fixed Unsloth MTP UD-IQ3_XXS model. The image encoder stayed on the GPU during API requests. Thinking, the model’s sampling defaults (temperature 0.7) and MTP 3 were enabled. Prompt chunks and Expert placement were automatic; context capacity was set for each VRAM class. Hybrid CUDA Graphs stayed at their default off; normal all-GPU CUDA Graphs remained enabled.

| Available VRAM | Context capacity | INT8 generation | Native FP8 generation | Notes |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 124.83 tok/s | 137.26 tok/s | All weights resident in VRAM |
| Approximately 16 GB | 256K | 101.04 tok/s | 103.07 tok/s | GPU and CPU Expert computation |
| Approximately 12 GB | 128K | 86.65 tok/s | 91.11 tok/s | GPU and CPU Expert computation |
| Approximately 8 GB | 64K | 66.24 tok/s | 68.99 tok/s | GPU and CPU Expert computation |

Each of the same three input texts was run twice per FP8/INT8 configuration, generating 128 tokens including thinking; sampling settings were left at their model defaults. Speed is total generated tokens divided by total decode time, excluding loading and prompt processing; context values are cache capacities. The reduced-memory rows reserve 8, 12 or 16 GiB on the same 24 GB GPU and use the context capacity for that VRAM class; they are not physical 16/12/8 GB GPU benchmarks.

## Measured speed — Gemma 4

Measured on the same Windows PC with 64 GB of system RAM, using Gemma 4 26B-A4B-it QAT Q4_0 and its matching QAT MTP assistant. The image encoder stayed on the GPU during API requests. MTP 3, CUDA Graphs and the model-specific thinking/sampling defaults were enabled; context capacity was set for each VRAM class; prompt chunks and Expert placement were automatic.

Native FP8 and INT8 were measured on **2026-10-27**.

| Available VRAM | Context capacity | INT8 (default) | Native FP8 (experimental) | Notes |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 115.89 tok/s | 115.46 tok/s | INT8 / FP8: Experts computed on the GPU |
| Approximately 16 GB | 256K | 102.83 tok/s | 100.86 tok/s | INT8 / FP8: GPU and CPU Expert computation |
| Approximately 12 GB | 128K | 85.53 tok/s | 91.71 tok/s | INT8 / FP8: GPU and CPU Expert computation |
| Approximately 8 GB | 64K | 52.98 tok/s | 55.09 tok/s | INT8 / FP8: GPU and CPU Expert computation |

Each of the same three input texts was run twice per FP8/INT8 configuration, generating 128 tokens including thinking; sampling settings were left at their model defaults. Speed is total generated tokens divided by total decode time, excluding loading and prompt processing; context values are cache capacities. The reduced-memory rows reserve 8, 12 or 16 GiB on the same 24 GB GPU and use the context capacity for that VRAM class; they are not physical 16/12/8 GB GPU benchmarks.

## Measured speed — LFM2.5

Measured on the same Windows PC with 64 GB of system RAM, using the official LFM2.5-24B-A2B Q4_K_M model, without MTP or Vision. CUDA Graphs and the model-specific thinking/sampling defaults were enabled; context capacity was set for each VRAM class; prompt chunks and Expert placement were automatic.

Native FP8 and INT8 were measured on **2026-10-27**.

| Available VRAM | Context capacity | INT8 | Native FP8 (default) | Notes |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 128K | 143.85 tok/s | 148.61 tok/s | INT8 / FP8: Experts computed on the GPU |
| Approximately 16 GB | 128K | 138.14 tok/s | 139.22 tok/s | INT8 / FP8: GPU and CPU Expert computation |
| Approximately 12 GB | 128K | 115.61 tok/s | 117.90 tok/s | INT8 / FP8: GPU and CPU Expert computation |
| Approximately 8 GB | 64K | 79.74 tok/s | 81.39 tok/s | INT8 / FP8: GPU and CPU Expert computation |

Each of the same three input texts was run twice per FP8/INT8 configuration, generating 128 tokens including thinking; sampling settings were left at their model defaults. Speed is total generated tokens divided by total decode time, excluding loading and prompt processing; context values are cache capacities. The reduced-memory rows reserve 8, 12 or 16 GiB on the same 24 GB GPU and use the context capacity for that VRAM class; they are not physical 16/12/8 GB GPU benchmarks.

## Measured speed — AMD Radeon RX 9060 XT (HIP)

Measured on **2026-10-28** using Windows 11, a Radeon RX 9060 XT (gfx1200, 16 GB; driver 32.0.32015.7029), a Ryzen 9 7945HX3D and 64 GB of system RAM, with the same three model files as above. INT8 KV and VMM growth with a 4 MiB allocation quantum were enabled. Context capacity, input chunks and Expert placement were automatic. Qwen and Gemma used MTP 3 and a GPU-resident image encoder; LFM used neither.

| Model | Context capacity | INT8 VMM generation |
| --- | ---: | ---: |
| Qwen3.6 UD-IQ3_XXS | 256K | 85.79 tok/s |
| Gemma 4 QAT Q4_0 | 256K | 80.77 tok/s |
| LFM2.5 Q4_K_M | 128K | 94.87 tok/s |

Thinking was enabled with the model sampling defaults: temperature 0.7 for Qwen, 1.0 for Gemma and 0.1 for LFM. Each value pools four independent API starts. On each start, the same three input texts were run twice, generating 128 tokens per request, including thinking: 24 requests and 3,072 generated tokens per model in total. Speed is total generated tokens divided by total decode time, excluding loading and prompt processing.

## Measured speed — NVIDIA GeForce RTX 5060 (8 GB)

Measured on **2026-10-25** on a Windows 11 PC with a GeForce RTX 5060 (8 GB, driver 616.92), a Core i7-10700K (8 cores), 32 GB of DDR4-3200 and the GPU on a PCIe 3.0 x8 link, with the same Qwen3.6 UD-IQ3_XXS, Gemma 4 QAT Q4_0 (with its MTP assistant) and LFM2.5 Q4_K_M files as above. Everything was left at the defaults through `run-api.bat` (CUDA backend, context capacity, KV type, input chunk, Expert placement and MTP chosen automatically; LFM2.5 has neither MTP nor Vision). Thinking was on and sampling was left at the model defaults, except that every request used `seed` 1234.

| Model | Context capacity | Decode, first 3 requests after start (two starts) | Decode, steady (median of 21 requests) | Prompt of about 2.5K tokens, first / later |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 UD-IQ3_XXS | 64K | 27.2 / 33.7 tok/s | 29.2 tok/s (24.1 to 35.0) | 61 and 62 / about 104 (89 to 118) tok/s |
| Gemma 4 QAT Q4_0 | 64K | 24.0 / 25.6 tok/s | 25.7 tok/s (22.2 to 28.7) | 333 and 369 / about 364 (361 to 371) tok/s |
| LFM2.5 Q4_K_M | 64K | 43.1 / 43.3 tok/s | 53.8 tok/s (42.4 to 56.8) | 595 and 685 / about 716 (690 to 762) tok/s |

Each request generated 128 tokens including thinking. Speed is generated tokens divided by `predicted_ms` from `/v1/status`, excluding loading and prompt processing. "First 3 requests" are three input texts in a fixed order right after the API reported the model loaded, on two separate starts (each value is total tokens divided by total time of the three). "Steady" is rounds 4 to 10 of ten rounds of the three texts on the first start (21 requests), with the range in brackets. The prompt rows use one input of 2,440 (Qwen3.6), 2,446 (Gemma 4) or 2,545 (LFM2.5) tokens, sent with a different first line each time so that nothing was cached: "first" is the first long request on each start, "later" the next three on both starts. The three short input texts and the long input were written for this run and are not the ones used in the sections above, so the rows are not directly comparable with them. GPU memory in use after the runs was 6.5 to 6.9 GiB of about 7.9 GiB, including the desktop.

- Qwen3.6 kept 12 (first start) and 22 (second start) of 256 Experts per layer in VRAM and the other 11.4 GiB in pinned system RAM, with an input chunk of 512. The two starts had 4.6 and 5.3 GiB of free VRAM, and their later prompt speeds were about 90 and 117 tok/s, which fits more resident Experts being faster; I did not test this separately.
- The prompt speeds are low compared with the 16 GB card above. The likely reasons are that most Experts are not in VRAM, that the Experts outside VRAM are computed on 8 CPU threads (Qwen3.6) or streamed to the GPU over a PCIe 3.0 x8 link of roughly 8 GB/s (Gemma 4 and LFM2.5, whose logs show staged Experts), and that free VRAM is too small to choose a larger chunk. Other `--chunk` values were not tried, so this is not confirmed.
- Single requests vary by 10% or more and the two starts differ by up to 6 tok/s: do not read small differences. These are one machine and one session.

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

This is a layout diagram for the default GPU image mode, not a fixed-size partition. At each model load, TinyStrata starts and warms up the GPU vision encoder before loading the language model; its weights and warmed work buffers stay resident. It then allocates the initial KV pages, other weights and work buffers, and sizes the Expert cache from the remaining free VRAM. On Windows, it uses the smaller of CUDA's free-memory reading and the NVIDIA driver's free dedicated VRAM when both are available.

Cache contents change as different Experts are used. With default KV growth, short conversations leave more VRAM for Experts; as the sequence grows, the Expert cache gives memory back to KV. New shorter requests can return unused KV pages to the Expert cache. `run-api.bat --kv-whole` allocates the whole KV capacity up front; `run-api.bat --kv-grow` restores growth. Other applications can still consume VRAM.

Expert placement normally targets 768 MiB of headroom. In Qwen’s `--kv-whole` mode, near-fit placement follows Strata and can release packed Q8 weight copies, shrink an Attention score buffer where present and reduce the target to 256 MiB so all main-model Experts fit in VRAM, while MTP Experts can run on the CPU. `--vram-reserve-mib N` fixes the reserve explicitly. This is a placement target, not a permanent guarantee of free memory.

More GPU-resident Experts usually means less CPU work. In hybrid mode, system RAM also retains backing copies of Experts that are cached in VRAM, so adding VRAM does not reduce RAM usage byte for byte. Leave RAM for the operating system and other applications too. More **RAM capacity** lets you hold model data, but speed also depends on the CPU, RAM bandwidth and GPU. Windows' "shared GPU memory" is different from TinyStrata deliberately computing Experts on the CPU.

The KV cache stays in VRAM; system RAM holds Expert data for this offload mechanism. A long conversation therefore still needs enough GPU memory for its KV cache.

## Automatic settings — Qwen3.6

New configurations choose KV/Attention and context capacity automatically. You do not need to select an 8 GB, 12 GB or 16 GB recipe.

For Qwen3.6, an RTX 3060 selects INT8. Supported RTX 40/50-series environments can select native FP8; the engine checks the compiled kernels and GPU capabilities as well as the hardware generation. If native FP8 is unavailable, automatic mode uses INT8. This chooses a supported path without running a speed or quality benchmark at startup.

The native engine estimates the prompt chunk size at startup, choosing 512, 1024, 2048 or 4096 tokens from available VRAM, context capacity and whether the Experts fit on the GPU. Larger chunks can reduce repeated RAM-to-GPU weight transfers during input processing. `run-api.bat --chunk N` fixes the size; `run-api.bat --chunk auto` restores native automatic selection.

Context capacity is the logical maximum for the input and generated output together. By default, physical KV memory is allocated as the conversation grows, rather than reserving the whole capacity for a short conversation. Long documents still require more computation and eventually more KV memory.

For scale, Qwen INT8 KV at full capacity takes about **0.71 GiB at 64K, 1.42 GiB at 128K and 2.84 GiB at 256K**, including the MTP layer; other weights and work buffers are additional. Native FP8 uses approximately the same KV storage as INT8 in this implementation; its benefit is a different computation path, rather than a substantially smaller cache.

Context capacity is chosen from total GPU VRAM; the Expert cache then fits into the free VRAM left at load time. Other applications using VRAM leave room for fewer GPU-cached Experts. The API rejects input that exceeds the context capacity instead of silently truncating it.

GGUF weight quantization and KV precision are separate. The selected UD-IQ3_XXS model has compressed weights; INT8 or FP8 describes the conversation cache and its Attention path. Lower-precision KV can change answers, so these modes do not promise identical output to FP32 KV. The implementation keeps all visible context positions and every model-selected Expert.

New launcher configurations use MTP 3, which proposes several tokens ahead and verifies them with the main model. It can improve generation speed, but the gain depends on how many proposals are accepted. Verification is intended to preserve the main model's decoding behavior; differences in numerical execution paths can still change tokens, so enabling MTP does not promise identical output to disabling it.

MTP supports both greedy generation and temperature-based sampling, including repetition, frequency and presence penalties. For sampled requests, the main model samples each verification row, and only the matching draft prefix is accepted.

Explicit settings override automatic choices, and existing manually selected settings remain in effect. For example, `run-api.bat --kv-type int8` fixes INT8; `run-api.bat --kv-type auto --max-context auto --expert-vram auto` restores automatic KV, context and Expert placement.

## API server

**TinyStrata is API-only.** It provides OpenAI-compatible Chat Completions (`/v1/chat/completions`) and Responses (`/v1/responses`) APIs and an Anthropic-compatible Messages API (`/v1/messages`), including streamed generation. Connect an API client or an external frontend; TinyStrata does not include a bundled chat UI.

New configurations use model-specific sampling defaults. Per-request settings take precedence; saved custom settings remain in effect.

| Setting | Qwen3.6 | Gemma 4 | LFM2.5 |
| --- | ---: | ---: | ---: |
| `temperature` | 0.7 | 1.0 | 0.1 |
| `top_p` | 0.95 | 0.95 | 1.0 |
| `top_k` | 20 | 64 | 50 |
| `min_p` | 0.0 | 0.0 | 0.0 |
| `presence_penalty` | 1.5 | 0.0 | 0.0 |
| `repetition_penalty` | 1.0 | 1.0 | 1.05 |
| `enable_thinking` | true | true | true (always enabled) |
| `max_tokens` (including thinking) | 32768 | 32768 | 32768 |

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
