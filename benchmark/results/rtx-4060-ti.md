# Measured speed — NVIDIA GeForce RTX 4060 Ti 16 GB (CUDA)

Measured on **2026-11-09** under Windows 11 Enterprise (build 26200), using a **GeForce RTX 4060 Ti (16 GB; 16,380 MiB; compute capability 8.9; 165 W power limit; PCIe 4.0 x8)**, NVIDIA driver **621.14**, an **Intel Core Ultra 7 265K (20 cores / 20 threads)** and **64 GiB of system RAM** (four DDR5-4800 modules; 63.4 GiB visible to Windows), with the same model files as in the main [README](../../README.md). INT8 KV and KV growth were enabled. Context capacity, input chunks and Expert placement were automatic. Qwen and Gemma used MTP 3 and a GPU-resident image encoder; LFM and the standard Clover row use neither.

Protocol: [tinystrata-bench-v1](../README.md#english), engine code [1ea7380](https://github.com/kbc20f11/TinyStrata/commit/1ea7380827797477f85ce873f4f8faf5f6e7d422) (clean `origin/main`), seed 1234, four independent API starts per model. The engine was built with `install-ui.ps1 -Rebuild` using the Microsoft C++ Build Tools and the installer's private **CUDA 13.2.2** compiler (nvcc **13.2.86**).

| Model | Context capacity | Generation (tok/s) | 4K input (tok/s) | 16K input (tok/s) | Image first / end (s) | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.6 UD-IQ3_XXS | 256K | 111.26 | 802.28 | 759.28 | 1.97 / 3.23 | Expert cache 7.58–7.66 → 7.58–7.66 GiB |
| Gemma 4 QAT Q4_0 | 256K | 87.70 | 2566.66 | 2424.65 | 0.94 / 1.72 | Expert cache 9.16 → 9.00 GiB |
| LFM2.5 Q4_K_M | 128K | 130.78 | 2784.59 | 2181.39 | — | Expert cache 12.90 → 12.81 GiB |
| Clover Preview IQ3_XXS (causal, standard v1) | 256K | 64.84 | 1858.90 | 1568.01 | — | Expert cache 8.87 → 8.48–8.50 GiB |

Image times show first output / response end, including image encoding; lower is faster. Expert cache values show the actual allocation at startup → immediately after the 16K input, in GiB. The automatic input chunks selected for these runs were 2,048 (Qwen, Clover), 1,120 (Gemma) and 256 (LFM) tokens.

All four rows are complete standard v1 results (four of four starts, 24 generation requests per model, zero KV reuse in the timed inputs). The Clover row is the fixed causal D0 profile without Vision.

## Tiel Coder 35B-A3B (saved-profile variant)

| Model | Context capacity | Generation (tok/s) | 4K input (tok/s) | 16K input (tok/s) | Image first / end (s) | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Tiel Coder 35B-A3B UD-IQ3_XXS (MTP 3) | 256K | 94.00 | 874.29 | 838.85 | 1.78 / 3.07 | Expert cache 7.73 → 7.73 GiB; FP8 KV |

This row is **`tinystrata-bench-v1-saved-profile`** (`standard: false`), measured with `benchmark/run.py --model tiel --saved-profile`. It keeps the model's saved native settings and sampler (temperature 1.0, top-p 0.95, top-k 20, automatic KV, which selected native FP8 on this GPU) and runs each canonical generation text once per start instead of twice, so it is not directly comparable with the four standard rows above. The saved profile was created by switching to Tiel Coder once through the API after the download, because a profile that has never been loaded still carries `model_bootstrap` and the runner then waits for an engine that is never started.

## Run-to-run spread

Per-start values (each start pools its own tokens and milliseconds), minimum–maximum over the four starts:

| Model | Generation (tok/s) | 4K input (tok/s) | 16K input (tok/s) |
| --- | ---: | ---: | ---: |
| Qwen3.6 | 107.66–117.83 | 780.8–831.9 | 729.5–771.4 |
| Gemma 4 | 85.19–89.49 | 2536.1–2587.5 | 2350.7–2469.2 |
| LFM2.5 | 128.96–134.61 | 2729.2–2880.7 | 2171.5–2193.3 |
| Clover (standard v1) | 64.25–65.92 | 1810.7–1920.2 | 1537.9–1579.2 |
| Tiel Coder (variant) | 91.99–95.10 | 862.7–880.7 | 833.8–846.0 |

## Earlier run on the same PC

The same protocol was run on **2026-11-05** with engine code [2e51521](https://github.com/kbc20f11/TinyStrata/commit/2e51521df4df3548ccefc989b90ee2ea9df07247), before several background applications were assigned to the integrated GPU (Intel Graphics) in the Windows graphics settings. Between the two runs the engine was updated to 1ea7380; both changed, so the difference below is not attributed to either one.

| Model | Generation (tok/s) | 4K input (tok/s) | 16K input (tok/s) | Image first / end (s) | Expert cache, startup → after 16K (GiB) |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.6 | 111.04 | 755.79 | 726.29 | 2.15 / 3.86 | 7.17–7.35 → 6.53–7.28 |
| Gemma 4 | 87.97 | 2406.97 | 2481.64 | 1.02 / 1.74 | 7.95–8.00 → 7.75–7.81 |
| LFM2.5 | 110.37 | 2260.98 | 1869.34 | — | 11.68–11.99 → 11.75–11.81 |
| Clover (standard v1) | 64.13 | 1749.34 | 1541.48 | — | 7.86 → 7.62–7.66 |

## Measurement conditions

- Before the runs about 0.7 GiB of the 16 GiB of VRAM was in use, almost all of it by the desktop compositor. A separate inference application that had been holding about 10 GiB of VRAM was stopped first; with it running, the engine could not start (0.14 GiB free at startup).
- Some background applications were assigned to the integrated GPU (Intel Graphics) in the Windows graphics settings. Other-application usage is not collected by the protocol.
- The Windows power plan was Balanced, and the system memory ran at DDR5-4800 (configured clock speed 4800 MT/s).
- These numbers come from one PC, one run set per row; use them as a reference for what a GeForce RTX 4060 Ti 16 GB can reach, not as a guarantee. Differences of a few percent between single runs are within the spread above.

## Artifacts

| Artifact | SHA-256 |
| --- | --- |
| Native engine (`tinystrata.exe`, Qwen, Clover, Tiel) | `37d9c0b6c9677d8f85eb3ebf4cb5257b68366f37c44f25bf15021a4306ed20e6` |
| Native engine (`tinystrata-gemma4.exe`) | `c742cc7a7b7c7d203469662acb756f3d0e25b50420dc3db626325ac2f8d363b3` |
| Native engine (`tinystrata-lfm2.exe`) | `96cb8ac0ae687ccc4af00ee4a3394c499325a6512e5ae675d989a2afc59a582b` |
| Vision encoder (`tinystrata-vision.exe`) | `4f9a117b908ae40a168c65768fff800681cc8be3f7f7ad43868f1ea4b57ba9e9` |
| Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf | `36f9ec0e4c775f6efd3a61c1ea76f0875128469c3d012e3d9e2495a90e7a7150` |
| Qwen projector `mmproj-F16.gguf` | `71f3cbc1f7cc0f30d09d41cfa924c0060827ebc33bf15ace7e86661e856f0160` |
| gemma-4-26B_q4_0-it.gguf | `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d` |
| Gemma projector `gemma-4-26B-it-mmproj.gguf` | `a359953a076b877db30c31dbbb4c6d93b4a6e017ee5db5784247e4d4c0dd4f3b` |
| Gemma MTP assistant `mtp-gemma-4-26B-A4B-it.gguf` | `7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a` |
| LFM2.5-24B-A2B-Q4_K_M.gguf | `d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d` |
| Clover-35B-A3B-Preview-AD-IQ3_S-IQ3_XXS.gguf | `03816cdc365892dfb628d49e0cae2c80003de903966b9d30db2ece231699c2d3` |
| Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf | `a4d8ffc7f030466c67db289543871a8f9e6d5340241b300c09fbaf16642273ca` |
| Tiel projector `mmproj-BF16.gguf` | `d9ce31026d1cb1f3f8d5152e2e2a014d9d2b302b6c93a7dc07bb0a0487f52837` |
| Benchmark runner (`benchmark/run.py`) | `fced0e499d8b822781846880893b26550dc3ab4ed310b4bdb8de4aacdad13905` |
| Canonical cases / image | `aa49a07349a72542a089222c755d4aa8203ab44b35e22fb3166838ef086eabde` / `b6983fe5adacff8a4fc2f8b7b0cd3b235d8961550e9f9bca7a669396b3d33138` |
