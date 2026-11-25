# Inkling 2 Mini KV, prefill and route audit on RTX 3060

The four-start CUDA v2 tests use the pinned Unsloth UD-IQ4_XS weights, the native Vision shard, one active request, and 131,072 device KV tokens on the RTX 3060 12 GB. Hardware is the Ryzen 9 7945HX3D Windows test PC with driver 610.88 and CUDA 13.2.86. The original four-start FP16 result is available separately. [FP16 v2 benchmark](rtx-3060-12gb-20261125-v2-inkling.md)

## Optimizations from reference engines

Each screen ran on the RTX 3060 with the same IQ4 weights, 128K INT8 KV, 1024-token chunks, CUDA Vision and greedy sampling. Each value averages two independent starts per setting in the same experimental binary. The short and long-input response text and reasoning matched their controls in these screens.

| Candidate | Measurement | Control → candidate (tok/s) | Result |
|---|---|---:|---|
| Larger Expert staging groups | 4K input | 496.71 → 528.22 (+6.34%) | Added to the 12 GB SM86 CUDA profile when the VRAM budget allows |
| Grouped Q/K/V projections | Short generation | 37.47 → 37.78 (+0.82%) | Added to eligible single-row Q8_0 projections on SM86 |
| Grouped Q/K/V projections | Generation after 4K | 36.02 → 36.23 (+0.58%) | Same Expert placement |
| Asynchronous staging reuse | 4K / 16K input | 496.85 → 499.14 (+0.46%) / 475.59 → 476.78 (+0.25%) | Held back |
| Batched Expert copies | 4K input | 496.71 → 500.83 (+0.83%) | Held back |

The staging-size change takes the larger transfer-group idea from Basalt. Two 128 MiB buffers replaced two 64 MiB buffers in the measured profile. Resident Expert capacity fell from 894 to 864, and short generation with just this change averaged 37.35 tok/s (−0.33%). The planner keeps its existing reserve and budgets the buffers before placing Experts. If the allocation fails, the engine tries its standard and minimum sizes. [Basalt staging](https://github.com/jesdga95/basalt/commit/071018a69ca40302c1913a9928c93b0d67010fc6)

The grouped projection uses Maya's separate-input Q/K/V scheduling and TinyStrata's existing Q8_0 quantization and MMA accumulation. Mixed-format layers use their existing individual calls. The separate batch-copy experiment queued 19,013 copies in 1,501 CUDA batches per start. [Project Maya](https://github.com/mw00/project-maya/commit/a6df38ddf2d43226bd7440d4b0114bd3ad2911f7) · [MoMoE-Strata](https://github.com/aflin/momoe/commit/c8a73fbf)

Both changes were tested together in the final release build. The following rates average one off/on pair and one on/off pair. The 4K and 16K completion timings include 32 generated tokens and the HTTP request. `STRATA_INK_PF_WIDE_STAGE=0` and `STRATA_INK_MAYA_QKV=0` reproduce the control.

| RTX 3060 12 GB | Control | Final default | Change |
|---|---:|---:|---:|
| Short code generation (tok/s) | 37.89 | 38.15 | +0.69% |
| 4K input (tok/s) | 493.35 | 528.12 | +7.05% |
| 16K input (tok/s) | 475.13 | 506.41 | +6.58% |
| Generation after 4K (tok/s) | 39.07 | 37.81 | −3.22% |
| Generation after 16K (tok/s) | 39.00 | 38.20 | −2.05% |
| 4K request completion (s) | 9.13 | 8.61 | −5.68% |
| 16K request completion (s) | 35.33 | 33.22 | −5.97% |

Short code, the exact 4K and 16K probes and the canonical image response matched across both settings. Two additional prompts reached their output limit during reasoning and their captured reasoning strings differed when the staging size and Expert placement changed together. Both settings used the same model weights and quantization.

The final managed profile returned all nine required facts from the 34,184-token workflow, reused that context for image input, read both predefined label values and recovered after invalid image and context overflow requests. A separate comparison at the same Expert placement matched all recorded GPU state hashes for 71 single-row forwards with the grouped projection off and on.

With 4 GiB reserved separately on the 12 GB card, the profile retained the standard 64 MiB staging size. With the 64K FP16 context target, the engine retained all 65,536 KV positions on the GPU and completed the 4K and GPU Vision workflows, including both label checks. This is an 8 GB-equivalent test on RTX 3060. The final CUDA executable SHA-256 is `ac56cab40fed0ba09a4a5c02d9eec02c284c8ac376305a017e1f9075b14ac198`.

## Existing-engine fast path follow-up

This follow-up screens mechanisms already used in TinyStrata. The comparisons use the same pinned IQ4 weights, GPU Vision, greedy sampling, and one slot. The 12 GB comparisons retain INT8, 128K device KV and chunk 1024.

| 12 GB configuration | Starts | Generation (tok/s) | 4K input (tok/s) | 16K input (tok/s) | Image first / complete (s) | GPU Experts |
|---|---:|---:|---:|---:|---:|---:|
| Separate Vision workspaces | 2 | 37.85 | 490.97 | 471.27 | 1.865 / 4.951 | 885 |
| Shared Vision workspace | 2 | 38.34 | 491.32 | 472.42 | 1.865 / 4.923 | 894 |
| Existing 64-token Expert tile disabled | 1 | 38.12 | 427.88 | 411.53 | 1.965 / 5.094 | 885 |

The workspace comparison averages the measured rates by case before averaging across starts. It is a development screen separate from v2. Sharing temporaries whose lifetimes do not overlap reduced GPU Vision workspace from 177.4 to 141.4 MiB, a saving of 37,765,120 bytes. Complete CUDA embedding arrays for the canonical image and two labelled images were bitwise identical with sharing off and on. The additional Expert residency can change floating-point rounding in subsequent language decoding. All fixture answers were checked separately from embedding equality. The small timing changes do not establish a universal speed gain. Disabling the existing Expert tile reduced 4K and 16K prefill throughput by about 13%, with identical captured outputs at the same Expert placement.

The Windows budget now applies the same minimum of CUDA and driver-reported dedicated VRAM used by other engines. Under a separate 4 GiB owned reservation on the 12 GB card, the earlier CUDA-only budget planned 873 GPU Experts and failed before model readiness. With the dedicated-memory bound, the Expert planning query used 4,204 MiB when CUDA alone reported 8,045 MiB. Both workspace configurations then retained the full 64K FP16 KV on the GPU at chunk 512, with zero resident Experts, and completed 4K input and three CUDA image requests. Both read the two label values correctly. This is a constrained-memory simulation on RTX 3060, with the 64K target set explicitly.

The initial same-binary screen found no dependable gain from changing CPU Expert worker count, slowing the Expert placement update cadence, or reusing packed KV tiles. The placement cadence change also changed the captured greedy code output. The existing verifier's 2/3/4/8-row INT8 windows produced different logits from singleton decoding for two probes on RTX 3060, so suffix lookup was held back.

The existing CPU Expert pool's shared intermediate was applied to Inkling's split GGUF_RAW and S_K down projections. One worker publishes the intermediate only after all producers finish. The same quantizer and per-row dot products run in both arms. A local CPU fixture passed 144 quantizer-byte and synchronization cases. The actual enabled engine reused the intermediate for 4,484 Expert jobs per 12 GB start and 6,816 per 8 GB start.

| Same-binary CPU Expert comparison | Starts per arm | Short code, off → on (tok/s) | Generation after 4K, off → on (tok/s) |
|---|---:|---:|---:|
| RTX 3060 12 GB, 128K INT8 | 2 | 37.50 → 37.54 | 35.58 → 36.20 (+1.7%) |
| RTX 3060 with 4 GiB reserved, 64K FP16 | 2 | 22.69 → 22.79 | 22.18 → 22.62 (+2.0%) |

The code and 4K answer and reasoning strings matched between the arms. Expert counts were fixed at 894 and zero for the 12 GB and 8 GB conditions respectively. Neither prefill nor short decoding showed a consistent speed gain. The executable SHA-256 for these comparisons is `18b33360c001b83c4edb1a62164d42cefe0378d9f8903609576fa3de4c4f7165`.

With the final managed settings (128K INT8, chunk 1024 and GPU Vision), the engine completed a 16K input and read both image labels correctly. It returned all nine required facts from a 34,184-token request, then reused the cache for a follow-up image. Invalid images and context overflow returned HTTP 400, and a valid retry completed.

Reference checkouts were refreshed to Strata `2af30ccaa`, Project Maya `1d0f8c5`, Basalt `48086c77`, PolyStrata `36e654b1`, and MoMoE-Strata `c8dd5a99`. This follow-up transfers mechanisms already present in TinyStrata.

## Four-start v2 result with GPU Vision

On 2026-11-25, the [CUDA Vision revision](https://github.com/kbc20f11/TinyStrata/commit/f2ba9b56b24fd6420401689278b194a3d33f0e6c) completed four independent starts under the same v2 extension, IQ4 weights, 128K INT8 KV, chunk 1024, and seed 1234 as the CPU Vision baseline below.

| RTX 3060 12 GB | CPU Vision | CUDA Vision | Change |
|---|---:|---:|---:|
| Image first output (s) | 24.11 | 1.87 | −92.2% |
| Image completion (s) | 27.41 | 5.31 | −80.6% |
| Generation (tok/s) | 38.68 | 36.84 | −4.8% |
| 4K input (tok/s) | 503.04 | 491.07 | −2.4% |
| 16K input (tok/s) | 483.86 | 472.28 | −2.4% |
| GPU Experts | 1,110 | 885 | −225 |

The 1024×768 image encoded on CUDA in 0.499–0.502 seconds on all four starts. Startup allocated 605.7 MiB for Vision weights and 177.4 MiB for its workspace before Expert placement and KV fitting. All 131,072 KV positions stayed on-device. The Expert cache occupied 3.73 GiB at startup and after 16K. The Vision allocation trades some text throughput for cold image latency.

The dedicated comparison decoded the canonical image and two independent shipping labels using the same image preparation and projector weights. All FP32 embeddings were finite. The largest per-image relative L2 error between CPU and CUDA was `1.05e-5`. The RTX 3060 correctly read both labels, reused the cache for a follow-up on the same image, and handled replacement with a different image of the same dimensions. A 34,184-token workflow returned all nine required facts. The following image request reused 34,288 tokens and read the label correctly. Invalid images and requests exceeding the context window returned HTTP 400. A valid retry completed.

The sampler, benchmark inputs and model hashes match the CPU Vision run below. The portable CUDA executable SHA-256 is `4a4319677114955789a9c03c7e7e3a95803ad4fe27ea0e098b4067cb249f1f4d`.

A 1120×920 label (644 image tokens) also encoded and answered on CUDA using temporary workspace. With CUDA Vision explicitly disabled, the same label workflow used the CPU and retained 128K. The image-disabled profile allocated no Vision memory, rejected an image request, and answered a subsequent text prompt. The portable HIP build passed for gfx1100, gfx1200 and gfx1201 after the shared image preparation change.

## Four-start v2 result with CPU Vision

Protocol `tinystrata-bench-v2-inkling` completed all four fresh starts at [5671776](https://github.com/kbc20f11/TinyStrata/commit/56717761024511a6f342d3b5deced2fbd04e5c35). Each start used seed 1234, ran the three generation prompts twice, and processed exact 4K and 16K inputs and the canonical cold image. This Inkling extension records `standard: false` because KV is selected by the backend and Vision runs in-process. Text rates pool native token counts and timings. Image times are medians including encoding.

| RTX 3060 12 GB | Generation (tok/s) | 4K input (tok/s) | 16K input (tok/s) | Image first output / complete (s) | GPU Experts |
|---|---:|---:|---:|---:|---:|
| Earlier automatic FP16, chunk 512 | 34.89 | 380.17 | 367.83 | 25.22 / 28.47 | 624 / 3,072 |
| Updated automatic INT8, chunk 1024 | 38.68 | 503.04 | 483.86 | 24.11 / 27.41 | 1,110 / 3,072 |
| Text throughput change | +10.9% | +32.3% | +31.5% | — | — |

The new portable build reported 4.68 GiB of GPU Expert allocation at startup and after 16K input in all four starts. It held 131,072 KV positions on the GPU, with 2,112 MiB of global KV and 2,387,607,552 bytes mapped including the other KV windows. CPU KV use was zero. Each start logged 204 eligible Expert staging visits, all routed. Staged bytes for those visits fell from an estimated 75.17 GB with full staging to 38.06 GB with routing. The new image measurements used 32 CPU workers and the older build used OpenMP.

| Input or artifact | SHA-256 |
|---|---|
| `benchmark/run.py` | `a27042683ba399f0f3684bcd32ed10856e6c4aa6649ac74eb7572bbea2f5eb9e` |
| `benchmark/cases.json` | `aa49a07349a72542a089222c755d4aa8203ab44b35e22fb3166838ef086eabde` |
| `benchmark/image.png` | `b6983fe5adacff8a4fc2f8b7b0cd3b235d8961550e9f9bca7a669396b3d33138` |
| `tinystrata-inkling2.exe` | `b71a58118351e0db93be0cdc6b929eca08f345f37214daad458f26de5df5bd48` |
| `Inkling-2-Mini-UD-IQ4_XS.gguf` | `a861d21e7c6d7bc69d8cdcdfb9eb195fd37ef1cd2ab31499db61cd7d30d9b3a1` |
| `Inkling-2-Mini-Vision.strata` | `6589eff47f9c20bc3a032faa3d449f10bf4f04a84cc6b2ee8a44d51d2b6fc677` |

## KV representation comparison

This screen used one fresh start per explicitly selected format, chunk 512, the same three generation prompts with seed 1234, and exact 4K and 16K inputs. Reported native formats were checked at startup.

| KV | Global device KV (MiB) | GPU Expert slots | Generation (tok/s) | 4K input (tok/s) | 16K input (tok/s) |
|---|---:|---:|---:|---:|---:|
| FP16 | 4,096 | 624 | 34.31 | 379.44 | 366.92 |
| INT8 (K8V8) | 2,112 | 1,110 | 37.97 | 410.82 | 398.51 |
| FP8 (E4M3) | 2,112 | 1,110 | 37.65 | 411.99 | 395.02 |

INT8 and FP8 free the same KV space for additional Experts. INT8 led generation and 16K input in this screen. The CUDA automatic-format policy uses that result for a single managed 128K slot on the same sm_86 memory class (11–13 GiB). Explicit KV settings retain their precedence. Results for other GPU classes require their own hardware measurements.

## IQ4 prefill and decode candidates

This screen uses INT8 KV, seed 1234 with greedy generation, a short code request and an exact 4,096-token prompt. The baseline and each candidate started in a new process. Rates shown are individual runs.

| Candidate | 4K input (tok/s) | Code generation (tok/s) | Free VRAM at startup (MiB) |
|---|---:|---:|---:|
| Chunk 512 | 419.61 | 38.2 | 846 |
| Chunk 768 | 460.83 | 37.9 | 635 |
| Chunk 1024 | 507.05 | 38.8 | 510 |
| Eight CPU Expert threads, chunk 512 | 418.31 | 37.4 | 769 |
| Routed-only Expert staging, chunk 512 | 455.48 | 38.9 | 784 |

Chunk 1024 and routed staging matched both captured baseline streams. At chunk 512, routed staging cut transferred Expert bytes from 151.14 GB to 109.32 GB. It applies to eligible sub-batches of up to 512 rows, so the improvement does not add to the chunk 1024 throughput increase. The changed automatic chunk rule keeps a startup memory check and allocation retry. The DSpark arm reached 130,752 context tokens and was excluded from the 128K comparison. [Basalt routed staging reference](https://github.com/jesdga95/basalt/commit/b78aadaaecb622396cb4716dbd997c3522ddde79)

## Windows portable Vision

The portable installer build has no OpenMP dependency. The CPU fallback now distributes independent rows among standard C++ workers and keeps each row's original dot product order. These comparisons use the same IQ4 and INT8 settings and an actual cold image per start.

| Candidate | Image first output (s) | Image completion (s) |
|---|---:|---:|
| One worker | Timed out at 301 s | — |
| Eight workers | 66.84 | 70.00 |
| Sixteen workers | 35.41 | 38.60 |
| Thirty-two workers | 24.10 | 27.23 |

The eight-, sixteen-, and thirty-two-worker streams matched for the canonical image and both labelled images. The portable Windows CPU fallback uses available logical processors up to 32, bounded by the number of image patches. The four-row weight-reuse helper passed a host arithmetic comparison, but the native arm did not activate and is excluded from the performance results. The older FP16 v2 binary uses OpenMP, making its image timing a different build comparison.

## Route checks

- The actual cached managed IQ3 migrated to the new IQ4 pin. IQ3 remained on disk, and an explicit local IQ3 selection stayed usable. The new IQ4 tokenizer and Vision artifact were prepared in an isolated configuration.
- With both options set to `auto`, the native engine reported INT8, chunk 1024, and a full 128K of device KV. With 256 MiB of VRAM held after Expert placement, the same route selected chunk 512 while keeping INT8 and the full context.
- The model returned all nine required facts from a 34K-token dossier. Tool call and continuation, image reuse and replacement, corrupt image handling, overflow and retry, and a structured Responses API completion passed.
- Native CUDA and HIP now publish actual Expert bytes and committed KV to the API. The benchmark records its native values before the workload and after 16K input. The UI renders `k8v8` as INT8 and uses the canonical model identifier for clients.
- The managed `--cpu-threads` option now reaches the native Expert worker pool in both GPU backends. A saved value of seven produced seven workers in the CUDA engine's statistics after generation.
- Managed discrete CUDA and HIP use the 64K, 128K and 256K context targets for the 8, 12 and 16+ GB classes. Inkling budgets its device KV at load time. Physical KV growth with Expert eviction, as used by other TinyStrata engines, is not implemented for Inkling.
- Windows-Portable HIP built and reported its capabilities with gfx1100, gfx1200 and gfx1201 device code. Hardware inference was exercised on RTX 3060. HIP runtime inference remains unmeasured.
