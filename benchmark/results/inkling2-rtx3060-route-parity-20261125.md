# Inkling 2 Mini KV, prefill and route audit on RTX 3060

The CUDA tests use the pinned Unsloth UD-IQ4_XS weights, the native Vision shard, one active request, and 131,072 device KV tokens on the RTX 3060 12 GB. Hardware is the Ryzen 9 7945HX3D Windows test PC with driver 610.88 and CUDA 13.2.86. The original four-start FP16 result is available separately. [FP16 v2 benchmark](rtx-3060-12gb-20261125-v2-inkling.md)

## Four-start v2 result with automatic settings

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
