# GPT-OSS SWA, long-prefix Attention and activation fusion — 2026-11-10

Measured on the RX 9060 XT 16 GB / gfx1200 Windows HIP PC, extending `86e0a3d`. Final native SHA-256: `13abded69a1a8a81ac57e9165a023f317e9661f7a1c8bf28dea7d4ef4f0e6fc7`.

## Selected implementation

Single-GPU HIP gfx1200 enables **SWA prefill, long-prefix full Attention and MMQ activation fusion** by default (`gptoss_fusion_requested=11`). Explicit environment zeroes override these defaults. Other GPUs/backends retain previous defaults; CUDA shaders compile to unsupported-path stubs. No weights, Expert counts, KV storage precision or context rows are deliberately reduced, and DFlash is not used.

* **SWA:** extend the existing 16-query / 32-key WMMA kernel with per-row 128-token causal windows and absolute-position-to-ring-slot mapping. Full-Attention specialization keeps its existing arithmetic. Small decode/window rows remain on their previous paths.
* **Long prefix:** use 32 queries / 512 threads / 32 keys and **62,976 bytes of LDS** to reuse each KV tile across more queries. Device thread/LDS limits are checked before launch. It serves full INT8 causal batches beyond 16K keys; unsupported cases fall back. Both Attention kernels retain FP16 Q/restored KV/probabilities, FP32 accumulation and one learned sink.
* **MMQ activation:** fuse GPT-OSS SwiGLU+bias/clamp with the existing MXFP4 MMQ D4 activation quantization. Cached and staged Expert row offsets/IDs are preserved. The resulting scales, codes and padding are byte-equivalent to the separate producers; no new quantization step is added. The intermediate FP32 activation write/read is removed.

Diagnostics: `TINYSTRATA_GPTOSS_SWA_PREFILL=0`, `TINYSTRATA_GPTOSS_LONG_PREFILL=0` and `TINYSTRATA_GPTOSS_ACT_QUANT=0` disable the corresponding additions. `TINYSTRATA_GPTOSS_ACT_MMQ` separately controls MMQ. Plain decode Q8_1 fusion passes byte parity but did not establish a consistent generation gain, so it remains off unless `TINYSTRATA_GPTOSS_ACT_Q8=1` is supplied. INFO's seen mask is engine-lifetime dispatch/capture evidence: **1=SWA, 2=long prefix, 4=plain Q8_1, 8=MMQ**; it is not a per-token count.

## Final canonical API observation

Before and final use `tinystrata-bench-v1-gptoss` (`standard: false`), medium reasoning, temperature/top-p/top-k **1/1/0**, seed 1234, neutral penalties, INT8 growing KV, **128K context / 1,120-row chunk**, MTP/lookup off and no images. Each start excludes a 32-token warm-up, measures three generation cases twice at 128 tokens and fresh exact 4K/16K cases at 32 tokens. Final sampler/down/Attention/fusion overrides are absent.

| Case | Before (tok/s) | Final default (tok/s) | Change |
| --- | ---: | ---: | ---: |
| Generation | 91.65 | 91.23 | -0.46% |
| 4K input | 3430.20 | 3829.83 | +11.65% |
| 16K input | 2458.10 | 2643.52 | +7.54% |

Both starts keep **768/768 Experts / 9.46 GiB** and **1,632 MiB mapped KV** throughout. All eight measured requests have zero reuse, 832 GPU draws, no CPU/fallback draws or host-logit copies, and normal unload with protected settings/history hashes unchanged. Final INFO reports requested 11/seen 9: these requests do not extend beyond 16K. Sampled outputs differ; these single observations are not identical-output causal estimates. Generation is effectively unchanged.

The following native development cases emit 32 tokens, use the same context/chunk/sampler and fresh IDs, and are separate from the canonical API protocol. Before is the actual mask 0 arm of `screen-long` on candidate02 (`26e956…`); after is one selected-mask 11 start of the final binary. Input hashes and all-resident geometry match. This is not a same-binary paired timing; the separate long-only mask 0/2 screen uses one candidate binary and measures **+71.85% / +108.31%** at 32K/64K.

| Fresh input | Processing seconds, before → final | Input tok/s, before → final | Change |
| --- | ---: | ---: | ---: |
| 32K | 29.99 → 16.43 | 1092.50 → 1994.76 | +82.59% |
| 64K | 112.28 → 52.06 | 583.70 → 1258.79 | +115.66% |

SWA alone measures **+9.85% / +6.74%** at 4K/16K on the earlier candidate01, with equal raw Expert-work counts/geometry but different sampled outputs. Historical component screens are retained with their actual binary hashes rather than relabeled as final-build results.

## Capacity checks

The same physical 16 GB GPU reserves real HIP memory to simulate 12/8 GiB. These are not separate cards. Final mask 0/11 uses normal automatic placement and growing KV:

| Simulated capacity / input tokens | Input tok/s, off → on | Change | Geometry / raw Expert work equal |
| --- | ---: | ---: | --- |
| 12 GiB / 4096 | 3293.66 → 3667.62 | +11.35% | no / no |
| 12 GiB / 16384 | 2361.76 → 2572.62 | +8.93% | no / no |
| 8 GiB / 4096 | 2869.96 → 3016.20 | +5.10% | yes / no |
| 8 GiB / 16384 | 2242.39 → 2377.97 | +6.05% | yes / no |

All these runs finish normally, but automatic placement/work changes. At 12 GiB, after-short/4K/16K cached Experts are **593/588/573 → 599/593/578**; at 8 GiB they are **310/304/294** in both arms. CPU/staged work and most continuations differ. These are practical same-capacity observations, not isolated matched-work gain estimates. Short/decode differences vary by workload and are not claimed as general generation gains.

A separate **MMQ-only** mask 0/8 comparison freezes Expert membership/KV, still using real16/12/8 GiB capacities. All corresponding output IDs, geometry and raw Expert-work counts match:

| Capacity | 4K input tok/s, off → on | Change |
| --- | ---: | ---: |
| 16 GiB | 3458.00 → 3535.00 | +2.23% |
| 12 GiB | 3274.18 → 3337.95 | +1.95% |
| 8 GiB | 1673.82 → 1687.05 | +0.79% |

These small one-pass gains do not establish repeated-run confidence. The initial activation-group mask 12 and initial combined mask 15 screens remain exploratory evidence; plain Q8_1 bit4 is not part of the selected default11.

## Numerical, state and compatibility evidence

The first activation prototype fails a native Q8_1 sum byte. HIP's `__fmul_rn` can still contract into a later sum; the fix uses the already-existing IQ fusion's `#pragma clang fp contract(off)` rule. Assertions were not weakened. The final unit passes **169 cases / 81,568,692 byte comparisons**, covering native IDs/slots and MMQ, scales/sums/codes, padding/untouched allocation tails, counts 1–32, actual 4480-row and odd 1837-row shapes, changed-data graph replay and guards. An earlier benchmark run passes 171 comparisons (165 base cases plus six benchmark comparisons), before those two additional shapes; its actual scope is retained separately.

SWA passes **90 cases**, including the original28, window/ring wrap, minimal storage, expired/future masks, graph replay, row counts larger than the window and position131000. Its selected FP64-oracle max error is **0.000203421**, versus **0.000203406** for the old path. Long-prefix Attention passes **35 cases**; all **4,587,520 outputs** at 1120 queries/32K prefix match the existing fused arithmetic bitwise. Late-prefix 32K/64K microchecks also validate the full matrix before timing.

Event-only times do not represent complete BLAS latency here: at 32K prefix, BLAS records **44.66 ms by GPU events / 148.62 ms synchronized wall**, versus the long-prefix kernel **55.76 / 55.81 ms**; at 64K, **44.31 / 294.59 ms** versus **108.73 / 108.79 ms**. The cause of this gap is not attributed to a specific driver mechanism. Selection relies on synchronized completion and native end-to-end input time, not an event-only extrapolation.

Both fixed-history replay arms execute the **final binary**, mask 0/11, with identical frozen inputs, geometry and work. Targets are fixed baseline-generated tokens from two earlier candidate02 starts; origin report/native hashes are retained separately from replay-native provenance. They are not independent accuracy labels or argmax oracles.

| Prefix tokens /32 positions | Argmax agreement | Logit RMS / max absolute | Target mean NLL, off → on | NLL delta |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 29/32 | 0.182385 / 3.820103 | 1.5853356 → 1.5969060 | +0.0115705 |
| 16384 | 32/32 | 0.167463 / 3.387082 | 1.5892523 → 1.5890933 | -0.0001590 |
| 32768 | 29/32 | 0.217843 / 3.416943 | 1.1648165 → 1.1827166 | +0.0179001 |
| 65536 | 32/32 | 0.337700 / 4.435641 | 0.4892394 → 0.5247224 | +0.0354829 |

Argmax agrees at **122/128** positions. Full-model outputs are not bitwise equal: Attention reduction/tile changes propagate through activation quantization and MoE routing. In particular,64K target mean NLL rises **0.4892394 →0.5247224** despite32/32 argmax agreement. This is bounded numerical evidence, not a claim of unchanged general quality or improved perplexity; no deliberate precision reduction/pruning is introduced.

Unweakened full-vocabulary sampling/window/state tests pass **338 checks** each with graphs off/on at position 193 using a mixed **324/768** cache, and **1,440 checks /242 draw comparisons /133 lazy host rows /24 boundary cases** with graphs on at position 32764 using all 768 Experts. Requested/seen masks are11/9 for the short cases and 11/11 for long. Accepted/rejected suffixes, state restore and two-sequence behavior are checked. Shared-Gemma controlled text/image smoke exactly matches its stored golden; this is a functional check, not a new Gemma speed benchmark. CUDA 13.2.86 compile-only passes **11 checks**: five shared host units and three shaders' fallback stubs for SM86/SM120. No NVIDIA hardware/performance result is claimed.

Raw evidence: `TinyStrata-work/gptoss-fused-followup-20261110/{api-before,api-final,screen-swa,screen-long,screen-activation,screen-mmq-fixed,screen-selected-low-vram,screen-selected-long,forced-final,unit,state,gemma-final-smoke,builds/cuda-fusions}` plus launcher/build/source-SHA receipts. The rejected activation prototype is preserved under `prototype/activation-candidate01`.
