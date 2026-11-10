# GPT-OSS MXFP4 cached-Expert down — 2026-11-10

**The selected implementation reuses staged Q8_1 activations for cached GPT-OSS MXFP4 down projections, with an automatic default on single-GPU HIP gfx1200.** It changes GPU scheduling and floating-point reduction order while retaining the stored weights, scales and existing activation quantization. The combined build keeps the [GPU sampler](gptoss-gpu-sampling-20261110.md) and [long-context attention](gptoss-sink-attention-20261110.md). The final combined build completes the canonical private API observation at **91.52 generation / 2801.83 4K-input / 1575.25 16K-input tok/s**. Development comparisons below remain labelled by build and are separate from this final-profile observation.

The layout adapts [Basalt commit `30ad59a`](https://github.com/jesdga95/basalt/commit/30ad59a67a9447449f0c2afa5251e356fb07837b), `src/models/deepseek41/dspark_kernels.cu`, under its retained MIT notice. Only activation staging and row scheduling are used; no drafter, DFlash or donor activation function is introduced. TinyStrata already used integer dot products. This change shares their activation reads across output rows and overlaps successive weight loads.

## Scope and selected grouping

The fast path accepts the existing **2880 × 2880 MXFP4 down matrix**, cached direct/host-Slots/device-IDs calls and counts **1–32**. Its 1,530-byte rows alternate alignment, so each 17-byte block load aligns the absolute address and handles the final block without assuming row padding. Existing scale conversion, including exponent byte 255, and `(weight_scale * activation_scale) * integer_dot` association are preserved. Stored E2M1 codes and Q8_1 bytes are unchanged. There is no weight repacking or new persistent VRAM allocation; the per-input staged activation occupies **4,320 shared-memory bytes**.

The selected **G2** schedule uses two output-row groups per warp for counts 1–3, exposing more thread blocks when only a few Experts are cached. Counts 4–32 retain the initial **G4** four-group schedule. Both use the same per-row dot/reduction arithmetic. This coverage matters for serial/window and accepted-prefix state consistency: GPT-OSS uses four Experts per token; wider fallback work is chunked into at most eight tokens / 32 entries. CPU Experts, gate/up, bias, activation, weighted combine and unsupported GPU shapes retain their existing paths.

On single-GPU **gfx1200 HIP**, the eligible path is automatic. `TINYSTRATA_GPTOSS_MXFP4_DOWN=0` selects the old kernel for diagnosis. **gfx1201 defaults off** and permits explicit `=1` as an unmeasured opt-in. CUDA, other GPU architectures, split-device operation and other models retain the generic path. `mxfp4_down_seen` bits identify direct/Slots/IDs submissions observed during capture; they are not token or graph-replay counts.

## Native development screens

Local RX 9060 XT **16 GB / gfx1200, Windows HIP**. Each screen is one same-binary A/B pass, changing only `TINYSTRATA_GPTOSS_MXFP4_DOWN=0/1`, with GPU sampling enabled, temperature/top-p/top-k **1/1/0**, min-p 0, neutral penalties, seed **1234**, INT8 KV, graphs, context **32,768 / chunk 1,120**, MTP/lookup off and no images. Both arms explicitly set **`STRATA_FREEZE_EXPERTS=1`**, reserve the full KV before placement and record zero swaps. Each fresh worker performs an excluded 32-token warm-up, then emits **64 tokens** after a 94-token short prompt and exactly **4,096 frozen input IDs**. Prefix reuse is zero. All reported workers unload normally and protected settings/history hashes match. Input rates remain in raw reports; no prefill gain is claimed.

| Build / capacity / frozen placement | Workload | Old down, tok/s | New down, tok/s | Observed change |
| --- | --- | ---: | ---: | ---: |
| Initial G4, uncapped 16 GB, 768 Experts / 9,682 MiB | Short | 89.84 | 92.30 | +2.74% |
| Initial G4, uncapped 16 GB, 768 Experts / 9,682 MiB | After 4K | 79.72 | 81.29 | +1.97% |
| G2, simulated 12 GiB, 611 Experts / 7,702 MiB | Short | 54.26 | 54.90 | +1.18% |
| G2, simulated 12 GiB, 611 Experts / 7,702 MiB | After 4K | 51.31 | 53.25 | +3.79% |
| G2, simulated 8 GiB, 285 Experts / 3,593 MiB | Short | 36.73 | 36.75 | +0.04% |
| G2, simulated 8 GiB, 285 Experts / 3,593 MiB | After 4K | 34.62 | 35.03 | +1.18% |
| G2, simulated 8 GiB, fixed 244 Experts / 3,076 MiB | Short | 34.29 | 34.77 | +1.40% |
| G2, simulated 8 GiB, fixed 244 Experts / 3,076 MiB | After 4K | 33.58 | 34.31 | +2.19% |

The capacity limits use real HIP reservations on the same physical 16 GB card, not separate 8/12 GB GPUs or a renamed Expert budget. A/B placement matches within the displayed rows. The G2 automatic placement of **611/285** Experts differs from the earlier G4 **569–570/244** placement, so rates across those historical screens do not isolate the grouping change. The fixed-244 G2 rerun addresses that geometry difference.

Sampled IDs differ except for G2's 285-Expert short row. Its IDs and work match and its +0.04% result is flat. The full-cache G4 rows have equal entry counts and no CPU work but different sampled IDs. In the hybrid rows, selected CPU/GPU Expert counts can also differ: fixed-244 short CPU entries are **4,480 → 4,479**, and after 4K **4,089 → 4,072**. Those changes are **same-geometry algorithm observations, not identical-output matched-work causal speed estimates**. These single passes do not establish repeatability. Initial G4 lower-capacity screens were flat/slower and motivated G2; their raw results are retained rather than pooled with the selected schedule.

The full-cache rows were measured on the initial G4 binary, SHA-256 `0d95fefd0eb4840c12c92aea49611f8ce0ec75590af6ad193fe3c95499a8c21b`. Count-four runtime arithmetic/scheduling is retained by G2, but these are not fresh final-build measurements. G2 rows use SHA-256 `80199f506589ea822b41a9959bf9388dac4effd6b897a2ee75c250ad8b86ef74`.

## Bounded numerical and state evidence

The G4 full-model comparison captures **64 full-vocabulary rows for each of short and 4K input**, with identical history, geometry and Expert work. The targets are sampled baseline tokens, also used as teacher-forced history; they are not independent accuracy labels. Actual A/B argmax values are calculated from their respective full logits.

| Fixed history | Argmax agreement | Logit RMS / max absolute | Target mean NLL: old → new | Mean NLL delta |
| --- | ---: | ---: | ---: | ---: |
| Short, 64 positions | 64/64 | 0.154938 / 6.079463 | 0.7916215 → 0.7885025 | −0.0031190 |
| 4K, 64 positions | 63/64 | 0.077190 / 1.322328 | 1.0152076 → 1.0220698 | +0.0068622 |

The one 4K argmax change is at row 21, where the old top-two margin is **0.007351**. Floating-point changes can propagate through activation quantization and MoE routing, so complete-model differences are larger than isolated dot errors. The implementation introduces no deliberate precision reduction or pruning. These 128 self-generated targets support a bounded numerical comparison, **not an independent quality evaluation or proof of equal general quality**. G2 retains the tested row arithmetic; these numerical captures are from the initial G4 build, while combined-build state/default checks are recorded below.

The G2 GPU unit passes **68 cases / 17,582,050 checks**. Coverage includes all 16 E2M1 codes and 256 scale bytes, existing-quantizer activations, FP64 reference calculations over the same stored Q8_1 bytes, counts 1–32 with bitwise prefix consistency, repeated/reordered IDs, both row alignments and the final block, output guards, unsupported-shape fallbacks and graph replay after changing IDs/table contents. The initial integrated G4 model tests pass with graphs off/on: mixed **324/768** cache state/window **52 checks** and device sampling **330 checks / 58 draws / 24 lazy host rows** per mode. All-resident **768/768** state tests also pass **52 checks** per mode, with device-IDs dispatch observed. G2 repeats the mixed state tests with **52 checks** per graph mode and direct/Slots dispatch observed. Exact serial/window and restoration expectations are retained.

## Final combined default build and API observation

The combined source is based on phase2 `0392149d536834c3505fa11659df06ac04e6ef15`, preserving long-context sink attention and GPU sampling, and selects G2 for counts 1–3. The final native SHA-256 is **`492cca194468a499cdd4c25e2e0750ae44bbfb26e1823d63eada0f0272e52b63`**.

The final HIP build passes both the **68-case / 17,582,050-check down unit** and the **150-check sink-attention unit**. Five model-backed runs from the combined source pass **1,438 assertions each**, with **242 draw comparisons, 133 lazy host-logit rows and 24 boundary cases per run**: graphs off/on at positions **8188/8193**, plus graphs on at **16380**, using context **32,768 / chunk 1,120**, INT8 KV and a mixed **324/768** Expert cache. Sampler/down/sink overrides are all absent. Default down activation is asserted, with direct/Slots dispatch observed (`requested=1, enabled=1, seen=3`). All-vocabulary exact serial/window, accepted-prefix/replacement, restore and two-sequence oracles remain unchanged.

CUDA **13.2.86 compile-only** verification passes **seven checks**: MSVC `engine.cpp`, `gemma4.cpp`, `gptoss.cpp`, `main.cpp`, `cpu_experts.cpp`, and the new down shader for **SM86/SM120**. The gfx1201 HIP shader also compiles as a syntax/compatibility check; no gfx1201 hardware correctness or performance is claimed. NVIDIA linking/runtime/performance remain unvalidated.

One fresh private API start completes **`tinystrata-bench-v1-gptoss`** (`standard: false`) on the same RX 9060 XT. The final profile uses medium reasoning, temperature/top-p/top-k **1/1/0**, min-p 0, neutral penalties, seed 1234, INT8 growing KV, automatic **128K context / 1,120-row chunk**, MTP/lookup off and no images. No GPU-sampler, down or sink environment override is supplied.

| Final combined profile | Generation, tok/s | 4K input, tok/s | 16K input, tok/s | Expert cache: startup → after 16K |
| --- | ---: | ---: | ---: | --- |
| GPT-OSS 20B MXFP4 | **91.52** | **2801.83** | **1575.25** | **9.46 → 9.46 GiB**, 768/768 Experts |

Three canonical generation prompts run twice at 128 tokens; exact fresh 4K/16K inputs run once each at 32 output tokens, after an excluded 32-token warm-up. All **eight measured requests / 832 emitted tokens** have zero reuse, **832 GPU draws, CPU/fallback draws 0 and host-logit copies 0**. Every request confirms the automatic down path (`requested=1, enabled=1, seen=4`, device-IDs dispatch). The seen mask is capture evidence, not a call count. Full 128K KV is mapped at startup (**1,632 MiB**). Normal unload succeeds without forced cleanup; protected settings/history hashes match. This is one final-profile observation, not a controlled comparison with previous API tables or a prefill gain claim.

A bounded **NEW-only shared-Gemma text/image smoke** also passes on this final native. It uses one private API start with **6 GiB frozen Expert cache, 4K context, MTP0 and GPU Vision**, reusing the preserved controlled configuration and profile snapshot. Text returns **56** (3 tokens); the image identifies the red square A, blue circle B and green triangle C (53 tokens). Both outputs exactly match that stored golden, with zero reuse, normal stop and `mxfp4_down_requested/enabled/seen=0/0/0`, confirming this GPT-OSS specialization stays inactive for Gemma. Native/Vision core-runtime identity, graceful unload and protected settings/history hashes pass. This is a bounded functional smoke, not a speed test, automatic-MTP equivalence test or comprehensive other-model regression. No deployment/publication claim is made here.

Raw evidence: `TinyStrata-work/gptoss-mxfp4-dot-20261110/{screen-integrated01,screen-lower-vram,screen-lower-vram-g2,screen-8-fixed244-g2,forced-integrated01,model-checks}`. `candidate02-snapshot.json` records the G2 source/body relationship; `attribution-fix.json` records the source-notice-only correction. Final evidence adds `unit-result-final.json`, `model-result-final.json`, `final-checks-v2`, `builds/cuda-final-7`, `final-api-benchmark`, `final-api-benchmark.launcher.json` and `gemma-final-smoke`. Full IDs, logits, INFO, commands and cleanup records remain available privately.
