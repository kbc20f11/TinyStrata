# GPT-OSS long-context sink attention — 2026-11-10

**The validated candidate02 implementation enables eligible long-context sink attention by default on HIP; CUDA defaults off with explicit opt-in.** The original one-token prototype violated serial/window consistency because wider windows used different attention arithmetic. Candidate02 applies the same per-row calculation to one through eight rows, handles the 8192-key transition explicitly, and passes the recorded native, numerical, state and API checks. Earlier-build lower-capacity observations are labelled separately. Deployment/publication are not claimed here.

Local development checks on **RX 9060 XT 16 GB / gfx1200, Windows HIP**, based on `231caa9` after the [GPU sampler change](gptoss-gpu-sampling-20261110.md). The new path splits full-attention keys into 512-row groups, shares existing INT8 KV reads across eight query heads, and adds the learned sink once when joining the partial sums. Queries stay FP32; no query quantization, KV omission, pruning or deliberate probability cutoff is introduced. Existing score scratch is reused without a new persistent VRAM allocation. DFlash is excluded.

## Conditions and candidate02 native screen

Both arms use candidate02 SHA-256 `c816d01b2c059a404d4d9a4eb0ac4837fe73d9b3626ec9131180fe8e1f252d4d`, with `TINYSTRATA_GPTOSS_SINK_ATTN=0/1`. Both keep GPU sampling on, temperature **1**, top-p **1**, top-k **0**, min-p **0**, neutral penalties, seed **1234**, INT8 KV, graphs on, MTP/lookup off and no images. Context is **32,768**, chunk **1,120**. Each arm starts once, runs an excluded 32-token warm-up, and emits **64 tokens** per request. Short input is 94 tokens; long inputs are exactly 4,096/16,384 frozen IDs. All requests have zero prefix reuse and normal EOF cleanup; protected settings/history hashes match. This is one A/B screen, not a repeatability estimate.

Both arms explicitly set **`STRATA_FREEZE_EXPERTS=1`**, disabling KV growth and fixing membership. The uncapped card holds **768/768 Experts / 9,682 MiB** throughout, with no CPU Expert entries. Entry counts and geometry match. Short/4K outputs match; 16K sampled outputs differ, so the latter is not an identical-output matched comparison.

| Candidate02 workload | Old attention, tok/s | Split attention, tok/s | Change | Output IDs |
| --- | ---: | ---: | ---: | --- |
| Short | 89.65 | 89.93 | +0.31% | All 64 match |
| After 4K | 79.61 | 79.74 | +0.16% | All 64 match |
| After 16K | 58.75 | 70.00 | +19.15% | Differ |

Short/4K remain on the old attention path and are near-flat. The initial prototype's 4K experiment (+0.78%, differing IDs) did not justify enabling that range and is not the final result. Input rates are preserved in the raw reports; no prefill gain is claimed.

## Bounded numerical evidence

The initial attention-only FP64 oracle passes **66 checks**, with maximum absolute error **1.422×10⁻⁷** and maximum RMS **4.365×10⁻⁸**. The expanded one-token suite passes **86 checks**, with maximum absolute/RMS error **3.704×10⁻⁶**, within the recorded `3e-5` absolute / `3e-4` relative tolerance. Candidate02 passes **150 checks**, including **64 bitwise multi-row-versus-one-row comparisons**, with the same maximum error. Full-model accumulation, activation quantization and routing can amplify small attention differences, so these unit results alone are insufficient to assess complete model output.

Candidate02 teacher-forced comparisons use the same sampled A tokens as fixed history and NLL targets; sampled IDs are **not** an argmax oracle. Actual A/B argmax values come from their respective full-vocabulary logits. Both arms freeze Expert membership and retain the same geometry/work. The 4K comparison has **byte-identical full logits**. For 16K, both newly captured candidate02 A/B dumps are **byte-identical to their respective initial-prototype dumps**, confirming the numerical figures below rather than assuming transfer between builds.

| Fixed history | Argmax agreement | Logit RMS / max absolute | Target mean NLL: old → new | Mean NLL delta |
| --- | ---: | ---: | ---: | ---: |
| 4K input, 64 positions | 64/64 | 0 / 0 | 1.2933231 → 1.2933231 | 0 |
| 16K input, 32 positions | 31/32 | 0.086119 / 2.125173 | 1.4379716 → 1.4403151 | +0.0023436 |

The 16K argmax difference occurs at position 14, whose old top-two margin is **0.133760**. Its mean NLL increases about **0.163%** on these 32 self-generated targets. This is bounded numerical evidence against the local baseline, not an independent quality evaluation or proof of identical quality. The implementation retains the mathematical attention operation while changing FP32 summation/normalization order; ordinary floating-point output variation is expected.

## Initial-prototype lower-capacity observations

These earlier-build observations use the same physical 16 GB card, with real HIP allocations reserving approximately **3.92/7.92 GiB** for simulated **12/8 GiB capacity**. They are not candidate02 measurements, separate smaller GPUs or manual Expert-budget simulations. The repeated lower-capacity screen explicitly sets **`STRATA_FREEZE_EXPERTS=1` in both arms**, reserves the full 32K KV before placement, and records zero swaps.

| Fixed-membership capacity / workload | Cached Experts / MiB | Old, tok/s | New, tok/s | Observed change |
| --- | ---: | ---: | ---: | ---: |
| 12 GiB, short | 570 / 7,186 | 50.31 | 50.12 | −0.38% |
| 12 GiB, after 16K | 570 / 7,186 | 40.33 | 44.98 | +11.52% |
| 8 GiB, short | 244 / 3,076 | 35.13 | 35.20 | +0.20% |
| 8 GiB, after 16K | 244 / 3,076 | 29.16 | 30.79 | +5.59% |

Long output IDs differ, changing the selected CPU/GPU Expert mix: at 12 GiB, CPU entries change **1,433 → 1,388**; at 8 GiB, **4,052 → 4,061**. Geometry and frozen membership match, but the long rows are **same-capacity algorithm observations, not matched-work causal speed estimates**. Short outputs/work match and remain flat.

The earlier `screen-lower-vram` used normal adaptive KV/cache behavior, with swaps and changing membership. Its raw results remain diagnostic only; they are not pooled with the frozen screen or used as controlled gain evidence.

## Candidate02 consistency and selected policy

The HIP default covers **D64/GQA8, INT8, full-attention calls of one through eight rows with actual key length greater than 8192**, preserving FP32 queries and all keys. Calls wider than eight rows, SWA/ring layers, shorter contexts, other shapes and non-INT8 caches retain the existing path. A window crossing the threshold computes its old prefix and new suffix separately; graph strategy keys include the old-prefix length as well as the rounded capacity. The same actual-key boundary is used with graphs on or off. This prevents both a premature 4097-key switch and a numerical mismatch between serial and window execution. `TINYSTRATA_GPTOSS_SINK_ATTN=0` retains old attention for diagnosis. **CUDA defaults off**; `TINYSTRATA_GPTOSS_SINK_ATTN=1` explicitly opts in where the same eligibility conditions hold.

Candidate02 native SHA-256 is `c816d01b2c059a404d4d9a4eb0ac4837fe73d9b3626ec9131180fe8e1f252d4d`. Its HIP build and 150-check unit pass. Five model-backed runs pass **1,433 checks each**, with **242 device draws, 133 lazy host-logit rows and 24 boundary cases per run**: graphs off/on at positions **8188/8193**, and graphs on at **16380**, using context **32,768 / chunk 1,120** and no sink environment override. Full logits match independent serial oracles bitwise for widths 2/3/4/8, partial/full keep, different replacement tokens after rejected suffixes, snapshot restoration and two-sequence stepping. The original default 768/64/193 configuration remains available.

CUDA **13.2.86** compile-only verification passes four candidate02 checks: MSVC `engine.cpp`/`gptoss.cpp` and `sink_attention.cu` for **SM86/SM120**. Source and cached build outputs remain unchanged. NVIDIA linking/runtime/performance are not validated.

## Candidate02 bounded 16K API sanity

Candidate02 runs one same-binary private API start per mode, reading exactly **16,384 fresh input tokens**, with medium reasoning, temperature/top-p/top-k **1/1/0**, INT8 KV, 32K context, all 768 Experts resident and a 1,024-token output cap. A explicitly disables split attention; B supplies **no `TINYSTRATA_GPTOSS_SINK_ATTN` environment variable**, checking the HIP default. The executable matches candidate02 SHA-256 `c816d01b…` above.

Both return the correct three labelled amounts for the facts near the beginning of the document: **12 × 180,000 = 2,160,000 yen**, budget **2,000,000 yen**, shortfall **160,000 yen**. Final answers happen to match exactly; matching wording is not a pass condition. Total output including reasoning is **213/192 tokens**. Both stop normally before the cap, have nonempty final answers and no sustained repetition, use GPU categorical sampling without host-logit copies, preserve protected settings/history hashes and unload gracefully. This single synthetic known-answer check supplements the NLL evidence; it is not an independent quality evaluation.

The candidate02 API rerun, same-binary native screen, captured numerical comparison, serial/window arithmetic and accepted-prefix state checks are complete within the stated scope. These bounded checks do not establish independent general model quality. Installed deployment/publication are not claimed. NVIDIA linking/runtime/performance remain unvalidated.

Raw evidence is retained under `TinyStrata-work/gptoss-sink-attn-20261110`: initial `unit-result.json`/`unit-result-final.json`, `screen-default`, `forced-history`, `screen-lower-vram`, `screen-lower-vram-fixed`, `api-long-sanity-final-retry`, plus `unit-result-candidate02.json`, `model-candidate02`, `builds/cuda-candidate02-4`, `screen-candidate02`, `forced-candidate02` and `api-long-sanity-candidate02`. Full commands/logits/IDs/cleanup records remain available. The first API fixture rejected an unsupported `--expert-vram auto` string before native startup; its failed record is retained, and the helper now uses default automatic placement. The candidate02 policy excludes the initial experimental 4K path.
