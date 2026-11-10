# GPT-OSS fused prompt Attention — 2026-11-10

Historical first-stage measurements; the current default policy is described in [SWA/long-prefix/activation fusion](gptoss-fusions-20261110.md).

RX 9060 XT 16 GB (gfx1200), Windows HIP, GPT-OSS 20B MXFP4. This extends `dee730f`; it retains the GPU sampler, cached-Expert down kernel and small-row sink Attention. Native SHA-256: `93724629461cb381b5e4a61ed42e78cfd7299822907ea5264cdcacd9223baf80`.

## Implementation and scope

The previous large-batch path expands INT8 KV into global FP16 tiles, writes FP32 scores and FP16 probabilities, and launches BLAS/softmax/value operations for each tile. A private, synchronized event diagnostic at 16K input records **6,564.7 ms** of full-Attention stream intervals and **518.6 ms** of SWA intervals. These intrusive GPU-event intervals are not complete end-to-end timing: later synchronized wall measurements show substantial additional BLAS elapsed time. Do not infer a fraction of total prompt time from these event values.

The selected HIP kernel fuses KV expansion, QK, online softmax and PV inside a CTA. It uses **16 query tokens / 256 threads / 32 keys**, with two lanes per softmax row. Queries, restored KV and probabilities retain the existing FP16 rounding; matrix products and online sums retain FP32 accumulation. The learned sink is included once. All keys are retained and causal masks are enforced. The existing INT8 storage, weights, Expert selection and global scratch allocations are unchanged.

It is automatic only on **HIP gfx1200**, for full causal D64/GQA8 INT8 batches of at least 32 rows with **the entire batch ending at or before 16,384 keys**. Later prefix batches, SWA, small decode/speculative rows, FP16 KV and unsupported devices/backends retain their previous paths. A longer input still benefits from its eligible earlier batches. `TINYSTRATA_GPTOSS_PREFILL_ATTN=0` selects the original path. INFO reports `prefill_attention_requested` and engine-lifetime `prefill_attention_seen`; the latter is dispatch evidence, not a per-request count.

Initial 8-token and parallel-softmax variants were screened. A 64-key tile regressed and was discarded; the 16-token/32-key schedule was retained. Synthetic long-prefix event comparisons have an unexplained plateau in the old path, so those are not used as quantitative end-to-end gain estimates. The conservative 16K eligibility bound avoids extrapolating the selected schedule to longer prefixes.

## Fresh canonical API observation

Both starts use `tinystrata-bench-v1-gptoss` (`standard: false`), medium reasoning, temperature/top-p/top-k **1/1/0**, seed 1234, neutral penalties, INT8 growing KV, **128K context / 1,120-row chunk**, MTP/lookup off and no images. The before native is `492cca…63`; the final native uses defaults with no prefill/sampler/down/sink override. Each start measures three generation cases twice at 128 tokens, fresh 4K/16K cases at 32 tokens, and an excluded 32-token warm-up.

| Case | Before (tok/s) | Final default (tok/s) | Change |
| --- | ---: | ---: | ---: |
| Generation | 91.70 | 91.20 | -0.56% |
| 4K input | 2819.77 | 3415.61 | +21.13% |
| 16K input | 1578.25 | 2437.91 | +54.47% |

Both starts have **768/768 Experts**, **9.46 GiB** Expert cache and **1,632 MiB** mapped KV throughout. All eight measured requests have zero reuse, 832 GPU draws, no CPU/fallback draws and no host-logit copies. The final requests report the fused prefill dispatch. Normal unload and protected-settings/history hashes pass. These single passes have identical settings/geometry but sampled outputs differ; they are not identical-output speed comparisons. Generation is effectively unchanged in this observation; the change targets input processing.

A separate same-final-binary, frozen all-resident native A/B (off/on), same 128K context and 1,120-row chunk, measures:

| Fresh input | Off (tok/s) | On (tok/s) | Change |
| --- | ---: | ---: | ---: |
| 4K | 2813.38 | 3443.75 | +22.41% |
| 16K | 1570.36 | 2443.44 | +55.60% |
| 32K | 975.42 | 1084.00 | +11.13% |

These development cases emit 32 tokens and do not replace the canonical API generation measurement. Expert geometry/work and fresh input IDs match; sampled continuations differ. The 32K result exercises the later-batch BLAS fallback rather than running the new kernel over the complete long prefix.

## Numerical and regression checks

The model-free GPU unit passes **28 cases**: analytic sink, random/large queries, row counts 1–65 and tails, 4K/16K prefixes, actual 64-head geometry, causal future rewrites, NaN padding, output/input guards, graph replay with changed queries and rejected host arguments. Its selected row/head FP64-oracle max absolute error is **0.00018129**, equal to the maximum observed for the old path; the independently coded tiled-rounding reference differs by at most **0.000000585**. This checks the existing precision policy, not bitwise equivalence between different tile/reduction orders.

The full-model comparison teacher-forces identical histories for 32 output positions at each input size. Targets are baseline-generated samples, not independent accuracy labels. Geometry and Expert work match.

| Fixed history | Argmax agreement | Logit RMS / max absolute | Mean target NLL: off → on | NLL delta |
| --- | ---: | ---: | ---: | ---: |
| 4096 tokens | 32/32 | 0.176294 / 3.739345 | 2.0143173 → 2.0295009 | +0.0151836 |
| 16384 tokens | 30/32 | 0.201942 / 4.480060 | 1.5310645 → 1.5133912 | -0.0176732 |
| 32768 tokens | 30/32 | 0.172325 / 2.573727 | 1.4115939 → 1.4194026 | +0.0078087 |

The four changed argmax positions have old top-two margins **0.14873, 0.05535, 0.04280 and 0.01847**. Floating-point changes propagate through activation quantization and MoE routing; full-model logits are not bitwise equal. No additional quantization or pruning is introduced. These 96 self-generated targets provide a bounded numerical check, not an independent general-quality evaluation.

Unweakened full-vocabulary sampling/window/state comparisons pass with the new path observed: **336 checks / 58 draws / 24 lazy host rows** each with graphs off/on at position 193 and a mixed **324/768** Expert cache; **1,438 checks / 242 draws / 133 host rows / 24 boundary cases** with graphs on at position 16,380 and all **768/768** Experts. They cover serial/window agreement, accepted/rejected suffixes, state restoration and two sequences. CUDA 13.2.86 compile-only checks pass for five shared host units and the new shader's fallback stub on **SM86/SM120**; no NVIDIA acceleration/runtime result is claimed.

A NEW-only shared-Gemma API smoke also passes with the final native: the controlled text response is **56** and the image identifies the red square A, blue circle B and green triangle C, exactly matching the stored golden. It uses a 6 GiB frozen cache, 4K context, MTP0 and GPU Vision; protected configuration/history hashes and normal unload pass. This is a bounded functional check, not a Gemma performance benchmark.

Raw evidence remains in `TinyStrata-work/gptoss-prefill-20261110`: `api-before`, `api-final`, `screen-final`, `forced-final`, `unit/final`, `state`, `builds/cuda-prefill`, `gemma-final-smoke` and their build/launcher receipts. The candidate screens and private profiling instrumentation are retained separately.
