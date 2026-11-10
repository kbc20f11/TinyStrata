# Upstream refresh — 2026-11-11

The reviewed update adds optional-image startup recovery, an owned Qwen-family command reader, and resource-aware Gemma mixed-image/text prompt scheduling. The six-model text-generation table is unchanged; these measurements are a separate mixed-input diagnostic.

## Reviewed snapshots

| Reference | Previously reviewed | Fetched HEAD |
| --- | --- | --- |
| Strata | `146df3c46873` | `dbb2d168cffa` |
| MoMoE | `4a4a0dca5ab3` | `bb30c69524d1` |
| PolyStrata | `4de79331b49e` | `83a7a3327241` |
| Basalt | `f19f69e68cdb` | `db0be88e91c3` |

Original reference checkouts were preserved; fetched refs and detached worktrees were used for review.

## Changes selected

- [MoMoE image-worker recovery](https://github.com/aflin/momoe/commit/b7d04596a8597b4bb3a93cb5f817e328208a6d5a): a recoverable encoder startup failure leaves text serving available. Image requests receive the failure reason, model discovery reflects current capability, and an explicit load retries the encoder. Initial load, download activation, model switching and rollback share this behavior. HIP SDK/configuration validation still fails closed.
- [Strata READY handling](https://github.com/Niko1221/Strata/commit/4cce1ee4d0057ff718cb0219e0114e5ddf73f296): one reader and total deadline tolerate bounded startup banners, retain useful diagnostics and bound the caller's wait when a dead worker's descendant retains stdout. [Quiet child crashes](https://github.com/Niko1221/Strata/commit/38ad0bcbd107fd92c77376f4715e5d54c3af8e7a) preserve explicit debugger overrides; [HTTP bind](https://github.com/Niko1221/Strata/commit/3e1fcaaf67e04c096b8172682817f427883515a0) avoids reverse DNS.
- [Basalt reader lifetime](https://github.com/jesdga95/basalt/commit/4039751fbd0e3535236fd5989ff9e20480ae3a76): Qwen, Tiel and Clover own and stop/join their stdin reader instead of leaving detached references to serve's stack. Queued STOP epochs, long lines, CRLF and partial EOF remain supported; QUIT exits cleanly.
- [PolyStrata image/text scheduling](https://github.com/simplexai-labs/PolyStrata/commit/f0aab99757ee49bf0d9b95385fa875048f23a90a): fresh image rows retain the original whole-image eval path; long text intervals around them can use the existing layer-major reader. Final logits, absolute positions, checkpoints and cancellation rollback are preserved.

## Gemma mixed-input measurements

RX 9060 XT 16 GB / gfx1200, Ryzen 9 7945HX3D, managed HIP 7.17.26391, verified 64 KiB VMM. The 8/12 GiB cases use real HIP allocations to reserve excess VRAM, not a smaller Expert-cache setting alone. Separate fixed Expert budgets are 1.5 / 3 / 6 GiB; actual matching A/B allocations are recorded in the companion JSON and can be lower under pressure. Context is 32K, chunk 1120, INT8 KV, frozen placement, greedy sampling, MTP/lookup off. Image encoder stays on GPU. Each request is fresh, capped at 96 output tokens; input timings exclude image encoding, while first output includes it. A is old image scheduling; B is final resource-auto scheduling. A single A/B start per capacity is a screening measurement, not a universal speed guarantee.

The 4K/16K label describes the template-counted text-only document before adding the image and final question. Total mixed prompts are about 4.46K / 16.75K tokens, and exactly match between each A/B pair. `image-first` and `image-last` place the same document after or before the image.

| Capacity limit | Workload | Old input tok/s | Auto input tok/s | Change | First output, seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 GiB | image-first-4096 | 927.84 | 934.08 | +0.67% | 4.83 → 4.81 |
| 8 GiB | image-last-4096 | 875.13 | 872.12 | -0.34% | 5.12 → 5.13 |
| 12 GiB | image-first-4096 | 1137.38 | 1133.83 | -0.31% | 3.94 → 3.97 |
| 12 GiB | image-last-4096 | 1073.88 | 1063.50 | -0.97% | 4.19 → 4.21 |
| 16 GiB | image-first-16384 | 1657.94 | 2302.69 | +38.89% | 10.13 → 7.35 |
| 16 GiB | image-last-16384 | 1629.70 | 2232.06 | +36.96% | 10.30 → 7.53 |

Unconditional enablement was rejected: forced mixed-text reading was about 9% slower at 4K and 4–5% slower at 16K in the constrained 8/12 GiB cells. The old path can overlap uploads more effectively there. The final auto policy inspects actual per-layer non-resident Expert bytes against the current staging buffer: only a one-round fit on every MMQ layer enables the new scheduling. Split devices and external drafters conservatively retain the original image path. No GPU-name or nominal-VRAM cutoff is used, and no extra VRAM is reserved for this feature.

The 8/12 GiB final logs show zero new read calls and identical complete answers; mixed-input rates remain within about 1% of their control. The 16 GiB final logs show one long text read and two image/short-text evals instead of 17 evals. A separate reversed-order 16K run also improved 36–38%. Image-only and text-only dispatch are retained. The 29-token text control is functional coverage, not a useful generation-speed benchmark.

Weight/KV formats, all Experts and Attention rows are retained; no tail skipping or approximate draft acceptance is introduced. Different batch boundaries can change floating-point rounding and some greedy wording: bit-identical output is not a general guarantee. All tested color/shape facts remained correct; final auto A/B answers matched. This is focused regression coverage, not a model-quality benchmark.

`TINYSTRATA_GEMMA_IMAGE_TEXT_PREFILL=0` forces the old image route; `1` forces the experimental route even when it is slower; unset or `auto` selects the resource check. `TINYSTRATA_GEMMA_IMAGE_TEXT_TRACE=1` prints mode, staging/missing bytes and read/eval counters to stderr. Eligibility uses current committed metadata; later KV growth can alter residency, so performance at every possible growth boundary is not established.

## Validation scope

CPU discovery: 419 passed / 420, one symlink-privilege skip. Added switch recovery cases: 3 passed. Owned-reader real-child tests: 14 passed, including held-open stdin, initialization failure, cancellation epochs, a 2.4 MB fragmented line and repeated lifetimes. Gemma scheduler: 202,512 schedules / 12,398,153 assertions, including image/mark boundaries and resource-overflow checks. Image startup: 18 cases; HIP integration: 17; stream stability: 7.

Actual model/API recovery, all-six-model functional checks and CUDA compile-only results are recorded with the final adoption receipt. Physical NVIDIA and POSIX execution are outside this machine's validation. An inherited-open stdout can still hold one isolated daemon reader until its descendant closes; the caller has a fixed deadline. Closing a buffered pipe from a different thread is deliberately avoided because it can deadlock.

New Qwen4/FlashNext architectures, NVFP4 artifacts, multi-request batching, Linux/UMA-only policies and output-changing draft policies were not transferred to the current six-model runtime. Basalt's separately tuned Q6_K multi-row kernel remains unadopted without matching backend measurements. Previously measured rejected fusions were not reintroduced.

Raw runs, private source/build receipts, source hashes and protected-profile checks are retained under `TinyStrata-work/upstream-refresh-20261111`. The [companion measurement JSON](upstream-refresh-20261111.json) contains compact timings, actual geometry, dispatch proof and raw-artifact hashes. The [six-model saved-profile table](rx-9060-xt.md) remains the earlier standard observation; mixed-input gains must not be applied to its text-generation rates.
