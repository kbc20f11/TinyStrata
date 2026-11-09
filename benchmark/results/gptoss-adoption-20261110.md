# GPT-OSS HIP adoption — 2026-11-10

Development measurements on **RX 9060 XT 16 GB / gfx1200, Windows HIP**. These are local native-engine screens, separate from `tinystrata-bench-v1` and the PR author's CUDA/SM86 measurements. DFlash is excluded.

## Conditions

The baseline is merged PR14 at `80919ec`, with the HIP build/correctness prerequisites and explicit `--allow-gpt-oss-hip`. Both arms use `ggml-org/gpt-oss-20b-GGUF` **MXFP4**, revision `15dcfc7e7c608c6161507211ed43c5b590801b4b`, file SHA-256 `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`.

Context **32,768**, chunk **512**, INT8 KV, graphs on, MTP off, greedy sampling, seed **1234**, identical penalties and frozen prompt IDs; no image encoder. Each arm starts once, runs an excluded 32-token warm-up, then emits 128 tokens per measured request. The short prompt contains 94 tokens; long inputs contain exactly 4,096/16,384 fresh tokens, with a recorded synthetic middle omission. Conversation caching/checkpoints are off and `reused=0`. Rates use emitted tokens and native elapsed time; proposals do not count as output. This small linear screen does not estimate run-to-run uncertainty.

The uncapped 16 GB runs cache **768/768 Experts, 9,682 MiB** in both arms. The verified managed HIP runtime reports version `71726391` and 64 KiB VMM backing. Every completed pair below exits normally via stdin EOF and preserves the protected configuration hashes.

## MXFP4 prompt kernels and device routing

Enable HIP's MXFP4 MMQ instance and the GPT-OSS device-routing path. This changes the prompt arithmetic kernel and removes host routing handoffs; it is a combined comparison rather than an isolated MMQ microbenchmark. Lookup and bias fusion are off in both arms.

| Workload | Baseline decode, tok/s | MMQ/device routing, tok/s | Change | Baseline input, tok/s | New input, tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 69.33 | 85.17 | +22.86% | 175.77 | 1,350.57 |
| After 4K input | 62.43 | 76.07 | +21.85% | 169.29 | 2,511.34 |
| After 16K input | 49.14 | 56.83 | +15.64% | 164.76 | 1,467.96 |

Cache geometry matches, but generated IDs differ and Expert execution moves from individual matvecs into batched MMQ. The harness therefore does **not** classify this as identical-output/identical-work evidence.

A separate **64-step fixed-history** short comparison evaluates both engines against the baseline's fed tokens. Candidate argmax differs at **2/64** positions, with baseline top-two margins **0.06659 / 0.04972**. Full-vocabulary logits have mean absolute difference **0.09452**, RMS **0.17870**, maximum absolute difference **5.09192**. Mean NLL on the same target tokens is **0.379884 → 0.379091** (delta **−0.000794**). This is numerical evidence against the local baseline, not an independent model-accuracy benchmark or a complete-answer evaluation.

A second fixed-history comparison uses the same **4,096-token prompt / 32 target tokens**, baseline versus the **final fusion-3 bundle**. Argmax agrees at **32/32** positions; mean absolute logit difference is **0.089174**, RMS **0.147354**, maximum **2.598005**. Target mean NLL is **0.2279763 → 0.2284091** (delta **+0.0004329**). Both processes exit normally and protected configuration hashes match. These two bounded numerical samples do not establish general perplexity or full long-context quality; no 16K fixed-history result is claimed.

## Small launch reductions and lookup

Two GPT-OSS bias fusions, selected together by `TINYSTRATA_GPTOSS_FUSE=3`, preserve output IDs and Expert work in the isolated short screens. Fusion 0→1 measures **85.17 → 86.02 tok/s (+0.99%)**; 1→3 measures **85.80 → 86.02 (+0.26%)**. These small single-pair differences should be read as near-flat, not a robust multi-percent gain.

An initial baseline lookup-3 short pair measures **70.34 → 71.75 tok/s (+2.00%)**, identical IDs, with **12/24** proposals accepted. On the MMQ-enabled binary, the following comparison combines **fusion 0 / lookup 0** against **fusion 3 / lookup 7**; it does not isolate lookup alone:

| Workload | Fusion 0 / lookup 0, tok/s | Fusion 3 / lookup 7, tok/s | Change | Accepted / offered |
| --- | ---: | ---: | ---: | ---: |
| Short | 85.23 | 88.08 | +3.34% | 15 / 25 |
| After 4K input | 76.11 | 83.01 | +9.07% | 33 / 45 |
| After 16K input | 56.86 | 58.78 | +3.38% | 26 / 39 |

All three corresponding 128-token outputs and cache geometries match. Rejected proposals increase evaluated Expert work, as expected. Input rates change by **+1.75% / +1.43% / +0.63%**.

A separate temperature-1, top-p-1, full-vocabulary top-k-0 test compares lookup 0/7 on the same candidate, with fusion 3 and the experimental KV pair enabled in both arms. All **five corresponding responses / 640 tokens per arm** match exactly, including prompts at ring positions **637/640/643** and a suffix-rejection case. The four diagnostics accept **76/101** proposals, so **25 rejected proposals** exercise continuation after rejection. The ordinary sampled short request accepts **0/6** and measures **71.51 → 70.43 tok/s (−1.51%)**; the synthetic rejection case changes **−1.93%**, while ring cases range **+1.11% to +9.20%**. These are correctness diagnostics, not a uniform speed gain. Lookup 7 remains an explicit `--lookup-draft 7` option; existing model defaults stay at lookup 0.

The additional paired KV-store experiment is **not adopted and is removed from the source**: short/4K/16K decode changes are **−0.09% / +0.21% / −0.08%**, with identical IDs, drafts, Expert work and geometry. It adds no measured benefit in this screen.

## Lower-capacity automatic placement

These use real HIP allocations on the same 16 GB Radeon to reserve about **3.92 / 7.92 GiB**, producing simulated **12 / 8 GiB capacity** cases. They are not measurements on separate 12/8 GB cards, and are not manual Expert-budget simulations. Both arms set `STRATA_FREEZE_EXPERTS=1`; the native engine confirms the whole-run freeze and allocates the full 32K KV first.

The baseline uses lookup 0. The experimental bundle uses MMQ/device routing, fusion 3, lookup 7 and the subsequently rejected KV-store pair. Automatic placement differs: **610→578 cached Experts at 12 GiB**, **285→252 at 8 GiB** (about 400 MiB less Expert payload with the MMQ workspace). Outputs also differ. These are **same-capacity end-to-end observations**, not isolated matched-placement A/B evidence.

| Simulated capacity / workload | Baseline decode, tok/s | Experimental bundle, tok/s | Baseline input, tok/s | Bundle input, tok/s |
| --- | ---: | ---: | ---: | ---: |
| 12 GiB, short | 53.33 | 54.18 | 183.49 | 468.59 |
| 12 GiB, after 4K | 50.44 | 54.57 | 199.10 | 2,419.66 |
| 8 GiB, short | 37.04 | 39.10 | 130.97 | 235.83 |
| 8 GiB, after 4K | 35.00 | 38.99 | 162.16 | 1,246.42 |

No 16K input is measured at these lower capacities in this draft.

Further **single-start** screens use the final candidate, lookup 7 and frozen Experts. `auto` resolves to **1,120 rows**:

| Capacity / chunk | Cached Experts / MiB | Short decode, tok/s | After 4K decode, tok/s | 4K input, tok/s |
| --- | ---: | ---: | ---: | ---: |
| 8 GiB / 256 | 261 / 3,290 | 38.87 | 39.04 | 944.63 |
| 8 GiB / auto 1,120 | 231 / 2,912 | 36.70 | 37.93 | 1,601.13 |
| 12 GiB / 256 | 587 / 7,400 | 55.04 | 55.11 | 2,161.14 |
| Uncapped / auto 1,120 | 768 / 9,682 | 87.37 | 82.18 | 2,815.12 |

At 8 GiB, chunk 256 saves enough scratch to cache **30 more Experts / 378 MiB** than auto. Decode is **+5.92% short / +2.91% after 4K**, but 4K input is **41.00% slower**: native prompt-plus-decode time for the 128-token response increases **5.933 → 7.615 seconds (+28.36%)**. Output IDs/draft acceptance differ and there is no repeat, so the decode difference is not established beyond run/workload variation. Against the earlier 512-row bundle, 256 gives **−0.58% / +0.13%** decode and **−24.21%** 4K input. At 12 GiB, it caches nine more Experts than the earlier 512-row bundle, but decode changes only **+1.59% / +0.99%** while 4K input falls **10.68%**. No 12 GiB auto-1,120 measurement was taken. The uncapped 4K prompt IDs differ from the earlier 512-row screen, so those rates are not an exact paired comparison. **Keep auto 1,120 unchanged**; chunk 256 does not justify a general policy change.

## Completed validation

- **State/window:** INT8, frozen partial cache (324/768 Experts), SWA window 128 and ring 192. Unfused graph-on passes **312 checks** at positions 127/128/191/192/193/384. Fusion 3 with the experimental KV pair enabled passes **312 checks with graphs off** and **52 with graphs on at position 191**. The final product removes that KV-pair experiment and retains the original separate KV stores. Saved-state restoration after an unrelated sequence, accepted prefixes of 1/4/8 rows, and continuation after rejected speculative suffixes match each tested configuration's serial logits bitwise.
- **Bias fusions:** **312 GPU cases pass**, including bitwise output, graph replay, finite values and guard storage. The paired KV experiment remains rejected despite matching outputs.
- **CPU sampler:** **302,594 checks pass**, covering full-vocabulary sampling beyond 64 IDs, signed logits, penalties/history, large offsets, tiny temperature, non-finite rejection and retained greedy/truncated compatibility.
- **CUDA compile-only:** final frozen source passes **five Release checks** with patched CUDA **13.2.86**: MSVC `engine.cpp`, `gptoss.cpp`, `main.cpp`, and `kernels.cu` for **SM86 / SM120**. Source hashes and cached-build outputs remain unchanged. This is not CUDA linking or NVIDIA runtime/performance validation.

## Policy and remaining checks

| Item | Decision |
| --- | --- |
| HIP MXFP4 MMQ and GPT-OSS device routing | Enable |
| GPT-OSS bias fusion 3 | Enable |
| Paired KV store | Reject; remove |
| Lookup 7 | Explicit opt-in; retain existing lookup-0 defaults |
| Automatic chunk | Retain 1,120; no new low-capacity chunk policy |

## Final API benchmark and integration

The final executable SHA-256 is `94ea53b2c054d42083bccb36a84e76a0eaa5a89fa08bf2dee5f981f8d1c0c094`. One fresh API start completes **`tinystrata-bench-v1-gptoss`**, a separately labelled extension with `standard: false`: three canonical generation texts × two passes at 128 output tokens, exact fresh 4K/16K inputs at 32 output tokens, and an excluded 32-token warm-up. All **eight measured requests / 832 emitted tokens** have zero prefix reuse and finish normally; unload succeeds without forced cleanup.

This run uses normal temperature **1**, top-p **1**, unrestricted top-k **0**, medium reasoning, INT8 growing KV, MTP/lookup off, no images, automatic **128K context / 1,120-row chunk**, and **768/768 Experts**. Expert payload stays **9.46 → 9.46 GiB** from startup through 16K; mapped KV grows to **20,480 rows / 255 MiB**, separate from logical context capacity.

| Final API profile | Generation, tok/s | 4K input, tok/s | 16K input, tok/s |
| --- | ---: | ---: | ---: |
| GPT-OSS 20B MXFP4 | **71.53** | **2,806.25** | **1,568.39** |

This is one final-profile rate observation, not a repeatability estimate or the greedy 32K/512-row A/B comparison above. The runner/cases hashes and measured asset identities are recorded in `final-benchmark/result.json`; its source HEAD field names base `80919ec` because the tested patch was not yet committed at measurement time.

Actual API integration passes **50/50 checks**: low/medium/high reasoning returns separated analysis/final text, named and required streaming tool calls preserve names/arguments/IDs and accept result continuations, tool-choice-none returns text, lookup 7 reaches native, full-vocabulary top-k 0 reaches native, and image input is rejected for this text-only model. Protected task configuration remains unchanged. Playwright verification covers light/dark desktop and mobile layouts, one private UI chat answering **2+3 → 5** with separate reasoning, temperature/top-p/top-k **1/1/0** and medium defaults, the plain **High** label, an installed-only picker, all six model icons including the official OpenAI logo, and active-GPT-OSS deletion being blocked.

The implementation is validated within these recorded conditions. Publication and installed deployment are separate actions; neither is claimed here.

Because GPT-OSS shares the Gemma executable, a bounded old/new Gemma regression compares prior `a5cd1e1a…` with final `94ea53b2…`, using the same Vision payload and runtime core/math identity. Both return exactly **`56` (three emitted tokens)** for the text check and the same **53-token** image description; image draft counts match at **37 accepted / 48 offered**. All four requests have zero reuse, both APIs unload normally, and installed profile hashes remain unchanged. These short checks make no performance claim. The first private Vision-bundle staging attempt was refused by the existing runtime guard; staging the same verified payload into the proper test bundle allowed the retry. This was a test-fixture issue, not a product failure.

Raw commands, binary/source identities, all output IDs, full forced logits, timings, cache geometry and normal-cleanup records are retained under `TinyStrata-work/gptoss-adopt-20261110`: the `screen-*` directories, `forced-mxfp4-short`, `forced-mxfp4-4k`, `checks`, `private-checks`, the state-test logs, `builds/cuda-final/build.json`, `final-benchmark`, `api-integration-01` and `gemma-regression`. Early native INFO incorrectly names Gemma while loading GPT-OSS; later binaries correct the label. The first forced-logit postprocess lacked NumPy; the existing GPU outputs were processed with a standard-library fallback, without rerunning inference.
