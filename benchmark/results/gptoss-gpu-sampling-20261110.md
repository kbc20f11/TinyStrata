# GPT-OSS GPU categorical sampling — 2026-11-10

Local native-engine measurements on **RX 9060 XT 16 GB / gfx1200, Windows HIP**. Moving eligible full-vocabulary sampling to the GPU removes the full-logit host transfer and CPU categorical work. These results extend the [GPT-OSS adoption measurements](gptoss-adoption-20261110.md); they are separate from the final API benchmark and CUDA validation. DFlash is excluded.

## Conditions

Both arms use the **same candidate01 executable**, SHA-256 `501123edb0bbb336e3855f0590ba0c7c374040135acc2e26de866992e0d911c5`, built from base `01c1d14` plus the sampler changes. A explicitly sets `TINYSTRATA_GPTOSS_GPU_SAMPLE=0`; B sets it to `1`. Model: GPT-OSS 20B MXFP4, the same verified GGUF used in the adoption report. HIP runtime/compiled version is `71726391`.

Sampling is **temperature 1, top-p 1, top-k 0, min-p 0, seed 1234**, repetition penalty 1 and presence/frequency penalties 0. Context is **32,768**, chunk **1,120**, INT8 KV, graphs on, MTP/lookup off unless stated otherwise, and no images. Each arm loads once and runs an excluded 32-token warm-up. **Every measured native response below emits 64 tokens**, including those after 4K/16K input. The short prompt has 94 tokens; long inputs have exactly 4,096/16,384 frozen IDs, with the harness's recorded synthetic middle omission. These output lengths differ from the separate API protocol's 32-token long-input responses.

Conversation caching/checkpoints are off and every request reports `reused=0`. `STRATA_FREEZE_EXPERTS=1` fixes Expert membership and reserves the complete **32K / 408 MiB KV** before placement; KV growth is off in these comparisons. All corresponding output IDs, finish reasons, draft counters, Expert work and cache geometries match. Rates are emitted tokens divided by native `DONE` decode time. These are single A/B screens, with a separate initial short recheck; they do not estimate run-to-run uncertainty. Input rates are retained in the raw data; no prefill speed gain is claimed.

## Decode results

The uncapped cell uses the physical 16 GB card. The 12/8 GiB cells reserve approximately **3.92/7.92 GiB with real HIP allocations on that same card**; they are capacity simulations, not separate 12/8 GB GPUs or manual Expert-budget limits. Automatic Expert placement is identical between arms, including the small sampler workspace allocated before Expert-cache sizing.

| Capacity / workload | Cached Experts / MiB | CPU sampler, tok/s | GPU sampler, tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Uncapped 16 GB, short | 768 / 9,682 | 72.39 | 90.13 | +24.50% |
| Uncapped 16 GB, after 4K | 768 / 9,682 | 65.13 | 79.67 | +22.33% |
| Uncapped 16 GB, after 16K | 768 / 9,682 | 50.54 | 59.08 | +16.91% |
| Simulated 12 GiB, short | 570 / 7,186 | 43.38 | 51.19 | +18.01% |
| Simulated 12 GiB, after 4K | 570 / 7,186 | 42.46 | 49.23 | +15.96% |
| Simulated 8 GiB, short | 244 / 3,076 | 32.35 | 36.52 | +12.87% |
| Simulated 8 GiB, after 4K | 244 / 3,076 | 30.75 | 34.32 | +11.62% |

The preceding short-only screen independently measured **70.87 → 90.22 tok/s (+27.30%)**, also with identical IDs/work/geometry. No 16K lower-capacity measurement is claimed.

Per-request `INFO` confirms that B samples **all 64 target tokens on GPU**, with CPU/fallback counts zero and **`logits_host_rows=0`, `logits_host_bytes=0`**. A samples all 64 on CPU and transfers **51,478,528 bytes**, exactly 64 × 201,088 float32 logits. The GPU path returns a small sampling result instead of the vocabulary vector.

## Probability, RNG and fallback checks

The eligible path computes categorical weights over **all 201,088 IDs**, with FP64 max-shifted exponentials and reductions. It adds no top-k truncation, manual probability cutoff, pruning or tail skip. The GPU receives a CPU-generated uniform; a successful device draw commits exactly that RNG advance. Failure retains the original RNG for the CPU path. Parallel reduction and exponential rounding can still affect a draw extremely close to a CDF boundary.

The GPU unit passes full-mass/equal-logit cases, endpoint uniforms, tiny temperatures, retained finite small tails, non-finite/invalid-input rejection, graph capture/replay and guard-storage checks. Reused real model logits cover **64 short rows × 16 draws** and **32 final-4K rows × 32 draws**, for **2,048 draws**: **zero token mismatches, zero error statuses and identical post-draw RNG states** against this MSVC STL's CPU `std::discrete_distribution`. CPU-CDF interval records are saved for every draw. This is bounded numerical/RNG evidence, not an independent model-quality benchmark.

One CPU-mode worker and one GPU-mode worker each run the same eight **32-token** requests without reloading: neutral full categorical → top-k 64 → top-p 0.9 → min-p 0.05 → greedy → repetition penalty 1.05 → presence penalty 0.1 → neutral again. All **256 output IDs per arm**, Expert work and geometry match. B uses GPU only for the two neutral requests; the six other requests report the expected CPU fallback reason and 32 CPU draws each. The final neutral request returns to GPU with zero host-logit copies and reproduces the initial neutral response exactly. Profiles, settings and history hashes remain unchanged.

## Lookup and SWA ring diagnostics

Both arms use lookup **7**, chunk **512**, the same temperature-1 full categorical settings and fixed Experts. Ring length is **640** (SWA window 128 plus chunk 512). All five **64-token** responses and all accepted/offered counts match; B uses GPU for every target draw, including verification rows, without host-logit copies. This compares CPU versus GPU sampling with lookup enabled in both arms; it does not measure enabling lookup itself.

| Request | CPU sampler, tok/s | GPU sampler, tok/s | Change | Accepted / offered, each arm |
| --- | ---: | ---: | ---: | ---: |
| Short | 69.19 | 86.35 | +24.80% | 0 / 6 |
| Ring position 637 | 76.43 | 99.32 | +29.95% | 17 / 21 |
| Ring position 640 | 73.51 | 96.08 | +30.70% | 15 / 21 |
| Ring position 643 | 76.71 | 100.08 | +30.46% | 16 / 18 |
| Synthetic suffix rejection | 70.61 | 88.13 | +24.81% | 2 / 6 |

The four synthetic diagnostics accept **50/66 proposals per arm**, exercising **16 rejected proposals**. Together with the ordinary short request, totals are **50/72**, including 22 rejections. These bounded continuation checks do not replace the separate saved-state/window test.

## Selected policy and remaining validation

The selected policy is **GPU sampling on by default for eligible single-GPU GPT-OSS requests**. `TINYSTRATA_GPTOSS_GPU_SAMPLE=0` remains the CPU diagnostic override. Eligibility requires positive temperature, top-k 0, top-p 1, min-p 0 and no active penalties. Filters, greedy requests and active penalties retain CPU sampling. Other model architectures and split/multi-GPU engines retain their existing path. Failure to allocate the optional sampler workspace is handled by CPU fallback; explicit allocation-failure injection is not part of these screens. The final default-on build is separate from the explicit-mode candidate01 measurements above.

## State retention and final validation

Model-backed checks exposed a **pre-existing SWA sequence-binding bug**: allocating a ring with `dmalloc` did not register it as sequence state, so rebinding a sequence could lose its window history. Registering those rings through `seq_alloc` restores them with the full-attention state. This is a correctness fix separate from probability computation.

The final default-on HIP CMake build and **2,799 sampler-unit checks** pass. With a frozen **4 GiB Expert budget / 324 of 768 cached Experts**, model-backed device-sampling tests pass **328 checks each** for graph-off INT8, graph-on INT8 and graph-on FP16. Each covers 58 device draws and 24 lazy host-logit rows; two independent sequence-step oracles now match bitwise. The separate state/window suite passes **312 checks** at positions **127/128/191/192/193/384**, including saved-state restoration and accepted/rejected window prefixes.

CUDA **13.2.86** compile-only verification passes **seven Release checks**: MSVC `engine.cpp`, `gptoss.cpp`, `main.cpp`, plus `kernels.cu` and `sampling.cu` for **SM86/SM120**. The changed `gptoss.cpp` passes again after the state fix. Python tests pass **381 with one skip**. These CUDA checks do not establish NVIDIA linking, runtime correctness or performance.

The initial automatic-cache/MTP-3 shared-Gemma comparison produces different image wording and is retained in `gemma-regression`, so it is not used as fixed-placement bitwise regression evidence. A bounded comparison with **6 GiB fixed Expert budget, frozen membership, 4K context and MTP 0** passes in `gemma-regression-controlled`: both executables return exactly `56` (three tokens) and the same 53-token image description, use the same Vision/runtime payloads, preserve configuration hashes and unload normally. This establishes neither general MTP equivalence nor a speed gain.

## Final-profile API result

The tested executable includes the SWA state-retention fix; SHA-256 is `1c64a2674f11028cea9a2c0dff1c5c9c303c25cb353fe477561f2befbce4b62e`. One fresh private API start completes **`tinystrata-bench-v1-gptoss`** (`standard: false`) with normal temperature **1**, top-p **1**, top-k **0**, medium reasoning, INT8 growing KV, MTP/lookup off and no images. Automatic context/chunk resolve to **128K / 1,120**. It supplies **no GPU-sampler environment override**.

The unchanged protocol runs three canonical generation prompts twice at 128 tokens and fresh 4K/16K inputs at 32 tokens; the 32-token warm-up is excluded. All **eight measured requests / 832 emitted tokens** have zero reuse and report default GPU dispatch: **832 GPU draws, 0 CPU draws, 0 fallback draws and 0 host-logit rows/bytes**. Expert cache stays **768/768 / 9.46 GiB**. The full 128K KV is mapped at startup and remains **1,632 MiB** after 16K; this is distinct from logical context capacity and the fixed-32K native screens.

| Final API profile | Generation, tok/s | 4K input, tok/s | 16K input, tok/s | Default dispatch |
| --- | ---: | ---: | ---: | --- |
| GPT-OSS 20B MXFP4 | **89.30** | **2,800.49** | **1,574.61** | GPU; host logits 0 |

Normal API unload completes without forced cleanup, and the launcher confirms unchanged protected profile/settings/history hashes. Profile, temporary/cache/log directories and Expert-profile writes are private. A private metrics-only shim retains the native sampler INFO fields that the unchanged benchmark snapshot otherwise omits; it changes no case, cap, sampler, lifecycle or rate calculation. Runner/shim/native hashes are recorded separately in `final-api-fixed.launcher.json`.

This is one final-profile rate observation, not a controlled comparison with the earlier 71.53 tok/s API result or a repeatability estimate. The matched native screens above remain the before/after evidence. The report names base source HEAD `01c1d14` because the sampler patch was uncommitted at measurement time.

Installed deployment and publication are separate actions and are not claimed here. NVIDIA linking/runtime/performance remain unvalidated.

Raw evidence is retained under `TinyStrata-work/gptoss-gpu-sampler-20261110`: `screen-short`, `screen-long`, `screen-12`, `screen-8`, `screen-lookup7`, `screen-mixed-fallback`, `sampling-real-short.jsonl`, `sampling-real-4k.jsonl`, `real-logits-manifest.json`, candidate01/final build logs, `final-native-checks.json`, `builds/cuda-final`, `builds/cuda-fixed`, `final-api-fixed` and `final-api-fixed.launcher.json`. The initial pre-state-fix `final-api` and both Gemma regression records are retained separately. Native comparisons report `complete=true` and `all_comparisons_matched=true`; mixed fallback reports `checks_passed=true`. Every worker exits normally through stdin EOF, capacity reservations are released normally, and protected configuration hashes match.
