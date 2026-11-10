# GPT-OSS EAGLE-3 and verification Expert reuse — 2026-11-10

On **RX 9060 XT 16 GB / gfx1200**, a lightweight one-layer EAGLE-3 drafter improves some capacity-limited decode workloads. The strongest short screen is **47.35 → 66.78 tok/s (+41.0%)** at simulated 8 GiB. It also adds prompt work and occupies VRAM: **16K input is slower overall in every tested capacity**, and full-VRAM 16K decode also regresses. **Autoregressive D0 remains the default; EAGLE-3 D1/D2 is a native opt-in.**

These are single-pass native development screens. They do not replace the canonical API benchmark or estimate run-to-run uncertainty. [Compact measurements and checks](gptoss-eagle3-20261110.json) retain rates, latency, acceptance, Expert placement, prompt/output hashes and model identity.

## Conditions

The measured candidate02 executable has SHA-256 `9e266cdc2152335d692554158a175057127130f67225ad31b1e7697e2790f4da`. Target: the pinned **12.110 GB GPT-OSS 20B MXFP4** GGUF.[^target] Drafter: **Nebius EAGLE-3 GPT-OSS 20B**, one layer, roughly 350M parameters, using the **336,157,216-byte mixed-quantization GGUF** (SHA-256 `b48723b81f44317264341df1ac7b95721dde4be88a1bba71ef9387f244c013c5`).[^draft]

All arms use **32,768 context, chunk 1,120, INT8 growing KV, graphs on, automatic unfrozen Expert placement**, MTP/lookup off and conversation caching off. HIP runtime and compiled versions are both `71726391`. Each arm loads once, excludes a **32-token warm-up**, then measures the same **94 / 4,096 / 16,384 input IDs**, with **128 output tokens in every row**, medium reasoning, no images and `reused=0`. Sampling is **temperature 1, top-p 1, top-k 0, min-p 0, seed 1234**, repetition penalty 1 and presence/frequency penalties 0.

The 16 GB arm is uncapped on the physical card. The **12/8 GiB** arms reserve approximately **3.92/7.92 GiB through real HIP allocations on the same card**; they do not emulate another card's bandwidth or compute capability. Expert placement is allowed to respond to the drafter and growing KV. For example, cached Experts after short input are **553 / 522 / 522** for 12 GiB D0/D1/D2 and **275 / 247 / 243** for 8 GiB; the target has 768 Experts. All 768 remain cached in the uncapped arms.

## Decode

Rates are emitted tokens divided by native decode time. Parenthesized changes compare with D0 in the same capacity/workload. **D0 does not load a drafter**; D1/D2 propose up to one/two tokens per verification window.

| Capacity | Workload | D0, tok/s | D1, tok/s | D2, tok/s |
| ---: | --- | ---: | ---: | ---: |
| 16 | Short (94) | 91.60 | 91.04 (-0.6%) | 94.33 (+3.0%) |
| 16 | After 4K | 81.55 | 85.74 (+5.1%) | 83.08 (+1.9%) |
| 16 | After 16K | 70.65 | 66.96 (-5.2%) | 64.56 (-8.6%) |
| 12 | Short (94) | 66.94 | 66.30 (-1.0%) | 72.63 (+8.5%) |
| 12 | After 4K | 63.71 | 66.86 (+4.9%) | 67.70 (+6.3%) |
| 12 | After 16K | 57.72 | 59.68 (+3.4%) | 56.95 (-1.3%) |
| 8 | Short (94) | 47.35 | 57.52 (+21.5%) | 66.78 (+41.0%) |
| 8 | After 4K | 43.77 | 55.21 (+26.1%) | 54.77 (+25.1%) |
| 8 | After 16K | 41.24 | 52.21 (+26.6%) | 42.17 (+2.3%) |

At 8 GiB, D1 after 16K is faster than D2 (**52.21 versus 42.17 tok/s**). D2's accepted/offered counts fall from **77/101** on short input to **60/136** after 16K; its larger verification work does not pay back uniformly. At 12 GiB, short D2 gains **8.5%**. At 16 GB, short D2 gains only **3.0%**, D1 wins the 4K decode screen, and neither depth wins after 16K. These observations support keeping the choice optional rather than selecting one depth for every workload.

## Prompt processing and total latency

EAGLE-3 captures three target hidden states, projects teacher context into draft KV and holds additional weights/KV. The current external-drafter path also uses ordinary prompt eval rather than the multi-call staged-read optimization. These costs are included below. In the lower-capacity arms, less target Expert storage remains available. Prompt rates therefore fall even where decode accelerates.

| Capacity | Input | D0 input, tok/s | D1 input, tok/s | D2 input, tok/s |
| ---: | --- | ---: | ---: | ---: |
| 16 | Short (94) | 1426.40 | 1358.38 | 1366.28 |
| 16 | After 4K | 3849.62 | 3583.55 | 3588.58 |
| 16 | After 16K | 2632.81 | 2512.77 | 2511.42 |
| 12 | Short (94) | 778.79 | 670.47 | 673.35 |
| 12 | After 4K | 3621.89 | 2849.19 | 2700.60 |
| 12 | After 16K | 2574.93 | 2237.79 | 2250.80 |
| 8 | Short (94) | 395.62 | 346.35 | 342.57 |
| 8 | After 4K | 2857.54 | 2022.72 | 2019.13 |
| 8 | After 16K | 2090.70 | 1748.75 | 1724.87 |

The following totals are **native prompt time + decode time** for 128 emitted tokens; they exclude model loading and client/protocol overhead. They make the input/decode tradeoff visible. In particular, the 8 GiB 16K D1 decode gain does not overcome its slower prompt processing.

| Capacity | Workload | D0 total, s | D1 total, s | D2 total, s |
| ---: | --- | ---: | ---: | ---: |
| 16 | Short (94) | 1.46 | 1.48 | 1.43 |
| 16 | After 4K | 2.63 | 2.64 | 2.68 |
| 16 | After 16K | 8.03 | 8.43 | 8.51 |
| 12 | Short (94) | 2.03 | 2.07 | 1.90 |
| 12 | After 4K | 3.14 | 3.35 | 3.41 |
| 12 | After 16K | 8.58 | 9.47 | 9.53 |
| 8 | Short (94) | 2.94 | 2.50 | 2.19 |
| 8 | After 4K | 4.36 | 4.34 | 4.37 |
| 8 | After 16K | 10.94 | 11.82 | 12.53 |

## Verification Expert reuse

The target verification kernel groups repeated cached MXFP4 Experts across the two/three rows of a window so their stored weights can be reused. It preserves the existing weight/activation codes, Expert selection and per-row accumulation result. A four-column group is selected for **single-GPU HIP gfx1200**; other GPUs retain their existing default path. `TINYSTRATA_GPTOSS_GROUPED_MXFP4=0` selects the previous implementation for comparison.

Fixed-work tests freeze Expert membership, use graphs and repeat 16 windows after setup. Their full-vocabulary logits match serial evaluation **bitwise**, with matching output hashes and Expert work between grouped-off/on runs. At full placement, a three-row target window falls from **20.362 → 18.173 ms (10.8% less time)**; a two-row window changes **15.481 → 15.361 ms**. With 324/768 cached Experts, three-row time changes **38.193 → 37.732 ms**. This is target-window time, not end-to-end generation speed.

A separate full-VRAM D2 grouped-off/on screen produces identical output IDs and accepted/offered counts. Decode changes **93.67 → 94.33 tok/s** on short input, **82.34 → 83.08** after 4K and **63.66 → 64.56** after 16K: roughly **0.7–1.4%**, far smaller than the isolated target-window gain. The drafter, sampling, commit work and other target kernels remain part of each generation round.

## Distribution and state checks

The enabled EAGLE-3 path defaults to **exact p/q rejection with residual sampling** for eligible positive finite-temperature, neutral full-vocabulary sampling. Its 64K draft vocabulary is mapped into the target's full **201,088 IDs**; IDs outside draft support remain available through the target residual distribution. Quantizing the drafter changes proposal efficiency, not the target acceptance rule. Approximate acceptance and Expert skipping are not used.

Exact rejection preserves the target sampling distribution; it does **not** promise the same output IDs at the same seed because speculative acceptance/residual draws consume RNG differently. Automatic unfrozen placement and the different sampled continuations can also change Expert work. Consequently, these stochastic rates are workload screens, not fixed-output kernel comparisons or a model-quality score. With temperature 0, two 128-token prompts (short and 4K) produce **identical complete output ID sequences for D0/D1/D2**.

Validation includes:

- **63 independent model checks:** actual GGUF shapes/taps, exact tokenizer match, full absolute vocabulary-map conversion and official HF Q/K weight samples
- **43 BF16 native-versus-independent NumPy stages**, with both proposal argmaxes matching; maximum final-logit error is **3.05e-5 / 4.96e-5** across D1/D2 recurrence
- **1,996,126 sparse-vocabulary probability/scatter checks** plus **221,364 exact rejection-sampler checks**, including zero draft support, CPU p/q reference, changed-input graph replay and guards
- **146 grouped-kernel cases / 47,179 checks** for the selected four-column layout, including repeated Experts, slots, graphs and exact baseline comparison
- Model-backed EAGLE-3 target/draft state checks: **700** for BF16 graph-on at position 17, **718** for quantized graph-off at position 193 and **1,463** for quantized graph-on with 16K context at position 8,191; these exercise proposal immutability, accepted/rejected prefixes, teacher rebuilding and external-state invalidation

The auxiliary tap indices are HF hidden-state/layer-input indices **[2,12,21]**, captured after target blocks **[1,11,20]**. GGUF Q/K rotary-row permutation is reversed without requantizing weights; fractional YaRN correction bounds, deferred teacher-token pairing and prenorm autoregressive recurrence follow the pinned checkpoint and reference implementations.[^implementation]

## Native use and scope

Start the native GPT-OSS executable with `--eagle-model GGUF --eagle 1` or `--eagle 2`. **`--eagle 0` remains the default.** When enabled, the default acceptance mode is `rejection`; `--eagle-accept rejection` makes that choice explicit. Greedy requests, filtered sampling and active penalties fall back to sequential target-draw verification. `--eagle-accept target` explicitly selects that diagnostic comparison mode. The same native arguments may be appended to a manual GPT-OSS API profile; profile validation and per-request admission reject incompatible drafters. Automatic model download, the managed API/UI defaults and the canonical benchmark table remain unchanged.

The grouped kernel defaults on only for verified single-GPU HIP gfx1200, with four columns. CUDA and other GPU families retain their previous default; these measurements establish no NVIDIA runtime or performance result. The measured candidate02 had explicit grouped/rejection settings. The separate final candidate04 executable has SHA-256 `14faf0bb2a8dee8b2142bec5b420f20695440b7c83989e927736fefe67dd9a88`: it changes those defaults, removes the earlier DFlash prototype and renames shared helpers without changing the measured EAGLE-3 equations. It is not the executable used for the tables above. The final release repeats the short screen at **D0 91.56 / D2 94.56 tok/s**, with all output IDs matching the measured candidate. Its default exact rejection is verified by real-model nonstreaming and streaming API requests; all seven integration checks, 80 CPU configuration/admission regressions, probability/kernel units and the 718-check quantized graph-state fixture pass. The API and model exit normally after the smoke test.

Raw evidence remains under `TinyStrata-work/gptoss-verify-eagle-20261110`: `screen-eagle-{full,capacity,greedy}-candidate02`, `screen-eagle-groupoff-candidate02b`, `grouped/*candidate02`, `unit`, `state`, `validation/eagle3-model-validation.json` and `validation/oracle-bf16-candidate01/result.json`. All measured workers and capacity holders exit normally, and protected configuration hashes remain unchanged. These native and API checks do not claim installer deployment.

[^target]: [Pinned target GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/blob/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf).
[^draft]: [Official Nebius checkpoint and CC-BY-4.0 model card](https://huggingface.co/nebius/EAGLE3-gpt-oss-20b/tree/9dd45bb1da8b1ddc8cabd52691d9ac170b41484f); [pinned GGUF conversion and license](https://huggingface.co/EntityDeletr/EAGLE3-gpt-oss-20b-GGUF/tree/59583c672a860e9d21ed87fff1cf1cf7c4351cd8). See the repository NOTICE for attribution.
[^implementation]: [Official vLLM EAGLE-3 Llama implementation](https://github.com/vllm-project/vllm/blob/3a104103ac70740824599453c7036edcca1e8548/vllm/model_executor/models/llama_eagle3.py), [GPT-OSS auxiliary-state capture](https://github.com/vllm-project/vllm/blob/3a104103ac70740824599453c7036edcca1e8548/vllm/model_executor/models/gpt_oss.py), and [pinned model config](https://huggingface.co/nebius/EAGLE3-gpt-oss-20b/blob/9dd45bb1da8b1ddc8cabd52691d9ac170b41484f/config.json).
