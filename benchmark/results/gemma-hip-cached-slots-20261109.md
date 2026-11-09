# Gemma HIP cached-Expert batching — 2026-11-09

## Change

Gemma's hybrid path now submits up to **32 cached token/Expert pairs** per launch on HIP, compared with eight before. With MTP 3, four target rows can select eight Experts each. Gate/up, GEGLU, Q8 activation quantization and down projection are still independent per pair, and the final weighted sum retains its existing order. No routed Experts, weights, sampling rules or draft verification are removed or approximated.

References: PolyStrata [43bbf3f1](https://github.com/VecSzn/PolyStrata/commit/43bbf3f1ffbaa1cb294f8ebd62edd287a3130add) and [5c0bd2c9](https://github.com/VecSzn/PolyStrata/commit/5c0bd2c991de7964b2cd4c9495e1be3346976f2c), MIT. Attribution is retained in `NOTICE`. The donor's separate warp kernel is not included in this comparison.

HIP's slot capacity is 32; CUDA keeps its eight-slot structure and default. Scratch below 32 rows retains eight-pair scheduling, or one pair below eight rows. Only the activation quantizer's independent-row limit is expanded; ordinary multi-column MMVQ remains limited to eight columns. No scratch allocation is enlarged.

`TINYSTRATA_GEMMA_SLOTS32=0` selects eight-pair scheduling on HIP for comparison or recovery. Unset/`1` uses the new default. This is a native diagnostic environment variable, not a new chat setting.

## Matched-compute development comparison

Windows 11 build 26300; **RX 9060 XT 16 GB / gfx1200**, Ryzen 9 7945HX3D, 64 GB RAM, AMD driver 32.0.32015.7029. The display was connected to the Radeon at boot. The existing private HIP 7.17 SDK/runtime and verified 64 KiB Expert backing were retained; no driver, clock or power setting was changed.

Both arms were rebuilt from TinyStrata baseline `79da37c` with the same SDK/compiler flags. The candidate changes only cached batching and its row-count validation. They use the same Google Gemma 4 26B-A4B QAT Q4_0 GGUF, QAT MTP assistant, INT8 KV, MTP 3, context 32,768, chunk 1,120 and manual **6 GiB Expert budget**. Each start has **1,925 / 3,840 Experts**, 6,142 MiB of cached weight payload. This manual Expert budget is not a simulation of a physical 8/12 GB card. Vision is absent from this native text comparison.

Private benchmark entry points disable Expert swaps, and `STRATA_FREEZE_EXPERTS=1` fixes cache membership and whole-window execution. Its existing policy allocates the whole 32K KV instead of growing it. The usual timed whole/halves selector is therefore excluded from this controlled comparison. An earlier swaps-only screen still varied that selector, even between unchanged baseline starts, and was not accepted as a matched-compute comparison.

Four fresh starts run **A1 / B1 / B2 / A2**. Each has one excluded 32-token warm-up, then the same frozen short, exact 4,096-token and exact 16,384-token inputs, with 128 output tokens each. Long inputs preserve the rendered chat's start/end with a recorded synthetic middle omission. Greedy sampling, seed 1234 and generation penalties match; checkpoints and prefix reuse are disabled. Rates pool actual emitted tokens and native decode milliseconds; speculative proposals do not count as output.

| Workload | Eight pairs, tok/s | 32 pairs, tok/s | Change |
| --- | ---: | ---: | ---: |
| Short generation | 67.55 | 70.25 | +4.00% |
| Generation after 4K input | 65.44 | 65.62 | +0.27% |
| Generation after 16K input | 62.71 | 63.58 | +1.39% |

All **12 measured responses / 1,536 output tokens** match token-for-token across the corresponding runs. Draft offered/accepted counts, GPU/CPU Expert execution counts, initial/final allocation and KV geometry also match. Every process exits normally. Prefill changes are **-0.14% at 4K / +0.10% at 16K**, effectively unchanged in this bounded comparison.

Per-start short generation: A **69.30 / 65.88**, B **70.51 / 69.99 tok/s**. After 4K: A **65.66 / 65.23**, B **65.57 / 65.67**. After 16K: A **62.65 / 62.76**, B **63.42 / 63.74**. These are small-sample engineering results, not a confidence interval or universal gain.

## Normal saved-profile integration

A separate **one-start-per-arm** `tinystrata-bench-v1-saved-profile` diagnostic retains automatic placement/context/chunk, KV growth, MTP 3, temperature 1.0 and the GPU-resident image encoder. Both runs complete generation, fresh exact 4K/16K inputs, the image request and graceful unload. Both describe the red square, blue circle and green triangle correctly. Saved application configurations remain unchanged.

Generation measures **74.64 -> 79.02 tok/s (+5.86%)**, but startup placement differs: **2,642 -> 2,784 cached Experts**. After 16K, cached payload is **8,255 / 8,191 MiB**. Do not attribute the entire observed difference to batching. The matched-placement table above is the basis for the performance decision. Normal-profile prefill measures **2,572.06 -> 2,573.35 tok/s at 4K** and **2,375.73 -> 2,398.13 at 16K**.

## Verification and scope

- HIP GPU regression passes **19 cases**: actual 2816 -> 1408 -> GEGLU -> 704 -> 2816 dimensions, 1/7/8/9/17/31/32 pairs, small/tail shapes, individual versus 8/32 batching, Q8 bytes, ordered sums, finite values, poisoned unused storage and guard bytes. Per-stage outputs match bitwise. Independent scalar stored-value reference maximum error is **1.1920929e-7**; fixture hash `778ac2dafd8b89ad`.
- Opt-in CMake target: `gemma_cached_slots_test` (excluded from normal user builds).
- CUDA compile-only checks pass for changed host code and **SM86 / SM120** matvecs. A host assertion confirms CUDA capacity eight and a 192-byte slots payload. NVIDIA runtime performance was not measured.
- Full code, commands, source/binary/runtime identities, failed test-fixture attempts, raw token IDs and timing logs are retained under `TinyStrata-work/gemma-slots32-20261109`. GPU test failures during development came from the test oracle/guard initialization and were corrected before the matched comparison; production arithmetic was not changed to satisfy the test.

**Decision:** enable 32-pair batching in the Gemma HIP hybrid path, retaining the eight-pair diagnostic fallback and unchanged CUDA default. Gains were measured on this Radeon only. LFM/Qwen/Clover paths are unchanged. These development measurements do not replace historical standard-v1 tables or claim equivalent speedups on every GPU/VRAM configuration.

## 日本語

GemmaのHIP経路で、MTP中にGPUへ渡す独立したtoken／Expertの組を8組から最大32組へまとめます。実際の重み・Expert数・演算・投機採否・加算順序は維持し、CUDAは8組のままです。

RX 9060 XT上でExpert配置と内部の計算分割を固定し、旧／新／新／旧の4起動で比較しました。短文生成は **67.55 -> 70.25 tok/s（+4.00%）**、4K入力後は **65.44 -> 65.62（+0.27%）**、16K入力後は **62.71 -> 63.58（+1.39%）**。入力処理はほぼ横ばいです。対応する出力トークン、投機採用数、CPU／GPU Expert処理数と配置が一致しました。

通常設定のAPIでも生成・4K／16K・画像・正常アンロードを確認しました。こちらの生成は74.64 -> 79.02 tok/sでしたが、自動Expert配置に差があるため、全差分を今回の変更の効果とは扱いません。19ケースのGPU数値検証、CUDA SM86／SM120のコンパイルも通過しています。設定・キーは変更せず、HIPの既定に採用します。既存ベンチ表は書き換えません。
