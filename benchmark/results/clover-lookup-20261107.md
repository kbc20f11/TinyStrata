# Clover exact suffix lookup — 2026-11-07

## English

Clover now has **optional prompt/history lookup**, without a draft model or a Clover preview forward. It proposes continuations already present in the current request's token history, then verifies them with the full causal model and the existing response-only penalties. All model-selected Experts are evaluated. Threshold acceptance and Expert omission are not used.

Use `--clover-lookup N` or the request field `clover_lookup`, where **N is the maximum number of proposals, 0–7**. Default **0/off** is unchanged. Positive values require Clover causal D0 and temperature 0; image input is supported. A suffix match shorter than eight tokens is capped at three proposals. Longer matches may use the requested width up to seven. This controls speculative work, not the acceptance rule.

The lookup index adapts [PolyStrata's suffix drafter](https://github.com/VecSzn/PolyStrata/tree/a82999fd60c2b6b3dbfb3924df60f8ff756c4371/src/spec) at `a82999fd`. Attribution is recorded in `NOTICE`. TinyStrata keeps its existing recurrent-state commit, image positions, checkpoint bookkeeping and generation penalties.

### Selected build and conditions

Base: TinyStrata `8ee33bb`. Selected native executable SHA-256:
`4ed238523adcc13f02571c9d264871e587c087d49a0834cf31ab4b1d28f016ca`.

Windows, RX 9060 XT 16 GB, Ryzen 9 7945HX3D, 64 GB RAM. Windows booted with the display connected to the Radeon. Original System32 HIP 7.12 and **4 MiB VMM allocations** were retained. No runtime, driver, clock or power-limit change is included.

These are **development comparisons, not tinystrata-bench-v1**. The same selected executable runs with lookup 0 and lookup 7. Each arm starts fresh with the same copied Expert rankings and 32-token warmup. The saved Clover IQ3_XXS/IQ3_S profile supplies context 262,144, chunk 2,048, FP8 MMA KV, automatic CPU workers (16 threads here), presence 0.9 and repetition 1.05. Temperature is 0, top-p 1, top-k 64, frequency penalty 0, seed 1234. Checkpoints are disabled in speed comparisons; all input tokens are read and reused tokens are zero.

The native engine accepts images, but **no image encoder is resident in the speed runs**. Image/API checks below use the real encoder separately. These timings must not replace Vision-resident historical tables or be treated as a controlled comparison with PolyStrata's different quantizations and benchmark inputs.

### Final automatic-placement comparison

There are 7,880 of 10,240 Experts initially resident, using 9.49 GiB of Expert cache in both arms. Long inputs are frozen exact-length token arrays from archive records plus a Python-copy instruction; a documented synthetic middle omission preserves the chat ending and code. The first three cases retain thinking and emit 128 tokens, mainly reasoning. The fourth disables thinking and emits 256 tokens of copied code, including numeric constants and comments.

| Case | Lookup off tok/s | Lookup on tok/s | Change | Accepted / offered |
|---|---:|---:|---:|---:|
| Short LRU question, thinking | 42.19 | 42.19 | −0.02% | 4 / 18 |
| After 4K input, thinking | 39.57 | 46.16 | +16.65% | 24 / 47 |
| After 16K input, thinking | 40.47 | 46.97 | +16.07% | 22 / 31 |
| After 4K input, code copying, no thinking | 42.42 | 173.05 | +307.96% | 220 / 220 |

Every output token ID, generated/read/reused count and finish condition matches its lookup-off control. **Only emitted tokens count as output**, including emitted thinking where enabled. Draft proposals are not counted as generated tokens. Lookup indexing and rejected verification work are included in the final native decode timer; there is no preview-model work.

The copying gain is workload-specific. Total request wall time, including prefill, changes from **10.303 to 5.665 seconds** for that case, rather than improving fourfold. Short-question wall time is **3.153 seconds** in both arms. The long-input thinking requests change from **7.553 to 7.077 seconds** at 4K and **21.680 to 21.263 seconds** at 16K. Lookup does not accelerate prefill itself.

### More CPU offload and the original singleton CPU policy

A separate pair uses a manual **3 GiB Expert budget**, with 2,440 resident Experts, and explicitly sets `STRATA_IQ_MT_MIN=2` in both arms. This increases CPU work on the same 16 GB card; it is **not a physical 8/12 GB VRAM simulation**.

| Case | Lookup off tok/s | Lookup on tok/s | Change |
|---|---:|---:|---:|
| Short LRU question, thinking, 128 tokens | 36.37 | 35.91 | −1.27% |
| 4K code copying, no thinking, 256 tokens | 33.67 | 115.00 | +241.59% |

Both output token arrays match. The short case's final Expert-table fingerprint also matches. An earlier candidate grouped CPU tokens under MT2 and diverged at output token 124. The selected implementation preserves singleton CPU jobs during lookup whenever the configured CPU kernel would otherwise switch arithmetic between one token and a group. It does not change the user's global CPU policy, ordinary causal decoding or the existing D3 path. The existing MT1 grouped path remains available.

### Validation and rejected intermediate behavior

- Unrestricted seven-token windows made the initial short-question screen **3.11% slower**. Restricting short suffix matches to three proposals removed that regression in the final automatic-placement case. This is why there is no unconditional widest-window default.
- CPU suffix-index tests pass **29,553 oracle comparisons**, including overlapping repeats, longest/recent matches, reset/rebuild, capacity and invalid-input checks. The CMake target is `q36_suffix_lookup_test`.
- **53 Python tests** pass: lookup request/config controls, existing Clover decode migrations and image settings. Positive temperature, self-speculation and invalid lookup counts are rejected; explicit zero disables lookup.
- Native state QA on the pre-singleton-compatibility candidate passes **11 raw-token pairs** with MT1: cached continuation, edited-turn restoration, limits 1/2, maintenance/KV growth, prefill/decode cancellation followed by retry, actual image input and its continuation. The final singleton compatibility change retains that MT1 arithmetic.
- The first automatic-placement KV test was invalid as a strict output comparison: trimming/restoring KV changed the Expert-table fingerprint between controls despite equal Expert counts. The controlled rerun holds membership fixed. Lookup also stops at ordinary causal decode's **64-row KV growth lookahead**, so it cannot move a cache resize across a verification window.
- A final active-boundary pair uses the selected executable, fresh processes and a full copy prompt padded to exactly 16,316 input tokens. Lookup accepts **78/78 proposals**, all **96 output IDs** match, and both arms grow physical KV from 16K to 24K immediately after the **fifth emitted token**. Fixed Expert-table fingerprints match. This directly exercises lookup and the growth guard together.
- The **selected executable** passes actual HTTP checks through OpenAI Chat Completions, Anthropic Messages and Responses, explicit lookup-off and three HTTP400 rejection cases. A real image plus a copy-code instruction exercises active lookup verification after image prefill; lookup 0/7 return identical answer/reasoning/stop fields. Image-history continuation also succeeds. HTTP output checks do not expose raw token IDs.
- The selected engine compiles for HIP gfx1200 and CUDA **SM86/SM120**. NVIDIA runtime speed is not measured on this Radeon machine. One initial CUDA compile attempt used an old cached command missing the current MSVC preprocessor/UTF-8 flags; the corrected current-flag build passes.

Each speed arm is a single screen, not a repeated statistical benchmark. Small differences around zero are not established gains. Exact causal acceptance avoids deliberately lossy speculation, but floating-point batching can still change near-tied choices on other inputs/hardware; fixture token identity is not a universal bitwise guarantee or a broad quality evaluation. Therefore lookup is adopted as an **opt-in**, while causal D0 and lookup off remain the defaults. Standard README v1 throughput tables are unchanged.

Private frozen prompts, raw outputs, request timings, placement/module fingerprints and build manifests are retained under `TinyStrata-work/clover-lookup-20261107`. Saved model settings, API keys and chat history are preserved.

## 日本語

Cloverに、入力・会話履歴の続きを候補として使うlookupを任意機能として追加しました。候補生成用のモデル実行はなく、因果モデル全体で完全一致を検証します。Expertの省略や近似採用は使いません。

最終版の自動配置では、短文は約42.2 tok/sで同等、4K／16K入力後の思考は約16%改善、入力中のコードを再掲する条件は **42.42→173.05 tok/s**でした。CPU処理を増やした3 GiB Expert条件でも、コピーは **33.67→115.00 tok/s**です。各速度比較の出力トークンは無効時と一致しました。コピーに適した入力での約4倍であり、一般会話すべての4倍を意味しません。入力処理を含むコピー要求全体は10.303→5.665秒です。

途中で見つかったMT2の単独／複数トークンCPU演算の差を修正し、通常の因果実行と同じ演算を維持しています。短い一致は最大3候補、8トークン以上の一致だけ最大7候補とし、KV拡張の64行先行確保や再配置の境界も越えないようにしました。画像と実APIでの稼働、会話復帰・キャンセル後の継続も確認しています。

効果が小さい会話や少し遅くなる条件もあるため、**既定は無効のまま**です。`--clover-lookup 7`／要求ごとの `clover_lookup: 7` で有効にできます（因果D0・温度0）。通常READMEの過去のベンチ表は変更せず、この比較を別に記録しています。異なるGPUでの速度や、すべての入力でのビット一致を保証するものではありません。
