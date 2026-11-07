# Qwen/Gemma target-sampled lookup and adaptive MTP — 2026-11-07

## English

Qwen and Gemma now offer **optional prompt/history lookup** and **experimental adaptive MTP**. Lookup improves the tested code-copy workloads, including with more CPU offload. It does not consistently accelerate ordinary conversation. The selected defaults remain **MTP 3, lookup off, automatic MTP off**; temperatures remain **Qwen 0.7 / Gemma 1.0**. These are controlled development comparisons, **not tinystrata-bench-v1**, and do not replace the historical README throughput tables.

### Implementation and controls

Use `--lookup-draft N` / request `lookup_draft: N` for **0–7 maximum proposed tokens**, or `--mtp-auto 0|1` / request `mtp_auto: 0|1` for measured D0/D1/configured-maximum selection. Both new controls default to **0/off**. Lookup's number is a proposal budget, not a suffix-match length. Gemma reserves its verification window at startup, so a request alone cannot enlarge `Engine.max_window() - 1`; launch with the intended lookup width. Qwen has eight verification rows, with lookup capped at three proposals on its INT8 FDEC route. Short suffix matches are also capped at three proposals.

Lookup copies continuations already present in the request's prompt/history. Each usable verification row uses the existing full target sampler, penalties and RNG; a proposal is accepted only when it equals the ordinary target draw. A mismatch returns that target draw and discards untrusted later rows. This supports normal positive temperatures and images without approximate acceptance, Expert omission or tail skipping. Only emitted tokens count toward throughput. Lookup indexing, rejected verification and MTP work remain inside decode time. Floating-point batching can still change logits and seeded output; this is not a universal bitwise-equivalence guarantee.

The suffix index and scheduling ideas reference pinned PolyStrata [suffix drafting](https://github.com/VecSzn/PolyStrata/blob/a82999fd60c2b6b3dbfb3924df60f8ff756c4371/src/spec/suffix_drafter.cpp), [draft policy](https://github.com/VecSzn/PolyStrata/blob/a82999fd60c2b6b3dbfb3924df60f8ff756c4371/src/spec/draft_policy.cpp) and [adaptive MTP control](https://github.com/VecSzn/PolyStrata/blob/a82999fd60c2b6b3dbfb3924df60f8ff756c4371/src/poly/guesses.hpp). Attribution is in `NOTICE`. Existing Gemma CPU/GPU overlap was already present; this work does not introduce that pipeline.

### Conditions and selected binaries

Base TinyStrata: `342655c`. Windows, **RX 9060 XT 16 GB**, Ryzen 9 7945HX3D, 64 GB RAM, display connected to the Radeon before Windows boot. Original **System32 HIP 7.12 and 4 MiB VMM allocations** are retained; no runtime, driver, clock or power-limit change is included. Loaded-module snapshots identify the same System32 HIP DLL in the compared arms.

The selected Qwen C3 executable SHA-256 is `fa7ddea6a160eeb3390d970e3ca96387f6c651314329b1f3372814a8d65c5dfd`; Gemma C5 is `bca5a24a610829a555f001aa05994698710734ab8ddd29ea4532963a5664945d`. Qwen's automatic-placement table below measures C2 (`8e81828112b0479d3c3a9a52125da7f6978585fc68d546ad524c1119de14d61d`); C3 changes stale-arm probing, and its actual final-binary checks include the CPU-offload and HTTP/image tests below. No unmeasured C3 automatic-MTP speed gain is claimed.

Each mode starts a fresh private native process, uses copied Expert profiles, identical eight-row lookup scratch and MTP 3, then receives a 32-token warmup. The 262,144-token logical context grows physically. Qwen uses the saved **UD-IQ3_XXS**, FP8 MMA KV, automatic chunk 512 and 16 CPU threads; Gemma uses **Google QAT Q4_0 plus its matching QAT MTP assistant**, INT8 KV and chunk 1,120. Qwen sampling is temperature 0.7/top-p 0.95/top-k 20/presence 1.5, with the API's 64-token penalty window; Gemma is temperature 1.0/top-p 0.95/top-k 64, with no presence penalty. Repetition penalty is 1.0, seed 1234, checkpoints disabled. Saved profiles are not rewritten.

These native speed fixtures retain the image-capable flag but **do not load a resident image encoder**. Actual image tests run separately through the HTTP API. Short and ordinary long-input cases emit 128 tokens with the normal thinking setting. The code-copy case disables thinking and emits 256 tokens from a frozen 4,096-token archive-plus-Python prompt, with a recorded synthetic middle omission. All inputs are read afresh, with zero reused tokens. All modes use the same prompt token arrays.

### Automatic-placement screens

`fixed` means configured MTP 3 and both new controls off. `auto` enables only adaptive MTP, `lookup` only lookup 7, and `both` enables both.

| Model / case | Fixed tok/s | Auto tok/s | Lookup tok/s | Both tok/s |
|---|---:|---:|---:|---:|
| Qwen, short | 102.54 | 101.74 | 101.15 | 100.35 |
| Qwen, after 4K input | 95.41 | 95.67 | 100.46 | 100.25 |
| Qwen, 4K code copying | 112.30 | 112.48 | **173.36** | 172.50 |
| Gemma, short | 82.38 | 83.80 | 85.28 | 76.30 |
| Gemma, after 4K input | 72.51 | 71.99 | 71.94 | 71.11 |
| Gemma, after 16K input | 65.72 | 66.90 | 64.96 | 61.15 |
| Gemma, 4K code copying | 95.61 | 95.01 | **122.78** | 121.66 |

Qwen begins with 9,840/10,496 Experts resident and Gemma with 3,451/3,840; initial placement counts and cache budgets match across their respective modes. Copying improves **54.37% for Qwen** and **28.41% for Gemma**, with all 256 output IDs matching the fixed controls in every mode. Qwen's auto-only screen also matches all fixed output IDs. Lookup/both after Qwen's 4K ordinary input first differ at zero-based output index 112. Gemma's ordinary seeded outputs differ between modes; lookup's first differences are indices 21/9/13 for short/4K/16K. Generated counts, read counts and finish conditions still match. These divergent ordinary outputs cannot establish equal-answer latency gains.

Lookup changes generation work, not prompt processing. In the copying pair, Qwen input throughput is 790.73→792.03 tok/s and Gemma 2,634.42→2,629.86 tok/s. Do not interpret the decode percentages as prefill or whole-request speedups.

### More CPU offload: final-binary lookup-only checks

A separate final-binary pair fixes the **Expert budget to 3 GiB** while preserving MTP 3 and the same positive-temperature sampling. This is a CPU-offload stress test on the 16 GB card, **not a physical 8/12 GB VRAM simulation**. Qwen retains 2,665 Experts and Gemma 962, with matching placement counts in each pair.

| Model / case | Lookup off tok/s | Lookup 7 tok/s | Change | Raw output IDs |
|---|---:|---:|---:|---|
| Qwen, short, 128 tokens | 79.07 | 78.68 | −0.49% | Differ from index 108 |
| Qwen, 4K code copying, 256 tokens | 84.91 | **124.43** | **+46.54%** | All match |
| Gemma, short, 128 tokens | 80.60 | 69.03 | **−14.35%** | Differ from index 21 |
| Gemma, 4K code copying, 256 tokens | 80.24 | **103.11** | **+28.50%** | All match |

The Gemma short regression rules out a global lookup default. Copy-friendly workloads can benefit, while ordinary conversation should retain the current default unless a user's own comparison justifies opting in.

### Improvement loop and rejected defaults

The first Gemma controller repeatedly calibrated D0/D1/Dmax and regressed roughly 9% on the short case. Later versions persist comparable-session evidence, exclude graph-recording costs, require enough fresh evidence before switching, and reduce probing when the leader is clearly faster. A discovered physical-KV boundary issue was corrected: ordinary MTP fallback keeps its existing full window and growth timing; only lookup receives the physical-row cap. Poor accidental lookup matches now back off, while strong copying keeps separate evidence.

A CPU policy fixture caught excessive exploration even when D0 was a clear winner. The schedule was fixed, keeping the original assertion: the last 1,096 simulated rounds now select D0 993 times instead of 891, while still remeasuring alternatives. Both actual C++ policy suites pass. These controller fixes do not alter the target acceptance rule.

Automatic MTP still has no reliable general gain. In the 512-token Qwen C2 long-answer screen, LRU explanation changes **90.00→88.52 tok/s** and a Japanese story **81.14→72.07 tok/s** (−11.18%); seeded outputs diverge at indices 212 and 51. Gemma C5's combined mode is slower on ordinary short/16K cases. Automatic MTP therefore remains **experimental/off**, and combining it with lookup is not a default recommendation. One-pass depth-3/5/7 probes likewise do not establish one best depth across workloads or placement geometries.

### Validation and limits

- Actual CPU tests pass: Gemma policy/gate/KV bounds; Qwen policy with **3,072 observations**; **19 Python option tests** (seven new plus 12 existing Clover tests), **26 Clover decode-setting tests**, and **15 Clover image-setting tests**.
- Both selected executables pass private HTTP tests through **OpenAI Chat Completions, Anthropic Messages and Responses**, including lookup-off overrides, bounds/type/family HTTP400 rejections and output limits 1/2. Positive-temperature settings are exercised.
- Real images plus a known-copy instruction exercise active lookup: lookup-off, lookup 7 and combined mode return identical content/reasoning/stop fields for both models. Image-history continuation succeeds. Native counters record **60 active lookup rounds per model**, with adaptive rounds **10 for Qwen / 13 for Gemma**. HTTP checks do not expose raw token IDs. Protected saved settings, keys and history remain unchanged.
- Native state checks pass for both selected binaries: active lookup plus adaptive MTP, raw-prefix continuation, an edited turn checkpoint, and STOP during decode followed by retry on the same process. Each of four resulting outputs per model matches a fresh lookup-off reference in every token ID, generated count and finish condition. The 32K-context / 3 GiB Expert test uses greedy sampling, fixed membership and no encoder; it verifies these state transitions, not broad numerical identity. Saved configurations remain unchanged.
- Final-answer checks at the saved positive temperatures, with thinking explicitly off and automatic placement, also pass. Both models answer 17×23 as 391 and provide syntactically valid `sum`/`range` Python for the sum of squares 1–10, with result 385 and a Japanese explanation. Lookup off/on matches all output IDs in these four pairs; all eight answers stop normally without repetition flags. The generated code was inspected and parsed, not executed. These small fixtures do not establish broad task accuracy.
- HIP gfx1200 private builds pass. CUDA **compile-only** validation passes for Qwen's engine translation unit targeting **SM86/SM120**, including its INT8 FDEC branch. Gemma's changed host sources compile against the CUDA build; unchanged Gemma GPU kernels are retained. This is not a CUDA runtime, linking or speed result.

Every speed cell is one development pass, not a repeated statistical result. Small differences around zero are not established gains. Seeded floating-point differences are disclosed rather than hidden; no broad quality or every-environment guarantee follows from the fixtures. Private prompts, IDs, timing, module and build manifests are retained under `TinyStrata-work/qwen-gemma-spec-20261107`.

### What the PolyStrata comparison does and does not establish

At pinned `a82999fd`, PolyStrata's [Qwen laptop table](https://github.com/VecSzn/PolyStrata/blob/a82999fd60c2b6b3dbfb3924df60f8ff756c4371/docs/models/qwen36.md) reports **94–100 tok/s with UD-Q3_K_M**, and its [Gemma table](https://github.com/VecSzn/PolyStrata/blob/a82999fd60c2b6b3dbfb3924df60f8ff756c4371/docs/models/gemma4.md) reports **92–93 tok/s with non-QAT UD-Q3_K_M** on an RTX 4070 Laptop 8 GB. Those are different weights from TinyStrata's Qwen IQ3_XXS and Gemma QAT Q4_0. The same Gemma document separately reports later QAT-UD-Q4_K_XL laptop chat at 50–62 tok/s under its stated conditions.

PolyStrata also uses greedy/warm 256-token short-chat measurements, a 32K context and CPU image encoding in the compared profile, versus TinyStrata's historical positive-temperature, resident-GPU-vision protocol. Its [setup defaults](https://github.com/VecSzn/PolyStrata/blob/a82999fd60c2b6b3dbfb3924df60f8ff756c4371/setup.py#L624) enable `--tail-skip 6` for Qwen, which can omit low-ranked CPU Experts; Gemma has no such default (tail skip 0). TinyStrata does not adopt that lossy option. Lookup and measured draft control were concrete portable candidates worth implementing and testing; these results do not quantify remaining CPU scheduling, dot-kernel or backend differences, or attribute the full cross-project gap to hardware composition.

An additional one-pass Tiny-only check uses PolyStrata's exact prompt, “Explain in three sentences why the sky is blue.”, fixed MTP 3, automatic placement and a 256-token cap. Qwen measures **97.99 tok/s at saved temperature 0.7 / 90.27 at greedy 0**; Gemma **81.81 at saved 1.0 / 90.20 at greedy 0**. These runs have no resident image encoder and all four reach the cap while still thinking, so they are reasoning throughput, not completed three-sentence-answer latency. Temperature changes the generated text and MTP acceptance, and is not uniformly faster in either direction. The test does not reproduce PolyStrata's weights/backend or isolate every cause of its laptop numbers; it shows why those numbers should not be compared directly with the old mixed-workload README table.

## 日本語

Qwen・Gemma に、入力や履歴の続きを候補にする lookup と、実測で幅を選ぶ自動 MTP を追加しました。既定は **MTP 3、lookup 無効、自動 MTP 無効**です。温度も **Qwen 0.7／Gemma 1.0** を維持します。通常のサンプラーで主モデルが選んだトークンと一致する候補だけを採用し、Expert の省略・近似採用は行いません。画像にも対応します。

自動配置のコピー入力では lookup 単独で、Qwen **112.30→173.36 tok/s（+54.37%）**、Gemma **95.61→122.78 tok/s（+28.41%）**でした。CPU 処理を増やした 3 GiB Expert 条件でも、Qwen **84.91→124.43（+46.54%）**、Gemma **80.24→103.11（+28.50%）**です。これらのコピー比較では 256 出力トークンの ID がすべて一致しました。3 GiB は CPU オフロードの検証であり、物理的な 8／12 GB GPU を再現した測定ではありません。

一方、3 GiB 条件の Gemma 短文は **80.60→69.03 tok/s（−14.35%）**となり、通常会話を一律に高速化するものではありません。自動 MTP も、Qwen の 512 トークンの物語で **81.14→72.07 tok/s**と遅くなりました。制御を改善しても一貫した利益が出ないため、既定への昇格はしません。通常文では同じ seed でも出力が異なる条件を表に隠さず記録し、コピーでの一致をすべての入力での一致保証とはしていません。

`--lookup-draft N`／要求の `lookup_draft: N` は最大提案数 0～7、`--mtp-auto 0|1`／`mtp_auto: 0|1` は実験的な自動 MTP です。両方とも既定 0 です。Gemma の幅は起動時の容量が上限なので、7 を使う場合は起動引数でも確保してください。Qwen の INT8 FDEC 経路は最大 3 提案です。既存の Clover 専用 lookup の条件は変更しません。

実画像と 3 種類の HTTP API、画像の続き、出力上限 1／2、無効値の拒否を確認しました。画像付きコピーでは無効／lookup／併用の回答・思考・終了理由が一致し、実際の lookup 稼働も確認しています。CPU テストも通過しています。CUDA は上記のコンパイル確認までで、Radeon 上の測定から NVIDIA の実行速度は主張しません。

生成途中の中断から同じプロセスで再開するケース、履歴の再利用、編集チェックポイントも両モデルで通過しました。各モデル 4 ケースの出力が、新規起動した無効側とトークン ID・生成数・終了理由まで一致しました。こちらは 32K・Expert 3 GiB・greedy・配置固定の状態検証であり、あらゆる入力での数値一致を保証するものではありません。

最終回答も通常の温度で確認し、両モデルが 17×23＝391、1～10 の二乗和＝385 と正しく回答しました。Python コードは構文解析を通過し、日本語の説明を含めて lookup 無効時と全トークンが一致しました。4 比較・8 回答とも正常終了し、ループは認めませんでした。小規模な確認であり、広範な正答率の評価ではありません。

PolyStrata の laptop の数字は量子化・入力・温度・画像エンコーダー常駐条件が異なります。Qwen の tail-skip 6 は Expert を省くため取り込みません。Gemma にはその既定がないので、tail-skip だけで差を説明することもできません。未測定の CPU スケジューリングや演算カーネルの寄与は断定していません。この報告は各条件 1 回の開発測定であり、固定 v1 の過去の表や全環境での速度保証とは区別します。
