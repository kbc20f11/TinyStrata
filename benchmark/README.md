# TinyStrata benchmark protocol v1

[English](#english) · [日本語](#日本語)

## English

Use **`tinystrata-bench-v1`** when submitting comparable TinyStrata measurements. The runner, fixed texts and image in this directory define the workload. Keep the protocol version and input SHA-256 hashes with the results; changing the workload requires a new protocol version. Older README measurements do not automatically become v1 results.

### Run

Download the models to be measured using the normal installer/API first, then stop the regular TinyStrata API. In 64-bit PowerShell:

```powershell
$python = "$env:LOCALAPPDATA\TinyStrata-work\runtime\.venv\Scripts\python.exe"
& $python "$env:LOCALAPPDATA\TinyStrata\benchmark\run.py" --model all
```

From a source checkout, use `.\benchmark\run.py` instead. `--model qwen`, `--model gemma`, `--model lfm` and `--model clover` select one model. Clover uses `config-clover.json`. The runner uses the existing model/configuration files, starts its own loopback API, and does not change the saved configuration. Results go under `%LOCALAPPDATA%\TinyStrata-work\benchmarks\`; generated logs and results do not belong in Git.

The app defaults to causal decoding without drafts, D0, with INT8 KV and `STRATA_IQ_MT_MIN=1`; exact D3 is optional. Its selected route and the fixed benchmark regulation are separate. The standard benchmark always uses **INT8 causal decoding, D0**, independently of the app profile. Native LowConfidence block diffusion is measured separately:

```powershell
& $python .\benchmark\run.py --model clover --clover-mode causal
& $python .\benchmark\run.py --model clover --clover-mode diffusion --clover-block-size 32 --clover-threshold 0.95
```

The first command retains the **`tinystrata-bench-v1` causal baseline**. The second uses the same inputs, repetitions, sampling and timing boundaries, but reports **`tinystrata-bench-v1-clover-diffusion`** with `standard: false`, so it cannot be mistaken for a causal v1 result. Its block size and confidence threshold are recorded with the result. The modes generate text differently; identical output token sequences are not a requirement. Count only final returned tokens, including thinking, once per token. Mask predictions and denoising iterations do not increase the output token count, and their execution time remains inside decode time.

Both profiles keep Clover's model-specific sampling settings; this protocol uses greedy generation and does not characterize sampled generation.

`--clover-block-size` accepts **1, 2, 4, 8, 16 or 32** (default **32**). `--clover-threshold` accepts **0–1** (default **0.95**); a pending masked position is filled when its confidence is strictly above the threshold, with at least one position filled per pass. These are diffusion settings. In the native engine and `run.py`, legacy `--clover-draft 0` selects causal decoding; a positive draft count selects the separate self-speculative route, which uses causal verification. The benchmark selects each mode explicitly.

Self-speculative generation is another separately labelled variant. Use exact acceptance as the control; threshold acceptance is an explicit experimental option:

```powershell
& $python .\benchmark\run.py --model clover --clover-mode self-spec --clover-draft 3 --clover-accept exact --engine-env STRATA_IQ_MT_MIN=1
& $python .\benchmark\run.py --model clover --clover-mode self-spec --clover-draft 3 --clover-accept threshold --clover-accept-ratio 0.9 --clover-accept-delta 0.6 --engine-env STRATA_IQ_MT_MIN=1
```

The first command accepts only drafts that match the causal verifier's selected token; the second measures the experimental threshold policy. Both report **`tinystrata-bench-v1-clover-self-spec`**, `standard: false`, with the same v1 workloads, sampling, repetitions and timers. In this benchmark runner, `--clover-draft` accepts **1–31** and defaults to **3** in self-spec mode; `--clover-accept` defaults to **exact**. Ratio and delta accept finite values in **0–1**, defaulting to **0.9** and **0.6**, and require self-spec mode. Specify these options explicitly when comparing policies.

The optional exact D3 profile sets `STRATA_IQ_MT_MIN=1` so single-row and grouped CPU IQ Expert work use the same accumulation path. For a matching causal control, add the same `--engine-env STRATA_IQ_MT_MIN=1` to the causal command; that control is also a variant. The standard causal v1 baseline keeps an empty tuning environment (`env: {}`), regardless of the saved app profile.

To measure Clover's default **GPU-resident Vision + causal D0** profile and optional **exact D3** profile, use these separately labelled variants:

```powershell
& $python .\benchmark\run.py --model clover --clover-mode causal --engine-env STRATA_IQ_MT_MIN=1 --clover-vision
& $python .\benchmark\run.py --model clover --clover-mode self-spec --clover-draft 3 --clover-accept exact --engine-env STRATA_IQ_MT_MIN=1 --clover-vision
```

`--clover-vision` reuses the installed Qwen3.6 projector, keeps the image encoder resident during every text test, and adds the same canonical image-description test as Qwen/Gemma. It uses 1,024 minimum / 4,096 maximum image tokens and records `vision: true`. The results are labelled **`tinystrata-bench-v1-clover-causal-variant-vision`** and **`tinystrata-bench-v1-clover-self-spec-vision`**, both `standard: false`; the tuning environment adds the `-variant` suffix to the causal result. Diffusion and threshold acceptance cannot use this flag. The fixed causal v1 baseline still has Vision off. Switching the app's default does not rewrite that historical comparison protocol.

`threshold` is approximate acceptance at temperature 0, not distribution-preserving speculative sampling. It uses target probabilities and entropy with the same response-count snapshot, including the already emitted anchor, for all preview and verification rows in a round; uncommitted drafts are excluded. It does not guarantee the causal/exact token sequence. A target EOS always stops, and an accepted draft EOS also remains a stopping boundary. Count final output tokens once and include preview, verification, transfers and acceptance work in decode time. The runner records the actual native settings, accepted/offered counts, transfers and kernel flags; report quality separately from throughput.

Clover-only optional overrides are `--kv-type int8|fp8-mma`, `--cpu-workers N` (0 = auto), `--chunk auto|N`, and repeatable `--engine-env STRATA_NAME=VALUE` / `TINYSTRATA_NAME=VALUE`. For example:

```powershell
& $python .\benchmark\run.py --model clover --clover-mode diffusion --kv-type fp8-mma --cpu-workers 9 --chunk 1024
```

Any such override makes the result a **variant** even when causal mode is selected. Saved profiles remain unchanged. The runner records the requested profile, checks the native mode/KV/block/threshold/explicit workers/chunk, and keeps the engine's reported worker count, total threads, block/pass counts and preview/replay times when available. Use the same overrides for the causal and diffusion comparison; the default command remains the INT8/automatic causal v1 baseline.

For a separate Clover comparison without thinking, add `--thinking off`. The default is `--thinking on`; disabling it marks even a causal run as a **variant**, records `thinking: false`, and applies to warm-up, token counting and every text request. The canonical short texts, sampler and seed stay fixed; the different chat template changes short-input token counts. Long inputs are still counted to exactly 4K/16K. This option does not change the saved profile or the standard v1 workload.

For a longer Clover comparison, add **`--include-64k`**:

```powershell
& $python .\benchmark\run.py --model clover --include-64k
& $python .\benchmark\run.py --model clover --include-64k --clover-mode self-spec --clover-draft 3 --clover-accept exact --engine-env STRATA_IQ_MT_MIN=1
```

This opt-in extension adds one fresh **65,536-token input including the chat template**, with the existing **32-token output limit**, after the six short-generation requests and the 4K/16K requests on each start. It keeps the warm-up, inputs, sampler, seed, thinking and default four starts unchanged. Startup context must hold at least **65,568 tokens**; otherwise measurement stops before any generation request. Results are nonstandard, carry a **`-64k` protocol suffix** and `include_64k: true`, and report 64K input processing plus generation after 64K using actual output counts. The 16K cache snapshot stays at the 16K boundary; a separate 64K snapshot records Expert cache and physically mapped KV. Without this flag, the v1 workload and report fields remain unchanged. The option requires `--model clover`.

### Clover repetition lookup (optional)

The launcher and API support `--clover-lookup N` / `clover_lookup: N`, with **0–7** as the maximum number of proposed tokens; this is not the suffix-match length. **0/off remains the default.** Use causal D0 and temperature **0**; images are supported. For example, from a source checkout:

```powershell
& $python .\run.py --model-family clover --clover-mode causal --clover-draft 0 --clover-lookup 3
```

Lookup copies candidate continuations already present in the prompt/history. The full causal model verifies them with the same generation penalties. Matches shorter than **8 tokens** are internally capped at **3 proposals**; requested widths **4–7** require a match of at least **8 tokens**. It uses no D3 preview, Expert skipping or threshold acceptance. Count only emitted output tokens; rejected proposals add work, not output. Compare **3** and **7** with lookup disabled on your own workload; gains depend on how much can be copied. The standard v1 runner does not enable lookup; [lookup measurements](results/clover-lookup-20261107.md) are separate experiments. Floating-point batching can change near-tied choices, so matching token IDs on the tested fixtures is evidence for those fixtures, not a universal bitwise-equivalence guarantee.

### Qwen/Gemma lookup and adaptive MTP (optional)

The launcher supports `--lookup-draft N` (**0–7**, maximum proposed tokens rather than match length) and experimental `--mtp-auto 0|1`; API request overrides are `lookup_draft: N` and `mtp_auto: 0|1`. **Both default to 0/off.** Default **MTP 3** and sampling temperatures (**Qwen 0.7 / Gemma 1.0**) are unchanged. Automatic MTP selects among D0, D1 and the configured maximum using measured round costs; it does not consistently improve every workload and remains experimental.

Lookup proposes known prompt/history continuations. Each accepted token must match an ordinary draw from the existing full target model, with the same sampler, penalties and normal temperature support; images are supported. All selected Experts are computed, without approximate acceptance or tail skipping. Count only emitted tokens, and include rejected verification work in decode time. Floating-point batching can still change token choices; agreement on tested fixtures is not a universal bitwise-equivalence guarantee.

Gemma allocates its verification capacity at startup: use the launcher's `--lookup-draft N` to reserve the desired window. A per-request override is clamped to that initialized capacity (`Engine.max_window() - 1`), so setting only a request to 7 does not enlarge a service started with ordinary MTP 3. Qwen's native verifier has eight rows (up to seven proposals); its INT8 FDEC route caps lookup at three proposals. Requested width is an upper bound, not a promise that every round uses it.

For example, from a source checkout:

```powershell
& $python .\run.py --model-family qwen36 --lookup-draft 3
& $python .\run.py --model-family gemma4 --lookup-draft 7
```

Compare lookup alone with the same saved profile and lookup off before also testing `--mtp-auto 1`; combining them is not guaranteed to be faster. The [controlled development runs](results/qwen-gemma-speculation-20261107.md) are separate from **tinystrata-bench-v1** and do not replace its historical tables or defaults.

### CPU worker probe (diagnostic)

Stop the regular API first, then use [probe.py](probe.py) for Qwen or Clover:

```powershell
& $python .\benchmark\probe.py --model clover
```

The probe preserves the saved profile's native arguments and tuning environment, including Expert layout and input chunk, except the worker count and explicit command-line overrides. It uses a controlled temperature-0 sampler, starts only its own benchmark services, and never saves a new application setting. Counts mean **background workers plus one participating host**: `--cpu-threads 7` uses eight CPU threads. If every Expert is GPU resident, it reports that worker tuning does not apply.

Candidates are powers of two, half the observed automatic count, and that count itself. Mirrored screens choose the fewest workers within 2% of the best timing only when the gain exceeds baseline variation and a 3% floor. A manual recommendation also requires fresh ABBA and BAAB confirmations using three short prompts and exact 4K/16K inputs, matching output/reasoning hashes, native counts and Expert byte capacity, with no material long-input regression. Results use **`tinystrata-worker-probe-v1`**, not standard v1; a printed `--cpu-threads N` applies only to the recorded model and conditions.

`--prepare-only --default-workers N` writes a bounded plan without starting services. `--expert-gib 4` explicitly tests a 4-GiB Expert budget; otherwise the profile's budget is preserved. `--chunk N` fixes one chunk size. This tool does not search chunk sizes or implement a fractional CPU/GPU prefill scheduler.

### Common configuration

Use the project's fixed model files: Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS, Google Gemma 4 26B-A4B-it QAT Q4_0 with its matching vision projector and QAT MTP assistant, LiquidAI LFM2.5-24B-A2B Q4_K_M, and [AtomicChat Clover-35B-A3B-Preview IQ3_XXS](https://huggingface.co/AtomicChat/Clover-35B-A3B-Preview-GGUF/blob/4f7e2918e5315634d9d9720ea4f6e8a9a627a875/Clover-35B-A3B-Preview-AD-IQ3_S-IQ3_XXS.gguf) (`Clover-35B-A3B-Preview-AD-IQ3_S-IQ3_XXS.gguf`; Expert gate/up IQ3_XXS, down IQ3_S). Keep older Clover Q4_0 measurements labeled separately; they do not represent this IQ3_XXS profile. Record the actual model and binary hashes, TinyStrata version/commit, GPU/VRAM, CPU, system RAM, OS and driver version with a submission.

- **INT8 KV** is the common comparison profile on NVIDIA and Radeon. FP8 or manually tuned settings belong in separately labelled results.
- KV growth, automatic context capacity, automatic input chunk size, automatic Expert placement and automatic CPU worker selection. Graphs use their normal defaults. Do not add tuning environment variables or omit Experts/tokens.
- MTP **3** and a **GPU-resident vision encoder** for Qwen/Gemma, including the three text benchmarks. LFM has neither. The fixed-v1 Clover control uses neither MTP nor Vision and explicitly uses causal decoding (`--clover-mode causal`, D0); the app's causal D0/Vision profile and optional exact D3/Vision profile use the separately labelled variants above.
- One request at a time; **seed 1234**. The following sampling values are fixed in `cases.json`, with the added Clover profile fixed in `run.py` so the existing v1 input hashes remain unchanged:

| Model | Temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6 | 0.7 | 0.95 | 20 | 0 | 1.5 | 1.0 |
| Gemma 4 | 1.0 | 0.95 | 64 | 0 | 0 | 1.0 |
| LFM2.5 | 0.1 | 1.0 | 50 | 0 | 0 | 1.05 |
| Clover | 0 | 1.0 | 64 | 0 | 0.9 | 1.05 |

Frequency penalty is zero. Thinking is enabled for generation and the two input-processing benchmarks; image description disables thinking so its first output is part of the description. The model's early EOS is respected: count the tokens actually generated, not an assumed output length.

### Four benchmarks

| Benchmark | Fixed workload | Report |
|---|---|---|
| Generation | The three `generation` texts in `cases.json`, in order, twice each; maximum 128 output tokens per request, including thinking | Total generated tokens ÷ total native decode seconds, tok/s |
| 4K input | The numbered document defined by `long_paragraph` and `long_question`, adjusted to exactly **4,096 input tokens including the chat template**; maximum 32 output tokens | Total input tokens actually read ÷ total native prompt-processing seconds, tok/s |
| 16K input | The same construction at exactly **16,384 input tokens including the chat template**; maximum 32 output tokens | Same input-processing rate, tok/s |
| Image description | Unmodified **1,024 × 768** `image.png` and `image_prompt`; maximum 128 output tokens; first real image request after each fresh API start | Median client first-output time and median completion time, seconds; LFM and the fixed-v1 Clover control are **N/A**; `--clover-vision` includes Clover |

The runner uses the normal Messages API, including MTP, rather than the native `--tokens` generation loop. It uses the same request with `/v1/messages/count_tokens` to construct the 4K/16K inputs, then checks the actual native counts. The published texts/image and their hashes identify the inputs; the native prompt counts identify their token lengths for each model.

### Repetitions and timing boundaries

Use **four independent API starts per model**. On each start, keep this order:

1. Wait for model loading and the built-in warm-up. Require the Qwen/Gemma GPU image encoder to be alive, and record the initial Expert cache size.
2. Send the separate `warmup_text` once with at most 32 output tokens; exclude it from all rates.
3. Run generation: three fixed texts × two passes.
4. Run the 4K request once, then the 16K request once. Record the Expert cache size immediately after 16K.
5. Run image description once for Qwen/Gemma. Restart the **whole API server** for the next repetition so the image-embedding hash cache is empty again.

There are **24 measured generation requests per model**, four 4K requests, four 16K requests, and four image requests for a vision model. Text requests are independent single-turn messages, with no reused KV: check `reused == 0` and `prompt_read == prompt_total`. Input-processing samples must also report exactly 4,096/16,384 newly read tokens.

Use native `/metrics` values for text rates:

```text
generation tok/s = 1000 × Σ engine_generated / Σ decode_ms
input tok/s      = 1000 × Σ prompt_read       / Σ prompt_ms
```

Do not average per-request tok/s or select the fastest repetition. Loading and the separate warm-up are excluded. Each benchmark request sets `strata_checkpoint: false` so conversation parking cannot turn a repeated input into a cache hit. Failed, cancelled, cached or wrong-length samples cannot silently be dropped from a standard result: report an incomplete run or repeat the complete protocol after fixing the cause.

For an image, start the client's monotonic clock immediately before POST. First-output time ends at the first nonempty streamed model text, not HTTP headers/keep-alives. Completion time ends at `message_stop`. These times include upload, image decoding/encoding, prompt processing and generation; they exclude server/model startup. Native `prompt_ms` alone excludes image encoding and must not replace the image times. Record the actual description and output length too.

### Expert cache and result format

Save the reported `expert_cache_bytes` before and after every benchmark; when the engine reports only `expert_cache_mib`, convert it using 1 MiB = 1,048,576 bytes. In a summary table's **Notes**, show **startup → after-16K** sizes in **GiB = bytes / 1,073,741,824**. If the four starts differ, show their minimum–maximum range; preserve the individual values in the JSON. For example: `Expert cache 9.21 → 9.04 GiB`.

This is the allocated GPU Expert cache, not the model's file size or an estimate of other applications' usage. KV growth, workspace and headroom adjustments can change it during a run; a later short image request may enlarge it again, which is why the second value is taken immediately after 16K. No other-application memory usage is collected by this protocol.

| Model | Context | Generation tok/s | 4K input tok/s | 16K input tok/s | Image first / complete, s | Notes |
|---|---:|---:|---:|---:|---:|---|
| … | … | … | … | … | … / … | Expert cache … → … GiB |

Use the same protocol/input hashes, model files and KV profile when comparing results. Report hardware and the actual automatic context/chunk settings alongside the table. Simulated VRAM limits are separate results from physical GPU measurements. Publish the complete run summary, not a single favourable sample.

## 日本語

TinyStrata の比較用計測は **`tinystrata-bench-v1`** に統一します。このフォルダーの実行スクリプト・固定テキスト・画像が計測条件です。結果には規約バージョンと入力の SHA-256 を残します。入力や手順を変更する場合は規約も改版し、以前の README の数値を後から v1 の結果として扱いません。

### 実行方法

通常のインストーラー／API で測定するモデルを先にダウンロードし、普段の TinyStrata API を終了します。64 ビット PowerShell で実行してください。

```powershell
$python = "$env:LOCALAPPDATA\TinyStrata-work\runtime\.venv\Scripts\python.exe"
& $python "$env:LOCALAPPDATA\TinyStrata\benchmark\run.py" --model all
```

ソースを取得したフォルダーでは、スクリプトのパスを `.\benchmark\run.py` に置き換えます。個別に測る場合は `--model qwen`、`--model gemma`、`--model lfm`、`--model clover` を使います。Clover の設定ファイルは `config-clover.json` です。既存のモデルと設定ファイルを使って計測専用 API を起動し、保存済みの設定は書き換えません。結果は `%LOCALAPPDATA%\TinyStrata-work\benchmarks\` に保存します。生成したログや結果は Git に含めません。

Clover のアプリの既定はドラフトを使わない因果生成 D0・INT8 KV・`STRATA_IQ_MT_MIN=1` で、exact D3 は任意選択です。アプリで選ぶ生成経路と固定ベンチ規約は別で、標準ベンチはアプリのプロファイルにかかわらず、**INT8・因果生成・D0** を使います。LowConfidence 方式のネイティブなブロック拡散は別々に測定します。

```powershell
& $python .\benchmark\run.py --model clover --clover-mode causal
& $python .\benchmark\run.py --model clover --clover-mode diffusion --clover-block-size 32 --clover-threshold 0.95
```

前者は **`tinystrata-bench-v1` の因果生成の基準値**です。後者は入力・反復数・サンプリング・計時範囲を同じにし、**`tinystrata-bench-v1-clover-diffusion`**・`standard: false` として保存します。因果生成の v1 結果とは混同せず、ブロックサイズと信頼度の閾値も結果へ記録します。生成方式が異なるため、出力トークン列の完全一致は求めません。思考を含む、最終的に返したトークンを一度ずつ数えます。マスク位置の予測回数や拡散の反復回数は出力数に加えず、その実行時間はデコード時間に含めます。

両方式とも Clover のモデル別サンプリング設定を維持します。本規約は greedy 生成を測るため、確率的にサンプリングする場合の結果を示すものではありません。

`--clover-block-size` は **1・2・4・8・16・32**（既定 **32**）、`--clover-threshold` は **0〜1**（既定 **0.95**）を指定できます。未確定のマスク位置のうち、信頼度が閾値を上回る位置を埋め、該当位置がなくても1回の反復で最低1位置は確定させます。どちらもブロック拡散の設定です。native エンジンと `run.py` の従来の `--clover-draft 0` は因果デコード、正のドラフト数は因果検証を使う別方式の自己投機デコードを選びます。ベンチでは生成方式を明示します。

自己投機デコードも別枠の variant として測定できます。exact を対照条件とし、閾値採用は実験的な明示選択として扱います。

```powershell
& $python .\benchmark\run.py --model clover --clover-mode self-spec --clover-draft 3 --clover-accept exact --engine-env STRATA_IQ_MT_MIN=1
& $python .\benchmark\run.py --model clover --clover-mode self-spec --clover-draft 3 --clover-accept threshold --clover-accept-ratio 0.9 --clover-accept-delta 0.6 --engine-env STRATA_IQ_MT_MIN=1
```

前者は因果検証が選んだトークンと一致するドラフトだけを採用し、後者は実験的な閾値採用を測ります。どちらも **`tinystrata-bench-v1-clover-self-spec`**・`standard: false` とし、v1 の入力・サンプリング・反復数・計時範囲を維持します。このベンチスクリプトでは `--clover-draft` は **1〜31**、自己投機を選んだ場合の既定は **3**、`--clover-accept` の既定は **exact** です。ratio と delta は有限の **0〜1**、既定は **0.9** と **0.6** で、自己投機のときだけ指定できます。採用規則の比較では各値を明示してください。

任意選択の exact D3 プロファイルは `STRATA_IQ_MT_MIN=1` を使い、CPU の IQ エキスパートを1行で計算するときと複数行で計算するときの積和経路をそろえます。同じ演算条件の因果対照には、causal のコマンドにも `--engine-env STRATA_IQ_MT_MIN=1` を付けて別枠として測定してください。標準の因果生成 v1 は、アプリの保存済み設定にかかわらず、調整用環境変数なし（`env: {}`）を維持します。

Clover の既定である **GPU 常駐 Vision＋因果 D0** と、任意選択の **exact D3** は、次の別枠の条件でそれぞれ測ります。

```powershell
& $python .\benchmark\run.py --model clover --clover-mode causal --engine-env STRATA_IQ_MT_MIN=1 --clover-vision
& $python .\benchmark\run.py --model clover --clover-mode self-spec --clover-draft 3 --clover-accept exact --engine-env STRATA_IQ_MT_MIN=1 --clover-vision
```

`--clover-vision` は既存の Qwen3.6 プロジェクタを再利用し、すべてのテキスト計測中も画像エンコーダーを常駐させ、Qwen・Gemma と同じ規定の画像説明を追加します。画像は最小 1,024／最大 4,096 トークン、結果には `vision: true` を記録します。プロトコル名はそれぞれ **`tinystrata-bench-v1-clover-causal-variant-vision`** と **`tinystrata-bench-v1-clover-self-spec-vision`**、どちらも `standard: false` とします。因果方式の結果には調整用環境変数により `-variant` が付きます。拡散・閾値採用との併用はできません。固定 v1 の因果対照は引き続き画像なしです。アプリの既定変更によって過去の比較規約を書き換えることはありません。

`threshold` は temperature 0 で使う近似的な採用規則で、確率分布を保存する投機サンプリングではありません。1 ラウンドの preview・検証の全行で、出力済み anchor を含む応答トークン数の履歴を固定し、未確定ドラフトを追加せずに本体の確率とエントロピーから採否を決めます。因果生成や exact と同じトークン列は保証しません。本体が選んだ EOS と採用されたドラフトの EOS は、いずれも応答終了として扱います。最終出力は各トークンを一度だけ数え、preview・因果検証・転送・採用判定もデコード時間に含めます。実際のエンジン設定・採用数／提案数・転送量・カーネル設定を保存し、品質は速度と分けて報告してください。

Clover 専用の調整項目として、`--kv-type int8|fp8-mma`、`--cpu-workers N`（0 は自動）、`--chunk auto|N`、繰り返し指定できる `--engine-env STRATA_NAME=VALUE`／`TINYSTRATA_NAME=VALUE` も使えます。例：

```powershell
& $python .\benchmark\run.py --model clover --clover-mode diffusion --kv-type fp8-mma --cpu-workers 9 --chunk 1024
```

いずれかを指定すると、因果生成の場合も **variant（条件を明示した別枠の測定）** として保存します。保存済みのモデル設定は変更しません。指定条件を記録し、実際の生成方式・KV・ブロック・閾値・明示したワーカー数とチャンクが一致することを確認します。エンジンが返す実ワーカー数・総スレッド数・ブロック数・反復数・preview／replay 時間も記録します。因果生成とブロック拡散の比較では同じ調整値を指定してください。調整指定のない既定コマンドは、INT8・自動設定・因果生成の v1 基準値のままです。

Clover で思考なしの比較を別途行う場合は、`--thinking off` を追加します。既定は `--thinking on` です。無効にすると因果生成でも **variant** とし、`thinking: false` を記録して、ウォームアップ・トークン数の確認・すべてのテキスト生成へ適用します。規定の短文本文・サンプラー・seed は同じですが、チャットテンプレートが変わるため短い入力のトークン数は変わります。長文入力は引き続き正確に 4K／16K に合わせます。保存済み設定や標準の v1 条件は変更しません。

Clover の長文比較には **`--include-64k`** を追加できます。

```powershell
& $python .\benchmark\run.py --model clover --include-64k
& $python .\benchmark\run.py --model clover --include-64k --clover-mode self-spec --clover-draft 3 --clover-accept exact --engine-env STRATA_IQ_MT_MIN=1
```

各起動で、短文生成 6 回・4K・16K の後に、**テンプレート込みで正確に 65,536 入力トークン**、既存の**出力上限 32 トークン**の新規入力を 1 回追加します。ウォームアップ・入力本文・サンプラー・seed・思考設定・既定の 4 回起動は同じです。起動時のコンテキスト容量が **65,568 トークン**未満なら、生成リクエストを送る前に停止します。結果は標準 v1 と区別し、プロトコル名に **`-64k`**、メタデータに `include_64k: true` を付け、64K の入力処理速度と入力後の生成速度を実出力数で集計します。16K 時点のキャッシュ記録は維持し、64K 後の Expert キャッシュ・実際に割り当てた KV 領域も別途記録します。フラグなしでは v1 の測定内容と報告項目を変更しません。`--model clover` の場合だけ指定できます。

### Clover の反復 lookup（任意）

ランチャーの `--clover-lookup N`／API の `clover_lookup: N` は、提案するトークン数の上限 **0～7** を指定します。一致を探す suffix の長さではありません。**既定は 0（無効）のまま**です。因果 D0・温度 **0** が必要で、画像にも対応します。ソースのチェックアウトから起動する例：

```powershell
& $python .\run.py --model-family clover --clover-mode causal --clover-draft 0 --clover-lookup 3
```

入力・履歴にある続きをコピーして候補とし、同じ生成ペナルティを適用した因果モデル全体で検証します。一致区間が **8 トークン未満**なら内部で**最大 3 トークン**に抑え、指定した幅 **4～7** を使うには **8 トークン以上**の一致が必要です。D3 の preview・Expert の省略・閾値採用は使いません。出力には実際に返したトークンだけを数え、不採用の提案は処理時間に含めます。コピーできる内容によって効果が変わるため、使う入力で無効時と **3**・**7** を比較してください。標準 v1 は lookup を有効にせず、[lookup の測定](results/clover-lookup-20261107.md)は別の実験として扱います。浮動小数点演算のバッチ化で僅差の候補が変わる可能性があるため、検証入力でのトークン ID 一致を、すべての入力でのビット一致保証とはしません。

### Qwen・Gemma の lookup と自動 MTP（任意）

ランチャーの `--lookup-draft N` は、一致を探す長さではなく提案トークン数の上限 **0～7**、実験的な `--mtp-auto 0|1` は自動 MTP の有効／無効を指定します。API のリクエストごとの指定は `lookup_draft: N` と `mtp_auto: 0|1` です。**どちらも既定は 0（無効）**で、既定の **MTP 3** と温度（**Qwen 0.7／Gemma 1.0**）は維持します。自動 MTP は実測したラウンドの処理時間から D0・D1・設定した最大幅を選びます。一律に高速化できないため、実験的な任意選択です。

lookup は入力・履歴にある続きを候補にします。採用には、既存の主モデル全体が通常のサンプラーと同じペナルティで選んだトークンとの一致が必要で、通常の温度設定と画像にも対応します。選ばれた Expert はすべて計算し、近似採用や末尾の省略は行いません。実際に出力したトークンだけを数え、不採用の検証時間も生成時間に含めます。浮動小数点演算のバッチ化で選択が変わる可能性はあるため、検証入力での一致をすべての入力でのビット一致保証とはしません。

Gemma の検証用の容量は起動時に確保します。必要な幅はランチャーの `--lookup-draft N` で指定してください。リクエストごとの幅は、起動時の容量（`Engine.max_window() - 1`）が上限です。通常の MTP 3 で起動したサービスへリクエストだけ 7 を指定しても容量は拡張しません。Qwen の native 検証器は 8 行（最大 7 提案）ですが、INT8 FDEC 経路では lookup を最大 3 提案に抑えます。指定幅は上限であり、毎回その幅を使うとは限りません。

ソースのチェックアウトから起動する例：

```powershell
& $python .\run.py --model-family qwen36 --lookup-draft 3
& $python .\run.py --model-family gemma4 --lookup-draft 7
```

まず同じ保存済みプロファイルで lookup 単独を無効時と比較し、その後 `--mtp-auto 1` も個別に試してください。併用が速いとは限りません。[開発用の条件を揃えた測定](results/qwen-gemma-speculation-20261107.md)は **tinystrata-bench-v1** と区別し、過去の表や既定値は変更しません。

### CPU ワーカー数の調査（診断用）

通常の API を停止してから、Qwen または Clover を [probe.py](probe.py) で調べます。

```powershell
& $python .\benchmark\probe.py --model clover
```

ワーカー数と明示したオプションを除き、保存済みプロファイルの native 引数・調整用環境変数・Expert レイアウト・入力チャンクを維持します。サンプリングは比較用の温度 0 とし、専用サービスだけを起動・終了します。設定は自動保存しません。数値は **バックグラウンドワーカー数**で、呼び出し元も計算するため `--cpu-threads 7` は合計 8 スレッドです。全 Expert が GPU に収まっている場合は、調査対象となる CPU 経路がないと表示します。

候補は 2 の累乗、自動選択数の半分、自動選択数です。順序を反転した選別試験で、最速の計時から 2% 以内にある最小の数を選びます。改善は基準値のばらつきと最低 3% を上回る必要があります。短文 3 種類と正確な 4K／16K 入力による新しい ABBA／BAAB 確認で、回答・思考のハッシュ、native の各トークン数、Expert の実容量が一致し、長文で明確に悪化しない場合だけ手動変更用の `--cpu-threads N` を提案します。結果は標準 v1 とは別の **`tinystrata-worker-probe-v1`** で、提案は記録したモデル・条件に限ります。

`--prepare-only --default-workers N` はサービスを起動せずに計画を保存します。`--expert-gib 4` は Expert の予算を明示的に 4 GiB にします。省略時はプロファイルの予算を維持します。`--chunk N` は 1 つのチャンクサイズを固定するだけで、チャンク探索や入力処理の CPU／GPU 割合調整は行いません。

### 共通設定

モデルはプロジェクト指定の Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS、Google Gemma 4 26B-A4B-it QAT Q4_0 と対応する画像プロジェクタ・QAT MTP アシスタント、LiquidAI LFM2.5-24B-A2B Q4_K_M、[AtomicChat Clover-35B-A3B-Preview IQ3_XXS](https://huggingface.co/AtomicChat/Clover-35B-A3B-Preview-GGUF/blob/4f7e2918e5315634d9d9720ea4f6e8a9a627a875/Clover-35B-A3B-Preview-AD-IQ3_S-IQ3_XXS.gguf)（`Clover-35B-A3B-Preview-AD-IQ3_S-IQ3_XXS.gguf`、エキスパートの gate/up は IQ3_XXS、down は IQ3_S）を使います。過去の Clover Q4_0 の計測は別途 Q4_0 と明記し、この IQ3_XXS プロファイルの結果として扱いません。モデルとバイナリのハッシュ、TinyStrata のバージョン／コミット、GPU・搭載 VRAM、CPU、RAM、OS、ドライバーを結果に併記します。

- NVIDIA・Radeon 共通の比較は **INT8 KV**。FP8 や手動調整を加えた結果は別枠です。
- KV は段階確保。コンテキスト容量・入力チャンク・Expert 配置・CPU ワーカー数は自動選択、Graph は既定の動作です。高速化用の環境変数や Expert／トークンの省略は追加しません。
- Qwen・Gemma は **MTP 3 と GPU 常駐 Vision** を使用します。テキストの計測でも画像エンコーダーを常駐させます。LFM はどちらも使いません。固定 v1 の Clover 対照は MTP・Vision なし、因果経路（`--clover-mode causal`、D0）で測り、アプリ既定の因果 D0＋Vision と任意選択の exact D3＋Vision は上記の別枠で測定します。
- 同時実行は 1 リクエスト、**seed は 1234**。サンプリングは下表で固定します。既存 v1 入力のハッシュを維持するため、追加した Clover の設定は `run.py` に、他のモデルは `cases.json` に定義します。

| モデル | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6 | 0.7 | 0.95 | 20 | 0 | 1.5 | 1.0 |
| Gemma 4 | 1.0 | 0.95 | 64 | 0 | 0 | 1.0 |
| LFM2.5 | 0.1 | 1.0 | 50 | 0 | 0 | 1.05 |
| Clover | 0 | 1.0 | 64 | 0 | 0.9 | 1.05 |

frequency_penalty は 0。生成・4K・16K は思考有効、画像説明は説明本文の開始時間を測るため思考無効です。早期 EOS はそのまま扱い、出力上限まで生成したと仮定せず、実生成トークン数を使います。

### 4 つの計測

| 項目 | 固定する入力と出力上限 | 掲載する値 |
|---|---|---|
| 生成速度 | `cases.json` の `generation` 3 入力を順に各 2 回。各リクエスト最大 128 トークン、思考を含む | 総生成トークン数 ÷ 総デコード秒数（tok/s） |
| 4K 入力 | `long_paragraph` と `long_question` から作る文書を、**テンプレート込みで 4,096 入力トークン**に調整。出力上限 32 | 新しく読んだ総入力トークン数 ÷ 総入力処理秒数（tok/s） |
| 16K 入力 | 同じ方法で **テンプレート込み 16,384 入力トークン**。出力上限 32 | 同じ入力処理速度（tok/s） |
| 画像説明 | 無加工の **1,024 × 768** `image.png` と `image_prompt`。出力上限 128。API 起動後、最初の実画像リクエスト | 最初の出力まで／説明完了までの秒数の中央値。LFM と固定 v1 の Clover 対照は対象外。`--clover-vision` で Clover を追加 |

MTP を含む通常の Messages API を使い、native の `--tokens` 生成ループでは測りません。4K・16K は同じリクエストを `/v1/messages/count_tokens` で数えて作成し、実際のエンジンの入力数でも確認します。テキスト・画像のハッシュで入力を識別し、各モデルの実トークン数で長さを確認します。

### 繰り返しと計時範囲

**各モデルで API を 4 回独立に起動**し、毎回次の順序を守ります。

1. 読み込みと組み込みウォームアップの完了を待つ。Qwen・Gemma は GPU 画像エンコーダーの稼働を確認し、起動時の Expert キャッシュ量を記録する。
2. 別の `warmup_text` を出力上限 32 で 1 回送る。このリクエストは集計しない。
3. 生成用の 3 入力を 2 周する。
4. 4K 入力を 1 回、16K 入力を 1 回測り、16K 終了直後の Expert キャッシュ量を記録する。
5. Qwen・Gemma は画像説明を 1 回測る。次の周回では **API サーバー全体を再起動**し、画像エンコード結果のキャッシュを引き継がない。

合計で、各モデルの生成は **24 リクエスト**、4K・16K は各 4 リクエスト、Vision 対応モデルの画像説明は 4 リクエストです。テキストは会話を継ぎ足さず独立した単発メッセージとして送り、`reused == 0` と `prompt_read == prompt_total` を確認します。4K・16K は新規入力数がそれぞれ 4,096／16,384 であることも確認します。

テキストの速度は `/metrics` のエンジン実測値から計算します。

```text
生成 tok/s = 1000 × Σ engine_generated / Σ decode_ms
入力 tok/s = 1000 × Σ prompt_read       / Σ prompt_ms
```

各回の tok/s の単純平均や最速値の選別はしません。モデルの読み込みと別枠のウォームアップは除外します。各リクエストに `strata_checkpoint: false` を指定し、会話キャッシュによって繰り返し入力がヒットしないようにします。失敗・中断・KV 再利用・入力数違いの試行を黙って除外せず、未完了として報告するか、原因を直して規約の全手順を測り直します。

画像はクライアントの単調増加時計で POST 直前から測ります。最初の出力は空でないモデルの文字列が届いた時点、完了は `message_stop` です。HTTP ヘッダーや keep-alive は出力に数えません。アップロード・画像の読み込み／エンコード・入力処理・生成を含み、サーバーとモデルの起動は含みません。エンジンの `prompt_ms` は画像エンコードを含まないため、画像の秒数の代わりには使いません。説明文と実出力数も保存します。

### Expert キャッシュと表の備考

各ベンチの前後の `expert_cache_bytes` を保存します。`expert_cache_mib` だけを返すエンジンでは 1 MiB = 1,048,576 バイトで換算します。表の備考には **起動時 → 16K 入力直後**を **GiB（バイト数 ÷ 1,073,741,824）** で載せます。4 回の起動で差があれば最小～最大の範囲を示し、個々の試行も JSON に残します。表記例は `Expert cache 9.21 → 9.04 GiB` です。

これは GPU に確保した Expert キャッシュの量であり、モデルファイルの容量や他アプリの使用量の推定値ではありません。KV・作業領域・予備領域の調整でも実行中に変わります。その後の短い画像リクエストで再び増える場合があるため、矢印の後ろは 16K 終了直後の値にします。他アプリのメモリ使用量は収集しません。

| モデル | コンテキスト | 生成 tok/s | 4K 入力 tok/s | 16K 入力 tok/s | 画像 最初／完了 秒 | 備考 |
|---|---:|---:|---:|---:|---:|---|
| … | … | … | … | … | … / … | Expert cache … → … GiB |

比較には同じ規約・入力ハッシュ・モデルファイル・KV 形式を使い、ハードウェアと実際に自動選択されたコンテキスト・チャンクも併記します。VRAM を擬似的に制限した結果は物理 GPU の結果と分けます。都合のよい 1 回ではなく、完了した全試行の集計を公開します。
