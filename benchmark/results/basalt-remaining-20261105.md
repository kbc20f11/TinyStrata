# Remaining Basalt ideas: measured adoption — 2026-11-05

## English

This update retains a smaller upload/direct combine for eager CPU-produced Expert
rows and adds a manual CPU-worker probe. GPU route/product fusion and fractional
prefill CPU/GPU scheduling were implemented and measured, but did not establish a
useful speed improvement and were removed.

These are controlled development comparisons, **not standard
tinystrata-bench-v1 results**. They supplement the existing measurements and do
not replace historical README benchmark tables. The frozen baseline already
contains the previous CPU phase fusion, checkpoint-buffer reuse and Clover GDN
improvements; their gains are not counted again here.

### Source, binaries and machine

- Frozen source: [`2e51521df4df3548ccefc989b90ee2ea9df07247`](https://github.com/kbc20f11/TinyStrata/commit/2e51521df4df3548ccefc989b90ee2ea9df07247).
- Baseline/worker-probe executable SHA256:
  `1f3ca9d253aca7afac7e6b16c7afca6e1fbe1b22f8f3cf7cb7b94751dd41778c`.
- Retained-kernel performance comparisons use the **same candidate-4 executable
  on both sides**, SHA256
  `1ced3f97444ef4b5c2c4db9712ce291c32ee548224122681aaed4a7fabb2e1b7`,
  with `TINYSTRATA_Q36_CPU_COMPACT=0/1`. This isolates the new upload/combine from
  changes between separate builds.
- Final candidate-5 executable SHA256:
  `5d864f0df103742e79f6a0023af4bb37cd56d16dcc425afa1d3e21ee7df22fac`.
  It retains candidate 4's arithmetic and makes the path automatic only on CUDA
  compute-capability major 12. The timing table below is the candidate-4 A/B
  measurement; final-binary quality checks are reported separately.
- Windows x64; RTX PRO 4000 Blackwell SFF, 24 GB, 70 W, 70 SMs, 48 MiB L2;
  driver 597.30; Ryzen 9 7945HX3D; 64 GB RAM; CUDA 13.2.86 / MSVC 14.51.

The native source HEAD and an executable hash identify different things: the
uncommitted development candidates are pinned by executable SHA256, rather than
being attributed to the frozen baseline commit alone.

### Retained change: compact eager CPU results

The previous eager hybrid path uploaded an output row for every routed entry,
then launched a scatter kernel to replace the CPU entries in the GPU output
array. The retained path packs only CPU-produced FP32 rows, uploads their
entry-to-row map, and lets a separate combine kernel read those rows directly.
CPU job/input order and each Expert's arithmetic remain unchanged. The new
combine preserves the legacy reduction, residual, normalization and Q8 activation
arithmetic; only the initial source-pointer selection differs. The original
combine kernel remains unchanged.

Automatic activation is limited to CUDA Blackwell, eager hybrid batches of
1–8 rows, hidden width 2,048, top-8 routing and the existing fast combine.
Mapped hybrid Graph replay/capture, preview, numerical debug verification, other shapes,
HIP and other NVIDIA generations retain their existing path. All-GPU execution
has no CPU rows to optimize. `TINYSTRATA_Q36_CPU_COMPACT=0` selects the legacy
upload/scatter path. Normal MTP/causal verification is eligible when the same
small-batch contract holds; it is not the numerical-debug exclusion.

Each model has fresh **ABBA and BAAB**, eight starts and **40 measured requests**:
three short fixtures capped at 256 output tokens, then exact 4,096/16,384-token
inputs capped at 128 output tokens per start. One identical warm-up precedes
measurement. Temperature is 0, seed 1234, thinking is enabled, prefix reuse is
zero, context capacity is 262,144 and chunk size is explicitly 2,048. All runs
hold **15 background CPU workers plus the participating host**; the worker-14
experiment below is separate. The image encoder remains resident.

The nominal Expert budget is manually fixed at **4 GiB**. Actual allocation,
constant at startup/completion and across peers, is:

| Model / settings | GPU Experts / total | Expert payload bytes | Expert arena bytes |
|---|---:|---:|---:|
| Clover IQ3, FP8 MMA KV, causal D0/MTP0, EXP_MODE8/IQ_MT_MIN1 | 2,760 / 10,240 | 3,570,336,000 | 3,570,499,840 |
| Qwen3.6 IQ3_XXS, INT8 KV, MTP3, EXP_MODE0 | 3,649 / 10,496 | 4,254,761,056 | 4,254,928,992 |

Throughput is pooled **final native generated tokens / decode time**, including
thinking; speculative proposals are not counted as generated output. Percentages
use unrounded pooled values. Each order's three short fixtures are pooled.

| Model / workload | Legacy | Compact | Pooled change | ABBA change | BAAB change |
|---|---:|---:|---:|---:|---:|
| Clover, short decode | 52.4233 tok/s | 53.1299 tok/s | +1.348% | +0.917% | +1.782% |
| Clover, decode after 4K | 49.0816 tok/s | 50.3130 tok/s | +2.509% | +3.939% | +1.090% |
| Clover, decode after 16K | 48.8666 tok/s | 49.7522 tok/s | +1.812% | +2.787% | +0.846% |
| Qwen, short decode | 91.1597 tok/s | 92.1189 tok/s | +1.052% | +1.659% | +0.458% |
| Qwen, decode after 4K | 91.8864 tok/s | 94.1817 tok/s | +2.498% | +1.055% | +3.963% |
| Qwen, decode after 16K | 90.7254 tok/s | 92.4038 tok/s | +1.850% | +1.962% | +1.738% |

Both orders have positive decode point estimates in every category. These are
small measured gains, not proof of statistical significance or a promise for
other hardware. Legacy pooled rates drift between the two orders by
0.015%/1.998%/1.065% for Clover and 0.886%/1.341%/0.797% for Qwen
(short/4K/16K). The per-order values expose this variation rather than hiding it
inside one pooled number.

| Prefill, prompt tokens / native prompt time | Legacy | Compact | Change |
|---|---:|---:|---:|
| Clover, 4K | 2,877.870 tok/s | 2,875.698 tok/s | -0.075% |
| Clover, 16K | 2,636.924 tok/s | 2,634.454 tok/s | -0.094% |
| Qwen, 4K | 804.265 tok/s | 805.558 tok/s | +0.161% |
| Qwen, 16K | 779.961 tok/s | 790.544 tok/s | +1.357% |

No bulk-prefill speedup is claimed: this change targets small eager CPU results,
and the prefill points include ordinary measurement variation.

All **80 requests** have corresponding request bodies, answer/reasoning hashes,
native generated/output counts, prompt totals/reads, reuse, offered/accepted
draft counts and finish fields equal across arms and orders. Response hashes do
not establish identity of raw native token arrays or full logits. The four saved
model-profile SHA256 hashes are unchanged, and the previously active production
Clover model is restored after the controllers.

For every enabled start, aggregate counter prefixes are identical at each saved
request boundary. The following deltas are **final minus startup**, including
the owned warm-up after startup. Each counter prefix satisfies
`unpacked_bytes = entries * (2048 + 1) * 4` and
`upload_bytes = CPU_rows * 2048 * 4 + entries * 4`; avoided merges equal calls.
Disabled-arm counters are zero.

| Model, per enabled start | Calls / avoided merges | CPU rows | Queued compact bytes | Counterfactual unpacked bytes | Reduction |
|---|---:|---:|---:|---:|---:|
| Clover | 40,886 | 173,472 | 1,422,390,976 | 2,680,813,248 | 46.942% |
| Qwen | 14,276 | 157,593 | 1,292,697,600 | 3,474,579,456 | 62.796% |

These are queued FP32 payload/map bytes for the same compact-eligible calls,
not measured physical PCIe traffic or a whole-request transfer reduction.
Runtime reports do not contain raw row maps; the standalone poisoned-buffer
test checks the map/combine results directly.

### Retained tool: manual environment-specific worker probe

[`benchmark/probe.py`](../probe.py) preserves the selected model profile's raw
native arguments, operational environment, layout, KV/context/chunk and Expert
capacity unless explicitly overridden. It varies the background-worker count,
uses fixed temperature-0 fixtures, checks actual count plus host participation,
and never writes a recommendation to saved settings. All-GPU placement is
reported as inapplicable to CPU-worker tuning. The tool currently supports
Qwen/Clover; it does not claim to search chunk sizes or fractional prefill share.

The initial fixed-4-GiB screens cover AUTO/1/2/4/7/8/15/AUTO, followed by
AUTO/12/14/AUTO, **120 measured requests total**. Reducing to 1–8 workers loses
decode throughput on this CPU. Nearby 14-worker point estimates justify a
confirmation, not a universal 14-worker default. The observed automatic choice
remains 15 workers plus host. Other machines must measure their own settings.

The real public-tool run explicitly uses `--workers 14,15 --minimum-gain 0.01`
and the 4-GiB budget, retaining Clover's FP8/EXP_MODE8/IQ_MT_MIN1/chunk2048
profile. It performs **21 mirrored-screen requests**, then **40 fresh ABBA/BAAB
confirmation requests**, all with exact response/count/settings checks.

| Clover confirmation | AUTO 15 workers | 14 workers | Change |
|---|---:|---:|---:|
| Pooled decode across measured workloads | 53.8390 tok/s | 55.4327 tok/s | +2.960% |
| Three short fixtures | 54.4584 tok/s | 56.3424 tok/s | +3.459% |
| Decode after 4K | 50.0939 tok/s | 50.4414 tok/s | +0.694% |
| Decode after 16K | 50.7091 tok/s | 50.6329 tok/s | -0.150% |

The pooled AUTO variation band is 2.054%; order gains are +3.961% and +1.969%.
4K/16K prefill changes are +0.131%/-0.218%, within baseline variation. The tool
therefore issues a **manual, recorded-profile-only** `--cpu-threads 14`
recommendation under the explicitly relaxed 1% floor. The pooled +2.960% would
**not pass the unchanged default 3% floor**. No global worker default is changed,
and this result is not multiplied into the compact-kernel table: they are
separate experiments, with different output caps and binaries.

The 61-request runtime pins tool SHA256
`600a27498329d55f7122991db04dff14a4bb3b77a2e37fe8ea4ae91b3614e6ff`,
source checkout HEAD `2e51521...`, and `source_worktree_dirty=true`.
Afterward, the final tool adds only `q36_cpu_compact` to static peer-geometry
validation; its SHA256 is
`6e47fef63082cf11691a51870815a599af362f804027dad64418f1534f5284e6`.
The runtime evidence is attributed to the executed tool, not relabeled as the
later source. See [usage and admission rules](../README.md).

### Implemented and rejected

| Candidate | Measured finding | Decision |
|---|---|---|
| GPU route/preparation + Expert GU fusion | Cooperative-grid and CTA-local schedules, 70/140/280 CTAs; all-GPU Clover and Qwen, 48 exact-response requests. Clover point estimates range -1.506% to -0.325%; Qwen -3.532% to +0.332% versus bracketed controls. Each enabled setting has one start. | No convincing end-to-end win; both prototypes removed. Correct arithmetic or fewer launches alone is insufficient. |
| Fractional native-prefill CPU/GPU scheduler | Numerical CPU-path oracles pass; packed-result screens test Qwen fractions 0.25/0.45/0.65/1.0 and Clover 0.3. Qwen 0.65 is +1.316% at 4K but -34.960% at 1K; its 4K controls drift 2.376%. Clover 0.3 loses 16.548%/8.389% at 1K/4K. | No robust prefill win; scheduler and its option removed. |

The prefill partition assigns a fixed fraction of nonresident Expert IDs; it is
not Basalt's few-row-first partition and is not the `--hybrid-rows` threshold.
Its CPU arithmetic is validated against the existing CPU-hybrid route, not
assumed bit-identical to the GPU route. Its permanent-staging Qwen experiment
holds 2,952 Experts / 3,442,053,888 payload bytes, whereas the real stage-loan
worker/compact configuration holds 3,649 / 4,254,761,056. Clover's native stream
control is also much slower than its actual MMQ prefill. Endpoint/fraction
screens cannot be advertised as matched-capacity improvements over the real
model default. Reducing requested copy bytes did not overcome these costs.

The first unpacked prefill endpoint test also loses at both lengths/model;
Clover's 4K output changes across CPU/GPU partitions. Later packing preserves
the CPU arithmetic but supplies no speed-admission win. These are limited
numerical/scheduling experiments, not broad model-quality benchmarks.

### Quality, compatibility and evidence status

| Check | Status |
|---|---|
| [Compact combine unit](../../tests/q36-cpu-combine.cu) | **PASS: 1,620** eager/Graph checks. Independent poisoned arms; all/no/mixed CPU entries, multiple row patterns, residual/norm/Q8 and inactive/guard checks; sampled FP64 weighted oracle. |
| [Worker-probe offline tests](../../tests/test_benchmark_probe.py) | **PASS: 16**; count selection/noise, real server-config normalization, full-controller logic, profile preservation and partial-constructor owned-child cleanup. |
| Final candidate-5 native build | **PASS**. |
| Final candidate-5 full-vocabulary logits | **PASS: four cases**, Qwen/Clover × INT8/FP8, 17 × 248,320 FP32 logits each: **16,885,760 floats total**. Raw bytes and greedy IDs match, maximum absolute difference 0. One GPU Expert per layer forces extreme CPU offload; 15 background workers, MTP0. Automatic-path counters are positive and saved profiles are restored. |
| Final candidate-5 checkpoint/edit/cancellation suite | **PASS: 31 paired cases / 62 executions**, baseline and candidate each run 11 Qwen INT8, 11 Qwen FP8 and nine Clover FP8 cases. All 20 successful pairs have equal raw native token IDs; case counts, reuse/read/resume, draft counts and finish fields match. |
| Final candidate-5 real-image regression | **PASS: eight requests / four starts**, two repeats per model/build, FP8, fixed 4-GiB Expert capacity, context 32,768, chunk 512, temperature 0, thinking off. Answer/reasoning hashes and native counts match; Qwen uses MTP3, Clover causal D0. Each image encoder keeps its identity and candidate compact counters are positive. |
| Final candidate-5 Clover exact-D3 smoke | **PASS: three paired fixtures**, six main native requests plus six unrelated-prefix reset requests. FP8, fixed 4-GiB Expert capacity, context 32,768, chunk 512, 15 workers plus host, no Vision. Raw token IDs and generated/read/reuse/draft/finish counts match; automatic compact execution is positive. |
| Final CUDA SM86 / HIP gfx1200 compile checks | **PASS**: SM86 engine + focused unit and HIP gfx1200 engine compile, exit 0. These are compile checks, not runtime or speed measurements on those devices. |

The earlier CPU-prefill scheduler separately passed 17 full-vocabulary rows per
model (248,320 FP32 logits/row), byte-identical to its matched CPU-hybrid oracle.
That gate belongs to the rejected prefill prototype, not to the final compact
candidate's separate quality checks above.

The state suite uses fixed 4-GiB Expert capacity, context 32,768, chunk 512 and
15 workers plus host. Qwen runs MTP3 and exercises accepted drafts, cancellation
inside synthetic image embeddings and a fresh image retry; Clover is causal
D0/MTP0. Checkpoint-matching retry and edited suffixes reproduce their cold
token IDs. Queued cancellation and the following request pass. Candidate
automatic compact counters are positive and satisfy the upload/map accounting;
baseline counters are zero, static geometry matches, and profiles are restored.
These are state-management checks, not semantic-answer or general vision tests.

The real-image fixture is [the benchmark image](../image.png). Manual inspection
finds the same red A square, blue B circle, green C triangle and bottom caption
in both builds' descriptions. Clover also reads `TinyStrata` and
`LOCAL AI - 3 SHAPES`; Qwen omits the heading in both builds. This is one
unchanged-output image regression, not general vision/OCR approval. It compares
rendered responses and native counts, not raw API token arrays.

The D3 fixtures report accepted/offered drafts of 23/23 (`repeat32`, 32 outputs),
3/6 (`ascending8`, six outputs, stop), and 17/35 (`mixed16`, 32 outputs), identical
before/after. Unrelated-prefix resets ensure cold reads. The candidate performs
1,120 compact calls/avoided merges during this smoke run, retaining the existing
preview fallback while normal small causal verification can use the compact
combine. This is an exact-state/ID compatibility check, not a D3 speed or broad
semantic-quality benchmark. Saved profiles remain unchanged and the root
controller restores production after the final image/D3 pipeline.

Completed evidence is retained privately in `*-cpu-compact4` and
`*-cpu-compact4-reverse`, `public-probe-clover`, the completed worker screens,
`*-route-screen`, `*-native-prefill-endpoints`, `*-prefill-ratios3-v2` and their
independent audit JSONs. Raw outputs/configurations/binaries remain outside Git.
Excluded failed controller attempts are `clover-workers-full` (string chunk
rejected), `clover-workers-full-v2` (server rejected the worker alias) and
`qwen-prefill-ratios3` (expected-setting assertion before an enabled measurement).
Their partial/failed results are not included in the timing tables. Saved
profiles were restored; no rejected prototype is retained as a product option.

Reference: [Basalt `bbf2f5bb`](https://github.com/jesdga95/basalt/tree/bbf2f5bb783adc5ffe5b49043f82c9399313bb46),
MIT, Copyright (c) 2026 Jesús Gulfo;
[`docs/probe.md`](https://github.com/jesdga95/basalt/blob/bbf2f5bb783adc5ffe5b49043f82c9399313bb46/docs/probe.md),
`src/program/probe_derive.cpp`, `src/prefill/cpu_experts.cpp` and
`src/kernels/cuda/dec_moe.cuh`. Its ideas guide the experiments; its thresholds
or reported performance are not TinyStrata speed guarantees.

## 日本語

今回の採用は、CPUで計算したExpert出力だけを詰めて転送し、その行をcombineから
直接読む処理と、環境ごとにCPUワーカー数を測る手動probeです。GPUのルーティングと
積演算の統合、長文prefillのCPU/GPU分担も実装・検証しましたが、実用的な速度改善が
得られず、試作品とその設定項目は製品コードから外しました。

**これは標準v1の再測定ではありません。既存READMEの通常ベンチ表は変更しません。**
基準は `2e51521...`。既に入っているCPU phase fusionやGDN改善を再計上しません。
速度比較は同一candidate-4バイナリの新経路OFF/ONで、最終candidate-5は演算を保ったまま
CUDAのCC major 12で自動選択する方針を加えています。各バイナリのSHA256は英語節に
記載し、基準コミットのビルドと混同していません。

Expert予算4 GiB、コンテキスト262,144、chunk2,048、15ワーカー＋呼び出し側1、温度0、
再利用0、画像エンコーダ常駐を固定しました。実割当はCloverが2,760/10,240個・
3,570,336,000 bytes、Qwenが3,649/10,496個・4,254,761,056 bytesです。CloverはFP8・
因果D0、QwenはINT8・MTP3。短文256出力、4K/16K入力後128出力をABBAとBAABで測り、
各モデル40件、合計80件を比較しています。

| モデル | 短文生成 | 4K入力後の生成 | 16K入力後の生成 |
|---|---:|---:|---:|
| Clover | 52.4233→53.1299 tok/s（+1.348%） | 49.0816→50.3130（+2.509%） | 48.8666→49.7522（+1.812%） |
| Qwen | 91.1597→92.1189 tok/s（+1.052%） | 91.8864→94.1817（+2.498%） | 90.7254→92.4038（+1.850%） |

両測定順とも各項目で改善方向ですが、小さい点推定であり、他環境の改善や統計的有意差を
保証する値ではありません。入力処理の高速化は主張しません。出力に数えるのは思考を含む
確定生成トークンで、投機候補数を足していません。全80件で回答・思考テキストのハッシュ、
生成数、入力総数／読取数、再利用数、投機の提示／採用数、終了理由が一致しました。
これだけを根拠に生トークン配列や全logitsまで一致したとは扱いません。

対応する小バッチの要求転送量はCloverで46.942%、Qwenで62.796%減っています。
これはFP32行＋mapのキュー投入バイト数で、実測PCIe帯域や要求全体の転送削減率では
ありません。旧combineは保持し、mapped Graph・preview・デバッグ照合・対象外形状・
HIP・他世代は既存経路を使います。全ExpertがGPU常駐する場合の改善を示す測定でも
ありません。`TINYSTRATA_Q36_CPU_COMPACT=0` で旧経路を選べます。

ワーカーprobeはQwen/Cloverに対応し、保存済み設定を保ったまま少数候補を往復測定して、
必要な場合だけ新しいABBA/BAAB確認を行います。1～8ワーカーはこのCPUでは遅く、
12/14も追加確認しました。公開ツールの実動作は61件で検証済みです。明示的に判断基準を
1%へ下げたClover試験では、14ワーカーが全体+2.960%、短文+3.459%、4K後+0.694%、
16K後-0.150%。対照群の変動幅2.054%を上回り、長文の差も許容範囲なので、その記録条件に
限った手動推奨を出しました。**既定の3%基準なら全体+2.960%は通りません。14を全環境の
既定値にせず、保存設定も自動変更しません。** この値をcompact経路の改善率へ足すことも
していません。

棄却したGPU統合は計48件のモデル実行で有用な改善を確認できませんでした。prefillの
分担も、Qwenの一例で4Kが+1.316%でも対照群変動2.376%内かつ1Kは約35%低下し、
Cloverも遅くなりました。コピー削減や数値検証の通過だけで採用していません。

compact単体の1,620件とprobeのオフライン16件は通過し、最終nativeビルドも成功しました。
最終candidate-5はQwen/Clover×INT8/FP8の4条件、計16,885,760個のFP32 logitsで生バイトと
greedy IDが一致し、最大絶対差0でした。1層あたりGPU Expertを1個に絞った強いCPU
オフロード条件で、自動経路の実行も確認しています。SM86のengine＋単体試験、HIP
gfx1200のengineもコンパイル成功です。状態復元／編集／キャンセルは31組・計62実行を
通過し、成功20組の生トークンID、読取／再利用／resume、投機数などが一致しました。
QwenはMTP3の採用と合成画像埋め込み内のキャンセル後再試行も確認しています。
実画像は4起動・計8要求で回答／思考ハッシュとnative数が一致し、エンコーダも各起動中に
維持されました。両モデルとも図形・色・文字ラベル・下の説明を同じように認識し、
Cloverは見出しも読み、Qwenは旧新版とも見出しを省略しています。単一画像の回帰確認で、
画像認識・OCR全般の品質承認ではありません。

Cloverのexact D3も3組・主要求6件＋別prefixのリセット6件で、生トークンIDと入力／出力／
投機数が一致しました。採用／提示は23/23、3/6、17/35で、新経路の実行も正でした。
画像なし・Expert4 GiB・FP8での小規模な互換性確認であり、D3の速度や一般的な回答品質を
示すものではありません。保存設定は変えず、最終パイプライン後に稼働モデルを復元しました。
コンパイル成功をAMD/Ampere実機の速度・動作保証として扱いません。
通常のMTP検証は対象の小バッチなら新経路を使えます。
除外する「デバッグ照合」は `STRATA_Q36_VERIFY` の数値診断です。
