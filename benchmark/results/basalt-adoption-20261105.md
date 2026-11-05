# Basalt reference adoption results — 2026-11-05

## English

This update adds CPU Expert phase fusion, reuse of the Qwen/Clover turn-checkpoint buffer, and paired Q8_0 GDN projections for eligible Clover execution on Blackwell. The CPU and GPU changes retain the existing per-row arithmetic; checkpoint reuse avoids zero-filling a buffer immediately before a full overwrite. Broader CPU task grains, GPU wait-side read-ahead and the first paired-projection layout were rejected after measurement.

These are development A/B comparisons, **not standard tinystrata-bench-v1 results**. They supplement the README's historical measurements; they do not replace the model tables or establish new headline speeds for the default configuration.

### Source and machine

- Frozen TinyStrata baseline: [`4ef81f7`](https://github.com/kbc20f11/TinyStrata/commit/4ef81f721ec8f9041e297352ff63b577af0dea8a).
- Reference: [Basalt `bbf2f5bb`](https://github.com/jesdga95/basalt/tree/bbf2f5bb783adc5ffe5b49043f82c9399313bb46), MIT; Copyright (c) 2026 Jesús Gulfo. Its [CPU pool](https://github.com/jesdga95/basalt/blob/bbf2f5bb783adc5ffe5b49043f82c9399313bb46/src/kernels/cpu/pool.cpp), [checkpoint allocation policy](https://github.com/jesdga95/basalt/blob/bbf2f5bb783adc5ffe5b49043f82c9399313bb46/include/basalt/core/conversation_cache.hpp) and [projection-pairing design](https://github.com/jesdga95/basalt/blob/bbf2f5bb783adc5ffe5b49043f82c9399313bb46/docs/unified-kernel.md) informed these changes. Basalt targets Qwen3.8-Flash-Next and Blackwell; TinyStrata's model drivers and existing arithmetic are retained.
- Windows x64; **RTX PRO 4000 Blackwell SFF, 24 GB, 70 W power limit**, 70 SMs and 48 MiB L2; NVIDIA driver **597.30**; Ryzen 9 **7945HX3D**, 64 GB system RAM.
- CUDA **13.2.86**, Visual Studio 18 Community / MSVC **14.51**.
- Qwen: `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`. Clover: `Clover-35B-A3B-Preview-AD-IQ3_S-IQ3_XXS.gguf`.

These measurements are from this Blackwell PC. They do not establish performance gains on Ampere or Radeon, or for the separate Gemma/LFM drivers.

### Final selected bundle versus frozen baseline

This final comparison uses the frozen baseline versus the selected implementation, rather than enabling one candidate at a time. Qwen uses **FP8 MMA KV and MTP3**; Clover uses **INT8 KV, causal D0 and MTP0**. Each model has four fresh **A1/B1/B2/A2** API starts and 20 measured requests: the three short prompts at a 128-token cap, plus exact 4K/16K inputs with a 32-token cap, per start. Temperature is 0, context 262,144, automatic chunk selects 2,048, the GPU image encoder is resident, and `strata_checkpoint:false` prevents checkpoint-buffer reuse from affecting these one-shot throughput measurements.

Both builds receive the same manual **4-GiB Expert budget**. Startup and final geometry match throughout: Qwen **3,649 / 10,496 Experts, 4,254,761,056 payload bytes**; Clover **2,760 / 10,240 Experts, 3,570,336,000 bytes**. The selected build uses its default CPU fusion and automatic projection eligibility, with no forced CPU/pair environment override. Qwen reports pairing disabled with zero eligible layers; Clover reports automatic pairing enabled with 30 eligible layers. Clover retains the same `STRATA_IQ_MT_MIN=1` setting on both builds.

| Model / decode workload | Frozen baseline | Selected bundle | Change |
|---|---:|---:|---:|
| Qwen FP8: three-prompt generation group | 98.28387 tok/s | 101.13514 tok/s | +2.901% |
| Qwen FP8: after 4K input | 100.66059 tok/s | 100.77153 tok/s | +0.110% |
| Qwen FP8: after 16K input | 98.99459 tok/s | 101.34600 tok/s | +2.375% |
| Clover INT8: three-prompt generation group | 51.37192 tok/s | 52.66948 tok/s | +2.526% |
| Clover INT8: after 4K input | 45.09265 tok/s | 47.20112 tok/s | +4.676% |
| Clover INT8: after 16K input | 37.27866 tok/s | 38.68706 tok/s | +3.778% |

| Input processing | Frozen baseline | Selected bundle | Change |
|---|---:|---:|---:|
| Qwen FP8, 4K | 828.39519 tok/s | 824.42687 tok/s | -0.479% |
| Qwen FP8, 16K | 823.33313 tok/s | 822.47757 tok/s | -0.104% |
| Clover INT8, 4K | 2,836.07409 tok/s | 2,829.99965 tok/s | -0.214% |
| Clover INT8, 16K | 2,474.84970 tok/s | 2,477.09474 tok/s | +0.091% |

Rates are recomputed from pooled native token counts and times, with percentages based on unrounded values. All **40 responses** have matching answer/reasoning hashes and generated/prompt/draft counts for corresponding requests across starts; prefix reuse is zero. The comparisons preserve saved profiles and restore production Clover. These small ABBA point estimates include run variance, particularly the nearly flat Qwen 4K result, and do not establish statistical significance or a prefill improvement. They retain the manual-capacity, greedy protocol and **do not replace the standard v1 README tables**.

Baseline executable SHA-256: `7ddb6bc10f25b729e0ced72431af1f3fcf58270454df3514e08c80c6d49534f7`; selected executable: `656da54c940fa0858ead9cc564b7fa5e78cc44382ba632fc06062b526c88cab9`.

### CPU phase fusion: fixed hybrid-capacity comparison

Each model uses one executable with the legacy and fused paths selected by `STRATA_POOL_FUSED=0/1`. There are four fresh API starts in **A1/B1/B2/A2** order. Each start measures the three [v1 generation prompts](../cases.json) once at a 128-token output cap, then exact 4,096- and 16,384-token inputs with a 32-token output cap. Thus each model has 20 measured requests, excluding the identical 32-token warm-up per start.

The comparisons use **temperature 0**, seed 1234, thinking enabled and the models' other fixed benchmark sampling parameters. Native KV reuse is zero and every complete input is read. `strata_checkpoint:false` makes these one-shot measurements; the checkpoint-buffer change is not active in this experiment. Input chunk size is automatic and selects 2,048. Context capacity is 262,144, INT8 KV, CPU workers 15 / total threads 16, with the GPU image encoder resident. Qwen uses MTP 3; Clover is **causal D0, MTP 0** and uses `STRATA_IQ_MT_MIN=1`. GPU read-ahead is off.

The Expert budget is manually fixed at **4 GiB**, rather than the app's automatic allocation. These are the actual rounded allocations, constant at startup and completion of all four starts:

| Model | GPU Experts / total | Actual Expert payload bytes |
|---|---:|---:|
| Qwen3.6 | 3,649 / 10,496 | 4,254,761,056 |
| Clover causal D0 | 2,760 / 10,240 | 3,570,336,000 |

Generated throughput is total final generated tokens, including thinking, divided by total native decode time. Speculative proposals are not counted as output tokens. Percentages below are recomputed from the unrounded pooled values.

| Model / workload | Legacy phases | Fused phases, grain 6 | Change |
|---|---:|---:|---:|
| Qwen: three-prompt generation group | 100.6883 tok/s | 101.9284 tok/s | +1.23% |
| Qwen: generation after 4K input | 98.0843 tok/s | 100.3449 tok/s | +2.30% |
| Qwen: generation after 16K input | 89.0621 tok/s | 92.5123 tok/s | +3.87% |
| Clover: three-prompt generation group | 52.4178 tok/s | 52.9608 tok/s | +1.04% |
| Clover: generation after 4K input | 45.6263 tok/s | 46.8384 tok/s | +2.66% |
| Clover: generation after 16K input | 38.0454 tok/s | 38.8255 tok/s | +2.05% |

| Input processing | Legacy | Fused | Change |
|---|---:|---:|---:|
| Qwen, 4K | 834.7940 tok/s | 830.2254 tok/s | -0.55% |
| Qwen, 16K | 817.6322 tok/s | 816.4445 tok/s | -0.15% |
| Clover, 4K | 2,906.5106 tok/s | 2,899.9257 tok/s | -0.23% |
| Clover, 16K | 2,555.2488 tok/s | 2,543.5859 tok/s | -0.46% |

The fused path improves decode in these samples; **no bulk-prefill improvement is claimed**. Both paths produce matching combined answer/reasoning text hashes for every corresponding request across all starts. Generated counts and offered/accepted draft counts also match; Clover's draft counts are zero. These response hashes do not prove identity of native token arrays. Separate CPU tests check bitwise equality of the per-row outputs. This is a bounded regression comparison, not a general model-quality improvement.

The same executable SHA-256 is used on both sides: `3f4879c67fc8b4189bbf0f372637f739f803af85f94892e5443b308a3f693b4e`. It includes the scheduling candidate; the legacy control is selected at runtime. The final published commit is therefore **not** identified solely by the baseline Git hash.

### Task-grain search

A separate four-start Qwen comparison tests fused grain 6 versus 12 with otherwise matched fixed-capacity settings. Grain 12 changes short generation by **+0.03%**, generation after 4K by **+0.81%**, and generation after 16K by **+1.65%**. Input processing changes by **-0.59% / -0.36%** at 4K / 16K. The per-start ranges overlap, and the short-workload gain is effectively absent. **Grain 12 is not selected as the global default.** The selected grain-6 path retains fallbacks for a single Expert or only one participating CPU thread.

### Checkpoint-buffer reuse: native latency comparison

The prior code cleared a turn snapshot's byte vector when its token/image prefix became unusable. `clear()` retained its allocation, but the next `resize()` value-initialized the full payload before the device copy overwrote it. The change invalidates the checkpoint position while retaining the vector's logical size. Capture still copies every convolution/recurrent/hidden-state byte and synchronizes before publishing the new checkpoint. The existing token-prefix and image-identity admission rules remain unchanged.

This experiment compares the frozen baseline executable with the newer executable in **four fresh native-engine starts, A1/B1/B2/A2**. It uses Qwen, FP8 KV, context 32,768, chunk 512, **MTP 0**, no Vision encoder, and all **10,496 Experts on GPU**. CPU fusion and GPU read-ahead are disabled. Each start warms two unrelated checkpointed requests, then measures 20 unrelated 32-token prompts, each with a turn boundary at token 16 and one output token. There are **80 measured requests**, 40 per build, with zero prefix reuse. Startup and first allocation are excluded.

| Latency, lower is faster | Previous clear/resize | Reused logical extent | Change |
|---|---:|---:|---:|
| Mean native prompt processing | 74.4200 ms | 72.8875 ms | -1.5325 ms / -2.06% |
| Mean client first output token | 75.2327 ms | 73.6865 ms | -1.5462 ms / -2.06% |

The **raw generated token IDs match** for every corresponding prompt across all four engines. The private extracted-structure poison/full-overwrite check also passes 504 assertions. This change does not add retained allocation capacity: the old vector already kept its capacity after `clear()`. It does not reduce the checkpoint's required payload size, establish an OS RSS saving, or accelerate ordinary decode. First-time allocation is not improved by this experiment.

Baseline executable SHA-256: `7ddb6bc10f25b729e0ced72431af1f3fcf58270454df3514e08c80c6d49534f7`. Checkpoint candidate: `605d34910522e53d24d481234e4e8b68939b8083928e7db4191afae2fac6ee7b`.

### Paired GDN projections: full-GPU Clover comparison

Clover's independent GDN qkv/z projections can share one CUDA launch. The selected layout assigns a separate matrix group to each CTA and retains TinyStrata's existing `q8_row_dot` arithmetic. Quantization, recurrence, reduction order, sampling and weights are unchanged. The first layout combined rows within a CTA and was rejected: decode changed by **-1.22% / -1.16% / -1.74%** for short generation / after 4K / after 16K.

The second layout is compared with separate launches using one executable, `TINYSTRATA_Q36_Q8_PAIR=0/1`, in four fresh **A1/B1/B2/A2** API starts. It uses the same temperature-0 workloads and output caps described above, Clover **causal D0**, FP8 MMA KV, context 262,144 and automatic chunk selection. Expert capacity is automatic; all **10,240 / 10,240 Experts** are GPU-resident, with **13,246,464,000 payload bytes** at startup and completion of every start. This isolates the paired CUDA path from CPU offload scheduling.

| Workload | Separate launches | Paired launches | Change |
|---|---:|---:|---:|
| Three-prompt generation group | 95.56875 tok/s | 96.40853 tok/s | +0.879% |
| Generation after 4K input | 95.38003 tok/s | 95.90889 tok/s | +0.554% |
| Generation after 16K input | 92.80742 tok/s | 93.91049 tok/s | +1.189% |

These are **small pooled point estimates**, not evidence of statistical significance or a guarantee for other GPUs, quantizations or offload budgets. Input processing changes by **-0.545% / -0.500%** at 4K / 16K; no prefill benefit is claimed. All corresponding answer/reasoning text hashes match, profiles are unchanged, and production Clover is restored. The response hashes have the same token-identity limitation as the CPU comparison.

Both controls use executable SHA-256 `5668e8ce6e0784d0f83b600fb3fbcc633ed65e7750c90c5a96007f87ff0c7353`, with the projection switch selecting the path.

Automatic eligibility is restricted to **Clover on CUDA Blackwell, compute-capability major 12**. Both projections must use raw Q8_0, input width at most 2,048, and a batch of 1–8 columns. The existing dispatch remains in use for packed Q8, other formats, larger batches, HIP and other NVIDIA generations. `TINYSTRATA_Q36_Q8_PAIR=0` disables the optimization; explicit `1` enables the diagnostic candidate where its format/shape checks allow it. The measured Qwen3.6 file uses **Q6_K** for these projections and does not activate this path.

`q36_q8_pair_eligible_layers` reports eligible model layers, 30 for measured Clover. `q36_q8_pair_dispatches` counts host dispatch/capture calls, including warm-up; it is not a count of GPU executions during CUDA Graph replay.

### GPU wait-side read-ahead: rejected screens

These Qwen INT8 screens retain the fused-6 CPU path, fixed 4-GiB Expert budget and the same workloads above. The order is **A1 / mode1 / mode2 / mode3 / A2**, five fresh starts. Each enabled mode has only one start; disabled A pools two starts. These are **screening measurements**, not four-start estimates for each enabled mode.

| Read-ahead mode | Short generation | After 4K | After 16K |
|---|---:|---:|---:|
| Disabled, pooled A | 101.8230 tok/s | 102.1548 tok/s | 92.0863 tok/s |
| Broad next-mixer ranges, 32 CTAs | 99.5283 (-2.25%) | 101.2658 (-0.87%) | 89.4604 (-2.85%) |
| Paired input ranges, 8 CTAs, submission query | 99.3943 (-2.39%) | 98.4615 (-3.62%) | 91.8221 (-0.29%) |
| Same paired ranges without submission query | 100.2192 (-1.58%) | 99.1019 (-2.99%) | 90.0141 (-2.25%) |

The point estimates are lower for every enabled mode on every decode workload. Inputs and answer/reasoning text hashes match, and Expert capacity stays matched. The screens provide no reason to adopt these variants; they do not prove that read-ahead cannot help other models or hardware. The recorded target-byte counter is the nominal queued range size, not the number of bytes the GPU completed before CPU work finished.

### Correctness and portability checks

- The final CPU pool regression passes **576 batches with MT1 and 576 with MT2**, with bitwise serial/legacy/fused equality. It covers the actual routed-Expert format pairs, mixed widths, multiple job batches, worker sleep/wake and the small-job fallbacks.
- The corrected public [paired-projection test](../../tests/q36-q8-pair.cu) passes **41 checks**: 32 eager shape/width cases and nine CUDA Graph replay checks. Candidate buffers are poisoned before every comparison/replay, every output row/column is compared bitwise with separate-launch outputs, unused columns retain their sentinel, and sampled values are checked against an independent CPU FP64 calculation.
- The final native checkpoint/cancellation suite passes **31 cases**: 11 Qwen INT8, 11 Qwen FP8 and nine Clover FP8 causal-D0 cases, with a fixed 4-GiB Expert budget, context 32,768 and chunk 512. It checks cold/resumed raw token IDs, edited-prefix restoration, unrelated-turn reset/recapture and cancellation. Qwen uses MTP3 and synthetic image-embedding cases; Clover uses MTP0 without images in this suite. These are state-management checks, not a general quality benchmark.
- A separate real-image baseline/candidate comparison passes **eight requests**, two per model/build, using [the benchmark image](../image.png), temperature 0, thinking off, FP8, context 32,768 and all Experts on GPU. Qwen uses MTP3; Clover uses causal D0. Answer/reasoning hashes and native counts match, each image encoder retains its process identity, and Expert capacity is unchanged. Manual review confirms the red A square, blue B circle and green C triangle. Clover reads the heading correctly; Qwen misreads `LOCAL AI - 3 SHAPES` as `Logical – 3 Shapes` identically before and after. This is an unchanged-output regression result, not perfect OCR or broad vision-quality validation.
- The touched q36 engine compiles for CUDA **SM86** and HIP **gfx1200**; the CPU pool also compiles with **clang-cl**. These are compile checks. Runtime correctness and performance on Ampere/Radeon are not established by them.

For an already configured CUDA build directory named `build-cuda`, with native Experts enabled, the two focused public tests can be built and run on Windows as follows:

```powershell
cmake --build build-cuda --target native_expert_pool_test q36_q8_pair_test
.\build-cuda\native_expert_pool_test.exe 1
.\build-cuda\native_expert_pool_test.exe 2
.\build-cuda\q36_q8_pair_test.exe
```

### Scope and data retention

Completed A/B controllers preserve the installed model-profile hashes and restore the previously active Clover model with images enabled. They use private loopback APIs and do not replace the UI history or search/tool settings. Raw outputs, temporary configurations, binaries and development helpers remain outside Git.

## 日本語

Basalt の実装を参考に、CPU Expert の処理段階の統合、Qwen/Clover のチェックポイント用バッファの再利用、Blackwell上のCloverで対象となるQ8_0 GDN投影の同時起動を追加しました。既存の行ごとの演算を保ち、同期・初期化・カーネル起動の余分な処理を減らす変更です。

**これは標準 v1 の再測定ではありません。** CPU融合の比較は温度0・Expert予算4 GiB固定で、通常の自動配置とは条件が異なります。既存READMEの速度表は置き換えません。この比較の実割当は Qwen が3,649個／4,254,761,056 bytes、Clover が2,760個／3,570,336,000 bytesで、各起動の前後を通して一致しています。

CPU融合では、短文生成が Qwen **+1.23%**、Clover **+1.04%**、4K入力後が **+2.30%／+2.66%**、16K入力後が **+3.87%／+2.05%** でした。入力処理の改善は確認できていません。各要求の回答・思考テキストのハッシュと生成数・採用数は一致しますが、テキストハッシュだけを根拠に生のトークンIDまで一致したとは扱いません。

最終採用版を凍結した旧版と直接比べた4起動ABBA・各モデル20要求では、Qwen FP8／MTP3が短文 **+2.901%**・4K後 **+0.110%**・16K後 **+2.375%**、Clover INT8／因果D0が **+2.526%／+4.676%／+3.778%** でした。CPU融合と対象CloverのGPU投影は既定の自動判定を使い、Expert予算4 GiB・実割当・コンテキスト262,144・chunk 2,048を揃えています。計40回答のテキストハッシュと生成・入力・投機数は一致しました。入力処理は約-0.48%～+0.09%の範囲で、改善は主張しません。これも測定ばらつきを含む点推定であり、既存の通常ベンチ表を差し替える値ではありません。

CPUタスクの粒度12は粒度6に対して短文でほぼ差がなく、長文後の差も小さいため、全環境の既定には採用しません。GPU待機中の先読みは3方式とも生成速度の点推定が低下し、今回の採用対象から外しています。各方式1起動の選別試験なので、他のGPUでも無効だという結論ではありません。

チェックポイントの比較は Qwen・FP8・全Expert GPU常駐・MTP0で、2回の準備後に各起動20回、合計80要求を測りました。入力32トークンの途中16トークンで保存し、1トークン生成します。入力時間は **74.4200→72.8875 ms**、最初の出力までは **75.2327→73.6865 ms**、約2.06%短縮しました。この試験では実際の出力トークンIDが全起動間で一致しています。従来も確保容量は保持していたので、保持RAMの増加・削減や通常decodeの高速化を示す測定ではありません。

GDN投影の同時起動は、最初の配置では生成速度が落ちたため棄却し、行列ごとにCTAを分ける配置を採用しました。Clover因果D0・FP8・全10,240 ExpertのGPU常駐・4起動ABBAでは、短文 **+0.879%**、4K入力後 **+0.554%**、16K入力後 **+1.189%** です。小さな点推定で、統計的有意差を示す結果ではありません。入力処理は約0.5%低下しました。自動適用はBlackwell上のClover・raw Q8_0の対象形状に限り、packed Q8・HIP・他世代などは既存経路に戻ります。現在のQwen3.6の該当重みはQ6_Kなので、この変更は動きません。

最終のCPU回帰試験はMT1／MT2それぞれ576バッチ、出力を毎回汚染して書き込み漏れも確認するGPU試験は41ケース、ネイティブのチェックポイント・キャンセル試験は31ケース通過しています。実画像も両モデルの旧版／新版で計8要求を比較し、回答・思考テキストと生成数などが一致、画像エンコーダのプロセスも維持されました。図形・色・ラベルは確認できましたが、Qwenの見出しの読み違いも旧版と同じで、OCR全般の正確さを保証する試験ではありません。q36のSM86／HIP gfx1200とCPUプールのclang-clコンパイルも通過しました。コンパイルの通過を実機動作・速度の検証とは扱いません。

現在の測定機は70 WのRTX PRO 4000 Blackwell SFFです。今回の数値はQwen/Cloverのもので、AMD・Ampere実機やGemma/LFMでの高速化を示す結果ではありません。
