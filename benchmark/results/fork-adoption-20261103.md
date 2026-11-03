# MoMoE / PolyStrata adoption results — 2026-11-03

## English

This update adds compensated CUDA INT8 decode for Qwen, bounded same-model conversation parking for Gemma/LFM, and reusable completed prompt chunks after Qwen cancellation. Gemma's CUDA host-upload buffers also use 512 MiB less configured RAM. The original FDEC implementation was selected after testing two further kernel variants.

These controlled comparisons supplement the README's historical model-default benchmark tables; they do not replace those measurements.

### Changes and defaults

| Change | Scope and default |
|---|---|
| Qwen INT8 FDEC decode | Automatic when the compiled CUDA kernels and SM80+ device support it: main Qwen attention, HD256/GQA8, 1–4 query rows, logical context capacity at least 4096. Reuses existing scratch. `TINYSTRATA_Q36_INT8_FDEC=0` disables it. FP8, Clover, MTP attention and unsupported shapes retain their existing paths. |
| Qwen prompt cancellation | Preserve completed chunks after returning staging resources; resume or edit from a valid prefix instead of retaining partially updated layers. |
| Gemma/LFM conversation parking | CUDA serve defaults: **Gemma 1024 MiB / LFM 512 MiB**, two parked slots, disk budget **0**. HIP defaults to **0 MiB** of additional parking. These are bounded cache limits, not eager allocations of the full allowance. |
| Gemma host-upload buffers | CUDA: **256 MiB per buffer**, down from 512 MiB. Two buffers save **512 MiB** of configured host RAM. LFM and HIP retain 512 MiB per buffer. |

Parking restores attention KV before recurrent state, including Gemma's sliding-window state, and distinguishes changed image embeddings. It works between chats using the same model in the same running engine; it does not persist through model switches or restarts. Budget exhaustion can evict a parked entry or skip saving it and recompute the prompt. Heap, optional temporary-file storage, metadata and temporary spill overlap share one bound. The default disk budget is zero.

The Responses API's completed streaming event also includes a copy of native timings. Benchmark profiles now preserve deliberate capacity settings through configuration migration and disable creation of conversation snapshots for one-shot measurements.

### Hardware and method

Windows; RTX PRO 4000 Blackwell SFF, 24 GB VRAM; Ryzen 9 7945HX3D; 64 GB RAM; nvcc **13.2.86**. CUDA SM86 and touched HIP sources compiled successfully. These compile checks are **not physical Ampere or Radeon measurements**.

TinyStrata baseline: `857477f56999e21846e2613c8ffebfe6879cb143`. Sources reviewed: [MoMoE `1fd5649b`](https://github.com/aflin/momoe/commit/1fd5649b5d74138254de6b32ec8739ee0dab1cd3) and [PolyStrata `f472d85a`](https://github.com/VecSzn/PolyStrata/commit/f472d85a30927301663401f7dc892e668578a90e), both dated November 2. FDEC adapts PolyStrata's decode structure to TinyStrata's head-major INT8 cache, FP16 scale per 64 values and int32 device position. Amplified residual compensation is retained for Q and P×Vscale; future unmapped KV rows are masked before reads.

The main Qwen and Gemma buffer comparisons use the same executable and native arguments with A1/B1/B2/A2 fresh starts. Generated throughput is total final generated tokens, including thinking, divided by total native decode time; draft proposals are not counted. Input time uses native `prompt_ms`. Startup and model loading are excluded. Requests use `strata_checkpoint:false`; these audited comparisons have zero reused tokens and zero RAM/file blobs.

### Qwen: INT8 decode comparison

Qwen3.6-35B-A3B UD-IQ3_XXS, INT8 KV, MTP 3, GPU image encoder enabled, context 262144, automatic chunk 2048, all **10,496 Experts** resident throughout. Only `TINYSTRATA_Q36_INT8_FDEC=0/1` changes. Thinking is enabled; temperature 0, seed 1234, top-p 0.95, top-k 20, presence penalty 1.5, repetition penalty 1.0. Each request has a 128-token output cap; the generation group contains three prompts run twice per start.

| Metric | Previous SIMT | FDEC | Change |
|---|---:|---:|---:|
| Six-request generation group | 142.10 tok/s | 146.37 tok/s | +3.01% |
| Generation after 4K input | 114.24 tok/s | 149.14 tok/s | **+30.55%** |
| Generation after 16K input | 64.81 tok/s | 129.46 tok/s | +99.75% |
| 4K input processing time | 1392.15 ms | 1404.60 ms | +0.89% |
| 16K input processing time | 6762.95 ms | 6786.35 ms | +0.35% |

The **4K responses match across all four starts**, comparing combined answer/reasoning text hashes. The generation group and **16K responses differ between modes**, although they repeat within each mode. Thus the 16K result is not a claim of twice the throughput on an identical token sequence. Text hashes also do not prove native token-array identity; MTP acceptance can change. FDEC accelerates decode, not bulk prefill, and these trials show no input-processing improvement.

Tested executable SHA-256: `bc5ac70ac7558ef50943be87d467fe219a71c13e714020264185bec2e5af7f9a`.

A final hybrid smoke comparison reserved 12 GiB of physical VRAM and used the selected build with INT8, context262144, MTP3, automatic chunk1024 and 3,649 GPU Experts. Both modes completed short, 4K and 16K requests; the FDEC run returned every staging loan to all 3,649 slots with exact mapping restoration, and the reservation retouch check passed. This single A/B run verifies hybrid operation, rather than establishing a repeated performance result.

### Numerical and functional checks

| Check | Result and limit |
|---|---|
| FDEC numerical suite | **48 cases pass**: 1–4 queries; split/tile tails; scale, subnormal, cancellation and saturation stress; guards and immutable inputs; device position; graphs; four VMM cases with future rows unmapped. Independent FP64 and production SIMT references use identical stored KV values. |
| Error criteria | Maximum error ≤ `max(2e-6 * max(absV, 1e-6), 8 * SIMT max error)`; relative L2 ≤ `max(5e-5, 8 * SIMT relative L2)`. Singleton and batched runs pass separately, and their difference fits the same error budget. Same-shape graph/eager output remains bit-identical. Moving queries between MMA fragment rows can change rounding. |
| Six bounded model tasks | FDEC **6/6 correct with natural termination**: three arithmetic tasks, Japanese bullet constraints, JSON extraction and interval-merging code with eight CPU assertions. Baseline is **5/6**: interval-merging reaches the 4096-token cap with an empty final answer. FDEC finishes that task at **4076 tokens**, only 20 below the cap. |
| Repetition | No offline exact-repetition hints or runtime reasoning recoveries in the twelve quality responses. This is not proof that loops cannot occur. |
| 4K/16K retrieval | Both modes return the correct fixed answers, 73195 / 48263, and terminate naturally. |
| Real image | Both correctly describe the benchmark image's red square A, blue circle B, green triangle C, white letters and lower caption; 91 / 86 generated tokens, natural termination. One image does not establish general vision quality. |
| Prompt cancellation | Qwen INT8/FP8 with MTP 3, including partial image-prompt cancellation, and Clover exact D3 pass protocol checks. Resumed/edited token sequences match cold controls in those cases. |
| Parking state | Successful LFM heap512 / Gemma heap1024 A→B→A2 checks reuse 4103 tokens; restored eight-token outputs match resident controls, including after cancellation. Gemma disk1024 with synthetic image embeddings also checks restore and changed-image identity. Gemma512's eviction/recomputation run is excluded from successful restore evidence. |
| Final selected build | FDEC GPU suite and Qwen INT8/FP8/Clover D3 cancellation pass again. CPU checks pass: 3,725 conversation-cache assertions, 34 Python tests and 14 benchmark tests. Real HTTP chat switching reuses 4,426 tokens for a Gemma image conversation and 4,095 for LFM text; both return the expected identifier. |

Quality checks used identical requests in both modes, temperature 0, seed 1234 and a 4096-token cap. These small fixed tests check regressions; the 5/6→6/6 result does not establish a general quality improvement or solve long thinking. Synthetic image parking checks state identity, separately from the real-image encoder test.

The first Japanese LFM HTTP fixture exhausted a 1024-token thinking cap before the first final answer, before any parked restoration was attempted. The successful LFM integration fixture uses English and a 4096-token cap; that success does not resolve the first fixture's long-thinking behavior.

### Host memory and rejected candidates

Gemma's 512→256 MiB buffer ABBA reserves 12 GiB on the physical 24 GB card and uses INT8, context32768, MTP3, vision, fixed Expert budget4 GiB and automatic chunk1120. Expert capacity transitions match across starts. Two identical-output short-code cases change decode by −0.27% / −0.64%; mean input time changes by −1.64% at4K and −0.24% at16K. The buffer reduction is adopted for its **512 MiB allocation saving**, without claiming zero speed regression or a reduction in total OS RSS.

Conversation snapshots have a separate RAM budget and can consume additional memory. The upload-buffer saving does not imply that total process RAM decreases by 512 MiB when parking is used.

| Candidate | Decision |
|---|---|
| LFM smaller upload buffers | Keep 512 MiB. The matched 256 MiB comparison gives generation −2.69%; one identical-output/capacity/Expert-demand code case gives −2.86%. Earlier64 MiB screening has changing Expert capacity. |
| Gemma64 MiB buffers | Reject; aggregate generation −6.37%, generation after16K −7.97%. Use256 MiB instead. |
| Two/eight/32 follower pinned slots | Reject. Savings did not produce a sufficiently consistent model-level benefit; early trials also had capacity confounders. |
| SSE non-temporal staging copies | Reject. CPU alignment/tail/canary tests pass, but model4K/16K prefill does not improve. |
| FDEC delayed V loads / paired MT1 launches | Numerical tests pass, but whole-model measurements do not improve over the selected original implementation. Compensation is retained. |
| Full-RAM Expert pool | Not adopted in this loop. The donor keeps this large-pool lifecycle optional; a benefit on this 64 GB machine was not demonstrated. No speed result is claimed. |

### Parking latency

| Same-model A→one-shot B→A2 | Parking disabled: PP / TTFT | Heap parking at the default budget: PP / TTFT |
|---|---:|---:|
| Gemma, 1024 MiB, slots2, disk0 | 1221.8 / 1223.5 ms | 82.7 / 84.1 ms |
| LFM, 512 MiB, slots2, disk0 | 1163.3 / 1164.4 ms | 37.3 / 38.5 ms |

These are single sequential diagnostic samples, not a repeated timing benchmark. Each model uses the same selected executable, inputs and arguments except the cache budget. Context65536, temperature0, seed1234, eight output tokens; Gemma INT8/MTP3/vision, LFM FP8/MTP0. A has 4096 input tokens, B4127, A2 4111; B explicitly uses checkpoint:false. The fixture sets min-tokens1, while the serve default remains1024. Returning to A reuses4103 tokens and reads8 with parking, versus reuse0/read4111 without it. Both restored eight-token outputs match resident controls. Cold recomputation can use a different execution shape and produce different output, as observed for Gemma; the cold output is not the correctness oracle.

A2 request wall time, including its eight-token decode, changes from1311.2→189.7 ms for Gemma and1209.6→82.5 ms for LFM. The parked runs' cache high-water counters are426.61/1024 MiB and45.54/512 MiB, respectively, with no evictions or spill. These are cache counters, not process RSS. On the first A, the heap run has PP+22.9 ms for Gemma and+8.4 ms for LFM in this sequence; other resident-start controls vary, so these differences do not isolate snapshot cost. Snapshot copying does add work and is included in request timing. Loading time is excluded throughout.

Reproduce FDEC comparisons with `TINYSTRATA_Q36_INT8_FDEC=0/1` and identical model, KV, MTP, context, chunk and Expert capacity. Check native `q36_int8_fdec`, reuse and memory counters. Compare host buffers with `TINYSTRATA_STAGE_CHUNK_MIB=512/256`; compare parking with `--conversation-cache-mib 0/1024` for Gemma or `0/512` for LFM, `--conversation-cache-slots 2 --conversation-cache-disk-mib 0`, and identical prompts/sampling. Raw logs, private harnesses, worktrees, models and binaries remain outside Git.

## 日本語

CUDA版QwenのINT8生成に補償付きFDECを採用しました。主モデルのAttention、HD256/GQA8、1〜4 query、論理context容量4096以上で、コンパイル済みカーネルとSM80以降のGPUが対応する場合に自動使用します。既存scratchを再利用し、`TINYSTRATA_Q36_INT8_FDEC=0` で無効化できます。FP8・Clover・MTP Attention・非対応shapeは従来経路を維持します。

実測はRTX PRO4000 Blackwell SFF24GB、Ryzen9 7945HX3D、RAM64GB、Windows、nvcc13.2.86です。SM86と変更対象のHIPソースはコンパイル成功ですが、Ampere/Radeon実機の速度測定ではありません。既存READMEのモデル既定条件による測定表は上書きしません。

| Qwen INT8、同一binaryのABBA | 従来 | FDEC | 変化 |
|---|---:|---:|---:|
| 生成用3prompt・各2回 | 142.10 tok/s | 146.37 tok/s | +3.01% |
| 4K入力後の生成 | 114.24 tok/s | 149.14 tok/s | +30.55% |
| 16K入力後の生成 | 64.81 tok/s | 129.46 tok/s | +99.75% |
| 4K入力処理時間 | 1392.15 ms | 1404.60 ms | +0.89% |
| 16K入力処理時間 | 6762.95 ms | 6786.35 ms | +0.35% |

Qwen3.6 UD-IQ3_XXS、INT8 KV、MTP3、GPU画像encoder、context262144、auto chunk2048、全10496 Expert常駐、temperature0、seed1234。出力上限は各128tokenです。ロード時間を除きnative時間から集計し、通常ベンチは `strata_checkpoint:false`、再利用token/RAM・file blobsはいずれも0でした。**4KはABBA全4回の回答＋思考テキストhashが一致。16Kと生成用promptはA/Bで出力が異なるため、同一token列で2倍になったという意味ではありません。** bulk prefill自体の改善は確認されません。

数値48ケースはFP64・現行SIMT対照でPASSし、guard・未map将来KV・graph・1〜4queryを確認しました。singleton/batchは同じ誤差基準で個別に判定し、同shapeのgraph/eagerはbit一致です。モデル6問は候補6/6正解・自然停止。baselineはintervalsだけ4096token上限で最終回答が空となり5/6、候補は4076tokenで自然停止してCPU8テストもPASSしました。上限まで20tokenであり、長考や品質全般の改善を保証する結果ではありません。反復のoffline疑い・runtime回復数は0。4K/16K needleと、赤四角A・青円B・緑三角Cの実画像説明は両方正しく自然停止しました。

| その他の採用変更 | 既定と範囲 |
|---|---|
| Qwenの入力処理取消 | 完了chunkを残して再送・編集へ復帰。不完全な層状態を再利用せず、Qwen INT8/FP8＋MTP3、Clover exact D3でプロトコル確認 |
| Gemma/LFMの同モデル会話parking | CUDAのserve既定上限は **Gemma1024 MiB / LFM512 MiB、slots2、disk0**。HIPの追加parking既定は0。同じ生存エンジン・モデル内のチャット切替用で、モデル切替・再起動をまたぐ保存ではない |
| Gemma CUDA host buffer | **1本256 MiB、2本で512 MiB削減**。LFMとHIPは1本512 MiBを維持。設定上の割当削減であり、OS全体の実測RAM削減値ではない |

parkingはfull KV import後にSWA/recurrent/MTP状態を復元し、変更画像のidentityも照合します。予算不足時はevict・保存見送り・再計算へ戻ります。LFM heap512 / Gemma heap1024では復帰時4103tokenを再利用し、8token出力がresident対照と一致。取消後や、合成画像埋め込みの変更も確認しました。同一binaryのA→単発B→A2では、A2の最初の出力までが **Gemma1223.5→84.1 ms、LFM1164.4→38.5 ms**。各1回の診断値であり、繰り返し速度ベンチではありません。context65536、temperature0、seed1234、出力8token、Bはcheckpoint:false、fixtureのmin-tokensは1（serve既定は1024）です。保存copyにも時間がかかり、最初のAのPPは今回の並びでGemma+22.9 ms/LFM+8.4 msでしたが、単発差から保存コストを厳密には分離できません。

Gemma256 MiB bufferは、同一出力コード2件で生成−0.27%/−0.64%という差と512 MiBの割当削減を踏まえて採用しました。LFM buffer縮小、Gemma64 MiB、2/8/32 pinned slot、SSE non-temporal copy、FDECのV読み込み遅延・MT1 pairは、model全体の改善が得られず不採用です。条件・却下理由の詳細は上の英語表を参照してください。raw logs・private計測補助・モデル・binaryはGitへ追加しません。

会話snapshotは別枠でRAMを使うため、転送bufferの削減がプロセス全体のRAMを512 MiB減らすという意味ではありません。全Expertを大きなRAM poolへ移す追加経路は、この64 GB環境での利点を実証しておらず、今回の採用対象から外しました。Responses APIの完了stream eventにはnative timingsを追加し、ベンチでは明示した容量設定を移行処理で保持し、単発測定のsnapshot作成を無効化しています。
