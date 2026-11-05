# Capacity-aware hybrid Expert staging — 2026-11-05

## English

This change keeps streamed GPU input processing available when a complete pair
of Expert-layer buffers does not fit during startup. It prefers the existing
two-buffer ring and tries one during capacity planning/allocation. Later
temporary-loan failure retains the existing CPU/GPU prompt fallback. Selection
uses the current memory budget, rather than a GPU name or nominal-VRAM table.

These are development comparisons, separate from `tinystrata-bench-v1`.
The historical README speed tables remain unchanged.

## Source and measurement conditions

- Reference source: [`273af6d`](https://github.com/kbc20f11/TinyStrata/commit/273af6d5773041e2864a570d4f67697c1bfc9680).
- Capacity implementation: [`b5aad2a`](https://github.com/kbc20f11/TinyStrata/commit/b5aad2ad5253f80fd505d0b04f6d845640c49d33);
  retained policy: [`2859376`](https://github.com/kbc20f11/TinyStrata/commit/2859376ded131e073e67c270f0dfd32ba6ec62ca).
- Final engine input SHA256: `939274d4e942a79518e99c4a89b677cd85de1f6f1a024cdfce55e6284995ae68`.
- Final timing/quality executable SHA256: `08daa6fa8c192dfe04556113f2b6046fa35aa4dc8423d3b5ea7f6760c410aa97`.
  It was built from reference273 plus that exact engine change. CPU-pool and
  product-kernel sources remain identical to273; documentation commits do not
  imply that the executable was rebuilt.
- Windows x64, RTX PRO4000 Blackwell SFF24GB/70W, driver597.30,
  Ryzen9 7945HX3D,64GB RAM, CUDA13.2.86 and MSVC14.51.

For nominal16/12/8GiB, a parent process allocates and touches8/12/16GiB on the
physical24GiB GPU and holds it across each complete comparison. CUDA/NVML
observations check the allocation, with another full touch outside each timed
request. These are memory-pressure simulations: they retain this GPU's compute,
bandwidth and driver, and are not physical16/12/8GB-board benchmarks. The24GiB
control holds no extra allocation.

Qwen3.6 uses IQ3_XXS, INT8 KV and MTP3. Clover uses IQ3, FP8-MMA KV and causal
D0/MTP0. Both keep their GPU image encoder resident. Context capacities are
262,144/262,144/131,072/65,536 for nominal24/16/12/8GiB. The8GiB comparison uses
automatic placement in the primary table, as do the12/16GiB rows. The24GiB control uses automatic
full-GPU placement without a fixed-slot argument, and verifies the native
all-GPU/graph route. Separate equal-slot diagnostics are identified below.
Chunk selection is automatic; the observed size matches within each A/B row
and is listed below. Hybrid CPU workers are15 plus the
host thread, with the existing reference synchronization and compact-combine
policy; the all-GPU route does not allocate a CPU Expert pool.

Each fresh engine runs an excluded warm-up, three short128-output fixtures,
exact4,096/16,384-token inputs capped at32 outputs, and a real-image request.
Text uses temperature0, seed1234, thinking enabled and checkpoint reuse disabled;
image requests disable thinking. Rates divide summed native final-token counts
by summed native time; unaccepted draft proposals are not output tokens.
The controls use the same final executable, with
`TINYSTRATA_Q36_STAGE_BUFFERS=2` versus `0`.

## Final executable results

### Automatic placement: normal budgets

**A is the forced pair; B is startup capacity-auto.** Each row has one fresh
**A1,B1 screen**, not a repeated statistical ranking. All16 starts completed
all96 measured requests, with no startup/request failures or excluded partial
requests. All rates below are tok/s; each entry is **A → B**.

| Model | Nominal GiB / parent-held GiB | Context | Short decode | Decode after4K | Decode after16K | 4K input | 16K input |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 | 24 /0 | 262,144 | 148.4001 →146.4922 | 154.3657 →153.1100 | 133.4445 →132.5052 | 3208.0201 →3180.8651 | 2606.7173 →2593.7183 |
| Qwen3.6 | 16 /8 | 262,144 | 107.7986 →113.4316 | 124.9024 →112.5572 | 100.7240 →100.5341 | 861.6084 →859.2406 | 797.2749 →790.8290 |
| Qwen3.6 | 12 /12 | 131,072 | 104.6265 →103.5096 | 103.6941 →101.0420 | 90.2425 →89.4855 | 766.7971 →761.4940 | 754.7135 →753.5576 |
| Qwen3.6 | 8 /16 | 65,536 | 80.0384 →79.5739 | 83.8355 →82.7729 | 69.4294 →70.6558 | 602.7518 →602.4150 | 615.0012 →614.4614 |
| Clover | 24 /0 | 262,144 | 98.3430 →97.5461 | 97.2349 →97.1168 | 94.7026 →94.7306 | 2276.1878 →2260.3609 | 2015.8720 →2016.2690 |
| Clover | 16 /8 | 262,144 | 63.4963 →63.4040 | 59.1607 →59.3692 | 58.0025 →58.2136 | 2380.0116 →2376.1457 | 2200.1397 →2197.7491 |
| Clover | 12 /12 | 131,072 | 56.2398 →56.1453 | 52.0156 →52.4848 | 50.7453 →51.3149 | 1644.5176 →1648.0245 | 1651.2134 →1651.1469 |
| Clover | 8 /16 | 65,536 | 43.7487 →43.5695 | 46.3432 →46.7768 | 41.9177 →41.6775 | 894.0303 →897.3207 | 890.3670 →897.6157 |

The ordinary8GiB Qwen row fits two buffers in both arms: it does **not** show a
twofold input-speed gain. The24GiB rows prove the actual all-GPU/Graph route,
with zero staging allocation in both arms. Their short-decode point differences
are -1.29% for Qwen and -0.81% for Clover; this single A/B screen cannot establish
a statistical ranking or a blanket no-regression result.

Qwen16GiB completed every request, but the existing cache-restoration budget
guard disabled temporary staging loans during startup warm-up in **both** arms.
Resident maps, CPU/GPU work, response hashes and some native counts differ, and
the strict clean-route gate fails. Those timings remain visible, but neither
the +5.23% short-decode point estimate nor the -9.88% post4K estimate certifies a
staging-policy speed change. The other seven cells,84 requests, match response
hashes, native counts, maps and CPU/GPU work across their paired arms.

### Added GPU pressure:8GiB context class plus256MiB

This separate stress case holds **17,448,304,640 bytes /16.25GiB**: the ordinary
16GiB parent reservation plus256MiB representing additional occupied GPU
capacity. Context remains65,536 and chunk512. It is neither an unqualified
normal8GiB result nor a physical8GB-board result, and does not simulate another
application's concurrent compute or PCIe traffic.

Each model uses fresh **A1,B1,B2,A2 (ABBA)**, two starts per arm. All8 starts and
all48 measured requests completed; **both Clover pair-policy starts worked**.
No startup/request failures are hidden or converted into0tok/s. Rates pool
actual final native tokens/time across the two starts per arm.

| Model | Short decode A→B | Decode after4K A→B | Decode after16K A→B | 4K input A→B | 16K input A→B |
|---|---:|---:|---:|---:|---:|
| Qwen3.6 | 70.6499 →75.3791 | 71.9344 →83.9344 | 69.3316 →70.7965 | 285.2090 →611.6400 | 286.3943 →616.7757 |
| Clover | 44.6281 →44.2756 | 47.1559 →47.3478 | 42.6468 →42.7379 | 894.8018 →897.4289 | 890.8948 →898.3023 |

Here Qwen's pair policy disables native streaming at startup, while auto selects
one buffer and retains native GPU prefill. Input processing rises **114.45% at4K
and115.36% at16K**. All10 single-buffer loans, including model warm-ups, restore
their original slot counts, map fingerprints and mapped cache bytes. Automatic
placement and execution routes differ, so only the16K fixture's hashes/counts
match across arms. This is an **end-to-end capacity/route benefit**, not an
isolated kernel or bitwise-equivalence speedup. No Experts are pruned and no
quantization, sampling or product kernels change.

Clover retains identical hashes/counts/maps/CPU-GPU work for all24 requests and
halves permanent staging from **662,331,392 to331,165,696 bytes**, saving
315.824MiB. Its small speed differences do not establish that single buffering
is universally faster or that a pair invariably fails.

### Image latency

These are API **first-output / complete-response** timings in seconds, with
thinking off. Normal rows have one image per arm; stress rows show means of two
per arm. Every image request completed. These wall timings include image handling
and differ from native output-decode throughput.

| Model / case | First output seconds A→B | Complete response seconds A→B |
|---|---:|---:|
| Qwen3.6 / normal24GiB | 0.6869 →0.6822 | 1.3962 →1.3959 |
| Qwen3.6 / normal16GiB | 1.7870 →1.7472 | 3.0118 →3.2597 |
| Qwen3.6 / normal12GiB | 1.6890 →1.6895 | 2.9385 →2.9324 |
| Qwen3.6 / normal8GiB | 2.5816 →2.5510 | 4.1504 →4.0998 |
| Clover / normal24GiB | 1.0842 →1.0925 | 1.9869 →1.9956 |
| Clover / normal16GiB | 1.2881 →1.2924 | 2.6078 →2.6205 |
| Clover / normal12GiB | 1.4037 →1.4148 | 2.9945 →3.0068 |
| Clover / normal8GiB | 2.0498 →2.0708 | 3.9362 →4.1544 |
| Qwen3.6 /8GiB class +256MiB pressure | 3.6646 →2.5013 | 5.3830 →4.1556 |
| Clover /8GiB class +256MiB pressure | 2.0363 →2.0365 | 3.9770 →4.0662 |

For the normal8GiB Clover image, native output decode is45.6112→41.2985tok/s
(-9.46%), with86 identical output tokens. Complete-response wall time increases
5.54%, from3.9362 to4.1544 seconds. This one-image A/B observation is retained;
the change is not presented as universally regression-free.

### Actual placement and route checks

Counts below are **resident Experts**, not omitted model Experts. Total counts
are10,496 for Qwen and10,240 for Clover. Startup→completion and ranges are observed
snapshots, including before/after warm-up and requests; they do not claim the
instantaneous minimum inside a temporary loan. A/B denote the two policies.

| Model / normal GiB | Chunk | Resident Experts startup→end, A /B | Snapshot range, A /B | Selected buffers A→B | Native-route / equivalence evidence |
|---|---:|---|---|---|---|
| Qwen3.6 /24 | 4096 | 10,496→10,496 / same | 10,496 / same | inactive / inactive (selected2/2; requested2/0) | All-GPU/Graph proof in both; no hybrid work, loans or allocated staging. |
| Qwen3.6 /16 | 2048 | 6,407→7,017 /6,361→7,016 | 6,407..7,017 /6,361..7,016 | 2→2 | Warm-up cache-restoration guard in both; later CPU/GPU fallback; unequal maps/hashes/work. |
| Qwen3.6 /12 | 1024 | 4,141→4,141 / same | 4,141 / same | 2→2 | Pair loans; matching hashes/native counts/maps/work. |
| Qwen3.6 /8 | 512 | 738→738 / same | 698..738 / same | 2→2 | Pair fits in both; matching hashes/native counts/maps/work. |
| Clover /24 | 512 | 10,240→10,240 / same | 10,240 / same | inactive / inactive (selected2/2; requested2/0) | All-GPU/Graph proof in both; no hybrid work, loans or allocated staging. |
| Clover /16 | 2048 | 5,640→5,223 / same | 5,134..5,640 / same | 2→2 | Permanent MMQ; matching hashes/native counts/maps/work. |
| Clover /12 | 1024 | 2,760→2,726 / same | 2,653..2,760 / same | 2→2 | Permanent MMQ; matching hashes/native counts/maps/work. |
| Clover /8 | 512 | 0→0 / same | 0 / same | 2→1 | Permanent MMQ allocation halves; matching hashes/native counts/maps/work. |

In the added-pressure ABBA case, Qwen's two A starts each go492→466 resident
Experts, with snapshot range394..492; the two B starts each retain410→410, with
range394..410. A's pair policy is inactive after startup rejection; B uses one
temporary buffer, with375,390,208 mapped bytes while borrowed and zero retained
stage allocation afterward. Clover remains0→0 resident Experts in every start,
using permanent MMQ staging2→1. Qwen B's single-buffer runs have no failed-loan/
restoration flag or terminal CPU-prefill fallback; A's startup-selected CPU/GPU
route remains distinct.

For normal budgets, initial CUDA/NVML drops match the full8/12/16GiB allocations;
180 nonzero-allocation retouches and60 unrestricted24GiB snapshots pass, with
zero observed free-memory drop. For stress, both initial drops match16.25GiB;
all120 retouches pass the configured256MiB tolerance. Three Clover B1 checks
show NVML-only decreases of28/64/64MiB, with CUDA free memory unchanged; the cause
was not established. Thus the owned allocation and bounded memory checks are
verified, without claiming byte-exact physical residency at every instant.
All reservations are released, services close and saved profiles/production are
restored after both datasets.

An initial24GiB diagnostic used explicit256 slots per layer. That forces hybrid
topology even with all Expert data resident; it is excluded from the true
automatic all-GPU table above. Equal-slot diagnostics and the discarded late
runtime-retry prototype also remain separate from these final measurements.

## Default policy and restoration

`TINYSTRATA_Q36_STAGE_BUFFERS` controls the ring:

| Setting | Behavior |
|---|---|
| Unset or `0` | Prefer two buffers; try one when startup capacity/allocation prevents the pair. |
| `1` | Force one buffer for diagnostics. |
| `2` | Force the previous pair policy. |
| Unrecognized explicit value | Conservatively use two. |

Two buffers retain copy/compute overlap when they fit. A single buffer waits for
its last compute reader before reusing storage. The count selected at startup
remains unchanged during later loans and forwards.
All-GPU placement does not allocate a staging ring.

For temporary native staging, an unsuccessful loan returns partial mapping and
uses the existing CPU/GPU prompt fallback. Successful loans are returned with
original slot maps and mapped-byte totals checked. A one-buffer ring selected at
startup also uses one for its temporary loans; a later failed pair is not
automatically converted to a single-buffer run.
Permanent MMQ staging can use the smaller ring too. Native INFO reports requested,
selected and active counts, allocated bytes, attempts and fallback outcomes.

## Correctness and portability

The final executable was compared with the frozen273 executable, using exactly
32 Expert slots per layer, chunk512, hybrid threshold128, growable KV and
context32,768. Prefixes exceed the threshold, and loan/allocation telemetry plus
native restoration logs prove that the tested one-buffer route actually ran.

| Gate | Result |
|---|---|
| Qwen INT8/FP8 MTP3 and Clover FP8 D0 state tests | PASS: cancellation, retry versus cold, edited-prefix restoration, unrelated input and synthetic Qwen image boundaries; successful raw IDs/counts match. |
| Qwen/Clover × INT8/FP8 full logits | PASS:17 complete FP32 vocabulary rows per case,16,885,760 finite values per arm; raw bytes and independently recomputed greedy argmax IDs match. |
| Real images, Qwen MTP3/Clover D0 | PASS:8 requests across4 fresh engine/encoder pairs; paired response hashes/native counts match. |
| Optional Clover exact D3 | PASS:3 fixtures per arm; raw IDs/counts match,69/69 offered drafts accepted in each arm. These fixtures are not a general acceptance-rate estimate. |
| Source-extracted GPU ring | Earlier candidate6 harness PASS:256 forwards, counts1/2, delayed readers, mixed host/device copies, MTP/singleton sequences and poisoned-buffer guards. Final7's four ring methods are byte-for-byte identical, with matching hashes; the harness was not rerun for final7. |
| CUDA SM86, HIP gfx1200 and gfx1201 | Compile PASS. No runtime or speed result on those devices is implied. |

Qwen misreads one image subtitle identically in both arms; the labelled colored
shapes remain correct. Clover reads the fixture's heading, subtitle, shapes and
footer correctly. This is a regression check, not a general vision-quality score.

The capacity implementation is shared CUDA/HIP host scheduling. Radeon runtime
measurements await the planned hardware change. Saved model profiles, keys and
history are preserved; every owned test controller restores the original model.
Gemma and LFM executables are unchanged.

## Other measured scheduling candidates

The following prototypes are preserved on the separate
`perf/portable-hybrid-latency` branch, rather than shipped in the adopted engine.
They build on ideas reviewed in the local Basalt reference; existing Basalt
attribution remains in NOTICE. These outcomes apply to the tested16GiB CUDA
conditions and do not establish their behavior on every CPU or Radeon.

| Prototype | Observation and disposition |
|---|---|
| Completion-only CPU handoff | No admitted speed gain. One16K comparison also changed to CPU/GPU prefill after a staging-loan failure, so it is not an isolated barrier result. |
| Before-only CPU park barrier | Equal-capacity/same-route ABBA: short111.96→110.31, after4K113.78→113.44, after16K100.11→99.35tok/s. Retain the existing synchronization. |
| FIFO asynchronous Expert promotion | Real copies/batches executed, but no observed gain; the16K comparison also had a route change. Copy submission alone does not prove useful overlap. |
| Reader-order asynchronous promotion | Equal-capacity/exact-output/same-route screen: short113.09→107.86, after4K115.98→101.14, after16K104.10→98.43tok/s. Keep it experimental. |
| Runtime pair-to-single loan retry | A16GiB equal-slot screen changed execution routes and measured4K/16K input878.35→805.89 /817.61→781.30tok/s. It did not establish a general advantage, so the adopted policy retains the previous late CPU/GPU fallback. |

A separate automatic16GiB diagnostic of the runtime-retry prototype completed all requests, but both arms
disabled staging loans during warmup after the existing cache-restoration budget
guard. Its strict native-route gate failed; it cannot certify the runtime-retry
prototype. Request success is kept separate from proof of a particular GPU route.

## 日本語

今回採用したのは、起動時にVRAMが足りない場合、入力処理用のExpert層バッファを
二重から一重へ切り替える共通処理です。空きがある場合は二重を優先し、GPU名で
分岐しません。全ExpertがGPUに収まる場合、一時リングは確保しません。

24GB実機のVRAMを別プロセスで確保して8／12／16GB相当を作っています。
物理的な小容量GPUとの性能比較ではなく、容量不足時の動作を調べる対照実験です。
上の最終実行ファイルの表を参照してください。通常のREADMEベンチ表とは別条件
なので、過去の数値は更新していません。

通常予算は各1回のA/Bスクリーニングで、順位や無劣化を統計的に保証する測定では
ありません。通常8GiB相当のQwenは両方とも二重バッファで、入力速度はほぼ同じです。
別途256MiBを占有したABBA試験では、一重化により4K入力が285.2090→611.6400、
16K入力が286.3943→616.7757tok/sになりました。この追加占有条件の効果を、
通常8GB環境全般の2倍という意味には広げていません。

Cloverの常設一時領域は662,331,392→331,165,696バイトに半減します。一方、
通常8GiB相当の画像応答では完了時間が3.9362→4.1544秒となった1回ずつの観測も
残しており、常に高速になるとはしていません。通常16GiB相当のQwenも、両条件で
既存のキャッシュ復元ガードが働いています。要求は完了していますが、同じGPU経路
での純粋な速度比較が成立したケースとしては扱いません。

今回の追加占有条件では、一重化でCPUへの入力処理フォールバックを回避し、入力処理が高速化しました。
起動後に二重から一重へ切り替える案は速くならない条件もあったため除外し、
後から一時領域を確保できない場合は従来のCPU/GPU併用処理を維持します。
自動配置のExpert数や実行経路が変わる効果も含むため、カーネル単体の高速化とは
区別しています。生成速度との兼ね合いも表に併記し、モデルのExpertを削ったり
量子化方式を変えたりする処理は追加していません。

配置を固定した全語彙の数値、キャンセル・再開、画像、任意のexact D3を比較し、
従来方式との一致を確認しました。リングの4関数は256試行を通過した先行候補と同一です。
このリング試験は最終候補では再実行していません。
CUDA SM86とHIP gfx1200／gfx1201はコンパイル確認済みで、Radeon実機での速度検証は
交換後に行います。速度改善が確認できなかったCPU同期・非同期転送の試作は別ブランチ
に保存し、正式経路には含めていません。
