# Measured speed — Clover on RTX PRO 4000 Blackwell SFF Edition

[English](#english) · [日本語](#日本語)

## English

Measured on **2026-11-02** under **Windows 11 Pro build 26300**, using an **RTX PRO 4000 Blackwell SFF Edition (24 GB; 24,467 MiB; 70 W power limit)**, NVIDIA driver **597.30**, a **Ryzen 9 7945HX3D (16 cores / 32 threads)** and **64 GiB installed system RAM**. Engine source: [7ad7904](https://github.com/kbc20f11/TinyStrata/commit/7ad7904175f50836383b020940aa321fba1e7e1c); the **CUDA compiler nvcc/NVVM is 13.2.86**. Runtime and cuBLAS DLLs were retained from the official **CUDA 13.2.0 redistributables**, with exact identities below; this measurement did not use a complete CUDA 13.2.2 component set. The model is [AtomicChat's pinned Clover AD-IQ3_S-IQ3_XXS](https://huggingface.co/AtomicChat/Clover-35B-A3B-Preview-GGUF/blob/4f7e2918e5315634d9d9720ea4f6e8a9a627a875/Clover-35B-A3B-Preview-AD-IQ3_S-IQ3_XXS.gguf): Expert gate/up IQ3_XXS, down IQ3_S. The **15.515 GB** language model retains Clover's own tokenizer and learned GDN parameters. The shared **0.899 GB Qwen3.6 projector** remains resident on the GPU throughout text tests. Image support is TinyStrata's extension; official Clover is text-only.

### Conditions and results

Protocol: **`tinystrata-bench-v1-clover-causal-vision-vram-variant`**, a controlled variant using the [v1 fixed workloads and timers](../README.md#english). It is separate from fixed standard v1 and uses explicit tuning and simulated VRAM limits. The current managed-profile default is **causal D0/INT8/MT1**, with D3 optional. All rows use **causal D0**, no MTP/draft model, **automatic CPU workers, prompt chunks and Expert placement**, physical KV growth, normal Graph selection, `STRATA_IQ_MT_MIN=1` and `STRATA_EXP_MODE=8`. The saved local fixed-2,048 tuning profile was not substituted for the automatic-chunk comparison.

| Nominal VRAM | KV | Successful / planned starts | Context | Generation tok/s | 4K input tok/s | 16K input tok/s | Image first / end s | Expert GiB: startup → 16K |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 24 GB | INT8 | 4/4 | 256K | 91.61 | 2216.87 | 1931.89 | 1.11 / 2.00 | 12.34 → 12.34 |
| 24 GB | FP8 | 4/4 | 256K | 94.67 | 2244.69 | 2029.67 | 1.11 / 1.94 | 12.34 → 12.34 |
| 16 GB | INT8 | 4/4 | 256K | 58.16 | 2235.75 | 2205.22 | 1.35 / 2.89 | 6.55–6.60 → 6.07–6.22 |
| 16 GB | FP8 | 4/4 | 256K | 58.17 | 2280.15 | 2253.31 | 1.36 / 2.84 | 6.55–6.60 → 6.11–6.19 |
| 12 GB | INT8 | 4/4 | 128K | 51.31 | 1605.06 | 1595.24 | 1.52 / 3.18 | 3.08–3.18 → 2.80–3.04 |
| 12 GB | FP8 | 4/4 | 128K | 50.89 | 1613.57 | 1640.68 | 1.48 / 3.13 | 2.94–3.18 → 2.80–3.04 |
| 8 GB | INT8 | 3/4 (successful subset only) | 64K | 39.46 | 881.25 | 883.78 | 2.12 / 4.26 | 0.00 → 0.00 |
| 8 GB | FP8 | 4/4 | 64K | 39.75 | 879.86 | 886.24 | 2.12 / 4.13 | 0.00 → 0.00 |

Text rates pool actual native tokens and milliseconds; loading and warm-up are excluded. The README shows numerical speed only for profiles with four successful starts. Any mixed-startup profile's numeric values in this detailed table describe only its explicitly labelled successful subset. No rate is invented for a startup failure. Image times are medians of successful client wall-time samples and include image encoding. Expert-cache ranges cover successful starts and show startup → immediately after 16K input, in GiB. **Zero Expert cache means decode Experts run on the CPU; GPU dense computation and streamed prompt work remain active.** Every row uses the same pinned GGUF, and every selected Expert is still computed on the CPU or GPU. Cache placement does not prune Experts, and zero cache does not mean the whole model runs on the CPU. FP8 and INT8 are throughput comparisons, not a claim of identical generated text or general quality equivalence.

### Memory classes and execution order

All rows use the **same physical 24 GB GPU**. For nominal **16 / 12 / 8 GB**, the benchmark parent allocates, writes and synchronizes **8 / 12 / 16 GiB** through CUDA before each fresh API. Native maximum context is **256K / 256K / 128K / 64K** for the **24 / 16 / 12 / 8 GB** rows; only the unrestricted 24 GB row uses automatic context selection. The reservation is retained through the image request, touched again, then released after the owned API is closed. These are available-memory experiments, not speed measurements of physical 16/12/8 GB cards; GPU compute units, bandwidth and system RAM remain those of this PC.

The initial NVML dedicated-memory drop must match the reservation within **512 MiB**; a successful start's final retouch must cause a drop of **less than 256 MiB**. Across the 24 included reserving attempts, the largest initial mismatch was **64.00 MiB**. Across 23 successful reserving starts, the largest signed retouch drop was **0.00 MiB**. These coarse endpoint checks do not prove byte-perfect physical residency at every instant under Windows WDDM.

There are **four planned independent API starts per KV/VRAM profile, 32 planned start IDs**. There were **33 actual attempts**, yielding **31 successful benchmark starts, 1 validated pre-READY startup OOM outcome, and one metadata-save interruption that was replaced by a fresh complete run**. A completed matrix means every planned start ID has a terminal outcome; it does not mean every profile could start. The first attempt was the **8 GB / FP8 pilot**. Its unchanged frozen workload and artifacts allowed it to be included as one of that profile's four starts. The remaining 31 planned starts followed the interleaved class/KV plan: class direction reverses each round and KV order alternates by class and round. This is not a 32-start ABBA experiment. After all nine timed requests of the pilot, two bounded diagnostic questions were run separately: **2/2 correct, 2/2 natural stops**. Their **979 output tokens** do not enter benchmark rates.

The first attempt of `r4-16gb-fp8-mma` stopped with **WinError 5 during the atomic metadata save**, after eight timed text requests and before the image request. Its owned API and reservation were released. The same frozen configuration was then run in a new API from warm-up through the complete workload. All **8 interrupted records are excluded together**, rather than selectively reused. The retained raw request lists contain **287 records**, of which **279** belong to complete formal starts. The exact cause of the save permission error is not established here.

### Startup outcomes

| Planned start | Required hybrid-buffer MiB | Free after initial model placement, MiB | Shortfall MiB |
| --- | ---: | ---: | ---: |
| r1-08gb-int8 | 663 | 615 | 48 |

Only the exact native error that insufficient VRAM remained for hybrid Expert buffers/slots, before READY or warm-up and before any measured request, is classified as startup OOM. Its configuration fingerprint, native log hash, empty measured records, reservation and owned-process cleanup were checked. Other failures are not accepted as completed outcomes. After the first such failure, controller-only handling was extended to record this outcome and continue the unchanged planned workload; existing measured records were retained. These observed margins do not identify an intrinsic INT8-versus-FP8 KV-memory difference, and they do not establish the behaviour of physical 8 GB cards. No 0 tok/s value is assigned to an unstarted model.

### Workload accounting

Each successful start uses one excluded warm-up, then the three canonical generation prompts twice (at most **128 output tokens** each), fresh exact **4,096 / 16,384-token inputs** (at most **32 output tokens** each), and one canonical **1,024 × 768 image** description (at most **128 output tokens**). There were **279 measured requests**, rather than the 288 that 32 successful starts would have produced; startup OOM attempts contributed no timed requests. All measured inputs had zero KV reuse, and all long-input counts matched. Early EOS is respected; actual returned tokens and stop reasons are shown below. Only final returned tokens count.

| Workload | Requests | Actual output tokens | Finish reasons |
| --- | ---: | ---: | --- |
| Generation | 186 | 23,808 | length 186 |
| 4K | 31 | 992 | length 31 |
| 16K | 31 | 992 | length 31 |
| Image | 31 | 2,435 | stop 31 |

Thinking is enabled for text and disabled for the image description. Sampler: temperature **0**, top_p **1**, top_k **64**, min_p **0**, presence penalty **0.9**, repetition penalty **1.05**, frequency penalty **0**, seed **1234**. The resident image encoder remained the same live process within each successful start. Every owned benchmark API and reservation was confirmed released before the next attempt. All planned start IDs have a terminal outcome; the interrupted raw attempt remains retained for provenance.

### Actual automatic settings

| Nominal VRAM | KV | Actual chunk | CPU workers / reported threads | GPU Expert slots: startup → 16K | Mapped KV GiB: startup → 16K | Starts with Graph enabled |
| --- | --- | ---: | --- | --- | --- | ---: |
| 24 GB | INT8 | 512 | — / — | 10240 → 10240 | 2.58 → 2.58 | 4/4 |
| 24 GB | FP8 | 512 | — / — | 10240 → 10240 | 2.58 → 2.58 | 4/4 |
| 16 GB | INT8 | 2048 | 15 / 16 | 5440–5480 → 5040–5161 | 0.23 → 0.31 | 0/4 |
| 16 GB | FP8 | 2048 | 15 / 16 | 5440–5480 → 5071–5139 | 0.23 → 0.31 | 0/4 |
| 12 GB | INT8 | 1024 | 15 / 16 | 2560–2640 → 2321–2526 | 0.20 → 0.27 | 0/4 |
| 12 GB | FP8 | 1024 | 15 / 16 | 2440–2640 → 2323–2527 | 0.20 → 0.27 | 0/4 |
| 8 GB | INT8 | 512 | 15 / 16 | 0 → 0 | 0.20 → 0.27 | 0/3 |
| 8 GB | FP8 | 512 | 15 / 16 | 0 → 0 | 0.20 → 0.27 | 0/4 |

A dash means the field was not reported; it is not a zero-worker setting. Expert slots count resident layer/Expert pairs out of **10,240**. The Graph column reports actual enabled starts, not merely the requested default.

### Reproduction

The public runner reproduces the unrestricted 24 GB workload/profile:

```powershell
python benchmark/run.py --model clover --clover-mode causal --clover-vision --kv-type int8 --chunk auto --engine-env STRATA_IQ_MT_MIN=1 --engine-env STRATA_EXP_MODE=8 --runs 4 --output benchmark-output/clover-24-int8
python benchmark/run.py --model clover --clover-mode causal --clover-vision --kv-type fp8-mma --chunk auto --engine-env STRATA_IQ_MT_MIN=1 --engine-env STRATA_EXP_MODE=8 --runs 4 --output benchmark-output/clover-24-fp8
```

For reduced-memory classes, retain the CUDA reservation in a parent process and set each isolated native configuration's `--max-context` to the class value before constructing `benchmark.run.Service`. The public runner has no one-option nominal-memory-class selector; the measured variant used its existing `controlled_config`, `request_body`, `exact_length_text`, `Service` and `aggregate_model` routines with the explicit context override. The reservation operation is:

```cpp
// Benchmark parent process; no allocation for nominal 24 GB.
void* reservation = nullptr;
const size_t bytes = (24 - nominal_gb) * (1ULL << 30);
cudaSetDevice(0);
if (bytes) {
    cudaMalloc(&reservation, bytes);
    cudaMemset(reservation, 0x5a, bytes);
    cudaDeviceSynchronize();
}
// Start one fresh API and run the fixed workload while retaining reservation.
if (bytes) {
    cudaMemset(reservation, 0x5a, bytes);
    cudaDeviceSynchronize();
}
// Close only the owned API/native/encoder processes, then release reservation.
if (reservation) cudaFree(reservation);
// Every CUDA return value must be checked; this is the reservation sequence.
```

Record the reservation size, initial/retouch checks, context, actual automatic settings and four fresh starts with any reproduction. Raw logs, generated harnesses, models and binaries stay outside Git.

### Artifact identities

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `tinystrata.exe` | 54143488 | `a66cc5474c37c1bf4bf9429993fd41651d5fab8b3b3114ba8db49941f54c15d9` |
| `tinystrata-vision.exe` | 59796480 | `1139019ecd8fae726a7473352750607a3d00d05369f163b69d70accb106e3392` |
| `Clover-35B-A3B-Preview-AD-IQ3_S-IQ3_XXS.gguf` | 15515157952 | `03816cdc365892dfb628d49e0cae2c80003de903966b9d30db2ece231699c2d3` |
| `mmproj-F16.gguf` | 899283584 | `71f3cbc1f7cc0f30d09d41cfa924c0060827ebc33bf15ace7e86661e856f0160` |
| `benchmark/run.py` | 51080 | `8adbc3ccce700deecd9fcb6cdcdd780007246df1d352a0a9848801b102aef2b2` |
| `benchmark/cases.json` | 8945 | `aa49a07349a72542a089222c755d4aa8203ab44b35e22fb3166838ef086eabde` |
| `benchmark/image.png` | 21652 | `b6983fe5adacff8a4fc2f8b7b0cd3b235d8961550e9f9bca7a669396b3d33138` |
| `cublas64_13.dll` | 51571824 | `b181c4ffb454840a49f802169d8826617841875607a2e2ba14ef9650e348c456` |
| `cublasLt64_13.dll` | 453003888 | `182baf9b1846630c2462cccdae744367b8ff4d2965651deb41a2beefb53f1de5` |
| `cudart64_13.dll` | 555120 | `7c50bf7f7398b5d1b654af19ff6a679988cf251cab677dbd8abac69fa1cce310` |

## 日本語

**2026-11-02** に **Windows 11 Pro build 26300**、**RTX PRO 4000 Blackwell SFF Edition（24 GB・24,467 MiB・電力上限 70 W）**、NVIDIA ドライバー **597.30**、**Ryzen 9 7945HX3D（16 コア／32 スレッド）**、システム RAM **64 GiB** で測定しました。エンジンは [7ad7904](https://github.com/kbc20f11/TinyStrata/commit/7ad7904175f50836383b020940aa321fba1e7e1c)、**CUDA コンパイラ nvcc／NVVM は 13.2.86** です。runtime／cuBLAS DLL は公式 **CUDA 13.2.0 再配布物**のものを保持しており、ハッシュを下表に記録しています。CUDA 13.2.2 一式での測定という意味ではありません。固定の AtomicChat Clover AD-IQ3_S-IQ3_XXS を使い、Expert の gate/up は IQ3_XXS、down は IQ3_S です。本体 **15.515 GB** と Clover 専用トークナイザー・学習済み GDN 重みを保持し、共有の **0.899 GB Qwen3.6 projector** をテキスト計測中も GPU に常駐させました。画像対応は TinyStrata による拡張で、公式 Clover はテキスト専用です。

### 条件と測定値

計測ラベルは **`tinystrata-bench-v1-clover-causal-vision-vram-variant`** です。[v1 の固定入力・計時方法](../README.md#日本語)を使う比較条件であり、固定の標準 v1 とは別で、調整用環境変数と仮想的な VRAM 制限を明示した条件です。現在の配布版の既定は **因果 D0／INT8／MT1** で、D3 は任意選択です。全条件で **因果実行 D0**、MTP／別 draft モデルなし、**CPU ワーカー数・入力チャンク・Expert 配置は自動**、KV 段階確保、通常の Graph 選択を使い、`STRATA_IQ_MT_MIN=1` と `STRATA_EXP_MODE=8` を指定しました。保存済みのチャンク 2048 固定設定は、このチャンク自動の比較へ持ち込んでいません。

| 名目 VRAM | KV | 成功／計画起動数 | コンテキスト | 生成 tok/s | 4K 入力 tok/s | 16K 入力 tok/s | 画像 最初／完了 秒 | Expert GiB：起動 → 16K |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 24 GB | INT8 | 4/4 | 256K | 91.61 | 2216.87 | 1931.89 | 1.11 / 2.00 | 12.34 → 12.34 |
| 24 GB | FP8 | 4/4 | 256K | 94.67 | 2244.69 | 2029.67 | 1.11 / 1.94 | 12.34 → 12.34 |
| 16 GB | INT8 | 4/4 | 256K | 58.16 | 2235.75 | 2205.22 | 1.35 / 2.89 | 6.55–6.60 → 6.07–6.22 |
| 16 GB | FP8 | 4/4 | 256K | 58.17 | 2280.15 | 2253.31 | 1.36 / 2.84 | 6.55–6.60 → 6.11–6.19 |
| 12 GB | INT8 | 4/4 | 128K | 51.31 | 1605.06 | 1595.24 | 1.52 / 3.18 | 3.08–3.18 → 2.80–3.04 |
| 12 GB | FP8 | 4/4 | 128K | 50.89 | 1613.57 | 1640.68 | 1.48 / 3.13 | 2.94–3.18 → 2.80–3.04 |
| 8 GB | INT8 | 3/4（成功分のみ） | 64K | 39.46 | 881.25 | 883.78 | 2.12 / 4.26 | 0.00 → 0.00 |
| 8 GB | FP8 | 4/4 | 64K | 39.75 | 879.86 | 886.24 | 2.12 / 4.13 | 0.00 → 0.00 |

テキスト速度は実際のトークン数とネイティブ処理時間を合算して算出し、読み込み・ウォームアップを含めません。README の数値は 4 回すべて起動できた条件だけを示します。成功・失敗が混在する場合、この詳細表の速度は明示した成功分だけの値です。起動できなかった条件には速度を付けていません。画像時間は成功した起動のエンコード込みクライアント実時間の中央値です。Expert キャッシュも成功した起動の実割当範囲を起動時 → 16K 入力直後の順に示します。**Expert キャッシュが 0 の場合、decode の Expert 計算は CPU で行い、dense 計算と入力時のストリーミング処理は GPU を使います。** 全行で同じ固定 GGUF を使い、選択された Expert は常駐配置に応じて CPU または GPU で計算しています。キャッシュ量を減らすために Expert を省略したり、モデル全体を CPU へ切り替えたりするものではありません。FP8／INT8 の表は速度比較であり、出力文の一致や一般的な回答品質の同等性を保証するものではありません。

### 仮想 VRAM と起動順序

すべて **同じ物理 24 GB GPU** を使います。名目 **16／12／8 GB** では、ベンチ親プロセスが CUDA で **8／12／16 GiB** を確保し、全領域への書き込みと同期を行ってから API を新規起動します。最大コンテキストは **24／16／12／8 GB** の順に **256K／256K／128K／64K** で、無制限の 24 GB だけコンテキスト自動選択です。予約領域は画像計測が終わるまで保持して再書き込みし、所有する API を閉じた後に解放します。これは利用可能メモリを減らす実験であり、物理的な 16／12／8 GB カードの速度測定ではありません。GPU の演算器・帯域・システム RAM はこの PC のままです。

NVML で見た初回の専用 VRAM 減少量と予約サイズの差が **512 MiB 以下**、成功した起動では終了時の再書き込みによる専用 VRAM 減少量が **256 MiB 未満**であることを確認しました。正式行に含めた予約あり 24 試行で、初回の差の最大値は **64.00 MiB**、成功した予約あり 23 起動で、再書き込み時の符号付き減少量の最大値は **0.00 MiB** でした。これらは端点での粗い確認であり、Windows WDDM 上で全瞬間の物理常駐をバイト単位で保証するものではありません。

KV／VRAM の各条件で **4 回ずつ、合計 32 件の独立起動を計画**しました。実際の起動試行は **33 回**で、**31 回は測定まで成功、1 回は READY 前の起動 OOM、1 回はメタデータ保存の中断後に新規起動で取り直し**となりました。「完了」は全計画 ID の結果が確定した意味であり、全条件で起動できた意味ではありません。最初は **8 GB／FP8 の試行**で、固定済みの入力・計測処理・成果物が変わらないため、その条件の正式な 4 起動のうち 1 回に含めました。残り 31 件の計画起動は、各ラウンドで VRAM クラスの方向を反転し、KV の順番もクラスとラウンドごとに交互にする計画で進めました。32 起動すべてを ABBA 順で実施したものではありません。最初の試行では 9 件の時間計測の後、別枠の品質問題を 2 問実行し、**2／2 問正答、2／2 問自然停止**でした。この **979 出力トークン**は速度の集計に含めていません。

`r4-16gb-fp8-mma` の初回は、テキスト 8 件の後、画像を送る前のメタデータの atomic 保存で **WinError 5** が発生しました。所有する API と予約領域を解放し、同じ固定設定の新しい API でウォームアップから全項目を取り直しました。中断した **8 件はまとめて集計から除外**し、一部だけを再利用していません。保持した生リクエスト配列は計 **287 件**で、そのうち正式な完了起動に属する **279 件**を集計しています。保存時の権限エラーの詳細な原因までは特定していません。

### 起動失敗の記録

| 計画起動 | 必要な hybrid buffer MiB | モデル初期配置後の空き MiB | 不足 MiB |
| --- | ---: | ---: | ---: |
| r1-08gb-int8 | 663 | 615 | 48 |

失敗の分類対象は、READY／ウォームアップ前かつ計測リクエスト 0 件の時点で、ネイティブエンジンが hybrid Expert 用バッファー／スロットに必要な VRAM の不足を明示した場合だけです。設定の fingerprint、エラーログの SHA、計測データが空であること、予約領域と所有プロセスの解放を確認しました。別の種類の失敗は完了結果として扱いません。最初の該当例の後、計測条件・入力を変えずに、この起動結果を記録して残りを続ける処理だけをコントローラーへ追加し、取得済みのデータを保持しました。これらの空き容量差だけでは INT8／FP8 の KV そのもののメモリ差を特定できず、物理 8 GB カードの挙動も断定できません。起動できなかった条件を 0 tok/s としていません。

### 入出力の会計

成功した各起動で、除外するウォームアップを 1 回実施した後、固定の短文 3 種類を各 2 回（各出力上限 **128 トークン**）、新規入力 **4,096／16,384 トークン**（出力上限 **32**）、固定 **1,024 × 768 画像**の説明（出力上限 **128**）を実行しました。計測できたのは **279 リクエスト**です。32 回すべて成功した場合の 288 件とは異なり、起動 OOM の試行は計測データを持ちません。計測リクエストはすべて KV 再利用 0、長文の新規入力数も規定どおりでした。EOS で早期終了した場合は実際の出力数を使用します。

| 処理 | リクエスト数 | 実際の出力トークン | 停止理由 |
| --- | ---: | ---: | --- |
| Generation | 186 | 23,808 | length 186 |
| 4K | 31 | 992 | length 31 |
| 16K | 31 | 992 | length 31 |
| Image | 31 | 2,435 | stop 31 |

テキストでは思考を有効、画像説明では無効にしています。サンプリングは温度 **0**、top_p **1**、top_k **64**、min_p **0**、presence penalty **0.9**、repetition penalty **1.05**、frequency penalty **0**、seed **1234** です。同じ成功起動内では画像エンコーダが同じ生存プロセスであることを確認し、次の試行前にベンチ所有の API と予約領域を解放しました。公開する計画試行の中に未確定の結果はありません。

### 自動選択された実設定

| 名目 VRAM | KV | 実チャンク | CPU ワーカー／報告スレッド | GPU Expert 数：起動 → 16K | KV 物理 GiB：起動 → 16K | Graph 有効な起動数 |
| --- | --- | ---: | --- | --- | --- | ---: |
| 24 GB | INT8 | 512 | — / — | 10240 → 10240 | 2.58 → 2.58 | 4/4 |
| 24 GB | FP8 | 512 | — / — | 10240 → 10240 | 2.58 → 2.58 | 4/4 |
| 16 GB | INT8 | 2048 | 15 / 16 | 5440–5480 → 5040–5161 | 0.23 → 0.31 | 0/4 |
| 16 GB | FP8 | 2048 | 15 / 16 | 5440–5480 → 5071–5139 | 0.23 → 0.31 | 0/4 |
| 12 GB | INT8 | 1024 | 15 / 16 | 2560–2640 → 2321–2526 | 0.20 → 0.27 | 0/4 |
| 12 GB | FP8 | 1024 | 15 / 16 | 2440–2640 → 2323–2527 | 0.20 → 0.27 | 0/4 |
| 8 GB | INT8 | 512 | 15 / 16 | 0 → 0 | 0.20 → 0.27 | 0/3 |
| 8 GB | FP8 | 512 | 15 / 16 | 0 → 0 | 0.20 → 0.27 | 0/4 |

「—」はネイティブ側で未報告の項目であり、ワーカー数 0 を意味しません。Expert 数はレイヤー／Expert の常駐組数で、全体は **10,240 組**です。Graph 欄は既定の要求値だけではなく、実際に有効だった起動数です。

### 再現方法

無制限の 24 GB 条件の入力・生成設定は、公開 runner の上記 2 コマンドで再現できます。制限条件では、別の親プロセスに CUDA の予約領域を保持させ、起動ごとのネイティブ設定で `--max-context` を対応する容量へ変更してから `benchmark.run.Service` を作ります。公開 runner には名目 VRAM クラスを指定する単独オプションがないため、この比較では既存の `controlled_config`、`request_body`、`exact_length_text`、`Service`、`aggregate_model` を使い、コンテキストだけ明示上書きしました。予約処理は英語欄の CUDA 呼び出しのとおりです。

再計測時も、予約サイズ・初回／終了時の確認・コンテキスト・実際の自動設定・4 回の独立起動をまとめて記録してください。成果物のバイト数と SHA-256 は上の表のとおりです。生ログ、モデル、バイナリ、ローカル生成スクリプトは Git に追加していません。
