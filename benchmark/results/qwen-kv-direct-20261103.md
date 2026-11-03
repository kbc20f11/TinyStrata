# Qwen: current INT8 versus native FP8 — 2026-11-03

## English

The latest INT8/FDEC route leads the standard generation group at 24 GB, while FP8 leads after the tested long inputs. At 16/12/8 GB-equivalent, FP8 leads the standard generation group. The 12 GB result after 4K input changes ranking between the 32-token standard and 128-token supplement. **KV defaults remain unchanged**: native FP8 when supported, otherwise INT8.

This compares **current INT8/FDEC and current FP8-MMA in the same executable**, separately from the temperature-0 [FDEC off/on adoption test](fork-adoption-20261103.md). It covers the **text-request portion of standard-v1**, plus a separate 128-token supplement. The image encoder remains on the GPU throughout; **image-description latency is not measured here**.

### Hardware and method

RTX PRO 4000 Blackwell SFF Edition, 24 GB (24,467 MiB reported); Ryzen 9 7945HX3D; 64 GB RAM; Windows; NVIDIA driver 597.30. Model: Qwen3.6-35B-A3B UD-IQ3_XXS; thinking and MTP 3 enabled. Sampling: temperature 0.7, seed 1234, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1.0, frequency penalty 0. Expert/CPU placement, prompt chunk, interleaved matvec and main-model MMQ remain automatic; KV format and per-class context capacity are explicitly fixed.

Each class uses **I1, F1, F2, I2, F3, I3, I4, F4**, four independent starts per format. Each start runs a 32-token warmup, then three standard prompts twice with 128-token output caps, exact 4K/16K inputs with 32-token output caps, then supplementary 4K/16K requests with 128-token caps. The three generation prompts contain 32, 44 and 1,597 input tokens. All requests use the same input text and sampling parameters across formats.

**TG pools actual final generated tokens / native decode time across all four starts**; PP pools input tokens / native prompt time. Per-start tok/s values are not averaged. Draft proposals, loading, startup and warmup are excluded. `strata_checkpoint:false`, exact input/read counts and zero prefix reuse are audited. The 128-token supplement follows all standard requests: Expert priorities and capacity can already have changed, so it is not an output-length-only comparison from an identical cold Expert state.

### Standard-v1 text results

All speeds are tok/s; paired cells are **INT8 / FP8**. The generation group contains 24 measured requests per format across four starts; each long-input row contains four requests per format.

| Memory class | Context capacity | Generation, 128 cap: INT8 | FP8 | After 4K, 32 cap: I / F | After 16K, 32 cap: I / F |
|---|---:|---:|---:|---:|---:|
| 24 GB | 262,144 | **141.52** | 134.41 | 133.46 / **157.99** | 125.28 / **149.90** |
| Approximately 16 GB | 262,144 | 103.69 | **105.15** | 99.70 / **108.01** | 97.34 / **109.39** |
| Approximately 12 GB | 131,072 | 92.89 | **94.74** | **90.92** / 78.81 | 82.94 / **88.91** |
| Approximately 8 GB | 65,536 | 67.17 | **69.95** | 60.63 / **65.04** | 65.39 / **71.62** |

### Supplement: 128-token generation after the standard sequence

| Memory class | After 4K: INT8 / FP8 | After 16K: INT8 / FP8 |
|---|---:|---:|
| 24 GB | 140.67 / **155.01** | 120.87 / **149.08** |
| Approximately 16 GB | 102.52 / **111.26** | 93.97 / **105.63** |
| Approximately 12 GB | 88.98 / **91.50** | 82.45 / **95.38** |
| Approximately 8 GB | 64.17 / **68.48** | 64.58 / **70.07** |

### Input processing: standard 4K/16K requests

| Memory class | 4K input PP: INT8 / FP8 | 16K input PP: INT8 / FP8 |
|---|---:|---:|
| 24 GB | 2,883.69 / 2,927.13 | 2,379.25 / 2,515.21 |
| Approximately 16 GB | 842.92 / 846.73 | 784.83 / 796.26 |
| Approximately 12 GB | 748.71 / 752.00 | 743.73 / 758.23 |
| Approximately 8 GB | 604.22 / 604.45 | 285.19 / 285.99 |

### Automatic capacity and pressure behavior

| Memory class | INT8 Expert slots (weight GiB) | FP8 Expert slots (weight GiB) | Auto chunk / interleaved matvec / main MMQ |
|---|---|---|---|
| 24 GB | 10,496 (11.398) | 10,496 (11.398) | 2,048 / on / on |
| Approximately 16 GB | 6,357–7,061 (6.904–7.662) | 6,404–7,061 (6.955–7.663) | 2,048 / off / off |
| Approximately 12 GB | 3,936–4,018 (4.274–4.363) | 3,936–4,018 (4.274–4.363) | 1,024 / off / off |
| Approximately 8 GB | 526–574 (0.571–0.623) | 528–574 (0.573–0.623) | 512 / off / off |

Ranges span measured request-boundary telemetry, including the 128-token supplement, not temporary intra-prefill loan minima. Expert placement is automatic in both formats; MTP-layer MMQ is off and FDEC is on only for INT8. This measures the usable automatic routes, including CPU/GPU capacity and workspace decisions; main-model MMQ is not forcibly disabled. Smaller-memory classes reserve 8/12/16 GiB on the same 24 GB card, with touched/retouched physical backing checked through NVML. They are **not physical 16/12/8 GB GPU measurements**.

Pressure fallback is retained, including its cost. In all eight 16 GB-equivalent starts, pressure during warmup reduces valid Expert capacity and disables further staging loans. At 8 GB-equivalent, each start records five completed borrows plus two optional preparation failures, with seven successful snapshot restores. The extra restores recover failed preparation attempts before cached CPU/GPU fallback; they are not loan-integrity failures. The slow 16K prefill cases using that fallback remain in the tables.

Two pilots, at 16 and 8 GB-equivalent, stopped on overly strict harness loan classification after ten completed native requests each. Both incomplete pilots and their 20 records are excluded, and each affected class is rerun in full. Only harness classification/cleanup changed; the engine and benchmark settings did not. The final audit contains **32 completed starts, 320 measured records and zero errors**. The controller releases its reservations, preserves all four saved profiles, and restores production Clover; its recorded health reports `loaded:true` and `images:true`.

### Interpretation

KV precision changes output trajectories and MTP acceptance. At 24 GB, the INT8 aggregate advantage coincides mainly with the 1,597-token prompt's MTP acceptance of 85.85% versus 59.56%; the code prompt instead favors FP8 (137.16 versus 143.41 tok/s). Different output hashes and automatic capacity changes preclude an identical-token, fixed-cache kernel interpretation. A 32-token timing sample can also rank differently from the 128-token supplement. These capped requests are not complete-answer quality tests; no format is claimed universally faster, and defaults are unchanged.

Source revision: `5113a763eb229fc596d9b12d677a36b5b16fd53e`. Executable SHA-256: `bc5ac70ac7558ef50943be87d467fe219a71c13e714020264185bec2e5af7f9a`. Cases SHA-256: `aa49a07349a72542a089222c755d4aa8203ab44b35e22fb3166838ef086eabde`. Raw logs, private harnesses, models and binaries stay outside Git.

## 日本語

最新INT8/FDECと最新FP8-MMAを同じbinaryで直接比較しました。温度0のFDEC旧新比較とは別です。24 GBでは通常生成の合算がINT8優位、今回の長文後はFP8優位でした。16/12/8 GB相当の通常生成はFP8が速く、12 GBの4K後は標準32tokenと補足128tokenで順位が変わりました。**既定のKV選択は変更していません**。上の共通表はすべてtok/s、`I / F` はINT8 / FP8です。

今回は **standard-v1のテキスト要求部分**と128token補足を測定しました。画像encoderはGPU常駐ですが、画像説明のlatencyは測定していません。実機・モデル・samplingは上記のとおり、temperature 0.7、MTP 3、思考有効。Expert/CPU配置・入力chunk・IL・MMQは自動のままです。各条件で **I1, F1, F2, I2, F3, I3, I4, F4** と独立起動し、各起動で3入力を2回・128token、標準4K/16Kを32token、最後に補足4K/16Kを128token上限で生成します。

TGは **4起動分の実生成token合計÷native decode時間合計**で、起動ごとのtok/sの単純平均ではありません。PPも入力token合計÷native prompt時間合計です。draft・ロード・起動・warmupは除き、prefix再利用0を確認しました。128token補足は標準要求後のため、Expertの学習・容量変化を含む別枠です。完全に同じ冷状態で出力長だけ変えた比較ではありません。

Expert表の範囲は補足も含む要求境界での実測値です。MMQは手動で無効化せず、24 GBでは自動で有効、16/12/8 GBではCPU/GPU併用経路により無効でした。小VRAM条件は同じ24 GB GPUへ8/12/16 GiBを実体予約したもので、物理16/12/8 GBカードの実測ではありません。16 GBでの有効slot縮小・loan停止と、8 GBでのoptional準備失敗→snapshot復元成功→CPU/GPU fallbackも、遅さを含めて保持しています。8 GBの16K prefillが遅い有効な要求も除外していません。

16 GBと8 GBのpilot各1起動は、native 10要求ずつ完了後にharnessの過剰なloan分類で停止したため、該当20要求を除外し、対象条件を全起動で再計測しました。engine・計測設定の変更はありません。最終監査は **32起動・320要求・エラー0**。controllerは予約を解放し、保存4profile不変を確認、元のCloverへ復帰しました。記録されたhealthも `loaded:true`・`images:true` です。

24 GBでは1,597token入力で生成速度とMTP採用率の差が大きく、採用率はINT8 **85.85%**、FP8 **59.56%** でした。一方コードpromptはFP8が速い結果です。KV形式で出力とMTP採用が変わり、自動Expert容量にも差があるため、同一token列・同一キャッシュ容量の演算速度差とは断定しません。性能用の出力上限と完全回答の品質確認も区別し、用途を問わず片方が速いとは主張しません。生ログ・private補助・モデル・binaryはGitへ追加しません。
