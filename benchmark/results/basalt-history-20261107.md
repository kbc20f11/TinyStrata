# Basalt history review — 2026-11-07

## English

Two correctness/portability fixes are retained. No throughput improvement is claimed, and the standard README benchmark tables are unchanged.

### Selected changes

- **Packed GPU bytes:** extract bytes from register values with shifts and masks instead of taking a byte pointer to a local packed scalar/vector. Reference: Basalt [5bdb6c61](https://github.com/jesdga95/basalt/commit/5bdb6c61ff99247899515c378856f49728e4a69b), [f83263da](https://github.com/jesdga95/basalt/commit/f83263da69e4e03145da3006e145b820836a6db3), and [dbc51581](https://github.com/jesdga95/basalt/commit/dbc515818ec5b74014efbbc69e4c42958a03414d). Upstream reported optimized-CUDA byte-read problems; the problem was **not reproduced in TinyStrata's tested HIP build**. The adapted implementation covers the existing IQ and K-quant paths without changing byte order, signed products or accumulation order.
- **CPU stall watchdog:** arm the timeout with a separate boolean, so wrapping the 32-bit spin counter cannot restart a stalled pool's timer. Reference: Basalt [2c4a0eb6](https://github.com/jesdga95/basalt/commit/2c4a0eb678265f0c5a473cdabfc808577dcaa6f5). Actual task progress still resets the timeout. The existing park waits, task claims, arithmetic and stall recovery remain in place.

The reference repository was fetched through `499fad25`. Recent DeepSeek/Blackwell-specific work was not copied into TinyStrata's different model drivers. Attribution is preserved in `NOTICE`.

### Development comparison conditions

Frozen baseline: TinyStrata `3004025`, executable SHA-256 `aa8dfec0563988dc2bedb1938ea4e615cf005a3f67f1f7869c31d19c191cf9a2`.
Selected executable: `75d5e0c1d4a2cfe9b3e72c15133437afcfea50f7daaee6d16ea2da1dfec6f0ac`.

Windows 11 build 26300, RX 9060 XT 16 GB, Ryzen 9 7945HX3D, 64 GB RAM. GPU power limit was not measured; no power or clock settings were changed. Windows started with the display connected to the Radeon. Both arms loaded the same System32 HIP/COMGR files, retaining HIP 7.12 and the existing **4 MiB VMM quantum**.

These are **development comparisons, not tinystrata-bench-v1**: native text-only execution, no resident image encoder, temperature 0, seed 1234, context 32,768, chunk 2,048, 16 CPU threads. A short LRU prompt and frozen exact 4K/16K token arrays each generate 128 tokens. Checkpoints are disabled; all prompt tokens are read and reused tokens are zero. Output rates count final generated tokens, including thinking, without counting speculative proposals as output. Qwen uses FP8 MMA KV and MTP3; Clover uses FP8 MMA KV and causal D0. The Expert budgets are **manual capacities, not simulations of physical 8/12 GB cards**. Each compared arm has identical placement and geometry.

### Rejected scheduling candidate

The candidate based on Basalt [c1850c40](https://github.com/jesdga95/basalt/commit/c1850c40bd5d8b1590c78a0f6306927d31ae5042) removed phase-boundary park waits after epoch-tagged job completion. A combined screen included this candidate and the packed-byte change, versus the frozen baseline, once per arm:

| Model / Expert budget | GPU Experts | Short decode change | After 4K | After 16K |
|---|---:|---:|---:|---:|
| Qwen / 3 GiB | 2,665 / 10,496 | −1.21% | −0.34% | −0.08% |
| Qwen / 6 GiB | 5,494 / 10,496 | −1.04% | −1.34% | −0.09% |
| Clover / 3 GiB | 2,440 / 10,240 | +0.90% | +2.13% | +1.14% |
| Clover / 6 GiB | 4,920 / 10,240 | +0.84% | −3.24% | −0.76% |

The result did not establish a consistent useful gain. The scheduling candidate and its diagnostic environment switch were removed. Its 3,520-batch CPU stress comparison passed bitwise parity, but correctness alone did not justify adoption. Shutdown admission locking and file-page discard hints were also deferred rather than added without sufficient validation.

### Selected package versus the baseline

The final package retains the old scheduling. Qwen has one baseline and one final start. Clover has a baseline, final start, and an additional original-executable control; its two controls are pooled by total tokens divided by total native time, not the arithmetic mean of rates.

| Model / case | Baseline decode | Selected decode | Change | Input-rate change |
|---|---:|---:|---:|---:|
| Qwen 6 GiB / short | 99.95 | 99.16 | −0.79% | −0.52% |
| Qwen 6 GiB / 4K | 85.54 | 85.40 | −0.16% | +1.72% |
| Qwen 6 GiB / 16K | 83.85 | 84.44 | +0.71% | +0.18% |
| Clover 3 GiB / short | 35.28 | 35.45 | +0.48% | −1.28% |
| Clover 3 GiB / 4K | 31.91 | 31.31 | −1.88% | −1.21% |
| Clover 3 GiB / 16K | 30.93 | 30.67 | −0.84% | −1.67% |

Rates are tok/s. The original Clover executable's 4K input speed itself moved from 310.28 to 291.92 tok/s (−5.92%); the selected build measured 297.18. These bounded measurements do not establish a causal speedup or regression. Adoption is for compiler robustness and correct stall detection, not new headline throughput.

Frozen input IDs, output token arrays/text, generated/read/reused counts, Qwen MTP acceptance/offering counts, placement, KV geometry, threads and runtime DLL hashes matched. No logits/PPL equivalence claim is made. The historical private harness emits an unsupported `QUIT` error only after the final completed request, then closes stdin; measured requests contain no protocol errors.

### Verification

- GPU regression: 65,536 packed words, 1,572,864 byte/scale checks, thread-index validation, and 112 real IQ/K-quant MMVQ cases. Baseline and selected builds match the scalar stored-value reference (maximum error 0 for these fixtures) and output hash `68c406b1054d5cad`.
- Final CPU pool: 1,408 batches with native token policies 1/2, up to 40 workers, sleep/wake, host participation, inactive-output poison guards, mixed formats and the 96-Expert batch boundary; bitwise serial parity passes.
- Production GPU kernels compile for HIP gfx1200 and CUDA sm86/sm120. CPU pool compilation passes Clang and MSVC. NVIDIA runtime performance was not tested on this Radeon machine.
- Saved-profile API smoke QA covers text on Qwen, Clover, Gemma, LFM and Tiel, plus image input on the four image-capable models. This is an operational smoke check, not an image-quality evaluation. The first LFM smoke request incorrectly disabled its mandatory thinking and was rejected with HTTP400; the corrected request was tested separately.
- Saved model profiles, API keys and chat history are preserved. No HIP runtime alignment, VMM-quantum change, CPU-worker default change or model sampling change is included.

## 日本語

Basaltの過去コミットから、量子化GPUデータのバイト読み出しをシフト・マスクへ置き換える修正と、CPU停止監視のスピンカウンタ周回によるタイマー再初期化を防ぐ修正を採用しました。CUDAで報告された誤読を避けるための修正ですが、TinyStrataの今回のHIP実行では誤読を再現していません。演算順序と数値形式は維持しています。

CPUのphase待機削減案は、Qwen/CloverのExpert容量3/6 GiBで有用な一律の改善を確認できず、実験用設定も含めて撤去しました。最終構成は上表のとおり大きな速度改善を示しておらず、高速化としては扱いません。旧実行ファイル自体にも入力速度の変動があったため、対照の合算値と変動を併記しています。Expert容量の制限は実物の8/12 GBカードを完全再現するものではありません。

最終構成のCPU 1,408バッチ、GPU 112ケースとpacked値65,536個の単体検証、Qwen/Cloverの生トークン・投機数・配置・DLLの比較を通過しました。HIP実機に加え、CUDA sm86/sm120とMSVCのコンパイルも確認しています。従来のSystem32 HIP、4 MiB割り当て、CPU設定、モデル設定を維持し、通常のREADMEベンチ表は更新していません。
