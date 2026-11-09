# HIP hybrid routing publication — 2026-11-09

## Change

Qwen3.6, Clover and Tiel share TinyStrata's Qwen-family hybrid Expert driver. Previously each hybrid layer copied routing IDs and normalized input to pinned host memory in two separate DMA calls. The new small-batch path publishes both from one GPU kernel. Each token row always publishes all its IDs; its input is published only if at least one selected Expert runs on the CPU. Expert selection, arithmetic, ordered combination, sampling and draft acceptance are retained.

The implementation adapts Niko1221/Strata's [conditional publication](https://github.com/Niko1221/Strata/commit/a36be1db2973b3e110fa2c7582c6393e2971d107) and [HIP system fences](https://github.com/Niko1221/Strata/commit/0f1390bd11119ae2875913efb4275842d36e142f), MIT; attribution is in `NOTICE`. TinyStrata keeps its existing route-completion event and CPU pool, uses one block per token row and copies stored input bits. It does not adopt the donor's doorbell/polling protocol.

The default is conditional publication on **Windows HIP**. Other HIP hosts retain DMA by default; their driver/handoff costs have not been measured. CUDA retains its existing handoff. `TINYSTRATA_HYBRID_PUBLISH=0` selects the old two-DMA path, `1` unconditional fused publication and `2` conditional publication on HIP. Invalid values select DMA. Batches above 32 rows use DMA, and unavailable mapped-host allocation/device aliases also fall back to DMA.

## Matched-work development measurements

RX 9060 XT 16 GB / gfx1200, Ryzen 9 7945HX3D, 64 GB RAM, Windows 11 build 26300, AMD driver 32.0.32015.7029. Display connected to the Radeon at boot. HIP SDK/runtime 7.17 and the existing verified 64 KiB backing policy are unchanged. Baseline `3b63f35` and candidate engine/main were rebuilt using the same compiler flags and verified common libraries from the installed generation.

These native text-only comparisons use a fixed **6 GiB routed-Expert budget**, context 32,768, chunk 1,120, automatic KV selection and GPU/CPU arithmetic from the existing profiles. Qwen/Tiel use MTP 3 with adaptation/lookup disabled; Clover uses causal D0. Each pair has an excluded 32-token warm-up, then 128 emitted output tokens per workload. Rank seeds are copied identically, `STRATA_Q36_FIXED_SLOTS=1` freezes membership, and greedy sampling/seed 1234 and penalties match. There is no resident image encoder. The Expert budget is not a simulation of a physical 8/12 GB GPU.

| Model / workload | Old DMA, tok/s | Conditional publish, tok/s | Change |
| --- | ---: | ---: | ---: |
| Qwen, short generation | 65.13 | 82.18 | +26.17% |
| Qwen, generation after exact 4K input | 68.06 | 85.32 | +25.35% |
| Qwen, generation after exact 16K input | 70.32 | 84.44 | +20.07% |
| Clover causal, short generation | 31.02 | 42.01 | +35.42% |
| Tiel, short generation | 58.62 | 72.59 | +23.84% |

Qwen's 4K/16K input rates change by +0.33%/+0.60%, effectively flat in this screen. Large hybrid prefill batches retain DMA. Rates count emitted tokens and native decode time, excluding speculative proposals. Each model/condition initially has one old/new start; these bounded development screens do not establish confidence intervals or universal speedups.

A second Qwen short pair with a smaller **4 GiB Expert budget** measures **67.34 → 79.51 tok/s (+18.07%)**, also with matching tokens, draft counts, Expert work and geometry. Different budgets have different CPU/GPU numerical paths and draft acceptance, so compare within each pair rather than ranking the 4/6 GiB absolute rates.

All **12 measured responses / 1,536 emitted tokens** match across their corresponding pairs, as do draft offered/accepted counts, CPU/GPU Expert execution counts, slot fingerprints and allocation/KV geometry. Normal settings and rank files are preserved. The CPU/GPU work is unchanged. In the 6 GiB Qwen short request only 17 of 6,480 published token rows omit hidden input; the gain primarily points to the combined handoff rather than extensive input omission.

## Normal saved-profile integration

One separate `tinystrata-bench-v1-saved-profile` Qwen start per arm retains automatic placement/context/chunk, saved sampling, KV growth, MTP and GPU-resident Vision. Both complete the three generation texts, exact 4K/16K inputs, image request and graceful unload. Configuration files remain unchanged.

Generation pools **87.47 → 107.64 tok/s (+23.06%)**. Startup Expert payload differs (**8,318,296,896 → 8,557,328,416 bytes**, about +228 MiB), so this observed difference is not entirely attributable to publication. The matched-work comparisons above are the adoption evidence. Normal 4K/16K prefill changes by **−0.45%/−0.58%**. Image first/complete time is **2.629/4.444 → 2.518/3.764 seconds**. Both answers identify the colored shapes and A/B/C correctly; both misread the heading text, so this is a compatibility check rather than evidence of improved OCR accuracy.

## Additional checks

- GPU regression: **62 grouped cases / 372 event handoffs**, rows 1/2/4/8/16/32, real width 2,048 and tail width 37; all-hit/all-miss/mixed rows/final-entry miss/null residency table. IDs and required input bits match the DMA reference, including signed zero/subnormals/NaN payloads. Skipped/unused rows retain poison, guards hold and repeated payload/residency changes reuse the same event safely. Fixture hash `526cbfbb7e6a13b2`.
- Isolated timing with mapped-host destinations shows roughly 237 microseconds per two-DMA host/event round trip versus 74–86 microseconds for fused/conditional publication at 1/4 rows. GPU service alone is slower for the mapped kernel (about 10 versus 1 microsecond when copying input), so the end-to-end native results above are the adoption evidence. This microbenchmark is not a generation benchmark or a Linux result.
- The final normal engine builds with the Windows-only default policy. Fresh HIP CMake configure/generate confirms the optional regression target's HIP language and exclusion from normal builds. CUDA engine and regression sources compile for **SM86 and SM120**; no NVIDIA runtime/performance result is claimed.

Raw commands, source/library/binary/runtime hashes, rank snapshots, token IDs and logs are retained under `TinyStrata-work/hybrid-publish-20261109`. Initial fixture issues are retained: the first harness assumed an already-running normal API, and the first Qwen pair sent unsupported `QUIT` before closing stdin. That pair completed all inference normally and exited 0 via EOF; later screens use stdin EOF directly. Neither issue changes measured generation work.

## 日本語

Qwen／Clover／Tielの小さいhybridバッチで、IDと入力を1回のGPU処理でCPUへ公開します。選択済みExpertがすべてGPUにある行は入力の転送も省きます。Expert選択・演算・投機採否は維持します。

RX 9060 XT上の同じ6 GiB Expert配置で、短文生成はQwen **65.13→82.18 tok/s（+26.17%）**、Clover因果 **31.02→42.01（+35.42%）**、Tiel **58.62→72.59（+23.84%）**でした。Qwenの4K／16K入力後も約20〜25%改善し、入力処理はほぼ横ばいです。対応する出力・処理件数・配置は一致しています。

Windows HIPで既定にし、CUDAと他OSの既定は維持します。32行超やmappedメモリを使えない場合は従来転送へ戻ります。既存の標準v1ベンチ表とは別枠の開発時比較です。
