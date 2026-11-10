# Other-model HIP optimization screens — 2026-11-10

Qwen3.6・Clover・Tiel Coderでは、CPUのExpert計算結果をGPUが直接読む受け渡しを採用します。Gemmaの3候補とLFMの2候補は、条件を変えた生成速度に一貫した利点がなく、採用を見送りました。

CloverのGQA共有は、gfx1200・INT8 KV・8,192トークン以上・単一行の通常の因果生成に限定して自動適用します。保存済みのKV自動選択は維持し、このRDNA4環境で通常選ばれるFP8経路にはGQA共有を適用しません。FP8での改善はCPU結果の受け渡しによるものとして、INT8の追加改善と区別します。

以下は配置と入力を固定したネイティブ比較です。通常のAPIベンチやREADMEの既存モデル一覧を測り直した値ではありません。モデル全体の比較は競合処理なしで行い、ユニット再検証の時間は性能の採否に使っていません。

Source baseline: `76a912b361cc926575446578b03a131f867aae56`. These are isolated native text A/B/C engineering screens with frozen Expert placement. They are not canonical `tinystrata-bench-v1`, saved-profile API measurements or replacement README benchmark values.

Rates pool emitted tokens and native elapsed milliseconds within the same screen/arm only. Rejected draft proposals do not count as output. Physical VRAM reservation and the separate manual Expert budget are recorded independently. Capacity reservations on this 16 GB GPU do not reproduce another GPU's compute speed or memory bandwidth.

No Expert pruning, weight requantization or approximate acceptance is introduced by these candidates. Equality below applies to the measured inputs and states; it is not a general answer-quality guarantee.

## Decisions

All three tested Gemma ideas were rejected and its native production path restored to the baseline: GEGLU/Q8 fusion, shared MLP input quantization, and grouped Q4_0 weight reuse. The grouped candidate did not produce a consistent end-to-end gain; its short-input screen regressed. This does not establish a general Gemma slowdown.

Both LFM fusion candidates were also rejected; its native production baseline is retained. Generation gain did not persist across reversed/capacity tests. Long prefill gained only0.3..0.8% in additional cells while short-prefill latency regressed. Cold first-MMQ module loading is a possible explanation, not a proven cause

The final default decisions are recorded separately from measured results:

```json
{
  "mapped_cpu_output": {
    "decision": "adopt",
    "default": "Automatic on Windows HIP after explicit coherent/mapped allocation flags, capability and device alias verification; unsupported or failed allocation falls back to DMA",
    "override": "TINYSTRATA_HYBRID_MAPPED_CPU_OUTPUT=0 disables; 1 requests; unset/auto selects automatic policy",
    "models": [
      "Qwen3.6",
      "Clover",
      "Tiel Coder"
    ],
    "precision": "Same existing residual/norm/Q8 combine and CPU results; no Expert pruning or precision change"
  },
  "clover_int8_gqa": {
    "decision": "adopt with measured scope",
    "default": "gfx1200, INT8 Clover, prefix >=8192, nt=1, non-verify and non-preview only; automatic KV selection remains unchanged",
    "override": "TINYSTRATA_CLOVER_INT8_GQA_HEADS=0 disables; 2/4 explicitly force diagnostic widths from4096 for1..4rows; unset/auto selects measured policy",
    "reason": "Forced4-head sharing gave -1.38% generation at4K and +4.82% at16K over mapped-output alone. Automatic policy avoids4K and multirow regressions",
    "scope_note": "Saved auto KV on this RDNA4 GPU normally resolves FP8; INT8 GQA applies when INT8 is selected. Its gains must not be attributed to the stock FP8 route"
  },
  "gemma_candidates": {
    "decision": "reject all three; production source unchanged",
    "candidates": [
      "GEGLU/native-Q8 fusion",
      "shared gate/up input quantization",
      "repeated cached Q4_0 Expert weight reuse"
    ],
    "reason": "No consistent useful end-to-end gain despite byte parity and faster grouped fixed-work kernels"
  },
  "lfm_candidates": {
    "decision": "reject both; production source unchanged",
    "candidates": [
      "native SwiGLU/Q8 fusion",
      "K-quant MMQ prefill activation fusion"
    ],
    "reason": "Generation gain did not persist across reversed/capacity tests. Long prefill gained only0.3..0.8% in additional cells while short-prefill latency regressed. Cold first-MMQ module loading is a possible explanation, not a proven cause"
  },
  "measurement_notes": [
    "Fixed-placement native text diagnostics, temperature0 with neutral penalties and ctx32768; not saved-profile normal API benchmark tables",
    "Capacity limitations include real physical HIP reservation as well as a manual Expert cache budget; no claim of actual8GB/12GB GPU bandwidth",
    "First qwen-map8 screen is invalid due unequal Expert placements; both engines exited0; replaced by qwen-map8b with equal placement",
    "Final model-free unit rechecks were correctness checks; their host-wall micro timings overlapped CUDA compile-only activity and are not adoption performance evidence",
    "All model A/B performance screens ran without native/GPU competition or CPU-heavy compilation"
  ],
  "final_validation": "PASS: API validation all four models, 13 functional cases including stock FP8 Clover image plus INT8 long-prefix GQA and short reset; positive sampling/streaming, normal unload/native exits0, configuration/history unchanged. Six CUDA compile-only checks SM86/SM120 passed; physical NVIDIA execution not performed. Final native SHA256 4ed3dcc333ac2d1d01b3d719eb5b599f3c83d47ae3bf7d38ef82350ce305cbc9."
}
```

## Controlled comparisons

### clover-abc16 — clover

Plan `A,B,C`; physical headroom target **16 GiB**; manual Expert budget **8.0 GiB**; KV `int8`; context `32768`; chunk `2048`; temperature `0`; MTP `0`.

A is the feature-off control; B enables `TINYSTRATA_HYBRID_MAPPED_CPU_OUTPUT`. C additionally uses `TINYSTRATA_CLOVER_INT8_GQA_HEADS=4`.

Startup Experts: `6600/10240`; cache `8142 MiB`; holder reservations `[0, 0, 0]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | C tok/s | B/A change | C/A change | C/B change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 46.70 | 55.42 | 55.53 | +18.66% | +18.90% | +0.21% |
| input_4096 | 46.45 | 54.69 | 53.93 | +17.74% | +16.11% | -1.38% |
| input_16384 | 41.79 | 48.85 | 51.20 | +16.88% | +22.52% | +4.82% |

Input processing:

| Workload | A tok/s | B tok/s | C tok/s | B/A change | C/A change | C/B change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 380.66 | 364.89 | 374.49 | -4.14% | -1.62% | +2.63% |
| input_4096 | 677.13 | 674.52 | 671.71 | -0.39% | -0.80% | -0.42% |
| input_16384 | 615.66 | 614.21 | 617.33 | -0.23% | +0.27% | +0.51% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `None`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

### clover-map-fp816 — clover

Plan `B,A`; physical headroom target **16 GiB**; manual Expert budget **8.0 GiB**; KV `fp8-mma`; context `32768`; chunk `2048`; temperature `0`; MTP `0`.

A is the feature-off control; B enables `TINYSTRATA_HYBRID_MAPPED_CPU_OUTPUT`.

Startup Experts: `6600/10240`; cache `8142 MiB`; holder reservations `[0, 0]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 45.52 | 54.76 | +20.31% |
| input_4096 | 48.75 | 57.91 | +18.79% |
| input_16384 | 47.13 | 55.57 | +17.91% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 356.45 | 358.87 | +0.68% |
| input_4096 | 676.18 | 676.02 | -0.02% |
| input_16384 | 610.48 | 608.93 | -0.26% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `None`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

### clover-map8 — clover

Plan `B,A`; physical headroom target **8 GiB**; manual Expert budget **1.5 GiB**; KV `int8`; context `32768`; chunk `2048`; temperature `0`; MTP `0`.

A is the feature-off control; B enables `TINYSTRATA_HYBRID_MAPPED_CPU_OUTPUT`.

Startup Experts: `1240/10240`; cache `1529 MiB`; holder reservations `[8506048512, 8506048512]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 36.10 | 42.62 | +18.06% |
| input_16384 | 32.22 | 37.50 | +16.40% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 164.88 | 144.81 | -12.17% |
| input_16384 | 241.77 | 240.71 | -0.44% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `None`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

### gemma-fusion16 — gemma

Plan `A,B`; physical headroom target **16 GiB**; manual Expert budget **8.0 GiB**; KV `int8`; context `32768`; chunk `1120`; temperature `0`; MTP `3`.

A is the feature-off control; B enables `TINYSTRATA_GEMMA_FUSED_GEGLU_Q8`.

Startup Experts: `2567/3840`; cache `8190 MiB`; holder reservations `[0, 0]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 81.61 | 81.59 | -0.04% |
| input_4096 | 69.24 | 69.18 | -0.09% |
| input_16384 | 65.62 | 65.52 | -0.15% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 68.71 | 69.59 | +1.28% |
| input_4096 | 2,679.93 | 2,689.43 | +0.35% |
| input_16384 | 2,498.48 | 2,504.59 | +0.24% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `True`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

Decision: rejected — GEGLU/native-Q8 fusion; baseline retained. This screen is not a claim about all Gemma workloads.

### gemma-group16 — gemma

Plan `B,A`; physical headroom target **16 GiB**; manual Expert budget **8.0 GiB**; KV `int8`; context `32768`; chunk `1120`; temperature `0`; MTP `3`.

A is the feature-off control; B enables `TINYSTRATA_GEMMA_GROUPED_Q4_0`.

Startup Experts: `2567/3840`; cache `8190 MiB`; holder reservations `[0, 0]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 81.12 | 75.88 | -6.46% |
| input_4096 | 68.83 | 69.34 | +0.74% |
| input_16384 | 65.66 | 65.61 | -0.08% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 68.77 | 61.75 | -10.21% |
| input_4096 | 2,686.08 | 2,685.90 | -0.01% |
| input_16384 | 2,510.27 | 2,501.79 | -0.34% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `True`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

Decision: rejected — repeated cached Q4_0 Expert weight reuse; baseline retained. This screen is not a claim about all Gemma workloads.

### gemma-share16 — gemma

Plan `A,B`; physical headroom target **16 GiB**; manual Expert budget **8.0 GiB**; KV `int8`; context `32768`; chunk `1120`; temperature `0`; MTP `3`.

A is the feature-off control; B enables `TINYSTRATA_GEMMA_SHARED_MLP_Q8`.

Startup Experts: `2567/3840`; cache `8190 MiB`; holder reservations `[0, 0]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 81.38 | 81.13 | -0.31% |
| input_4096 | 69.23 | 69.24 | +0.02% |
| input_16384 | 65.86 | 65.72 | -0.21% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 69.52 | 69.13 | -0.56% |
| input_4096 | 2,673.28 | 2,675.20 | +0.07% |
| input_16384 | 2,502.06 | 2,503.28 | +0.05% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `True`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

Decision: rejected — shared dense gate/up input quantization; baseline retained. This screen is not a claim about all Gemma workloads.

### lfm-combined12 — lfm

Plan `A,B`; physical headroom target **12 GiB**; manual Expert budget **6.5 GiB**; KV `fp8-mma`; context `32768`; chunk `256`; temperature `0`; MTP `0`.

A is the feature-off control; B enables `TINYSTRATA_LFM_FUSED_SWIGLU_Q8, TINYSTRATA_LFM_FUSED_MMQ_ACTIVATION`.

Startup Experts: `1225/2432`; cache `6653 MiB`; holder reservations `[4211081216, 4211081216]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 54.50 | 54.31 | -0.36% |
| input_4096 | 52.63 | 52.51 | -0.22% |
| input_16384 | 46.72 | 47.20 | +1.03% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 118.51 | 111.55 | -5.88% |
| input_4096 | 1,850.30 | 1,862.92 | +0.68% |
| input_16384 | 1,698.58 | 1,710.28 | +0.69% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `True`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

Decision: rejected; LFM's production baseline is retained. Isolated small gains in this table do not override the final rejection.

### lfm-combined16 — lfm

Plan `B,A`; physical headroom target **16 GiB**; manual Expert budget **10.0 GiB**; KV `fp8-mma`; context `32768`; chunk `256`; temperature `0`; MTP `0`.

A is the feature-off control; B enables `TINYSTRATA_LFM_FUSED_SWIGLU_Q8, TINYSTRATA_LFM_FUSED_MMQ_ACTIVATION`.

Startup Experts: `1885/2432`; cache `10235 MiB`; holder reservations `[0, 0]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 71.68 | 69.11 | -3.58% |
| input_4096 | 68.76 | 68.78 | +0.03% |
| input_16384 | 59.83 | 59.36 | -0.77% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 192.81 | 118.36 | -38.61% |
| input_4096 | 2,195.89 | 2,209.04 | +0.60% |
| input_16384 | 1,850.55 | 1,865.87 | +0.83% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `True`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

Decision: rejected; LFM's production baseline is retained. Isolated small gains in this table do not override the final rejection.

### lfm-combined8 — lfm

Plan `B,A`; physical headroom target **8 GiB**; manual Expert budget **3.0 GiB**; KV `fp8-mma`; context `32768`; chunk `256`; temperature `0`; MTP `0`.

A is the feature-off control; B enables `TINYSTRATA_LFM_FUSED_SWIGLU_Q8, TINYSTRATA_LFM_FUSED_MMQ_ACTIVATION`.

Startup Experts: `566/2432`; cache `3071 MiB`; holder reservations `[8506048512, 8506048512]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 45.08 | 45.44 | +0.79% |
| input_4096 | 43.83 | 44.13 | +0.70% |
| input_16384 | 39.11 | 39.28 | +0.45% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 85.21 | 82.20 | -3.53% |
| input_4096 | 757.80 | 760.59 | +0.37% |
| input_16384 | 759.94 | 762.40 | +0.32% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `True`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

Decision: rejected; LFM's production baseline is retained. Isolated small gains in this table do not override the final rejection.

### lfm-fusion16 — lfm

Plan `A,B`; physical headroom target **16 GiB**; manual Expert budget **12.0 GiB**; KV `fp8-mma`; context `32768`; chunk `256`; temperature `0`; MTP `0`.

A is the feature-off control; B enables `TINYSTRATA_LFM_FUSED_SWIGLU_Q8`.

Startup Experts: `2213/2432`; cache `12019 MiB`; holder reservations `[0, 0]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 84.80 | 85.55 | +0.88% |
| input_4096 | 79.47 | 80.50 | +1.30% |
| input_16384 | 68.10 | 68.91 | +1.19% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 280.09 | 278.20 | -0.68% |
| input_4096 | 2,310.73 | 2,310.86 | +0.01% |
| input_16384 | 1,903.88 | 1,905.27 | +0.07% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `True`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

Decision: rejected; LFM's production baseline is retained. Isolated small gains in this table do not override the final rejection.

### lfm-mmq16 — lfm

Plan `A,B`; physical headroom target **16 GiB**; manual Expert budget **12.0 GiB**; KV `fp8-mma`; context `32768`; chunk `256`; temperature `0`; MTP `0`.

A is the feature-off control; B enables `TINYSTRATA_LFM_FUSED_MMQ_ACTIVATION`.

Startup Experts: `2213/2432`; cache `12019 MiB`; holder reservations `[0, 0]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 84.89 | 85.83 | +1.10% |
| input_4096 | 79.83 | 80.92 | +1.37% |
| input_16384 | 68.50 | 69.20 | +1.03% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 282.66 | 283.96 | +0.46% |
| input_4096 | 2,302.81 | 2,340.97 | +1.66% |
| input_16384 | 1,896.38 | 1,927.10 | +1.62% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `True`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

Decision: rejected; LFM's production baseline is retained. Isolated small gains in this table do not override the final rejection.

### qwen-map-fp816 — qwen

Plan `B,A`; physical headroom target **16 GiB**; manual Expert budget **8.0 GiB**; KV `fp8-mma`; context `32768`; chunk `2048`; temperature `0`; MTP `3`.

A is the feature-off control; B enables `TINYSTRATA_HYBRID_MAPPED_CPU_OUTPUT`.

Startup Experts: `7339/10496`; cache `8160 MiB`; holder reservations `[0, 0]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 102.63 | 119.39 | +16.33% |
| input_4096 | 100.94 | 117.01 | +15.92% |
| input_16384 | 98.50 | 111.80 | +13.50% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 231.11 | 223.30 | -3.38% |
| input_4096 | 656.31 | 656.50 | +0.03% |
| input_16384 | 594.91 | 598.21 | +0.56% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `None`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

### qwen-map12 — qwen

Plan `A,B`; physical headroom target **12 GiB**; manual Expert budget **5.0 GiB**; KV `fp8-mma`; context `32768`; chunk `2048`; temperature `0`; MTP `3`.

A is the feature-off control; B enables `TINYSTRATA_HYBRID_MAPPED_CPU_OUTPUT`.

Startup Experts: `4592/10496`; cache `5106 MiB`; holder reservations `[4211081216, 4211081216]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 81.25 | 94.83 | +16.71% |
| input_4096 | 77.02 | 89.68 | +16.44% |
| input_16384 | 78.71 | 89.76 | +14.03% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 179.00 | 175.94 | -1.71% |
| input_4096 | 397.59 | 395.55 | -0.51% |
| input_16384 | 371.77 | 374.22 | +0.66% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `None`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

### qwen-map16b — qwen

Plan `A,B`; physical headroom target **16 GiB**; manual Expert budget **8.0 GiB**; KV `int8`; context `32768`; chunk `2048`; temperature `0`; MTP `3`.

A is the feature-off control; B enables `TINYSTRATA_HYBRID_MAPPED_CPU_OUTPUT`.

Startup Experts: `7339/10496`; cache `8160 MiB`; holder reservations `[0, 0]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 101.61 | 116.45 | +14.61% |
| input_4096 | 90.66 | 101.34 | +11.78% |
| input_16384 | 72.45 | 79.15 | +9.24% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 235.82 | 232.41 | -1.44% |
| input_4096 | 675.66 | 673.92 | -0.26% |
| input_16384 | 614.92 | 609.91 | -0.82% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `None`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

### qwen-map8b — qwen

Plan `B,A`; physical headroom target **8 GiB**; manual Expert budget **1.5 GiB**; KV `fp8-mma`; context `32768`; chunk `2048`; temperature `0`; MTP `3`.

A is the feature-off control; B enables `TINYSTRATA_HYBRID_MAPPED_CPU_OUTPUT`.

Startup Experts: `1353/10496`; cache `1504 MiB`; holder reservations `[8506048512, 8506048512]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 68.18 | 77.45 | +13.61% |
| input_4096 | 65.18 | 73.66 | +13.01% |
| input_16384 | 65.48 | 72.46 | +10.66% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 142.69 | 144.14 | +1.01% |
| input_4096 | 275.85 | 278.55 | +0.98% |
| input_16384 | 264.95 | 265.37 | +0.16% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `None`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

### tiel-map16 — tiel

Plan `A,B`; physical headroom target **16 GiB**; manual Expert budget **8.0 GiB**; KV `fp8-mma`; context `32768`; chunk `2048`; temperature `0`; MTP `3`.

A is the feature-off control; B enables `TINYSTRATA_HYBRID_MAPPED_CPU_OUTPUT`.

Startup Experts: `7708/10496`; cache `8158 MiB`; holder reservations `[0, 0]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 83.10 | 95.81 | +15.30% |
| input_4096 | 89.28 | 101.89 | +14.12% |
| input_16384 | 88.92 | 102.61 | +15.40% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 529.28 | 549.62 | +3.84% |
| input_4096 | 652.71 | 659.67 | +1.07% |
| input_16384 | 596.76 | 600.60 | +0.64% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `None`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

### tiel-map8 — tiel

Plan `A,B`; physical headroom target **8 GiB**; manual Expert budget **1.5 GiB**; KV `fp8-mma`; context `32768`; chunk `2048`; temperature `0`; MTP `3`.

A is the feature-off control; B enables `TINYSTRATA_HYBRID_MAPPED_CPU_OUTPUT`.

Startup Experts: `1435/10496`; cache `1518 MiB`; holder reservations `[8506048512, 8506048512]` bytes (run order).

Generation:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 53.71 | 60.53 | +12.68% |
| input_16384 | 56.89 | 64.51 | +13.41% |

Input processing:

| Workload | A tok/s | B tok/s | B/A change |
| --- | ---: | ---: | ---: |
| short | 209.99 | 207.25 | -1.30% |
| input_16384 | 233.15 | 227.75 | -2.32% |

Evidence: raw output IDs equal `True`; placement differences `[]`; work differences `{}`; execution-state differences `{}`; native Expert counters equal `None`. Per-run binary hashes, output-ID hashes/lengths, exact arguments, effective INFO deltas, capacity metadata and shutdown receipts are retained in the companion JSON.

## Verification and limits

Shutdown receipts: 70/70 recorded native/reservation exits are clean (exit 0, not forced, not alive). Protected configurations unchanged in all completed screens: `True`.

- `clover_attention_gpu_test.log`: passed `True`; [{"passed": true, "gqa": true, "cases": 27, "checks": 5203, "max_FP64_error": 7.99350719e-07, "H2T4_registers": 69, "H2T4_shared_bytes": 24576, "H4T4_registers": 89, "H4T4_shared_bytes": 49152, "baseline": "current T1 independent; T2/T4 row-shared"}]
- `gemma_activation_quant_test.log`: passed `True`; PASS Gemma GEGLU/shared-Q8 cases=103 compared_bytes=1518660 (Q8 scale/sum/codes; shared gate/up/GEGLU/down; tail/guards; graph replay)
- `gemma_grouped_q4_test.log`: passed `True`; PASS Gemma grouped Q4 cases=31 grouped_layouts=11 fallbacks=16 bitwise gate/up/GEGLU/Q8/down/ordered-sum; guards, graph replay, invalid no-launch; hash=ed451ec9c20d7655
- `lfm_activation_quant_test.log`: passed `True`; PASS LFM SwiGLU/native-Q8: 192 cases, 2377296 compared bytes; guards, inputs and changed-data graphs hold
- `lfm_mmq_activation_test.log`: passed `True`; PASS LFM K-quant MMQ activation: 130 cases, 45522720 compared bytes; DS4/D4, pad512, row offsets, untouched tails, input/guards and changed-data graphs hold
- `q36_cpu_combine_test.log`: passed `True`; PASS 192 coherent mapped residual/norm/Q8 checks; repeated payload/map rewrites and guards

Model-free unit logs are correctness evidence only. Their final rechecks overlapped CUDA compile-only activity, so all unit microtimings are excluded from this report and adoption evidence. The main-model A/B performance screens ran in isolation. Small percentage differences from few process starts are engineering observations, not confidence intervals. The native INFO vision capability flag is retained as reported; it alone does not establish image-encoder residency. No cross-project/laptop/historical throughput comparison is inferred.

Final functional API aggregate: `complete`; configurations/history unchanged `True`; remaining native workers `0`. API text, reasoning override, streaming and image checks are functional validation and do not supply the throughput tables above.

- API `qwen`: pass `True`; cases `['text', 'stream', 'image']`; native SHA-256 `4ed3dcc333ac2d1d01b3d719eb5b599f3c83d47ae3bf7d38ef82350ce305cbc9`.
- API `clover`: pass `True`; cases `['text', 'stream', 'long_text', 'short_reset', 'image']`; native SHA-256 `4ed3dcc333ac2d1d01b3d719eb5b599f3c83d47ae3bf7d38ef82350ce305cbc9`.
- API `tiel`: pass `True`; cases `['text', 'stream', 'image']`; native SHA-256 `4ed3dcc333ac2d1d01b3d719eb5b599f3c83d47ae3bf7d38ef82350ce305cbc9`.
- API `lfm`: pass `True`; cases `['text', 'stream']`; native SHA-256 `a102387a254102aaa9aca92d37a16874a922032ba26ca1804d700f56b354f39b`.

The aggregate preserves and audits the original Clover harness-only log-literal mismatch (`off` versus `disabled`); original artifacts are not edited. Its separate Clover image check explicitly covers stock-auto FP8, while the long/short GQA checks select INT8.

CUDA 13.2.86 compile-only validation: `complete`, 6/6 successful objects (Qwen engine and changed GPU units, SM86/SM120). No NVIDIA execution or speed result is claimed.

Incomplete top-level screens were excluded from speed claims: `qwen-map16, qwen-map8`.

The companion JSON contains no output/prompt ID arrays, protected profile contents or user-specific local paths. Its INFO-after dictionaries are deltas to each run's INFO-before dictionary. Native-provided ID hashes are preserved; a missing hash is derived from the compact JSON ID array and labelled accordingly.
