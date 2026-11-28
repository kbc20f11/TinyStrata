# RTX PRO 4000 Blackwell SFF: Gemma KV and Inkling 2 Mini

Measured on **2026-11-28** with an RTX PRO 4000 Blackwell SFF 24 GB, Ryzen 9 7945HX3D, 61.7 GiB usable RAM, and Windows build 26300. Benchmark and API source [`3c38119`](https://github.com/kbc20f11/TinyStrata/commit/3c38119e7b4607833f941a14f56f857473ab5e5a). The installed CUDA `sm_120` build came from [`0b8f14a`](https://github.com/kbc20f11/TinyStrata/commit/0b8f14a28759cbfadbb780f5430128a8fadeb03c), with the same native source files.

The 24 GiB row uses the physical card without a reservation. The 16, 12 and 8 GiB rows retain real CUDA VRAM reservations verified with NVML. The GPU's compute resources and bandwidth are those of the same Blackwell card in every row. For virtual 12/8 GiB, the private harness explicitly requests the 128K/64K contexts selected by the production capacity policy on equivalent physical cards. Results report the actual context when the engine reduces it.

## Gemma 4 26B-A4B CUDA KV comparison

Gemma's CUDA default remains **INT8**. The existing FP8 route increased generation speed by 0.3–6.8% in this comparison and reduced 16K input throughput by 26.1–29.9% at every capacity. The 24 GiB comparison used two fresh starts per format in INT8 → FP8 → FP8 → INT8 order. The virtual conditions each used one fresh start per format. Both arms use the v3 workloads, seed 1234, equivalent model and assistant weights, MTP 3, automatic Expert placement and GPU Vision. Rates are INT8 / native FP8-MMA in each cell, followed by the FP8 change.

| VRAM | Actual context | Generation tok/s | 4K input tok/s | 16K input tok/s |
| --- | ---: | ---: | ---: | ---: |
| Physical 24 GiB | 256K | 114.05 / 115.49 (+1.26%) | 3,443.46 / 2,923.83 (-15.09%) | 3,037.84 / 2,128.18 (-29.94%) |
| Virtual 16 GiB | 256K | 106.70 / 110.03 (+3.12%) | 2,647.88 / 2,349.57 (-11.27%) | 2,586.96 / 1,887.82 (-27.03%) |
| Virtual 12 GiB | 128K | 96.57 / 96.84 (+0.28%) | 2,289.80 / 2,094.60 (-8.52%) | 2,530.62 / 1,856.69 (-26.63%) |
| Virtual 8 GiB | 64K | 54.68 / 58.37 (+6.75%) | 2,035.89 / 1,894.19 (-6.96%) | 2,520.77 / 1,862.58 (-26.11%) |

### Gemma FP8 prefill candidate and quality

An experimental CUDA `sm_120` build kept FP8 KV storage and dequantized its tiles to FP16 for the prefill BLAS path. Each state below was measured once on the same executable at 24 GiB. Its dispatch log confirmed the experimental route was active only when enabled. Both runs had all 3,840 Experts resident.

| FP8 prefill route | Generation tok/s | 4K input tok/s | 16K input tok/s |
| --- | ---: | ---: | ---: |
| Existing native FP8 | 112.42 | 2,813.38 | 2,058.21 |
| FP8 storage, tiled FP16 BLAS prefill | 113.21 (+0.70%) | 3,279.42 (+16.57%) | 2,911.73 (+41.47%) |

The experimental prefill recovered much of the lost input throughput, with the measured rates still below the INT8 baseline. A separate fixed-input coding check with MTP disabled reproduced a concrete correctness error in the existing FP8 path. The Python function was valid, but one generated assertion omitted `-1` from the expected output. INT8 passed all three authored assertions, existing FP8 passed three of four, and the tiled prefill candidate passed all four while retaining FP8 storage. This isolates the observed failure to a change in how the attention computation is performed for this case.

On three RULER retrieval cases with approximately 32K input tokens, the existing FP8 route matched the expected output on two cases and INT8 on one. These small checks establish the reproduced failure and its correction in the candidate. They provide limited evidence about overall answer quality. The tested candidate does not change the default selection.

## Inkling 2 Mini default capacity baseline

`tinystrata-bench-v3-inkling` workloads, one fresh start per capacity. The same pinned Unsloth UD-IQ4_XS model, native Vision shard, automatic KV and chunk selection, automatic Expert placement, medium reasoning and drafting off are used in every row. Vision timings include the in-process encoder. Generation and input speeds pool the actual native token and timing measurements within each start.

| VRAM | Effective KV | Actual context | Chunk | Generation tok/s | 4K input tok/s | 16K input tok/s | Image first / complete (s) | Resident Experts startup → after 16K | Vision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Physical 24 GiB | FP16 | 256K | 512 | 42.66 | 953.42 | 885.16 | 1.21 / 10.06 | 1,869 → 1,869 of 3,072 | CUDA |
| Virtual 16 GiB | FP16 | 256K | 512 | 21.23 | 450.49 | 436.30 | 2.53 / 19.60 | 0 → 0 of 3,072 | CUDA |
| Virtual 12 GiB | FP16 | 128K | 512 | 21.21 | 450.35 | 436.14 | 2.54 / 19.70 | 0 → 0 of 3,072 | CUDA |
| Virtual 8 GiB | FP16 | 29,120 (64K requested) | 512 | 20.87 | 450.29 | 436.58 | 2.53 / 19.76 | 0 → 0 of 3,072 | CUDA |

On the managed Blackwell route, automatic selection resolved to FP16, allocating about 8.23 GiB of GPU KV at 256K and 4.23 GiB at 128K. This accompanies a drop to zero resident Experts in the 16/12/8 GiB tests. At 8 GiB, the first attempt reached 28,992 tokens, so the exact-context verification aborted that run. The separate clamped-capacity run completed all requests at 29,120 tokens, including an image encoded on CUDA. The observed limits reflect the available memory during each fresh start.

### Physical 24 GiB: lending unused KV rows to Experts

The same binary ran the v3 workloads at temperature 0 in off → on → on → off order. Each arm used a fresh start, the same FP16 KV, 256K context, 512-token input chunks, CUDA Vision, and 1,872 resident Experts at startup. The two enabled runs lent the unused KV space to all 1,200 remaining Experts by the 16K measurement. The slots are returned before attention writes those KV rows. The 16K workload alone does not reach the loaned rows in a 256K context.

| KV lending | Generation tok/s | 4K input tok/s | 16K input tok/s | Image first / complete (s) | Resident Experts after 16K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | 44.37 | 953.86 | 882.05 | 1.209 / 8.888 | 1,872 / 3,072 |
| On | 45.66 (+2.89%) | 1,181.73 (+23.89%) | 1,057.96 (+19.94%) | 0.858 / 8.278 | 3,072 / 3,072 |

The table shows the median of two starts per arm. The first measured code response took 3.42 s with lending and 3.23 s without it. The repeated code response took 2.78 s with lending and 3.25 s without it. Greedy code, long-text and fixed-length input answers and reasoning matched their baseline hashes. Explanation reasoning and full image answer text varied. The four image answers correctly identified the text, colors, shapes, labels and layout.

A separate virtual 16 GiB run used an explicit 64K FP16 context and forced lending to exercise actual reclamation. During a 54,172-token logistics workflow, 384 Experts were reclaimed as KV expanded. Resident Experts went from 1,422 at startup to 1,838 after priming, down to 1,454 after the document, and back to 1,838 in a fresh conversation. The model matched all nine reference facts, completed the cached image follow-up with CUDA Vision, and handled image replacement, invalid-image retry and context-overflow retry.

The new CUDA build automatically enables lending for the single-slot, GPU-first FP16/256K profile on 24 GiB Blackwell when dedicated VRAM availability before loading is at least 20 GiB and Expert placement is automatic. Explicit `STRATA_INK_KV_LEND=0` or `=1` takes precedence under the existing eligibility checks. With the normal managed options and no environment override, the new build activated lending and passed the 34,184-token document workflow with nine out of nine facts, a cached image follow-up, image replacement, invalid-image recovery and context-overflow recovery. All 3,072 Experts became resident, and the reported context stayed at 256K.

### Virtual 16 GiB: KV formats and routed Expert staging

This screen used one fresh start per row, temperature 0, thinking off, the same compiled CUDA binary, an abbreviated generation workload, exact 4K and 16K inputs, and a short image OCR request. All runs kept a 256K context, 512-token chunks and CUDA Vision. No run activated automatic KV lending.

| KV and staging | Resident Experts | Generation tok/s | 4K input tok/s | 16K input tok/s |
| --- | ---: | ---: | ---: | ---: |
| FP16, full | 0 | 22.45 | 451.42 | 437.14 |
| INT8, full | 944 | 37.43 | 661.46 | 626.95 |
| FP8, full | 944 | 37.01 | 659.39 | 619.54 |
| FP16, routed only | 0 | 22.22 | 487.03 | 473.76 |

Routed staging reduced the recorded transfer bytes by 26.7%, with identical output hashes in this screen. FP8 also matched those hashes. INT8 differed on the capped 4K and 16K responses. All four arms passed the three expected-value assertions and returned the correct image heading, `TinyStrata`, from the reference image. The automated OCR check used a stale heading, so the raw outputs were rescored against the actual image. These single starts on virtual 16 GiB establish a speed signal. The adopted default applies to the separately repeated and validated 24 GiB lending configuration.

## Evidence

Raw measurements are retained privately under `blackwell-followup-20261128`. Gemma throughput is summarized in `gemma-speed-summary.json`, with raw v3 runs and the two experimental dispatch receipts. Coding and retrieval results are in `gemma-quality-code/summary.json`, `gemma-quality-code-candidate/summary.json` and `gemma-quality-ruler/summary.json`. Inkling's per-capacity `result.json` files contain actual prompts, outputs, placements, settings and workload hashes. `inkling/results-1x*.json` contains reservation and image-path evidence. The same-binary comparisons are in `inkling-ab-lend-24/ab.json` and `inkling-quick-kv-v16/matrix.json`. Reclamation and managed-default workflows are saved under `inkling-candidate/qa-lend-reclaim-v16/` and `inkling-candidate/qa-lend-managed-default-24/`. The original Inkling CUDA executable's SHA-256 was `27029e368d0af368e9cf52af0a8742dd09ada8c4620483857f5fc6c24a7d4705`. The updated executable's SHA-256 was `ec5776b002fa2ddda126896212b11273cb6947fceea26ae4cc7a321f591324be`.
