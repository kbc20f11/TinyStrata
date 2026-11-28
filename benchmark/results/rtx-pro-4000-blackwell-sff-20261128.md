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

The FP16 CUDA build automatically enabled lending for the single-slot, GPU-first FP16/256K profile on 24 GiB Blackwell when dedicated VRAM availability before loading was at least 20 GiB and Expert placement was automatic. Explicit `STRATA_INK_KV_LEND=0` or `=1` takes precedence under the existing eligibility checks. The build passed the 34,184-token document workflow with nine out of nine facts, a cached image follow-up, image replacement, invalid-image recovery and context-overflow recovery. All 3,072 Experts became resident, and the reported context stayed at 256K.

### Virtual 16 GiB: KV formats and routed Expert staging

This screen used one fresh start per row, temperature 0, thinking off, the same compiled CUDA binary, an abbreviated generation workload, exact 4K and 16K inputs, and a short image OCR request. All runs kept a 256K context, 512-token chunks and CUDA Vision. No run activated automatic KV lending.

| KV and staging | Resident Experts | Generation tok/s | 4K input tok/s | 16K input tok/s |
| --- | ---: | ---: | ---: | ---: |
| FP16, full | 0 | 22.45 | 451.42 | 437.14 |
| INT8, full | 944 | 37.43 | 661.46 | 626.95 |
| FP8, full | 944 | 37.01 | 659.39 | 619.54 |
| FP16, routed only | 0 | 22.22 | 487.03 | 473.76 |

Routed staging reduced the recorded transfer bytes by 26.7%, with identical output hashes in this screen. FP8 also matched those hashes. INT8 differed on the capped 4K and 16K responses. All four arms passed the three expected-value assertions and returned the correct image heading, `TinyStrata`, from the reference image. The automated OCR check used a stale heading, so the raw outputs were rescored against the actual image. These single starts on virtual 16 GiB establish a speed signal.

### Managed FP8 KV and lending on Blackwell

The follow-up sets FP8 KV and automatic Expert lending for the managed 256K profile on 16–24 GiB Blackwell CUDA GPUs. The app starts each profile with `--kv auto` and no lending environment override. Each row uses one fresh start at temperature zero with two short code requests, fixed 4K and 16K inputs, and a capped CUDA image OCR request. The virtual row uses the original 24 GiB GPU with a dedicated-memory reservation.

| Capacity | Effective KV | Generation tok/s | 4K input tok/s | 16K input tok/s | GPU Experts at startup → after 16K |
| --- | --- | ---: | ---: | ---: | ---: |
| Physical 24 GiB | FP8 / 256K | 48.92 | 1,205.63 | 1,090.91 | 2,821 → 3,072 |
| Virtual 16 GiB | FP8 / 256K | 41.07 | 892.16 | 835.00 | 950 → 1,814 |

The automatic virtual 16 GiB profile passed the 54,172-token document workflow with nine of nine reference facts. The engine reclaimed 192 Expert loans as the document grew, then restored the earlier Expert count in a fresh conversation. Cached and fresh CUDA image requests, an invalid-image retry and a context-overflow retry completed. Separate starts confirmed that lending can be explicitly disabled with FP8 active and that an explicit INT8 request selects K8V8.

The first singleton step used fused Split-K Flash-GQA with FP8 KV. An instrumented physical 24 GiB comparison held FP8, lending and the 512-token chunk constant and verified both prefill dispatches in the native log. The standard prefill uses BF16 Tensor Core tiles and online softmax. Each route used one fresh start with the same requests.

| Prefill route | 4K input tok/s | 16K input tok/s |
| --- | ---: | ---: |
| Standard tiled prefill | 1,187.90 | 1,069.11 |
| Forced fused Flash-GQA prefill | 907.54 (−23.60%) | 693.00 (−35.18%) |

The two routes matched the output hashes on all cases except the capped 4K response. The measured long-input default uses the higher-throughput tiled route.

### Blackwell input chunk and generation follow-up

The study kept the pinned IQ4 weights, FP8 KV, 256K context, one slot, CUDA Vision and KV lending. A real CUDA allocation reserved approximately 8 GiB on the 24 GB card for the virtual 16 GiB runs. Prompt inputs were exactly 4,096 and 16,384 tokens. The following rates are diagnostic starts and use the same quick workload.

| Capacity | Prefill chunk | 4K input tok/s | 16K input tok/s |
| --- | ---: | ---: | ---: |
| Physical 24 GiB | 512 | 1,172.33 | 1,053.35 |
| Physical 24 GiB | 1024 | 1,228.70 | 1,101.37 |
| Physical 24 GiB | 1536 | 1,235.41 | 1,105.70 |
| Physical 24 GiB | 2048 | 937.41 | 718.74 |
| Virtual 16 GiB | 512 | 870.99 | 816.53 |
| Virtual 16 GiB | 1024 | 1,037.88 | 950.63 |
| Virtual 16 GiB | 1536 | 1,076.93 | 983.92 |
| Virtual 16 GiB | 2048 | 1,129.74 | 1,023.14 |

The 1536-token virtual 16 GiB run retained 256K and left 546 MiB after the runner initialized. The 2048-token run left 407 MiB. Automatic tuning uses 1024 with sufficient memory, and considers 1536 when most Experts are in RAM and the available workspace is larger. The initialization checks the actual device context and remaining memory before keeping a larger chunk.

On virtual 16 GiB, selecting only routed Experts at chunk 512 moved about 30% fewer bytes and improved 16K input from 816.53 to 839.11 tok/s. BF16 KV tile reuse at chunk 1024 gave 954.31 vs 950.63 tok/s. The wider staging buffer at 1024 reached 993.99 tok/s but reduced short generation from 40.85 to 39.29 tok/s. Visit-weighted Expert promotion had 41.16 tok/s generation and 925.23 tok/s at 16K. The release uses the measured chunk policy. [Basalt visit weighting](https://github.com/jesdga95/basalt/commit/40000f2de47032c63f0c5053b20a3417c315941f)

Warmed singleton generation on physical 24 GiB, after two independent conversation requests populated all 3,072 Experts, was 47.1 tok/s on the control. Skipping empty Expert staging measured 47.3, grouped BF16 Q/K/V measured 47.1, and combining the staging skip with fused MoE reduction and ShortConv measured 47.3. These arms used the same binary with dedicated switches and a separate 256-token warm response.

A separate same-binary comparison computed only the relative-position distances reachable by the current singleton query. Warm generation measured 46.8 tok/s on the control and 47.0 with cropping. The short generation candidates had sub-percent warm differences and were held back.

The virtual 16 GiB context comparison explicitly held FP8, KV lending, and chunk 512 constant. Each setting used one fresh start.

| Context | Global KV MiB | GPU Experts at startup → after 16K | Generation tok/s | First code generation tok/s | Generation after 16K tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256K | 4,224 | 930 → 1,794 | 41.22 | 37.0 | 37.6 |
| 128K | 2,112 | 1,419 → 1,803 | 42.48 | 40.5 | 37.7 |

The shorter cache placed 489 more Experts at startup and helped the first response. At 16K the loaned placement was similar and the generation rates converged. The managed 256K policy is used in the chunk tuning results.

The managed build selected chunk 1024 with no overrides at both capacities. Physical 24 GiB measured 46.93 generation, 1,233.47 4K and 1,103.38 16K tok/s. Virtual 16 GiB measured 41.10 generation, 1,048.91 4K and 959.62 16K tok/s. A separate virtual 16 GiB run allocated another 640 MiB after Expert placement. The engine attempted chunk 1024, discarded it for insufficient headroom, and retried with chunk 512. It retained 262,144 device KV rows and completed the code, 16K input and CUDA image workloads.

Compared with the virtual 16 GiB 256K context control above, the release run held the same 930 startup and 1,794 post-16K GPU Experts. It improved 4K input by 17.6% and 16K input by 15.6%, with a 0.3% generation difference. The capped 16K answer changed wording. The other response hashes matched, and the actual 54,172-token logistics workflow returned all nine expected facts, reclaimed 192 Expert loans, restored the initial warm residency in a new conversation, and completed cached and fresh GPU image requests and error retries.

The physical 24 GiB release run improved 4K and 16K input by 5.2% and 4.7% against the earlier 512-token start. Startup Expert residency was higher in the release run, so these are deployment measurements with the same requests. In the same-binary physical 24 GiB screen, explicit 1024 improved 16K input from 1,053.35 to 1,097.73 tok/s.

### Native Blackwell FP4 conversion

The Inkling CUDA runner adds an `sm_120a` device image alongside the portable image. The Blackwell Expert kernels use the hardware FP4-to-FP16 conversion. Both builds selected FP8 KV, 256K context, CUDA Vision, and a 1024-token chunk automatically. The diagnostic uses greedy generation with the same prompts and GPU Expert counts within each pair. [Strata change](https://github.com/Niko1221/Strata/commit/2cbe802195d8c7f7545d5ac76b27e8778dd981f7)

| Capacity | Generation tok/s | 4K input tok/s | 16K input tok/s |
| --- | ---: | ---: | ---: |
| Physical 24 GiB | 46.91 → 50.95 (+8.6%) | 1,232.25 → 1,230.66 | 1,098.59 → 1,098.24 |
| Virtual 16 GiB | 41.32 → 43.73 (+5.8%) | 1,041.92 → 1,040.76 | 949.40 → 949.44 |

Greedy outputs matched in both pairs. At virtual 16 GiB the 54K logistics workflow returned all nine expected facts, reclaimed 192 Expert loans in each build, and matched responses across 15 requests covering text, CUDA images, and a retry after invalid input.

The v3 workload confirmed the gain with application-default sampling and medium reasoning. Each build used one fresh start per capacity in the reverse order of the diagnostic. Expert counts, effective chunk size, requests, text and reasoning matched across each pair.

| Capacity | Generation tok/s | 4K input tok/s | 16K input tok/s | Image completion s |
| --- | ---: | ---: | ---: | ---: |
| Physical 24 GiB | 44.57 → 48.38 (+8.6%) | 1,224.73 → 1,220.65 | 1,099.06 → 1,099.67 | 7.89 → 7.31 |
| Virtual 16 GiB | 40.99 → 43.98 (+7.3%) | 1,044.92 → 1,051.55 | 961.97 → 968.42 | 6.55 → 6.15 |

## Evidence

Raw measurements are retained privately under `blackwell-followup-20261128`. Gemma throughput is summarized in `gemma-speed-summary.json`, with raw v3 runs and the two experimental dispatch receipts. Coding and retrieval results are in `gemma-quality-code/summary.json`, `gemma-quality-code-candidate/summary.json` and `gemma-quality-ruler/summary.json`. Inkling's per-capacity `result.json` files contain actual prompts, outputs, placements, settings and workload hashes. `inkling/results-1x*.json` contains reservation and image-path evidence. The same-binary comparisons are in `inkling-ab-lend-24/ab.json` and `inkling-quick-kv-v16/matrix.json`. Reclamation and managed-default workflows are saved under `inkling-candidate/qa-lend-reclaim-v16/` and `inkling-candidate/qa-lend-managed-default-24/`. The original Inkling CUDA executable's SHA-256 was `27029e368d0af368e9cf52af0a8742dd09ada8c4620483857f5fc6c24a7d4705`. The updated executable's SHA-256 was `ec5776b002fa2ddda126896212b11273cb6947fceea26ae4cc7a321f591324be`.

Follow-up measurements are under `inkling-blackwell-loop-20261128`. `prefill-screen`, `native-screen`, and `chunk1536` retain the chunk and transfer experiments. `decode-abcd` and `relative-decode` retain the generated sequences and activation logs. `context-compare`, `release-chunk-qa`, and `release-long-qa` retain the fixed-format 128K/256K comparison, managed starts, and long workflow.

The native FP4 conversion comparison is under `inkling-sm120a-20261128`. `speed/ab/report.json` and `speed/v3-ab/report.json` record the matched runs. `qa/realmodel-ab/summary.json` records the long workflow and response checks.
