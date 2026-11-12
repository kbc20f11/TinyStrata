# RTX PRO 4000 Blackwell SFF: five-model capacity measurement

Measured on **2026-11-12**, Windows build 26300, **RTX PRO 4000 Blackwell SFF 24 GB**, driver **597.30**, **Ryzen 9 7945HX3D / 64 GB RAM**, source [`811bd9d`](https://github.com/kbc20f11/TinyStrata/commit/811bd9de7fa6c67bd61303e1fece72cbb12705df). The CUDA compiler is nvcc/NVVM 13.2.86. Runtime/cuBLAS components have their own versions; exact DLL resource versions and hashes are listed below.

**Complete: 20/20 model/capacity starts, one per condition.** The three nonzero VRAM reservations were released, and the installed configuration files remained unchanged.

One fresh API start per model at each nominal 24/16/12/8 GB condition. This is a managed-default capacity variant using the saved-profile runner, separate from standard four-start tinystrata-bench-v1. Private overrides to the saved profiles are described below.

Each start excludes its 32-token warm-up, runs the three canonical generation texts once each (up to 128 output tokens including thinking), then fresh exact 4096/16384-token inputs and an image for models with Vision. Rates pool actual native tokens and milliseconds. Loading and warm-up are excluded; early EOS uses actual emitted counts.

KV policy: Qwen3.6/LFM2.5/Tiel native FP8; Gemma 4/Clover INT8. Gemma FP8 remains experimental and was not tested here. MTP 3 is retained for Qwen/Gemma/Tiel; Clover uses causal D0 without drafts. Automatic placement and chunk selection are retained; the private comparison resets a saved local Clover 2048 chunk to auto.

Context capacities are 256K/256K/128K/64K for Qwen/Gemma/Clover/Tiel and 128K/128K/128K/64K for LFM. Smaller conditions retain a real CUDA allocation equal to observed NVML total memory minus the target 16/12/8 GiB budget. Exact reserved bytes and release results are recorded in the [JSON result](rtx-pro-4000-blackwell-sff-20261112.json); the 24 GB row is unrestricted. The same GPU's bandwidth and compute units are retained in every condition.

| Nominal VRAM | Model | Actual KV / context | Generation tok/s | 4K input tok/s | 16K input tok/s | Image first / end s | Expert GiB: startup → 16K |
|---|---|---|---:|---:|---:|---:|---|
| 24 | Qwen3.6 | fp8-mma / 262144 | 145.48 | 3243.84 | 2783.27 | 0.65 / 1.50 | 11.40 → 11.40 |
| 24 | Gemma 4 | int8 / 262144 | 113.24 | 3461.21 | 3062.26 | 0.57 / 1.22 | 11.96 → 11.96 |
| 24 | LFM2.5 | fp8-mma / 131072 | 162.21 | 2991.09 | 2677.04 | N/A | 12.90 → 12.90 |
| 24 | Clover | int8 / 262144 | 96.80 | 2421.52 | 2086.18 | 1.02 / 1.83 | 12.34 → 12.34 |
| 24 | Tiel Coder | fp8-mma / 262144 | 112.20 | 892.12 | 840.64 | 1.33 / 2.45 | 10.85 → 10.85 |
| 16 | Qwen3.6 | fp8-mma / 262144 | 107.72 | 891.69 | 848.64 | 1.71 / 3.18 | 7.06 → 7.06 |
| 16 | Gemma 4 | int8 / 262144 | 110.39 | 2675.38 | 2585.74 | 0.86 / 1.54 | 8.61 → 8.44 |
| 16 | LFM2.5 | fp8-mma / 131072 | 147.22 | 2698.46 | 2398.86 | N/A | 12.69 → 12.50 |
| 16 | Clover | int8 / 262144 | 63.92 | 2286.86 | 2094.52 | 1.35 / 2.64 | 6.60 → 6.32 |
| 16 | Tiel Coder | fp8-mma / 262144 | 83.83 | 967.91 | 896.78 | 1.55 / 3.09 | 7.07 → 7.07 |
| 12 | Qwen3.6 | fp8-mma / 131072 | 107.80 | 785.50 | 797.08 | 1.72 / 3.29 | 4.36 → 4.31 |
| 12 | Gemma 4 | int8 / 131072 | 82.10 | 2289.42 | 2528.67 | 1.03 / 1.77 | 4.63 → 4.44 |
| 12 | LFM2.5 | fp8-mma / 131072 | 121.19 | 1984.02 | 1985.24 | N/A | 8.71 → 8.50 |
| 12 | Clover | int8 / 131072 | 56.64 | 1656.02 | 1631.00 | 1.45 / 2.88 | 3.18 → 3.04 |
| 12 | Tiel Coder | fp8-mma / 131072 | 84.49 | 802.81 | 814.09 | 1.80 / 3.02 | 4.36 → 4.36 |
| 8 | Qwen3.6 | fp8-mma / 65536 | 72.81 | 613.05 | 320.82 | 2.55 / 4.73 | 0.62 → 0.62 |
| 8 | Gemma 4 | int8 / 65536 | 54.03 | 2058.19 | 2559.96 | 1.22 / 2.23 | 0.66 → 0.50 |
| 8 | LFM2.5 | fp8-mma / 65536 | 76.89 | 1743.28 | 1898.85 | N/A | 4.74 → 4.56 |
| 8 | Clover | int8 / 65536 | 42.56 | 902.14 | 902.34 | 2.02 / 3.81 | 0.00 → 0.00 |
| 8 | Tiel Coder | fp8-mma / 65536 | 60.52 | 632.49 | 648.27 | 2.85 / 5.11 | 0.68 → 0.67 |

Tiel prompt processing retains the existing correctness fallback: its IQ3_XXS down projections are excluded from the faster MMQ route after repeated-input checks found output differences. See the [format guard](../../src/qwen35/q36_engine.cu#L1762) in measured source `811bd9d`. This measurement does not change that route.

Historical README tables retain their original dates and repeated/KV-comparison designs. The old Clover 8 GB INT8 profile had only 3/4 successful starts; the current single successful start is recorded as 1/1.

The JSON summary records actual output counts, sampler, MTP, allocation geometry and per-capacity source/hash metadata. This run measures throughput; it does not include a separate answer-quality evaluation.

Sources: run-24gb/result.json, run-16gb/result.json, run-12gb/result.json, run-08gb/result.json; reservation/lifecycle evidence: progress.json; frozen overrides and provenance: plan.json.


## MTP generation accounting

| Nominal GiB | Qwen accepted / offered | Gemma accepted / offered | Tiel accepted / offered |
|---|---:|---:|---:|
| 24 | 255 / 371 | 256 / 374 | 225 / 465 |
| 16 | 255 / 354 | 265 / 354 | 218 / 443 |
| 12 | 261 / 335 | 254 / 383 | 229 / 416 |
| 8 | 252 / 361 | 267 / 341 | 228 / 427 |

LFM and causal D0 Clover do not use MTP. Counts cover the three generation requests, excluding warm-up, long-input decode and images. Actual returned generation counts are recorded per row in the JSON.

## Native and runtime identities

| Artifact | Bytes / file-version resource | SHA-256 |
|---|---:|---|
| `tinystrata-gemma4.exe` | 28700160 | `6adb5c3fe36e4b19f7eace3002d7b369de62e0b3cd9a9a095a6138e4a49858b5` |
| `tinystrata-lfm2.exe` | 28624384 | `54b6ab5539c84e18af00d11c269b5b0d032d15dc5adebf6614adfdc96c328768` |
| `tinystrata.exe` | 56561664 | `a8e9e0f38aac33917a078a6981bc21eed3330beb57f4097a48c3a1b0fcc8e81f` |
| `tinystrata-vision.exe` | 59828736 | `60f68f52feeca1573305a9aa4ffe8be9981c5edb0df44cfe0809c1eb84854d66` |
| `strata_kq.dll` | 13072384 | `a431e69f669b143fa93c0e484c0a5944cb664e80e38f320fc273238cc29c3c1a` |
| `cudart64_13.dll` | 6,14,11,13020 | `7c50bf7f7398b5d1b654af19ff6a679988cf251cab677dbd8abac69fa1cce310` |
| `cublas64_13.dll` | 6,14,11,1330 | `b181c4ffb454840a49f802169d8826617841875607a2e2ba14ef9650e348c456` |
| `cublasLt64_13.dll` | 6,14,11,1330 | `182baf9b1846630c2462cccdae744367b8ff4d2965651deb41a2beefb53f1de5` |

These are verified existing compiled artifacts whose source guards match the measured commit. This run did not perform a new full installer/toolkit build. The detailed [JSON result](rtx-pro-4000-blackwell-sff-20261112.json) includes exact allocation amounts, KV mapping, cache sizes, sampler, MTP counts, raw-result hashes and compiled-receipt hashes. Raw private logs and model files stay outside Git.

## Reproduction scope

Use the fixed `benchmark/run.py` workloads with `--saved-profile --runs 1 --model all` and a private configuration catalog for each capacity. Set the documented KV/context and automatic chunk overrides before starting; retain the CUDA allocation for the entire capacity pass and verify it with NVML before/after retouch. The public runner alone does not create the reservation. The 24 GB pass has no reservation.
