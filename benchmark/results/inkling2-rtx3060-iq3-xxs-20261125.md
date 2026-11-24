# Inkling 2 Mini IQ3_XXS on RTX 3060 12 GB

The experiment starts from TinyStrata `main` at `f3c29b1`. The standard v2 workload inputs ran on CUDA sm_86 with one active request, FP16 device KV, a 131,072-token native context, a 512-token input chunk, and the native Vision shard present. The native engine reserved 4,096 MiB of global device KV. The screening results use one code prompt and one 2,048-token input per candidate. Decode excludes prefill and measures generated tokens including thinking. The rows are single measurements.

## Weight comparison

| Distribution | Download with Vision (GB) | GPU experts at launch | Code decode (tok/s) | 2K input (tok/s) |
|---|---:|---:|---:|---:|
| Previous Unsloth UD-IQ4_XS | 16.37 | 624 / 3,072 | 34.3 | 384.85 |
| InterpreticAI IQ3_XXS | 12.38 | 534 / 3,072 | 21.3 | 94.70 |
| Unsloth UD-IQ3_XXS | 12.30 | 519 / 3,072 | 32.5 | 372.17 |

InterpreticAI stores IQ3_XXS gate/up and IQ4_NL down matrices. The engine loads these Expert matrices directly. Unsloth mixes IQ2 and IQ3 gate/up tensors with NVFP4 down tensors in 20 of the 24 physical layers. The native engine requantizes those two Expert planes to SF4 on CUDA and S4_K by default on HIP/SYCL. The device kernels and loaded memory footprints differ by format. The CUDA Expert arena is 10.83 GB for InterpreticAI and 13.27 GB for Unsloth UD-IQ3_XXS. On HIP the corresponding estimates are 10.83 and 12.48 GB. The other four layers of Unsloth UD-IQ3_XXS are loaded directly.

## CUDA screening on InterpreticAI IQ3_XXS

| Candidate | Code decode (tok/s) | 2K input (tok/s) | Observation |
|---|---:|---:|---|
| Default | 21.3 | 94.70 | 99.64 GB of staged Expert data in the screen |
| Maya spare KV for Experts | 18.5 | 96.93 | 53.04 GB staged. The input gain came with a decode regression |
| Basalt-inspired KV tile sharing | 21.3 | 94.54 | Comparable to the default |
| Basalt-inspired routed Expert staging | 21.4 | 95.71 | 67.63 GB staged. The 1.1% input difference is within the precision of a single screen |
| Two-row GGUF weight reuse | 21.2 | 51.21 | Input regression from the upstream 16-row group at chunk 512 |
| Strata DSpark | 11.3 | 93.81 | 51 of 226 draft tokens accepted for the code prompt. GPU Expert slots fell from 534 to 154 |
| Basalt packed IQ3 sign expansion | 21.2 | 98.90 | 4.4% higher input throughput. Both captured greedy streams matched the baseline |

The packed sign expansion was kept in the CUDA GGUF kernel. An independent CPU check verified 65,536 codebook and sign combinations against the previous scalar expansion. KV lending, KV tile sharing, routed staging and DSpark stay off by default. [Basalt packed signs](https://github.com/jesdga95/basalt/commit/d9915bbc8dc9126b04e06d58d3746312efddea32)

Strata supplies the native engine. The reference checkouts were Strata `d46f5f2`, MoMoE `c8dd5a9`, PolyStrata `36e654b`, Basalt `bc36673`, and Project Maya `1016876`. [Strata](https://github.com/Niko1221/Strata/commit/d46f5f268f8ed797e83a5c63a69bffbd7a581cf6) · [MoMoE](https://github.com/aflin/momoe/commit/c8dd5a99fe8dc7aba240b4ba34960a3723f1f455) · [PolyStrata](https://github.com/simplexai-labs/PolyStrata/commit/36e654b1b7eb4cb462202445be296ba012c75edf) · [Basalt](https://github.com/jesdga95/basalt/commit/bc36673bd3fea4ef6d753649316d7d025c4f3d86) · [Project Maya](https://github.com/mw00/project-maya/commit/1016876a387e332fc5bf8bb82030fca7dc2821b2)

MoMoE's shared improvements were already included in the tested TinyStrata baseline. PolyStrata's tuned IQ3 kernels use a different activation packing layout. Its GPU Vision path uses a different projector format. Maya's KV lending and Basalt's two prefill candidates were screened independently. [Maya KV lending](https://github.com/mw00/project-maya/commit/381301c5c87d4c5ca026bbd679191e84168e3258) · [Basalt KV tiles](https://github.com/jesdga95/basalt/commit/73c95c10650d428fde0fd23ba10ed9bbd481647f) · [Basalt Expert routing](https://github.com/jesdga95/basalt/commit/b78aadaaecb622396cb4716dbd997c3522ddde79)

## Standard workloads

| Distribution | Decode, three prompts (tok/s) | 4,096-token input (tok/s) | 16,384-token input (tok/s) | Cold image, total (s) |
|---|---:|---:|---:|---:|
| InterpreticAI IQ3_XXS, scalar signs | 20.9 | 94.16 | 93.80 | 37.9 |
| Unsloth UD-IQ3_XXS, packed signs | 32.7 | 373.89 | 362.28 | 30.1 |

The native automatic decode probe on this PC selected CPU computation for cold Experts, which is also the explicit mode used in the full InterpreticAI baseline. The first image's encoder runs on CPU on the Strata CUDA path. Reported image time is end-to-end, including the encoder. HIP built and returned native capabilities on the same machine. Hardware inference was exercised on CUDA sm_86. CUDA sm_120 was compiled. The Intel build remains experimental and the oneAPI compiler was unavailable for this run.

The branch uses the pinned Unsloth UD-IQ3_XXS download. It retains most of the previous IQ4 screening throughput while reducing the combined download from 16.37 GB to 12.30 GB. It also completed the RTX 3060 API workflow checks below. [Pinned GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/800c946d7b9c830c86290df49afbbe91db8455f5)

- A 34,184-token Japanese shipment dossier returned all nine approved facts, with no extra keys, in 102.2 seconds. The fixture included older and similarly numbered records as distractors.
- The model issued the requested shipment lookup with the correct arguments, and the next completion reported the values from the tool result.
- A 132,262-token prompt was rejected with HTTP 400. The next short request succeeded on the same engine.
- A damaged image returned HTTP 400. A follow-up reused the original image's context and read back its label. An equal-sized replacement image used a fresh context and returned its own label.
