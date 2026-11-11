# Blackwell CUDA build and Windows charset — 2026-11-11

The reported installer failed while compiling Gemma/GPT-OSS's GGML MXFP4 MMQ instance. Its command requested generic `compute_120` / `sm_120`, but the generated `mma.kind::mxf4.block_scale` instruction requires an architecture-specific target. The repeated PTX messages came from one failed translation unit. The preceding C4819 messages were a separate Windows source-encoding warning.

## Fix

Only the MXFP4 template instance is now a separate OBJECT target, included in the existing Gemma MMQ archive. It maps numeric 12x and family-specific 12xf requests to the corresponding 12xa target, retaining real/virtual suffixes. Explicit 12xa targets are idempotent; 86/89 remain present in portable builds. Ordinary Gemma MMQ and the global architecture selection retain their requested targets. The HIP source list, optional MXFP4 fallback and definitions are unchanged.

This mirrors the architecture handling that pinned GGML already applies inside its own CUDA directory; TinyStrata's custom MXFP4 target had bypassed that directory's CMake handling. NVIDIA documents the [MXF4 target restriction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma). No model format, sampling setting, kernel arithmetic or approximate-acceptance policy was changed.

Before creating local targets or GGML dependencies, Windows builds now pass `/utf-8` to MSVC C/C++ and `-Xcompiler=/utf-8` to NVIDIA CUDA with an MSVC host. This sets source and execution encodings according to [Microsoft's documentation](https://learn.microsoft.com/en-us/cpp/build/reference/utf-8-set-source-and-executable-character-sets-to-utf-8). HIP Clang and non-Windows builds receive neither MSVC option. C4819 is not disabled, and other compiler warnings remain visible.

## Validation

The exact MXFP4 source and CMake-generated compiler options reproduced the original PTX failure with generic SM120 on NVCC 13.2.86. Replacing only architecture code generation with 86/89/120a compiled successfully in the same environment. The preserved user log uses a different toolkit path whose failed managed directory is no longer present; its patch version cannot be inferred from the directory name. The existing installer and CMake rejection of affected NVCC 13.2 versions remains enabled.

Actual full CUDA compilation/linking and warning counts are recorded in the [companion validation receipt](cuda-mxfp4-build-20261111.json). CPU CMake regressions exercise the real production Gemma target graph across 15 architecture/HIP conditions, including the archive's OBJECT input, portable fallbacks and unresolved symbolic targets. Seven charset checks cover all generated C/C++/CUDA commands and external GGML, with negative HIP-Clang and non-Windows profiles. The installer entrypoint suite passed 107 checks without installing or making network requests.

At validation time the discrete NVIDIA adapter was using Microsoft's basic display driver. Consequently CUDA model execution, output quality and speed are not verified by this build test; those require an active NVIDIA driver. No installed model profile or running model was switched. Raw logs, hashes, full-build outputs and the reported failure are retained under `TinyStrata-work/cuda-mxfp4-target-20261111`.
