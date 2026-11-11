// Instantiate GGML's MXFP4-weight / Q8-activation MMQ path on CUDA too.
// TinyStrata's Mmq::quantize writes block_q8_1_mmq; Blackwell's native FP4
// activation kernel instead expects block_fp4_mmq. Their byte sizes agree but
// their scales, codes and elements-per-block do not. Never interpret Q8 as FP4.
#include "common.cuh"

// Keep this policy local to this one translation unit and one instantiated
// weight type. common.cuh has already defined its internal-linkage helpers;
// its include guard prevents the function-like macro from rewriting them.
// Select the matching host AND device Q8 configuration in mmq.cuh. Other
// Gemma and upstream GGML kernels retain their ordinary feature selection.
#undef BLACKWELL_MMA_AVAILABLE
#define blackwell_mma_available(cc) false
#include "mmq.cuh"
#undef blackwell_mma_available

static_assert(QK_FP4_MMQ == 2 * QK8_1_MMQ, "Review the pinned GGML activation layouts");
DECL_MMQ_CASE(GGML_TYPE_MXFP4);
