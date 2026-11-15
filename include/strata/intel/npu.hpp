#pragma once
#include <sycl/sycl.hpp>
#include <memory>

namespace strata::intel {
class PrefillGemm;
// Optional OpenVINO Intel NPU candidate. Owns bounded projection scratch,
// not another persistent model or expert weight store.
// Multiplicands and the internal MatMul result are FP16. The public output
// buffer is F32 for the surrounding GPU operations; qualify model quality.
class NpuProjection {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit NpuProjection(sycl::queue& queue);
    ~NpuProjection();
    sycl::event run(int type, const uint8_t* weights, const float* input, float* output,
                    int rows, int width, int tokens);
#ifdef TINYSTRATA_INTEL_MKL
    bool run_split(PrefillGemm& gpu, int type, const uint8_t* weights, const float* input,
                   float* output, int rows, int width, int tokens, int npu_rows,
                   bool parallel, bool remote, sycl::event& completion, bool interleaved = false);
#endif
};
} // namespace strata::intel
