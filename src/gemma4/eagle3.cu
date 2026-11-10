// Sparse vocabulary expansion for an independently sampled EAGLE3 proposal.
#include "strata/gemma4/eagle3.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace strata::gemma4 {
namespace {

__global__ void scatter_probabilities(const double* __restrict__ q,
                                      const int32_t* __restrict__ d2t,
                                      double* __restrict__ out, int count) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < count) out[d2t[i]] = q[i];
}

void check(cudaError_t error, const char* what) {
    if (error == cudaSuccess) return;
    cudaGetLastError();
    throw std::runtime_error(std::string("EAGLE3: ") + what + ": " + cudaGetErrorString(error));
}

}  // namespace

void eagle3_scatter_probabilities(const double* q, const int32_t* d2t, double* out,
                                  int draft_vocab, int target_vocab, void* stream) {
    if (!q || !d2t || !out || draft_vocab <= 0 || target_vocab < draft_vocab ||
        static_cast<const void*>(q) == static_cast<const void*>(out) ||
        static_cast<const void*>(d2t) == static_cast<const void*>(out))
        throw std::invalid_argument("EAGLE3: invalid probability vocabulary expansion");
    const auto st = static_cast<cudaStream_t>(stream);
    check(cudaMemsetAsync(out, 0, size_t(target_vocab) * sizeof(double), st), "clear target probabilities");
    scatter_probabilities<<<(draft_vocab + 255) / 256, 256, 0, st>>>(q, d2t, out, draft_vocab);
    check(cudaGetLastError(), "expand draft probabilities");
}

}  // namespace strata::gemma4
