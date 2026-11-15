// Model semantics adapted from src/lfm2/lfm2moe.cpp and kernels.cu.
// Their PolyStrata attribution is retained in LICENSE and NOTICE.
#include "strata/intel/lfm_ops.hpp"
#include <cmath>
#include <stdexcept>

namespace strata::intel {
namespace {
void ordered(const sycl::queue& q) {
    if (!q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("LFM operations require an in-order queue");
}
}
sycl::event lfm_shortconv(sycl::queue& q, const float* bcx, float* state,
                         const float* weights, float* output, int channels, int tokens) {
    ordered(q);
    if (!bcx || !state || !weights || !output || channels <= 0 || tokens <= 0)
        throw std::invalid_argument("Invalid LFM short-convolution shape/buffer");
    // Every output reads the old state; update it only in the following
    // dispatch. Parallel token evaluation avoids a serial T-step channel loop.
    q.parallel_for(sycl::range<1>(size_t(tokens) * channels), [=](sycl::id<1> ix) {
        const size_t at = ix[0];
        const int c = int(at % channels), t = int(at / channels);
        const auto bx = [&](int row) {
            if (row < 0) return state[size_t(c) * 2 + 2 + row];
            const size_t base = size_t(row) * channels * 3;
            return bcx[base + c] * bcx[base + channels * 2 + c];
        };
        const float* w = weights + size_t(c) * 3;
        output[at] = bcx[size_t(t) * channels * 3 + channels + c] *
            (bx(t - 2) * w[0] + bx(t - 1) * w[1] + bx(t) * w[2]);
    });
    return q.parallel_for(sycl::range<1>(size_t(channels)), [=](sycl::id<1> ix) {
        const size_t c = ix[0];
        const auto bx = [&](int row) {
            if (row < 0) return state[c * 2 + 2 + row];
            const size_t base = size_t(row) * channels * 3;
            return bcx[base + c] * bcx[base + channels * 2 + c];
        };
        // Read both before modifying state: a single-token call keeps old s1.
        const float a = bx(tokens - 2), b = bx(tokens - 1);
        state[c * 2] = a;
        state[c * 2 + 1] = b;
    });
}
sycl::event lfm_route(sycl::queue& q, const float* logits, const float* bias,
                     int32_t* ids, float* weights, int tokens, int experts,
                     int picks, bool normalize, float scale) {
    ordered(q);
    if (!logits || !bias || !ids || !weights || tokens <= 0 || experts <= 0 ||
        experts > 64 || picks <= 0 || picks > experts || !std::isfinite(scale))
        throw std::invalid_argument("Invalid LFM sigmoid-routing shape/buffer");
    return q.parallel_for(sycl::range<1>(size_t(tokens)), [=](sycl::id<1> ix) {
        const size_t row = ix[0];
        float probability[64], rank[64];
        for (int e = 0; e < experts; ++e) {
            probability[e] = 1.f / (1.f + sycl::exp(-logits[row * experts + e]));
            rank[e] = probability[e] + bias[e];
        }
        float total = 0.f;
        for (int k = 0; k < picks; ++k) {
            int best = 0;
            for (int e = 1; e < experts; ++e)
                if (rank[e] > rank[best]) best = e;
            rank[best] = -INFINITY;
            ids[row * picks + k] = best;
            weights[row * picks + k] = probability[best];
            total += probability[best];
        }
        const float divisor = normalize ? sycl::fmax(total, 6.103515625e-5f) : 1.f;
        for (int k = 0; k < picks; ++k) {
            float w = weights[row * picks + k] / divisor;
            if (scale != 0.f) w *= scale;
            weights[row * picks + k] = w;
        }
    });
}
}
