#include "strata/intel/prefill_gemm.hpp"
#include "strata/intel/q6_interleave.hpp"
#include <oneapi/mkl/blas.hpp>
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace strata::intel {
struct PrefillGemm::Impl {
    sycl::queue& q;
    sycl::half *weights = nullptr, *input = nullptr;
    float* output = nullptr;
    float* scores = nullptr;
    size_t score_elements;
    static constexpr size_t weight_elements = 8192 * 2048, input_elements = 1024 * 8192, output_elements = 1024 * 2048;
    explicit Impl(sycl::queue& queue, int context, int max_tokens) : q(queue), score_elements(size_t(context) * max_tokens) {
        if (context < 1 || context > 32768 || max_tokens < 1 || max_tokens > 1024) throw std::invalid_argument("Invalid GEMM attention scratch shape");
        try {
            weights = sycl::malloc_device<sycl::half>(weight_elements, q);
            input = sycl::malloc_device<sycl::half>(input_elements, q);
            output = sycl::malloc_device<float>(output_elements, q);
            scores = sycl::malloc_device<float>(score_elements, q);
            if (!weights || !input || !output || !scores) throw std::runtime_error("Intel GEMM scratch allocation failed");
        } catch (...) {
            if (weights) sycl::free(weights, q);
            if (input) sycl::free(input, q);
            if (output) sycl::free(output, q);
            if (scores) sycl::free(scores, q);
            throw;
        }
        std::fprintf(stderr, "Intel prefill_gemm=oneMKL FP16_multiplicands=1 FP32_output=1 bounded_scratch_bytes=%zu\n",
            (weight_elements + input_elements) * sizeof(sycl::half) + (output_elements + score_elements) * sizeof(float));
    }
    ~Impl() {
        try { q.wait_and_throw(); } catch (...) {}
        for (void* p : {static_cast<void*>(weights), static_cast<void*>(input), static_cast<void*>(output), static_cast<void*>(scores)})
            try { sycl::free(p, q); } catch (...) {}
    }
    void validate(int rows, int width, int tokens, bool selected) const {
        if (rows < 1 || width < 1 || tokens < 1 || size_t(rows) * width > weight_elements ||
            size_t(tokens) * width > input_elements || (selected && size_t(tokens) * rows > output_elements))
            throw std::invalid_argument("Intel GEMM exceeds bounded scratch");
    }
    sycl::event gemm(float* destination, int rows, int width, int tokens, const std::vector<sycl::event>& dependencies, bool transposed_storage = false) {
        return oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans,
            transposed_storage ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans,
            tokens, rows, width, 1.f, input, width, weights, transposed_storage ? rows : width, 0.f, destination, rows, oneapi::mkl::blas::compute_mode::standard, dependencies);
    }
};
PrefillGemm::PrefillGemm(sycl::queue& q, int context, int max_tokens) : impl_(std::make_unique<Impl>(q, context, max_tokens)) {}
PrefillGemm::~PrefillGemm() = default;
sycl::event PrefillGemm::dense(int type, const uint8_t* weights, const float* input, float* output, int rows, int width, int tokens) {
    auto& p = *impl_;
    p.validate(rows, width, tokens, false);
    auto w = dequantize_half(p.q, type, weights, p.weights, size_t(rows) * width);
    auto x = convert_half(p.q, input, p.input, size_t(tokens) * width);
    return p.gemm(output, rows, width, tokens, {w,x});
}
sycl::event PrefillGemm::dense_interleaved_q6(const uint8_t* weights, const float* input, float* output, int rows, int width, int tokens) {
    auto& p = *impl_;
    p.validate(rows, width, tokens, false);
    if (rows % 16 || width % 256) throw std::invalid_argument("Interleaved dense GEMM needs complete tiles");
    sycl::half* decoded = p.weights;
    auto w = p.q.parallel_for(sycl::range<1>(size_t(rows) * width), [=](sycl::id<1> id) {
        // Enumerate adjacent output rows for coalesced reads and writes. The
        // FP16 values are unchanged; GEMM consumes their transposed storage.
        const size_t row = id[0] % rows, col = id[0] / rows;
        const size_t block = col / 256, local = col % 256;
        const size_t start = (((row / 16) * (width / 256) + block) * 53 * 16 + row % 16) * 4;
        const auto byte = [=](size_t offset) { return weights[start + (offset / 4) * 64 + offset % 4]; };
        const size_t segment = local / 128, part = (local % 128) / 32, lane = local % 32;
        const int low = (byte(segment * 64 + (part % 2) * 32 + lane) >> (part >= 2 ? 4 : 0)) & 15;
        const int high = (byte(128 + segment * 32 + lane) >> (2 * part)) & 3;
        const int scale = int8_t(byte(192 + segment * 8 + part * 2 + lane / 16));
        const uint16_t bits = uint16_t(byte(208)) | (uint16_t(byte(209)) << 8);
        const float delta = float(sycl::bit_cast<sycl::half>(bits));
        decoded[col * rows + row] = sycl::half(delta * float(scale) * float((low | (high << 4)) - 32));
    });
    auto x = convert_half(p.q, input, p.input, size_t(tokens) * width);
    return p.gemm(output, rows, width, tokens, {w,x}, true);
}
sycl::event PrefillGemm::selected(int type, const uint8_t* weights, size_t expert_bytes, ExpertPlan plan,
    const float* input, float* output, int rows, int width, bool separate_inputs, bool interleaved) {
    auto& p = *impl_;
    p.validate(rows, width, plan.tokens, true);
    const int block_bytes = type == 14 ? 210 : type == 18 ? 98 : type == 21 ? 110 : type == 22 ? 82 : 0;
    const size_t original_bytes = block_bytes ? size_t(rows) * (width / 256) * block_bytes : 0;
    if (interleaved && (!block_bytes || rows % 16 || width % 256 ||
            original_bytes > Impl::output_elements * sizeof(float) ||
            expert_bytes != quant_interleaved_bytes(rows, width, block_bytes)))
        throw std::invalid_argument("Invalid interleaved expert GEMM layout/scratch");
    // Read only the small device-generated counts, once per projection. The
    // token/expert lists and all gathered/scattered activations stay on GPU.
    p.q.wait_and_throw();
    sycl::event last;
    for (int expert = 0; expert < plan.experts; ++expert) {
        const int count = plan.counts[expert];
        if (!count) continue;
        if (count < 0 || count > plan.tokens) throw std::runtime_error("Invalid expert token count");
        const uint8_t* source = weights + size_t(expert) * expert_bytes;
        if (interleaved) {
            // The output scratch is idle until GEMM. Restore one compressed
            // expert here, dequantize it, then reuse that same memory for the
            // result. No second persistent expert/model weight store.
            auto restored = reinterpret_cast<uint8_t*>(p.output);
            p.q.parallel_for(sycl::range<1>(original_bytes), [=](sycl::id<1> id) {
                restored[id[0]] = source[quant_interleaved_offset(id[0], width, block_bytes)];
            });
            source = restored;
        }
        auto w = dequantize_half(p.q, type, source, p.weights, size_t(rows) * width);
        sycl::half* gathered = p.input;
        const int32_t* slots = plan.slots + size_t(expert) * plan.tokens;
        const int picks = plan.picks;
        auto x = p.q.parallel_for(sycl::range<1>(size_t(count) * width), [=](sycl::id<1> i) {
            const size_t slot = i[0] / width, col = i[0] % width;
            const size_t source = separate_inputs ? size_t(slots[slot]) : size_t(slots[slot]) / picks;
            gathered[i] = sycl::half(input[source * width + col]);
        });
        const auto dot = p.gemm(p.output, rows, width, count, {w,x});
        const float* result = p.output;
        last = p.q.submit([&](sycl::handler& h) {
            h.depends_on(dot);
            h.parallel_for(sycl::range<1>(size_t(count) * rows), [=](sycl::id<1> i) {
                const size_t slot = i[0] / rows, row = i[0] % rows;
                output[size_t(slots[slot]) * rows + row] = result[i];
            });
        });
    }
    return last;
}
sycl::event PrefillGemm::attention_small(const float* queries, const float* gates, const float* keys, const float* values,
    float* output, int heads, int kv_heads, int context, int position, int tokens) {
    auto& p = *impl_;
    if (heads < 1 || kv_heads < 1 || heads % kv_heads || tokens < 1 || tokens > 8 || position < 0 || position + tokens > context)
        throw std::invalid_argument("Invalid grouped GEMM attention geometry");
    const int group = heads / kv_heads, rows = tokens * group;
    // The caller zeroes the padded KV tail. Keep GEMM shapes stable while the
    // causal prefix grows; future scores are masked before the value product.
    const int visible = std::min(context, (position + tokens + 255) / 256 * 256);
    if (size_t(rows) * visible > p.score_elements || size_t(rows) * 256 > p.output_elements)
        throw std::invalid_argument("Grouped GEMM attention exceeds bounded scratch");
    float* scratch = p.output;
    float* scores = p.scores;
    sycl::event last;
    for (int kv = 0; kv < kv_heads; ++kv) {
        p.q.parallel_for(sycl::range<1>(size_t(rows) * 256), [=](sycl::id<1> id) {
            const size_t row = id[0] / 256, column = id[0] % 256;
            scratch[id[0]] = queries[((row / group) * heads + kv * group + row % group) * 256 + column];
        });
        auto qk = oneapi::mkl::blas::row_major::gemm(p.q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
            rows, visible, 256, 1.f / 16.f, scratch, 256, keys + size_t(kv) * context * 256, 256,
            0.f, scores, visible, oneapi::mkl::blas::compute_mode::standard);
        p.q.submit([&](sycl::handler& handler) {
            handler.depends_on(qk);
            handler.parallel_for(sycl::nd_range<1>(size_t(rows) * 128, 128), [=](sycl::nd_item<1> it) {
                const size_t row = it.get_group_linear_id(), lane = it.get_local_linear_id(), end = position + row / group;
                float maximum = -INFINITY;
                for (size_t cell = lane; cell <= end; cell += 128) maximum = sycl::fmax(maximum, scores[row * visible + cell]);
                maximum = sycl::reduce_over_group(it.get_group(), maximum, sycl::maximum<float>());
                float sum = 0;
                for (size_t cell = lane; cell <= end; cell += 128) sum += sycl::exp(scores[row * visible + cell] - maximum);
                sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
                for (size_t cell = lane; cell < size_t(visible); cell += 128)
                    scores[row * visible + cell] = cell <= end ? sycl::exp(scores[row * visible + cell] - maximum) / sum : 0.f;
            });
        });
        auto pv = oneapi::mkl::blas::row_major::gemm(p.q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
            rows, 256, visible, 1.f, scores, visible, values + size_t(kv) * context * 256, 256,
            0.f, scratch, 256, oneapi::mkl::blas::compute_mode::standard);
        last = p.q.submit([&](sycl::handler& handler) {
            handler.depends_on(pv);
            handler.parallel_for(sycl::range<1>(size_t(rows) * 256), [=](sycl::id<1> id) {
                const size_t row = id[0] / 256, column = id[0] % 256;
                const size_t index = ((row / group) * heads + kv * group + row % group) * 256 + column;
                output[index] = scratch[id[0]] / (1.f + sycl::exp(-gates[index]));
            });
        });
    }
    return last;
}
sycl::event PrefillGemm::attention(const float* queries, const float* gates, const float* keys, const float* values,
    float* output, int heads, int kv_heads, int context, int position, int tokens) {
    auto& p = *impl_;
    const int visible = position + tokens;
    if (heads < 1 || kv_heads < 1 || heads % kv_heads || position < 0 || tokens < 1 || visible > context || size_t(tokens) * visible > p.score_elements)
        throw std::invalid_argument("Invalid GEMM causal attention shape");
    float* scores = p.scores;
    sycl::event last;
    for (int head = 0; head < heads; ++head) {
        const int kv = head / (heads / kv_heads);
        auto qk = oneapi::mkl::blas::row_major::gemm(p.q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
            tokens, visible, 256, 1.f / 16.f, queries + head * 256, heads * 256,
            keys + size_t(kv) * context * 256, 256, 0.f, scores, visible, oneapi::mkl::blas::compute_mode::standard);
        auto softmax = p.q.submit([&](sycl::handler& h) {
            h.depends_on(qk);
            h.parallel_for(sycl::nd_range<1>(size_t(tokens) * 128, 128), [=](sycl::nd_item<1> it) {
                const size_t token = it.get_group_linear_id(), lane = it.get_local_linear_id(), end = position + token;
                float maximum = -INFINITY;
                for (size_t cell = lane; cell <= end; cell += 128) maximum = sycl::fmax(maximum, scores[token * visible + cell]);
                maximum = sycl::reduce_over_group(it.get_group(), maximum, sycl::maximum<float>());
                float sum = 0;
                for (size_t cell = lane; cell <= end; cell += 128) sum += sycl::exp(scores[token * visible + cell] - maximum);
                sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
                for (size_t cell = lane; cell < size_t(visible); cell += 128)
                    scores[token * visible + cell] = cell <= end ? sycl::exp(scores[token * visible + cell] - maximum) / sum : 0.f;
            });
        });
        auto pv = oneapi::mkl::blas::row_major::gemm(p.q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
            tokens, 256, visible, 1.f, scores, visible, values + size_t(kv) * context * 256, 256,
            0.f, output + head * 256, heads * 256, oneapi::mkl::blas::compute_mode::standard, {softmax});
        last = p.q.submit([&](sycl::handler& h) {
            h.depends_on(pv);
            h.parallel_for(sycl::range<1>(size_t(tokens) * 256), [=](sycl::id<1> id) {
                const size_t index = (id[0] / 256 * heads + head) * 256 + id[0] % 256;
                output[index] /= 1.f + sycl::exp(-gates[index]);
            });
        });
    }
    return last;
}
} // namespace strata::intel
