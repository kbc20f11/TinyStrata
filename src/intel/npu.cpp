#include "strata/intel/npu.hpp"
#include "strata/intel/qwen_ops.hpp"
#ifdef TINYSTRATA_INTEL_MKL
#include "strata/intel/prefill_gemm.hpp"
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#endif
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <tuple>

namespace strata::intel {
struct NpuProjection::Impl {
    sycl::queue& q;
    ov::Core core;
    sycl::half *weights = nullptr, *input = nullptr;
    float* output = nullptr;
    static constexpr size_t weight_elements = 8192 * 2048, activation_elements = 1024 * 8192;
    struct Graph { ov::CompiledModel compiled; ov::InferRequest request; };
    std::map<std::tuple<int,int,int>, Graph> graphs;
    size_t calls = 0;
    double wall_ms = 0;
    sycl::half* split_weights = nullptr;
    float* split_gpu_output = nullptr;
    static constexpr size_t split_cache_elements = 64 * 1024 * 1024;
    size_t split_cache_used = 0, split_calls = 0, split_hits = 0;
    double split_compile_ms = 0, split_prepare_ms = 0, split_wait_ms = 0;
    ov::Tensor split_remote_input, split_remote_output, split_remote_weights;
    size_t split_gpu_done_npu_pending = 0;
    std::map<std::tuple<const uint8_t*, int, int, int>, sycl::half*> split_cache;
    explicit Impl(sycl::queue& queue) : q(queue) {
        const auto devices = core.get_available_devices();
        if (std::none_of(devices.begin(), devices.end(), [](const std::string& name) { return name.rfind("NPU", 0) == 0; }))
            throw std::runtime_error("--prefill-npu requires an available Intel NPU; CPU fallback is disabled");
        try {
            weights = sycl::malloc_host<sycl::half>(weight_elements, q);
            input = sycl::aligned_alloc_host<sycl::half>(4096, activation_elements, q);
            output = sycl::aligned_alloc_host<float>(4096, activation_elements, q);
            if (!weights || !input || !output) throw std::runtime_error("Intel NPU projection scratch allocation failed");
        } catch (...) {
            if (weights) sycl::free(weights, q);
            if (input) sycl::free(input, q);
            if (output) sycl::free(output, q);
            throw;
        }
        std::fprintf(stderr, "Intel prefill_npu=enabled device=%s scratch_bytes=%zu runtime_weights=1 FP16_multiplicands=1\n",
            core.get_property("NPU", ov::device::full_name).c_str(), weight_elements * sizeof(sycl::half) + activation_elements * (sizeof(sycl::half) + sizeof(float)));
    }
    ~Impl() {
        try { q.wait_and_throw(); } catch (...) {}
        const size_t shapes = graphs.size();
        graphs.clear();
        split_remote_input = {}; split_remote_output = {}; split_remote_weights = {};
        if (split_weights) sycl::free(split_weights, q);
        if (split_gpu_output) sycl::free(split_gpu_output, q);
        for (void* buffer : {static_cast<void*>(output), static_cast<void*>(input), static_cast<void*>(weights)}) {
            try { sycl::free(buffer, q); } catch (const std::exception& error) { std::fprintf(stderr, "NPU scratch release: %s\n", error.what()); }
        }
        std::fprintf(stderr, "Intel NPU projection calls=%zu wall_ms=%.3f shapes=%zu\n", calls, wall_ms, shapes);
        if (split_calls) std::fprintf(stderr, "Intel NPU split calls=%zu cache_hits=%zu cache_bytes=%zu compile_ms=%.3f prepare_ms=%.3f wait_ms=%.3f gpu_done_npu_pending=%zu\n",
            split_calls, split_hits, split_cache_used * sizeof(sycl::half), split_compile_ms, split_prepare_ms, split_wait_ms, split_gpu_done_npu_pending);
    }
    Graph& graph(int rows, int width, int tokens) {
        const auto key = std::make_tuple(rows, width, tokens);
        if (auto found = graphs.find(key); found != graphs.end()) return found->second;
        const ov::Shape input_shape{size_t(tokens), size_t(width)}, weight_shape{size_t(rows), size_t(width)}, output_shape{size_t(tokens), size_t(rows)};
        auto x = std::make_shared<ov::opset13::Parameter>(ov::element::f16, input_shape);
        auto w = std::make_shared<ov::opset13::Parameter>(ov::element::f16, weight_shape);
        auto dot = std::make_shared<ov::opset13::MatMul>(x, w, false, true);
        auto result = std::make_shared<ov::opset13::Convert>(dot, ov::element::f32);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{x,w}, "TinyStrata runtime-weight projection");
        Graph entry;
        entry.compiled = core.compile_model(model, "NPU", {{"PERFORMANCE_HINT", "LATENCY"}});
        entry.request = entry.compiled.create_infer_request();
        entry.request.set_input_tensor(0, ov::Tensor(ov::element::f16, input_shape, input));
        entry.request.set_input_tensor(1, ov::Tensor(ov::element::f16, weight_shape, weights));
        entry.request.set_output_tensor(ov::Tensor(ov::element::f32, output_shape, output));
        return graphs.emplace(key, std::move(entry)).first->second;
    }
};
NpuProjection::NpuProjection(sycl::queue& q) : impl_(std::make_unique<Impl>(q)) {}
NpuProjection::~NpuProjection() = default;
sycl::event NpuProjection::run(int type, const uint8_t* weights, const float* input, float* output, int rows, int width, int tokens) {
    auto& p = *impl_;
    if (rows < 1 || width < 1 || tokens < 1 || size_t(rows) * width > p.weight_elements ||
        size_t(tokens) * width > p.activation_elements || size_t(tokens) * rows > p.activation_elements)
        throw std::invalid_argument("Intel NPU projection exceeds bounded scratch");
    const auto start = std::chrono::steady_clock::now();
    auto& graph = p.graph(rows, width, tokens);
    dequantize_half(p.q, type, weights, p.weights, size_t(rows) * width);
    convert_half(p.q, input, p.input, size_t(tokens) * width);
    p.q.wait_and_throw();
    graph.request.infer();
    const auto event = p.q.memcpy(output, p.output, size_t(tokens) * rows * sizeof(float));
    ++p.calls;
    p.wall_ms += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start).count();
    return event;
}
#ifdef TINYSTRATA_INTEL_MKL
bool NpuProjection::run_split(PrefillGemm& gpu, int type, const uint8_t* weights, const float* input,
    float* output, int rows, int width, int tokens, int npu_rows, bool parallel, bool remote, sycl::event& completion, bool interleaved) {
    auto& p = *impl_;
    // The experiment is bounded to original Q6_K projection rows. Other
    // types/layouts keep the existing GPU path rather than guessing strides.
    if (type != 14 || width != 2048 || tokens < 128 || tokens > 1024 ||
        npu_rows < 256 || npu_rows % 256 || npu_rows >= rows ||
        size_t(rows) * tokens > p.activation_elements) return false;
    if (!p.split_weights) {
        auto* weights_buffer = sycl::aligned_alloc_host<sycl::half>(4096, p.split_cache_elements, p.q);
        float* output_buffer = nullptr;
        try {
            output_buffer = sycl::malloc_device<float>(p.activation_elements, p.q);
            if (!weights_buffer || !output_buffer) throw std::runtime_error("NPU split scratch allocation failed");
        } catch (...) {
            if (weights_buffer) sycl::free(weights_buffer, p.q);
            if (output_buffer) sycl::free(output_buffer, p.q);
            throw;
        }
        p.split_weights = weights_buffer;
        p.split_gpu_output = output_buffer;
    }
    const auto key = std::make_tuple(weights, type + (interleaved ? 1000 : 0), npu_rows, width);
    const size_t elements = size_t(npu_rows) * width;
    auto found = p.split_cache.find(key);
    if (found == p.split_cache.end() && p.split_cache_used + elements > p.split_cache_elements) return false;
    auto started = std::chrono::steady_clock::now();
    auto& graph = p.graph(npu_rows, width, tokens);
    p.split_compile_ms += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - started).count();
    started = std::chrono::steady_clock::now();
    sycl::half* prepared = nullptr;
    if (found == p.split_cache.end()) {
        prepared = p.split_weights + p.split_cache_used;
        if (!interleaved) dequantize_half(p.q, type, weights, prepared, elements);
        else p.q.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> id) {
            const size_t row = id[0] / width, col = id[0] % width;
            const size_t block = col / 256, local = col % 256;
            const size_t start = (((row / 16) * (width / 256) + block) * 53 * 16 + row % 16) * 4;
            const auto byte = [=](size_t at) { return weights[start + (at / 4) * 64 + at % 4]; };
            const size_t segment = local / 128, part = (local % 128) / 32, lane = local % 32;
            const int low = (byte(segment * 64 + (part % 2) * 32 + lane) >> (part >= 2 ? 4 : 0)) & 15;
            const int high = (byte(128 + segment * 32 + lane) >> (2 * part)) & 3;
            const int scale = int8_t(byte(192 + segment * 8 + part * 2 + lane / 16));
            const uint16_t bits = uint16_t(byte(208)) | (uint16_t(byte(209)) << 8);
            prepared[id[0]] = sycl::half(float(sycl::bit_cast<sycl::half>(bits)) * float(scale) * float((low | high << 4) - 32));
        });
    } else { prepared = found->second; ++p.split_hits; }
    convert_half(p.q, input, p.input, size_t(tokens) * width);
    p.q.wait_and_throw();
    if (found == p.split_cache.end()) {
        p.split_cache.emplace(key, prepared);
        p.split_cache_used += elements;
    }
    const ov::Shape xshape{size_t(tokens), size_t(width)}, wshape{size_t(npu_rows), size_t(width)}, yshape{size_t(tokens), size_t(npu_rows)};
    if (remote) {
        auto context = graph.compiled.get_context().as<ov::intel_npu::level_zero::ZeroContext>();
        const auto kind = ov::intel_npu::MemType::CPU_VA;
        // Import each COMPLETE allocation once. Keeping a small first-shape
        // import alive otherwise prevents a later larger graph from wrapping
        // the same address range. All requests are destroyed before the owners.
        if (!p.split_remote_input) {
            auto input_region = context.create_tensor(ov::element::f16, {p.activation_elements}, p.input, kind);
            auto output_region = context.create_tensor(ov::element::f32, {p.activation_elements}, p.output, kind);
            auto weights_region = context.create_tensor(ov::element::f16, {p.split_cache_elements}, p.split_weights, kind);
            p.split_remote_input = std::move(input_region);
            p.split_remote_output = std::move(output_region);
            p.split_remote_weights = std::move(weights_region);
        }
        graph.request.set_input_tensor(0, context.create_tensor(ov::element::f16, xshape, p.input, kind));
        graph.request.set_input_tensor(1, context.create_tensor(ov::element::f16, wshape, prepared, kind));
        graph.request.set_output_tensor(context.create_tensor(ov::element::f32, yshape, p.output, kind));
    } else {
        graph.request.set_input_tensor(0, ov::Tensor(ov::element::f16, xshape, p.input));
        graph.request.set_input_tensor(1, ov::Tensor(ov::element::f16, wshape, prepared));
        graph.request.set_output_tensor(ov::Tensor(ov::element::f32, yshape, p.output));
    }
    p.split_prepare_ms += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - started).count();
    // Independent buffers: GPU computes every remaining row while NPU is in
    // flight. The request is joined only after useful GPU work is submitted.
    const int gpu_rows = rows - npu_rows;
    const size_t stride = size_t(width / 256) * (interleaved ? 212 : 210);
    try {
        if (parallel) graph.request.start_async();
        else graph.request.infer();
        if (interleaved) gpu.dense_interleaved_q6(weights + size_t(npu_rows) * stride, input, p.split_gpu_output, gpu_rows, width, tokens);
        else gpu.dense(type, weights + size_t(npu_rows) * stride, input, p.split_gpu_output, gpu_rows, width, tokens);
        const float* gpu_result = p.split_gpu_output;
        auto gpu_end = p.q.parallel_for(sycl::range<1>(size_t(tokens) * gpu_rows), [=](sycl::id<1> id) {
            const size_t token = id[0] / gpu_rows, row = id[0] % gpu_rows;
            output[token * rows + npu_rows + row] = gpu_result[id[0]];
        });
        // Numerical micro tests enable profiling and can observe the request
        // still in flight after useful GPU work completes. Normal model runs
        // preserve asynchronous submission and join only at the dependency.
        if (parallel && p.q.has_property<sycl::property::queue::enable_profiling>()) {
            gpu_end.wait_and_throw();
            if (!graph.request.wait_for(std::chrono::milliseconds(0))) ++p.split_gpu_done_npu_pending;
        }
        started = std::chrono::steady_clock::now();
        if (parallel) graph.request.wait();
    } catch (...) {
        if (parallel) { try { graph.request.wait(); } catch (...) {} }
        throw;
    }
    p.split_wait_ms += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - started).count();
    const float* npu_result = p.output;
    completion = p.q.parallel_for(sycl::range<1>(size_t(tokens) * npu_rows), [=](sycl::id<1> id) {
        const size_t token = id[0] / npu_rows, row = id[0] % npu_rows;
        output[token * rows + row] = npu_result[id[0]];
    });
    ++p.split_calls;
    return true;
}
#endif
} // namespace strata::intel
