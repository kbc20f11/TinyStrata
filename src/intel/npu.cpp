#include "strata/intel/npu.hpp"
#include "strata/intel/qwen_ops.hpp"
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
    explicit Impl(sycl::queue& queue) : q(queue) {
        const auto devices = core.get_available_devices();
        if (std::none_of(devices.begin(), devices.end(), [](const std::string& name) { return name.rfind("NPU", 0) == 0; }))
            throw std::runtime_error("--prefill-npu requires an available Intel NPU; CPU fallback is disabled");
        try {
            weights = sycl::malloc_host<sycl::half>(weight_elements, q);
            input = sycl::malloc_host<sycl::half>(activation_elements, q);
            output = sycl::malloc_host<float>(activation_elements, q);
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
        for (void* buffer : {static_cast<void*>(output), static_cast<void*>(input), static_cast<void*>(weights)}) {
            try { sycl::free(buffer, q); } catch (const std::exception& error) { std::fprintf(stderr, "NPU scratch release: %s\n", error.what()); }
        }
        std::fprintf(stderr, "Intel NPU projection calls=%zu wall_ms=%.3f shapes=%zu\n", calls, wall_ms, shapes);
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
} // namespace strata::intel
