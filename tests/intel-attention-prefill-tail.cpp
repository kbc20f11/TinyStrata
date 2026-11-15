// Model-free matched screen of the existing causal vs oneMKL FP32 attention.
#include "strata/intel/device.hpp"
#include "strata/intel/prefill_gemm.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>
using namespace strata::intel;
namespace {
constexpr int max_context = 32768, max_tokens = 1024, heads = 16, kv_heads = 2, width = 256;
constexpr size_t query_capacity = size_t(max_tokens) * heads * width;
constexpr size_t cache_capacity = size_t(max_context) * kv_heads * width;
constexpr uint32_t canary = 0x7fc12345u;
struct Memory {
    sycl::queue& queue; std::vector<void*> pointers;
    template<class T> T* device(size_t count) {
        T* value = sycl::malloc_device<T>(count, queue);
        if (!value) throw std::bad_alloc();
        try { pointers.push_back(value); } catch (...) { sycl::free(value, queue); throw; }
        return value;
    }
    ~Memory() { try { queue.wait_and_throw(); } catch (...) {} for (void* value : pointers) sycl::free(value, queue); }
};
float query_value(size_t index) { return float(int((index * 13 + 11) % 257) - 128) / 1024.f; }
float key_value(size_t index) { return float(int((index * 17 + 3) % 255) - 127) / 128.f; }
float value_value(size_t index) { return float(int((index * 29 + 11) % 255) - 127) / 64.f; }
float gate_value(size_t index) { return float(int(index % 17) - 8) / 4.f; }
struct Timing { double wall_ms, gpu_span_ms, returned_event_ms; };
template<class Launch> Timing measure(sycl::queue& queue, uint32_t* marker, Launch launch) {
    const auto start = std::chrono::steady_clock::now();
    auto begin = queue.single_task([=] { marker[0] = 1; });
    auto end = launch();
    end.wait_and_throw(); queue.wait_and_throw();
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const auto device_begin = begin.template get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto device_end = end.template get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto returned_start = end.template get_profiling_info<sycl::info::event_profiling::command_start>();
    return {wall, double(device_end - device_begin) * 1e-6, double(device_end - returned_start) * 1e-6};
}
double reference(int context, int position, int token, int head, int column) {
    const int visible = position + token + 1, kv = head / (heads / kv_heads);
    std::vector<double> probabilities(size_t(visible), 0.);
    double maximum = -INFINITY;
    for (int cell = 0; cell < visible; ++cell) {
        double dot = 0;
        for (int col = 0; col < width; ++col)
            dot += double(query_value((size_t(token) * heads + head) * width + col)) *
                   double(key_value((size_t(kv) * context + cell) * width + col));
        probabilities[size_t(cell)] = dot / 16.; maximum = std::max(maximum, dot / 16.);
    }
    double total = 0;
    for (double& value : probabilities) total += value = std::exp(value - maximum);
    double result = 0;
    for (int cell = 0; cell < visible; ++cell)
        result += probabilities[size_t(cell)] * double(value_value((size_t(kv) * context + cell) * width + column));
    const size_t index = (size_t(token) * heads + head) * width + column;
    return result / total / (1. + std::exp(-double(gate_value(index))));
}
struct Case { int context, position, tokens; };
}
int main(int argc, char** argv) {
    try {
        const bool near_full_short = argc == 2 && std::string_view(argv[1]) == "--near-full-short";
        if (argc != 1 && !near_full_short) throw std::invalid_argument("Expected no arguments or --near-full-short");
        const auto device = select_device(0);
        sycl::queue queue(device, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
        // About400 MiB including the unchanged oneMKL C1024 scratch capacity.
        constexpr uint64_t reserve = 512ull * 1024 * 1024;
        if (reserve > allocation_budget(memory_budget(inspect_device(device)))) throw std::runtime_error("Insufficient bounded micro budget");
        Memory memory{queue};
        float* queries = memory.device<float>(query_capacity);
        float* gates = memory.device<float>(query_capacity);
        float* keys = memory.device<float>(cache_capacity);
        float* values = memory.device<float>(cache_capacity);
        float* old_output = memory.device<float>(query_capacity + 64);
        float* mkl_output = memory.device<float>(query_capacity + 64);
        uint32_t* stats = memory.device<uint32_t>(3);
        uint32_t* marker = memory.device<uint32_t>(1);
        float* samples = memory.device<float>(48);
        PrefillGemm gemm(queue, max_context, max_tokens);
        queue.parallel_for(sycl::range<1>(query_capacity), [=](sycl::id<1> id) {
            const size_t index = id[0];
            queries[index] = float(int((index * 13 + 11) % 257) - 128) / 1024.f;
            gates[index] = float(int(index % 17) - 8) / 4.f;
        });
        queue.parallel_for(sycl::range<1>(cache_capacity), [=](sycl::id<1> id) {
            const size_t index = id[0];
            keys[index] = float(int((index * 17 + 3) % 255) - 127) / 128.f;
            values[index] = float(int((index * 29 + 11) % 255) - 127) / 64.f;
        }).wait_and_throw();
        const std::vector<Case> cases = near_full_short
            ? std::vector<Case>{{8192,0,1008}, {8192,0,1019}, {8192,7168,1008}, {8192,7168,1019}}
            : std::vector<Case>{{67,0,7}, {1031,257,33}, {32768,15360,1000},
                                 {32768,15360,1008}, {32768,15360,1019}, {32768,15360,1024}};
        int passed_cases = 0;
        for (const auto config : cases) {
            const int context = config.context, position = config.position, tokens = config.tokens;
            const size_t active = size_t(tokens) * heads * width;
            const auto old = [&] { return causal_attention(queue, queries, gates, keys, values, old_output,
                                                          heads, kv_heads, context, position, tokens); };
            const auto mkl = [&] { return gemm.attention(queries, gates, keys, values, mkl_output,
                                                       heads, kv_heads, context, position, tokens); };
            // Shape warm-up is excluded; reset outputs afterward so every tail
            // check observes only this shape's measured calls.
            old().wait_and_throw(); mkl().wait_and_throw();
            queue.parallel_for(sycl::range<1>(query_capacity + 64), [=](sycl::id<1> id) {
                old_output[id[0]] = mkl_output[id[0]] = sycl::bit_cast<float>(canary);
            }).wait_and_throw();
            for (int round = 0; round < 2; ++round) {
                Timing a{}, b{};
                if (round == 0) { a = measure(queue, marker, old); b = measure(queue, marker, mkl); }
                else { b = measure(queue, marker, mkl); a = measure(queue, marker, old); }
                std::printf("{\"context\":%d,\"position\":%d,\"tokens\":%d,\"round\":%d,\"old_wall_ms\":%.9f,\"old_gpu_span_ms\":%.9f,\"old_kernel_ms\":%.9f,\"mkl_wall_ms\":%.9f,\"mkl_gpu_span_ms\":%.9f,\"mkl_returned_gate_event_ms\":%.9f}\n",
                    context, position, tokens, round, a.wall_ms, a.gpu_span_ms, a.returned_event_ms,
                    b.wall_ms, b.gpu_span_ms, b.returned_event_ms);
                std::fflush(stdout);
            }
            queue.memset(stats, 0, 3 * sizeof(uint32_t));
            queue.parallel_for(sycl::range<1>(query_capacity + 64), [=](sycl::id<1> id) {
                const size_t index = id[0];
                using Atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                               sycl::access::address_space::global_space>;
                if (index < active) {
                    const float a = old_output[index], b = mkl_output[index], difference = sycl::fabs(a - b);
                    if (!sycl::isfinite(a) || !sycl::isfinite(b) || difference > 2e-5f + 2e-4f * sycl::fabs(a)) Atomic(stats[0]).fetch_add(1);
                    if (sycl::isfinite(difference)) {
                        const uint32_t bits = sycl::bit_cast<uint32_t>(difference);
                        Atomic maximum(stats[2]);
                        if (bits > maximum.load()) maximum.fetch_max(bits);
                    }
                } else if (sycl::bit_cast<uint32_t>(old_output[index]) != canary || sycl::bit_cast<uint32_t>(mkl_output[index]) != canary)
                    Atomic(stats[1]).fetch_add(1);
            });
            queue.parallel_for(sycl::range<1>(48), [=](sycl::id<1> id) {
                const size_t sample = id[0] % 24;
                constexpr int selected_heads[4] = {0, 7, 8, 15}, columns[3] = {0, 127, 255};
                const int token = sample < 12 ? 0 : tokens - 1;
                const size_t index = (size_t(token) * heads + selected_heads[(sample % 12) / 3]) * width + columns[sample % 3];
                samples[id[0]] = id[0] < 24 ? old_output[index] : mkl_output[index];
            }).wait_and_throw();
            std::array<uint32_t,3> observed{}; std::array<float,48> observed_samples{};
            queue.memcpy(observed.data(), stats, sizeof(observed));
            queue.memcpy(observed_samples.data(), samples, sizeof(observed_samples)).wait_and_throw();
            if (observed[0] || observed[1]) throw std::runtime_error("Attention complete-output or tail-canary mismatch");
            double worst_reference_error = 0;
            constexpr int selected_heads[4] = {0,7,8,15}, columns[3] = {0,127,255};
            for (int sample = 0; sample < 24; ++sample) {
                const int token = sample < 12 ? 0 : tokens - 1;
                const double expected = reference(context, position, token, selected_heads[(sample % 12) / 3], columns[sample % 3]);
                for (int path = 0; path < 2; ++path) {
                    const double error = std::abs(double(observed_samples[size_t(path * 24 + sample)]) - expected);
                    worst_reference_error = std::max(worst_reference_error, error);
                    if (error > 2e-5 + 2e-4 * std::abs(expected)) throw std::runtime_error("Attention independent FP64 sample mismatch");
                }
            }
            ++passed_cases;
            std::printf("{\"case_passed\":true,\"context\":%d,\"position\":%d,\"tokens\":%d,\"all_active_outputs_compared\":%zu,\"tail_canaries_intact\":true,\"fp64_samples_per_path\":24,\"max_path_difference\":%.9g,\"max_fp64_error\":%.9g}\n",
                context, position, tokens, active, double(sycl::bit_cast<float>(observed[2])), worst_reference_error);
        }
        std::printf("{\"passed\":true,\"cases\":%d,\"mode\":\"%s\",\"scope\":\"isolated existing FP32 attention paths; not full-model prefill gain\",\"mkl_capacity_tokens\":1024,\"all_causal_keys_retained\":true,\"models_loaded\":false,\"gpu_span_note\":\"marker-end through final-completion includes every oneMKL stage; returned gate event alone is not GEMM time\"}\n", passed_cases, near_full_short ? "near-full-short" : "long-and-boundary");
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"Attention tail micro failed: %s\n",error.what()); return 1; }
}
