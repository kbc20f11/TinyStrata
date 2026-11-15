// PRIVATE model-free screen of the existing full-prefix INT8->FP32 expansion.
// No new attention kernel, truncation or quantization policy is introduced.
#include "strata/intel/device.hpp"
#include "strata/intel/kv_q8.hpp"
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace strata::intel;
namespace {
constexpr int context = 32768, heads = 2, dimension = 256, chunk = 1024, layers = 10;
constexpr uint32_t canary = 0x7fc12345u;
constexpr size_t cells = size_t(context) * heads * dimension;
struct Allocations {
    sycl::queue& queue;
    std::vector<void*> pointers;
    template<class T> T* device(size_t count) {
        T* result = sycl::malloc_device<T>(count, queue);
        if (!result) throw std::bad_alloc();
        try { pointers.push_back(result); } catch (...) { sycl::free(result, queue); throw; }
        return result;
    }
    ~Allocations() { try { queue.wait_and_throw(); } catch (...) {} for (void* value : pointers) sycl::free(value, queue); }
};
}
int main() {
    try {
        const auto device = select_device(0);
        sycl::queue queue(device, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
        Allocations allocations{queue};
        std::vector<Q8Kv> caches;
        const size_t bytes = layers * (cells * 2 + cells / 32 * sizeof(sycl::half) * 2) + cells * sizeof(float) * 2 + sizeof(uint32_t);
        if (bytes > allocation_budget(memory_budget(inspect_device(device))))
            throw std::runtime_error("Insufficient budget for bounded model-free expansion fixture");
        for (int layer = 0; layer < layers; ++layer) {
            Q8Kv cache{allocations.device<int8_t>(cells), allocations.device<int8_t>(cells),
                       allocations.device<sycl::half>(cells / 32), allocations.device<sycl::half>(cells / 32), context, heads};
            caches.push_back(cache);
            queue.parallel_for(sycl::range<1>(cells), [=](sycl::id<1> id) {
                const size_t index = id[0];
                cache.key[index] = int8_t(int((index * 17 + 3 + size_t(layer)) % 255) - 127);
                cache.value[index] = int8_t(int((index * 29 + 11 + size_t(layer)) % 255) - 127);
            });
            queue.parallel_for(sycl::range<1>(cells / 32), [=](sycl::id<1> id) {
                cache.key_scale[id[0]] = sycl::half(float(id[0] % 11 + 1) / 1024);
                cache.value_scale[id[0]] = sycl::half(float(id[0] % 7 + 1) / 256);
            });
        }
        float* expanded_key = allocations.device<float>(cells);
        float* expanded_value = allocations.device<float>(cells);
        uint32_t* errors = allocations.device<uint32_t>(1);
        queue.wait_and_throw();
        // Exclude JIT/first-use warm-up and all fixture initialization.
        dequantize_q8_kv(queue, caches.front(), expanded_key, expanded_value, chunk).wait_and_throw();
        queue.parallel_for(sycl::range<1>(cells), [=](sycl::id<1> id) {
            expanded_key[id[0]] = expanded_value[id[0]] = sycl::bit_cast<float>(canary);
        }).wait_and_throw();
        struct Row { int visible, layer; sycl::event event; };
        std::vector<Row> rows;
        rows.reserve(160);
        const auto start = std::chrono::steady_clock::now();
        for (int visible = chunk; visible <= 16384; visible += chunk)
            for (int layer = 0; layer < layers; ++layer)
                rows.push_back({visible, layer, dequantize_q8_kv(queue, caches[size_t(layer)], expanded_key, expanded_value, visible)});
        queue.wait_and_throw();
        const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        double gpu_ms = 0;
        for (int visible = chunk; visible <= 16384; visible += chunk) {
            double prefix_ms = 0;
            for (const auto& row : rows) if (row.visible == visible) {
                const auto begin = row.event.get_profiling_info<sycl::info::event_profiling::command_start>();
                const auto end = row.event.get_profiling_info<sycl::info::event_profiling::command_end>();
                prefix_ms += double(end - begin) * 1e-6;
            }
            gpu_ms += prefix_ms;
            std::printf("{\"prefix\":%d,\"dispatches\":10,\"gpu_ms\":%.9f}\n", visible, prefix_ms);
        }
        queue.memset(errors, 0, sizeof(uint32_t));
        const Q8Kv final_cache = caches.back();
        queue.parallel_for(sycl::range<1>(cells), [=](sycl::id<1> id) {
            const size_t index = id[0], cell = (index / dimension) % context;
            bool bad;
            if (cell < 16384) {
                const float expected_key = float(final_cache.key[index]) * float(final_cache.key_scale[index / 32]);
                const float expected_value = float(final_cache.value[index]) * float(final_cache.value_scale[index / 32]);
                bad = !sycl::isfinite(expanded_key[index]) || !sycl::isfinite(expanded_value[index]) ||
                      expanded_key[index] != expected_key || expanded_value[index] != expected_value;
            } else bad = sycl::bit_cast<uint32_t>(expanded_key[index]) != canary || sycl::bit_cast<uint32_t>(expanded_value[index]) != canary;
            if (bad) sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                sycl::access::address_space::global_space>(errors[0]).fetch_add(1);
        }).wait_and_throw();
        uint32_t error_count = 0;
        queue.memcpy(&error_count, errors, sizeof(error_count)).wait_and_throw();
        if (error_count) throw std::runtime_error("Expansion scalar/canary verification failed");
        constexpr double prior_full_prefill_ms = 59600.;
        std::printf("{\"passed\":true,\"scope\":\"isolated existing expansion micro; not full-model gain\",\"context\":32768,\"dimension\":256,\"kv_heads\":2,\"chunk\":1024,\"layers\":10,\"dispatches\":%zu,\"gpu_ms\":%.9f,\"synchronized_wall_ms\":%.9f,\"allocated_bytes\":%zu,\"all_visible_keys_checked\":true,\"tail_canary_unchanged\":true,\"prior_full_prefill_reference_ms\":%.1f,\"micro_gpu_over_reference_percent\":%.9f,\"micro_wall_over_reference_percent\":%.9f,\"models_loaded\":false}\n",
            rows.size(), gpu_ms, wall_ms, bytes, prior_full_prefill_ms, 100 * gpu_ms / prior_full_prefill_ms, 100 * wall_ms / prior_full_prefill_ms);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "KV expansion micro failed: %s\n", error.what()); return 1; }
}
