#pragma once
#include <sycl/sycl.hpp>
#include <memory>
#include <cstddef>

namespace strata::intel {
class PrefillGemm;
struct NpuSplitOptions {
    bool parallel = true, remote = false, interleaved = false;
    bool reuse_tensors = true, profile_stages = false;
};
struct NpuSplitTiming {
    bool valid = false, cache_hit = false, tensor_reused = false, profiled = false;
    int rows = 0, npu_rows = 0, tokens = 0;
    double compile_ms = 0, weight_prepare_ms = 0, input_prepare_ms = 0;
    double prepare_wait_ms = 0, tensor_bind_ms = 0, npu_submit_ms = 0;
    double gpu_submit_ms = 0, npu_wait_ms = 0, npu_host_span_ms = 0, host_ms = 0;
    double gpu_status_query_ms = 0;
    double begin_host_ms = 0, deferred_gap_ms = 0;
    // Profiling-only prior-work fence perturbs scheduling; not a speedup claim.
    double prior_gpu_drain_ms = -1, weight_decode_gpu_ms = -1, input_convert_gpu_ms = -1;
    // GPU timestamps include conversion/dequantization in the remainder span.
    // Negative means unavailable/incomplete, never an invented zero duration.
    double gpu_remainder_ms = -1, gpu_compute_event_ms = -1, gpu_gather_ms = -1, npu_merge_ms = -1;
};
struct NpuSplitCounters {
    size_t eligible = 0, served = 0, cache_hits = 0;
    size_t cache_capacity_skips = 0, graph_capacity_skips = 0;
    size_t max_pending_requests = 0;
};
struct NpuMemoryUsage {
    size_t host_usm_bytes = 0, device_usm_bytes = 0;
    size_t fp16_cache_capacity_bytes = 0, fp16_cache_used_bytes = 0;
    size_t compiled_graphs = 0, tensor_bindings = 0;
    size_t request_slots = 1, infer_requests = 0;
    // Driver/compiler allocations are opaque, not measured as zero.
    size_t opaque_runtime_reservation_bytes = 96ull << 20;
};
// Optional OpenVINO Intel NPU candidate. Owns bounded projection scratch,
// not another persistent model or expert weight store.
// Multiplicands and the internal MatMul result are FP16. The public output
// buffer is F32 for the surrounding GPU operations; qualify model quality.
class NpuProjection {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit NpuProjection(sycl::queue& queue, unsigned request_slots = 1);
    ~NpuProjection();
    sycl::event run(int type, const uint8_t* weights, const float* input, float* output,
                    int rows, int width, int tokens);
#ifdef TINYSTRATA_INTEL_MKL
    bool run_split(PrefillGemm& gpu, int type, const uint8_t* weights, const float* input,
                   float* output, int rows, int width, int tokens, int npu_rows,
                   bool parallel, bool remote, sycl::event& completion, bool interleaved = false);
    bool run_split(PrefillGemm& gpu, int type, const uint8_t* weights, const float* input,
                   float* output, int rows, int width, int tokens, int npu_rows,
                   const NpuSplitOptions& options, sycl::event& completion);
    // Convert both bank inputs before starting their GPU suffixes. This keeps
    // a later bank's readiness wait from draining earlier useful GPU work.
    sycl::event prepare_split_input(unsigned bank, const float* input, int width, int tokens);
    // One pending request per admitted slot. GPUcompletion covers only the
    // GPU suffix; join the same bank before consuming its NPU prefix.
    bool begin_split(PrefillGemm& gpu, int type, const uint8_t* weights, const float* input,
                     float* output, int rows, int width, int tokens, int npu_rows,
                     const NpuSplitOptions& options, sycl::event& GPUcompletion, unsigned bank = 0);
    sycl::event finish_split(unsigned bank = 0);
    bool split_pending(unsigned bank = 0) const noexcept;
    unsigned pending_split_count() const noexcept;
    // Does not synchronize. Query immediately after completion.wait() in a
    // profiling micro; normal callers need not collect device stage timings.
    NpuSplitTiming last_split_timing(unsigned bank = 0) const;
    NpuSplitCounters split_counters() const;
#endif
    NpuMemoryUsage memory_usage() const;
};
} // namespace strata::intel
