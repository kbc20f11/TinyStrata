// Exercises the exact production asynchronous upload helper, with guarded
// allocations and pinned sources. Invalid requests must submit neither copy.
#include <cuda_runtime.h>
#include "strata/qwen35/hybrid_buffer_dma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <vector>

namespace H = strata::qwen35::hybrid_buffer_policy;
#define CK(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)
static constexpr uint32_t POISON = 0xa5a5a5a5u;
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
static constexpr auto CAPTURE_MODE = hipStreamCaptureModeGlobal;
#else
static constexpr auto CAPTURE_MODE = cudaStreamCaptureModeGlobal;
#endif
static size_t comparisons = 0;
static int cases = 0;

static void run(int capacity, int T, size_t entries, size_t top, size_t width,
                bool accepted, bool graph = false) {
    const size_t value_words = (size_t(capacity) + 1) * top * width + 64;
    const size_t map_words = (size_t(capacity) + 1) * top + 64;
    uint32_t *host_values = nullptr, *host_map = nullptr;
    uint32_t *device_values = nullptr, *device_map = nullptr;
    CK(cudaHostAlloc((void**) &host_values, value_words * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &host_map, map_words * 4, cudaHostAllocDefault));
    CK(cudaMalloc((void**) &device_values, value_words * 4));
    CK(cudaMalloc((void**) &device_map, map_words * 4));
    for (size_t i = 0; i < value_words; ++i) host_values[i] = 0x3f000000u + uint32_t(i % 65521);
    for (size_t i = 0; i < map_words; ++i) host_map[i] = uint32_t(i % 31);
    cudaStream_t stream{};
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CK(cudaMemsetAsync(device_values, 0xa5, value_words * 4, stream));
    CK(cudaMemsetAsync(device_map, 0xa5, map_words * 4, stream));
    CK(cudaStreamSynchronize(stream));
    cudaGraph_t captured{};
    cudaGraphExec_t executable{};
    if (graph) CK(cudaStreamBeginCapture(stream, CAPTURE_MODE));
    const auto status = H::upload(reinterpret_cast<float*>(device_values), reinterpret_cast<int32_t*>(device_map),
        reinterpret_cast<const float*>(host_values), reinterpret_cast<const int32_t*>(host_map),
        capacity, T, entries, top, width, stream);
    const bool valid_status = accepted ? status == cudaSuccess : status ==
#if defined(STRATA_USE_HIP) || defined(__HIPCC__)
        hipErrorInvalidValue;
#else
        cudaErrorInvalidValue;
#endif
    if (!valid_status) {
        std::fprintf(stderr, "FAIL DMA status R=%d T=%d entries=%zu accepted=%d actual=%d\n",
                     capacity, T, entries, accepted, int(status)); std::exit(1);
    }
    if (graph) {
        CK(cudaStreamEndCapture(stream, &captured));
        CK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        CK(cudaGraphLaunch(executable, stream));
    }
    CK(cudaStreamSynchronize(stream));
    std::vector<uint32_t> actual_values(value_words), actual_map(map_words);
    CK(cudaMemcpy(actual_values.data(), device_values, value_words * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(actual_map.data(), device_map, map_words * 4, cudaMemcpyDeviceToHost));
    const size_t copied_values = accepted ? entries * width : 0;
    const size_t copied_map = accepted ? size_t(T) * top : 0;
    for (size_t i = 0; i < value_words; ++i) {
        const uint32_t expected = i < copied_values ? host_values[i] : POISON;
        if (actual_values[i] != expected) {
            std::fprintf(stderr, "FAIL value/guard R=%d T=%d entries=%zu word=%zu\n", capacity, T, entries, i); std::exit(1);
        }
    }
    for (size_t i = 0; i < map_words; ++i) {
        const uint32_t expected = i < copied_map ? host_map[i] : POISON;
        if (actual_map[i] != expected) {
            std::fprintf(stderr, "FAIL map/guard R=%d T=%d entries=%zu word=%zu\n", capacity, T, entries, i); std::exit(1);
        }
    }
    comparisons += value_words + map_words;
    ++cases;
    if (executable) CK(cudaGraphExecDestroy(executable));
    if (captured) CK(cudaGraphDestroy(captured));
    CK(cudaStreamDestroy(stream));
    CK(cudaFree(device_values)); CK(cudaFree(device_map));
    CK(cudaFreeHost(host_values)); CK(cudaFreeHost(host_map));
}

int main() {
    for (int R : {1, 8, 32, 128}) {
        run(R, R, size_t(R) * 8, 8, 2048, true);
        run(R, R, size_t(R) * 4, 8, 2048, true);
        run(R, R, 0, 8, 2048, true);
        run(R, R + 1, 0, 8, 2048, false);
        run(R, R + 1, size_t(R + 1) * 8, 8, 2048, false);
        run(R, R, size_t(R) * 8 + 1, 8, 2048, false);
    }
    run(128, 8, 65, 8, 2048, false);
    run(128, 1, std::numeric_limits<size_t>::max(), 8, 2048, false);
    run(128, 128, 512, 8, 2048, true, true);
    run(32, 32, 128, 4, 2880, true);
    std::printf("PASS %d production DMA canary cases, %zu compared words\n", cases, comparisons);
}
