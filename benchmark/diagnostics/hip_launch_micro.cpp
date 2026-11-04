// Cost of empty-kernel launches on this HIP runtime (issue #7): back to back, onto an idle stream, and as a graph.
// Every number is a stream span between two events, so it includes submission gaps; it is not a kernel-active time.
//
//   hip_launch_micro [repetitions]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                                           \
    do {                                                                                                \
        hipError_t e_ = (x);                                                                            \
        if (e_ != hipSuccess) {                                                                         \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_));              \
            std::exit(1);                                                                               \
        }                                                                                               \
    } while (0)

__global__ void nop(float* p) {
    if (threadIdx.x == 99999) p[0] = 1.f;
}

static float elapsed_ms(hipEvent_t a, hipEvent_t b) {
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    if (!std::isfinite(ms) || ms <= 0.f) {
        std::fprintf(stderr, "error: an event pair gave a non-positive or non-finite time (%f ms)\n", ms);
        std::exit(2);
    }
    return ms;
}

int main(int argc, char** argv) {
    int reps = 300;
    if (argc > 1) {
        char* end = nullptr;
        const long v = std::strtol(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || v < 1 || v > 1000000) {
            std::fprintf(stderr, "error: repetitions must be a positive integer, got '%s'\n", argv[1]);
            return 2;
        }
        reps = (int) v;
    }
    hipStream_t st;
    CK(hipStreamCreate(&st));
    float* d = nullptr;
    CK(hipMalloc(&d, 64));
    for (int i = 0; i < 100; ++i) nop<<<1, 64, 0, st>>>(d);
    CK(hipGetLastError());
    CK(hipStreamSynchronize(st));
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));

    for (const int n : {5, 50, 1000}) {
        const auto h0 = std::chrono::steady_clock::now();
        CK(hipEventRecord(a, st));
        for (int i = 0; i < n; ++i) nop<<<1, 64, 0, st>>>(d);
        CK(hipGetLastError());
        CK(hipEventRecord(b, st));
        const auto h1 = std::chrono::steady_clock::now();
        CK(hipEventSynchronize(b));
        std::printf("n=%4d back to back: host issue %.1f us/launch, stream span %.1f us/launch\n", n,
                    std::chrono::duration<double, std::micro>(h1 - h0).count() / n, elapsed_ms(a, b) * 1000.0 / n);
    }

    // One empty kernel onto a drained stream, after the host has spent idle_us.  The event pair is recorded around
    // the launch exactly as in the other modes.
    for (const int idle_us : {0, 20, 100, 300}) {
        double total_ms = 0;
        for (int r = 0; r < reps; ++r) {
            CK(hipStreamSynchronize(st));
            const auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < idle_us) {}
            CK(hipEventRecord(a, st));
            nop<<<1, 64, 0, st>>>(d);
            CK(hipGetLastError());
            CK(hipEventRecord(b, st));
            CK(hipEventSynchronize(b));
            total_ms += elapsed_ms(a, b);
        }
        std::printf("one launch onto an idle stream, host idle %3d us before: %.1f us between the events\n", idle_us,
                    total_ms * 1000.0 / reps);
    }

    hipGraph_t g = nullptr;
    hipGraphExec_t ge = nullptr;
    CK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
    for (int i = 0; i < 5; ++i) nop<<<1, 64, 0, st>>>(d);
    CK(hipStreamEndCapture(st, &g));
    CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CK(hipGraphLaunch(ge, st));
    CK(hipStreamSynchronize(st));
    CK(hipEventRecord(a, st));
    for (int i = 0; i < reps; ++i) CK(hipGraphLaunch(ge, st));
    CK(hipEventRecord(b, st));
    CK(hipEventSynchronize(b));
    const double per_graph = elapsed_ms(a, b) * 1000.0 / reps;
    std::printf("graph of 5 empty kernels, launched back to back: %.1f us per graph (%.1f us per kernel)\n", per_graph,
                per_graph / 5);

    CK(hipGraphExecDestroy(ge));
    CK(hipGraphDestroy(g));
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    CK(hipFree(d));
    CK(hipStreamDestroy(st));
    return 0;
}
