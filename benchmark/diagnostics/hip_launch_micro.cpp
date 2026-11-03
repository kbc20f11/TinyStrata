#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
__global__ void nop(float* p) { if (threadIdx.x == 99999) p[0] = 1.f; }
int main() {
    hipStream_t st; hipStreamCreate(&st);
    float* d; hipMalloc(&d, 64);
    for (int i = 0; i < 100; ++i) nop<<<1, 64, 0, st>>>(d);
    hipStreamSynchronize(st);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    for (int n : {5, 50, 1000}) {
        // GPU-side time for n back-to-back empty launches, and the host time to issue them
        auto h0 = std::chrono::steady_clock::now();
        hipEventRecord(a, st);
        for (int i = 0; i < n; ++i) nop<<<1, 64, 0, st>>>(d);
        hipEventRecord(b, st);
        auto h1 = std::chrono::steady_clock::now();
        hipEventSynchronize(b);
        float ms; hipEventElapsedTime(&ms, a, b);
        std::printf("n=%4d  host issue %.1f us/launch, stream span %.1f us/launch\n", n,
                    std::chrono::duration<double, std::micro>(h1 - h0).count() / n, ms * 1000.0 / n);
    }
    // an empty kernel launched onto an idle stream: event delta around one launch, after the host spent `idle_us` since the stream drained
    for (int idle_us : {0, 20, 100, 300}) {
        double tot = 0; const int reps = 300;
        for (int r = 0; r < reps; ++r) {
            hipStreamSynchronize(st);
            auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < idle_us) {}
            hipEventRecord(a, st);
            nop<<<1, 64, 0, st>>>(d);
            hipEventRecord(b, st);
            hipEventSynchronize(b);
            float ms; hipEventElapsedTime(&ms, a, b); tot += ms;
        }
        std::printf("one empty launch on an idle stream (host idle %3d us before): %.1f us between the events\n", idle_us, tot * 1000.0 / reps);
    }
    // a graph of 5 empty kernels, launched repeatedly
    hipGraph_t g; hipGraphExec_t ge;
    hipStreamBeginCapture(st, hipStreamCaptureModeGlobal);
    for (int i = 0; i < 5; ++i) nop<<<1, 64, 0, st>>>(d);
    hipStreamEndCapture(st, &g); hipGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    hipGraphLaunch(ge, st); hipStreamSynchronize(st);
    hipEventRecord(a, st);
    for (int i = 0; i < 200; ++i) hipGraphLaunch(ge, st);
    hipEventRecord(b, st); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    std::printf("graph of 5 empty kernels: %.1f us per graph (%.1f us per kernel)\n", ms * 1000.0 / 200, ms * 1000.0 / 200 / 5);
    return 0;
}
