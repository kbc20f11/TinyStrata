// Per-shape timing of gemma_native_mmvq (single) and gemma_native_mmvq_slots (8 matrices) on the real Gemma 4
// 26B-A4B Q4_0 shapes.  Weights are random bytes (timing only).  Several copies of each matrix are rotated through so
// that the card's last-level cache cannot hold the working set.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include "strata/gemma4/matvec.hpp"
using namespace strata::kernels;

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); std::exit(1); } } while (0)

struct Shape { const char* name; int n_in, n_out, slots; };

int main(int argc, char** argv) {
    const int type = 2;                       // Q4_0
    hipStream_t st; CK(hipStreamCreate(&st));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s, %d CUs, clock %d MHz, mem clock %d MHz, bus %d bit\n", prop.name, prop.multiProcessorCount,
                prop.clockRate / 1000, prop.memoryClockRate / 1000, prop.memoryBusWidth);
    const Shape shapes[] = {
        {"expert gate+up (8 slots)", 2816, 1408, 8}, {"expert down (8 slots)", 704, 2816, 8},
        {"expert gate+up (1)", 2816, 1408, 1},       {"expert down (1)", 704, 2816, 1},
        {"dense gate/up", 2816, 2112, 1},            {"dense down", 2112, 2816, 1},
        {"attn q (swa)", 2816, 4096, 1},             {"attn q (global)", 2816, 8192, 1},
        {"attn k/v", 2816, 2048, 1},                 {"attn o (swa)", 4096, 2816, 1},
        {"lm_head slice 65536", 2816, 65536, 1},
    };
    const int iters = argc > 1 ? std::atoi(argv[1]) : 400;
    std::printf("%-28s %7s %7s %9s %10s %10s %9s\n", "shape", "n_in", "n_out", "MB/call", "us/call", "GB/s", "us floor");
    for (const Shape& s : shapes) {
        const size_t wbytes = gemma_native_mmvq_weight_bytes(type, s.n_in, s.n_out);
        // working set of at least 256 MB, so every call reads from VRAM
        const int copies = (int) std::max<size_t>(8, (256u << 20) / (wbytes * s.slots) + 1);
        const size_t total = wbytes * (size_t) copies * s.slots;
        uint8_t* w = nullptr; CK(hipMalloc(&w, total));
        std::vector<uint8_t> host(std::min<size_t>(total, 64u << 20));
        for (size_t i = 0; i < host.size(); ++i) host[i] = (uint8_t) (rand() & 0x3f);
        for (size_t off = 0; off < total; off += host.size()) CK(hipMemcpy(w + off, host.data(), std::min(host.size(), total - off), hipMemcpyHostToDevice));
        const size_t xb = gemma_native_q8_1_bytes(s.n_in);
        uint8_t* x = nullptr; CK(hipMalloc(&x, xb * 8));
        std::vector<float> hx((size_t) s.n_in); for (auto& v : hx) v = (float) (rand() % 100) / 100.f - 0.5f;
        float* dx = nullptr; CK(hipMalloc(&dx, hx.size() * 4)); CK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        gemma_native_quantize_q8_1(dx, x, s.n_in, 1, st);
        for (int j = 1; j < 8; ++j) CK(hipMemcpyAsync(x + xb * j, x, xb, hipMemcpyDeviceToDevice, st));
        float* y = nullptr; CK(hipMalloc(&y, (size_t) s.n_out * 8 * 4));
        auto once = [&](int c) {
            if (s.slots == 1) {
                gemma_native_mmvq(type, w + (size_t) c * wbytes, x, y, s.n_in, s.n_out, 1, st);
            } else {
                MmvqSlots m{};
                for (int j = 0; j < s.slots; ++j) { m.w[j] = w + ((size_t) c * s.slots + j) * wbytes; m.x[j] = x + xb * j; m.y[j] = y + (size_t) j * s.n_out; }
                if (!gemma_native_mmvq_slots(type, m, s.slots, s.n_in, s.n_out, st)) { std::fprintf(stderr, "slots unsupported\n"); std::exit(2); }
            }
        };
        for (int i = 0; i < 20; ++i) once(i % copies);
        CK(hipStreamSynchronize(st));
        hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
        const int idle_us = argc > 2 ? std::atoi(argv[2]) : 0;
        double us = 0;
        if (idle_us == 0) {
            CK(hipEventRecord(a, st));
            for (int i = 0; i < iters; ++i) once(i % copies);
            CK(hipEventRecord(b, st)); CK(hipEventSynchronize(b));
            float ms = 0; CK(hipEventElapsedTime(&ms, a, b)); us = ms * 1000.0 / iters;
        } else {
            double tot = 0;
            for (int i = 0; i < iters; ++i) {
                CK(hipStreamSynchronize(st));
                auto t0 = std::chrono::steady_clock::now();
                while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < idle_us) {}
                CK(hipEventRecord(a, st));
                once(i % copies);
                CK(hipEventRecord(b, st)); CK(hipEventSynchronize(b));
                float ms = 0; CK(hipEventElapsedTime(&ms, a, b)); tot += ms;
            }
            us = tot * 1000.0 / iters;
        }
        const double mb = (double) wbytes * s.slots / 1e6;
        std::printf("%-28s %7d %7d %9.2f %10.2f %10.1f\n", s.name, s.n_in, s.n_out, mb, us, mb * 1e6 / (us * 1e-6) / 1e9);
        hipFree(w); hipFree(x); hipFree(dx); hipFree(y);
    }
    return 0;
}
