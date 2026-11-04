// Standalone timing of gemma_native_mmvq / gemma_native_mmvq_slots on the Gemma 4 26B-A4B Q4_0 shapes (issue #7).
// Weights are random bytes (timing only).  Two modes:
//
//   gemma_mmvq_micro shapes [calls] [host_idle_us]
//       Single shapes.  Matrices are copies in one hipMalloc block of at least 256 MB, rotated through.
//   gemma_mmvq_micro layout [steps] [host_idle_us] [layers]
//       The routed-expert step as Engine::moe() issues it for one token: 8 picks among the slots of one layer, each
//       slot = [gate | up | down] back to back (slot stride 2 * gu_bytes + d_bytes), the same activation pointer for all
//       gate+up picks, separate rows for the down picks.  Run on a contiguous hipMalloc arena and on a VMM arena
//       mapped like the engine's expert cache (hipMemAddressReserve + hipMemCreate/Map in 64 MiB pieces).
//
// Timing pattern, identical for every number printed: an event pair around each call or step, summed after the last
// one.  With host_idle_us = 0 the steps are submitted back to back and the host waits only at the end.  With a
// non-zero value the host waits until the stream has drained and then spins host_idle_us before each step.  The
// numbers are stream spans between the two events, so they include submission gaps; they are not kernel-active times.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "strata/gemma4/matvec.hpp"

using namespace strata::kernels;

#define CK(x)                                                                                           \
    do {                                                                                                \
        hipError_t e_ = (x);                                                                            \
        if (e_ != hipSuccess) {                                                                         \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_));              \
            std::exit(1);                                                                               \
        }                                                                                               \
    } while (0)

namespace {

constexpr int kQ4_0 = 2;

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "error: %s\n", what);
    std::exit(2);
}

int positive_int(const char* text, const char* name) {
    char* end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < 1 || v > 10000000) {
        std::fprintf(stderr, "error: %s must be a positive integer, got '%s'\n", name, text);
        std::exit(2);
    }
    return (int) v;
}

int non_negative_int(const char* text, const char* name) {
    char* end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < 0 || v > 10000000) {
        std::fprintf(stderr, "error: %s must be a non-negative integer, got '%s'\n", name, text);
        std::exit(2);
    }
    return (int) v;
}

void spin_us(int us) {
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < us) {}
}

// Runs `count` items; item i is issued by issue(i).  Returns the mean microseconds between the event pair around an
// item, and rejects timings that are not positive finite numbers.
template <class F>
double time_items(hipStream_t st, int count, int idle_us, F&& issue) {
    std::vector<hipEvent_t> a((size_t) count), b((size_t) count);
    for (int i = 0; i < count; ++i) {
        CK(hipEventCreate(&a[(size_t) i]));
        CK(hipEventCreate(&b[(size_t) i]));
    }
    CK(hipStreamSynchronize(st));
    for (int i = 0; i < count; ++i) {
        if (idle_us > 0) {
            CK(hipStreamSynchronize(st));
            spin_us(idle_us);
        }
        CK(hipEventRecord(a[(size_t) i], st));
        issue(i);
        CK(hipGetLastError());
        CK(hipEventRecord(b[(size_t) i], st));
    }
    CK(hipStreamSynchronize(st));
    double total_ms = 0;
    for (int i = 0; i < count; ++i) {
        float ms = 0;
        CK(hipEventElapsedTime(&ms, a[(size_t) i], b[(size_t) i]));
        if (!std::isfinite(ms) || ms <= 0.f) fail("an event pair gave a non-positive or non-finite time");
        total_ms += ms;
        CK(hipEventDestroy(a[(size_t) i]));
        CK(hipEventDestroy(b[(size_t) i]));
    }
    return total_ms * 1000.0 / count;
}

void fill_random(uint8_t* dev, size_t bytes) {
    std::vector<uint8_t> host(std::min<size_t>(bytes, 64u << 20));
    std::mt19937 rng(1234);
    for (auto& v : host) v = (uint8_t) (rng() & 0x3f);
    for (size_t off = 0; off < bytes; off += host.size())
        CK(hipMemcpy(dev + off, host.data(), std::min(host.size(), bytes - off), hipMemcpyHostToDevice));
}

struct Shape {
    const char* name;
    int n_in, n_out, slots;
};

void run_shapes(hipStream_t st, int calls, int idle_us) {
    const Shape shapes[] = {
        {"expert gate+up (8 slots)", 2816, 1408, 8}, {"expert down (8 slots)", 704, 2816, 8},
        {"expert gate+up (1)", 2816, 1408, 1},       {"expert down (1)", 704, 2816, 1},
        {"dense gate/up", 2816, 2112, 1},            {"dense down", 2112, 2816, 1},
        {"attn q (swa)", 2816, 4096, 1},             {"attn q (global)", 2816, 8192, 1},
        {"attn k/v", 2816, 2048, 1},                 {"attn o (swa)", 4096, 2816, 1},
        {"lm_head slice 65536", 2816, 65536, 1},
    };
    std::printf("%-28s %7s %7s %9s %10s %10s\n", "shape", "n_in", "n_out", "MB/call", "us/call", "GB/s");
    for (const Shape& s : shapes) {
        const size_t wbytes = gemma_native_mmvq_weight_bytes(kQ4_0, s.n_in, s.n_out);
        const int copies = (int) std::max<size_t>(8, (256u << 20) / (wbytes * s.slots) + 1);
        const size_t total = wbytes * (size_t) copies * s.slots;
        uint8_t* w = nullptr;
        CK(hipMalloc(&w, total));
        fill_random(w, total);
        const size_t xb = gemma_native_q8_1_bytes(s.n_in);
        uint8_t* x = nullptr;
        CK(hipMalloc(&x, xb * 8));
        std::vector<float> hx((size_t) s.n_in);
        std::mt19937 rng(7);
        for (auto& v : hx) v = (float) (rng() % 100) / 100.f - 0.5f;
        float* dx = nullptr;
        CK(hipMalloc(&dx, hx.size() * 4));
        CK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        gemma_native_quantize_q8_1(dx, x, s.n_in, 1, st);
        for (int j = 1; j < 8; ++j) CK(hipMemcpyAsync(x + xb * j, x, xb, hipMemcpyDeviceToDevice, st));
        float* y = nullptr;
        CK(hipMalloc(&y, (size_t) s.n_out * 8 * 4));
        const auto once = [&](int c) {
            c %= copies;
            if (s.slots == 1) {
                gemma_native_mmvq(kQ4_0, w + (size_t) c * wbytes, x, y, s.n_in, s.n_out, 1, st);
            } else {
                MmvqSlots m{};
                for (int j = 0; j < s.slots; ++j) {
                    m.w[j] = w + ((size_t) c * s.slots + j) * wbytes;
                    m.x[j] = x + xb * j;
                    m.y[j] = y + (size_t) j * s.n_out;
                }
                if (!gemma_native_mmvq_slots(kQ4_0, m, s.slots, s.n_in, s.n_out, st)) fail("slots kernel unsupported");
            }
        };
        for (int i = 0; i < 20; ++i) once(i);
        CK(hipStreamSynchronize(st));
        const double us = time_items(st, calls, idle_us, once);
        const double mb = (double) wbytes * s.slots / 1e6;
        std::printf("%-28s %7d %7d %9.2f %10.2f %10.1f\n", s.name, s.n_in, s.n_out, mb, us, mb * 1e6 / (us * 1e-6) / 1e9);
        CK(hipFree(w));
        CK(hipFree(x));
        CK(hipFree(dx));
        CK(hipFree(y));
    }
}

struct Arena {
    uint8_t* base = nullptr;
    size_t bytes = 0;
    bool vmm = false;
    std::vector<std::pair<hipMemGenericAllocationHandle_t, size_t>> pieces;
};

Arena make_arena(size_t bytes, bool vmm, int device) {
    Arena a;
    a.vmm = vmm;
    if (!vmm) {
        a.bytes = bytes;
        CK(hipMalloc((void**) &a.base, bytes));
        return a;
    }
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    size_t grain = 0;
    CK(hipMemGetAllocationGranularity(&grain, &prop, hipMemAllocationGranularityMinimum));
    const size_t piece = std::max<size_t>(grain, (size_t(64) << 20) / grain * grain);
    a.bytes = (bytes + grain - 1) / grain * grain;
    void* p = nullptr;
    CK(hipMemAddressReserve(&p, a.bytes, 0, nullptr, 0));
    a.base = (uint8_t*) p;
    hipMemAccessDesc access = {};
    access.location = prop.location;
    access.flags = hipMemAccessFlagsProtReadWrite;
    for (size_t at = 0; at < a.bytes;) {
        const size_t n = std::min(piece, a.bytes - at);
        hipMemGenericAllocationHandle_t h = nullptr;
        CK(hipMemCreate(&h, n, &prop, 0));
        CK(hipMemMap(a.base + at, n, 0, h, 0));
        CK(hipMemSetAccess(a.base + at, n, &access, 1));
        a.pieces.emplace_back(h, n);
        at += n;
    }
    std::printf("  VMM arena: granularity %zu bytes, %zu pieces of up to %zu MiB\n", grain, a.pieces.size(), piece >> 20);
    return a;
}

void free_arena(Arena& a) {
    if (!a.vmm) {
        CK(hipFree(a.base));
        return;
    }
    size_t at = 0;
    for (auto& [h, n] : a.pieces) {
        CK(hipMemUnmap(a.base + at, n));
        CK(hipMemRelease(h));
        at += n;
    }
    CK(hipMemAddressFree(a.base, a.bytes));
}

// The routed-expert step of Engine::moe() for one token.
void run_layout(hipStream_t st, int steps, int idle_us, int layers) {
    const int n_embd = 2816, ff = 704, experts = 128, used = 8;
    const size_t gu = gemma_native_mmvq_weight_bytes(kQ4_0, n_embd, ff);      // gate (or up) of one expert
    const size_t dn = gemma_native_mmvq_weight_bytes(kQ4_0, ff, n_embd);      // down of one expert
    const size_t slot = 2 * gu + dn;
    const size_t layer_bytes = slot * experts;
    const size_t total = layer_bytes * (size_t) layers;
    std::printf("slot = %zu bytes (gate+up %zu, down %zu), %d layers of %d slots = %.2f GiB\n", slot, 2 * gu, dn, layers,
                experts, (double) total / (1 << 30));
    int device = 0;
    CK(hipGetDevice(&device));
    const size_t xq = gemma_native_q8_1_bytes(n_embd), q8_row = gemma_native_q8_1_bytes(ff);
    uint8_t *x = nullptr, *q8 = nullptr;
    float *g = nullptr, *d = nullptr, *dx = nullptr;
    CK(hipMalloc(&x, xq));
    CK(hipMalloc(&q8, q8_row * used));
    CK(hipMalloc(&g, (size_t) used * 2 * ff * 4));
    CK(hipMalloc(&d, (size_t) used * n_embd * 4));
    CK(hipMalloc(&dx, (size_t) std::max(n_embd, ff * used) * 4));
    {
        std::vector<float> h((size_t) std::max(n_embd, ff * used));
        std::mt19937 rng(11);
        for (auto& v : h) v = (float) (rng() % 100) / 100.f - 0.5f;
        CK(hipMemcpy(dx, h.data(), h.size() * 4, hipMemcpyHostToDevice));
        gemma_native_quantize_q8_1(dx, x, n_embd, 1, st);
        gemma_native_quantize_q8_1(dx, q8, ff * used, 1, st);   // the rows back to back, as the engine does
    }
    // 85% of each layer's experts are resident (the rest are not picked), as in the measured run
    std::mt19937 rng(99);
    std::vector<std::vector<int>> resident((size_t) layers);
    for (auto& r : resident) {
        std::vector<int> all((size_t) experts);
        for (int e = 0; e < experts; ++e) all[(size_t) e] = e;
        std::shuffle(all.begin(), all.end(), rng);
        r.assign(all.begin(), all.begin() + experts * 85 / 100);
    }
    std::printf("%-34s %12s %12s\n", "arena / submission", "us per step", "GB/s");
    for (const bool vmm : {false, true}) {
        Arena arena = make_arena(total, vmm, device);
        fill_random(arena.base, total);
        // the same sequence of (layer, picks) for every configuration
        std::mt19937 pick_rng(5);
        std::vector<std::pair<int, std::vector<int>>> plan((size_t) steps);
        for (auto& p : plan) {
            p.first = (int) (pick_rng() % (unsigned) layers);
            auto r = resident[(size_t) p.first];
            std::shuffle(r.begin(), r.end(), pick_rng);
            p.second.assign(r.begin(), r.begin() + used);
        }
        const auto step = [&](int i) {
            const auto& [il, picks] = plan[(size_t) i];
            MmvqSlots gate_up{}, down{};
            for (int j = 0; j < used; ++j) {
                const uint8_t* s = arena.base + (size_t) il * layer_bytes + (size_t) picks[(size_t) j] * slot;
                gate_up.w[j] = s;                          // gate then up, as one 1408-row matrix
                gate_up.x[j] = x;                          // the same activation for every pick
                gate_up.y[j] = g + (size_t) j * 2 * ff;
                down.w[j] = s + 2 * gu;
                down.x[j] = q8 + (size_t) j * q8_row;
                down.y[j] = d + (size_t) j * n_embd;
            }
            if (!gemma_native_mmvq_slots(kQ4_0, gate_up, used, n_embd, 2 * ff, st)) fail("slots kernel unsupported");
            if (!gemma_native_mmvq_slots(kQ4_0, down, used, ff, n_embd, st)) fail("slots kernel unsupported");
        };
        for (int i = 0; i < 20; ++i) step(i % steps);
        CK(hipStreamSynchronize(st));
        const double bytes = (double) used * slot;
        std::vector<int> idles{0};
        if (idle_us > 0) idles.push_back(idle_us);
        for (const int idle : idles) {
            const double us = time_items(st, steps, idle, step);
            std::printf("%-34s %12.1f %12.1f\n",
                        (std::string(vmm ? "VMM, 64 MiB pieces" : "hipMalloc") +
                         (idle ? ", host idle before step" : ", back to back")).c_str(),
                        us, bytes / (us * 1e-6) / 1e9);
        }
        free_arena(arena);
    }
    CK(hipFree(x));
    CK(hipFree(q8));
    CK(hipFree(g));
    CK(hipFree(d));
    CK(hipFree(dx));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s shapes [calls] [host_idle_us]\n       %s layout [steps] [host_idle_us] [layers]\n",
                     argv[0], argv[0]);
        return 2;
    }
    hipStream_t st;
    CK(hipStreamCreate(&st));
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s, %d multiprocessors, core clock %d MHz, memory clock %d MHz, bus %d bit\n", prop.name,
                prop.multiProcessorCount, prop.clockRate / 1000, prop.memoryClockRate / 1000, prop.memoryBusWidth);
    const std::string mode = argv[1];
    if (mode == "shapes") {
        run_shapes(st, argc > 2 ? positive_int(argv[2], "calls") : 400, argc > 3 ? non_negative_int(argv[3], "host_idle_us") : 0);
    } else if (mode == "layout") {
        run_layout(st, argc > 2 ? positive_int(argv[2], "steps") : 400, argc > 3 ? non_negative_int(argv[3], "host_idle_us") : 0,
                   argc > 4 ? positive_int(argv[4], "layers") : 12);
    } else {
        std::fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
        return 2;
    }
    return 0;
}
