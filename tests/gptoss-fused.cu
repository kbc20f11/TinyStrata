// Bitwise regression of the optional GPT-OSS bias fusions against their unfused kernels.
// No model file is required. Run on CUDA and HIP; every buffer has checked red zones.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include <cuda_runtime.h>
#include "strata/gemma4/kernels.hpp"
#include "strata/gemma4/matvec.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace G = strata::gemma4;
namespace K = strata::kernels;
#define CK(x) do { const auto e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "line %d: %s\n", __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
static unsigned cases = 0;
static uint32_t rng = 0x64B1A50Du;
static uint32_t random_word() { rng ^= rng << 13; rng ^= rng >> 17; return rng ^= rng << 5; }
static std::vector<float> values(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const int scale = int(random_word() % 21) - 10;
        v[i] = std::ldexp(float(int(random_word() % 4097) - 2048) / 2048.f, scale);
    }
    return v;
}
struct Device {
    uint8_t* p = nullptr;
    size_t bytes;
    cudaStream_t s;
    Device(size_t n, cudaStream_t stream) : bytes(n), s(stream) {
        CK(cudaMalloc(reinterpret_cast<void**>(&p), bytes + 128));
        CK(cudaMemsetAsync(p, 0xA7, bytes + 128, s));
    }
    ~Device() { if (p) cudaFree(p); }
    template<class T> T* data() const { return reinterpret_cast<T*>(p + 64); }
    template<class T> void put(const std::vector<T>& v) {
        CHECK(v.size() * sizeof(T) == bytes);
        CK(cudaMemcpyAsync(p + 64, v.data(), bytes, cudaMemcpyHostToDevice, s));
    }
    std::vector<uint8_t> get() const {
        CK(cudaStreamSynchronize(s));
        std::vector<uint8_t> all(bytes + 128);
        CK(cudaMemcpy(all.data(), p, all.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < 64; ++i) CHECK(all[i] == 0xA7 && all[bytes + 64 + i] == 0xA7);
        return {all.begin() + 64, all.end() - 64};
    }
};
template<class F> static void launch(F&& work, bool graph, cudaStream_t s) {
    if (!graph) { work(); return; }
    cudaGraph_t g = nullptr;
    cudaGraphExec_t exec = nullptr;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    work();
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    // Replay the graph, not the work used to construct it.
    CK(cudaGraphLaunch(exec, s));
    CK(cudaStreamSynchronize(s));
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(g));
}
static void equal(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const char* stage) {
    CHECK(a.size() == b.size());
    for (size_t i = 0; i < a.size(); i += sizeof(float)) {
        float x, y;
        std::memcpy(&x, a.data() + i, sizeof(float));
        std::memcpy(&y, b.data() + i, sizeof(float));
        CHECK(std::isfinite(x) && std::isfinite(y));
        if (std::memcmp(a.data() + i, b.data() + i, sizeof(float))) {
            std::fprintf(stderr, "FAIL %s float=%zu reference=%.9g fused=%.9g\n", stage, i / 4, x, y);
            std::exit(1);
        }
    }
    ++cases;
}
static void one_case(int n, int tokens, int used, bool graph, cudaStream_t s) {
    const int experts = 32, rr = tokens * used;
    const auto input = values(size_t(rr) * n), bias = values(size_t(experts) * n);
    const auto initial = values(size_t(tokens) * n);
    auto weights = values(rr);
    std::vector<int32_t> ids(rr);
    for (int i = 0; i < rr; ++i) { ids[i] = random_word() % experts; if (i % 5 == 0) weights[i] = 0.f; }
    Device raw(input.size() * 4, s), changed(input.size() * 4, s), db(bias.size() * 4, s);
    Device dw(weights.size() * 4, s), di(ids.size() * 4, s), ref(initial.size() * 4, s), got(initial.size() * 4, s);
    raw.put(input); changed.put(input); db.put(bias); dw.put(weights); di.put(ids); ref.put(initial); got.put(initial);
    CK(cudaStreamSynchronize(s));
    launch([&] {
        G::add_expert_bias_ids(changed.data<float>(), n, rr, di.data<int32_t>(), db.data<float>(), s);
        G::moe_sum(changed.data<float>(), dw.data<float>(), ref.data<float>(), n, used, tokens, s);
    }, graph, s);
    launch([&] { G::moe_sum_bias(raw.data<float>(), dw.data<float>(), di.data<int32_t>(), db.data<float>(),
                                got.data<float>(), n, used, tokens, s); }, graph, s);
    equal(ref.get(), got.get(), "down bias / ordered moe sum");

    // Cached slots can repeatedly add into the same token, in a nontrivial order.
    const int count = std::min(rr, 32);
    G::ScaledRows reference{};
    G::BiasedScaledRows fused{};
    for (int j = 0; j < count; ++j) {
        const int token = (j * 7 + 1) % tokens;
        reference.x[j] = changed.data<float>() + size_t(j) * n;
        reference.acc[j] = ref.data<float>() + size_t(token) * n;
        reference.w[j] = weights[j];
        fused.rows.x[j] = raw.data<float>() + size_t(j) * n;
        fused.rows.acc[j] = got.data<float>() + size_t(token) * n;
        fused.rows.w[j] = weights[j];
        fused.bias[j] = db.data<float>() + size_t(ids[j]) * n;
    }
    ref.put(initial); got.put(initial); CK(cudaStreamSynchronize(s));
    launch([&] { G::add_scaled(reference, count, n, s); }, graph, s);
    launch([&] { G::add_scaled_bias(fused, count, n, s); }, graph, s);
    equal(ref.get(), got.get(), "cached bias / ordered repeated destination");

    // The first tokens rows suffice; use a dedicated view to keep the buffers guarded.
    std::vector<float> output(input.begin(), input.begin() + size_t(tokens) * n);
    Device dx(output.size() * 4, s), dy(output.size() * 4, s);
    dx.put(output); dy.put(output); ref.put(initial); got.put(initial); CK(cudaStreamSynchronize(s));
    launch([&] {
        G::rope_bias(dx.data<float>(), db.data<float>(), dx.data<float>(), n, 1, 0, 1.f, nullptr, 1.f, 0, tokens, s);
        G::axpy(dx.data<float>(), 1.f, ref.data<float>(), n * tokens, true, s);
    }, graph, s);
    launch([&] { G::add_bias_residual(dy.data<float>(), db.data<float>(), got.data<float>(), n, tokens, s); }, graph, s);
    equal(ref.get(), got.get(), "attention output bias / residual");
    for (const Device* d : {&raw, &changed, &db, &dw, &di, &ref, &got, &dx, &dy}) d->get();
}
static void kv_pair_case(int cols, bool graph, cudaStream_t s) {
    constexpr int in = 2880, out = 512;
    const size_t bytes = K::gemma_native_mmvq_weight_bytes(8, in, out);
    std::vector<uint8_t> a(bytes), b(bytes);
    for (auto* w : {&a, &b}) {
        for (size_t at = 0; at < bytes; at += 34) {
            const uint16_t scale = 0x1C00; // half 2^-8
            std::memcpy(w->data() + at, &scale, 2);
            for (int j = 2; j < 34; ++j) (*w)[at + j] = uint8_t(random_word());
        }
    }
    const auto input = values(size_t(cols) * in);
    Device da(bytes, s), db(bytes, s), dx(input.size() * 4, s);
    Device q(K::gemma_native_q8_1_bytes(in) * cols, s);
    Device ra(size_t(cols) * out * 4, s), rb(size_t(cols) * out * 4, s);
    Device ga(size_t(cols) * out * 4, s), gb(size_t(cols) * out * 4, s);
    da.put(a); db.put(b); dx.put(input);
    K::gemma_native_quantize_q8_1(dx.data<float>(), q.data<void>(), in, cols, s);
    CK(cudaStreamSynchronize(s));
    launch([&] {
        K::gemma_native_mmvq(8, da.data<void>(), q.data<void>(), ra.data<float>(), in, out, cols, s);
        K::gemma_native_mmvq(8, db.data<void>(), q.data<void>(), rb.data<float>(), in, out, cols, s);
    }, graph, s);
    launch([&] { CHECK(K::gemma_native_mmvq_pair(8, da.data<void>(), db.data<void>(), q.data<void>(),
                                               ga.data<float>(), gb.data<float>(), in, out, cols, s)); }, graph, s);
    equal(ra.get(), ga.get(), "K projection pair");
    equal(rb.get(), gb.get(), "V projection pair");
    for (const Device* d : {&da, &db, &dx, &q, &ra, &rb, &ga, &gb}) d->get();
}
int main() {
    int devices = 0; CK(cudaGetDeviceCount(&devices)); if (!devices) return 77;
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    K::gemma_native_mmvq_set_multi_exact(true);
    for (const int n : {1, 31, 32, 33, 127, 128, 257, 2880})
        for (const int tokens : {1, 2, 8})
            for (const int used : {1, 4})
                for (const bool graph : {false, true}) one_case(n, tokens, used, graph, s);
    for (const int cols : {2, 3, 4, 5, 7, 8})
        for (const bool graph : {false, true}) kv_pair_case(cols, graph, s);
    CK(cudaStreamDestroy(s));
    std::printf("PASS GPT-OSS bias fusions cases=%u bitwise, graph replay, finite and guards\n", cases);
}
#endif
