// Cached Gemma experts: individual products versus 8/32-slot launches, including
// the GEGLU/Q8_1/down/ordered-add pipeline. No model files are required.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/gemma4/matvec.hpp"
#include "strata/gemma4/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace K = strata::kernels;
namespace G = strata::gemma4;
#define CK(x) do { auto e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "line %d: %s\n", __LINE__, cudaGetErrorString(e)); std::exit(2); } } while (0)
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
static constexpr int kType = 2; // GGUF Q4_0, the deployed Gemma expert format.
static constexpr int kCapacity = K::kMmvqSlotsMax;
static constexpr int kTokens = 4;
static_assert(kCapacity == 32, "both backends must exercise the 32-slot payload");
static unsigned cases = 0;
static uint64_t checksum = 1469598103934665603ull;
static double maximum_scalar_error = 0;
static uint32_t rng = 0xD391A507u;
static uint32_t random_word() { rng ^= rng << 13; rng ^= rng >> 17; return rng ^= rng << 5; }
static uint16_t half_bits(float x) { const __half h = __float2half(x); uint16_t b; std::memcpy(&b, &h, 2); return b; }
static double half_value(uint16_t h) {
    const int e = (h >> 10) & 31, f = h & 1023;
    const double v = e ? std::ldexp(1.0 + f / 1024.0, e - 15) : std::ldexp(double(f), -24);
    return h & 0x8000 ? -v : v;
}
struct Q8 { uint16_t d, sum; int8_t q[32]; };
static_assert(sizeof(Q8) == 36, "Q8_1 block layout");

struct Device {
    uint8_t* base = nullptr;
    size_t bytes;
    explicit Device(size_t n) : bytes(n) {
        CK(cudaMalloc(reinterpret_cast<void**>(&base), n + 128));
        CK(cudaMemset(base, 0xA7, n + 128));
        // The working stream is nonblocking. Finish default-stream guard fills
        // before it can write these buffers; otherwise HIP can race the fill.
        CK(cudaDeviceSynchronize());
    }
    ~Device() { if (base) cudaFree(base); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void* data() const { return base + 64; }
    template<class T> T* as() const { return static_cast<T*>(data()); }
    void put(const void* p, size_t n, cudaStream_t s) { CHECK(n <= bytes); CK(cudaMemcpyAsync(data(), p, n, cudaMemcpyHostToDevice, s)); }
    void poison(cudaStream_t s) { CK(cudaMemsetAsync(data(), 0xFF, bytes, s)); }
    std::vector<uint8_t> get() const { std::vector<uint8_t> out(bytes); CK(cudaMemcpy(out.data(), data(), bytes, cudaMemcpyDeviceToHost)); return out; }
    void check_guards() const {
        uint8_t a[64], b[64]; CK(cudaMemcpy(a, base, 64, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b, base + 64 + bytes, 64, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 64; ++i) CHECK(a[i] == 0xA7 && b[i] == 0xA7);
    }
};
static std::vector<uint8_t> weights(int n_in, int n_out, int experts) {
    const size_t stride = K::gemma_native_mmvq_weight_bytes(kType, n_in, n_out);
    std::vector<uint8_t> result(stride * experts);
    for (size_t b = 0; b < result.size(); b += 18) {
        const uint16_t d = half_bits(std::ldexp((b / 18) % 3 == 0 ? -1.0f : 1.0f, -8 + int((b / 18) % 3)));
        std::memcpy(result.data() + b, &d, 2);
        for (int j = 2; j < 18; ++j) result[b + j] = uint8_t(random_word());
    }
    return result;
}
static std::vector<float> inputs(int n_in) {
    std::vector<float> result(size_t(kTokens) * n_in);
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = float(int(random_word() % 2049) - 1024) / 1024.0f;
    return result;
}
static int token(int expert) { return (expert * 7 + 3) % kTokens; }
static float coefficient(int expert) { return (expert % 3 == 0 ? -1.0f : 1.0f) * float(expert + 1) / 64.0f; }
static void equal_bytes(const std::vector<uint8_t>& expected, const std::vector<uint8_t>& got,
                        size_t used, bool floats, const char* stage, int count, int split) {
    CHECK(expected.size() == got.size() && used <= got.size());
    if (floats) {
        CHECK(used % sizeof(float) == 0);
        for (size_t i = 0; i < used; i += sizeof(float)) {
            float a, b; std::memcpy(&a, expected.data() + i, 4); std::memcpy(&b, got.data() + i, 4);
            CHECK(std::isfinite(a) && std::isfinite(b));
        }
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (expected[i] != got[i] || (i >= used && got[i] != 0xFF)) {
            std::fprintf(stderr, "FAIL %s count=%d split=%d byte=%zu expected=%02x got=%02x used=%zu\n",
                         stage, count, split, i, unsigned(expected[i]), unsigned(got[i]), used);
            std::exit(1);
        }
        if (i < used) checksum = (checksum ^ got[i]) * 1099511628211ull;
    }
}

// Small reduction and odd output count exercise the two-row kernel and its tail.
// Check its result against an independent scalar decode as well as singleton GPU calls.
static void matrix_case(cudaStream_t s) {
    constexpr int n = 320, out = 13;
    const size_t wb = K::gemma_native_mmvq_weight_bytes(kType, n, out), qb = K::gemma_native_q8_1_bytes(n);
    auto w = weights(n, out, kCapacity); auto x = inputs(n);
    Device dw(w.size()), dx(x.size() * 4), dq(kTokens * qb), reference(size_t(kCapacity) * out * 4), actual(reference.bytes);
    dw.put(w.data(), w.size(), s); dx.put(x.data(), dx.bytes, s);
    K::gemma_native_quantize_q8_1(dx.as<float>(), dq.data(), n, kTokens, s);
    reference.poison(s); actual.poison(s);
    K::MmvqSlots slots{};
    for (int j = 0; j < kCapacity; ++j) {
        slots.w[j] = dw.as<uint8_t>() + size_t(j) * wb;
        slots.x[j] = dq.as<uint8_t>() + size_t(token(j)) * qb;
        slots.y[j] = actual.as<float>() + size_t(j) * out;
        K::gemma_native_mmvq(kType, slots.w[j], slots.x[j], reference.as<float>() + size_t(j) * out, n, out, 1, s);
    }
    CHECK(K::gemma_native_mmvq_slots(kType, slots, kCapacity, n, out, s));
    CK(cudaStreamSynchronize(s));
    CHECK(dw.get() == w);
    const auto copied_input = dx.get();
    CHECK(std::memcmp(copied_input.data(), x.data(), dx.bytes) == 0);
    const auto a = reference.get(), b = actual.get(), qbytes = dq.get();
    equal_bytes(a, b, a.size(), true, "small-tail MMVQ", kCapacity, kCapacity);
    for (int j = 0; j < kCapacity; ++j) for (int r = 0; r < out; ++r) {
        double expected = 0, bound = 0;
        for (int b = 0; b < n / 32; ++b) {
            const uint8_t* block = w.data() + size_t(j) * wb + (size_t(r) * (n / 32) + b) * 18;
            uint16_t wd; std::memcpy(&wd, block, 2);
            Q8 q; std::memcpy(&q, qbytes.data() + size_t(token(j)) * qb + size_t(b) * sizeof(Q8), sizeof(q));
            int dot = 0;
            for (int i = 0; i < 32; ++i) {
                const int code = (block[2 + i % 16] >> (4 * (i / 16))) & 15;
                dot += code * q.q[i];
                bound += std::fabs(half_value(wd) * (code - 8) * half_value(q.d) * q.q[i]);
            }
            // Q4_0's affine correction consumes Q8_1's FP16 sum of the
            // original float input, rather than d * sum(quantized codes).
            // SmallTraits<Q40Block,4> applies half this correction in each
            // of its two dot fragments; the full-block scalar formula is:
            expected += half_value(wd) * (half_value(q.d) * dot - 8.0 * half_value(q.sum));
        }
        float got; std::memcpy(&got, b.data() + (size_t(j) * out + r) * 4, 4);
        const double error = std::fabs(double(got) - expected);
        if (error > 2e-6 * std::max(1.0, bound))
            std::fprintf(stderr, "scalar j=%d row=%d expected=%.12g got=%.12g error=%.12g bound=%.12g\n",
                         j, r, expected, double(got), error, bound);
        if (error > 2e-6 * std::max(1.0, bound)) {
            for (int bi = 0; bi < 2; ++bi) {
                uint16_t wd; std::memcpy(&wd, w.data() + size_t(j) * wb + (size_t(r) * (n / 32) + bi) * 18, 2);
                Q8 q; std::memcpy(&q, qbytes.data() + size_t(token(j)) * qb + size_t(bi) * sizeof(Q8), sizeof(q));
                std::fprintf(stderr, "block=%d wd=%04x d4=%.9g d8=%04x/%.9g sum=%04x/%.9g q0=%d\n",
                             bi, unsigned(wd), half_value(wd), unsigned(q.d), half_value(q.d), unsigned(q.sum), half_value(q.sum), int(q.q[0]));
            }
        }
        CHECK(error <= 2e-6 * std::max(1.0, bound));
        maximum_scalar_error = std::max(maximum_scalar_error, error);
    }
    for (const Device* d : {&dw, &dx, &dq, &reference, &actual}) d->check_guards();
    ++cases;
}

struct Pipeline {
    Device gate_up, h, q, down, sum;
    Pipeline(int n, int ff) : gate_up(size_t(kCapacity) * 2 * ff * 4), h(size_t(kCapacity) * ff * 4),
        q(size_t(kCapacity) * K::gemma_native_q8_1_bytes(ff)), down(size_t(kCapacity) * n * 4), sum(size_t(kTokens) * n * 4) {}
    void reset(const std::vector<float>& initial, cudaStream_t s) {
        gate_up.poison(s); h.poison(s); q.poison(s); down.poison(s); sum.put(initial.data(), sum.bytes, s);
    }
    void guards() const { for (const Device* d : {&gate_up, &h, &q, &down, &sum}) d->check_guards(); }
};
struct Snapshot { std::vector<uint8_t> gate_up, h, q, down, sum; };
static Snapshot snapshot(const Pipeline& p) { return {p.gate_up.get(), p.h.get(), p.q.get(), p.down.get(), p.sum.get()}; }

static void pipeline_case(int n, int ff, const std::vector<int>& counts, cudaStream_t s) {
    const size_t gu_bytes = K::gemma_native_mmvq_weight_bytes(kType, n, 2 * ff);
    const size_t down_bytes = K::gemma_native_mmvq_weight_bytes(kType, ff, n);
    const size_t xq_bytes = K::gemma_native_q8_1_bytes(n), hq_bytes = K::gemma_native_q8_1_bytes(ff);
    auto gu = weights(n, 2 * ff, kCapacity), dw = weights(ff, n, kCapacity); auto x = inputs(n);
    Device gate_weights(gu.size()), down_weights(dw.size()), dx(x.size() * 4), xq(kTokens * xq_bytes);
    gate_weights.put(gu.data(), gu.size(), s); down_weights.put(dw.data(), dw.size(), s); dx.put(x.data(), dx.bytes, s);
    K::gemma_native_quantize_q8_1(dx.as<float>(), xq.data(), n, kTokens, s);
    std::vector<float> initial(size_t(kTokens) * n);
    for (size_t i = 0; i < initial.size(); ++i) initial[i] = float(int(i % 17) - 8) / 1024.0f;
    Pipeline p(n, ff);
    for (int count : counts) {
        CHECK(count <= kCapacity);
        p.reset(initial, s);
        for (int j = 0; j < count; ++j) {
            float* gate = p.gate_up.as<float>() + size_t(j) * 2 * ff;
            K::gemma_native_mmvq(kType, gate_weights.as<uint8_t>() + size_t(j) * gu_bytes,
                xq.as<uint8_t>() + size_t(token(j)) * xq_bytes, gate, n, 2 * ff, 1, s);
            G::geglu(gate, gate + ff, p.h.as<float>() + size_t(j) * ff, ff, s);
            K::gemma_native_quantize_q8_1(p.h.as<float>() + size_t(j) * ff, p.q.as<uint8_t>() + size_t(j) * hq_bytes, ff, 1, s);
            K::gemma_native_mmvq(kType, down_weights.as<uint8_t>() + size_t(j) * down_bytes,
                p.q.as<uint8_t>() + size_t(j) * hq_bytes, p.down.as<float>() + size_t(j) * n, ff, n, 1, s);
            G::axpy(p.down.as<float>() + size_t(j) * n, coefficient(j), p.sum.as<float>() + size_t(token(j)) * n, n, true, s);
        }
        CK(cudaStreamSynchronize(s)); const auto expected = snapshot(p); p.guards();
        for (int split : {8, kCapacity}) {
            p.reset(initial, s);
            G::ScaledRows sum{};
            for (int at = 0; at < count; at += split) {
                const int m = std::min(split, count - at);
                K::MmvqSlots gate_up{}, down{};
                for (int j = 0; j < m; ++j) {
                    const int k = at + j;
                    gate_up.w[j] = gate_weights.as<uint8_t>() + size_t(k) * gu_bytes;
                    gate_up.x[j] = xq.as<uint8_t>() + size_t(token(k)) * xq_bytes;
                    gate_up.y[j] = p.gate_up.as<float>() + size_t(k) * 2 * ff;
                    down.w[j] = down_weights.as<uint8_t>() + size_t(k) * down_bytes;
                    down.x[j] = p.q.as<uint8_t>() + size_t(k) * hq_bytes;
                    down.y[j] = p.down.as<float>() + size_t(k) * n;
                    sum.x[k] = down.y[j]; sum.acc[k] = p.sum.as<float>() + size_t(token(k)) * n; sum.w[k] = coefficient(k);
                }
                CHECK(K::gemma_native_mmvq_slots(kType, gate_up, m, n, 2 * ff, s));
                G::geglu_rows(p.gate_up.as<float>() + size_t(at) * 2 * ff, p.h.as<float>() + size_t(at) * ff, ff, m, s);
                K::gemma_native_quantize_q8_1(p.h.as<float>() + size_t(at) * ff, p.q.as<uint8_t>() + size_t(at) * hq_bytes, ff, m, s);
                CHECK(K::gemma_native_mmvq_slots(kType, down, m, ff, n, s));
            }
            // Flush in expert order once, as Engine::moe does for its 32-entry sum.
            G::add_scaled(sum, count, n, s);
            CK(cudaStreamSynchronize(s)); const auto got = snapshot(p);
            equal_bytes(expected.gate_up, got.gate_up, size_t(count) * 2 * ff * 4, true, "gate/up", count, split);
            equal_bytes(expected.h, got.h, size_t(count) * ff * 4, true, "GEGLU", count, split);
            equal_bytes(expected.q, got.q, size_t(count) * hq_bytes, false, "Q8_1 bytes", count, split);
            equal_bytes(expected.down, got.down, size_t(count) * n * 4, true, "down", count, split);
            equal_bytes(expected.sum, got.sum, got.sum.size(), true, "ordered add", count, split);
            p.guards(); ++cases;
        }
    }
    for (const Device* d : {&gate_weights, &down_weights, &dx, &xq}) d->check_guards();
}

int main() {
    int devices = 0; CK(cudaGetDeviceCount(&devices)); if (!devices) return 77;
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    K::gemma_native_mmvq_set_multi_exact(true);
    // A compact shape covers non-256 tails; actual Gemma dimensions cover the
    // 2816 -> 2*704 gate/up and 704 -> 2816 down reduction layouts.
    pipeline_case(288, 96, {7, kCapacity}, s);
    pipeline_case(2816, 704, {1, 7, 8, 9, 17, 31, 32}, s);
    std::printf("PASS pipeline cases=%u\n", cases); std::fflush(stdout);
    matrix_case(s);
    CK(cudaStreamDestroy(s));
    std::printf("PASS Gemma cached slots cases=%u capacity=%d bitwise singleton/split parity; Q8 bytes, guards, finite, ordered sums; scalar_max_error=%.9g hash=%016llx\n",
        cases, kCapacity, maximum_scalar_error, (unsigned long long)checksum);
    return 0;
}
#endif
