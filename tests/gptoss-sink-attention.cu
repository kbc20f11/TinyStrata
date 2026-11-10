// Model-free full-attention INT8 D64/GQA8 regression. The split path deliberately changes reduction order;
// compare both paths with a double oracle, not with an unqualified bitwise expectation.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/gemma4/sink_attention.hpp"
#include "strata/gemma4/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace G = strata::gemma4;
#define CK(x) do { const auto check_status = (x); if (check_status != cudaSuccess) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, cudaGetErrorString(check_status)); std::exit(2); } } while (0)
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
constexpr int D = 64, GQA = 8, SPLIT = 512, STRIDE = D + 2;
static unsigned cases = 0;
static double worst_error = 0., worst_rms = 0.;
static uint32_t state = 0xD64A771Bu;
static uint32_t word() { state ^= state << 13; state ^= state >> 17; return state ^= state << 5; }
// Exercise all mantissa bits; short dyadic fixtures can hide contraction/rounding differences.
static float real() { return float(double(word()) / 4294967296.0 * 2.0 - 1.0); }
static uint16_t half_bits(float x) { const __half h = __float2half_rn(x); uint16_t b; std::memcpy(&b, &h, 2); return b; }
static float half_value(uint16_t b) { __half h; std::memcpy(&h, &b, 2); return __half2float(h); }

struct Device {
    uint8_t* p = nullptr;
    size_t bytes;
    cudaStream_t stream;
    Device(size_t n, cudaStream_t s) : bytes(n), stream(s) {
        CK(cudaMalloc(reinterpret_cast<void**>(&p), bytes + 128));
        CK(cudaMemsetAsync(p, 0xA7, bytes + 128, stream));
    }
    ~Device() { if (p) (void)cudaFree(p); }
    template<class T> T* data() const { return reinterpret_cast<T*>(p + 64); }
    template<class T> void put(const std::vector<T>& v) {
        CHECK(v.size() * sizeof(T) == bytes);
        CK(cudaMemcpyAsync(p + 64, v.data(), bytes, cudaMemcpyHostToDevice, stream));
    }
    void poison() { CK(cudaMemsetAsync(p + 64, 0xFF, bytes, stream)); }
    std::vector<uint8_t> get() const {
        CK(cudaStreamSynchronize(stream));
        std::vector<uint8_t> all(bytes + 128);
        CK(cudaMemcpy(all.data(), p, all.size(), cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < 64; ++j) CHECK(all[j] == 0xA7 && all[bytes + 64 + j] == 0xA7);
        return {all.begin() + 64, all.end() - 64};
    }
    std::vector<float> floats() const {
        const auto bytes_out = get();
        CHECK(bytes_out.size() % 4 == 0);
        std::vector<float> out(bytes_out.size() / 4);
        std::memcpy(out.data(), bytes_out.data(), bytes_out.size());
        return out;
    }
};

struct Fixture {
    int kv, heads, capacity;
    cudaStream_t stream;
    size_t part_count;
    std::vector<float> q, sinks;
    std::vector<int8_t> kc, vc;
    std::vector<uint16_t> ks, vs;
    Device dq, dkc, dvc, dks, dvs, dsinks, output, reference, part, score, at;
    long long position = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    Fixture(int kv_heads, int max_keys, cudaStream_t s)
        : kv(kv_heads), heads(kv_heads * GQA), capacity(max_keys), stream(s),
          part_count(G::sink_attention_workspace_floats(heads, capacity)), q(size_t(heads) * D), sinks(heads),
          kc(size_t(capacity) * kv * D), vc(kc.size()), ks(kc.size() / 32), vs(ks.size()),
          dq(q.size() * 4, s), dkc(kc.size(), s), dvc(vc.size(), s), dks(ks.size() * 2, s), dvs(vs.size() * 2, s),
          dsinks(sinks.size() * 4, s), output(q.size() * 4, s), reference(q.size() * 4, s),
          part(part_count * 4, s), score(size_t(heads) * capacity * 4, s), at(sizeof(long long), s) {}
    ~Fixture() { destroy_graph(); }
    void destroy_graph() {
        if (executable) CK(cudaGraphExecDestroy(executable));
        if (graph) CK(cudaGraphDestroy(graph));
        executable = nullptr; graph = nullptr;
    }
    void upload() {
        dq.put(q); dkc.put(kc); dvc.put(vc); dks.put(ks); dvs.put(vs); dsinks.put(sinks);
        CK(cudaStreamSynchronize(stream));
    }
    void analytic() {
        for (auto& x : q) x = real();
        std::fill(kc.begin(), kc.end(), int8_t(0));
        std::fill(vc.begin(), vc.end(), int8_t(1));
        std::fill(ks.begin(), ks.end(), half_bits(1.f));
        std::fill(vs.begin(), vs.end(), half_bits(1.f));
        upload();
    }
    void random() {
        for (auto& x : q) x = real();
        for (auto& x : kc) x = int8_t(int(word() % 255) - 127);
        for (auto& x : vc) x = int8_t(int(word() % 255) - 127);
        for (auto& x : ks) x = half_bits(0.001f + (real() + 1.f) * 0.01f);
        for (auto& x : vs) x = half_bits(0.001f + (real() + 1.f) * 0.01f);
        for (auto& x : sinks) x = real() * 4.f;
        upload();
    }
    bool launch(int keys, bool dynamic, size_t available = std::numeric_limits<size_t>::max()) {
        return G::sink_attention_q8(dq.data<float>(), dkc.data<int8_t>(), dks.data<uint16_t>(),
            dvc.data<int8_t>(), dvs.data<uint16_t>(), dsinks.data<float>(), output.data<float>(),
            part.data<float>(), std::min(part_count, available), heads, kv, keys - 1,
            dynamic ? at.data<long long>() : nullptr, dynamic ? capacity : 0, 0.125f, stream);
    }
    void capture() {
        destroy_graph();
        CK(cudaStreamSynchronize(stream));
        CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        CHECK(launch(1, true));
        CK(cudaStreamEndCapture(stream, &graph));
        CK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    }
    void run(int keys, bool dynamic) {
        CHECK(keys >= 1 && keys <= capacity);
        position = keys - 1;
        CK(cudaMemcpyAsync(at.data<long long>(), &position, sizeof(position), cudaMemcpyHostToDevice, stream));
        part.poison(); output.poison();
        if (dynamic) { CHECK(executable); CK(cudaGraphLaunch(executable, stream)); }
        else CHECK(launch(keys, false));
        CK(cudaStreamSynchronize(stream));
        // Dynamic graphs launch for the room, but unused split slots must remain untouched poison.
        const auto partials = part.floats();
        const int blocks = ((dynamic ? capacity : keys) + SPLIT - 1) / SPLIT;
        const int used = (keys + SPLIT - 1) / SPLIT;
        for (int h = 0; h < heads; ++h)
            for (int b = 0; b < blocks; ++b)
                for (int d = 0; d < STRIDE; ++d)
                    CHECK(b < used ? std::isfinite(partials[(size_t(h) * blocks + b) * STRIDE + d])
                                   : std::isnan(partials[(size_t(h) * blocks + b) * STRIDE + d]));
        // Check all input/output red zones as well as scratch.
        (void)dq.get(); (void)dkc.get(); (void)dvc.get(); (void)dks.get(); (void)dvs.get();
        (void)dsinks.get(); (void)at.get(); (void)score.get(); (void)reference.get();
    }
    std::vector<double> oracle(int keys) const {
        std::vector<double> expected(size_t(heads) * D), logits(keys), weights(keys);
        for (int h = 0; h < heads; ++h) {
            double top = sinks[h];
            for (int j = 0; j < keys; ++j) {
                const size_t row = size_t(j) * kv + h / GQA;
                double dot = 0.;
                for (int d = 0; d < D; ++d) {
                    const float value = float(kc[row * D + d]) * half_value(ks[row * 2 + d / 32]);
                    dot += double(q[size_t(h) * D + d]) * value;
                }
                logits[j] = dot * 0.125;
                top = std::max(top, logits[j]);
            }
            double total = std::exp(double(sinks[h]) - top);
            for (int j = 0; j < keys; ++j) { weights[j] = std::exp(logits[j] - top); total += weights[j]; }
            for (int d = 0; d < D; ++d) {
                double sum = 0.;
                for (int j = 0; j < keys; ++j) {
                    const size_t row = size_t(j) * kv + h / GQA;
                    const float value = float(vc[row * D + d]) * half_value(vs[row * 2 + d / 32]);
                    sum += weights[j] * value;
                }
                expected[size_t(h) * D + d] = sum / total;
            }
        }
        return expected;
    }
    void old_path(int keys) {
        G::attend(dq.data<float>(), nullptr, nullptr, keys - 1, 1, score.data<float>(), 1, reference.data<float>(),
                  D, heads, kv, 0.125f, stream, 0, 0, false, nullptr, 0, 0,
                  G::Kvq8{dkc.data<int8_t>(), dks.data<uint16_t>()},
                  G::Kvq8{dvc.data<int8_t>(), dvs.data<uint16_t>()}, dsinks.data<float>());
    }
};

static void near(const std::vector<float>& got, const std::vector<double>& expected, const char* tag, int keys) {
    CHECK(got.size() == expected.size());
    double worst = 0., squared = 0.;
    for (size_t j = 0; j < got.size(); ++j) {
        const double error = std::abs(double(got[j]) - expected[j]);
        worst = std::max(worst, error); squared += error * error;
        if (!std::isfinite(got[j]) || error > 0.00003 + 0.0003 * std::abs(expected[j])) {
            std::fprintf(stderr, "FAIL %s keys=%d index=%zu got=%.9g expected=%.17g error=%.9g\n",
                         tag, keys, j, got[j], expected[j], error);
            std::exit(1);
        }
    }
    worst_error = std::max(worst_error, worst);
    worst_rms = std::max(worst_rms, std::sqrt(squared / got.size()));
    ++cases;
}


static void multi_case(int rows, int position, bool graph, cudaStream_t stream) {
    Fixture f(2, 32768, stream);
    f.random();
    // Future slots are deliberately non-finite; even partial tail tiles must not consume them.
    const size_t first_future = size_t(position + rows) * f.kv * 2;
    std::fill(f.ks.begin() + first_future, f.ks.end(), uint16_t(0x7fff));
    std::fill(f.vs.begin() + first_future, f.vs.end(), uint16_t(0x7fff));
    f.dks.put(f.ks); f.dvs.put(f.vs);
    std::vector<float> queries(size_t(rows) * f.heads * D);
    for (auto& x : queries) x = real();
    std::vector<long long> positions(size_t(rows) * 2);
    for (int r = 0; r < rows; ++r) { positions[2 * r] = position + r; positions[2 * r + 1] = -991; }
    const size_t scratch = G::sink_attention_workspace_floats(f.heads, f.capacity, rows);
    Device dq(queries.size() * 4, stream), all(queries.size() * 4, stream), one(queries.size() * 4, stream);
    Device parts(scratch * 4, stream), at(positions.size() * sizeof(long long), stream);
    dq.put(queries); at.put(positions); parts.poison(); all.poison(); one.poison();
    CK(cudaStreamSynchronize(stream));
    auto work = [&] {
        CHECK(G::sink_attention_q8(dq.data<float>(), f.dkc.data<int8_t>(), f.dks.data<uint16_t>(),
            f.dvc.data<int8_t>(), f.dvs.data<uint16_t>(), f.dsinks.data<float>(), all.data<float>(),
            parts.data<float>(), scratch, f.heads, f.kv, position, graph ? at.data<long long>() : nullptr,
            graph ? f.capacity : 0, 0.125f, stream, rows));
        for (int r = 0; r < rows; ++r)
            CHECK(G::sink_attention_q8(dq.data<float>() + size_t(r) * f.heads * D,
                f.dkc.data<int8_t>(), f.dks.data<uint16_t>(), f.dvc.data<int8_t>(), f.dvs.data<uint16_t>(),
                f.dsinks.data<float>(), one.data<float>() + size_t(r) * f.heads * D,
                f.part.data<float>(), f.part_count, f.heads, f.kv, position + r,
                graph ? at.data<long long>() + 2 * r : nullptr, graph ? f.capacity : 0, 0.125f, stream));
    };
    if (graph) {
        cudaGraph_t captured = nullptr; cudaGraphExec_t executable = nullptr;
        CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); work();
        CK(cudaStreamEndCapture(stream, &captured));
        CK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        CK(cudaGraphLaunch(executable, stream)); CK(cudaStreamSynchronize(stream));
        CK(cudaGraphExecDestroy(executable)); CK(cudaGraphDestroy(captured));
    } else work();
    const auto a = all.floats(), b = one.floats();
    for (size_t j = 0; j < a.size(); ++j)
        if (!std::isfinite(a[j]) || std::memcmp(a.data() + j, b.data() + j, 4)) {
            std::fprintf(stderr, "FAIL multirow exact rows=%d pos=%d graph=%d index=%zu multi=%.9g one=%.9g\n",
                         rows, position, int(graph), j, a[j], b[j]); std::exit(1);
        }
    const auto partials = parts.floats();
    const int blocks = ((graph ? f.capacity : position + rows) + SPLIT - 1) / SPLIT;
    for (int r = 0; r < rows; ++r) {
        const int used = (position + r + 1 + SPLIT - 1) / SPLIT;
        for (int h = 0; h < f.heads; ++h)
            for (int b0 = 0; b0 < blocks; ++b0)
                for (int d = 0; d < STRIDE; ++d) {
                    const float x = partials[((size_t(r) * f.heads + h) * blocks + b0) * STRIDE + d];
                    CHECK(b0 < used ? std::isfinite(x) : std::isnan(x));
                }
    }
    (void)dq.get(); (void)at.get(); (void)f.dkc.get(); (void)f.dvc.get(); (void)f.dks.get(); (void)f.dvs.get();
    ++cases;
}

int main() {
    cudaStream_t stream = nullptr;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        Fixture f(8, 16385, stream); // actual model's 64 query heads / 8 KV heads, non-multiple graph room
        CHECK(G::sink_attention_workspace_floats(0, 1) == 0);
        CHECK(G::sink_attention_workspace_floats(64, 0) == 0);
        CHECK(!f.launch(2048, false, 0));
        CHECK(!G::sink_attention_q8(f.dq.data<float>(), f.dkc.data<int8_t>(), f.dks.data<uint16_t>(),
            f.dvc.data<int8_t>(), f.dvs.data<uint16_t>(), f.dsinks.data<float>(), f.output.data<float>(),
            f.part.data<float>(), f.part_count, 63, 8, 1, nullptr, 0, 0.125f, stream));
        f.analytic(); f.capture();
        const int boundaries[] = {1, 2, 31, 32, 63, 64, 65, 511, 512, 513, 1023, 1024, 1025,
                                  2047, 2048, 2049, 4095, 4096, 4097, 16384, 16385};
        for (int keys : boundaries) {
            // A sink of log(keys) makes exactly one sink contribution comparable to all keys combined.
            std::fill(f.sinks.begin(), f.sinks.end(), std::log(float(keys)));
            f.dsinks.put(f.sinks);
            const double expected = keys / (keys + std::exp(double(f.sinks[0])));
            for (bool graph : {false, true}) {
                f.run(keys, graph);
                near(f.output.floats(), std::vector<double>(f.q.size(), expected), "sink-once", keys);
            }
        }
        f.destroy_graph(); f.random(); f.capture();
        for (int keys : {1, 63, 65, 511, 513, 2049, 4097, 16385}) {
            const auto expected = f.oracle(keys);
            for (bool graph : {false, true}) {
                f.run(keys, graph);
                near(f.output.floats(), expected, graph ? "split graph" : "split", keys);
            }
            f.old_path(keys);
            near(f.reference.floats(), expected, "existing attention", keys);
        }
    }
    {
        Fixture f(8, 131072, stream);
        f.analytic(); f.capture();
        // Include both sides of split boundaries and shrink a graph's actual key count after full capacity.
        const int keys_list[] = {32767, 32768, 32769, 65535, 65536, 65537, 131071, 131072, 65536, 32768};
        for (int keys : keys_list) {
            std::fill(f.sinks.begin(), f.sinks.end(), std::log(float(keys)));
            f.dsinks.put(f.sinks);
            const double expected = keys / (keys + std::exp(double(f.sinks[0])));
            for (bool graph : {false, true}) {
                f.run(keys, graph);
                near(f.output.floats(), std::vector<double>(f.q.size(), expected), "long sink-once", keys);
            }
        }
    }
    for (int rows : {2, 3, 4, 8})
        for (int position : {510, 511, 512, 8188, 8191, 8192, 8193, 16380})
            for (bool graph : {false, true}) multi_case(rows, position, graph, stream);
    CK(cudaStreamDestroy(stream));
    std::printf("PASS %u checks max_abs=%.9g max_rms=%.9g (atol3e-5 + rtol3e-4)\n", cases, worst_error, worst_rms);
    return 0;
}
#endif
