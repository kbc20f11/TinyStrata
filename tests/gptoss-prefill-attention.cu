// Model-free INT8/sink prefill regression. Both paths round Q, restored KV and P to FP16;
// compare with independent CPU references instead of demanding equal reduction order.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/gemma4/kernels.hpp"
#include "strata/gemma4/prefill_attention.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace G = strata::gemma4;
namespace P = strata::gemma4::prefill_attention;
#define CK(x) do { const auto e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "line %d: %s\n", __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
constexpr int D = 64, GROUP = 8;
static unsigned cases = 0;
static double max_error = 0, max_rms = 0, baseline_error = 0, tile_error = 0;
static uint32_t rng = 0x374BB211u;
static uint32_t word() { rng ^= rng << 13; rng ^= rng >> 17; return rng ^= rng << 5; }
static float real() { return float(double(word()) / 4294967296.0 * 2.0 - 1.0); }
static uint16_t hb(float x) { const __half h = __float2half_rn(x); uint16_t b; std::memcpy(&b, &h, 2); return b; }
static float hv(uint16_t b) { __half h; std::memcpy(&h, &b, 2); return __half2float(h); }
static float rounded(float x) { return hv(hb(x)); }

struct Device {
    uint8_t* p = nullptr; size_t n; cudaStream_t stream;
    Device(size_t bytes, cudaStream_t s) : n(bytes), stream(s) {
        CK(cudaMalloc(reinterpret_cast<void**>(&p), n + 128));
        CK(cudaMemsetAsync(p, 0xA7, n + 128, stream));
    }
    ~Device() { if (p) (void) cudaFree(p); }
    template<class T> T* data() const { return reinterpret_cast<T*>(p + 64); }
    template<class T> void put(const std::vector<T>& v) {
        CHECK(v.size() * sizeof(T) == n);
        CK(cudaMemcpyAsync(p + 64, v.data(), n, cudaMemcpyHostToDevice, stream));
    }
    void poison() { CK(cudaMemsetAsync(p + 64, 0xFF, n, stream)); }
    std::vector<uint8_t> bytes() const {
        CK(cudaStreamSynchronize(stream));
        std::vector<uint8_t> result(n + 128);
        CK(cudaMemcpy(result.data(), p, result.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < 64; ++i) CHECK(result[i] == 0xA7 && result[n + 64 + i] == 0xA7);
        return {result.begin() + 64, result.end() - 64};
    }
    std::vector<float> floats() const {
        const auto b = bytes(); std::vector<float> result(n / 4);
        std::memcpy(result.data(), b.data(), n); return result;
    }
    template<class T> void unchanged(const std::vector<T>& v) const {
        const auto b = bytes(); CHECK(b.size() == v.size() * sizeof(T));
        CHECK(std::memcmp(b.data(), v.data(), b.size()) == 0);
    }
};

struct Fixture {
    int rows, pos, kv, heads, padded, keys, window, ring; cudaStream_t stream; void* blas;
    size_t scores, halves;
    std::vector<float> q, sinks;
    std::vector<int8_t> kc, vc;
    std::vector<uint16_t> ks, vs;
    Device dq, dk, dv, dks, dvs, ds, out, old, score, work, kw, vw;
    Fixture(int r, int p, int k, cudaStream_t st, int w = 0, int rg = 0)
        : rows(r), pos(p), kv(k), heads(k * GROUP), padded((r + 63) / 64 * 64 + 1), keys(rg > 0 ? rg : p + r + 33),
          window(w), ring(rg), stream(st),
          blas(G::blas_create(st)), scores(std::min(size_t(r) * heads * 512, size_t(16) << 20)), halves(size_t(k) * D * 1024),
          q(size_t(padded) * heads * D, std::numeric_limits<float>::quiet_NaN()), sinks(heads),
          kc(size_t(keys) * kv * D), vc(kc.size()), ks(kc.size() / 32), vs(ks.size()),
          dq(q.size() * 4, st), dk(kc.size(), st), dv(vc.size(), st), dks(ks.size() * 2, st), dvs(vs.size() * 2, st),
          ds(sinks.size() * 4, st), out(q.size() * 4, st), old(q.size() * 4, st), score(scores * 4, st),
          work((size_t(r) * heads * (D + D / 2 + 3) + 4) * 4, st), kw(halves * 2, st), vw(halves * 2, st) {}
    ~Fixture() { G::blas_destroy(blas); }
    void init(bool analytic, bool large = false) {
        for (size_t j = 0; j < size_t(rows) * heads * D; ++j) q[j] = analytic ? 0.f : real() * (large ? 16.f : 1.f);
        for (auto& x : kc) x = analytic ? int8_t(0) : int8_t(int(word() % 256) - 128);
        for (auto& x : vc) x = analytic ? int8_t(1) : int8_t(int(word() % 256) - 128);
        for (auto& x : ks) x = hb(analytic ? 1.f : 0.002f + (real() + 1.f) * (large ? 0.05f : 0.01f));
        for (auto& x : vs) x = hb(analytic ? 1.f : 0.002f + (real() + 1.f) * 0.01f);
        for (int h = 0; h < heads; ++h) sinks[h] = analytic ? std::log(float(pos + rows)) + (h % 3 - 1) * 12.f : real() * 8.f;
        // Last tile and query padding are not readable keys/queries. NaNs expose unmasked loads.
        if (window == 0 && ring == 0) {
            std::fill(ks.begin() + size_t(pos + rows) * kv * 2, ks.end(), uint16_t(0x7fff));
            std::fill(vs.begin() + size_t(pos + rows) * kv * 2, vs.end(), uint16_t(0x7fff));
        } else {
            // Poison every physical row outside the batch union, including ring holes.
            std::vector<bool> visible(keys, false);
            for (int j = window > 0 ? std::max(0,pos-window+1) : 0; j < pos+rows; ++j)
                visible[physical(j)] = true;
            for (int j = 0; j < keys; ++j) if (!visible[j]) for (int h = 0; h < kv; ++h)
                for (int d = 0; d < 2; ++d) ks[(size_t(j)*kv+h)*2+d] = vs[(size_t(j)*kv+h)*2+d] = 0x7fff;
        }
        upload();
    }
    void upload() { dq.put(q); dk.put(kc); dv.put(vc); dks.put(ks); dvs.put(vs); ds.put(sinks); }
    bool launch(int r = -1, int64_t p = -2, int h = -1) {
        return P::run(dq.data<float>(), dk.data<int8_t>(), dks.data<uint16_t>(), dv.data<int8_t>(), dvs.data<uint16_t>(),
                      ds.data<float>(), out.data<float>(), h < 0 ? heads : h, kv, p == -2 ? pos : p,
                      r < 0 ? rows : r, 0.125f, stream, window, ring);
    }
    void baseline() {
        G::attend_batch(blas, dq.data<float>(), nullptr, nullptr, pos, rows, score.data<float>(), scores,
                        work.data<float>(), old.data<float>(), D, heads, kv, 0.125f, stream, window, ring, false,
                        G::Kvq8{dk.data<int8_t>(), dks.data<uint16_t>()}, G::Kvq8{dv.data<int8_t>(), dvs.data<uint16_t>()},
                        kw.data<uint16_t>(), vw.data<uint16_t>(), halves, ds.data<float>());
    }
    int physical(int key) const { return ring > 0 ? key % ring : key; }
    float restored(bool value, int key, int kh, int d) const {
        const size_t j = size_t(physical(key)) * kv + kh;
        return rounded(float((value ? vc : kc)[j * D + d]) * hv((value ? vs : ks)[j * 2 + d / 32]));
    }
    // Exact FP64 softmax over the same rounded Q/KV, and a separately coded tiled online FP32/half-P oracle.
    void oracle(int r, int h, std::vector<double>& exact, std::vector<float>& tiled) const {
        const int count = pos + r + 1, lower = window > 0 ? std::max(0,count-window) : 0, kh = h / GROUP;
        std::vector<double> a(count); std::vector<float> af(count);
        double top = sinks[h]; float running = -INFINITY, mass = 0;
        exact.assign(D, 0); tiled.assign(D, 0);
        for (int j = lower; j < count; ++j) {
            double sum = 0; float sf = 0;
            for (int d = 0; d < D; ++d) {
                const float x = rounded(q[(size_t(r) * heads + h) * D + d]), k = restored(false, j, kh, d);
                sum += double(x) * k; sf = std::fma(x, k, sf);
            }
            a[j] = sum * 0.125; af[j] = sf * 0.125f; top = std::max(top, a[j]);
        }
        double total = std::exp(double(sinks[h]) - top);
        for (int j = lower; j < count; ++j) {
            const double w = std::exp(a[j] - top); total += w;
            for (int d = 0; d < D; ++d) exact[d] += w * restored(true, j, kh, d);
        }
        for (auto& x : exact) x /= total;
        // The CTA begins at the first query's window lower bound. Individual rows
        // mask older keys, including any initial tiles with no keys for this row.
        const int tile_first = window > 0 ? std::max(0,pos+(r/P::kQueryTile)*P::kQueryTile-window+1) : 0;
        for (int first = tile_first; first < count; first += P::kKeyTile) {
            const int end = std::min(first + P::kKeyTile, count);
            const int begin = std::max(first, lower);
            float now = running;
            for (int j = begin; j < end; ++j) now = std::max(now, af[j]);
            const float alpha = running == -INFINITY ? 0.f : std::exp(running - now);
            mass *= alpha; for (auto& x : tiled) x *= alpha;
            for (int j = begin; j < end; ++j) {
                const float w = rounded(std::exp(af[j] - now)); mass += w;
                for (int d = 0; d < D; ++d) tiled[d] = std::fma(w, restored(true, j, kh, d), tiled[d]);
            }
            running = now;
        }
        const float final_top = std::max(running, sinks[h]), factor = std::exp(running - final_top);
        const float denominator = mass * factor + std::exp(sinks[h] - final_top);
        for (auto& x : tiled) x = (x * factor) / denominator;
    }
    void padding(const std::vector<float>& values) const {
        for (size_t j = 0; j < values.size(); ++j) CHECK(j < size_t(rows) * heads * D ? std::isfinite(values[j]) : std::isnan(values[j]));
    }
    void guards() const {
        dq.unchanged(q); dk.unchanged(kc); dv.unchanged(vc); dks.unchanged(ks); dvs.unchanged(vs); ds.unchanged(sinks);
        (void) score.bytes(); (void) work.bytes(); (void) kw.bytes(); (void) vw.bytes();
    }
    void check(bool analytic, bool graph) {
        out.poison(); old.poison(); CK(cudaStreamSynchronize(stream));
        if (graph) {
            cudaGraph_t g = nullptr; cudaGraphExec_t e = nullptr;
            CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); CHECK(launch());
            CK(cudaStreamEndCapture(stream, &g)); CK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
            CK(cudaGraphLaunch(e, stream)); CK(cudaStreamSynchronize(stream));
            const auto first = out.floats();
            // Replay with changed data, same pointers and scalar parameters.
            for (size_t j = 0; j < size_t(rows) * heads * D; ++j) q[j] *= -0.731f;
            dq.put(q); out.poison(); CK(cudaGraphLaunch(e, stream)); CK(cudaStreamSynchronize(stream));
            CK(cudaGraphExecDestroy(e)); CK(cudaGraphDestroy(g)); (void)first;
        } else CHECK(launch());
        baseline(); const auto got = out.floats(), before = old.floats(); padding(got); padding(before); guards();
        double squared = 0; size_t checked = 0;
        std::vector<int> rr{0, rows / 2, rows - 1}, hh{0, heads - 1};
        if (window > 0) for (int r : {1,15,16,31,32,63,64,127,128,129}) if (r < rows) rr.push_back(r);
        std::sort(rr.begin(), rr.end()); rr.erase(std::unique(rr.begin(), rr.end()), rr.end());
        for (int r : rr) for (int h : hh) {
            std::vector<double> exact; std::vector<float> tiled; oracle(r, h, exact, tiled);
            for (int d = 0; d < D; ++d) {
                const size_t j = (size_t(r) * heads + h) * D + d;
                const double error = std::abs(double(got[j]) - exact[d]);
                const double terr = std::abs(double(got[j]) - tiled[d]);
                const double berr = std::abs(double(before[j]) - exact[d]);
                // Probability half-rounding is intentional in both GPU paths. The tile oracle tests its recurrence separately.
                if (error > 4e-4 + 2e-3 * std::abs(exact[d]) || terr > 4e-5 + 3e-4 * std::abs(tiled[d]) ||
                    berr > 4e-4 + 2e-3 * std::abs(exact[d])) {
                    std::fprintf(stderr, "FAIL oracle rows=%d pos=%d kv=%d window=%d ring=%d r=%d h=%d d=%d new=%.9g old=%.9g fp64=%.17g tile=%.9g err=%.9g olderr=%.9g tileerr=%.9g\n",
                                 rows,pos,kv,window,ring,r,h,d,got[j],before[j],exact[d],tiled[d],error,berr,terr); std::exit(1);
                }
                if (analytic) {
                    const double count = window > 0 ? std::min(pos+r+1,window) : pos+r+1;
                    const double answer = count / (count + std::exp(double(sinks[h])));
                    CHECK(std::abs(got[j] - answer) < 4e-5);
                }
                max_error = std::max(max_error,error); baseline_error = std::max(baseline_error,berr);
                tile_error = std::max(tile_error,terr); squared += error * error; ++checked;
            }
        }
        max_rms = std::max(max_rms, std::sqrt(squared / checked)); ++cases;
    }
    void causal() {
        CHECK(rows >= 2); out.poison(); CHECK(launch()); const auto a = out.floats();
        const int cut = pos + rows / 2;
        for (int j = cut; j < pos + rows; ++j) for (int h = 0; h < kv; ++h) {
            const size_t row = size_t(physical(j))*kv+h;
            for (int d = 0; d < D; ++d) { kc[row*D+d] = (d&1)?127:-128; vc[row*D+d] = (d&1)?-128:127; }
            for (int d = 0; d < 2; ++d) { ks[row*2+d] = hb(0.25f); vs[row*2+d] = hb(0.125f); }
        }
        dk.put(kc); dv.put(vc); dks.put(ks); dvs.put(vs); out.poison(); CHECK(launch()); const auto b = out.floats();
        const size_t earlier = size_t(rows / 2) * heads * D;
        CHECK(std::memcmp(a.data(), b.data(), earlier * sizeof(float)) == 0);
        CHECK(std::memcmp(a.data()+earlier, b.data()+earlier, (size_t(rows)*heads*D-earlier)*4) != 0);
        padding(b); guards(); ++cases;
    }
    void expired() {
        CHECK(window > 0 && pos >= window && rows >= 2);
        out.poison(); CHECK(launch()); const auto a = out.floats();
        const int cut = rows/2, lower = pos+cut-window+1;
        for (int j = pos-window+1; j < lower; ++j) for (int h = 0; h < kv; ++h) {
            const size_t row = size_t(physical(j))*kv+h;
            for (int d = 0; d < D; ++d) { kc[row*D+d] = (d&1)?127:-128; vc[row*D+d] = (d&1)?-128:127; }
            for (int d = 0; d < 2; ++d) { ks[row*2+d] = hb(0.25f); vs[row*2+d] = hb(0.125f); }
        }
        dk.put(kc); dv.put(vc); dks.put(ks); dvs.put(vs); out.poison(); CHECK(launch()); const auto b = out.floats();
        const size_t earlier = size_t(cut)*heads*D, later = size_t(rows-cut)*heads*D;
        CHECK(std::memcmp(a.data(),b.data(),earlier*4) != 0);
        CHECK(std::memcmp(a.data()+earlier,b.data()+earlier,later*4) == 0);
        padding(b); guards(); ++cases;
    }
    void ring_equivalence() {
        CHECK(window > 0 && ring > 0);
        Fixture linear(rows,pos,kv,stream,window,0); linear.init(false);
        linear.q = q; linear.sinks = sinks;
        const int first = std::max(0,pos-window+1);
        // Deliberately restore logical rows in reverse order across the physical wrap.
        for (int key = pos+rows-1; key >= first; --key) for (int h = 0; h < kv; ++h) {
            const size_t src = size_t(physical(key))*kv+h, dst = size_t(key)*kv+h;
            std::copy_n(kc.data()+src*D,D,linear.kc.data()+dst*D);
            std::copy_n(vc.data()+src*D,D,linear.vc.data()+dst*D);
            std::copy_n(ks.data()+src*2,2,linear.ks.data()+dst*2);
            std::copy_n(vs.data()+src*2,2,linear.vs.data()+dst*2);
        }
        linear.upload(); out.poison(); linear.out.poison(); CHECK(launch()); CHECK(linear.launch());
        const auto a = out.floats(), b = linear.out.floats();
        CHECK(std::memcmp(a.data(),b.data(),size_t(rows)*heads*D*4) == 0);
        padding(a); linear.padding(b); guards(); linear.guards(); ++cases;
    }
    double timing(bool fused, int repeat) {
        cudaEvent_t start = nullptr, stop = nullptr; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
        for (int i = 0; i < 3; ++i) { if (fused) CHECK(launch()); else baseline(); }
        CK(cudaEventRecord(start,stream));
        for (int i = 0; i < repeat; ++i) { if (fused) CHECK(launch()); else baseline(); }
        CK(cudaEventRecord(stop,stream)); CK(cudaEventSynchronize(stop)); float ms=0; CK(cudaEventElapsedTime(&ms,start,stop));
        CK(cudaEventDestroy(stop)); CK(cudaEventDestroy(start)); return double(ms)*1000/repeat;
    }
};

int main(int argc, char** argv) {
    const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    const bool long_bench = argc == 2 && std::strcmp(argv[1], "--long-bench") == 0;
    CHECK(argc == 1 || bench || long_bench);
    if (!P::available()) { std::puts("SKIP: prefill attention unavailable on this backend/device"); return 77; }
    cudaStream_t st=nullptr; CK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    {
        Fixture f(33,0,1,st); f.init(false);
        CHECK(!f.launch(0)); CHECK(!f.launch(1,-1)); CHECK(!f.launch(1,0,7));
        CHECK(!P::run(nullptr,f.dk.data<int8_t>(),f.dks.data<uint16_t>(),f.dv.data<int8_t>(),f.dvs.data<uint16_t>(),
                     f.ds.data<float>(),f.out.data<float>(),8,1,0,1,0.125f,st));
        f.guards(); ++cases;
    }
    for (int rows : {1,7,8,9,31,32,33,63,64,65}) {
        Fixture f(rows,0,1,st); f.init(true); f.check(true,false); f.init(false); f.check(false,false);
        if (rows == 33) { f.causal(); f.init(false,true); f.check(false,false); }
    }
    for (int pos : {4096,16384}) {
        Fixture f(33,pos,2,st); f.init(false); f.check(false,pos==4096); f.causal();
    }
    { Fixture f(32,4096,8,st); f.init(false); f.check(false,false); }
    // Keep the original 28 full-attention checks above unchanged. SWA additionally
    // exercises a batch larger than its window, exact/minimal and padded rings,
    // lower-bound masking, future masking, and restored/reordered physical caches.
    {
        Fixture f(33,4096,1,st,128,161); f.init(false);
        auto invalid = [&](int window, int ring) {
            return P::run(f.dq.data<float>(),f.dk.data<int8_t>(),f.dks.data<uint16_t>(),f.dv.data<int8_t>(),f.dvs.data<uint16_t>(),
                          f.ds.data<float>(),f.out.data<float>(),8,1,f.pos,f.rows,0.125f,st,window,ring);
        };
        CHECK(!invalid(-1,161)); CHECK(!invalid(128,-1)); CHECK(!invalid(0,161)); CHECK(!invalid(128,159));
        f.guards(); ++cases;
    }
    for (int rows : {1,7,31,32,33,127,128,129,257}) {
        Fixture f(rows,0,1,st,128,128+rows); f.init(true); f.check(true,false);
        f.init(false); f.check(false,rows==129);
        if (rows == 33 || rows == 257) { f.causal(); f.ring_equivalence(); }
    }
    for (int pos : {127,128,129,259,4096,16384,131000}) {
        Fixture f(65,pos,2,st,128,193); f.init(false); f.check(false,pos==4096);
        f.ring_equivalence(); f.causal();
        if (pos >= 128) { f.init(false); f.expired(); }
    }
    { Fixture f(33,4096,1,st,128,160); f.init(false,true); f.check(false,false); f.ring_equivalence(); }
    { Fixture f(33,0,1,st,128,33); f.init(false); f.check(false,false); f.ring_equivalence(); }
    for (int window : {1,17}) {
        Fixture f(65,4096,1,st,window,window+65); f.init(false); f.check(false,true); f.ring_equivalence(); f.expired();
    }
    { Fixture f(1120,4096,1,st,128,1248); f.init(false); f.check(false,false); f.ring_equivalence(); }
    const std::vector<int> positions = long_bench ? std::vector<int>{32768,65536} : std::vector<int>{0,4096,8192,12288,16384};
    if (bench || long_bench) for (int pos : positions) {
        Fixture f(1120,pos,8,st); f.init(false); CHECK(f.launch()); f.baseline(); CK(cudaStreamSynchronize(st));
        // A/B/B/A balanced event timings; resident synthetic data is not end-to-end model throughput.
        const double a0=f.timing(false,3), b0=f.timing(true,3), b1=f.timing(true,3), a1=f.timing(false,3);
        std::printf("MICRO rows=1120 pos=%d keytile=%d old_us=%.6f new_us=%.6f A0=%.6f B0=%.6f B1=%.6f A1=%.6f\n",pos,P::kKeyTile,(a0+a1)/2,(b0+b1)/2,a0,b0,b1,a1);
    }
    CK(cudaStreamDestroy(st));
    std::printf("PASS %u checks max_abs=%.9g max_rms=%.9g baseline_max_abs=%.9g tile_max_abs=%.9g keytile=%d (FP16 Q/KV/P; selected FP64 row/head oracles)\n",cases,max_error,max_rms,baseline_error,tile_error,P::kKeyTile);
    return 0;
}
#endif
