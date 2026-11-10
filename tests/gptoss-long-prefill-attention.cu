// Independent long full-Attention candidate regression. Timings follow full-shape parity checks.
// Baseline/current/candidate compare stored FP16 Q/KV/P precision; no input re-quantization.
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
#include "strata/gemma4/long_prefill_attention.hpp"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace G = strata::gemma4;
namespace P = strata::gemma4::long_prefill_attention;
namespace Current = strata::gemma4::prefill_attention;
#define CK(x) do { const auto e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "line %d: %s\n", __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
constexpr int D = 64, GROUP = 8;
static unsigned cases = 0;
static double max_error = 0, max_rms = 0, baseline_error = 0, tile_error = 0, max_vs_blas = 0, max_vs_current = 0;
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
    int rows, pos, kv, heads, padded, keys; cudaStream_t stream; void* blas;
    size_t scores, halves;
    std::vector<float> q, sinks;
    std::vector<int8_t> kc, vc;
    std::vector<uint16_t> ks, vs;
    Device dq, dk, dv, dks, dvs, ds, out, old, score, work, kw, vw;
    Fixture(int r, int p, int k, cudaStream_t st)
        : rows(r), pos(p), kv(k), heads(k * GROUP), padded((r + 63) / 64 * 64 + 1), keys(p + r + 33), stream(st),
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
        std::fill(ks.begin() + size_t(pos + rows) * kv * 2, ks.end(), uint16_t(0x7fff));
        std::fill(vs.begin() + size_t(pos + rows) * kv * 2, vs.end(), uint16_t(0x7fff));
        upload();
    }
    void upload() { dq.put(q); dk.put(kc); dv.put(vc); dks.put(ks); dvs.put(vs); ds.put(sinks); }
    bool launch(int r = -1, int64_t p = -2, int h = -1) {
        return P::run(dq.data<float>(), dk.data<int8_t>(), dks.data<uint16_t>(), dv.data<int8_t>(), dvs.data<uint16_t>(),
                      ds.data<float>(), out.data<float>(), h < 0 ? heads : h, kv, p == -2 ? pos : p,
                      r < 0 ? rows : r, 0.125f, stream);
    }
    void baseline() {
        G::attend_batch(blas, dq.data<float>(), nullptr, nullptr, pos, rows, score.data<float>(), scores,
                        work.data<float>(), old.data<float>(), D, heads, kv, 0.125f, stream, 0, 0, false,
                        G::Kvq8{dk.data<int8_t>(), dks.data<uint16_t>()}, G::Kvq8{dv.data<int8_t>(), dvs.data<uint16_t>()},
                        kw.data<uint16_t>(), vw.data<uint16_t>(), halves, ds.data<float>());
    }
    void current() {
        CHECK(Current::run(dq.data<float>(),dk.data<int8_t>(),dks.data<uint16_t>(),dv.data<int8_t>(),dvs.data<uint16_t>(),
                           ds.data<float>(),old.data<float>(),heads,kv,pos,rows,0.125f,stream));
    }
    float restored(bool value, int key, int kh, int d) const {
        const size_t j = size_t(key) * kv + kh;
        return rounded(float((value ? vc : kc)[j * D + d]) * hv((value ? vs : ks)[j * 2 + d / 32]));
    }
    // Exact FP64 softmax over the same rounded Q/KV, and a separately coded tiled online FP32/half-P oracle.
    void oracle(int r, int h, std::vector<double>& exact, std::vector<float>& tiled) const {
        const int count = pos + r + 1, kh = h / GROUP;
        std::vector<double> a(count); std::vector<float> af(count);
        double top = sinks[h]; float running = -INFINITY, mass = 0;
        exact.assign(D, 0); tiled.assign(D, 0);
        for (int j = 0; j < count; ++j) {
            double sum = 0; float sf = 0;
            for (int d = 0; d < D; ++d) {
                const float x = rounded(q[(size_t(r) * heads + h) * D + d]), k = restored(false, j, kh, d);
                sum += double(x) * k; sf = std::fma(x, k, sf);
            }
            a[j] = sum * 0.125; af[j] = sf * 0.125f; top = std::max(top, a[j]);
        }
        double total = std::exp(double(sinks[h]) - top);
        for (int j = 0; j < count; ++j) {
            const double w = std::exp(a[j] - top); total += w;
            for (int d = 0; d < D; ++d) exact[d] += w * restored(true, j, kh, d);
        }
        for (auto& x : exact) x /= total;
        for (int first = 0; first < count; first += P::kKeyTile) {
            const int end = std::min(first + P::kKeyTile, count);
            float now = running;
            for (int j = first; j < end; ++j) now = std::max(now, af[j]);
            const float alpha = std::exp(running - now);
            mass *= alpha; for (auto& x : tiled) x *= alpha;
            for (int j = first; j < end; ++j) {
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
            const auto replayed=out.floats();
            CHECK(std::memcmp(first.data(),replayed.data(),size_t(rows)*heads*D*sizeof(float))!=0);
            CK(cudaGraphExecDestroy(e)); CK(cudaGraphDestroy(g));
        } else CHECK(launch());
        baseline(); const auto got = out.floats(), before = old.floats(); padding(got); padding(before);
        current(); const auto prior = old.floats(); padding(prior); guards();
        // Validate every output in the complete production-shaped matrix, not only the timed launch.
        // Selected independent FP64/tile oracles below additionally validate both BLAS and candidate.
        double baseline_max = 0, current_max = 0, baseline_sq = 0;
        for (size_t j = 0; j < size_t(rows)*heads*D; ++j) {
            const double be = std::abs(double(got[j])-before[j]), ce = std::abs(double(got[j])-prior[j]);
            if (be > 8e-4 + 4e-3*std::abs(double(before[j])) || ce > 8e-4 + 4e-3*std::abs(double(prior[j]))) {
                std::fprintf(stderr,"FAIL full matrix rows=%d pos=%d kv=%d at=%zu long=%.9g BLAS=%.9g current=%.9g\n",
                             rows,pos,kv,j,got[j],before[j],prior[j]); std::exit(1);
            }
            baseline_max=std::max(baseline_max,be);current_max=std::max(current_max,ce);baseline_sq+=be*be;
        }
        max_vs_blas=std::max(max_vs_blas,baseline_max);max_vs_current=std::max(max_vs_current,current_max);
        std::printf("PARITY rows=%d pos=%d kv=%d all_outputs=%zu max_vs_blas=%.9g rms_vs_blas=%.9g max_vs_current=%.9g keytile=%d querytile=%d\n",
                    rows,pos,kv,size_t(rows)*heads*D,baseline_max,std::sqrt(baseline_sq/(size_t(rows)*heads*D)),current_max,P::kKeyTile,P::kQueryTile);
        double squared = 0; size_t checked = 0;
        std::vector<int> rr{0, rows / 2, rows - 1}, hh{0, heads - 1};
        std::sort(rr.begin(), rr.end()); rr.erase(std::unique(rr.begin(), rr.end()), rr.end());
        for (int r : rr) for (int h : hh) {
            std::vector<double> exact; std::vector<float> tiled; oracle(r, h, exact, tiled);
            for (int d = 0; d < D; ++d) {
                const size_t j = (size_t(r) * heads + h) * D + d;
                const double error = std::abs(double(got[j]) - exact[d]);
                const double terr = std::abs(double(got[j]) - tiled[d]);
                const double berr = std::abs(double(before[j]) - exact[d]);
                const double cerr = std::abs(double(prior[j]) - exact[d]);
                // Probability half-rounding is intentional in both GPU paths. The tile oracle tests its recurrence separately.
                if (error > 4e-4 + 2e-3 * std::abs(exact[d]) || terr > 4e-5 + 3e-4 * std::abs(tiled[d]) ||
                    berr > 4e-4 + 2e-3 * std::abs(exact[d]) || cerr > 4e-4 + 2e-3 * std::abs(exact[d])) {
                    std::fprintf(stderr, "FAIL oracle rows=%d pos=%d kv=%d r=%d h=%d d=%d new=%.9g old=%.9g fp64=%.17g tile=%.9g err=%.9g olderr=%.9g tileerr=%.9g\n",
                                 rows,pos,kv,r,h,d,got[j],before[j],exact[d],tiled[d],error,berr,terr); std::exit(1);
                }
                if (analytic) {
                    const double count = pos + r + 1, answer = count / (count + std::exp(double(sinks[h])));
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
            for (int d = 0; d < D; ++d) { kc[(size_t(j)*kv+h)*D+d] = (d&1)?127:-128; vc[(size_t(j)*kv+h)*D+d] = (d&1)?-128:127; }
            for (int d = 0; d < 2; ++d) { ks[(size_t(j)*kv+h)*2+d] = hb(0.25f); vs[(size_t(j)*kv+h)*2+d] = hb(0.125f); }
        }
        dk.put(kc); dv.put(vc); dks.put(ks); dvs.put(vs); out.poison(); CHECK(launch()); const auto b = out.floats();
        const size_t earlier = size_t(rows / 2) * heads * D;
        CHECK(std::memcmp(a.data(), b.data(), earlier * sizeof(float)) == 0);
        CHECK(std::memcmp(a.data()+earlier, b.data()+earlier, (size_t(rows)*heads*D-earlier)*4) != 0);
        padding(b); guards(); ++cases;
    }
    struct Timing { double event_us, wall_us; };
    Timing timing(int path, int repeat) {
        const auto launch_path = [&] {
            if (path == 0) baseline();
            else if (path == 1) current();
            else CHECK(launch());
        };
        cudaEvent_t start=nullptr,stop=nullptr; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
        for (int i=0;i<3;++i) launch_path();
        CK(cudaStreamSynchronize(stream)); // No warm-up/copy work may leak into either measurement.
        const auto wall0=std::chrono::steady_clock::now();
        CK(cudaEventRecord(start,stream));
        for (int i=0;i<repeat;++i) launch_path();
        CK(cudaEventRecord(stop,stream));
        CK(cudaStreamSynchronize(stream)); // Independent host-clock bound on completion of all timed work.
        const auto wall1=std::chrono::steady_clock::now();
        float ms=0;CK(cudaEventElapsedTime(&ms,start,stop));
        CK(cudaEventDestroy(stop)); CK(cudaEventDestroy(start));
        const double event_us=double(ms)*1000/repeat;
        const double wall_us=std::chrono::duration<double,std::micro>(wall1-wall0).count()/repeat;
        CHECK(std::isfinite(event_us) && event_us>0 && std::isfinite(wall_us) && wall_us>0);
        return {event_us,wall_us};
    }

};

int main(int argc,char** argv) {
    bool bench=false,only_bench=false;int requested_pos=-1,requested_rows=1120,repeat=3;
    for (int i=1;i<argc;++i) {
        if (!std::strcmp(argv[i],"--bench") || !std::strcmp(argv[i],"--long-bench")) bench=true;
        else if (!std::strcmp(argv[i],"--bench-only")) bench=only_bench=true;
        else if (!std::strcmp(argv[i],"--position") || !std::strcmp(argv[i],"--rows") || !std::strcmp(argv[i],"--repeat")) {
            CHECK(i+1<argc);const char* option=argv[i++];char* end=nullptr;const long value=std::strtol(argv[i],&end,10);
            CHECK(end && end!=argv[i] && *end=='\0' && value>=0 && value<=131072);
            if (!std::strcmp(option,"--position")) requested_pos=int(value);
            else if (!std::strcmp(option,"--rows")) requested_rows=int(value);
            else repeat=int(value);
        } else { std::fprintf(stderr,"unknown argument: %s\n",argv[i]);return 2; }
    }
    CHECK(requested_rows>=1 && requested_rows<=1120 && repeat>=1 && repeat<=20);
    if (!P::available()) { std::puts("SKIP: long prefill unavailable on this backend/device");return 77; }
    CHECK(Current::available());
    cudaStream_t st=nullptr;CK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    if (!only_bench) {
        {
            Fixture f(33,0,1,st);f.init(false);f.out.poison();
            CHECK(!f.launch(0));CHECK(!f.launch(1,-1));CHECK(!f.launch(1,0,7));CHECK(!f.launch(2,INT_MAX));CHECK(!f.launch(1,INT_MAX-1));
            CHECK(!P::run(nullptr,f.dk.data<int8_t>(),f.dks.data<uint16_t>(),f.dv.data<int8_t>(),f.dvs.data<uint16_t>(),
                          f.ds.data<float>(),f.out.data<float>(),8,1,0,1,0.125f,st));
            CHECK(!P::run(f.dq.data<float>(),f.dk.data<int8_t>(),f.dks.data<uint16_t>(),f.dv.data<int8_t>(),f.dvs.data<uint16_t>(),
                          f.ds.data<float>(),f.out.data<float>(),8,0,0,1,0.125f,st));
            CHECK(!P::run(f.dq.data<float>(),f.dk.data<int8_t>(),f.dks.data<uint16_t>(),f.dv.data<int8_t>(),f.dvs.data<uint16_t>(),
                          f.ds.data<float>(),f.out.data<float>(),8,1,0,1,std::numeric_limits<float>::quiet_NaN(),st));
            for (int null_input=1;null_input<7;++null_input) {
                CHECK(!P::run(f.dq.data<float>(),null_input==1?nullptr:f.dk.data<int8_t>(),
                              null_input==2?nullptr:f.dks.data<uint16_t>(),null_input==3?nullptr:f.dv.data<int8_t>(),
                              null_input==4?nullptr:f.dvs.data<uint16_t>(),null_input==5?nullptr:f.ds.data<float>(),
                              null_input==6?nullptr:f.out.data<float>(),8,1,0,1,0.125f,st));
            }
            CK(cudaGetLastError());
            const auto untouched=f.out.floats();for (float x:untouched) CHECK(std::isnan(x));
            f.guards();++cases;
        }
        for (int rows:{1,7,8,15,16,17,31,32,33,63,64,65}) {
            Fixture f(rows,0,1,st);f.init(true);f.check(true,false);f.init(false);f.check(false,false);
            if (rows==33) { f.causal();f.init(false,true);f.check(false,false); }
        }
        for (int pos:{4095,16383,32767}) {
            Fixture f(33,pos,2,st);f.init(false);f.check(false,pos==16383);f.causal();
        }
        { Fixture f(32,16384,8,st);f.init(false);f.check(false,false); }
        // The real 1120-row geometry at a late prefix is validated in the default unit, not timing-only.
        { Fixture f(1120,32768,8,st);f.init(false);f.check(false,false); }
    }
    if (bench) {
        const std::vector<int> positions=requested_pos>=0?std::vector<int>{requested_pos}:std::vector<int>{16384,32768,65536};
        for (int pos:positions) {
            Fixture f(requested_rows,pos,8,st);f.init(false);f.check(false,false);
            // Every output and selected FP64 rows above MUST pass before any timing is emitted.
            // Six balanced samples use both GPU events and synchronized CPU wall time from the SAME work.
            const auto a0=f.timing(0,repeat),c0=f.timing(1,repeat),b0=f.timing(2,repeat);
            const auto b1=f.timing(2,repeat),c1=f.timing(1,repeat),a1=f.timing(0,repeat);
            const Fixture::Timing values[]={a0,c0,b0,b1,c1,a1};const char* names[]={"BLAS0","CURRENT0","LONG0","LONG1","CURRENT1","BLAS1"};
            for (int i=0;i<6;++i) std::printf("LONG_PREFILL_SAMPLE rows=%d pos=%d name=%s repeat=%d event_us=%.6f wall_us=%.6f\n",
                                             requested_rows,pos,names[i],repeat,values[i].event_us,values[i].wall_us);
            std::printf("LONG_PREFILL_MICRO rows=%d pos=%d keytile=%d querytile=%d LDS=%zu repeat=%d validated_before_timing=1 blas_event_us=%.6f current_event_us=%.6f long_event_us=%.6f blas_wall_us=%.6f current_wall_us=%.6f long_wall_us=%.6f\n",
                        requested_rows,pos,P::kKeyTile,P::kQueryTile,P::kSharedBytes,repeat,
                        (a0.event_us+a1.event_us)/2,(c0.event_us+c1.event_us)/2,(b0.event_us+b1.event_us)/2,
                        (a0.wall_us+a1.wall_us)/2,(c0.wall_us+c1.wall_us)/2,(b0.wall_us+b1.wall_us)/2);
        }
    }
    CK(cudaStreamDestroy(st));
    std::printf("PASS long prefill cases=%u max_abs=%.9g max_rms=%.9g baseline_max_abs=%.9g tile_max_abs=%.9g max_vs_blas=%.9g max_vs_current=%.9g keytile=%d querytile=%d LDS=%zu (FP16 Q/KV/P; all-output parity + selected FP64 rows)\n",
                cases,max_error,max_rms,baseline_error,tile_error,max_vs_blas,max_vs_current,P::kKeyTile,P::kQueryTile,P::kSharedBytes);
    return 0;
}
#endif
