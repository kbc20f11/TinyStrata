// Auxiliary-drafter Q5_1 read-once parity against the existing type7 IQ MMVQ.
// Actual drafter widths, every column count, zero/mixed/minimum terms, guards,
// graph replay with changed weights/inputs, and optional synchronized timings.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include "strata/gemma4/draft_q51.hpp"
#include "strata/gemma4/matvec.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace G = strata::gemma4;
namespace K = strata::kernels;
#define CK(x) do {const auto err_=(x);if(err_!=cudaSuccess){std::fprintf(stderr,"line%d %s\n",__LINE__,cudaGetErrorString(err_));std::exit(2);}}while(0)
#define REQUIRE(x) do {if(!(x)){std::fprintf(stderr,"FAIL line%d %s\n",__LINE__,#x);std::exit(1);}}while(0)
unsigned cases = 0;
uint64_t values_compared = 0;
uint32_t state = 0x56A83715;
uint32_t random_word() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
float random_float() { return float(double(random_word()) / 4294967296.0 * 2.0 - 1.0); }

struct Q51 { uint16_t d, m; uint8_t high[4], low[16]; };
static_assert(sizeof(Q51) == 24);

struct Buffer {
    uint8_t* p = nullptr;
    size_t bytes;
    cudaStream_t stream;
    Buffer(size_t n, cudaStream_t s) : bytes(n), stream(s) {
        CK(cudaMalloc(reinterpret_cast<void**>(&p), n + 128));
        CK(cudaMemsetAsync(p, 0xA7, n + 128, s));
    }
    ~Buffer() { if (p) (void)cudaFree(p); }
    template<class T> T* data() const { return reinterpret_cast<T*>(p + 64); }
    void upload(const void* src) { CK(cudaMemcpyAsync(p + 64, src, bytes, cudaMemcpyHostToDevice, stream)); }
    std::vector<uint8_t> read() const {
        CK(cudaStreamSynchronize(stream));
        std::vector<uint8_t> all(bytes + 128);
        CK(cudaMemcpy(all.data(), p, all.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < 64; ++i) REQUIRE(all[i] == 0xA7 && all[64 + bytes + i] == 0xA7);
        return {all.begin() + 64, all.end() - 64};
    }
};

struct Fixture {
    int input, output, columns;
    cudaStream_t stream;
    std::vector<Q51> weights;
    std::vector<float> x;
    Buffer dw, dx, dq, before, after;
    Fixture(int in, int out, int nc, cudaStream_t s) : input(in), output(out), columns(nc), stream(s),
        weights(size_t(in / 32) * out), x(size_t(in) * nc), dw(weights.size() * sizeof(Q51), s),
        dx(x.size() * sizeof(float), s), dq(size_t(nc) * K::gemma_native_q8_1_bytes(in), s),
        before(size_t(nc) * out * sizeof(float), s), after(size_t(nc) * out * sizeof(float), s) {}
    void fill(int pattern) {
        for (Q51& w : weights) {
            w.d = __half_as_ushort(__float2half(pattern == 1 ? 0.f : random_float() * .45f));
            w.m = __half_as_ushort(__float2half(pattern == 2 ? 0.f : random_float() * .75f));
            for (uint8_t& v : w.high) v = uint8_t(random_word());
            for (uint8_t& v : w.low) v = uint8_t(random_word());
        }
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = pattern == 1 ? (i % 3 ? 0.f : -0.f) : random_float() * (pattern == 2 ? 16384.f : 3.f);
        dw.upload(weights.data()); dx.upload(x.data());
        K::gemma_native_quantize_q8_1(dx.data<float>(), dq.data<void>(), input * columns, 1, stream);
    }
    void original() {
        K::gemma_native_mmvq(7, dw.data<void>(), dq.data<void>(), before.data<float>(), input, output, columns, stream);
    }
    void candidate() {
        G::draft_q51_mmvq(dw.data<void>(), dq.data<void>(), after.data<float>(), input, output, columns, stream);
    }
    void equal(const char* phase) {
        const auto a = before.read(), b = after.read();
        REQUIRE(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) {
            float f, g; const size_t first = i / 4 * 4;
            std::memcpy(&f, a.data() + first, 4); std::memcpy(&g, b.data() + first, 4);
            std::fprintf(stderr,"FAIL %s in%d out%d cols%d index%zu old%.9g new%.9g byte%zu\n",
                         phase,input,output,columns,i / 4,double(f),double(g),i);std::exit(1);
        }
        for (size_t i = 0; i < a.size(); i += 4) {float f;std::memcpy(&f,a.data()+i,4);REQUIRE(std::isfinite(f));}
        values_compared += a.size() / 4; ++cases;
        dw.read(); dx.read(); dq.read();
    }
};

void graph_case(int in, int out, int nc, cudaStream_t stream) {
    Fixture f(in, out, nc, stream); f.fill(0);
    CK(cudaStreamSynchronize(stream));
    cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));f.candidate();
    CK(cudaStreamEndCapture(stream, &graph));CK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int pattern = 0; pattern < 3; ++pattern) {
        f.fill(pattern);f.original();CK(cudaGraphLaunch(executable, stream));f.equal("graph changed weights and data");
    }
    CK(cudaGraphExecDestroy(executable));CK(cudaGraphDestroy(graph));
}

void benchmark(cudaStream_t stream) {
    Fixture f(2880, 11520, 8, stream);f.fill(0);f.original();f.candidate();f.equal("bench parity");
    for (int i = 0; i < 10; ++i) { f.original(); f.candidate(); }
    CK(cudaStreamSynchronize(stream));
    for (bool readonce : {false, true}) {
        constexpr int repeats = 40;
        cudaEvent_t start, end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
        const auto wall0 = std::chrono::steady_clock::now();CK(cudaEventRecord(start, stream));
        for (int i = 0; i < repeats; ++i) { if (readonce) f.candidate();else f.original(); }
        CK(cudaEventRecord(end, stream));CK(cudaStreamSynchronize(stream));
        const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-wall0).count();
        float event = 0;CK(cudaEventElapsedTime(&event, start, end));
        std::printf("Q51_BENCH mode=%s input=2880 output=11520 cols=8 repeats=%d event_ms=%.6f wall_ms=%.6f\n",
                     readonce ? "readonce" : "original",repeats,double(event)/repeats,wall/repeats);
        CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
    }
}

int main(int argc, char** argv) {
    cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int shapes[][2] = {{8640,2880},{5760,4096},{5760,512},{2880,11520},{2880,513},{2880,1},{32,5},{96,17}};
    for (const auto& shape : shapes) for (int nc = 1; nc <= 8; ++nc) {
        Fixture f(shape[0],shape[1],nc,stream);
        for (int p = 0; p < 3; ++p) { f.fill(p);f.original();f.candidate();f.equal("direct"); }
    }
    graph_case(2880,513,1,stream);graph_case(2880,513,3,stream);graph_case(8640,2880,8,stream);
    // Invalid calls must fail before launching and must leave no sticky GPU error.
    for (int nc : {0,9}) {
        bool threw=false;try{G::draft_q51_mmvq(reinterpret_cast<void*>(4),reinterpret_cast<void*>(4),reinterpret_cast<float*>(4),32,1,nc,stream);}
        catch(const std::invalid_argument&){threw=true;}REQUIRE(threw);REQUIRE(cudaGetLastError()==cudaSuccess);++cases;
    }
    if (argc > 1 && std::strcmp(argv[1],"--bench") == 0) benchmark(stream);
    CK(cudaStreamSynchronize(stream));CK(cudaStreamDestroy(stream));
    std::printf("PASS draft Q5_1 read-once cases=%u values=%llu bitwise columns1..8 EAGLE3 widths, guards, changed-data graphs\n",
                 cases,static_cast<unsigned long long>(values_compared));
}
#endif
