// Model-free hybrid routing publish regression. CPU visibility is checked after
// the caller's route event only; there is no doorbell or whole-device wait.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include <cuda_runtime.h>
#include "strata/qwen35/hybrid_publish.cuh"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace P = strata::qwen35::hybrid_publish;
#define CK(x) do { const auto err_ = (x); if (err_ != cudaSuccess) { std::fprintf(stderr, "line %d: %s\n", __LINE__, cudaGetErrorString(err_)); std::exit(2); } } while (0)
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
using Clock = std::chrono::steady_clock;
static constexpr int kExperts = 256, kTopK = 8, kCapacity = P::kMaxRows;
static constexpr uint32_t kHiddenPoison = 0x7FC12345u, kIdsPoison = 0xC0DEC0DEu;
static_assert(kCapacity >= 32, "The regression covers 32-row Clover previews");
static unsigned cases = 0, handoffs = 0;
static uint64_t checksum = 1469598103934665603ull;

static void wait_route(cudaEvent_t event) {
    const auto deadline = Clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto result = cudaEventQuery(event);
        if (result == cudaSuccess) return;
        if (result != cudaErrorNotReady) CK(result);
        CHECK(Clock::now() < deadline);
        std::this_thread::yield();
    }
}
struct Device {
    uint8_t* base = nullptr;
    size_t bytes;
    Device(size_t n, cudaStream_t stream) : bytes(n) {
        CK(cudaMalloc(reinterpret_cast<void**>(&base), bytes + 128));
        // Keep setup on the working stream, including its guard fill.
        CK(cudaMemsetAsync(base, 0xA7, bytes + 128, stream));
    }
    ~Device() { if (base) (void) cudaFree(base); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void* data() const { return base + 64; }
    template<class T> T* as() const { return static_cast<T*>(data()); }
    void put(const void* source, size_t n, cudaStream_t stream) {
        CHECK(n <= bytes); CK(cudaMemcpyAsync(data(), source, n, cudaMemcpyHostToDevice, stream));
    }
    void guards() const {
        uint8_t prefix[64], suffix[64];
        CK(cudaMemcpy(prefix, base, 64, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(suffix, base + 64 + bytes, 64, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 64; ++i) CHECK(prefix[i] == 0xA7 && suffix[i] == 0xA7);
    }
};
struct Mapped {
    uint8_t* host_base = nullptr;
    uint8_t* device_base = nullptr;
    size_t bytes;
    explicit Mapped(size_t n) : bytes(n) {
        CK(cudaHostAlloc(reinterpret_cast<void**>(&host_base), bytes + 128, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_base), host_base, 0));
        std::memset(host_base, 0xA7, bytes + 128);
    }
    ~Mapped() { if (host_base) (void) cudaFreeHost(host_base); }
    Mapped(const Mapped&) = delete;
    Mapped& operator=(const Mapped&) = delete;
    template<class T> T* host() const { return reinterpret_cast<T*>(host_base + 64); }
    template<class T> T* device() const { return reinterpret_cast<T*>(device_base + 64); }
    void poison(uint32_t word) { CHECK(bytes % 4 == 0); std::fill_n(host<uint32_t>(), bytes / 4, word); }
    void guards() const {
        for (int i = 0; i < 64; ++i) CHECK(host_base[i] == 0xA7 && host_base[64 + bytes + i] == 0xA7);
    }
};

enum Pattern { AllHit, AllMiss, MixedRows, LastEntryMiss, NullTable };
static const char* name(Pattern pattern) {
    switch (pattern) {
    case AllHit: return "all-hit"; case AllMiss: return "all-miss";
    case MixedRows: return "mixed-rows"; case LastEntryMiss: return "last-entry-miss";
    default: return "null-table";
    }
}
struct Payload {
    std::vector<uint32_t> hidden;
    std::vector<int32_t> ids, slots;
    Payload(int rows, int width, Pattern pattern, unsigned epoch) : hidden(size_t(rows) * width),
        ids(size_t(rows) * kTopK), slots(kExperts) {
        for (size_t i = 0; i < hidden.size(); ++i) {
            const uint32_t mixed = uint32_t(i * 2654435761ull + epoch * 2246822519ull);
            hidden[i] = 0x3F000000u | (mixed & 0x007FFFFFu) | ((mixed & 1u) << 31);
            // Exact bit-copy contract, including signed zero, subnormal and NaN payload.
            if (i % 127 == 0) hidden[i] = 0x80000000u;
            if (i % 127 == 1) hidden[i] = 0;
            if (i % 127 == 2) hidden[i] = 0x7FC05678u;
            if (i % 127 == 3) hidden[i] = 1;
        }
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = int32_t((i + epoch * 11u) % kExperts);
        for (int i = 0; i < kExperts; ++i) slots[i] = pattern == AllMiss ? -1 : (i + int(epoch)) % kExperts;
        if (pattern == MixedRows) {
            for (int row = 0; row < rows; ++row) if (row % 2 == 0)
                slots[size_t(ids[size_t(row) * kTopK + row % kTopK])] = -1;
        }
        if (pattern == LastEntryMiss) slots[size_t(ids.back())] = -1;
    }
    bool miss(int row, Pattern pattern) const {
        if (pattern == NullTable) return true;
        for (int k = 0; k < kTopK; ++k) if (slots[size_t(ids[size_t(row) * kTopK + k])] < 0) return true;
        return false;
    }
};

struct Fixture {
    int width;
    cudaStream_t stream;
    cudaEvent_t route;
    Device hidden, ids, slots;
    Mapped host_hidden, host_ids;
    Fixture(int n, cudaStream_t st) : width(n), stream(st), hidden(size_t(kCapacity) * n * 4, st),
        ids(size_t(kCapacity) * kTopK * 4, st), slots(kExperts * 4, st),
        host_hidden(size_t(kCapacity) * n * 4), host_ids(size_t(kCapacity) * kTopK * 4) {
        CK(cudaEventCreateWithFlags(&route, cudaEventDisableTiming));
    }
    ~Fixture() { (void) cudaEventDestroy(route); }
    void upload(const Payload& payload) {
        hidden.put(payload.hidden.data(), payload.hidden.size() * 4, stream);
        ids.put(payload.ids.data(), payload.ids.size() * 4, stream);
        // Rebalance visibility: slot-table upload precedes publish on the same stream.
        slots.put(payload.slots.data(), payload.slots.size() * 4, stream);
    }
    void launch(int rows, Pattern pattern, int mode) {
        if (mode == 0) {
            CK(cudaMemcpyAsync(host_ids.host<int32_t>(), ids.data(), size_t(rows) * kTopK * 4, cudaMemcpyDeviceToHost, stream));
            CK(cudaMemcpyAsync(host_hidden.host<float>(), hidden.data(), size_t(rows) * width * 4, cudaMemcpyDeviceToHost, stream));
        } else P::publish(hidden.as<float>(), ids.as<int32_t>(), pattern == NullTable ? nullptr : slots.as<int32_t>(),
            host_hidden.device<float>(), host_ids.device<int32_t>(), rows, width, kTopK, kExperts, mode == 2, stream);
        CK(cudaGetLastError());
    }
    void verify(const Payload& payload, int rows, Pattern pattern, int mode) const {
        for (size_t i = 0; i < size_t(kCapacity) * kTopK; ++i) {
            const uint32_t expected = i < payload.ids.size() ? uint32_t(payload.ids[i]) : kIdsPoison;
            const uint32_t got = host_ids.host<uint32_t>()[i];
            if (got != expected) {
                std::fprintf(stderr, "IDs rows=%d width=%d pattern=%s mode=%d index=%zu expected=%08x got=%08x\n", rows, width, name(pattern), mode, i, expected, got);
                std::exit(1);
            }
            checksum = (checksum ^ got) * 1099511628211ull;
        }
        for (int row = 0; row < kCapacity; ++row) for (int column = 0; column < width; ++column) {
            const size_t i = size_t(row) * width + column;
            const bool copied = row < rows && (mode != 2 || payload.miss(row, pattern));
            const uint32_t expected = copied ? payload.hidden[i] : kHiddenPoison;
            const uint32_t got = host_hidden.host<uint32_t>()[i];
            if (got != expected) {
                std::fprintf(stderr, "hidden rows=%d width=%d pattern=%s mode=%d row=%d column=%d expected=%08x got=%08x\n", rows, width, name(pattern), mode, row, column, expected, got);
                std::exit(1);
            }
            checksum = (checksum ^ got) * 1099511628211ull;
        }
        host_hidden.guards(); host_ids.guards();
    }
    void exercise(int rows, Pattern pattern, int mode, unsigned epoch) {
        const Payload payload(rows, width, pattern, epoch);
        // The preceding route event has completed before these CPU stores occur.
        host_hidden.poison(kHiddenPoison); host_ids.poison(kIdsPoison);
        upload(payload); launch(rows, pattern, mode);
        CK(cudaEventRecord(route, stream));
        wait_route(route); // The sole producer/consumer handoff in the measured path.
        verify(payload, rows, pattern, mode); ++handoffs;
    }
    void guards() const { hidden.guards(); ids.guards(); slots.guards(); host_hidden.guards(); host_ids.guards(); }
};

static void correctness(cudaStream_t stream) {
    unsigned epoch = 0;
    for (int width : {2048, 37}) {
        Fixture fixture(width, stream);
        for (int rows : {1, 2, 4, 8, 16, 32}) for (Pattern pattern : {AllHit, AllMiss, MixedRows, LastEntryMiss, NullTable}) {
            for (int mode : {0, 1, 2}) fixture.exercise(rows, pattern, mode, ++epoch);
            ++cases;
        }
        // Reuse the SAME event and mapped buffers; switch misses and membership
        // every handoff, with no intervening stream/device synchronization.
        for (unsigned turn = 0; turn < 96; ++turn) {
            const int rows = 1 << (turn % 6);
            const Pattern pattern = static_cast<Pattern>(turn % 5);
            fixture.exercise(rows, pattern, 2, ++epoch);
        }
        ++cases;
        fixture.guards();
    }
}

static void benchmark(cudaStream_t stream) {
    constexpr int repeats = 64, warmup = 4;
    std::puts("BENCH micro_only=1 destination=mapped_host modes=0:two_D2H,1:fused,2:conditional repeats=64; GPU-event service and CPU enqueue+route-event roundtrip are separate measurements");
    Fixture fixture(2048, stream);
    cudaEvent_t start, end; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&end));
    for (int rows : {1, 4, 8, 32}) for (Pattern pattern : {AllHit, MixedRows, AllMiss}) {
        const Payload payload(rows, fixture.width, pattern, 5);
        fixture.upload(payload); CK(cudaEventRecord(fixture.route, stream)); wait_route(fixture.route);
        for (int mode : {0, 1, 2}) {
            for (int i = 0; i < warmup; ++i) fixture.launch(rows, pattern, mode);
            CK(cudaEventRecord(fixture.route, stream)); wait_route(fixture.route);
            CK(cudaEventRecord(start, stream));
            for (int i = 0; i < repeats; ++i) fixture.launch(rows, pattern, mode);
            CK(cudaEventRecord(end, stream)); wait_route(end);
            float milliseconds = 0; CK(cudaEventElapsedTime(&milliseconds, start, end));
            const auto begun = Clock::now();
            for (int i = 0; i < repeats; ++i) {
                fixture.launch(rows, pattern, mode);
                CK(cudaEventRecord(fixture.route, stream)); wait_route(fixture.route);
            }
            const double roundtrip_us = std::chrono::duration<double, std::micro>(Clock::now() - begun).count() / repeats;
            int missed_rows = 0; for (int row = 0; row < rows; ++row) missed_rows += payload.miss(row, pattern);
            const size_t copied = size_t(rows) * kTopK * 4 + size_t(mode == 2 ? missed_rows : rows) * fixture.width * 4;
            std::printf("BENCH rows=%d width=%d pattern=%s mode=%d published_bytes=%zu GPU_us=%.6f route_roundtrip_us=%.6f\n",
                rows, fixture.width, name(pattern), mode, copied, milliseconds * 1000.0 / repeats, roundtrip_us);
            std::fflush(stdout);
        }
    }
    CK(cudaEventDestroy(start)); CK(cudaEventDestroy(end)); fixture.guards();
}
int main(int argc, char** argv) {
    const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    if (argc > 1 && !bench) { std::fputs("usage: hybrid-publish-gpu [--bench]\n", stderr); return 2; }
    int devices = 0; CK(cudaGetDeviceCount(&devices)); if (!devices) return 77;
    cudaStream_t stream; CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (bench) benchmark(stream); else correctness(stream);
    CK(cudaStreamDestroy(stream));
    if (!bench) std::printf("PASS hybrid publish cases=%u event_handoffs=%u max_rows=%d modes=two_D2H/fused/conditional; exact bits, skipped-row poison, guards, slot updates hash=%016llx\n",
        cases, handoffs, P::kMaxRows, (unsigned long long)checksum);
    return 0;
}
#endif
