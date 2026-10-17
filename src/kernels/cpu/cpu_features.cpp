#include "strata/kernels/cpu/cpu_features.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#endif

#include "strata/kernels/cpu/native_expert.hpp"

namespace strata::kernels::cpu {
int cpu_isa_cap() {
    // STRATA_FORCE_ISA (tests, the experimental older-CPU builds): the engine's own dispatch acts as if this CPU
    // stopped at that level.  ggml-cpu is not affected: it runs what the build compiled it for.
    static const int cap = [] {
        const char* f = std::getenv("STRATA_FORCE_ISA");
        if (f == nullptr || f[0] == '\0') return 3;
        const std::string v(f);
        if (v == "avx2") return 2;
        if (v == "avx") return 1;
        if (v == "sse" || v == "sse4.2" || v == "none") return 0;
        std::fprintf(stderr, "strata: STRATA_FORCE_ISA=%s is not avx2, avx or sse; ignored\n", f);
        return 3;
    }();
    return cap;
}

bool cpu_avx512_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        if (cpu_isa_cap() < 3) return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
#if !defined(__aarch64__)   // arm64 (the Metal build): no XCR0; cpuid reads zero there, so this is never reached
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
#endif
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1], ecx = r[2];
        return ((ebx >> 16) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u) && ((ecx >> 11) & 1u) && ((ecx >> 1) & 1u);
    }();
    return ok;
}

bool cpu_avx512bw_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
#if !defined(__aarch64__)   // arm64 (the Metal build): no XCR0; cpuid reads zero there, so this is never reached
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
#endif
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1];
        return ((ebx >> 16) & 1u) && ((ebx >> 17) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u);
    }();
    return ok;
}

bool cpu_avx2_ok() {
    static const bool ok = [] {
        if (cpu_isa_cap() < 2) return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        const unsigned ecx1 = r[2];
        // FMA (12), OSXSAVE (27), AVX (28), F16C (29)
        if (!((ecx1 >> 12) & 1u) || !((ecx1 >> 27) & 1u) || !((ecx1 >> 28) & 1u) || !((ecx1 >> 29) & 1u)) return false;
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
#if !defined(__aarch64__)   // arm64 (the Metal build): no XCR0; cpuid reads zero there, so this is never reached
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
#endif
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0x6) != 0x6) return false;            // the OS saves the SSE and AVX state
        cpuid(7, 0);
        return ((r[1] >> 5) & 1u) != 0;                    // AVX2
    }();
    return ok;
}

bool cpu_avx1_ok() {
    // AVX (Sandy Bridge, 2011): AVX + OSXSAVE with the OS saving the YMM state; FMA, F16C and AVX2 not needed.
    // From the Strata_Dirigo fork (rwkeyes).
    static const bool ok = [] {
        if (cpu_isa_cap() < 1) return false;
        unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, 1, 0);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
#endif
        if (!((r[2] >> 27) & 1u) || !((r[2] >> 28) & 1u)) return false;   // OSXSAVE, AVX
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
#if !defined(__aarch64__)   // arm64 (the Metal build): no XCR0; cpuid reads zero there, so this is never reached
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
#endif
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        return (xcr0 & 0x6) == 0x6;
    }();
    return ok;
}

bool cpu_sse42_ok() {
    static const bool ok = [] {
        unsigned r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, 1, 0);
        for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
        __cpuid_count(1, 0, r[0], r[1], r[2], r[3]);
#endif
        return ((r[2] >> 20) & 1u) && ((r[2] >> 23) & 1u);   // SSE4.2, POPCNT
    }();
    return ok;
}

const char* isa_floor_build() {
#if defined(STRATA_ISA_FLOOR_AVX)
    return "avx";
#elif defined(STRATA_ISA_FLOOR_NONE)
    return "sse4.2";
#else
    return "";
#endif
}

namespace {
void cpuid_regs(unsigned leaf, unsigned sub, unsigned r[4]) {
#if defined(_MSC_VER)
    int x[4];
    __cpuidex(x, (int) leaf, (int) sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}
}  // namespace

int iq256_gather_setting() {
    static const int s = [] {
        const char* v = std::getenv("STRATA_IQ256_GATHER");
        if (v == nullptr || v[0] == '\0') return -1;
        return std::atoi(v) != 0 ? 1 : 0;
    }();
    return s;
}

bool cpu_gather_fast() {
    static const bool ok = [] {
        if (cpu_isa_cap() < 3) return false;   // STRATA_FORCE_ISA: as on a CPU that stops at AVX2
        unsigned r[4];
        cpuid_regs(0, 0, r);
        const unsigned max_leaf = r[0];
        if (!(r[1] == 0x756e6547u && r[3] == 0x49656e69u && r[2] == 0x6c65746eu)) return false;   // "GenuineIntel"
        if (max_leaf < 7) return false;
        cpuid_regs(7, 0, r);
        if (r[0] < 1) return false;            // no sub-leaf 1
        cpuid_regs(7, 1, r);
        if (!((r[0] >> 4) & 1u)) return false; // AVX-VNNI: Alder Lake, Sapphire Rapids and later
        // The E-core-only parts with AVX-VNNI are not hybrid, so the core type below may not tell them apart: Alder
        // Lake-N / Twin Lake (6/BEh), Grand Ridge (6/B6h), Sierra Forest (6/AFh), Clearwater Forest (6/DDh).
        cpuid_regs(1, 0, r);
        const unsigned family = (r[0] >> 8) & 0xf, model = ((r[0] >> 4) & 0xf) | (((r[0] >> 16) & 0xf) << 4);
        if (family == 6 && (model == 0xBE || model == 0xB6 || model == 0xAF || model == 0xDD)) return false;
        return true;
    }();
    return ok;
}

bool cpu_gather_fast_here() {
    if (!cpu_gather_fast()) return false;
    static const unsigned max_leaf = [] { unsigned r[4]; cpuid_regs(0, 0, r); return r[0]; }();
    static const bool hybrid = [] { unsigned r[4]; cpuid_regs(7, 0, r); return ((r[3] >> 15) & 1u) != 0; }();
    // A CPUID can cost a microsecond under a hypervisor (Windows with VBS), so once per thread.
    thread_local int here = -1;
    if (here < 0) {
        unsigned type = 0;
        if (max_leaf >= 0x1A) {
            unsigned r[4];
            cpuid_regs(0x1A, 0, r);
            type = r[0] >> 24;
        }
        // 40h a performance core, 20h an E-core; no core type is a non-hybrid part (P-cores, see cpu_gather_fast)
        here = type == 0x40 || (type == 0 && !hybrid) ? 1 : 0;
    }
    return here == 1;
}

bool cpu_avxvnni_ok() {
    static const bool ok = [] {
        if (const char* v = std::getenv("STRATA_NO_AVXVNNI"); v != nullptr && std::atoi(v) != 0) return false;
        if (isa_floor_build()[0] != '\0' || cpu_isa_cap() < 3 || !cpu_avx2_ok()) return false;
        unsigned r[4];
        cpuid_regs(0, 0, r);
        if (r[0] < 7) return false;
        cpuid_regs(7, 0, r);
        if (r[0] < 1) return false;            // no sub-leaf 1
        cpuid_regs(7, 1, r);
        return ((r[0] >> 4) & 1u) != 0;        // AVX-VNNI; VEX-encoded, so the OS state is AVX's (cpu_avx2_ok)
    }();
    return ok;
}

std::string cpu_name() {
    unsigned r[12] = {};
#if defined(_MSC_VER)
    int x[4];
    __cpuid(x, (int) 0x80000000u);
    if ((unsigned) x[0] < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) {
        __cpuid(x, (int) (0x80000002u + i));
        for (int j = 0; j < 4; ++j) r[i * 4 + j] = (unsigned) x[j];
    }
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid(0x80000000u, a, b, c, d);
    if (a < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) __cpuid(0x80000002u + i, r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3]);
#endif
    char s[49] = {};
    std::memcpy(s, r, 48);
    std::string name(s);
    const size_t b0 = name.find_first_not_of(' '), b1 = name.find_last_not_of(' ');
    return b0 == std::string::npos ? std::string("unknown") : name.substr(b0, b1 - b0 + 1);
}

#if !defined(STRATA_NATIVE_EXPERTS)
// CPU expert support is disabled in this build.
bool native_experts_available() noexcept { return false; }
bool native_fmt(int, int, int64_t, int64_t, NativeFmt&, std::string& err) { err = "built without native experts"; return false; }
void native_quant_act(const NativeFmt&, const float*, void*) { std::abort(); }
void native_quant_h(const NativeFmt&, const float*, void*) { std::abort(); }
int native_gu_mt_min(int) { return 2; }
void native_gu_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
void native_down_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
#endif

}  // namespace strata::kernels::cpu
