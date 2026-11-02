// CPU-only accounting for the CUDA Qwen automatic prompt-chunk rule.
// Geometry follows Engine::alloc_kv and kv_grow::Range::mask. This describes
// requested/cache-backed bytes, not a claim about allocator/driver overhead.
#pragma once
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace strata::qwen35::chunk_policy {
using Bytes = std::uint64_t;
constexpr Bytes KiB = 1024;
constexpr Bytes MiB = 1024 * KiB;
constexpr Bytes GiB = 1024 * MiB;

inline bool add(Bytes a, Bytes b, Bytes& out) {
    if (b > std::numeric_limits<Bytes>::max() - a) return false;
    out = a + b;
    return true;
}
inline bool mul(Bytes a, Bytes b, Bytes& out) {
    if (a && b > std::numeric_limits<Bytes>::max() / a) return false;
    out = a * b;
    return true;
}
inline bool rounded(Bytes bytes, Bytes granularity, Bytes& out) {
    if (!granularity) return false;
    const Bytes remainder = bytes % granularity;
    return !remainder ? (out = bytes, true) : add(bytes, granularity - remainder, out);
}

struct Geometry {
    Bytes context = 0;
    Bytes attention_layers = 0; // main layers which actually own K/V
    bool mtp_attention = false; // only a loaded MTP block owns another cache
    Bytes kv_heads = 0;
    Bytes head_dim = 0;
    int kv_format = -1;         // 0 FP32, 1 legacy FP8, 2 INT8, 3 native FP8
    bool tiled_prompt = false; // same runtime shape/kernel probe as Engine::init
    bool growth = false;       // true only when VMM is actually available
    Bytes granularity = 0;
};

struct Memory {
    Bytes logical_kv = 0;
    Bytes initial_backing = 0;  // per-head prefixes, including page-boundary unions
    Bytes full_backing = 0;     // all cache planes; whole-allocation request if !growth
    Bytes fallback_fp16 = 0;    // one reusable K/V pair; never multiplied by layers
    Bytes initial_rows = 0;
};

// Range::mask rounds each head's prefix to pages in one plane. Adjacent heads
// may share a boundary page, so summing separately rounded prefixes is wrong.
inline bool prefix_bytes(Bytes context, Bytes rows, Bytes heads, Bytes row_bytes,
                         Bytes granularity, Bytes& out) {
    if (!context || rows > context || !heads || heads > 65536 || !row_bytes || !granularity) return false;
    if (!rows) { out = 0; return true; }
    Bytes stride = 0, logical = 0, prefix = 0;
    if (!mul(context, row_bytes, stride) || !mul(stride, heads, logical) || !mul(rows, row_bytes, prefix)) return false;
    Bytes pages = 0, last_end = 0;
    for (Bytes h = 0; h < heads; ++h) {
        Bytes begin = 0, end = 0, end_rounded = 0;
        if (!mul(h, stride, begin) || !add(begin, prefix, end) || !rounded(end, granularity, end_rounded)) return false;
        const Bytes first_page = begin / granularity;
        const Bytes end_page = end_rounded / granularity;
        const Bytes added = end_page > last_end ? end_page - std::max(first_page, last_end) : 0;
        if (!add(pages, added, pages)) return false;
        last_end = std::max(last_end, end_page);
    }
    return mul(pages, granularity, out);
}

inline bool estimate(const Geometry& g, Memory& out) {
    if (!g.context || !g.kv_heads || g.kv_heads > 65536 || !g.head_dim ||
        g.kv_format < 0 || g.kv_format > 3 || (g.growth && !g.granularity) ||
        ((g.kv_format == 2 || g.kv_format == 3) && g.head_dim % 64)) return false;
    Memory result;
    result.initial_rows = g.growth ? std::min<Bytes>(g.context, 16384) : g.context;
    Bytes layers = 0;
    if (!add(g.attention_layers, g.mtp_attention ? 1 : 0, layers)) return false;
    Bytes rows[4] = {};
    if (g.kv_format == 0) {
        if (!mul(g.head_dim, 4, rows[0])) return false;
        rows[1] = rows[0];
    } else {
        rows[0] = rows[1] = g.head_dim;
        if (g.kv_format == 1) rows[2] = rows[3] = 4;
        else {
            if (!mul(g.head_dim / 64, 2, rows[2])) return false;
            rows[3] = rows[2];
        }
    }
    for (const Bytes row : rows) if (row) {
        Bytes plane = 0, initial = 0, full = 0;
        if (!mul(g.context, g.kv_heads, plane) || !mul(plane, row, plane)) return false;
        if (g.growth) {
            if (!rounded(plane, g.granularity, full) ||
                !prefix_bytes(g.context, result.initial_rows, g.kv_heads, row, g.granularity, initial)) return false;
        } else initial = full = plane;
        if (!mul(plane, layers, plane) || !mul(initial, layers, initial) || !mul(full, layers, full) ||
            !add(result.logical_kv, plane, result.logical_kv) ||
            !add(result.initial_backing, initial, result.initial_backing) ||
            !add(result.full_backing, full, result.full_backing)) return false;
    }
    if (!g.tiled_prompt) {
        // k16/v16 are shared prompt staging arrays even with growing byte KV.
        if (!mul(g.context, g.kv_heads, result.fallback_fp16) ||
            !mul(result.fallback_fp16, g.head_dim, result.fallback_fp16) ||
            !mul(result.fallback_fp16, 4, result.fallback_fp16)) return false;
    }
    out = result;
    return true;
}

struct Plan {
    Memory memory;
    Bytes growth_extra = 0;
    Bytes required = 0;
    int chunk = 512;
    bool full_resident_fits = false;
};

// Preserve the established conservative non-KV buffer envelope. Its 8K cache
// allowance is already included; only the *actual* cache/staging growth above
// that baseline is added. The envelope also covers score scratch for fallback
// attention; this rule does not reclaim it on tiled attention.
// Budget the full logical cache backing rather than only its initial VMM
// prefix, so a larger chunk does not borrow the future KV-growth allowance.
inline bool plan(Bytes free_raw, Bytes experts, Bytes reserve, const Geometry& g, Plan& out) {
    Plan result;
    Memory baseline;
    Geometry old = g;
    old.context = std::min<Bytes>(8192, g.context);
    Bytes full = 0, base = 0;
    if (!estimate(g, result.memory) || !estimate(old, baseline) ||
        !add(result.memory.full_backing, result.memory.fallback_fp16, full) ||
        !add(baseline.full_backing, baseline.fallback_fp16, base)) return false;
    // The legacy envelope contains only this fixed 8K allowance. On larger
    // geometries, crediting their entire 8K cache would hide real allocation.
    base = std::min<Bytes>(base, (45056 + 5120) * 8192ull);
    result.growth_extra = full > base ? full - base : 0;
    // The old 2.75 GiB envelope includes the default 768 MiB reserve.
    const Bytes extra_reserve = reserve > 768 * MiB ? reserve - 768 * MiB : 0;
    Bytes fixed = 0;
    if (!add(experts, 11 * GiB / 4, fixed) || !add(fixed, result.growth_extra, fixed) ||
        !add(fixed, extra_reserve, fixed)) return false;
    for (const int chunk : {4096, 2048, 1024, 512}) {
        Bytes need = 0;
        // A large candidate can overflow while a smaller one is still valid.
        if (!add(fixed, static_cast<Bytes>(chunk - 512) * MiB / 2, need)) continue;
        result.required = need;
        result.chunk = chunk;
        if (need <= free_raw) { result.full_resident_fits = true; break; }
    }
    out = result;
    return true;
}
} // namespace strata::qwen35::chunk_policy
