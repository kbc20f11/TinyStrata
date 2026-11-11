#include "strata/qwen35/hybrid_buffer_policy.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace H = strata::qwen35::hybrid_buffer_policy;
static int checks = 0;
static void check(bool ok, const char* name) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", name); std::exit(1); }
}

int main() {
    H::Layout full, bounded;
    check(H::layout(2048, 8, 2048, full), "full layout");
    check(H::layout(128, 8, 2048, bounded), "bounded layout");
    check(full.values_bytes == 134217728 && full.map_bytes == 65536, "independent full bytes");
    check(bounded.values_bytes == 8388608 && bounded.map_bytes == 4096, "independent bounded bytes");
    check(full.total_bytes - bounded.total_bytes == 125890560, "independent bytes saved");
    for (int unavailable = 0; unavailable < 3; ++unavailable)
        check(H::capacity_rows(444, 387, 2048, 128, 32,
            unavailable != 0, unavailable != 1, unavailable == 2) == 2048,
            "unbounded routes unchanged");
    check(H::capacity_rows(387, 387, 2048, 128, 32, true, true, false) == 2048, "exact fit unchanged");
    check(H::capacity_rows(386, 387, 2048, 128, 32, true, true, false) == 2048, "fitting unchanged");
    check(H::capacity_rows(388, 387, 2048, 128, 32, true, true, false) == 128, "failure rescue");
    check(H::capacity_rows(388, 387, 2048, 8, 32, true, true, false) == 32, "verify bound");
    check(H::capacity_rows(388, 387, 16, 128, 32, true, true, false) == 16, "tmax cap");
    const bool unbacked = H::protect_unbacked_kv_growth(0, true, 16384, 32768);
    check(unbacked, "zero-slot lazy KV needs growth headroom");
    check(H::capacity_rows(443, 545, 2048, 128, 32, true, true, false, unbacked) == 128,
          "fitting startup retains otherwise unbacked future KV headroom");
    check(!H::protect_unbacked_kv_growth(1, true, 16384, 32768), "Expert backing is reclaimable");
    check(!H::protect_unbacked_kv_growth(0, false, 16384, 32768), "whole KV unchanged");
    check(!H::protect_unbacked_kv_growth(0, true, 32768, 32768), "full KV already committed");
    check(!H::protect_unbacked_kv_growth(-1, true, 16384, 32768), "invalid Expert slot count not treated as zero");
    check(!H::protect_unbacked_kv_growth(0, true, -1, 32768), "invalid KV count");
    check(H::capacity_rows(443, 545, 2048, 128, 32, true, true, false,
          H::protect_unbacked_kv_growth(1533, true, 16384, 32768)) == 2048, "fitting12GB cache-backed path unchanged");
    check(H::capacity_rows(443, 545, 2048, 128, 32, true, true, false,
          H::protect_unbacked_kv_growth(0, true, 32768, 32768)) == 2048, "fitting fully committed path unchanged");
    for (int unavailable = 0; unavailable < 3; ++unavailable)
        check(H::capacity_rows(443, 545, 2048, 128, 32,
            unavailable != 0, unavailable != 1, unavailable == 2, unbacked) == 2048,
            "future-KV protection cannot shrink unbounded CPU fallback routes");
    H::Layout invalid;
    const auto maximum = std::numeric_limits<size_t>::max();
    check(!H::layout(0, 8, 2048, invalid), "zero rows");
    check(!H::layout(1, 0, 2048, invalid), "zero experts");
    check(!H::layout(1, 8, 0, invalid), "zero width");
    check(!H::layout(maximum, 2, 1, invalid), "entry overflow");
    check(!H::layout(1, 2, maximum, invalid), "value count overflow");
    check(!H::layout(1, 1, maximum / sizeof(float), invalid), "total byte overflow");
    H::Transfer copy;
    check(H::transfer(128, 128, 1024, 8, 2048, copy), "R copy allowed");
    check(copy.values_bytes == bounded.values_bytes && copy.map_bytes == bounded.map_bytes, "R exact copy bytes");
    check(H::transfer(128, 8, 3, 8, 2048, copy), "compact copy allowed");
    check(copy.values_bytes == 3 * 2048 * 4 && copy.map_bytes == 8 * 8 * 4, "compact map remains full");
    check(H::transfer(128, 8, 0, 8, 2048, copy) && copy.values_bytes == 0 && copy.map_bytes == 256,
          "zero compact values still has map");
    check(!H::transfer(128, 129, 0, 8, 2048, copy), "R+1 map-only rejected");
    check(!H::transfer(128, 129, 1032, 8, 2048, copy), "R+1 full rejected");
    check(!H::transfer(128, 8, 65, 8, 2048, copy), "compact beyond request rejected");
    check(!H::transfer(128, 128, 1025, 8, 2048, copy), "output beyond capacity rejected");
    check(!H::transfer(128, 1, maximum, 8, 2048, copy), "huge output rejected");
    check(!H::transfer(128, 0, 0, 8, 2048, copy), "zero request rejected");
    check(!H::transfer(-1, 1, 1, 8, 2048, copy), "negative capacity rejected");
    check(!H::transfer(128, -1, 0, 8, 2048, copy), "negative request rejected");
    std::printf("PASS %d portable hybrid buffer policy checks\n", checks);
}
