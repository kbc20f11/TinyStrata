// Standalone CPU test: no CUDA headers, GPU runtime, model files or test framework.
#include "strata/qwen35/chunk_policy.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace CP = strata::qwen35::chunk_policy;
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

// Independent page-mask oracle, following the actual union of head intervals.
CP::Bytes mask_oracle(CP::Bytes ctx, CP::Bytes rows, CP::Bytes heads, CP::Bytes row, CP::Bytes gran) {
    const CP::Bytes bytes = ctx * heads * row;
    std::vector<bool> used((std::size_t) ((bytes + gran - 1) / gran));
    for (CP::Bytes h = 0; h < heads; ++h)
        for (CP::Bytes b = h * ctx * row; b < (h * ctx + rows) * row; ++b)
            used[(std::size_t) (b / gran)] = true;
    CP::Bytes pages = 0;
    for (bool x : used) pages += x ? 1 : 0;
    return pages * gran;
}

int main() {
    CP::Geometry q;
    q.context = 262144; q.attention_layers = 10; q.mtp_attention = true;
    q.kv_heads = 2; q.head_dim = 256; q.kv_format = 3;
    q.tiled_prompt = true; q.growth = true; q.granularity = 65536;
    CP::Memory m;
    REQUIRE(CP::estimate(q, m));
    REQUIRE(m.logical_kv == 3045064704ull);
    REQUIRE(m.full_backing == m.logical_kv);
    REQUIRE(m.initial_rows == 16384 && m.initial_backing == 190316544ull);
    REQUIRE(m.fallback_fp16 == 0);
    q.kv_format = 2;
    CP::Memory int8;
    REQUIRE(CP::estimate(q, int8));
    REQUIRE(int8.logical_kv == m.logical_kv && int8.initial_backing == m.initial_backing);
    q.mtp_attention = false;
    REQUIRE(CP::estimate(q, m) && m.logical_kv == 2768240640ull);
    q.mtp_attention = true; q.kv_format = 0; q.tiled_prompt = false;
    REQUIRE(CP::estimate(q, m));
    REQUIRE(m.logical_kv == 11 * CP::GiB);
    REQUIRE(m.fallback_fp16 == CP::GiB / 2); // shared staging, not 11 pairs
    q.kv_format = 1;
    REQUIRE(CP::estimate(q, m));
    REQUIRE(m.logical_kv == 2998927360ull && m.fallback_fp16 == CP::GiB / 2);

    // Real format/layout accounting across less convenient shapes and contexts.
    for (const CP::Bytes ctx : {1ull, 17ull, 8192ull, 16385ull, 262144ull})
        for (const CP::Bytes dim : {64ull, 128ull, 256ull})
            for (int format = 0; format != 4; ++format) {
                auto g = q;
                g.context = ctx; g.head_dim = dim; g.kv_heads = 3;
                g.attention_layers = 7; g.mtp_attention = false; g.kv_format = format;
                const CP::Bytes per_head = format == 0 ? 8 * dim : format == 1 ? 2 * (dim + 4) : 2 * (dim + dim / 64 * 2);
                REQUIRE(CP::estimate(g, m));
                REQUIRE(m.logical_kv == ctx * 3 * 7 * per_head);
                REQUIRE(m.initial_backing <= m.full_backing && m.logical_kv <= m.full_backing);
                REQUIRE(m.fallback_fp16 == ctx * 3 * dim * 4);
                g.growth = false; g.granularity = 0;
                REQUIRE(CP::estimate(g, m));
                REQUIRE(m.initial_rows == ctx && m.initial_backing == m.logical_kv && m.full_backing == m.logical_kv);
            }

    // Including partial rows and head-boundary pages shared by adjacent heads.
    unsigned masks = 0;
    for (const CP::Bytes ctx : {1ull, 17ull, 63ull, 1001ull})
        for (const CP::Bytes heads : {1ull, 2ull, 7ull})
            for (const CP::Bytes row : {2ull, 3ull, 8ull, 64ull})
                for (const CP::Bytes gran : {1ull, 64ull, 4096ull})
                    for (const CP::Bytes rows : std::initializer_list<CP::Bytes>{0, 1, ctx / 2, ctx}) {
                        CP::Bytes actual = 99;
                        REQUIRE(CP::prefix_bytes(ctx, rows, heads, row, gran, actual));
                        REQUIRE(actual == mask_oracle(ctx, rows, heads, row, gran));
                        ++masks;
                    }

    q.kv_format = 3; q.tiled_prompt = true;
    CP::Plan roomy;
    REQUIRE(CP::plan(24 * CP::GiB, 11 * CP::GiB, 768 * CP::MiB, q, roomy));
    REQUIRE(roomy.chunk == 4096 && roomy.full_resident_fits);
    const CP::Bytes delta = 3045064704ull - 95158272ull; // actual full256K minus8K byte KV
    REQUIRE(roomy.growth_extra == delta);
    REQUIRE(roomy.required == 11 * CP::GiB + 11 * CP::GiB / 4 + 1792 * CP::MiB + delta);
    CP::Plan edge;
    REQUIRE(CP::plan(roomy.required, 11 * CP::GiB, 768 * CP::MiB, q, edge) && edge.chunk == 4096);
    REQUIRE(CP::plan(roomy.required - 1, 11 * CP::GiB, 768 * CP::MiB, q, edge) && edge.chunk == 2048);
    // Including loaded MTP Experts preserves drafter residency rather than
    // spending their room on a larger prompt. Exercise the exact boundary.
    CP::Bytes main_plus_mtp = 0;
    REQUIRE(CP::add(11 * CP::GiB, 1 * CP::GiB, main_plus_mtp));
    REQUIRE(CP::plan(roomy.required, main_plus_mtp, 768 * CP::MiB, q, edge));
    REQUIRE(edge.chunk == 2048 && edge.required == roomy.required && edge.full_resident_fits);
    REQUIRE(CP::plan(roomy.required - 1, main_plus_mtp, 768 * CP::MiB, q, edge) && edge.chunk == 1024);
    REQUIRE(CP::plan(0, 11 * CP::GiB, 768 * CP::MiB, q, edge) && edge.chunk == 512 && !edge.full_resident_fits);
    REQUIRE(CP::plan(24 * CP::GiB, 11 * CP::GiB, 2 * CP::GiB, q, edge));
    REQUIRE(edge.required == roomy.required + 1280 * CP::MiB);
    // The fixed legacy envelope must not credit a larger geometry's entire8K cache.
    {
        auto g=q; g.context=8192; g.kv_heads=16; g.attention_layers=32;
        g.mtp_attention=false; g.kv_format=0; g.growth=false; g.tiled_prompt=false;
        REQUIRE(CP::plan(8 * CP::GiB, 0, 768 * CP::MiB, g, edge));
        REQUIRE(edge.memory.logical_kv == 8 * CP::GiB);
        REQUIRE(edge.growth_extra == 8 * CP::GiB + 128 * CP::MiB - 392 * CP::MiB);
        REQUIRE(edge.chunk == 512 && !edge.full_resident_fits);
    }
    {
        auto g=q; g.context=8192;
        const auto max=std::numeric_limits<CP::Bytes>::max();
        REQUIRE(CP::plan(max, max - 3 * CP::GiB, 768 * CP::MiB, g, edge));
        REQUIRE(edge.chunk == 1024 && edge.required == max && edge.full_resident_fits);
    }
    CP::Bytes previous = 0;
    for (const CP::Bytes ctx : {4096ull, 8192ull, 16384ull, 65536ull, 262144ull}) {
        auto g = q; g.context = ctx;
        REQUIRE(CP::plan(24 * CP::GiB, 11 * CP::GiB, 768 * CP::MiB, g, edge));
        REQUIRE(edge.growth_extra >= previous);
        REQUIRE(edge.chunk >= 512 && edge.chunk <= 4096);
        previous = edge.growth_extra;
    }

    // Invalid geometry and all overflow paths fail without publishing partial output.
    const auto bad = [&](CP::Geometry g) {
        CP::Memory marker; marker.logical_kv = 123;
        REQUIRE(!CP::estimate(g, marker)); REQUIRE(marker.logical_kv == 123);
    };
    { auto g=q; g.context=0; bad(g); }
    { auto g=q; g.kv_heads=0; bad(g); }
    { auto g=q; g.kv_heads=65537; bad(g); }
    { auto g=q; g.head_dim=65; bad(g); }
    { auto g=q; g.kv_format=-1; bad(g); }
    { auto g=q; g.kv_format=4; bad(g); }
    { auto g=q; g.granularity=0; bad(g); }
    { auto g=q; g.context=std::numeric_limits<CP::Bytes>::max(); bad(g); }
    { auto g=q; g.attention_layers=std::numeric_limits<CP::Bytes>::max(); bad(g); }
    { auto g=q; g.kv_format=0; g.head_dim=std::numeric_limits<CP::Bytes>::max(); bad(g); }
    REQUIRE(!CP::plan(24*CP::GiB, std::numeric_limits<CP::Bytes>::max(), 768*CP::MiB, q, edge));
    CP::Bytes marker=123;
    REQUIRE(!CP::prefix_bytes(17,18,1,3,64,marker) && marker==123);
    std::printf("PASS Qwen chunk policy: 4 KV formats, MTP cache/drafter residency, 60 shapes, %u page unions, growth/fallback, planner bounds and overflow\n", masks);
}
