// CPU-only MMQ row-map bounds regression. No GPU runtime, model files or dependencies.
#include "strata/prefill/mmq_ids_margin.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace M = strata::prefill::mmq;
static bool legacy = false;
static size_t checks = 0, fixtures = 0, old_outside = 0;
#define REQUIRE(x) do { ++checks; if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

// Independent reader bound: the main kernel reads complete tiles of each nonempty expert. Stream-k fixup
// may read any tile of the widest expert, even for an empty expert at the end; its write test comes AFTER
// the ID load. This deliberately does not call the production round-up helper to compute the read indices.
static void read_span(size_t base, size_t count, size_t widest, int J, size_t capacity, size_t old_capacity) {
    for (size_t tile = 0; tile < std::max(count, widest); tile += (size_t) J) {
        const size_t first = base + tile, last = first + (size_t) J - 1;
        if (last >= old_capacity) ++old_outside;
        REQUIRE(first < capacity && last < capacity);
    }
}

static void fixture(const std::vector<size_t>& counts, size_t token_capacity, size_t top_k) {
    const size_t logical_capacity = token_capacity * top_k;
    M::IdsLayout ids;
    REQUIRE(M::ids_layout(logical_capacity, token_capacity, ids));
    REQUIRE(ids.logical_rows == logical_capacity);
    REQUIRE(ids.storage_bytes == ids.storage_rows * sizeof(int32_t));
    const size_t used = std::accumulate(counts.begin(), counts.end(), size_t(0));
    const size_t widest = *std::max_element(counts.begin(), counts.end());
    REQUIRE(used <= logical_capacity && widest <= token_capacity);
    const size_t capacity = legacy ? logical_capacity : ids.storage_rows;

    // Explicit-startup and automatic q36 workspace share this exact layout; inverse routing stays logical.
    std::vector<int32_t> identity(ids.storage_rows), inverse(ids.logical_rows);
    std::iota(identity.begin(), identity.end(), 0);
    std::iota(inverse.begin(), inverse.end(), 0);
    for (size_t i = 0; i < logical_capacity; ++i) REQUIRE(identity[i] == inverse[i]);
    for (int J = 8; J <= 128; J += 8) {
        size_t at = 0;
        for (size_t rows : counts) {
            read_span(at, rows, widest, J, capacity, logical_capacity);
            at += rows;
        }
    }

    // Gemma/LFM pack a destination map followed by bounds/weights. Their staged down launches use
    // interior subspans of this map: [row0, row0+rows), with bounds relative to row0.
    M::PackedIdsLayout p;
    REQUIRE(M::packed_ids_layout(logical_capacity, token_capacity, counts.size(), 3, p));
    REQUIRE(p.ids.storage_rows == ids.storage_rows);
    REQUIRE(p.place == logical_capacity && p.bounds == p.place + ids.storage_rows);
    REQUIRE(p.relative == p.bounds + counts.size() + 1);
    REQUIRE(p.weights == p.relative + 2 * counts.size() + 2);
    REQUIRE(p.classes == p.weights + logical_capacity);
    REQUIRE(p.words == p.classes + 3 * (counts.size() + 1));
    const size_t old_words = 3 * logical_capacity + 3 * counts.size() + 3 + 3 * (counts.size() + 1);
    REQUIRE(p.words - old_words == ids.storage_rows - logical_capacity);
    std::vector<int32_t> packed(p.words + 8, 0);
    for (size_t i = p.bounds; i < packed.size(); ++i) packed[i] = 0x51515151;
    for (size_t i = 0; i < used; ++i) packed[p.place + i] = (int32_t) (used - i - 1);
    for (size_t i = used; i < ids.storage_rows; ++i) REQUIRE(packed[p.place + i] == 0);
    for (size_t begin = 0; begin < counts.size(); ++begin) {
        const size_t row0 = std::accumulate(counts.begin(), counts.begin() + begin, size_t(0));
        size_t at = row0;
        for (size_t e = begin; e < counts.size(); ++e) {
            for (int J = 8; J <= 128; J += 8)
                read_span(at, counts[e], widest, J, capacity, logical_capacity);
            at += counts[e];
        }
    }
    for (size_t i = p.bounds; i < packed.size(); ++i) REQUIRE(packed[i] == 0x51515151);
    ++fixtures;
}

int main(int argc, char** argv) {
    legacy = argc == 2 && std::strcmp(argv[1], "--legacy-exact") == 0;
    REQUIRE(M::kMmqTileColsMax == 128 && M::kMmqTileColsStep == 8); // pinned llama.cpp switch_J bound

    // Historical late-empty-expert failure, plus awkward and full-capacity layouts. --legacy-exact must
    // fail this first fixture, independently proving the old exact-sized table does not cover the reads.
    fixture({100, 100, 0, 0}, 100, 2);
    for (size_t T : {size_t(1), size_t(7), size_t(31), size_t(127), size_t(128), size_t(129),
                     size_t(255), size_t(512), size_t(1024), size_t(1120), size_t(2048), size_t(4096), size_t(32768)}) {
        for (size_t K : {size_t(1), size_t(2), size_t(8), size_t(10)}) {
            std::vector<size_t> rows(K + 3, T);
            rows[K] = rows[K + 1] = rows[K + 2] = 0; // full capacity, followed by empty experts
            fixture(rows, T, K);
            std::fill(rows.begin(), rows.end(), 0);
            fixture(rows, T, K); // no rows: no MMQ launch and no speculative read
            for (size_t e = 0; e < K; ++e) rows[e] = (T + 3 * e) % (T + 1);
            fixture(rows, T, K); // sparse/uneven routing and a short final prompt chunk
        }
    }

    // Dense HIP MMQ packs its fixed identity map immediately before bounds: even its last partial tile
    // must remain in the map's initialized span, never in the bounds or outside the allocation.
    M::IdsLayout dense;
    REQUIRE(M::ids_layout(1024, 1024, dense));
    const size_t bounds_word = (dense.storage_bytes + 255) / 256 * 256 / sizeof(int32_t);
    REQUIRE(bounds_word >= dense.storage_rows);
    for (size_t rows : {size_t(1), size_t(33), size_t(1023), size_t(1024)})
        for (int J = 8; J <= 128; J += 8) read_span(0, rows, rows, J, bounds_word, 1024);

    // Overflow/invalid shapes fail without publishing a partial layout or clobbering the output.
    int64_t margin = 987;
    REQUIRE(!M::mmq_ids_margin_rows(-1, 128, &margin) && margin == 987);
    REQUIRE(!M::mmq_ids_margin_rows(10, 0, &margin) && margin == 987);
    REQUIRE(!M::mmq_ids_margin_rows(INT64_MAX, 128, &margin) && margin == 987);
    REQUIRE(!M::mmq_ids_margin_rows(1, 128, nullptr));
    REQUIRE(M::mmq_ids_table_rows(UINT64_MAX, 1) == 0);
    REQUIRE(M::mmq_ids_table_rows(1, UINT64_MAX) == 0);
    M::IdsLayout keep{11, 22, 88};
    REQUIRE(!M::ids_layout(INT32_MAX, 1, keep));
    REQUIRE(keep.logical_rows == 11 && keep.storage_rows == 22 && keep.storage_bytes == 88);
    M::PackedIdsLayout p;
    REQUIRE(!M::packed_ids_layout(32, 32, UINT64_MAX, 3, p));
    REQUIRE(!M::packed_ids_layout(32, 32, 256, UINT64_MAX, p));
    REQUIRE(!M::packed_ids_layout(32, 32, INT32_MAX, INT32_MAX, p));
    REQUIRE(old_outside > 0);
    std::printf("mmq_ids_margin: PASS %zu fixtures, %zu checks; old exact-span over-read intervals=%zu\n",
                fixtures, checks, old_outside);
    return 0;
}
