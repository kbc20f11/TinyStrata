// CPU-only image/text scheduling and checkpoint boundaries; no model/GPU.
#include "strata/gemma4/image_text_prefill.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Image { int64_t at; int n; };
using strata::gemma4::image_prompt_piece;
static int checks = 0, schedules = 0;
#define REQUIRE(x) do { ++checks; if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

bool intersects(int64_t at, int n, const Image& im) {
    return at < im.at + im.n && at + n > im.at;
}

// Independent image-call contract: stop before the next image; an image
// starting at this position is one indivisible eval(), even with a small cap.
int whole(int64_t at, int limit, const std::vector<Image>& images) {
    for (const auto& im : images) {
        if (at == im.at) return im.n;
        REQUIRE(!(at > im.at && at < im.at + im.n));
        if (im.at > at && im.at < at + limit) return (int) (im.at - at);
    }
    return limit;
}

void schedule(int64_t begin, int64_t final, int rows, bool enabled,
              const std::vector<Image>& images, int64_t mark) {
    int64_t at = begin;
    bool saved = mark <= begin;
    int pictures = 0, expected_pictures = 0, saves = 0;
    for (const auto& im : images) expected_pictures += im.at >= begin && im.at < final;
    while (at < final) {
        const auto piece = image_prompt_piece(at, (int) (final - at), rows, rows * 4,
            true, enabled, images.data(), images.size(), saved ? -1 : mark);
        REQUIRE(piece.limit > 0);
        REQUIRE(piece.limit <= final - at);
        REQUIRE(piece.limit <= (piece.read ? rows * 4 : rows));
        const int n = whole(at, piece.limit, images);
        REQUIRE(n > 0 && n <= final - at);
        bool image_call = false;
        for (const auto& im : images) if (intersects(at, n, im)) {
            REQUIRE(!piece.read);
            REQUIRE(at == im.at && n == im.n);
            image_call = true;
        }
        pictures += image_call;
        if (!enabled) REQUIRE(!piece.read);
        if (piece.read) REQUIRE(n == piece.limit);
        if (!saved && mark > at && mark < at + n) {
            REQUIRE(piece.read); // read() records its interior checkpoint.
            saved = true; ++saves;
        }
        at += n;
        if (!saved && at == mark) { saved = true; ++saves; }
        // This committed boundary can be used to restart the planner after a
        // cancel/retry; all image positions remain absolute, not suffix-local.
        if (at < final) {
            const auto next = image_prompt_piece(at, (int) (final - at), rows, rows * 4,
                true, enabled, images.data(), images.size(), saved ? -1 : mark);
            REQUIRE(next.limit > 0);
        }
    }
    REQUIRE(at == final); // The final logits token has not been consumed.
    REQUIRE(pictures == expected_pictures);
    if (mark > begin && mark <= final) REQUIRE(saved && saves == 1);
    ++schedules;
}

int main() {
    // One staging round must hold every non-resident Expert of a layer. Exact
    // fit succeeds, the next Expert fails, independently of GPU name or VRAM.
    using strata::gemma4::image_text_stage_fits;
    size_t bytes = 0;
    const size_t gate = 1024 * 1024, down = 1024 * 1024 + 200000, one = 2 * gate + down;
    const size_t stage = 80 * one;
    REQUIRE(image_text_stage_fits(stage, 64, gate, down, &bytes) && bytes == 64 * one);
    REQUIRE(image_text_stage_fits(stage, 80, gate, down, &bytes) && bytes == stage);
    REQUIRE(!image_text_stage_fits(stage, 81, gate, down, &bytes) && bytes == 81 * one);
    REQUIRE(!image_text_stage_fits(stage, 124, gate, down));
    REQUIRE(!image_text_stage_fits(stage - 1, 80, gate, down));
    REQUIRE(image_text_stage_fits(0, 0, gate, down, &bytes) && bytes == 0);
    REQUIRE(!image_text_stage_fits(0, 1, gate, down));
    REQUIRE(!image_text_stage_fits(stage, 1, 0, 0, &bytes));
    const size_t maximum = std::numeric_limits<size_t>::max();
    REQUIRE(!image_text_stage_fits(maximum, 1, maximum, 1, &bytes) && bytes == maximum);
    REQUIRE(!image_text_stage_fits(maximum, maximum, 1, 1, &bytes) && bytes == maximum);
    REQUIRE(image_text_stage_fits(maximum, maximum, 0, 1, &bytes) && bytes == maximum);
    REQUIRE(!image_text_stage_fits(maximum - 1, maximum, 0, 1, &bytes) && bytes == maximum);
    const std::vector<Image> images{{4096, 1120}, {9500, 64}};
    auto plan = [&](int64_t at, int left, bool on, int64_t mark = -1) {
        return image_prompt_piece(at, left, 1120, 16384, true, on, images.data(), images.size(), mark);
    };
    auto p = plan(0, 17000, false); REQUIRE(!p.read && p.limit == 1120);
    p = plan(0, 17000, true); REQUIRE(p.read && p.limit == 4096);
    p = plan(0, 17000, true, 1234); REQUIRE(p.read && p.limit == 4096);
    p = plan(4090, 12910, true); REQUIRE(!p.read && p.limit == 1120);
    REQUIRE(whole(4090, p.limit, images) == 6);
    p = plan(4096, 12904, true); REQUIRE(!p.read && p.limit == 1120);
    p = plan(5216, 11784, true); REQUIRE(p.read && p.limit == 4284);
    p = plan(9564, 7436, true); REQUIRE(p.read && p.limit == 7436);
    // Exact threshold, shorter interval and a suffix ending before an image.
    p = plan(2976, 10000, true); REQUIRE(p.read && p.limit == 1120);
    p = plan(2977, 10000, true); REQUIRE(!p.read && p.limit == 1120);
    p = plan(0, 1500, true); REQUIRE(p.read && p.limit == 1500);
    p = plan(0, 17, true, 5); REQUIRE(!p.read && p.limit == 5);
    // Flag OFF retains every legacy piece cap, including checkpoint caps.
    p = plan(0, 17000, false, 37); REQUIRE(!p.read && p.limit == 37);
    // Missing span information fails back to legacy eval() even with flag ON.
    p = image_prompt_piece<Image>(0, 2000, 1024, 16384, true, true, nullptr, 0);
    REQUIRE(!p.read && p.limit == 1024);
    // A cached image is behind the suffix; fresh_pictures=false is still the
    // pre-existing ordinary text reader regardless of the screening switch.
    for (bool enabled : {false, true}) {
        p = image_prompt_piece(10000, 17000, 1120, 16384, false, enabled, images.data(), images.size(), 12345);
        REQUIRE(p.read && p.limit == 16384);
    }
    // Image at the start/end and multiple pictures, with nonzero reuse and
    // every legal checkpoint. Include tiny one-row pictures and no text gap.
    for (int rows : {8, 16, 32}) for (int64_t origin : {0, 5, 10000})
        for (int before : {0, 1, rows - 1, rows, rows * 5})
            for (int width : {1, rows}) for (int gap : {0, 1, rows * 5}) {
                std::vector<Image> spans{{origin + before, width}, {origin + before + width + gap, 1}};
                const int64_t final = spans.back().at + spans.back().n + rows * 5 + 1;
                for (int64_t begin : {origin, spans.front().at + spans.front().n, spans.back().at + spans.back().n})
                    for (int64_t mark = begin + 1; mark <= final; ++mark) {
                        bool legal = true;
                        for (const auto& im : spans) if (mark > im.at && mark < im.at + im.n) legal = false;
                        if (legal) for (bool enabled : {false, true}) schedule(begin, final, rows, enabled, spans, mark);
                    }
            }
    std::printf("gemma image text prefill: %d schedules, %d checks passed\n", schedules, checks);
}
