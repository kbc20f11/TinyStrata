#pragma once
#include "strata/intel/turn_checkpoint.hpp"
#include "strata/qwen35/vision.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace strata::intel::qwen_vision {
namespace canonical = strata::qwen35::vision;
using Input = canonical::Input;
constexpr uint64_t max_embedding_bytes = 256ull << 20;
inline bool clover_workspace_aliases(bool clover,bool vision,int mtp) {
    return clover&&vision&&mtp==0;
}
// Full-attention query gates and GDN gates belong to mutually exclusive
// layer branches. Return a non-owning view only for exactly equal row extents.
inline float* clover_layer_gate_alias(float* recurrent_gate,size_t query_width,size_t recurrent_width) {
    if(!recurrent_gate||!query_width||query_width!=recurrent_width)
        throw std::logic_error("Clover layer gate workspace extents differ");
    return recurrent_gate;
}
// Intel's validated Qwen-family geometry is 2,048-wide. Even a one-row
// picture consumes one pad, bounding the old/new record arrays by this count.
inline uint64_t host_metadata_reservation(uint64_t context) {
    return 2 * std::min(context, max_embedding_bytes / sizeof(float) / 2048) * sizeof(canonical::Image);
}
// Admit vision before the compressed weight arena is allocated. The actual
// tokenizer spelling takes precedence over a declared PLE token identity.
inline int admit_metadata(int width, double rotation, bool image_rope, int vocabulary,
                          int64_t actual_pad, double declared_pad = -1) {
    int pad = -1;
    if (actual_pad >= 0 && actual_pad < vocabulary) pad = int(actual_pad);
    else if (std::isfinite(declared_pad) && declared_pad >= 0 && declared_pad < vocabulary &&
             declared_pad == std::floor(declared_pad)) pad = int(declared_pad);
    if (width != 2048 || rotation != 64 || !image_rope || pad < 0)
        throw std::runtime_error("Intel vision requires 2048-wide Qwen image input, a real image-pad token and 11/11/10 M-RoPE");
    return pad;
}
inline bool valid_image_keys(const std::string& keys) {
    return !keys.empty() && keys.size() <= 8192 && keys.front() != ',' && keys.back() != ',' &&
        keys.find(",,") == std::string::npos && keys.find_first_not_of("0123456789abcdef,") == std::string::npos;
}

// Validate the total size before the shared SVE1 reader allocates any rows.
// Old and incoming inputs must coexist until exact image-prefix comparison.
inline bool load(const std::string& path, const std::vector<int>& ids, int width,
                 int context, int image_pad, Input& result, std::string& error) {
    const auto rows = size_t(std::count(ids.begin(), ids.end(), image_pad));
    if (width <= 0 || rows > max_embedding_bytes / sizeof(float) / uint64_t(width)) {
        error = "image embeddings exceed the 256 MiB request limit";
        return false;
    }
    return canonical::load(path, ids, width, context, result, error, image_pad, true);
}

inline bool overlaps(const Input& input, size_t begin, size_t count) {
    for (const auto& image : input.images)
        if (image.cell < begin + count && begin < image.cell + size_t(image.rows)) return true;
    return false;
}

inline bool saved_prefix(const std::vector<int>& live, const std::vector<int>& prompt,
                         const std::vector<int>& saved, size_t position, bool enabled) {
    return enabled && position && position == saved.size() && saved.size() <= live.size() &&
        saved.size() <= prompt.size() && std::equal(saved.begin(), saved.end(), live.begin()) &&
        std::equal(saved.begin(), saved.end(), prompt.begin());
}

// Image-pad token IDs alone cannot identify a picture. A changed late image
// can still reuse an earlier complete turn before that picture.
inline turn_checkpoint::Reuse choose(const std::vector<int>& live, const std::vector<int>& prompt,
    const std::vector<int>& saved, size_t position, bool enabled,
    const Input& old, const Input& next, int width) {
    if (live.size() <= prompt.size() && std::equal(live.begin(), live.end(), prompt.begin()) &&
        canonical::same_prefix(old, next, live.size(), width))
        return {turn_checkpoint::Source::live, live.size()};
    if (saved_prefix(live, prompt, saved, position, enabled) &&
        canonical::same_prefix(old, next, saved.size(), width))
        return {turn_checkpoint::Source::saved_turn, saved.size()};
    return {};
}
} // namespace strata::intel::qwen_vision
