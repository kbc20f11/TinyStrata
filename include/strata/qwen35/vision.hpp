// Qwen multimodal input adapted from Strata's GENI/SVE1 path (MIT; see LICENSE and NOTICE).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace strata::qwen35::vision {
constexpr int32_t kImagePad = 248056;
struct Image {
    size_t cell = 0, offset = 0;
    int32_t rows = 0, nx = 0, ny = 0;
};
struct Input {
    std::vector<float> embeddings;
    std::vector<Image> images;
    // Interleaved [cell][temporal, height, width], including future generated cells.
    std::vector<int32_t> positions;
};

inline bool whole_prefix(const Input& input, size_t end) {
    for (const Image& image : input.images)
        if (image.cell < end && end - image.cell < (size_t) image.rows) return false;
    return true;
}

// A run of image-pad IDs does not identify a picture. Check the validated
// embedding rows and their grid/position instead. The old input remains owned
// by the engine until this comparison, so no second image cache or hash is
// needed. Pictures appended after the reused prefix do not invalidate it.
inline bool same_prefix(const Input& old, const Input& next, size_t end, int width) {
    if (width <= 0 || !whole_prefix(old, end) || !whole_prefix(next, end)) return false;
    size_t a = 0, b = 0;
    while (a < old.images.size() && old.images[a].cell < end) {
        if (b >= next.images.size()) return false;
        const Image& x = old.images[a++];
        const Image& y = next.images[b++];
        if (x.cell != y.cell || x.rows != y.rows || x.nx != y.nx || x.ny != y.ny || x.rows <= 0) return false;
        const size_t n = (size_t) x.rows * (size_t) width;
        if (x.offset > old.embeddings.size() || n > old.embeddings.size() - x.offset ||
            y.offset > next.embeddings.size() || n > next.embeddings.size() - y.offset ||
            std::memcmp(old.embeddings.data() + x.offset, next.embeddings.data() + y.offset,
                        n * sizeof(float)) != 0) return false;
    }
    if (b < next.images.size() && next.images[b].cell < end) return false;
    if ((!old.positions.empty() && old.positions.size() / 3 < end) ||
        (!next.positions.empty() && next.positions.size() / 3 < end)) return false;
    if (!old.positions.empty() && !next.positions.empty())
        return std::equal(old.positions.begin(), old.positions.begin() + end * 3, next.positions.begin());
    // Text-only inputs have the identity position table, even when a previous
    // request had an image later than this checkpoint.
    const auto& positions = old.positions.empty() ? next.positions : old.positions;
    if (!positions.empty())
        for (size_t cell = 0; cell < end; ++cell)
            for (size_t axis = 0; axis < 3; ++axis)
                if (positions[cell * 3 + axis] != (int32_t) cell) return false;
    return true;
}

inline std::vector<int32_t> identity_positions(int context) {
    std::vector<int32_t> positions((size_t) context * 3);
    for (int cell = 0; cell < context; ++cell)
        positions[(size_t) cell * 3] = positions[(size_t) cell * 3 + 1] = positions[(size_t) cell * 3 + 2] = cell;
    return positions;
}

// SVE1: int32 magic, rows, nx, ny, embedding width; then row-major FP32 embeddings.
// Bound every allocation by image-pad tokens already present in a validated prompt.
inline bool load(const std::string& path, const std::vector<int32_t>& ids, int width, int context,
                 Input& result, std::string& error, int32_t image_pad = kImagePad) {
    Input input;
    const size_t pads = (size_t) std::count(ids.begin(), ids.end(), image_pad);
    if (width <= 0 || context <= 0 || ids.empty() || ids.size() > (size_t) context || pads == 0) {
        error = "the image prompt has no image-pad tokens or exceeds the context";
        return false;
    }
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
    if (!file) { error = "cannot open image embeddings file"; return false; }
    size_t remaining = pads;
    while (file.peek() != std::char_traits<char>::eof()) {
        int32_t header[5] = {};
        file.read(reinterpret_cast<char*>(header), sizeof header);
        if (!file || header[0] != 0x31455653 || header[1] < 1 || header[2] < 1 || header[3] < 1 ||
            (int64_t) header[2] * header[3] != header[1] || header[4] != width || (size_t) header[1] > remaining) {
            error = "bad image embeddings file (expected SVE1 records matching image-pad tokens and model width)";
            return false;
        }
        Image image;
        image.rows = header[1]; image.nx = header[2]; image.ny = header[3]; image.offset = input.embeddings.size();
        const size_t count = (size_t) image.rows * (size_t) width;
        input.embeddings.resize(image.offset + count);
        file.read(reinterpret_cast<char*>(input.embeddings.data() + image.offset), (std::streamsize) (count * sizeof(float)));
        if (!file) { error = "short image embeddings file"; return false; }
        for (size_t i = image.offset; i < input.embeddings.size(); ++i)
            if (!std::isfinite(input.embeddings[i])) { error = "non-finite image embedding"; return false; }
        input.images.push_back(image);
        remaining -= (size_t) image.rows;
    }
    if (file.bad()) { error = "cannot read image embeddings file"; return false; }
    input.positions.resize((size_t) context * 3);
    auto put = [&](size_t cell, int32_t t, int32_t h, int32_t w) {
        input.positions[cell * 3] = t; input.positions[cell * 3 + 1] = h; input.positions[cell * 3 + 2] = w;
    };
    size_t cell = 0, image_index = 0;
    int32_t position = 0;
    while (cell < ids.size()) {
        if (ids[cell] != image_pad) { put(cell++, position, position, position); ++position; continue; }
        if (image_index >= input.images.size()) { error = "the prompt has more images than the embeddings file"; return false; }
        Image& image = input.images[image_index++];
        image.cell = cell;
        if ((size_t) image.rows > ids.size() - cell) { error = "fewer image-pad tokens than image rows"; return false; }
        for (int32_t row = 0; row < image.rows; ++row) {
            if (ids[cell + (size_t) row] != image_pad) { error = "fewer image-pad tokens than image rows"; return false; }
            put(cell + (size_t) row, position, position + row / image.nx, position + row % image.nx);
        }
        cell += (size_t) image.rows;
        position += std::max(image.nx, image.ny);
    }
    if (image_index != input.images.size() || remaining != 0) {
        error = "the embeddings file does not match the prompt's images";
        return false;
    }
    if (ids.back() == image_pad) { error = "the prompt cannot end in an image"; return false; }
    for (; cell < (size_t) context; ++cell, ++position) put(cell, position, position, position);
    result = std::move(input);
    return true;
}
} // namespace strata::qwen35::vision
