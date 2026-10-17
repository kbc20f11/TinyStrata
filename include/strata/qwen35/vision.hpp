// Qwen multimodal input adapted from Strata's GENI/SVE1 path (MIT; see LICENSE and NOTICE).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
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
