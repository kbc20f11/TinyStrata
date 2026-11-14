#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strata::intel {
// Preserve every Q6_K byte; only pad each 210-byte block to 53 words and
// interleave those words across 16 output rows for coalesced XMX loads.
inline size_t quant_interleaved_offset(size_t source_byte, int width, int block_bytes) {
    const size_t blocks = size_t(width / 256), row_bytes = blocks * block_bytes;
    const size_t row = source_byte / row_bytes, within = source_byte % row_bytes;
    return ((((row / 16) * blocks + within / block_bytes) * ((block_bytes + 3) / 4) + (within % block_bytes) / 4) * 16 + row % 16) * 4 + within % block_bytes % 4;
}
inline size_t quant_interleaved_bytes(int rows, int width, int block_bytes) {
    if (rows < 1 || rows % 16 || width < 256 || width % 256 || (block_bytes != 82 && block_bytes != 110 && block_bytes != 210))
        throw std::invalid_argument("Quant interleave needs complete 16-row/256-column tiles and a supported block size");
    return size_t(rows) * size_t(width / 256) * ((block_bytes + 3) / 4 * 4);
}
inline void interleave_quant(const uint8_t* source, uint8_t* target, int rows, int width, int block_bytes) {
    const size_t bytes = quant_interleaved_bytes(rows, width, block_bytes);
    std::memset(target, 0, bytes);
    // source is a bounded batch of complete row tiles, never a model replica.
    const int blocks = width / 256, words = (block_bytes + 3) / 4;
    for (int tile = 0; tile < rows / 16; ++tile) for (int block = 0; block < blocks; ++block)
        for (int word = 0; word < words; ++word) for (int row = 0; row < 16; ++row)
            std::memcpy(target + (((size_t(tile) * blocks + block) * words + word) * 16 + row) * 4,
                source + (size_t(tile * 16 + row) * blocks + block) * block_bytes + word * 4,
                size_t(block_bytes - word * 4 < 4 ? block_bytes - word * 4 : 4));
}
inline size_t q6_interleaved_offset(size_t byte, int width) { return quant_interleaved_offset(byte, width, 210); }
inline size_t q6_interleaved_bytes(int rows, int width) { return quant_interleaved_bytes(rows, width, 210); }
inline void interleave_q6(const uint8_t* source, uint8_t* target, int rows, int width) { interleave_quant(source, target, rows, width, 210); }
}
