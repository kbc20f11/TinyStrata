#pragma once
#include <string>
#include <string_view>
#include <cstdint>
#include <limits>

namespace strata::vision {
// Match the image decoder's dimension ceiling and signed RGB byte-size bound.
inline constexpr uint32_t max_count_dimension = 1u << 24;
inline constexpr uint64_t max_count_pixels = uint64_t(std::numeric_limits<int32_t>::max()) / 3;

inline bool parse_count(std::string_view line, uint32_t& width, uint32_t& height) {
    if (line.substr(0, 6) != "COUNT\t") return false;
    line.remove_prefix(6);
    const auto split = line.find('\t');
    if (split == std::string_view::npos) return false;
    const auto number = [](std::string_view text, uint32_t& result) {
        if (text.empty() || text.size() > 8) return false;
        uint32_t value = 0;
        for (char c : text) {
            if (c < '0' || c > '9') return false;
            const uint32_t digit = uint32_t(c - '0');
            if (value > (max_count_dimension - digit) / 10) return false;
            value = value * 10 + digit;
        }
        if (value == 0) return false;
        result = value;
        return true;
    };
    uint32_t w = 0, h = 0;
    if (!number(line.substr(0, split), w) || !number(line.substr(split + 1), h) ||
        uint64_t(w) * h > max_count_pixels) return false;
    width = w; height = h;
    return true;
}

inline bool parse_enc(std::string_view line, std::string& image, std::string& output) {
    std::string_view first, second;
    if (line.substr(0, 4) == "ENC\t") {
        line.remove_prefix(4);
        const auto split = line.find('\t');
        if (split == std::string_view::npos) return false;
        first = line.substr(0, split); second = line.substr(split + 1);
    } else if (line.substr(0, 4) == "ENC ") {
        line.remove_prefix(4);
        const auto split = line.rfind(' ');
        if (split == std::string_view::npos) return false;
        first = line.substr(0, split); second = line.substr(split + 1);
    } else return false;
    const auto valid = [](std::string_view value) {
        return !value.empty() && value.find_first_of(std::string_view("\0\r\n\t", 4)) == std::string_view::npos;
    };
    if (!valid(first) || !valid(second)) return false;
    image.assign(first); output.assign(second);
    return true;
}
}
