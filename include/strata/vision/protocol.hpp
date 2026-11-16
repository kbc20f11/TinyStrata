#pragma once
#include <string>
#include <string_view>

namespace strata::vision {
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
