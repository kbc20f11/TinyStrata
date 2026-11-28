#pragma once
#include <array>
#include <string>
#include <utility>

namespace strata::intel::attention_usage {
enum Route : unsigned {
    int8_split = 1u << 0, int8_fused_gqa = 1u << 1, int8_token_reuse = 1u << 2,
    int8_tiled = 1u << 3, int8_expanded_tiled = 1u << 4, int8_expanded_mkl = 1u << 5,
    fp32_split = 1u << 6, fp32_tiled = 1u << 7, fp32_mkl = 1u << 8,
    int8_split_pv = 1u << 9,
};

inline std::string format(unsigned routes) {
    static constexpr std::array names{
        std::pair{int8_split,"int8_split"},std::pair{int8_fused_gqa,"int8_fused_gqa"},
        std::pair{int8_token_reuse,"int8_token_reuse"},std::pair{int8_tiled,"int8_tiled"},
        std::pair{int8_expanded_tiled,"int8_expanded_tiled"},std::pair{int8_expanded_mkl,"int8_expanded_mkl"},
        std::pair{fp32_split,"fp32_split"},std::pair{fp32_tiled,"fp32_tiled"},
        std::pair{fp32_mkl,"fp32_mkl"},std::pair{int8_split_pv,"int8_split_pv"},
    };
    std::string result;
    for (auto [route,name] : names) {
        if (!(routes & route)) continue;
        if (!result.empty()) result += ',';
        result += name;
    }
    return result.empty() ? "none" : result;
}
}
