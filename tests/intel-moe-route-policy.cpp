#include "strata/intel/moe_route_policy.hpp"
#include <array>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>

namespace p = strata::intel::moe_route_policy;
static int checks = 0;
static void check(bool ok) { ++checks; if (!ok) throw std::runtime_error("MoE route policy check failed"); }
template<class F> static void rejects(F f) { bool failed = false; try { f(); } catch (const std::invalid_argument&) { failed = true; } check(failed); }
static std::array<int32_t,256> uniform(int tokens, int active) {
    std::array<int32_t,256> result{};
    for (int slot = 0; slot < tokens * 8; ++slot) ++result[slot % active];
    return result;
}
static std::array<int32_t,256> recorded_route(int tokens, int mode) {
    std::array<int32_t,256> result{};
    for (int token = 0; token < tokens; ++token) {
        std::array<bool,256> seen{};
        for (int pick = 0; pick < 8; ++pick) {
            int shift = mode == 0 ? token * 3 : 0;
            if (mode == 2) shift = pick < 2 ? 0 : pick < 4 ? (token / 2) * 3 : pick < 6 ? (token / 3) * 3 : token * 3;
            const int id = 32 * pick + shift % 32;
            check(!seen[id]); seen[id] = true; ++result[id];
        }
    }
    return result;
}
int main() { try {
    check(!p::enabled(nullptr)); check(!p::enabled("0")); check(p::enabled("1"));
    for (const char* value : {"", "2", "true", "01", " 1", "1\n"}) rejects([&] { p::enabled(value); });
    for (int tokens : {512,1024}) {
        for (int mode = 0; mode < 3; ++mode) {
            const auto counts = recorded_route(tokens,mode);
            const auto route = p::analyze(counts.data(),256,tokens,8);
            check(route.slots == unsigned(tokens * 8));
            check(route.active == (mode == 0 ? 256 : mode == 1 ? 8 : 194));
            check(p::prefer_packed_q8(route) == (mode == 0 || (tokens == 512 && mode == 2)));
            auto permuted = counts;
            for (int i = 0; i < 256; ++i) permuted[(i * 73 + 19) % 256] = counts[i];
            check(p::prefer_packed_q8(p::analyze(permuted.data(),256,tokens,8)) == p::prefer_packed_q8(route));
        }
        for (bool separate : {false,true}) {
            check(p::measured_projection(separate ? 21 : 22,separate ? 2048 : 512,separate ? 512 : 2048,separate,true,tokens,256,8));
            check(!p::measured_projection(separate ? 21 : 22,separate ? 2048 : 512,separate ? 512 : 2048,separate,false,tokens,256,8));
        }
        for (int active : {8,191,192,239,240,256}) {
            const auto counts = uniform(tokens,active);
            const auto route = p::analyze(counts.data(),256,tokens,8);
            check(p::prefer_packed_q8(route) == (tokens == 512 ? active >= 192 : active >= 240));
        }
    }
    auto peak = uniform(1024,240);
    for (int target : {64,65}) {
        auto counts = peak; const int delta = target - counts[0]; counts[0] = target;
        for (int i = 0; i < delta; ++i) --counts[1 + i % 239];
        const auto route = p::analyze(counts.data(),256,1024,8);
        check(route.active == 240 && route.peak == target); check(p::prefer_packed_q8(route) == (target == 64));
    }
    for (int hot : {1024,1025}) {
        std::array<int32_t,256> counts{};
        const int hot_experts = hot == 1024 ? 2 : 3;
        counts[0] = 512; counts[1] = hot == 1024 ? 512 : 448;
        if (hot_experts == 3) counts[2] = 65;
        for (int slot = 0; slot < 4096 - hot; ++slot) ++counts[hot_experts + slot % (192 - hot_experts)];
        const auto route = p::analyze(counts.data(),256,512,8);
        check(route.active == 192 && route.hot_slots == unsigned(hot)); check(p::prefer_packed_q8(route) == (hot == 1024));
    }
    auto invalid = uniform(512,256); invalid[0] = -1; rejects([&] { p::analyze(invalid.data(),256,512,8); });
    invalid[0] = 513; rejects([&] { p::analyze(invalid.data(),256,512,8); });
    invalid.fill(0); rejects([&] { p::analyze(invalid.data(),256,512,8); });
    rejects([&] { p::analyze(nullptr,256,512,8); });
    for (int tokens : {0,1,128,511,513,1008,1023,1025,2147483647}) {
        check(!p::measured_projection(22,512,2048,false,true,tokens,256,8));
        rejects([&] { p::analyze(invalid.data(),256,tokens,8); });
    }
    check(!p::measured_projection(22,512,2048,false,true,512,256,4));
    check(!p::measured_projection(22,512,2048,false,true,512,64,8));
    check(!p::measured_projection(22,768,2048,false,true,512,256,8));
    check(!p::measured_projection(14,512,2048,false,true,512,256,8));
    check(!p::measured_projection(22,512,2048,true,true,512,256,8));
    check(!p::prefer_packed_q8({512,256,16,0,0}));
    check(!p::prefer_packed_q8({512,256,513,4096,0}));
    check(!p::prefer_packed_q8({1024,256,32,8192,8193}));
    std::printf("MoE route policy PASS checks=%d\n",checks);return 0;
} catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; } }
