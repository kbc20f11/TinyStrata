#include "strata/intel/turn_checkpoint.hpp"
#include <iostream>
#include <stdexcept>
using namespace strata::intel::turn_checkpoint;
void require(bool value) { if (!value) throw std::runtime_error("Turn checkpoint policy regression"); }
int main() {
    const std::vector<int> live{1,2,3,4,5}, saved{1,2,3};
    auto check = [&](std::vector<int> prompt, std::vector<int> cp, size_t pos, bool on, Source kind, size_t tokens) {
        const auto result = choose(live,prompt,cp,pos,on);
        require(result.source == kind && result.tokens == tokens);
    };
    check({1,2,3,4,5,6},saved,3,true,Source::live,5);
    check(live,saved,3,true,Source::live,5);
    check({1,2,3,8},saved,3,true,Source::saved_turn,3);
    check({1,2,3},saved,3,true,Source::saved_turn,3);
    check({1,2},saved,3,true,Source::reset,0);
    check({1,9,3,8},saved,3,true,Source::reset,0);
    check({1,2,3,8},{1,2,9},3,true,Source::reset,0);
    check({1,2,3,8},saved,0,true,Source::reset,0);
    check({1,2,3,8},saved,2,true,Source::reset,0);
    check({1,2,3,8},saved,3,false,Source::reset,0);
    check({1,2,3,8},{1,2,3,4,5,6},6,true,Source::reset,0);
    require(choose({}, {1,2}, {}, 0, true).source == Source::live);
    require(enabled(nullptr) && enabled("1") && !enabled("0"));
    for (const auto invalid : {"", "2", "true", "-1"}) {
        bool rejected=false;
        try { (void)enabled(invalid); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected);
    }
    std::cout << "turn checkpoint prefix, opt-out and invalid setting checks passed\n";
}
