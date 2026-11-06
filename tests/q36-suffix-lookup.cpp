// Longest-suffix oracle scans the token history directly; it has no hash table,
// occurrence limit, or shared matching code with the implementation.
#include "strata/qwen35/suffix_lookup.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using strata::qwen35::SuffixLookup;
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);std::exit(1); } } while (0)
unsigned comparisons = 0;

struct Expected { int length = 0; std::vector<int32_t> tokens; };
Expected oracle(const std::vector<int32_t>& history, int minimum, int maximum, int draft) {
    Expected result;
    if (draft <= 0 || history.size() <= size_t(minimum)) return result;
    const size_t n = history.size();
    size_t best = 0;
    for (size_t end = 0; end + 1 < n; ++end) {
        size_t begin = end + 1;
        while (begin && end + 1 - begin < size_t(maximum) &&
               history[begin - 1] == history[n - (end + 1 - begin) - 1]) --begin;
        const int length = int(end + 1 - begin);
        if (length >= minimum && length >= result.length) { result.length = length; best = end; }
    }
    if (!result.length) return result;
    for (size_t i = best + 1; i < n && result.tokens.size() < size_t(std::min(draft,7)); ++i)
        result.tokens.push_back(history[i]);
    return result;
}
void check(SuffixLookup& lookup, const std::vector<int32_t>& history, int minimum, int maximum, int draft) {
    std::array<int32_t,10> output;output.fill(0x12345678);
    const Expected expected = oracle(history,minimum,maximum,draft);
    const int got = lookup.propose(draft,output.data());
    REQUIRE(lookup.size() == history.size());
    REQUIRE(got == int(expected.tokens.size()));
    REQUIRE(lookup.last_match() == expected.length);
    for (int i = 0; i < got; ++i) REQUIRE(output[size_t(i)] == expected.tokens[size_t(i)]);
    for (size_t i = size_t(got); i < output.size(); ++i) REQUIRE(output[i] == 0x12345678);
    ++comparisons;
}
template<class Exception, class F> void rejects(F function) {
    bool caught = false;
    try { function(); } catch (const Exception&) { caught = true; }
    REQUIRE(caught);
}
void fixed_cases() {
    SuffixLookup lookup(3,64,1024);
    std::vector<int32_t> history = {10,11,12,13,14,15,91};
    // The best match predates more than four newer occurrences of its trigram.
    for (int i = 0; i < 9; ++i) {
        history.insert(history.end(), {100+i,13,14,15,200+i});
    }
    history.insert(history.end(), {10,11,12,13,14,15});
    lookup.rebuild(history.data(),history.size());
    check(lookup,history,3,64,7);
    int32_t output[7];REQUIRE(lookup.propose(7,output) == 7 && output[0] == 91);
    history = {1,2,3,8,0,1,2,3,9,0,1,2,3};
    SuffixLookup tied(3,3,64);tied.append(history.data(),history.size());
    check(tied,history,3,3,7);REQUIRE(tied.propose(7,output) > 0 && output[0] == 9);
    for (int period : {1,2,3,7,11}) {
        history.clear();lookup.reset();
        for (int i = 0; i < 150; ++i) { history.push_back(i%period-4);lookup.append(history.back()); }
        for (int draft : {0,1,3,7,100}) check(lookup,history,3,64,draft);
        REQUIRE(lookup.last_match() == 64);
    }
    lookup.reset();REQUIRE(lookup.size() == 0 && lookup.last_match() == 0);
    check(lookup,{},3,64,7);
    history = {std::numeric_limits<int32_t>::min(),0,std::numeric_limits<int32_t>::max(),-1,
               std::numeric_limits<int32_t>::min(),0,std::numeric_limits<int32_t>::max()};
    lookup.rebuild(history.data(),history.size());check(lookup,history,3,64,7);
}
void randomized_oracle() {
    std::mt19937 random(0x4A61732);
    for (int trial = 0; trial < 80; ++trial) {
        const int minimum = 3 + int(random()%6), maximum = minimum + int(random()%unsigned(65-minimum));
        SuffixLookup lookup(minimum,maximum,512);
        std::vector<int32_t> history;
        for (int step = 0; step < 360; ++step) {
            int32_t token = int32_t(random()%unsigned(2+trial%17))-10;
            if (step > 20 && random()%5 == 0) token = history[size_t(random()%history.size())];
            lookup.append(token);history.push_back(token);
            check(lookup,history,minimum,maximum,int(random()%13));
            if (step%41 == 0) {
                SuffixLookup rebuilt(minimum,maximum,512);
                rebuilt.rebuild(history.data(),history.size());
                check(rebuilt,history,minimum,maximum,7);
            }
        }
    }
}
void bounds_and_reset() {
    rejects<std::invalid_argument>([] { SuffixLookup x(2,64,64); });
    rejects<std::invalid_argument>([] { SuffixLookup x(3,65,64); });
    rejects<std::invalid_argument>([] { SuffixLookup x(6,5,64); });
    rejects<std::invalid_argument>([] { SuffixLookup x(3,64,0); });
    rejects<std::invalid_argument>([] { SuffixLookup x(3,64,std::numeric_limits<size_t>::max()); });
    rejects<std::invalid_argument>([] { SuffixLookup x(3,64,SuffixLookup::kMaxCapacity+1); });
    SuffixLookup tiny(3,64,4);
    const std::vector<int32_t> full = {9,9,9,9};
    tiny.append(full.data(),full.size());check(tiny,full,3,64,7);
    const int32_t token = 4;
    rejects<std::length_error>([&] { tiny.append(token); });
    rejects<std::length_error>([&] { tiny.append(&token,std::numeric_limits<size_t>::max()); });
    rejects<std::length_error>([&] { tiny.rebuild(&token,5); });
    rejects<std::invalid_argument>([&] { tiny.append(nullptr,1); });
    rejects<std::invalid_argument>([&] { tiny.rebuild(nullptr,1); });
    rejects<std::invalid_argument>([&] { tiny.propose(7,nullptr); });
    tiny.append(nullptr,0);check(tiny,full,3,64,7);
    REQUIRE(tiny.propose(0,nullptr) == 0 && tiny.last_match() == 0);
    REQUIRE(tiny.propose(-1,nullptr) == 0 && tiny.last_match() == 0);
    tiny.rebuild(nullptr,0);REQUIRE(tiny.size() == 0 && tiny.last_match() == 0);
    std::vector<int32_t> short_history = {1,2,3};
    tiny.append(short_history.data(),short_history.size());check(tiny,short_history,3,64,7);
    tiny.reset();tiny.append(full.data(),full.size());check(tiny,full,3,64,100);
}
int main() {
    fixed_cases();randomized_oracle();bounds_and_reset();
    std::printf("PASS suffix lookup oracle comparisons=%u\n",comparisons);
}
