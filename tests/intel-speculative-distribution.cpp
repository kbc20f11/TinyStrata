#include "strata/intel/speculative_distribution.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
using strata::intel::speculative_distribution::Distribution;
using strata::intel::speculative_distribution::verify;
void check(const Distribution& p, const Distribution& q, unsigned seed) {
    constexpr int count = 400000;
    std::mt19937_64 proposal(seed), target(seed ^ 0x9e3779b9u);
    std::array<int, 8> histogram{};
    int accepted = 0;
    for (int i = 0; i < count; ++i) {
        const int draft = q.draw(proposal), output = verify(p, q, draft, target);
        if (output < 0 || output >= int(histogram.size())) throw std::runtime_error("Out-of-support sample");
        ++histogram[size_t(output)];
        accepted += output == draft;
    }
    double expected_acceptance = 0;
    for (int token = 0; token < int(histogram.size()); ++token) {
        const double actual = double(histogram[size_t(token)]) / count, expected = p.mass(token) / p.total();
        if (std::abs(actual - expected) > .004) throw std::runtime_error("Empirical output distribution differs from target");
        expected_acceptance += std::min(expected, q.mass(token) / q.total());
    }
    if (std::abs(double(accepted) / count - expected_acceptance) > .004)
        throw std::runtime_error("Acceptance differs from independent probability-overlap formula");
    std::printf("seed=%u samples=%d acceptance=%.6f expected=%.6f result=passed\n",seed,count,double(accepted)/count,expected_acceptance);
}
int main(){try{
    // Different ordering, partial/disjoint support, zero weights and identical
    // distributions: output frequencies must match p, not a mixture with q.
    check({{0,1,2},{.2,.3,.5}},{{2,0,1},{.1,.7,.2}},1);
    check({{2,4},{.1,.9}},{{4,3},{.2,.8}},2);
    check({{0,1},{.4,.6}},{{2,3},{.5,.5}},3);
    check({{0,1,2},{0.,1.,0.}},{{0,1,2},{.2,.3,.5}},4);
    check({{0,1,2},{.2,.3,.5}},{{2,0,1},{5.,2.,3.}},5);
    check({{7},{1.}},{{7},{1.}},6);
    bool rejected = false;std::mt19937_64 rng(7);
    try { (void)verify({{1},{1.}},{{0},{1.}},1,rng); } catch(const std::runtime_error&) { rejected=true; }
    if(!rejected)throw std::runtime_error("Invalid proposal probability was not rejected");
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"speculative distribution test failed: %s\n",e.what());return 1;}}
