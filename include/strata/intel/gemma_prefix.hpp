#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::intel::gemma_prefix {
inline uint64_t product(uint64_t a,uint64_t b){
    if(a&&b>UINT64_MAX/a)throw std::overflow_error("Gemma turn snapshot byte overflow");
    return a*b;
}
inline uint64_t snapshot_bytes(uint64_t capacity,uint64_t heads,uint64_t dim,uint64_t layers=1){
    if(!capacity||!heads||!dim||dim%32)throw std::invalid_argument("Invalid Gemma turn snapshot geometry");
    const auto cells=product(product(capacity,heads),dim);
    return product(product(cells/32,68),layers); // two code arrays and two FP16 scale arrays
}
// The first fresh query at logical needs keys [logical-window+1,logical).
// A whole-chunk snapshot may also contain later keys. They are masked until
// rewritten, but must not already have overwritten that required past interval.
inline bool ring_retains(size_t logical,size_t written,size_t capacity,size_t window){
    if(!capacity||!window||logical>written)return false;
    const size_t first=logical>=window?logical-window+1:0;
    if(first==logical)return true;
    const size_t retained=written>capacity?written-capacity:0;
    return retained<=first;
}
inline size_t turn_boundary(size_t requested,size_t prompt_size){
    if(!prompt_size||requested>prompt_size)throw std::invalid_argument("Gemma turn boundary exceeds prompt");
    return std::min(requested,prompt_size-1); // restoring always leaves fresh logits work
}
enum class Source { reset,live,saved_turn,replay_last };
struct Reuse {Source source=Source::reset;size_t tokens=0;};
inline bool prefix(const std::vector<int>& a,const std::vector<int>& b){
    return a.size()<=b.size()&&std::equal(a.begin(),a.end(),b.begin());
}
inline Reuse choose(const std::vector<int>& live,const std::vector<int>& prompt,
                    const std::vector<int>& saved,size_t position,bool logits_valid,
                    bool snapshot_valid,bool enabled){
    if(!enabled||position!=live.size())return {};
    if(prefix(live,prompt)){
        if(live.size()<prompt.size()||logits_valid)return {Source::live,live.size()};
        // With no cached output row, replay one token instead of sampling stale
        // logits. A one-token rewind always fits the engine's window+chunk ring.
        if(!live.empty())return {Source::replay_last,live.size()-1};
    }
    if(snapshot_valid&&!saved.empty()&&saved.size()<prompt.size()&&prefix(saved,live)&&prefix(saved,prompt))
        return {Source::saved_turn,saved.size()};
    return {};
}
}
