#pragma once
#include <algorithm>
#include <stdexcept>

namespace strata::intel {
// A cached decode graph can be replayed at every position in its extent
// bucket. The previous extent, rather than extent/2, is the lower bound when
// the last bucket is capped by a non-power-of-two context.
inline int graph_minimum_visible(int context,int extent) {
    if(context<1||context>262144||extent<1||extent>context)
        throw std::invalid_argument("Invalid Intel decode graph extent");
    int current=std::min(context,128),previous=0;
    while(current<extent){previous=current;current=std::min(context,current*2);}
    if(current!=extent)throw std::invalid_argument("Intel decode graph extent is not a replay bucket");
    return previous+1;
}
}
