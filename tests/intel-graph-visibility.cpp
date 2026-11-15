#include "strata/intel/graph_visibility.hpp"
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

int main(){
    using strata::intel::graph_minimum_visible;
    const int rows[][3]={{1,1,1},{127,127,1},{128,128,1},{129,128,1},{129,129,129},
        {32768,256,129},{32768,8192,4097},{32768,16384,8193},{32768,32768,16385},
        {10000,10000,8193},{16385,16385,16385},{20000,20000,16385},
        {157696,157696,131073},{159744,159744,131073},{262144,262144,131073}};
    for(const auto& row:rows)if(graph_minimum_visible(row[0],row[1])!=row[2])return 1;
    const int contexts[]={1,127,128,129,10000,16384,16385,20000,32768,131072,157696,159744,262144};
    unsigned checked=0;
    for(int context:contexts)for(int position=0;position<context;++position){
        int extent=std::min(context,128);
        while(extent<=position)extent=std::min(context,extent*2);
        const int hint=graph_minimum_visible(context,extent);
        if(hint>position+1||hint<1)return 2;
        if(hint>=16384&&position+1<16384)return 3;
        ++checked;
    }
    const int bad[][2]={{0,0},{32768,129},{20000,16385},{32768,65536},{262145,128}};
    for(const auto& row:bad){bool rejected=false;try{(void)graph_minimum_visible(row[0],row[1]);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)return 4;}
    std::printf("Intel graph visibility: %u replay positions and boundary/refusal cases PASS\n",checked);
    return 0;
}
