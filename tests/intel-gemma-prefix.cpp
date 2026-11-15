#include "strata/intel/gemma_prefix.hpp"
#include "strata/intel/context_policy.hpp"
#include <cstdio>
#include <functional>
#include <string>
using namespace strata::intel;
namespace gp=strata::intel::gemma_prefix;
namespace cp=strata::intel::context_policy;
static size_t checks=0;
static void check(bool value){++checks;if(!value)throw std::runtime_error("Gemma prefix check "+std::to_string(checks));}
template<class F>static void reject(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed);}
int main(){try{
    check(gp::snapshot_bytes(1536,8,256,25)==167116800);
    check(gp::snapshot_bytes(1,8,256,25)==108800);
    reject([]{gp::snapshot_bytes(0,8,256);});reject([]{gp::snapshot_bytes(10,8,255);});
    reject([]{gp::snapshot_bytes(UINT64_MAX,8,256);});reject([]{gp::snapshot_bytes(10,8,256,UINT64_MAX);});
    check(gp::turn_boundary(83,83)==82);check(gp::turn_boundary(80,83)==80);check(gp::turn_boundary(0,83)==0);
    reject([]{gp::turn_boundary(84,83);});reject([]{gp::turn_boundary(0,0);});
    const std::vector<int> live{1,2,3,4,5},saved{1,2,3},append{1,2,3,4,5,6},thought_strip{1,2,3,7,8},edit{1,9,3,7,8};
    auto reuse=gp::choose(live,append,saved,5,true,true,true);check(reuse.source==gp::Source::live&&reuse.tokens==5);
    reuse=gp::choose(live,live,saved,5,true,true,true);check(reuse.source==gp::Source::live&&reuse.tokens==5);
    reuse=gp::choose(live,live,saved,5,false,true,true);check(reuse.source==gp::Source::replay_last&&reuse.tokens==4);
    reuse=gp::choose(live,thought_strip,saved,5,true,true,true);check(reuse.source==gp::Source::saved_turn&&reuse.tokens==3);
    check(gp::choose(live,edit,saved,5,true,true,true).source==gp::Source::reset);
    check(gp::choose(live,append,saved,5,true,true,false).source==gp::Source::reset); // ckpt0 is always cold
    check(gp::choose(live,thought_strip,saved,5,true,false,true).source==gp::Source::reset); // cancelled capture
    check(gp::choose(live,thought_strip,saved,4,true,true,true).source==gp::Source::reset); // partial state
    check(gp::choose(live,saved,saved,5,true,true,true).source==gp::Source::reset); // at least one fresh token
    check(gp::choose(std::vector<int>{1,8,3,4,5},thought_strip,saved,5,true,true,true).source==gp::Source::reset);
    // Independently simulate physical modulo slots. Check the exact past keys
    // required by the first fresh query, including wrap and non-power-of-two.
    for(size_t window=1;window<=8;++window)for(size_t capacity=1;capacity<=20;++capacity)
        for(size_t logical=1;logical<=32;++logical)for(size_t written=logical;written<=logical+10;++written){
            std::vector<size_t> ring(capacity,SIZE_MAX);for(size_t p=0;p<written;++p)ring[p%capacity]=p;
            bool retained=true;const size_t first=logical>=window?logical-window+1:0;
            for(size_t p=first;p<logical;++p)retained=retained&&ring[p%capacity]==p;
            check(gp::ring_retains(logical,written,capacity,window)==retained);
        }
    // A saved whole chunk may contain future rows, but restoring its logical
    // mark and writing a new complete chunk must preserve every causal window.
    for(size_t window:{size_t(1),size_t(3),size_t(8),size_t(1024)})for(size_t chunk:{size_t(1),size_t(3),size_t(17),size_t(512)})
        for(size_t logical:{size_t(1),size_t(80),size_t(1023),size_t(1024),size_t(16379)}){
            const size_t capacity=window+chunk,written=logical+chunk-1;
            std::vector<size_t> ring(capacity,SIZE_MAX);for(size_t p=0;p<written;++p)ring[p%capacity]=p;
            check(gp::ring_retains(logical,written,capacity,window));
            const auto snapshot=ring;
            // Decode long enough to overwrite the old live ring completely.
            for(size_t p=written;p<written+capacity+7;++p)ring[p%capacity]=1000000+p;
            ring=snapshot;
            for(size_t p=logical;p<logical+chunk;++p)ring[p%capacity]=1000000+p;
            for(size_t q=logical;q<logical+chunk;++q){const size_t first=q>=window?q-window+1:0;
                for(size_t p=first;p<=q;++p)check(ring[p%capacity]==(p<logical?p:1000000+p));}
        }
    for(uint64_t context:{1,1023,1024,1535,1536,32768,262144})for(uint64_t chunk:{1,17,512}){
        cp::GemmaSpec s;s.chunk=chunk;
        const auto p=cp::gemma(context,s);
        const uint64_t independent=25*std::min(context,uint64_t(1024)+chunk)*8*256*68/32;
        check(p.host_only==(16ull<<20)+independent);
        MemoryBudget memory;memory.integrated=true;memory.available_physical=memory.available_commit=4*GiB+p.device+p.host_only;
        memory.wddm_known=true;memory.wddm_available=p.device;
        check(cp::fits(memory,p.largest,p)); // host snapshot does not consume WDDM
        --memory.available_commit;check(!cp::fits(memory,p.largest,p));
        ++memory.available_commit;--memory.wddm_available;check(!cp::fits(memory,p.largest,p));
        memory.integrated=false;memory.wddm_available=p.device+GiB/4;memory.available_commit=4*GiB+p.host_visible+p.host_only;
        memory.wddm_nonlocal_known=true;memory.wddm_nonlocal_available=p.host_visible;
        check(cp::fits(memory,p.largest,p));--memory.available_commit;check(!cp::fits(memory,p.largest,p));
    }
    std::printf("Gemma prefix policy PASS checks=%zu snapshot_bytes=167116800\n",checks);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
