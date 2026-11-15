#include "strata/intel/context_policy.hpp"
#include <cstdio>
#include <functional>
#include <string>
using namespace strata::intel;
namespace cp=strata::intel::context_policy;
static int checks=0;
static void check(bool value){++checks;if(!value)throw std::runtime_error("context policy check "+std::to_string(checks));}
template<class F>static void rejected(F f){bool no=false;try{f();}catch(const std::exception&){no=true;}check(no);}
int main(){try{
    check(cp::parse("auto",262144)==0);check(cp::parse("32768",262144)==32768);
    check(cp::parse("262144",262144)==262144);check(cp::parse("131072",131072)==131072);
    check(cp::model_limit(262144,262144)==262144);check(cp::model_limit(131072,262144)==131072);
    rejected([]{cp::model_limit(0,262144);});rejected([]{cp::model_limit(INFINITY,262144);});
    for(const char* s:{"","0","-1","+1","1.5"," 8192","8192x","262145","99999999999999999999999999999999999"})rejected([&]{cp::parse(s,262144);});
    rejected([]{cp::mul(UINT64_MAX,2);});rejected([]{cp::add(UINT64_MAX,1);});rejected([]{cp::round256(UINT64_MAX);});
    rejected([]{auto s=cp::GemmaSpec{};s.chunk=UINT64_MAX;cp::gemma(262144,s);});
    rejected([]{auto s=cp::QwenSpec{};s.vocabulary=UINT64_MAX;cp::qwen(262144,s);});
    // Independently sum the actual declared Gemma arrays, including the
    // packed2*E gate/up allocation and bounded tile/control arrays.
    for(uint64_t c:{1,1023,1024,1535,1536,1537,32768,131073,262144}){
        const uint64_t ring=std::min(c,uint64_t(1536));
        const uint64_t cache=(25*ring*8*256+5*c*2*512)*2*34/32;
        const uint64_t floats=512ull*(8*2816+3*16*512+4*8*256+3*2112+3*8*704+8*2816+128+8);
        const uint64_t ints=512*8+512+1+128+512*128+2*((512*8+3)/4+128)+1;
        const uint64_t shared=262144*4+512*4+4;
        const auto p=cp::gemma(c);
        check(p.cache==cache);check(p.owned_device==cache+(floats+262144+ints)*4);
        check(p.owned_host_visible==shared);
        cp::GemmaSpec scalar;scalar.mkl=false;const auto old=cp::gemma(c,scalar);
        const uint64_t expansion=std::max(ring*8*256,c*2*512);
        const uint64_t gemm=8192ull*2048*2+1024ull*8192*2+1024ull*2048*4+512*4;
        check(p.device-old.device==2*expansion*4+512*512*4+gemm);
        check(old.device==old.owned_device+c*512*4);
    }
    for(uint64_t c:{1,32767,32768,65537,131072}){
        const auto p=cp::lfm(c);const uint64_t cache=10ull*c*8*64*2*34/32;
        const uint64_t floats=30*4*2048+512ull*(8*2048+3*32*64+3*8*64+3*11776+4*2048+64+4)+2*2048;
        const uint64_t ints=512*4+512+1+64+512*64+2*((512*4+3)/4+64)+1;
        check(p.cache==cache);check(p.owned_device==cache+(floats+128000+ints)*4);
        check(p.owned_host_visible==128000*4+512*4+4+64*4);
    }
    cp::QwenSpec q;q.chunk=1024;q.int8=true;q.mtp=true;q.mkl=true;q.dp4a=true;q.support=true;q.draft_vocabulary=2048;
    for(uint64_t c:{1,255,256,257,32768,262144}){
        const auto p=cp::qwen(c,q);check(p.cache==11ull*2*c*256*2*34/32);
        const uint64_t n=1024,rows=8*16,cells=2*c*256;
        const uint64_t states=2*30ull*(32*128*128+64*128*3)*4;
        const uint64_t base=(n*80713+248320+2048+n*270+770)*4; // includes INT8 error word
        const uint64_t attention=(rows*(c+((c+255)/256)*256)+2*cells)*4;
        const uint64_t mtp=(n*12288+4096+2048+2048+8*248320+2*8*2048+30*(2*8*8192+2*8*32))*4;
        if(p.owned_device!=p.cache+states+base+attention+mtp)std::fprintf(stderr,"Qwen ledger c=%llu got=%llu expected=%llu\n",(unsigned long long)c,(unsigned long long)p.owned_device,(unsigned long long)(p.cache+states+base+attention+mtp));
        check(p.owned_device==p.cache+states+base+attention+mtp);
        check(p.owned_host_visible==p.owned_device);
        auto d=q;d.device_state=true;const auto discrete=cp::qwen(c,d);
        check(discrete.device==p.device);check(discrete.host_visible<p.host_visible);
        auto off=q;off.support=false;const auto no=cp::qwen(c,off);
        check(p.device-no.device==248320*12+65536*4+8192*4+8);
        check(p.host_visible-no.host_visible==p.device-no.device);
        auto snapshot=q;snapshot.turn_checkpoint=false;
        check(p.host_only-cp::qwen(c,snapshot).host_only==30ull*(32*128*128+64*128*3)*4+2048*4*2+c*4);
    }
    // A remaining10GiB RAM/commit snapshot means6GiB after the existing OS
    // floor, once. All full model limits fit this synthetic UMA budget.
    MemoryBudget uma{10*GiB,10*GiB,12*GiB,true,true};
    check(cp::select(0,262144,uma,16*GiB,[](uint64_t c){return cp::gemma(c);}).tokens==262144);
    check(cp::select(0,131072,uma,16*GiB,[](uint64_t c){return cp::lfm(c);}).tokens==131072);
    check(cp::select(0,262144,uma,16*GiB,[&](uint64_t c){return cp::qwen(c,q);}).tokens==262144);
    const auto linear=[](uint64_t c){cp::Footprint p;p.allocation(c,1024);p.host(63ull<<20);return p;};
    const uint64_t need=65537ull*1024+(63ull<<20);
    MemoryBudget edge{4*GiB+need,4*GiB+need,0,false,true};
    auto chosen=cp::select(0,262144,edge,GiB,linear);check(chosen.automatic&&chosen.tokens==65536);
    check(cp::select(65537,262144,edge,GiB,linear).tokens==65537); // explicit unaligned retained
    rejected([&]{cp::select(65538,262144,edge,GiB,linear);});
    rejected([&]{cp::select(262145,262144,edge,GiB,linear);});
    auto none=edge;none.available_physical=none.available_commit=4*GiB;
    rejected([&]{cp::select(0,262144,none,GiB,linear);});
    // WDDM and max single allocation can independently limit auto. There is
    // no sum of RAM and shared/local GPU capacity.
    auto wddm=uma;wddm.wddm_available=65537ull*1024;check(cp::select(0,262144,wddm,GiB,linear).tokens==65536);
    // Replay the first real B390 auto-startup budgets. Device USM is WDDM
    // charged; the ordinary83MiB snapshot/work is RAM-only. This is a CPU
    // admission fixture, not a claim of a new real GPU allocation.
    MemoryBudget measured{7494410240ull+4*GiB,7494410240ull+4*GiB,4039076372ull,true,true};
    q.device_state=true;q.dense_split=true;
    const auto improved=cp::select(0,262144,measured,17677697024ull,[&](uint64_t c){return cp::qwen(c,q);});
    check(improved.tokens==157696);check(improved.bytes.device==4014184612ull);
    check(improved.bytes.host_only==83288064ull);
    check(!cp::fits(measured,17677697024ull,cp::qwen(159744,q)));
    check(cp::select(0,262144,uma,65536ull*1024,linear).tokens==65536);
    rejected([&]{cp::select(65537,262144,uma,65536ull*1024,linear);});
    MemoryBudget separate{4*GiB+(63ull<<20),4*GiB+(63ull<<20),8*GiB,true,false,GiB,true};
    cp::Footprint p;p.allocation(GiB,1);p.host(63ull<<20);check(cp::fits(separate,2*GiB,p));
    p.allocation(1,1,true);check(!cp::fits(separate,2*GiB,p)); // one RAM byte below OS floor
    separate.available_physical++;separate.available_commit++;check(cp::fits(separate,2*GiB,p));
    separate.wddm_nonlocal_known=false;check(!cp::fits(separate,2*GiB,p));
    std::printf("Intel auto-context CPU checks=%d PASS\n",checks);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
