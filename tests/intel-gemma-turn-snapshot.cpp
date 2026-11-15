#include "strata/intel/gemma_turn_snapshot.hpp"
#include "strata/intel/device.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace strata::intel;
static size_t checks=0;
static void check(bool value){++checks;if(!value)throw std::runtime_error("snapshot check "+std::to_string(checks));}
struct Array{
    sycl::queue& q;uint8_t* allocation;size_t size;std::vector<uint8_t> original;
    Array(sycl::queue& queue,size_t bytes,unsigned seed):q(queue),size(bytes),original(bytes+128){
        for(size_t i=0;i<original.size();++i)original[i]=uint8_t((i*131+seed*17+(i>>8))&255);
        allocation=sycl::malloc_device<uint8_t>(original.size(),q);if(!allocation)throw std::bad_alloc();
        q.memcpy(allocation,original.data(),original.size()).wait();
    }
    ~Array(){sycl::free(allocation,q);}
    uint8_t* data(){return allocation+64;}
    void mutate(){q.memset(data(),0x5a,size);}
    void unchanged(){std::vector<uint8_t> actual(original.size());q.memcpy(actual.data(),allocation,actual.size()).wait();check(actual==original);}
};
int main(){try{
    sycl::queue q(select_device(0),sycl::property::queue::in_order{});
    for(int capacity:{19,1536}){
        const int window=capacity==19?11:1024,dim=256,heads=8;const size_t cells=size_t(capacity)*heads*dim;
        Array k(q,cells,1),v(q,cells,2),ks(q,cells/32*2,3),vs(q,cells/32*2,4);
        Array fk(q,37*2*512,5),fv(q,37*2*512,6),fks(q,37*2*512/32*2,7),fvs(q,37*2*512/32*2,8);
        GemmaKv sliding{reinterpret_cast<int8_t*>(k.data()),reinterpret_cast<int8_t*>(v.data()),reinterpret_cast<sycl::half*>(ks.data()),reinterpret_cast<sycl::half*>(vs.data()),capacity,heads,dim,window};
        GemmaKv full{reinterpret_cast<int8_t*>(fk.data()),reinterpret_cast<int8_t*>(fv.data()),reinterpret_cast<sycl::half*>(fks.data()),reinterpret_cast<sycl::half*>(fvs.data()),37,2,512,0};
        GemmaTurnSnapshot snapshot({sliding,full});check(snapshot.bytes()==size_t(cells)*68/32);check(!snapshot.valid());
        const size_t logical=capacity==19?43:16379,chunk=capacity==19?8:512;
        check(snapshot.save(q,logical,logical+chunk-1,chunk));check(snapshot.position()==logical);
        k.mutate();v.mutate();ks.mutate();vs.mutate();q.wait_and_throw();
        check(snapshot.restore(q)==logical);k.unchanged();v.unchanged();ks.unchanged();vs.unchanged();
        fk.unchanged();fv.unchanged();fks.unchanged();fvs.unchanged(); // full KV never copied or changed
        std::atomic<bool> cancel{true};check(!snapshot.save(q,logical,logical+chunk-1,chunk,&cancel));check(!snapshot.valid());
        bool refused=false;try{snapshot.restore(q);}catch(const std::logic_error&){refused=true;}check(refused);
        check(snapshot.save(q,logical,logical+chunk-1,chunk));snapshot.invalidate();check(!snapshot.valid());
        refused=false;try{snapshot.save(q,logical,logical+size_t(capacity)+size_t(window),size_t(capacity)+size_t(window));}catch(const std::invalid_argument&){refused=true;}check(refused);check(!snapshot.valid());
    }
    q.wait_and_throw();std::printf("Gemma turn snapshot GPU PASS checks=%zu full_kv_unchanged=1 guards=128\n",checks);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
