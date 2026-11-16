// Model-free GPU gate for Gemma image masks. The FP64 reference uses the
// canonical CUDA attn_rows formula, independently of gemma_ops.cpp.
#include "strata/intel/gemma_ops.hpp"
#include "strata/intel/device.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace si=strata::intel;
namespace {
int checks=0,cases=0;double maximum_error=0;
void check(bool x){++checks;if(!x)throw std::runtime_error("Gemma image attention check "+std::to_string(checks));}
template<class T>struct Buffer{
    sycl::queue& q;T* base;T* data;size_t size;
    Buffer(sycl::queue& queue,size_t n):q(queue),base(sycl::malloc_shared<T>(n+32,q)),data(base?base+16:nullptr),size(n){if(!base)throw std::bad_alloc();std::fill_n(base,n+32,T(7));}
    ~Buffer(){try{q.wait_and_throw();sycl::free(base,q);}catch(...){}}
    void guards(){for(int i=0;i<16;++i)check(base[i]==T(7)&&data[size+i]==T(7));}
};
int8_t value(int logical,int col){return int8_t((logical+3*col)%19-9);}
void run(sycl::queue& q,int context,int position,int tokens,int window,int chunk,bool image,bool mkl){
    constexpr int dim=32,heads=2,kv_heads=1;
    const int capacity=window?std::min(context,window+std::max(chunk,tokens)):context;
    const size_t cells=size_t(capacity)*kv_heads*dim,rows=size_t(tokens)*heads*dim;
    Buffer<int8_t> keys(q,cells),values(q,cells);Buffer<sycl::half> ks(q,cells/32),vs(q,cells/32);
    Buffer<float> queries(q,rows),output(q,rows);
    std::fill_n(keys.data,cells,int8_t(0));std::fill_n(values.data,cells,int8_t(0));
    std::fill_n(ks.data,cells/32,sycl::half(1));std::fill_n(vs.data,cells/32,sycl::half(.125f));std::fill_n(queries.data,rows,0.f);
    for(int logical=0;logical<position+tokens;++logical)for(int col=0;col<dim;++col)values.data[size_t(logical%capacity)*dim+col]=value(logical,col);
    const std::vector<int8_t> immutable_values(values.data,values.data+cells);
    si::GemmaKv cache{keys.data,values.data,ks.data,vs.data,capacity,kv_heads,dim,window};
    si::GemmaAttention attention(q,context,chunk,heads,dim,cells,mkl);
    attention.run(queries.data,cache,output.data,heads,position,tokens,1.f,image).wait_and_throw();
    for(int t=0;t<tokens;++t){
        const int own=position+t;
        // CUDA: lo=p-window+1; last=pos+rows-1 only for a SWA picture.
        const int low=window&&own>=window?own-window+1:0;
        const int high=image&&window?position+tokens-1:own;
        for(int head=0;head<heads;++head)for(int col=0;col<dim;++col){
            double sum=0;for(int key=low;key<=high;++key)sum+=double(value(key,col))*.125;
            const double reference=sum/double(high-low+1),actual=output.data[(size_t(t)*heads+head)*dim+col];
            maximum_error=std::max(maximum_error,std::abs(actual-reference));check(std::isfinite(actual)&&std::abs(actual-reference)<=2e-5*(1+std::abs(reference)));
        }
    }
    check(std::equal(immutable_values.begin(),immutable_values.end(),values.data));
    if(image&&tokens>chunk){bool refused=false;try{attention.run(queries.data,cache,output.data,heads,position,tokens,1.f,false);}catch(const std::invalid_argument&){refused=true;}check(refused);}
    keys.guards();values.guards();ks.guards();vs.guards();queries.guards();output.guards();
    ++cases;std::printf("Gemma image attention case=%d window=%d rows=%d text_chunk=%d image=%d mkl=%d PASS\n",cases,window,tokens,chunk,int(image),int(mkl));
}
}
int main(){try{
    sycl::queue q(si::select_device(0),sycl::property::queue::in_order{});
    for(bool mkl:{false,true}){
#ifndef TINYSTRATA_INTEL_MKL
        if(mkl)continue;
#endif
        run(q,64,33,6,4,4,true,mkl); // wrapped ring and a picture wider than SWA
        run(q,64,33,6,0,4,true,mkl); // full layers stay causal for pictures
        run(q,64,33,6,4,8,false,mkl); // ordinary text path remains causal
        run(q,2048,900,1120,1024,512,true,mkl); // late queries drop the first96 patches
        run(q,2048,900,1120,0,512,true,mkl);
    }
    std::printf("Gemma vision independent attention PASS cases=%d checks=%d max_FP64_abs=%.9g\n",cases,checks,maximum_error);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
