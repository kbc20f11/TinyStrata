// Independent scalar double-precision references for LFM-specific semantics.
// Explicit hardware test: never in the automatic CPU CTest group.
#include "strata/intel/device.hpp"
#include "strata/intel/lfm_ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void need(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
struct Buffer {
    sycl::queue& q;
    float *base,*p;size_t count;
    Buffer(sycl::queue& queue, size_t n):q(queue),base(sycl::malloc_shared<float>(n+16, queue)),p(base?base+8:nullptr),count(n) {
        if (!base) throw std::bad_alloc();
        std::fill_n(base,n+16,7.f);
    }
    ~Buffer() { try { q.wait(); sycl::free(base, q); } catch (...) {} }
    void guards()const{for(size_t i=0;i<8;++i)need(base[i]==7.f&&p[count+i]==7.f,"LFM input/state/output canary changed");}
};
void convolution(sycl::queue& q, int channels, int tokens, int split) {
    const size_t n = size_t(tokens) * channels;
    Buffer x(q, n * 3), w(q, size_t(channels) * 3), st(q, size_t(channels) * 2), out(q,n+8);
    std::vector<double> state(size_t(channels) * 2), expected(n);
    for (size_t i=0;i<n*3;++i) x.p[i]=float(std::sin(double(i)*.071)*.7);
    for (int c=0;c<channels;++c) {
        st.p[2*c]=float(std::cos(c*.1)*.2); st.p[2*c+1]=float(std::sin(c*.13)*.3);
        state[2*c]=st.p[2*c]; state[2*c+1]=st.p[2*c+1];
        for (int j=0;j<3;++j) w.p[3*c+j]=float((j-1.25)*.17+(c%7)*.03);
    }
    std::fill(out.p,out.p+n+8,12345.f);
    for(int t=0;t<tokens;++t) for(int c=0;c<channels;++c) {
        const size_t base=size_t(t)*channels*3;
        const double bx=double(x.p[base+c])*x.p[base+channels*2+c];
        expected[size_t(t)*channels+c]=double(x.p[base+channels+c]) *
            (state[2*c]*w.p[3*c]+state[2*c+1]*w.p[3*c+1]+bx*w.p[3*c+2]);
        state[2*c]=state[2*c+1]; state[2*c+1]=bx;
    }
    if(split>0 && split<tokens) {
        strata::intel::lfm_shortconv(q,x.p,st.p,w.p,out.p,channels,split);
        strata::intel::lfm_shortconv(q,x.p+size_t(split)*channels*3,st.p,w.p,
            out.p+size_t(split)*channels,channels,tokens-split).wait_and_throw();
    } else strata::intel::lfm_shortconv(q,x.p,st.p,w.p,out.p,channels,tokens).wait_and_throw();
    for(size_t i=0;i<n;++i) need(std::abs(out.p[i]-expected[i])<2e-6,"LFM convolution FP64 mismatch");
    for(int i=0;i<channels*2;++i) need(std::abs(st.p[i]-state[i])<2e-7,"LFM convolution state mismatch");
    for(size_t i=n;i<n+8;++i) need(out.p[i]==12345.f,"LFM convolution output overrun");
    x.guards();w.guards();st.guards();out.guards();
}
void routing(sycl::queue& q, bool norm, float scale) {
    constexpr int E=64,K=4,T=7;
    Buffer logits(q,E*T),bias(q,E),weights(q,T*K+8);
    auto* id_base=sycl::malloc_shared<int32_t>(T*K+24,q);
    if(!id_base) throw std::bad_alloc();
    auto* ids=id_base+8;std::fill_n(id_base,T*K+24,-777);
    for(int e=0;e<E;++e) bias.p[e]=float((e%11-5)*.13);
    for(int t=0;t<T;++t) for(int e=0;e<E;++e)
        logits.p[t*E+e]=t==0?0.f:t==1?-1000.f:float(std::sin(t*17+e*.31)*7);
    std::fill(ids,ids+T*K+8,-777); std::fill(weights.p,weights.p+T*K+8,12345.f);
    strata::intel::lfm_route(q,logits.p,bias.p,ids,weights.p,T,E,K,norm,scale).wait_and_throw();
    for(int t=0;t<T;++t) {
        std::vector<int> order(E); std::iota(order.begin(),order.end(),0);
        std::vector<double> probability(E);
        for(int e=0;e<E;++e) probability[e]=1./(1.+std::exp(-double(logits.p[t*E+e])));
        std::stable_sort(order.begin(),order.end(),[&](int a,int b) {
            return probability[a]+bias.p[a]>probability[b]+bias.p[b];
        });
        double sum=0; for(int k=0;k<K;++k) sum+=probability[order[k]];
        const double div=norm?std::max(sum,6.103515625e-5):1.;
        for(int k=0;k<K;++k) {
            need(ids[t*K+k]==order[k],"LFM biased sigmoid pick mismatch");
            const double expected=probability[order[k]]/div*(scale==0.f?1.:scale);
            need(std::abs(weights.p[t*K+k]-expected)<3e-6,"LFM unbiased mixture weight mismatch");
        }
    }
    for(int i=T*K;i<T*K+8;++i) need(ids[i]==-777 && weights.p[i]==12345.f,"LFM routing overrun");
    for(int i=0;i<8;++i)need(id_base[i]==-777&&ids[T*K+8+i]==-777,"LFM router ID guard changed");
    logits.guards();bias.guards();weights.guards();sycl::free(id_base,q);
}
}
int main() {
    try {
        sycl::queue q(strata::intel::select_device(0), sycl::property::queue::in_order{});
        int checks=0;
        for(int c:{1,31,257,2048}) for(int t:{1,2,3,17,129,512,1024}) {
            convolution(q,c,t,0); ++checks;
            if(t>1) { convolution(q,c,t,1); ++checks; }
            if(t>3) { convolution(q,c,t,t-1); ++checks; }
        }
        for(bool normalize:{false,true}) for(float scale:{0.f,.7f,1.5f}) {
            routing(q,normalize,scale); ++checks;
        }
        std::printf("LFM independent numerical cases: %d passed\n",checks);
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
