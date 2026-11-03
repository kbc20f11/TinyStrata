// Independent FP64 oracle + production SIMT incumbent for compensated FDEC.
// No model, timing claim or additional test framework. Run outside other GPU jobs.
#include "strata/qwen35/int8_fdec.hpp"
#include "strata/qwen35/attention.cuh"
#include "strata/qwen35/kv_grow.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace F = strata::qwen35::int8_fdec;
namespace A = strata::qwen35::attention;
namespace VMM = strata::qwen35::kv_grow;
namespace {
constexpr int D=256, H=16, KVH=2, Guard=64;
constexpr std::uint32_t Canary=0x7fc3a55au;
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool x,const char* s) { if(!x) throw std::runtime_error(s); }
float bits(std::uint32_t n) { float x;std::memcpy(&x,&n,4);return x; }
std::uint32_t bits(float x) { std::uint32_t n;std::memcpy(&n,&x,4);return n; }
std::uint16_t half_bits(float x) { return __half_as_ushort(__float2half_rn(x)); }
float as_float(std::uint16_t x) { return __half2float(__ushort_as_half(x)); }
float uniform(std::mt19937& r) { return (float)((r()+.5)/4294967296.0); }
struct Buffer {
    float* base=nullptr;std::size_t count;
    explicit Buffer(std::size_t n):count(n) {
        ck(cudaMalloc(&base,(n+2*Guard)*4));poison();
    }
    ~Buffer(){if(base)cudaFree(base);}
    Buffer(const Buffer&)=delete;
    float* data()const{return base+Guard;}
    void poison(){std::vector<float>v(count+2*Guard,bits(Canary));ck(cudaMemcpy(base,v.data(),v.size()*4,cudaMemcpyHostToDevice));}
    void put(const std::vector<float>&v){require(v.size()<=count,"float input capacity");ck(cudaMemcpy(data(),v.data(),v.size()*4,cudaMemcpyHostToDevice));}
    std::vector<float> get(std::size_t n)const{std::vector<float>v(n);ck(cudaMemcpy(v.data(),data(),n*4,cudaMemcpyDeviceToHost));return v;}
    void guard(std::size_t written)const{
        std::vector<float>v(count+2*Guard);ck(cudaMemcpy(v.data(),base,v.size()*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<Guard;++i)require(bits(v[i])==Canary,"leading guard");
        for(std::size_t i=Guard+written;i<v.size();++i)require(bits(v[i])==Canary,"tail/output guard");
    }
};
struct Data {
    int context,visible,pattern;
    std::vector<std::int8_t> k,v;
    std::vector<std::uint16_t> ks,vs;
    std::vector<float>q,gate;
    Data(int ctx,int end,int mode):context(ctx),visible(end),pattern(mode),k((size_t)KVH*ctx*D),v(k.size()),
        ks(k.size()/64,0x7e00u),vs(ks.size(),0x7e00u),q(4*H*D),gate(q.size()) {
        std::mt19937 r(713+mode);
        for(float&x:q)x=(2*uniform(r)-1)*(mode==3?1000.f:mode==4?1e-12f:mode==1?1e-4f:.5f);
        for(float&x:gate)x=(2*uniform(r)-1)*3.f;
        for(int h=0;h<KVH;++h)for(int t=0;t<end;++t)for(int b=0;b<D/64;++b){
            const size_t scale=((size_t)h*ctx+t)*4+b;
            float sk=.003f+.025f*uniform(r),sv=.002f+.035f*uniform(r);
            if(mode==1){const float values[]={0x1p-24f,0x1p-10f,1.f,256.f};sk=values[(b+t+h)%4];sv=values[(3*b+t+h)%4];}
            if(mode==2){sk=.01f;sv=(t&1)?-.02f:.02f;}
            if(mode==4){sk=0x1p-14f;sv=(b&1)?0x1p-24f:0x1p-10f;}
            if(mode==5){sk=(h+1)*(b+1)*.001f;sv=(h+1)*(4-b)*.003f;}
            if(mode==6){sk=(t%17==0)?0.f:.02f;sv=(t%13==0)?0.f:65504.f;}
            ks[scale]=half_bits(sk);vs[scale]=half_bits(sv);
            for(int d=0;d<64;++d){const size_t at=scale*64+d;k[at]=(std::int8_t)((int)(r()%255)-127);v[at]=(std::int8_t)((int)(r()%255)-127);}
        }
        if(mode==2)for(int t=0;t<4;++t)for(int h=0;h<H;++h)for(int d=0;d<D;++d)
            q[((size_t)t*H+h)*D+d]=(d&1)?-.125f:.125f;
    }
    float key(int h,int t,int d)const{const size_t i=((size_t)h*context+t)*D+d;return(float)k[i]*as_float(ks[i/64]);}
    float value(int h,int t,int d)const{const size_t i=((size_t)h*context+t)*D+d;return(float)v[i]*as_float(vs[i/64]);}
};
struct DeviceCache {
    F::Cache view; std::int8_t *k=nullptr,*v=nullptr;std::uint16_t *ks=nullptr,*vs=nullptr;
    std::unique_ptr<VMM::Range> kr,vr,ksr,vsr;
    DeviceCache(const Data&d,bool sparse=false){
        if(!sparse){
            ck(cudaMalloc(&k,d.k.size()));ck(cudaMalloc(&v,d.v.size()));
            ck(cudaMalloc(&ks,d.ks.size()*2));ck(cudaMalloc(&vs,d.vs.size()*2));
            ck(cudaMemcpy(k,d.k.data(),d.k.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(v,d.v.data(),d.v.size(),cudaMemcpyHostToDevice));
            ck(cudaMemcpy(ks,d.ks.data(),d.ks.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(vs,d.vs.data(),d.vs.size()*2,cudaMemcpyHostToDevice));
        }else{
            require(VMM::api().ok,"VMM unavailable for unmapped-future test");
            const auto plane=[&](size_t row,std::unique_ptr<VMM::Range>&r){
                r=std::make_unique<VMM::Range>();require(r->reserve((size_t)KVH*d.context*row),"VMM reserve");
                require(r->map(r->mask(d.visible,KVH,(size_t)d.context*row,row),nullptr),"VMM prefix map");return r->base();};
            k=(std::int8_t*)plane(D,kr);v=(std::int8_t*)plane(D,vr);
            ks=(std::uint16_t*)plane(8,ksr);vs=(std::uint16_t*)plane(8,vsr);
            ck(cudaDeviceSynchronize());
            for(int h=0;h<KVH;++h){
                const size_t off=(size_t)h*d.context*D,sc=(size_t)h*d.context*4;
                ck(cudaMemcpy(k+off,d.k.data()+off,(size_t)d.visible*D,cudaMemcpyHostToDevice));
                ck(cudaMemcpy(v+off,d.v.data()+off,(size_t)d.visible*D,cudaMemcpyHostToDevice));
                ck(cudaMemcpy(ks+sc,d.ks.data()+sc,(size_t)d.visible*8,cudaMemcpyHostToDevice));
                ck(cudaMemcpy(vs+sc,d.vs.data()+sc,(size_t)d.visible*8,cudaMemcpyHostToDevice));
            }
        }
        view={k,v,ks,vs};
    }
    void immutable(const Data& d)const{
        // Copy only the mapped prefix in sparse mode, and compare stored bits
        // (including half scales), rather than recomputing floating values.
        std::vector<std::int8_t> codes((size_t)d.visible*D);
        std::vector<std::uint16_t> scales((size_t)d.visible*4);
        for(int h=0;h<KVH;++h){
            const size_t off=(size_t)h*d.context*D,sc=(size_t)h*d.context*4;
            ck(cudaMemcpy(codes.data(),k+off,codes.size(),cudaMemcpyDeviceToHost));
            require(std::memcmp(codes.data(),d.k.data()+off,codes.size())==0,"immutable K changed");
            ck(cudaMemcpy(codes.data(),v+off,codes.size(),cudaMemcpyDeviceToHost));
            require(std::memcmp(codes.data(),d.v.data()+off,codes.size())==0,"immutable V changed");
            ck(cudaMemcpy(scales.data(),ks+sc,scales.size()*2,cudaMemcpyDeviceToHost));
            require(std::memcmp(scales.data(),d.ks.data()+sc,scales.size()*2)==0,"immutable K scales changed");
            ck(cudaMemcpy(scales.data(),vs+sc,scales.size()*2,cudaMemcpyDeviceToHost));
            require(std::memcmp(scales.data(),d.vs.data()+sc,scales.size()*2)==0,"immutable V scales changed");
        }
    }
    ~DeviceCache(){if(!kr){cudaFree(k);cudaFree(v);cudaFree(ks);cudaFree(vs);}}
};
std::vector<double> exact(const Data&d,int pos,int n,int query_first=0){
    std::vector<double>out((size_t)n*H*D),scores(pos+n);
    for(int t=0;t<n;++t)for(int h=0;h<H;++h){
        const int kh=h/8,keys=pos+t+1;double top=-INFINITY;
        for(int k=0;k<keys;++k){double s=0;for(int x=0;x<D;++x)s+=(double)d.q[((size_t)(query_first+t)*H+h)*D+x]*d.key(kh,k,x);
            scores[k]=s/16.;top=std::max(top,scores[k]);}
        double den=0;for(int k=0;k<keys;++k){scores[k]=std::exp(scores[k]-top);den+=scores[k];}
        for(int x=0;x<D;++x){double a=0;for(int k=0;k<keys;++k)a+=scores[k]*d.value(kh,k,x);
            const size_t at=((size_t)t*H+h)*D+x, qi=((size_t)(query_first+t)*H+h)*D+x;
            out[at]=a/den/(1.+std::exp(-(double)d.gate[qi]));}
    }return out;
}
struct Error{double max=0,l2=0,relative=0;};
Error error(const std::vector<float>&v,const std::vector<double>&ref){
    Error e;double norm=0;for(size_t i=0;i<v.size();++i){require(std::isfinite(v[i]),"nonfinite output");
        const double d=v[i]-ref[i];e.max=std::max(e.max,std::abs(d));e.l2+=d*d;norm+=ref[i]*ref[i];}
    e.relative=std::sqrt(e.l2/std::max(norm,1e-30));return e;
}
struct Difference {size_t count=0,first=0;double max=0;uint64_t ulp=0;};
uint32_t ordered(float x){const auto u=bits(x);return(u&0x80000000u)?~u:u^0x80000000u;}
Difference difference(const std::vector<float>&a,const std::vector<float>&b){
    require(a.size()==b.size(),"comparison shape");Difference d;
    for(size_t i=0;i<a.size();++i)if(bits(a[i])!=bits(b[i])){
        if(!d.count)d.first=i;++d.count;
        const uint64_t ua=ordered(a[i]),ub=ordered(b[i]);d.ulp=std::max(d.ulp,ua>ub?ua-ub:ub-ua);
        d.max=std::max(d.max,std::abs((double)a[i]-b[i]));
    }return d;
}
void check_case(int ctx,int pos,int n,int mode,bool graph=false,bool sparse=false){
    Data data(ctx,pos+n,mode);DeviceCache cache(data,sparse);
    Buffer q(data.q.size()),gate(data.gate.size()),out(data.q.size()),old(data.q.size());q.put(data.q);gate.put(data.gate);
    Buffer part(F::part_floats(ctx,KVH)),meta(F::meta_floats(ctx,KVH));
    const int nc=(ctx+255)/256;
    Buffer pold((size_t)n*H*nc*D),mold((size_t)n*H*nc),lold((size_t)n*H*nc);
    std::int32_t *at=nullptr;ck(cudaMalloc(&at,4));ck(cudaMemcpy(at,&pos,4,cudaMemcpyHostToDevice));
    A::KvCache ac;ac.key=cache.k;ac.value=cache.v;ac.key_scale16=cache.ks;ac.value_scale16=cache.vs;
    A::k_attn_part<2><<<dim3((pos+n+255)/256,n*H),256>>>(q.data(),ac,at,0,H,KVH,ctx,1.f/16,nc,pold.data(),mold.data(),lold.data());
    A::k_attn_merge<<<n*H,D>>>(pold.data(),mold.data(),lold.data(),gate.data(),at,0,H,nc,old.data());
    ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    require(F::run(q.data(),cache.view,at,pos,n,H,KVH,ctx,1.f/16,part.data(),part.count,meta.data(),meta.count,gate.data(),out.data(),nullptr),"FDEC eligible call refused");
    ck(cudaDeviceSynchronize());
    const auto eager=out.get((size_t)n*H*D), incumbent=old.get((size_t)n*H*D);const auto ref=exact(data,pos,n);
    const auto ef=error(eager,ref),ei=error(incumbent,ref);
    double vmax=0;for(int h=0;h<KVH;++h)for(int k=0;k<pos+n;++k)for(int x=0;x<D;++x)vmax=std::max(vmax,std::abs((double)data.value(h,k,x)));
    // Absolute floor is scaled by actual stored V, not an arbitrary loose
    // donor 1e-2 bound. Relative checks catch cancellation/tiny-scale failures.
    const double limit=std::max(2e-6*std::max(vmax,1e-6),8*ei.max);
    std::printf("NUMERIC n=%d pos=%d ctx=%d pattern=%d max=%.9g incumbent=%.9g limit=%.9g relL2=%.9g incumbentRel=%.9g\n",n,pos,ctx,mode,ef.max,ei.max,limit,ef.relative,ei.relative);std::fflush(stdout);
    require(ef.max<=limit,"FDEC FP64 max error exceeded");
    require(ef.relative<=std::max(5e-5,8*ei.relative),"FDEC FP64 relative L2 exceeded");
    const auto batch_part=part.get(part.count),batch_meta=meta.get(meta.count);
    if(graph){
        cudaStream_t st;cudaGraph_t g;cudaGraphExec_t exec;ck(cudaStreamCreate(&st));
        ck(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
        require(F::run(q.data(),cache.view,at,0,n,H,KVH,ctx,1.f/16,part.data(),part.count,meta.data(),meta.count,gate.data(),out.data(),st,true),"graph eligible call refused");
        ck(cudaStreamEndCapture(st,&g));ck(cudaGraphInstantiate(&exec,g,0));
        for(int replay_pos:{pos,0,63,1023}){
            ck(cudaMemcpyAsync(at,&replay_pos,4,cudaMemcpyHostToDevice,st));
            ck(cudaGraphLaunch(exec,st));ck(cudaStreamSynchronize(st));const auto captured=out.get((size_t)n*H*D);
            require(F::run(q.data(),cache.view,at,replay_pos,n,H,KVH,ctx,1.f/16,part.data(),part.count,meta.data(),meta.count,gate.data(),out.data(),nullptr),"dynamic eager refused");
            ck(cudaDeviceSynchronize());const auto direct=out.get((size_t)n*H*D);
            require(std::memcmp(captured.data(),direct.data(),captured.size()*4)==0,"graph/eager bits differ");
        }
        ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(g));ck(cudaStreamDestroy(st));
    }
    // Same query alone versus in an n-row launch, including unequal masks.
    ck(cudaMemcpy(at,&pos,4,cudaMemcpyHostToDevice));
    for(int t=0;t<n;++t){
        const int p=pos+t;ck(cudaMemcpy(at,&p,4,cudaMemcpyHostToDevice));
        require(F::run(q.data()+(size_t)t*H*D,cache.view,at,p,1,H,KVH,ctx,1.f/16,part.data(),part.count,meta.data(),meta.count,gate.data()+(size_t)t*H*D,out.data(),nullptr),"singleton refused");
        ck(cudaDeviceSynchronize());const auto one=out.get(H*D);
        const std::vector<float> batch(eager.begin()+(size_t)t*H*D,eager.begin()+(size_t)(t+1)*H*D);
        const auto one_ref=exact(data,p,1,t);const auto es=error(one,one_ref),eb=error(batch,one_ref);
        const std::vector<float> batch_old(incumbent.begin()+(size_t)t*H*D,incumbent.begin()+(size_t)(t+1)*H*D);
        const auto ib=error(batch_old,one_ref);
        A::k_attn_part<2><<<dim3((p+1+255)/256,H),256>>>(q.data()+(size_t)t*H*D,ac,at,0,H,KVH,ctx,1.f/16,nc,pold.data(),mold.data(),lold.data());
        A::k_attn_merge<<<H,D>>>(pold.data(),mold.data(),lold.data(),gate.data()+(size_t)t*H*D,at,0,H,nc,old.data());
        ck(cudaGetLastError());ck(cudaDeviceSynchronize());const auto is=error(old.get(H*D),one_ref);
        const auto delta=difference(one,batch);
        const auto single_part=part.get(part.count),single_meta=meta.get(meta.count);
        std::vector<float> bp,sp,bm,sm;
        for(int split=0;split<=p/1024;++split)for(int h=0;h<H;++h){
            const size_t bi=((size_t)split*KVH+h/8)*32+t*8+h%8,si=((size_t)split*KVH+h/8)*32+h%8;
            bm.push_back(batch_meta[2*bi]);bm.push_back(batch_meta[2*bi+1]);
            sm.push_back(single_meta[2*si]);sm.push_back(single_meta[2*si+1]);
            for(int d=0;d<D;++d){bp.push_back(batch_part[bi*D+d]);sp.push_back(single_part[si*D+d]);}
        }
        const auto dp=difference(sp,bp),dm=difference(sm,bm);
        std::printf("SINGLE n=%d pos=%d row=%d outputBits=%zu maxDelta=%.9g maxULP=%llu singleFP64Max=%.9g batchFP64Max=%.9g incumbentFP64Max=%.9g singleL2=%.9g batchL2=%.9g singleRel=%.9g batchRel=%.9g partBits=%zu partMax=%.9g partULP=%llu metaBits=%zu metaMax=%.9g metaULP=%llu\n",n,pos,t,delta.count,delta.max,(unsigned long long)delta.ulp,es.max,eb.max,is.max,std::sqrt(es.l2),std::sqrt(eb.l2),es.relative,eb.relative,dp.count,dp.max,(unsigned long long)dp.ulp,dm.count,dm.max,(unsigned long long)dm.ulp);
        if(delta.count)std::printf("FIRST output row=%d head=%zu dim=%zu single=%.9g batch=%.9g singleBits=%08x batchBits=%08x\n",t,delta.first/D,delta.first%D,one[delta.first],batch[delta.first],bits(one[delta.first]),bits(batch[delta.first]));
        if(dp.count)std::printf("FIRST part row=%d split=%zu head=%zu dim=%zu single=%.9g batch=%.9g singleBits=%08x batchBits=%08x\n",t,dp.first/(H*D),(dp.first/D)%H,dp.first%D,sp[dp.first],bp[dp.first],bits(sp[dp.first]),bits(bp[dp.first]));
        if(dm.count)std::printf("FIRST meta row=%d split=%zu head=%zu component=%s single=%.9g batch=%.9g singleBits=%08x batchBits=%08x\n",t,dm.first/(H*2),(dm.first/2)%H,dm.first%2?"denominator":"max",sm[dm.first],bm[dm.first],bits(sm[dm.first]),bits(bm[dm.first]));
        std::fflush(stdout);
        const double single_max=std::max(2e-6*std::max(vmax,1e-6),8*is.max),batch_max=std::max(2e-6*std::max(vmax,1e-6),8*ib.max);
        const double single_rel=std::max(5e-5,8*is.relative),batch_rel=std::max(5e-5,8*ib.relative);
        require(es.max<=single_max,"singleton FP64 max error exceeded");
        require(es.relative<=single_rel,"singleton FP64 relative L2 exceeded");
        require(eb.max<=batch_max&&eb.relative<=batch_rel,"batch row FP64 error exceeded");
        // A query moves between MMA fragment accumulator rows when run alone.
        // PTX leaves MMA's internal accumulation/rounding order unspecified.
        // Keep the original FP64 thresholds for BOTH executions, and require
        // their difference to fit one original error budget (not twice it).
        const std::vector<double> batch_double(batch.begin(),batch.end());
        const auto pair=error(one,batch_double);
        require(pair.max<=std::max(single_max,batch_max)&&pair.relative<=std::max(single_rel,batch_rel),"singleton/batch numerical error budget exceeded");
    }
    q.guard(q.count);gate.guard(gate.count);out.guard((size_t)n*H*D);old.guard((size_t)n*H*D);part.guard(part.count);meta.guard(meta.count);
    pold.guard(pold.count);mold.guard(mold.count);lold.guard(lold.count);cache.immutable(data);
    require(q.get(q.count)==data.q && gate.get(gate.count)==data.gate,"immutable input changed");
    // Unsupported capacity/n/shape calls may not write partial results.
    out.poison();require(!F::run(q.data(),cache.view,at,pos,5,H,KVH,ctx,1.f/16,part.data(),part.count,meta.data(),meta.count,gate.data(),out.data(),nullptr),"n5 accepted");
    require(!F::run(q.data(),cache.view,at,pos,n,H,KVH,ctx,1.f/16,part.data(),part.count-1,meta.data(),meta.count,gate.data(),out.data(),nullptr),"short scratch accepted");out.guard(0);
    require(!F::run(q.data(),cache.view,at,pos,n,H/2,KVH,ctx,1.f/16,part.data(),part.count,meta.data(),meta.count,gate.data(),out.data(),nullptr),"unsupported GQA accepted");out.guard(0);
    ck(cudaFree(at));
    std::printf("PASS n=%d pos=%d ctx=%d pattern=%d graph=%d vmm=%d max=%.9g incumbent=%.9g relL2=%.9g\n",n,pos,ctx,mode,graph,sparse,ef.max,ei.max,ef.relative);
}
} // namespace
int main(int argc,char**argv){try{
    require(F::available(),"SM80+ FDEC unavailable");
    require(F::part_floats(4095,KVH)==0 && F::meta_floats(4095,KVH)==0,"capacity lower bound");
    require(F::part_floats(262144,KVH)==(size_t)256*KVH*32*D,"part geometry");
    if(argc==2&&std::strcmp(argv[1],"--focus")==0){check_case(16387,1023,2,0);return 0;}
    for(int n=1;n<=4;++n){
        for(int p:{0,63,1023,4095,8191})check_case(16387,p,n,0,p==4095);
        for(int mode=1;mode<=6;++mode)check_case(8195,4095,n,mode);
    }
    if(VMM::api().ok)for(int n=1;n<=4;++n)check_case(65536,4095,n,5,true,true);
    else std::puts("SKIP VMM unmapped-future checks: runtime reports no VMM");
    std::puts("PASS compensated FDEC: FP64+production SIMT, masks/tails/scales, n1..4/singleton, graph, guards, VMM");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
