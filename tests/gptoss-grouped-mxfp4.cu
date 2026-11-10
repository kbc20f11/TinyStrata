// Compare every output against the existing native IDs/slots and optimized down
// kernels. Includes repeated Experts, unchanged rows/padding, and graph replay.
#include <cstdio>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
int main() { std::puts("SKIP: CUDA or HIP required"); return 77; }
#else
#include "strata/gemma4/grouped_mxfp4.hpp"
#include "strata/gemma4/mxfp4_down.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace K = strata::kernels;
namespace G = strata::kernels::grouped_mxfp4;
#define CK(x) do { const auto err=(x); if(err!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(err)); } while(0)
static unsigned checks=0,cases=0;
static uint64_t compared=0;
static void need(bool condition,const char* why) { if(!condition)throw std::runtime_error(why);++checks; }
struct Device {
    uint8_t* base=nullptr;
    size_t bytes;
    cudaStream_t stream;
    Device(size_t n,cudaStream_t s):bytes(n),stream(s) {
        CK(cudaMalloc(reinterpret_cast<void**>(&base),bytes+128));
        CK(cudaMemsetAsync(base,0xA7,bytes+128,stream));
    }
    ~Device(){if(base)(void)cudaFree(base);}
    template<class T>T* data()const{return reinterpret_cast<T*>(base+64);}
    void guards()const {
        uint8_t left[64],right[64];CK(cudaStreamSynchronize(stream));
        CK(cudaMemcpy(left,base,64,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(right,base+64+bytes,64,cudaMemcpyDeviceToHost));
        for(int i=0;i<64;++i)need(left[i]==0xA7&&right[i]==0xA7,"device red zone overwritten");
    }
};
struct Fixture {
    static constexpr int width=2880,rows=32,expert_count=4;
    static constexpr size_t up_bytes=size_t(5760)*90*17,down_bytes=size_t(2880)*90*17;
    static constexpr size_t matrix_bytes=up_bytes+down_bytes;
    static constexpr int stride=5760+16;
    cudaStream_t stream;
    Device weights,table,ids,source,q8,workspace,reference,actual;
    std::vector<const void*> pointers;
    std::vector<int32_t> host_ids;
    size_t qrow;
    Fixture(cudaStream_t s):stream(s),weights(expert_count*(matrix_bytes+256),s),table(32*sizeof(void*),s),
        ids(rows*sizeof(int32_t),s),source(size_t(rows)*width*sizeof(float),s),
        q8(rows*K::gemma_native_q8_1_bytes(width),s),workspace(G::workspace_bytes(),s),
        reference(size_t(rows)*stride*sizeof(float),s),actual(size_t(rows)*stride*sizeof(float),s),
        pointers(32),host_ids(rows),qrow(K::gemma_native_q8_1_bytes(width)) {
        std::vector<uint8_t> data(expert_count*(matrix_bytes+256),0);
        for(int expert=0;expert<expert_count;++expert) {
            uint8_t* dst=data.data()+size_t(expert)*(matrix_bytes+256);
            uint32_t rng=1234+expert*991;
            for(size_t block=0;block<matrix_bytes/17;++block) {
                rng=rng*1664525u+1013904223u;
                dst[block*17]=block%509==0?0:block%1021==0?1:uint8_t(118+(rng%15));
                for(int j=0;j<16;++j){rng=rng*1664525u+1013904223u;dst[block*17+1+j]=uint8_t(rng>>24);}
            }
        }
        CK(cudaMemcpyAsync(weights.data<void>(),data.data(),data.size(),cudaMemcpyHostToDevice,stream));
        for(int expert=0;expert<32;++expert)
            pointers[size_t(expert)]=weights.data<uint8_t>()+size_t(expert%expert_count)*(matrix_bytes+256);
        CK(cudaMemcpyAsync(table.data<void>(),pointers.data(),table.bytes,cudaMemcpyHostToDevice,stream));
        change_input(0);
        CK(cudaStreamSynchronize(stream));
    }
    void change_input(int phase) {
        std::vector<float> values(size_t(rows)*width);
        for(size_t i=0;i<values.size();++i)
            values[i]=i%103==0?0.f:std::sin(float((i*17+phase*11)%997)*.071f)*float(1+(i%3));
        CK(cudaMemcpyAsync(source.data<void>(),values.data(),source.bytes,cudaMemcpyHostToDevice,stream));
        K::gemma_native_quantize_q8_1(source.data<float>(),q8.data<void>(),width*rows,1,stream);
        CK(cudaStreamSynchronize(stream));
    }
    void pattern(int count,int kind) {
        for(int row=0;row<count;++row)
            host_ids[size_t(row)]=kind==0?2:kind==1?(row*7+row/3)%4:row;
        CK(cudaMemcpyAsync(ids.data<void>(),host_ids.data(),count*sizeof(int32_t),cudaMemcpyHostToDevice,stream));
    }
    K::MmvqIds ids_bundle(bool down,bool output_actual)const {
        return {table.data<const void*>(),ids.data<int32_t>(),down?up_bytes:0,q8.data<void>(),qrow,
                down?1:4,output_actual?actual.data<float>():reference.data<float>(),size_t(stride)};
    }
    K::MmvqSlots slots_bundle(int count,bool down,bool output_actual)const {
        K::MmvqSlots bundle{};
        for(int row=0;row<count;++row) {
            bundle.w[row]=static_cast<const uint8_t*>(pointers[size_t(host_ids[size_t(row)])])+(down?up_bytes:0);
            bundle.x[row]=q8.data<uint8_t>()+size_t(down?row:(row*7+row/3)%8)*qrow;
            bundle.y[row]=(output_actual?actual.data<float>():reference.data<float>())+size_t(row)*stride;
        }
        return bundle;
    }
    void clear() {
        CK(cudaMemsetAsync(reference.data<void>(),0x5A,reference.bytes,stream));
        CK(cudaMemsetAsync(actual.data<void>(),0x5A,actual.bytes,stream));
    }
    void compare(int count,int out,const char* label) {
        std::vector<float> a(size_t(rows)*stride),b(a.size());
        CK(cudaMemcpyAsync(a.data(),reference.data<void>(),reference.bytes,cudaMemcpyDeviceToHost,stream));
        CK(cudaMemcpyAsync(b.data(),actual.data<void>(),actual.bytes,cudaMemcpyDeviceToHost,stream));
        CK(cudaStreamSynchronize(stream));
        for(size_t i=0;i<a.size();++i) {
            const bool used=i/stride<size_t(count)&&i%stride<size_t(out);
            if((used&&(!std::isfinite(a[i])||!std::isfinite(b[i])))||std::memcmp(&a[i],&b[i],sizeof(float))) {
                std::fprintf(stderr,"FAIL %s count=%d out=%d at=%zu reference=%.9g grouped=%.9g\n",label,count,out,i,a[i],b[i]);
                throw std::runtime_error("grouped Expert output/padding is not bitwise equal");
            }
        }
        compared+=a.size()*sizeof(float);++cases;++checks;
    }
    void baseline(bool slots,int count,bool down,bool optimized_down) {
        if(slots) {
            const auto b=slots_bundle(count,down,false);
            const bool done=optimized_down?K::mxfp4_down::slots(39,b,count,width,width,stream):
                K::gemma_native_mmvq_slots(39,b,count,width,down?width:2*width,stream);
            need(done,"reference slots kernel unsupported");
        } else {
            const auto b=ids_bundle(down,false);
            const bool done=optimized_down?K::mxfp4_down::ids(39,b,count,width,width,stream):
                K::gemma_native_mmvq_ids(39,b,count,width,down?width:2*width,stream);
            need(done,"reference IDs kernel unsupported");
        }
    }
    void grouped(bool slots,int count,bool down,bool optimized_down) {
        if(slots) {
            // Prepare with the gate/up base pointers and reuse that same map
            // for down, whose per-Expert pointers all have a common offset.
            const auto gate=slots_bundle(count,false,true),b=slots_bundle(count,down,true);
            need(G::prepare_slots(gate,count,workspace.data<void>(),workspace.bytes,stream),"prepare slots unsupported");
            need(G::slots(39,b,count,width,down?width:2*width,workspace.data<void>(),optimized_down,stream),"grouped slots unsupported");
        } else {
            need(G::prepare_ids(ids.data<int32_t>(),count,32,workspace.data<void>(),workspace.bytes,stream),"prepare IDs unsupported");
            need(G::ids(39,ids_bundle(down,true),count,width,down?width:2*width,workspace.data<void>(),optimized_down,stream),"grouped IDs unsupported");
        }
    }
    void one(bool slots,int count,int kind,bool down,bool optimized_down,bool graph) {
        pattern(count,kind);clear();baseline(slots,count,down,optimized_down);
        if(graph)CK(cudaStreamSynchronize(stream));
        cudaGraph_t plan=nullptr;cudaGraphExec_t exec=nullptr;
        if(graph)CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        grouped(slots,count,down,optimized_down);
        if(graph) {
            CK(cudaStreamEndCapture(stream,&plan));CK(cudaGraphInstantiate(&exec,plan,nullptr,nullptr,0));
            CK(cudaGraphLaunch(exec,stream));
        }
        compare(count,down?width:2*width,slots?"slots":"ids");
        if(graph) {CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(plan));}
    }
    void changed_graph() {
        const int count=32;pattern(count,0);clear();baseline(false,count,false,false);CK(cudaStreamSynchronize(stream));
        cudaGraph_t plan=nullptr;cudaGraphExec_t exec=nullptr;
        CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));grouped(false,count,false,false);
        CK(cudaStreamEndCapture(stream,&plan));CK(cudaGraphInstantiate(&exec,plan,nullptr,nullptr,0));
        CK(cudaGraphLaunch(exec,stream));compare(count,2*width,"graph before changed IDs");
        pattern(count,1);change_input(1);clear();baseline(false,count,false,false);
        CK(cudaGraphLaunch(exec,stream));compare(count,2*width,"graph after changed IDs/data");
        CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(plan));
    }
    void invalid_ids() {
        pattern(8,1);host_ids[3]=32;CK(cudaMemcpyAsync(ids.data<void>(),host_ids.data(),8*sizeof(int32_t),cudaMemcpyHostToDevice,stream));
        clear();grouped(false,8,false,false);
        std::vector<float> output(actual.bytes/sizeof(float));
        CK(cudaMemcpyAsync(output.data(),actual.data<void>(),actual.bytes,cudaMemcpyDeviceToHost,stream));CK(cudaStreamSynchronize(stream));
        for(int row=0;row<8;++row)for(int col=0;col<5760;++col)
            need(std::isnan(output[size_t(row)*stride+col]),"out-of-range Expert ID was not rejected safely");
        need(!G::prepare_ids(ids.data<int32_t>(),1,32,workspace.data<void>(),workspace.bytes,stream),"single-row preparation must stay unsupported");
        need(!G::ids(39,ids_bundle(false,true),1,width,2*width,workspace.data<void>(),false,stream),"single-row product must keep original path");
        bool refused=false;
        try{G::prepare_ids(ids.data<int32_t>(),8,32,workspace.data<void>(),workspace.bytes-1,stream);}catch(const std::invalid_argument&){refused=true;}
        need(refused,"undersized metadata workspace accepted");
    }
    void guards(){weights.guards();table.guards();ids.guards();source.guards();q8.guards();workspace.guards();reference.guards();actual.guards();}
};
int main() {
    try {
        int devices=0;CK(cudaGetDeviceCount(&devices));if(!devices)return 77;
        cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        {
            Fixture f(stream);
            for(bool slots:{false,true}) {
                const std::vector<int> counts=slots?std::vector<int>{2,3,8,17,32}:std::vector<int>{8,16,32};
                for(int count:counts) {
                    if(slots&&count>K::kMmvqSlotsMax)continue;
                    for(int kind:{0,1,2})for(int product:{0,1,2}) {
#if !defined(__HIPCC__)
                        if(product==2)continue;
#endif
                        for(bool graph:{false,true})f.one(slots,count,kind,product!=0,product==2,graph);
                    }
                }
            }
            f.changed_graph();f.invalid_ids();f.guards();
        }
        CK(cudaStreamDestroy(stream));
        std::printf("PASS grouped MXFP4 cases=%u checks=%u bytes=%llu exact native/optimized-down IDs/slots, repeated Experts, graphs and guards\n",
                    cases,checks,(unsigned long long)compared);
        return 0;
    }catch(const std::exception&error){std::fprintf(stderr,"grouped_mxfp4_test: %s\n",error.what());return 1;}
}
#endif
