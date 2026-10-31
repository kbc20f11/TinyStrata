// GPU exact-verifier regression against the production CPU penalty oracle.
// From the repository root (then run the resulting executable on a GPU):
// CUDA: nvcc -std=c++20 -O2 -Iinclude -Isrc/qwen35 tests/clover-self-spec-gpu.cu -o clover-self-spec-gpu
// HIP: hipcc -std=c++20 -O2 -ffp-contract=off -Iinclude/strata/hip_compat -Iinclude -Isrc/qwen35 tests/clover-self-spec-gpu.cu -o clover-self-spec-gpu
// Set the compiler's GPU architecture option for the device being tested.
// This checks identical-logit verification, not model-level equality between
// sequential causal decoding and a batched forward pass. No weights are needed.
#include "clover_diffusion.cuh"
#include "strata/qwen35/clover_decode.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
namespace G = strata::qwen35::clover_diffusion_gpu;
namespace C = strata::qwen35::clover_decode;
#define GPU(call) do { const auto error=(call); if(error!=cudaSuccess)throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(error)); } while(0)
static int checks=0;
static void require(bool condition,const std::string& label) { if(!condition)throw std::runtime_error(label);++checks; }
struct Buffers {
    float *logits=nullptr,*penalized=nullptr,*length_bias=nullptr,*think_bias=nullptr;
    int64_t *counts=nullptr,*effective=nullptr;
    int32_t *candidates=nullptr,*prediction=nullptr,*bad=nullptr;
    cudaStream_t stream=nullptr;
    Buffers(int rows,int vocab) {
        GPU(cudaStreamCreate(&stream));
        GPU(cudaMalloc((void**)&logits,(size_t)rows*vocab*4));
        GPU(cudaMalloc((void**)&counts,(size_t)vocab*8));
        GPU(cudaMalloc((void**)&candidates,(size_t)rows*4));
        GPU(cudaMalloc((void**)&prediction,(size_t)rows*4));
        GPU(cudaMalloc((void**)&bad,4));
        if(vocab<1000) {
            GPU(cudaMalloc((void**)&penalized,(size_t)rows*vocab*4));
            GPU(cudaMalloc((void**)&effective,(size_t)rows*vocab*8));
            GPU(cudaMalloc((void**)&length_bias,(size_t)rows*4));
            GPU(cudaMalloc((void**)&think_bias,(size_t)rows*4));
        }
    }
    ~Buffers() {
        for(void* ptr:{(void*)logits,(void*)penalized,(void*)length_bias,(void*)think_bias,(void*)counts,(void*)effective,(void*)candidates,(void*)prediction,(void*)bad})
            if(ptr)(void)cudaFree(ptr);
        if(stream)(void)cudaStreamDestroy(stream);
    }
};
// Isolate float operation order as well as argmax. Prefix counts/biases for
// this diagnostic are supplied from the independently constructed CPU prefix.
__global__ void capture_values(const float* raw,const int64_t* counts,const float* lb,const float* tb,
    float* out,int rows,int vocab,G::Controls controls) {
    const int at=blockIdx.x*blockDim.x+threadIdx.x;
    if(at>=rows*vocab)return;
    const int row=at/vocab,id=at%vocab;bool invalid=false;
    out[at]=G::self_spec_exact_value(raw[at],id,counts[at],controls,lb[row],tb[row],invalid);
}
static void one(const std::string& name,int rows,int vocab,int threads,int pattern,int length=17) {
    G::Controls controls;controls.mask=2;controls.eos0=3;controls.eos1=4;controls.think_end=5;
    controls.frequency=.15f;controls.presence=.9f;controls.repetition=1.05f;
    std::mt19937 rng(7331+rows+vocab+pattern*17);std::uniform_real_distribution<float> random(-8,8);
    std::vector<float> raw((size_t)rows*vocab);
    for(float& value:raw)value=random(rng);
    std::vector<int32_t> response(length),candidates(rows,-1),prediction(rows);
    for(int i=0;i<length;++i)response[i]=6+(i%13);
    for(int row=0;row<rows-1;++row)candidates[row]=9+(row%3);
    bool invalid=false;
    if(pattern==1) { // Repeated proposals must change subsequent frequency penalties.
        std::fill(response.begin(),response.end(),6);
        std::fill(raw.begin(),raw.end(),-10.f);
        controls.frequency=.25f;controls.presence=.1f;
        for(int row=0;row<rows;++row) {
            raw[(size_t)row*vocab+9]=2.f;raw[(size_t)row*vocab+1]=1.7f;
            if(row<rows-1)candidates[row]=9;
        }
    }else if(pattern==2) { // Negative seen logits must multiply, not divide, by repetition.
        std::fill(response.begin(),response.end(),9);
        std::fill(raw.begin(),raw.end(),-100.f);
        for(int row=0;row<rows;++row) {raw[(size_t)row*vocab+9]=-2.f;raw[(size_t)row*vocab+1]=-2.3f;}
        controls.frequency=0;controls.presence=0;controls.repetition=1.2f;
    }else if(pattern==3 || pattern==8) { // Per-row causal think bias, then prefix suppression.
        std::fill(response.begin(),response.end(),6);
        if(pattern==8 && length)response.back()=controls.think_end;
        std::fill(raw.begin(),raw.end(),-100.f);
        controls.frequency=controls.presence=0;controls.repetition=1;
        for(int row=0;row<rows;++row) {raw[(size_t)row*vocab+1]=.001f;raw[(size_t)row*vocab+controls.think_end]=0;}
        if(rows>1)candidates[std::min(1,rows-2)]=controls.think_end;
    }else if(pattern==4) { // Duplicated EOS must receive its length bias once.
        controls.eos1=controls.eos0;controls.frequency=controls.presence=0;controls.repetition=1;
        std::fill(raw.begin(),raw.end(),-100.f);
        for(int row=0;row<rows;++row) {raw[(size_t)row*vocab+1]=.0018f;raw[(size_t)row*vocab+controls.eos0]=0;}
    }else if(pattern==5) {
        for(int row=0;row<rows;++row)raw[(size_t)row*vocab+controls.mask]=1000.f;
    }else if(pattern==6) {
        std::fill(raw.begin(),raw.end(),1.f);controls.frequency=controls.presence=0;
        controls.repetition=1;controls.length_strength=controls.think_strength=0;
    }else if(pattern==7) {
        for(int row=0;row<rows;++row)raw[(size_t)row*vocab+1]=-INFINITY;
    }else if(pattern==10) {raw[0]=NAN;invalid=true;}
    else if(pattern==11) {raw[0]=INFINITY;invalid=true;}
    else if(pattern==12) {std::fill(raw.begin(),raw.end(),-INFINITY);invalid=true;}
    else if(pattern==14) {candidates[0]=-2;invalid=true;}
    else if(pattern==15) {candidates[0]=vocab;invalid=true;}
    else if(pattern==16) { // A raw-preview mask proposal is a valid causal token.
        std::fill(response.begin(),response.end(),6);
        std::fill(raw.begin(),raw.end(),-10.f);
        controls.frequency=.25f;controls.presence=.1f;
        for(int row=0;row<rows;++row) {
            raw[(size_t)row*vocab+controls.mask]=2.f;raw[(size_t)row*vocab+1]=1.7f;
            if(row<rows-1)candidates[row]=controls.mask;
        }
    }
    else if(pattern==17) {candidates.back()=0;invalid=true;}
    std::vector<int64_t> counts(vocab);
    for(int token:response)++counts[token];
    if(pattern==13) {counts[0]=-1;invalid=true;}
    if(pattern==18) {counts[9]=std::numeric_limits<int64_t>::max();candidates[0]=9;invalid=true;}
    if(pattern==19) {counts[controls.mask]=-1;invalid=true;}
    Buffers device(rows,vocab);
    GPU(cudaMemcpyAsync(device.logits,raw.data(),raw.size()*4,cudaMemcpyHostToDevice,device.stream));
    GPU(cudaMemcpyAsync(device.counts,counts.data(),counts.size()*8,cudaMemcpyHostToDevice,device.stream));
    GPU(cudaMemcpyAsync(device.candidates,candidates.data(),candidates.size()*4,cudaMemcpyHostToDevice,device.stream));
    GPU(G::launch_self_spec_exact(device.logits,device.counts,response.size(),device.candidates,rows,vocab,controls,
        device.prediction,device.bad,device.stream,threads));
    int32_t bad=0;std::vector<float> unchanged(raw.size());std::vector<int64_t> unchanged_counts(counts.size());
    GPU(cudaMemcpyAsync(&bad,device.bad,4,cudaMemcpyDeviceToHost,device.stream));
    GPU(cudaMemcpyAsync(prediction.data(),device.prediction,prediction.size()*4,cudaMemcpyDeviceToHost,device.stream));
    GPU(cudaMemcpyAsync(unchanged.data(),device.logits,raw.size()*4,cudaMemcpyDeviceToHost,device.stream));
    GPU(cudaMemcpyAsync(unchanged_counts.data(),device.counts,counts.size()*8,cudaMemcpyDeviceToHost,device.stream));
    GPU(cudaStreamSynchronize(device.stream));
    require((bad!=0)==invalid,name+": bad flag");
    require(std::memcmp(raw.data(),unchanged.data(),raw.size()*4)==0,name+": raw logits changed");
    require(counts==unchanged_counts,name+": base count snapshot changed");
    if(invalid)return;
    C::PenaltyOptions options;options.frequency=controls.frequency;options.presence=controls.presence;options.repetition=controls.repetition;
    options.length_strength=controls.length_strength;options.length_start=controls.length_start;options.length_window=controls.length_window;
    options.think_strength=controls.think_strength;options.think_start=controls.think_start;options.think_window=controls.think_window;
    options.think_end=controls.think_end;options.stops={controls.eos0,controls.eos1};
    std::vector<float> expected(raw.size()),lb(rows),tb(rows);
    std::vector<int64_t> effective(device.effective?(size_t)rows*vocab:0);
    auto prefix=response;
    for(int row=0;row<rows;++row) {
        std::vector<float> values(raw.begin()+(size_t)row*vocab,raw.begin()+(size_t)(row+1)*vocab);
        // Use the actual CPU target, including its ability to select <|mask|>.
        // Preview proposal suppression is intentionally a separate policy.
        C::apply_penalties(values,prefix,options);
        const int selected=(int)(std::max_element(values.begin(),values.end())-values.begin());
        require(prediction[row]==selected,name+": argmax vs apply_penalties row "+std::to_string(row));
        if(pattern==6)require(selected==0,name+": lowest-ID tie");
        if(pattern==5)require(selected==controls.mask,name+": CPU target mask argmax preserved");
        std::copy(values.begin(),values.end(),expected.begin()+(size_t)row*vocab);
        lb[row]=C::ramp(prefix.size()+1,options.length_start,options.length_window,options.length_strength);
        tb[row]=std::find(prefix.begin(),prefix.end(),options.think_end)==prefix.end()
            ?C::ramp(prefix.size()+1,options.think_start,options.think_window,options.think_strength):0;
        if(device.effective)for(int token:prefix)++effective[(size_t)row*vocab+token];
        if(row<rows-1)prefix.push_back(candidates[row]);
    }
    if(device.effective) {
        GPU(cudaMemcpyAsync(device.effective,effective.data(),effective.size()*8,cudaMemcpyHostToDevice,device.stream));
        GPU(cudaMemcpyAsync(device.length_bias,lb.data(),lb.size()*4,cudaMemcpyHostToDevice,device.stream));
        GPU(cudaMemcpyAsync(device.think_bias,tb.data(),tb.size()*4,cudaMemcpyHostToDevice,device.stream));
        capture_values<<<(rows*vocab+255)/256,256,0,device.stream>>>(device.logits,device.effective,device.length_bias,device.think_bias,device.penalized,rows,vocab,controls);
        GPU(cudaGetLastError());std::vector<float> actual(raw.size());
        GPU(cudaMemcpyAsync(actual.data(),device.penalized,actual.size()*4,cudaMemcpyDeviceToHost,device.stream));
        GPU(cudaStreamSynchronize(device.stream));
        require(std::memcmp(expected.data(),actual.data(),actual.size()*4)==0,name+": penalty values differ bitwise");
    }
}
int main() {
    try {
        for(int threads:{256,512,1024}) {
            for(int rows:{1,2,4,8,16,32}) {
                for(int pattern=0;pattern<=8;++pattern)one("ordinary-"+std::to_string(pattern),rows,257,threads,pattern);
                one("mask-candidate-prefix",rows,257,threads,16);
                for(int length:{4094,4095,4096,8192}) {
                    one("think-prefix",rows,257,threads,3,length);
                    one("think-already-seen",rows,257,threads,8,length);
                    one("length-duplicate-eos",rows,257,threads,4,length);
                }
            }
            for(int pattern=10;pattern<=19;++pattern)if(pattern!=16)one("invalid-"+std::to_string(pattern),4,257,threads,pattern);
            one("full-vocabulary",4,248320,threads,0,4096);
            one("full-vocabulary-repeated",4,248320,threads,1,8192);
        }
        std::printf("{\"passed\":true,\"checks\":%d}\n",checks);
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
