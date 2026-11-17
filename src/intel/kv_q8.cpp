#include "strata/intel/kv_q8.hpp"
#include "strata/intel/fp8_codec.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/hf8.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>
namespace strata::intel {
namespace {
bool fused_default(const sycl::device& device) {
    // Cache immutable SYCL facts by actual device. This hot path never calls
    // inspect_device, Level Zero/DXGI identity or Windows memory telemetry.
    // Thread-local storage avoids a lock on each layer of the single request.
    thread_local std::vector<std::pair<sycl::device,bool>> cache;
    for(const auto& entry:cache)if(entry.first==device)return entry.second;
    bool enabled=false;
    try {
        namespace exp=sycl::ext::oneapi::experimental;
        const auto subgroups=device.get_info<sycl::info::device::sub_group_sizes>();
        // PTL-H is the measured architecture with GRF256 kernel support. SYCL
        // exposes the GRF kernel property, not a supported-GRF-size device list;
        // never infer this support from the executable's AOT compilation flag.
        enabled=device.is_gpu() && device.get_info<sycl::info::device::vendor_id>()==0x8086 &&
            device.get_info<exp::info::device::architecture>()==exp::architecture::intel_gpu_ptl_h &&
            device.has(sycl::aspect::ext_intel_esimd) &&
            std::find(subgroups.begin(),subgroups.end(),size_t(32))!=subgroups.end() &&
            device.get_info<sycl::info::device::max_work_group_size>()>=128;
    } catch(const sycl::exception&) {
        // Missing architecture/capability metadata keeps the generic baseline.
    }
    cache.emplace_back(device,enabled);
    return enabled;
}
bool fused_requested(sycl::queue& queue) {
    if(const char* requested=std::getenv("TINYSTRATA_INTEL_FUSED_Q8_ATTN")) {
        if(std::strcmp(requested,"0")==0)return false;
        if(std::strcmp(requested,"1")==0)return true;
    }
    return fused_default(queue.get_device());
}
bool native_fp8_conversion(const sycl::device& device) {
    if(const char* setting=std::getenv("TINYSTRATA_INTEL_FP8_NATIVE_CVT"))
        if(std::strcmp(setting,"0")==0)return false;
    thread_local std::vector<std::pair<sycl::device,bool>> cache;
    for(const auto& entry:cache)if(entry.first==device)return entry.second;
    bool enabled=false;
    try {
        namespace exp=sycl::ext::oneapi::experimental;
        // Only PTL-H's vector HF8->half instruction was verified on hardware.
        // Header availability and a generic ESIMD aspect do not prove this.
        enabled=device.is_gpu()&&device.get_info<sycl::info::device::vendor_id>()==0x8086&&
            device.get_info<exp::info::device::architecture>()==exp::architecture::intel_gpu_ptl_h&&
            device.has(sycl::aspect::ext_intel_esimd);
    } catch(const sycl::exception&) {}
    cache.emplace_back(device,enabled);
    return enabled;
}
void validate(const sycl::queue& q, Q8Kv c) {
    if (!q.has_property<sycl::property::queue::in_order>() || c.context < 1 || c.heads < 1 ||
        !c.key || !c.value || !c.key_scale || !c.value_scale)
        throw std::invalid_argument("Invalid Intel Q8 KV geometry/queue");
}
// Preserve each query's original FMA/reduction order while sharing decoded
// KV words across three/four queries. Admission uses the CPU-known visible end.
template<bool FP8> inline float coefficient(int8_t code,float scale) {
    if constexpr(FP8)return fp8::decode(uint8_t(code))*scale;
    else return float(code)*scale;
}
template<int Tokens,bool FP8> sycl::event token_reuse_attention(sycl::queue& q,const float* query,const float* gate,Q8Kv c,
    float* scores,float* partial,float* output,int heads,const int32_t* position,int extent,sycl::event* stages) {
    static_assert(Tokens>=3 && Tokens<=4);
    constexpr size_t group=128,subgroup=16,per_group=group/subgroup,block=256;
    const size_t entries=size_t(heads)*extent,splits=(c.context+block-1)/block;
    auto qk=q.parallel_for(sycl::nd_range<1>(((entries+per_group-1)/per_group)*group,group),[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const auto sg=it.get_sub_group();const size_t entry=it.get_group_linear_id()*per_group+sg.get_group_linear_id();
        const size_t head=entry%heads,cell=entry/heads,lane=sg.get_local_linear_id();
        if(cell>=size_t(extent))return;
        bool active[Tokens];bool any=false;float sum[Tokens]={};
#pragma unroll
        for(int token=0;token<Tokens;++token){active[token]=cell<=size_t(position[token]);any=any||active[token];}
        if(!any)return;
        const size_t kv=head/(heads/c.heads);
        for(size_t word=lane;word<64;word+=subgroup){
            const size_t col=word*4,i=(kv*c.context+cell)*256+col;
            uint32_t codes;std::memcpy(&codes,c.key+i,sizeof(codes));
            const float scale=float(c.key_scale[i/32]);
#pragma unroll
            for(int j=0;j<4;++j){
                const float value=coefficient<FP8>(int8_t(codes>>(8*j)),scale);
#pragma unroll
                for(int token=0;token<Tokens;++token)if(active[token])
                    sum[token]=sycl::fma(query[(size_t(token)*heads+head)*256+col+j],value,sum[token]);
            }
        }
#pragma unroll
        for(int token=0;token<Tokens;++token){
            const float reduced=sycl::reduce_over_group(sg,sum[token],sycl::plus<float>());
            if(lane==0 && active[token])scores[(size_t(token)*heads+head)*c.context+cell]=reduced/16.f;
        }
    });
    auto softmax=q.parallel_for(sycl::nd_range<2>({size_t(Tokens),size_t(heads)*group},{1,group}),[=](sycl::nd_item<2> it){
        const size_t head=it.get_group(0)*heads+it.get_group(1),lane=it.get_local_linear_id(),end=position[it.get_group(0)];float maximum=-INFINITY;
        for(size_t cell=lane;cell<=end;cell+=group)maximum=sycl::fmax(maximum,scores[head*c.context+cell]);
        maximum=sycl::reduce_over_group(it.get_group(),maximum,sycl::maximum<float>());float sum=0;
        for(size_t cell=lane;cell<=end;cell+=group)sum+=sycl::exp(scores[head*c.context+cell]-maximum);
        sum=sycl::reduce_over_group(it.get_group(),sum,sycl::plus<float>());
        for(size_t cell=lane;cell<=end;cell+=group)scores[head*c.context+cell]=sycl::exp(scores[head*c.context+cell]-maximum)/sum;
    });
    constexpr size_t pv_group=64;
    auto pv=q.parallel_for(sycl::nd_range<1>(size_t((extent+block-1)/block)*heads*pv_group,pv_group),[=](sycl::nd_item<1> it){
        const size_t head=it.get_group_linear_id()%heads,split=it.get_group_linear_id()/heads,lane=it.get_local_linear_id(),kv=head/(heads/c.heads);
        const size_t col=lane*4,begin=split*block;float a[Tokens][4]={};
        int latest=position[0];
#pragma unroll
        for(int token=1;token<Tokens;++token)latest=sycl::max(latest,position[token]);
        const size_t end=sycl::min(begin+block,size_t(latest+1));
        for(size_t cell=begin;cell<end;++cell){
            const size_t i=(kv*c.context+cell)*256+col;
            uint32_t codes;std::memcpy(&codes,c.value+i,sizeof(codes));
            const float scale=float(c.value_scale[i/32]);
#pragma unroll
            for(int j=0;j<4;++j){
                const float value=coefficient<FP8>(int8_t(codes>>(8*j)),scale);
#pragma unroll
                for(int token=0;token<Tokens;++token)if(cell<=size_t(position[token]))
                    a[token][j]=sycl::fma(scores[(size_t(token)*heads+head)*c.context+cell],value,a[token][j]);
            }
        }
#pragma unroll
        for(int token=0;token<Tokens;++token){
            const size_t target=((size_t(token)*heads+head)*splits+split)*256+col;
#pragma unroll
            for(int j=0;j<4;++j)partial[target+j]=a[token][j];
        }
    });
    if(stages){stages[0]=qk;stages[1]=softmax;stages[2]=pv;}
    return q.parallel_for(sycl::range<2>(size_t(Tokens),size_t(heads)*256),[=](sycl::id<2> id){
        const size_t head=id[0]*heads+id[1]/256,col=id[1]%256,used=(size_t(position[id[0]]+1)+block-1)/block;float sum=0;
        for(size_t split=0;split<used;++split)sum+=partial[(head*splits+split)*256+col];
        const size_t index=head*256+col;
        output[index]=sum/(1.f+sycl::exp(-gate[index]));
    });
}
}
template<bool FP8> sycl::event store_kv_impl(sycl::queue& q,const float* key,const float* value,Q8Kv c,
    const int32_t* position,int tokens,int32_t* error) {
    validate(q,c);
    if(tokens<1 || tokens>1024 || !error)throw std::invalid_argument("Invalid Q8 KV store");
    return q.parallel_for(sycl::nd_range<1>(size_t(tokens)*c.heads*256,32),[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const size_t i=it.get_global_linear_id(), row=i/256, col=i%256, token=row/c.heads, head=row%c.heads;
        const int cell=position[0]+int(token);
        const auto sg=it.get_sub_group();
        if(cell<0 || cell>=c.context){
            if(sg.get_local_linear_id()==0)sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                sycl::access::address_space::global_space>(error[0]).store(1);
            return;
        }
        const float k=key[i],v=value[i];
        const bool nonfinite=sycl::any_of_group(sg,!sycl::isfinite(k)||!sycl::isfinite(v));
        const float km=sycl::reduce_over_group(sg,sycl::fabs(k),sycl::maximum<float>());
        const float vm=sycl::reduce_over_group(sg,sycl::fabs(v),sycl::maximum<float>());
        if constexpr(FP8){
            sycl::half kd=sycl::half(km/fp8::maximum),vd=sycl::half(vm/fp8::maximum);
            // Retain tiny finite blocks without a zero denominator. The
            // smallest FP16 subnormal is representable in the scale array.
            if(km>0.f&&float(kd)==0.f)kd=sycl::half(0x1p-24f);
            if(vm>0.f&&float(vd)==0.f)vd=sycl::half(0x1p-24f);
            const bool bad=nonfinite||!sycl::isfinite(float(kd))||!sycl::isfinite(float(vd));
            const size_t target=(head*c.context+size_t(cell))*256+col;
            c.key[target]=!bad&&km>0.f?int8_t(fp8::encode(k/float(kd))):0;
            c.value[target]=!bad&&vm>0.f?int8_t(fp8::encode(v/float(vd))):0;
            if(sg.get_local_linear_id()==0){
                c.key_scale[target/32]=bad?sycl::half(0.f):kd;
                c.value_scale[target/32]=bad?sycl::half(0.f):vd;
                if(bad)sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(error[0]).store(1);
            }
            return;
        }
        const sycl::half kd=sycl::half(km/127.f),vd=sycl::half(vm/127.f);
        const bool bad=nonfinite || !sycl::isfinite(float(kd)) || !sycl::isfinite(float(vd));
        const size_t target=(head*c.context+size_t(cell))*256+col;
        const float kc=!bad && km>0 ? sycl::rint(k/km*127.f) : 0.f;
        const float vc=!bad && vm>0 ? sycl::rint(v/vm*127.f) : 0.f;
        c.key[target]=int8_t(sycl::clamp(kc,-127.f,127.f));
        c.value[target]=int8_t(sycl::clamp(vc,-127.f,127.f));
        if(sg.get_local_linear_id()==0){
            c.key_scale[target/32]=bad ? sycl::half(0.f) : kd;
            c.value_scale[target/32]=bad ? sycl::half(0.f) : vd;
            if(bad)sycl::atomic_ref<int32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                sycl::access::address_space::global_space>(error[0]).store(1);
        }
    });
}
template<bool FP8> sycl::event dequantize_kv_impl(sycl::queue& q,Q8Kv c,float* key,float* value,int visible){
    validate(q,c);
    if(visible<1||visible>c.context)throw std::invalid_argument("Invalid Q8 KV visibility");
    return q.parallel_for(sycl::range<1>(size_t(c.heads)*visible*256),[=](sycl::id<1> id){
        const size_t head=id[0]/(size_t(visible)*256),within=id[0]%(size_t(visible)*256),i=head*c.context*256+within;
        key[i]=coefficient<FP8>(c.key[i],float(c.key_scale[i/32]));
        value[i]=coefficient<FP8>(c.value[i],float(c.value_scale[i/32]));
    });
}
sycl::event dequantize_fp8_native(sycl::queue& q,Q8Kv c,float* key,float* value,int visible){
    validate(q,c);
    if(visible<1||visible>c.context)throw std::invalid_argument("Invalid FP8 KV visibility");
    return q.parallel_for(sycl::range<1>(size_t(c.heads)*visible*8),[=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
        namespace es=sycl::ext::intel::esimd;
        using hf8=sycl::ext::intel::experimental::esimd::hf8;
        const size_t head=id[0]/(size_t(visible)*8),within=id[0]%(size_t(visible)*8);
        const size_t offset=head*size_t(c.context)*256+within*32;
        es::simd<uint8_t,32> raw_key(reinterpret_cast<const uint8_t*>(c.key+offset));
        es::simd<uint8_t,32> raw_value(reinterpret_cast<const uint8_t*>(c.value+offset));
        es::simd<hf8,32> fp8_key=raw_key.template bit_cast_view<hf8>();
        es::simd<hf8,32> fp8_value=raw_value.template bit_cast_view<hf8>();
        es::simd<sycl::half,32> half_key=fp8_key,half_value=fp8_value;
        es::simd<float,32> decoded_key=half_key,decoded_value=half_value;
        (decoded_key*float(c.key_scale[offset/32])).copy_to(key+offset);
        (decoded_value*float(c.value_scale[offset/32])).copy_to(value+offset);
    });
}
template<bool FP8> sycl::event fused_attention_impl(sycl::queue& q,const float* query,const float* gate,Q8Kv c,
    float* stats,float* partial,float* output,int heads,const int32_t* position,int extent,
    sycl::event* partial_stage,int tokens) {
    validate(q,c);
    if(!q.has_property<sycl::property::queue::in_order>() || heads!=16 || c.heads!=2 ||
       extent<1 || extent>c.context || c.context<2 || tokens<1 || tokens>8)
        throw std::invalid_argument("Invalid register Q8 D256/GQA8 geometry");
    constexpr size_t width=256,lanes=32,block=256;
    const size_t capacity_splits=(c.context+block-1)/block;
    const size_t launched_splits=(extent+block-1)/block;
    auto fused=q.submit([&](sycl::handler& h){
        const auto properties=sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::grf_size<256>};
        h.parallel_for(sycl::nd_range<1>(size_t(tokens)*c.heads*launched_splits*lanes,lanes),properties,
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const size_t job=it.get_group_linear_id(),split=job%launched_splits;
            const size_t kv=(job/launched_splits)%2,token=job/(launched_splits*2);
            const size_t begin=split*block,end=sycl::min(begin+block,size_t(position[token]+1));
            if(begin>=end)return;
            const auto sg=it.get_sub_group();
            const size_t lane=sg.get_local_linear_id();
            // Keep the small read-only Q working set in cache. Hoisting all
            // 64 Q coefficients per lane would double the accumulator pressure.
            const volatile float* query_stream=query;
            float accumulator[8][8]={},maximum[8],denominator[8]={};
#pragma unroll
            for(int head=0;head<8;++head)maximum[head]=-INFINITY;
            for(size_t cell=begin;cell<end;++cell) {
                float key[8],value[8];
#pragma unroll
                for(int j=0;j<8;++j) {
                    const size_t source=(kv*c.context+cell)*width+lane+j*lanes;
                    key[j]=coefficient<FP8>(c.key[source],float(c.key_scale[source/32]));
                    value[j]=coefficient<FP8>(c.value[source],float(c.value_scale[source/32]));
                }
#pragma unroll
                for(int head=0;head<8;++head) {
                    const size_t row=token*heads+kv*8+head;
                    float dot=0.f;
#pragma unroll
                    for(int j=0;j<8;++j)
                        dot=sycl::fma(query_stream[row*width+lane+j*lanes],key[j],dot);
                    const float score=sycl::reduce_over_group(sg,dot,sycl::plus<float>())/16.f;
                    const float next_maximum=sycl::fmax(maximum[head],score);
                    const float previous_scale=sycl::exp(maximum[head]-next_maximum);
                    const float weight=sycl::exp(score-next_maximum);
                    denominator[head]=sycl::fma(denominator[head],previous_scale,weight);
#pragma unroll
                    for(int j=0;j<8;++j)
                        accumulator[head][j]=sycl::fma(value[j],weight,accumulator[head][j]*previous_scale);
                    maximum[head]=next_maximum;
                }
            }
#pragma unroll
            for(int head=0;head<8;++head) {
                const size_t row=token*heads+kv*8+head;
                if(lane==0){stats[row*c.context+split*2]=maximum[head];stats[row*c.context+split*2+1]=denominator[head];}
#pragma unroll
                for(int j=0;j<8;++j)
                    partial[(row*capacity_splits+split)*width+lane+j*lanes]=accumulator[head][j];
            }
        });
    });
    if(partial_stage)*partial_stage=fused;
    return q.submit([&](sycl::handler& h){
        sycl::local_accessor<float,1> factors(sycl::range<1>(capacity_splits+1),h);
        h.parallel_for(sycl::nd_range<1>(size_t(tokens)*heads*128,128),[=](sycl::nd_item<1> it){
            const size_t row=it.get_group_linear_id(),lane=it.get_local_linear_id();
            const size_t used=(size_t(position[row/heads]+1)+block-1)/block;
            if(lane==0) {
                float maximum=-INFINITY,denominator=0.f;
                for(size_t split=0;split<used;++split)maximum=sycl::fmax(maximum,stats[row*c.context+split*2]);
                for(size_t split=0;split<used;++split) {
                    const float factor=sycl::exp(stats[row*c.context+split*2]-maximum);
                    factors[split]=factor;
                    denominator=sycl::fma(stats[row*c.context+split*2+1],factor,denominator);
                }
                factors[capacity_splits]=denominator;
            }
            it.barrier(sycl::access::fence_space::local_space);
            for(size_t col=lane;col<width;col+=128) {
                float numerator=0.f;
                for(size_t split=0;split<used;++split)
                    numerator=sycl::fma(partial[(row*capacity_splits+split)*width+col],factors[split],numerator);
                const size_t index=row*width+col;
                output[index]=(numerator/factors[capacity_splits])/(1.f+sycl::exp(-gate[index]));
            }
        });
    });
}
template<bool FP8> sycl::event decode_attention_impl(sycl::queue& q,const float* query,const float* gate,Q8Kv c,
    float* scores,float* partial,float* output,int heads,const int32_t* position,int extent,sycl::event* stages,int tokens,int actual_visible){
    validate(q,c);
    if(heads<1||heads%c.heads||extent<1||extent>c.context||tokens<1||tokens>8)throw std::invalid_argument("Invalid Q8 decode attention geometry");
    if(actual_visible<0 || actual_visible>extent)throw std::invalid_argument("Invalid Q8 actual visibility");
    // The measured PTL-H device defaults to register GQA8 fusion. Explicit0/1
    // overrides that device policy. Old profiling expects three stage events;
    // graphs with a zero visibility hint retain the original implementation.
    if(!stages && heads==16 && c.heads==2 && c.context>=2 && actual_visible>=16384 && fused_requested(q))
        return fused_decode_attention_q8(q,query,gate,c,scores,partial,output,heads,position,extent,nullptr,tokens);
    if(actual_visible>=16384 && tokens==3)return token_reuse_attention<3,FP8>(q,query,gate,c,scores,partial,output,heads,position,extent,stages);
    if(actual_visible>=16384 && tokens==4)return token_reuse_attention<4,FP8>(q,query,gate,c,scores,partial,output,heads,position,extent,stages);
    constexpr size_t group=128,subgroup=16,per_group=group/subgroup,block=256;
    const size_t entries=size_t(heads)*extent,splits=(c.context+block-1)/block;
    auto qk=q.parallel_for(sycl::nd_range<2>({size_t(tokens),((entries+per_group-1)/per_group)*group},{1,group}),[=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        const auto sg=it.get_sub_group();const size_t entry=it.get_group(1)*per_group+sg.get_group_linear_id();
        const size_t head=entry%heads,cell=entry/heads,token=it.get_group(0),lane=sg.get_local_linear_id();
        if(cell>=size_t(extent)||cell>size_t(position[token]))return;
        const size_t row=token*heads+head;
        const size_t kv=head/(heads/c.heads);float sum=0;
        for(size_t word=lane;word<64;word+=subgroup){
            const size_t col=word*4,i=(kv*c.context+cell)*256+col;
            uint32_t codes;std::memcpy(&codes,c.key+i,sizeof(codes));
            const float scale=float(c.key_scale[i/32]);
#pragma unroll
            for(int j=0;j<4;++j)
                sum=sycl::fma(query[row*256+col+j],coefficient<FP8>(int8_t(codes>>(8*j)),scale),sum);
        }
        sum=sycl::reduce_over_group(sg,sum,sycl::plus<float>());
        if(lane==0)scores[row*c.context+cell]=sum/16.f;
    });
    auto softmax=q.parallel_for(sycl::nd_range<2>({size_t(tokens),size_t(heads)*group},{1,group}),[=](sycl::nd_item<2> it){
        const size_t head=it.get_group(0)*heads+it.get_group(1),lane=it.get_local_linear_id(),end=position[it.get_group(0)];float maximum=-INFINITY;
        for(size_t cell=lane;cell<=end;cell+=group)maximum=sycl::fmax(maximum,scores[head*c.context+cell]);
        maximum=sycl::reduce_over_group(it.get_group(),maximum,sycl::maximum<float>());float sum=0;
        for(size_t cell=lane;cell<=end;cell+=group)sum+=sycl::exp(scores[head*c.context+cell]-maximum);
        sum=sycl::reduce_over_group(it.get_group(),sum,sycl::plus<float>());
        for(size_t cell=lane;cell<=end;cell+=group)scores[head*c.context+cell]=sycl::exp(scores[head*c.context+cell]-maximum)/sum;
    });
    constexpr size_t pv_group=64;
    auto pv=q.parallel_for(sycl::nd_range<2>({size_t(tokens),size_t((extent+block-1)/block)*heads*pv_group},{1,pv_group}),[=](sycl::nd_item<2> it){
        const size_t local_head=it.get_group(1)%heads,head=it.get_group(0)*heads+local_head,split=it.get_group(1)/heads,lane=it.get_local_linear_id(),kv=local_head/(heads/c.heads);
        const size_t col=lane*4,begin=split*block,end=sycl::min(begin+block,size_t(position[it.get_group(0)]+1));float a[4]={};
        for(size_t cell=begin;cell<end;++cell){
            const float weight=scores[head*c.context+cell];const size_t i=(kv*c.context+cell)*256+col;
            uint32_t codes;std::memcpy(&codes,c.value+i,sizeof(codes));
            const float scale=float(c.value_scale[i/32]);
#pragma unroll
            for(int j=0;j<4;++j)a[j]=sycl::fma(weight,coefficient<FP8>(int8_t(codes>>(8*j)),scale),a[j]);
        }
        const size_t target=(head*splits+split)*256+col;
#pragma unroll
        for(int j=0;j<4;++j)partial[target+j]=a[j];
    });
    if(stages){stages[0]=qk;stages[1]=softmax;stages[2]=pv;}
    return q.parallel_for(sycl::range<2>(size_t(tokens),size_t(heads)*256),[=](sycl::id<2> id){
        const size_t head=id[0]*heads+id[1]/256,col=id[1]%256,used=(size_t(position[id[0]]+1)+block-1)/block;float sum=0;
        for(size_t split=0;split<used;++split)sum+=partial[(head*splits+split)*256+col];
        const size_t index=head*256+col;
        output[index]=sum/(1.f+sycl::exp(-gate[index]));
    });
}
// Format selection occurs once per enqueue; the original INT8 arithmetic
// remains a separate compile-time instantiation, including its PTL fused policy.
sycl::event store_q8_kv(sycl::queue& q,const float* key,const float* value,Q8Kv c,
    const int32_t* position,int tokens,int32_t* error){
    return c.fp8_e4m3?store_kv_impl<true>(q,key,value,c,position,tokens,error)
        :store_kv_impl<false>(q,key,value,c,position,tokens,error);
}
sycl::event dequantize_q8_kv(sycl::queue& q,Q8Kv c,float* key,float* value,int visible){
    if(c.fp8_e4m3&&native_fp8_conversion(q.get_device()))return dequantize_fp8_native(q,c,key,value,visible);
    return c.fp8_e4m3?dequantize_kv_impl<true>(q,c,key,value,visible)
        :dequantize_kv_impl<false>(q,c,key,value,visible);
}
sycl::event fused_decode_attention_q8(sycl::queue& q,const float* query,const float* gate,Q8Kv c,
    float* stats,float* partial,float* output,int heads,const int32_t* position,int extent,sycl::event* partial_stage,int tokens){
    return c.fp8_e4m3?fused_attention_impl<true>(q,query,gate,c,stats,partial,output,heads,position,extent,partial_stage,tokens)
        :fused_attention_impl<false>(q,query,gate,c,stats,partial,output,heads,position,extent,partial_stage,tokens);
}
sycl::event decode_attention_q8(sycl::queue& q,const float* query,const float* gate,Q8Kv c,
    float* scores,float* partial,float* output,int heads,const int32_t* position,int extent,sycl::event* stages,int tokens,int actual_visible){
    return c.fp8_e4m3?decode_attention_impl<true>(q,query,gate,c,scores,partial,output,heads,position,extent,stages,tokens,actual_visible)
        :decode_attention_impl<false>(q,query,gate,c,scores,partial,output,heads,position,extent,stages,tokens,actual_visible);
}
}
