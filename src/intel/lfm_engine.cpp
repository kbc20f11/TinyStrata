// Native Windows LFM2.5 causal text engine. LFM formulas follow the existing
// src/lfm2 implementation (PolyStrata/Strata attribution in LICENSE/NOTICE).
#include "strata/intel/lfm_model.hpp"
#include "strata/intel/lfm_ops.hpp"
#include "strata/intel/gemma_ops.hpp" // Generic head RMS/RoPE and causal Q8 attention; scale supplied explicitly.
#include "strata/intel/weight_store.hpp"
#include "strata/intel/qwen_ops.hpp"
#include "strata/intel/execution_request.hpp"
#include "strata/intel/sampling_policy.hpp"
#ifdef TINYSTRATA_INTEL_MKL
#include "strata/intel/prefill_gemm.hpp"
#endif
#include "strata/common/command_inbox.hpp"
#include "strata/common/utf8_args.hpp"
#include "strata/common/utf8_path.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>

namespace strata::intel {
namespace {
class LfmEngine {
    sycl::queue q_;
    DeviceInfo device_;
    LfmGeometry geometry_;
    std::unique_ptr<WeightStore> weights_;
    std::unique_ptr<GemmaAttention> attention_;
#ifdef TINYSTRATA_INTEL_MKL
    std::unique_ptr<PrefillGemm> gemm_;
#endif
    struct State { GemmaKv kv;float* convolution=nullptr;float* backup=nullptr; };
    std::vector<State> states_;
    std::vector<void*> allocations_;
    int context_,chunk_,position_=0;
    bool logits_valid_=false;
    float *residual_,*norm_,*bcx_,*mixed_,*qraw_,*kraw_,*v_,*query_,*key_,*attended_,*attention_out_;
    float *gate_,*up_,*hidden_,*expert_down_,*ffn_out_,*router_,*selected_weights_,*last_hidden_,*backup_hidden_,*logits_;
    int32_t *ids_,*input_ids_,*quant_error_,*counts_,*slots_,*owners_,*offsets_,*tiles_;
    template<class T>T* allocate(size_t n,bool shared=false){
        if(!n||n>std::numeric_limits<size_t>::max()/sizeof(T))throw std::runtime_error("LFM allocation overflow");
        auto* p=shared?sycl::malloc_shared<T>(n,q_):sycl::malloc_device<T>(n,q_);
        if(!p)throw std::runtime_error("LFM USM allocation failed");
        try{allocations_.push_back(p);}catch(...){sycl::free(p,q_);throw;}
        q_.memset(p,0,n*sizeof(T));return p;
    }
    const ResidentTensor& tensor(const std::string& name)const{
        const auto at=weights_->tensors().find(name);if(at==weights_->tensors().end())throw std::runtime_error("Missing LFM tensor "+name);return at->second;
    }
    const float* f32(const std::string& name)const{
        const auto& t=tensor(name);if(t.info.type!=0)throw std::runtime_error("Expected F32 LFM tensor "+name);return reinterpret_cast<const float*>(t.data);
    }
    void matrix(const std::string& name,const float* input,float* output,int tokens){
        const auto& t=tensor(name);const int width=int(t.info.shape[0]),rows=int(t.info.shape[1]);
#ifdef TINYSTRATA_INTEL_MKL
        if(gemm_&&tokens>=16&&t.info.type!=0){gemm_->dense(int(t.info.type),t.data,input,output,rows,width,tokens);return;}
#endif
        quantized_matmul(q_,int(t.info.type),t.data,input,output,rows,width,tokens);
    }
    void expert_matrix(const ResidentTensor& t,ExpertPlan plan,const float* input,float* output,int rows,int width,bool separate,int tokens){
#ifdef TINYSTRATA_INTEL_MKL
        if(gemm_&&tokens>=16){gemm_->selected(int(t.info.type),t.data,t.bytes/geometry_.experts,plan,input,output,rows,width,separate);return;}
#endif
        if(tokens>=16)selected_matmul(q_,int(t.info.type),t.data,plan,input,output,rows,width,separate);
        else selected_matvec(q_,int(t.info.type),t.data,ids_,geometry_.picks,input,output,rows,width,separate,tokens);
    }
    void add(const float* source,float* target,size_t n){q_.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){target[i]+=source[i];});}
    void swiglu(size_t n){const float* gate=gate_,*up=up_;float* hidden=hidden_;
        q_.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){hidden[i]=gate[i]/(1.f+sycl::exp(-gate[i]))*up[i];});}
    void checkpoint(bool restore){
        for(auto& state:states_)if(state.convolution)
            q_.memcpy(restore?state.convolution:state.backup,restore?state.backup:state.convolution,size_t(2*geometry_.width)*sizeof(float));
        q_.memcpy(restore?last_hidden_:backup_hidden_,restore?backup_hidden_:last_hidden_,size_t(geometry_.width)*sizeof(float));
    }
    void output_head(){
        rms_norm(q_,last_hidden_,f32("token_embd_norm.weight"),norm_,1,geometry_.width,geometry_.epsilon);
        const auto& table=tensor("token_embd.weight");
        quantized_matvec(q_,int(table.info.type),table.data,norm_,logits_,geometry_.vocabulary,geometry_.width);
    }
    void check_logits(){for(int i=0;i<geometry_.vocabulary;++i)if(!std::isfinite(logits_[i]))throw std::runtime_error("Non-finite LFM logits");}
    void release()noexcept{
        try{q_.wait_and_throw();}catch(...){}attention_.reset();
#ifdef TINYSTRATA_INTEL_MKL
        gemm_.reset();
#endif
        for(auto* p:allocations_)try{sycl::free(p,q_);}catch(...){}allocations_.clear();weights_.reset();
    }
public:
    LfmEngine(const std::string& path,unsigned device,int context,int chunk,WeightMemory memory,bool mkl)
        :q_(select_device(device),[](sycl::exception_list errors){for(const auto& e:errors)std::rethrow_exception(e);},sycl::property::queue::in_order{}),
         device_(inspect_device(q_.get_device())),context_(context),chunk_(chunk){
        if(context<1||context>32768||chunk<1||chunk>512)throw std::runtime_error("Initial LFM context1..32768/chunk1..512 required");
#ifdef TINYSTRATA_INTEL_AOT_TARGET
        if(std::string(TINYSTRATA_INTEL_AOT_TARGET)=="intel_gpu_ptl_h"&&!device_.ptl_h_aot)
            throw std::runtime_error("PTL-H AOT requires Panther Lake; use the generic runtime on this GPU");
#endif
        try{
            ExecutionRequest loading;
            weights_=WeightStore::load_validated(q_,path,memory,[&](const strata::GgufModel& model){geometry_=LfmGeometry::validate(model);});
            const auto& g=geometry_;const size_t n=size_t(chunk_),width=g.width,cache_elements=size_t(context)*g.kv_heads*g.dim;
            const uint64_t scratch=n*(20*width+3*size_t(g.dense_ff)+g.picks*(3ull*g.expert_ff+width))*sizeof(float)
                +4ull*30*width*sizeof(float)+2*cache_elements*sizeof(float)+uint64_t(context)*std::max(chunk,8*g.heads)*sizeof(float)+256*1024*1024ULL;
            const uint64_t shared_bytes=uint64_t(g.vocabulary)*sizeof(float)+n*sizeof(int32_t)+sizeof(int32_t)+uint64_t(g.experts)*sizeof(int32_t);
            if(!execution_allocation_fits(memory_budget(device_),g.cache_bytes(context)+scratch+shared_bytes,shared_bytes,16*1024*1024ULL))
                throw std::runtime_error("LFM state exceeds separate RAM/device budgets; unload another model first");
            states_.resize(g.layers);
            for(int l=0;l<g.layers;++l){auto& s=states_[l];if(g.attention(l)){
                s.kv={allocate<int8_t>(cache_elements),allocate<int8_t>(cache_elements),allocate<sycl::half>(cache_elements/32),
                      allocate<sycl::half>(cache_elements/32),context,g.kv_heads,g.dim,0};
            }else{s.convolution=allocate<float>(2*width);s.backup=allocate<float>(2*width);}}
            residual_=allocate<float>(n*width);norm_=allocate<float>(n*width);bcx_=allocate<float>(n*3*width);mixed_=allocate<float>(n*width);
            qraw_=allocate<float>(n*g.heads*g.dim);query_=allocate<float>(n*g.heads*g.dim);attended_=allocate<float>(n*g.heads*g.dim);
            kraw_=allocate<float>(n*g.kv_heads*g.dim);key_=allocate<float>(n*g.kv_heads*g.dim);v_=allocate<float>(n*g.kv_heads*g.dim);
            attention_out_=allocate<float>(n*width);
            const size_t ff_elements=n*std::max(size_t(g.dense_ff),size_t(g.picks)*g.expert_ff);
            gate_=allocate<float>(ff_elements);up_=allocate<float>(ff_elements);hidden_=allocate<float>(ff_elements);
            expert_down_=allocate<float>(n*g.picks*width);ffn_out_=allocate<float>(n*width);router_=allocate<float>(n*g.experts);
            selected_weights_=allocate<float>(n*g.picks);ids_=allocate<int32_t>(n*g.picks);
            input_ids_=allocate<int32_t>(n,true);quant_error_=allocate<int32_t>(1,true);logits_=allocate<float>(g.vocabulary,true);
            last_hidden_=allocate<float>(width);backup_hidden_=allocate<float>(width);
            // The bounded oneMKL selected-expert scheduler reads these small
            // counts on the host; the expert slots/activations stay on device.
            counts_=allocate<int32_t>(g.experts,true);slots_=allocate<int32_t>(n*g.experts);
            const size_t max_tiles=(n*g.picks+3)/4+g.experts;owners_=allocate<int32_t>(max_tiles);offsets_=allocate<int32_t>(max_tiles);tiles_=allocate<int32_t>(1);
            attention_=std::make_unique<GemmaAttention>(q_,context,chunk,g.heads,g.dim,cache_elements,mkl);
#ifdef TINYSTRATA_INTEL_MKL
            if(mkl)gemm_=std::make_unique<PrefillGemm>(q_,1,chunk);
#else
            if(mkl)throw std::runtime_error("LFM oneMKL requires an MKL build");
#endif
            q_.wait_and_throw();
            std::fprintf(stderr,"Intel LFM2.5 layers40 conv30 attention10 width2048 head64 picks4of64 KV=int8 MTP0 mandatory_thinking=1 single_weight_store=1 cache_bytes=%llu\n",(unsigned long long)g.cache_bytes(context));
        }catch(...){release();throw;}
    }
    ~LfmEngine(){release();}
    const DeviceInfo& device_info()const{return device_;}
    int context()const{return context_;}int chunk()const{return chunk_;}int consumed()const{return position_;}int vocabulary()const{return geometry_.vocabulary;}
    bool terminal(int token)const{return std::find(geometry_.terminal_tokens.begin(),geometry_.terminal_tokens.end(),token)!=geometry_.terminal_tokens.end();}
    void reset(){for(auto& s:states_)if(s.convolution)q_.memset(s.convolution,0,size_t(2*geometry_.width)*sizeof(float));q_.wait_and_throw();position_=0;logits_valid_=false;}
    const float* logits(){if(!position_)throw std::runtime_error("No committed LFM prompt");if(!logits_valid_){output_head();q_.wait_and_throw();check_logits();logits_valid_=true;}return logits_;}
    const float* step_many(const int* tokens,int count,bool final=true,const std::atomic<bool>* cancelled=nullptr){
        if(count<1||count>chunk_||position_+count>context_)throw std::runtime_error("LFM step exceeds context/chunk");
        for(int i=0;i<count;++i)if(tokens[i]<0||tokens[i]>=vocabulary())throw std::runtime_error("LFM token outside vocabulary");
        if(cancelled&&cancelled->load())return nullptr;
        checkpoint(false);logits_valid_=false;std::copy_n(tokens,count,input_ids_);*quant_error_=0;
        const auto& g=geometry_;const int width=g.width,picks=g.picks;const size_t all=size_t(count)*width;
        const auto& table=tensor("token_embd.weight");embedding_lookup(q_,int(table.info.type),table.data,input_ids_,residual_,width,count);
        bool complete=true;
        for(int l=0;l<g.layers;++l){
            if(cancelled&&cancelled->load()){complete=false;break;}
            const std::string p="blk."+std::to_string(l)+".";
            rms_norm(q_,residual_,f32(p+"attn_norm.weight"),norm_,count,width,g.epsilon);
            if(g.attention(l)){
                matrix(p+"attn_q.weight",norm_,qraw_,count);matrix(p+"attn_k.weight",norm_,kraw_,count);matrix(p+"attn_v.weight",norm_,v_,count);
                gemma_head_norm_rope(q_,qraw_,f32(p+"attn_q_norm.weight"),query_,g.heads,g.dim,g.dim,position_,count,g.epsilon,g.rope_base);
                gemma_head_norm_rope(q_,kraw_,f32(p+"attn_k_norm.weight"),key_,g.kv_heads,g.dim,g.dim,position_,count,g.epsilon,g.rope_base);
                // LFM values stay raw: no Gemma V RMS normalization.
                gemma_store_q8(q_,key_,v_,states_[l].kv,position_,count,quant_error_);
                attention_->run(query_,states_[l].kv,attended_,g.heads,position_,count,.125f);
                matrix(p+"attn_output.weight",attended_,attention_out_,count);
            }else{
                matrix(p+"shortconv.in_proj.weight",norm_,bcx_,count);
                lfm_shortconv(q_,bcx_,states_[l].convolution,f32(p+"shortconv.conv.weight"),mixed_,width,count);
                matrix(p+"shortconv.out_proj.weight",mixed_,attention_out_,count);
            }
            add(attention_out_,residual_,all);
            rms_norm(q_,residual_,f32(p+"ffn_norm.weight"),norm_,count,width,g.epsilon);
            if(l<g.dense_layers){
                matrix(p+"ffn_gate.weight",norm_,gate_,count);matrix(p+"ffn_up.weight",norm_,up_,count);
                swiglu(size_t(count)*g.dense_ff);matrix(p+"ffn_down.weight",hidden_,ffn_out_,count);
            }else{
                matrix(p+"ffn_gate_inp.weight",norm_,router_,count);
                lfm_route(q_,router_,f32(p+"exp_probs_b.bias"),ids_,selected_weights_,count,g.experts,picks,g.normalize,g.expert_scale);
                ExpertPlan plan{counts_,slots_,owners_,offsets_,tiles_,g.experts,count,picks};if(count>=16)plan_experts(q_,ids_,plan);
                expert_matrix(tensor(p+"ffn_gate_exps.weight"),plan,norm_,gate_,g.expert_ff,width,false,count);
                expert_matrix(tensor(p+"ffn_up_exps.weight"),plan,norm_,up_,g.expert_ff,width,false,count);
                swiglu(size_t(count)*picks*g.expert_ff);
                expert_matrix(tensor(p+"ffn_down_exps.weight"),plan,hidden_,expert_down_,width,g.expert_ff,true,count);
                const float *down=expert_down_,*weights=selected_weights_;float* out=ffn_out_;
                q_.parallel_for(sycl::range<1>(all),[=](sycl::id<1> i){const size_t token=i[0]/width,col=i[0]%width;float sum=0;
                    for(int k=0;k<picks;++k)sum+=weights[token*picks+k]*down[(token*picks+k)*width+col];out[i]=sum;});
            }
            add(ffn_out_,residual_,all);
        }
        if(complete){q_.memcpy(last_hidden_,residual_+size_t(count-1)*width,size_t(width)*sizeof(float));if(final)output_head();}
        q_.wait_and_throw();if(*quant_error_)throw std::runtime_error("Nonfinite/overflow LFM INT8 KV input");
        if(!complete||(cancelled&&cancelled->load())){checkpoint(true);q_.wait_and_throw();return nullptr;}
        if(final)check_logits();position_+=count;logits_valid_=final;return final?logits_:residual_;
    }
};

std::vector<int> parse_tokens(std::string text){std::replace(text.begin(),text.end(),',',' ');std::istringstream in(text);std::vector<int> ids;int64_t id;
    while(in>>id){if(id<0||id>std::numeric_limits<int>::max())throw std::runtime_error("Invalid token ID");ids.push_back(int(id));}
    if(!in.eof()||ids.empty())throw std::runtime_error("Expected nonempty token list");return ids;}
int integer(const std::string& text,int maximum){if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("Expected nonnegative integer");
    const auto n=std::stoull(text);if(n>uint64_t(maximum))throw std::runtime_error("Integer exceeds supported range");return int(n);}
void option(sampling_policy::Sampling& s,const std::string& text){const auto at=text.find('=');const auto k=text.substr(0,at),v=text.substr(at+1);
    if(k=="seed"){if(v.empty()||v.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("Invalid seed");s.seed=std::stoull(v);return;}
    size_t used=0;const float n=std::stof(v,&used);if(used!=v.size()||!std::isfinite(n))throw std::runtime_error("Invalid sampling field");
    if(k=="temperature"&&n>=0)s.temperature=n;else if(k=="top_p"&&n>0&&n<=1)s.top_p=n;else if(k=="min_p"&&n>=0&&n<=1)s.min_p=n;
    else if(k=="top_k")s.top_k=integer(v,64);else if(k=="penalty_repeat"&&n>0)s.repeat=n;
    else if(k=="penalty_freq"&&n>=0)s.frequency=n;else if(k=="penalty_present"&&n>=0)s.presence=n;
    else if(k=="penalty_last_n"&&n>=-1&&n<=32768&&n==std::floor(n))s.last_n=int(n);
    else if((k=="ckpt"||k=="turn_at")&&n>=0&&n==std::floor(n)){} // exact current-prefix reuse only in this milestone
    else if((k=="mtp_auto"||k=="lookup_draft")&&n==0){}
    else throw std::runtime_error("Unsupported LFM request option: "+text);}
int serve(LfmEngine& e){const auto& d=e.device_info();
    std::printf("INFO backend=sycl experimental=1 model=lfm2.5-24b-a2b family=lfm2moe kv=int8 mtp=0 vision=0 mandatory_thinking=1 memory_model=%s single_weight_store=1 device_vendor=%u device_id=%u device_integrated=%d device_architecture=%s adapter_luid=%s adapter_node_mask=%u kv_used_tokens=0\nREADY %d stop\n",
        d.integrated?"uma":"discrete",d.vendor_id,d.device_id,int(d.integrated),d.architecture.c_str(),d.adapter_luid.empty()?"unknown":d.adapter_luid.c_str(),d.adapter_node_mask,e.context());std::fflush(stdout);
    strata::common::CommandInbox inbox;std::string line;std::vector<int> live;int reported=0;
    const auto report=[&]{if(reported!=e.consumed()){reported=e.consumed();std::printf("INFO kv_used_tokens=%d\n",reported);std::fflush(stdout);}};
    while(inbox.next(line)){if(line=="QUIT")return 0;if(line.empty())continue;
        std::vector<int> prompt;int count=0;sampling_policy::Sampling sampling;sampling.temperature=.1f;sampling.top_k=50;sampling.repeat=1.05f;sampling.last_n=64;
        try{if(line.rfind("GEN ",0)!=0)throw std::runtime_error("LFM supports text GEN only");std::istringstream in(line.substr(4));std::string field,ids;
            if(!(in>>field))throw std::runtime_error("Missing output count");count=integer(field,e.context());
            while(in>>field){if(ids.empty()&&field.find('=')!=std::string::npos)option(sampling,field);else{if(!ids.empty())ids+=' ';ids+=field;}}
            prompt=parse_tokens(ids);if(!count||prompt.size()+size_t(count)>size_t(e.context()))throw std::runtime_error("LFM request exceeds context");
            for(int id:prompt)if(id>=e.vocabulary())throw std::runtime_error("LFM token outside vocabulary");
        }catch(const std::exception& error){std::printf("ERR %s\n",error.what());std::fflush(stdout);continue;}
        ExecutionRequest active;const auto start=std::chrono::steady_clock::now();
        if(live.size()!=size_t(e.consumed()))throw std::runtime_error("LFM committed history/position mismatch");
        if(live.size()>prompt.size()||!std::equal(live.begin(),live.end(),prompt.begin())){e.reset();live.clear();}
        report();const size_t reused=live.size();std::printf("RESUME %zu\n",reused);std::fflush(stdout);
        const float* logits=nullptr;
        for(size_t index=reused;index<prompt.size()&&!inbox.cancelled().load();){const int n=int(std::min(size_t(e.chunk()),prompt.size()-index));
            logits=e.step_many(prompt.data()+index,n,index+n==prompt.size(),&inbox.cancelled());if(!logits)break;
            live.insert(live.end(),prompt.begin()+index,prompt.begin()+index+n);index+=n;report();
            const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            std::printf("PP %zu %zu %.3f\n",index,prompt.size(),elapsed);std::fflush(stdout);}
        const auto decode_start=std::chrono::steady_clock::now();const double prefill=std::chrono::duration<double,std::milli>(decode_start-start).count();
        const size_t fresh=live.size()-reused;std::vector<int> history(prompt);std::mt19937_64 random(sampling.seed?sampling.seed:std::random_device{}());
        int produced=0;const char* finish="length";
        if(inbox.cancelled().load())finish="cancel";
        else for(int n=0;n<count;++n){if(inbox.cancelled().load()){finish="cancel";break;}if(n==0)logits=e.logits();
            const int token=sampling_policy::sample(logits,e.vocabulary(),sampling,history,random);++produced;history.push_back(token);
            std::printf("T %d\n",token);std::fflush(stdout);if(e.terminal(token)){finish="stop";break;}
            if(n+1<count){logits=e.step_many(&token,1,true,&inbox.cancelled());if(!logits){finish="cancel";break;}live.push_back(token);report();}}
        report();const double decode=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-decode_start).count();
        std::printf("DONE %d %zu %.3f %.3f %s 0 0 %zu 0 0 0 0 0.0 %zu 0\n",produced,prompt.size(),prefill,decode,finish,reused,fresh);std::fflush(stdout);
    }return 0;}
} // namespace
} // namespace strata::intel

int main(int argc,char** argv){try{
    strata::paths::Utf8Arguments utf8(argc,argv);argc=utf8.argc();argv=utf8.argv();
    if(strata::intel::print_device_report_if_requested(argc,argv))return 0;
    std::string model,tokens,file,dump;unsigned device=0;int context=8192,chunk=512;bool server=false;
#ifdef TINYSTRATA_INTEL_MKL
    bool mkl=true;
#else
    bool mkl=false;
#endif
    auto memory=strata::intel::WeightMemory::Host;
    for(int i=1;i<argc;++i){const std::string arg=argv[i];
        if(arg=="--capabilities"){
#ifdef TINYSTRATA_INTEL_MKL
            constexpr const char* mkl="true";
            constexpr const char* profiles="{\"b390-lfm25\":1,\"lfm25-causal\":1}";
#else
            constexpr const char* mkl="false";
            constexpr const char* profiles="{\"lfm25-causal\":1}";
#endif
#ifdef TINYSTRATA_INTEL_INT8_XMX
            constexpr const char* xmx="true";
#else
            constexpr const char* xmx="false";
#endif
#ifdef TINYSTRATA_INTEL_AOT_TARGET
            constexpr const char* aot=TINYSTRATA_INTEL_AOT_TARGET;
#else
            constexpr const char* aot="";
#endif
            std::printf("{\"backend\":\"sycl\",\"experimental\":true,\"model_families\":[],"
                "\"experimental_models\":[\"lfm2.5-24b-a2b\"],\"experimental_profiles\":%s,"
                "\"experimental_model_contracts\":{\"lfm2.5-24b-a2b\":{\"version\":1,\"architecture\":\"lfm2moe\",\"kv_types\":[\"int8\"],\"mtp\":false,\"experimental_mtp_max_depth\":0,\"images\":false,\"thinking\":\"mandatory\",\"mode\":\"target_only\"}},"
                "\"kv_types\":[\"int8\"],\"images\":false,\"mtp\":false,\"experimental_mtp_max_depth\":0,\"serve\":true,\"capabilities_schema\":1,\"mandatory_thinking\":true,"
                "\"build\":{\"mkl\":%s,\"int8_xmx\":%s,\"aot_target\":\"%s\"},"
                "\"weight_types\":{\"scalar\":[0,12,14],\"dp4a\":[],\"int8_xmx\":[]}}\n",profiles,mkl,xmx,aot);
            return 0;
        }
        if(arg=="--help"){std::puts("tinystrata-intel-lfm2 --model PINNED.gguf [--serve | --tokens IDS | --tokens-file FILE] [--dump-full-logits FILE] [--max-context N] [--chunk N] [--kv-type int8] [--mtp 0] [--prefill-mkl | --no-prefill-mkl] [--device N] [--weight-memory host|shared|device]\nExperimental LFM2.5 target-only causal text. INT8 KV baseline; mandatory thinking; no MTP/images. Exact live-prefix reuse; edited/regenerated histories replay cleanly.");return 0;}
        if(arg=="--serve"){server=true;continue;}if(arg=="--prefill-mkl"){mkl=true;continue;}if(arg=="--no-prefill-mkl"){mkl=false;continue;}
        if(i+1>=argc)throw std::runtime_error("Missing value for "+arg);const std::string value=argv[++i];
        if(arg=="--model")model=value;else if(arg=="--tokens")tokens=value;else if(arg=="--tokens-file")file=value;
        else if(arg=="--dump"||arg=="--dump-full-logits")dump=value;
        else if(arg=="--device")device=unsigned(strata::intel::integer(value,1024));
        else if(arg=="--max-context")context=strata::intel::integer(value,32768);else if(arg=="--chunk")chunk=strata::intel::integer(value,512);
        else if(arg=="--kv-type"){if(value!="int8"&&value!="auto")throw std::runtime_error("Initial LFM SYCL baseline validates INT8 KV only");}
        else if(arg=="--mtp"){if(value!="0")throw std::runtime_error("LFM MTP is unimplemented; use --mtp 0");}
        else if(arg=="--weight-memory"){if(value=="host")memory=strata::intel::WeightMemory::Host;else if(value=="shared")memory=strata::intel::WeightMemory::Shared;else if(value=="device")memory=strata::intel::WeightMemory::Device;else throw std::runtime_error("Invalid weight memory kind");}
        else throw std::runtime_error("Unsupported LFM option "+arg);
    }
    if(model.empty())throw std::runtime_error("--model is required");if(server&&(!tokens.empty()||!file.empty()||!dump.empty()))throw std::runtime_error("CLI tokens cannot accompany --serve");
    strata::intel::LfmEngine engine(model,device,context,chunk,memory,mkl);if(server)return strata::intel::serve(engine);
    if(!file.empty()){std::ifstream in(strata::paths::native_path(file));if(!in)throw std::runtime_error("Cannot read tokens file");std::ostringstream text;text<<in.rdbuf();tokens=text.str();}
    const auto ids=strata::intel::parse_tokens(tokens);strata::intel::ExecutionRequest active;
    for(size_t index=0;index<ids.size();){const int n=int(std::min(size_t(engine.chunk()),ids.size()-index));engine.step_many(ids.data()+index,n,index+n==ids.size());index+=n;}
    const auto* logits=engine.logits();if(!dump.empty()){std::ofstream out(strata::paths::native_path(dump),std::ios::binary);if(!out.write(reinterpret_cast<const char*>(logits),size_t(engine.vocabulary())*sizeof(float)))throw std::runtime_error("Cannot write logits");}
    const int best=int(std::max_element(logits,logits+engine.vocabulary())-logits);std::printf("LOGITS tokens=%zu vocabulary=%d argmax=%d value=%.9g\n",ids.size(),engine.vocabulary(),best,logits[best]);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"Intel LFM2.5: %s\n",error.what());return 1;}}
