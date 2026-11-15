#pragma once
#include "strata/artifact/gguf_reader.hpp"
#include <cmath>
#include <set>
#include <stdexcept>

namespace strata::intel {
// The pinned LFM2.5 target has its own geometry and tensor contract. It has
// neither Gemma's V normalization/softcap nor Qwen's GDN/shared expert/MTP.
struct LfmGeometry {
    int layers=40,width=2048,vocabulary=128000,heads=32,kv_heads=8,dim=64;
    int dense_layers=2,dense_ff=11776,experts=64,picks=4,expert_ff=1536,model_context=131072;
    float epsilon=1e-5f,rope_base=1e7f;
    bool normalize=true;float expert_scale=0;
    bool attention(int layer) const { return layer%4==2; }
    uint64_t cache_bytes(int context) const { return 10ull*context*kv_heads*dim*2*34/32; }
    static LfmGeometry validate(const strata::GgufModel& model) {
        const auto fail=[](const std::string& why){throw std::runtime_error("LFM2.5 GGUF: "+why);};
        const auto& meta=model.meta();
        const auto text=[&](const char* key,const char* value){const auto* x=meta.get(key);return x&&x->type==strata::MetaType::STRING&&x->s==value;};
        if(!text("general.architecture","lfm2moe")||!text("general.name","LFM2.5-24B-A2B")||
           !text("tokenizer.ggml.model","gpt2")||!text("tokenizer.ggml.pre","lfm2"))fail("expected the pinned LFM2.5 architecture/tokenizer");
        if(model.size()!=1||model.shard(0).file_size()!=14526258464ULL)fail("expected complete pinned single-file Q4_K_M target");
        const auto number=[&](const char* key,double expected){const auto* v=meta.get(key);if(!v||!v->is_num()||!std::isfinite(v->num())||v->num()!=expected)fail(std::string("invalid ")+key);};
        LfmGeometry g;
        number("lfm2moe.block_count",40);number("lfm2moe.embedding_length",2048);number("lfm2moe.vocab_size",128000);
        number("lfm2moe.context_length",131072);number("lfm2moe.feed_forward_length",11776);
        number("lfm2moe.attention.head_count",32);number("lfm2moe.rope.freq_base",1e7);
        number("lfm2moe.attention.layer_norm_rms_epsilon",double(float(1e-5)));
        number("lfm2moe.expert_count",64);number("lfm2moe.expert_used_count",4);number("lfm2moe.expert_feed_forward_length",1536);
        number("lfm2moe.leading_dense_block_count",2);number("lfm2moe.expert_gating_func",2);number("lfm2moe.shortconv.l_cache",3);
        number("tokenizer.ggml.bos_token_id",124894);number("tokenizer.ggml.eos_token_id",124900);
        const auto* kv=meta.get("lfm2moe.attention.head_count_kv");
        if(!kv||kv->type!=strata::MetaType::ARRAY||kv->items.size()!=40)fail("missing per-layer key-head pattern");
        for(int l=0;l<40;++l)if(!kv->items[l].is_num()||kv->items[l].num()!=(g.attention(l)?8:0))fail("unexpected attention/shortconv layer pattern");
        if(const auto* v=meta.get("lfm2moe.expert_weights_norm"))g.normalize=v->u!=0;
        if(const auto* v=meta.get("lfm2moe.expert_weights_scale")){if(!v->is_num()||!std::isfinite(v->num()))fail("invalid expert scale");g.expert_scale=float(v->num());}
        std::set<std::string> expected;
        const auto tensor=[&](const std::string& name,std::vector<uint64_t> shape,int type){size_t shard=0;const auto* t=model.find(name,&shard);
            if(!t||t->shape!=shape||int(t->type)!=type||!model.in_bounds(*t,shard))fail("missing/invalid tensor "+name);expected.insert(name);};
        tensor("token_embd.weight",{2048,128000},14);tensor("token_embd_norm.weight",{2048},0);
        const std::set<int> high_v={2,14,26,34,38};
        const std::set<int> high_down={2,3,4,7,10,13,16,19,22,25,28,31,34,35,36,37,38,39};
        for(int l=0;l<40;++l){const std::string p="blk."+std::to_string(l)+".";
            tensor(p+"attn_norm.weight",{2048},0);tensor(p+"ffn_norm.weight",{2048},0);
            if(g.attention(l)){
                tensor(p+"attn_q.weight",{2048,2048},12);tensor(p+"attn_k.weight",{2048,512},12);
                tensor(p+"attn_v.weight",{2048,512},high_v.contains(l)?14:12);tensor(p+"attn_output.weight",{2048,2048},12);
                tensor(p+"attn_q_norm.weight",{64},0);tensor(p+"attn_k_norm.weight",{64},0);
            }else{tensor(p+"shortconv.in_proj.weight",{2048,6144},12);tensor(p+"shortconv.conv.weight",{3,2048},0);tensor(p+"shortconv.out_proj.weight",{2048,2048},12);}
            if(l<2){tensor(p+"ffn_gate.weight",{2048,11776},12);tensor(p+"ffn_up.weight",{2048,11776},12);tensor(p+"ffn_down.weight",{11776,2048},14);}
            else{tensor(p+"ffn_gate_inp.weight",{2048,64},0);tensor(p+"exp_probs_b.bias",{64},0);
                tensor(p+"ffn_gate_exps.weight",{2048,1536,64},12);tensor(p+"ffn_up_exps.weight",{2048,1536,64},12);
                tensor(p+"ffn_down_exps.weight",{1536,2048,64},high_down.contains(l)?14:12);}
        }
        size_t count=0;for(size_t s=0;s<model.size();++s)for(const auto& t:model.shard(s).tensors()){
            if(!expected.contains(t.name))fail("unhandled tensor "+t.name);++count;}
        if(count!=428||count!=expected.size())fail("unexpected tensor count");return g;
    }
};
} // namespace strata::intel
