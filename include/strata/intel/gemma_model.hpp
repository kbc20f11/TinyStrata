#pragma once
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace strata::intel {
// Geometry of the pinned Google Gemma4 26B QAT Q4_0 target. It deliberately
// does not inherit the Qwen/GDN geometry, cache layout, or routing defaults.
struct GemmaGeometry {
    int layers = 0, width = 0, vocabulary = 0, heads = 0;
    int experts = 0, picks = 0, dense_ff = 0, expert_ff = 0, window = 0;
    int full_dim = 0, swa_dim = 0, full_rot = 0, swa_rot = 0, model_context = 0;
    float epsilon = 0, full_base = 0, swa_base = 0, softcap = 0;
    std::vector<int> kv_heads;
    std::vector<bool> sliding;
    int turn_end=-1;
    std::vector<int> terminal_tokens;
    int dim(int layer) const { return sliding.at(size_t(layer)) ? swa_dim : full_dim; }
    int rotation(int layer) const { return sliding.at(size_t(layer)) ? swa_rot : full_rot; }
    float base(int layer) const { return sliding.at(size_t(layer)) ? swa_base : full_base; }
    size_t capacity(int layer, int context, int chunk) const {
        return sliding.at(size_t(layer)) ? size_t(std::min(context, window + chunk)) : size_t(context);
    }
    uint64_t cache_bytes(int context, int chunk) const {
        uint64_t result = 0;
        for (int l = 0; l < layers; ++l)
            result += capacity(l, context, chunk) * size_t(kv_heads[l]) * dim(l) * 2 * 34 / 32;
        return result;
    }
    static GemmaGeometry validate(const strata::GgufModel& model) {
        const auto fail = [](const std::string& text) { throw std::runtime_error("Gemma4 GGUF: " + text); };
        const auto& meta = model.meta();
        const auto* arch = meta.get("general.architecture");
        if (!arch || arch->type != strata::MetaType::STRING || arch->s != "gemma4") fail("expected gemma4 target architecture");
        if (model.size() != 1 || model.shard(0).file_size() != 14439363584ULL)
            fail("expected the pinned single-file QAT Q4_0 target (not a partial file or assistant)");
        auto number = [&](const char* key) {
            const auto* v = meta.get(std::string("gemma4.") + key);
            if (!v || !v->is_num() || !std::isfinite(v->num())) fail(std::string("missing/invalid ") + key);
            return v->num();
        };
        auto integer = [&](const char* key) {
            const double v = number(key);
            if (v < 0 || v > 1000000 || v != std::floor(v)) fail(std::string("invalid integer ") + key);
            return int(v);
        };
        GemmaGeometry g;
        g.layers=integer("block_count"); g.width=integer("embedding_length");
        g.heads=integer("attention.head_count"); g.experts=integer("expert_count"); g.picks=integer("expert_used_count");
        g.dense_ff=integer("feed_forward_length"); g.expert_ff=integer("expert_feed_forward_length");
        g.full_dim=integer("attention.key_length"); g.swa_dim=integer("attention.key_length_swa");
        g.full_rot=integer("rope.dimension_count"); g.swa_rot=integer("rope.dimension_count_swa");
        g.window=integer("attention.sliding_window"); g.model_context=integer("context_length");
        g.epsilon=float(number("attention.layer_norm_rms_epsilon"));
        g.full_base=float(number("rope.freq_base")); g.swa_base=float(number("rope.freq_base_swa"));
        g.softcap=float(number("final_logit_softcapping"));
        if (g.layers!=30 || g.width!=2816 || g.heads!=16 || g.experts!=128 || g.picks!=8 ||
            g.dense_ff!=2112 || g.expert_ff!=704 || g.full_dim!=512 || g.swa_dim!=256 ||
            g.full_rot!=512 || g.swa_rot!=256 || g.window!=1024 || g.model_context!=262144 ||
            g.full_base!=1e6f || g.swa_base!=1e4f || g.softcap!=30.f || std::abs(g.epsilon-1e-6f)>1e-12f ||
            integer("attention.shared_kv_layers")!=0 || integer("embedding_length_per_layer_input")!=0)
            fail("metadata differs from the pinned26B target contract");
        const auto* kh = meta.get("gemma4.attention.head_count_kv");
        const auto* sw = meta.get("gemma4.attention.sliding_window_pattern");
        if (!kh || !sw || kh->type!=strata::MetaType::ARRAY || sw->type!=strata::MetaType::ARRAY ||
            kh->items.size()!=size_t(g.layers) || sw->items.size()!=size_t(g.layers)) fail("invalid per-layer attention arrays");
        for(int l=0;l<g.layers;++l) {
            const bool sliding=(l%6)!=5;
            if (!kh->items[l].is_num() || !sw->items[l].is_num() ||
                kh->items[l].num()!=(sliding?8:2) || sw->items[l].num()!=(sliding?1:0)) fail("unexpected layer attention geometry");
            g.kv_heads.push_back(sliding?8:2); g.sliding.push_back(sliding);
        }
        std::set<std::string> expected;
        auto tensor = [&](const std::string& name, std::vector<uint64_t> shape, uint32_t type) {
            size_t shard=0; const auto* t=model.find(name,&shard);
            if (!t || t->shape!=shape || t->type!=type || !model.in_bounds(*t,shard)) fail("missing/invalid tensor " + name);
            expected.insert(name);
        };
        g.vocabulary=262144;
        tensor("token_embd.weight",{g.width,g.vocabulary},14);
        tensor("output_norm.weight",{g.width},0);
        tensor("rope_freqs.weight",{g.full_rot/2},0);
        for(int l=0;l<g.layers;++l) {
            const auto p="blk."+std::to_string(l)+"."; const int d=g.dim(l), kv=g.kv_heads[l];
            for(const char* n:{"attn_norm.weight","post_attention_norm.weight","ffn_norm.weight","post_ffw_norm.weight",
                "ffn_gate_inp.scale","post_ffw_norm_1.weight","post_ffw_norm_2.weight","pre_ffw_norm_2.weight"})
                tensor(p+n,{g.width},0);
            tensor(p+"attn_q_norm.weight",{d},0);tensor(p+"attn_k_norm.weight",{d},0);
            tensor(p+"attn_q.weight",{g.width,g.heads*d},2);tensor(p+"attn_k.weight",{g.width,kv*d},2);
            if(g.sliding[l])tensor(p+"attn_v.weight",{g.width,kv*d},2);
            tensor(p+"attn_output.weight",{g.heads*d,g.width},2);
            tensor(p+"ffn_gate.weight",{g.width,g.dense_ff},2);tensor(p+"ffn_up.weight",{g.width,g.dense_ff},2);
            tensor(p+"ffn_down.weight",{g.dense_ff,g.width},2);
            tensor(p+"ffn_gate_inp.weight",{g.width,g.experts},0);
            tensor(p+"ffn_gate_up_exps.weight",{g.width,2*g.expert_ff,g.experts},2);
            tensor(p+"ffn_down_exps.weight",{g.expert_ff,g.width,g.experts},2);
            tensor(p+"ffn_down_exps.scale",{g.experts},0);tensor(p+"layer_output_scale.weight",{1},0);
        }
        size_t count=0;
        for(size_t s=0;s<model.size();++s)for(const auto& t:model.shard(s).tensors()) {
            if(!expected.count(t.name))fail("unhandled tensor "+t.name);
            ++count;
        }
        if(count!=658 || count!=expected.size())fail("unexpected tensor count");
        // GgufFile metadata arrays intentionally retain only64 sample items.
        // Turn106 and every other vocabulary lookup must use the mapped full
        // token directory, never metadata().items as a vocabulary table.
        const auto* bos=meta.get("tokenizer.ggml.bos_token_id");
        const auto* eos=meta.get("tokenizer.ggml.eos_token_id");
        if(!bos||!eos||!bos->is_num()||!eos->is_num()||bos->num()!=2||eos->num()!=1||
            meta.token_id("<bos>")!=int64_t(bos->num())||meta.token_id("<eos>")!=int64_t(eos->num()))fail("BOS/EOS metadata does not match full vocabulary");
        const int64_t turn=meta.token_id("<turn|>");
        if(turn<0||turn>=g.vocabulary||meta.token_id("<|turn>")<0)fail("missing full-vocabulary turn boundaries");
        g.turn_end=int(turn);g.terminal_tokens.push_back(g.turn_end);
        for(const char* key:{"tokenizer.ggml.eos_token_id","tokenizer.ggml.eot_token_id","tokenizer.ggml.eom_token_id"}){
            if(const auto* token=meta.get(key)){
                if(!token->is_num()||token->num()<0||token->num()>=g.vocabulary||token->num()!=std::floor(token->num()))fail(std::string("invalid ")+key);
                g.terminal_tokens.push_back(int(token->num()));
            }
        }
        std::sort(g.terminal_tokens.begin(),g.terminal_tokens.end());
        g.terminal_tokens.erase(std::unique(g.terminal_tokens.begin(),g.terminal_tokens.end()),g.terminal_tokens.end());
        return g;
    }
};
}
