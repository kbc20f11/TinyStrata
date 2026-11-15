// CPU-only LFM geometry/header/tensor-directory gate. No SYCL queue or payload
// allocation is used, including the optional real pinned GGUF validation.
#include "strata/intel/lfm_model.hpp"
#include "strata/common/utf8_args.hpp"
#include <cstdio>
int main(int argc,char** argv){try{
    strata::paths::Utf8Arguments utf8(argc,argv);argc=utf8.argc();argv=utf8.argv();
    strata::intel::LfmGeometry g;int attention=0;
    for(int l=0;l<g.layers;++l)attention+=g.attention(l);
    if(attention!=10||g.layers-attention!=30||g.dim!=64||g.picks!=4||g.experts!=64||
       g.cache_bytes(32768)!=10ull*32768*8*64*2*34/32)
        throw std::runtime_error("LFM hybrid geometry or INT8 budget differs");
    if(argc==3&&std::string(argv[1])=="--model"){
        const auto model=strata::GgufModel::open(argv[2]);
        g=strata::intel::LfmGeometry::validate(model);
        const auto* sample=model.meta().get("tokenizer.ggml.tokens");
        if(!sample||sample->count!=128000||sample->items.size()>64||
           model.meta().token_id("<|im_end|>")!=124900||model.meta().token_id("<|endoftext|>")!=124895||
           std::find(g.terminal_tokens.begin(),g.terminal_tokens.end(),124895)==g.terminal_tokens.end()||
           std::find(g.terminal_tokens.begin(),g.terminal_tokens.end(),124900)==g.terminal_tokens.end())
            throw std::runtime_error("LFM full-vocabulary lookup lost EOS/end-of-text beyond metadata sample");
        std::printf("Pinned LFM2.5 metadata + 428 tensor shapes/types/bounds passed; no tensor payload loaded\n");
    }else if(argc!=1)throw std::runtime_error("Usage: intel_lfm_model_test [--model PINNED.gguf]");
    std::puts("PASS LFM geometry CPU checks; no Intel device opened");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"LFM model fixture: %s\n",e.what());return 1;}}
