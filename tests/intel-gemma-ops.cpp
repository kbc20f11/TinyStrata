// Explicit model-free Intel GPU gate. CPU/FP64 formulas are independent of
// gemma_ops.cpp. Never run automatically with the CPU-only CTest group.
#include "strata/intel/gemma_ops.hpp"
#include "strata/intel/gemma_model.hpp"
#include "strata/intel/qwen_ops.hpp"
#include "strata/intel/device.hpp"
#include "strata/common/utf8_path.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace si = strata::intel;
namespace {
int checks = 0;
double maximum_error = 0;
void require(bool value,const char* text) { ++checks; if(!value)throw std::runtime_error(text); }
void close(float actual,double expected,double tolerance=1e-5) {
    ++checks;
    const double error=std::abs(double(actual)-expected);
    maximum_error=std::max(maximum_error,error);
    if(!std::isfinite(actual)||error>tolerance*(1+std::abs(expected))) {
        std::fprintf(stderr,"numeric actual=%.10g expected=%.14g error=%.9g tolerance=%.9g\n",actual,expected,error,tolerance);
        throw std::runtime_error("Gemma independent numerical reference mismatch");
    }
}
template<class T> struct Buffer {
    sycl::queue& q;T* allocation;T* data;size_t count;
    Buffer(sycl::queue& queue,size_t n):q(queue),allocation(sycl::malloc_shared<T>(n+32,q)),data(allocation?allocation+16:nullptr),count(n) {
        if(!allocation)throw std::runtime_error("Gemma fixture USM allocation failed");
        std::fill_n(allocation,n+32,T(7));
    }
    ~Buffer(){try{q.wait_and_throw();sycl::free(allocation,q);}catch(...){}}
    Buffer(const Buffer&)=delete;
    T& operator[](size_t i){return data[i];}
    void guards() const {
        for(size_t i=0;i<16;++i)require(allocation[i]==T(7)&&data[count+i]==T(7),"Gemma kernel overwrote a canary");
    }
    void finite() const { for(size_t i=0;i<count;++i)require(std::isfinite(float(data[i])),"Gemma output has a non-finite value"); }
};

void geometry() {
    si::GemmaGeometry g;g.layers=30;g.window=1024;g.full_dim=512;g.swa_dim=256;
    for(int l=0;l<30;++l){g.sliding.push_back(l%6!=5);g.kv_heads.push_back(l%6!=5?8:2);}
    require(g.capacity(0,4096,128)==1152&&g.capacity(5,4096,128)==4096,"Gemma ring/full capacity contract changed");
    require(g.capacity(0,512,128)==512,"Gemma short context must bound its ring capacity");
    const uint64_t expected=(25ull*1152*8*256+5ull*4096*2*512)*2*34/32;
    require(g.cache_bytes(4096,128)==expected,"Gemma per-layer Q8 geometry budget differs");
}

void norm(sycl::queue& q,int dim,int heads,int tokens,int rotation,bool weighted,bool frequencies) {
    const size_t count=size_t(dim)*heads*tokens;
    Buffer<float> x(q,count),w(q,dim),freq(q,std::max(1,rotation/2)),y(q,count);
    for(size_t i=0;i<count;++i)x[i]=i<size_t(dim)?0.f:float(std::sin(i*.031)*.7+std::cos(i*.11)*.2);
    for(int i=0;i<dim;++i)w[i]=.8f+float(i%17)*.01f;
    for(size_t i=0;i<freq.count;++i)freq[i]=1.f+float(i%11)*.125f;
    const float epsilon=1e-6f,base=dim==512?1e6f:1e4f;
    constexpr int position=7;
    si::gemma_head_norm_rope(q,x.data,weighted?w.data:nullptr,y.data,heads,dim,rotation,position,tokens,epsilon,base,frequencies?freq.data:nullptr).wait_and_throw();
    const float theta=std::pow(base,-2.f/float(std::max(rotation,1)));
    for(int row=0;row<heads*tokens;++row) {
        double squares=0;for(int c=0;c<dim;++c)squares+=double(x[size_t(row)*dim+c])*x[size_t(row)*dim+c];
        const double scale=1/std::sqrt(squares/dim+epsilon);
        for(int c=0;c<dim;++c) {
            auto v=[&](int col){return double(x[size_t(row)*dim+col])*scale*(weighted?w[col]:1.f);};
            double expected=v(c);
            if(c<rotation) {
                const int half=rotation/2,j=c<half?c:c-half;
                const float power=float(std::pow(double(theta),double(j)));
                const float angle=float(position+row/heads)*power/(frequencies?freq[j]:1.f);
                expected=c<half?v(j)*std::cos(double(angle))-v(j+half)*std::sin(double(angle)):
                                    v(j)*std::sin(double(angle))+v(j+half)*std::cos(double(angle));
            }
            close(y[size_t(row)*dim+c],expected,6e-6);
        }
    }
    x.guards();w.guards();freq.guards();y.guards();
    std::printf("head_norm D%d H%d T%d rotation%d weighted%d freq%d passed\n",dim,heads,tokens,rotation,weighted,frequencies);
}

void gelu(sycl::queue& q) {
    constexpr int width=704,rows=3;constexpr size_t n=size_t(width)*rows;
    Buffer<float> gate(q,n),up(q,n),packed(q,2*n),a(q,n),b(q,n);
    for(size_t i=0;i<n;++i) {
        gate[i]=float(int(i%41)-20)*.31f;up[i]=float(int(i%13)-6)*.2f;
        packed[(i/width)*2*width+i%width]=gate[i];packed[(i/width)*2*width+width+i%width]=up[i];
    }
    si::gemma_geglu(q,gate.data,up.data,a.data,n).wait_and_throw();
    si::gemma_geglu_packed(q,packed.data,b.data,width,rows).wait_and_throw();
    for(size_t i=0;i<n;++i) {
        const double x=gate[i],expected=.5*x*(1+std::tanh(std::sqrt(2/std::acos(-1.))*x*(1+.044715*x*x)))*up[i];
        close(a[i],expected,4e-6);close(b[i],expected,4e-6);
    }
    gate.guards();up.guards();packed.guards();a.guards();b.guards();
}

void route(sycl::queue& q) {
    constexpr int experts=128,picks=8,tokens=3,width=2816;
    Buffer<float> logits(q,experts*tokens),scales(q,experts),weights(q,picks*tokens);
    Buffer<int32_t> ids(q,picks*tokens);
    for(int e=0;e<experts;++e)scales[e]=.5f+float(e%7)*.1f;
    for(int t=0;t<tokens;++t)for(int e=0;e<experts;++e)
        logits[t*experts+e]=t==0?0.f:t==1?float(int(e%17)-8):float(std::sin(e*.17)*4);
    for(bool have_scale:{false,true}) {
        si::gemma_route(q,logits.data,have_scale?scales.data:nullptr,ids.data,weights.data,experts,picks,tokens,width).wait_and_throw();
        for(int t=0;t<tokens;++t) {
            std::vector<int> order(experts);std::iota(order.begin(),order.end(),0);
            std::sort(order.begin(),order.end(),[&](int a,int b){const float x=logits[t*experts+a],y=logits[t*experts+b];return x==y?a<b:x>y;});
            const double scale=1/std::sqrt(double(width)),maximum=logits[t*experts+order[0]]*scale;
            double total=0,selected=0;
            for(int e=0;e<experts;++e)total+=std::exp(logits[t*experts+e]*scale-maximum);
            for(int k=0;k<picks;++k)selected+=std::exp(logits[t*experts+order[k]]*scale-maximum)/total;
            for(int k=0;k<picks;++k) {
                require(ids[t*picks+k]==order[k],"Gemma router changed top-k tie order or omitted an expert");
                const double expected=std::exp(logits[t*experts+order[k]]*scale-maximum)/total/
                    std::max(selected,std::ldexp(1.,-14))*(have_scale?scales[order[k]]:1.f);
                close(weights[t*picks+k],expected,2e-6);
            }
        }
    }
    logits.guards();scales.guards();weights.guards();ids.guards();
}

void store(sycl::queue& q,int dim,int heads,int window,int position,int tokens) {
    const int capacity=window?window+std::max(tokens,17):position+tokens+3;
    const size_t n=size_t(tokens)*heads*dim,cells=size_t(capacity)*heads*dim;
    Buffer<float> key(q,n),value(q,n);Buffer<int8_t> k(q,cells),v(q,cells);
    Buffer<sycl::half> ks(q,cells/32),vs(q,cells/32);Buffer<int32_t> error(q,1);
    for(size_t i=0;i<n;++i){key[i]=i<32?0.f:float(std::sin(i*.17)*(1+i%3));value[i]=float(std::cos(i*.13)*(1+i%5));}
    error[0]=0;si::GemmaKv cache{k.data,v.data,ks.data,vs.data,capacity,heads,dim,window};
    si::gemma_store_q8(q,key.data,value.data,cache,position,tokens,error.data).wait_and_throw();
    require(error[0]==0,"finite Gemma cache input was rejected");
    for(int head=0;head<heads;++head)for(int cell=0;cell<capacity;++cell) {
        int token=-1;for(int t=0;t<tokens;++t)if((position+t)%capacity==cell)token=t;
        for(int block=0;block<dim/32;++block)for(int which=0;which<2;++which) {
            const size_t at=(size_t(head)*capacity+cell)*dim+block*32;
            const float* x=which?value.data:key.data;const int8_t* codes=which?v.data:k.data;const sycl::half* scales=which?vs.data:ks.data;
            if(token<0) { require(scales[at/32]==sycl::half(7),"Gemma cache store changed an inactive row scale");for(int j=0;j<32;++j)require(codes[at+j]==7,"Gemma cache store changed an inactive row code");continue; }
            const size_t input=(size_t(token)*heads+head)*dim+block*32;float maximum=0;
            for(int j=0;j<32;++j)maximum=std::max(maximum,std::abs(x[input+j]));
            const float d=maximum/127.f,inverse=d>0?1.f/d:0.f;
            require(float(scales[at/32])==float(sycl::half(d)),"Gemma persistent Q8 scale does not round to FP16");
            for(int j=0;j<32;++j) {
                const int expected=int(std::clamp(std::nearbyint(double(float(x[input+j]*inverse))),-127.,127.));
                require(codes[at+j]==expected,"Gemma Q8 codes used the rounded scale or wrong layout");
            }
        }
    }
    for(float invalid:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),1e10f}) {
        key[0]=invalid;error[0]=0;
        si::gemma_store_q8(q,key.data,value.data,cache,position,tokens,error.data).wait_and_throw();
        require(error[0]==(std::isfinite(invalid)?2:1),"Gemma Q8 non-finite/FP16-overflow flag differs");
    }
    key.guards();value.guards();k.guards();v.guards();ks.guards();vs.guards();error.guards();
    std::printf("q8_store D%d KV%d window%d pos%d T%d passed\n",dim,heads,window,position,tokens);
}

void attention(sycl::queue& q,int dim,int heads,int kv,int window,int position,int tokens,float scale) {
    const int context=std::max(4096,position+tokens+1),capacity=window?window+tokens:position+tokens;
    const size_t cells=size_t(kv)*capacity*dim,n=size_t(tokens)*heads*dim;
    Buffer<int8_t> k(q,cells),v(q,cells);Buffer<sycl::half> ks(q,cells/32),vs(q,cells/32);
    Buffer<float> query(q,n),fallback(q,n),mkl(q,n);
    const int first=window?std::max(0,position-window+1):0;
    for(int p=first;p<position+tokens;++p)for(int h=0;h<kv;++h)for(int c=0;c<dim;++c) {
        const size_t at=(size_t(h)*capacity+p%capacity)*dim+c;
        k[at]=int8_t((p*3+h*5+c*7)%127-63);v[at]=int8_t((p*11+h*13+c*3)%127-63);
        ks[at/32]=sycl::half(.002f*float(1+(p+h+c/32)%3));vs[at/32]=sycl::half(.003f*float(1+(p+h+c/32)%5));
    }
    for(size_t i=0;i<n;++i)query[i]=float(std::sin(i*.037)*.04);
    si::GemmaKv cache{k.data,v.data,ks.data,vs.data,capacity,kv,dim,window};
    si::GemmaAttention baseline(q,context,tokens,heads,dim,cells,false);
    if(window&&position>=window&&tokens>1) {
        auto insufficient=cache;insufficient.capacity=window;
        bool rejected=false;
        try{baseline.run(query.data,insufficient,fallback.data,heads,position,tokens,scale);}
        catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Gemma silently accepted an overwritten multi-token SWA ring");
    }
    baseline.run(query.data,cache,fallback.data,heads,position,tokens,scale).wait_and_throw();fallback.finite();
#ifdef TINYSTRATA_INTEL_MKL
    if(tokens>=16) {
        si::GemmaAttention candidate(q,context,tokens,heads,dim,cells,true);
        candidate.run(query.data,cache,mkl.data,heads,position,tokens,scale).wait_and_throw();mkl.finite();
        for(size_t i=0;i<n;++i)close(mkl[i],fallback[i],2e-5);
    }
#endif
    for(int token:{0,tokens/2,tokens-1})for(int head:{0,heads/2,heads-1}) {
        const int end=position+token,low=window?std::max(0,end-window+1):0,kh=head/(heads/kv);
        std::vector<double> scores(size_t(end-low+1));double maximum=-INFINITY;
        for(int p=low;p<=end;++p) {
            double sum=0;for(int c=0;c<dim;++c) {
                const size_t at=(size_t(kh)*capacity+p%capacity)*dim+c;
                sum+=double(query[(size_t(token)*heads+head)*dim+c])*double(k[at])*float(ks[at/32]);
            }
            scores[size_t(p-low)]=sum*scale;maximum=std::max(maximum,sum*scale);
        }
        double total=0;for(double& s:scores){s=std::exp(s-maximum);total+=s;}
        for(int c:{0,1,dim/2,dim-1}) {
            double expected=0;for(int p=low;p<=end;++p) {
                const size_t at=(size_t(kh)*capacity+p%capacity)*dim+c;
                expected+=scores[size_t(p-low)]/total*double(v[at])*float(vs[at/32]);
            }
            const size_t out=(size_t(token)*heads+head)*dim+c;
            close(fallback[out],expected,1e-5);
#ifdef TINYSTRATA_INTEL_MKL
            if(tokens>=16)close(mkl[out],expected,1e-5);
#endif
        }
    }
    k.guards();v.guards();ks.guards();vs.guards();query.guards();fallback.guards();mkl.guards();
    std::printf("attention D%d H%d KV%d window%d pos%d T%d scale%.3f passed\n",dim,heads,kv,window,position,tokens,scale);
}

double q4_value(const uint8_t* raw,size_t col) {
    const auto* b=raw+(col/32)*18;uint16_t bits=uint16_t(b[0])|uint16_t(b[1])<<8;
    const unsigned exponent=(bits>>10)&31,mantissa=bits&1023;double scale;
    if(exponent==0)scale=std::ldexp(double(mantissa),-24);
    else if(exponent==31)scale=mantissa?NAN:INFINITY;
    else scale=std::ldexp(double(1024+mantissa),int(exponent)-25);
    if(bits&0x8000)scale=-scale;
    const size_t local=col%32;const int code=(b[2+local%16]>>(local<16?0:4))&15;
    return scale*(code-8);
}
void q4(sycl::queue& q,int width,int tokens) {
    constexpr int rows=17,experts=3,picks=2;
    const size_t bytes=size_t(rows)*width/32*18;
    Buffer<uint8_t> raw(q,bytes*experts);Buffer<float> decoded(q,size_t(rows)*width),input(q,size_t(tokens)*width),output(q,size_t(tokens)*rows);
    for(size_t block=0;block<bytes*experts/18;++block) {
        const uint16_t bits=block%4==0?0x0000:block%4==1?0x3000:block%4==2?0xb400:0x3800;
        raw[block*18]=uint8_t(bits);raw[block*18+1]=uint8_t(bits>>8);
        for(int j=0;j<16;++j)raw[block*18+2+j]=uint8_t((j+block)%16|((15-j+block)%16)<<4);
    }
    for(size_t i=0;i<input.count;++i)input[i]=float(std::sin(i*.017)*.3);
    si::dequantize(q,2,raw.data,decoded.data,size_t(rows)*width).wait_and_throw();
    for(size_t i=0;i<decoded.count;++i)close(decoded[i],q4_value(raw.data,i),0);
    si::quantized_matmul(q,2,raw.data,input.data,output.data,rows,width,tokens).wait_and_throw();
    for(int t=0;t<tokens;++t)for(int row=0;row<rows;++row) {
        double expected=0;for(int col=0;col<width;++col)expected+=q4_value(raw.data+size_t(row)*width/32*18,col)*input[size_t(t)*width+col];
        close(output[size_t(t)*rows+row],expected,2e-5);
    }
    if(tokens==1) {
        const std::vector<float> matrix(output.data,output.data+output.count);
        si::quantized_matvec(q,2,raw.data,input.data,output.data,rows,width).wait_and_throw();
        for(int row=0;row<rows;++row)close(output[row],matrix[row],2e-5);
    }
    Buffer<int32_t> embedding_ids(q,tokens);Buffer<float> embeddings(q,size_t(tokens)*width);
    for(int t=0;t<tokens;++t)embedding_ids[t]=(t*7+16)%rows;
    si::embedding_lookup(q,2,raw.data,embedding_ids.data,embeddings.data,width,tokens).wait_and_throw();
    for(int t=0;t<tokens;++t)for(int col=0;col<width;++col)
        close(embeddings[size_t(t)*width+col],q4_value(raw.data+size_t(embedding_ids[t])*width/32*18,col),0);
    Buffer<int32_t> ids(q,tokens*picks);Buffer<float> selected(q,size_t(tokens)*picks*rows);
    for(int i=0;i<tokens*picks;++i)ids[i]=(i*2+1)%experts;
    si::selected_matvec(q,2,raw.data,ids.data,picks,input.data,selected.data,rows,width,false,tokens).wait_and_throw();
    for(int t=0;t<tokens;++t)for(int pick=0;pick<picks;++pick)for(int row=0;row<rows;++row) {
        double expected=0;const size_t offset=size_t(ids[t*picks+pick])*bytes+size_t(row)*width/32*18;
        for(int col=0;col<width;++col)expected+=q4_value(raw.data+offset,col)*input[size_t(t)*width+col];
        close(selected[(size_t(t)*picks+pick)*rows+row],expected,2e-5);
    }
    const std::vector<float> independent_selected(selected.data,selected.data+selected.count);
    Buffer<int32_t> counts(q,experts),slots(q,experts*tokens),owners(q,(tokens*picks+3)/4+experts),offsets(q,owners.count),tiles(q,1);
    si::ExpertPlan plan{counts.data,slots.data,owners.data,offsets.data,tiles.data,experts,tokens,picks};
    si::plan_experts(q,ids.data,plan).wait_and_throw();
    si::selected_matmul(q,2,raw.data,plan,input.data,selected.data,rows,width,false).wait_and_throw();
    for(size_t i=0;i<selected.count;++i)close(selected[i],independent_selected[i],2e-5);
    raw.guards();decoded.guards();input.guards();output.guards();ids.guards();selected.guards();
    embedding_ids.guards();embeddings.guards();counts.guards();slots.guards();owners.guards();offsets.guards();tiles.guards();
    std::printf("q4_0 independent byte decode/matrix/embedding/selected/tiled width%d rows17 tokens%d passed\n",width,tokens);
}
template<class T> T read(std::ifstream& in){T value{};if(!in.read(reinterpret_cast<char*>(&value),sizeof value))throw std::runtime_error("Truncated NumPy quant fixture");return value;}
template<class T> std::vector<T> read_vector(std::ifstream& in,size_t n){if(n>32*1024*1024)throw std::runtime_error("Oversized NumPy fixture");std::vector<T> v(n);if(!in.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*sizeof(T))))throw std::runtime_error("Truncated NumPy fixture array");return v;}
void reference(sycl::queue& q,const std::string& path) {
    std::ifstream in(strata::paths::native_path(path),std::ios::binary);require(bool(in),"Cannot open independent NumPy quant reference");
    const auto count=read<uint32_t>(in);require(count>0&&count<=64,"Invalid quant fixture case count");int used=0;
    for(uint32_t n=0;n<count;++n) {
        const int type=read<int32_t>(in),rows=read<int32_t>(in),width=read<int32_t>(in);const auto bytes=read<uint64_t>(in);
        require(rows>0&&rows<=64&&width>0&&width<=8192&&bytes<32*1024*1024,"Invalid NumPy matrix geometry");
        auto raw=read_vector<uint8_t>(in,size_t(bytes));auto x=read_vector<float>(in,width);
        auto decoded=read_vector<float>(in,size_t(rows)*width);auto expected=read_vector<double>(in,rows);
        if(type!=2)continue;
        require(bytes==uint64_t(rows)*width/32*18&&width%32==0,"Invalid NumPy Q4 block layout");
        Buffer<uint8_t> w(q,size_t(bytes));Buffer<float> input(q,width),dequant(q,size_t(rows)*width),out(q,rows);
        std::copy(raw.begin(),raw.end(),w.data);std::copy(x.begin(),x.end(),input.data);
        si::dequantize(q,2,w.data,dequant.data,dequant.count).wait_and_throw();
        for(size_t i=0;i<dequant.count;++i){close(dequant[i],decoded[i],0);close(dequant[i],q4_value(raw.data(),i),0);}
        si::quantized_matvec(q,2,w.data,input.data,out.data,rows,width).wait_and_throw();
        for(int row=0;row<rows;++row)close(out[row],expected[row],2e-5);
        w.guards();input.guards();dequant.guards();out.guards();++used;
    }
    require(used>0,"NumPy fixture contained no Q4_0 rows");std::printf("independent NumPy GGUF Q4_0 cases%d passed\n",used);
}
} // namespace
int main(int argc,char** argv){try{
    int device=0;std::string fixture,model_layout;bool cpu_only=false;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--cpu-only")cpu_only=true;
        else if(arg=="--device"&&i+1<argc)device=std::stoi(argv[++i]);
        else if(arg=="--q4-reference"&&i+1<argc)fixture=argv[++i];
        else if(arg=="--model-layout"&&i+1<argc)model_layout=argv[++i];
        else throw std::runtime_error("Usage: intel_gemma_ops_test [--cpu-only] [--device N] [--q4-reference FILE] [--model-layout FILE]");
    }
    geometry();
    if(!model_layout.empty()) {
        // Header/tensor-directory/bounds admission only. No tensor payload is
        // copied, no weight arena is allocated, and no GPU queue exists yet.
        const auto model=strata::GgufModel::open(model_layout);
        const auto g=si::GemmaGeometry::validate(model);
        require(g.layers==30&&g.vocabulary==262144&&g.cache_bytes(4096,128)>0,"Pinned Gemma metadata admission differs");
        const auto* sampled=model.meta().get("tokenizer.ggml.tokens");
        require(sampled&&sampled->count==262144&&sampled->items.size()<=64,"Gemma fixture must exercise sampled metadata arrays");
        require(model.meta().token_id("<|turn>")==105&&model.meta().token_id("<turn|>")==106,
                "Gemma full vocabulary lookup missed turn controls beyond metadata sample");
        require(model.meta().token_id("<eos>")==1&&model.meta().token_id("<bos>")==2,"Gemma full vocabulary BOS/EOS mismatch");
        require(g.turn_end==106&&std::find(g.terminal_tokens.begin(),g.terminal_tokens.end(),106)!=g.terminal_tokens.end()&&
                std::find(g.terminal_tokens.begin(),g.terminal_tokens.end(),1)!=g.terminal_tokens.end(),
                "Gemma admitted geometry dropped its normal turn/EOS termination");
        std::printf("pinned Gemma metadata/layout/bounds admission passed; no payload loaded\n");
    }
    if(cpu_only){std::printf("PASS %d Gemma geometry CPU checks; no device opened\n",checks);return 0;}
    sycl::queue q(si::select_device(unsigned(device)),[](sycl::exception_list errors){for(auto& e:errors)std::rethrow_exception(e);},sycl::property::queue::in_order{});
    norm(q,64,8,3,64,true,false);norm(q,256,16,3,256,true,false);
    norm(q,512,16,3,512,true,true);norm(q,512,2,17,0,false,false);
    gelu(q);route(q);
    store(q,64,8,0,17,8);store(q,256,8,1024,1023,3);store(q,256,8,1024,1024,3);
    store(q,256,8,1024,4097,512);store(q,512,2,1024,2101,17);
    for(const auto shape:{std::array{64,32,8,0,0,1},std::array{64,32,8,0,31,17},std::array{256,16,8,1024,1023,3},std::array{256,16,8,1024,1024,3},
                         std::array{256,16,8,1024,1024,8},std::array{256,16,8,1024,1025,512},std::array{256,16,8,1024,4097,512},
                         std::array{256,16,8,1024,2101,17},std::array{512,16,2,0,23,3},std::array{512,16,2,0,1030,17}})
        attention(q,shape[0],shape[1],shape[2],shape[3],shape[4],shape[5],shape[0]==64?.125f:1.f);
    for(int width:{32,704,2816})for(int tokens:{1,3,4,1024})q4(q,width,tokens);
    if(!fixture.empty())reference(q,fixture);
    q.wait_and_throw();std::printf("PASS %d Gemma independent numeric/canary checks max_abs_error=%.9g; no model loaded\n",checks,maximum_error);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"Gemma fixture failed: %s\n",e.what());return 1;}}
