// Regression for the complete persistent GDN chain against the ORIGINAL engine
// launches. No model files or API are needed. Link strata_kernels and cudart.
// Optional --bench runs only after all quality checks pass, and reports eager
// and CUDA Graph ABBA events against the current optimized engine schedule.
// --bench-cold conditions the cache with 96 MiB cudaMemsetAsync before each
// single measured call, outside its CUDA events; complete L2 eviction is not
// guaranteed across devices/drivers. --bench-causal-cold additionally restores
// initial recurrent state outside the events and measures state-writing calls.
// --reference-wide selects the proposed wide Q8 output as an explicitly labeled
// timing stand-in; the default output reference is the actual native MMVQ path.
// Compile with the SAME arithmetic flags as q36_engine.cu. In particular do not
// infer its flags from comments in q36_kernels.cuh; check the actual build.
// The copied reference kernels below are from TinyStrata 1fb4ca2 and intentionally
// retain the operation order, float storage boundaries, and state-write behavior.
#include <cstdio>
#if !defined(__CUDACC__) || defined(__HIPCC__)
int main() { std::puts("SKIP: persistent GDN regression requires CUDA"); return 0; }
#else
#include "q36_gdn_chain.cuh"
#include "q36_pair.cuh"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call) do { const auto q36g_error=(call); if(q36g_error!=cudaSuccess) { std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(q36g_error));std::exit(2); } } while(0)
namespace old {
constexpr int kConvTile=16;
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }
__global__ void k_gdn_ab(const float* Wa, const float* Wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int n_in, int HV) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= 2 * HV) return;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane != 0) return;
    if (isb) beta[(size_t) blockIdx.y * HV + r] = sigm(acc);
    else alpha[(size_t) blockIdx.y * HV + r] = softplus(acc + dt[r]) * ssm_a[r];
}

__global__ void __launch_bounds__(128) k_conv_seq(float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T, bool write) {
    __shared__ float part[4];
    const int c = blockIdx.x * 128 + threadIdx.x;
    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f;
    if (gridDim.y == 1 || blockIdx.y == 0) {   // other tiles read only qkv; avoid racing tile zero's history write
        v0 = hist[c * 3]; v1 = hist[c * 3 + 1]; v2 = hist[c * 3 + 2];
    }
    // Prompt tiles run independently. Only tile zero writes the final history; each later tile's history is
    // the same three preceding inputs as in the sequential loop. Decode / verify / commit use one tile.
    int t_first = 0, t_end = T;
    const bool tiled = gridDim.y > 1;
    if (tiled) {
        t_first = (int) blockIdx.y * kConvTile;
        t_end = min(T, t_first + kConvTile);
        if (t_first == 0 && write) {
            const float old3[3] = {v0, v1, v2};
            for (int k = 0; k < 3; ++k) {
                const int idx = T - 3 + k;
                hist[c * 3 + k] = idx >= 0 ? qkv[(size_t) idx * C + c] : old3[idx + 3];
            }
        }
        if (t_first > 0) {
            v0 = qkv[(size_t) (t_first - 3) * C + c];
            v1 = qkv[(size_t) (t_first - 2) * C + c];
            v2 = qkv[(size_t) (t_first - 1) * C + c];
        }
        if (t_first >= T) return;
    }
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const bool qk = (int) blockIdx.x < qk_heads;
    for (int t = t_first; t < t_end; ++t) {
        const float x = qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (qk) {
            float sq = y * y;
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            const float ss = part[0] + part[1] + part[2] + part[3];
            __syncthreads();
            y *= rsqrtf(ss + eps);
        }
        h[(size_t) t * C + c] = y;
    }
    if (!write || tiled) return;   // verify: commit owns it; tile zero already stored it
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

__global__ void __launch_bounds__(512) k_gdn_seq(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write) {
    constexpr int S = 128, RG = 4, RPG = 32;
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % h_k;
    const int ZV = h_v * S;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            y[(size_t) t * ZV + head * S + col] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
        }
        __syncthreads();   // sk / sq / the reductions are rewritten by the next token
    }
    if (!write) return;   // a verify window: the state is commit()'s to write
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

} // namespace old
namespace K = strata::kernels;
constexpr int N=2048, HK=16, HV=32, S=128, C=8192, Z=4096, MT=5, GUARD=32;
constexpr size_t NS=size_t(S)*HV*S, NH=size_t(C)*3;
constexpr float EPS=1e-6f;
int selected_blocks=0; // test-only override; zero keeps the ordinary SM default
bool wide_reference=false; // explicit stand-in; default matches the current engine
struct Q80 { half d; int8_t q[32]; };
struct Q6 { uint8_t ql[128],qh[64]; int8_t scales[16]; half d; };
static_assert(sizeof(Q80)==34 && sizeof(Q6)==210,"native GGUF layouts");
struct Device {
 void* p=nullptr;size_t bytes;
 explicit Device(size_t n):bytes(n) { CK(cudaMalloc(&p,n)); }
 Device(const Device&)=delete;Device& operator=(const Device&)=delete;
 ~Device(){cudaFree(p);}
 template<class T> T* as() const {return static_cast<T*>(p);}
 void poison(cudaStream_t s)const {CK(cudaMemsetAsync(p,0xff,bytes,s));}
};
struct Arm {
 Device hist{(NH+GUARD)*4},state{(NS+GUARD)*4};
 Device qkv{(size_t(MT)*C+GUARD)*4},z{(size_t(MT)*Z+GUARD)*4};
 Device alpha{(size_t(MT)*HV+GUARD)*4},beta{(size_t(MT)*HV+GUARD)*4};
 Device hc{(size_t(MT)*C+GUARD)*4},gdn{(size_t(MT)*Z+GUARD)*4};
 Device out{(size_t(MT)*N+GUARD)*4},oq{size_t(MT)*Z/32*36+GUARD*4};
 Device il{K::native_q8_1_il_bytes(C,4)};
 void reset(const Device& ih,const Device& is,cudaStream_t st) {
  for(auto* d:{&hist,&state,&qkv,&z,&alpha,&beta,&hc,&gdn,&out,&oq})d->poison(st);
  CK(cudaMemcpyAsync(hist.p,ih.p,NH*4,cudaMemcpyDeviceToDevice,st));
  CK(cudaMemcpyAsync(state.p,is.p,NS*4,cudaMemcpyDeviceToDevice,st));
 }
};
struct Fixture {
 int type;uint32_t rng=0x5316a2c9;
 std::vector<uint8_t> wq,wz,wo;
 std::vector<float> input,wa,wb,dt,sa,cw,norm,ih,is;
 Device dwq,dwz,dwo,dx{size_t(MT)*N*4},dxq{size_t(MT)*N/32*36};
 Device dwa{size_t(HV)*N*4},dwb{size_t(HV)*N*4},ddt{HV*4},dsa{HV*4},dcw{size_t(C)*4*4},dno{S*4};
 Device dih{NH*4},dis{NS*4};
 unsigned next(){rng=rng*1664525u+1013904223u;return rng;}
 float value(float scale){return float(int(next()%2001)-1000)*scale/1000.f;}
 void weights(std::vector<uint8_t>& bytes, int ni,int no){
  if(type==8){std::vector<Q80> v(size_t(ni/32)*no);for(auto& b:v){b.d=__float2half(.0001f*float(1+next()%20));for(auto& q:b.q)q=int8_t(int(next()%255)-127);}std::memcpy(bytes.data(),v.data(),bytes.size());}
  else {std::vector<Q6> v(size_t(ni/256)*no);for(auto& b:v){for(auto& x:b.ql)x=uint8_t(next()>>24);for(auto& x:b.qh)x=uint8_t(next()>>24);for(auto& x:b.scales)x=int8_t(int(next()%31)-15);b.d=__float2half(.00001f*float(1+next()%20));}std::memcpy(bytes.data(),v.data(),bytes.size());}
 }
 explicit Fixture(int ty):type(ty),wq(K::native_mmvq_weight_bytes(ty,N,C)),wz(K::native_mmvq_weight_bytes(ty,N,Z)),wo(K::native_mmvq_weight_bytes(ty,Z,N)),input(size_t(MT)*N),wa(size_t(HV)*N),wb(wa.size()),dt(HV),sa(HV),cw(size_t(C)*4),norm(S),ih(NH),is(NS),dwq(wq.size()),dwz(wz.size()),dwo(wo.size()) {
  weights(wq,N,C);weights(wz,N,Z);weights(wo,Z,N);
  for(auto& x:input)x=value(.3f);for(auto& x:wa)x=value(.02f);for(auto& x:wb)x=value(.02f);
  for(int i=0;i<HV;++i){dt[i]=value(1.2f);sa[i]=-.1f-std::abs(value(.5f));}
  for(auto& x:cw)x=value(.2f);for(auto& x:norm)x=.9f+value(.2f);
  for(auto& x:ih)x=value(.1f);for(auto& x:is)x=value(.02f);
  auto copy=[](const Device& d,const void* p){CK(cudaMemcpy(d.p,p,d.bytes,cudaMemcpyHostToDevice));};
  copy(dwq,wq.data());copy(dwz,wz.data());copy(dwo,wo.data());copy(dx,input.data());copy(dwa,wa.data());copy(dwb,wb.data());copy(ddt,dt.data());copy(dsa,sa.data());copy(dcw,cw.data());copy(dno,norm.data());copy(dih,ih.data());copy(dis,is.data());
 }
 q36g::Args args(Arm& a,int T,bool write,int token=0){
  q36g::Args x{};x.type=type;x.T=T;x.n=N;x.hk=HK;x.hv=HV;x.write_state=write;
  x.wqkv=dwq.p;x.wz=dwz.p;x.wo=dwo.p;x.input=dx.as<float>()+size_t(token)*N;
  x.input_q=dxq.as<q36k::QB>()+size_t(token)*N/32;
  x.wa=dwa.as<float>();x.wb=dwb.as<float>();x.dt=ddt.as<float>();x.ssm_a=dsa.as<float>();x.conv_w=dcw.as<float>();x.norm=dno.as<float>();
  x.hist=a.hist.as<float>();x.state=a.state.as<float>();x.eps=EPS;
  x.qkv=a.qkv.as<float>()+size_t(token)*C;x.z=a.z.as<float>()+size_t(token)*Z;
  x.alpha=a.alpha.as<float>()+size_t(token)*HV;x.beta=a.beta.as<float>()+size_t(token)*HV;
  x.hc=a.hc.as<float>()+size_t(token)*C;x.gdn_y=a.gdn.as<float>()+size_t(token)*Z;
  x.out_q=a.oq.as<q36k::QB>()+size_t(token)*Z/32;x.out=a.out.as<float>()+size_t(token)*N;
  return x;
 }
};
void reference(const q36g::Args& a,cudaStream_t st,bool interleaved=false,void* scratch=nullptr){
 const auto project=[&](const void* w,const void* x,float* out,int ni,int no){
  if(interleaved && K::native_mmvq_il_supported(a.type,a.T,no)){
   K::native_q8_1_interleave(x,scratch,ni,a.T,st);
   K::native_mmvq_il(a.type,w,x,scratch,out,ni,no,a.T,st);
  }else K::native_mmvq(a.type,w,x,out,ni,no,a.T,st);
 };
 project(a.wqkv,a.input_q,a.qkv,N,C);
 project(a.wz,a.input_q,a.z,N,Z);
 old::k_gdn_ab<<<dim3((2*HV+7)/8,a.T),256,0,st>>>(a.wa,a.wb,a.input,a.dt,a.ssm_a,a.alpha,a.beta,N,HV);
 old::k_conv_seq<<<C/128,128,0,st>>>(a.hist,a.qkv,a.conv_w,a.hc,C,2*HK,a.eps,a.T,a.write_state);
 old::k_gdn_seq<<<HV,dim3(128,4),0,st>>>(a.state,a.hc,C,a.alpha,a.beta,a.z,a.norm,a.eps,a.gdn_y,HK,HV,a.T,a.write_state);
 K::native_quantize_q8_1(a.gdn_y,a.out_q,Z*a.T,1,st);
 project(a.wo,a.out_q,a.out,Z,N);
}
// The current optimized engine is the timing reference, rather than the
// generic MMVQ oracle above. Clover already uses paired Q8 input projections;
// its 4096-input ssm_out exceeds q8_small's current 2048-input bound and uses
// native MMVQ. --reference-wide explicitly selects the proposed wider Q8 path.
// Qwen may use the exact interleaved Q6 path.
template<int NC> void engine_q8_reference(const q36g::Args& a,cudaStream_t st){
 q36k::launch_q8_rows_pair<NC,2>(a.wqkv,a.wz,a.input_q,a.qkv,a.z,N/32,C,Z,st);
 old::k_gdn_ab<<<dim3((2*HV+7)/8,NC),256,0,st>>>(a.wa,a.wb,a.input,a.dt,a.ssm_a,a.alpha,a.beta,N,HV);
 old::k_conv_seq<<<C/128,128,0,st>>>(a.hist,a.qkv,a.conv_w,a.hc,C,2*HK,a.eps,NC,a.write_state);
 old::k_gdn_seq<<<HV,dim3(128,4),0,st>>>(a.state,a.hc,C,a.alpha,a.beta,a.z,a.norm,a.eps,a.gdn_y,HK,HV,NC,a.write_state);
 K::native_quantize_q8_1(a.gdn_y,a.out_q,Z*NC,1,st);
 if(wide_reference)q36k::launch_q8_rows<NC,4,false>(a.wo,nullptr,a.out_q,a.out,Z/32,N,st);
 else K::native_mmvq(8,a.wo,a.out_q,a.out,Z,N,NC,st);
}
void wide_output(const void* weights,const void* input,float* output,int T,cudaStream_t st){
 switch(T){
  case 1:q36k::launch_q8_rows<1,4,false>(weights,nullptr,input,output,Z/32,N,st);break;
  case 2:q36k::launch_q8_rows<2,4,false>(weights,nullptr,input,output,Z/32,N,st);break;
  case 3:q36k::launch_q8_rows<3,4,false>(weights,nullptr,input,output,Z/32,N,st);break;
  case 4:q36k::launch_q8_rows<4,4,false>(weights,nullptr,input,output,Z/32,N,st);break;
 }
}
void engine_reference(const q36g::Args& a,cudaStream_t st,bool il,void* scratch){
 if(a.type!=8){reference(a,st,il,scratch);return;}
 switch(a.T){
  case 1:engine_q8_reference<1>(a,st);break;
  case 2:engine_q8_reference<2>(a,st);break;
  case 3:engine_q8_reference<3>(a,st);break;
  case 4:engine_q8_reference<4>(a,st);break;
 }
}
std::vector<uint32_t> download(const Device& d,cudaStream_t st){
 std::vector<uint32_t> v(d.bytes/4);CK(cudaMemcpyAsync(v.data(),d.p,d.bytes,cudaMemcpyDeviceToHost,st));CK(cudaStreamSynchronize(st));return v;
}
void fail(const char* what,int type,int T,bool write,int replay,size_t index=0){
 std::fprintf(stderr,"FAIL %s type=%d T=%d write=%d replay=%d word=%zu\n",what,type,T,int(write),replay,index);std::exit(1);
}
void compare(const Device& ref,const Device& cand,size_t active,const char* name,int type,int T,bool write,int replay,cudaStream_t st,bool floats=true){
 auto a=download(ref,st),b=download(cand,st);
 for(size_t i=0;i<a.size();++i){
  if(a[i]!=b[i]){
   float av,bv;std::memcpy(&av,&a[i],4);std::memcpy(&bv,&b[i],4);
   std::fprintf(stderr,"  ref=%08x %.9g candidate=%08x %.9g\n",a[i],double(av),b[i],double(bv));
   fail(name,type,T,write,replay,i);
  }
  if(i>=active && b[i]!=0xffffffffu)fail("unused/guard write",type,T,write,replay,i);
  if(floats && i<active){float f;std::memcpy(&f,&b[i],4);if(!std::isfinite(f))fail("nonfinite output/state",type,T,write,replay,i);}
 }
}
void compare_arms(const Arm& a,const Arm& b,int type,int T,bool write,int replay,cudaStream_t st){
 compare(a.qkv,b.qkv,size_t(T)*C,"qkv",type,T,write,replay,st);
 compare(a.z,b.z,size_t(T)*Z,"z",type,T,write,replay,st);
 compare(a.alpha,b.alpha,size_t(T)*HV,"alpha",type,T,write,replay,st);
 compare(a.beta,b.beta,size_t(T)*HV,"beta",type,T,write,replay,st);
 compare(a.hc,b.hc,size_t(T)*C,"hc",type,T,write,replay,st);
 compare(a.gdn,b.gdn,size_t(T)*Z,"gdn_y",type,T,write,replay,st);
 compare(a.oq,b.oq,size_t(T)*Z/32*9,"gdn_q8",type,T,write,replay,st,false);
 compare(a.out,b.out,size_t(T)*N,"out",type,T,write,replay,st);
 compare(a.hist,b.hist,NH,"history",type,T,write,replay,st);
 compare(a.state,b.state,NS,"SSM",type,T,write,replay,st);
}
void unchanged(const Device& d,const std::vector<float>& initial,const char* name,int type,int T,int replay,cudaStream_t st){
 const auto words=download(d,st);if(std::memcmp(words.data(),initial.data(),initial.size()*4))fail(name,type,T,false,replay);
}
double weight(const std::vector<uint8_t>& w,int type,int ni,int row,int k){
 if(type==8){Q80 b;std::memcpy(&b,w.data()+(size_t(row)*(ni/32)+k/32)*34,34);return double(__half2float(b.d))*b.q[k%32];}
 Q6 b;std::memcpy(&b,w.data()+(size_t(row)*(ni/256)+k/256)*210,210);
 const int j=k%256,chunk=j/128,p=j%128,l=p%32,part=p/32;
 const uint8_t ql=b.ql[chunk*64+l+(part%2)*32],qh=b.qh[chunk*32+l];
 const int q=int((part<2 ? ql&15 : ql>>4)|(((qh>>(2*part))&3)<<4))-32;
 return double(__half2float(b.d))*b.scales[chunk*8+l/16+part*2]*q;
}
void cpu_dots(const Fixture& f,const Device& xq,const Device& out,const std::vector<uint8_t>& weights,int ni,int no,int T,cudaStream_t st,double& maxerr){
 std::vector<q36k::QB> in(size_t(ni/32)*T);std::vector<float> y(size_t(no)*T);
 CK(cudaMemcpyAsync(in.data(),xq.p,in.size()*36,cudaMemcpyDeviceToHost,st));
 CK(cudaMemcpyAsync(y.data(),out.p,y.size()*4,cudaMemcpyDeviceToHost,st));CK(cudaStreamSynchronize(st));
 for(int t=0;t<T;++t)for(int sample=0;sample<8;++sample){
  const int row=sample*(no-1)/7;double exact=0,mag=0;
  for(int k=0;k<ni;++k){const auto& xb=in[size_t(t)*(ni/32)+k/32];const double v=weight(weights,f.type,ni,row,k)*double(__low2float(xb.ds))*xb.qs[k%32];exact+=v;mag+=std::abs(v);}
  const double err=std::abs(double(y[size_t(t)*no+row])-exact);maxerr=std::max(maxerr,err);
  if(!std::isfinite(err)||err>mag*4e-6+2e-6)fail("CPU dequantized dot bound",f.type,T,true,0,row);
 }
}
void commit(const q36g::Args& a,int keep,cudaStream_t st){
 old::k_conv_seq<<<C/128,128,0,st>>>(a.hist,a.qkv,a.conv_w,a.hc,C,2*HK,a.eps,keep,true);
 old::k_gdn_seq<<<HV,dim3(128,4),0,st>>>(a.state,a.hc,C,a.alpha,a.beta,a.z,a.norm,a.eps,nullptr,HK,HV,keep,true);
}
uint64_t checksum(const Device& d,cudaStream_t st){
 uint64_t h=1469598103934665603ull;for(uint32_t w:download(d,st)){h^=w;h*=1099511628211ull;}return h;
}
void benchmark(Fixture& f,Arm& ref,Arm& cand,cudaStream_t st){
 constexpr int calls=32;
 for(int T=1;T<=4;++T)for(bool il:{false,true}){
  if(il && !(f.type==14 && T>1 && (K::native_mmvq_il_supported(f.type,T,C) ||
             K::native_mmvq_il_supported(f.type,T,Z) || K::native_mmvq_il_supported(f.type,T,N))))continue;
  auto ra=f.args(ref,T,false),ca=f.args(cand,T,false);q36g::Plan device{};CK(q36g::prepare(device,ca,selected_blocks));
  std::vector<int> tested;
  for(int requested:{std::max(1,device.sms/2),selected_blocks?selected_blocks:device.sms,2*device.sms}){
   q36g::Plan p{};CK(q36g::prepare(p,ca,requested));const int ti=f.type==8?0:1,b=p.blocks[ti][T-1];
   if(std::find(tested.begin(),tested.end(),b)!=tested.end()){
    std::printf("BENCH_SKIP type=%d T=%d requested=%d actual=%d occupancy_cap_or_duplicate\n",f.type,T,requested,b);continue;
   }
   tested.push_back(b);ref.reset(f.dih,f.dis,st);cand.reset(f.dih,f.dis,st);
   engine_reference(ra,st,il,ref.il.p);CK(q36g::launch(p,ca,st));CK(cudaStreamSynchronize(st));
   compare_arms(ref,cand,f.type,T,false,20,st);
   cudaGraph_t g[2]{};cudaGraphExec_t e[2]{};
   for(int mode=0;mode<2;++mode){
    CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
    if(mode)CK(q36g::launch(p,ca,st));else engine_reference(ra,st,il,ref.il.p);
    CK(cudaStreamEndCapture(st,&g[mode]));CK(cudaGraphInstantiate(&e[mode],g[mode],nullptr,nullptr,0));
   }
   cudaEvent_t begin{},end{};CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
   for(bool graph:{false,true}){
    const auto run=[&](int mode){
     if(graph)CK(cudaGraphLaunch(e[mode],st));
     else if(mode)CK(q36g::launch(p,ca,st));else engine_reference(ra,st,il,ref.il.p);
    };
    for(int mode=0;mode<2;++mode)for(int i=0;i<8;++i)run(mode);
    CK(cudaStreamSynchronize(st));double total[2]{};
    // State writes are disabled and all inputs stay fixed. No copies, resets,
    // initialization or L2 flush are included in either event interval.
    for(int mode:{0,1,1,0}){
     CK(cudaEventRecord(begin,st));for(int i=0;i<calls;++i)run(mode);
     CK(cudaEventRecord(end,st));CK(cudaEventSynchronize(end));float ms=0;CK(cudaEventElapsedTime(&ms,begin,end));total[mode]+=ms;
    }
    compare_arms(ref,cand,f.type,T,false,21,st);
    std::printf("BENCH %s type=%d T=%d reference=%s requested=%d blocks=%d regs=%d shared=%zu old_us=%.3f chain_us=%.3f ABBA=2+2x%d checksum=%016llx\n",
      graph?"graph":"eager",f.type,T,f.type==8?(wide_reference?"Q8pair+wide_out":"Q8pair+native_out"):il?"Q6_IL":"Q6_native",requested,b,p.registers[ti][T-1],p.shared_bytes[ti][T-1],
      total[0]*1000/(2*calls),total[1]*1000/(2*calls),calls,static_cast<unsigned long long>(checksum(cand.out,st)));
    std::fflush(stdout);
   }
   CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
   for(int mode=0;mode<2;++mode){CK(cudaGraphExecDestroy(e[mode]));CK(cudaGraphDestroy(g[mode]));}
  }
 }
}
void benchmark_cold(Fixture& f,Arm& ref,Arm& cand,cudaStream_t st,bool write=false){
 constexpr int cycles=8;constexpr size_t flush_bytes=96u*1024u*1024u;
 Device flush(flush_bytes);
 for(int T=1;T<=4;++T)for(bool il:{false,true}){
  if(il && !(f.type==14 && T>1 && (K::native_mmvq_il_supported(f.type,T,C) ||
             K::native_mmvq_il_supported(f.type,T,Z) || K::native_mmvq_il_supported(f.type,T,N))))continue;
  auto ra=f.args(ref,T,write),ca=f.args(cand,T,write);q36g::Plan p{};CK(q36g::prepare(p,ca,selected_blocks));
  const int ti=f.type==8?0:1;
  ref.reset(f.dih,f.dis,st);cand.reset(f.dih,f.dis,st);
  engine_reference(ra,st,il,ref.il.p);CK(q36g::launch(p,ca,st));CK(cudaStreamSynchronize(st));
  compare_arms(ref,cand,f.type,T,write,50,st);
  cudaGraph_t g[2]{};cudaGraphExec_t exec[2]{};
  for(int mode=0;mode<2;++mode){
   CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
   if(mode)CK(q36g::launch(p,ca,st));else engine_reference(ra,st,il,ref.il.p);
   CK(cudaStreamEndCapture(st,&g[mode]));CK(cudaGraphInstantiate(&exec[mode],g[mode],nullptr,nullptr,0));
  }
  cudaEvent_t begin{},end{};CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
  for(bool graph:{false,true}){
   const auto run=[&](int mode){
    if(graph)CK(cudaGraphLaunch(exec[mode],st));
    else if(mode)CK(q36g::launch(p,ca,st));else engine_reference(ra,st,il,ref.il.p);
   };
   for(int mode=0;mode<2;++mode)for(int i=0;i<4;++i)run(mode);
   CK(cudaStreamSynchronize(st));std::vector<double> samples[2];
   for(int cycle=0;cycle<cycles;++cycle)for(int mode:{0,1,1,0}){
    // Same stream ordering completes the eviction before the begin timestamp.
    // No flush, state reset, transfer or initialization lies between events.
    if(write){
     Arm& arm=mode?cand:ref;
     CK(cudaMemcpyAsync(arm.hist.p,f.dih.p,NH*4,cudaMemcpyDeviceToDevice,st));
     CK(cudaMemcpyAsync(arm.state.p,f.dis.p,NS*4,cudaMemcpyDeviceToDevice,st));
    }
    CK(cudaMemsetAsync(flush.p,(cycle*17+mode*41)&255,flush_bytes,st));
    CK(cudaEventRecord(begin,st));run(mode);CK(cudaEventRecord(end,st));
    CK(cudaEventSynchronize(end));float ms=0;CK(cudaEventElapsedTime(&ms,begin,end));samples[mode].push_back(double(ms)*1000);
   }
   double means[2]{},medians[2]{};
   for(int mode=0;mode<2;++mode){
    for(double us:samples[mode])means[mode]+=us;means[mode]/=samples[mode].size();
    std::sort(samples[mode].begin(),samples[mode].end());const size_t mid=samples[mode].size()/2;
    medians[mode]=(samples[mode][mid-1]+samples[mode][mid])*.5;
   }
   compare_arms(ref,cand,f.type,T,write,51,st);
   std::printf("%s %s type=%d T=%d reference=%s requested=%d blocks=%d regs=%d shared=%zu flush_bytes=%zu projection_weight_bytes=%zu nominal_ssm_read_bytes=%zu nominal_hist_read_bytes=%zu state_write_bytes=%zu old_us=%.3f chain_us=%.3f old_median_us=%.3f chain_median_us=%.3f ABBA=%dx1 checksum=%016llx\n",
     write?"BENCH_CAUSAL_COLD_L2":"BENCH_COLD_L2",graph?"graph":"eager",f.type,T,f.type==8?(wide_reference?"Q8pair+wide_out":"Q8pair+native_out"):il?"Q6_IL":"Q6_native",
     selected_blocks,p.blocks[ti][T-1],p.registers[ti][T-1],p.shared_bytes[ti][T-1],flush_bytes,f.wq.size()+f.wz.size()+f.wo.size(),NS*4,NH*4,write?(NS+NH)*4:0,
     means[0],means[1],medians[0],medians[1],cycles,static_cast<unsigned long long>(checksum(cand.out,st)));
   std::fflush(stdout);
  }
  CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
  for(int mode=0;mode<2;++mode){CK(cudaGraphExecDestroy(exec[mode]));CK(cudaGraphDestroy(g[mode]));}
 }
}
int main(int argc,char** argv){
 bool bench=false,cold=false,causal=false,valid=true;
 for(int i=1;i<argc;++i){
  if(std::strcmp(argv[i],"--bench")==0)bench=true;
  else if(std::strcmp(argv[i],"--bench-cold")==0)cold=true;
  else if(std::strcmp(argv[i],"--bench-causal-cold")==0){cold=true;causal=true;}
  else if(std::strcmp(argv[i],"--reference-wide")==0)wide_reference=true;
  else if(std::strncmp(argv[i],"--blocks=",9)==0){
   char* end=nullptr;errno=0;const long value=std::strtol(argv[i]+9,&end,10);
   if(errno || end==argv[i]+9 || *end || value<0 || value>INT_MAX)valid=false;
   else selected_blocks=int(value);
  }else valid=false;
 }
 if(!valid || (bench&&cold)){std::fputs("usage: q36-gdn-chain [--bench|--bench-cold|--bench-causal-cold] [--blocks=N] [--reference-wide]\n",stderr);return 2;}
 std::printf("TEST_BLOCKS requested=%d (0=ordinary SM default)\n",selected_blocks);
 cudaStream_t st{};CK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));K::native_mmvq_set_multi_exact(true);
 int checks=0,il_checks=0;double maxerr=0;
 for(int type:{8,14}){
  Fixture f(type);Arm ref,cand;
  K::native_quantize_q8_1(f.dx.as<float>(),f.dxq.p,MT*N,1,st);CK(cudaStreamSynchronize(st));
  if(type==8)for(int T=1;T<=4;++T){
   // Isolate the proposed nb=128/NJ=4 output helper from fusion. The native
   // output and candidate output never alias, and every inactive column/guard
   // starts poisoned. The helper receives no recurrent state pointers.
   ref.reset(f.dih,f.dis,st);cand.reset(f.dih,f.dis,st);
   reference(f.args(ref,T,false),st);
   wide_output(f.dwo.p,ref.oq.p,cand.out.as<float>(),T,st);CK(cudaStreamSynchronize(st));
   compare(ref.out,cand.out,size_t(T)*N,"wide Q8 output",type,T,false,60,st);
   ++checks;
   cudaGraph_t graph{};cudaGraphExec_t executable{};
   CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
   wide_output(f.dwo.p,ref.oq.p,cand.out.as<float>(),T,st);
   CK(cudaStreamEndCapture(st,&graph));CK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
   for(int replay=0;replay<3;++replay){
    cand.out.poison(st);CK(cudaGraphLaunch(executable,st));CK(cudaStreamSynchronize(st));
    compare(ref.out,cand.out,size_t(T)*N,"wide Q8 Graph output",type,T,false,61+replay,st);++checks;
   }
   CK(cudaGraphExecDestroy(executable));CK(cudaGraphDestroy(graph));
   unchanged(ref.hist,f.ih,"wide reference history mutated",type,T,60,st);
   unchanged(ref.state,f.is,"wide reference SSM mutated",type,T,60,st);
   unchanged(cand.hist,f.ih,"wide helper history mutated",type,T,60,st);
   unchanged(cand.state,f.is,"wide helper SSM mutated",type,T,60,st);
   std::printf("WIDE_Q8_OUTPUT_PASS T=%d eager=1 graph=3\n",T);
  }
  if(type==14)for(int T=1;T<=4;++T)
   std::printf("Q6_IL_SUPPORT T=%d qkv=%d z=%d out=%d\n",T,
    int(K::native_mmvq_il_supported(type,T,C)),int(K::native_mmvq_il_supported(type,T,Z)),int(K::native_mmvq_il_supported(type,T,N)));
  for(int T=1;T<=4;++T)for(bool write:{false,true})for(bool il:{false,true}){
   if(il && !(T>=2 && type==14 && (K::native_mmvq_il_supported(type,T,C) ||
              K::native_mmvq_il_supported(type,T,Z) || K::native_mmvq_il_supported(type,T,N))))continue;
   auto ra=f.args(ref,T,write),ca=f.args(cand,T,write);q36g::Plan plan{};
   const auto status=q36g::prepare(plan,ca,selected_blocks);
   if(status==cudaErrorNotSupported){std::printf("SKIP unsupported device/format: type=%d T=%d\n",type,T);CK(cudaStreamDestroy(st));return 77;}
   CK(status);if(!q36g::supported(plan,ca))fail("prepare/support mismatch",type,T,write,-1);
   if(!write&&!il)std::printf("QUALITY_PLAN type=%d T=%d requested=%d blocks=%d\n",type,T,selected_blocks,plan.blocks[type==8?0:1][T-1]);
   // Unsupported geometry/types must choose the ordinary chain, without ever
   // enqueueing a kernel that dereferences buffers under the wrong layout.
   for(int invalid=0;invalid<5;++invalid){
    auto bad=ca;
    if(invalid==0)bad.type=2;
    if(invalid==1)bad.T=0;
    if(invalid==2)bad.n=N+1;
    if(invalid==3)bad.hk=0;
    if(invalid==4)bad.input_q=nullptr;
    if(q36g::supported(plan,bad))fail("unsupported guard",type,T,write,invalid);
   }
   ref.reset(f.dih,f.dis,st);cand.reset(f.dih,f.dis,st);reference(ra,st,il,ref.il.p);CK(q36g::launch(plan,ca,st));CK(cudaStreamSynchronize(st));
   compare_arms(ref,cand,type,T,write,-1,st);if(!write){unchanged(cand.hist,f.ih,"verify history mutated",type,T,-1,st);unchanged(cand.state,f.is,"verify SSM mutated",type,T,-1,st);}
   cpu_dots(f,f.dxq,cand.qkv,f.wq,N,C,T,st,maxerr);cpu_dots(f,f.dxq,cand.z,f.wz,N,Z,T,st,maxerr);cpu_dots(f,cand.oq,cand.out,f.wo,Z,N,T,st,maxerr);++checks;if(il)++il_checks;
   cudaGraph_t graphs[2]{};cudaGraphExec_t exec[2]{};
   for(int mode=0;mode<2;++mode){
    CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));if(mode)CK(q36g::launch(plan,ca,st));else reference(ra,st,il,ref.il.p);
    CK(cudaStreamEndCapture(st,&graphs[mode]));CK(cudaGraphInstantiate(&exec[mode],graphs[mode],nullptr,nullptr,0));
   }
   for(int replay=0;replay<3;++replay){
    // Reset inputs to both arms outside capture. Outputs are never shared and
    // are poisoned before EVERY eager or captured launch, so omitted writes fail.
    ref.reset(f.dih,f.dis,st);cand.reset(f.dih,f.dis,st);
    CK(cudaGraphLaunch(exec[0],st));CK(cudaGraphLaunch(exec[1],st));CK(cudaStreamSynchronize(st));
    compare_arms(ref,cand,type,T,write,replay,st);
    if(!write){unchanged(cand.hist,f.ih,"graph verify history mutated",type,T,replay,st);unchanged(cand.state,f.is,"graph verify SSM mutated",type,T,replay,st);}++checks;if(il)++il_checks;
   }
   for(int mode=0;mode<2;++mode){CK(cudaGraphExecDestroy(exec[mode]));CK(cudaGraphDestroy(graphs[mode]));}
  }
  // A four-token window must equal four causal single-token steps, including
  // every intermediate output and both recurrent states (MTP exactness).
  ref.reset(f.dih,f.dis,st);cand.reset(f.dih,f.dis,st);
  reference(f.args(ref,4,true),st);
  q36g::Plan single{};auto sa=f.args(cand,1,true);CK(q36g::prepare(single,sa,selected_blocks));
  for(int t=0;t<4;++t){auto a=f.args(cand,1,true,t);CK(q36g::launch(single,a,st));}
  CK(cudaStreamSynchronize(st));compare_arms(ref,cand,type,4,true,10,st);++checks;
  // A verify window leaves the old state intact. Committing a kept prefix using
  // the engine's original commit kernels must match causal generation of it.
  for(int keep:{1,2,3}){
   ref.reset(f.dih,f.dis,st);cand.reset(f.dih,f.dis,st);reference(f.args(ref,keep,true),st);
   auto a=f.args(cand,4,false);q36g::Plan verify{};CK(q36g::prepare(verify,a,selected_blocks));CK(q36g::launch(verify,a,st));commit(a,keep,st);CK(cudaStreamSynchronize(st));
   compare(ref.hist,cand.hist,NH,"committed history",type,keep,true,11,st);
   compare(ref.state,cand.state,NS,"committed SSM",type,keep,true,11,st);++checks;
  }
  // Exercise nondefault resident grid sizes even without requesting timings.
  // Smaller grids reuse one CTA's shared storage across several heads.
  for(int T=1;T<=4;++T)for(bool write:{false,true}){
   auto ra=f.args(ref,T,write),ca=f.args(cand,T,write);q36g::Plan base{};CK(q36g::prepare(base,ca,selected_blocks));
   std::vector<int> tested{base.blocks[type==8?0:1][T-1]};
   for(int requested:{std::max(1,base.sms/2),2*base.sms}){
    q36g::Plan p{};CK(q36g::prepare(p,ca,requested));const int blocks=p.blocks[type==8?0:1][T-1];
    if(std::find(tested.begin(),tested.end(),blocks)!=tested.end())continue;tested.push_back(blocks);
    ref.reset(f.dih,f.dis,st);cand.reset(f.dih,f.dis,st);engine_reference(ra,st,false,ref.il.p);CK(q36g::launch(p,ca,st));
    CK(cudaStreamSynchronize(st));compare_arms(ref,cand,type,T,write,30+requested,st);++checks;
    cudaGraph_t g{};cudaGraphExec_t e{};CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));CK(q36g::launch(p,ca,st));
    CK(cudaStreamEndCapture(st,&g));CK(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
    cand.reset(f.dih,f.dis,st);CK(cudaGraphLaunch(e,st));CK(cudaStreamSynchronize(st));compare_arms(ref,cand,type,T,write,40+requested,st);++checks;
    CK(cudaGraphExecDestroy(e));CK(cudaGraphDestroy(g));
   }
  }
 }
 std::printf("PASS %d exact GDN/wide-output eager/Graph/window/commit checks (%d against interleaved Q6); sampled CPU dot max_abs=%.9g\n",checks,il_checks,maxerr);std::fflush(stdout);
 if(bench||cold)for(int type:{8,14}){
  Fixture f(type);Arm ref,cand;
  K::native_quantize_q8_1(f.dx.as<float>(),f.dxq.p,MT*N,1,st);CK(cudaStreamSynchronize(st));
  if(bench)benchmark(f,ref,cand,st);else benchmark_cold(f,ref,cand,st,causal);
 }
 CK(cudaStreamDestroy(st));return 0;
}
#endif
