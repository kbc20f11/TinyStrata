#include "strata/intel/q8_xmx.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <algorithm>
#include <stdexcept>

namespace strata::intel {
namespace {
namespace es = sycl::ext::intel::esimd;
template<int N> using U = es::simd<uint32_t, N>;
template<int N> using I = es::simd<int32_t, N>;
template<int N> using F = es::simd<float, N>;
template<class T, int N> SYCL_ESIMD_FUNCTION ESIMD_INLINE es::simd<T,N>
read(const T* p, U<N> offsets, es::simd_mask<N> mask) {
    if constexpr (N <= 32) {
        return es::gather<T,N>(p, offsets, mask, es::simd<T,N>(0), es::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>});
    } else {
        es::simd<T,N> result;
#pragma unroll
        for (int i=0;i<N;i+=32) result.template select<32,1>(i) =
            read<T,32>(p, offsets.template select<32,1>(i).read(), mask.template select<32,1>(i).read());
        return result;
    }
}
template<int N> SYCL_ESIMD_FUNCTION ESIMD_INLINE U<N>
words(const uint8_t* p, U<N> offsets, es::simd_mask<N> mask) {
    auto low=read<uint16_t,N>(reinterpret_cast<const uint16_t*>(p), offsets, mask);
    auto high=read<uint16_t,N>(reinterpret_cast<const uint16_t*>(p), offsets+2, mask);
    return U<N>(low) | (U<N>(high)<<16);
}
template<int N> SYCL_ESIMD_FUNCTION ESIMD_INLINE U<N>
packed_q6_words(const uint32_t* p, U<N> words) {
    return read<uint32_t,N>(p, words*4, es::simd_mask<N>(1));
}
// All weight gathers below enumerate row = SIMD lane % 16. Preserve
// that mapping when reading original blocks or interleaved four-byte words.
template<int N> SYCL_ESIMD_FUNCTION ESIMD_INLINE U<N>
selected_q6_words(const uint32_t* p,U<N> offsets,es::simd_mask<N> mask) {
    return read<uint32_t,N>(p,offsets*4,mask);
}
template<int Type,bool Interleaved> struct WeightView { const uint8_t* pointer; uint32_t row_bytes; };
template<class T,int N,int Type,bool Interleaved> SYCL_ESIMD_FUNCTION ESIMD_INLINE es::simd<T,N>
read(WeightView<Type,Interleaved> view,U<N> offsets,es::simd_mask<N> mask) {
    if constexpr(!Interleaved) return read<T,N>(reinterpret_cast<const T*>(view.pointer),offsets,mask);
    else {
        constexpr uint32_t bytes=Type==14?210:Type==21?110:82,words=(bytes+3)/4;
        U<N> lanes(0,1),row=lanes%16,within=offsets-row*view.row_bytes;
        U<N> block=within/bytes,byte=within%bytes;
        U<N> physical=((block*words+byte/4)*16+row)*4+byte%4;
        return read<T,N>(reinterpret_cast<const T*>(view.pointer),physical,mask);
    }
}

// Offsets are logical block bytes; physical words are interleaved over rows.
// This helper is only used for Interleaved IQ2_S/IQ3_S. Whole words stay inside
// each padded 84/112-byte block, including the final partially occupied word.
template<int Type> SYCL_ESIMD_FUNCTION ESIMD_INLINE U<16>
packed_iq_word(WeightView<Type,true> view,int block,int byte,es::simd_mask<16> mask) {
    constexpr int bytes=Type==21?110:82,words=(bytes+3)/4;
    U<16> nr(0,1),physical=uint32_t((block*words+byte/4)*16*4)+nr*4;
    return read<uint32_t,16>(reinterpret_cast<const uint32_t*>(view.pointer),physical,mask);
}
template<int Type> SYCL_ESIMD_FUNCTION ESIMD_INLINE U<16>
packed_iq_byte(WeightView<Type,true> view,int block,int byte,es::simd_mask<16> mask) {
    return (packed_iq_word<Type>(view,block,byte,mask)>>((byte&3)*8))&255;
}
template<int Count> SYCL_ESIMD_FUNCTION ESIMD_INLINE U<Count> repeat_iq_rows(U<16> values) {
    U<Count> expanded;
#pragma unroll
    for(int first=0;first<Count;first+=16) expanded.template select<16,1>(first)=values;
    return expanded;
}

template<int N,int Type,bool Interleaved> SYCL_ESIMD_FUNCTION ESIMD_INLINE U<N>
words(WeightView<Type,Interleaved> view,U<N> offsets,es::simd_mask<N> mask) {
    auto low=read<uint16_t,N>(view,offsets,mask),high=read<uint16_t,N>(view,offsets+2,mask);
    return U<N>(low)|(U<N>(high)<<16);
}
template<int N> SYCL_ESIMD_FUNCTION ESIMD_INLINE U<N> signed_words(U<N> grid,U<N> signs) {
    U<N> mask=(signs&1)*0xffu | ((signs>>1)&1)*0xff00u |
        ((signs>>2)&1)*0xff0000u | ((signs>>3)&1)*0xff000000u;
    return (grid^mask)+(mask&0x01010101u);
}
template<int Type> constexpr int block_bytes() { return Type==14 ? 210 : Type==21 ? 110 : 82; }
template<int Type,int M=4,bool Interleaved=false,bool PackedIq=false> sycl::event run(sycl::queue& q, const uint8_t* weights,
    const int8_t* input, const float* scales, float* output, int rows, int width,
    ExpertPlan plan, bool separate, const uint32_t* iq2, const uint32_t* iq3) {
    constexpr int N=16,Step=Type==21 ? 32 : 16,Words=Step/4,Count=Words*N;
    const size_t row_bytes=size_t(width/256)*block_bytes<Type>(), row_tiles=(rows+N-1)/N;
    const int bound=std::min(plan.tokens*plan.picks,(plan.tokens*plan.picks+plan.tile_width-1)/plan.tile_width+plan.experts);
    const size_t entries=row_tiles*bound;
    return q.parallel_for(sycl::nd_range<1>((entries+3)/4*4,4),[=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        const size_t entry=it.get_global_linear_id();
        if(entry>=entries)return;
        const size_t tile=entry/row_tiles, first_row=(entry%row_tiles)*N;
        if(tile>=size_t(plan.tiles[0]))return;
        const size_t expert=size_t(plan.owners[tile]);
        const size_t storage_stride=Interleaved ? size_t(width/256)*((block_bytes<Type>()+3)/4*4) : row_bytes;
        const WeightView<Type,Interleaved> w{weights+(expert*rows+first_row)*storage_stride,uint32_t(row_bytes)};
        int input_rows[M]={}, output_slots[M]={};
        int count;
        {
            const int first=plan.offsets[tile], left=plan.counts[expert]-first;
            count=left<M ? left : M;
#pragma unroll
            for(int t=0;t<M;++t)if(t<count){
                const int slot=plan.slots[expert*plan.tokens+first+t];
                output_slots[t]=slot;input_rows[t]=separate ? slot : slot/plan.picks;
            }
        }
        U<N> nr(0,1);const auto row_mask=nr+uint32_t(first_row)<uint32_t(rows);
        U<Count> index(0,1), row_index=index%N, column_word=index/N;
        const auto weight_mask=row_index+uint32_t(first_row)<uint32_t(rows);
        F<M*N> sum(0);
        for(int base=0;base<width;base+=Step){
            const int block=base/256, col=base%256, group=col/32;
            U<Count> row_offset=row_index*uint32_t(row_bytes)+uint32_t(block*block_bytes<Type>());
            U<N> scale_offset=nr*uint32_t(row_bytes)+uint32_t(block*block_bytes<Type>());
            U<Count> packed;
            F<N> dw;
            if constexpr(Type==14){
                U<Count> c=column_word*4+uint32_t(col), segment=c>>7, part=(c&127)>>5, within=c&31;
                U<Count> low=(words<Count>(w,row_offset+segment*64+(part&1)*32+within,weight_mask)>>((part>>1)*4))&0x0f0f0f0fu;
                U<Count> high=(words<Count>(w,row_offset+128+segment*32+within,weight_mask)>>(part*2))&0x03030303u;
                U<Count> value=low|(high<<4);packed=((value|0x80808080u)-0x20202020u)^0x80808080u;
                auto dh=read<uint16_t,N>(w,scale_offset+208,row_mask);
                auto scale=read<int8_t,N>(w,scale_offset+192+uint32_t(col/16),row_mask);
                dw=F<N>(dh.template bit_cast_view<sycl::half>().read())*F<N>(scale);
            }else{
                auto dh=read<uint16_t,N>(w,scale_offset,row_mask);
                F<N> delta(dh.template bit_cast_view<sycl::half>().read());
                U<Count> code,sign,grid;
                if constexpr(Type==21){
                    if constexpr(Interleaved && PackedIq) {
                        const int code_start=2+group*8,sign_start=74+group*4;
                        const U<Count> code_byte=uint32_t(code_start)+column_word;
                        const U<Count> code_part=code_byte/4-uint32_t(code_start/4);
                        U<Count> codes=repeat_iq_rows<Count>(packed_iq_word<Type>(w,block,code_start,row_mask));
                        codes.merge(repeat_iq_rows<Count>(packed_iq_word<Type>(w,block,(code_start/4+1)*4,row_mask)),code_part==1);
                        codes.merge(repeat_iq_rows<Count>(packed_iq_word<Type>(w,block,(code_start/4+2)*4,row_mask)),code_part==2);
                        code=(codes>>((code_byte&3)*8))&255;
                        const auto high=repeat_iq_rows<Count>(packed_iq_byte<Type>(w,block,66+group,row_mask));
                        code|=((high>>column_word)&1)<<8;
                        const U<Count> sign_byte=uint32_t(sign_start)+column_word/2;
                        U<Count> signs=repeat_iq_rows<Count>(packed_iq_word<Type>(w,block,sign_start,row_mask));
                        signs.merge(repeat_iq_rows<Count>(packed_iq_word<Type>(w,block,(sign_start/4+1)*4,row_mask)),
                                    sign_byte/4!=uint32_t(sign_start/4));
                        sign=(signs>>((sign_byte&3)*8+(column_word&1)*4))&15;
                    } else {
                        code=U<Count>(read<uint8_t,Count>(w,row_offset+2+uint32_t(group*8)+column_word,weight_mask));
                        U<Count> high(read<uint8_t,Count>(w,row_offset+66+uint32_t(group),weight_mask));
                        code|=((high>>column_word)&1)<<8;
                        sign=(U<Count>(read<uint8_t,Count>(w,row_offset+74+uint32_t(group*4)+column_word/2,weight_mask))>>((column_word&1)*4))&15;
                    }
                    grid=read<uint32_t,Count>(iq3,code*4,weight_mask);
                    auto scale=read<uint8_t,N>(w,scale_offset+106+uint32_t(group/2),row_mask);
                    dw=delta*F<N>(1+2*((U<N>(scale)>>((group&1)*4))&15));
                }else{
                    U<Count> c=column_word*4+uint32_t(col),octet=(c&31)>>3;
                    if constexpr(Interleaved && PackedIq) {
                        const int first_octet=(col&31)>>3;
                        const U<Count> code_byte=uint32_t(2+group*4)+octet;
                        const auto codes=repeat_iq_rows<Count>(packed_iq_word<Type>(w,block,2+group*4+first_octet,row_mask));
                        code=(codes>>((code_byte&3)*8))&255;
                        const auto high=repeat_iq_rows<Count>(packed_iq_byte<Type>(w,block,66+group,row_mask));
                        code|=((high>>(octet*2))&3)<<8;
                        const U<Count> sign_byte=uint32_t(34+group*4)+octet;
                        const auto signs=repeat_iq_rows<Count>(packed_iq_word<Type>(w,block,34+group*4+first_octet,row_mask));
                        sign=(signs>>((sign_byte&3)*8+(c&7)))&15;
                    } else {
                        code=U<Count>(read<uint8_t,Count>(w,row_offset+2+uint32_t(group*4)+octet,weight_mask));
                        U<Count> high(read<uint8_t,Count>(w,row_offset+66+uint32_t(group),weight_mask));
                        code|=((high>>(octet*2))&3)<<8;
                        sign=(U<Count>(read<uint8_t,Count>(w,row_offset+34+uint32_t(group*4)+octet,weight_mask))>>(c&7))&15;
                    }
                    grid=read<uint32_t,Count>(iq2,(code*2+((c&7)>>2))*4,weight_mask);
                    auto scale=read<uint8_t,N>(w,scale_offset+74+uint32_t(group),row_mask);
                    dw=(F<N>((U<N>(scale)>>((col&16)>>2))&15)+.5f)*(delta*.25f);
                }
                packed=signed_words<Count>(grid,sign);
            }
            I<128> b(0);b.template select<Count,1>(0)=I<Count>(packed);
            I<M*8> a(0);F<M*N> repeated_w,repeated_x;
#pragma unroll
            for(int t=0;t<M;++t){
                if(t<count){
                    a.template select<Words,1>(t*8)=es::block_load<int32_t,Words>(reinterpret_cast<const int32_t*>(input+size_t(input_rows[t])*width+base));
                    repeated_x.template select<N,1>(t*N)=scales[size_t(input_rows[t])*(width/32)+base/32];
                }else repeated_x.template select<N,1>(t*N)=0.f;
                repeated_w.template select<N,1>(t*N)=dw;
            }
            // DPAS consumes row-major A and VNNI B. Its accumulator is first.
            I<M*N> dot(0);
            if constexpr(M==4){
                // Short verification windows often route only one or two
                // rows to an expert. Keep the grouped weight load but avoid
                // doing a four-row DPAS for those partially occupied tiles.
                if(count==1){
                    I<8> small_a=a.template select<8,1>(0).read();
                    dot.template select<N,1>(0)=es::xmx::dpas<8,1,int32_t>(I<N>(0),b.template bit_cast_view<int8_t>().read(),small_a.template bit_cast_view<int8_t>().read());
                }else if(count==2){
                    I<16> small_a=a.template select<16,1>(0).read();
                    dot.template select<2*N,1>(0)=es::xmx::dpas<8,2,int32_t>(I<2*N>(0),b.template bit_cast_view<int8_t>().read(),small_a.template bit_cast_view<int8_t>().read());
                }else dot=es::xmx::dpas<8,M,int32_t>(I<M*N>(0),b.template bit_cast_view<int8_t>().read(),a.template bit_cast_view<int8_t>().read());
            }else dot=es::xmx::dpas<8,M,int32_t>(I<M*N>(0),b.template bit_cast_view<int8_t>().read(),a.template bit_cast_view<int8_t>().read());
            sum+=F<M*N>(dot)*repeated_w*repeated_x;
        }
#pragma unroll
        for(int t=0;t<M;++t)if(t<count)
            es::scatter<float,N>(output+size_t(output_slots[t])*rows+first_row,nr*4,sum.template select<N,1>(t*N).read(),row_mask);
    });
}
// Only short interleaved IQ2/IQ3 verification windows use packed gathers.
// Keep T1, large prefill, Q6 and original-layout products on their old paths.
template<int Type,int M> sycl::event dispatch_expert_product(sycl::queue& q,const uint8_t* weights,
    const int8_t* input,const float* scales,float* output,int rows,int width,ExpertPlan plan,
    bool separate,const uint32_t* iq2,const uint32_t* iq3,bool interleaved) {
    if(interleaved) {
        if constexpr(Type==21 || Type==22) {
            if(plan.tokens>=2 && plan.tokens<=8)
                return run<Type,M,true,true>(q,weights,input,scales,output,rows,width,plan,separate,iq2,iq3);
        }
        return run<Type,M,true,false>(q,weights,input,scales,output,rows,width,plan,separate,iq2,iq3);
    }
    return run<Type,M,false,false>(q,weights,input,scales,output,rows,width,plan,separate,iq2,iq3);
}
}
sycl::event q8_xmx_product(sycl::queue& q,int type,const uint8_t* weights,const int8_t* input,
    const float* scales,float* output,int rows,int width,ExpertPlan plan,bool separate,const uint32_t* iq2,const uint32_t* iq3,bool interleaved){
    if(!q.has_property<sycl::property::queue::in_order>()||rows<1||width<256||width%256||plan.tokens<1||plan.tokens>1024||
        plan.experts<1||plan.experts>256||(plan.tile_width!=4&&plan.tile_width!=1)||plan.picks<1||plan.picks>8)throw std::invalid_argument("Invalid INT8 XMX product geometry");
    if(interleaved && rows%16) throw std::invalid_argument("Interleaved experts need complete 16-row tiles");
#define PRODUCT(T,M) dispatch_expert_product<T,M>(q,weights,input,scales,output,rows,width,plan,separate,iq2,iq3,interleaved)
#define RUN(T) (plan.tile_width==1 ? PRODUCT(T,1) : PRODUCT(T,4))
    switch(type){case 14:return RUN(14);case 21:return RUN(21);case 22:return RUN(22);default:throw std::invalid_argument("Unsupported INT8 XMX GGUF type");}
#undef RUN
#undef PRODUCT
}
template<int M> sycl::event interleaved_q6_impl(sycl::queue& q,const uint8_t* source,const int8_t* x,const float* dx,float* y,int rows,int width,int tokens){
    if (!q.has_property<sycl::property::queue::in_order>() || rows < 1 || rows % 16 || width < 256 || width > 8192 || width % 256 || tokens < 1 || tokens > 8)
        throw std::invalid_argument("Invalid interleaved Q6 XMX geometry");
    const auto* weights=reinterpret_cast<const uint32_t*>(source);
    const int blocks=width/256,row_tiles=rows/16;
    const size_t entries=size_t(row_tiles)*((tokens+M-1)/M);
    return q.parallel_for(sycl::nd_range<1>((entries+3)/4*4,4),[=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        if(it.get_global_linear_id()>=entries)return;
        const size_t tile=it.get_global_linear_id(),first_row=(tile%row_tiles)*16,first_token=(tile/row_tiles)*M;
        const auto* w=weights+(tile%row_tiles)*blocks*53*16;
        U<64> index(0,1),row=index%16,colword=index/16;
        U<16> nr(0,1);F<M*16> sum(0);
        for(int base=0;base<width;base+=16){
            const int block=base/256,col=base%256;
            U<64> c=colword*4+uint32_t(col),segment=c>>7,part=(c&127)>>5,within=c&31;
            U<64> lowbyte=segment*64+(part&1)*32+within,highbyte=128+segment*32+within;
            U<64> low=(packed_q6_words<64>(w,uint32_t(block*53*16)+(lowbyte/4)*16+row)>>((part>>1)*4))&0x0f0f0f0fu;
            U<64> high=(packed_q6_words<64>(w,uint32_t(block*53*16)+(highbyte/4)*16+row)>>(part*2))&0x03030303u;
            U<64> value=low|(high<<4),packed=((value|0x80808080u)-0x20202020u)^0x80808080u;
            auto dword=packed_q6_words<16>(w,uint32_t((block*53+52)*16)+nr);
            es::simd<uint16_t,16> dh=dword;
            const int scale_byte=192+col/16;
            auto sword=packed_q6_words<16>(w,uint32_t((block*53+scale_byte/4)*16)+nr);
            es::simd<int8_t,16> sc=(sword>>((scale_byte%4)*8))&255;
            F<16> dw=F<16>(dh.template bit_cast_view<sycl::half>().read())*F<16>(sc);
            I<128> b(0);b.template select<64,1>(0)=I<64>(packed);
            I<M*8> a(0);F<M*16> factors;
            for(int t=0;t<M;++t){
                if(first_token+t<size_t(tokens)){
                    a.template select<4,1>(t*8)=es::block_load<int32_t,4>(reinterpret_cast<const int32_t*>(x+(first_token+t)*width+base));
                    factors.template select<16,1>(t*16)=dw*dx[(first_token+t)*(width/32)+base/32];
                }else factors.template select<16,1>(t*16)=0.f;
            }
            auto dot=es::xmx::dpas<8,M,int32_t>(I<M*16>(0),b.template bit_cast_view<int8_t>().read(),a.template bit_cast_view<int8_t>().read());
            sum+=F<M*16>(dot)*factors;
        }
        for(int t=0;t<M;++t)if(first_token+t<size_t(tokens))
            es::scatter<float,16>(y+(first_token+t)*rows+first_row,nr*4,sum.template select<16,1>(t*16).read());
    });
}
template<int M> sycl::event split_q6_impl(sycl::queue& q,const uint8_t* source,const int8_t* x,const float* dx,float* y,int rows,int width,int tokens,float* partial,sycl::event* first_stage){
    if (!q.has_property<sycl::property::queue::in_order>() || rows < 1 || rows % 16 || width < 256 || width > 8192 || width % 256 || tokens < 1 || tokens > 8)
        throw std::invalid_argument("Invalid interleaved Q6 XMX geometry");
    constexpr int partitions=4;
    if(rows>8192 || !partial)throw std::invalid_argument("Split-K Q6 exceeds bounded partial scratch");
    const auto* weights=reinterpret_cast<const uint32_t*>(source);
    const int blocks=width/256,row_tiles=rows/16;
    const size_t token_tiles=(tokens+M-1)/M,entries=size_t(row_tiles)*token_tiles*partitions;
    auto dot=q.parallel_for(sycl::nd_range<1>((entries+3)/4*4,4),[=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        if(it.get_global_linear_id()>=entries)return;
        const size_t tile=it.get_global_linear_id(),first_row=(tile%row_tiles)*16,first_token=((tile/row_tiles)%token_tiles)*M,partition=tile/(row_tiles*token_tiles);
        const auto* w=weights+(tile%row_tiles)*blocks*53*16;
        U<64> index(0,1),row=index%16,colword=index/16;
        U<16> nr(0,1);F<M*16> sum(0);
        const int span=width/partitions;
        for(int base=int(partition)*span;base<(int(partition)+1)*span;base+=16){
            const int block=base/256,col=base%256;
            U<64> c=colword*4+uint32_t(col),segment=c>>7,part=(c&127)>>5,within=c&31;
            U<64> lowbyte=segment*64+(part&1)*32+within,highbyte=128+segment*32+within;
            U<64> low=(packed_q6_words<64>(w,uint32_t(block*53*16)+(lowbyte/4)*16+row)>>((part>>1)*4))&0x0f0f0f0fu;
            U<64> high=(packed_q6_words<64>(w,uint32_t(block*53*16)+(highbyte/4)*16+row)>>(part*2))&0x03030303u;
            U<64> value=low|(high<<4),packed=((value|0x80808080u)-0x20202020u)^0x80808080u;
            auto dword=packed_q6_words<16>(w,uint32_t((block*53+52)*16)+nr);
            es::simd<uint16_t,16> dh=dword;
            const int scale_byte=192+col/16;
            auto sword=packed_q6_words<16>(w,uint32_t((block*53+scale_byte/4)*16)+nr);
            es::simd<int8_t,16> sc=(sword>>((scale_byte%4)*8))&255;
            F<16> dw=F<16>(dh.template bit_cast_view<sycl::half>().read())*F<16>(sc);
            I<128> b(0);b.template select<64,1>(0)=I<64>(packed);
            I<M*8> a(0);F<M*16> factors;
            for(int t=0;t<M;++t){
                if(first_token+t<size_t(tokens)){
                    a.template select<4,1>(t*8)=es::block_load<int32_t,4>(reinterpret_cast<const int32_t*>(x+(first_token+t)*width+base));
                    factors.template select<16,1>(t*16)=dw*dx[(first_token+t)*(width/32)+base/32];
                }else factors.template select<16,1>(t*16)=0.f;
            }
            auto dot=es::xmx::dpas<8,M,int32_t>(I<M*16>(0),b.template bit_cast_view<int8_t>().read(),a.template bit_cast_view<int8_t>().read());
            sum+=F<M*16>(dot)*factors;
        }
        for(int t=0;t<M;++t)if(first_token+t<size_t(tokens))
            es::scatter<float,16>(partial+(partition*tokens+first_token+t)*rows+first_row,nr*4,sum.template select<16,1>(t*16).read());
    });
    if(first_stage)*first_stage=dot;
    return q.parallel_for(sycl::range<1>(size_t(tokens)*rows),[=](sycl::id<1> id){float sum=0;for(int p=0;p<partitions;++p)sum+=partial[size_t(p)*tokens*rows+id[0]];y[id[0]]=sum;});
}

sycl::event q6_interleaved_splitk_product(sycl::queue& q,const uint8_t* weights,const int8_t* input,
    const float* scales,float* output,int rows,int width,int tokens,float* partial,sycl::event* compute_stage) {
    if(tokens==1)return split_q6_impl<1>(q,weights,input,scales,output,rows,width,tokens,partial,compute_stage);
    if(tokens==2)return split_q6_impl<2>(q,weights,input,scales,output,rows,width,tokens,partial,compute_stage);
    if(tokens<=4)return split_q6_impl<4>(q,weights,input,scales,output,rows,width,tokens,partial,compute_stage);
    return split_q6_impl<8>(q,weights,input,scales,output,rows,width,tokens,partial,compute_stage);
}

sycl::event q6_interleaved_selected_product(sycl::queue& q,const uint8_t* source,const int8_t* input,const float* scales,
    const int32_t* selected_ids,float* output,int vocab_rows,int width,int selected_count){
    if(!q.has_property<sycl::property::queue::in_order>()||!source||!input||!scales||!selected_ids||!output||
        vocab_rows<16||vocab_rows%16||width!=2048||selected_count<1||selected_count>4096||selected_count>vocab_rows)
        throw std::invalid_argument("Invalid selected draft Q6 vocabulary geometry");
    const auto* weights=reinterpret_cast<const uint32_t*>(source);
    const int blocks=width/256;
    const size_t tiles=(size_t(selected_count)+15)/16;
    return q.parallel_for(sycl::nd_range<1>((tiles+3)/4*4,4),[=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        const size_t tile=it.get_global_linear_id();if(tile>=tiles)return;
        U<16> nr(0,1),logical=uint32_t(tile*16)+nr;
        auto active=logical<uint32_t(selected_count);
        I<16> real=read<int32_t,16>(selected_ids,logical*4,active);
        active=active&(real>=0)&(real<vocab_rows);
        U<16> rows(real),physical_base=(rows/16)*uint32_t(blocks*53*16)+(rows%16);
        U<64> index(0,1),lane=index%16,colword=index/16;
        U<64> physical64=physical_base.iselect(es::simd<uint16_t,64>(lane));
        es::simd_mask<64> active64=es::simd_mask<64>(logical.iselect(es::simd<uint16_t,64>(lane))<uint32_t(selected_count));
        active64=active64&(U<64>(rows.iselect(es::simd<uint16_t,64>(lane)))<uint32_t(vocab_rows));
        F<16> sum(0);
        for(int base=0;base<width;base+=16){
            const int block=base/256,col=base%256;
            U<64> c=colword*4+uint32_t(col),segment=c>>7,part=(c&127)>>5,within=c&31;
            U<64> lowbyte=segment*64+(part&1)*32+within,highbyte=128+segment*32+within;
            U<64> low=(selected_q6_words<64>(weights,physical64+uint32_t(block*53*16)+(lowbyte/4)*16,active64)>>((part>>1)*4))&0x0f0f0f0fu;
            U<64> high=(selected_q6_words<64>(weights,physical64+uint32_t(block*53*16)+(highbyte/4)*16,active64)>>(part*2))&0x03030303u;
            U<64> value=low|(high<<4),packed=((value|0x80808080u)-0x20202020u)^0x80808080u;
            auto dword=selected_q6_words<16>(weights,physical_base+uint32_t((block*53+52)*16),active);
            es::simd<uint16_t,16> delta=dword;
            const int scale_byte=192+col/16;
            auto sword=selected_q6_words<16>(weights,physical_base+uint32_t((block*53+scale_byte/4)*16),active);
            es::simd<int8_t,16> scale=(sword>>((scale_byte%4)*8))&255;
            F<16> dw=F<16>(delta.template bit_cast_view<sycl::half>().read())*F<16>(scale);
            I<128> b(0);b.template select<64,1>(0)=I<64>(packed);
            I<8> a(0);a.template select<4,1>(0)=es::block_load<int32_t,4>(reinterpret_cast<const int32_t*>(input+base));
            auto dot=es::xmx::dpas<8,1,int32_t>(I<16>(0),b.template bit_cast_view<int8_t>().read(),a.template bit_cast_view<int8_t>().read());
            sum+=F<16>(dot)*(dw*scales[base/32]);
        }
        es::scatter<float,16>(output+tile*16,nr*4,sum,active);
    });
}

sycl::event q6_interleaved_product(sycl::queue& q,const uint8_t* weights,const int8_t* input,
    const float* scales,float* output,int rows,int width,int tokens) {
    if(tokens==1)return interleaved_q6_impl<1>(q,weights,input,scales,output,rows,width,tokens);
    if(tokens==2)return interleaved_q6_impl<2>(q,weights,input,scales,output,rows,width,tokens);
    if(tokens<=4)return interleaved_q6_impl<4>(q,weights,input,scales,output,rows,width,tokens);
    return interleaved_q6_impl<8>(q,weights,input,scales,output,rows,width,tokens);
}

}
