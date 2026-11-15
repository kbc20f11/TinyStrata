// CPU interpretation of the new Q4_K decoder against independently generated
// gguf/NumPy rows. No GPU/device/model is opened; only a bounded fixture is read.
#include "strata/intel/quant.hpp"
#include "strata/common/utf8_args.hpp"
#include "strata/common/utf8_path.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#include <stdexcept>
template<class T>T read(std::ifstream& in){T v{};if(!in.read(reinterpret_cast<char*>(&v),sizeof v))throw std::runtime_error("Truncated NumPy fixture");return v;}
template<class T>std::vector<T> array(std::ifstream& in,size_t n){if(n>32*1024*1024)throw std::runtime_error("Oversized fixture");std::vector<T> v(n);if(!in.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*sizeof(T))))throw std::runtime_error("Truncated fixture array");return v;}
int main(int argc,char** argv){try{
    strata::paths::Utf8Arguments utf8(argc,argv);argc=utf8.argc();argv=utf8.argv();
    if(argc!=2)throw std::runtime_error("Usage: intel_lfm_quant_test NUMPY_REFERENCE.bin");
    std::ifstream in(strata::paths::native_path(argv[1]),std::ios::binary);if(!in)throw std::runtime_error("Missing reference");
    const auto cases=read<uint32_t>(in);if(!cases||cases>64)throw std::runtime_error("Invalid case count");
    size_t checked=0;int used=0;double maximum=0;
    for(uint32_t i=0;i<cases;++i){const auto type=read<int32_t>(in),rows=read<int32_t>(in),width=read<int32_t>(in);const auto bytes=read<uint64_t>(in);
        if(rows<1||rows>64||width<1||width>16384||bytes>32*1024*1024)throw std::runtime_error("Invalid matrix geometry");
        const auto raw=array<uint8_t>(in,size_t(bytes));const auto x=array<float>(in,width);
        const auto expected=array<float>(in,size_t(rows)*width);const auto dots=array<double>(in,rows);
        if(type!=12)continue;if(width%256||bytes!=uint64_t(rows)*width/256*144)throw std::runtime_error("Invalid Q4_K block extent");
        for(int row=0;row<rows;++row){double dot=0;
            for(int col=0;col<width;++col){const size_t index=size_t(row)*width+col;
                const float actual=strata::intel::quant::value<12>(raw.data(),index);const double error=std::abs(double(actual)-expected[index]);
                if(!std::isfinite(actual)||error>2e-6*(1+std::abs(double(expected[index]))))throw std::runtime_error("Q4_K scale/minimum/nibble differs from NumPy");
                dot+=double(actual)*x[col];maximum=std::max(maximum,error);++checked;
            }
            if(std::abs(dot-dots[row])>5e-5*(1+std::abs(dots[row])))throw std::runtime_error("Q4_K independent FP64 dot differs");
        }++used;
    }
    if(!used||in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Missing Q4_K cases or trailing data");
    std::printf("PASS Q4_K CPU vs NumPy cases=%d values=%zu max_abs_error=%.9g; no GPU/model opened\n",used,checked,maximum);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"LFM quant fixture: %s\n",e.what());return 1;}}
