#pragma once
#include <string>

namespace strata::kernels::cpu {
/// Plan v0.3 P6: whether this CPU (and its OS) runs the AVX-512 kernels (F, BW, VL, VNNI, VBMI).  Probed in a
/// file compiled without AVX-512, so asking is safe everywhere; STRATA_FORCE_AVX2=1 answers no (for tests).
bool cpu_avx512_ok();
/// Whether this CPU runs AVX-512F + AVX-512BW + AVX-512VL + AVX-512DQ (e.g. Skylake-X i9-9920X) for iq_avx512.cpp.
bool cpu_avx512bw_ok();
/// Whether this CPU (and its OS) runs the AVX2 kernels (AVX, AVX2, FMA, F16C): the floor of every expert kernel
/// (iq_avx2.cpp and ggml-cpu in the portable build).  STRATA_FORCE_AVX2 does not change it.
bool cpu_avx2_ok();
/// Whether this CPU (and its OS) runs AVX (Sandy Bridge and newer): the AVX1 router dot (kq_avx1.cpp) and ggml-cpu in
/// a STRATA_ISA_FLOOR=avx build.  The other expert rows then run on ggml-cpu (experimental, slow).
bool cpu_avx1_ok();
/// SSE4.2 + POPCNT: ggml-cpu in a STRATA_ISA_FLOOR=none build.
bool cpu_sse42_ok();
/// STRATA_FORCE_ISA=avx2|avx|sse caps the probes above (3 = no cap, 2 = AVX2, 1 = AVX, 0 = SSE4.2), so the older-CPU
/// paths run on any PC: tests and the experimental builds' checks.  ggml-cpu keeps what it was compiled for.
int cpu_isa_cap();
/// The experimental older-CPU build this engine is ("avx", "sse4.2"), or "" for the normal build (AVX2 floor).
const char* isa_floor_build();
/// STRATA_IQ256_GATHER, the AVX-2 i-quant kernels' gathered grid decode (iq_avx2.cpp): -1 unset (auto, see
/// cpu_gather_fast_here), 0 the scalar decode on every core, 1 the gathered one on every core.
int iq256_gather_setting();
/// Whether this CPU's performance cores gather the IQ grid entries faster than they assemble them from scalar loads:
/// an Intel CPU with AVX-VNNI, i.e. Alder Lake / Sapphire Rapids or newer, and not one of the E-core-only parts.
/// The older Intel cores gather slowly (Haswell, Broadwell) or under the Downfall (GDS) microcode (Skylake to Tiger
/// Lake), AMD Zen 2/3 gather slowly, and Zen 4/5 run the AVX-512 kernels.  STRATA_FORCE_ISA answers no.
bool cpu_gather_fast();
/// cpu_gather_fast() and the CALLING THREAD runs on a performance core: CPUID 1Ah core type 40h on a hybrid CPU (its
/// E-cores, 20h, gather slower than they assemble).  Probed once per thread: the pool pins each worker to one core.
bool cpu_gather_fast_here();
/// Whether the AVX-2 expert kernels take their AVX-VNNI forms (vpdpwssd / vpdpbusd: Alder Lake, Sapphire Rapids and
/// later, P- and E-cores alike; the same integer sums): cpu_avx2_ok() and CPUID 7.1:EAX[4], not capped by
/// STRATA_FORCE_ISA, and not an older-CPU build (STRATA_ISA_FLOOR, which stays on its floor and AVX2).
/// STRATA_NO_AVXVNNI=1 answers no.
bool cpu_avxvnni_ok();
std::string cpu_name();
}  // namespace strata::kernels::cpu
