#pragma once
// CUDA virtual-memory KV growth, following Strata's core/vmm.hpp and vmm.cpp
// (Niko1221 and the Strata contributors, MIT; see LICENSE and NOTICE).
// Virtual addresses and logical head strides never move. Only physical pages
// behind each head's used prefix are committed; expert arenas may return pages.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::qwen35::kv_grow {
struct Api {
    bool ok = false;
    size_t gran = 0;
    int device = -1;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
    template<class F> static bool resolve(const char* name, F& f) {
        cudaDriverEntryPointQueryResult q{};
        void* p = nullptr;
#if CUDART_VERSION >= 12050
        const auto e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else
        const auto e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
        if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || !p) {
            (void) cudaGetLastError();
            return false;
        }
        f = reinterpret_cast<F>(p);
        return true;
    }
    Api() {
#if !defined(__HIPCC__)
        decltype(&cuCtxGetDevice) current_device = nullptr;
        decltype(&cuDeviceGetAttribute) attr = nullptr;
        decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
        // The model has already initialized the current runtime context. Driver
        // allocations must use its CUdevice, rather than a runtime ordinal.
        if (!resolve("cuCtxGetDevice", current_device) || current_device(&device) != CUDA_SUCCESS ||
            !resolve("cuDeviceGetAttribute", attr) ||
            !resolve("cuMemGetAllocationGranularity", granularity) || !resolve("cuMemAddressReserve", reserve) ||
            !resolve("cuMemAddressFree", free_va) || !resolve("cuMemCreate", create) ||
            !resolve("cuMemRelease", release) || !resolve("cuMemMap", map) ||
            !resolve("cuMemUnmap", unmap) || !resolve("cuMemSetAccess", access)) return;
        int supports = 0;
        if (attr(&supports, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device) != CUDA_SUCCESS || !supports)
            return;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        ok = granularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS && gran != 0;
#endif
    }
};
inline const Api& api() { static const Api a; return a; }

class Range {
    CUdeviceptr base_ = 0;
    std::vector<CUmemGenericAllocationHandle> pages_;
public:
    Range() = default;
    Range(const Range&) = delete;
    Range& operator=(const Range&) = delete;
    ~Range() {
        if (!base_) return;
        // A partially initialized range may still have queued zeroing/copies.
        // This also covers local ranges destroyed before Engine cleanup.
        (void) cudaDeviceSynchronize();
        const auto& a = api();
        for (size_t i = 0; i < pages_.size(); ++i) if (pages_[i]) {
            a.unmap(base_ + i * a.gran, a.gran);
            a.release(pages_[i]);
        }
        a.free_va(base_, pages_.size() * a.gran);
    }
    bool reserve(size_t bytes) {
        const auto& a = api();
        if (!a.ok || base_) return false;
        const size_t n = (bytes + a.gran - 1) / a.gran;
        if (!n || a.reserve(&base_, n * a.gran, 0, 0, 0) != CUDA_SUCCESS) return false;
        pages_.resize(n, 0);
        return true;
    }
    void* base() const { return reinterpret_cast<void*>(base_); }
    size_t mapped() const {
        size_t n = 0;
        for (auto h : pages_) n += h != 0;
        return n * api().gran;
    }
    // Prefixes can share a page at a head boundary when ctx is not aligned.
    // A union mask avoids double mapping or unmapping such boundary pages.
    std::vector<uint8_t> mask(size_t rows, size_t heads, size_t stride, size_t row_bytes) const {
        const size_t g = api().gran;
        std::vector<uint8_t> out(pages_.size(), 0);
        if (!rows) return out;
        for (size_t h = 0; h < heads; ++h) {
            const size_t lo = h * stride / g;
            const size_t hi = (h * stride + rows * row_bytes + g - 1) / g;
            for (size_t i = lo; i < hi && i < out.size(); ++i) out[i] = 1;
        }
        return out;
    }
    size_t need(const std::vector<uint8_t>& want) const {
        size_t n = 0;
        for (size_t i = 0; i < pages_.size(); ++i) n += want[i] && !pages_[i];
        return n * api().gran;
    }
    bool map(const std::vector<uint8_t>& want, cudaStream_t stream) {
        const auto& a = api();
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = a.device;
        CUmemAccessDesc access{};
        access.location = prop.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        for (size_t i = 0; i < pages_.size(); ++i) if (want[i] && !pages_[i]) {
            CUmemGenericAllocationHandle h = 0;
            if (a.create(&h, a.gran, &prop, 0) != CUDA_SUCCESS) return false;
            const CUdeviceptr p = base_ + i * a.gran;
            if (a.map(p, a.gran, 0, h, 0) != CUDA_SUCCESS) { a.release(h); return false; }
            if (a.access(p, a.gran, &access, 1) != CUDA_SUCCESS) {
                a.unmap(p, a.gran); a.release(h); return false;
            }
            pages_[i] = h;
            if (cudaMemsetAsync(reinterpret_cast<void*>(p), 0, a.gran, stream) != cudaSuccess) return false;
        }
        return true;
    }
    // The caller must have drained every stream that could access these pages.
    bool trim(const std::vector<uint8_t>& want) {
        const auto& a = api();
        for (size_t i = 0; i < pages_.size(); ++i) if (!want[i] && pages_[i]) {
            if (a.unmap(base_ + i * a.gran, a.gran) != CUDA_SUCCESS) return false;
            a.release(pages_[i]);
            pages_[i] = 0;
        }
        return true;
    }
    std::vector<uint8_t> prefix(size_t bytes) const { return mask(bytes, 1, 0, 1); }
};
} // namespace strata::qwen35::kv_grow
