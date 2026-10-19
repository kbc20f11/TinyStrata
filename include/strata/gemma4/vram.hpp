// Dedicated VRAM accounting shared with TinyStrata Qwen3.6 placement.
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#if defined(_WIN32) && !defined(STRATA_USE_HIP) && !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace strata::gemma4 {
// WDDM can satisfy CUDA allocations with pageable GPU memory. cudaMemGetInfo is
// still an allocation estimate, but is not a promise that the whole allocation
// will remain in dedicated VRAM. Automatic expert placement needs the latter:
// bound the estimate by the active adapter's device-wide dedicated-memory free
// count. NVML is supplied by the NVIDIA driver; neither its SDK nor an import
// library is required. An unavailable query keeps the existing CUDA estimate.
#if defined(_WIN32) && !defined(STRATA_USE_HIP) && !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
struct ResidentVram {
    struct Memory { unsigned long long total, free, used; };
    using Init = int (*)();
    using Shutdown = int (*)();
    using HandleByPci = int (*)(const char*, void**);
    using MemoryInfo = int (*)(void*, Memory*);
    HMODULE module = nullptr;
    Shutdown shutdown = nullptr;
    HandleByPci handle_by_pci = nullptr;
    MemoryInfo memory_info = nullptr;
    bool initialized = false;

    ResidentVram() {
        // Never search the working/model directory or PATH for a driver DLL.
        module = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (module == nullptr) return;
        const auto init = reinterpret_cast<Init>(GetProcAddress(module, "nvmlInit_v2"));
        shutdown = reinterpret_cast<Shutdown>(GetProcAddress(module, "nvmlShutdown"));
        handle_by_pci = reinterpret_cast<HandleByPci>(GetProcAddress(module, "nvmlDeviceGetHandleByPciBusId_v2"));
        memory_info = reinterpret_cast<MemoryInfo>(GetProcAddress(module, "nvmlDeviceGetMemoryInfo"));
        if (init && shutdown && handle_by_pci && memory_info) initialized = init() == 0;
    }
    ~ResidentVram() {
        if (initialized) shutdown();
        if (module) FreeLibrary(module);
    }
    bool query(size_t& free_bytes) const {
        if (!initialized) return false;
        int device = 0;
        char pci[32] = {};
        if (cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetPCIBusId(pci, sizeof pci, device) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        // NVML ordinals need not match CUDA ordinals (e.g. CUDA_VISIBLE_DEVICES).
        void* nvml_device = nullptr;
        Memory memory{};
        if (handle_by_pci(pci, &nvml_device) != 0 || nvml_device == nullptr ||
            memory_info(nvml_device, &memory) != 0 || memory.total == 0 || memory.free > memory.total)
            return false;
        free_bytes = (size_t) memory.free;
        return true;
    }
};
inline ResidentVram& resident_vram() {
    static ResidentVram resident;
    return resident;
}
#endif

inline cudaError_t resident_mem_info(size_t* free_bytes, size_t* total_bytes) {
    const cudaError_t status = cudaMemGetInfo(free_bytes, total_bytes);
#if defined(_WIN32) && !defined(STRATA_USE_HIP) && !defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
    if (status == cudaSuccess && free_bytes != nullptr && total_bytes != nullptr) {
        size_t physical = 0;
        const size_t raw = *free_bytes;
        const bool known = resident_vram().query(physical);
        if (known) *free_bytes = std::min(raw, physical);
        if (std::getenv("STRATA_LFM_PLACE_DEBUG")) {
            if (known)
                std::fprintf(stderr, "tinystrata-lfm: resident VRAM budget: CUDA %.2f GiB, physical %.2f GiB, using %.2f GiB\n",
                             raw / 1073741824.0, physical / 1073741824.0, *free_bytes / 1073741824.0);
            else
                std::fprintf(stderr, "tinystrata-lfm: resident VRAM budget: NVML unavailable, using CUDA %.2f GiB\n",
                             raw / 1073741824.0);
        }
    }
#endif
    return status;
}
} // namespace strata::gemma4
