#include "strata/intel/device.hpp"
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <level_zero/ze_api.h>
#include <windows.h>
#include <dxgi1_4.h>
#include <psapi.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::intel {
using Microsoft::WRL::ComPtr;
sycl::device select_device(unsigned index) {
    std::vector<sycl::device> devices;
    for (const auto& platform : sycl::platform::get_platforms()) {
        if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : platform.get_devices(sycl::info::device_type::gpu)) {
            if (d.get_info<sycl::info::device::vendor_id>() == 0x8086) devices.push_back(d);
        }
    }
    if (index >= devices.size()) throw std::runtime_error("Requested Intel Level Zero GPU is unavailable");
    return devices[index];
}
DeviceInfo inspect_device(const sycl::device& d) {
    if (!d.is_gpu() || d.get_backend() != sycl::backend::ext_oneapi_level_zero)
        throw std::runtime_error("An Intel Level Zero GPU is required");
    ze_device_properties_t properties{};
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    if (zeDeviceGetProperties(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d), &properties) != ZE_RESULT_SUCCESS)
        throw std::runtime_error("zeDeviceGetProperties failed");
    DeviceInfo out;
    out.vendor_id = properties.vendorId;
    out.device_id = properties.deviceId;
    out.integrated = (properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0;
#ifdef ZE_DEVICE_LUID_EXT_NAME
    // This optional query must not make inference depend on telemetry support.
    // The returned LUID is the identity of the actual selected Level Zero GPU,
    // not a guess from Windows display order or matching vendor/device IDs.
    try {
        const auto driver = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d.get_platform());
        uint32_t count = 0;
        if (zeDriverGetExtensionProperties(driver, &count, nullptr) == ZE_RESULT_SUCCESS && count <= 1024) {
            std::vector<ze_driver_extension_properties_t> extensions(count);
            if (zeDriverGetExtensionProperties(driver, &count, extensions.data()) == ZE_RESULT_SUCCESS) {
                for (const auto& extension : extensions) {
                    if (std::strcmp(extension.name, ZE_DEVICE_LUID_EXT_NAME) != 0) continue;
                    ze_device_luid_ext_properties_t identity{};
                    identity.stype = ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES;
                    properties.pNext = &identity;
                    if (zeDeviceGetProperties(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d), &properties) == ZE_RESULT_SUCCESS &&
                        identity.nodeMask && !(identity.nodeMask & (identity.nodeMask - 1))) {
                        LUID luid{};
                        static_assert(sizeof(luid) == sizeof(identity.luid.id));
                        std::memcpy(&luid, identity.luid.id, sizeof(luid));
                        if (luid.HighPart || luid.LowPart) {
                            char text[18];
                            std::snprintf(text, sizeof(text), "%08x:%08x", unsigned(luid.HighPart), unsigned(luid.LowPart));
                            out.adapter_luid = text;
                            out.adapter_node_mask = identity.nodeMask;
                        }
                    }
                    break;
                }
            }
        }
    } catch (const sycl::exception&) {
        // Unsupported native identity stays unknown, never another adapter.
    }
#endif
    out.name = d.get_info<sycl::info::device::name>();
    out.driver = d.get_info<sycl::info::device::driver_version>();
    // Query the runtime architecture. A B390 marketing name must not receive
    // Battlemage tuning merely because it contains the letter B.
    try {
        namespace exp = sycl::ext::oneapi::experimental;
        const auto architecture = d.get_info<exp::info::device::architecture>();
        if (architecture == exp::architecture::intel_gpu_ptl_h || architecture == exp::architecture::intel_gpu_ptl_u)
            out.architecture = "panther_lake_xe3";
        else if (architecture == exp::architecture::intel_gpu_bmg_g21 || architecture == exp::architecture::intel_gpu_bmg_g31)
            out.architecture = "battlemage_xe2";
        else if (architecture == exp::architecture::intel_gpu_acm_g10 || architecture == exp::architecture::intel_gpu_acm_g11 ||
                 architecture == exp::architecture::intel_gpu_acm_g12)
            out.architecture = "alchemist_xe_hpg";
    } catch (const sycl::exception&) {
        // An older runtime may not expose this extension. No name heuristic.
    }
    out.reported_global_memory = d.get_info<sycl::info::device::global_mem_size>();
    out.max_allocation = d.get_info<sycl::info::device::max_mem_alloc_size>();
    if (out.vendor_id != 0x8086) throw std::runtime_error("The selected GPU is not Intel");
    return out;
}
MemoryBudget memory_budget(const DeviceInfo& device) {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) throw std::runtime_error("GlobalMemoryStatusEx failed");
    MemoryBudget out{status.ullAvailPhys, status.ullAvailPageFile, 0, false};
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return out;
    ComPtr<IDXGIAdapter3> selected;
    unsigned matches = 0;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(result)) return out;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) continue;
        if (desc.VendorId == device.vendor_id && desc.DeviceId == device.device_id) {
            ++matches;
            adapter.As(&selected);
        }
    }
    // A PCI device ID is not a unique adapter identity on multi-card systems.
    // Refuse to borrow another card's budget when the match is ambiguous.
    if (matches != 1 || !selected) return out;
    DXGI_QUERY_VIDEO_MEMORY_INFO video{};
    if (SUCCEEDED(selected->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &video))) {
        out.wddm_known = true;
        out.wddm_available = video.Budget > video.CurrentUsage ? video.Budget - video.CurrentUsage : 0;
    }
    return out;
}
void print_memory(const char* stage, const DeviceInfo& d) {
    const auto m = memory_budget(d);
    PROCESS_MEMORY_COUNTERS_EX process{};
    process.cb = sizeof(process);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&process), sizeof(process)))
        throw std::runtime_error("GetProcessMemoryInfo failed");
    std::printf("memory stage=%s available_physical=%llu available_commit=%llu wddm_known=%d wddm_available=%llu allocation_budget=%llu working_set=%llu private_bytes=%llu\n",
        stage, (unsigned long long)m.available_physical, (unsigned long long)m.available_commit,
        int(m.wddm_known), (unsigned long long)m.wddm_available, (unsigned long long)allocation_budget(m),
        (unsigned long long)process.WorkingSetSize, (unsigned long long)process.PrivateUsage);
    std::fflush(stdout);
}
} // namespace strata::intel
