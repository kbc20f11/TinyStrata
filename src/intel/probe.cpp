// Issue #11: native Windows allocation/visibility/addressing foundation.
// This executable is deliberately not an inference engine.
#include "strata/intel/device.hpp"
#include <chrono>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace strata::intel;
struct Allocation {
    sycl::queue& queue;
    void* pointer = nullptr;
    ~Allocation() { if (pointer) sycl::free(pointer, queue); }
};
uint64_t positive_integer(const char* input) {
    std::string value(input);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected an unsigned decimal integer");
    size_t consumed = 0;
    const auto result = std::stoull(value, &consumed);
    if (!result || consumed != value.size()) throw std::runtime_error("Expected a positive integer");
    return result;
}
uint64_t tag(uint64_t page, unsigned round) {
    return (page + 1) * 0x9e3779b97f4a7c15ull ^ (uint64_t(round + 1) << 48);
}
void run_probe(sycl::queue& q, const DeviceInfo& device, const std::string& kind,
               size_t bytes, unsigned rounds) {
    const auto budget = memory_budget(device);
    // Diagnostics must not induce paging just to get a successful allocation.
    if (!budget.wddm_known) throw std::runtime_error("Cannot uniquely identify the selected GPU's WDDM budget");
    if (!allocation_fits(budget, bytes, device.max_allocation))
        throw std::runtime_error("Probe exceeds the allocation limit or available RAM/commit/WDDM budget with 4 GiB headroom");
    const auto d = q.get_device();
    if ((kind == "host" && !d.has(sycl::aspect::usm_host_allocations)) ||
        (kind == "shared" && !d.has(sycl::aspect::usm_shared_allocations)) ||
        (kind == "device" && !d.has(sycl::aspect::usm_device_allocations)))
        throw std::runtime_error("The selected USM allocation kind is unsupported");
    constexpr size_t page_bytes = 4096;
    if (bytes % page_bytes) throw std::runtime_error("Probe byte count must be a multiple of 4096");
    for (unsigned round = 0; round < rounds; ++round) {
        print_memory("before", device);
        const auto start = std::chrono::steady_clock::now();
        {
            Allocation arena{q};
            if (kind == "host") arena.pointer = sycl::malloc_host(bytes, q);
            else if (kind == "shared") arena.pointer = sycl::malloc_shared(bytes, q);
            else arena.pointer = sycl::malloc_device(bytes, q);
            if (!arena.pointer) throw std::runtime_error("USM allocation refused");
            Allocation errors{q, sycl::malloc_shared<uint32_t>(1, q)};
            if (!errors.pointer) throw std::runtime_error("Result allocation refused");
            auto* bad = static_cast<uint32_t*>(errors.pointer);
            auto* arena_bytes = static_cast<unsigned char*>(arena.pointer);
            *bad = 0;
            const size_t pages = bytes / page_bytes;
            constexpr size_t chunk_pages = (64 * 1024 * 1024) / page_bytes;
            // Touch every physical page, with 64-bit indexing on both sides of
            // 4 GiB. Chunk dispatches to keep individual Windows jobs bounded.
            for (size_t first = 0; first < pages; first += chunk_pages) {
                const size_t count = std::min(chunk_pages, pages - first);
                q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
                    const uint64_t page = first + id[0];
                    auto* location = reinterpret_cast<uint64_t*>(arena_bytes + page * page_bytes);
                    *location = (page + 1) * 0x9e3779b97f4a7c15ull ^ (uint64_t(round + 1) << 48);
                }).wait_and_throw();
            }
            for (size_t first = 0; first < pages; first += chunk_pages) {
                const size_t count = std::min(chunk_pages, pages - first);
                q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
                    const uint64_t page = first + id[0];
                    const auto actual = *reinterpret_cast<const uint64_t*>(arena_bytes + page * page_bytes);
                    const auto expected = (page + 1) * 0x9e3779b97f4a7c15ull ^ (uint64_t(round + 1) << 48);
                    if (actual != expected) {
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> counter(*bad);
                        counter.fetch_add(1);
                    }
                }).wait_and_throw();
            }
            if (*bad) throw std::runtime_error("GPU page tags mismatch (possible alias/addressing defect)");
            if (kind != "device") {
                for (size_t page = 0; page < pages; ++page) {
                    auto* location = reinterpret_cast<uint64_t*>(arena_bytes + page * page_bytes);
                    if (*location != tag(page, round)) throw std::runtime_error("GPU-to-host visibility failed");
                    *location = ~tag(page, round);
                }
                for (size_t first = 0; first < pages; first += chunk_pages) {
                    const size_t count = std::min(chunk_pages, pages - first);
                    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
                        const uint64_t page = first + id[0];
                        const auto actual = *reinterpret_cast<const uint64_t*>(arena_bytes + page * page_bytes);
                        const auto expected = ~((page + 1) * 0x9e3779b97f4a7c15ull ^ (uint64_t(round + 1) << 48));
                        if (actual != expected) {
                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> counter(*bad);
                            counter.fetch_add(1);
                        }
                    }).wait_and_throw();
                }
                if (*bad) throw std::runtime_error("Host-to-GPU visibility failed");
            }
            print_memory("resident", device);
            q.wait_and_throw();
        }
        print_memory("released", device);
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("probe kind=%s bytes=%zu round=%u seconds=%.6f result=passed\n", kind.c_str(), bytes, round + 1, seconds);
        std::fflush(stdout);
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        std::string kind = "shared";
        uint64_t bytes = 64 * 1024 * 1024;
        unsigned index = 0, rounds = 6;
        bool inspect_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option(argv[i]);
            if (option == "--inspect") { inspect_only = true; continue; }
            if (option == "--large") { bytes = 4 * GiB + 64 * 1024; continue; }
            if (option == "--help") {
                std::puts("tinystrata-intel-probe [--inspect] [--device N] [--kind host|shared|device] [--bytes N|--large] [--rounds N]");
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("Missing option value");
            if (option == "--kind") kind = argv[++i];
            else if (option == "--bytes") bytes = positive_integer(argv[++i]);
            else if (option == "--rounds") {
                auto n = positive_integer(argv[++i]);
                if (n > 100) throw std::runtime_error("At most 100 rounds are supported");
                rounds = static_cast<unsigned>(n);
            } else if (option == "--device") {
                const std::string value(argv[++i]);
                if (value == "0") index = 0;
                else {
                    auto n = positive_integer(value.c_str());
                    if (n > 1024) throw std::runtime_error("Invalid GPU index");
                    index = static_cast<unsigned>(n);
                }
            } else throw std::runtime_error("Unknown option: " + option);
        }
        if (kind != "host" && kind != "shared" && kind != "device") throw std::runtime_error("Unknown USM kind");
        if (bytes > std::numeric_limits<size_t>::max()) throw std::runtime_error("Allocation size overflows size_t");
        const auto selected = strata::intel::select_device(index);
        const auto device = strata::intel::inspect_device(selected);
        sycl::queue queue(selected, [](sycl::exception_list exceptions) {
            for (const auto& exception : exceptions) std::rethrow_exception(exception);
        }, sycl::property::queue::in_order{});
        std::printf("device=%s vendor_id=0x%04x device_id=0x%04x architecture=%s integrated=%d driver=%s backend=level_zero compiler=%d\n",
            device.name.c_str(), device.vendor_id, device.device_id, device.architecture.c_str(), int(device.integrated), device.driver.c_str(), __INTEL_LLVM_COMPILER);
        std::printf("global_memory_reported=%llu max_allocation=%llu usm_host=%d usm_shared=%d usm_device=%d subgroup_sizes=",
            (unsigned long long)device.reported_global_memory, (unsigned long long)device.max_allocation,
            int(selected.has(sycl::aspect::usm_host_allocations)), int(selected.has(sycl::aspect::usm_shared_allocations)),
            int(selected.has(sycl::aspect::usm_device_allocations)));
        for (const auto size : selected.get_info<sycl::info::device::sub_group_sizes>()) std::printf("%zu,", size);
        std::puts("");
        strata::intel::print_memory("inspect", device);
        if (!inspect_only) run_probe(queue, device, kind, static_cast<size_t>(bytes), rounds);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Intel probe failed: %s\n", error.what());
        return 1;
    }
}
