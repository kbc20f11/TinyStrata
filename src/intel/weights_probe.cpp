#include "strata/intel/weight_store.hpp"
#include "strata/common/utf8_args.hpp"
#include <cstdio>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        // Keep the native CLI's UTF-8 convention for Unicode Windows paths.
        strata::paths::Utf8Arguments arguments(argc, argv);
        argc = arguments.argc(); argv = arguments.argv();
        if (argc < 2 || argc > 3) throw std::runtime_error("Usage: tinystrata-intel-weights MODEL.gguf [host|shared|device]");
        const std::string kind = argc == 3 ? argv[2] : "host";
        using namespace strata::intel;
        WeightMemory memory;
        if (kind == "host") memory = WeightMemory::Host;
        else if (kind == "shared") memory = WeightMemory::Shared;
        else if (kind == "device") memory = WeightMemory::Device;
        else throw std::runtime_error("Unknown weight memory kind");
        sycl::queue queue(select_device(0), [](sycl::exception_list errors) {
            for (const auto& error : errors) std::rethrow_exception(error);
        }, sycl::property::queue::in_order{});
        const auto device = inspect_device(queue.get_device());
        print_memory("before_weights", device);
        {
            auto store = WeightStore::load(queue, argv[1], memory);
            print_memory("resident_weights", device);
        }
        print_memory("released_weights", device);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Intel weight probe failed: %s\n", error.what());
        return 1;
    }
}
