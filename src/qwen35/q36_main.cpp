#include "strata/qwen35/engine.hpp"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    // This probe must work without a model, GPU or initialized GPU runtime.
    if (argc == 2 && std::strcmp(argv[1], "--capabilities") == 0) {
#if defined(STRATA_USE_HIP)
        std::puts("{\"backend\":\"hip\",\"model_family\":\"qwen35moe\",\"model_families\":[\"qwen35moe\",\"clover_moe\"],\"kv_types\":[\"int8\",\"fp8-mma\"],\"images\":true,\"mtp\":true,\"self_speculative\":true}");
#else
        std::puts("{\"backend\":\"cuda\"}");
#endif
        return 0;
    }
    return strata::qwen35::run(argc, argv);
}
