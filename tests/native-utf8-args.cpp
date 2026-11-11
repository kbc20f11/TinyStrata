#include "strata/common/utf8_args.hpp"
#include <cstdio>
#include <exception>
#include <string>

// Real child-process argv probe. It deliberately starts with ordinary main(),
// just like the shipping engines; no model, driver or GPU library is linked.
int main(int argc, char** argv) {
    try {
        strata::paths::Utf8Arguments arguments(argc, argv);
        argc = arguments.argc(); argv = arguments.argv();
        for (int i = 1; i < argc; ++i) {
            for (const unsigned char value : std::string(argv[i]))
                std::printf("%02x", static_cast<unsigned>(value));
            std::putchar('\n');
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
}
