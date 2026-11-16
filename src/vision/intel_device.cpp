#include "strata/vision/intel_device.hpp"
#include "ggml-sycl.h"
#include "dpct/helper.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::vision {
strata::intel::DeviceInfo intel_device(const char* name) {
    constexpr const char* prefix = GGML_SYCL_NAME;
    if (!name || std::strncmp(name, prefix, std::strlen(prefix)))
        throw std::invalid_argument("An enumerated GGML SYCL device is required");
    const std::string number(name + std::strlen(prefix));
    size_t end = 0;
    if (number.empty() || number.front() < '0' || number.front() > '9')
        throw std::invalid_argument("Invalid GGML SYCL device index");
    const auto index = std::stoul(number, &end);
    if (end != number.size() || index >= unsigned(ggml_backend_sycl_get_device_count()) ||
        index > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Invalid GGML SYCL device index");
    // This is the same pinned dpct singleton used by ggml_backend_sycl_reg().
    // inspect_device rejects CPU, non-Intel and non-Level-Zero devices and
    // obtains the physical Windows adapter LUID through the driver extension.
    return strata::intel::inspect_device(dpct::dev_mgr::instance().get_device(unsigned(index)));
}
}
