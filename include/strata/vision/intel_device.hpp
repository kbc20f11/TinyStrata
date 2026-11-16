#pragma once
#include "strata/intel/device.hpp"

namespace strata::vision {
// Resolve the exact SYCL device used by the pinned GGML backend. Its SYCLn
// ordering is not assumed to equal the language engine's device ordering.
strata::intel::DeviceInfo intel_device(const char* backend_device_name);
}
