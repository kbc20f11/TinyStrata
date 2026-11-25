#include "strata/gemma4/vram.hpp"

namespace strata::ink {

bool query_resident_vram(size_t& bytes) {
    return gemma4::resident_vram().query(bytes);
}

}  // namespace strata::ink
