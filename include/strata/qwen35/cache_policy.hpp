// Host-side Qwen Expert cache admission scheduling.
// Copyright (c) 2026 Niko1221 and the Strata contributors. See LICENSE and NOTICE.
#pragma once
#include <cstddef>
#include <vector>

namespace strata::qwen35::cache_policy {
// Continue after the last layer visited when a layer exhausts the shared swap
// budget. Disabled: preserve the original 0..layers-1 scan and leave the cursor.
class LayerScan {
public:
    LayerScan(size_t layers, bool fair, size_t& next_layer)
        : layers_(layers), fair_(fair), next_layer_(next_layer),
          start_(fair && layers ? next_layer % layers : 0) {}
    bool next(int remaining, size_t& layer) {
        if (remaining <= 0 || visited_ >= layers_) return false;
        layer = fair_ ? (start_ + visited_) % layers_ : visited_;
        ++visited_;
        if (fair_) next_layer_ = (layer + 1) % layers_;
        return true;
    }
private:
    size_t layers_, visited_ = 0;
    bool fair_;
    size_t& next_layer_;
    size_t start_;
};
inline void age(std::vector<double>& score) {
    for (double& value : score) value *= 0.5;
}
} // namespace strata::qwen35::cache_policy
