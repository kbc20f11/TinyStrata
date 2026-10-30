#pragma once
#include <string>
namespace strata {
enum class ModelFamily { Unknown = 0, Qwen35Moe, CloverMoe };
inline ModelFamily model_family_from_arch(const std::string& arch) {
    return arch == "qwen35moe" ? ModelFamily::Qwen35Moe : arch == "clover_moe" ? ModelFamily::CloverMoe : ModelFamily::Unknown;
}
inline const char* model_family_arch(ModelFamily f) {
    return f == ModelFamily::Qwen35Moe ? "qwen35moe" : f == ModelFamily::CloverMoe ? "clover_moe" : "";
}
inline const char* model_family_title(ModelFamily f) {
    return f == ModelFamily::Qwen35Moe ? "Qwen3.6-35B-A3B" : f == ModelFamily::CloverMoe ? "Clover-35B-A3B-Preview" : "an unsupported model";
}
inline bool model_family_runs(ModelFamily f) { return f == ModelFamily::Qwen35Moe || f == ModelFamily::CloverMoe; }
inline std::string model_family_not_supported(ModelFamily f) {
    return model_family_runs(f) ? "" : "TinyStrata requires a Qwen3.6 qwen35moe or Clover clover_moe GGUF";
}
}  // namespace strata
