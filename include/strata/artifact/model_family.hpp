#pragma once
#include <string>
namespace strata {
enum class ModelFamily { Unknown = 0, Qwen35Moe };
inline ModelFamily model_family_from_arch(const std::string& arch) {
    return arch == "qwen35moe" ? ModelFamily::Qwen35Moe : ModelFamily::Unknown;
}
inline const char* model_family_arch(ModelFamily f) {
    return f == ModelFamily::Qwen35Moe ? "qwen35moe" : "";
}
inline const char* model_family_title(ModelFamily f) {
    return f == ModelFamily::Qwen35Moe ? "Qwen3.6-35B-A3B" : "an unsupported model";
}
inline bool model_family_runs(ModelFamily f) { return f == ModelFamily::Qwen35Moe; }
inline std::string model_family_not_supported(ModelFamily f) {
    return model_family_runs(f) ? "" : "TinyStrata requires a Qwen3.6 qwen35moe GGUF";
}
}  // namespace strata
