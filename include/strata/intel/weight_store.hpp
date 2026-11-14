#pragma once
#include "strata/intel/device.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <map>
#include <memory>

namespace strata::intel {
enum class WeightMemory { Host, Shared, Device };
enum class WeightLayout { Original, Q6Interleaved16, ExpertInterleaved16 };
struct ResidentTensor {
    strata::TensorInfo info;
    const uint8_t* data = nullptr;
    uint64_t bytes = 0;
    WeightLayout layout = WeightLayout::Original;
};
// One compressed weight arena. The GGUF mappings are closed before allocating
// the arena; host/shared storage is filled directly from the file. Device
// storage uses one bounded transfer buffer which is released before returning.
class WeightStore {
public:
    static std::unique_ptr<WeightStore> load(sycl::queue queue, const std::string& gguf,
                                            WeightMemory kind = WeightMemory::Host, bool interleave_output = false, bool interleave_experts = false, bool interleave_dense = false, bool interleave_split_output = false);
    ~WeightStore();
    WeightStore(const WeightStore&) = delete;
    const std::map<std::string, ResidentTensor>& tensors() const { return tensors_; }
    const std::map<std::string, strata::MetaValue>& metadata() const { return metadata_; }
    uint64_t allocated_bytes() const { return bytes_; }
    uint64_t upload_bytes() const { return upload_bytes_; }
    const strata::core::ModelGeometry& geometry() const { return geometry_; }
private:
    explicit WeightStore(sycl::queue queue) : queue_(std::move(queue)) {}
    sycl::queue queue_;
    uint8_t* arena_ = nullptr;
    uint64_t bytes_ = 0;
    uint64_t upload_bytes_ = 0;
    strata::core::ModelGeometry geometry_;
    std::map<std::string, ResidentTensor> tensors_;
    std::map<std::string, strata::MetaValue> metadata_;
};
} // namespace strata::intel
