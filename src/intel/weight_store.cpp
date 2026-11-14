#include "strata/intel/weight_store.hpp"
#include "strata/intel/q6_interleave.hpp"
#include "strata/common/utf8_path.hpp"
#include "strata/core/family_layout.hpp"
#include <windows.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::intel {
namespace {
struct File {
    HANDLE handle = INVALID_HANDLE_VALUE;
    explicit File(const std::string& path) {
        handle = CreateFileW(strata::paths::utf8_to_wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ,
                             nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (handle == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open GGUF tensor source");
    }
    ~File() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
};
struct Usm {
    sycl::queue& queue;
    void* pointer;
    ~Usm() { if (pointer) sycl::free(pointer, queue); }
};
struct PlannedTensor {
    strata::TensorInfo tensor;
    size_t shard;
    uint64_t file_offset;
    uint64_t arena_offset;
    uint64_t bytes;
    uint64_t stored_bytes;
    WeightLayout layout;
};
}
WeightStore::~WeightStore() {
    if (arena_) {
        try { sycl::free(arena_, queue_); }
        catch (const std::exception& error) { std::fprintf(stderr, "Intel weight release failed: %s\n", error.what()); }
    }
}
std::unique_ptr<WeightStore> WeightStore::load(sycl::queue queue, const std::string& gguf, WeightMemory kind, bool interleave_output, bool interleave_experts, bool interleave_dense, bool interleave_split_output) {
    if (!queue.has_property<sycl::property::queue::in_order>()) throw std::invalid_argument("Weight loading requires an in-order queue");
    const auto device = inspect_device(queue.get_device());
    if (!device.integrated) throw std::runtime_error("Single-pool Intel weight storage currently requires an integrated GPU");
    const auto paths = strata::gguf_split_paths(gguf);
    // Hold read-only handles across metadata planning and loading. This closes
    // the replace/write window while still releasing every memory mapping
    // before the persistent arena is allocated.
    std::vector<std::unique_ptr<File>> files;
    for (const auto& path : paths) files.push_back(std::make_unique<File>(path));
    auto store = std::unique_ptr<WeightStore>(new WeightStore(queue));
    std::vector<PlannedTensor> plan;
    size_t packed_expert_tensors = 0, original_expert_tensors = 0, packed_dense_tensors = 0;
    const auto is_expert = [](const std::string& name) {
        return name.ends_with(".ffn_gate_exps.weight") || name.ends_with(".ffn_up_exps.weight") || name.ends_with(".ffn_down_exps.weight");
    };
    {
        strata::GgufModel model(paths);
        const auto* architecture = model.meta().get("general.architecture");
        if (!architecture || architecture->s != "qwen35moe") throw std::runtime_error("The initial Intel weight loader accepts qwen35moe only");
        const auto* name = model.meta().get("general.name");
        if (!name || name->s.find("Qwen3.6") == std::string::npos)
            throw std::runtime_error("The initial Intel route is enabled only for Qwen3.6; Tiel and other models are unvalidated");
        const auto error = strata::check_architecture_qwen35moe(model.meta());
        if (!error.empty()) throw std::runtime_error("Qwen GGUF: " + error);
        int64_t top_k = 0;
        std::string layout_error;
        if (!strata::core::geometry_from_gguf(model.meta(), store->geometry_, top_k, layout_error) ||
            !strata::core::check_gguf_layout(model, store->geometry_, layout_error))
            throw std::runtime_error("Qwen GGUF layout: " + layout_error);
        store->metadata_ = model.meta().metadata();
        // Keep each layer's expert trio in one layout. Mixed GGUF quantization
        // (including the draft layer) retains the original GPU path when any
        // of its three matrices has no interleaved XMX decoder.
        std::map<std::string, std::pair<int, bool>> expert_layers;
        if (interleave_experts) for (size_t shard = 0; shard < model.size(); ++shard)
            for (const auto& tensor : model.shard(shard).tensors()) if (is_expert(tensor.name)) {
                const auto prefix = tensor.name.substr(0, tensor.name.rfind(".ffn_"));
                auto [it, inserted] = expert_layers.try_emplace(prefix, std::make_pair(0, true));
                ++it->second.first;
                it->second.second = it->second.second && (tensor.type == 14 || tensor.type == 21 || tensor.type == 22) &&
                    tensor.shape.size() == 3 && tensor.shape[0] <= 8192 && tensor.shape[1] % 16 == 0 && tensor.shape[2] <= 256;
            }
        for (size_t shard = 0; shard < model.size(); ++shard) {
            for (const auto& tensor : model.shard(shard).tensors()) {
                if (!model.in_bounds(tensor, shard)) throw std::runtime_error("Invalid tensor payload: " + tensor.name);
                const auto bytes = strata::tensor_payload_bytes(tensor);
                const auto layer = expert_layers.find(tensor.name.substr(0, tensor.name.rfind(".ffn_")));
                const bool expert = interleave_experts && is_expert(tensor.name) && layer != expert_layers.end() && layer->second.first == 3 && layer->second.second;
                if (is_expert(tensor.name)) { if (expert) ++packed_expert_tensors; else ++original_expert_tensors; }
                const bool dense = interleave_dense && tensor.type == 14 && tensor.shape.size() == 2 && tensor.name != "token_embd.weight" &&
                    // B390 alternating-kernel measurements: short/wide output
                    // matrices benefit; small output counts and width 4096
                    // regress at T1 and retain their original GPU kernels.
                    (tensor.shape[0] <= 2048 || (interleave_split_output && tensor.shape[0] == 4096 && tensor.shape[1] == 2048)) && tensor.shape[1] >= 2048 && tensor.shape[0] % 256 == 0 && tensor.shape[1] % 16 == 0 &&
                    (tensor.name == "output.weight" || tensor.shape[0] * tensor.shape[1] <= 8192 * 2048);
                if (dense) ++packed_dense_tensors;
                const bool packed = expert || dense || (interleave_output && tensor.name == "output.weight");
                if (packed && ((expert ? tensor.type != 14 && tensor.type != 21 && tensor.type != 22 : tensor.type != 14) ||
                    tensor.shape.size() != size_t(expert ? 3 : 2) || tensor.shape[0] > 8192 || tensor.shape[1] > 1000000 ||
                    (expert && (tensor.shape[1] % 16 || tensor.shape[2] > 256))))
                    throw std::runtime_error("Interleaved weights need validated Q6_K/IQ2_S/IQ3_S geometry");
                const int block_bytes = tensor.type == 14 ? 210 : tensor.type == 21 ? 110 : 82;
                const uint64_t stored_bytes = packed ? quant_interleaved_bytes(int(tensor.shape[1] * (expert ? tensor.shape[2] : 1)), int(tensor.shape[0]), block_bytes) : bytes;
                const uint64_t aligned = (stored_bytes + 255) & ~uint64_t{255};
                if (stored_bytes > std::numeric_limits<uint64_t>::max() - 255 || aligned > std::numeric_limits<uint64_t>::max() - store->bytes_)
                    throw std::runtime_error("Tensor arena size overflows");
                plan.push_back({tensor, shard, model.shard(shard).data_start() + tensor.offset, store->bytes_, bytes, stored_bytes,
                    expert ? WeightLayout::ExpertInterleaved16 : packed ? WeightLayout::Q6Interleaved16 : WeightLayout::Original});
                store->bytes_ += aligned;
            }
        }
    } // No persistent file mapping alongside the weight arena.
    constexpr uint64_t transfer_bytes = 8 * 1024 * 1024;
    const auto memory = memory_budget(device);
    if (!memory.wddm_known || !allocation_fits(memory, store->bytes_, device.max_allocation) ||
        allocation_budget(memory) - store->bytes_ < transfer_bytes)
        throw std::runtime_error("Compressed model exceeds the single-pool RAM/commit/WDDM budget with headroom");
    if (store->bytes_ > std::numeric_limits<size_t>::max()) throw std::runtime_error("Tensor arena exceeds address space");
    auto& q = store->queue_;
    if (kind == WeightMemory::Host) store->arena_ = sycl::malloc_host<uint8_t>(size_t(store->bytes_), q);
    else if (kind == WeightMemory::Shared) store->arena_ = sycl::malloc_shared<uint8_t>(size_t(store->bytes_), q);
    else store->arena_ = sycl::malloc_device<uint8_t>(size_t(store->bytes_), q);
    if (!store->arena_) throw std::runtime_error("Compressed weight arena allocation refused");
    Usm staging{q, kind == WeightMemory::Device || interleave_output || interleave_experts || interleave_dense ? sycl::malloc_host(size_t(transfer_bytes), q) : nullptr};
    if ((kind == WeightMemory::Device || interleave_output || interleave_experts || interleave_dense) && !staging.pointer) throw std::runtime_error("Weight staging allocation refused");
    Usm checksum{q, sycl::malloc_shared<uint64_t>(1, q)};
    if (!checksum.pointer) throw std::runtime_error("Weight check allocation refused");
    auto* actual = static_cast<uint64_t*>(checksum.pointer);
    uint64_t checked = 0;
    for (const auto& tensor : plan) {
        auto* target = store->arena_ + tensor.arena_offset;
        LARGE_INTEGER offset;
        offset.QuadPart = static_cast<LONGLONG>(tensor.file_offset);
        if (!SetFilePointerEx(files[tensor.shard]->handle, offset, nullptr, FILE_BEGIN))
            throw std::runtime_error("Cannot seek to tensor: " + tensor.tensor.name);
        uint64_t expected = 0;
        const bool packed = tensor.layout != WeightLayout::Original;
        const int width = packed ? int(tensor.tensor.shape[0]) : 0;
        const int block_bytes = tensor.tensor.type == 14 ? 210 : tensor.tensor.type == 21 ? 110 : 82;
        // Sample each 64 KiB plus the final byte independently on CPU and GPU.
        // This is a visibility/transfer check, not a cryptographic checksum.
        constexpr uint64_t sample_stride = 64 * 1024;
        if (packed) {
            const size_t source_tile = size_t(width / 256) * block_bytes * 16;
            const size_t target_tile = quant_interleaved_bytes(16, width, block_bytes);
            const size_t batch_tiles = size_t(transfer_bytes / 2) / target_tile;
            if (!batch_tiles) throw std::runtime_error("Interleave staging is too small");
            auto* source = static_cast<uint8_t*>(staging.pointer);
            auto* converted = source + transfer_bytes / 2;
            const size_t rows = size_t(tensor.tensor.shape[1]) * (tensor.layout == WeightLayout::ExpertInterleaved16 ? size_t(tensor.tensor.shape[2]) : 1);
            for (size_t tile = 0, tiles = rows / 16; tile < tiles;) {
                const size_t count = std::min(batch_tiles, tiles - tile), first = tile * source_tile;
                DWORD received = 0, requested = DWORD(count * source_tile);
                if (!ReadFile(files[tensor.shard]->handle, source, requested, &received, nullptr) || received != requested)
                    throw std::runtime_error("Short interleaved vocabulary read");
                for (size_t local = (sample_stride - first % sample_stride) % sample_stride; local < requested; local += sample_stride)
                    expected += uint64_t(source[local]) * (first + local + 1);
                if (first + requested == tensor.bytes) expected += uint64_t(source[requested - 1]) * tensor.bytes;
                interleave_quant(source, converted, int(count * 16), width, block_bytes);
                if (kind == WeightMemory::Device) {
                    q.memcpy(target + tile * target_tile, converted, count * target_tile).wait_and_throw();
                    store->upload_bytes_ += count * target_tile;
                } else std::memcpy(target + tile * target_tile, converted, count * target_tile);
                const auto current = memory_budget(device);
                if (std::min(current.available_physical, current.available_commit) < 4 * GiB)
                    throw std::runtime_error("Memory pressure while interleaving vocabulary");
                tile += count;
            }
        } else for (uint64_t first = 0; first < tensor.bytes; first += transfer_bytes) {
            const DWORD count = static_cast<DWORD>(std::min<uint64_t>(transfer_bytes, tensor.bytes - first));
            auto* destination = kind == WeightMemory::Device ? static_cast<uint8_t*>(staging.pointer) : target + first;
            DWORD received = 0;
            if (!ReadFile(files[tensor.shard]->handle, destination, count, &received, nullptr) || received != count)
                throw std::runtime_error("Short tensor read: " + tensor.tensor.name);
            for (uint64_t local = 0; local < count; local += sample_stride)
                expected += uint64_t(destination[local]) * (first + local + 1);
            if (first + count == tensor.bytes) expected += uint64_t(destination[count - 1]) * tensor.bytes;
            if (kind == WeightMemory::Device) {
                q.memcpy(target + first, destination, count).wait_and_throw();
                store->upload_bytes_ += count;
            }
            // Recheck system pressure while loading. Leave the desktop reserve
            // intact even if another application starts after the first check.
            const auto current = memory_budget(device);
            if (std::min(current.available_physical, current.available_commit) < 4 * GiB)
                throw std::runtime_error("Memory pressure while loading compressed weights");
        }
        *actual = 0;
        const uint64_t bytes = tensor.bytes;
        const uint64_t samples = (bytes + sample_stride - 1) / sample_stride;
        q.parallel_for(sycl::range<1>(size_t(samples)), [=](sycl::id<1> id) {
            const uint64_t offset = id[0] * sample_stride;
            uint64_t value = uint64_t(target[packed ? quant_interleaved_offset(offset, width, block_bytes) : offset]) * (offset + 1);
            if (id[0] == 0) value += uint64_t(target[packed ? quant_interleaved_offset(bytes - 1, width, block_bytes) : bytes - 1]) * bytes;
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device> sum(*actual);
            sum.fetch_add(value);
        }).wait_and_throw();
        if (*actual != expected) throw std::runtime_error("GPU compressed-weight visibility failed: " + tensor.tensor.name);
        store->tensors_.emplace(tensor.tensor.name, ResidentTensor{tensor.tensor, target, tensor.stored_bytes, tensor.layout});
        if (++checked % 100 == 0) {
            std::printf("weights checked=%llu/%zu\n", (unsigned long long)checked, plan.size());
            std::fflush(stdout);
        }
    }
    std::printf("weights tensors=%zu arena_bytes=%llu gpu_upload_bytes=%llu sampled_visibility=passed\n", store->tensors_.size(),
        (unsigned long long)store->bytes_, (unsigned long long)store->upload_bytes_);
    if (interleave_experts) std::printf("interleaved_expert_tensors=%zu original_expert_tensors=%zu\n", packed_expert_tensors, original_expert_tensors);
    if (interleave_dense) std::printf("interleaved_dense_tensors=%zu\n", packed_dense_tensors);
    return store;
}
} // namespace strata::intel
