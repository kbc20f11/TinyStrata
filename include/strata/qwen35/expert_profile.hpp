// Learned Expert rankings; no request text, activations, or fixed VRAM capacity.
// Ranking policy informed by Basalt (MIT), src/core/expert_cache.cpp. See NOTICE.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace strata::qwen35::expert_profile {
using Order = std::vector<std::vector<int32_t>>;
// The caller supplies a model-artifact key and an architecture/quant-layout key.
// A sampled content fingerprint is not a full-file digest; include file identity
// metadata in such a key, or use a verified artifact digest when one is available.
struct Identity {
    std::string model, layout;
    uint32_t layers = 0, experts = 0;
    bool operator==(const Identity&) const = default;
};
struct Profile { Identity identity; Order order; };
constexpr uint32_t kVersion = 1, kMaxEntries = 4 * 1024 * 1024, kMaxKeyBytes = 4096;

inline bool valid_identity(const Identity& id) {
    return id.layers && id.experts && id.layers <= kMaxEntries / id.experts &&
           !id.model.empty() && !id.layout.empty() &&
           id.model.size() <= kMaxKeyBytes && id.layout.size() <= kMaxKeyBytes;
}
inline bool valid_order(const Identity& id, const Order& order) {
    if (!valid_identity(id) || order.size() != id.layers) return false;
    std::vector<uint8_t> seen(id.experts);
    for (const auto& row : order) {
        if (row.size() != id.experts) return false;
        std::fill(seen.begin(), seen.end(), 0);
        for (int32_t e : row)
            if (e < 0 || (uint32_t) e >= id.experts || seen[(size_t) e]++) return false;
    }
    return true;
}

// Preserve the current admitted set first, then recent routing frequency. Prior
// rank and expert ID make ties stable across saves and different slot budgets.
inline bool rank(const Identity& id, const std::vector<std::vector<double>>& heat,
                 const Order& resident, const Order& prior, Profile& out, std::string& error) {
    out = {};
    if (!valid_identity(id) || heat.size() != id.layers || resident.size() != id.layers ||
        (!prior.empty() && !valid_order(id, prior))) {
        error = "invalid Expert profile inputs"; return false;
    }
    Order order(id.layers);
    for (uint32_t l = 0; l < id.layers; ++l) {
        if (heat[l].size() != id.experts) { error = "invalid Expert heat shape"; return false; }
        std::vector<uint8_t> held(id.experts, 0);
        for (int32_t e : resident[l]) {
            if (e < 0 || (uint32_t) e >= id.experts || held[(size_t) e]++) {
                error = "invalid resident Expert list"; return false;
            }
        }
        for (double value : heat[l])
            if (!std::isfinite(value) || value < 0) { error = "invalid Expert heat"; return false; }
        std::vector<uint32_t> before(id.experts);
        std::iota(before.begin(), before.end(), 0u);
        if (!prior.empty())
            for (uint32_t r = 0; r < id.experts; ++r) before[(size_t) prior[l][r]] = r;
        auto& row = order[l]; row.resize(id.experts); std::iota(row.begin(), row.end(), 0);
        std::stable_sort(row.begin(), row.end(), [&](int32_t a, int32_t b) {
            if (held[(size_t) a] != held[(size_t) b]) return held[(size_t) a] > held[(size_t) b];
            if (heat[l][(size_t) a] != heat[l][(size_t) b]) return heat[l][(size_t) a] > heat[l][(size_t) b];
            if (before[(size_t) a] != before[(size_t) b]) return before[(size_t) a] < before[(size_t) b];
            return a < b;
        });
    }
    out = {id, std::move(order)}; error.clear(); return true;
}

inline bool select(const Profile& profile, const std::vector<size_t>& slots, Order& selected,
                   std::string& error) {
    selected.clear();
    if (!valid_order(profile.identity, profile.order) || slots.size() != profile.identity.layers) {
        error = "invalid Expert profile selection"; return false;
    }
    Order result(slots.size());
    for (size_t l = 0; l < slots.size(); ++l) {
        if (slots[l] > profile.identity.experts) { error = "Expert slots exceed model geometry"; return false; }
        result[l].assign(profile.order[l].begin(), profile.order[l].begin() + slots[l]);
    }
    selected = std::move(result); error.clear(); return true;
}

namespace detail {
constexpr std::array<uint8_t, 8> kMagic{{'T','S','E','X','P','R','F','1'}};
inline void u32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (int s = 0; s < 32; s += 8) bytes.push_back((uint8_t) (value >> s));
}
inline void u64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (int s = 0; s < 64; s += 8) bytes.push_back((uint8_t) (value >> s));
}
inline uint32_t get32(const std::vector<uint8_t>& b, size_t at) {
    uint32_t v = 0; for (int s = 0; s < 32; s += 8) v |= uint32_t(b[at++]) << s; return v;
}
inline uint64_t get64(const std::vector<uint8_t>& b, size_t at) {
    uint64_t v = 0; for (int s = 0; s < 64; s += 8) v |= uint64_t(b[at++]) << s; return v;
}
inline uint64_t checksum(const uint8_t* data, size_t size) {
    uint64_t value = 14695981039346656037ull;
    for (size_t i = 0; i < size; ++i) value = (value ^ data[i]) * 1099511628211ull;
    return value;
}
inline bool encode(const Profile& p, std::vector<uint8_t>& bytes, std::string& error) {
    if (!valid_order(p.identity, p.order)) { error = "invalid Expert profile"; return false; }
    const auto& id = p.identity;
    bytes.assign(kMagic.begin(), kMagic.end());
    for (uint32_t v : {kVersion, id.layers, id.experts, (uint32_t) id.model.size(),
                       (uint32_t) id.layout.size(), id.layers * id.experts}) u32(bytes, v);
    bytes.insert(bytes.end(), id.model.begin(), id.model.end());
    bytes.insert(bytes.end(), id.layout.begin(), id.layout.end());
    for (const auto& row : p.order) for (int32_t e : row) u32(bytes, (uint32_t) e);
    u64(bytes, checksum(bytes.data(), bytes.size())); return true;
}
inline bool decode(const std::vector<uint8_t>& bytes, const Identity& expected,
                   Profile& out, std::string& error) {
    if (bytes.size() < 40 || !std::equal(kMagic.begin(), kMagic.end(), bytes.begin()) ||
        get32(bytes, 8) != kVersion) { error = "unsupported Expert profile header"; return false; }
    const uint32_t nl = get32(bytes, 12), ne = get32(bytes, 16),
                   model_n = get32(bytes, 20), layout_n = get32(bytes, 24), count = get32(bytes, 28);
    if (nl != expected.layers || ne != expected.experts || model_n > kMaxKeyBytes ||
        layout_n > kMaxKeyBytes || count != (uint64_t) nl * ne || count > kMaxEntries ||
        bytes.size() != 40ull + model_n + layout_n + 4ull * count ||
        get64(bytes, bytes.size() - 8) != checksum(bytes.data(), bytes.size() - 8)) {
        error = "corrupt or mismatched Expert profile"; return false;
    }
    size_t at = 32;
    Identity id{std::string((const char*) bytes.data() + at, model_n), {}, nl, ne}; at += model_n;
    id.layout.assign((const char*) bytes.data() + at, layout_n); at += layout_n;
    if (!(id == expected)) { error = "Expert profile belongs to another model or layout"; return false; }
    Order order(nl, std::vector<int32_t>(ne));
    for (auto& row : order) for (int32_t& e : row) { e = (int32_t) get32(bytes, at); at += 4; }
    if (!valid_order(id, order)) { error = "invalid Expert profile permutation"; return false; }
    out = {std::move(id), std::move(order)}; error.clear(); return true;
}
} // namespace detail

inline bool load(const std::filesystem::path& path, const Identity& expected, Profile& out,
                 std::string& error) {
    out = {};
    if (!valid_identity(expected)) { error = "invalid Expert model identity"; return false; }
    try {
        std::error_code ec;
        const uintmax_t size = std::filesystem::file_size(path, ec);
        if (ec || size < 40 || size > 40ull + 2 * kMaxKeyBytes + 4ull * kMaxEntries) {
            error = "Expert profile unavailable or invalid size"; return false;
        }
        std::ifstream file(path, std::ios::binary);
        std::vector<uint8_t> bytes((size_t) size);
        if (!file || !file.read((char*) bytes.data(), (std::streamsize) bytes.size()) ||
            file.peek() != std::char_traits<char>::eof()) {
            error = "cannot read complete Expert profile"; return false;
        }
        return detail::decode(bytes, expected, out, error);
    } catch (const std::exception&) { error = "cannot load Expert profile"; return false; }
}

// A sibling temporary file is published only after its full payload closes.
// Failed writes preserve an existing valid profile; concurrent writers publish
// complete snapshots and the last successful rename wins.
inline bool save(const std::filesystem::path& path, const Profile& profile, std::string& error) {
    std::filesystem::path temporary;
    try {
        std::vector<uint8_t> bytes;
        if (!detail::encode(profile, bytes, error)) return false;
        std::error_code ec;
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) { error = "cannot create Expert profile directory"; return false; }
        static std::atomic<uint64_t> serial{0};
        std::random_device random;
        temporary = path;
        temporary += ".tmp." + std::to_string((uint64_t(random()) << 32) ^ random()) +
                     "." + std::to_string(serial.fetch_add(1, std::memory_order_relaxed));
        {
            std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
            if (!file || !file.write((const char*) bytes.data(), (std::streamsize) bytes.size())) {
                error = "cannot write Expert profile";
                file.close(); std::filesystem::remove(temporary, ec); return false;
            }
            file.flush(); file.close();
            if (!file) { error = "cannot close Expert profile"; std::filesystem::remove(temporary, ec); return false; }
        }
        std::filesystem::rename(temporary, path, ec);
        if (ec) { error = "cannot publish Expert profile"; std::filesystem::remove(temporary, ec); return false; }
        error.clear(); return true;
    } catch (const std::exception&) {
        if (!temporary.empty()) { std::error_code ignored; std::filesystem::remove(temporary, ignored); }
        error = "cannot save Expert profile"; return false;
    }
}
} // namespace strata::qwen35::expert_profile
