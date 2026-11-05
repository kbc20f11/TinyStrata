// CPU-only ranking/file-format regression; no models, GPU runtime or services.
#include "strata/qwen35/expert_profile.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace EP = strata::qwen35::expert_profile;
namespace fs = std::filesystem;
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

struct PrivateDirectory {
    fs::path base = fs::absolute(fs::temp_directory_path()).lexically_normal(), path;
    PrivateDirectory() {
        std::random_device random;
        for (int attempt = 0; attempt < 8; ++attempt) {
            path = base / ("tinystrata-expert-profile-test-" + std::to_string(random()) + "-" + std::to_string(random()));
            std::error_code ec;
            if (fs::create_directory(path, ec)) return;
        }
        REQUIRE(false);
    }
    ~PrivateDirectory() {
        // Only the exact unique directory created by this test can be removed.
        const auto target = fs::absolute(path).lexically_normal();
        if (target.parent_path() == base && target.filename().string().rfind("tinystrata-expert-profile-test-", 0) == 0) {
            std::error_code ec; fs::remove_all(target, ec);
        }
    }
};

std::vector<uint8_t> bytes(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    REQUIRE(file.good());
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), {});
}
void put(const fs::path& path, const std::vector<uint8_t>& data) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    REQUIRE(file.good());
    if (!data.empty()) file.write((const char*) data.data(), (std::streamsize) data.size());
    file.close(); REQUIRE(!file.fail());
}
void word(std::vector<uint8_t>& data, size_t at, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) data[at++] = uint8_t(value >> shift);
}
void repair_checksum(std::vector<uint8_t>& data) {
    const uint64_t hash = EP::detail::checksum(data.data(), data.size() - 8);
    for (int i = 0; i < 8; ++i) data[data.size() - 8 + (size_t) i] = uint8_t(hash >> (8 * i));
}

int main() {
    PrivateDirectory temporary;
    const fs::path target = temporary.path / "nested" / fs::path(u8"配置.profile");
    const fs::path badfile = temporary.path / "invalid.profile";
    const EP::Identity id{"verified-artifact-sha256:0123456789abcdef", "qwen35moe:width2048:gu18:down21", 2, 5};
    const std::vector<std::vector<double>> heat{{1, 4, 3, 2, 0}, {0, 0, 0, 0, 0}};
    const EP::Order resident{{1, 4}, {0}}, prior{{4, 3, 2, 1, 0}, {4, 3, 2, 1, 0}};
    EP::Profile profile;
    std::string error;
    REQUIRE(EP::rank(id, heat, resident, prior, profile, error));
    REQUIRE(error.empty() && profile.identity == id);
    REQUIRE((profile.order[0] == std::vector<int32_t>{1, 4, 2, 3, 0}));
    REQUIRE((profile.order[1] == std::vector<int32_t>{0, 4, 3, 2, 1}));
    REQUIRE(EP::valid_order(id, profile.order));

    EP::Order selected;
    REQUIRE(EP::select(profile, {1, 3}, selected, error));
    REQUIRE((selected == EP::Order{{1}, {0, 4, 3}}));
    REQUIRE(EP::select(profile, {4, 0}, selected, error));
    REQUIRE((selected == EP::Order{{1, 4, 2, 3}, {}}));
    REQUIRE(EP::select(profile, {5, 5}, selected, error) && selected == profile.order);
    REQUIRE(!EP::select(profile, {6, 2}, selected, error) && selected.empty() && !error.empty());
    REQUIRE(!EP::select(profile, {1}, selected, error) && selected.empty());

    EP::Profile loaded = profile;
    REQUIRE(!EP::load(target, id, loaded, error) && loaded.order.empty());
    REQUIRE(EP::save(target, profile, error));
    REQUIRE(EP::load(target, id, loaded, error));
    REQUIRE(loaded.identity == profile.identity && loaded.order == profile.order);
    const auto original = bytes(target);
    REQUIRE(original.size() == 40 + id.model.size() + id.layout.size() + 4 * id.layers * id.experts);
    REQUIRE(std::string(original.begin(), original.begin() + 8) == "TSEXPRF1");
    REQUIRE(original[8] == 1 && original[9] == 0 && original[10] == 0 && original[11] == 0);

    // Capacity does not appear in a saved snapshot: selecting a larger GPU's
    // allocation from a ranking saved on a smaller GPU preserves its prefix.
    REQUIRE(EP::select(loaded, {2, 2}, selected, error));
    const auto small = selected;
    REQUIRE(EP::select(loaded, {4, 5}, selected, error));
    REQUIRE(std::equal(small[0].begin(), small[0].end(), selected[0].begin()));
    REQUIRE(std::equal(small[1].begin(), small[1].end(), selected[1].begin()));

    for (int change = 0; change < 4; ++change) {
        auto different = id;
        if (change == 0) different.model += ":different-weight";
        if (change == 1) different.layout += ":different-quant";
        if (change == 2) { different.layers = 1; different.experts = 10; }
        if (change == 3) different.experts = 6;
        loaded = profile;
        REQUIRE(!EP::load(target, different, loaded, error));
        REQUIRE(loaded.order.empty() && loaded.identity.layers == 0 && !error.empty());
    }

    auto other = profile;
    for (auto& row : other.order) std::reverse(row.begin(), row.end());
    REQUIRE(EP::save(target, other, error));
    REQUIRE(EP::load(target, id, loaded, error) && loaded.order == other.order);
    const auto replacement = bytes(target);
    REQUIRE(replacement != original);

    // Invalid in-memory state never replaces a previously published snapshot.
    auto invalid = profile;
    invalid.order[0][1] = invalid.order[0][0];
    REQUIRE(!EP::save(target, invalid, error) && bytes(target) == replacement);
    auto badheat = heat; badheat[0][0] = -1;
    REQUIRE(!EP::rank(id, badheat, resident, prior, loaded, error) && loaded.order.empty());
    badheat = heat; badheat[0][0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE(!EP::rank(id, badheat, resident, prior, loaded, error) && loaded.order.empty());
    auto badresident = resident; badresident[0].push_back(1);
    REQUIRE(!EP::rank(id, heat, badresident, prior, loaded, error) && loaded.order.empty());
    auto badprior = prior; badprior[0].pop_back();
    REQUIRE(!EP::rank(id, heat, resident, badprior, loaded, error) && loaded.order.empty());

    int malformed = 0;
    const auto reject = [&](std::vector<uint8_t> raw) {
        put(badfile, raw); loaded = profile;
        REQUIRE(!EP::load(badfile, id, loaded, error));
        REQUIRE(loaded.order.empty() && loaded.identity.layers == 0 && !error.empty());
        ++malformed;
    };
    for (size_t length : {size_t(0), size_t(1), size_t(7), size_t(8), size_t(31), size_t(32), original.size() - 1})
        reject(std::vector<uint8_t>(original.begin(), original.begin() + length));
    { auto raw = original; raw[0] ^= 1; reject(raw); }
    { auto raw = original; raw.back() ^= 1; reject(raw); }
    { auto raw = original; raw[32] ^= 1; reject(raw); }
    { auto raw = original; raw.push_back(0); reject(raw); }
    { auto raw = original; word(raw, 8, 999); repair_checksum(raw); reject(raw); }
    { auto raw = original; word(raw, 20, UINT32_MAX); repair_checksum(raw); reject(raw); }
    { auto raw = original; word(raw, 28, UINT32_MAX); repair_checksum(raw); reject(raw); }
    { auto raw = original; word(raw, 12, UINT32_MAX); repair_checksum(raw); reject(raw); }
    const size_t first = 32 + id.model.size() + id.layout.size();
    { auto raw = original; std::copy_n(raw.begin() + first, 4, raw.begin() + first + 4); repair_checksum(raw); reject(raw); }
    { auto raw = original; word(raw, first, id.experts); repair_checksum(raw); reject(raw); }
    { auto raw = original; word(raw, first, UINT32_MAX); repair_checksum(raw); reject(raw); }

    // A failed publish cannot remove its destination or strand temporary files.
    const auto occupied = temporary.path / "occupied";
    REQUIRE(fs::create_directory(occupied));
    put(occupied / "keep", {1, 2, 3});
    REQUIRE(!EP::save(occupied, profile, error));
    REQUIRE((bytes(occupied / "keep") == std::vector<uint8_t>{1, 2, 3}));
    for (const auto& item : fs::directory_iterator(temporary.path))
        REQUIRE(item.path().filename().string().find(".tmp.") == std::string::npos);
    for (const auto& item : fs::directory_iterator(target.parent_path()))
        REQUIRE(item.path() == target);

    REQUIRE(!EP::valid_identity({"", "layout", 1, 1}));
    REQUIRE(!EP::valid_identity({"model", "layout", 0, 1}));
    REQUIRE(!EP::valid_identity({"model", "layout", 1, 0}));
    REQUIRE(!EP::valid_identity({"model", "layout", UINT32_MAX, UINT32_MAX}));
    REQUIRE(!EP::valid_identity({std::string(EP::kMaxKeyBytes + 1, 'a'), "layout", 1, 1}));
    std::printf("PASS Expert profile ranking, capacity reuse, identity, atomic replacement and %d malformed files\n", malformed);
    return 0;
}
