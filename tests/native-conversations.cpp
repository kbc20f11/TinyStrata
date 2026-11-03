// CPU-only reference model for native same-model parking and bounded spill.
#include "strata/common/conversations.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
int checks = 0;
void require(bool value, const char* what) { ++checks; if (!value) throw std::runtime_error(what); }
struct Fake {
    struct State { uint64_t carry = 0; int32_t ring[4] = {}; int64_t pos = 0; } state;
    std::vector<int32_t> rows;
    int imports = 0, restores = 0;
    int64_t pos() const { return state.pos; }
    size_t state_bytes() const { return sizeof(State); }
    size_t row_bytes() const { return sizeof(int32_t); }
    void reset() { state = {}; rows.clear(); }
    void eval(int32_t token) {
        require(state.pos >= 0 && state.pos < 512, "position bound");
        if (rows.size() <= size_t(state.pos)) rows.resize(size_t(state.pos) + 1);
        rows[size_t(state.pos)] = token;
        state.ring[size_t(state.pos) % 4] = token;
        state.carry = (state.carry ^ uint32_t(token)) * 1099511628211ull;
        ++state.pos;
    }
    void save_state(uint8_t* p) { std::memcpy(p, &state, sizeof state); }
    void load_state(const uint8_t* p, int64_t at) {
        State saved{}; std::memcpy(&saved, p, sizeof saved);
        require(saved.pos == at, "state position matches checkpoint");
        require(rows.size() >= size_t(at), "KV import precedes state load");
        state = saved; ++restores;
    }
    void save_rows(int64_t first, int64_t n, uint8_t* p) {
        require(first >= 0 && first + n <= (int64_t) rows.size(), "saved rows bounds");
        std::memcpy(p, rows.data() + first, size_t(n) * sizeof(int32_t));
    }
    void load_rows(const uint8_t* p, int64_t from, int64_t count, int64_t first, int64_t n) {
        require(first >= from && first + n <= from + count, "loaded rows bounds");
        rows.resize(size_t(first + n));
        std::memcpy(rows.data() + first, p + size_t(first - from) * sizeof(int32_t), size_t(n) * sizeof(int32_t));
        ++imports;
    }
};
using Cache = strata::conversation::Cache<Fake>;
std::vector<int32_t> tokens(int first, int n) {
    std::vector<int32_t> out(size_t(n), 0); std::iota(out.begin(), out.end(), first); return out;
}
void exact(const Fake& model, const std::vector<int32_t>& seq) {
    Fake reference; for (int32_t x : seq) reference.eval(x);
    require(model.pos() == reference.pos(), "reference position");
    require(model.state.carry == reference.state.carry, "reference convolution/recurrent carry");
    require(std::memcmp(model.state.ring, reference.state.ring, sizeof model.state.ring) == 0, "reference sliding ring");
    require(model.rows.size() >= seq.size() && std::equal(seq.begin(), seq.end(), model.rows.begin()), "reference full attention rows");
}
void bounded(const Cache& c, size_t limit) {
    const auto m = c.memory(); require(m.heap + m.disk + m.reserved <= limit && m.peak <= limit, "hard combined payload/metadata bound");
}
int64_t request(Cache& c, Fake& e, const std::vector<int32_t>& ids, int turn, bool checkpoints = true) {
    const auto at = c.resume(ids, checkpoints);
    require(c.seq().size() == size_t(at), "resume history/state contract");
    uint8_t* mark = turn > at ? c.prepare_mark(turn) : nullptr;
    for (int64_t i = at; i < (int64_t) ids.size(); ++i) {
        if (i == turn && mark) { e.save_state(mark); c.publish_mark(ids); }
        if (i == (int64_t) ids.size() - 1) c.capture(ids);
        e.eval(ids[size_t(i)]);
    }
    c.set_seq(ids); c.release(); exact(e, ids); return at;
}
} // namespace

int main() {
    try {
        strata::conversation::Options o; o.bytes = 1 << 20; o.disk_bytes = o.bytes; o.slots = 2;
        o.max_tokens = 512; o.min_tokens = 1; o.checkpoints = 3;
        Fake e; Cache c(e, o);
        auto a = tokens(1, 40), b = tokens(701, 45);
        require(request(c, e, a, 12) == 0, "first A fresh");
        require(request(c, e, b, 13) == 0, "first B fresh");
        require(c.parked() == 1, "A parked"); bounded(c, o.bytes);
        auto a2 = a; a2.push_back(999);
        require(request(c, e, a2, 12) == 40, "A-B-A restores full committed A");
        require(e.imports > 0, "parked restore imports full KV"); bounded(c, o.bytes);
        auto edit = a2; edit[20] = 123456;
        require(request(c, e, edit, 12) == 12, "edit rewinds recurrent and ring state before changed token");
        auto next = edit; next.push_back(111); next.push_back(222); next.push_back(333);
        const auto reuse = c.resume(next); require(reuse == (int64_t) edit.size(), "live extension");
        uint8_t* saved = c.prepare_mark(reuse + 1); require(saved != nullptr, "reserve cancel checkpoint");
        e.eval(next[size_t(reuse)]); e.save_state(saved); c.publish_mark(next);
        e.state.carry ^= 123; e.state.pos += 1; // a cancelled read advanced only some layers
        require(c.rewind(), "cancel restores only published safe state");
        exact(e, c.seq()); require(e.pos() == reuse + 1, "cancel kept completed checkpoint");
        require(request(c, e, next, 12) == reuse + 1, "cancelled request resumes safely");
        for (int i = 0; i < 5; ++i) { request(c, e, tokens(2000 + i * 100, 50), 15); require(c.parked() <= 2, "slot eviction bound"); bounded(c, o.bytes); }
        // Equal grid length with changed embedding fingerprint must not restore the old image.
        auto image = tokens(5000, 30); image[4] = -1001; request(c, e, image, 3);
        auto other = image; other[4] = -2002; const auto ir = request(c, e, other, 3);
        require(ir <= 3, "different image fingerprint never reuses image rows");
        c.clear(); const auto empty = c.memory(); require(empty.heap == 0 && empty.disk == 0, "clear releases every blob/file");
        require(!c.rewind(), "no checkpoint cancellation resets safely");
        request(c, e, a, 12);
        require(request(c, e, b, 13, false) == 0, "one-shot B fresh");
        require(c.parked() == 1 && c.prepare_mark(10) == nullptr, "one-shot retains A and allocates no B snapshot");
        require(request(c, e, a2, 12) == 40, "A-one-shotB-A keeps A parking");
        require(c.parked() == 0, "one-shot B is never parked");
        bounded(c, o.bytes); c.clear();
        e.reset(); request(c, e, a, 12);
        auto helper = a; helper.push_back(808); helper.push_back(809);
        require(request(c, e, helper, 12, false) == 40, "one-shot can reuse active GPU prefix");
        require(c.parked() == 1 && c.prepare_mark(10) == nullptr, "prefix-extending one-shot preserves old A only");
        require(request(c, e, a2, 12) == 40, "return from prefix-extending helper restores A");
        bounded(c, o.bytes); c.clear();
        // Short prompts keep the existing explicit turn snapshot, but not the
        // additional before-logits/end capture introduced by parking.
        auto short_options = o; short_options.min_tokens = 32;
        Fake short_engine; Cache short_cache(short_engine, short_options);
        auto short_ids = tokens(100, 16);
        require(request(short_cache, short_engine, short_ids, 5) == 0, "short prompt starts fresh");
        require(short_cache.memory().heap == short_engine.state_bytes(), "short prompt keeps explicit turn snapshot only");
        short_cache.capture(short_ids);
        require(short_cache.memory().heap == short_engine.state_bytes(), "short end capture does not allocate");
        require(short_cache.rewind() && short_engine.pos() == 5, "short explicit turn snapshot remains usable");
        exact(short_engine, short_cache.seq());
        auto at_min = tokens(100, 32);
        require(request(short_cache, short_engine, at_min, 5) == 5, "short snapshot resumes through threshold");
        short_cache.capture(at_min);
        require(short_cache.memory().heap == 2 * short_engine.state_bytes(), "automatic capture starts at minimum inclusive");
        bounded(short_cache, short_options.bytes); short_cache.clear();
        require(short_cache.memory().heap == 0, "short snapshot cleanup");
        // Actual immutable file/map roundtrip and destructor accounting.
        auto budget = std::make_shared<strata::conversation::Budget>(4096, 4096, 0);
        auto blob = strata::conversation::Blob::make(budget, 512); require(bool(blob), "blob allocation");
        for (int i = 0; i < 512; ++i) blob->data()[i] = uint8_t(i * 17);
        require(blob->spill(), "spill within combined overlap budget");
        require(budget->stats().heap == 0 && budget->stats().disk == 512, "file payload replaces heap payload");
        for (int pass = 0; pass < 2; ++pass) { const auto* p = blob->data(); require(p != nullptr, "file map"); for (int i = 0; i < 512; ++i) require(p[i] == uint8_t(i * 17), "file bit parity"); blob->unmap(); }
        blob.reset(); require(budget->stats().disk == 0, "delete-on-close file release");
        auto tight = std::make_shared<strata::conversation::Budget>(128, 128, 0);
        auto t = strata::conversation::Blob::make(tight, 96); require(bool(t), "tight allocation");
        std::memset(t->data(), 43, 96); require(!t->spill(), "temporary heap+file overlap never exceeds cap");
        require(t->data()[0] == 43 && tight->stats().heap == 96 && tight->stats().disk == 0, "spill quota failure retains valid heap");
        auto nodisk = std::make_shared<strata::conversation::Budget>(4096, 0, 0);
        auto n = strata::conversation::Blob::make(nodisk, 64); require(bool(n), "no-disk allocation");
        std::memset(n->data(), 7, 64); require(!n->spill() && n->data()[63] == 7, "disk budget exhaustion fallback");
        strata::conversation::Options off; off.max_tokens = 512; Fake f; Cache disabled(f, off);
        require(!disabled.enabled() && disabled.prepare_mark(1) == nullptr, "default off");
        auto undersized_options = o; undersized_options.max_tokens = 262144;
        undersized_options.bytes = undersized_options.disk_bytes = 1 << 20;
        Fake undersized_engine; Cache undersized(undersized_engine, undersized_options);
        require(!undersized.enabled() && undersized.prepare_mark(1) == nullptr, "metadata above budget disables cache");
        undersized.set_seq(a); undersized.capture(a); undersized.release();
        require(undersized.resume(a) == 0 && undersized.seq().empty(), "disabled cache retains no token sequence");
        const auto undersized_memory = undersized.memory();
        require(undersized_memory.heap == 0 && undersized_memory.disk == 0 && undersized_memory.reserved == 0 &&
                undersized_memory.peak <= undersized_options.bytes, "disabled cache has no charged allocation and stays within requested cap");
        // A long conversation whose rows cannot ever fit must not evict a
        // previously parked short conversation merely while trying to save it.
        auto limited_options = o; limited_options.disk_bytes = 0;
        limited_options.bytes = size_t(o.slots + 4) * size_t(o.max_tokens) * sizeof(int32_t) + (size_t(64) << 10) + 1024;
        Fake limited_engine; Cache limited(limited_engine, limited_options);
        require(request(limited, limited_engine, a, 12) == 0, "limited A fresh");
        require(request(limited, limited_engine, tokens(10000, 400), 50) == 0, "oversized B fresh");
        require(limited.parked() == 1, "short A parked before oversized B");
        require(request(limited, limited_engine, b, 13) == 0, "limited C fresh");
        require(limited.parked() == 1, "impossible B rows preserve parked A");
        require(request(limited, limited_engine, a2, 12) == 40, "A survives B allocation failure and C");
        bounded(limited, limited_options.bytes); limited.clear();
        std::cout << "PASS native-conversations " << checks << " checks\n";
        return 0;
    } catch (const std::exception& ex) { std::cerr << "FAIL after " << checks << ": " << ex.what() << '\n'; return 1; }
}
