// Same-model native conversation parking. Algorithm adapted from PolyStrata
// conversations.hpp at 6f279e9f4b82ce9c20a27df02b72601e690ce4ea (MIT).
// Model identity is the owning engine lifetime; no cross-model persistence.
#pragma once
#include "strata/common/conversation_spill.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::conversation {
struct Options {
    size_t bytes = 0, disk_bytes = 0;
    int slots = 4, checkpoints = 3;
    int64_t min_tokens = 1024, max_tokens = 262144;
};

template<class E> class Cache {
    using BlobPtr = std::shared_ptr<Blob>;
    struct Mark { int64_t pos = 0; BlobPtr state; };
    struct Entry {
        std::vector<int32_t> seq;
        std::vector<Mark> marks;
        BlobPtr rows;
        int64_t row_count = 0;
        uint64_t stamp = 0;
        bool allow_checkpoint = true;
    };
public:
    Cache(E& engine, Options options) : e_(engine), o_(options) {
        if (o_.slots < 1 || o_.slots > 16 || o_.checkpoints < 1 || o_.checkpoints > 8 ||
            o_.min_tokens < 0 || o_.max_tokens < 1 || o_.max_tokens > 262144)
            throw std::invalid_argument("conversation cache options outside supported range");
        // Bound token-vector copies and metadata independently of payload sizes.
        // At most slots parked entries, current, selected and transient copies.
        const size_t metadata = size_t(o_.slots + 4) * size_t(o_.max_tokens) * sizeof(int32_t) + (size_t(64) << 10);
        if (o_.bytes > 0 && metadata > o_.bytes) {
            std::fprintf(stderr, "tinystrata: conversation cache disabled: requested %zu bytes is below %zu bytes of metadata for %lld tokens and %d slots\n",
                         o_.bytes, metadata, (long long) o_.max_tokens, o_.slots);
            o_.bytes = 0; o_.disk_bytes = 0;
        }
        budget_ = std::make_shared<Budget>(o_.bytes, o_.disk_bytes, metadata);
    }
    bool enabled() const { return o_.bytes > 0; }
    const std::vector<int32_t>& seq() const { return cur_.seq; }
    MemoryStats memory() const { return budget_->stats(); }
    size_t parked() const { return parked_.size(); }
    void clear() { cur_ = {}; parked_.clear(); pending_ = {}; }
    void drop_current() { cur_ = {}; pending_ = {}; }
    void set_seq(const std::vector<int32_t>& ids) {
        if (!enabled()) return;
        if (ids.size() > size_t(o_.max_tokens)) throw std::logic_error("conversation token budget exceeded");
        cur_.seq = ids;
        trim_marks((int64_t) ids.size());
    }

    // Reserve first; the caller publishes only a fully completed state snapshot.
    uint8_t* prepare_mark(int64_t at) {
        pending_ = {};
        if (!enabled() || !cur_.allow_checkpoint || at < 1 || at > o_.max_tokens) return nullptr;
        auto b = allocate(e_.state_bytes());
        if (!b) return nullptr;
        pending_ = {at, b}; return b->data();
    }
    void publish_mark(const std::vector<int32_t>& ids) {
        if (!pending_.state) return;
        if (pending_.pos > (int64_t) ids.size()) throw std::logic_error("checkpoint outside prompt");
        if (cur_.seq.size() < size_t(pending_.pos)) cur_.seq.assign(ids.begin(), ids.begin() + pending_.pos);
        add_mark(pending_); pending_ = {};
    }
    void discard_mark() { pending_ = {}; }
    void capture(const std::vector<int32_t>& ids) {
        // Short turns retain only their explicit UI turn checkpoint. Avoid the
        // additional logits/end snapshots until parking can actually use them.
        if (!enabled() || !cur_.allow_checkpoint || e_.pos() < 1 || e_.pos() < o_.min_tokens ||
            e_.pos() > (int64_t) ids.size()) return;
        const int64_t at = e_.pos();
        if (std::any_of(cur_.marks.begin(), cur_.marks.end(), [&](const Mark& m) { return m.pos == at; })) return;
        auto b = allocate(e_.state_bytes());
        if (!b) return;
        e_.save_state(b->data());
        if (cur_.seq.size() < size_t(at)) cur_.seq.assign(ids.begin(), ids.begin() + at);
        add_mark({at, std::move(b)});
    }

    // Restoring rows maps/covers VMM before restoring recurrent/ring/MTP state.
    int64_t resume(const std::vector<int32_t>& ids, bool allow_checkpoint = true) {
        if (!enabled()) return 0;
        if (ids.empty() || ids.size() > size_t(o_.max_tokens)) throw std::logic_error("invalid cached prompt size");
        pending_ = {};
        if (e_.pos() != (int64_t) cur_.seq.size()) cur_ = {};
        if (!cur_.seq.empty() && cur_.seq.size() < ids.size() && prefix(cur_.seq, ids) == (int64_t) cur_.seq.size()) {
            // A helper can extend the active chat while opting out of its own
            // snapshots. Preserve the old chat before it changes the KV suffix.
            if (!allow_checkpoint && cur_.allow_checkpoint) park_current();
            request_policy(allow_checkpoint); ++hits_; reused_ += cur_.seq.size(); return cur_.seq.size();
        }
        int best = -1; int64_t at = reach(cur_, ids);
        for (size_t i = 0; i < parked_.size(); ++i) {
            const int64_t p = reach(parked_[i], ids);
            if (p > at) { best = (int) i; at = p; }
        }
        Entry selected;
        if (at > 0) {
            selected = best < 0 ? cur_ : parked_[size_t(best)];
            if (best >= 0) parked_.erase(parked_.begin() + best);
        }
        park_current();
        if (at > 0) {
            const auto mark = std::find_if(selected.marks.begin(), selected.marks.end(), [&](const Mark& m) { return m.pos == at; });
            const uint8_t* state = mark == selected.marks.end() ? nullptr : mark->state->data();
            const uint8_t* rows = best < 0 ? nullptr : selected.rows ? selected.rows->data() : nullptr;
            if (state && (best < 0 || (rows && at <= selected.row_count))) {
                if (best >= 0) { e_.reset(); e_.load_rows(rows, 0, selected.row_count, 0, at); }
                e_.load_state(state, at);
                cur_ = std::move(selected); cur_.seq.resize(size_t(at)); cur_.rows.reset(); cur_.row_count = 0;
                trim_marks(at); cur_.stamp = ++clock_;
                request_policy(allow_checkpoint);
                ++hits_; reused_ += size_t(at); say("restored", at); return at;
            }
        }
        cur_ = {}; request_policy(allow_checkpoint); e_.reset(); return 0;
    }

    // A failed layer-major read may have advanced only some layers. Only the
    // last published checkpoint is reusable; never capture its current state.
    bool rewind() {
        pending_ = {};
        const bool allow = cur_.allow_checkpoint;
        if (cur_.marks.empty()) { cur_ = {}; request_policy(allow); e_.reset(); return false; }
        const Mark m = cur_.marks.back();
        const auto* data = m.state->data();
        if (!data || m.pos > (int64_t) cur_.seq.size()) { cur_ = {}; request_policy(allow); e_.reset(); return false; }
        e_.load_state(data, m.pos); cur_.seq.resize(size_t(m.pos)); trim_marks(m.pos); return true;
    }
    void release() {
        // One-shot/probe requests do not introduce storage work of their own.
        if (!cur_.allow_checkpoint) return;
        for (auto& p : parked_) {
            if (p.rows) p.rows->spill();
            for (auto& m : p.marks) {
                const bool active = std::any_of(cur_.marks.begin(), cur_.marks.end(), [&](const Mark& c) { return c.state == m.state; });
                if (!active) m.state->spill();
            }
        }
    }
    void info() const {
        const auto m = memory();
        std::printf("INFO conversation_cache_mib=%zu conversation_cache_slots=%d conversation_cache_parked=%zu "
                    "conversation_cache_heap_bytes=%zu conversation_cache_disk_bytes=%zu conversation_cache_metadata_bytes=%zu "
                    "conversation_cache_used_bytes=%zu conversation_cache_peak_bytes=%zu conversation_cache_spill_failures=%zu "
                    "conversation_cache_hits=%llu conversation_cache_reused_tokens=%llu conversation_cache_evictions=%zu "
                    "conversation_cache_state_bytes=%zu conversation_cache_row_bytes=%zu\n",
                    o_.bytes >> 20, o_.slots, parked_.size(), m.heap, m.disk, m.reserved, m.heap + m.disk + m.reserved,
                    m.peak, m.spill_failures, (unsigned long long) hits_, (unsigned long long) reused_, evictions_,
                    e_.state_bytes(), e_.row_bytes());
    }
private:
    void request_policy(bool allow) {
        cur_.allow_checkpoint = allow;
        if (!allow) { cur_.marks.clear(); cur_.rows.reset(); cur_.row_count = 0; pending_ = {}; }
    }
    static int64_t prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        size_t n = 0; while (n < a.size() && n < b.size() && a[n] == b[n]) ++n; return (int64_t) n;
    }
    int64_t reach(const Entry& s, const std::vector<int32_t>& ids) const {
        const auto c = std::min<int64_t>(prefix(s.seq, ids), (int64_t) ids.size() - 1);
        int64_t at = 0; for (const auto& m : s.marks) if (m.pos <= c) at = std::max(at, m.pos); return at;
    }
    void trim_marks(int64_t at) {
        cur_.marks.erase(std::remove_if(cur_.marks.begin(), cur_.marks.end(), [&](const Mark& m) { return m.pos > at; }), cur_.marks.end());
    }
    void add_mark(Mark mark) {
        cur_.marks.erase(std::remove_if(cur_.marks.begin(), cur_.marks.end(), [&](const Mark& m) { return m.pos == mark.pos; }), cur_.marks.end());
        cur_.marks.push_back(std::move(mark));
        std::sort(cur_.marks.begin(), cur_.marks.end(), [](const Mark& a, const Mark& b) { return a.pos < b.pos; });
        while (cur_.marks.size() > size_t(o_.checkpoints)) cur_.marks.erase(cur_.marks.begin());
    }
    bool evict() {
        if (parked_.empty()) return false;
        const auto i = std::min_element(parked_.begin(), parked_.end(), [](const Entry& a, const Entry& b) { return a.stamp < b.stamp; });
        parked_.erase(i); ++evictions_; return true;
    }
    BlobPtr allocate(size_t bytes) {
        // Eviction cannot help a single blob that exceeds the whole payload
        // allowance. Preserve other conversations when this one cannot fit.
        if (!bytes || bytes > budget_->limit() - budget_->stats().reserved) return nullptr;
        for (;;) { if (auto b = Blob::make(budget_, bytes)) return b; if (!evict()) return nullptr; }
    }
    void park_current() {
        if (!cur_.allow_checkpoint || cur_.seq.empty() || cur_.seq.size() < size_t(o_.min_tokens) || e_.pos() != (int64_t) cur_.seq.size()) return;
        capture(cur_.seq);
        if (cur_.marks.empty()) return;
        const int64_t end = cur_.marks.back().pos;
        const size_t row = e_.row_bytes();
        if (row && size_t(end) > std::numeric_limits<size_t>::max() / row) return;
        auto rows = allocate(size_t(end) * row);
        // Architectures without attention rows can still preserve recurrent state.
        if (row && !rows) { say("skipped", end); return; }
        if (rows) e_.save_rows(0, end, rows->data());
        Entry saved = cur_; saved.seq.resize(size_t(end)); saved.rows = std::move(rows); saved.row_count = end; saved.stamp = ++clock_;
        while (parked_.size() >= size_t(o_.slots)) evict();
        parked_.push_back(std::move(saved)); say("parked", end);
    }
    void say(const char* action, int64_t tokens) const {
        const auto m = memory();
        std::fprintf(stderr, "tinystrata: conversation cache: %s %lld tokens; parked=%zu bytes=%zu evictions=%zu heap=%zu disk=%zu peak=%zu\n",
                     action, (long long) tokens, parked_.size(), m.heap + m.disk, evictions_, m.heap, m.disk, m.peak);
    }
    E& e_; Options o_; std::shared_ptr<Budget> budget_;
    Entry cur_; std::vector<Entry> parked_; Mark pending_;
    uint64_t clock_ = 0, hits_ = 0, reused_ = 0; size_t evictions_ = 0;
};
} // namespace strata::conversation
