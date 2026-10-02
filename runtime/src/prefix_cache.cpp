#include "sparkinfer/prefix_cache.h"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace sparkinfer {

PrefixCache::PrefixCache(KVCacheManager* kv, const Limits& limits) : kv_(kv), limits_(limits) {
    if (limits_.max_host_kv_bytes == 0 || !kv_ || kv_->windowed() || kv_->kv_slots() <= 0) return;
    const size_t eb = kv_->int8_kv() ? 1 : 2;
    layout_.k_pool = kv_->k_pool();
    layout_.v_pool = kv_->v_pool();
    layout_.slot_stride_bytes = kv_->layer_stride_elems() * eb;
    layout_.block_bytes = kv_->block_elems() * eb;
    if (kv_->int8_kv()) {
        layout_.k_scale = kv_->k_scale_pool();
        layout_.v_scale = kv_->v_scale_pool();
        layout_.scale_slot_stride_bytes = kv_->scale_layer_stride_elems() * 2;
        layout_.scale_block_bytes = kv_->block_scale_elems() * 2;
    }
    layout_.slots = kv_->kv_slots();
    host_stride_ = (kv_host_block_bytes(layout_) + 255) & ~(size_t)255;
    host_cap_ = host_stride_ ? (int)std::min<size_t>(limits_.max_host_kv_bytes / host_stride_, 1 << 30) : 0;
    if (host_cap_ <= 0) host_stride_ = 0;
}

PrefixCache::~PrefixCache() {
    std::lock_guard<std::mutex> lock(mu_);
    for (const Entry& e : entries_) kv_->release_blocks(e.blocks);
    entries_.clear();
    for (char* s : slabs_) cudaFreeHost(s);
    if (d_ids_) cudaFree(d_ids_);
    if (d_ptrs_) cudaFree(d_ptrs_);
    if (stream_) cudaStreamDestroy(stream_);
}

PrefixCache::Entry* PrefixCache::find_locked(uint64_t id) {
    for (Entry& e : entries_)
        if (e.id == id) return &e;
    return nullptr;
}

PrefixCache::Hit PrefixCache::lookup(const std::vector<int>& prompt) {
    std::lock_guard<std::mutex> lock(mu_);
    Hit hit;
    stats_.lookups++;
    Entry* best = nullptr;
    for (Entry& e : entries_) {
        if (e.tokens.size() >= prompt.size()) continue;
        if (best && e.tokens.size() <= best->tokens.size()) continue;
        if (!std::equal(e.tokens.begin(), e.tokens.end(), prompt.begin())) continue;
        best = &e;
    }
    if (!best) return hit;
    best->last_used = ++clock_;
    stats_.hits++;
    stats_.tokens_reused += best->tokens.size();
    hit.tokens = (int)best->tokens.size();
    hit.blocks = best->blocks;
    hit.state = best->state;   // shares the pinned buffer; nothing ever writes to a stored snapshot
    hit.entry_id = best->id;
    hit.on_host = best->blocks.empty();
    if (hit.on_host) stats_.host_hits++;
    return hit;
}

bool PrefixCache::restore_host_hit(const Hit& hit, uint64_t seq_id) {
    std::lock_guard<std::mutex> lock(mu_);
    const int bs = kv_->block_size();
    Entry* e = find_locked(hit.entry_id);
    if (!e || bs <= 0 || hit.tokens % bs) return false;
    const int n = hit.tokens / bs;
    if (!e->blocks.empty() || (int)e->host_blocks.size() != n) return false;
    const std::vector<int>& phys = kv_->physical_block_ids(seq_id);
    if ((int)phys.size() < n) return false;
    if (!copy_locked(phys, e->host_blocks, n, /*to_host=*/false)) return false;
    // The sequence's first blocks now hold the prefix: the entry keeps them, as an entry that was
    // never off the device keeps the blocks it was inserted with.
    e->blocks = kv_->retain_prefix_blocks(seq_id, n);
    if ((int)e->blocks.size() != n) {
        kv_->release_blocks(e->blocks);
        e->blocks.clear();
        return false;
    }
    for (int b : e->blocks) held_[b]++;
    enforce_limits_locked();
    return true;
}

void PrefixCache::insert(std::vector<int> tokens, std::vector<int> blocks,
                         Qwen35Model::RecurrentStateSnapshot state) {
    std::lock_guard<std::mutex> lock(mu_);
    const size_t bs = (size_t)kv_->block_size();
    const bool well_formed = !tokens.empty() && bs > 0 && tokens.size() % bs == 0 &&
                             blocks.size() * bs == tokens.size();
    Entry* duplicate = nullptr;
    for (Entry& e : entries_) {
        if (e.tokens.size() == tokens.size() && e.tokens == tokens) {
            e.last_used = ++clock_;
            duplicate = &e;
            break;
        }
    }
    // A duplicate whose KV is on the host only takes these blocks: it is back on the device.
    if (well_formed && duplicate && duplicate->blocks.empty()) {
        duplicate->blocks = std::move(blocks);
        for (int b : duplicate->blocks) held_[b]++;
        enforce_limits_locked();
        return;
    }
    if (!well_formed || duplicate || state.bytes() > limits_.max_host_bytes) {
        kv_->release_blocks(blocks);
        return;
    }
    Entry e;
    e.id = ++next_id_;
    e.tokens = std::move(tokens);
    e.blocks = std::move(blocks);
    e.state = std::move(state);
    e.last_used = ++clock_;
    host_bytes_ += e.state.bytes();
    for (int b : e.blocks) held_[b]++;
    entries_.push_back(std::move(e));
    stats_.inserts++;
    enforce_limits_locked();
}

bool PrefixCache::evict_for(int need_blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    bool changed = false;
    while (kv_->num_free_blocks() < need_blocks) {
        const uint64_t id = lru_locked(Pick::ON_DEVICE, 0);
        if (!id) break;
        demote_or_drop_locked(id);
        changed = true;
    }
    return changed;
}

int PrefixCache::evictable_blocks() const {
    std::lock_guard<std::mutex> lock(mu_);
    int n = 0;
    for (const auto& kv : held_)
        if (kv_->block_refs(kv.first) == kv.second) ++n;
    return n;
}

PrefixCache::Stats PrefixCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    Stats s = stats_;
    s.entries = entries_.size();
    s.host_bytes = host_bytes_;
    s.blocks = (int)held_.size();
    s.host_entries = 0;
    for (const Entry& e : entries_) s.host_entries += e.blocks.empty();
    s.host_kv_bytes = (size_t)host_used_ * host_stride_;
    return s;
}

void PrefixCache::release_device_locked(Entry& e) {
    kv_->release_blocks(e.blocks);
    for (int b : e.blocks) {
        auto it = held_.find(b);
        if (it != held_.end() && --it->second == 0) held_.erase(it);
    }
    e.blocks.clear();
}

void PrefixCache::release_host_locked(Entry& e) {
    for (int i : e.host_blocks) host_free_.push_back(i);
    host_used_ -= (int)e.host_blocks.size();
    e.host_blocks.clear();
}

uint64_t PrefixCache::lru_locked(Pick pick, uint64_t keep_id) const {
    const Entry* lru = nullptr;
    for (const Entry& e : entries_) {
        if (e.id == keep_id) continue;
        if (pick == Pick::ON_DEVICE && e.blocks.empty()) continue;
        if (pick == Pick::HOST_ONLY && !e.blocks.empty()) continue;
        if (!lru || e.last_used < lru->last_used) lru = &e;
    }
    return lru ? lru->id : 0;
}

void PrefixCache::drop_locked(uint64_t id) {
    for (size_t i = 0; i < entries_.size(); ++i) {
        Entry& e = entries_[i];
        if (e.id != id) continue;
        release_device_locked(e);
        release_host_locked(e);
        host_bytes_ -= e.state.bytes();
        entries_.erase(entries_.begin() + (std::ptrdiff_t)i);
        stats_.evictions++;
        return;
    }
}

void PrefixCache::demote_or_drop_locked(uint64_t id) {
    Entry* e = find_locked(id);
    if (!e) return;
    if (host_tier_on() && e->host_blocks.empty()) {
        const int n = (int)e->blocks.size();
        std::vector<int> hb;
        // May drop other entries (host-only ones), which moves this one: look it up again.
        const bool got = host_alloc_locked(n, hb, id);
        e = find_locked(id);
        if (!e) return;
        if (got) {
            e->host_blocks = std::move(hb);
            if (!copy_locked(e->blocks, e->host_blocks, n, /*to_host=*/true)) release_host_locked(*e);
        }
    }
    if (e->host_blocks.empty()) {   // no host tier, no room in it, or the copy failed
        drop_locked(id);
        return;
    }
    release_device_locked(*e);
    stats_.demotions++;
}

void PrefixCache::enforce_limits_locked() {
    // Entry count and snapshot memory: an entry costs those wherever its KV is, so drop.
    while (!entries_.empty() &&
           (entries_.size() > limits_.max_entries || host_bytes_ > limits_.max_host_bytes))
        drop_locked(lru_locked(Pick::ANY, 0));
    // Device blocks: move entries off the device.
    while (limits_.max_blocks > 0 && (int)held_.size() > limits_.max_blocks) {
        const uint64_t id = lru_locked(Pick::ON_DEVICE, 0);
        if (!id) break;
        demote_or_drop_locked(id);
    }
}

char* PrefixCache::host_block(int i) const {
    return slabs_[(size_t)(i / kSlabBlocks)] + (size_t)(i % kSlabBlocks) * host_stride_;
}

bool PrefixCache::host_alloc_locked(int n, std::vector<int>& out, uint64_t keep_id) {
    if (n <= 0 || n > host_cap_) return false;
    // Room within the limit: drop the oldest entries whose KV is on the host only.
    while (host_used_ + n > host_cap_) {
        const uint64_t id = lru_locked(Pick::HOST_ONLY, keep_id);
        if (!id) return false;
        drop_locked(id);
    }
    // Grow the pinned pool a slab at a time. A failed allocation just means no host copy.
    while ((int)host_free_.size() < n) {
        const int have = (int)slabs_.size() * kSlabBlocks;
        if (have >= host_cap_) return false;
        void* p = nullptr;
        if (cudaHostAlloc(&p, (size_t)kSlabBlocks * host_stride_,
                          cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            fprintf(stderr, "[prefix-cache] host tier: pinned allocation failed at %.1f GB; it stays "
                            "at that size\n", (double)have * host_stride_ / 1e9);
            host_cap_ = have;
            return false;
        }
        slabs_.push_back(static_cast<char*>(p));
        for (int i = kSlabBlocks - 1; i >= 0; --i) host_free_.push_back(have + i);
    }
    out.assign(host_free_.end() - n, host_free_.end());
    host_free_.resize(host_free_.size() - (size_t)n);
    host_used_ += n;
    return true;
}

bool PrefixCache::copy_locked(const std::vector<int>& dev, const std::vector<int>& host, int n, bool to_host) {
    if (n <= 0) return true;
    if ((int)dev.size() < n || (int)host.size() < n) return false;
    if (!stream_ && cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        stream_ = nullptr;
        return false;
    }
    if (n > d_cap_) {
        if (d_ids_) cudaFree(d_ids_);
        if (d_ptrs_) cudaFree(d_ptrs_);
        d_ids_ = nullptr;
        d_ptrs_ = nullptr;
        d_cap_ = 0;
        const int cap = std::max(n, kv_->max_blocks_per_seq());
        if (cudaMalloc(&d_ids_, (size_t)cap * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_ptrs_, (size_t)cap * sizeof(char*)) != cudaSuccess) {
            cudaGetLastError();
            if (d_ids_) cudaFree(d_ids_);
            d_ids_ = nullptr;
            d_ptrs_ = nullptr;
            return false;
        }
        d_cap_ = cap;
    }
    std::vector<char*> ptrs((size_t)n);
    for (int i = 0; i < n; ++i) {
        void* d = nullptr;
        // Mapped pinned memory: its device address (the same pointer under UVA).
        if (cudaHostGetDevicePointer(&d, host_block(host[(size_t)i]), 0) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        ptrs[(size_t)i] = static_cast<char*>(d);
    }
    bool ok = cudaMemcpyAsync(d_ids_, dev.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice,
                              stream_) == cudaSuccess &&
              cudaMemcpyAsync(d_ptrs_, ptrs.data(), (size_t)n * sizeof(char*), cudaMemcpyHostToDevice,
                              stream_) == cudaSuccess &&
              launch_kv_blocks_host_copy(layout_, d_ids_, d_ptrs_, n, to_host, stream_);
    ok = (cudaStreamSynchronize(stream_) == cudaSuccess) && ok;
    if (!ok) cudaGetLastError();
    return ok;
}

}  // namespace sparkinfer
