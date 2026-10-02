#pragma once

// Automatic prefix cache for ContinuousBatchEngine.
//
// Chat and agent clients resend the whole conversation on every turn, so each request's prompt is
// the previous request's prompt plus what happened since. Without a cache the server recomputes
// all of it. An entry here is a prefix that was already computed: its KV blocks, shared with any
// later sequence that starts from it (KVCacheManager's refcounted blocks, no copy), and a snapshot
// of the Gated-DeltaNet recurrent state at the end of it.
//
// The snapshot is why an entry is taken at ONE chosen position rather than at every block. KV can
// be cut at any block boundary after the fact, but a recurrent layer's state is a single running
// value -- it exists only at the position the sequence is at, and copying it costs ~205 MB of
// pinned host memory on Qwen3.8-27B. So the engine snapshots each request at a checkpoint the
// caller picks (the server picks the start of the final assistant turn, which is where the next
// turn's prompt stops matching this one), and the cache keeps the most recently used of those.
//
// HOST TIER (Limits::max_host_kv_bytes). The KV pool holds one --ctx worth of tokens, so it keeps
// only a handful of long prompts: eleven 8K prompts filled the cache's share on a 32 GB card, and
// AIPerf's 8K-prompt cells hit 1% where vLLM, with a larger pool, hits 35%. With a host tier, an
// entry pushed off the device keeps its KV in pinned host memory instead of being dropped, and a
// hit on it copies the blocks back into the new sequence (~270 MB for 8K tokens: milliseconds,
// where prefilling them again takes about half a second). The copies are one kernel each way
// over host-mapped memory, so they need no device staging.
//
// Every method that touches blocks must run under the model's device mutex, the same lock every
// other KVCacheManager mutation already takes. The cache's own mutex only guards its bookkeeping,
// so stats() can be read from an HTTP thread.

#include "sparkinfer/kv_cache.h"
#include "sparkinfer/kv_ops.h"
#include "sparkinfer/models/qwen35.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sparkinfer {

class PrefixCache {
public:
    struct Limits {
        size_t max_entries = 32;
        size_t max_host_bytes = size_t(8) << 30;   // pinned host memory for recurrent snapshots
        int max_blocks = 0;                        // distinct KV blocks entries may hold; 0 = no cap
        size_t max_host_kv_bytes = 0;              // pinned host memory for KV off the device; 0 = none
    };

    struct Hit {
        int tokens = 0;                            // cached prefix length; 0 on a miss
        std::vector<int> blocks;                   // tokens / block_size physical blocks to share
        Qwen35Model::RecurrentStateSnapshot state;
        // The entry's KV is on the host only: `blocks` is empty, and restore_host_hit() copies
        // the KV into a sequence opened without a prefix.
        bool on_host = false;
        uint64_t entry_id = 0;
    };

    struct Stats {
        uint64_t lookups = 0;
        uint64_t hits = 0;
        uint64_t tokens_reused = 0;
        uint64_t inserts = 0;
        uint64_t evictions = 0;
        size_t entries = 0;
        size_t host_bytes = 0;
        int blocks = 0;          // distinct KV blocks held: entries on one conversation share most
        uint64_t host_hits = 0;  // hits served from the host tier
        uint64_t demotions = 0;  // entries moved off the device to the host tier
        size_t host_entries = 0; // entries whose KV is on the host only
        size_t host_kv_bytes = 0;
    };

    PrefixCache(KVCacheManager* kv, const Limits& limits);
    ~PrefixCache();   // releases every block the entries hold
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    // The longest entry that is a PROPER prefix of `prompt` -- strictly shorter, so at least one
    // prompt token is still prefilled and produces the first generated token. Refreshes its LRU
    // position. The returned blocks are not retained by the hit; share them before anything can
    // evict (both happen under the device mutex).
    Hit lookup(const std::vector<int>& prompt);

    // Adopt `blocks` (already retained for this entry) and `state` as the entry for `tokens`, whose
    // length must be a whole number of blocks. A prefix already present, or a malformed entry, is
    // dropped and its blocks released. Evicts least-recently-used entries past the limits.
    void insert(std::vector<int> tokens, std::vector<int> blocks,
                Qwen35Model::RecurrentStateSnapshot state);

    // A host-tier hit: copy the entry's KV into seq_id's first hit.tokens / block_size blocks (the
    // sequence was opened without a prefix) and keep those blocks as the entry's again. False when
    // the entry is gone or the copy failed; the caller then prefills the whole prompt.
    bool restore_host_hit(const Hit& hit, uint64_t seq_id);

    // Free device blocks until the pool has at least need_blocks free or no entry holds any:
    // least-recently-used entries move to the host tier, or are dropped without one. True if
    // anything changed. Blocks still shared with a live sequence do not return to the pool until
    // that sequence finishes, so this can empty the device side and still fall short.
    bool evict_for(int need_blocks);

    // Blocks that evicting every entry would return to the pool: those only cache entries hold.
    int evictable_blocks() const;

    Stats stats() const;

private:
    struct Entry {
        uint64_t id = 0;
        std::vector<int> tokens;
        std::vector<int> blocks;        // device blocks; empty while the KV is on the host only
        std::vector<int> host_blocks;   // host tier copy (kept once made: the KV never changes)
        Qwen35Model::RecurrentStateSnapshot state;
        uint64_t last_used = 0;
    };

    // Entries are named by id across calls that can drop others: dropping erases from entries_,
    // which moves the rest.
    Entry* find_locked(uint64_t id);
    // The least-recently-used entry (other than keep_id): any, one with device blocks, or one
    // whose KV is on the host only. 0 when there is none.
    enum class Pick { ANY, ON_DEVICE, HOST_ONLY };
    uint64_t lru_locked(Pick pick, uint64_t keep_id) const;
    void drop_locked(uint64_t id);
    // Take an entry off the device: to the host tier when there is one and it has room (made by
    // dropping host-only entries), else dropped. Frees its device blocks either way.
    void demote_or_drop_locked(uint64_t id);
    void release_device_locked(Entry& e);
    void release_host_locked(Entry& e);
    void enforce_limits_locked();
    // Host tier: pinned, mapped slabs of whole blocks, allocated as needed up to the limit.
    bool host_tier_on() const { return limits_.max_host_kv_bytes > 0 && host_stride_ > 0; }
    bool host_alloc_locked(int n, std::vector<int>& out, uint64_t keep_id);
    char* host_block(int i) const;
    bool copy_locked(const std::vector<int>& dev, const std::vector<int>& host, int n, bool to_host);

    KVCacheManager* kv_;
    Limits limits_;
    mutable std::mutex mu_;
    std::vector<Entry> entries_;
    uint64_t clock_ = 0;
    Stats stats_;
    size_t host_bytes_ = 0;
    // Entries holding each block. Entries along one conversation share most of their blocks, so a
    // cap on the sum would evict for memory that is not actually used twice.
    std::unordered_map<int, int> held_;
    uint64_t next_id_ = 0;
    // Host tier.
    KVHostCopy layout_;
    size_t host_stride_ = 0;          // bytes per host block, 256-aligned
    int host_cap_ = 0;                // blocks the limit allows
    static constexpr int kSlabBlocks = 64;
    std::vector<char*> slabs_;
    std::vector<int> host_free_;
    int host_used_ = 0;
    int* d_ids_ = nullptr;            // device copies of one transfer's block ids and host pointers
    char** d_ptrs_ = nullptr;
    int d_cap_ = 0;
    cudaStream_t stream_ = nullptr;
};

}  // namespace sparkinfer
