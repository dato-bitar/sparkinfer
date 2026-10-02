#pragma once
#include <cuda_runtime.h>

namespace sparkinfer {

// Append one new token's K/V into the paged cache for each sequence.
//   k_new/v_new:  [num_seqs, num_kv_heads, head_dim]   (bf16)
//   k_pool/v_pool base of this layer's sub-pool (caller adds layer offset)
//   block_table:  [num_seqs, max_blocks_per_seq]       (int32, device)
//   write_pos:    [num_seqs]  position index for the new token (= old seq_len)
void launch_kv_append(
    void* k_pool, void* v_pool,
    const void* k_new, const void* v_new,
    const int* block_table, const int* write_pos,
    int num_seqs, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    cudaStream_t stream = nullptr);

// Elementwise residual add: out[i] = a[i] + b[i]  (bf16). out may alias a.
void launch_residual_add(const void* a, const void* b, void* out, int n,
                         cudaStream_t stream = nullptr);

// Whole KV blocks between the device pool and host-mapped (cudaHostAllocMapped) memory, in one
// launch and with no device staging. Block i is device block dev_blocks[i] in every slot, and
// host_blocks[i] on the host, laid out per slot: its K bytes, its V bytes, then (int8 pools) its K
// and V scales -- kv_host_block_bytes() in all. dev_blocks and host_blocks are device-readable
// arrays of n entries.
struct KVHostCopy {
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    void* k_scale = nullptr;          // null without int8 scales
    void* v_scale = nullptr;
    size_t slot_stride_bytes = 0;     // between consecutive slots' K (or V) sub-pools
    size_t scale_slot_stride_bytes = 0;
    size_t block_bytes = 0;           // one block's K (or V) in one slot
    size_t scale_block_bytes = 0;     // one block's K (or V) scales in one slot; 0 without
    int slots = 0;
};
inline size_t kv_host_block_bytes(const KVHostCopy& p) {
    return (size_t)p.slots * 2 * (p.block_bytes + p.scale_block_bytes);
}
// to_host: device -> host; else host -> device. False when the launch fails.
bool launch_kv_blocks_host_copy(const KVHostCopy& p, const int* dev_blocks, char* const* host_blocks,
                                int n, bool to_host, cudaStream_t stream);

} // namespace sparkinfer
