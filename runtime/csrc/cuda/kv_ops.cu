// KV-cache append and residual add — small device ops the runtime uses to wire
// attention + MoE into a decode step.
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cstdint>
#include <cuda_runtime.h>
#include "sparkinfer/kv_ops.h"
#endif

namespace sparkinfer {

// grid = (num_seqs, num_kv_heads); blockDim = head_dim (<=1024).
__global__ void kv_append_kernel(
    __nv_bfloat16* __restrict__ k_pool, __nv_bfloat16* __restrict__ v_pool,
    const __nv_bfloat16* __restrict__ k_new, const __nv_bfloat16* __restrict__ v_new,
    const int* __restrict__ block_table, const int* __restrict__ write_pos,
    int num_kv_heads, int head_dim, int block_size, int max_blocks_per_seq
) {
    const int seq = blockIdx.x;
    const int h   = blockIdx.y;
    const int d   = threadIdx.x;
    if (d >= head_dim) return;

    const int pos    = write_pos[seq];
    const int blk    = pos / block_size;
    const int within = pos % block_size;
    const int phys   = block_table[seq * max_blocks_per_seq + blk];
    const size_t dst = ((size_t)(phys * block_size + within) * num_kv_heads + h) * head_dim + d;
    const size_t src = ((size_t)seq * num_kv_heads + h) * head_dim + d;
    k_pool[dst] = k_new[src];
    v_pool[dst] = v_new[src];
}

__global__ void residual_add_kernel(const __nv_bfloat16* __restrict__ a,
                                    const __nv_bfloat16* __restrict__ b,
                                    __nv_bfloat16* __restrict__ out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

// grid = (blocks, slots). One region of `bytes` between a device and a host-mapped pointer; VEC
// when both are 16-byte aligned and the size a multiple of 16.
template <bool VEC>
__device__ __forceinline__ void kv_region_copy(char* dst, const char* src, size_t bytes) {
    if (VEC) {
        uint4* d = reinterpret_cast<uint4*>(dst);
        const uint4* s = reinterpret_cast<const uint4*>(src);
        for (size_t i = threadIdx.x; i < bytes / 16; i += blockDim.x) d[i] = s[i];
    } else {
        for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) dst[i] = src[i];
    }
}

template <bool VEC>
__global__ void kv_blocks_host_copy_kernel(char* k_pool, char* v_pool, char* k_scale, char* v_scale,
                                           size_t slot_stride, size_t scale_slot_stride,
                                           size_t block_bytes, size_t scale_block_bytes,
                                           const int* __restrict__ dev_blocks,
                                           char* const* __restrict__ host_blocks, bool to_host) {
    const int i = blockIdx.x, s = blockIdx.y;
    const size_t b = (size_t)dev_blocks[i];
    char* host = host_blocks[i] + (size_t)s * 2 * (block_bytes + scale_block_bytes);
    char* dev[4] = { k_pool + s * slot_stride + b * block_bytes, v_pool + s * slot_stride + b * block_bytes,
                     k_scale ? k_scale + s * scale_slot_stride + b * scale_block_bytes : nullptr,
                     v_scale ? v_scale + s * scale_slot_stride + b * scale_block_bytes : nullptr };
    const size_t len[4] = { block_bytes, block_bytes, scale_block_bytes, scale_block_bytes };
    for (int r = 0; r < 4; ++r) {
        if (!dev[r] || !len[r]) continue;
        if (to_host) kv_region_copy<VEC>(host, dev[r], len[r]);
        else kv_region_copy<VEC>(dev[r], host, len[r]);
        host += len[r];
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
bool launch_kv_blocks_host_copy(const KVHostCopy& p, const int* dev_blocks, char* const* host_blocks,
                                int n, bool to_host, cudaStream_t stream) {
    if (n <= 0) return true;
    if (!p.k_pool || !p.v_pool || p.slots <= 0 || !dev_blocks || !host_blocks) return false;
    const bool vec = !(p.block_bytes % 16) && !(p.scale_block_bytes % 16) && !(p.slot_stride_bytes % 16) &&
                     !(p.scale_slot_stride_bytes % 16) && !((uintptr_t)p.k_pool % 16) &&
                     !((uintptr_t)p.v_pool % 16) && !((uintptr_t)p.k_scale % 16) && !((uintptr_t)p.v_scale % 16);
    const dim3 grid((unsigned)n, (unsigned)p.slots);
    char* kp = static_cast<char*>(p.k_pool);
    char* vp = static_cast<char*>(p.v_pool);
    char* ks = static_cast<char*>(p.k_scale);
    char* vs = static_cast<char*>(p.v_scale);
    if (vec)
        kv_blocks_host_copy_kernel<true><<<grid, 256, 0, stream>>>(
            kp, vp, ks, vs, p.slot_stride_bytes, p.scale_slot_stride_bytes, p.block_bytes,
            p.scale_block_bytes, dev_blocks, host_blocks, to_host);
    else
        kv_blocks_host_copy_kernel<false><<<grid, 256, 0, stream>>>(
            kp, vp, ks, vs, p.slot_stride_bytes, p.scale_slot_stride_bytes, p.block_bytes,
            p.scale_block_bytes, dev_blocks, host_blocks, to_host);
    return cudaPeekAtLastError() == cudaSuccess;
}

void launch_kv_append(void* k_pool, void* v_pool, const void* k_new, const void* v_new,
                      const int* block_table, const int* write_pos,
                      int num_seqs, int num_kv_heads, int head_dim,
                      int block_size, int max_blocks_per_seq, cudaStream_t stream) {
    dim3 grid(num_seqs, num_kv_heads);
    kv_append_kernel<<<grid, head_dim, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(k_pool), reinterpret_cast<__nv_bfloat16*>(v_pool),
        reinterpret_cast<const __nv_bfloat16*>(k_new), reinterpret_cast<const __nv_bfloat16*>(v_new),
        block_table, write_pos, num_kv_heads, head_dim, block_size, max_blocks_per_seq);
}

void launch_residual_add(const void* a, const void* b, void* out, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    residual_add_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a), reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_bfloat16*>(out), n);
}
#endif

} // namespace sparkinfer
