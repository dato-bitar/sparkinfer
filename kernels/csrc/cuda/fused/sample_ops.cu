// Gumbel-max temperature sampling: mutates logits in place before argmax, using cuRAND's
// Philox4_32_10 device API (header-only, stateless-per-call -- no persistent curandState needed
// across CUDA graph replays; the caller re-derives the same draw from (seed, row*vocab+v, step)
// every launch). See fused.h for the full design rationale (why temp/seed/step are read from
// device memory on every launch rather than gating the kernel launch itself on temperature).
//
// Sampling from softmax(logits/T) is equivalent to argmax_v(logits[v]/T + G_v), G_v =
// -log(-log(U_v)), U_v ~ Uniform(0,1] i.i.d. per vocab entry (the standard Gumbel-max trick) --
// so this kernel is a thin elementwise transform feeding the existing, unmodified launch_argmax,
// mirroring how launch_logit_softcap already mutates logits in place immediately before argmax.

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "sparkinfer/kernels/fused.h"

namespace sparkinfer {
namespace kernels {

// One block per row, grid-stride over vocab -- same layout as logit_softcap_kernel.
__global__ void temperature_sample_kernel(float* __restrict__ logits, int vocab,
                                          const float* __restrict__ temp_f32,
                                          const unsigned long long* __restrict__ seed_u64,
                                          const unsigned long long* __restrict__ step_u64) {
    const float T = *temp_f32;
    if (T <= 0.f) return;  // greedy: leave logits untouched, byte-identical to plain argmax

    const unsigned long long seed = *seed_u64;
    const unsigned long long step = *step_u64;
    float* L = logits + (size_t)blockIdx.y * vocab;
    const float inv_t = 1.f / T;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab; v += gridDim.x * blockDim.x) {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, (unsigned long long)((size_t)blockIdx.y * vocab + v), step, &st);
        // curand_uniform is (0, 1], and u == 1 is not harmless: -log(1) is -0, log(-0) is -inf, and the
        // noise comes out +inf -- that token wins the argmax whatever its logit, a logit_bias of -100
        // or a constrained-decoding mask included. At 2^-24 per draw and 248K draws per step it hit
        // about one sampled step in 70. Clamp to the largest float below 1 (noise ~16.6, finite).
        const float u = fminf(curand_uniform(&st), 0.99999994f);
        const float g = -logf(-logf(u));      // Gumbel(0,1), finite for every u in (0, 1)
        L[v] = L[v] * inv_t + g;
    }
}

void launch_temperature_sample(float* logits, int n_rows, int vocab,
                               const float* temp_f32, const unsigned long long* seed_u64,
                               const unsigned long long* step_u64, cudaStream_t stream) {
    const int bx = (vocab + 255) / 256 > 1024 ? 1024 : (vocab + 255) / 256;
    temperature_sample_kernel<<<dim3(bx < 1 ? 1 : bx, n_rows), 256, 0, stream>>>(
        logits, vocab, temp_f32, seed_u64, step_u64);
}

// ---- batched top-k/top-p sampling (see fused.h, launch_sample_rows_topk) --------------------
//
// Reproduces launch_topk_topp_mask -> launch_temperature_sample -> launch_argmax, each launched
// with n_rows = 1 on the row, without touching the logits: every entry outside the top-k is
// masked to -inf there and can never win the argmax, so only the k candidates need noise.
//
// 1. Each thread takes the max key of its strided slice. The k-th largest of those 1024 maxima
//    is a lower bound on the row's k-th largest entry (they are k distinct entries), so every
//    top-k entry -- ties at the boundary included -- has a key at or above it.
// 2. Entries at or above the bound are collected (normally a few times k) and sorted by
//    (key desc, id asc): the order CUB's stable descending radix sort gives the old path.
// 3. Thread 0 walks the first k in that order: stable-softmax numerators against rank 0, the
//    running sum, and the top_p cut exactly as topk_topp_mask_kernel applies it.
// 4. The surviving candidates get forward_token's Gumbel noise with the same Philox key
//    (seed, subsequence = id, offset = step) and the same arithmetic; the argmax keeps the max
//    value and the min id, as argmax_merge does.
//
// The running sum is sequential here where the old path used a CUB scan, so the top_p cut can
// differ when a partial sum lands within float rounding of top_p * total -- a measure-zero case
// the test counts. A row whose candidate set overflows kSrCap (a mass of identical keys above
// the bound) writes -1 and the caller samples it the old way, or, with overflow_keeps_out,
// leaves out_id as the caller set it.
namespace {
constexpr int kSrThreads = 1024;
constexpr int kSrCap = 2048;

// The order-preserving float -> uint32 map CUB's radix sort uses for float keys.
__device__ __forceinline__ unsigned int sr_key(float f) {
    const unsigned int b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float sr_val(unsigned int key) {
    return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}
__device__ __forceinline__ int sr_id(unsigned long long c) {
    return (int)(0xffffffffu - (unsigned int)(c & 0xffffffffull));
}
}  // namespace

__global__ void __launch_bounds__(kSrThreads)
sample_rows_topk_kernel(const float* __restrict__ logits, int vocab,
                        const float* __restrict__ temp_f32,
                        const unsigned long long* __restrict__ seed_u64,
                        const unsigned long long* __restrict__ step_u64,
                        const int* __restrict__ top_k_i32, const float* __restrict__ top_p_f32,
                        int* __restrict__ out_id, bool overflow_keeps_out) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float T = temp_f32[row];
    const int k = top_k_i32[row];
    if (!sample_rows_topk_eligible(T, k, vocab)) return;  // the caller samples this row
    const float* L = logits + (size_t)row * vocab;

    __shared__ unsigned int s_max[kSrThreads];
    __shared__ unsigned long long s_cand[kSrCap];
    __shared__ float s_score[kSampleRowsTopkMax];
    __shared__ int s_n;
    __shared__ int s_keep;

    unsigned int mk = 0u;
    for (int v = tid; v < vocab; v += kSrThreads) mk = max(mk, sr_key(L[v]));
    s_max[tid] = mk;
    if (tid == 0) s_n = 0;
    __syncthreads();
    // Descending bitonic sort of the 1024 thread maxima.
    for (int size = 2; size <= kSrThreads; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int partner = tid ^ stride;
            if (partner > tid) {
                const bool desc = (tid & size) == 0;
                const unsigned int a = s_max[tid], b = s_max[partner];
                if ((a < b) == desc) { s_max[tid] = b; s_max[partner] = a; }
            }
            __syncthreads();
        }
    }
    const unsigned int bound = s_max[k - 1];

    for (int v = tid; v < vocab; v += kSrThreads) {
        const unsigned int key = sr_key(L[v]);
        if (key >= bound) {
            const int slot = atomicAdd(&s_n, 1);
            if (slot < kSrCap)
                s_cand[slot] = ((unsigned long long)key << 32) | (0xffffffffu - (unsigned int)v);
        }
    }
    __syncthreads();
    const int n = s_n;
    if (n > kSrCap) {
        if (tid == 0 && !overflow_keeps_out) out_id[row] = -1;
        return;
    }
    int p2 = 1;
    while (p2 < n) p2 <<= 1;
    for (int i = n + tid; i < p2; i += kSrThreads) s_cand[i] = 0ull;
    __syncthreads();
    // Descending bitonic sort of the candidates: key desc, then id asc (the low word is ~id).
    for (int size = 2; size <= p2; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < p2; i += kSrThreads) {
                const int partner = i ^ stride;
                if (partner > i) {
                    const bool desc = (i & size) == 0;
                    const unsigned long long a = s_cand[i], b = s_cand[partner];
                    if ((a < b) == desc) { s_cand[i] = b; s_cand[partner] = a; }
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        // topk_topp_exp_kernel, the scan and topk_topp_mask_kernel, over the first k ranks.
        const float row_max = sr_val((unsigned int)(s_cand[0] >> 32));
        float cum = 0.f;
        for (int i = 0; i < k; i++) {
            cum += __expf(sr_val((unsigned int)(s_cand[i] >> 32)) - row_max);
            s_score[i] = cum;  // the inclusive running sum; overwritten by the scores below
        }
        const float top_p = top_p_f32[row];
        const bool top_p_active = top_p >= 0.f && top_p < 1.f;
        const float cut = top_p * s_score[k - 1];
        int keep = 1;
        while (keep < k && (!top_p_active || s_score[keep - 1] < cut)) keep++;
        s_keep = keep;
    }
    __syncthreads();
    const int keep = s_keep;
    if (tid < keep) {
        const unsigned long long c = s_cand[tid];
        const int v = sr_id(c);
        const float x = sr_val((unsigned int)(c >> 32));
        curandStatePhilox4_32_10_t st;
        curand_init(seed_u64[row], (unsigned long long)v, step_u64[row], &st);
        // temperature_sample_kernel's arithmetic, term for term (see the clamp's comment there).
        const float inv_t = 1.f / T;
        const float u = fminf(curand_uniform(&st), 0.99999994f);
        const float g = -logf(-logf(u));
        s_score[tid] = x * inv_t + g;
    }
    __syncthreads();
    if (tid == 0) {
        float best = -1e30f;
        int bi = 0;
        for (int i = 0; i < keep; i++) {
            const int v = sr_id(s_cand[i]);
            const float sc = s_score[i];
            if (sc > best || (sc == best && v < bi)) { best = sc; bi = v; }
        }
        out_id[row] = bi;
    }
}

void launch_sample_rows_topk(const float* logits, int n_rows, int vocab,
                             const float* temp_f32, const unsigned long long* seed_u64,
                             const unsigned long long* step_u64, const int* top_k_i32,
                             const float* top_p_f32, int* out_id, cudaStream_t stream,
                             bool overflow_keeps_out) {
    if (n_rows < 1) return;
    sample_rows_topk_kernel<<<n_rows, kSrThreads, 0, stream>>>(
        logits, vocab, temp_f32, seed_u64, step_u64, top_k_i32, top_p_f32, out_id,
        overflow_keeps_out);
}

}  // namespace kernels
}  // namespace sparkinfer
