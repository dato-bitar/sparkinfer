#pragma once
// DFlash block-diffusion draft model for Qwen3.6-35B-A3B.
// Loads official z-lab BF16 safetensors; reuses target embed + lm_head.

#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace sparkinfer {

struct DFlashDraftConfig {
    int hidden = 2048;
    int intermediate = 6144;
    int n_layers = 6;
    int n_q_heads = 32;
    int n_kv_heads = 8;
    int head_dim = 128;
    int block_size = 16;
    int mask_token_id = 248077;
    int vocab = 248320;
    float rms_eps = 1e-6f;
    float rope_theta = 10000000.f;
    int sliding_window = 4096;
    int max_seq = 8192;
    std::vector<int> target_layer_ids = {1, 6, 11, 16, 22, 27, 32, 37};
    // Per-layer: true = sliding_attention (window), false = full_attention.
    std::vector<bool> sliding_layers;
    // false (default) = existing Qwen3.6 draft behavior: NeoX split-half RoPE pairing
    // (rotate h[i] with h[i+half]), via launch_rms_heads_rope. true = consecutive-pair
    // ("normal"/LLAMA_ROPE_TYPE_NORM) pairing, via launch_rms_heads_rope_normal -- set for the
    // Muse Glimmer draft (see museglimmer_dflash_config_from_gguf). This mirrors a fix already
    // made and validated on the Muse Glimmer TARGET model; on the draft it is an informed but
    // UNVERIFIED carry-over (no DFlash accuracy/SPEC_AGREE evaluation has run yet) -- if Muse
    // Glimmer draft proposals look wrong, check this flag first.
    bool rope_normal = false;

    // YaRN rotary scaling (RadixArk/Qwen3.8-27B-DSpark ships rope_type: "yarn"). factor <= 1
    // disables it and the draft uses plain theta^(-2i/d), so existing checkpoints are unaffected.
    // These cannot be folded into rope_theta: YaRN's NTK-by-parts ramp scales each frequency band
    // differently -- measured on this checkpoint, 36 of 64 bands are divided by `factor` and only
    // 15 are untouched -- and it additionally scales cos/sin magnitude by 0.1*ln(factor)+1.
    float yarn_factor = 0.f;          // "factor" (32.0 for DSpark); <= 1 => no YaRN
    int   yarn_orig_max_pos = 0;      // "original_max_position_embeddings" (8192)
    float yarn_beta_fast = 32.f;
    float yarn_beta_slow = 1.f;

    // DFlash2 (z-lab DFlash2DraftModel): a grouped dynamic causal conv around each attention and
    // MLP sublayer, and a candidate selector over the top selector_top_k logits of each slot.
    // `is_causal` is the checkpoint's own field (-1 when absent: the per-layer default applies).
    bool dflash2 = false;
    int is_causal = -1;
    int conv_kernel = 0;       // "conv_kernel_size" (taps)
    int conv_group = 0;        // "conv_group_size" (channels per dynamic kernel group)
    int selector_rank = 0;
    int selector_top_k = 0;
};

class DFlashDraftModel {
public:
    explicit DFlashDraftModel(const DFlashDraftConfig& cfg);
    ~DFlashDraftModel();

    // Load model.safetensors (+ optional config.json) from a HF draft directory.
    bool load(const std::string& dir);

    // Load a GGUF-packed draft checkpoint (e.g. Muse Glimmer's dflash-kquant.gguf). Self-contained
    // like load(): opens the file, derives config from its metadata (see
    // museglimmer_dflash_config_from_gguf in runtime/examples/dflash_gguf_config.h), and uploads
    // dequantized weights. Does not touch/alter load()'s HF-safetensors path.
    bool load_gguf(const std::string& path);

    const DFlashDraftConfig& config() const;

    // Bind shared target embed / lm_head (non-owning device pointers).
    void set_shared_weights(const void* embed_bf16_or_null,
                            const void* lm_head,
                            int lm_head_type,
                            int vocab,
                            int hidden);

    // Reset draft KV length to 0.
    void reset();

    // Crop draft KV to the first `keep` tokens (speculative accept boundary).
    void crop(int keep);

    // Slots: independent per-generation draft state -- the KV cache, the projected target
    // context, the position and context floor -- sharing the weights and block scratch, one per
    // request speculated concurrently. Slot 0 is the state allocated at load. use_slot(i, need)
    // makes slot i current, allocating it for `need` positions (~(2 * layers * kv_dim + hidden) *
    // need bf16) on first use or when it holds fewer; need 0 takes a live slot as it is and a new
    // one at max_seq. It returns false, leaving the current slot as it was, when that allocation
    // fails. reset/crop/forward_block act on the current slot, and forward_block declines past its
    // positions. free_slot(i) gives slot i back (not 0).
    bool use_slot(int i, int need = 0);
    void free_slot(int i);
    int current_slot() const;

    int seq_len() const;

    // One parallel block forward.
    //   target_hidden: [ctx_len, n_capture * hidden] bf16 (concat features before fc)
    //   noise_ids:     [block_size] token ids (mask-filled block; position 0 = seed)
    //   pos0:          absolute position of noise_ids[0]
    //   out_argmax:    [block_size] host argmax (only [1..] are draft proposals; [0] unused)
    // Returns false on failure.
    // Build the quantized weight copies now rather than on the first forward_block, so a caller
    // that primes the draft outside a timed region does not pay for it inside one. Idempotent.
    void ensure_quant();

    //   proposals:     how many rows after the seed to score (0 = the built-in default). The
    //                  verifier picks this by context length, so the draft has to be told rather
    //                  than deciding for itself, or the two disagree on how long a block is.
    //   out_confidence: optional, [1..proposals] host logits from DSpark's confidence head (raw
    //                  logit, not sigmoid'd -- sigmoid on the caller side if a probability is
    //                  needed). Left untouched (whatever the caller passed in) for checkpoints
    //                  without a confidence head, or when nullptr.
    bool forward_block(const void* target_hidden, int ctx_len,
                       const int* noise_ids, int pos0,
                       int* out_argmax, cudaStream_t stream = nullptr,
                       int proposals = 0, float* out_confidence = nullptr,
                       int target_hidden_start = 0);

    // Couple the next forward_block's proposals to a sampled request's sampler: proposal r (1-based)
    // becomes the token that sampler would draw from the draft's own logits -- top_k/top_p mask,
    // temperature, Gumbel noise from Philox(seed, vocab id, step0 + r - 1) -- instead of their
    // argmax. The target's verify draws its token at that same step with the same noise, so a
    // draft that agrees with the target's distribution now lands on the target's sampled token.
    // Draft-only: what is emitted is still decided by the verify, so LOSSLESS is unaffected.
    // temperature <= 0, a top_k the batched sampler does not take (sample_rows_topk_eligible), or
    // SPARKINFER_DFLASH_COUPLED=0 keeps the argmax.
    void set_sampling(float temperature, unsigned long long seed, unsigned long long step0,
                      int top_k, float top_p);

    // Apply target lm_head to last forward's hidden states; writes device logits [block, vocab]
    // and host argmax. Called internally by forward_block; exposed for debugging.
    const float* last_logits() const;

private:
    struct Impl;
    Impl* p_;
};

} // namespace sparkinfer
