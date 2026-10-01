// Check for Qwen35Model::verify_grouped: two sessions' speculative blocks verified in one pass
// must give each session what verifying it alone gives -- the same verified tokens, the same
// accepted prefix, and (after the commit) the same state, shown by decoding on from both.
//
// Each block is [seed, its 3 true greedy continuations, 2 junk tokens], so the expected accepted
// prefix is 4 (the seed plus three proposals). Three runs from fresh sessions:
//   truth    plain greedy decode, to get the true continuations
//   alone    batched_forward per session, then greedy decode on
//   grouped  one verify_grouped for both, then greedy decode on
//
// usage: grouped_verify_check <model_dir> <ids_file> [bench]
#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "qwen38_hf_config.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <model_dir> <ids_file>\n", argv[0]); return 2; }
    // Bit-exact comparison: without this, ordinary run-to-run reduction order flips near-tie
    // tokens (the junk rows, a late decoded token) in the alone run as well as the grouped one.
    setenv("SPARKINFER_DETERMINISTIC", "1", 0);
    // ...and the grouped pass on the same row kernels the alone pass takes at its width.
    setenv("SPARKINFER_GROUPED_WIDE", "0", 0);
    std::vector<int> ids;
    { std::ifstream f(argv[2]); int v; while (f >> v) ids.push_back(v); }
    const int lenA = 300, lenB = 517, offA = 0, offB = 400, kTrue = 3, kJunk = 2, kOn = 12;
    if ((int)ids.size() < offB + lenB) { printf("[FAIL] need %d ids\n", offB + lenB); return 1; }

    sparkinfer::Qwen35Config cfg;
    std::string err;
    if (!qwen38_config_from_hf_json(argv[1], cfg, err)) { printf("[FAIL] config: %s\n", err.c_str()); return 1; }
    cfg.max_seq = 1024;
    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads; kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16; kvc.int8_kv = true;
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(cfg.n_layers, cfg.hybrid, cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, cfg.n_layers);
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    sparkinfer::KVCacheManager kv(kvc, (size_t)kvL * 2 * epb * 2 * (4 * ((size_t)cfg.max_seq / 16) + 32));
    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k; mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);
    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!model.load_compressed_tensors(argv[1])) { printf("[FAIL] load\n"); return 1; }

    if (argc > 3 && std::string(argv[3]) == "bench") {
        // Timing: verify_grouped over G groups of 8 rows (G = 1..4) against one batched_forward
        // of 8 rows, each repeated with positions advancing by the accepted prefix.
        const int kRows = 8, kReps = 20, kLen = 256;
        for (int G = 0; G <= 4; ++G) {
            const int ng = G == 0 ? 1 : G;
            uint64_t sids[4];
            int pos[4];
            std::vector<int> blk[4];
            for (int g = 0; g < ng; ++g) {
                sids[g] = model.open_session(kLen + kRows * kReps + 64);
                model.activate_session(sids[g]);
                model.reset_mrope_offset();
                int p = 0;
                const int seed = model.ingest_prompt_range(ids.data() + 100 * g, 0, kLen, 0, &p);
                pos[g] = kLen;
                blk[g].assign(kRows, 11);
                blk[g][0] = seed;
            }
            std::vector<int> out((size_t)ng * kRows);
            int keep[4];
            double ms = 0;
            for (int r = 0; r < kReps; ++r) {
                cudaDeviceSynchronize();
                const auto t0 = std::chrono::steady_clock::now();
                if (G == 0) {
                    model.activate_session(sids[0]);
                    model.batched_forward(blk[0].data(), kRows, pos[0], false, out.data());
                    keep[0] = 1;
                } else {
                    const int* toks[4];
                    int lens[4];
                    for (int g = 0; g < ng; ++g) { toks[g] = blk[g].data(); lens[g] = kRows; }
                    if (!model.verify_grouped(ng, sids, toks, lens, pos, nullptr, out.data(), keep)) {
                        printf("[FAIL] verify_grouped declined\n"); return 1;
                    }
                }
                cudaDeviceSynchronize();
                if (r >= 2) ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                for (int g = 0; g < ng; ++g) { pos[g] += keep[g]; blk[g][0] = out[(size_t)g * kRows + keep[g] - 1]; }
            }
            printf("%s %d x %d rows: %.2f ms\n", G == 0 ? "batched_forward" : "verify_grouped", ng, kRows,
                   ms / (kReps - 2));
            for (int g = 0; g < ng; ++g) model.close_session(sids[g]);
        }
        return 0;
    }

    const int lens_p[2] = {lenA, lenB}, offs[2] = {offA, offB};
    auto open_and_prefill = [&](uint64_t sid[2], int seed[2]) {
        for (int g = 0; g < 2; ++g) {
            sid[g] = model.open_session(lens_p[g] + 64);
            model.activate_session(sid[g]);
            model.reset_mrope_offset();
            int pos = 0;
            seed[g] = model.ingest_prompt_range(ids.data() + offs[g], 0, lens_p[g], 0, &pos);
        }
    };
    auto decode_on = [&](uint64_t sid, int tok, int pos, int n) {
        std::vector<int> out;
        model.activate_session(sid);
        for (int i = 0; i < n; ++i) { tok = model.forward_token(tok, pos++, true); out.push_back(tok); }
        return out;
    };

    // truth
    uint64_t sid[2];
    int seed[2];
    open_and_prefill(sid, seed);
    std::vector<int> truth[2];
    for (int g = 0; g < 2; ++g) truth[g] = decode_on(sid[g], seed[g], lens_p[g], kTrue);
    for (int g = 0; g < 2; ++g) model.close_session(sid[g]);
    std::vector<int> block[2];
    for (int g = 0; g < 2; ++g) {
        block[g] = {seed[g]};
        for (int t : truth[g]) block[g].push_back(t);
        for (int j = 0; j < kJunk; ++j) block[g].push_back(11 + j);   // junk proposals
    }
    const int blen = (int)block[0].size();

    // alone
    std::vector<int> post_alone[2], on_alone[2];
    int keep_alone[2];
    open_and_prefill(sid, seed);
    for (int g = 0; g < 2; ++g) {
        model.activate_session(sid[g]);
        post_alone[g].assign(blen, -1);
        if (!model.batched_forward(block[g].data(), blen, lens_p[g], false, post_alone[g].data())) {
            printf("[FAIL] batched_forward\n"); return 1;
        }
        int k = 1;
        while (k < blen && block[g][k] == post_alone[g][k - 1]) ++k;
        keep_alone[g] = k;
    }
    for (int g = 0; g < 2; ++g)
        on_alone[g] = decode_on(sid[g], post_alone[g][keep_alone[g] - 1], lens_p[g] + keep_alone[g], kOn);
    for (int g = 0; g < 2; ++g) model.close_session(sid[g]);

    // grouped
    open_and_prefill(sid, seed);
    const int* toks[2] = {block[0].data(), block[1].data()};
    const int lens_v[2] = {blen, blen};
    const int starts[2] = {lenA, lenB};
    std::vector<int> post_g(2 * blen, -1);
    int keep_g[2] = {0, 0};
    if (!model.verify_grouped(2, sid, toks, lens_v, starts, nullptr, post_g.data(), keep_g)) {
        printf("[FAIL] verify_grouped declined\n"); return 1;
    }
    std::vector<int> on_g[2];
    for (int g = 0; g < 2; ++g)
        on_g[g] = decode_on(sid[g], post_g[g * blen + keep_g[g] - 1], lens_p[g] + keep_g[g], kOn);
    for (int g = 0; g < 2; ++g) model.close_session(sid[g]);

    bool ok = true;
    for (int g = 0; g < 2; ++g) {
        bool same_post = true;
        for (int i = 0; i < blen; ++i) same_post = same_post && post_alone[g][i] == post_g[g * blen + i];
        const bool same_on = on_alone[g] == on_g[g];
        if (!same_post || !same_on) {
            printf("  post alone:");
            for (int i = 0; i < blen; ++i) printf(" %d", post_alone[g][i]);
            printf("\n  post group:");
            for (int i = 0; i < blen; ++i) printf(" %d", post_g[g * blen + i]);
            printf("\n  on alone:");
            for (int t : on_alone[g]) printf(" %d", t);
            printf("\n  on group:");
            for (int t : on_g[g]) printf(" %d", t);
            printf("\n");
        }
        printf("session %c: keep alone %d grouped %d (expected %d) | verified tokens %s | %d tokens decoded on %s\n",
               'A' + g, keep_alone[g], keep_g[g], kTrue + 1, same_post ? "same" : "DIFF", kOn,
               same_on ? "same" : "DIFF");
        ok = ok && same_post && same_on && keep_alone[g] == keep_g[g];
    }
    printf(ok ? "[OK] grouped verify matches verifying each session alone\n" : "[FAIL] grouped verify differs\n");
    return ok ? 0 : 1;
}
