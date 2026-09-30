// Check for Qwen35Model::mixed_step: decode rows riding a prompt chunk's prefill pass.
//
// Three sessions on a compressed-tensors checkpoint: A and B are decoding, C has a prompt of
// which the first part is already prefilled and the next chunk is pending. The same state is
// driven two ways, each from freshly reopened sessions:
//   apart   decode_packed for A and B, then C's chunk through ingest_prompt_range
//   mixed   one mixed_step carrying A and B's rows and C's chunk
// and then every session decodes on (decode_packed, greedy) for `steps` more tokens. Reports how
// many of each session's tokens agree. The two are different pass shapes -- in a mixed step the
// decode rows go through the prefill's GEMMs -- so identity is not guaranteed; a layout slip (a
// row's state or KV in another's slot, the chunk at the wrong rows) shows up as divergence from
// the first token, not a late near-tie flip.
//
// usage: mixed_step_check <model_dir> <ids_file> [steps] [chunk]
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

namespace {

struct Run {
    std::vector<int> a, b, c;   // generated tokens: A and B from the step on, C from its seed on
    double step_ms = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <model_dir> <ids_file> [steps] [chunk]\n", argv[0]);
        return 2;
    }
    const std::string model_dir = argv[1];
    const int steps = argc > 3 ? atoi(argv[3]) : 24;
    std::vector<int> ids;
    {
        std::ifstream f(argv[2]);
        int v;
        while (f >> v) ids.push_back(v);
    }
    // A: 300-token prompt, B: 517, C: 700 prefilled then a 250-token chunk (+2 decode rows = 252,
    // not a multiple of 8, so the unaligned arms run too), then the rest of C's prompt.
    const int lenA = 300, lenB = 517, c_pre = 700, c_rest = 37;
    const int c_chunk = argc > 4 ? atoi(argv[4]) : 250;
    const int offA = 0, offB = 400, offC = 1000;
    const int lenC = c_pre + c_chunk + c_rest;
    if ((int)ids.size() < offC + lenC) { printf("[FAIL] need %d ids\n", offC + lenC); return 1; }

    sparkinfer::Qwen35Config cfg;
    std::string err;
    if (!qwen38_config_from_hf_json(model_dir, cfg, err)) { printf("[FAIL] config: %s\n", err.c_str()); return 1; }
    cfg.max_seq = 2048;

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers;
    kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    kvc.int8_kv = true;
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(cfg.n_layers, cfg.hybrid, cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, cfg.n_layers);
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = 4 * ((size_t)(cfg.max_seq + 15) / 16) + 16;
    sparkinfer::KVCacheManager kv(kvc, (size_t)kvL * 2 * epb * 2 * blocks);
    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts;
    mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn;
    mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);
    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!model.load_compressed_tensors(model_dir)) { printf("[FAIL] load_compressed_tensors\n"); return 1; }

    auto run = [&](bool mixed, Run& out) -> bool {
        const uint64_t sA = model.open_session(lenA + steps + 8);
        const uint64_t sB = model.open_session(lenB + steps + 8);
        const uint64_t sC = model.open_session(lenC + steps + 8);
        if (!sA || !sB || !sC) { printf("[FAIL] open sessions\n"); return false; }
        int pos = 0;
        model.activate_session(sA); model.reset_mrope_offset();
        const int tA = model.ingest_prompt_range(ids.data() + offA, 0, lenA, 0, &pos);
        model.activate_session(sB); model.reset_mrope_offset();
        const int tB = model.ingest_prompt_range(ids.data() + offB, 0, lenB, 0, &pos);
        model.activate_session(sC); model.reset_mrope_offset();
        model.ingest_prompt_range(ids.data() + offC, 0, c_pre, 0, &pos);
        if (tA < 0 || tB < 0 || pos != c_pre) { printf("[FAIL] prefill\n"); return false; }
        out.a = {tA};
        out.b = {tB};
        // The step under test.
        int toks[2] = {tA, tB};
        int poss[2] = {lenA, lenB};
        uint64_t seqs[2] = {sA, sB};
        int dec[2] = {-1, -1};
        int seedC = -1;
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        if (mixed) {
            if (!model.mixed_step(toks, poss, seqs, 2, dec, nullptr, sC, ids.data() + offC + c_pre,
                                  c_pre, c_chunk, &seedC)) {
                printf("[FAIL] mixed_step declined\n");
                return false;
            }
        } else {
            if (!model.decode_packed(toks, poss, seqs, 2, dec)) { printf("[FAIL] decode_packed\n"); return false; }
            model.activate_session(sC);
            int p2 = c_pre;
            seedC = model.ingest_prompt_range(ids.data() + offC, c_pre, c_pre + c_chunk, 0, &p2, false,
                                              /*allow_batched_resume=*/true);
            if (p2 != c_pre + c_chunk) { printf("[FAIL] chunk prefill\n"); return false; }
        }
        cudaDeviceSynchronize();
        out.step_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        out.a.push_back(dec[0]);
        out.b.push_back(dec[1]);
        out.c = {seedC};   // the chunk's last-position argmax (C's prompt continues below)
        // The rest of C's prompt, then everyone decodes on together.
        model.activate_session(sC);
        int p3 = c_pre + c_chunk;
        const int tC = model.ingest_prompt_range(ids.data() + offC, c_pre + c_chunk, lenC, 0, &p3, false, true);
        if (tC < 0 || p3 != lenC) { printf("[FAIL] rest of C\n"); return false; }
        out.c.push_back(tC);
        int t3[3] = {dec[0], dec[1], tC};
        int p[3] = {lenA + 1, lenB + 1, lenC};
        uint64_t q[3] = {sA, sB, sC};
        for (int k = 0; k < steps; ++k) {
            int o[3];
            if (!model.decode_packed(t3, p, q, 3, o)) { printf("[FAIL] decode_packed at %d\n", k); return false; }
            out.a.push_back(o[0]); out.b.push_back(o[1]); out.c.push_back(o[2]);
            for (int i = 0; i < 3; ++i) { t3[i] = o[i]; ++p[i]; }
        }
        model.close_session(sA); model.close_session(sB); model.close_session(sC);
        return true;
    };

    // Each way twice; the second of each is reported (the first pays one-time scratch and graph
    // setup).
    Run apart, mixed;
    if (!run(false, apart) || !run(true, mixed) || !run(false, apart) || !run(true, mixed)) return 1;
    auto agree = [](const std::vector<int>& x, const std::vector<int>& y, int* first_diff) {
        int n = 0;
        *first_diff = -1;
        for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
            if (x[i] == y[i]) ++n;
            else if (*first_diff < 0) *first_diff = (int)i;
        }
        return n;
    };
    int fa, fb, fc;
    const int na = agree(apart.a, mixed.a, &fa), nb = agree(apart.b, mixed.b, &fb),
              nc = agree(apart.c, mixed.c, &fc);
    printf("session  agree/total  first-diff   (index 1 = the mixed step's own token for A/B; 0 = C's chunk seed)\n");
    printf("A        %3d/%-3zu      %d\nB        %3d/%-3zu      %d\nC        %3d/%-3zu      %d\n",
           na, apart.a.size(), fa, nb, apart.b.size(), fb, nc, apart.c.size(), fc);
    printf("step ms: apart %.1f  mixed %.1f\n", apart.step_ms, mixed.step_ms);
    if (getenv("MIXED_CHECK_DUMP")) {
        for (const auto* r : {&apart, &mixed}) {
            printf("%s C:", r == &apart ? "apart" : "mixed");
            for (int t : r->c) printf(" %d", t);
            printf("\n");
        }
    }
    // A layout slip diverges at once; allow late flips from the different pass shape.
    const bool ok = fa != 1 && fb != 1 && fc != 0 && fc != 1 &&
                    (fa < 0 || fa > 4) && (fb < 0 || fb > 4) && (fc < 0 || fc > 4);
    printf(ok ? "[OK] mixed step matches the separate step and prefill\n"
              : "[FAIL] mixed step diverges early\n");
    return ok ? 0 : 1;
}
