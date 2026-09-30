// GPU test for launch_sample_rows_topk (batched top_k/top_p + temperature sampling). Every row
// must draw the token the per-row path draws: launch_topk_topp_mask -> launch_temperature_sample
// -> launch_argmax, each launched with n_rows = 1 on that row, as decode_packed did before. Also
// checks the rows it must leave alone, the overflow fallback, the sampled distribution, and times
// 32 rows both ways. Calls the kernels directly on synthetic logits -- no model needed.
#include "sparkinfer/kernels/fused.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace {

namespace K = sparkinfer::kernels;

// forward_token's load-time scratch for launch_topk_topp_mask.
struct Scratch {
    int vocab = 0;
    int* vocab_iota = nullptr;
    float* sorted_logits = nullptr;
    int* sorted_idx = nullptr;
    float* topk_exp = nullptr;
    float* topk_cumsum = nullptr;
    void* sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    void* scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    int* rank_by_id = nullptr;

    explicit Scratch(int v) : vocab(v) {
        cudaMalloc(&vocab_iota, vocab * sizeof(int));
        cudaMalloc(&sorted_logits, vocab * sizeof(float));
        cudaMalloc(&sorted_idx, vocab * sizeof(int));
        cudaMalloc(&topk_exp, vocab * sizeof(float));
        cudaMalloc(&topk_cumsum, vocab * sizeof(float));
        cudaMalloc(&rank_by_id, vocab * sizeof(int));
        K::launch_vocab_iota_init(vocab_iota, vocab);
        sort_temp_bytes = K::topk_sort_temp_storage_bytes(vocab);
        scan_temp_bytes = K::topk_scan_temp_storage_bytes(vocab);
        cudaMalloc(&sort_temp, sort_temp_bytes);
        cudaMalloc(&scan_temp, scan_temp_bytes);
        cudaDeviceSynchronize();
    }
    ~Scratch() {
        cudaFree(vocab_iota); cudaFree(sorted_logits); cudaFree(sorted_idx);
        cudaFree(topk_exp); cudaFree(topk_cumsum); cudaFree(rank_by_id);
        cudaFree(sort_temp); cudaFree(scan_temp);
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
};

struct Rows {
    int n = 0, vocab = 0;
    std::vector<float> logits;
    std::vector<float> temp, top_p;
    std::vector<int> top_k;
    std::vector<unsigned long long> seed, step;
};

// Device copies of a Rows batch, in decode_packed's layout (one array per parameter).
struct DevRows {
    float* logits = nullptr; float* temp = nullptr; float* top_p = nullptr;
    int* top_k = nullptr; int* out = nullptr;
    unsigned long long* seed = nullptr; unsigned long long* step = nullptr;
    explicit DevRows(const Rows& r) {
        const size_t lv = (size_t)r.n * r.vocab;
        cudaMalloc(&logits, lv * sizeof(float));
        cudaMalloc(&temp, r.n * sizeof(float)); cudaMalloc(&top_p, r.n * sizeof(float));
        cudaMalloc(&top_k, r.n * sizeof(int)); cudaMalloc(&out, r.n * sizeof(int));
        cudaMalloc(&seed, r.n * sizeof(unsigned long long));
        cudaMalloc(&step, r.n * sizeof(unsigned long long));
        cudaMemcpy(logits, r.logits.data(), lv * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(temp, r.temp.data(), r.n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(top_p, r.top_p.data(), r.n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(top_k, r.top_k.data(), r.n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(seed, r.seed.data(), r.n * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(step, r.step.data(), r.n * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    }
    ~DevRows() {
        cudaFree(logits); cudaFree(temp); cudaFree(top_p); cudaFree(top_k); cudaFree(out);
        cudaFree(seed); cudaFree(step);
    }
    DevRows(const DevRows&) = delete;
    DevRows& operator=(const DevRows&) = delete;
};

// The per-row path, as decode_packed ran it: mask, noise and argmax on a scratch copy of the row.
std::vector<int> sample_alone(Scratch& s, const Rows& r, const DevRows& d) {
    std::vector<int> ids(r.n, -2);
    float* row = nullptr;
    int* out = nullptr;
    cudaMalloc(&row, r.vocab * sizeof(float));
    cudaMalloc(&out, sizeof(int));
    for (int i = 0; i < r.n; i++) {
        if (!(r.temp[i] > 0.f)) continue;
        cudaMemcpy(row, d.logits + (size_t)i * r.vocab, r.vocab * sizeof(float), cudaMemcpyDeviceToDevice);
        if ((r.top_k[i] > 0 && r.top_k[i] < r.vocab) || r.top_p[i] < 1.f)
            K::launch_topk_topp_mask(row, r.vocab, s.vocab_iota, s.sorted_logits, s.sorted_idx,
                                     s.topk_exp, s.topk_cumsum, s.sort_temp, s.sort_temp_bytes,
                                     s.scan_temp, s.scan_temp_bytes, d.top_k + i, d.top_p + i,
                                     s.rank_by_id);
        K::launch_temperature_sample(row, 1, r.vocab, d.temp + i, d.seed + i, d.step + i);
        K::launch_argmax(row, out, 1, r.vocab);
        cudaMemcpy(&ids[i], out, sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaFree(row);
    cudaFree(out);
    return ids;
}

std::vector<int> sample_batched(const Rows& r, const DevRows& d, int sentinel) {
    std::vector<int> ids(r.n, sentinel);
    cudaMemcpy(d.out, ids.data(), r.n * sizeof(int), cudaMemcpyHostToDevice);
    K::launch_sample_rows_topk(d.logits, r.n, r.vocab, d.temp, d.seed, d.step, d.top_k, d.top_p, d.out);
    cudaMemcpy(ids.data(), d.out, r.n * sizeof(int), cudaMemcpyDeviceToHost);
    return ids;
}

// Logit rows shaped like an LM head's: a Gaussian bulk, a few peaked tokens, and per-row quirks
// that stress the ordering (ties at the top_k boundary, +0/-0, -inf entries).
Rows make_rows(int n, int vocab, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> bulk(0.f, 2.5f);
    std::uniform_int_distribution<int> any_id(0, vocab - 1);
    const int ks[] = {1, 2, 5, 20, 20, 40, 64, 100, 128};
    const float ps[] = {0.f, 0.3f, 0.8f, 0.9f, 0.95f, 0.95f, 1.f};
    const float ts[] = {0.2f, 0.6f, 0.7f, 1.f, 1.f, 1.3f};
    Rows r;
    r.n = n; r.vocab = vocab;
    r.logits.resize((size_t)n * vocab);
    for (int i = 0; i < n; i++) {
        float* L = r.logits.data() + (size_t)i * vocab;
        for (int v = 0; v < vocab; v++) L[v] = bulk(rng);
        const int peaks = 1 + (int)(rng() % 30);
        for (int j = 0; j < peaks; j++) L[any_id(rng)] = 8.f + 6.f * std::generate_canonical<float, 24>(rng);
        const int k = ks[rng() % (sizeof(ks) / sizeof(ks[0]))];
        switch (i % 5) {
            case 1: {  // ties straddling the top_k boundary, at scattered ids
                const float tie = 9.5f;
                for (int j = 0; j < k + 7; j++) L[any_id(rng)] = tie;
                break;
            }
            case 2:  // signed zeros and -inf among the candidates
                for (int j = 0; j < 40; j++) L[any_id(rng)] = (j & 1) ? 0.f : -0.f;
                for (int j = 0; j < 40; j++) L[any_id(rng)] = -std::numeric_limits<float>::infinity();
                break;
            case 3:  // quantized values: many exact duplicates everywhere
                for (int v = 0; v < vocab; v++) L[v] = std::round(L[v] * 4.f) / 4.f;
                break;
            default: break;
        }
        r.top_k.push_back(k);
        r.top_p.push_back(ps[rng() % (sizeof(ps) / sizeof(ps[0]))]);
        r.temp.push_back(ts[rng() % (sizeof(ts) / sizeof(ts[0]))]);
        r.seed.push_back(rng());
        r.step.push_back(rng() % 100000);
    }
    return r;
}

bool test_matches_per_row_path(Scratch& s, int vocab, int batches, int rows_per_batch) {
    int compared = 0, mismatched = 0;
    for (int b = 0; b < batches; b++) {
        const Rows r = make_rows(rows_per_batch, vocab, 1000u + 17u * b + (unsigned)vocab);
        DevRows d(r);
        const std::vector<int> want = sample_alone(s, r, d);
        const std::vector<int> got = sample_batched(r, d, -7);
        for (int i = 0; i < r.n; i++) {
            if (!(r.temp[i] > 0.f)) continue;
            if (got[i] == -1) continue;  // overflow fallback, checked separately
            compared++;
            if (got[i] != want[i]) {
                mismatched++;
                if (mismatched <= 5)
                    printf("  mismatch vocab=%d batch=%d row=%d k=%d p=%.2f T=%.2f: batched %d, per-row %d\n",
                           vocab, b, i, r.top_k[i], r.top_p[i], r.temp[i], got[i], want[i]);
            }
        }
    }
    printf("[%s] per-row equivalence, vocab %d: %d rows, %d mismatched\n",
           mismatched == 0 ? "PASS" : "FAIL", vocab, compared, mismatched);
    return mismatched == 0 && compared > 0;
}

bool test_leaves_ineligible_rows_alone(int vocab) {
    Rows r = make_rows(4, vocab, 77u);
    r.temp[0] = 0.f;                          // greedy
    r.top_k[1] = 0;                           // top_k disabled
    r.top_k[2] = K::kSampleRowsTopkMax + 1;   // above the batched limit
    r.top_k[3] = 20;                          // handled
    DevRows d(r);
    const std::vector<int> got = sample_batched(r, d, -7);
    const bool ok = got[0] == -7 && got[1] == -7 && got[2] == -7 && got[3] >= 0 && got[3] < vocab;
    printf("[%s] ineligible rows untouched (%d %d %d), eligible row sampled (%d)\n",
           ok ? "PASS" : "FAIL", got[0], got[1], got[2], got[3]);
    return ok;
}

bool test_overflow_falls_back(int vocab) {
    Rows r = make_rows(1, vocab, 5u);
    for (int v = 0; v < vocab; v++) r.logits[v] = 1.5f;  // every key equal: the set overflows
    r.temp[0] = 1.f; r.top_k[0] = 20; r.top_p[0] = 0.9f;
    DevRows d(r);
    const std::vector<int> got = sample_batched(r, d, -7);
    const bool ok = got[0] == -1;
    printf("[%s] overflowing candidate set reports -1 (got %d)\n", ok ? "PASS" : "FAIL", got[0]);
    return ok;
}

// launch_topk_rows (DFlash2's candidate selector) must return each row's top k in the sort's
// order -- value descending, id ascending among equal values -- and -1 ids on overflow.
bool test_topk_rows(int vocab) {
    bool ok = true;
    for (int k : {1, 16, K::kSampleRowsTopkMax}) {
        const Rows r = make_rows(24, vocab, 4242u + (unsigned)k);
        float* dl = nullptr; int* di = nullptr; float* dv = nullptr;
        cudaMalloc(&dl, r.logits.size() * sizeof(float));
        cudaMalloc(&di, (size_t)r.n * k * sizeof(int));
        cudaMalloc(&dv, (size_t)r.n * k * sizeof(float));
        cudaMemcpy(dl, r.logits.data(), r.logits.size() * sizeof(float), cudaMemcpyHostToDevice);
        K::launch_topk_rows(dl, r.n, vocab, k, di, dv);
        std::vector<int> ids((size_t)r.n * k);
        std::vector<float> vals((size_t)r.n * k);
        cudaMemcpy(ids.data(), di, ids.size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(vals.data(), dv, vals.size() * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < r.n; i++) {
            const float* L = r.logits.data() + (size_t)i * vocab;
            std::vector<int> order(vocab);
            for (int v = 0; v < vocab; v++) order[v] = v;
            std::partial_sort(order.begin(), order.begin() + k, order.end(),
                              [&](int a, int b) { return L[a] != L[b] ? L[a] > L[b] : a < b; });
            for (int j = 0; j < k; j++)
                if (ids[(size_t)i * k + j] != order[j] || vals[(size_t)i * k + j] != L[order[j]]) { bad++; break; }
        }
        printf("[%s] topk_rows k=%d: %d/%d rows match the sorted top k\n", bad ? "FAIL" : "PASS", k,
               r.n - bad, r.n);
        ok = ok && bad == 0;
        cudaFree(dl); cudaFree(di); cudaFree(dv);
    }
    {
        std::vector<float> flat(vocab, 1.5f);  // every key equal: the candidate set overflows
        float* dl = nullptr; int* di = nullptr; float* dv = nullptr;
        cudaMalloc(&dl, vocab * sizeof(float));
        cudaMalloc(&di, 16 * sizeof(int));
        cudaMalloc(&dv, 16 * sizeof(float));
        cudaMemcpy(dl, flat.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
        K::launch_topk_rows(dl, 1, vocab, 16, di, dv);
        int got[16];
        cudaMemcpy(got, di, sizeof(got), cudaMemcpyDeviceToHost);
        const bool o = std::all_of(got, got + 16, [](int v) { return v == -1; });
        printf("[%s] topk_rows overflow reports -1 ids\n", o ? "PASS" : "FAIL");
        ok = ok && o;
        cudaFree(dl); cudaFree(di); cudaFree(dv);
    }
    return ok;
}

// The batched kernel on its own must sample softmax(logits / T) over the top_k/top_p survivors.
bool test_distribution() {
    const int vocab = 64, draws = 40000, k = 12;
    const float T = 0.8f, p = 0.9f;
    std::vector<float> base(vocab);
    for (int v = 0; v < vocab; v++) base[v] = 0.37f * (float)((v * 29) % vocab) / 8.f;
    // Expected survivors and probabilities, computed on the host the way the mask defines them.
    std::vector<int> order(vocab);
    for (int v = 0; v < vocab; v++) order[v] = v;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return base[a] > base[b]; });
    std::vector<double> cum(k);
    double run = 0;
    for (int i = 0; i < k; i++) { run += std::exp((double)base[order[i]] - base[order[0]]); cum[i] = run; }
    int keep = 1;
    while (keep < k && cum[keep - 1] < p * cum[k - 1]) keep++;
    std::vector<double> want(vocab, 0.0);
    double z = 0;
    for (int i = 0; i < keep; i++) z += std::exp(base[order[i]] / T);
    for (int i = 0; i < keep; i++) want[order[i]] = std::exp(base[order[i]] / T) / z;

    Rows r;
    r.n = 256; r.vocab = vocab;
    for (int i = 0; i < r.n; i++) {
        r.logits.insert(r.logits.end(), base.begin(), base.end());
        r.temp.push_back(T); r.top_k.push_back(k); r.top_p.push_back(p);
        r.seed.push_back(12345); r.step.push_back(0);
    }
    std::vector<double> got(vocab, 0.0);
    DevRows d(r);
    for (int b = 0; b < draws / r.n; b++) {
        for (int i = 0; i < r.n; i++) r.step[i] = (unsigned long long)(b * r.n + i);
        cudaMemcpy(d.step, r.step.data(), r.n * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        const std::vector<int> ids = sample_batched(r, d, -7);
        for (int id : ids) if (id >= 0 && id < vocab) got[id] += 1.0;
    }
    double tv = 0;
    const double total = (double)(draws / r.n) * r.n;
    for (int v = 0; v < vocab; v++) tv += std::fabs(got[v] / total - want[v]);
    tv *= 0.5;
    bool outside = false;
    for (int v = 0; v < vocab; v++) if (want[v] == 0.0 && got[v] > 0) outside = true;
    const bool ok = tv < 0.02 && !outside;
    printf("[%s] distribution: %d survivors, total variation %.4f, draws outside the set: %s\n",
           ok ? "PASS" : "FAIL", keep, tv, outside ? "yes" : "no");
    return ok;
}

void time_32_rows(Scratch& s, int vocab) {
    Rows r = make_rows(32, vocab, 99u);
    for (int i = 0; i < r.n; i++) { r.top_k[i] = 20; r.top_p[i] = 0.95f; r.temp[i] = 1.f; }
    DevRows d(r);
    float* work = nullptr;
    cudaMalloc(&work, (size_t)r.n * vocab * sizeof(float));
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    auto per_row = [&]() {
        cudaMemcpy(work, d.logits, (size_t)r.n * vocab * sizeof(float), cudaMemcpyDeviceToDevice);
        for (int i = 0; i < r.n; i++) {
            float* row = work + (size_t)i * vocab;
            K::launch_topk_topp_mask(row, vocab, s.vocab_iota, s.sorted_logits, s.sorted_idx,
                                     s.topk_exp, s.topk_cumsum, s.sort_temp, s.sort_temp_bytes,
                                     s.scan_temp, s.scan_temp_bytes, d.top_k + i, d.top_p + i,
                                     s.rank_by_id);
            K::launch_temperature_sample(row, 1, vocab, d.temp + i, d.seed + i, d.step + i);
            K::launch_argmax(row, d.out + i, 1, vocab);
        }
    };
    const int reps = 20;
    per_row();
    cudaDeviceSynchronize();
    // The per-row timing includes a 32-row device copy (restoring the masked logits); time it
    // alone and subtract.
    cudaEventRecord(a);
    for (int t = 0; t < reps; t++)
        cudaMemcpy(work, d.logits, (size_t)r.n * vocab * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float copy_ms = 0; cudaEventElapsedTime(&copy_ms, a, b);
    cudaEventRecord(a);
    for (int t = 0; t < reps; t++) per_row();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float old_ms = 0; cudaEventElapsedTime(&old_ms, a, b);
    K::launch_sample_rows_topk(d.logits, r.n, vocab, d.temp, d.seed, d.step, d.top_k, d.top_p, d.out);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int t = 0; t < reps; t++)
        K::launch_sample_rows_topk(d.logits, r.n, vocab, d.temp, d.seed, d.step, d.top_k, d.top_p, d.out);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float new_ms = 0; cudaEventElapsedTime(&new_ms, a, b);
    printf("[INFO] 32 rows, vocab %d, top_k 20 / top_p 0.95: per-row path %.1f us, batched %.1f us\n",
           vocab, 1000.f * (old_ms - copy_ms) / reps, 1000.f * new_ms / reps);
    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(work);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device\n");
        return 0;
    }
    const int big_vocab = 248320;  // Qwen3.8's vocab
    bool ok = true;
    {
        Scratch s(big_vocab);
        ok = test_matches_per_row_path(s, big_vocab, 8, 32) && ok;
        time_32_rows(s, big_vocab);
    }
    {
        Scratch s(5003);  // not a multiple of the block size
        ok = test_matches_per_row_path(s, 5003, 20, 32) && ok;
    }
    ok = test_leaves_ineligible_rows_alone(big_vocab) && ok;
    ok = test_overflow_falls_back(big_vocab) && ok;
    ok = test_distribution() && ok;
    ok = test_topk_rows(big_vocab) && ok;
    ok = test_topk_rows(5003) && ok;
    if (cudaGetLastError() != cudaSuccess) { printf("[FAIL] CUDA error\n"); ok = false; }
    if (!ok) return 1;
    printf("sample_rows_topk_gpu_test: OK\n");
    return 0;
}
