# Roadmap: the fastest inference engine on the RTX 5090

Status: plan written 2026-09-30; the progress log at the end records what has shipped since.
Scope: one GeForce RTX 5090 (sm_120, 32 GB, 1792 GB/s). The RTX PRO 6000 Blackwell comes second
(see "After the 5090" at the end).
Primary model: Qwen3.8-27B (NVFP4, ModelOpt release checkpoint). Ternary-Bonsai-2-27B, Muse
Glimmer and Qwen3.6-35B-A3B follow the same work where it applies.

## 1. Where we stand

Figures for Qwen3.8-27B. Ours come from main (e952756) and the 2026-09-28 AIPerf run against vLLM
0.30.0. Competitor figures are published community measurements, so they are rough (±10%) and
their workloads differ.

| Axis | sparkinfer | Best published | Position |
|---|---|---|---|
| Single-stream decode, no speculation | ~96 tok/s | SGLang 93, vLLM ~79, llama.cpp 74–78 | lead |
| Decode at 256K context | ~65 tok/s | no comparable plain figure | lead |
| Prefill, 4K prompt | 15–16K tok/s | SGLang 13.4–15K, NInfer 12.8K | lead |
| Speculative, greedy, single stream | 130 on hard 16K prose (DSpark) | vLLM + DFlash2 ~304 on code @8K, 221 @60K | behind |
| Speculative, sampled (T>0), single stream | none: DSpark is greedy-only | SGLang + DFlash2 228 @T=0.7 | large gap |
| 8 concurrent users | 344 on our public card (stale) | NInfer + MTP 922 | behind ~2.7x |
| 16 concurrent users, TTFT p50 | 2.3 s | vLLM 0.34 s | behind |
| 32 concurrent users, inter-token latency | 21.6 ms | vLLM 17.7 ms | behind |
| Throughput vs vLLM, c16/c32 | 0.69–0.91x | — | behind |

Plain single-stream decode already runs at roughly 85–90% of what the memory bus allows. A copy
kernel reaches about 1.53 TB/s on a 5090, so the ceiling is near. The remaining single-user gains
must come from speculation. Throughput gains come from the scheduler and from batched-decode
kernels.

### Phase 0.2 profile (2026-09-30, main f74b5ac, RTX 5090, Qwen3.8-27B ModelOpt)

nsys traces with CUDA-graph node tracing, taken on the eval box. Per decode step the engine reads
about 14.5 GB of weights: 13.7 GB of layers plus the 0.7–0.8 GB NVFP4 LM head. A copy kernel
reaches about 1.53 TB/s on a 5090.

| Workload | Step | GPU busy | Where the time goes | Verdict |
|---|---|---|---|---|
| Single stream, 32K context | 10.7 ms | 99.5% | weight GEMVs 9.5 ms (~1.42 TB/s), LM head 0.7 ms, attention 0.8 ms | at the bandwidth ceiling |
| Single stream, 128K | 12.5 ms | 99.6% | attention 2.9 ms (~1.48 TB/s over 4.3 GB of int8 KV) | at the ceiling |
| Single stream, 262K | 15.2 ms | 99.7% | attention 5.6 ms (~1.53 TB/s over 8.6 GB of KV), 33% of the step | at the ceiling; only fewer bytes can help |
| Batched, 8 rows, greedy | 12.3 ms | 92% | CUTLASS NVFP4 tensor-core GEMMs dominate; GDN 0.8 ms | GEMMs ~1.2 TB/s: ~15–20% headroom |
| Batched, 32 rows, greedy | 15.3 ms | 95% | GEMMs ~12 ms; GDN recurrent state 2.8 ms (bandwidth-bound); attention 0.5 ms | same GEMM headroom; GDN only via fewer bytes |
| Batched, 32 rows, **sampled** (T=1.0, top_k 20, top_p 0.95) | **18.7 ms** | 87% | per-row sampling: each row runs a full-vocabulary radix sort plus mask/temperature/argmax (~50 µs), rows in series, ~1.6 ms of kernels and ~2.4 ms more idle | **the largest fixable loss** |
| Prefill, 128K / 262K prompt | — | — | prefill attention is 50% / 68% of prefill time, running at roughly a quarter to a third of INT8 tensor peak | main lever for long-prompt TTFT |

Engine throughput (cb_bench, 256-token prompts and answers):

| Rows | Greedy tok/s (ITL) | Sampled tok/s (ITL) |
|---|---|---|
| 1 | 102 (10.1 ms) | 102 (10.1 ms) |
| 8 | 650 (11.9 ms) | 617 (12.6 ms) |
| 16 | 1,148 (12.9 ms) | 1,062 (14.1 ms) |
| 32 | 1,840 (15.5 ms) | 1,504 (18.7 ms) |

What this changes:
- The server's 21.6 ms inter-token latency at 32 users is not a kernel problem. About 3 ms of it is
  per-row sampling. Most of the rest is prompts arriving mid-decode: engine max ITL reaches
  230–415 ms at 16–32 rows. Phase 1.0 and Phase 2 target these.
- Kernels are already near the bandwidth ceiling for single-stream decode, decode attention and
  GDN state. The remaining kernel headroom is batched GEMMs at 8–32 rows (≤1.2x) and long-prompt
  prefill attention (2–3x on that kernel).
- Decode attention at long context is at copy speed, so windowing can only help by reading fewer
  bytes (4-bit KV, or query-aware selection behind a recall gate), never by a faster kernel.

## 2. Targets (the scorecard)

| # | Axis (Qwen3.8-27B, one RTX 5090) | Now | Target |
|---|---|---|---|
| T1 | Single-stream decode, default sampling (T=1.0), speculative | ~90 | ≥ 230 tok/s |
| T2 | Single-stream decode, greedy, hard prose @16K | 130 | ≥ 200 tok/s |
| T3 | Aggregate at 8 concurrent, default sampling | 563 (bot engine bench, greedy, c8); sampled serving not measured | ≥ 1,000 tok/s |
| T4 | TTFT p50 at 16 concurrent chat (AIPerf 1024/256) | 2.3 s | < 0.5 s |
| T5 | Inter-token latency at 32 concurrent | 21.6 ms | < 17.7 ms |
| T6 | AIPerf throughput vs vLLM and SGLang, every cell c1–c32 | 0.69–1.15x | ≥ 1.0x everywhere |
| T7 | Plain decode, prefill, 256K decode | lead | keep the lead (no regression) |

The target is met only when it is measured on the eval box, lossless (or behind an explicit
opt-in), and repeatable across two runs.

## 3. Rules for every phase

1. **Lossless by default.** Speculation must keep the output distribution exact: greedy stays
   byte-identical, and sampling uses the standard min(1, p/q) acceptance test. Anything lossy
   (windowing, eviction, lower-precision state) is opt-in and must pass a long-context recall
   gate (RULER/NIAH plus a multi-turn agent test). #1088 showed why: a sink+window KV mode made
   Qwen3.8 misread a 31K-token tool result.
2. **5090 instruction choices.** On GeForce Blackwell, FP8 and BF16 MMAs with FP32 accumulate
   run at half rate (419 and 209.5 TFLOPS). Block-scaled FP4 (`kind::mxf4nvf4`), INT8, and FP8
   with FP16 accumulate run at full rate. New kernels should prefer the full-rate paths. The
   hardware limits are: no wgmma, no TMEM or tcgen05, no TMA multicast, and at most 99 KB of
   shared memory per block.
3. **Measure before building.** Every phase starts from a profile (nsys/ncu) on the box. Every
   PR carries the same-box before/after numbers.
4. **Never trade away T7.** The existing no-regression floors in the eval bots stay in force.

## 4. Phases

### Phase 0: Measurement foundation (week 1)

| Step | Work | Output |
|---|---|---|
| 0.1 | Finish the new eval box to match the old one: Rust (server build), safetensors, transformers | the server and Bonsai checks run again |
| 0.2 | Profile Qwen3.8 on the box. Batched decode at c8/c16/c32: time per GEMM, which GEMM path runs at 8–32 rows, SM fill, and GDN state bytes. Long-context decode at 32K/128K/262K: attention share and achieved GB/s | a short profile report that sets the order of Phases 1–3 |
| 0.3 | Head-to-head harness: AIPerf cells (chat 1024/256, answer 128/1024, 8K/128) at c1–c32 against vLLM (MTP/DFlash2 on), SGLang (DFlash2/MTP on), llama.cpp (draft-mtp) and NInfer, with default sampling | the public baseline for T1–T6 |
| 0.4 | Add eval-bot axes: sampled serving throughput (c1/c8/c16), TTFT at c16, speculative decode with sampling. Contributor PRs then aim at the real gaps | bots score what users feel |
| 0.5 | Refresh the public model card and README tables. The card still shows v0.5.5 numbers (c8 344, c16 345), and search results mix them up with competitors' | correct public numbers |

### Phase 1.0: Batched sampler (quick win, days). T5, T3

Packed decode samples each row separately (`decode_packed`, qwen35.cpp ~4957–4992):
- `launch_topk_topp_mask` accepts n_rows == 1 only, and runs a full CUB radix sort over the
  248K-token vocabulary;
- then temperature, Gumbel sample and argmax;
- then one sync.

At 32 rows that costs about 3 ms of an 18.7 ms step (−18% throughput against greedy).

Work:
- One launch for all rows: per-row top-k selection (k ≤ 64 by default; fall back to the current
  path above that), top-p inside the k candidates, temperature, and Gumbel-max with the existing
  Philox key (seed, step, vocab index).
- The same kernel serves `forward_token`'s tail.

Gates:
- Token-identical to the current sampler for the same seed on a fixed logit corpus.
- The existing distribution tests (`temperature_sample_gpu_test`, `topk_topp_sample_gpu_test`).
- Sampled c32 cb throughput within 3% of greedy.

### Phase 1: Speculation for every request (weeks 1–3). Biggest gap: T1, T2, T3

Today DSpark runs only for greedy requests. It requires all of the following:
- exactly one live request;
- no penalties, logprobs or logit bias;
- no prefix-cache hit.

The rules are in `ContinuousBatchEngine::worker_loop` and `spec_eligible` (inference_engine.cpp).
A second request interrupts it. Requests that omit a temperature take generation_config's T=1.0,
and the prefix cache is on by default, so most real traffic never speculates today.

Key design point, from the code map. Sampling is Gumbel-max with a counter-based Philox key
(seed, step, vocab index). A verify row can therefore apply the same mask, temperature and noise
at its own step and accept on equality with the draft token, exactly like greedy. This is exactly
lossless: it reproduces the non-speculative token stream for the same seed, so the existing
byte-identical LOSSLESS gate carries over to sampled requests. Adding the same noise to the
draft's logits ("coupling") raises acceptance without affecting correctness.

Project rule (CONTRIBUTING.md): MTP must never replace DSpark on the scored DSpark axis (#912 was
reverted for this). MTP is a separately selected, separately labelled, opt-in drafter.

| Step | Work | Gate |
|---|---|---|
| 1.1 | **Sampled speculation (single request).** Split `dflash_verify_short_run` into a forward and a commit(keep), so the caller chooses `keep`. Export verify-row logits in non-packed mode. Per verify row, apply the forward_token sampling tail (logit bias, top-k/top-p, temperature, Gumbel at step = base + i) and accept on equality with the draft. Couple the draft with the same Gumbel noise. Relax `spec_eligible` for temperature/top-k/top-p. Also allow prefix-cache hits, where the draft ingests only the uncached suffix | LOSSLESS: token-identical to non-speculative decode with the same seed, at T=0.7 and T=1.0; greedy unchanged; tau and tok/s at T=1.0 reported |
| 1.2 | **MTP head as a separate, opt-in drafter** (CONTRIBUTING rule: never on the DSpark axis, own flag, own metric label, own bot axis). Revive #912's loader (commit 6a3d10b) with a per-sequence MTP KV prefilled over the prompt. One layer, cheap in VRAM, so it fits beside a 262K context where DSpark forces 131K | acceptance length ≥ 2.5 on the workload corpus; no context reduction; owner approval of the drafter-selection policy |
| 1.3 | **Batched verify for concurrent requests.** Speculate for up to N live requests in one packed verify pass. The packed decode path (`decode_packed`, per-row block tables, per-row GDN state) is the base. GDN state rollback per row on rejection | lossless per row; aggregate ≥ plain packed decode at every concurrency |
| 1.4 | **Load-adaptive draft length.** Per step, choose the draft length per row from predicted acceptance and batch size: long drafts when idle, short or none at 16–32 rows. DSpark's confidence scheduler already predicts per-token acceptance | never slower than no speculation at any c1–c32 |
| 1.5 | **Evaluate a DFlash2 drafter** against DSpark and MTP on the same corpus. It is the current leader: 228 tok/s sampled on SGLang, ~304 on code on vLLM | choose the default drafter per model by measurement |
| 1.6 | **Default it on.** `serve` speculates by default, and `serve-dspark` becomes the full-context MTP or DSpark choice. Update `/metrics` and the README | T1 and T3 measured on the box |

### Phase 2: Serving under load (weeks 3–4). T4, T6

| Step | Work | Gate |
|---|---|---|
| 2.1 | **Mixed prefill + decode steps** (shipped opt-in as #1224). Each step carries all decode rows plus the next chunk of the oldest pending prompt. Row-wise work (GEMMs, LM head) runs once over all rows; attention and GDN run per segment (recurrent for decode rows, chunked scan for the prompt chunk). The chunk contains its prefix-cache checkpoint | a correctness tool comparing mixed-step decode rows with `decode_packed` and the chunk's KV/state with prefill resume |
| 2.2 | **CPU/GPU overlap.** Prepare step N+1 on the host while step N runs, and remove host–device syncs from the hot path (SGLang's zero-overhead scheduler gave about 1.1x) | no idle gaps between steps in nsys at c16/c32 |
| 2.3 | **Token budget and chunk coalescing** for many short prompts arriving together | TTFT p99 at c32 below vLLM's |

### Phase 3: Batched-decode kernels (weeks 2–6, bot contributors can carry most of it). T5, T3

From the Phase 0.2 profile:
- Batched decode already runs CUTLASS NVFP4 tensor-core GEMMs, at about 1.2 TB/s of weight
  traffic against a ~1.5 TB/s copy ceiling. That is ≤1.2x headroom on the GEMM share (~80% of a
  32-row step).
- The GDN state is already bandwidth-bound (2.8 ms/step at 32 rows), so only fewer bytes help.

Items, in order:
- **Tensor-core skinny GEMM for 8–64 rows.** NVFP4 `mma.sync m16n8k64 kind::mxf4nvf4`, weights
  on the MMA's 16-row side and tokens on the 8 side ("swap-AB"). Multi-stage weight streaming and
  split-K sized to fill 170 SMs, with activation quantization fused into the preceding norm.
  CUDA-core GEMVs lose 3–5x to tensor cores at 32 rows (RaZeR, RTX 5090). vLLM's b12x kernel is
  the open reference.
- **GDN state traffic.** Fuse conv1d, gating, the recurrent update and the gated norm into one
  pass over the state. Consider DeltaLog-style deferred write-back (1.3–1.9x on the kernel). The
  bf16-compacted state already helps packed rows.
- **Attention at 16–64 rows.** Pack the 6 query heads per KV head so each KV byte loads once.
  Split counts are tuned per context length.

### Phase 4: KV capacity and safe long context (weeks 4–6). T6 at long prompts, T7

| Step | Work | Note |
|---|---|---|
| 4.1 | **Long-prompt prefill attention.** `pf_attn_mma_gqa` is 50% of prefill time at 128K and 68% at 262K, at roughly a quarter to a third of INT8 tensor peak. Target: FA2-style tiling with TMA and warp specialization inside the 99 KB shared-memory limit, and GQA packing (6 query heads per KV head) | lossless, first choice. (Profile correction: decode attention is already at copy speed, ~1.53 TB/s at 262K, so a faster decode kernel is not a lever) |
| 4.2 | FP8 or 4-bit KV for the full-attention layers, as an opt-in for capacity on 32 GB (FP8 is about 91–100% of the speed of the current int8 path) | recall gate |
| 4.3 | Skip-softmax block skipping in prefill and decode attention (near-lossless at ~50% sparsity per TensorRT-LLM; ~1.3–1.4x at 128K) | opt-in, RULER gate |
| 4.4 | Quest-style query-aware block selection for decode. It keeps all KV, so there is no multi-turn failure. Mainly for Bonsai and batched long context | opt-in, RULER plus multi-turn gate |
| 4.5 | Make the lossy Qwen3.6 sink+window decode (default on past 16K today) opt-in, like Qwen3.8 | quality fix |

### Phase 5: Prove it (week 6 onward)

- Re-run the Phase 0 head-to-head with the same versions and settings, and publish the tables:
  README, model card and release notes.
- Claims are made per axis (single-user, c8/c16/c32 throughput, TTFT, long context), never as one
  number.
- Release it. (v0.6.0 shipped earlier, after Phase 1.3–1.6, with the vLLM head-to-head only; the
  full multi-engine head-to-head goes in a later release.)

## 5. Timeline (estimates)

| Week | Work |
|---|---|
| 1 | Phase 0 (0.2 profile done 2026-09-30); Phase 1.0 batched sampler; start 1.1 |
| 2 | 1.1, 1.2 done; 1.3 underway; Phase 3 items opened to contributors |
| 3 | 1.3, 1.4, 1.5; start 2.1 |
| 4 | 1.6; 2.1, 2.2 |
| 5–6 | 2.3; Phase 4; Phase 3 PRs landing |
| 6+ | Phase 5: head-to-head, publish |

## 6. Risks

- **Speculation at high concurrency.** Speculation stops paying off at about 8 concurrent users
  for every engine today. Step 1.4 must fall back cleanly, or we lose T5 and T6.
- **VRAM on 32 GB.** A drafter, the batch arena and the KV pool compete. The MTP head is the
  low-memory option; DSpark needs a reduced context.
- **Sampled acceptance is lower than greedy.** Published T=0.7–1.0 runs show lower acceptance.
  T1 may land nearer 200 than 230 on hard prose.
- **Numerics.** Mixed steps and batched verify are not bit-identical to unmixed steps. Accuracy
  is judged differentially (top-1/KL), as the bots already do.
- **Benchmark noise.** The 5090 hits its power cap on prefill and high concurrency. Every
  published run logs power, clocks and throttle reasons, and repeats twice.

## 7. Not now

- **Megakernel rewrite.** We already use CUDA graphs and programmatic launch. Published results
  suggest a few percent at 27B, with nothing shown at batch 8–32 on sm_120.
- **New sub-4-bit weight formats.** Bonsai already covers 1.75-bit.
- **Lossy defaults of any kind** (StreamingLLM, KV eviction).

## 8. After the 5090: RTX PRO 6000

- Same memory bandwidth, so single-user decode is equal. The 188 SMs, 128 MB L2 and full-rate
  BF16/FP8 need their own tile, split and path choices.
- 96 GB allows a 262K context with a drafter and large batches. Remove the 32 GB compromises
  (the 131K default, the DSpark context cut).
- Add a PRO 6000 eval box (note which edition: Workstation 600 W, Max-Q 300 W, or Server with
  1597 GB/s).
- No one has published single-GPU c16/c32 numbers for Qwen3.8-27B on it, so we could be first.

## 9. Decisions needed from the owner

1. Go-ahead for Phase 0, including the bot-axis changes in 0.4, which change contributor scoring.
2. Whether the lossy Qwen3.6 default (4.5) changes now.
3. Whether speculation becomes the default in `serve` (1.6) once gates pass.
4. Budget for a PRO 6000 box, when the 5090 phases are done.

## 10. Progress log

### 2026-10-01

Merged (each independently reviewed, CI green, box-tested):
- **Phase 1.0, #1214:** batched sampler.
- **Phase 1.1:**
  - #1215 sampled speculation;
  - #1221 speculation on prefix-cache hits, plus checkpoints taken by the speculative prefill.
- **Prefill, TTFT and engine:**
  - #1217 prefill overheads;
  - #1222 short-resume tail;
  - #1223 NVFP4-head graph key.
- **Phase 1.5, #1220:** DFlash2 drafter, opt-in.
- **Phase 2.1, #1224:** mixed prefill+decode, opt-in `SPARKINFER_MIXED_CHUNK`. It is a latency mode: chat c32 TTFT 1240 → 332 ms, but tok/s −15%. It carries one prompt per mixed pass.
- **Engine, #1225:** tokens streamed off the engine thread. The emission callbacks were 4 ms of every 19 ms step at 32 rows. longanswer c32 1600 → 1986 tok/s (vLLM 1785).

Scoreboard vs vLLM 0.30 before #1225 (vLLM tok/s ÷ ours; < 1 = sparkinfer faster):

| | c1 | c4 | c16 | c32 |
|---|---|---|---|---|
| chat | 0.88 | 0.86 | 0.93 | 1.19 |
| 8K | 0.90 | 1.16 | 1.03 | 1.01 |
| long answers | 0.86 | 0.85 | 1.04 | 1.17 |

#1225 should flip both c32 cells.

Rejected or deferred, measured:
- **Keeping the NVFP4 head resident for chat-sized prompts:** worse (the prefill arena loses its VRAM).
- **Tail-forward cap 8 → 32 alone:** flat.

Open:
- **8K at c4:** mixing loses ~20 ms per chunk to a 12 ms decode step saved.
- **Concurrent speculation (Phase 1.3).**
- **Single-stream sampled speculation:** 155 (DSpark) / 177 (DFlash2) tok/s against SGLang + DFlash2's published 228.
- **Long-prompt prefill attention (Phase 4.1).**

Later on 2026-10-01:
- **#1225 merged:** tokens streamed off the engine thread.
- **#1227 merged:** a long prompt's prefill scratch is given back. The full sweep's chat-c32 loss was this state effect: 849 tok/s in the sweep, ~1,180 on a fresh server.
- **Single-stream speculation is a tie with vLLM 0.30 + DFlash2** (same draft, same 7 real prompts):

  | | T=0 tok/s | T=0.7 tok/s | acceptance |
  |---|---:|---:|---|
  | sparkinfer | 211 | 196 | 3.4–3.8 code/math |
  | vLLM | 215 | 193 | 3.5–4.0 |

  - vLLM can't load our DSpark checkpoint (shape mismatch).
  - Bug: our DFlash2 at `SPARKINFER_DFLASH_PROPOSALS=6` drops to plain-decode speed (94 tok/s).
- **Concurrent speculation (aggregate tok/s, real prompts, DFlash2):**

  | | c1 | c2 | c4 | c8 |
  |---|---:|---:|---:|---:|
  | sparkinfer | 183 | **178** | 328 | **622** |
  | vLLM | 191 | **276** | 341 | 345 |

  Phase 1.3 (speculation for 2–4 live requests) is next. It is the one place a speculating vLLM beats us.

Later still on 2026-10-01:
- **#1228 merged: concurrent speculation (Phase 1.3).**
  - Up to 4 fresh prompts speculate together, `SPARKINFER_SPEC_GROUP`, default 4:
    - one draft slot each;
    - one grouped verify, with per-row attention tables, a compact GDN scan and commit per sequence, and the wide GEMM arms (4×8 rows: 56 → 16.6 ms).
  - Single requests take the same path and stay lossless. They also got faster: DFlash2 207/193 → 227/227 tok/s.

  | conc_bench DFlash2 T=0.7 | c1 | c2 | c4 | c8 |
  |---|---:|---:|---:|---:|
  | before | 177 | 178 | 319 | 633 |
  | #1228 | 208 | 349 | 485 | ≈ |
  | #1228 + #1229 | 206 | **372** | **522** | **651** |
  | vLLM + DFlash2 | 191 | 276 | 341 | 345 |

- **#1229 (open):** the short-prompt aligned prefill split now applies from an 8-token body (it was 128). TTFT at 9–100 prompt tokens: 81–88 → 25–31 ms.
- **#1230 (open): batched draft and groups of up to 8.**
  - `DFlashDraftModel::forward_blocks` runs every member's projections as one NVFP4 GEMM: the draft costs 2.2 ms per step at c4 instead of ~6.
  - DFlash2's MLP is NVFP4 for every draft.
  - `SPARKINFER_SPEC_GROUP` now defaults to 8.
  - conc_bench (DFlash2, T=0.7), aggregate tok/s:

    | | c2 | c4 | c6 | c8 |
    |---|---:|---:|---:|---:|
    | #1228 | 349 | 485 | ~405 | 633 |
    | #1230 | 367 | 573 | 669 | 733 |
    | #1230 + #1229 | 416 | 626–652 | 768 | **861** |
    | vLLM + DFlash2 | 276 | 341 | | 345 |

  - T3 (≥1,000 tok/s at c8) is not met yet. A c8 step costs:

    | part | per step |
    |---|---:|
    | verify (32 rows) | 18.6 ms |
    | draft | 3.6 ms |
    | joins | 2.8 ms |
    | kept tokens | 21.7 |

    Next levers:
    - a 64-row verify, for full depth at 8 members;
    - cheaper joins;
    - the per-member GDN scan in one launch.
- **Prefix cache, measured and shelved:**
  - The AIPerf cells re-send earlier cells' prompts (same seed), and vLLM's cache keeps them: a 35% hit rate in the 8K-prompt cells.
  - Ours keeps at most half the KV pool, 7 entries of 8K tokens.
  - Raising that to 95% gives 15 entries, but only 6 of 36 hits at 8K prompts and c4: the re-scan thrashes LRU, and our pool is 131K tokens against vLLM's larger one. +8% on that cell.
  - Not worth the capacity-reporting changes yet (branch perf/prefix-cache-pool).

Merged later on 2026-10-01 (each reviewed, CI green, box-tested on the combined state):
- **#1229:** short-prompt aligned prefill split. TTFT at 9–100 tokens: 81–88 → 25–31 ms.
- **#1230:** batched draft and groups of up to 8. The review's head-prefix finding was disproved on the GPU (max |diff| 0).
- **#1231:** one-launch GDN conv/scan for grouped verifies, bit-identical.

main a420039, conc_bench (DFlash2, real prompts, T=0.7), aggregate tok/s:

| | c1 | c2 | c4 | c6 | c8 |
|---|---:|---:|---:|---:|---:|
| main a420039 | 212 | 428 | 676 | 843 | 956–964 |
| main 7cb1301 (before #1228) | 177 | 178 | 319 | ~405 | 633 |
| vLLM 0.30 + DFlash2 | 191 | 276 | 341 | | 345 |

Every lossless check passes on a420039, with both drafts.

T3 (≥1,000 at c8) is 4% away. Remaining levers:
- c8 joins (~2.8 ms a step);
- a 64-row verify;
- DSpark has no batched draft yet (DFlash2 only).

Later still, 2026-10-01:
- **#1232 merged:** one launch for every member's selector walk (picks identical). c8 ~960 → 982–998.
- **#1233 merged:**
  - a joining prompt's tail rides the group's verify;
  - full-attention k/v use the GEMM in wide passes (`SPARKINFER_ATTN_GEMM` default 7).
  - c8 at T=1.0, the T3 setting: 964 → 1,013–1,017 tok/s. A later suite run gave 993, so the box spread is ±2% and T3 is met, but only just.
- **Measured and rejected:**
  - acceptance-weighted verify-row allocation: no change, members' acceptance is too similar;
  - a 48/64-row verify at c8: 930/973 against 1,012 at 32 rows. More kept tokens, but the verify cost grows faster.
- **c8 step now** (32-row verify, T=1.0): 20.7 ms. Verify 15.7, draft 3.3, join 1.6.
- **Open levers:**
  - per-member draft kernels (convs, RoPE, attention, KV copies; ~440 launches a step) as segment-batched kernels;
  - a deferred GDN commit (replay the accepted rows inside the next scan: saves a third of the state traffic);
  - T1 (single stream at T=1.0, 192 vs 230) can't use the wide verify without breaking bit-identity, so it needs acceptance or draft cost.

2026-10-01, continued:
- **AIPerf scoreboard, main ffe544b vs vLLM:**
  - chat c32 at 0.94× was the only chat loss;
  - 8K prompts at c4 / c16 / c32: 0.86 / 0.97 / 0.99×;
  - every other cell 1.11–1.21×;
  - our ITL beats vLLM in every cell.
- **#1234 merged: prefix cache sized by memory.** 256 entries, snapshots up to a quarter of RAM, 75% of the KV pool. `/v1/capacity` counts cache-only blocks as free, and a prefix-session turn evicts before refusing. Chat c32: 971 → 1,070 tok/s (1.04× vLLM), TTFT p50 1,575 → 1,121 ms; c16 TTFT 792 → 553 ms. The review found a refcount data race and a missing eviction; both fixed.
- **#1235 merged: a lone speculating request verifies the depth that pays.** Against its measured cost curve (rows 1–4 nearly free):
  - 14.5K-token prose: 96 → 119 tok/s;
  - DSpark chats: +16–20%;
  - DFlash2 single request: 231 / 210 / 205 tok/s at T=0 / 0.7 / 1.0.

  The review found that the tier-cut length wasn't reported back; fixed.
- **Measured and dropped:**
  - more weight rows per CTA in the verify's exact LM head: no change;
  - acceptance-weighted rows inside groups: −7–9% at c4/c8, because rows are nearly free in a group.
- **Remaining gaps:**
  - The 8K-prompt AIPerf cells: an LRU cache scan of 18 prompts against room for ~12. vLLM's larger KV pool keeps them; ours is sized from `--ctx`.
  - T1 (230 at T=1.0, single stream): 205–224 depending on the prompt set.
  - T2 (greedy 16K hard prose ≥200): 119. That needs better draft acceptance at long context.
- **Measured and dropped:** a lone member through the batched draft. Draft 1.83 → 1.53 ms, but acceptance 3.38 → 3.25: a wash.
- **T1 on main e819e76, spec_bench DFlash2 single stream:** T=1.0 at 228–230 tok/s, so 230 is met only within noise. T=0 is 231–235.

0.6.0 regression pass, v0.5.14 against main e819e76 (2026-10-01; five models, engine benches, scores and DSpark lossless):
- Teacher-forced scores identical on four models. Qwen3.6 differs run to run on the same build, even under `SPARKINFER_DETERMINISTIC=1` (top-1 0.78–0.81 between repeats), so that is not a regression.
- Qwen3.8 NVFP4 decode +5–8% at every context; Bonsai decode +8–13%, cb c32 +23%; DSpark LOSSLESS on both, 94.1 → 99.1 tok/s.
- Three single-run drops were rechecked with repeats:
  - Qwen3.8 GGUF cb c16 (987 → 928): noise, 978–982 on both builds.
  - Bonsai prefill @512 (10,152 → 9,460): noise, ~10,100 on both.
  - Muse cb c32 (2,136 → 388): real, and v0.5.14 has it too. The run is bimodal on an out-of-memory edge; fixed in #1236.
- Final speculation on main (DFlash2, real prompts): c1/c2/c4/c8 212/410/714/1,040 tok/s at T=0.7, 215/402/675/1,011 at T=1.0.
