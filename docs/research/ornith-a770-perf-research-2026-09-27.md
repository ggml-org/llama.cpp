# Ornith-1.5-35B-A3B on Arc A770: where the time goes and what to try (2026-09-27)

Read-only research pass over the production Ornith service and a review of
`egorcherkasoff/local-ai` (a Docker CUDA recipe for the same model on a 12 GB
RTX 2060). No service restart, no source change, no timing campaign. All
numbers below are either (a) live source and log evidence, (b) op-level probes
run with `test-backend-ops perf` on this box while the production server was
resident but idle (`requests_processing 0`, `fuser` also showed Slack and codex
holding the render node), or (c) prior dated research in this directory. Per
the evidence rules in `CLAUDE.md`, (b) is order-of-magnitude evidence, not
campaign data; every lever below still has to go through
`~/projects/local-models/experiments/run-config.sh` before promotion.

Source: `589bf18cb` (ornith-improve). Installed service binary: b12275
`bb6908513`, IntelLLVM 2026.1.1, `GGML_SYCL_DNNL: no`.

## 1. Production baseline and config

Service `llama-gpu@Ornith-1.5-35B-Q4_K_M` (port 8089), effective args:
`--ctx-size 131072 --parallel 1 --fit on --fit-target 1024 --flash-attn on
--threads 12 --threads-batch 12 -ctk q8_0 -ctv q8_0 --spec-type ngram-mod
--moe-cache off`, env `GGML_SYCL_ENABLE_GRAPH=1
GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300`. Default `-b 2048 -ub 512`. Verbosity 3.

Measured by the harness (results.md run 22/23, b12275, graphs on):

| metric | value |
|---|---|
| tg128 | 20.7 t/s (48 ms/token) |
| tg128 @ d8192 | 19.5 t/s |
| pp512 | 126 t/s (4.07 s) |
| code decode with ngram-mod | 34.4 t/s (88% acceptance) |

Runs 16/17/18/22 show FA on/off, KV type (f16, q8_0, turbo4) and context
(32k vs 128k) all within noise of each other at decode. Attention is not the
bottleneck; the per-token floor is elsewhere.

## 2. Model shape (from the GGUF)

`qwen35moe`, 41 blocks (blk.40 is the unused MTP block, 521 MiB), n_embd 2048,
256 experts / 8 used, n_ff_exp 512, 10 full-attention layers (every 4th,
16 q-heads, 2 kv-heads, d=256), 30 gated-delta-net layers (16 k-heads,
32 v-heads, S=128). Vocab 248320.

| tensor group | bytes | note |
|---|---|---|
| routed experts, 40 layers | 18600 MiB | gate/up Q4_K all layers; down Q6_K in 21 layers, Q4_K in 19 |
| per expert layer | 498 MiB | 15.6 MiB of it read per token (8 of 256) |
| non-expert weights total | 1576 MiB | attn 611, ssm 141, shexp 73, embd 273, output 398 (Q6_K) |
| KV at 128k q8_0 | ~1.4 GiB | 10 layers only |

The routed-expert share of that is roughly 0.4 GiB (VRAM) + 265 MiB (host RAM)
per token, but every decode step also reads the non-expert weights except the
embedding table (attn 611 + ssm 141 + shexp 73 + output 398 = 1223 MiB;
embedding is a per-token row lookup, not a full read, so it's excluded).
Bandwidth floor per decode token is therefore closer to 1.9 GiB, not 0.7 GiB:
about 22 ms at the same rate, not 8 ms. Observed is 48 ms - still a real gap,
smaller than first stated, and still not fully explained by bytes alone.

## 3. Structural facts verified in source and live output

- **Fit is blind to free VRAM.** `llama-server --list-devices` reports
  `15473 MiB, 15473 MiB free` while the production process holds ~14 GiB, with
  and without `ZES_ENABLE_SYSMAN=1`. `--fit` therefore produces the same plan
  regardless of other tenants; `--fit-target 1024` is the only margin.
  Fit's per-layer plan is logged only at `-lv 4`; the production journal has
  no placement line. Estimate from sizes: ~22-23 expert layers on the GPU,
  ~17 on the CPU. **Estimate, not measured.**
- **CPU-resident experts live in the SYCL pinned host buffer**, not repacked
  (`make_cpu_buft_list` adds the host buffer type before the CPU extra bufts).
  Consequences: CPU decode uses the generic Q4_K/Q6_K vec_dot, op-offload
  streams pinned weights, and the `--load-mode none` warning is about the mmap
  copy, not about the compute path.
- **SYCL `supports_op` accepts MUL_MAT_ID for Q4_K/Q6_K weights wherever they
  live**, so the scheduler's op-offload (`GGML_OP_OFFLOAD_MIN_BATCH`, default
  32) can move CPU-resident expert layers to the GPU at prefill, copying up to
  498 MiB per layer per ubatch.
- **Prefill MUL_MAT_ID on SYCL is a per-expert loop**: gather rows, then one
  `ggml_sycl_mul_mat` per touched expert (256 x 3 matrices per layer), with a
  blocking `stream->wait()` per node. Only the single-token path
  (`ggml_sycl_mul_mat_id_mmvq_fused`) is fused; it covers Q4_K/Q5_K/Q6_K plus
  the IQ and legacy types.
- **Queues are plain in-order** (`dpct/helper.hpp create_in_order_queue`),
  no submission property. The dual-queue design from
  `sycl-a770-round2-decode-candidates-2026-07-25.md` section C0 was never
  implemented; the 2026-08-13 probe that confirmed its premise is the last
  word on it.
- **SYCL graph capture rejects a segment containing MUL_MAT_ID unless
  `ext_oneapi_async_memory_alloc` is available** (lazy reorder allocates USM).
  Whether Ornith's segments replay at all is unknown: results.md shows graphs
  on vs off at +2%, and `GGML_SYCL_GRAPH_PROFILE` prints only in a destructor
  at process exit, so it has never been captured for this model.
- **Scheduler split cost**: `.cpy_tensor_async` is NULL in the SYCL backend
  interface, so every CPU<->GPU input copy at a split is a synchronous
  `memcpy` + wait. `event_wait` also does a host wait. Each CPU expert layer
  costs two splits.
- GDN prefill uses the fused single-op path (`fused_gdn_ch` default on),
  which on SYCL is the recurrent kernel; the chunked kernel is a TODO.

## 4. Op-level probes on this A770 (shared tenancy, idle server)

`~/build-ornith-probe/bin/test-backend-ops perf` with Ornith shapes added
locally (patch reverted afterwards). "batched" =
`UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 UR_L0_BATCH_SIZE=64`.

### 4.1 Routed-expert MUL_MAT_ID, 256 experts / 8 used

| shape | SYCL default | SYCL batched | CPU 12 thr |
|---|---:|---:|---:|
| Q4_K gate/up, n=1 | 57.5 us | 27.7 us | 316 us (a) |
| Q6_K down, n=1 | 68.8 us | 39.3 us | 277 us (a) |
| Q4_K, n=4 (spec verify) | 1224 us | 860 us | 517 us |
| Q4_K, n=17 (spec verify) | 4248 us | 2271 us | 1406 us |
| Q4_K, n=512 (ub=512 prefill) | 50.0 ms | 13.6 ms | 28.3 ms |
| Q6_K, n=512 | 51.7 ms | 17.6 ms | 23.6 ms |
| Q4_K, n=2048 (ub=2048 prefill) | 57.8 ms | 17.2 ms | 64.5 ms |

(a) test-backend-ops spins a fresh CPU threadpool per graph, so the CPU n=1
figures include thread start-up; the server's persistent pool will be lower.

**Caveat:** these were collected under shared tenancy (the production server
resident and idle, Slack and codex also holding the render node per `fuser`)
per the read-only, no-service-restart scope of this pass. Labeling them
order-of-magnitude bounds the absolute error but does not remove contention as
a confound for the relative comparisons below (default vs batched, GPU vs
CPU) - two runs on the same box can be affected unevenly. None of the readings
or ranked levers below should be treated as a settled conclusion; each needs
re-verification under confirmed sole tenancy (stop the service, `fuser`
check, dmesg check per this fork's GPU-discipline rules) through
`run-config.sh` before it drives an implementation decision.

Readings:

- Decode-shaped MoE matvecs are fine (~180 us per layer for three matrices,
  ~4 ms per token across ~23 GPU layers). They halve under batched submission.
- Anything with n>1 falls into the per-expert loop. At ub=512 one layer costs
  ~150 ms on the GPU; 23 layers is ~3.5 s of the observed 4.07 s pp512. This is
  the prefill bottleneck. Going to n=2048 costs only 15% more per matrix, so
  the per-token cost drops ~3.5x with `-ub 2048`.
- The CPU at 12 threads beats the GPU's default loop for prefill (80 ms vs
  150 ms per layer at ub=512). If op-offload is engaging for the CPU layers
  today, it is a net loss until the GPU loop is fixed.
- Speculative-decoding verify batches (n=4..17) pay 20-75x the single-token
  cost per matrix. This is a concrete reason draft-mtp lost in runs 02/03/19-21
  and why ngram-mod only pays off at very high acceptance.

### 4.2 Everything else probed

| op | cost | per-token contribution |
|---|---:|---|
| GDN recurrent kernel, n=1 (16/32 heads, S=128) | 33 us | ~1 ms over 30 layers |
| GDN, n=512 | 1.6 ms | ~48 ms per 512-token ubatch (1% of prefill) |
| output head Q6_K 248320x2048 matvec | 1.01 ms | bandwidth-bound (~400 GB/s), fine |
| small dense matvec 512x2048 Q4_K, n=1 | 24 us | launch-latency floor per kernel |

### 4.3 Runtime latency probe (standalone icpx program)

| measurement | immediate (default) | batched cmd lists |
|---|---:|---:|
| 30 tiny kernels + 1 wait, per kernel | 21.5 us | 7.4 us |
| host submit cost per kernel | 6.8 us | 1.8 us |
| kernel + wait round trip | 39-93 us | 43 us |
| D2H + wait + H2D + kernel + wait (one split boundary) | 115 us | 208 us |

A qwen35moe decode step is roughly 40 layers x 35-45 nodes plus head, so on
the order of 1500 kernels; at ~21 us each that is ~30 ms, which together with
~17 CPU expert layers (~5-15 ms) and ~34 split boundaries (~4-7 ms) accounts
for the observed 48 ms. The 2026-08-13 probe measured the same lever on dense
Mistral: `imm=0 + UR_L0_BATCH_SIZE=64` gave +32% tg128 at depth 0 and +21.6%
at 16k, and found the fork's default to be immediate mode.

## 5. What the sibling repo does that transfers, and what does not

`egorcherkasoff/local-ai` (RTX 2060 12 GB target, verified on a 5070 Ti):

- Transfers: bigger `-ub` because expert weights are re-streamed once per
  ubatch (their `-ub 1500`, `-b 8192`); fewer decode threads than SMT count
  (they found 12 > 16 on decode); `--no-op-offload` as an A/B knob; keeping
  experts in host RAM is viable when the GPU-side per-layer overhead is small.
  Their 45-52 t/s with all 40 expert layers on CPU says the CUDA per-split cost
  is small; on SYCL it is not, so their placement is not directly portable.
- Does not transfer: YaRN 512K (KV at 512k q8_0 is ~5.6 GiB here, which would
  push ~11 more expert layers to the CPU); `FIT=off` with hand placement is
  optional here because fit is deterministic on this box; the CUDA image,
  PTX-JIT cache and `--cache-ram` notes.
- Their decode 45-52 t/s vs 20.7 here is the gap to close; the analysis above
  attributes it to SYCL launch cost and split cost rather than to weights.

## 6. Ranked levers

Config-only (run through `run-config.sh`, paired against the current
baseline, one change per run):

1. **Batched submission.** Add to a drop-in:
   `Environment=UR_L0_USE_IMMEDIATE_COMMANDLISTS=0`
   `Environment=UR_L0_BATCH_SIZE=64`. Cross with `GGML_SYCL_ENABLE_GRAPH`
   in {0,1} as a paired 2x2: both cut per-kernel submission cost, and whether
   their gains add is unmeasured until this run.
   Watch pp512 too: the 2026-07-19 campaign saw pp512 -3% from graphs.
2. **Diagnostic run**: `-lv 4` once, with `GGML_SYCL_GRAPH_PROFILE=1
   GGML_SYCL_FA_PROFILE=1`, then stop the service so the profile prints.
   Yields the real fit placement (n_gpu_layers, per-layer overrides, buffer
   sizes) and `replay_calls` vs `direct_calls`. If `direct_calls` dominates,
   graph capture is not engaging for Ornith and the compat gate in
   `check_graph_compatibility` (MUL_MAT_ID / async alloc) is the place to look.
3. **Prefill**: `-ub 2048` (fit will move one or two expert layers to the CPU
   to pay for the compute buffer; check tg does not regress), and
   `--no-op-offload` as a separate A/B. Also `GGML_OP_OFFLOAD_MIN_BATCH=1024`
   as a softer variant.
4. **CPU side**: `-t 6 --cpu-mask 3F` (V-cache CCD only) vs `-t 12`; `--poll 100`;
   `-tb 24` for prefill only; CPU governor `performance` (amd-pstate-epp is on
   `powersave` / `balance_performance`). Each is a cheap paired run.
5. Leave `--moe-cache` off: fit already fills VRAM, and `MOE-CACHE.md`
   documents why the fit/provider budget disconnect can regress decode.

Code (each needs the CPU oracle green and a paired campaign):

1. **Grouped MUL_MAT_ID for n>1 on SYCL.** One launch per matrix over the
   expert-sorted row mapping (which `mmid_counting_sort_rows` already builds),
   using the existing `mul_mat_vec_q_reorder_ncols` machinery for n<=8 per
   expert and an MMQ tile for larger groups. Fixes both the ub=512 prefill
   floor (256 launches per matrix per layer) and the spec-decode verify cost,
   which would make draft-mtp worth re-testing.
2. **Dual-queue submission** (C0 in the 2026-07-25 candidates doc): a batched
   in-order queue for decode-shaped graphs, an immediate one for prefill and
   batch-1 latency. Same effect as lever 1 above without a process-wide env.
3. **Split cost**: implement `cpy_tensor_async` for host<->SYCL copies and use
   events instead of `queue.wait()` at split boundaries. Saves at most a few
   ms per token; only worth it after the two above.
4. **Graph capture for MoE segments**: if the diagnostic shows `direct_calls`,
   accept MUL_MAT_ID nodes whose weights already carry the reorder flag (no
   allocation left to do) instead of requiring async USM alloc.

## 7. Not claimed

- No placement was measured; the ~23/17 GPU/CPU layer split is arithmetic from
  tensor sizes and the 1 GiB fit margin.
- Op probes ran with the production server resident and other DRM clients
  open. They are consistent with each other and with the 2026-08-13 campaign,
  but they are not paired campaign samples and should not be quoted as such.
- The 1500-kernel decode estimate is a static count of graph builders, not a
  measured node count; `GGML_SYCL_GRAPH_PROFILE` `nodes=` would give the real
  number.
- Whether op-offload is currently engaging for CPU expert layers at prefill is
  inferred from `supports_op`, not observed; the `--no-op-offload` A/B decides.
- On Ornith, batched submission was probed only in a synthetic loop (the dense
  Mistral +32% is a real paired tg128 measurement on a different model); its
  real gain on Ornith decode depends on segment length between CPU splits and
  may be smaller.

## 8. Addendum (same day): upstream PRs merged and what they measured on the A770

Cherry-picked onto `ornith-improve` with `-x` provenance (authorship preserved; #29375 squashed
because its later commits rewrite its earlier ones); `master` received the same ports through the
stacked `upstream-pr/*` branches (#64, #65, #68 merged; #66, #67, #70 open at the time of writing). Probes: `test-backend-ops` on this A770 with
the production SYCL server resident but idle, default perf shapes, pre-PR binary as baseline.

| PR | what | A770 result |
|---|---|---|
| ggml-org#29476 vulkan GDN Intel tuning | subgroup-16 GDN pipeline on Intel | GDN 32 heads S=128: 24.9 -> 6.2 us decode, 6986 -> 1500 us at 512 tokens; 40/40 correct |
| ggml-org#29186 Q8_0 ESIMD DMMV + wide MMVQ | `GGML_SYCL_MMVQ_WIDE` (default 1) | q8_0 n=1 matvec 177 -> 157 us, 187 -> 172 us (+9-13%); the gain is the ESIMD DMMV, wide MMVQ alone is neutral to -7% |
| ggml-org#29375 Q5_K reorder MMVQ + fused GLU | Q5_K vec_dot restructure, row pairing at 3..5 cols | as-is: pairing halves throughput at n=3..5 (275 -> 529, 298 -> 676, 335 -> 823 us); unpaired the restructured vec_dot is slow at n=8 (2109 us). Fork pairs Q5_K only from 6 columns: n=1..5 at 1.00-1.05x of pre-PR, n=8 0.93x (1261 vs 1167 us), n=512 1.00x |
| ggml-org#29245 grouped MoE XMX GEMM | `GGML_SYCL_XMX_GATHER_TYPES`, IQ4_NL / IQ3_S only | correct (joint_matrix at SG16 JIT-compiles and passes here); no measurable gain at MoE shapes with 256 experts x 512 tokens; dense fused dequant GEMM 0.99x |
| ggml-org#29506 ExternalProject SYCL build | `GGML_SYCL_SEPARATE_BUILD` (default OFF) | build-system only; its `#if GGML_SYCL_DNNL` fix was already in the fork |

Correctness: q5_K / q8_0 / iq4_nl / iq3_s `MUL_MAT`, `MUL_MAT_VEC_FUSION` (1265 cases) and
`MUL_MAT_ID` all pass against CPU; `test-sycl-turbo-correctness` default sweep 0 GATE-FAIL. The
`MUL_MAT_ID` sweep aborts at the TQ3_1S / TQ4_1S cases (`unsupport src0 data type` in
`ggml_sycl_op_mul_mat_vec_q`) on the pre-PR binary as well: SYCL `supports_op` accepts MoE
matmuls on the TQ weight types but no matvec kernel exists. Pre-existing fork bug, not caused
by these PRs.

**Pre-existing multi-column MMVQ cliff on the A770 (Q4_K, the Ornith weight type).** Same probe,
`MUL_MAT` q4_K, us per call:

| m x k | n=1 | n=2 | n=3 | n=4 | n=5 | n=8 | n=512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4096 x 14336 | 108 | 199 | 191 | **660** | 278 | **2169** | 1825 |
| 4096 x 4096 | 46 | 82 | 85 | **225** | - | **654** | - |
| 14336 x 4096 | 100 | 173 | 177 | **590** | - | **2068** | - |

n=4 (Q4_K pairs rows at 3..4) costs 3.5x n=3, and n=8 (last MMVQ column count before the GEMM
route) costs more than n=512. Speculative-decoding verify batches of 4 and 8 tokens land
exactly there for every dense projection. The Q5_K experiment above points at the mechanism for
n=8: `mul_mat_vec_q_reorder_ncols` has a "shared weights" single-row path (load the weight block
once, loop activations over the columns) that types with `reorder_vec_dot_shared_weights` take;
at 8 columns that path is ~2x slower than either the plain per-column vec_dot (Q5_K before the
PR, 1167 us) or the two-rows-per-subgroup variant (1261 us). Q4_K has used the shared-weights
path all along. Follow-up candidates, in order: pair rows or use the plain vec_dot for n >= 6,
restrict Q4_K pairing to n=3 (the n=4 cliff), or route n >= 6 to the GEMM path on this arch;
re-measure each. Not done here.
