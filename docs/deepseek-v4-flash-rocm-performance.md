# DeepSeek-V4-Flash ROCm prompt-processing and raw-decode performance plan

Status: living canonical engineering record
Owner branch: `perf/dsv4-rocm-pp-20260803`  
Base: `b88a59fbc6ac255e6bf5e2dd790f559c89ce911c` in Edwin's llama.cpp fork  
Target host: `edwin@192.168.1.161` (`webhie`)  
Last updated: 2026-08-04
Current phase: accepted four-optimization PP stack is closed; target-only raw-decode baseline is blocking; indexed CSA is on hold pending raw-TG evidence.

## 1. Objective and success criteria

The completed phase optimized DeepSeek-V4-Flash prompt processing on the four-V620 ROCm host. The current primary objective is target-only raw-decode throughput and latency on the accepted four-optimization stack, without changing model math or sacrificing existing llama.cpp deployment modes. Keep generic/non-HIP fallbacks and structure upstream work as independently reviewable changes when practical.

A phase-specific change is successful only when all of the following hold:

1. A matched before/after run shows a repeatable gain in the target metric: PP for closed PP work, or raw TG and ms/token for current decode work.
2. The changed operation is proven to execute in a trace or targeted test.
3. DSV4 layer-reference versus tensor-split validation remains green.
4. Non-target contexts/modes do not regress materially; context-specific wins are reported separately.
5. Exact source/build/model/runtime identities and raw logs are preserved.

Phase boundary (2026-08-04): the initial PP loop is complete with four guarded optimizations (J16 MMQ, exact-shape HC, LID subwave-4, and IQ3 J16 T128). Existing PP throughput and PP context scaling are not raw token-generation evidence and must not be used as the TG baseline.

The supplied vLLM throughput numbers are not an acceptance gate. The reference vLLM hardware, checkpoint, GPU count, batching, and residency are unmatched. Four PCIe-connected gfx1030 GPUs have very different low-precision and matrix capabilities from B200/B300.

## 2. Fixed target facts

### Hardware and system

| Item | Value |
|---|---|
| GPUs | 4 x AMD Radeon Pro V620, Navi 21, `gfx1030` |
| VRAM | 34,342,961,152 bytes per GPU |
| PCIe | GPU0-2 Gen4 x16; GPU3 Gen4 x8 |
| GPU topology | all GPU pairs are PCIe, 2 hops, ROCm topology weight 40; no XGMI |
| CPU | AMD Ryzen Threadripper PRO 3945WX, 12C/24T |
| RAM / NUMA | 123 GiB, one NUMA node |
| OS | Ubuntu 26.04 LTS |
| Kernel | `7.0.12-v620all140test1` |
| HIP | 7.14.60850-0000000 |
| ROCm compiler | clang 23 snapshot, target `gfx1030` |
| Disk constraint | root filesystem had about 17 GiB free on 2026-08-03 |

Implication: inter-GPU collectives and mirrored tensors cross PCIe. GPU3's x8 link is a likely imbalance source. Avoid assumptions based on NVLink/XGMI systems.

### Source and build

- Sole base checkout: `/home/edwin/llama.cpp-rdna2`.
- Base commit: `b88a59fbc6ac255e6bf5e2dd790f559c89ce911c`.
- Base branch contained two important local fixes:
  - `b72684c0d`: avoid gfx1030 DSV4 top-k index corruption.
  - `b88a59fbc`: extend bitonic argsort/top-k beyond 1024 columns.
- Existing build: Release, `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1030`, `CMAKE_HIP_ARCHITECTURES=gfx1030`, `GGML_NATIVE=ON`, LTO off.
- Preserve `/home/edwin/baseline-bin` as a frozen A/B control. Record its version before use.

### Primary model in the current service

- Target: `DeepSeek-V4-Flash-0731-UD-IQ2_M`, three GGUF shards, about 90.9 GB total.
- MTP: `DeepSeek-V4-Flash-MTP-Q4_0.gguf`, about 4.2 GB.
- Main checkpoint tensor inventory includes IQ2_XXS/IQ3_XXS experts and Q8_0 indexer projections; this is not equivalent to a native vLLM FP4/FP8 checkpoint.
- The last recorded production service used all four GPUs, tensor split `1,1,1,1`, FA on, F16 K/V, context 262144, batch 2048, ubatch 1024, and MTP n-max 2 at sampling temperature 1.0. It was stopped before this new phase. The raw-decode baseline must load no draft model and use no speculative flags.

## 3. Baseline evidence available before this project

The only attested DSV4 sweep found so far is `/home/edwin/dsv4-dspark-sweep-20260802/results.tsv`:

| Main model / draft | Prompt | Generation | Notes |
|---|---:|---:|---|
| UD-IQ3_XXS + BF16 DSpark, n-max 16 | 13.9 t/s | 20.1 t/s | one fixed 512-token CLI run |
| UD-IQ3_XXS + Haraldh MXFP4/Q8 DSpark, n-max 16 | 9.2 t/s | 19.4 t/s | one fixed 512-token CLI run |

These are provenance data, not a statistical baseline. The user reports roughly 80-300 PP t/s from the current IQ2_M server under real inference, but the prompt length, cache state, concurrent load, and timing definition vary; record this as workload-dependent field evidence, not a controlled result. The new baseline must use the IQ2_M target, current base commit, repeated PP-only runs, and no speculative draft during PP attribution unless explicitly testing draft overhead.

Reusable assets:

- `/home/edwin/run_dsv4_mtp_sweep.sh`: deterministic four-GPU CLI command pattern.
- `/home/edwin/llama.cpp-rdna2/tests/test-dsv4-validation.sh`: layer-reference/tensor-split, replay, and KV-reuse correctness gate.
- `/home/edwin/baseline-bin`: frozen binaries for A/B testing.
- `/home/edwin/llama-jobs`: established command/result/log artifact convention.

Do not cite incomplete `haraldh_n16_p02.log` as a result.

### Raw-decode evidence status: BLOCKING / NOT YET ACCEPTED

No stable, speculation-disabled context-depth TG sweep has been accepted on the current four-optimization stack. The historical 20.1 t/s value used DSpark, and the 20.208 t/s main-only observation came from a single failed MTP diagnostic; neither is a raw-decode baseline.

Required closure: identical target model/quantization/layer split; MTP and DSpark absent; fixed measured generation count; at least five valid repetitions at every required depth, including 32K and 64K; scheduler/backend residency attested; exact command, commit, clocks/power state, and artifacts recorded.

Harness status (2026-08-04): M5.0 tooling and its non-GPU validation are complete, and the restore-equivalence gate is now closed, but no raw-TG sweep has run. `scripts/dsv4-rocm/run-tg.sh` provides separate performance and residency modes, manifests, phase-aware setup/sample watchdogs, telemetry, exact-depth/repetition summaries, and measured-decode scheduler parsing. Static fixtures and fake-runner tests passed complete, incomplete, unstable, CPU-residency, measured-timeout, and setup-timeout cases. Model-dependent testing found that llama-bench's prior sequence-only restore is invalid for DSV4, while full context restore is bit-identical; `f97f5cdb0` makes context state explicit and fail-closed. The blocking next evidence is the actual five-sample TG/residency sweep.

## 4. Current DSV4 execution facts

The load-bearing source path is `src/models/deepseek4.cpp:606-846`.

### CSA

Current graph behavior:

1. Project LID Q and weights.
2. `GGML_OP_LIGHTNING_INDEXER` scans every visible compressed LID K row and emits every F32 score.
3. A separate `GGML_OP_TOP_K` selects up to 512 indices.
4. `build_top_k_mask()` fills a full-size `-INF` mask and scatters zeros at selected indices.
5. Raw SWA K and every compressed CSA K are concatenated.
6. Generic `build_attn_mha()` invokes flash attention with the dense arbitrary mask.

Therefore source inspection proves dense score materialization and dense attention operands. Runtime counters still must establish the fraction of wall time and whether HIP flash attention performs any useful mask pruning.

#### CSA / DSA external fact-check (research note, 2026-08)

Independent sources confirm the architecture in both directions: our source
facts are correct, and the dense-masked HIP path is a genuine departure from
DeepSeek's intended index/sparse design.

1. **The intended design is indexed-sparse, not dense-masked.** The
   DeepSeek-V4 paper and HuggingFace `DeepSeekV4` docs describe CSA as: a
   lightning indexer scores queries against the low-compression pool, then
   `gather`s the top `index_topk` blocks per query *before they reach core
   attention*, and "a sparse attention kernel reads only those keys." The CSA
   mask is `-inf` everywhere except the `k` indexer picks per query. Our HIP
   `build_csa_lid_attention` instead concatenates `raw_k` with **all** `T =
   ctx/4` compressed CSA keys and calls generic flash attention with a dense
   `-INF` mask scattering zeros only at the selected indices (`deepseek4.cpp`
   732-789). The graph therefore presents every compressed key to core
   attention rather than a compact selected operand. Source inspection alone
   does not prove the HIP flash kernel physically reads every masked element.
2. **Ratios/top-k match.** `DSV4_CSA_RATIO=4` (overlap), `DSV4_HCA_RATIO=128`
   (non-overlap), `indexer_top_k<=512` per query, and 64 indexer heads all
   match the paper and the Transformers implementation are correct in our
   graph.
3. **Full-score materialization is a reference-design warning, not this
   implementation's peak-allocation measurement.** A full-sequence reference
   tensor `[B,S,H_I,T]` would reach **256 GB at S=65,536** with the published
   V4-Flash dimensions, and StreamIndex (arXiv 2605.02568) reports that a
   streaming top-k design avoids that reference OOM. llama.cpp's fused
   `GGML_OP_LIGHTNING_INDEXER`, however, reduces indexer heads inside the
   kernel and materializes `[T,n_batch,1,n_stream]` for one ubatch at a time.
   It therefore does not allocate the 256 GB full-sequence tensor. It still
   performs O(S*T) total indexer work and F32 score traffic over a complete PP
   run, and O(T) scanning for each raw decode token.
4. **Attention cost consequence at production context is plausible but not
   yet locally attributed for TG.** Dense-masked flash receives `T = ctx/4`
   compressed entries (65,536 at 256K) rather than ~512 selected entries. The
   accepted 16K PP trace measured flash 10.29% and indexer 9.12%. Those shares
   motivate a long-context test; they do not prove either component dominates
   raw decode at 32K-256K.

Verdict: source and external architecture evidence keep indexed CSA and
streaming selection as credible long-context candidates, but they no longer
select the next patch. For the raw-decode track, CSA is carried/deferred until
the target-only TG sweep and a measured component breakdown establish its
share and crossover.

#### Local measured scaling (2026-08): whole-graph PP is super-linear

A same-day full-stack PP sweep (J16+HC+LID, non-traced llama-bench) measured:

| Native context | PP t/s (single run) | measured prompt time | cost vs 16K |
|---|---:|---:|---:|
| 16,384 | 372.1 t/s | 44 s | 1.0x |
| 32,768 | 117.4 t/s | 279 s | ~6.3x |

Doubling context 16K->32K multiplied whole-graph PP time by ~6.3x. This is
strong evidence of super-linear PP scaling and is consistent with dense CSA
attention plus LID/indexer work, but it is **not** component attribution: no
successful 32K measured-region profile exists. The 64K job exited 137 during
`warmup prompt run` before `measurement-start.ns` was written, so it provides
no 64K throughput, no measured elapsed time, and no additional scaling point.

Artifacts:
- `$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260804T025141.681162476Z-csa-scaling-16k-16384-d032b943d185-13070/`
- `$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260804T025414.444077827Z-csa-scaling-32k-32768-d032b943d185-7005/`
- `$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260804T030440.549424222Z-csa-scaling-64k-65536-d032b943d185-6545/` (incomplete warmup; no result)

### HCA and raw attention

HCA concatenates raw SWA entries with all visible 128:1 compressed entries and calls generic MHA. At practical context this is far smaller than CSA. Initial layers use raw 128-token SWA.

### HIP lightning indexer

`ggml/src/ggml-cuda/lightning-indexer.cu` has an NVIDIA WMMA path, but HIP is explicitly routed to the vector kernel. On gfx1030 it:

- accepts F32 Q and weights;
- loads/dequantizes K to F32 vectors;
- loops over all 64 indexer heads;
- writes one F32 score for every visible compressed entry;
- performs separate top-k afterward.

Do not plan around rocWMMA until support and generated instructions are verified for gfx1030. RDNA2 lacks the same matrix acceleration assumed by CUDA designs. Wave32/vector-dot and memory-layout tuning are more plausible architecture-specific paths.

### Multi-GPU

The meta backend has special handlers for LID, TOP_K, and flash attention. Source inspection alone does not prove the communication pattern or whether scores are replicated/materialized across devices. Profile P2P and collective traffic. Because all four V620s communicate through two-hop PCIe, communication can dominate otherwise fast local kernels.

## 5. Working hypotheses, ordered by decision value

### H0 - short-prompt PP is dominated outside sparse CSA

At 512 prompt tokens, visible compressed CSA length is below or near top-k, so indexed sparse CSA cannot explain or repair a 13.9 t/s result. Quantized MoE GEMMs, tensor-split communication, graph/scheduler copies, and compression/projection work are more likely.

Test: profile 512 and 2048 tokens, attribute time and bytes to routed/shared MoE, all-reduce/P2P, attention, LID, top-k, and graph copies.

### H1 - dense masked CSA dominates as context grows

Above the 512-compressed-entry crossover (roughly above 2K native positions), generic attention is passed every compressed entry. Source inspection establishes O(ctx/4) dense operands. The 16K->32K PP observation supports super-linear whole-graph scaling but does not establish component dominance. **Verdict: structurally supported for PP, carried/deferred for raw decode pending M5.0 and M5.6.**

Test: target-only raw decode plus measured component timing at 8K/16K/32K/64K (128K only if practical). Record CSA flash, indexer, TOP_K, mask construction, complete attention block, and whole-model TG separately.

Candidate implementation if confirmed: start with a decode-only selected-KV proof, then a direct indexed HIP attention kernel. Multi-query PP needs an indexed-attention layout/kernel; existing common-KV flash plus a simple `GET_ROWS` gather is not established for per-query selections.

### H2 - LID plus top-k is an independent long-context bottleneck

Even perfect sparse attention must scan all LID entries. The HIP vector kernel emits all reduced scores and top-k rereads them. **Verdict: partially confirmed for PP.** LID reached 14.87% before optimization and subwave-4 passed; separate selection was only ~0.31% at 16K. Raw-decode LID/TOP_K share and scheduler residency remain unknown.

Test: profile `lightning_indexer_kernel_vec`, rocPRIM top-k/argsort, score traffic, scheduler placement, and graph splits over the target-only context sweep.

Candidate implementation if confirmed: keep TOP_K GPU-resident, optimize the exact observed selector path, reduce score-buffer traffic, then fuse tiled LID scoring with hierarchical top-512 merge. Preserve a full-score reference path.

### H3 - PCIe tensor-split overhead limits PP

Four GPUs have no XGMI and GPU3 is x8. Multi-GPU collectives, mirrored activations, and meta scheduler transfers may explain low PP at short context.

Test: capture HIP copies/RCCL and per-GPU kernel timelines; compare one/two/four GPUs only with model residency/offload effects explicitly separated. Measure device imbalance.

**Verdict: rejected as the first-order limiter for the tested four-GPU PP configuration, not universally rejected.** The accepted trace measured RCCL near 9.8%, explicit copies near 0.2%, and did not identify the x8 GPU as slowest. Different topology, split, batch, or raw-decode graphs can differ.

### H4 - RDNA2 quantized MoE kernels dominate

The 284B model activates about 13B parameters/token. The IQ2/IQ3 expert path and routed MMQ shape may underutilize V620s during PP.

Test: attribute PP/TG to `MUL_MAT_ID`/MMQ, shared expert, activation quantization, and routing; record exact shapes, expert skew, achieved bandwidth/waves, dispatch count, and total time.

**Verdict: confirmed for the original PP profile** (routed MMQ ~40% and therefore the measured first target). It is not a current raw-decode verdict; the old percentage breakdown is stale after the accepted whole-model optimizations.

### Hypothesis verdict summary

| Hypothesis | Current verdict |
|---|---|
| H0 short PP outside sparse CSA | Confirmed for the tested initial PP profile; routed MMQ, not CSA, was selected first. |
| H1 dense CSA at long context | PP structure/scaling supported; raw-decode indexed CSA carried/deferred pending 32K/64K TG attribution. |
| H2 LID + TOP_K | LID confirmed and optimized for PP; TOP_K raw-decode residency/cost not yet established. |
| H3 multi-GPU/PCIe | Rejected as first-order for the tested PP configuration only. |
| H4 quantized MoE/MMQ | Confirmed for the original PP profile; fresh raw-decode profile required. |

## 6. Milestone plan

### M0 - reproducible baseline and profiler harness

Harness source under `scripts/dsv4-rocm/`; timestamped raw artifacts under `$HOME/llama-jobs/dsv4-rocm-pp/`:

- `manifest.sh`: hardware, source, build, model shard identities/optional hashes, environment, command.
- `run-pp.sh`: PP-only deterministic runner with repetitions and unique output directory.
- `summarize.py`: median/p95, raw timings, failures, and identities.
- `profile-pp.sh`: rocprofv3 trace wrapper plus rocm-smi sampling.
- `README.md`: safe use, including refusal to run while another llama process owns GPUs unless explicitly overridden.

Initial quick-iteration matrix:

- one complete short grid: prompt 512 and 2048, ubatch 256, batch 512;
- one separate long-context probe: prompt 8192, ubatch 256, two repetitions;
- modes: PP-only (benchmark-native PP), then fixed-prefix TG separately;
- repetitions: one built-in warmup plus three measured samples per completed short shape;
- measurement budget: absolute five-minute cap beginning at the first measured prompt run, explicitly excluding initial model load/context creation/warmup; JSONL preserves completed cases when truncated;
- truncated or shape-incomplete summaries are marked `complete=false`, exit nonzero after preserving evidence, and cannot be used as matched A/B results;
- quick summaries report median/range and latency; p05/p95 remain unavailable until at least 20 samples;
- final-validation expansion only: ubatch 128/512 and prompt 32K/64K/128K after the fast loop stabilizes;
- main run: IQ2_M, four GPUs, tensor split 1,1,1,1, FA on, no draft for PP attribution;
- comparison: draft enabled only in a separate PP+TG run.

Metrics: PP t/s and measured latency, peak VRAM/RAM, per-GPU utilization/clocks/power, kernel time/calls, HIP memcpy/P2P bytes, RCCL time, attention/LID/top-k/MoE stage percentages, and failures. Whole-process rocprof summaries include model-load/setup events and are not valid for stage thresholds; filter traces to the harness-recorded first measured-run and completed-record timestamps.

#### Controlled M0 results (2026-08-03)

The rebuilt `04c01936c` default dispatch produced matched medians of 367.246
t/s at 512 tokens, 364.486 t/s at 2K, and 292.528 t/s at 8K (batch 512,
ubatch 256, four-way tensor split, three/three/two repetitions). The 512 case
had a slow first sample; cite medians and preserve raw samples rather than
selecting the fastest value. The separate traced 8K run measured 238.528 t/s,
which demonstrates roughly 18% profiler overhead relative to its nearby
ordinary baseline and is not an A/B throughput number.

The harness-recorded measured interval for the trace is 34.345 seconds. After
mapping its realtime markers to rocprof monotonic timestamps, it contains
978,828 kernel events and 85.912 summed device-seconds across four GPUs. This
legacy trace's single clock calibration was backfilled on the same boot about
32 minutes later, so the interval duration is unaffected but boundary
uncertainty is unknown; new traces capture start/end mappings and reject more
than 1 ms of offset drift. Routed
`mul_mat_q` kernels account for approximately 40% of summed kernel time:
IQ2_XXS 18.12%, IQ3_XXS 9.66%, type 8 6.14%, type 13 3.43%, and smaller
quantized types the remainder. The largest single Tensile Cijk GEMM is 28.44%,
RCCL device all-reduce 8.01%, lightning indexer 6.30%, and explicit flash
attention 5.57%. Within the interval, H2D copies total only 46.7 ms and RCCL
host API calls 248.6 ms; the multi-second whole-process copy/init totals were
model-load pollution. This satisfies the >=35% M1 MoE/MMQ rule and does not
satisfy the communication or LID thresholds.

A compact measured-region trace at 16K confirms the crossover without the
multi-GiB HIP API/JSON output: 1,793,356 kernel events, 185.325 summed
device-seconds, and 70.382 seconds wall time. The dominant Cijk kernel remains
26.43%; lightning indexer rises to 11.11%, explicit flash attention to 7.06%,
routed MMQ totals about 34.8%, and RCCL device work is 7.28%. The attention
subsystem is growing, but neither explicit CSA nor LID/top-k individually
crosses 15% at 16K. The complete artifact is 809 MiB. A later ordinary
(non-profiled) 32K PP sample completed in 279 seconds, but no successful 32K
measured-region attribution exists; full rocprof output at that size also
exhausted disk. Thus 16K remains the longest accepted attribution point.

Measured-region artifacts:

- `$HOME/llama-jobs/dsv4-rocm-pp/20260803T155051.535173991Z-trace-base-8k-04c01936c-04c01936cfc8-2974/`
- `measured-region-summary.txt` and `measured-region-summary.json` in that run.
- `$HOME/llama-jobs/dsv4-rocm-pp/20260803T175617.261510915Z-kernel-trace-base-16k-e62763ecbf21-14872/`

### M1 - select one optimization from profile evidence

Decision rule:

- dense CSA attention >=15% of PP at 8K+ and scales with context: implement indexed CSA;
- LID/top-k >=15% and score traffic is material: fuse/tune LID selection;
- MoE/MMQ >=35% at 512-8K: tune the dominant routed expert path;
- copies/collectives >=20%: optimize tensor split/communication first.

Do not select by architectural elegance alone. Apply thresholds only to a trace interval aligned to measured PP, never to whole-process profiler totals. Compare only identical complete shapes using paired/interleaved base-candidate-base ordering.

**Selected first path:** RDNA2 routed-MMQ tile width, initially as an opt-in
screening control rather than a broad default. The existing selector chooses
J=64 from the full 256-token width even though compact routed tiles see far
fewer rows per expert. A focused 256-expert/top-6 fixture found J=16 reduced
IQ2_XXS latency from a paired mean 3049.7 us to 1575.8 us and IQ3_XXS from
3312.5 us to 1787.0 us, with zero A/B mismatches. Setting J=16 for the whole
model improved medians to 410.547 t/s at 512, 403.849 t/s at 2K, and 323.151
t/s at 8K. Trailing default controls measured 367.977, 364.810, and 294.796
t/s respectively, versus leading controls of 367.246, 364.486, and 292.528.
Relative to the midpoint of the bracketing controls, J=16 gained 11.7%, 10.8%,
and 10.0%; control drift was 0.1% at 2K and 0.8% at 8K. J=8 regressed
whole-model 2K PP to 350.359 t/s despite winning the isolated IQ3 case.

The control remains opt-in because expert skew changes the optimum: in a
16-hot-expert fixture, J=16 was roughly 43-45% slower than J=64 while J=32 was
best. Any automatic selector must account for routing concentration or be
narrowly guarded; do not make J=16 a generic RDNA2 routed-MMQ default from the
balanced fixture alone.

**Selected second path:** tune the exact skinny F32 hidden-channel mixer GEMM,
not indexed CSA yet. rocBLAS profile logging identifies the dominant Tensile
kernel as `rocblas_sgemm(transA=T, transB=N, M=24, N=256, K=16384)` with 344
calls per 256-token microbatch. DeepSeek-V4 has 43 blocks and calls
`build_hc_pre()` twice per block; across four tensor-parallel GPUs this is
`43*2*4 = 344`, exactly matching the Cijk, RCCL all-reduce, and fused
`dsv4_hc_post` counts. The source operation is
`ggml_mul_mat(hc_fn, flat_norm)`, named `hc_mixes`. Its selected Tensile macro
tile is 128x256, wasting most output rows for M=24. At 16K there are 22,016
calls (`344*64` microbatches), again exactly matching the model structure.

Commit `560635e3b` implements an explicit `GGML_HIP_RDNA2_HC_MIXES=1` screen
for only contiguous F32 M=24,N=256,K=16384 on wave32 RDNA2. The 12x16x256
LDS-tiled kernel launches 32 workgroups instead of using Tensile's single
128x256 macro-tile; unset/`0`, all other shapes/layouts/types/devices, and
non-HIP backends retain the existing dispatcher. Invalid values fail closed.
The focused graph benchmark brackets rocBLAS at 2052.95/2047.29 us around a
535.95 us candidate: 3.83x faster, with bit-identical output and max absolute
error 9.54e-6 versus a double-accumulation CPU reference. Graph-off measured
542.30 us. Near-shape N=1/128/255/257, M=23/25, and K=16383 all remained on
generic paths and passed the CPU reference; rocBLAS profile logging proves the
exact candidate emits no SGEMM while N=255 does. Independent review reported
PASS with no blocker/high/medium finding; its low nonfinite-reference test
finding was fixed before commit.

With `GGML_HIP_RDNA2_MMQ_J=16` fixed in all arms, bracketed whole-model PP
improved from control midpoints 414.468/409.075/325.116 t/s to
501.063/494.149/389.632 t/s at 512/2K/8K: +20.9%/+20.8%/+19.8% from the
second optimization alone. The combined gain over the earlier no-J16 default
control midpoint is approximately +36%/+35%/+33%.

The deployment gate is now complete. Commits `4dd19713f` and `56dd4177e`
parameterize the attested arms and pin the validation server to batch/ubatch
512/256; without the explicit physical batch, llama-server's default ubatch
512 would only exercise the generic HC fallback. At clean commit `56dd4177e`,
the fully hashed J16-only versus J16+HC run matched all six natural-text proxy
responses in content, token IDs, prompt bytes, counts, continuation, replay,
and KV reuse. Its single first-prompt observations were 326.653→370.314 t/s
(+13.37%) in tensor mode and 208.128→222.924 (+7.11%) in layer-reference mode.
These remain correctness/dispatch-sanity observations rather than statistical
speed evidence; bracketed llama-bench is the throughput evidence.

A fresh combined-stack compact trace at 16K confirms dispatch and sets the
third-phase direction. Its 59.514-second measured region contains 1,793,356
kernel events and 140.528 summed device-seconds. Name-matched `mul_mat_q`
remains 40.19%; the custom HC kernel is 10.28%, lightning indexer is 14.87%,
RCCL device work is 9.84%, and explicit flash attention is 9.40%. The old
rocBLAS HC family has zero calls; the custom kernel has the expected 22,016
calls and 14.445 device-seconds versus 48.981 seconds for the old Cijk in the
prior 16K trace, a trace-local 3.39x reduction. LID plus its separate top-k
kernel families is approximately 15.2%, just crossing the M1 threshold after
the first two compute improvements. Because raw LID scanning dominates its
small selection overhead and scales rapidly, the next evidence-backed local
phase is LID vector-kernel tiling first, not communication or top-k fusion. Name-matched MMQ remains the largest family and stays in the roadmap;
any further J-width automation must handle routing skew.

Per-agent analysis maps KFD agents through `agent_info.csv` to PCI BDF. Summed
all-kernel duration ranges only 34.016–35.772 seconds; the x8 device at
`0000:46:00.0` is 35.385 seconds and is not the slowest. RCCL sums are
3.670/3.355/5.232/1.568 seconds for buses 43/46/23/03 respectively, so role or
algorithm asymmetry persists but does not single out the x8 link. H2D duration
to bus 46 is 34.137 ms versus 23.722–25.258 ms for the others, but all measured
copy events total only 108.966 ms (0.18% wall), and the trace exposes no direct
D2D copy events for RCCL. These data reject communication-first under the 20%
rule; they do not measure bytes or prove PCIe bandwidth causality.

The flash-attention mean grows from 869.8 us/call at 8K to 1,200.1 us/call at
16K while call count doubles. Together with dense operands in the graph, this
is inconsistent with complete fixed-top-k arbitrary-mask tile pruning, though
it does not exclude partial pruning; CSA remains a later long-context phase.

The focused LID baseline closes the first limiter question. Commit `05a7e5731`
adds the exact batch-256 production shape to `test-backend-ops`: CPU-reference
correctness passes at KV=256, and ordinary ROCm0 performance is 504.82 us at
KV=256 and 7,856.11 us at KV=4096 (2.14/2.20 effective TFLOPS by the fixture's
operation convention). Trace launches use eight wave32s, 80 VGPRs, 128 SGPRs,
8,704 bytes LDS, and zero scratch. Filtered KV=4096 counters measure 74.135%
occupancy, 23.726 mean waves/CU, 6,481 VALU instructions/work-item, 16.456%
memory-unit busy, 95.474% L2 hit rate, 11,328 KiB fetches, 3,837 KiB writes,
zero LDS bank conflict, and 2.532% ALU stall by LDS. Therefore DRAM/LDS are not
the first-order limiters; FP32 vector/reduction instruction work and the
VGPR-limited 74% occupancy are. Selection kernels are only about 0.31%, so
phase three began with K-vectors-per-wave 4 versus the current 8, then a
32-head inner tile versus 16. Both temporary opt-in prototypes were screened,
rejected, preserved, and removed from the clean source. K4 reduced reported
VGPRs 80→72 but left occupancy unchanged (74.135→74.186%), doubled wavefronts,
and regressed A/B/A control-midpoint latency 9.21% at KV=256 and 10.48% at
KV=4096. H32 used 72 VGPRs and 16,896 trace-reported LDS bytes with no scratch,
but demonstrated no gain: +0.39%/+0.26% latency, within short-shape control
drift. Only the retained KV=256 CPU-reference fixtures passed; the losers did
not undergo the comprehensive correctness contract and are not deployment
candidates. Simple launch/head tiling is therefore exhausted. A final bounded
reduction/instruction screen was justified before pivoting because the current
kernel executes eight width-32 reductions per lane/head.

The temporary `GGML_HIP_RDNA2_LID_SUBWAVE=4` prototype assigns one K vector to
each four-lane subgroup while retaining eight K vectors/wave and the same
launch. Each lane computes the original partials for logical lanes
`r,r+4,...,r+28`; explicit register additions reproduce XOR offsets 16, 8, and
4, followed by the existing width-4 XOR offsets 2 and 1. Thus subgroup lane 0
has the same FP32 tree and head accumulation order as the current kernel.
Read-only review confirmed the arithmetic/index/tail mapping and caught a
host-preprocessor guard that initially eliminated the candidate; the corrected
HIP guard then proved dispatch.

The corrected fast screen was unusually strong: control-midpoint latency fell
509.26→271.69 us at KV=256 (-46.65%, 1.87x) and 7,910.92→4,158.95 us at
KV=4096 (-47.43%, 1.90x). A new deterministic graph fixture then found the
first candidate was not yet bitwise: KV=1 differed in 91/256 values by at most
5.96e-8 because global `-funsafe-math-optimizations` reassociated the explicit
register tree. That attempt is excluded. A kernel-local
`#pragma clang fp reassociate(off)` restores the intended tree without changing
the generic kernel.

Commit `7a75d8a5a` contains the opt-in guarded path and manual differential
fixture. The authoritative artifact proves separate generic/subwave kernel
names with rocprof and bitwise equality for KV=1/63/64/65/256/4096, graphs off
at 256/4096, batch-1 fallback, and ineligible batch 255/257 plus KV=4097.
Four executed generic fallback cases cover F16/F32, 32/64 heads, and 1/4
streams. Ten process-level A/B/A cycles per shape (30 observations) give
437.90→254.95 us at KV=256 (-41.78%, 1.72x; 0.046% control drift) and
6,884.95→3,611.25 us at KV=4096 (-47.55%, 1.91x; -0.151% drift); every process
has one stable output hash per shape. The production CPU-reference fixture
passes. Candidate counters show the intended mechanism: VALU instructions per
work-item fall 6,481→2,923 while wavefront count stays 131,072. The pragma
raises trace-reported VGPRs to 88 and measured occupancy falls 74.135→61.505%,
but instruction reduction dominates; SGPR 128, LDS 8,704 B, and scratch zero
remain. Three review rounds end with no blocker/high/medium finding; the final
fixture binds the expected path to env/shape/reference-produced headers and
the exact validated V620, while retained kernel-name traces close dispatch
attestation.

The J16+HC-held-constant whole-model and corpus gates now pass at clean commit
`9f4808637`. In the single-observation A/B/A whole-model bracket, candidate
LID subwave-4 is +0.94% at 512, -0.62% at 2K, +4.79% at 8K, and +10.18% at
16K versus the control midpoint. Short/mid control drift is elevated
(-2.52% to -3.50%), so those three points prove no material (>2%) midpoint
regression rather than a statistical gain; the 16K controls are stable at
+0.14% drift and establish the long-context whole-model win. The fully hashed
2,527-token proxy gate matches content, token IDs, prompts, counts, replay, and
KV reuse for all six layer/tensor responses. Its single first-prompt
observations are +0.21% layer and +1.70% tensor and remain correctness/routing
sanity, not statistical performance evidence.

Subwave-4 is therefore promoted as guarded optimization three for the known
stack, still explicit via `GGML_HIP_RDNA2_LID_SUBWAVE=4`; generic and
near-shape fallbacks remain intact.

The fresh compact 16K J16+HC+LID trace closes the attribution gate. It records
the same 1,793,356 measured kernel events/call families as the pre-LID trace,
while measured wall falls 59.514→50.679 s and summed device time
140.528→126.093 s. Lightning-indexer time falls 20.890→11.499 device-seconds
(-44.96%), matching the focused mechanism and accounting for most of the
14.435-device-second reduction. Post-LID summed-kernel shares are routed/MMQ
name matches 43.15%, HC custom 11.41%, flash tile 10.29%, RCCL 9.76%, and LID
9.12%; explicit copies remain only about 0.21% of wall. The x8 bus remains
neither the slowest all-kernel nor RCCL agent. Traced throughput is 323.294 t/s
and remains profiler-perturbed, so bracketed non-profiled runs are the accepted
performance evidence.

Routed MMQ is selected again as optimization four: the two J16 type-16/type-18
kernels alone are 27.14% of summed device time, and all `mul_mat_q` name matches
are 43.15%. Communication, LID follow-up, and flash each remain below their
selection thresholds.

A bounded RDNA2 configuration screen then tested mechanisms beyond J width for
the J16 IQ2_XXS/IQ3_XXS kernels. I=64 regresses all uniform/hot cases by
16.0-21.6% and changes low-order output bits; I=256 exceeds the supported
shared-memory configuration and aborts fail-closed. Launch-bound occupancy 1
and 3 are neutral (within 0.36%). Reducing block threads 256→128 is neutral to
+1.30% for IQ2 but improves IQ3_XXS by 16.08% uniform and 16.38% hot, with exact
outputs (`max_abs=0`) for both quant types/routes. The final candidate therefore
changes only RDNA2 IQ3_XXS J16 `fallback=false`; fallback=true and every other
J/type/backend configuration are untouched.

Commit `803a41c37` contains that one-line optimization-four candidate. With
J16+HC+LID fixed, whole-model control-midpoint gains are +2.37% at 2K, +1.70%
at 8K, and +1.69% at 16K; control drift is -1.49% to -1.87%. The initial
single 512 observation is excluded for short-context inference because every
arm's first repetition is graph-cold and candidate cold cost was unusually
large. A dedicated three-repetition 512 A/B/A uses the stable repetition
median and shows +2.11% with 1.01% control drift (508.109/513.265 controls
around 521.471 t/s candidate). Focused review finds no correctness blocker;
its residual medium is performance coverage on older/unclassified AMD targets
that share the RDNA2 config fallback. The change is functionally thread-count
parametric and exact on the V620, but remains guarded by the existing J16
specialization decision rather than becoming a new generic J default.

Both remaining gates for this candidate pass. The fully attested natural-proxy
gate at commit `fb2a0c85d` (`20260803-225603-iq3-t128-corpus-fb2a0c85d`, `complete=1`)
authorized only the committed T128 stack, so its two identical arms serve as a
determinism/correctness gate: all six responses match in content/tokens/prompts/
counts, and layer/tensor reference-vs-tensor equality passes; its -0.57%/-0.50%
first-prompt timings are re-run noise between identical builds, not an A/B.
A compact rocprof resource proof (`20260804-001910`/`002309` corrected runs via
`20260804-002602-iq3-t128-dispatch4-fb2a0c85d`) confirms the IQ3_XXS J16 kernel
dispatches with the 128-thread block: wavefronts 11,264→5,632 versus the IQ2
256-thread control while 25 hot-routing events/WF counters are recorded, and
the focused exact-output contract is unchanged. Tree is clean at `fb2a0c85d`.

Optimization four is therefore promoted: repeatable matched PP gain, focused
dispatch/counters, natural-proxy determinism, whole-model throughput, review,
and full source/build artifacts are complete, with J16/HC/LID guards intact.
Fused top-k, reduced-precision Q, rocWMMA, and device-local candidate merging
remain deferred.

Mapping and screening artifacts:

- `$HOME/edwin/llama-jobs/dsv4-rocm-rocblas/20260803T175331Z-ec1b7e64c-map-cijk/` (three-line aggregate rocBLAS profile)
- `$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260803T175022.376406151Z-trace-map-cijk-256-ec1b7e64c2cc-24970/` (single-microbatch trace)
- `$HOME/edwin/llama-jobs/dsv4-hc-mixes-sweep/20260803T181014Z-1d6a42983-prototype/` (tile sweep, correctness, fallback, graph, and dispatch proof)
- `$HOME/edwin/llama-jobs/dsv4-hc-mixes-sweep/20260803T182030Z-560635e3b-whole-model/` (J16-held-constant whole-model A/B)
- `$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260803T191856.045376424Z-kernel-trace-j16-hc-16k-52e0121043ad-23195/` (combined-stack 16K compact trace, aggregate and per-agent measured-region summaries)
- `$HOME/edwin/llama-jobs/dsv4-lid-study/20260803T195000Z-bd4d1b9aa-baseline/` (launch scaling, exact fixture correctness/performance, hardware counters, raw DBs, counter command, and screen contract)
- `$HOME/edwin/llama-jobs/dsv4-lid-study/20260803T201000Z-k4-217f2a271-prototype/` (discarded K4 source, correctness/fallback/invalid-env screen, A/B/A, trace resources, occupancy counters)
- `$HOME/edwin/llama-jobs/dsv4-lid-study/20260803T202000Z-h32-217f2a271-prototype/` (discarded H32 source, retained KV=256 reference test, A/B/A, trace resources)
- `$HOME/edwin/llama-jobs/dsv4-lid-study/20260803T204500Z-subwave4-087813f76-prototype/` (excluded: host-side `RDNA2` preprocessor condition removed the candidate; invalid env did not fail)
- `$HOME/edwin/llama-jobs/dsv4-lid-study/20260803T205000Z-subwave4-087813f76-prototype/` (corrected temporary source, CPU-reference/fallback fast screen, distinct trace dispatch/resources, restoration proof)
- `$HOME/edwin/llama-jobs/dsv4-lid-study/20260803T211000Z-subwave4-validation-3276edc81/` (excluded first deterministic attempt: detected 1-ULP reassociation drift at KV=1)
- `$HOME/edwin/llama-jobs/dsv4-lid-study/20260803T212000Z-subwave4-validation-3276edc81/` (authoritative bitwise/path/counter/repeated-process/fallback artifact; final source patch and binary hashes)
- `$HOME/edwin/llama-jobs/dsv4-lid-study/20260803T220000Z-subwave4-whole-model-9f4808637/` (J16+HC-held-constant 512/2K/8K/16K A/B/A; stable 16K +10.18%)
- `$HOME/edwin/llama-jobs/dsv4-corpus-validation/20260803T212823.803707936Z-attested-9f4808637e55-20974/` (fully hashed LID-off/on corpus acceptance; all six responses equal)
- `$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260803T215054.700650714Z-kernel-trace-j16-hc-lid-16k-fdde31252a63-8573/` (fresh post-LID 16K compact trace; aggregate/per-agent summaries)
- `$HOME/edwin/llama-jobs/20260803-221516-mmq-config-screen-6af98d65b/` (excluded strict-zero-tolerance first screen; correctly stops on I64 low-order drift)
- `$HOME/edwin/llama-jobs/20260803-221906-mmq-config-screen-tolerant-6af98d65b/` (I64 16.0-21.6% regressions and I256 fail-closed unsupported evidence; source restored)
- `$HOME/edwin/llama-jobs/20260803-222442-mmq-config-screen-core-6af98d65b/` (authoritative T128/occupancy screen; exact focused outputs, three process timings per type/route)
- `$HOME/edwin/llama-jobs/20260803-223310-iq3-t128-whole-model-6af98d65b/` (J16+HC+LID-held-constant 512/2K/8K/16K A/B/A; initial 512 single-observation excluded)
- `$HOME/edwin/llama-jobs/20260803-224633-iq3-t128-short-repeat-6af98d65b/` (dedicated three-repetition 512 A/B/A; stable-median +2.11%)

Screening artifacts:

- `$HOME/llama-jobs/dsv4-rocm-mmq-sweep/20260803T160812Z-04c01936c/`
- `$HOME/llama-jobs/dsv4-rocm-mmq-sweep/20260803T160916Z-full-model/`
- Full-model harness runs labeled `mmq-jauto-*`, `mmq-j8-*`, `mmq-j16-*`, and trailing controls `mmq-jauto-*-post`.
- Fast `prototype1024-validation/` screening and a separate `unique-fixture-validation/` below the focused sweep directory. The latter gives every expert/output row independently quantized weights at target N=512, batch=256, experts=256, and top-6 (K reduced to one 256-value quant block to keep setup bounded). J16 and J64 outputs match bit-for-bit for both IQ types and uniform/hot routes.

### M2 - correctness proof and microbenchmark

Before integration, add a focused backend op test or deterministic reference for the changed operation. Required cases include short visible length, exactly/above top-k, chunk boundaries, unequal sequence lengths where supported, and gfx1030-specific dispatch fallback. Because llama-bench uses synthetic token IDs and MoE routing is input-dependent, any exploratory win must also pass one fixed, recorded production-representative token corpus before acceptance.

The repository now includes `scripts/dsv4-rocm/corpus/technical-proxy.txt`, a
9,807-byte natural-text engineering proxy with SHA-256
`396c178b3f77e7a920473fedaa54d79d3c98df5a27baebfa9b7de62a793a71df`.
It tokenizes to 2,527 prompt tokens. This approximates the user's technical
assistant workload but is explicitly not a user-supplied production corpus.
`test-dsv4-validation.sh` can preserve server logs/responses via
`DSV4_OUTPUT_DIR`; `compare-validation.py` compares base/candidate content,
token IDs, prompts, counts, and first-prompt timings.

Default and J=16 runs both passed layer-reference versus tensor split,
continuation, replay, and KV reuse. All six base/candidate responses matched in
content and token IDs. The initial artifact used the pre-normalized corpus and
had incomplete attestation. The acceptance rerun at clean commit `ec1b7e64c`
uses the normalized 2,527-token bytes, strict response/timing validation,
exclusive output directories, full SHA-256 hashes of all three GGUF shards,
and hashes for the server executable plus seven resolved llama/ggml DSOs. Its
single first-prompt observation was 399.268→420.273 t/s (+5.26%) in tensor mode
and 259.319→291.886 (+12.56%) in layer-reference mode. These are correctness
and routing-sanity evidence, not a statistical performance result; bracketed
llama-bench runs remain the throughput evidence.

Artifacts:

- `$HOME/llama-jobs/dsv4-corpus-validation/20260803T173909.127798287Z-attested-ec1b7e64c2cc-6101/` (J-width acceptance artifact)
- `$HOME/edwin/llama-jobs/dsv4-corpus-validation/20260803T190100.516971835Z-attested-56dd4177e501-15597/` (J16-only versus J16+HC acceptance artifact; full hashes, batch/ubatch 512/256, `complete=1`)
- `$HOME/edwin/llama-jobs/dsv4-corpus-validation/20260803T212823.803707936Z-attested-9f4808637e55-20974/` (J16+HC+LID-off versus LID-on acceptance; six exact response matches; +0.21% layer/+1.70% tensor single observations)
- `$HOME/edwin/llama-jobs/dsv4-corpus-validation/20260803T184944.079084014Z-attested-56dd4177e501-25863/` (excluded interrupted attempt: all responses exist and match, but no final status/comparison artifact and `candidate.rc=120`)
- `$HOME/llama-jobs/dsv4-corpus-validation/20260803T165136Z-88c415d91/` (superseded pre-normalization artifact)

### Production server and MTP gate

Commit `eff519d09` adds a fail-closed production-config validator and
`38880f985` ensures failed comparisons preserve diagnostics. It runs fresh
main-only and Q4 MTP servers with the accepted J16+HC stack, tensor split,
262144 context, unified F16 KV, batch/ubatch 512/256, draft-mtp n-max 3 on
ROCm0+ROCm1, and a fixed 2,527-token prompt followed by 128 generated tokens.
It requires actual draft counters and exact generated content/token equality.

The first real run is a **failed acceptance**, not a deployment pass. Both arms
completed, and MTP genuinely drafted (71 attempts, 211 drafted, 55 accepted,
71 verification steps; 26.07% token acceptance), but greedy output diverged
from main-only at generated token index 41 (`32907` versus `12275`). A single
main-then-MTP observation also showed PP 381.585→345.794 t/s (-9.38%) and TG
20.208→19.951 t/s (-1.27%). Those deltas are diagnostic only, but they provide
no MTP speed evidence and exact deterministic equivalence failed. Production
MTP acceptance therefore remains blocked pending isolation of speculative
batching/verification, graph/cache behavior, or configuration effects. The
main J16+HC PP stack itself remains accepted by the non-draft gates above.

Failed diagnostic artifact (fully hashed source/server/DSOs/main shards/draft
model/corpus and exact commands):

- `$HOME/edwin/llama-jobs/20260803-194659-dsv4-prod-mtp-j16-hc-r3/`

Commit `3fc2c17f6` parameterizes n-max, quick/full hashes, and the actual runtime
graph-disable variable. A four-cell, 64-token bounded matrix confirms that
neither runtime graph capture nor n-max 1 restores equivalence. Main-only token
IDs are identical across all cells. MTP graph-on/off IDs are also identical for
each n-max: n-max 3 always forks at token 41 (`32907`→`12275`), while n-max 1
always forks at token 6 (`124793`→`14`). Graph toggling is therefore not the
cause. N-max changes the verification batch width/cadence and moves the fork,
which favors target batch-shape arithmetic or DSV4 recurrent rollback state as
the remaining mechanisms. Source inspection confirms every draft position is
sampled from target logits and emitted IDs come from the verifier, so there is
no evidence that accepted draft IDs intentionally bypass target verification.
Single-cell TG deltas range from -4.7% to +19.2% and are contradictory; they
remain unusable as a speed claim.

Matrix summary (quick identities linked to the full-hash run above):

- `$HOME/edwin/llama-jobs/dsv4-mtp-matrix-20260803T200941Z-3fc2c17f6/`
- source runs `$HOME/edwin/llama-jobs/20260803-200941-mtp-matrix-{g1n3,g1n1,g0n3,g0n1}/`

Prefix replay now narrows this further. Slot erase is not implemented for this
recurrent model (HTTP 501), and a two-token cap-1 replay made zero draft
attempts, so both attempts are explicitly excluded. The accepted diagnostic
uses a fresh target+draft server per arm, recomputes the original prompt plus
common generated indices 0–4 as a 2,532-token prefill, and requests eight
tokens with request-level n-max 0 versus 1. Both produce the exact main-only
continuation `[22648,124793,305,3676,18333,29,23393,1950]`; importantly, the
cap-1 server drafted four tokens and accepted two. Pure request-level target
verification batch arithmetic is therefore insufficient to reproduce the
fork after the prefix is recomputed as prefill. The remaining distinction is
accumulated sequential/recurrent state and earlier speculative rollback or
history.

Artifact:

- `$HOME/edwin/llama-jobs/20260803-204032-mtp-prefix-replay8-fresh-087813f76/` (`diagnostic-evidence.json` is authoritative)

The stateful continuation diagnostic now isolates the mechanism. Three fresh
servers generated the common first five tokens, then reused the resident slot;
each continuation processed exactly one prompt token and ended with 2,539
cached tokens. Cap-0 prefix→cap-0 continuation matches main. Cap-0 prefix→cap-1
also matches main even though the continuation drafts four and accepts two.
But cap-1 prefix (two drafted, one accepted)→cap-1 continuation reproduces the
exact original fork and continuation:
`[22648,14,3676,18333,14,41933,32907,38593]`. All arms emitted identical first
five tokens. Earlier speculative verification/rollback therefore leaves a
different DSV4 recurrent state even when visible tokens match; current-step
verification batching alone is insufficient. This is causal bug isolation,
not MTP acceptance or TG evidence.

Artifact:

- `$HOME/edwin/llama-jobs/20260803-210548-mtp-stateful-7a75d8a5a/` (`stateful-continuation/diagnostic-evidence.json`)

Iteration-14 diagnostics narrow this again. A fresh cap-1-prefix→cap-0
continuation still produces the forked trajectory
`[22648,14,3676,18333,14,305,14961,32907]` even though continuation drafting is
zero. The poisoned state is therefore in the target trajectory left by the
speculative prefix, not merely a stale draft context affecting later verifier
choices.

Three source-level experiments preserve exact patches, builds, commands, and
fresh-server output. First, forcing target and draft checkpoints for every
speculative attempt still forks. Second, removing CSA/HCA/LID compressed-cache
rows alongside the raw cache during bounded rollback still forks. Third,
forcing checkpoint serialization/restoration with `LLAMA_STATE_SEQ_FLAGS_NONE`
(including compressed caches) rather than `PARTIAL_ONLY` also still forks.
Each test restores and rebuilds the clean source afterward. This excludes
bounded `rs_idx` restoration, omitted compressed-cache removal, and partial
checkpoint contents as individually sufficient causes. It does **not** prove
those paths ideal in all cases; the stronger surviving hypothesis is
accepted-token re-evaluation/replay or multi-token recurrent verification
semantics, including an accepted draft processed in the same target batch.

Artifacts:

- `$HOME/edwin/llama-jobs/20260803-214024-mtp-prefix1-cont0-9f4808637/` (target-state isolation; continuation cap zero)
- `$HOME/edwin/llama-jobs/20260803-212523-mtp-force-checkpoint-9f4808637/` (forced target+draft checkpoint path; partial state)
- `$HOME/edwin/llama-jobs/20260803-214302-mtp-compressed-cache-fix-9f4808637/` (bounded compressed-cache removal experiment)
- `$HOME/edwin/llama-jobs/20260803-214553-mtp-full-state-checkpoint-9f4808637/` (forced full-state checkpoint experiment)

Iteration-15 sequential-replay experiments close the production decision
without claiming a repair. A reviewed candidate forced checkpoints for every
recurrent speculative attempt and, after a rejected draft, restored the
pre-verification target/draft state, discarded the precomputed replacement,
and evaluated the already-emitted target token alone. The focused
cap-1-prefix→cap-0 arm becomes exactly main. In the four-arm rerun, the two
cap-0-prefix controls and cap-1-prefix→cap-0 are exact, but
cap-1-prefix→cap-1 still reproduces the original fork. Rejection replay repairs
one poisoned-target path, but later recurrent target verification remains
batch/history dependent.

A stronger diagnostic then discarded every recurrent verification result,
restored the checkpoint, and advanced only through target single-token decodes.
It accepted zero drafts and matched six continuation tokens before forking at
index 6 (`23393`→`14961`). Repeating with full DSV4 state serialization rather
than `PARTIAL_ONLY` gives the same result. Thus even zero-accept speculative
verification/checkpoint history is not exactly target-only equivalent; combining
full checkpoints and target-single advancement is still insufficient. None of
these candidates is committed, and every wrapper restored/rebuilt clean HEAD.

Artifacts:

- `$HOME/edwin/llama-jobs/20260803-215521-mtp-sequential-replay-fdde31252/` (focused rejection-replay success)
- `$HOME/edwin/llama-jobs/20260803-215800-mtp-seq-replay-matrix-fdde31252/` (four arms; only cap-1-prefix→cap-1 still fails)
- `$HOME/edwin/llama-jobs/20260803-220554-mtp-target-single-verify-fdde31252/` (zero-accept target-single diagnostic; delayed fork)
- `$HOME/edwin/llama-jobs/20260803-220904-mtp-fullstate-target-single-fdde31252/` (full-state variant; same delayed fork)

Production MTP is therefore explicitly rejected/deferred for this exact-greedy
DSV4 stack. It has no accepted TG result and must remain disabled when token
identity is required. Reopening it requires a focused model-level recurrent
state/logit equivalence fixture that proves speculative verification plus
checkpoint round trips are identical to target-only single-token decoding;
server token plumbing patches alone are not sufficient. Do not relax token
equality or represent the zero-accept diagnostics as TG evidence.

### M3 - implementation and matched A/B

One implementation branch and one writer. Keep the base binary and build artifacts. A frozen baseline is valid only when its sibling llama/ggml DSOs are selected and hashed; an executable that resolves candidate-build libraries is not frozen. Run static/backend tests first, then DSV4 validation, then matched benchmark/profile. Report local kernel speed separately from the applicable whole-model metric (PP or raw TG).

### M4 - independent review and next decision

Fresh review must cover correctness/causality, HIP synchronization/memory safety, generic backend behavior, multi-GPU index semantics, and benchmark validity. Update this document with accepted findings and raw artifact paths.

## 7. Revised next-phase roadmap: raw decode first

### Evidence boundary

The accepted four-optimization stack and PP reproduction record remain valid. The next phase must not conflate three different evidence classes:

| Evidence | Status | What it can decide |
|---|---|---|
| Repeated PP through 16K | accepted | Existing PP optimizations and previous bottleneck choices. |
| Single-run PP at 16K/32K | accepted as scaling observation only | Whole-graph PP is super-linear; not CSA attribution. |
| PP 64K | missing | The attempt died during warmup before measurement start. |
| Target-only raw TG sweep | **blocking / missing** | Next decode optimization, TOP_K residency, TG cliffs, and CSA crossover. |

### Revised milestone order

| Order | Milestone | Required evidence |
|---|---|---|
| M5.0 | Raw TG baseline | Stable target-only median TG with MTP/DSpark absent. |
| M5.1 | ROCm backend-residency audit | DSV4 LID and TOP_K remain GPU-resident; CPU/GPU split counts recorded. |
| M5.2 | Large HIP TOP_K, **only if reproduced** | Exact selected indices and GPU residency through 64K; no CPU split. |
| M5.3 | Fresh whole-model raw-decode profile | Ranked elapsed device-time breakdown on the full accepted branch. |
| M5.4 | Next dominant MMQ/MoE optimization | Exact hot shape and end-to-end TG gain, only if MoE remains dominant. |
| M5.5 | LID + TOP_K optimization | Only if raw profile shows material share. |
| M5.6 | 32K/64K attention-scaling study | CSA/indexer/TOP_K/mask/block/TG timings and measured crossover. |
| M5.7 | Indexed CSA | Only if M5.6 supports it. |
| M5.8 | Decode-chain fusion | Only if measured launch/elementwise cost dominates. |
| M5.9 | Multi-stream overlap | After single-stream paths are optimized. |
| Separate | DSpark/MTP | Acceptance/state-management workstream; never baseline evidence. |

### M5.0 / P0 - establish a valid raw-decode baseline

Run the accepted target stack with **no draft model or speculative flags** (MTP off, DSpark off), identical model/quantization, exact `--split-mode tensor --tensor-split 1,1,1,1`, F16 K/V, flash attention on, identical batch/ubatch settings, and identical clocks/power policy. Record the exact seed/sampling contract if sampling is used (temperature, top-k/top-p/min-p, penalties, grammar, and EOS policy). Prefer benchmark-native fixed-token evaluation so exactly 32 target evaluations occur without sampler/EOS ambiguity; otherwise any early stop or differing generated count is incomplete. Record fixed input/generated token IDs where the harness supplies them and the actual KV depth before the first measured token.

Depth sweep: minimal/empty prompt (label it by the actual starting KV depth, not blindly `0`), 2048, 3072, 4096, 8192, 16384, 32768, and 65536. Use **tg32 for the entire sweep** to keep generation count fixed; use tg64 for the entire sweep only if it fits. At least five measured repetitions per depth. Context setup/warmup is outside the measured TG interval, but its time and failures remain recorded. Every repetition starts from a fresh equivalent KV state; reused/restored prompt state is allowed only after target-only logit/token equivalence to fresh prefill is attested.

Set `GGML_SCHED_DEBUG=2` for the residency audit and parse the measured decode graph, not model-load/prefill noise. Record for every depth:

- five raw samples, median tokens/s, median ms/token, range and MAD/median;
- actual initial KV depth and fixed generated-token count;
- every DSV4 `TOP_K` backend assignment and fused lightning-indexer backend;
- CPU and per-GPU graph-split counts, plus introduced copies;
- per-GPU utilization samples, clocks, power cap/state, VRAM, temperature;
- exact source/build/DSO/model identities, environment, and command.

P0 has two acceptance states:

1. **Observational pre-fix baseline accepted:** untouched target-only measurement integrity, identities, fixed token count, timing boundaries, and five repetitions are valid. A reproduced CPU LID/TOP_K assignment or repeatable TG cliff is recorded as the finding that routes to M5.2/M5.3; it does not erase the observation.
2. **Post-fix/deployment baseline accepted:** no context-dependent migration of DSV4 TOP_K/lightning indexer to CPU and no unexplained context-step TG cliff.

Both states require no speculative decoding/draft model; a stable five-run median (initial target MAD/median <=3%, otherwise increase tg/repetitions and retain instability as evidence); and an exact externally rerunnable command/artifact directory. A GPU split count need not be zero under four-way tensor execution; it must be stable/explained. A CPU LID/TOP_K split blocks post-fix/deployment acceptance until corrected or causally justified.

#### M5.0 harness readiness (tooling accepted; TG still missing)

Performance mode uses llama-bench `n_prompt=0`, `n_depth=<sweep>`, `n_gen=32`, and `--no-warmup` under one model load. llama-bench computes/restores the requested depth before its timer, then `test_gen` performs exactly 32 one-token target `llama_decode` calls with no sampler or EOS stop. Six raw repetitions are preserved; the first target-depth graph-cold sample is predeclared and excluded, leaving five accepted samples. `summarize-tg.py` validates the exact depth/config/repetition contract and recomputes t/s, ms/token, ranges, and latency MAD/median from raw nanoseconds.

Residency mode is a separate non-performance run (`n_gen=1`, one repetition, `GGML_SCHED_DEBUG=2 --verbose`). `parse-sched-debug.py` changes phase only on llama-bench progress markers, ignores setup/prefill assignments, and records measured-decode LID/TOP_K backends, CPU/ROCm-meta split counts, and scheduled split-input copies. Separating it prevents verbose scheduler output from perturbing accepted TG. The tensor split is spelled `1/1/1/1` because llama-bench reserves commas for parameter variants; this is its exact four-device equivalent of `1,1,1,1`.

The non-GPU monitor rerun passed for the code committed at `1e5519bf1`; its precommit source patch and file hashes are preserved at
`$HOME/edwin/llama-jobs/dsv4-rocm-tg/static-validation-20260804T0415Z-0376a55aacd6/`.
It preserved a reproduced/rejected FIFO notification deadlock: a fast child could exit before a later FIFO event had a reader. The accepted implementation uses an append-only phase log plus an atomically replaced latest-phase file. Expected/observed exits were 0/0 for stable performance and residency, 3/3 for missing depth and measured timeout, 4/4 for instability, and 124/124 for setup timeout. These are tooling results only.

The model-dependent restore gate (`3d23fff4a`, controlled diagnostic `400c47cd6`, full-context extension `5d80b8662`) used deterministic fresh prefixes at 2K/3K/16K, four greedy target steps, every full-vocabulary logit, exact token IDs, and a fresh re-prefill control. The fresh repeat was bit-identical at all depths (state byte mismatches 0, logit tolerance violations 0, max absolute difference 0). Sequence-only `llama_state_seq_get/set_data` then failed at every depth: 516,983 / 516,900 / 516,930 logit tolerance violations, maximum absolute differences 4.118656 / 0.818019 / 0.736117, and a 2K final argmax divergence. Therefore the sequence API is rejected for DSV4 depth reuse.

Full `llama_state_get/set_data` context state passed bit-identically at all three depths: state sizes 60,674,301 / 67,567,869 / 157,184,253 bytes; zero state-byte mismatch against fresh re-prefill; zero repeat/restore bitwise or tolerance/non-finite mismatches; identical four-token argmax paths. Accepted artifact:
`$HOME/edwin/llama-jobs/dsv4-rocm-state-equivalence/20260804T044114.706244382Z-context-state-controlled-5d80b8662a95-16000/`.
Rejected controlled sequence artifact:
`$HOME/edwin/llama-jobs/dsv4-rocm-state-equivalence/20260804T043609.677826936Z-state-restore-controlled-400c47cd68ce-27056/`.
The fixed sweep has one generation/batch configuration and unique depths, so reuse occurs only for repetitions 2-6 inside the same context. Commit `f97f5cdb0` adds explicit llama-bench full-context depth state and makes `run-tg.sh` reject any sequence-mode override.

### M5.1-M5.2 / P0-A - TOP_K residency, then fix only if reproduced

The premise that this branch lacks large HIP TOP_K is **not currently true**. Source audit at commit `925d93700` (implementation source unchanged by the subsequent documentation edits):

- `ggml/src/ggml-cuda/common.cuh:114-120` enables `GGML_CUDA_USE_CUB` on HIP when hipCUB is present;
- `ggml/src/ggml-cuda/ggml-cuda.cu:5129-5143` advertises all TOP_K/ARGSORT widths under that path;
- `ggml/src/ggml-cuda/top-k.cu:6-16,60-105,161-173` uses `rocprim::topk_pairs` for supported K and the DSV4 large-row shape, with hipCUB argsort fallback;
- commits `b60551777` and `b72684c0d` added the HIP partial top-k path and fixed gfx1030 DSV4 index corruption.

Therefore M5.2 is conditional. First reproduce a CPU fallback or incorrect result in the raw graph. If reproduced, determine whether the cause is a build lacking hipCUB/rocPRIM, a support-gate mismatch, a meta-scheduler assignment, or a shape/correctness bug. Do **not** replace a bounded support test with unconditional `return true` unless every advertised shape has a correct implementation.

If a new fallback is genuinely required, prefer in order: repair/use the existing `rocprim::topk_pairs` path; a stable segmented descending key-value radix sort as a correctness reference; then a specialized hierarchical k=512 selector. AMD documents that rocPRIM `partial_sort` does not support streams in graph-capture mode, so it is not the sole production answer for graph-captured execution.

TOP_K gates: candidate sizes below/above 1024 and around 4096; 2K-64K native contexts; fewer/exactly/more than 512 valid entries; multiple rows/streams; tied scores; `-INFINITY`; CPU-reference score/index agreement; deterministic repeated output; no CPU split. Exact index-set equality is required for unique scores. For boundary ties, preserve the accepted implementation's explicit index tie-break where one exists. If tie policy intentionally changes, require dense-reference downstream attention/logit/token equivalence in tied fixtures; score-threshold validity plus determinism alone is insufficient. Do not assume rocPRIM stability.

### M5.3 / P1 - fresh profile of the accepted branch

The original ~40% routed-MMQ profile is stale after J16, HC, LID subwave-4, and IQ3 T128; LID alone changed 16K whole-model PP by 10.18%. After any accepted whole-model optimization above 3%, prior percentage attribution is considered stale and the next target must come from a new profile of the complete accepted stack.

Use target-only decode measured regions and disk-safe profiling (targeted counters/kernel filters; never repeat the full >=32K rocprof CSV that exhausted disk). Rank by **total elapsed device time**, while also reporting wall time and dispatch count:

- routed expert MMQ and shared-expert FFN;
- activation quantization;
- lightning indexer and TOP_K separately;
- dense-mask construction;
- CSA/HCA attention and complete attention block;
- HC operations and output projections;
- copies, RCCL, synchronization, and graph/scheduler overhead.

Branch selection rule: use at least two independent targeted profiles at the decision context(s). Select a family only when it is the largest reproducible measured-region elapsed-device-time family, contributes at least 15%, and leads the runner-up by at least 3 percentage points. If ranks disagree or the lead is smaller, collect more evidence or keep both candidates open; do not select by elegance. Communication requires the existing 20% threshold. Any candidate must later show >=3% median whole-model TG gain at its target depth with <=2% median regression at required shorter depths, unless a different threshold is declared before measurement.

Decision branches:

- **A: routed/shared MoE dominant.** Record weight type, M/N/K, expert count, tokens/expert, wave size, dispatch count, total time, and achieved bandwidth. Optimize only the hottest observed combination; measure activation quantization with MMQ. Shared/routed overlap or final-add fusion is eligible only if its standalone cost is material.
- **B: LID/TOP_K dominant.** Keep TOP_K GPU-resident, optimize the actual GPU selector, reduce score-buffer traffic, then consider fused indexer + hierarchical selection (`tile -> wave/workgroup candidates -> global merge -> final 512`). Exact selection still scans all candidates; streaming bounds intermediate traffic, not asymptotic scan time.
- **C: attention meets the predeclared M5.6 gate at long context.** Then consider indexed CSA. CUDA draft PR #25917 is design/correctness reference only (open draft, CUDA MMA, author measurements); its tiling and crossover are not ROCm evidence.
- **D: elementwise/launch work meets the branch threshold.** Only then pursue decode-chain fusion (compression/norm/RoPE/cache conversion/insertion or inverse-RoPE/packing/projection preparation).

### M5.6 / P2 - mandatory 32K and 64K raw TG

32K and 64K target-only TG are mandatory before indexed CSA is accepted or permanently rejected. The existing complete 32K result is **PP-only** (`n_gen=0`, one sample), and the 64K PP attempt has no result; neither closes this gate.

Use tg32 (or tg64 uniformly if it fits), **five valid decode repetitions at both 32K and 64K**, one model load where valid, and separate context setup from measured generation. Disable duplicate 64K warmup if the harness supports a verified `--no-warmup`/equivalent path. Reuse/cache a prompt state only after proving target logits/tokens match fresh prefill. Keep A/B commands identical. A cap-limited, unstable, early-stop, or fewer-than-five point is incomplete and permits no CSA selection/acceptance/permanent rejection; repair the harness/resource issue or leave CSA on hold.

For 8K/16K/32K/64K (128K optional), record lightning indexer, TOP_K, mask construction, CSA flash, complete attention block, and whole-model TG. Indexed CSA may be selected only if repeated profiles show either (a) CSA flash >=15% of measured raw-decode device time at 32K or 64K, or (b) CSA flash ms/token grows >=1.5x when context doubles and its Amdahl-limited removable share projects >=3% whole-model TG gain. It must still satisfy the branch reproducibility rule and then demonstrate the actual >=3% TG gate.

### M5.7 - indexed CSA, only after selection

Phase A is a **decode-only**, one-query-per-stream selected-KV gather proof. Gather selected compressed K/V plus valid local-window K/V; keep attention sinks in the existing separate `sinks` argument (not a physical KV row). Preserve selected mask values or compact invalid `-INFINITY` candidates; top-k can contain invalid entries when fewer than 512 positions are visible. Map logical compressed indices through the allocated per-stream cache stride.

Prefill remains dense-masked in Phase A because each query has a different selected set and existing flash consumes common K/V per stream. Phase B is direct indexed ROCm attention; Phase C is fused/streaming indexer selection. Require a sparse-attention microbenchmark, dense-vs-indexed numeric/logit/token gates, scheduler residency, sinks/SWA/inverse-RoPE/causal/cache-wrap checks, and a measured dense/indexed crossover. Keep dense below that crossover.

### P3 - DSpark and MTP remain separate

Raw baseline and kernel selection use one target token per target evaluation with MTP/DSpark absent. Zero-accept speculative runs cannot establish TG, select kernels, or judge MMQ/CSA/fusion changes. Production MTP remains deferred for exact-greedy state divergence; reopen DSpark/MTP only after the target-only raw baseline is stable.

## 8. Correctness contract for indexed CSA (on hold until M5.6 selects it)

If indexed CSA becomes M5.7, it must preserve:

- shared K=V MQA, 64 query heads, 512-dimensional heads;
- per-query selected index sets and top-k <=512;
- local 128-token raw/SWA branch;
- one stable softmax over selected compressed entries + local entries + per-head sink, with sinks passed separately;
- original mask validity for selected entries, including partially valid/all-`-INF` top-k tails;
- causal visibility and compression completion boundaries;
- inverse partial RoPE/Hadamard rotation behavior;
- allocated cache stream stride, wrap/reuse/reset, and unequal stream lengths;
- deterministic duplicate/tie policy compatible with the accepted TOP_K contract;
- layer-owner device residency with no CPU fallback or unintended peer transfer;
- dense generic fallback and a force-reference switch for testing.

A dense mask is not a sparse performance implementation. The first gather proof is decode-only. Multi-query PP requires a supported per-query indexed-KV representation or a direct indexed kernel. Success requires numeric/logit/token equivalence gates and runtime/traffic scaling with selected + local entries after a measured crossover.

## 9. Decision log

| Date | Decision | Evidence | Status |
|---|---|---|---|
| 2026-08-03 | Use `/home/edwin/llama.cpp-rdna2` as sole base. | User direction. | final |
| 2026-08-03 | Branch from `b88a59fbc`, retaining gfx1030 top-k fixes. | Active fork history and source audit. | final |
| 2026-08-03 | Do not interrupt current server; request a GPU window before controlled runs. | External client connected; GPUs 99% busy. | final |
| 2026-08-03 | PP first; TG secondary. | Initial user objective; the four-optimization PP loop completed. | superseded by raw-decode phase 2026-08-04 |
| 2026-08-03 | Profile short-prompt MoE/communication and long-context attention/LID separately. | Existing 512-token 13.9 t/s result plus graph inspection. | final |
| 2026-08-03 | Direct indexed CSA remains the leading long-context architectural candidate, not yet the selected first patch. | Dense operands proven in source; wall-time dominance unmeasured. | provisional |
| 2026-08-03 | Do not treat historical 13.9 t/s as current production throughput; user observes about 80-300 PP t/s under varying live IQ2_M workloads. | User report; controlled conditions not yet recorded. | final |
| 2026-08-03 | Cap measured PP at five minutes while excluding initial load/warmup; run 8K separately and reject incomplete runs for matched A/B. | User direction plus independent benchmark-validity review. | final |
| 2026-08-03 | Trace whole process but apply attribution thresholds only after filtering to recorded measured-run timestamps. | Independent review found model load would bias whole-process totals. | final |
| 2026-08-03 | Select routed MMQ as M1 rather than CSA/LID/communication. | Measured-region trace assigns about 40% of summed kernel time to `mul_mat_q`; RCCL is 8%, LID 6.3%, explicit flash attention 5.6%, and measured H2D only 47 ms. | final |
| 2026-08-03 | Keep J=16 explicit for the known DSV4 IQ2_M service; do not make it a generic RDNA2 default. | Bracketed synthetic PP gains 10.0-11.7% and the attested 2,527-token natural proxy gains 5.26% in tensor mode, but hot-routing microbench regresses with J=16. | final |
| 2026-08-03 | Select the M=24,N=256,K=16384 F32 `hc_mixes` GEMM as optimization two. | Exact rocBLAS dimensions/call count map the 26.43-28.44% Cijk kernel to two `build_hc_pre` calls in each of 43 layers on four GPUs; explicit CSA is 7.06% and LID is 11.11% at 16K. | accepted |
| 2026-08-03 | Keep the skinny hc-mixer path explicit and exact-shape for the known service. | Custom 12x16x256 kernel is bit-identical and 3.83x faster locally; J16-held-constant PP gains 19.8-20.9%; the fully attested 256-ubatch corpus gate matches all responses; all near-shapes retain rocBLAS/generic fallback. | final |
| 2026-08-03 | Do not switch to communication-first after the local compute wins. | Fresh 16K trace assigns RCCL device work 9.84% and explicit copies 0.18%; x8 bus 46 is not the slowest by total/RCCL kernel sums. | final |
| 2026-08-03 | Tune the LID vector kernel before considering fused selection. | Lightning indexer is 14.87% while selection is only about 0.31%; exact counters show 74.1% occupancy, 16.5% memory busy, 95.5% L2 hits, no LDS conflicts, and 6,481 VALU instructions/work-item. | selected |
| 2026-08-03 | Reject simple LID K4/H32 tiling; do not combine or deploy it. | K4 regresses 9.2-10.5% and does not improve occupancy despite fewer VGPRs; H32 is neutral within drift and doubles LDS. Both fail the 10% local promotion gate. | final |
| 2026-08-03 | Promote same-tree LID subwave-4 as guarded optimization three for the known stack. | Focused exact/path/fallback/counter gates pass; J16+HC-held-constant whole model is +10.18% at 16K with 0.14% control drift and has no >2% short midpoint regression; fully hashed LID-off/on proxy outputs match in all six layer/tensor cases. | accepted guarded |
| 2026-08-03 | Promote RDNA2 IQ3_XXS J16 128-thread blocks as optimization four. | Exact focused outputs; IQ3 uniform/hot -16.08/-16.38%; whole-model +2.11/+2.37/+1.70/+1.69% at 512/2K/8K/16K; natural-proxy gate `complete=1` (all six equal); compact rocprof dispatch shows IQ3 wavefronts 11,264→5,632. I64 regresses, I256 unsupported, occupancy 1/3 neutral. | accepted |
| 2026-08-03 | Reject/defer production MTP for the exact-greedy DSV4 stack. | Production and n-max matrix diverge; rejection-only sequential replay fixes target-only continuation but not continued speculation; even zero-accept target-single advancement with full-state checkpoints later forks. No exact output or TG acceptance exists. | final deferred |
| 2026-08-03 | Reframe indexed CSA as a credible long-context candidate after an external fact-check. | Source facts confirmed (dense-masked operands, ratios 4/128, top-k<=512). Paper/Transformers describe indexed-sparse intent; StreamIndex supports streaming selection. Local TG dominance remains unmeasured. | provisional / on hold |
| 2026-08-04 | Record 16K->32K super-linear **whole-graph PP** scaling without assigning component dominance. | Single PP observations: 16K=372.1 t/s (44 s), 32K=117.4 t/s (279 s). No successful 32K attribution trace. 64K exited 137 during warmup before measurement start and supplies no timing. | accepted, qualified evidence |
| 2026-08-04 | Make target-only raw decode the blocking next phase and hold indexed CSA. | No accepted MTP/DSpark-disabled repeated TG sweep exists; PP scaling cannot select the raw-decode bottleneck. | selected |
| 2026-08-04 | Accept the M5.0 harness mechanics, not a raw-TG result. | Dry runs, source/CLI audit, parser fixtures, and fake end-to-end runs passed success/incomplete/unstable/setup-timeout/measured-timeout/residency cases without loading a model or launching GPU work. A reproduced FIFO deadlock was fixed before acceptance. | tooling accepted; TG pending |
| 2026-08-04 | Reject sequence-only depth-state restore for DSV4; require full context state. | Controlled 2K/3K/16K gate: fresh re-prefill is bit-identical, sequence restore has ~516.9K logit violations/depth and one 2K argmax divergence, while full context restore is bit-identical with zero token/logit/state mismatches. | accepted correctness fix `f97f5cdb0` |
| 2026-08-04 | Treat large HIP TOP_K work as conditional, not presumed missing. | Current branch enables HIP hipCUB, uses rocPRIM top-k for DSV4 large rows, and advertises TOP_K support; scheduler residency must still be attested in raw decode. | selected diagnostic |
| 2026-08-04 | Invalidate old percentage profiles for target selection after >3% whole-model gains. | Accepted J16/HC/LID/T128 changes materially altered Amdahl shares; LID alone changed 16K PP by +10.18%. | final rule |
| 2026-08-04 | Keep MTP/DSpark outside raw-decode baselines and kernel selection. | Exact-greedy MTP state diverges; speculative acceptance/checkpoint behavior is a separate workstream. | final |
| 2026-08-04 | Exclude in-band `rocm-smi` telemetry from accepted TG samples. | First full sweep showed 32K/64K MAD 13.7%/12.2% while polling every 1s inside 1.5-3s samples; with setup-only telemetry the same long points re-measure 0.33-3.97% and 64K stabilizes at 0.33%. Telemetry now samples setup + the discarded first repetition only. | accepted fix `81b072481` |
| 2026-08-04 | Raw-TG residency is attested at the composite backend, not per-GPU. | Scheduler exposes one `Meta(ROCm0..ROCm3)` backend; independent review: top-level split/copy counts cannot prove per-GPU execution or copies. | final caveat |
| 2026-08-04 | Residency parser counts real TOP_K/LIGHTNING_INDEXER ops only, with exact expected counts enforced. | Parser previously counted CONT/SET_ROWS consumers of `lid_top_k`, inflating 21 real nodes to 63; exact-op reparse of the preserved log requires and confirms exactly 21 TOP_K + 21 LID per measured graph at every depth 2048-65536, all on `Meta(ROCm0..ROCm3)`, zero CPU/unknown. Depth 0 has neither op. | accepted fix `4936a8673` |
| 2026-08-04 | Evidence runs require full GGUF shard hashes, all-resolved-DSO hashes, and recorded power/performance policy. | Independent review: metadata-only hashing and absent power profile failed strict identity attestation; added `DSV4_HASH_MODE=full` plus expanded rocm-smi policy snapshot. | accepted fix `4936a8673` |
| 2026-08-04 | Reject the fully-hashed 16-rep full sweep: 4K/8K over the stability gate. | MAD/median 4K=3.80%, 8K=3.15% with ordered warm-to-steady regime shift; 15 accepted at every depth; identities fully hashed (3 GGUF shard + ~40 ROCm/system DSO hashes, perf level auto, no power cap set). Remedy per policy: more tg32 repetitions. | rejected; full31 launched |
| 2026-08-04 | Pause all GPU work at user direction; full31 sweep aborted cleanly. | Job terminated (no KFD PIDs, lock released, no result). Raw TG remains pending; CSA undecided; next action on resume recorded below. | paused |

## 10. Closed decisions and open questions

Closed: IQ3_XXS J16 T128 passed focused exact-output, dispatch/counter, natural-proxy, and whole-model gates and is accepted as guarded optimization four. Its older/unclassified-AMD performance coverage remains an upstreaming-scope caveat, not an open local acceptance question.

Open questions:

1. Does J16 hold on a future user-supplied production corpus? The committed technical proxy is positive, but no user corpus exists.
2. Can a later expert-concentration signal select J16/J32/J64 without host synchronization? The accepted patch intentionally stays explicit.
3. What is the stable target-only raw TG curve and actual starting KV depth at minimal/2K/3K/4K/8K/16K/32K/64K?
4. Do all DSV4 LID TOP_K nodes remain ROCm-resident at every raw-decode depth, and how many CPU/GPU splits are present?
5. Which subsystem dominates the fresh target-only decode profile after all four accepted PP optimizations?
6. Does HIP flash attention perform partial arbitrary-mask tile pruning, and at what raw-decode context does CSA become material?
7. How are LID scores and top-k indices assigned across the four meta devices at runtime?
8. If MTP is reopened separately, which recurrent state/logit component changes after verification/checkpoint round trips with zero accepted drafts?
9. Which fixed corpus best represents production once the user supplies one?

## 11. Reproduction record

The controlled PP baseline, fixed natural-text proxy, and four accepted PP
optimizations are complete. This section records the externally
monitor-rerunnable PP verification command. A complete single-run 32K PP
result now exists, but repeated matched 32K A/B, successful 32K attribution,
and every 64K PP measurement remain missing. The 64K attempt terminated in
warmup before measurement start. No target-only repeated raw-TG sweep or
user-supplied production corpus has been accepted. The raw-decode harness now
preserves its measured-generation cap separately from context setup. Its
full-context restored-state equivalence gate is accepted; sequence-only restore
is rejected and fails closed.

**Current non-GPU M5.0 harness monitor command** (tooling verification only):

```bash
cd /home/edwin/llama.cpp-rdna2
ARTIFACT=$HOME/llama-jobs/dsv4-rocm-tg/static-validation-20260804T0415Z-0376a55aacd6
OUT=/tmp/dsv4-tg-static-rerun
ARTIFACT="$ARTIFACT" OUT="$OUT" "$ARTIFACT/commands.sh"
```

It uses fake llama-bench/model/`rocm-smi` fixtures and launches no GPU work.

**Accepted restored-state monitor command:**

```bash
cd /home/edwin/llama.cpp-rdna2
cmake --build build --target test-state-restore-equivalence -j 12
DSV4_STATE_API=context DSV4_LABEL=context-state-controlled \
  scripts/dsv4-rocm/run-state-restore-equivalence.sh
```

The accepted result is the `5d80b8662` artifact listed above; all original/fresh-repeat/restored logits and argmax tokens are bit-identical. The next real commands are `DSV4_LABEL=raw-tg-baseline scripts/dsv4-rocm/run-tg.sh` and `DSV4_TG_MODE=residency DSV4_LABEL=raw-tg-residency scripts/dsv4-rocm/run-tg.sh`. Recheck GPU ownership immediately before each.

**Final externally rerunnable PP verification command** (recorded on ancestor
`77ef7c2d1`; implementation source unchanged since `803a41c37`, with
acceptance recorded at `c98197389`; later commits are documentation only):

```bash
cd /home/edwin/llama.cpp-rdna2
cmake --build build --target llama-bench -j 12
export HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_NO_SCRATCH_RECLAIM=1 \
  GGML_HIP_GRAPHS=1 GGML_CUDA_ALLREDUCE=nccl GGML_CUDA_P2P=1 \
  GGML_HIP_RDNA2_MMQ_J=16 GGML_HIP_RDNA2_HC_MIXES=1 \
  GGML_HIP_RDNA2_LID_SUBWAVE=4
DSV4_PROMPTS=512,2048,8192,16384 DSV4_UBATCHES=256 DSV4_BATCH=512 \
DSV4_REPS=1 DSV4_TIMEOUT=300 DSV4_LABEL=final-full-stack-4opt \
scripts/dsv4-rocm/run-pp.sh
```

The recorded run (`$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260804T003038.699935606Z-final-full-stack-4opt-c98197389511-25811/`) was complete with all four shapes and median 293.744 / 523.352 / 437.512 / 365.332 t/s at 512/2K/8K/16K. Correctness of the committed stack is attested by the natural-proxy gate `20260803-225603-iq3-t128-corpus-fb2a0c85d` (`complete=1`).

Current 8K control/candidate commands:

```bash
cd /home/edwin/llama.cpp-rdna2
unset GGML_HIP_RDNA2_MMQ_J
DSV4_LABEL=mmq-jauto-8k DSV4_PROMPTS=8192 DSV4_UBATCHES=256 \
DSV4_REPS=2 DSV4_TIMEOUT=300 scripts/dsv4-rocm/run-pp.sh

GGML_HIP_RDNA2_MMQ_J=16 \
DSV4_LABEL=mmq-j16-8k DSV4_PROMPTS=8192 DSV4_UBATCHES=256 \
DSV4_REPS=2 DSV4_TIMEOUT=300 scripts/dsv4-rocm/run-pp.sh
```

Current combined-stack correctness and profile commands, rerunnable from a
fresh shell at clean implementation/runner commit `9f4808637` (later commits
only update this evidence record):

```bash
cd /home/edwin/llama.cpp-rdna2
cmake --build build --target llama-server llama-bench -j 12

DSV4_BASE_MMQ_J=16 DSV4_CANDIDATE_MMQ_J=16 \
DSV4_BASE_HC_MIXES=1 DSV4_CANDIDATE_HC_MIXES=1 \
DSV4_BASE_LID_SUBWAVE=0 DSV4_CANDIDATE_LID_SUBWAVE=4 \
DSV4_BATCH_SIZE=512 DSV4_UBATCH_SIZE=256 DSV4_HASH_MODE=full \
scripts/dsv4-rocm/run-corpus-validation.sh

trace_log=$(mktemp)
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_NO_SCRATCH_RECLAIM=1 \
GGML_HIP_GRAPHS=1 GGML_CUDA_ALLREDUCE=nccl GGML_CUDA_P2P=1 \
GGML_HIP_RDNA2_MMQ_J=16 GGML_HIP_RDNA2_HC_MIXES=1 \
GGML_HIP_RDNA2_LID_SUBWAVE=4 \
DSV4_PROFILE=kernel DSV4_LABEL=kernel-trace-j16-hc-lid-16k \
DSV4_PROMPTS=16384 DSV4_UBATCHES=256 DSV4_BATCH=512 \
DSV4_REPS=1 DSV4_TIMEOUT=300 \
scripts/dsv4-rocm/profile-pp.sh | tee "$trace_log"
run_dir=$(awk -F= '$1 == "run_dir" { print $2 }' "$trace_log" | tail -1)
test -n "$run_dir"
scripts/dsv4-rocm/summarize-trace.py "$run_dir" --top 40 \
  --json "$run_dir/measured-region-summary.json" \
  | tee "$run_dir/measured-region-summary.txt"
scripts/dsv4-rocm/analyze-trace-agents.py "$run_dir" \
  --json "$run_dir/measured-region-agents.json" \
  | tee "$run_dir/measured-region-agents.txt"
```

Accepted correctness artifacts are
`$HOME/edwin/llama-jobs/dsv4-corpus-validation/20260803T190100.516971835Z-attested-56dd4177e501-15597/`
for HC and
`$HOME/edwin/llama-jobs/dsv4-corpus-validation/20260803T212823.803707936Z-attested-9f4808637e55-20974/`
for LID. The accepted post-LID attribution trace is
`$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260803T215054.700650714Z-kernel-trace-j16-hc-lid-16k-fdde31252a63-8573/`;
the directly comparable pre-LID trace remains
`$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260803T191856.045376424Z-kernel-trace-j16-hc-16k-52e0121043ad-23195/`.
Optimization four is accepted (commits `803a41c37`/`fb2a0c85d`); production
MTP remains explicitly deferred. These commands are a reproducible verification
checkpoint, not a claim that every deferred roadmap item is complete: 32K/64K
full-context A/B and the nonselected CSA/communication/MTP items are recorded
as deferred elsewhere in this document.

Current production diagnostic, rerunnable from the clean worktree (it is
expected to exit nonzero while the recorded MTP divergence persists):

```bash
cd /home/edwin/llama.cpp-rdna2
cmake --build build --target llama-server -j 12
timeout --signal=TERM --kill-after=30s 2400s env \
  GGML_HIP_RDNA2_MMQ_J=16 GGML_HIP_RDNA2_HC_MIXES=1 \
  DSV4_OUTER_TIMEOUT=2400 \
  scripts/dsv4-rocm/run-production-mtp-validation.sh
```

Current exact-shape LID baseline:

```bash
cd /home/edwin/llama.cpp-rdna2
cmake --build build --target test-backend-ops -j 12
export HSA_OVERRIDE_GFX_VERSION=10.3.0 LD_LIBRARY_PATH=$PWD/build/bin
build/bin/test-backend-ops test -b ROCm0 -o LIGHTNING_INDEXER \
  -p 'hsk=128,nh=64,kv=256,nb=256,ns=1,nm=1,type_K=f16'
for kv in 256 4096; do
  build/bin/test-backend-ops perf -b ROCm0 -o LIGHTNING_INDEXER \
    -p "hsk=128,nh=64,kv=$kv,nb=256,ns=1,nm=1,type_K=f16"
done
```

The counter commands and raw DBs are preserved under
`$HOME/edwin/llama-jobs/dsv4-lid-study/20260803T195000Z-bd4d1b9aa-baseline/`.
The production failure is under
`$HOME/edwin/llama-jobs/20260803-194659-dsv4-prod-mtp-j16-hc-r3/`.

### Artifact and loop index

```text
Canonical master:
  /home/edwin/llama.cpp-rdna2/docs/deepseek-v4-flash-rocm-performance.md

Completed PP Ralph log:
  /Users/edwin/.ralph/dsv4-flash-rocm.md
Completed PP Ralph state:
  /Users/edwin/.ralph/dsv4-flash-rocm.state.json (status=completed)
Repository implementation/evidence chain:
  803a41c37 (optimization-four implementation) ->
  c98197389 (optimization-four acceptance record) ->
  77ef7c2d1 (final PP verification) ->
  3cf35253f (initial 16K/32K PP scaling record; interpretation corrected here) ->
  925d93700 (indexed-CSA design note, now held by this roadmap) ->
  5df30a53e (raw-decode-first roadmap reset) ->
  0376a55aa (Ralph loop registration) ->
  1e5519bf1 (M5.0 target-only TG harness + static validation) ->
  3d23fff4a (restore-equivalence gate) ->
  400c47cd6 (fresh-repeat control) ->
  5d80b8662 (full-context diagnostic) ->
  f97f5cdb0 (llama-bench/run-tg full-context integration) ->
  3a2ab230f (iteration-2 master sync) ->
  0b0c8e4cf (launch telemetry race fix) ->
  81b072481 (telemetry excluded from accepted TG) ->
  4936a8673 (evidence provenance hardening: exact-node parser counts, full hashes, power policy)

Raw-decode Ralph log:
  /Users/edwin/.ralph/dsv4-raw-decode-roadmap.md
Raw-decode Ralph state:
  /Users/edwin/.ralph/dsv4-raw-decode-roadmap.state.json
Raw-decode Ralph status:
  active, iteration 3/50, PAUSED by user 2026-08-04; started 2026-08-04T03:47:49Z
Revised roadmap / loop-registration commits:
  5df30a53e / 0376a55aa
M5.0 harness / corrected depth-state commits:
  1e5519bf1 / f97f5cdb0
M5.0 static-validation artifacts:
  $HOME/edwin/llama-jobs/dsv4-rocm-tg/static-validation-20260804T0415Z-0376a55aacd6/
Current next action (PAUSED; no GPU work until directed):
  On resume: run one accepted full-depth tg32 sweep (31 raw / 30 accepted
  repetitions was in flight when paused and was aborted cleanly), re-run the
  residency audit at the hardened parser commit `4936a8673`, then M5.1/M5.3
  classification. No baseline accepted yet; CSA undecided.

Purpose:
  Ralph files contain per-iteration checkpoints, rejected variants, commands,
  and blockers. This repository document is canonical for accepted evidence,
  current decisions, and the next action; every Ralph iteration must update
  and commit it before advancing.
```

### Iteration 3 checkpoint (PAUSED by user 2026-08-04; no GPU work)

- Fixed a launch telemetry race (`0b0c8e4cf`): sampler/watchdog liveness now checks leader PID in addition to process group; background consumers close the lock-bearing fd (`9>&-`) so `flock` on `$HOME/llama-jobs/gpu.lock` releases synchronously. Static matrix re-passed all six exit cases; rebuilt `llama-bench` at clean HEAD.
- First real full sweep (continuous telemetry) REJECTED: 0-16K stable (MAD <=1.1%, ~22.5-24.2 t/s) but 32K/64K MAD 13.7%/12.2% from 1s in-band `rocm-smi` polling inside 1.5-3s samples. Fix `81b072481` restricts telemetry to setup + discarded first repetition (`TELEMETRY_SCOPE=setup-and-discarded-first-repetition`); static validation re-passed.
- Accepted composite performance observations (same build `81b072481`, setup-only telemetry, tg32, context-state API):
  - 0-16K: `20260804T054358.914659309Z-raw-tg-baseline-short-performance-81b072481f7a-18833` (5 accepted each): medians 24.030 / 24.168 / 23.809 / 23.855 / 23.552 / 22.522 t/s, MAD/median <=1.06%.
  - 32K: `20260804T053717.265093462Z-raw-tg-32k-stability-16-performance-81b072481f7a-13173` (15 accepted): median 20.311 t/s, 49.235 ms/token, MAD/median 1.23%.
  - 64K: retained from `20260804T051715.402695167Z-raw-tg-stability-11-performance-81b072481f7a-14164` (10 accepted): median 18.398 t/s, 54.355 ms/token, MAD/median 0.33%; that joint job is globally rejected (32K row 3.97%) and must not be cited as accepted.
- Residency audit at `81b072481` (`20260804T054747.800048172Z-raw-tg-residency-residency-81b072481f7a-2297`, rc=0): every depth has 1 decode graph; split #0 is a CPU empty-input split (token embedding GET_ROWS), split #1 is the Meta graph with 22-25 inputs. Re-run at `4936a8673` with the hardened exact-op parser: exactly 21 TOP_K + 21 LIGHTNING_INDEXER per measured graph at 2048-65536 on `Meta(ROCm0..ROCm3)`, zero CPU/unknown; depth 0 has neither op. Per-GPU execution/copy counts are NOT provable from the top-level Meta log. `scheduler-summary.pre-4936a8673.{json,tsv}` backups, the parser command (`scheduler-parser-command.sh`) and commit (`scheduler-parser-commit.txt`) are preserved in the run dir.
- Independent reviewer findings repaired at `4936a8673` (see decision log): parser consumer-count inflation (63 vs 21), metadata-only hashing, missing DSO hashes/power policy. `manifest.sh` now hashes every resolved DSO and records `--showperflevel --showprofile --showmaxpower --showoverdrive --showmemoverdrive`; `run-tg.sh` gains `DSV4_HASH_MODE` (metadata default, full for evidence runs), `DSV4_EXPECTED_DSV4_NODES=21` wiring, and final policy snapshot.
- Fully-hashed 16-rep sweep REJECTED: `20260804T062521.895434970Z-raw-tg-baseline-full-performance-4936a8673daf-11900` (15 accepted/depth; 4K MAD 3.80%, 8K 3.15%; ordered warm-to-steady regime; 3 GGUF shard hashes + ~40 ROCm/system DSO hashes; perf level auto, no power cap set). Full31 sweep (31 raw/30 accepted, `DSV4_HASH_MODE=full DSV4_TG_REPS=31 DSV4_LABEL=raw-tg-baseline-full31`) was launched and ABORTED by the user pause; no result.
- No accepted raw-TG baseline exists; CSA remains undecided. GPUs idle, lock free, repo clean at `4936a8673` at pause time.

Planned final record:

```text
source commit:
build command:
model files and hashes:
environment:
baseline command:
candidate command:
correctness command:
profile command:
raw artifact directory:
summary:
```