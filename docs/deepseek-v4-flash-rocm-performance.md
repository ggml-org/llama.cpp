# DeepSeek-V4-Flash ROCm prompt-processing and raw-decode performance plan

Status: living canonical engineering record
Owner branch: `perf/dsv4-rocm-pp-20260803`  
Base: `b88a59fbc6ac255e6bf5e2dd790f559c89ce911c` in Edwin's llama.cpp fork  
Target host: `edwin@192.168.1.161` (`webhie`)  
Last updated: 2026-08-05
Current phase: PP plus M5.0/M5.1 raw-decode baseline/residency are accepted; M5.3 selected communication and routine profiling remains frozen. M5.4 Tree+LL and Ring+LL failed screen advancement, and the final predeclared guarded BF16 hidden-reduction candidate failed its short numerical correctness gate before performance. No decode optimization is accepted; no longer run is authorized; indexed CSA remains held.

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

### Raw-decode evidence status: M5.0/M5.1 ACCEPTED; M5.3 COMPLETE; M5.4 ACTIVE

A stable, speculation-disabled target-only context-depth sweep is accepted on the four-optimization stack. M5.0 uses tg32, 31 raw/30 accepted repetitions at every depth 0/2K/3K/4K/8K/16K/32K/64K, full-context state, MTP/DSpark/draft absent, and exact four-way tensor split. Medians are 24.063 / 23.669 / 22.841 / 21.924 / 22.637 / 22.133 / 19.729 / 18.402 t/s; every MAD/median is <=1.77%. M5.1 separately attests exact composite-Meta LID/TOP_K scheduler residency. Full binary/model/46-DSO/power identities and telemetry boundaries are preserved under the artifact paths in sections 7 and 11.

Historical 20.1 t/s DSpark and 20.208 t/s failed-MTP observations remain non-baselines. They are superseded for controlled raw TG by M5.0, not retroactively accepted.

Harness status (2026-08-04): `scripts/dsv4-rocm/run-tg.sh` has accepted separate performance/residency mechanics, manifests, phase-aware watchdogs, setup-only telemetry, exact-depth/repetition summaries, strict scheduler parsing, full-context state, and fail-closed raw/JSON stdout isolation. Static fixtures, real FIFO fake-runner tests, and model-dependent restore equivalence pass. M5.3 is complete: two profiles at each of 16K/64K select communication in all 30 retained ranks. Routine profiling is frozen; M5.4 runtime RCCL controls are exhausted without a screen winner and source work is next.

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
- `$HOME/llama-jobs/dsv4-rocm-pp/20260804T025141.681162476Z-csa-scaling-16k-16384-d032b943d185-13070/`
- `$HOME/llama-jobs/dsv4-rocm-pp/20260804T025414.444077827Z-csa-scaling-32k-32768-d032b943d185-7005/`
- `$HOME/llama-jobs/dsv4-rocm-pp/20260804T030440.549424222Z-csa-scaling-64k-65536-d032b943d185-6545/` (incomplete warmup; no result)

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

- `$HOME/llama-jobs/dsv4-rocm-rocblas/20260803T175331Z-ec1b7e64c-map-cijk/` (three-line aggregate rocBLAS profile)
- `$HOME/llama-jobs/dsv4-rocm-pp/20260803T175022.376406151Z-trace-map-cijk-256-ec1b7e64c2cc-24970/` (single-microbatch trace)
- `$HOME/llama-jobs/dsv4-hc-mixes-sweep/20260803T181014Z-1d6a42983-prototype/` (tile sweep, correctness, fallback, graph, and dispatch proof)
- `$HOME/llama-jobs/dsv4-hc-mixes-sweep/20260803T182030Z-560635e3b-whole-model/` (J16-held-constant whole-model A/B)
- `$HOME/llama-jobs/dsv4-rocm-pp/20260803T191856.045376424Z-kernel-trace-j16-hc-16k-52e0121043ad-23195/` (combined-stack 16K compact trace, aggregate and per-agent measured-region summaries)
- `$HOME/llama-jobs/dsv4-lid-study/20260803T195000Z-bd4d1b9aa-baseline/` (launch scaling, exact fixture correctness/performance, hardware counters, raw DBs, counter command, and screen contract)
- `$HOME/llama-jobs/dsv4-lid-study/20260803T201000Z-k4-217f2a271-prototype/` (discarded K4 source, correctness/fallback/invalid-env screen, A/B/A, trace resources, occupancy counters)
- `$HOME/llama-jobs/dsv4-lid-study/20260803T202000Z-h32-217f2a271-prototype/` (discarded H32 source, retained KV=256 reference test, A/B/A, trace resources)
- `$HOME/llama-jobs/dsv4-lid-study/20260803T204500Z-subwave4-087813f76-prototype/` (excluded: host-side `RDNA2` preprocessor condition removed the candidate; invalid env did not fail)
- `$HOME/llama-jobs/dsv4-lid-study/20260803T205000Z-subwave4-087813f76-prototype/` (corrected temporary source, CPU-reference/fallback fast screen, distinct trace dispatch/resources, restoration proof)
- `$HOME/llama-jobs/dsv4-lid-study/20260803T211000Z-subwave4-validation-3276edc81/` (excluded first deterministic attempt: detected 1-ULP reassociation drift at KV=1)
- `$HOME/llama-jobs/dsv4-lid-study/20260803T212000Z-subwave4-validation-3276edc81/` (authoritative bitwise/path/counter/repeated-process/fallback artifact; final source patch and binary hashes)
- `$HOME/llama-jobs/dsv4-lid-study/20260803T220000Z-subwave4-whole-model-9f4808637/` (J16+HC-held-constant 512/2K/8K/16K A/B/A; stable 16K +10.18%)
- `$HOME/llama-jobs/dsv4-corpus-validation/20260803T212823.803707936Z-attested-9f4808637e55-20974/` (fully hashed LID-off/on corpus acceptance; all six responses equal)
- `$HOME/llama-jobs/dsv4-rocm-pp/20260803T215054.700650714Z-kernel-trace-j16-hc-lid-16k-fdde31252a63-8573/` (fresh post-LID 16K compact trace; aggregate/per-agent summaries)
- `$HOME/llama-jobs/20260803-221516-mmq-config-screen-6af98d65b/` (excluded strict-zero-tolerance first screen; correctly stops on I64 low-order drift)
- `$HOME/llama-jobs/20260803-221906-mmq-config-screen-tolerant-6af98d65b/` (I64 16.0-21.6% regressions and I256 fail-closed unsupported evidence; source restored)
- `$HOME/llama-jobs/20260803-222442-mmq-config-screen-core-6af98d65b/` (authoritative T128/occupancy screen; exact focused outputs, three process timings per type/route)
- `$HOME/llama-jobs/20260803-223310-iq3-t128-whole-model-6af98d65b/` (J16+HC+LID-held-constant 512/2K/8K/16K A/B/A; initial 512 single-observation excluded)
- `$HOME/llama-jobs/20260803-224633-iq3-t128-short-repeat-6af98d65b/` (dedicated three-repetition 512 A/B/A; stable-median +2.11%)

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
- `$HOME/llama-jobs/dsv4-corpus-validation/20260803T190100.516971835Z-attested-56dd4177e501-15597/` (J16-only versus J16+HC acceptance artifact; full hashes, batch/ubatch 512/256, `complete=1`)
- `$HOME/llama-jobs/dsv4-corpus-validation/20260803T212823.803707936Z-attested-9f4808637e55-20974/` (J16+HC+LID-off versus LID-on acceptance; six exact response matches; +0.21% layer/+1.70% tensor single observations)
- `$HOME/llama-jobs/dsv4-corpus-validation/20260803T184944.079084014Z-attested-56dd4177e501-25863/` (excluded interrupted attempt: all responses exist and match, but no final status/comparison artifact and `candidate.rc=120`)
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

- `$HOME/llama-jobs/20260803-194659-dsv4-prod-mtp-j16-hc-r3/`

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

- `$HOME/llama-jobs/dsv4-mtp-matrix-20260803T200941Z-3fc2c17f6/`
- source runs `$HOME/llama-jobs/20260803-200941-mtp-matrix-{g1n3,g1n1,g0n3,g0n1}/`

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

- `$HOME/llama-jobs/20260803-204032-mtp-prefix-replay8-fresh-087813f76/` (`diagnostic-evidence.json` is authoritative)

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

- `$HOME/llama-jobs/20260803-210548-mtp-stateful-7a75d8a5a/` (`stateful-continuation/diagnostic-evidence.json`)

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

- `$HOME/llama-jobs/20260803-214024-mtp-prefix1-cont0-9f4808637/` (target-state isolation; continuation cap zero)
- `$HOME/llama-jobs/20260803-212523-mtp-force-checkpoint-9f4808637/` (forced target+draft checkpoint path; partial state)
- `$HOME/llama-jobs/20260803-214302-mtp-compressed-cache-fix-9f4808637/` (bounded compressed-cache removal experiment)
- `$HOME/llama-jobs/20260803-214553-mtp-full-state-checkpoint-9f4808637/` (forced full-state checkpoint experiment)

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

- `$HOME/llama-jobs/20260803-215521-mtp-sequential-replay-fdde31252/` (focused rejection-replay success)
- `$HOME/llama-jobs/20260803-215800-mtp-seq-replay-matrix-fdde31252/` (four arms; only cap-1-prefix→cap-1 still fails)
- `$HOME/llama-jobs/20260803-220554-mtp-target-single-verify-fdde31252/` (zero-accept target-single diagnostic; delayed fork)
- `$HOME/llama-jobs/20260803-220904-mtp-fullstate-target-single-fdde31252/` (full-state variant; same delayed fork)

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
| Target-only raw TG sweep | **M5.0 accepted; M5.1 residency accepted** | Controlled TG curve and residency are closed; M5.3/M5.6 profiles select the next decode optimization/crossover. |

### Revised milestone order

| Order | Milestone | Required evidence |
|---|---|---|
| M5.0 | Raw TG baseline | Stable target-only median TG with MTP/DSpark absent. |
| M5.1 | ROCm backend-residency audit | DSV4 LID and TOP_K remain GPU-resident; CPU/GPU split counts recorded. |
| M5.2 | Large HIP TOP_K, **only if reproduced** | Exact selected indices and GPU residency through 64K; no CPU split. |
| M5.3 | Fresh whole-model raw-decode profile | Ranked elapsed device-time breakdown on the full accepted branch. |
| M5.4 | Selected communication optimization | Reversible RCCL screen, then source change only if needed; accept only end-to-end TG gain. |
| M5.5 | LID + TOP_K optimization | Deferred because communication won the locked branch rule. |
| M5.6 | 32K/64K attention-scaling study | CSA/indexer/TOP_K/mask/block/TG timings and measured crossover. |
| M5.7 | Indexed CSA | Only if M5.6 supports it. |
| M5.8 | Decode-chain fusion | Only if measured launch/elementwise cost dominates. |
| M5.9 | Multi-stream overlap | After single-stream paths are optimized. |
| Separate | DSpark/MTP | Acceptance/state-management workstream; never baseline evidence. |

### M5.0 / P0 - establish a valid raw-decode baseline

Run the accepted target stack with **no draft model or speculative flags** (MTP off, DSpark off), identical model/quantization, exact `--split-mode tensor --tensor-split 1,1,1,1`, F16 K/V, flash attention on, identical batch/ubatch settings, and identical clocks/power policy. Record the exact seed/sampling contract if sampling is used (temperature, top-k/top-p/min-p, penalties, grammar, and EOS policy). Prefer benchmark-native fixed-token evaluation so exactly 32 target evaluations occur without sampler/EOS ambiguity; otherwise any early stop or differing generated count is incomplete. Record fixed input/generated token IDs where the harness supplies them and the actual KV depth before the first measured token.

Depth sweep: minimal/empty prompt (label it by the actual starting KV depth, not blindly `0`), 2048, 3072, 4096, 8192, 16384, 32768, and 65536. Use **tg32 for the entire sweep** to keep generation count fixed; use tg64 for the entire sweep only if it fits. At least five measured repetitions per depth. Context setup/warmup is outside the measured TG interval, but its time and failures remain recorded. Every repetition starts from a fresh equivalent KV state; reused/restored prompt state is allowed only after target-only logit/token equivalence to fresh prefill is attested.

Set `GGML_SCHED_DEBUG=2` for the residency audit and parse the measured decode graph, not model-load/prefill noise. Record for every depth:

- at least five accepted samples after any predeclared discard, median tokens/s, median ms/token, range and MAD/median;
- actual initial KV depth and fixed generated-token count;
- every DSV4 `TOP_K` backend assignment and fused lightning-indexer backend;
- CPU and per-GPU graph-split counts, plus introduced copies;
- per-GPU utilization samples, clocks, power cap/state, VRAM, temperature;
- exact source/build/DSO/model identities, environment, and command.

P0 has two acceptance states:

1. **Observational pre-fix baseline accepted:** untouched target-only measurement integrity, identities, fixed token count, timing boundaries, and five repetitions are valid. A reproduced CPU LID/TOP_K assignment or repeatable TG cliff is recorded as the finding that routes to M5.2/M5.3; it does not erase the observation.
2. **Post-fix/deployment baseline accepted:** no context-dependent migration of DSV4 TOP_K/lightning indexer to CPU and no unexplained context-step TG cliff.

Both states require no speculative decoding/draft model; a stable five-run median (initial target MAD/median <=3%, otherwise increase tg/repetitions and retain instability as evidence); and an exact externally rerunnable command/artifact directory. A GPU split count need not be zero under four-way tensor execution; it must be stable/explained. A CPU LID/TOP_K split blocks post-fix/deployment acceptance until corrected or causally justified.

#### M5.0 harness and accepted target-only baseline

Performance mode uses llama-bench `n_prompt=0`, `n_depth=<sweep>`, `n_gen=32`, and `--no-warmup` under one model load. llama-bench computes/restores the requested depth before its timer, then `test_gen` performs exactly 32 one-token target `llama_decode` calls with no sampler or EOS stop. Six raw repetitions are preserved; the first target-depth graph-cold sample is predeclared and excluded, leaving five accepted samples. `summarize-tg.py` validates the exact depth/config/repetition contract and recomputes t/s, ms/token, ranges, and latency MAD/median from raw nanoseconds.

Residency mode is a separate non-performance run (`n_gen=1`, one repetition, `GGML_SCHED_DEBUG=2 --verbose`). `parse-sched-debug.py` changes phase only on llama-bench progress markers, ignores setup/prefill assignments, and records measured-decode LID/TOP_K backends, CPU/ROCm-meta split counts, and scheduled split-input copies. Separating it prevents verbose scheduler output from perturbing accepted TG. The tensor split is spelled `1/1/1/1` because llama-bench reserves commas for parameter variants; this is its exact four-device equivalent of `1,1,1,1`.

The non-GPU monitor rerun passed for the code committed at `1e5519bf1`; its precommit source patch and file hashes are preserved at
`$HOME/llama-jobs/dsv4-rocm-tg/static-validation-20260804T0415Z-0376a55aacd6/`.
It preserved a reproduced/rejected FIFO notification deadlock: a fast child could exit before a later FIFO event had a reader. The accepted implementation uses an append-only phase log plus an atomically replaced latest-phase file. Expected/observed exits were 0/0 for stable performance and residency, 3/3 for missing depth and measured timeout, 4/4 for instability, and 124/124 for setup timeout. These are tooling results only.

The model-dependent restore gate (`3d23fff4a`, controlled diagnostic `400c47cd6`, full-context extension `5d80b8662`) used deterministic fresh prefixes at 2K/3K/16K, four greedy target steps, every full-vocabulary logit, exact token IDs, and a fresh re-prefill control. The fresh repeat was bit-identical at all depths (state byte mismatches 0, logit tolerance violations 0, max absolute difference 0). Sequence-only `llama_state_seq_get/set_data` then failed at every depth: 516,983 / 516,900 / 516,930 logit tolerance violations, maximum absolute differences 4.118656 / 0.818019 / 0.736117, and a 2K final argmax divergence. Therefore the sequence API is rejected for DSV4 depth reuse.

Full `llama_state_get/set_data` context state passed bit-identically at all three depths: state sizes 60,674,301 / 67,567,869 / 157,184,253 bytes; zero state-byte mismatch against fresh re-prefill; zero repeat/restore bitwise or tolerance/non-finite mismatches; identical four-token argmax paths. Accepted artifact:
`$HOME/llama-jobs/dsv4-rocm-state-equivalence/20260804T044114.706244382Z-context-state-controlled-5d80b8662a95-16000/`.
Rejected controlled sequence artifact:
`$HOME/llama-jobs/dsv4-rocm-state-equivalence/20260804T043609.677826936Z-state-restore-controlled-400c47cd68ce-27056/`.
The fixed sweep has one generation/batch configuration and unique depths, so reuse occurs only for later repetitions inside the same context. Commit `f97f5cdb0` adds explicit llama-bench full-context depth state and makes `run-tg.sh` reject any sequence-mode override.

**M5.0 accepted target-only baseline.** At clean source commit `1cd80107ee7659ede72b9487e3bd00f24527e93b`, the fully hashed resumed sweep ran tg32 at all eight required actual starting KV depths with 31 raw samples, the first predeclared graph-cold sample discarded, and 30 accepted samples per depth. It used no draft/MTP/DSpark/speculative path; full-context state restore; J16/HC1/LID-subwave4; F16 K/V; FA on; batch/ubatch 512/256; and exact tensor split `1/1/1/1`. The run completed without timeout or truncation. Its 7,680 accepted target decode calls are:

| Actual starting KV depth | Accepted | Median t/s (range) | Median ms/token (range) | MAD/median |
|---:|---:|---:|---:|---:|
| 0 | 30 | 24.0625 (23.9278-24.1856) | 41.5584 (41.3469-41.7925) | 0.1406% |
| 2048 | 30 | 23.6687 (21.1673-24.2190) | 42.2499 (41.2898-47.2428) | 0.8770% |
| 3072 | 30 | 22.8409 (20.3960-24.0128) | 43.7819 (41.6444-49.0293) | 1.7694% |
| 4096 | 30 | 21.9241 (19.7031-23.8758) | 45.6120 (41.8834-50.7535) | 1.5912% |
| 8192 | 30 | 22.6371 (20.6265-23.5240) | 44.1752 (42.5098-48.4812) | 1.4514% |
| 16384 | 30 | 22.1328 (17.5235-22.6453) | 45.1819 (44.1593-57.0663) | 0.9778% |
| 32768 | 30 | 19.7291 (13.9684-20.4276) | 50.6865 (48.9534-71.5901) | 0.2659% |
| 65536 | 30 | 18.4023 (9.0068-18.5772) | 54.3411 (53.8295-111.0270) | 0.1380% |

Every depth is below the predeclared 3% latency-MAD gate. Wide 32K/64K ranges retain isolated slow samples rather than post-hoc removal; 30-sample medians/MADs remain stable. Of 1,054 telemetry query starts, zero falls in an accepted measurement interval. The exact executable SHA-256 is `386adefc9aa74fd762c7aaafb64eed647176db94a29b5890f1f5fdc26010f7df`; all 46 resolved DSO hashes match the residency arm. Full GGUF hashes are shard 1 `057a3aacf912e079f22d07b94bc3b4ef46c6632476bc0bd1761347eb08edb2aa`, shard 2 `700405274473b58fa26be4f14e4a194c2e7554fa3a052f62a0c50c568e89fc1f`, and shard 3 `a69102ddfaf4a84426e11fdb66716654f4260dc3a1de3ade9fd50e006b8691d3`. All four GPUs recorded performance level `auto`, boot-default power profile, 140 W maximum package power, and no supported overdrive.

Performance artifact:
`$HOME/llama-jobs/dsv4-rocm-tg/20260804T124716.565555325Z-raw-tg-baseline-full31-resumed-performance-1cd80107ee76-27396/`.
Its `pair-acceptance-validation.{py,json,txt}` is the exact rerunnable M5.0/M5.1 pair monitor and reports every check PASS.

### M5.1-M5.2 / P0-A - TOP_K residency, then fix only if reproduced

**M5.1 accepted at composite-backend scope.** The separate fully hashed residency arm at source `1cd80107ee76` used `n_gen=1`, one repetition/depth, `GGML_SCHED_DEBUG=2 --verbose`, and no performance timing. At each depth it found exactly one ordered measured decode graph: split `#0 = CPU/0 inputs` and split `#1 = Meta(ROCm0,ROCm1,ROCm2,ROCm3)` with 22 inputs at depth 0 and 25 at 2K-64K. Depth 0 has no TOP_K/LIGHTNING_INDEXER operation. Every 2K-64K graph has exactly 21 real TOP_K plus 21 real LIGHTNING_INDEXER operations on the aggregate Meta backend, with zero CPU/unknown DSV4 assignment and no parser warning. The CPU split is token-embedding `GET_ROWS`; zero split inputs means no scheduler-introduced input enters it, not that no CPU computation/CPU-origin transfer exists. This top-level log does **not** attest per-GPU execution ownership, per-GPU copies, peer transfer, or selector correctness.

Residency artifact:
`$HOME/llama-jobs/dsv4-rocm-tg/20260804T131957.468937324Z-raw-tg-residency-hardened-full-residency-1cd80107ee76-22574/`.
Offline parser commit `6f7115360e3c` fails closed on marker order/counters, exact two-split structure/backend/input counts, 2K selector presence, abbreviated-node-to-full-Meta correlation, extra/unknown splits, and any warning. Negative fixtures and a postcommit reparse of the preserved 529 MiB log pass. Therefore M5.0 observational and post-fix/deployment states are accepted; M5.2 implementation is **not triggered** because no LID/TOP_K fallback reproduced. M5.2 selector correctness remains a separate conditional gate.

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

**M5.3 checkpoint — communication selected at 16K and 64K for investigation only.** Commit `35ee01b7e` adds the disk-safe ROCm-only selected-region hook/harness. For accepted repetitions only, llama-bench calls `roctxProfilerResume(0)` immediately before target-only synchronized `test_gen`, records authoritative monotonic boundaries, and pauses after generation. Kernel, memory-copy, RCCL, and HIP-runtime CSV domains exclude load, depth setup, context restore, and the first discarded sample. Commits `667bc100a` and `09f575d10` preserve only scope-valid attribution when profiler wall timing fails, while forcing throughput and CSA eligibility false.

The first coarse classifier mixed shared experts, projections, and context work. Commit `27f432de9` fails closed on the fully hashed 43-layer V4-Flash IQ2_M routed/shared normal and anomalous type/fusion/workgroup/grid signatures, globally, per GPU, and per repetition. Commit `a789f138c` then maps the exact F16 `concat_cont` signatures to source-proven roles: axis 2 is CSA/HCA K materialization (`csa_k_all`/`hca_k_all`), axis 0 is their final mask concatenation; state restore uses another axis/type. CSA/HCA require exactly 21/20 layers x target tokens x four GPUs, globally/per GPU/per repetition at 16K and 64K. Unknown names, near-grids, missing calls, or redistribution fail closed. Two review rounds closed unanchored-name and missing 64K/secondary-grid/generic-depth/per-agent/per-repetition fixture gaps; all four real traces satisfy the exact contract.

Both fully hashed 16K runs remain **wall-unstable exit-4 diagnostics**, so neither establishes TG or permits a CSA decision. Exact attribution agrees:

| 16K artifact | Profiled reps / tokens | Wall MAD/median | Aggregate top | Runner-up | Lead | Per-repetition top |
|---|---:|---:|---:|---:|---:|---|
| `...T145326...-16k-a-...-35ee01b7edf9-6030` | 5 / 160 | 6.385% | NCCL 23.020% | non-MoE quantized 19.749% | 3.271 pt | NCCL 5/5 |
| `...T150438...-16k-b11-...-667bc100a5cc-32436` | 10 / 320 | 6.511% | NCCL 28.871% | non-MoE quantized 20.086% | 8.785 pt | NCCL 10/10 |

NCCL satisfies the locked 16K rule in two independent profiles and is selected **for investigation only**. Routed experts are 9.392% / 8.499% and shared experts 7.446% / 6.539%; neither wins. No implementation or throughput gain is accepted.

Exact concat-aware 64K attribution now reproduces in two independent wall-unstable profiles. The second (`...T180714...-raw-tg-profile-64k-b11-...-877a73b581c9-31443`, 10 profiled tg32 repetitions / 320 tokens, 2.6 GiB) is process-clean, exact-count, zero-outside-boundary, and exit 4 at 6.406% wall MAD; no sample is removed.

| 64K artifact | Profiled reps / tokens | Wall MAD/median | NCCL | Runner-up | Lead | Per-repetition top |
|---|---:|---:|---:|---:|---:|---|
| `...T161130...-64k-a-...27f432de97fc-17771` | 5 / 160 | 5.356% | 20.173% | non-MoE quantized 17.080% | 3.092 pt | NCCL 5/5 |
| `...T180714...-64k-b11-...877a73b581c9-31443` | 10 / 320 | 6.406% | 22.762% | non-MoE quantized 16.502% | 6.260 pt | NCCL 10/10 |

Both aggregates clear the 20% communication threshold and 3-point lead, with NCCL first in all 15 retained repetitions. Per-repetition minima (19.041% share / 1.727-point lead) are reported but do not add an undeclared every-repetition threshold; reproducibility is established by the two qualifying aggregates plus consistent first rank. Communication is therefore selected at 64K/global **for investigation only**. Critical-path proof, implementation, throughput, and CSA acceptance remain false.

| Exact component | 16K device ms/token | 64K-A device ms/token | 64K-A/16K | 64K-A share |
|---|---:|---:|---:|---:|
| NCCL kernels | 25.484 | 26.075 | 1.023x | 20.173% |
| lightning indexer | 3.149 | 11.051 | 3.509x | 8.550% |
| TOP_K | 2.123 | 2.419 | 1.140x | 1.871% |
| flash attention | 4.218 | 8.942 | 2.120x | 6.918% |
| CSA K materialization | 1.722 | 5.958 | 3.459x | 4.609% |
| CSA final-mask concat | 0.142 | 0.164 | 1.155x | 0.127% |
| unclassified `other` | 18.793 | 18.837 | 1.002x | 14.573% |

The exact named CSA path (LID + TOP_K + flash + CSA K/mask concat) sums to 22.075% at 64K, but this is **not a removable-share projection**: components have different remedies and combining them cannot select indexed CSA. Flash alone is 6.918%, CSA K materialization 4.609%, and no indexed implementation is selected.

Commit `fa1e98ba2` adds a fail-closed read-only communication forensic analyzer. `ncclAllReduce` is the only traced collective API function. Every repetition has exactly 86 traced AllReduce groups/token, 344 rank calls/token (86/GPU/token), and 2,752 groups plus 11,008 rank calls/device kernels per tg32 repetition; cadence is invariant at 16K/64K. NCCL kernel median/p95/p99 is 58.281/121.842/346.284 us in 16K-A, 58.840/122.281/331.286 us in 16K-B, 57.681/137.762/362.404 us in 64K-A, and 57.921/128.762/336.043 us in 64K-B. 16K-B repetition 8 retains three ~1.0618 s long intervals on Agents 2/3/4; 64K-B repetition 7 has four ~346.8 ms intervals and repetition 10 one 79.195 ms interval. These are **long intervals / stall candidates**, not proven stalls or causes. Each GPU uses one RCCL stream (14/15/16/17), and same-agent non-NCCL timestamp overlap is at most 0.160% at 16K / 0.033% at 64K; timestamps do not prove dependencies, resource overlap, rank causality, or critical-path membership.

The exact supported RCCL CSV has no count/datatype/buffer/communicator/rank/stream/message-byte fields, and RCCL API/device-kernel correlation-ID sets are disjoint. Message size, algorithm/protocol, direct API-to-kernel mapping, and the critical path therefore remain unproven. API durations remain separate and are never added to device time. Next: capture predeclared RCCL argument/dependency evidence and choose only the proven communication remedy before a matched TG candidate. Indexed CSA remains held.

Decision branches:

- **A: routed/shared MoE dominant.** Record weight type, M/N/K, expert count, tokens/expert, wave size, dispatch count, total time, and achieved bandwidth. Optimize only the hottest observed combination; measure activation quantization with MMQ. Shared/routed overlap or final-add fusion is eligible only if its standalone cost is material.
- **B: LID/TOP_K dominant.** Keep TOP_K GPU-resident, optimize the actual GPU selector, reduce score-buffer traffic, then consider fused indexer + hierarchical selection (`tile -> wave/workgroup candidates -> global merge -> final 512`). Exact selection still scans all candidates; streaming bounds intermediate traffic, not asymptotic scan time.
- **C: attention meets the predeclared M5.6 gate at long context.** Then consider indexed CSA. CUDA draft PR #25917 is design/correctness reference only (open draft, CUDA MMA, author measurements); its tiling and crossover are not ROCm evidence.
- **D: elementwise/launch work meets the branch threshold.** Only then pursue decode-chain fusion (compression/norm/RoPE/cache conversion/insertion or inverse-RoPE/packing/projection preparation).
- **E: communication — selected for implementation at 16K/64K.** Exact source and end-to-end A/B now drive the remedy. Further tracing is exception-only; generic kernel duration and PCIe topology alone still do not accept an algorithm.

### M5.4 / P1-A - stop profiling and test communication solutions

The user explicitly ended routine profiling after two independent 16K and two independent 64K exact-role artifacts selected NCCL in all 30 retained repetitions. That evidence is sufficient to try solutions; additional trace collection is allowed only to diagnose an anomalous candidate. Causal unprofiled whole-model TG now outranks another attribution pass.

The source closes the basic payload question without a new trace. `ggml_backend_cuda_comm_allreduce_nccl` calls `ncclAllReduce(..., ne, ncclFloat, ncclSum, ...)` for four-way tensors with `ne < 262144`. Direct GGUF metadata reads establish `deepseek4.block_count=43` and `deepseek4.embedding_length=4096`; the completed dispatch audit then identifies 344 small FP32 calls across four target inputs, exactly two row-parallel hidden outputs per layer/token. Thus the 86 groups/token (86 rank calls/GPU/token) are 4,096-element FP32 reductions (16,384 bytes of input per rank per call). The attention output must be reduced before its residual/norm becomes the FFN input, and the FFN output is only available later; blindly combining the two collectives would violate dependencies. The output-head force-FP32 flag is separate and is not one of these 86 layer collectives.

Installed-library attestation, not an upstream-version guess: `/opt/rocm/core-7.14/lib/librccl.so.1` reports RCCL 2.30.4, accepts `NCCL_ALGO`/`NCCL_PROTO`, and contains an explicit diagnostic that `NCCL_MIN_NCHANNELS` is ignored for fewer than eight GPUs. Therefore the first predeclared reversible set is exactly:

1. `tree-ll`: `NCCL_ALGO=Tree NCCL_PROTO=LL` — run first;
2. `auto`: both unset — run only if tree-ll is plausibly >=3% over the accepted historical 64K median, to obtain a contemporaneous control;
3. `ring-ll`: `NCCL_ALGO=Ring NCCL_PROTO=LL` — one fallback if tree-ll fails the matched gate.

No 1/2-channel sweep and no rocprof run participate. `screen-rccl-tg.sh` fails closed on inherited `NCCL_*`/`RCCL_*` and `GGML_CUDA_DISABLE_GRAPHS`, forces one load with exact 16K/32K/64K tg32, six raw/one discarded/five accepted repetitions, full hashes, accepted stack, and no speculative path. Commits `a9c80dd84` and `4b8caa954` replace the false setup-only `ENV,TUNING` claim with `INFO/ENV`, preserve complete raw stdout plus separated non-JSON output, timestamp only parsed JSON records, fail on malformed/unterminated/excessive output or consumer errors, normalize run-local manifest identities, require `GGML_HIP_GRAPHS:BOOL=ON`, and reject runtime graph disable. The comparator permits RCCL's small setup matrix but rejects the observed per-collective `AllReduce:`/`threadThreshold`/channel-tuning families. It requires stable complete identity-matched results and predeclares >=3% median TG gain at 64K with no >2% median regression at 16K/32K. A five-repetition pass selects full 31-repetition validation only; `optimization_accepted=0` until validation and correctness pass.

The first Tree+LL attempt at source `45064b0d3` is **invalid measurement instrumentation**, not a candidate result: `$HOME/llama-jobs/dsv4-rocm-tg/20260804T192247.461599490Z-raw-tg-rccl-screen-tree-ll-performance-45064b0d3397-4656/` contains 1,497,247 non-JSON stdout lines, including 88,064 AllReduce and 4 x 352,256 per-call tuning lines. The old consumer mixed them into `result.jsonl` and spawned `date` per line; no `summary.json` exists and the three ~0.159 t/s records are ineligible. `screen-invalid.txt` and `screen-invalid-analysis.json` preserve the rejection. Do not salvage or compare it.

A metadata-only depth-0/tg1 non-evidence smoke at `a9c80dd84` then proved the repaired installed-library behavior: `$HOME/llama-jobs/dsv4-rocm-tg/20260804T214242.500623821Z-raw-tg-rccl-env-smoke-tree-ll-residency-a9c80dd84f74-24936/` has one JSON/one timestamp, 27 non-JSON lines, zero malformed/excessive output, capture rc0, exact Tree/LL acknowledgements on all four ranks, `GGML_HIP_GRAPHS:BOOL=ON`, and no per-collective tuning markers. The setup matrix is permitted evidence of environment processing, not per-collective path proof.

The repaired forced screens are complete but produce no candidate eligible for a matched control or full validation:

- Tree+LL: `$HOME/llama-jobs/dsv4-rocm-tg/20260804T214837.068203556Z-raw-tg-rccl-screen-tree-ll-performance-4b8caa954627-16362/`; medians 22.7059 / 16.4285 / 14.8470 t/s at 16K/32K/64K; MAD/median 0.137% / 4.668% / 15.576%. It is globally unstable, regresses 64K by 19.320% versus the accepted historical median, and even its best 64K sample (18.3567 t/s) is below the predeclared 18.9544 t/s plausibility threshold. Per protocol, `auto` was not run.
- Ring+LL fallback: `$HOME/llama-jobs/dsv4-rocm-tg/20260804T221418.737155192Z-raw-tg-rccl-screen-ring-ll-performance-4b8caa954627-214/`; medians 23.0775 / 16.4675 / 18.3176 t/s; MAD/median 0.364% / 14.023% / 0.352%. It is globally unstable, regresses 32K by 16.532%, and its stable 64K median is 0.460% below historical rather than >=3% above.

Both repaired artifacts have three JSON/three timestamps, zero malformed/excessive output, capture rc0, twelve exact algorithm plus twelve LL acknowledgements, no forbidden per-collective diagnostics, clean process status, and graph build/runtime attestation. They are **NO-GO for advancement**; no matched comparison JSON, 31-repetition validation, or optimization acceptance is allowed. This exhausts only the predeclared runtime-control branch, not generic RCCL behavior.

The final predeclared source candidate is a shape-scoped RDNA2/four-way guard that tests BF16 for these unforced hidden reductions, halving collective input to 8,192 bytes while paying conversion kernels. Exact opt-in `GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE=1` requires HIP+RCCL, explicit `GGML_CUDA_ALLREDUCE=nccl`, four distinct unshared physical RDNA2 devices, and contiguous F32 rank tensors of exact shape `[4096,1,1,1]`; every miss retains the old size heuristic. Force-FP32 is ORed across ranks and wins before the candidate. The implementation reuses the existing F32-to-BF16 / BF16 RCCL sum / BF16-to-F32 body, cleans partially initialized communicators before any fallback, and offers a correctness-only per-context audit. It does not collapse the two dependency-separated reductions.

Per the user's iteration-time override, the initial gate is intentionally short and can reject or select further work but cannot accept the optimization. Correctness runs one explicit FP32 control and one BF16 candidate at only 2K context with four deterministic fixed target inputs. It captures raw full-vocabulary F32 logits and requires identical argmax tokens, finite values, every element within `0.05 + 0.01*scale`, RMSE <=0.02, exactly 344 eligible hidden reductions, zero candidate dispatches in control, 344 in candidate, and the observed exact dynamic force-FP32 count of zero; rank-wise force precedence is covered by the pure host selector test because this model path does not emit a forced AllReduce in the capture. Only a correctness pass permits a matched 0/2K/8K screen using tg8, six raw/one discarded/five retained samples per arm. Any >2% regression is a NO-GO; less than 4% gain at 8K is not worth longer testing. A pass is only `PROMISING_SHORT_SCREEN`, with `optimization_accepted=0`. No 16K/32K/64K run occurs in this stage; any longer confirmation requires a new explicit user decision.

The first post-commit 2K A/B artifact at source `19373e2bf` is **invalid candidate-path evidence**, not a BF16 correctness result: `$HOME/llama-jobs/dsv4-rocm-bf16-equivalence/20260805T001143.656347516Z-bf16-hidden-short-correctness-19373e2bfb15-293/`. Both arms completed in about 70 seconds and produced logits, but the candidate audit recorded 1,032 total AllReduce calls, zero candidate-eligible calls, zero BF16 candidate calls, and no dispatch marker; the control was identical. The 344 small FP32 plus 688 legacy BF16 calls across four target inputs show the original 7,168-element premise was wrong. A diagnostic rerun at `d4cec50f5`, `$HOME/llama-jobs/dsv4-rocm-bf16-equivalence/20260805T001920.668484789Z-bf16-hidden-short-correctness-d4cec50f54af-17349/`, again proved zero 7,168-element calls while explicitly attesting the four-device RDNA2 topology. Direct GGUF metadata then showed this model's hidden width is 4,096, not 7,168. No numeric or performance conclusion is allowed from either artifact. Correct the exact guard and audit to `[4096,1,1,1]`, require 344 matching calls and 344 dispatches, and rerun the same short correctness gate.

The corrected clean run at source `3e2861d85` is a complete, provenance-valid **NO-GO**: `$HOME/llama-jobs/dsv4-rocm-bf16-equivalence/20260805T003241.023493282Z-bf16-hidden-short-correctness-3e2861d85e15-11945/`. Audit proves the intended path exactly: both arms have 1,032 total calls, 344 exact `[4096,1,1,1]` eligible F32 reductions, and 688 unchanged legacy BF16 reductions; control routes all 344 eligible calls to FP32 while candidate routes all 344 to candidate BF16. All four argmax tokens match and no value is nonfinite, but every record fails the numerical gate. RMSE is 0.07808 / 0.18557 / 0.06170 / 0.17095 versus the 0.02 limit, with 35,662 / 83,471 / 24,359 / 81,175 combined-tolerance violations out of 129,280 logits per record; maximum absolute differences are 0.3744 / 1.0524 / 0.3635 / 1.0006. Therefore no 0/2K/8K TG performance screen is permitted. The guarded BF16 candidate is rejected, and no 16K/32K/64K confirmation or acceptance run is allowed.

### M5.6 / P2 - mandatory 32K and 64K raw TG

The target-only TG half of this gate is now satisfied by accepted M5.0: tg32 with 30 accepted repetitions at both 32K and 64K, one model load, full-context state, median 19.729 / 18.402 t/s, and MAD/median 0.266% / 0.138%. Older PP-only 32K and failed 64K PP observations do not contribute. The remaining M5.6 blocker is a predeclared component-removable-share projection tied to an indexed-CSA candidate; the two 64K diagnostics above rank components but do not make their summed 22.075% a removable share.

Any future matched candidate must retain tg32 (or declare tg64 uniformly before measurement), **at least five valid decode repetitions at both 32K and 64K**, one model load where valid, and separate context setup from measured generation. Disable duplicate 64K warmup if the harness supports a verified `--no-warmup`/equivalent path. Reuse/cache a prompt state only after proving target logits/tokens match fresh prefill. Keep A/B commands identical. A cap-limited, unstable, early-stop, or fewer-than-five point is incomplete and permits no CSA acceptance/permanent rejection; repair the harness/resource issue or leave CSA on hold.

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
| 2026-08-04 | Make target-only raw decode the blocking next phase and hold indexed CSA. | No accepted MTP/DSpark-disabled repeated TG sweep existed at that checkpoint; PP scaling could not select the raw-decode bottleneck. | superseded by M5.0 acceptance; CSA hold remains |
| 2026-08-04 | Accept the M5.0 harness mechanics, not a raw-TG result. | Dry runs, source/CLI audit, parser fixtures, and fake end-to-end runs passed success/incomplete/unstable/setup-timeout/measured-timeout/residency cases without loading a model or launching GPU work. A reproduced FIFO deadlock was fixed before acceptance. | tooling accepted; TG later accepted by M5.0 |
| 2026-08-04 | Reject sequence-only depth-state restore for DSV4; require full context state. | Controlled 2K/3K/16K gate: fresh re-prefill is bit-identical, sequence restore has ~516.9K logit violations/depth and one 2K argmax divergence, while full context restore is bit-identical with zero token/logit/state mismatches. | accepted correctness fix `f97f5cdb0` |
| 2026-08-04 | Treat large HIP TOP_K work as conditional, not presumed missing. | Current branch enables HIP hipCUB, uses rocPRIM top-k for DSV4 large rows, and advertises TOP_K support; scheduler residency must still be attested in raw decode. | selected diagnostic |
| 2026-08-04 | Invalidate old percentage profiles for target selection after >3% whole-model gains. | Accepted J16/HC/LID/T128 changes materially altered Amdahl shares; LID alone changed 16K PP by +10.18%. | final rule |
| 2026-08-04 | Keep MTP/DSpark outside raw-decode baselines and kernel selection. | Exact-greedy MTP state diverges; speculative acceptance/checkpoint behavior is a separate workstream. | final |
| 2026-08-04 | Exclude in-band `rocm-smi` telemetry from accepted TG samples. | First full sweep showed 32K/64K MAD 13.7%/12.2% while polling every 1s inside 1.5-3s samples; with setup-only telemetry the same long points re-measure 0.33-3.97% and 64K stabilizes at 0.33%. Telemetry now samples setup + the discarded first repetition only. | accepted fix `81b072481` |
| 2026-08-04 | Raw-TG residency is attested at the composite backend, not per-GPU. | Scheduler exposes one `Meta(ROCm0..ROCm3)` backend; independent review: top-level split/copy counts cannot prove per-GPU execution or copies. | final caveat |
| 2026-08-04 | Residency parser counts real TOP_K/LIGHTNING_INDEXER ops only, with exact expected counts enforced. | Parser previously counted CONT/SET_ROWS consumers of `lid_top_k`, inflating 21 real nodes to 63; exact-op reparse of the preserved log requires and confirms exactly 21 TOP_K + 21 LID per measured graph at every depth 2048-65536, all on `Meta(ROCm0..ROCm3)`, zero CPU/unknown. Depth 0 has neither op. | accepted fix `4936a8673` |
| 2026-08-04 | Evidence runs require full GGUF shard hashes, all-resolved-DSO hashes, and recorded power/performance policy. | Independent review: metadata-only hashing and absent power profile failed strict identity attestation; added `DSV4_HASH_MODE=full` plus expanded rocm-smi policy snapshot. | accepted fix `4936a8673` |
| 2026-08-04 | Reject the fully-hashed 16-rep full sweep: 4K/8K over the stability gate. | MAD/median 4K=3.80%, 8K=3.15% with ordered warm-to-steady regime shift; 15 accepted at every depth; identities fully hashed (3 GGUF shard + ~40 ROCm/system DSO hashes, perf level auto, no custom power cap set). Remedy per policy: more tg32 repetitions. | rejected; full31 launched |
| 2026-08-04 | Pause all GPU work at user direction; full31 sweep aborted cleanly. | Job terminated (no KFD PIDs, lock released, no result). Raw TG remained pending at that checkpoint. | superseded by resumed acceptance |
| 2026-08-04 | Accept the fully hashed target-only M5.0 raw-TG baseline. | Clean `1cd80107ee76`; tg32, 31 raw/30 accepted at all eight depths; all latency MAD/median <=1.77%; 7,680 accepted target decodes; zero of 1,054 telemetry query starts in accepted intervals; full model/binary/46-DSO/power identities. | accepted |
| 2026-08-04 | Accept M5.1 composite Meta scheduler residency; do not trigger M5.2. | Separate tg1 log: exactly 21 TOP_K + 21 LIGHTNING_INDEXER on `Meta(ROCm0..ROCm3)` at 2K-64K, none at depth 0, zero CPU/unknown DSV4 assignment; exact split/marker structure. Does not attest per-GPU ownership or selector correctness. | accepted, qualified |
| 2026-08-04 | Fail closed on complete residency graph structure. | `6f7115360`: default selector requirement starts at 2K; exact benchmark/depth/generation markers, CPU0 + exact Meta1 split/input counts, operation/backend correlation, no extras/warnings; negative fixtures and real 529 MiB log reparse pass. | accepted tooling fix |
| 2026-08-04 | Use ROCTx selected regions for disk-safe target-only raw-decode profiling. | `35ee01b7e`: authoritative in-process monotonic boundaries surround accepted `test_gen` only; exact full-hash/stack/config contract; CSV kernel/copy/RCCL/HIP domains; fail-closed summarizer. Three independent reviews; selected-region smoke captured exactly 1 of 3 kernels. | accepted tooling |
| 2026-08-04 | Stabilize profiles by increasing tg32 repetitions only; preserve attribution on unstable exit 4. | Default 5-profiled-rep 16K run missed at 6.385% MAD; 10-profiled-rep retry still missed at 6.511%. `667bc100a` keeps one discard and tg32; `09f575d10` emits scope-valid diagnostic attribution but marks throughput and CSA ineligible. | accepted policy/tooling |
| 2026-08-04 | Do not select an M5.3 implementation branch from the first coarse 16K classification. | Initial type-only grouping mixed shared FFN and projections, producing apparent aggregate rank disagreement. All samples were retained. | superseded by exact-role parser `27f432de9` |
| 2026-08-04 | Correct every documented artifact prefix to `$HOME/llama-jobs`. | Final-monitor execution exposed the inherited impossible `$HOME/edwin/llama-jobs` expansion (`/home/edwin/edwin/...`). All 60 canonical occurrences now match the actual preserved directories. | accepted documentation fix `2c8bab656` |
| 2026-08-04 | Select communication as the 16K M5.3 investigation branch, not an accepted optimization. | Exact 43-layer normal/anomalous MoE signatures and counts leave NCCL first in both independent profiles: 23.020% vs 19.749% (3.271-point lead) and 28.871% vs 20.086% (8.785-point lead), NCCL first in all 15 repetitions. | selected at 16K `27f432de9` |
| 2026-08-04 | Keep the 64K/global M5.3 branch unresolved under the coarse concat classification. | First 64K profile originally ranked NCCL 20.173% vs mixed `other` 19.722%, only 0.451-point lead and split 3/5 vs 2/5. | superseded by exact concat attribution `a789f138c` |
| 2026-08-04 | Preserve and pin exact profile parser provenance in every new artifact. | `7355f8bcc` records parser provenance/command; `85833a12d` pins reruns to the last parser-changing commit. Current artifacts pin exact concat parser `a789f138c`. | accepted tooling fix |
| 2026-08-04 | Attribute 16K/64K CSA/HCA K and final-mask concat exactly. | Source-unique axis/type plus exact depth grids and 21/20-layer x token x four-GPU counts globally/per GPU/per repetition. Real traces and expanded negative fixtures pass after two review rounds. `other` is flat 1.002x, while CSA K concat scales 3.459x. | accepted diagnostic tooling `a789f138c` |
| 2026-08-04 | Treat communication as a provisional 64K candidate, not a global selection. | Reclassified first 64K profile: NCCL 20.173% vs non-MoE quantized 17.080%, 3.092-point lead, NCCL first 5/5. Only one independent 64K artifact existed at that checkpoint. | superseded by replicated 64K selection |
| 2026-08-04 | Select communication as the 64K/global M5.3 investigation branch, not an accepted optimization. | Second independent 64K profile: NCCL 22.762% vs non-MoE quantized 16.502%, 6.260-point lead, NCCL first 10/10; first profile also qualifies, for 15/15 consistent ranks. Both wall curves remain ineligible. | selected at 64K for investigation only |
| 2026-08-04 | Freeze routine profiling and pivot to unprofiled communication candidates. | Four independent 16K/64K profiles already select NCCL in 30/30 ranks; later GGUF metadata plus dispatch audit identify 86 FP32 16,384-byte layer collectives/token. More attribution cannot establish a TG gain. | tree-ll first; matched auto only if promising |
| 2026-08-04 | Invalidate the first Tree+LL attempt and harden stdout/graph/identity evidence. | `ENV,TUNING` emitted 1,497,247 non-JSON lines and the old consumer spawned `date` per line. `a9c80dd84`/`4b8caa954` add raw capture/classification, ENV-only acknowledgement, normalized manifests, compiled/runtime graph gates, and adversarial fixtures; repaired smoke passes. | invalid instrumentation; never throughput evidence |
| 2026-08-04 | Do not advance Tree+LL or Ring+LL and do not run auto. | Repaired Tree has 64K median 14.847 t/s and 15.576% MAD; Ring has stable 64K 18.318 t/s (-0.460% historical) but unstable/regressed 32K. Neither meets preliminary plausibility/stability, so conditional auto and matched comparison are not allowed. | runtime-control branch exhausted; guarded BF16 source candidate next |
| 2026-08-05 | Reject guarded BF16 hidden AllReduce before performance. | Corrected exact `[4096,1,1,1]` audit proves 344/344 candidate dispatches, but all four full-vocabulary records exceed the numerical tolerance and RMSE limit (0.06170-0.18557 vs 0.02). Argmax equality alone is insufficient. | final predeclared decode candidate closed NO-GO; 0/2K/8K TG screen skipped |
| 2026-08-04 | Bound communication evidence without inventing payload or critical-path claims. | Exact cadence is 86 AllReduce groups/token and 11,008 rank calls/device kernels per tg32 rep. RCCL schema lacks message arguments; API/kernel correlation IDs are disjoint. Long intervals and near-zero same-agent compute overlap do not prove cause/dependencies. | accepted forensics `fa1e98ba2`; critical path open |

## 10. Closed decisions and open questions

Closed: IQ3_XXS J16 T128 passed focused exact-output, dispatch/counter, natural-proxy, and whole-model gates and is accepted as guarded optimization four. Its older/unclassified-AMD performance coverage remains an upstreaming-scope caveat, not an open local acceptance question. M5.0 now supplies the stable target-only tg32 curve at every required actual starting KV depth. M5.1 confirms all measured DSV4 TOP_K/LIGHTNING_INDEXER operations remain on the composite four-device Meta backend through 64K with no CPU/unknown assignment; per-GPU execution/copy ownership remains outside that attestation.

Open questions:

1. Does J16 hold on a future user-supplied production corpus? The committed technical proxy is positive, but no user corpus exists.
2. Can a later expert-concentration signal select J16/J32/J64 without host synchronization? The accepted patch intentionally stays explicit.
3. Does Tree+LL or Ring+LL improve causal whole-model TG for the metadata/audit-derived 86 x 16,384-byte FP32 layer collectives/token; if not, does guarded BF16 reduction pass correctness and the same gate?
4. Does HIP flash attention perform partial arbitrary-mask tile pruning, and can its 2.120x growth or the source-proven 3.459x CSA K materialization support a predeclared >=3% removable whole-model projection after replication?
5. How are LID scores and top-k indices assigned across the four meta devices at runtime?
6. If MTP is reopened separately, which recurrent state/logit component changes after verification/checkpoint round trips with zero accepted drafts?
7. Which fixed corpus best represents production once the user supplies one?

## 11. Reproduction record

The controlled PP baseline, fixed natural-text proxy, four accepted PP
optimizations, full-context restore gate, target-only M5.0 raw-TG baseline,
and M5.1 composite-backend residency attestation are complete. PP caveats
remain: repeated matched 32K A/B, successful 32K attribution, and every 64K PP
measurement are missing; the prior 64K PP attempt terminated in warmup. No
user-supplied production corpus has been accepted. Raw TG now has 30 accepted
tg32 samples at every required depth and a separate strict tg1 scheduler audit.
Sequence-only restore remains rejected and fails closed. Selected-region M5.3
tooling, exact MoE/concat attribution, and bounded communication forensics are
accepted. The two unstable 16K and two unstable 64K profiles select
communication for investigation by exact device attribution (first in all 30
retained repetitions), never throughput. The next blocking evidence is explicit
RCCL message/algorithm/dependency and critical-path attribution, not another
branch-selection or raw-TG baseline sweep.

**Current non-GPU raw-TG tooling monitors:**

```bash
cd /home/edwin/llama.cpp-rdna2
scripts/dsv4-rocm/test-tg-tools.py
scripts/dsv4-rocm/test-tg-profile.py
scripts/dsv4-rocm/test-tg-communication.py
DSV4_TG_DEPTHS=65536 DSV4_TG_REPS=6 scripts/dsv4-rocm/profile-tg.sh --dry-run
```

These launch no GPU/model work. The old static-validation artifact remains
preserved, but its embedded fake residency split (`Meta/2 inputs`) predates the
strict `6f7115360` 22/25-input parser and its complete `commands.sh` is no
longer the canonical monitor.

**Accepted restored-state monitor command:**

```bash
cd /home/edwin/llama.cpp-rdna2
cmake --build build --target test-state-restore-equivalence -j 12
DSV4_STATE_API=context DSV4_LABEL=context-state-controlled \
  scripts/dsv4-rocm/run-state-restore-equivalence.sh
```

The accepted result is the `5d80b8662` artifact listed above; all original/fresh-repeat/restored logits and argmax tokens are bit-identical. M5.0/M5.1 no longer require rerunning; use their preserved pair monitor. Recheck GPU ownership immediately before any new M5.3 profile.

**Current exact non-GPU M5.3 diagnostic monitor:**

```bash
A=$HOME/llama-jobs/dsv4-rocm-tg-profile/20260804T145326.793240911Z-raw-tg-profile-16k-a-performance-35ee01b7edf9-6030
B=$HOME/llama-jobs/dsv4-rocm-tg-profile/20260804T150438.588831000Z-raw-tg-profile-16k-b11-performance-667bc100a5cc-32436
L1=$HOME/llama-jobs/dsv4-rocm-tg-profile/20260804T161130.092022210Z-raw-tg-profile-64k-a-performance-27f432de97fc-17771
L2=$HOME/llama-jobs/dsv4-rocm-tg-profile/20260804T180714.788127093Z-raw-tg-profile-64k-b11-performance-877a73b581c9-31443
for d in "$A" "$B" "$L1" "$L2"; do "$d/profile-parser-command.sh"; done
python3 "$B/profile-comparison-monitor.py"
python3 "$L2/profile-replication-monitor.py"
"$L2/communication-forensics-command.sh"
```

Expected results are `M5.3 16K COMMUNICATION BRANCH: SELECTED
(INVESTIGATION ONLY)` with `branch_selected=1 selected_branch=communication
optimization_accepted=0`, followed by `M5.3 64K COMMUNICATION BRANCH: SELECTED
(INVESTIGATION ONLY)` with `independent_64k_profiles=2 profiled_repetitions=15
branch_selected=1 selected_branch=communication critical_path_proven=0`, then
`M5.3 DSV4 NCCL FORENSICS: COMPLETE (CRITICAL PATH NOT PROVEN)` with four run
rows and `cross_run_cadence_invariant=1 critical_path_proven=0`. Every unstable wall
sample remains; only exact selected-region device attribution participates in
branch selection. Exact GPU commands were:

```bash
cd /home/edwin/llama.cpp-rdna2
DSV4_TG_DEPTHS=16384 DSV4_LABEL=raw-tg-profile-16k-a \
  scripts/dsv4-rocm/profile-tg.sh
DSV4_TG_DEPTHS=16384 DSV4_TG_REPS=11 DSV4_LABEL=raw-tg-profile-16k-b11 \
  scripts/dsv4-rocm/profile-tg.sh
DSV4_TG_DEPTHS=65536 DSV4_TG_REPS=6 DSV4_LABEL=raw-tg-profile-64k-a \
  scripts/dsv4-rocm/profile-tg.sh
DSV4_TG_DEPTHS=65536 DSV4_TG_REPS=11 DSV4_LABEL=raw-tg-profile-64k-b11 \
  scripts/dsv4-rocm/profile-tg.sh
```

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

The recorded run (`$HOME/llama-jobs/dsv4-rocm-pp/20260804T003038.699935606Z-final-full-stack-4opt-c98197389511-25811/`) was complete with all four shapes and median 293.744 / 523.352 / 437.512 / 365.332 t/s at 512/2K/8K/16K. Correctness of the committed stack is attested by the natural-proxy gate `20260803-225603-iq3-t128-corpus-fb2a0c85d` (`complete=1`).

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
`$HOME/llama-jobs/dsv4-corpus-validation/20260803T190100.516971835Z-attested-56dd4177e501-15597/`
for HC and
`$HOME/llama-jobs/dsv4-corpus-validation/20260803T212823.803707936Z-attested-9f4808637e55-20974/`
for LID. The accepted post-LID attribution trace is
`$HOME/llama-jobs/dsv4-rocm-pp/20260803T215054.700650714Z-kernel-trace-j16-hc-lid-16k-fdde31252a63-8573/`;
the directly comparable pre-LID trace remains
`$HOME/llama-jobs/dsv4-rocm-pp/20260803T191856.045376424Z-kernel-trace-j16-hc-16k-52e0121043ad-23195/`.
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
`$HOME/llama-jobs/dsv4-lid-study/20260803T195000Z-bd4d1b9aa-baseline/`.
The production failure is under
`$HOME/llama-jobs/20260803-194659-dsv4-prod-mtp-j16-hc-r3/`.

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
  4936a8673 (evidence provenance hardening: exact-node parser counts, full hashes, power policy) ->
  1cd80107e (iteration-3 pause/evidence checkpoint; source of accepted resumed artifacts) ->
  6f7115360 (strict marker/split/backend residency parser) ->
  7390b73d0 (M5.0/M5.1 acceptance record) ->
  35ee01b7e (selected-region target-only profile harness) ->
  667bc100a (repetition-only profile stabilization) ->
  09f575d10 (unstable-attribution preservation + decode MMVQ classification) ->
  10ce4d660 (unresolved coarse 16K profile evidence record) ->
  2c8bab656 (actual `$HOME/llama-jobs` artifact paths) ->
  43427c6cf (first M5.3 diagnostic checkpoint) ->
  27f432de9 (exact DSV4 MoE/non-MoE profile attribution) ->
  7355f8bcc (automatic profile-parser provenance) ->
  85833a12d (pin parser reruns to recorded implementation commit) ->
  ba3bf23a3 (16K selected / first-64K unresolved checkpoint) ->
  a789f138c (exact CSA/HCA concat profile attribution) ->
  fa1e98ba2 (fail-closed decode communication forensics) ->
  877a73b58 (bounded-communication checkpoint; source of second 64K profile)

Raw-decode Ralph log:
  /Users/edwin/.ralph/dsv4-raw-decode-roadmap.md
Raw-decode Ralph state:
  /Users/edwin/.ralph/dsv4-raw-decode-roadmap.state.json
Raw-decode Ralph status:
  active, iteration 3/50; M5.0/M5.1 accepted, communication selected at 16K/64K for investigation; critical path and optimization open; started 2026-08-04T03:47:49Z
Revised roadmap / loop-registration commits:
  5df30a53e / 0376a55aa
M5.0 harness / corrected depth-state commits:
  1e5519bf1 / f97f5cdb0
M5.0 static-validation artifacts:
  $HOME/llama-jobs/dsv4-rocm-tg/static-validation-20260804T0415Z-0376a55aacd6/
Accepted M5.0 performance artifact:
  $HOME/llama-jobs/dsv4-rocm-tg/20260804T124716.565555325Z-raw-tg-baseline-full31-resumed-performance-1cd80107ee76-27396/
Accepted M5.1 residency artifact:
  $HOME/llama-jobs/dsv4-rocm-tg/20260804T131957.468937324Z-raw-tg-residency-hardened-full-residency-1cd80107ee76-22574/
Diagnostic M5.3 profile artifacts (all exit 4 / unstable wall):
  $HOME/llama-jobs/dsv4-rocm-tg-profile/20260804T145326.793240911Z-raw-tg-profile-16k-a-performance-35ee01b7edf9-6030/
  $HOME/llama-jobs/dsv4-rocm-tg-profile/20260804T150438.588831000Z-raw-tg-profile-16k-b11-performance-667bc100a5cc-32436/
  $HOME/llama-jobs/dsv4-rocm-tg-profile/20260804T161130.092022210Z-raw-tg-profile-64k-a-performance-27f432de97fc-17771/
  $HOME/llama-jobs/dsv4-rocm-tg-profile/20260804T180714.788127093Z-raw-tg-profile-64k-b11-performance-877a73b581c9-31443/
Invalid/repaired M5.4 runtime artifacts:
  INVALID: $HOME/llama-jobs/dsv4-rocm-tg/20260804T192247.461599490Z-raw-tg-rccl-screen-tree-ll-performance-45064b0d3397-4656/
  ENV smoke: $HOME/llama-jobs/dsv4-rocm-tg/20260804T214242.500623821Z-raw-tg-rccl-env-smoke-tree-ll-residency-a9c80dd84f74-24936/
  Tree+LL NO-GO: $HOME/llama-jobs/dsv4-rocm-tg/20260804T214837.068203556Z-raw-tg-rccl-screen-tree-ll-performance-4b8caa954627-16362/
  Ring+LL NO-GO: $HOME/llama-jobs/dsv4-rocm-tg/20260804T221418.737155192Z-raw-tg-rccl-screen-ring-ll-performance-4b8caa954627-214/
Current next action:
  Routine profiling remains frozen. Runtime Tree+LL/Ring+LL and guarded BF16
  candidates are all NO-GO. The BF16 failure occurred at the deterministic 2K
  logit gate, so its 0/2K/8K TG screen and every longer run are skipped. No
  predeclared decode candidate remains and no decode optimization is accepted.
  M5.2 is not triggered; indexed CSA remains held pending a new user decision.

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
- No accepted raw-TG baseline existed at the pause checkpoint; CSA remained undecided. GPUs were idle, lock free, and the repo clean at `4936a8673`.

### Iteration 3 resumed — M5.0/M5.1 accepted

- Revalidated clean `1cd80107ee76`, no KFD/llama owner, free GPU lock, and 112 GiB free. Marked the earlier aborted full31 artifact explicitly `acceptance=none` (seven complete rows; stopped in 64K setup).
- Rebuilt `llama-bench` and ran the exact fully hashed resumed command. Performance artifact `20260804T124716.565555325Z-raw-tg-baseline-full31-resumed-performance-1cd80107ee76-27396` exits 0, complete/stable: tg32, 31 raw/30 accepted per depth, every MAD/median <=1.77%, 7,680 accepted target decode calls. No timeout/truncation; zero of 1,054 telemetry query starts is inside an accepted interval.
- Ran the separate fully hashed tg1 residency audit at the same source/binary/model/46-DSO/power identity. Artifact `20260804T131957.468937324Z-raw-tg-residency-hardened-full-residency-1cd80107ee76-22574` exits 0 and confirms exact composite-Meta DSV4 residency through 64K.
- Independent read-only review found no evidence blocker and recommended M5.0/M5.1 closure, but identified residual parser false-pass cases. Commit `6f7115360` closes them with exact marker, split, backend/input, 2K-node, no-extra, and warning-free requirements; negative fixtures and the 529 MiB real log reparse pass.
- Preserved monitor: `pair-acceptance-validation.py` in the performance artifact checks source/config/status, 3 model hashes, 46 matching DSO hashes, binary, power policy, telemetry boundaries, summary contract, and strict residency JSON. Its rerun prints `M5.0/M5.1 PAIR ACCEPTANCE: PASS`. A later exact monitor run exposed inherited `$HOME/edwin/llama-jobs` documentation paths; every canonical artifact prefix is corrected to the actual `$HOME/llama-jobs`.
- Accepted: M5.0 observational + post-fix/deployment baseline and M5.1 composite scheduler residency. M5.2 is not triggered. Per-GPU copy/execution ownership and selector correctness are not claimed. Next: M5.3 fresh raw-decode profiling; indexed CSA remains held.

### Iteration 3 continued — M5.3 profiler and first diagnostics

- Added the fail-closed selected-region profiler at `35ee01b7e`; three independent reviews found and closed exact-contract bypasses, asynchronous-boundary false attestation, missing/malformed trace domains, unprofiled command regression, and non-HIP build portability. Non-GPU fixtures, negative dry-run matrix, CMake build, direct ROCTx smoke (exactly one of three kernels captured), and process-group safety pass.
- First fully hashed clean 16K run (`...T145326...35ee01b7edf9-6030`, five profiled tg32 reps) proved real domain emission and exact boundary scope but exited 4 at 6.385% profiled-wall MAD. Per policy, no tokens/discard changed; `667bc100a` permits repetition-only expansion.
- The 11-raw/10-profiled retry (`...T150438...667bc100a5cc-32436`) also exited 4 at 6.511% wall MAD. Both runs preserve every sample, three GGUF and 53 DSO hashes, clean source, exact accepted runtime stack, all four trace domains, zero outside-boundary events, and 713,244 dispatches per repetition.
- Offline classifier/parser `09f575d10` recognizes decode MMVQ and all `dsv4_hc_*` kernels and emits per-repetition ranks. Run A aggregate: other quantized matmul 27.446%, NCCL 23.020%; all five reps rank quantized first. Run B aggregate: NCCL 28.871%, quantized 26.842%; eight reps rank quantized, two rank NCCL due retained stalls. Flash/LID/TOP_K are only 3.3-3.8% / 2.4-2.8% / 1.7-1.9% at 16K.
- Initial decision: no branch from the coarse classifier. That diagnostic and `final-monitor-rerun-2c8bab656.txt` remain preserved, but the conclusion is superseded below by exact-role attribution; no samples changed or were removed.

### Iteration 3 continued — exact 16K roles and first 64K profile

- Audited every quantized kernel shape against the fully hashed GGUF tensor inventory. The apparent `other_quantized_matmul` winner mixed shared experts with projections. Commit `27f432de9` splits routed/shared/non-MoE families using exact type/fusion/workgroup/grid signatures and requires the 43-layer normal/anomalous subtype counts globally, per GPU, and per repetition. Missing/wrong block count, subtype, grid, fusion, or count fails closed. Two independent reviewer passes: initial NO-GO defects repaired; final GO; fixture suites and real-trace reparses pass.
- Corrected 16K result: NCCL 23.020% vs non-MoE quantized 19.749% (3.271-point lead; first 5/5 reps) and NCCL 28.871% vs 20.086% (8.785-point lead; first 10/10). Communication clears its 20% threshold and is selected for **16K investigation only**. Wall throughput and CSA remain ineligible; no optimization is accepted. The old monitor/JSON are backed up with `.pre-27f432de9` names and the new exact monitor is preserved in run B.
- Added automatic parser commit/command provenance at `7355f8bcc`, then pinned reruns to the recorded last parser-changing commit at `85833a12d`; tests and 64K dry-run pass.
- Rechecked clean source, idle GPUs, free lock, and >100 GiB disk; rebuilt `llama-bench` at `27f432de9`. Disk-safe 64K command used six raw/one discard/five profiled tg32 repetitions and produced only 1.3 GiB, not a full whole-process rocprof CSV.
- 64K artifact `20260804T161130.092022210Z-raw-tg-profile-64k-a-performance-27f432de97fc-17771`: process clean, zero outside-boundary events, exact per-agent/per-repetition MoE counts, but wall MAD 5.356% -> exit 4 diagnostic. NCCL 20.173% leads `other` 19.722% by only 0.451 point; NCCL is first 3/5 reps, `other` 2/5. No 64K/global branch selection.
- Scaling: NCCL device ms/token is nearly flat (1.023x 16K->64K), while LID is 3.509x, flash 2.120x, TOP_K 1.140x, and unclassified context concat 3.452x. Named LID+TOP_K+flash is 17.339% at 64K, but flash alone is 6.918% and no >=3% removable whole-model projection exists; indexed CSA remains held.
- Preserved `profile-scaling-monitor.py` and exact parser command in the 64K artifact. End state after tooling pin: clean `85833a12d`, no KFD PID, GPU lock free, 106 GiB free.

### Iteration 3 continued — exact attention roles and communication bounds

- `a789f138c` maps source-unique CSA/HCA K axis-2 and final-mask axis-0 concat signatures at 16K/64K with exact 21/20-layer x token x four-GPU contracts. Initial reviewer NO-GO (near-name and coverage gaps) was repaired with exact demangled names and 64K/secondary-grid/generic-depth/per-agent/per-repetition negatives; follow-up GO. All three real traces reparse exactly.
- CSA K concat is 1.722 -> 5.958 device ms/token (3.459x); CSA final-mask concat is 0.142 -> 0.164 (1.155x); remaining `other` is 18.793 -> 18.837 (1.002x). This proves the context-scaling concat is K materialization, not restore, but selects no indexed implementation.
- Reclassification changes the first 64K rank to NCCL 20.173% vs non-MoE quantized 17.080%, 3.092-point lead, NCCL first 5/5. Marked `single_64k_branch_rule_met=1` but `independent_64k_profiles=1 replication_required=1 global_branch_selected=0`.
- `fa1e98ba2` adds exact-schema communication forensics. Initial review NO-GO (schema/claim/test gaps) was repaired; follow-up GO. Three real artifacts prove invariant 86 AllReduce groups/token and 11,008 rank calls/device kernels per tg32 repetition. The RCCL schema has no message arguments and API/kernel correlation IDs are disjoint; critical-path proof remains false.
- Preserved exact reparses/monitors plus `communication-forensics.{json,txt}`, its pinned command/commit, and `.pre-a789f138c` summaries/monitors. All unstable samples remain; no throughput, optimization, or CSA acceptance.

### Iteration 3 continued — second 64K profile selects communication

- Rechecked clean `877a73b58`, no KFD/llama/rocprof owner, free GPU lock, 106 GiB disk, 108 GiB available RAM; rebuilt `llama-bench` (SHA-256 `46476bbad97a1d9c788af96bdee0f0cef76dd847a306317ade20f0896f9237d1`).
- Ran `DSV4_TG_DEPTHS=65536 DSV4_TG_REPS=11 DSV4_LABEL=raw-tg-profile-64k-b11 scripts/dsv4-rocm/profile-tg.sh`. The 2.6 GiB artifact has 10 profiled tg32 repetitions, 320 target tokens, 7,147,520 dispatches, zero outside events, exact MoE/concat counts, clean process/identity, and exit 4 wall MAD 6.406%; all samples retained.
- Exact attribution reproduces: NCCL 22.762% vs non-MoE quantized 16.502%, lead 6.260 points, NCCL first 10/10. Together with 64K-A (20.173%, 3.092-point lead, first 5/5), two independent aggregates and all 15 ranks select communication at 64K for investigation. Reviewer GO confirmed aggregate thresholds are the declared rule; per-repetition minima 19.041%/1.727 points do not add an undeclared stricter rule.
- Repaired the durable replication monitor to use explicit `ValueError` gates under Python `-O`, exact MAD/raw instability, repetition lengths, dispatches, trace/role contracts, and eligibility flags; follow-up reviewer GO.
- Four-run forensics retain invariant cadence. 64K-B has four ~346.8 ms long NCCL intervals in repetition 7 and one 79.195 ms interval in repetition 10; neither establishes cause/dependency/critical path. Preserved `profile-replication-64k.{json,txt}`, exact monitor/hash, and four-run `communication-forensics.{json,txt}` in the second 64K artifact.

### Iteration 3 continued — pivot from profiling to causal A/B

- User correctly identified diminishing returns: no decode throughput has improved during M5.3. Profiling delivered branch selection, not speed. Routine profiling is now frozen and future progress is measured by unprofiled whole-model TG.
- GGUF metadata plus dispatch audit identify exactly two dependency-separated 4,096-element FP32 layer reductions across each of 43 blocks. The existing backend threshold is based on RTX 4090 PCIe and has not been tuned for four RDNA2 V620s. Combining collectives is rejected; reversible RCCL selection is the fastest safe first solution.
- Installed RCCL is 2.30.4, not the initially researched 2.28 baseline. Local binary strings attest Tree/Ring and LL support and reject a useful min-channel experiment below eight GPUs. The predeclared order is tree-ll -> conditional matched auto -> one ring-ll fallback; no profiler.
- Added fail-closed wrapper, environment provenance, matched comparator, and fixtures. Screen gate: stable five accepted tg32 samples at 16K/32K/64K, >=3% 64K median gain, <=2% shorter regression; pass advances only to full 31-repetition correctness/performance validation.

### Iteration 3 continued — RCCL screens fail; source candidate selected

- First Tree+LL attempt at `45064b0d3` is invalid instrumentation: 1,497,247 non-JSON stdout lines under `ENV,TUNING`, per-line `date` process creation, no summary, and no eligible throughput. Preserved explicit rejection in the artifact.
- `a9c80dd84` adds the persistent fail-closed stdout classifier, raw/non-JSON logs, consumer-status propagation, ENV-only control acknowledgement, exact graph/runtime state, normalized run-local identities, and real FIFO failure fixtures. Follow-up `4b8caa954` permits the observed small ENV setup matrix while continuing to reject per-collective tuning. Multiple blocker reviews end GO.
- Depth-0/tg1 metadata smoke proves installed RCCL's ENV-only behavior with 27 diagnostics, exact Tree/LL acknowledgements, no per-collective markers, and clean capture/process/graph state. It is validation only, not performance evidence.
- Repaired Tree+LL screen exits 4: medians 22.7059/16.4285/14.8470 t/s; MAD 0.137%/4.668%/15.576%. It fails preliminary plausibility and stability; auto is correctly skipped.
- Ring+LL fallback exits 4: medians 23.0775/16.4675/18.3176 t/s; MAD 0.364%/14.023%/0.352%. Its stable 64K result is -0.460% historical, not >=3%, and 32K is unstable/regressed. No candidate advances, no matched comparator is run, and no optimization is accepted.
- Implemented the guarded RDNA2/four-way BF16 conversion/reduction path, corrected the original 7,168-width assumption to GGUF-proven 4,096, and proved exactly 344 eligible/dispatch calls in the four-target gate. Full-vocabulary correctness is NO-GO despite identical argmax: all four records exceed tolerance and RMSE 0.02. The 0/2K/8K screen is intentionally skipped; no decode optimization is accepted and no predeclared candidate remains.

**Exact accepted GPU commands:**

```bash
cd /home/edwin/llama.cpp-rdna2
DSV4_HASH_MODE=full DSV4_TG_REPS=31 \
  DSV4_LABEL=raw-tg-baseline-full31-resumed scripts/dsv4-rocm/run-tg.sh
DSV4_HASH_MODE=full DSV4_TG_MODE=residency \
  DSV4_LABEL=raw-tg-residency-hardened-full scripts/dsv4-rocm/run-tg.sh
```

**Exact non-GPU pair monitor:**

```bash
PERF=$HOME/llama-jobs/dsv4-rocm-tg/20260804T124716.565555325Z-raw-tg-baseline-full31-resumed-performance-1cd80107ee76-27396
python3 "$PERF/pair-acceptance-validation.py"
```

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