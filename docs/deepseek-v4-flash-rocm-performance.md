# DeepSeek-V4-Flash ROCm prompt-processing performance plan

Status: living engineering record  
Owner branch: `perf/dsv4-rocm-pp-20260803`  
Base: `b88a59fbc6ac255e6bf5e2dd790f559c89ce911c` in Edwin's llama.cpp fork  
Target host: `edwin@192.168.1.161` (`webhie`)  
Last updated: 2026-08-03

## 1. Objective and success criteria

Primary objective: improve DeepSeek-V4-Flash prompt-processing throughput on the four-V620 ROCm host without changing model math or sacrificing existing llama.cpp deployment modes.

Secondary objectives: improve decode only when it does not distract from PP; keep generic and non-HIP fallbacks; upstream work as independently reviewable changes when practical.

A change is successful only when all of the following hold:

1. A matched before/after run shows a repeatable PP gain on this host.
2. The changed operation is proven to execute in a trace or targeted test.
3. DSV4 layer-reference versus tensor-split validation remains green.
4. Short-context performance does not regress materially; long-context wins are reported separately.
5. Exact source/build/model/runtime identities and raw logs are preserved.

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
- The active service uses all four GPUs, tensor split `1,1,1,1`, FA on, F16 K/V, context 262144, batch 512, ubatch 256, and DSpark/MTP max 3.

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

Above the 512-compressed-entry crossover (roughly above 2K native positions), generic attention is passed every compressed entry. Long-context CSA attention work should scale with context/4 rather than selected 512 + local 128.

Test: fixed ubatch, sweep 2K/8K/32K/64K. Compare flash-attention kernel duration and memory traffic against compressed length.

Candidate implementation if confirmed: introduce a backend-agnostic indexed shared-KV attention operation with a generic correctness fallback, then implement a HIP direct-index kernel. A gather proof is acceptable only if it handles per-query index sets and the union softmax over selected compressed entries, local SWA, and sinks correctly.

### H2 - LID plus top-k is an independent long-context bottleneck

Even perfect sparse attention must scan all LID entries. The HIP vector kernel emits all scores and top-k rereads them. At long context this is a large F32 round trip and selection cost.

Test: profile `lightning_indexer_kernel_vec`, rocPRIM top-k, argsort/copies, and score tensor bytes over the same context sweep.

Candidate implementation if confirmed: tiled fused LID + local top-k, then hierarchical merge. Keep a debug/reference path that emits full scores. For tensor split, merge device-local candidates with correct global indices.

### H3 - PCIe tensor-split overhead limits PP

Four GPUs have no XGMI and GPU3 is x8. Multi-GPU collectives, mirrored activations, and meta scheduler transfers may explain low PP at short context.

Test: capture HIP copies/RCCL and per-GPU kernel timelines; compare one/two/four GPUs only with model residency/offload effects explicitly separated. Measure device imbalance.

Candidate implementation if confirmed: reduce mirrored activations/collectives, aggregate transfers, revisit tensor ownership, or choose an asymmetric split that compensates for GPU3 x8. Do not optimize a compute kernel before proving communication is not dominant.

### H4 - RDNA2 quantized MoE kernels dominate

The 284B model activates about 13B parameters/token. The IQ2/IQ3 expert path and routed MMQ shape may underutilize V620s during PP.

Test: attribute PP to `MUL_MAT_ID`/MMQ and routing; record achieved bandwidth/waves, expert tile occupancy, and per-layer time. Compare relevant existing RDNA2 K-quant commits rather than restarting from upstream.

Candidate implementation if confirmed: tune only the measured dominant GGUF quant/shape, guarded by architecture/shape dispatch with generic fallback.

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
crosses 15% at 16K. The complete artifact is 809 MiB. A complete 32K sample
exceeds the fixed five-minute measured cap even with warmup disabled, so 16K
is the longest complete attribution point under the current rule.

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
phase is LID vector-kernel/fused-selection investigation, not communication
first. Name-matched MMQ remains the largest family and stays in the roadmap;
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

Mapping and screening artifacts:

- `$HOME/edwin/llama-jobs/dsv4-rocm-rocblas/20260803T175331Z-ec1b7e64c-map-cijk/` (three-line aggregate rocBLAS profile)
- `$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260803T175022.376406151Z-trace-map-cijk-256-ec1b7e64c2cc-24970/` (single-microbatch trace)
- `$HOME/edwin/llama-jobs/dsv4-hc-mixes-sweep/20260803T181014Z-1d6a42983-prototype/` (tile sweep, correctness, fallback, graph, and dispatch proof)
- `$HOME/edwin/llama-jobs/dsv4-hc-mixes-sweep/20260803T182030Z-560635e3b-whole-model/` (J16-held-constant whole-model A/B)
- `$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260803T191856.045376424Z-kernel-trace-j16-hc-16k-52e0121043ad-23195/` (combined-stack 16K compact trace, aggregate and per-agent measured-region summaries)

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
- `$HOME/edwin/llama-jobs/dsv4-corpus-validation/20260803T184944.079084014Z-attested-56dd4177e501-25863/` (excluded interrupted attempt: all responses exist and match, but no final status/comparison artifact and `candidate.rc=120`)
- `$HOME/llama-jobs/dsv4-corpus-validation/20260803T165136Z-88c415d91/` (superseded pre-normalization artifact)

### M3 - implementation and matched A/B

One implementation branch and one writer. Keep the base binary and build artifacts. A frozen baseline is valid only when its sibling llama/ggml DSOs are selected and hashed; an executable that resolves candidate-build libraries is not frozen. Run static/backend tests first, then DSV4 validation, then matched benchmark/profile. Report local kernel speed separately from whole-model PP.

### M4 - independent review and next decision

Fresh review must cover correctness/causality, HIP synchronization/memory safety, generic backend behavior, multi-GPU index semantics, and benchmark validity. Update this document with accepted findings and raw artifact paths.

## 7. Correctness contract for indexed CSA

If indexed CSA becomes M1, it must preserve:

- shared K=V MQA, 64 query heads, 512-dimensional heads;
- per-query selected index sets and top-k <=512;
- local 128-token raw/SWA branch;
- one stable softmax over selected compressed entries + local entries + per-head sink;
- causal visibility and compression completion boundaries;
- inverse partial RoPE behavior;
- stream/batch sequences with different lengths/phases;
- deterministic duplicate/tie policy compatible with the existing top-k reference;
- dense generic fallback and a force-reference switch for testing.

A dense mask is not a sparse performance implementation. Success requires runtime or traffic scaling with selected + local entries after the crossover.

## 8. Decision log

| Date | Decision | Evidence | Status |
|---|---|---|---|
| 2026-08-03 | Use `/home/edwin/llama.cpp-rdna2` as sole base. | User direction. | final |
| 2026-08-03 | Branch from `b88a59fbc`, retaining gfx1030 top-k fixes. | Active fork history and source audit. | final |
| 2026-08-03 | Do not interrupt current server; request a GPU window before controlled runs. | External client connected; GPUs 99% busy. | final |
| 2026-08-03 | PP first; TG secondary. | User objective. | final |
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
| 2026-08-03 | Investigate LID vector/fused selection as optimization phase three. | Lightning indexer is 14.87% and LID plus separate top-k name families is about 15.2% at 16K, up from 6.30% at 8K; MMQ remains 40.19% but has already received its first routing-sensitive optimization. | provisional |

## 9. Open questions

1. Does the J=16 win hold on a user-supplied production corpus? The attested engineering proxy and through-16K synthetic scaling are positive, but no user corpus exists and 32K cannot complete under the five-minute cap.
2. Can a later expert-concentration signal select J=16/J=32/J=64 without a host synchronization? This patch intentionally stays explicit.
3. Which part of the HIP LID vector kernel (K dequantization, 64-head dot products, reduction, or score write) is limiting, and can score production be fused with local top-k without breaking global-index/tie semantics?
4. Does HIP flash attention perform partial arbitrary-mask tile pruning? Scaling rejects complete fixed-top-k pruning, but source/counters do not yet quantify partial pruning.
5. How are LID scores and global top-k distributed across the four meta devices at runtime?
6. RCCL kernels do not make x8 bus 46 the slowest, but what algorithm roles and actual peer bytes/bandwidth explain the per-agent RCCL asymmetry?
7. Which short-prompt/long-context mix and fixed corpus best represent the user's production workload?

## 10. Reproduction record

The first controlled window and fixed natural-text proxy are complete. These
are not yet the final acceptance command; 32K+ and any user-supplied production
corpus remain deferred. All harness runs cap measured time at five minutes
while excluding model load.

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
fresh shell at clean commit `52e012104` (the validation code is unchanged from
the acceptance build at `56dd4177e`):

```bash
cd /home/edwin/llama.cpp-rdna2
cmake --build build --target llama-server llama-bench -j 12

DSV4_BASE_MMQ_J=16 DSV4_CANDIDATE_MMQ_J=16 \
DSV4_BASE_HC_MIXES=0 DSV4_CANDIDATE_HC_MIXES=1 \
DSV4_BATCH_SIZE=512 DSV4_UBATCH_SIZE=256 DSV4_HASH_MODE=full \
scripts/dsv4-rocm/run-corpus-validation.sh

trace_log=$(mktemp)
HSA_OVERRIDE_GFX_VERSION=10.3.0 HSA_NO_SCRATCH_RECLAIM=1 \
GGML_HIP_GRAPHS=1 GGML_CUDA_ALLREDUCE=nccl GGML_CUDA_P2P=1 \
GGML_HIP_RDNA2_MMQ_J=16 GGML_HIP_RDNA2_HC_MIXES=1 \
DSV4_PROFILE=kernel DSV4_LABEL=kernel-trace-j16-hc-16k \
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

Accepted artifacts are
`$HOME/edwin/llama-jobs/dsv4-corpus-validation/20260803T190100.516971835Z-attested-56dd4177e501-15597/`
and
`$HOME/edwin/llama-jobs/dsv4-rocm-pp/20260803T191856.045376424Z-kernel-trace-j16-hc-16k-52e0121043ad-23195/`.
These commands are a phase checkpoint, not the project's final completion
command; production-MTP and the selected LID phase remain.

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