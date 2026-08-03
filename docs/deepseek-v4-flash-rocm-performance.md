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

These are provenance data, not a statistical baseline. The new baseline must use the IQ2_M target, current base commit, repeated PP-only runs, and no speculative draft during PP attribution unless explicitly testing draft overhead.

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

Artifacts under `perf/dsv4-rocm/`:

- `manifest.sh`: hardware, source, build, model hashes/metadata, environment, command.
- `run-pp.sh`: PP-only deterministic runner with repetitions and unique output directory.
- `summarize.py`: median/p95, raw timings, failures, and identities.
- `profile.sh`: rocprofv3 trace wrapper plus rocm-smi sampling.
- `README.md`: safe use, including refusal to run while another llama process owns GPUs unless explicitly overridden.

Initial practical matrix:

- prompt tokens: 512, 2048, 8192, 32768; add 64K/128K only after stability;
- ubatch: 128, 256, 512; batch fixed at least as large;
- modes: PP-only (`-n 1` or benchmark-native PP), then fixed-prefix TG separately;
- repetitions: two warmups and five measured runs for short cases; at least three measured for expensive cases;
- main run: IQ2_M, four GPUs, tensor split 1,1,1,1, FA on, no draft for PP attribution;
- comparison: draft enabled only in a separate PP+TG run.

Metrics: PP t/s and wall time, peak VRAM/RAM, per-GPU utilization/clocks/power, kernel time/calls, HIP memcpy/P2P bytes, RCCL time, attention/LID/top-k/MoE stage percentages, and failures.

### M1 - select one optimization from profile evidence

Decision rule:

- dense CSA attention >=15% of PP at 8K+ and scales with context: implement indexed CSA;
- LID/top-k >=15% and score traffic is material: fuse/tune LID selection;
- MoE/MMQ >=35% at 512-8K: tune the dominant routed expert path;
- copies/collectives >=20%: optimize tensor split/communication first.

Do not select by architectural elegance alone.

### M2 - correctness proof and microbenchmark

Before integration, add a focused backend op test or deterministic reference for the changed operation. Required cases include short visible length, exactly/above top-k, chunk boundaries, unequal sequence lengths where supported, and gfx1030-specific dispatch fallback.

### M3 - implementation and matched A/B

One implementation branch and one writer. Keep the base binary and build artifacts. Run static/backend tests first, then DSV4 validation, then matched benchmark/profile. Report local kernel speed separately from whole-model PP.

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

## 9. Open questions

1. What is the current IQ2_M PP baseline without speculative draft at 512/2K/8K/32K?
2. What fraction of PP is MoE/MMQ, communication/copies, CSA attention, LID, top-k, compression, and HC?
3. Does HIP flash attention perform any arbitrary-mask tile pruning? Source and counters must answer.
4. How are LID scores and global top-k distributed across the four meta devices at runtime?
5. Are peer copies direct and what bandwidth does GPU3's x8 path achieve?
6. Which existing RDNA2 MMQ branch is already proven correct and beneficial for this exact IQ2_M expert shape?
7. Is rocprofv3 stable with this custom kernel/ROCm stack, and which counters are available on V620?
8. What short-prompt/long-context split best represents the user's production workload?

## 10. Reproduction record

Pending the first controlled GPU window. Do not populate this section with the active production service's throughput because it has an external client and speculative decode enabled.

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