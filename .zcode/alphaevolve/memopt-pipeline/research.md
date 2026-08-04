# research.md - run: memopt-pipeline

Goal: "optimize the memory usage of the pipeline"

Synthesized from two deep research passes (OSS engines + frontier academia,
2023-2026) cross-referenced against a codebase survey of tessera's actual
memory surfaces. Each finding is tagged with a tessera relevance verdict:
DONE / PARTIAL / MISSING.

A note on trust: sources marked [verified] were resolved against arXiv/official
sites during this phase. Sources marked [agent-reported] came from a background
research agent and look plausible but were not independently re-resolved beyond
the spot-checks noted.

---

## Findings

### A. Paged / virtual-memory KV management

**A1. PagedAttention (vLLM, SOSP 2023).** KV cache broken into fixed-size blocks
stored in non-contiguous memory, addressed per-sequence by a block table;
blocks refcounted for copy-on-write sharing. Cuts KV waste from 60-80% to <4%.
[verified] https://arxiv.org/abs/2309.06180

**A2. vAttention (ASPLOS 2025).** Keeps KV contiguous in virtual address space
but maps to physical GPU pages lazily via CUDA virtual-memory APIs
(cuMemMap/cuMemSetAccess). Up to 1.23x over paged kernels; works with stock
attention kernels. [verified] https://arxiv.org/abs/2405.04437

### B. Chunked prefill and scheduling

**B1. Sarathi-Serve (OSDI 2024).** Chunked prefill packed with decodes in the
same batch; stall-free scheduling. 2.6-5.6x serving capacity. [verified]
https://arxiv.org/abs/2403.02310

### C. Disaggregation / distributed KV

**C1. DistServe (OSDI 2024).** Separate prefill and decode GPU pools, transfer
KV once. 7.4x requests / 12.6x tighter SLO. [verified]
https://arxiv.org/abs/2401.09670

**C2. DistAttention / Infinite-LLM (ASPLOS 2025).** KV cache as cluster-wide
virtual memory spilling across GPUs and host DRAM. 1.35-3.4x throughput.
[verified] https://arxiv.org/abs/2401.02669

**C3. Mooncake (FAST 2025, Moonshot/Kimi).** KVCache-centric disaggregated pool
on reclaimed CPU/DRAM/SSD. Up to +525% simulated throughput. [verified]
https://arxiv.org/abs/2407.00079

### D. Sparse KV eviction

**D1. H2O (NeurIPS 2023).** Evict lowest cumulative-attention-score tokens each
step; keep heavy hitters + recent window. Up to 29x throughput, 1.9x lower
latency. [verified] https://arxiv.org/abs/2306.14048

**D2. StreamingLLM (ICLR 2024).** Keep attention-sink tokens + rolling recent
window; constant-memory unbounded generation. Stable to 4M+ tokens. [verified]
https://arxiv.org/abs/2309.17453

### E. KV cache quantization

**E1. KIVI (ICML 2024).** Asymmetric 2-bit: keys per-channel, values per-token.
2.6x less peak memory, 4x batch size. [verified] https://arxiv.org/abs/2402.02750

**E2. KVQuant (NeurIPS 2024).** Pre-RoPE key quant, non-uniform per-layer
sensitivity, dense+sparse outlier split. 1M context on 1xA100-80GB. [verified]
https://arxiv.org/abs/2401.18079

**E3. CacheGen (SIGCOMM 2024).** KV cache compressed via delta + entropy coding;
streamed over network. 3.5-4.3x smaller KV. [verified]
https://arxiv.org/abs/2310.07240

### F. Architectural KV compression

**F1. MLA / DeepSeek-V2.** Low-rank joint KV latent compression; only the latent
vector is cached. 93.3% KV reduction at iso-quality. Requires retraining.
[verified] https://arxiv.org/abs/2405.04434

### G. Weight + KV offloading

**G1. FlexGen (ICML 2023).** GPU+CPU+SSD hierarchy; LP-solved offload schedule;
4-bit weight+activation compression. OPT-175B on a single 16GB GPU. [verified]
https://arxiv.org/abs/2303.06865

**G2. InfiniGen (OSDI 2024).** Predict next-layer KV touches via partial
rehearsal; prefetch only those from CPU. Up to 3.0x over baselines. [verified]
https://arxiv.org/abs/2406.19707

**G3. vLLM V1 RECOMPUTE.** On memory pressure, drop KV blocks and replay the
prompt instead of swapping to pinned CPU. Avoids pinned-mem cost. [agent-reported]
https://docs.vllm.ai/en/stable/configuration/optimization/

**G4. TensorRT-LLM host KV offload.** Reusable blocks offloaded to host before
eviction; opt-in. Up to 14x/28x TTFT. [agent-reported]
https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html

### H. MoE expert offloading

**H1. Mixtral-Offloading (Eliseev & Mazur, 2023).** LRU GPU expert cache;
speculative expert prefetch (apply next-layer gate to current hidden states);
mixed HQQ precision (4-bit attention, 2-3 bit experts). Mixtral-8x7B in 12-16GB
VRAM. [verified] https://arxiv.org/abs/2312.17238

**H2. KTransformers (SOSP 2025).** Attention/MLA on GPU, sparse MoE experts on
CPU; Intel AMX for CPU expert GEMM. DeepSeek-V3/R1 671B on a single GPU; up to
286 tok/s prefill. [verified]
https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-bla/private/SOSP25-chen.pdf

**H3. Pre-gated MoE (ISCA 2024).** Pre-gating router decides routing one layer
ahead; prefetch overlaps with compute. Large MoE on a single GPU at ~23%
overhead vs oracle. Requires retraining. [verified] https://arxiv.org/abs/2308.12066

**H4. HOBBIT / Fiddler / fMoE.** Mixed-precision and fine-grained expert swap.
8-19x single-batch latency improvement on Mixtral. [agent-reported]
https://arxiv.org/abs/2411.01433

### I. Apple Silicon / unified memory

**I1. MLX.** Unified memory model; CPU/GPU/ANE share one address space with no
copies. Eliminates duplicated weight/KV copies (~2x RSS reduction vs
copy-in/copy-out frameworks). [verified] https://github.com/ml-explore/mlx

**I2. vLLM-MLX (2026 preprint).** vLLM-style server on MLX. 21-87% higher
throughput than llama.cpp; 525 tok/s on M4 Max. [verified]
https://arxiv.org/abs/2601.19139

**I3. Core ML StateType (iOS 18 / macOS 15).** KV as state input read/written
in place across prediction calls; avoids per-token I/O copy. Toy attention
4245ms -> 238ms. [agent-reported]
https://apple.github.io/coremltools/docs-guides/source/stateful-models.html

**I4. Open-TQ-Metal (2026 preprint).** On-the-fly int4 KV quantization fused
into the Metal attention kernel. KV ~40GB -> ~12.5GB on Apple UMA. [verified]
https://arxiv.org/html/2604.16957v1

### J. Hybrid recompute

**J1. HybridServe (arXiv 2501.01792).** Stores intermediate activation
checkpoints to rapidly recompute KV while parameters transfer; hybrid caching
balances KV-vs-activation ratio. ~50% cached-context memory cut. [verified]
https://arxiv.org/abs/2501.01792

---

## Relevance to tessera

Tessera is more memory-advanced than a typical llama.cpp fork. Four major
memory techniques are already landed. This narrows the productive search space
considerably - the wins are in what is NOT done yet.

### Already DONE in tessera

1. **Paged attention (A1), deeply integrated.** `GGML_OP_TESSERA_PAGED_ATTN`
   (`ggml/src/ggml.c:1086`) is implemented on CPU (`ggml-cpu/ops.cpp:9250`),
   Metal (`kernel_tessera_paged_attn` with f32/f16 specializations at
   `ggml-metal/ggml-metal.metal:11240`), and referenced by ANE
   (`ggml-ane.mm:1202`). The block-table boundary lives at
   `src/llama-kv-cache.h:101-130` (`tessera_kv_block_span`,
   `tessera_kv_block_table`, `make_page_map`), wired in at
   `src/llama-kv-cache.cpp:2776`. So PagedAttention-the-paper is DONE. vLLM and
   TRT-LLM's basic paging is not a target.

2. **Radix-tree KV prefix sharing.** `tools/server/server-context.cpp:831-896`
   tracks `n_kv_radix_hits`, `n_kv_prefix_shares`, `n_kv_block_radix_attaches`.
   That is the prefix-cache / KV-reuse win already shipped. vLLM APC and
   TRT-LLM KV-reuse are not targets.

3. **KV cache quantization (8-bit and 4-bit).** `--cache-type-k`/`-v` accept
   q8_0, q4_0, q4_1, iq4_nl, q5_* (`common/arg.cpp:2694`,
   `common/common.h:604`). So llama.cpp-style KV quant is DONE; the frontier
   (KIVI E1, KVQuant E2) is the additive part - see candidates below.

4. **Streaming weight loading.** Commits `02ac74294`, `770bddee4`,
   `cf0c49fbf` - streaming load for the screening pipeline already landed as
   the 16GB-OOM fix.

5. **Chunked prefill cap (B1, partial).** `tools/server/server-prefill-policy.{h,cpp}`
   is a per-iteration shared prefill cap (referenced as the
   vLLM-concurrency-parity fix). So Sarathi-Serve's *memory-spike* benefit is
   partially addressed at the scheduling layer.

### PARTIAL - existing scaffolding, incomplete payoff

6. **Paged-attention block table (A1) - the "compressed block reader" is incomplete.**
   The block-table comments at `src/llama-kv-cache.h:97-101` explicitly say the
   table is "the common boundary for direct quantized/paged attention. A span
   is bounded by block_size so a backend can substitute a compressed block
   reader without materializing a contiguous K/V staging tensor." But the
   paged-attn CPU impl (`ops.cpp:9250`) asserts `k->type == F32 || F16` and
   `v->type == F32 || F16` - i.e. it reads K/V as f16/f32, not quantized
   blocks. The boundary exists; the compressed reader does not. This is the
   single biggest tessera-native opening: a quantized-KV block reader behind
   the existing TESSERA_PAGED_ATTN op would let q8_0/q4_0 KV flow through the
   same path with no f16 staging buffer.

7. **MoE expert offload (H1-H4) - design only.** `docs/moe-disk-offload-study.md`
   studies WASTE (Marco Bambini) for NVMe streaming of Kimi K3, but no code.
   The upstream `--cpu-moe` flag may or may not be present in tessera; needs
   confirmation. KTransformers-style AMX expert GEMM is absent.

### MISSING - not present, additive if adopted

8. **vAttention-style virtual-memory paging (A2).** Tessera's paging is at the
   block-table level, not the OS/driver level. On Apple Silicon there is no
   CUDA driver API to mirror, but the *philosophy* (lazy physical commit
   rather than eager reservation) maps to mmap-backed KV residency, which is
   MISSING. llama.cpp RFC issue 20757 proposes exactly this.

9. **Sparse KV eviction (D1, D2).** H2O heavy-hitter eviction and StreamingLLM
   attention sinks are not present. Tessera's KV cache keeps full history per
   sequence; for unbounded/long-streaming workloads these would cap KV memory
   independently of sequence length.

10. **Frontier 2-3 bit KV quant (E1, E2).** KIVI (asymmetric per-channel K,
    per-token V) and KVQuant (pre-RoPE, non-uniform) go below q4_0. Not present.

11. **KV cache compression (E3).** CacheGen-style delta+entropy coding is not
    present; orthogonal to paging and quantization.

12. **KV offload to host with speculative prefetch (G2, G4).** InfiniGen's
    next-layer prefetch and TRT-LLM's host-offload-before-eviction are not
    present. Relevant if/when tessera targets GPU+host hybrid deployment.

13. **MLA (F1).** Architectural; requires retraining. Out of scope for an
    inference-only evolutionary agent unless the target model already uses MLA
    (DeepSeek-V2/V3). Worth confirming whether the model in the workload does.

14. **Disaggregation (C1-C3).** DistServe/Infinite-LLM/Mooncake are
    cluster-scale; tessera is a single-node server. Out of scope for a
    single-host memory-optimization run.

15. **Core ML StateType (I3).** Only relevant if tessera routes through Core ML;
    it routes through its own Metal/ANE backends. Not applicable.

### Adoptions with no payoff for this goal

- vLLM V1 RECOMPUTE (G3): trades memory for compute; useful under pressure but
  not a peak-RSS reducer at fixed workload.
- RevNet/Reformer (training lineage): inference-irrelevant.
- MLX (I1) / vLLM-MLX (I2): a different framework, not an in-tessera technique.
- FlexGen (G1): offline throughput focus, single-GPU orientation; not a fit for
  the interactive server workload.

---

## Candidate approaches

Ordered by expected payoff x tessera-fit, highest first. Each lists the region
it would touch - this drives Phase 5 stack-compatibility analysis.

**S1. Quantized-KV block reader behind TESSERA_PAGED_ATTN.** [tessera-native,
PARTIAL -> DONE] Wire q8_0/q4_0 K/V through the existing paged-attn op instead
of f16. The block-table boundary is already there; the kernel side asserts f16
and must learn to dequant on the fly. Largest tessera-native win because it
unlocks KV quant (already a flag) without a staging buffer.
Source: A1 + E1/E2 + Open-TQ-Metal (I4).
Region: `ggml/src/ggml-cpu/ops.cpp:9250+`, `ggml/src/ggml-metal/ggml-metal.metal:11240+`,
        `ggml/src/ggml-ane/ggml-ane.mm:1202+`, `src/llama-kv-cache.cpp:2776+`.
Expected: ~50% KV memory (q8) to ~75% (q4) with no f16 staging buffer; peak RSS
          drop proportional to KV share of total memory.
Risk: kernel correctness on quantized block boundaries; quality with q4 KV.

**S2. mmap-backed KV residency (lazy physical commit).** [MISSING -> DONE]
Mirror vAttention's lazy-commit philosophy on Apple UMA using mmap for KV
buffers rather than eager allocation. Reduces peak RSS when n_ctx is set high
but the actual sequence is short.
Source: A2 + llama.cpp issue 20757.
Region: `src/llama-kv-cache.cpp` (buffer allocation), `src/llama-memory*.cpp`.
Expected: peak RSS tracks touched KV pages, not n_ctx * head_dim.
Risk: page-fault stalls during decode; interaction with the existing block
      table needs design.

**S3. H2O / StreamingLLM KV eviction policy.** [MISSING -> DONE]
Add an eviction policy to the unified KV cache that drops low-attention or
non-sink tokens under a configurable memory budget. Caps KV growth
independently of sequence length.
Source: D1, D2.
Region: `src/llama-kv-cache.{h,cpp}` (eviction hook off `seq_rm` path at
        `kv-cache.cpp:379`), `src/llama-kv-cells.h`.
Expected: constant-memory unbounded generation for streaming workloads.
Risk: quality degradation on long-context recall; needs an opt-in policy knob
      and the correctness gate must cover a long-context recall fixture.

**S4. KIVI-style asymmetric 2-bit KV.** [MISSING, frontier extension of S1]
Below q4_0: per-channel K, per-token V at 2 bits. Builds on S1's quantized
reader.
Source: E1.
Region: same as S1 plus a new ggml KV type and per-axis quant metadata.
Expected: ~2.6x KV reduction vs f16; composes with S1.
Risk: kernel complexity; needs the dequant fused into attention to avoid
      latency regression (KIVI paper is explicit on this).

**S5. InfiniGen-style speculative KV prefetch (host offload path).** [MISSING]
If a host-offload path is added, predict next-layer KV touches via partial
rehearsal and prefetch only those entries. Most useful when S2 or a host-tier
        KV cache makes residency dynamic.
Source: G2.
Region: new prefetch controller, hooks into `pre_decode()` in
        `tools/server/server-context.cpp`.
Expected: hides host-KV latency, makes offload viable for interactive use.
Risk: only meaningful once there is a host KV tier to prefetch from.

**S6. MoE expert offload to disk (WASTE-style).** [MISSING, design exists]
Implement the design in `docs/moe-disk-offload-study.md`: stripe expert weights
        across NVMe, RVQ out-of-codebook multiply, resident-trunk split.
Source: H1-H4 + `docs/moe-disk-offload-study.md`.
Region: new module under `src/`, integration with the MoE graph build in
        `src/llama-graph*`.
Expected: only relevant if the workload model is MoE; if so, fits multi-T-param
          models in host RAM. If the workload model is dense, SKIP.
Risk: PCIe/disk bandwidth wall on Apple Silicon (the study is explicit that
      this is the hard ceiling); large implementation surface.

**S7. CacheGen-style KV compression for prefix-cache storage.** [MISSING]
Delta + entropy code the radix-cached KV blocks so the prefix cache footprint
shrinks. Orthogonal to S1/S3.
Source: E3.
Region: `tools/server/server-context.cpp` (radix cache store at line 831+).
Expected: 3.5-4.3x smaller prefix cache.
Risk: compression/decompression CPU cost; benefit depends on prefix-cache hit
      patterns in the workload.

**S8. Pre-gated / speculative expert prefetch (if MoE).** [MISSING]
Apply next-layer gate to current hidden states to prefetch the next expert.
Composes with S6.
Source: H1 (Mixtral-Offloading), H3 (Pre-gated MoE).
Region: MoE eval path in `src/llama.cpp`.
Expected: hides expert load latency; only meaningful for MoE workloads.
Risk: mispredicts force synchronous loads; architecture-dependent accuracy.

### Compatibility notes for Phase 5

- **S1, S3, S4** all touch the KV cache and are mutually *incompatible at the
  kernel level* (S1 changes the dtype the reader expects; S3 changes what's
  retained; S4 changes the dtype further). They are good island candidates but
  will not stack trivially; finalize will likely pick one.
- **S2 and S6** touch allocation/dispatch and are largely *disjoint* from S1/S3/S4
  at the file level, so they stack cleanly - the canonical compounding case.
- **S5 depends on S2 or S6** - do not start S5 before one of those lands.
- **S7** is disjoint from all of the above (server radix store).
- **S6 and S8** are MoE-gated: skip both if the workload model is dense.

---

## Sources

- PagedAttention / vLLM - https://arxiv.org/abs/2309.06180
- vAttention - https://arxiv.org/abs/2405.04437
- Sarathi-Serve - https://arxiv.org/abs/2403.02310
- DistServe - https://arxiv.org/abs/2401.09670
- DistAttention / Infinite-LLM - https://arxiv.org/abs/2401.02669
- Mooncake - https://arxiv.org/abs/2407.00079
- H2O - https://arxiv.org/abs/2306.14048
- StreamingLLM - https://arxiv.org/abs/2309.17453
- KIVI - https://arxiv.org/abs/2402.02750
- KVQuant - https://arxiv.org/abs/2401.18079
- CacheGen - https://arxiv.org/abs/2310.07240
- MLA / DeepSeek-V2 - https://arxiv.org/abs/2405.04434
- FlexGen - https://arxiv.org/abs/2303.06865
- InfiniGen - https://arxiv.org/abs/2406.19707
- Mixtral-Offloading - https://arxiv.org/abs/2312.17238
- KTransformers (SOSP 2025) - https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-bla/private/SOSP25-chen.pdf
- Pre-gated MoE - https://arxiv.org/abs/2308.12066
- HOBBIT - https://arxiv.org/abs/2411.01433
- MLX - https://github.com/ml-explore/mlx
- vLLM-MLX - https://arxiv.org/abs/2601.19139
- Core ML StateType - https://apple.github.io/coremltools/docs-guides/source/stateful-models.html
- Open-TQ-Metal - https://arxiv.org/html/2604.16957v1
- HybridServe - https://arxiv.org/abs/2501.01792
- vLLM optimization docs (RECOMPUTE) - https://docs.vllm.ai/en/stable/configuration/optimization/
- TensorRT-LLM KV cache docs - https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html
- llama.cpp mmap discussion - https://github.com/ggml-org/llama.cpp/discussions/638
- llama.cpp mmap-KV RFC - https://github.com/ggml-org/llama.cpp/issues/20757
- tessera MoE disk-offload study (internal) - docs/moe-disk-offload-study.md
