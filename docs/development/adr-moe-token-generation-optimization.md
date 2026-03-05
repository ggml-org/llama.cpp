# ADR: MoE Token Generation Optimization for Strix Halo

**Status**: Accepted
**Date**: 2026-03-05
**Decision makers**: AI-assisted development session
**Supersedes**: N/A

## Context

The Strix Halo fork of llama.cpp has accumulated significant optimizations for dense
model inference on AMD Ryzen AI Max+ 395 hardware: UMA auto-configuration, APEX runtime
scheduling, flash attention tuning, RDNA 3.5 Vulkan pipelines, and iGPU weight
prefetching. However, none of these optimizations specifically target the MoE (Mixture
of Experts) inference pattern.

MoE models like Qwen3-30B-A3B (30B total params, ~3B active per token) are an
increasingly important workload. They are uniquely suited to Strix Halo's unified memory
architecture because:

- **Bandwidth-dominated decode**: Only top-K experts (~3B params) are read per token,
  making token generation purely bandwidth-bound. Strix Halo's ~256 GB/s unified
  LPDDR5X is the primary constraint.
- **High parameter efficiency**: 30B total params fit in ~18 GB (Q4_K_M), well within
  the unified memory pool, while delivering quality competitive with dense 13B-30B
  models.
- **Favorable pp/tg ratio**: Prompt processing activates all experts (compute-heavy)
  while decode activates few (bandwidth-heavy), creating a distinctive optimization
  profile.

Community benchmarks show ~52 tok/s on Qwen3-30B-A3B with current code. Analysis of the
inference pipeline reveals MoE-specific bottlenecks not addressed by existing
optimizations.

## Decision

We introduce a structured plan for MoE token generation optimization through five user
stories (US-MOE-1 through US-MOE-5) and a dedicated MoE benchmark script. The specific
decisions are:

### 1. MoE-specific benchmark suite (US-MOE-5, implemented)

**Created** `scripts/bench-strix-halo-moe.sh` — a benchmark script that captures
MoE-specific performance characteristics:

- Batch size sweep to measure expert coalescing benefits
- PP/TG ratio analysis revealing routing overhead
- Thread count sweep exposing bandwidth contention between CPU and GPU expert reads
- KV cache quantization impact specific to MoE memory patterns
- MMQ vs hipBLAS comparison for per-expert matrix sizes

This establishes the measurement baseline before any optimization work begins.

### 2. Expert weight prefetching (US-MOE-1, planned)

**Decision**: Implement asynchronous prefetching of predicted expert weights using
router frequency tracking.

**Alternatives considered**:
- *Static expert pinning*: Pin the most frequently used experts in cache. Rejected
  because routing patterns vary by input and layer — static pinning wastes cache on
  cold experts.
- *Full expert caching*: Cache all experts in Infinity Cache. Rejected because at
  Q4_K_M, each Qwen3-30B-A3B expert slice is ~1.4 MB, and with 128 experts per layer
  the full set (180 MB/layer) far exceeds the 32 MB MALL.

### 3. Fused gating kernel for RDNA 3.5 (US-MOE-3, planned)

**Decision**: Port and tune the existing `topk-moe.cu` fused kernel for HIP/RDNA 3.5
with wave64-optimal reductions, eliminating 2 kernel launches per layer per token.

**Alternatives considered**:
- *Vulkan compute shader*: A Vulkan fused gating shader already exists
  (`topk_moe.comp`). Rejected as primary path because HIP provides better tuning
  control on RDNA 3.5 and the CUDA/HIP kernel is more mature (403 lines with full
  test coverage).
- *Keep separate ops*: The current 3-op pipeline (softmax → top-K → gather) works
  correctly. Rejected because the per-token overhead of 96 extra kernel launches
  (48 layers × 2 launches) is measurable at the ~52 tok/s operating point.

### 4. Sparse batch coalescing for server workloads (US-MOE-4, planned)

**Decision**: Sort tokens by expert assignment before FFN dispatch in multi-token
batches to maximize weight cache reuse.

**Alternatives considered**:
- *Expert-parallel dispatch*: Dispatch all tokens for each expert in sequence (expert
  outer loop). Rejected because it requires restructuring the graph builder and
  changes the computation order in ways that complicate correctness verification.
- *No change*: Current batched `ggml_mul_mat` already processes all tokens. However,
  it doesn't reorder for cache locality — tokens routed to different experts cause
  thrashing.

### 5. MoE-aware KV cache budget (US-MOE-2, planned)

**Decision**: Adjust default context size recommendation upward for MoE models based
on the sparsity ratio `n_expert_used / n_expert`.

**Rationale**: This is a low-risk quality-of-life improvement. MoE models use the same
KV cache structure as dense models but leave more memory headroom due to sparse expert
activation.

## Consequences

### Positive
- Establishes a principled, measurable approach to MoE optimization
- Benchmark script enables before/after comparison for every change
- User stories are independent — they can be implemented in any order and each
  delivers standalone value
- Fused kernel (US-MOE-3) and prefetching (US-MOE-1) target the two largest
  per-token overhead sources
- All optimizations are MoE-specific and do not regress dense model performance

### Negative
- Expert prefetching (US-MOE-1) adds complexity to the HIP backend with per-layer
  frequency tracking state
- Fused kernel (US-MOE-3) is a maintenance burden — changes to gating logic must be
  reflected in the fused kernel
- Batch coalescing (US-MOE-4) adds a token-reordering step that could introduce
  subtle ordering bugs if not carefully tested

### Risks
- Prefetch hints (`hipMemPrefetchAsync`) may have no effect on some ROCm versions or
  driver configurations — the Infinity Cache is hardware-managed
- Wave64 tuning of the fused kernel may not provide significant gains over the
  existing CUDA kernel compiled via HIP, since hipcc already handles wave64 dispatch
- The 52 tok/s baseline may already be near the theoretical bandwidth limit for
  Qwen3-30B-A3B Q4_K_M (~18 GB read at ~256 GB/s ≈ 14 reads/sec × ~4 tokens/read)

## Files Changed

| File | Change |
|------|--------|
| `docs/development/moe-token-generation-stories.md` | New — 5 user stories for MoE optimization |
| `scripts/bench-strix-halo-moe.sh` | New — MoE-specific benchmark script |
| `docs/development/adr-moe-token-generation-optimization.md` | This ADR |

## Related Work

- `docs/development/apex-runtime-scheduling-stories.md` — APEX user stories (CPU-GPU scheduling)
- `docs/development/uma-bandwidth-aware-splitting.md` — UMA layer splitting
- `ggml/src/ggml-cuda/topk-moe.cu` — Existing fused top-K MoE kernel (CUDA)
- `ggml/src/ggml-vulkan/vulkan-shaders/topk_moe.comp` — Vulkan MoE shader
- `src/models/qwen3moe.cpp` — Qwen3 MoE model implementation
