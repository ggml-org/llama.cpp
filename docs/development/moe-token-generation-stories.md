# MoE Token Generation Optimization — User Stories

Target model: **Qwen3-30B-A3B** (30B params, ~3B active per token, 128 experts, top-8 routing)
Target hardware: **AMD Strix Halo** (Ryzen AI Max+ 395, RDNA 3.5 iGPU, unified LPDDR5X)

## Context

Mixture-of-Experts models like Qwen3-30B-A3B are a sweet spot for Strix Halo's unified
memory architecture. With only ~3B parameters active per token, the decode bottleneck is
reading expert weights from memory — dominated by bandwidth, not compute. The ~256 GB/s
unified memory bus and 32 MB Infinity Cache create opportunities for MoE-specific
optimizations that differ from dense model tuning.

Current baseline: ~52 tok/s token generation (Q4_K_M, GPU-primary, `-ngl 99`).

---

## US-MOE-1: Expert Weight Prefetching for Sequential Decode

**As a** user running MoE inference on Strix Halo,
**I want** the system to prefetch the next token's likely expert weights into cache
while the current token's experts are being computed,
**so that** memory latency is hidden and token generation throughput increases.

### Rationale
During single-token decode, the router selects top-K experts per layer. The 32 MB Infinity
Cache (MALL) can hold ~2 expert slices for a Q4_K_M Qwen3-30B-A3B layer. By issuing
asynchronous prefetch hints for the next layer's experts based on routing patterns observed
in prior tokens, we can warm the cache and reduce stalls.

### Acceptance Criteria
- Implement a prefetch mechanism in the HIP backend that issues `__builtin_prefetch` or
  `hipMemPrefetchAsync` for expert weight tensors based on router predictions
- Router prediction uses an exponential moving average of expert selection frequency
  per layer (updated every token)
- Prefetch fires for the top-K most-likely experts of the *next* layer while the current
  layer's FFN is executing
- Token generation throughput improves by ≥5% on Qwen3-30B-A3B Q4_K_M (measured via
  `llama-bench -n 128 -ngl 99`)
- No correctness impact — prefetch is a hint, not a semantic change
- Overhead of tracking expert frequencies < 1% of per-token time

---

## US-MOE-2: MoE-Aware KV Cache Memory Budget

**As a** user running large MoE models with long contexts on a memory-constrained system,
**I want** the KV cache allocation to account for MoE models' lower active-parameter footprint
and allocate more memory to KV cache,
**so that** I can run longer contexts without OOM while maintaining token generation speed.

### Rationale
Dense 30B models consume ~18 GB for weights (Q4_K_M) and need substantial KV cache memory.
Qwen3-30B-A3B also uses ~18 GB for all expert weights, but only ~3B active params contribute
to KV cache pressure per token. The memory budget heuristic should account for MoE sparsity
when sizing the default KV cache, allowing larger context windows by default.

### Acceptance Criteria
- When loading an MoE model, the memory estimator uses `n_expert_used / n_expert` as a
  sparsity factor to estimate active memory bandwidth during decode
- The default context size recommendation adjusts upward for MoE models (e.g., suggest
  `-c 16384` instead of `-c 8192` when memory permits)
- `llama-server` with `-np 4 -c 16384` runs stably on Qwen3-30B-A3B Q4_K_M on a 96 GB
  Strix Halo system
- Memory usage reported at startup includes both total weight size and active-parameter
  estimate

---

## US-MOE-3: Fused Expert-Gating + Top-K Kernel for RDNA 3.5

**As a** developer optimizing MoE decode throughput,
**I want** the gating softmax, top-K selection, and expert weight gathering to execute in a
single fused HIP kernel,
**so that** kernel launch overhead and redundant memory traffic are minimized during
token-by-token decode.

### Rationale
Qwen3-30B-A3B uses softmax gating with top-8 selection from 128 experts. The current
implementation runs separate ops for softmax, top-K, and gather. On RDNA 3.5 with wave64
wavefronts, a single kernel can compute gating logits, find top-8, and produce the
weighted expert mask — eliminating 2 kernel launches per layer per token (48 layers ×
2 = 96 saved launches per token).

The existing `topk-moe.cu` fused kernel supports this for CUDA. This story ports and
tunes it for the HIP/RDNA 3.5 path with wave64-optimal reductions.

### Acceptance Criteria
- Port `topk-moe.cu` to use wave64 intrinsics (`__shfl_xor` with 64-wide mask) where
  beneficial on RDNA 3.5
- Fused kernel handles the Qwen3 MoE configuration: 128 experts, top-8, softmax gating,
  with weight normalization
- Kernel dispatch uses a single launch per layer instead of 3 separate ops
- Per-token latency for the gating+topk stage decreases by ≥30%
- No accuracy regression (output matches unfused path within FP32 tolerance)
- Benchmark with `scripts/bench-strix-halo-moe.sh` shows measurable tg improvement

---

## US-MOE-4: Sparse Expert Batch Coalescing for Decode

**As a** user running MoE models in `llama-server` with multiple concurrent sessions,
**I want** the scheduler to coalesce expert computations across tokens in the same batch
that route to the same expert,
**so that** expert weight reads are amortized across tokens and GPU utilization improves.

### Rationale
When `llama-server` runs with `-np 4`, multiple decode tokens are batched together. If
tokens 0 and 2 both route to expert #37, their FFN computations for that expert can share
a single weight read. This is especially impactful on bandwidth-bound Strix Halo where
reducing total bytes read directly improves throughput.

The existing `build_moe_ffn` implementation already batches via `ggml_mul_mat` on the
expert tensors, but the token dimension is processed uniformly. Explicit coalescing
reorders tokens by expert assignment to maximize cache reuse.

### Acceptance Criteria
- When batch size > 1, tokens are sorted by their expert assignment before FFN dispatch
- Expert weight tensors are read once per unique expert in the batch (not once per token)
- Multi-slot server throughput (`-np 4`) improves by ≥10% on Qwen3-30B-A3B compared to
  uncoalesced baseline
- Single-token decode (batch=1) has zero overhead from coalescing logic
- Correctness: output tokens are reordered back to original sequence order after FFN

---

## US-MOE-5: MoE-Specific Benchmark Suite for Strix Halo

**As a** developer or benchmarker,
**I want** a dedicated benchmark script that tests MoE-specific performance characteristics
on Strix Halo,
**so that** I can measure the impact of MoE optimizations separately from dense model tuning.

### Rationale
The existing `bench-strix-halo.sh` sweeps ngl, threads, and KV cache types but doesn't
capture MoE-specific metrics: expert routing overhead, sparse vs dense throughput
comparison, expert cache hit rates, or multi-slot decode coalescing benefits.

### Acceptance Criteria
- New script `scripts/bench-strix-halo-moe.sh` runs benchmarks tailored for MoE models
- Tests include:
  - Token generation at varying batch sizes (1, 2, 4, 8) to measure coalescing
  - Prompt processing vs token generation ratio comparison
  - Thread count sweep showing bandwidth contention
  - KV cache quantization impact on MoE models specifically
  - Side-by-side comparison of FORCE_MMQ vs hipBLAS for MoE expert matmuls
- Script outputs CSV results compatible with the existing benchmark format
- Results directory includes model metadata (n_expert, n_expert_used, quant type)

---

## Priority Order

1. **US-MOE-5** — Benchmark suite (establishes baseline, measures all other improvements)
2. **US-MOE-3** — Fused gating kernel (low-hanging fruit: reduces per-layer dispatch overhead)
3. **US-MOE-1** — Expert prefetching (addresses the primary bottleneck: memory latency)
4. **US-MOE-4** — Batch coalescing (server workload optimization)
5. **US-MOE-2** — KV cache budget (quality-of-life improvement for long-context usage)

## Expected Cumulative Impact

| Story | Estimated tg Improvement | Compound tok/s |
|-------|--------------------------|----------------|
| Baseline | — | ~52 t/s |
| US-MOE-3 (fused kernel) | +5-8% | ~55-56 t/s |
| US-MOE-1 (prefetch) | +5-10% | ~58-62 t/s |
| US-MOE-4 (coalescing, np=4) | +10-15% server throughput | ~64-71 t/s aggregate |

## Non-Goals (for now)

- Expert pruning or dynamic expert count reduction (changes model semantics)
- NPU offload of gating computation (XDNA 2 support not yet in llama.cpp)
- Cross-layer expert caching (requires model-specific analysis of routing patterns)
- Quantization-aware expert routing (would require retraining)
