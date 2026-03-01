# CPU Offload Architecture Options

**Date**: Feb 11, 2026
**Problem**: Pure unified cache path gives 1 tok/s at 30% VRAM. fit_params + ggml-cpu gives 6.6 tok/s.
**Goal**: Make the unified cache path fast without depending on fit_params for layer pre-splitting.

---

## Current State

| Path | TG tok/s | How it works |
|------|----------|-------------|
| Default (100% VRAM) | 70.9 | All weights on GPU, graph replay |
| fit_params auto (30% budget) | 6.6 | ngl=19, 14 layers on ggml-cpu (AVX2) |
| Pure unified cache (30% budget) | 1.0 | ngl=99, cache evicts to host, SYCL CPU dispatch |

The 6.5x gap is because ggml-cpu's AVX2 quantized kernels are much faster than our SYCL OpenCL CPU queue + dnnl_sgemm path.

---

## Option A: Route evicted layers to ggml-cpu via backend scheduler

**How**: When unified cache evicts weights to host, use llama.cpp's existing `ggml_backend_sched` to route those operations to the ggml-cpu backend instead of our slow SYCL CPU path.

**The scheduler already exists** — llama.cpp creates one in `llama_init_from_model()` for multi-backend dispatch. It handles tensor copying, synchronization, and per-node backend assignment. We'd just need to make it cache-aware.

**Implementation**:
1. Expose per-tensor eviction status from unified cache
2. When cache has evictions, switch from SYCL-only `graph_compute` to scheduler-based `sched_graph_compute`
3. Scheduler auto-routes host-resident tensors to ggml-cpu

**Estimate**: ~500 lines, ~14 tok/s at 30% budget
**Pros**: Proven infrastructure, handles all op types, clean architecture
**Cons**: Per-node scheduler dispatch overhead (~10-100us), disables graph replay for mixed graphs

---

## Option B: Call ggml-cpu quantized kernels directly

**How**: When a MUL_MAT weight is host-resident, call `ggml_vec_dot_q4_0_q8_0()` (and friends) directly from within the SYCL backend. These are plain C functions that operate on host memory with AVX2 SIMD — no backend framework needed.

**Implementation**:
1. Include `ggml-cpu/quants.h` in SYCL backend
2. Build a dispatch table: `ggml_type -> vec_dot function pointer`
3. In `graph_compute`, check if weight is evicted; if so, call CPU kernel directly
4. Handle SOA->AOS layout conversion (unified cache stores SOA for GPU)

**Estimate**: ~200 lines, ~15-20 tok/s at 30% budget
**Pros**: Minimal code, 6x faster than SYCL CPU path, no scheduler overhead
**Cons**: Only handles MUL_MAT (not other ops), layout conversion cost, fragile coupling to CPU kernel signatures

---

## Option C: Hybrid — scheduler + direct kernels

**How**: Use Option A (scheduler) as the general path, with Option B (direct kernels) as a fast path for the common case (quantized MUL_MAT on Q4_0/Q4_K/Q5_K/Q6_K).

**Estimate**: ~700 lines, ~15-20 tok/s
**This is the recommended approach.**

---

## What about the iGPU?

**Answer: No.** Research findings:

- Arrow Lake iGPU has 64 EUs (below the 80 EU minimum for practical inference)
- TG performance is **identical or slower** than CPU (~10-14 tok/s on both) because batch=1 is memory-latency bound
- PP is 2x faster on iGPU, but 17x slower than dGPU — not worth the complexity
- Multi-device SYCL dispatch breaks graph replay (our 12.5x TG speedup)
- Memory bandwidth is shared with CPU — iGPU would slow down CPU too

**Bottom line**: iGPU is not faster than CPU for token generation, and adds thermal/coordination overhead.

---

## What about sub-layer splitting?

**Answer: Layer-level splitting is near-optimal.** Research findings:

| Strategy | TG Impact | Why |
|----------|-----------|-----|
| Tensor-level (attention on GPU, FFN on CPU) | -10% to -50% | PCIe latency 0.5ms per boundary x 50 tensors = 25ms overhead |
| Row/column tensor parallelism | -10% (TG) | Communication overhead exceeds compute benefit for batch=1 |
| Pipeline parallelism | -8x latency | Must buffer 40+ tokens, destroys interactive latency |
| Speculative decoding (draft model) | +15-25% (low VRAM only) | Only helps when GPU VRAM < 6GB |
| Attention vs FFN split | -10% to -50% | Data dependencies force sequential, PCIe sync cost dominates |
| Expert-level (MoE) | -20x to -100x | Experts are compute-bound FFNs, CPU can't compete |

**Key insight**: PCIe **latency** (0.5ms per transfer), not bandwidth, is the bottleneck. Each device boundary crossing costs 0.5ms regardless of data size. Layer-level splitting minimizes boundary crossings (1 per layer vs 50 per layer for tensor-level).

**PowerInfer** (neuron-level splitting) achieves 11.7x over CPU-only on OPT-175B but requires offline profiling, has 30% misprediction rate, and isn't integrated into llama.cpp.

---

## Recommendation

**Option C (scheduler + direct kernels)** is the path forward:

1. **Phase 1**: Integrate `ggml_backend_sched` as the dispatch path when cache has evictions (~500 lines)
2. **Phase 2**: Add direct `ggml_vec_dot_*` fast path for Q4_0/Q4_K/Q5_K/Q6_K MUL_MAT (~200 lines)
3. **Phase 3**: Benchmark and tune — target 15+ tok/s at 30% VRAM budget

This gives us the unified cache architecture (no fit_params needed) with ggml-cpu-level performance for the common quantized MUL_MAT operations.

**Key architectural requirement**: The unified cache needs to expose per-tensor eviction status, not just a boolean `has_evictions()`. The scheduler needs to know which specific tensors are on host to route them correctly.

---

## Sources

- iGPU research: /tmp/igpu-research.md (15 sources, Intel specs + benchmarks)
- Splitting strategies: /tmp/splitting-research.md (12 sources, academic + industry)
- Cross-backend dispatch: /tmp/cross-backend-research.md (codebase analysis)
