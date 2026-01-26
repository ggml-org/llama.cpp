# Unified Kernel with Auto-Tuning Architecture

**Created:** 2026-01-26
**Status:** Design Complete - Ready for Implementation

## Executive Summary

This document describes a comprehensive redesign of the SYCL kernel dispatch and memory management systems. The goal is to replace the current 200+ line dispatch logic with 11 kernel variants with a data-driven architecture: one unified kernel whose parameters come from benchmarks, not hardcoded heuristics.

### Key Design Decisions

1. **Single unified kernel** with internal branching based on runtime parameters
2. **XMX-first design** - matrix extensions are primary, scalar is fallback
3. **Progressive background tuning** - no startup delay, improves during inference
4. **Intelligent memory manager** - eviction, prefetch, budget enforcement
5. **Hybrid benchmark pipeline** - ship defaults, allow runtime tuning to override
6. **User-local tuning cache** at `~/.cache/llama.cpp/sycl-tune/`

### Optimization Targets

- **Balance** throughput, latency, and memory efficiency with configurable priorities
- **XMX utilization** for all batch sizes, including batch=1
- **Simplified dispatch** - from 200 lines to ~20 lines

---

## Section 1: Architecture Overview

### 1.1 Current State Problems

| Issue | Impact |
|-------|--------|
| 11 kernel variants | Hard to maintain, test, and reason about |
| Hardcoded batch thresholds | Suboptimal for different hardware |
| 8+ environment variables | Confusing for users |
| No data-driven selection | Leaves performance on the table |
| Unbounded memory cache | OOM with large models |
| Manual layout selection | Fragile eligibility checks |

### 1.2 New Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SYCL Backend Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Request (mul_mat)                                                  │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────┐                                                    │
│  │ Operation Context│  M, N, K, quant_type, batch, device                │
│  │     Builder      │                                                    │
│  └────────┬─────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────┐      ┌──────────────────┐                         │
│  │  Kernel Selector │◄────►│  Tuning Engine   │                         │
│  │                  │      │                  │                         │
│  │ • Unified path   │      │ • Get/set params │                         │
│  │ • oneDNN path    │      │ • Background tune│                         │
│  │ • Fallback chain │      │ • Cache lookup   │                         │
│  └────────┬─────────┘      └────────┬─────────┘                         │
│           │                         │                                    │
│           │         ┌───────────────┘                                    │
│           │         │                                                    │
│           ▼         ▼                                                    │
│  ┌──────────────────────────┐    ┌──────────────────┐                   │
│  │   Memory Manager         │◄──►│  Stability       │                   │
│  │                          │    │  Controller      │                   │
│  │ • ensure_ready(tensor)   │    │                  │                   │
│  │ • Layout conversion      │    │ • Prevent        │                   │
│  │ • Eviction policy        │    │   oscillation    │                   │
│  │ • Prefetch scheduling    │    │ • Convergence    │                   │
│  │ • Budget enforcement     │    │   guarantee      │                   │
│  └────────┬─────────────────┘    └──────────────────┘                   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Unified Matmul Kernel                         │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │                    XMX Core (dpas)                         │  │   │
│  │  │   batch=1-7: Wide N-tiles    batch=8-63: Standard tiles    │  │   │
│  │  │   batch=64+: Persistent threads, large accumulation        │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │   Scalar Fallback (non-XMX devices, edge cases)            │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│       Output                                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Component Summary

| Component | Current State | New Design |
|-----------|---------------|------------|
| **Kernels** | 11 variants (DMMV×2, MMVQ×3, MMQ×3, XMX×2, oneDNN) | 1 unified kernel + oneDNN fallback |
| **Dispatch** | ~200 lines, 8 env vars, manual eligibility checks | ~20 lines, data-driven selection |
| **Tuning** | None (hardcoded thresholds) | Progressive background tuning with persistent cache |
| **Memory** | Simple cache, grows unbounded | Intelligent 3-tier system with eviction, prefetch, budgets |
| **XMX Usage** | Large batch only (>32) | XMX-first for all batch sizes |

---

## Section 2: Unified Adaptive Kernel

### 2.1 Design Principle

Treat XMX as the primary compute engine. The kernel's job is to keep the systolic array busy, not to "decide whether to use XMX."

Intel XMX uses systolic arrays with fixed tile dimensions: **8×16×32 (M×N×K)** for dpas instructions. Even batch=1 can use XMX by accumulating multiple output elements in parallel across the N dimension.

### 2.2 Kernel Structure

```cpp
void unified_matmul_kernel(sycl::nd_item<2> item,
                           const UnifiedKernelArgs& args,
                           sycl::local_accessor<uint8_t, 1>& slm) {

    // Compute tile boundaries with boundary handling
    int tile_m_start = item.get_group(0) * args.tile_m;
    int tile_n_start = item.get_group(1) * args.tile_n;
    int actual_tile_m = min(args.tile_m, (int)(args.M - tile_m_start));
    int actual_tile_n = min(args.tile_n, (int)(args.N - tile_n_start));

    float acc[TILE_M_MAX][TILE_N_MAX] = {0};

    // Main K-loop with prefetching
    for (int k_tile = 0; k_tile < args.K; k_tile += args.tile_k) {
        // Prefetch next K-tile
        if (args.prefetch_depth > 0) {
            prefetch_weights(args, k_tile + args.tile_k * args.prefetch_depth);
        }

        // Load weights to SLM (layout-aware)
        load_weights_to_slm(args, slm, k_tile);
        item.barrier(sycl::access::fence_space::local_space);

        // Compute: XMX or scalar path
        if (args.use_xmx) {
            compute_tile_xmx(sg, slm, acc, args);
        } else {
            compute_tile_scalar(slm, acc, args);
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    // Write output with boundary masking
    write_output_bounded(args, acc, tile_m_start, tile_n_start,
                         actual_tile_m, actual_tile_n);
}
```

### 2.3 Batch Strategies

| Batch | XMX Strategy | Parallelism Source |
|-------|--------------|-------------------|
| 1-7 | Wide N-tiles (16+ output columns per dpas) | N-dimension parallelism |
| 8-63 | Standard tiles, multiple work-groups per row | M×N parallelism |
| 64+ | Persistent threads, large tile accumulation | Full M×N×K parallelism |

### 2.4 Supported Quantization Types

| Type | Optimization Level | XMX Path | Notes |
|------|-------------------|----------|-------|
| Q4_0 | Full | Dequant→INT8→dpas | Most common, optimize heavily |
| Q8_0 | Full | Native INT8 dpas | Native XMX support |
| Q6_K | Full | Dequant→INT8→dpas | K-quant, different patterns |
| Q4_K | Full | Dequant→INT8→dpas | K-quant variant |
| Q5_0, Q5_1 | Scalar | No | Supported but not optimized |
| Q2_K, Q3_K | Scalar | No | Supported but not optimized |
| F16, BF16 | oneDNN preferred | Native dpas | Delegate to oneDNN |
| IQ2_*, IQ3_* | Dequant→oneDNN | No | Exotic types |

### 2.5 Layout-Aware Weight Loading

Single function handles all layouts:

```cpp
void load_weights_to_slm(const UnifiedKernelArgs& args,
                         sycl::local_accessor<uint8_t, 1>& slm,
                         int k_start) {
    switch (args.layout) {
        case LAYOUT_AOS:       load_weights_aos(args, slm, k_start);       break;
        case LAYOUT_SOA:       load_weights_soa(args, slm, k_start);       break;
        case LAYOUT_COALESCED: load_weights_coalesced(args, slm, k_start); break;
        case LAYOUT_XMX_COALESCED: load_weights_xmx_coalesced(args, slm, k_start); break;
    }
}
```

---

## Section 3: Tuning Engine

### 3.1 Three-Tier Parameter Selection

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parameter Selection                           │
├─────────────────────────────────────────────────────────────────┤
│  Tier 1: Runtime Cache (in-memory)                              │
│  - Current session's tuning results                             │
│  - Highest priority, most recent data                           │
│                        ↓ miss                                   │
│  Tier 2: Persistent Cache (~/.cache/llama.cpp/sycl-tune/)       │
│  - Previous sessions' results for this GPU                      │
│  - Loaded at startup, updated on session end                    │
│                        ↓ miss                                   │
│  Tier 3: Compiled Defaults (baked into binary)                  │
│  - Pre-tuned for known GPUs (A770, A750, B580)                  │
│  - Conservative settings, always work                           │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Tuning Data Model

```cpp
struct TuningEntry {
    // Hardware context
    uint64_t gpu_id_hash;           // PCI ID + driver version hash

    // Operation context
    ggml_type quant_type;           // Q4_0, Q8_0, Q6_K, etc.
    int64_t K;                      // Weight matrix K dimension
    int64_t N;                      // Weight matrix N dimension
    int batch_bucket;               // 1, 2-4, 5-8, 9-32, 33-128, 129+

    // Optimal parameters (discovered via tuning)
    int tile_m, tile_n, tile_k;     // XMX tile dimensions
    int prefetch_depth;             // How many K-tiles to prefetch
    int slm_kb;                     // SLM allocation
    bool use_dpas;                  // XMX vs scalar path

    // Performance metrics
    float throughput_tops;          // Measured TOPS
    float memory_bw_gbps;           // Measured bandwidth
    float confidence;               // 0.0-1.0 based on sample count
};
```

### 3.3 Progressive Tuning Strategy

Tuning happens in the background without user-visible latency:

**Phase 1: Observation (first 100 tokens)**
- Record batch sizes, dimensions seen in real workload
- No tuning yet, use defaults
- Build histogram of actual usage patterns

**Phase 2: Targeted Micro-benchmarks (during inference gaps)**
- Identify most frequent (batch, K, N) combinations
- Run 3-5 iteration micro-benchmark during idle time
- Compare 2-3 parameter configurations
- Update cache with winner

**Phase 3: Refinement (ongoing)**
- Periodically re-test as confidence decays
- Detect workload shifts (batch size changes)
- Adapt to thermal/power throttling effects

### 3.4 Micro-benchmark Methodology

A/B comparison protocol with statistical rigor:

```
Config A (current): tile_m=8, tile_n=16, prefetch=2
Config B (candidate): tile_m=8, tile_n=32, prefetch=3

Round 1: Run A(5 iters), Run B(5 iters) → B wins
Round 2: Run B(5 iters), Run A(5 iters) → B wins  (reversed order)
Round 3: Run A(5 iters), Run B(5 iters) → B wins

Result: B wins 3/3 rounds, speedup = 1.054x (5.4%)
        → Adopt Config B, confidence = 0.85
```

**Acceptance Criteria:**
- Win 2/3+ rounds
- At least 5% faster
- Consistent improvement (low variance)

### 3.5 Cold-Start Strategy

When no tuning data exists:

1. **Hardware heuristics**: Derive initial config from GPU specs (EU count, SLM size, bandwidth)
2. **GPU family defaults**: Pre-tuned settings for Arc Battlemage (B580), Arc Alchemist (A770, A750)
3. **Transfer learning**: Check for similar model caches, apply with discounted confidence
4. **First-inference tuning**: Aggressive tuning during first prompt (users expect startup delay)

### 3.6 Multi-GPU Tuning

- Per-GPU profiles even for identical hardware (silicon lottery, thermal position)
- Tensor parallelism context in tuning key (world_size, split_dim)
- Cross-GPU learning: successful configs suggested to similar GPUs with 60% confidence discount
- Serialized tuning: only one GPU tunes at a time to avoid interference

### 3.7 Cache File Format

Location: `~/.cache/llama.cpp/sycl-tune/{gpu_hash}.json`

```json
{
  "version": 1,
  "gpu_id": "8086:56a0:v2.1.0",
  "created": "2026-01-20T10:00:00Z",
  "updated": "2026-01-25T14:30:00Z",
  "entries": [
    {
      "quant": "Q4_0", "K": 4096, "N": 4096, "batch": "1",
      "params": {"tile_m": 8, "tile_n": 16, "tile_k": 32, "prefetch": 2},
      "perf": {"tops": 45.2, "bw_gbps": 380.1},
      "confidence": 0.95,
      "samples": 47
    }
  ]
}
```

---

## Section 4: Intelligent Memory Manager

### 4.1 Tensor Classification

| Heat Level | Examples | Policy |
|------------|----------|--------|
| HOT | Attention Q/K/V, layer norms | Never evict |
| WARM | FFN weights | LRU eviction |
| COLD | MoE experts | Stream on-demand |
| FROZEN | Embeddings | One-time load |

### 4.2 Memory Pressure Levels

| Level | Condition | Adaptation |
|-------|-----------|------------|
| NONE | Budget > HOT + WARM + workspace | Full caching |
| MODERATE | Budget > HOT + workspace | Partial WARM eviction, 90-95% perf |
| HIGH | Budget > HOT (minimal) | Layer streaming, 50-70% perf |
| CRITICAL | Budget < HOT | Full streaming, 20-40% perf |
| IMPOSSIBLE | Budget < single layer | Fail with guidance |

### 4.3 Three-Tier Storage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Device Memory (GPU)                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   HOT Tier      │  │   WARM Tier     │  │   COLD Tier     │ │
│  │   (Pinned)      │  │   (Evictable)   │  │   (Streaming)   │ │
│  │                 │  │                 │  │                 │ │
│  │ • Attn weights  │  │ • FFN weights   │  │ • MoE experts   │ │
│  │ • Layer norms   │  │ • Less-used     │  │ • On-demand     │ │
│  │                 │  │   attention     │  │   upload        │ │
│  │ Never evicted   │  │ LRU eviction    │  │ Immediate evict │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Eviction Policy

Priority score (lower = evict first):

```cpp
float compute_priority(const CacheEntry& entry) {
    float heat_score = heat_multiplier(entry.heat);  // HOT=1000, WARM=100, COLD=10
    float recency = time_since_last_access(entry);
    float frequency = entry.access_count / session_duration;
    float size_penalty = log2(entry.size_bytes / 1MB);

    return heat_score * frequency / (recency + 1.0) - size_penalty;
}
```

### 4.5 Predictive Prefetching

Use model graph structure to prefetch weights 2-3 layers ahead:

```cpp
void on_kernel_start(int current_node) {
    for (int ahead = 2; ahead <= 3; ahead++) {
        int target = current_node + ahead;
        if (target < prefetch_schedule.size()) {
            auto& entry = prefetch_schedule[target];
            if (!is_resident(entry.tensor)) {
                async_upload(entry.tensor);  // Non-blocking
            }
        }
    }
}
```

### 4.6 KV Cache Coordination

KV cache grows linearly with context; dynamic rebalancing as context grows:

```cpp
void on_token_generated(size_t current_kv_size) {
    size_t available_for_weights = total_budget - current_kv_size - fixed_overhead;

    if (current_weight_usage > available_for_weights) {
        size_t to_evict = current_weight_usage - available_for_weights;
        evict_weights(to_evict);
        memory_manager.update_budget(available_for_weights);
    }
}
```

**Memory Comparison with Flash Attention:**

| Context | Standard Attn | Flash Attn | Flash + Sliding (4K) |
|---------|--------------|------------|---------------------|
| 8K tokens | Attn: 8 GB | Attn: 130 MB | Attn: 130 MB, KV: 512 MB |
| 32K tokens | IMPOSSIBLE | KV: 4 GB | KV: 512 MB (constant!) |

### 4.7 Layout Conversion Scheduling

Convert layouts during idle periods without blocking inference:

```cpp
void try_process_conversions(float available_time_ms) {
    while (!pending_queue.empty() && available_time_ms > 0) {
        auto& next = pending_queue.top();

        // Skip if tensor was evicted or already converted
        if (!is_resident(next.tensor)) { pending_queue.pop(); continue; }
        if (get_layout(next.tensor) == next.target) { pending_queue.pop(); continue; }

        // Skip if would exceed time budget
        if (next.estimated_time_ms > available_time_ms) break;

        execute_conversion(next.tensor, next.target_layout);
        available_time_ms -= next.estimated_time_ms;
        pending_queue.pop();
    }
}
```

### 4.8 Tuning ↔ Memory Integration

Bidirectional relationship with convergence guarantee:

- **Memory → Tuning**: Memory pressure affects which configs are viable
- **Tuning → Memory**: Tuning results inform layout preferences for caching
- **Stability Controller**: Prevents oscillation between configs/pressure levels

---

## Section 5: Kernel Dispatch Simplification

### 5.1 New Dispatch Code (~20 lines)

```cpp
void ggml_sycl_mul_mat_unified(ggml_backend_sycl_context* ctx,
                                const ggml_tensor* src0,
                                const ggml_tensor* src1,
                                ggml_tensor* dst) {
    // 1. Build operation context
    OperationContext op_ctx = build_op_context(src0, src1, dst);

    // 2. Get tuned parameters (handles cold-start internally)
    TunedParams params = ctx->tuning_engine->get_params(op_ctx);

    // 3. Ensure weights are ready (handles memory/layout internally)
    ctx->memory_manager->ensure_ready(src0, params.required_layout);

    // 4. Launch unified kernel
    launch_unified_matmul(ctx->queue, op_ctx, params, src0, src1, dst);

    // 5. Record observation for tuning (non-blocking)
    ctx->tuning_engine->record_observation_async(op_ctx, params);
}
```

### 5.2 Comparison

| Aspect | Before | After |
|--------|--------|-------|
| Dispatch lines | ~200 | ~20 |
| Kernel variants | 11 | 1 + oneDNN fallback |
| Environment variables | 8+ | 2 user-facing |
| Selection logic | Hardcoded eligibility checks | Data-driven lookup |
| Debug workflow | "Check 5 env vars, layout, batch, type..." | "Check trace log" |

### 5.3 Traceability

```
[mul_mat] M=1 N=4096 K=4096 type=Q4_0 → tile=8x32x32 xmx=yes layout=SOA conf=87% src=CACHED
[mul_mat] M=32 N=4096 K=4096 type=Q4_0 → tile=32x32x32 xmx=yes layout=SOA conf=92% src=TUNED
```

---

## Section 6: oneDNN Fallback Integration

### 6.1 When to Use oneDNN

- FP16/BF16/FP32 native GEMM
- Large batch where oneDNN GEMM wins (crossover discovered via benchmarking)
- Exotic quant types after dequantization
- Fallback when unified kernel fails

### 6.2 Zero-Copy Integration

```cpp
// Wrap existing SYCL allocations as oneDNN memory (no copy)
dnnl::memory wrap_sycl_buffer(void* sycl_ptr, const dnnl::memory::desc& md,
                               dnnl::engine& engine) {
    return dnnl::sycl_interop::make_memory(
        md, engine, dnnl::sycl_interop::memory_kind::usm, sycl_ptr
    );
}
```

### 6.3 Crossover Discovery

During progressive tuning, benchmark both paths to find where oneDNN wins:

```cpp
// Test batches: 1, 4, 16, 32, 64, 128, 256, 512
// Find crossover point where unified_time / onednn_time >= 1.0
// Store in crossover_cache for future routing decisions
```

---

## Section 7: Edge Case Handling

### 7.1 Validation at Entry

```cpp
ValidationResult validate(const OperationContext& ctx) {
    if (ctx.M * ctx.N > INT32_MAX)
        return {false, DIM_VERY_LARGE, USE_CPU_FALLBACK};
    if (!is_supported_quant_type(ctx.weight_type))
        return {false, TYPE_UNSUPPORTED_QUANT, USE_ONEDNN_FALLBACK};
    if (!ctx.xmx_available && requires_xmx(ctx))
        return {false, HW_NO_XMX, USE_SCALAR_PATH};
    return {true, NONE, NONE};
}
```

### 7.2 Fallback Chain

```
Unified Kernel → Unified (scalar) → oneDNN → CPU
```

### 7.3 Boundary Handling

Non-aligned dimensions handled via boundary masking in kernel (not actual padding):

```cpp
int actual_tile_m = min(args.tile_m, (int)(args.M - tile_m_start));
int actual_tile_n = min(args.tile_n, (int)(args.N - tile_n_start));
if (actual_tile_m <= 0 || actual_tile_n <= 0) return;  // Skip OOB workgroups
```

---

## Section 8: File Structure

### 8.1 New Files

```
ggml/src/ggml-sycl/
├── unified-kernel.cpp          # Unified matmul kernel
├── unified-kernel.hpp          # Kernel parameters, launcher
├── tuning-engine.cpp           # Parameter selection, caching
├── tuning-engine.hpp           # TunedParams, TuningCache
├── onednn-fallback.cpp         # oneDNN integration
├── onednn-fallback.hpp         # Bridge, error handling
├── dispatch.cpp                # Simplified dispatch (~20 lines)
├── cold-start.cpp              # Heuristics, transfer learning
```

### 8.2 Modified Files

```
ggml/src/ggml-sycl/
├── memory-manager.cpp          # Add eviction, prefetch, budgets
├── memory-manager.hpp          # New interfaces
├── common.hpp                  # Remove old kernel enums
├── ggml-sycl.cpp               # Wire up new dispatch
```

### 8.3 Deprecated (Keep for Validation)

```
ggml/src/ggml-sycl/
├── dmmv.cpp                    # Legacy DMMV kernels
├── mmvq.cpp                    # Legacy MMVQ kernels
├── mmq.cpp                     # Legacy MMQ kernels
```

---

## Section 9: Configuration

### 9.1 Environment Variables (Simplified)

| Variable | Purpose | Default |
|----------|---------|---------|
| `GGML_SYCL_MEM_BUDGET` | Device memory budget in bytes | Auto-detect |
| `GGML_SYCL_TUNE_CACHE` | Custom cache location | `~/.cache/llama.cpp/sycl-tune/` |
| `GGML_SYCL_DEBUG_PARAMS` | Debug override (dev only) | Unset |

### 9.2 Cache Structure

```
~/.cache/llama.cpp/sycl-tune/
├── gpu_8086_56a0_pcie_0000_03_00.json    # Per-GPU tuning data
├── gpu_8086_56a0_pcie_0000_04_00.json    # Second GPU
├── multi_gpu_2x_8086_56a0.json           # Multi-GPU TP configs
└── topology.json                          # GPU interconnect info
```

---

## Section 10: Migration Path

1. **Phase 1**: Implement unified kernel alongside existing kernels
2. **Phase 2**: A/B testing with `GGML_SYCL_UNIFIED_KERNEL=1`
3. **Phase 3**: Validation mode (run both, compare results)
4. **Phase 4**: Default to unified, keep legacy as fallback
5. **Phase 5**: Remove legacy kernels after stability period

---

## Section 11: Success Metrics

| Metric | Target |
|--------|--------|
| Dispatch code complexity | -80% (200 → 20 lines) |
| Environment variables | -75% (8 → 2 user-facing) |
| XMX utilization (batch=1) | >50% (currently ~0%) |
| Cold-start overhead | <5s for initial tuning |
| Memory efficiency | <10% waste under budget |
| Fallback rate | <5% of operations |

---

## Section 12: Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Unified kernel slower than specialized | Extensive benchmarking, keep legacy as fallback |
| Tuning oscillation | Stability controller with convergence guarantee |
| Cold-start too slow | Hardware heuristics + transfer learning |
| Memory budget too aggressive | Graceful degradation with user feedback |
| oneDNN version incompatibility | Runtime version detection, feature flags |

---

## Appendix A: Tuning Engine State Machine

```
┌─────────────┐
│  ICE_COLD   │ ─── no cache files exist
└─────────────┘
       │ load_model()
       ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Hardware Heuristics                                │
│ • Derive configs from GPU specs                             │
│ • Check for similar model caches (transfer learning)        │
│ • Load GPU family defaults                                  │
└─────────────────────────────────────────────────────────────┘
       │ first_inference_start()
       ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: First-Inference Tuning                             │
│ • Use heuristic defaults for first layer                    │
│ • Measure actual performance                                │
│ • If underperforming, quick-test alternatives               │
│ • Record observations for all operations                    │
└─────────────────────────────────────────────────────────────┘
       │ prompt_complete()
       ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Cache Building                                     │
│ • Flush observations to cache                               │
│ • Confidence starts low (~0.3)                              │
│ • Background tuning continues                               │
└─────────────────────────────────────────────────────────────┘
       │ confidence > 0.7 for most operations
       ▼
┌─────────────┐
│     HOT     │ ─── high-confidence tuning data available
└─────────────┘
```

---

## Appendix B: Memory Manager State Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│           Memory Budget Impact: Flash vs Standard                │
├──────────────────────────────────────────────────────────────────┤
│  Example: 12GB GPU, Mistral 7B, various context lengths          │
│                                                                   │
│  Context   │ Standard Attn      │ Flash Attn         │ Savings  │
│  ──────────┼────────────────────┼────────────────────┼──────────│
│  2K tokens │ Attn: 512 MB       │ Attn: 130 MB       │ 382 MB   │
│            │ KV: 256 MB         │ KV: 256 MB         │          │
│            │ Weights: 8.2 GB    │ Weights: 8.6 GB    │          │
│  ──────────┼────────────────────┼────────────────────┼──────────│
│  8K tokens │ Attn: 8 GB         │ Attn: 130 MB       │ 7.9 GB   │
│            │ KV: 1 GB           │ KV: 1 GB           │          │
│            │ Weights: STREAMING │ Weights: 7.9 GB    │ ← huge!  │
│  ──────────┼────────────────────┼────────────────────┼──────────│
│  32K tokens│ Attn: 128 GB (!)   │ Attn: 130 MB       │ N/A      │
│            │ IMPOSSIBLE         │ KV: 4 GB           │          │
│            │                    │ Weights: 4.9 GB    │          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Appendix C: Dispatch Complexity Comparison

```
BEFORE (Current System)              AFTER (Unified System)
────────────────────────             ─────────────────────
11 kernel variants                   1 unified kernel
5 layout modes × 3 families          Layouts handled in loader
8+ environment overrides             2 user-facing env vars
~200 lines dispatch logic            ~20 lines dispatch logic
Manual eligibility checks            Data-driven selection
Hardcoded batch thresholds           Learned thresholds
Fragile fallback chains              Single code path

Files touched for new quant type:    Files touched:
• dmmv.cpp (DMMV kernel)             • unified_kernel.cpp
• mmvq.cpp (MMVQ kernel)               (add dequant routine)
• mmq.cpp (MMQ kernel)               • quant_types.hpp
• ggml-sycl.cpp (dispatch)             (register type)
• common.hpp (constants)
```
