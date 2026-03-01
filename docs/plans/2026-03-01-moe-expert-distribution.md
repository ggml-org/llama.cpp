# MoE Expert Parallelism: Multi-Device Concurrent Expert Dispatch

**Date**: 2026-03-01 (revised)
**Status**: Research complete, architecture redesigned
**Beads**: llama.cpp-i5o2 (P0)
**Branch**: feature/sycl-coalescing

## Problem Statement

GPT-OSS 120B MoE model achieves only ~2.5 tok/s PP and ~3.1 tok/s TG on our multi-device setup (B580 + B50 + CPU), while dense Mistral 7B achieves ~1300 tok/s PP on the same B580. The PP path crashes with SIGSEGV when VRAM is exhausted during expert prestaging.

### Root Cause: Synchronization-Bound, Not Bandwidth-Bound

Code investigation reveals **the PP bottleneck is excessive GPU synchronization**, not compute or memory bandwidth:

1. **180 full GPU drains per PP pass**: `update_moe_ptr_table()` calls `stream->wait()` at every MUL_MAT_ID entry (ggml-sycl.cpp:18852)
2. **Serial H2D staging**: `prestage_routed_experts()` does `q.memcpy().wait()` per cache miss (expert-cache.cpp:294). Its `queue_ptr` parameter for async staging is **unused** (unified-cache.cpp:6154: `(void) queue_ptr;`)
3. **Fused kernel path dead**: `ggml_sycl_mul_mat_id_fused()` requires `sycl::usm::alloc::device` but mmap'd weights return `unknown` — always rejected (ggml-sycl.cpp:22899)
4. **ExpertPrefetcher not wired to PP**: Async prefetch infrastructure exists but only activates for TG hybrid path (ggml-sycl.cpp:25020-25046), never for PP
5. **Zero compute-DMA overlap in PP**: Expert load → wait → GEMM → wait → next expert, fully serialized

### Why the Dense PP Path Is Fast (~1300 tok/s)

The Mistral 7B PP path achieves ~1300 tok/s through five compounding factors:
1. **oneDNN XMX GEMM**: JIT-compiled FP16 matmul on Intel Matrix Extensions (~6x faster than ESIMD)
2. **Per-op SYCL graph replay**: When model fits in VRAM (`g_model_exceeds_vram=false`), the per-op dispatch sequence is recorded once and replayed with single `ext_oneapi_graph()` call (~5% contribution for compute-bound PP)
3. **All weights in VRAM**: Stable device pointers, zero DMA overhead
4. **Pre-cached oneDNN primitives**: Warmup pass JIT-compiles before recording
5. **Pre-allocated scratch**: Dequant buffers reserved during model init

**Note on graph replay vs persistent TG micro-graph**: These are separate mechanisms.
- Per-op graph replay (line 35928): records hundreds of individual op dispatches, replays as one `ext_oneapi_graph()` call. Disabled when `g_model_exceeds_vram=true` (stale pointers).
- Persistent TG micro-graph (line 9577): records a single monolithic kernel submit, replays it. Only for TG (batch=1). Was designed for multi-GPU support.

The 120B MoE breaks **all five dense-path factors**: model exceeds VRAM (per-op graph replay disabled at line 35484), expert weights stream on demand from host memory, MUL_MAT_ID uses expert cache lookup with sync stalls, and sparse activation creates variable-size GEMMs.

---

## Hardware Inventory

| Device | Memory | Bandwidth | PCIe | Role |
|--------|--------|-----------|------|------|
| Arc B580 | 12 GB VRAM | 170 GB/s | x16 Gen4 (25 GB/s) | Primary GPU |
| Arc Pro B50 | 4 GB VRAM | 110 GB/s | x8 Gen4 (12 GB/s) | Secondary GPU |
| CPU (DDR5) | 256 GB RAM | 50 GB/s | N/A | Host compute + storage |

### Critical Insight: Activation Shipping vs Weight Shipping

*Example sizes for GPT-OSS 120B MXFP4; actual sizes computed from `ggml_tensor` metadata at runtime:*

| Transfer | Size (example) | PCIe x16 | PCIe x8 |
|----------|---------------|----------|---------|
| Expert weight | `expert_bytes` (e.g. 12.6 MB) | 0.50 ms | 1.05 ms |
| Activation (batch=1) | `activation_bytes` (e.g. 11.2 KB) | 0.0004 ms | 0.001 ms |
| Result (batch=1) | `activation_bytes` (e.g. 11.2 KB) | 0.0004 ms | 0.001 ms |

**Activation shipping is ~1000x cheaper than weight shipping** for typical MoE models. (Fiddler, ICLR 2025)

Never transfer expert weights at inference time. Send the tiny activation to the device that already has the weights.

**GPT-OSS 120B Example** (verified from GGUF metadata):
- `embedding_length` = 2880, `expert_feed_forward_length` = 2880
- `block_count` = 36 layers, `expert_count` = 128, `expert_used_count` = 4 (top-4 routing)
- Expert size (MXFP4): 12.6 MB per expert, activation = 11.2 KB

**Generic model support**: All dimensions are queried at runtime, never hardcoded. Other MoE models (Qwen-MoE, Kimi, Mixtral, DeepSeek-V3, DBRX) have different expert counts, dimensions, and routing strategies. The implementation reads these from `ggml_tensor` shapes and model metadata.

---

## Architecture: Unified Tiered Weight Management with Expert Parallelism

### Core Architectural Principle: Unified Cache Owns ALL Weights

**All weights — dense and expert — flow through the unified cache as the single source of truth.** The current `ExpertCache` (expert-cache.cpp) is a separate parallel system that bypasses the unified cache entirely; it should be subsumed into the unified cache. This eliminates dual bookkeeping and enables consistent tiered storage.

### Runtime Parameters (Never Hardcoded)

All model and hardware parameters are **queried at runtime**. The implementation must work for any MoE model (GPT-OSS, Qwen-MoE, Kimi, Mixtral, DeepSeek-V3, DBRX, etc.) and any hardware configuration.

| Parameter | Source | API | Example (GPT-OSS 120B) |
|-----------|--------|-----|----------------------|
| `n_expert` | Model metadata | `src0->ne[2]` in MUL_MAT_ID | 128 |
| `n_expert_used` (top-K) | Model metadata | Router output tensor dims | 4 |
| `n_layer` | Model metadata | `llama_model_n_layer()` | 36 |
| `hidden_dim` (K) | Tensor shape | `src0->ne[0]` (weight columns) | 2880 |
| `ffn_dim` (N) | Tensor shape | `src0->ne[1]` (weight rows) | 2880 |
| Expert byte size | Computed | `ggml_nbytes(expert_tensor)` per expert slice | 12.6 MB |
| Activation byte size | Computed | `hidden_dim * ggml_type_size(src1->type)` | 11.2 KB |
| VRAM per device | Hardware | `dpct::dev_mgr::get_memory_info()` | 12 GB (B580) |
| System RAM | Hardware | `get_total_system_memory_bytes()` (sysconf) | 256 GB |
| Device count | Hardware | `dpct::dev_mgr::instance().device_count()` | 3 |
| PCIe bandwidth | Measured or estimated | Optional: timed H2D memcpy at init | ~25 GB/s (x16) |

**Derived at runtime** (no constants):
- `expert_slots_per_device = (vram_available - dense_weight_bytes - kv_bytes - scratch) / expert_byte_size`
- `total_gpu_expert_slots = sum(expert_slots_per_device[i] for i in gpu_devices)`
- `layers_cached = total_gpu_expert_slots / n_expert`
- `cache_sufficient = (total_gpu_expert_slots >= n_expert)` — if true, entire current layer fits in VRAM

**Model diversity examples** (all must work without code changes):

| Model | Experts | Top-K | Hidden | FFN | Expert Size (Q4_0) |
|-------|---------|-------|--------|-----|-------------------|
| GPT-OSS 120B | 128 | 4 | 2880 | 2880 | ~12.6 MB (MXFP4) |
| Qwen2.5-MoE-A2.7B | 64 | 4 | 2048 | 1408 | ~5.2 MB |
| Mixtral 8x7B | 8 | 2 | 4096 | 14336 | ~106 MB |
| DeepSeek-V3 | 256 | 8 | 7168 | 2048 | ~26.5 MB |
| DBRX | 16 | 4 | 6144 | 10752 | ~119 MB |

Note how expert sizes vary by **20x** (5 MB to 119 MB). Slot calculations, device partitioning, and prefetch strategy must all adapt dynamically.

### Tiered Weight Storage

```
Tier 0: HOST-PINNED (canonical store)
  ┌──────────────────────────────────────────────────────┐
  │  sycl::malloc_host via pinned_chunk_pool (8GB chunks) │
  │  Budget: 90% of system RAM (~230 GB on 256 GB system) │
  │  Layout: AOS (CPU-optimal, native ggml block format)  │
  │  Contents: ALL weights — dense + expert               │
  │  CPU compute reads directly from this tier             │
  └──────────────────────────────────────────────────────┘
          ↓ H2D copy (SOA-converted)
Tier 1: GPU VRAM (hot cache)
  ┌──────────────────────────────────────────────────────┐
  │  sycl::malloc_device (B580 12GB + B50 4GB)            │
  │  Layout: SOA (GPU-optimal, coalesced memory access)   │
  │  Contents: dense weights + predicted hot experts      │
  │  GPU compute reads from this tier                     │
  │  Eviction: LFU+staleness, dense pinned, experts by   │
  │  popularity rank                                       │
  └──────────────────────────────────────────────────────┘

Fallback: HOST_MMAP (when model exceeds system RAM)
  ┌──────────────────────────────────────────────────────┐
  │  mmap'd GGUF file (zero-copy alias, no pinning)       │
  │  Used only when host-pinned budget exhausted           │
  │  OS page cache manages residency                      │
  └──────────────────────────────────────────────────────┘
```

### Why AOS for Host, SOA for Device

**Research confirmed** (CPU layout agent, KTransformers SOSP'25):

| Property | CPU (AOS) | GPU (SOA) |
|----------|-----------|-----------|
| Access pattern | 1 thread processes block sequentially: d → qs → next block | 32+ threads read same field from different blocks simultaneously |
| Cache line efficiency | scale `d` and quants `qs` in same 64B cache line (18B Q4_0 block) | All quants contiguous for coalesced 128B memory transactions |
| vec_dot match | `_mm256_dpbssd_epi32` operates on contiguous 32-byte blocks | MMVQ kernel reads SOA quant arrays with `sub_group::load` |
| Repack for GEMM | Optional `block_q4_0x8` interleaving for multi-column GEMM (batch>1) | SOA is already the optimal layout for all batch sizes |

Storing the same weight twice (AOS in host-pinned, SOA in VRAM) is standard practice — confirmed across MoE-Infinity, KTransformers, HOBBIT, Pre-gated MoE, and ZeRO-Inference. The memory cost is justified by each format serving its processor's access pattern optimally.

### Why This Works (Industry Validation)

| System | Architecture | Our Equivalent |
|--------|-------------|----------------|
| MoE-Infinity (2024) | All experts in host DRAM, GPU expert cache, pinned memory + DMA | Unified cache host_cache → device cache |
| KTransformers (SOSP'25) | Expert weights in CPU DRAM, dense on GPU, AMX kernels for CPU | Host-pinned AOS + AVX-VNNI, dense in VRAM SOA |
| Fiddler (ICLR'25) | CPU-resident experts, activation shipping, static popularity | Host-pinned experts, activation shipping, prediction-based GPU fill |
| Pre-gated MoE (ISCA'24) | All MoE params in CPU DRAM, GPU for dense, pre-gate prediction | Unified cache hosts all, VRAM caches hot, next-layer prediction |
| ZeRO-Inference | Entire model pinned in CPU DRAM, GPU holds 1-2 layers | Unified cache pins all, VRAM holds all dense + hot experts |
| FlexGen | 3-tier (GPU/CPU/disk), LP solver for placement | 3-tier (VRAM/pinned/mmap), popularity-based placement |

### Existing Infrastructure (Already Built)

The unified cache's `host_cache` class already implements most of this architecture:

| Feature | Status | Location |
|---------|--------|----------|
| Pinned allocation via `sycl::malloc_host` | DONE | `pinned_chunk_pool` (pinned-pool.hpp:25), 8GB chunks |
| 90% system RAM budget | DONE | unified-cache.cpp:37 (`g_unified_cache_host_budget_pct`) |
| `expert_id` tracking in host entries | DONE | unified-cache.hpp:342 (`host_cache_entry.expert_id`) |
| AOS and SOA layout support on host | DONE | unified-cache.hpp:343 (`host_cache_entry.layout`) |
| Dual storage (host-pinned → DMA → device) | DONE for dense | ggml-sycl.cpp:5779-5895 |
| `MOE_EXPERT` cache entry type | DONE | unified-cache.hpp enum |
| Fallback chain: pinned → unpinned → mmap alias | DONE | unified-cache.cpp:1526-1577 |
| Expert eviction priority (lower than dense) | DONE | unified-cache.cpp:3968 (2x score multiplier for dense) |

**The gap**: Expert weights bypass all of this. The `ExpertCache` (expert-cache.cpp) is a completely separate system that stores mmap pointers and does its own H2D copies. CPU dispatch reads raw mmap. None of this flows through the unified cache.

### Practical Limits: Pinned Memory on Linux

| Parameter | Value | Notes |
|-----------|-------|-------|
| `ulimit -l` on this system | unlimited | Already configured; `pinned_chunk_pool` handles failures gracefully |
| Safe pinned budget (256GB system) | ~230 GB | 90% default. Leaves ~26GB for OS, KV cache, activations |
| Level Zero per-alloc limit | ~11 GB | `pinned_chunk_pool` uses 8GB chunks (pinned-pool.hpp:18) |
| Level Zero allocation count limit | ~1005 | 8GB chunks → 29 allocs for 230GB budget (well under limit) |
| Pinned vs mmap CPU bandwidth | **Same** | Both in DRAM. Pinned advantage: no page faults, ~3x GPU DMA speed |
| Fallback on alloc failure | Automatic | `ensure_cached_alloc()`: pinned → unpinned malloc → mmap alias (line 1526-1577) |

### Key Principles (from research)

1. **Expert parallelism** (Fiddler, KTransformers, DeepSpeed-MoE): Experts within an MoE layer have **zero data dependencies between them**. All `top_k` active experts can compute simultaneously on different devices.
2. **Activation shipping** (Fiddler ICLR'25): Send tiny `activation_bytes` to device with weights, not large `expert_bytes` to GPU. CPU computes locally using host-pinned AOS weights + AVX-VNNI.
3. **Static popularity-based placement** (Fiddler, Aurora): Profile expert activation frequency, fill GPU VRAM with hottest experts at model load time. Dense weights always resident.
4. **Expert deferral** (KTransformers SOSP'25): Defer slow CPU experts to overlap with next layer's GPU attention. Residual connections make this accuracy-safe (<0.5% degradation).
5. **Async prefetching** (MoE-Infinity, PreScope): While computing current layer, predict and prefetch next layer's experts.

### Execution Model

```
Per MoE Layer (n_expert experts, top-K routing — values from model metadata):

  1. Router (GPU0) selects K experts per token

  2. Partition by where weights reside (placement table query):
     ┌─ GPU0-resident → dispatch oneDNN GEMM on GPU0 queue
     ├─ GPU1-resident → ship 11.2KB activation to GPU1, dispatch GEMM on GPU1 queue
     └─ CPU-resident  → ship 11.2KB activation to CPU, dispatch on CPU thread pool

  3. All three compute IN PARALLEL (zero cross-device dependencies)

  4. Gather results (tiny: K × activation_bytes total)
     ├─ GPU0 results: already in place
     ├─ GPU1 results: D2H (activation_bytes, negligible)
     └─ CPU results:  H2D memcpy (activation_bytes, negligible)

  5. Weighted sum → continue to next layer
```

### Memory Layout (Computed at Runtime; GPT-OSS 120B MXFP4 Example)

```
Expert size: 12.6 MB each (24.9M params × MXFP4 4.25 bits/param)
Total experts: 128 per layer × 36 layers = 4,608 experts total
Total expert weight: ~56.7 GB (96% of model)
Dense weight (attention + norms + embeddings): ~2.3 GB

HOST-PINNED (Tier 0, canonical store, AOS layout):
  Dense weights:         ~2.3 GB (copied from mmap at load, AOS for CPU fallback)
  ALL expert weights:    ~56.7 GB (copied from mmap at load, AOS for CPU vec_dot)
  Total pinned:          ~59 GB (within 90% budget of 256 GB system = ~230 GB)
  Fallback:              mmap alias for any weight that doesn't fit

B580 VRAM (Tier 1, hot cache, SOA layout):
  Dense layer weights:   ~2.3 GB (always resident, SOA, never evicted)
  KV cache:              ~0.5 GB (with -c 4096)
  Hot expert pool:       ~7.5 GB → ~595 expert slots (most popular by frequency, SOA)
  Compute scratch:       ~0.5 GB (oneDNN + activation staging)
  Headroom:              ~1.2 GB

B50 VRAM (Tier 1, warm cache, SOA layout):
  Warm expert pool:      ~3.0 GB → ~238 expert slots (moderately popular, SOA)
  Compute scratch:       ~0.5 GB
  Headroom:              ~0.5 GB

Total GPU expert slots:  ~833 (595 + 238)
Total experts:           4,608 (128 × 36 layers)
Active per token:        4 (top-4 routing) × 1 layer = 4 lookups
GPU cache hit rate:      ~100% for TG (833 >> 128 experts/layer, entire hot layer fits)
```

**Weight lifecycle**: GGUF file → mmap → `host_cache::ensure_cached_alloc()` allocates pinned space → caller fills via callback (`needs_fill=true`) → `unified_cache::ensure_cached_layout()` copies to device SOA (for hot experts/dense). CPU dispatch reads from pinned AOS. GPU dispatch reads from device SOA. Both copies managed by the unified cache.

**Key insight**: The ratio `total_gpu_expert_slots / n_expert` determines how many full layers of experts fit in GPU VRAM simultaneously. For GPT-OSS 120B (12.6 MB experts, 833 slots, 128 experts/layer): ~6.5 layers. For Mixtral 8x7B (106 MB experts, ~70 slots, 8 experts/layer): ~8.7 layers. Since TG processes one layer at a time, the GPU can prefetch the next layer's experts while computing the current layer — achieving near-100% cache hit rate. Models with very large experts (e.g. DBRX, 119 MB/expert) may have fewer cached layers, making prefetch prediction more critical.

### Per-MoE-Layer Timeline (TG, batch=1) — GPT-OSS 120B Example

```
Experts per token: 4 (top-4, from model metadata)
Expert GEMM: M=1, K=2880, N=2880 (MXFP4, 12.6 MB weight)
(Other models will have different K, N, expert_bytes — computed at runtime)

TIME →  0ms      0.5ms    1ms      1.5ms    2ms
GPU0:   [exp0 0.3ms][exp1 0.3ms]                    ← 2 hot experts, cached in VRAM
GPU1:   [exp2 0.5ms]                                 ← 1 warm expert, cached in VRAM
CPU:    [exp3    1.5ms       ]                        ← 1 cold expert, host-pinned AOS

Gather: ................[merge 0.01ms]                ← tiny results (top_k × activation_bytes)
Total:  max(0.6, 0.5, 1.5) = ~1.5 ms per MoE layer  ← CPU-bound
With deferral: defer CPU expert → 0.6 ms visible     ← overlapped with next attention
```

Note: When total_gpu_expert_slots >> n_expert (e.g. 833 >> 128 for GPT-OSS on B580+B50),
ALL experts for the current layer fit in GPU VRAM. CPU dispatch only activates for cache
misses during layer transitions, unusual routing patterns, or models with very large experts.

### Per-MoE-Layer Timeline (PP, batch=64) — GPT-OSS 120B Example

```
For PP, batch=64 means each expert processes up to 64 tokens (top_k routing).
Expert matmul: M=batch, K=hidden_dim, N=ffn_dim (dimensions from model metadata)
GPT-OSS example: M=64, K=2880, N=2880 (MXFP4, 12.6 MB weight)

With top_k routing and 64 tokens, up to ~(batch × top_k / n_expert × n_expert) unique experts may activate.
GPT-OSS example: top-4, 128 experts → up to ~57 unique experts per layer.
With 833 GPU expert slots, most/all are already cached.

GPU0:   [exp0..exp30 batched GEMMs, oneDNN XMX]  ~15 ms total  (30 experts)
GPU1:   [exp31..exp45 batched GEMMs]              ~20 ms total  (15 experts)
CPU:    [exp46..exp57 host-pinned, AVX-VNNI]      ~30 ms total  (12 experts, only on cache miss)

  All parallel → max(15, 20, 30) = ~30 ms/layer
  36 layers × 30 ms + dense overhead = ~1.1 sec
  PP64: ~58 tok/s (vs 2.5 tok/s current)

If ALL experts fit in GPU VRAM (likely for most layers):
  GPU0:  [all 57 experts batched]  ~30 ms total
  36 layers × 30 ms + dense overhead = ~1.1 sec
  PP64: ~58 tok/s (GPU-only, no CPU needed)
```

---

## Implementation Plan

### Phase 1: Remove Synchronization Stalls (Quick Win)

**Goal**: Eliminate the 180 GPU drains per PP pass. No architecture change — just make existing code async.
**Effort**: 1-2 days
**Expected improvement**: 2.5 → ~8-12 tok/s PP (removing stalls alone)

#### Changes

| File | Change | Lines |
|------|--------|-------|
| `unified-cache.cpp` | Make `prestage_routed_experts()` use its `queue_ptr` for async H2D (remove `(void) queue_ptr;`) | ~20 |
| `ggml-sycl.cpp` | Remove unconditional `stream->wait()` in `update_moe_ptr_table()` — use `depends_on(event)` | ~15 |
| `expert-cache.cpp` | Add `ensure_cached_async()` that returns `sycl::event` instead of doing `.wait()` | ~30 |
| `ggml-sycl.cpp` | Wire `ExpertPrefetcher` to PP path (currently TG-only) | ~20 |

#### Detail: Async `prestage_routed_experts()`

```cpp
// BEFORE (current, serial):
for (auto & expert : experts_to_stage) {
    cache->ensure_cached(key, ptr, size, MOE_EXPERT, ...);  // contains .wait()
}

// AFTER (async, pipelined):
std::vector<sycl::event> staging_events;
for (auto & expert : experts_to_stage) {
    staging_events.push_back(
        cache->ensure_cached_async(key, ptr, size, MOE_EXPERT, queue_ptr));
}
// Single barrier for all staging, not per-expert.
// NOTE: Do NOT use ext_oneapi_submit_barrier() — known to corrupt Level Zero event state
// (see common.hpp:172, ggml-sycl.cpp:4594). Use in-order queue marker or depends_on instead:
for (auto & e : staging_events) {
    queue_ptr->submit([&](sycl::handler & h) { h.depends_on(e); h.single_task([]{}); });
}
```

### Phase 2: Expert Parallelism with Activation Shipping (Core Architecture)

**Goal**: All three devices compute experts in parallel. Zero idle time. No OOM crash.
**Effort**: 5-7 days
**Expected improvement**: ~8-12 → ~40-60 tok/s PP, ~3 → ~15-30 tok/s TG

#### 2a: Unified Cache Expert Registration (Replace ExpertCache)

At model load time, route ALL expert weights through the unified cache:

1. **Host-pinned AOS**: `host_cache::ensure_cached_alloc()` allocates pinned space, returns `needs_fill=true` to the caller, who copies data from mmap to pinned host memory via a fill callback in AOS format. This is the canonical store for CPU dispatch.
2. **Device SOA (hot experts)**: `unified_cache::ensure_cached_layout()` copies the most popular experts from pinned AOS to device VRAM in SOA format. Slot count per device = `(vram_available - dense - kv - scratch) / expert_bytes` — computed at runtime from hardware query + model metadata. Example (GPT-OSS 120B MXFP4 on B580+B50): ~595 + ~238 = ~833 slots.
3. **Placement table**: Maps `(layer_id, expert_id)` → `{device_id, device_ptr_soa, host_ptr_aos, popularity_rank}`.

```cpp
// New: unified expert placement (replaces ExpertCache)
// Stored in unified_cache, queried at dispatch time.
// Generic: works for any MoE model (Mixtral, Qwen, DeepSeek, GPT-OSS, etc.)
struct ExpertPlacement {
    int    device_id;       // 0..n_gpu-1 for GPU devices, -1=CPU-only (runtime device count)
    void * device_ptr;      // SOA device pointer (nullptr if CPU-only)
    void * host_ptr;        // AOS host-pinned pointer (always valid)
    size_t weight_bytes;    // per-expert size in bytes (varies by model & quantization)
    int    popularity_rank; // for GPU eviction priority
};

// Placement table: maps (layer_id, expert_id) → ExpertPlacement
// Built at model load from ggml_tensor metadata:
//   - n_expert = src0->ne[2] (from first MUL_MAT_ID tensor)
//   - expert_bytes = ggml_row_size(src0->type, src0->ne[0]) * src0->ne[1]
//   - n_layer = model.n_layer (or count of MoE layers)
// Device assignment based on: available VRAM / expert_bytes = slots_per_device
```

The separate `ExpertCache` class (expert-cache.cpp) is eliminated. Its responsibilities are absorbed:
- `register_expert()` → `host_cache::ensure_cached_alloc()` with `MOE_EXPERT` type
- `ensure_cached()` → `unified_cache::ensure_cached_layout()` with SOA conversion
- `lookup()` → placement table query (O(1) hash lookup)
- LFU eviction → unified cache already has LFU+staleness scoring (line 3968)

#### 2b: Per-Expert Device Routing in MUL_MAT_ID

Replace the host-side serial expert loop with parallel per-device dispatch:

```cpp
// ggml-sycl.cpp — new dispatch inside ggml_sycl_mul_mat_id()
// Generic: works for any n_expert, top_k, n_devices — no hardcoded model or hardware constants
void moe_parallel_dispatch(ctx, dst, src0, src1, ids, n_as, top_k) {
    // 1. Read router decisions (top_k from model metadata, not hardcoded)
    auto active_experts = get_active_experts(ids, n_as, top_k);

    // 2. Partition by device — dynamic number of GPU devices + CPU
    const int n_gpu = dpct::dev_mgr::instance().device_count();
    std::vector<std::vector<ExpertWork>> gpu_work(n_gpu);
    std::vector<ExpertWork> cpu_work;
    for (auto & [expert_id, token_indices] : active_experts) {
        auto placement = unified_cache->get_expert_placement(layer_id, expert_id);
        if (placement.device_id >= 0 && placement.device_id < n_gpu) {
            gpu_work[placement.device_id].push_back({expert_id, token_indices, placement});
        } else {
            cpu_work.push_back({expert_id, token_indices, placement});
        }
    }

    // 3. Ship activations + dispatch ALL devices in parallel
    // activation_bytes = src1->ne[0] * ggml_type_size(src1->type) — not hardcoded
    std::vector<sycl::event> gpu_events;
    for (int d = 0; d < n_gpu; d++) {
        if (!gpu_work[d].empty()) {
            gpu_events.push_back(dispatch_experts_gpu(gpu_queues[d], gpu_work[d], src1_data));
        }
    }
    auto cpu_future = cpu_work.empty() ? std::future<void>{}
                                       : dispatch_experts_cpu(cpu_pool, cpu_work, src1_data);

    // 4. Gather results (activation_bytes per expert, not hardcoded sizes)
    gather_expert_results(dst, gpu_events, cpu_future);
}
```

#### 2c: Activation Shipping for GPU1 and CPU

For experts on GPU1 or CPU, ship the activation (`activation_bytes`, e.g. 11.2 KB for GPT-OSS) instead of the weight (`expert_bytes`, e.g. 12.6 MB):

```cpp
sycl::event dispatch_experts_gpu(sycl::queue & q, vector<ExpertWork> & work,
                                  const float * src1_device) {
    // 1. Copy activation to target device (11.2KB, negligible latency)
    auto act_event = q.memcpy(act_staging, src1_device, K * sizeof(float));

    // 2. Dispatch GEMM on target device's queue (depends on activation copy)
    return q.submit([&](sycl::handler & h) {
        h.depends_on(act_event);
        // oneDNN GEMM or MMVQ kernel using expert weights already in this device's VRAM
        launch_expert_gemm(h, work, act_staging, result_staging);
    });
}
```

#### 2d: Fused Kernel Path Fix

Enable `ggml_sycl_mul_mat_id_fused()` for expert-cache-resident weights:

```cpp
// ggml-sycl.cpp:22899 — fix the pointer type check
// BEFORE: rejects non-device weights (mmap returns sycl::usm::alloc::unknown)
if (sycl::get_pointer_type(src0_weight_ptr, ctx) != sycl::usm::alloc::device) return false;

// AFTER: accept unified-cache-managed device pointers via placement table
auto placement = unified_cache->get_expert_placement(layer_id, expert_id);
if (placement.device_ptr != nullptr) {
    src0_weight_ptr = placement.device_ptr;
    // SOA device pointer from unified cache placement table → fused path OK
}
```

#### 2e: ExpertCache Removal & Cleanup

Once Phases 2a-2d are validated (correct output, no performance regression), **remove all ExpertCache code** so that only the unified cache expert path exists. This prevents accidental use of the old parallel system.

**Files to DELETE entirely:**

| File | Contents | Why delete |
|------|----------|-----------|
| `ggml/src/ggml-sycl/expert-cache.cpp` | `ExpertCache` class implementation (~750 lines, 30+ methods) | All functionality absorbed by unified cache |
| `ggml/src/ggml-sycl/expert-cache.hpp` | `ExpertCache` class definition, `ExpertLookup` struct, `expert_key` struct | Types replaced by `ExpertPlacement` struct in unified cache |

**Files to MODIFY:**

| File | Line(s) | What to remove/change |
|------|---------|----------------------|
| `ggml-sycl.cpp` | 223 | Remove `#include "ggml-sycl/expert-cache.hpp"` |
| `ggml-sycl.cpp` | 233 | Remove `static ggml_sycl::ExpertCache g_expert_caches[GGML_SYCL_MAX_DEVICES]` |
| `ggml-sycl.cpp` | 235 | Remove `ggml_sycl_get_expert_cache()` accessor function |
| `ggml-sycl.cpp` | 286 | Remove `ggml_sycl_init_expert_cache()` function |
| `ggml-sycl.cpp` | 542 | Remove multi-GPU expert cache initialization loop |
| `ggml-sycl.cpp` | 25010, 25059 | Replace `ExpertCache` dispatch calls with unified cache placement table lookups |
| `ggml-sycl.cpp` | 27852 | Remove first-graph expert cache initialization trigger |
| `expert-prefetch.cpp` | 37, 73 | Remove `ExpertCache *` parameter from `ExpertPrefetcher::init()` — refactor to use unified cache placement table |
| `expert-prefetch.hpp` | includes | Remove `#include "expert-cache.hpp"`, remove `ExpertCache * cache_` member |
| `unified-cache.hpp` | 1537 | Update comment that references ExpertCache |
| `CMakeLists.txt` | (ggml-sycl sources) | Remove `expert-cache.cpp` from build |

**Migration strategy**: Do NOT delete ExpertCache and refactor in one step. Instead:

1. First: add a compile-time `#ifdef GGML_SYCL_LEGACY_EXPERT_CACHE` guard around all ExpertCache code
2. Build without the define → verifies nothing depends on ExpertCache
3. Run full test suite to confirm correctness with only unified cache path
4. Then: delete the guarded code, delete the files, remove from CMakeLists.txt

**ExpertPrefetcher refactoring detail**: The prefetcher's core logic (predicting next-layer experts from gating logits) is independent of ExpertCache. Refactor it to:
- Accept a placement table reference instead of `ExpertCache *`
- Call `unified_cache->ensure_cached_layout()` for predicted experts (same async H2D)
- This preserves the prefetching capability while eliminating the ExpertCache dependency

### Phase 3: Expert Deferral + Prefetching (Maximum Overlap)

**Goal**: Hide CPU expert latency behind GPU attention. Predict next-layer experts.
**Effort**: 3-5 days
**Expected improvement**: Eliminates CPU as bottleneck for TG

#### 3a: Expert Deferral (KTransformers technique)

When CPU experts are the bottleneck, defer 1-2 slowest CPU experts:
- Their partial results accumulate in a deferral buffer
- Next layer's GPU attention runs concurrently with deferred CPU expert compute
- Results are added to the residual stream when ready (accuracy-safe via residual connections)

#### 3b: Next-Layer Expert Prediction

Use current layer's gating logits to predict next-layer expert activation (~94% accuracy):
- While current layer computes, prefetch predicted experts for next layer
- On cache hit: zero latency. On miss with correct prediction: expert already streaming.

---

## Comparison: Old Plan vs New Plan

*All sizes below are GPT-OSS 120B MXFP4 examples; actual values computed from model metadata at runtime.*

| Aspect | Old Plan (Streaming) | New Plan (Expert Parallelism) |
|--------|---------------------|------------------------------|
| Core idea | Stream weights through ping-pong VRAM | Compute on device that has weights |
| Weight transfer per expert | `expert_bytes` per expert (e.g. 12.6 MB) | 0 (weights stay put) |
| Activation transfer | 0 | `activation_bytes` per expert (e.g. 11.2 KB) |
| Devices utilized | Primary GPU only (serial) | All GPUs + CPU (parallel) |
| PP64 estimate | ~2 tok/s (serial streaming) | ~58 tok/s (parallel compute) |
| TG estimate | ~3 tok/s | ~15-30 tok/s |
| GPU idle time | High (waiting for PCIe) | Near-zero (all devices busy) |
| VRAM requirement | Streaming buffers | Expert cache pools |
| Complexity | New MoEStreamingEngine class | Extends existing expert cache + dispatch |

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| Host-pinned allocation fails | LOW | Already working: `pinned_chunk_pool` allocates 8GB chunks via `sycl::malloc_host`, `ulimit -l` = unlimited on this system. On failure, `ensure_cached_alloc()` auto-falls back to unpinned malloc → mmap alias. No crash risk |
| Model-scale pinned allocation | LOW | Budget capped at `host_budget_pct=90%` of system RAM. For 256GB system: ~230GB budget. Even GPT-OSS 120B's ~64GB expert weight total fits easily. Smaller MoE models (Mixtral ~30GB, Qwen ~12GB) have even more headroom |
| Level Zero ~1005 allocation count limit | LOW | `pinned_chunk_pool` allocates 8GB chunks. Even largest models need ~8-10 chunks. Already solved in existing code (pinned-pool.hpp:25) |
| B50 GEMM dispatch reliability | MEDIUM | Test single-expert matmul on B50 first. Known: B50 queue works for tensor split merges (commit b322792f4) |
| Cross-device result gathering | LOW | Results are tiny (16KB). Use proven BCS-mediated copy from tensor split work |
| Expert popularity profiling | LOW | Start with uniform distribution, add runtime profiling later |
| CPU expert throughput for PP | MEDIUM | PP batch=64 means large matmuls — CPU may be slow. Expert deferral (Phase 3) mitigates |
| oneDNN primitive cache for B50 | LOW | B50 uses same oneDNN backend. Warmup pass populates cache per-device |
| ExpertCache removal breaks TG fast-path | MEDIUM | Phase 2e uses `#ifdef` guard → build without → test → delete. Incremental validation before removal |
| SOA conversion overhead at load time | LOW | One-time cost at model init. Throughput ~2 GB/s CPU reorder → scales linearly with total expert weight (e.g. ~29s for GPT-OSS 120B's ~57 GB, ~15s for Mixtral's ~30 GB). Acceptable for model load |

---

## References

- **Fiddler** (ICLR 2025): Activation shipping, static popularity placement. [arXiv:2402.07033](https://arxiv.org/html/2402.07033)
- **KTransformers** (SOSP 2025): Expert deferral, AMX kernels, CPU/GPU hybrid. [GitHub](https://github.com/kvcache-ai/ktransformers)
- **HybriMoE** (DAC 2025): Score-based caching, impact-driven prefetching. [arXiv:2504.05897](https://arxiv.org/abs/2504.05897)
- **MoE-Infinity** (2024): Sparsity-aware cache, layer-proximity eviction. [arXiv:2401.14361](https://arxiv.org/html/2401.14361v3)
- **HOBBIT** (2024): Mixed-precision fallback for cache misses. [arXiv:2411.01433](https://arxiv.org/html/2411.01433v2)
- **Aurora** (2024): Optimal heterogeneous device assignment (Theorem 5.1). [arXiv:2410.17043](https://arxiv.org/html/2410.17043v1)
- **PreScope** (2025): Learnable predictor + async chunked IO. [arXiv:2509.23638](https://arxiv.org/html/2509.23638v1)
- **DeepSpeed-MoE** (ICML 2022): Expert parallelism + expert slicing. [Paper](https://arxiv.org/pdf/2201.05596)
- **MoETuner** (2025): ILP-based expert placement optimization. [arXiv:2502.06643](https://arxiv.org/html/2502.06643v1)

## De-Risking Tests (Pre-Implementation Validation)

These small standalone tests validate risky assumptions before writing the full implementation. Run them first to catch showstoppers early.

### Test 1: B50 GEMM Dispatch (de-risks: B50 GEMM dispatch reliability, MEDIUM)

Verify that oneDNN GEMM works on B50 (device 1) for expert-sized matmuls. We've used B50 for tensor split merges but never for independent GEMM dispatch.

```cpp
// tests/test-sycl-b50-expert-gemm.cpp
// Dispatches a single expert-sized GEMM on each visible device using dimensions
// queried from the actual model's expert tensors (e.g. M=1, K=hidden_dim, N=ffn_dim).
// For GPT-OSS 120B: M=1, K=2880, N=2880 — but test should read from model metadata.
// Run: ONEAPI_DEVICE_SELECTOR="level_zero:0,1" ./build/bin/test-sycl-b50-expert-gemm \
//      -m /path/to/model.gguf
```

**What to verify**: (1) B50 queue accepts oneDNN GEMM, (2) output matches CPU ref within FP16 tolerance, (3) no hangs or crashes

### Test 2: Cross-Device Activation Shipping Round-Trip (de-risks: activation shipping, LOW)

Ship an activation vector from B580 device memory to B50 via H2D, compute a GEMM on B50, then gather the result back to B580 via D2H. Validates the complete activation shipping pipeline.

```cpp
// tests/test-sycl-activation-shipping.cpp
// Activation size = hidden_dim * sizeof(float) — queried from model, not hardcoded
// For GPT-OSS 120B: hidden_dim=2880 → 11,520 bytes (11.2 KB)
// For Mixtral 8x7B: hidden_dim=4096 → 16,384 bytes (16 KB)
//
// 1. Allocate hidden_dim floats on B580 (device mem)
// 2. q_b50.memcpy(b50_staging, b580_src, activation_bytes)  // H2D to B50
// 3. Launch small GEMM on B50 queue (M=1, K=hidden_dim, N=ffn_dim)
// 4. q_b580.memcpy(b580_dst, b50_result, activation_bytes)  // D2H back
// 5. Verify result matches single-device reference
// Measure round-trip latency.
// Run: ./build/bin/test-sycl-activation-shipping -m /path/to/model.gguf
```

**What to verify**: (1) cross-device memcpy works without in-order queue requirement (results are tiny, no BCS concerns), (2) latency < 0.1 ms for typical activation sizes (~11-16 KB), (3) GEMM output correct

### Test 3: Pinned Pool Stress Test at Model Scale (de-risks: pinned allocation at model scale, LOW)

Allocate model-scale pinned memory from pinned_chunk_pool in 8 GB chunks (e.g. ~60 GB for GPT-OSS 120B, ~30 GB for Mixtral, varies by model). This tests the actual allocation path used for expert weights.

```bash
# Can be done with existing infrastructure — just configure host budget high
GGML_SYCL_HOST_BUDGET_PCT=50 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-cli \
  -m /Storage/GenAI/models/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  -sys "Hello" -c 512 -n 1 -p "test"
# Check logs for: "Allocated pinned chunk N (size=8192.0 MB, total=XX.X GB)"
# Should see multiple 8GB chunks allocated without failure (count depends on model size)
```

**What to verify**: (1) 8 × 8GB chunks allocate successfully, (2) no OOM or timeout, (3) unified cache logs show expected budget

### Test 4: ensure_cached_layout() with MOE_EXPERT Type (de-risks: unified cache expert path, MEDIUM)

Verify that the existing `ensure_cached_layout()` → SOA conversion → device upload path works for MOE_EXPERT cache entries (currently only used for DENSE_WEIGHT).

```bash
# The existing model load already exercises this for dense weights.
# To test for experts specifically, use the existing prestage_routed_experts()
# with tracing enabled:
GGML_SYCL_DEBUG=2 GGML_SYCL_CPU_EXPERT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-cli \
  -m /Storage/GenAI/models/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  -c 512 -n 5 -p "Hello" 2>&1 | grep -E 'MOE_EXPERT|expert.*layout|expert.*cached'
```

**What to verify**: (1) MOE_EXPERT entries are created in host cache, (2) SOA device copies are materialized, (3) lookup returns valid device pointers

### Test 5: CPU vec_dot on Host-Pinned AOS Expert Weights (de-risks: CPU expert compute, LOW)

Verify that CPU vec_dot (AVX2/AVX-VNNI) works correctly when reading from `sycl::malloc_host` memory instead of mmap. The pointer types differ but both are host-addressable. Works for any quantization type — vec_dot dispatches based on `ggml_type`, not hardcoded dimensions.

```bash
# Already works for HOST_COMPUTE mode (commit 65ff78521).
# The CPU path reads from host-pinned memory via g_leaf_staging_cache.
# Verify by running CPU-only expert dispatch:
GGML_SYCL_CPU_EXPERT_TG=1 GGML_SYCL_VRAM_BUDGET_PCT=5 \
  ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu" \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  -p "1, 2, 3," -n 10 --seed 42 --temp 0
# Verify: correct output (not garbage), no SEGFAULT
```

**What to verify**: (1) CPU vec_dot reads from pinned memory correctly, (2) output is coherent text, (3) no alignment or page fault issues

---

## Verification Plan

```bash
source /opt/intel/oneapi/setvars.sh --force

# Phase 1: PP correctness after removing sync stalls (must not crash)
GGML_SYCL_CPU_EXPERT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-cli \
  -m /Storage/GenAI/models/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  -sys "You are a helpful assistant." -c 4096 -n 20 \
  -p "What is the capital of France?"

sleep 60

# Phase 2: Multi-device expert parallelism TG
GGML_SYCL_CPU_EXPERT_TG=1 ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  -c 4096 -n 32
# Target: TG > 15 tok/s (up from 3.1)

sleep 60

# Phase 2: Multi-device PP
GGML_SYCL_CPU_EXPERT_TG=1 ONEAPI_DEVICE_SELECTOR="level_zero:0,1" \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  -c 4096 -p 64
# Target: PP > 40 tok/s (up from 2.5)

sleep 60

# Regression: Mistral 7B must not regress
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Target: PP512 >= 1300, TG128 >= 68
```
