# Unified Memory Manager Design for llama.cpp SYCL Backend

**Date**: 2026-04-03
**Status**: Design Document (Pre-Implementation)
**Scope**: Complete VRAM/host memory lifecycle management for the SYCL inference backend

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research: Production Inference Engine Memory Management](#2-research-production-inference-engine-memory-management)
3. [Research: SYCL-Specific Memory Patterns](#3-research-sycl-specific-memory-patterns)
4. [Current State Analysis](#4-current-state-analysis)
5. [Gap Analysis](#5-gap-analysis)
6. [Target Architecture](#6-target-architecture)
7. [Phased Implementation Plan](#7-phased-implementation-plan)
8. [Risks and Mitigations](#8-risks-and-mitigations)
9. [Sources](#9-sources)

---

## 1. Executive Summary

The vision is a **single unified memory manager** that owns ALL GPU VRAM and host-pinned RAM for inference. At init time, it pre-allocates the full VRAM budget (and a host-pinned pool sized to the model). At runtime, zero direct `sycl::malloc_device` or `sycl::malloc_host` calls occur -- everything flows through the unified cache. Higher-priority data (compute scratch, KV cache) can evict lower-priority data (cold weights, MoE experts) from VRAM to host memory. Callers always re-query data location before use, since data may have moved between tiers.

The current unified cache already implements roughly 60-70% of this vision. The key gaps are: (a) VRAM is allocated per-entry rather than pre-allocated as a single block, (b) pool_leg (the compute scratch allocator) still makes independent `sycl::malloc_device` calls that bypass the cache, (c) priority-based eviction exists in embryonic form (tier-based scoring) but is not fully priority-driven, and (d) KV cache allocations are tracked via `update_reserved_bytes` budget accounting but do not flow through the cache's allocation API.

---

## 2. Research: Production Inference Engine Memory Management

### 2.1 TensorRT-LLM (NVIDIA)

**Pre-allocation**: TensorRT-LLM pre-allocates a **paged KV cache pool** at initialization. By default, it claims 90% of remaining free GPU memory for KV cache after engine weights are loaded. The pool is divided into configurable block sizes (2-64 tokens per block). Weights are compiled into the TRT engine and occupy fixed GPU memory.

**Workspace/Activation**: Uses CUDA's stream-ordered memory allocator (cuMemAllocAsync) via the BufferManager, which pools allocations through the CUDA driver's default memory pool. Activations (intermediate tensors) are workspace that gets reused across layers.

**Memory pressure**: TRT-LLM implements priority-based KV block eviction. When a new blank block is needed and the pool is full, the runtime evicts the lowest-priority blocks first. Each block carries a priority score (0-100); within a priority tier, eviction follows LRU ordering. Host offloading is supported via the `host_cache_size` parameter (disabled by default) -- evicted blocks are copied to pinned host memory and can be restored on demand. If eviction cannot free sufficient space, the request is rejected or the batch size is reduced.

**Key takeaway**: Single-owner model. The engine statically owns weight memory; KV cache is pre-allocated as a fixed pool with priority-based eviction and optional host offloading; activations use stream-ordered recycling. No runtime malloc during inference.

### 2.2 vLLM / PagedAttention

**Pre-allocation**: vLLM pre-allocates all KV cache memory as a **block pool** at startup. Blocks are fixed-size (typically 16 tokens = 12.8KB for a 13B model). A free-block queue (doubly-linked list with sentinel nodes) provides O(1) alloc/free. Model weights are loaded into GPU memory separately and stay fixed.

**Memory pressure**: When GPU memory is full, vLLM uses one of two preemption strategies:
- **Swap**: Move KV cache pages to CPU memory (pinned host) via async DMA, resume later.
- **Recompute**: Evict the request entirely and recompute KV cache when memory becomes available.

Freed blocks return to the free pool via LRU ordering. The free queue maintains LRU order automatically -- freed blocks append to tail, allocations pop from head.

**Location transparency**: PagedAttention uses a **block table** (virtual-to-physical mapping per sequence) so attention kernels can access non-contiguous KV cache blocks transparently. The block table is updated when pages are swapped.

**Key takeaway**: Fixed-size block pool eliminates fragmentation. O(1) alloc/free via linked list. Swap-to-CPU provides graceful degradation. Block table provides location indirection.

### 2.3 vAttention (Microsoft, ASPLOS 2025)

**Innovation**: Uses **CUDA Virtual Memory Management** (cuMemMap/cuMemUnmap) to decouple virtual and physical address spaces. Pre-allocates a large virtual address range (enough for max batch size), then maps physical pages on demand.

**Advantage over PagedAttention**: KV cache is contiguous in virtual memory, so standard attention kernels (FlashAttention, FlashInfer) work without modification. Physical memory is still managed in pages to avoid fragmentation.

**Key takeaway**: Virtual memory indirection is the cleanest location-transparency mechanism. Level Zero does have equivalent virtual memory APIs (`zeVirtualMemReserve`, `zeVirtualMemMap`, `zePhysicalMemCreate`, etc. -- in the L0 core spec since 1.3.7+), but SYCL the language does not expose them. Accessing these APIs requires L0 interop from SYCL. See Section 3.4 for details on this opportunity.

### 2.4 FlexGen

**Three-tier offloading**: FlexGen manages GPU, CPU, and disk as a memory hierarchy. It uses a **linear programming solver** at startup to compute optimal placement of weights, activations, and KV cache across tiers.

**I/O optimization**: Multiple CUDA streams overlap compute with DMA transfers. Block-level scheduling pipelines layer execution: while GPU processes layer N, CPU/disk prefetches layer N+1.

**Compression**: 4-bit weight and KV cache compression reduces I/O bandwidth requirements between tiers.

**Key takeaway**: Static planning via LP at init time. The "compute where data lives" principle -- if weights are on CPU, compute on CPU. Overlapped DMA is critical for hiding tier-transition latency.

### 2.5 DeepSpeed ZeRO-Inference

**Layer streaming**: Pins entire model weights in CPU memory (or NVMe). Streams weights layer-by-layer into GPU, keeping only 1-2 layers resident on GPU at a time. This reduces GPU memory by 100x (e.g., 530B model: 1TB to 10GB).

**Prefetching**: Double-buffering -- while layer N executes on GPU, layer N+1 is being DMA'd from CPU. Uses pinned host memory as staging.

**Recent improvements**: Weight quantization + KV cache offloading for up to 20x throughput improvement (DeepSpeed >= 0.10.3).

**Key takeaway**: Pure streaming model -- no caching, no eviction. Works when model vastly exceeds GPU memory. Simple but high-latency.

### 2.6 KTransformers (SOSP 2025)

**CPU/GPU hybrid for MoE**: Specifically designed for sparse MoE models. Dense components (attention, shared experts) run on GPU. Sparse experts run on CPU using AMX/AVX-512 optimized kernels.

**Memory tiering**: Expert weights live in CPU DRAM. Attention weights and KV cache live in GPU VRAM. The system dynamically schedules which experts run on CPU vs GPU based on activation patterns.

**Optimizations**: AMX tiling-aware memory layout for cache-optimal CPU expert execution. Dynamic AMX/AVX-512 kernel selection based on arithmetic intensity.

**Performance**: 4.6-19.7x prefill speedups, 1.25-4.09x decode speedups vs existing systems. Supports DeepSeek-R1/V3 on single 24GB GPU + 382GB DRAM.

**Key takeaway**: "Compute where data lives" for MoE. Expert placement is semi-static (based on model architecture), not dynamically migrated. This aligns well with our existing CPU offload architecture.

### 2.7 ORCA / Sarathi-Serve

**Continuous batching**: ORCA introduced iteration-level scheduling -- batch composition changes every forward pass. New sequences enter when old ones complete.

**Memory model**: ORCA reserves KV cache for max sequence length upfront (wasteful). Sarathi-Serve adds chunked prefill to limit batch token count, reducing peak activation memory.

**Key takeaway**: Scheduling innovation, not memory management. The KV cache reservation model (max length upfront) is what PagedAttention was designed to replace.

### 2.8 Synthesis: Common Patterns Across Production Systems

| System | Pre-allocate VRAM? | Eviction mechanism | Location transparent? | "Data moved" handling |
|--------|-------------------|--------------------|-----------------------|----------------------|
| TensorRT-LLM | Yes (KV pool) | Yes (priority-based + host offload) | No | Evicted blocks copied to host; restored on demand |
| vLLM | Yes (block pool) | Swap to CPU or recompute | Yes (block table) | Block table updated atomically |
| vAttention | Yes (virtual range) | Physical page reclaim | Yes (virtual addresses stable) | Physical pages remapped |
| FlexGen | Yes (LP-planned) | Static placement | No (placement fixed at init) | N/A |
| DeepSpeed ZeRO | No (stream from CPU) | N/A (nothing cached) | No | N/A |
| KTransformers | Partial (GPU-resident fixed) | No (CPU weights stay on CPU) | No | N/A |
| llama.cpp CUDA | No (pool_leg caches) | LRU pool eviction | No | Stale pointers crash |
| **Our target** | **Yes (single alloc)** | **Priority-based async DMA** | **Yes (re-query API)** | **Callers re-query; kernel dispatch adapts** |

**Critical insight**: No production system does exactly what we want. vLLM comes closest with its block pool + swap, but it only applies to KV cache. Our design is more ambitious: ALL memory types (weights, KV, compute scratch, staging) in one unified pool with priority-based migration.

---

## 3. Research: SYCL-Specific Memory Patterns

### 3.1 USM Allocation Types

| API | Location | GPU access | CPU access | VRAM consumption |
|-----|----------|------------|------------|-----------------|
| `sycl::malloc_device` | Device VRAM | Direct | None | Yes |
| `sycl::malloc_host` | Host RAM (pinned) | PCIe zero-copy | Direct | No |
| `sycl::malloc_shared` | Migrates | Direct (after migration) | Direct (after migration) | Yes (when migrated) |

**Best practice** (Intel optimization guide): Use `malloc_device` for compute-intensive data. Use `malloc_host` for staging/transfer buffers. Avoid `malloc_shared` for performance-critical paths (migration overhead is unpredictable).

### 3.2 Pre-allocation Pool Strategy for SYCL

Intel's documentation recommends allocating device memory once during initialization. Each `sycl::malloc_device` call goes through the Level Zero driver, which:
1. Allocates virtual address space
2. Maps physical pages (may be lazy/residency-managed)
3. Registers the allocation in the driver's internal tracking

This has non-trivial overhead (100us to tens of ms per call -- large allocations regularly hit 10-30ms) and can fail under fragmentation. Pre-allocating a single large block and sub-allocating from it avoids this overhead entirely.

**Level Zero specifics**:
- `zeMemAllocDevice` allocations use residency management (lazy physical page mapping)
- Maximum single allocation size is device-dependent. On Arc B580 (BMG), `maxMemAllocSize` = 11.3 GB = 100% of visible VRAM. The ~75% limit was an Alchemist-era (DG2) restriction that BMG eliminated. Fallback chunking is retained for portability to older hardware, not needed on B580.
- UMF (Unified Memory Framework) provides pool allocator primitives, but we need custom priority-based eviction that UMF does not support

### 3.3 Current Pre-allocation in Our Code

We already have three pre-allocation mechanisms:
1. **`sycl_device_pool` (layout_pool_)**: 256MB chunk bump allocator for weight layout entries
2. **Compute arena**: Single `malloc_device` block for compute scratch, bump-allocated
3. **`pinned_chunk_pool`**: 8GB chunk pool for host-pinned memory

These are the building blocks. The design goal is to unify them into a single coherent system.

### 3.4 Level Zero Virtual Memory APIs

Level Zero (since spec version 1.3.7+) provides full virtual memory management APIs equivalent to CUDA's `cuMemMap`/`cuMemUnmap`:

- **`zeVirtualMemReserve`** / **`zeVirtualMemFree`**: Reserve/release contiguous virtual address ranges without backing physical memory.
- **`zeVirtualMemMap`** / **`zeVirtualMemUnmap`**: Map/unmap physical memory objects into reserved virtual ranges.
- **`zePhysicalMemCreate`** / **`zePhysicalMemDestroy`**: Allocate/free physical memory pages independently of virtual addresses.

These APIs enable a **vAttention-style** virtual memory management strategy for KV cache:
- Reserve a large contiguous virtual address range at init (sized for max context length).
- Allocate physical pages on demand as KV cache grows.
- FlashAttention and other kernels operate on the contiguous virtual KV buffer without modification -- no block tables or scatter/gather needed.
- Pages can be remapped between VRAM and host physical memory without changing virtual pointers, enabling transparent KV offloading.

**Access from SYCL**: SYCL the language does not expose these APIs. Using them requires Level Zero interop (`sycl::get_native<sycl::backend::ext_oneapi_level_zero>(...)`) to obtain the `ze_context_handle_t` and `ze_device_handle_t`, then calling the L0 APIs directly. The resulting virtual addresses can be used with SYCL kernels as raw pointers.

**Risk**: The BMG compute-runtime implementation of these APIs needs verification -- virtual memory support varies by driver version and has had bugs in past releases. Thorough testing with the patched compute-runtime (26.09.37435.10) is required before relying on this path.

**Status**: Stretch goal beyond this epic's scope. The current arena-based design (Phases 1-7) does not depend on virtual memory APIs, but a future phase could replace the VRAM arena with a virtual-memory-backed pool for zero-copy KV growth and transparent tier migration.

### 3.5 Micro-Benchmark: Host-Resident Weight Access Strategy

**Test system**: Arc B580 (12 GB VRAM, PCIe 4.0 x8), Arrow Lake Core Ultra 7 265K (20 cores, DDR5-5600), Level Zero, `ONEAPI_DEVICE_SELECTOR=level_zero:0`

**Benchmark source**: `tests/test-host-weight-strategy.cpp`

When weights are evicted from VRAM to host-pinned memory, the inference path must still read them. There are two options: (a) the GPU reads them over PCIe via zero-copy, or (b) the CPU reads them directly from host memory. The micro-benchmark below measures both paths across three representative tensor sizes (attention projection, FFN gate, MoE expert block).

#### Results

| Benchmark | attn 9 MB | FFN 33 MB | MoE 116 MB |
|-----------|-----------|-----------|------------|
| GPU VRAM read (baseline) | 0.083 ms (113.5 GB/s) | 0.492 ms (67.1 GB/s) | 2.006 ms (57.6 GB/s) |
| GPU zero-copy SOA | 0.837 ms (11.3 GB/s) | 2.923 ms (11.3 GB/s) | 10.104 ms (11.4 GB/s) |
| GPU zero-copy AOS | 0.836 ms (11.3 GB/s) | 2.897 ms (11.4 GB/s) | 10.116 ms (11.4 GB/s) |
| CPU AOS (1 thread) | 1.458 ms (6.5 GB/s) | 6.079 ms (5.4 GB/s) | 21.699 ms (5.3 GB/s) |
| CPU AOS (18 threads) | 0.320 ms (29.5 GB/s) | 1.778 ms (18.6 GB/s) | 3.879 ms (29.8 GB/s) |
| CPU SOA (1 thread) | 1.534 ms (6.2 GB/s) | 5.772 ms (5.7 GB/s) | 20.450 ms (5.7 GB/s) |
| CPU SOA (18 threads) | 0.340 ms (27.7 GB/s) | 0.860 ms (38.4 GB/s) | 3.808 ms (30.4 GB/s) |
| SOA-to-AOS conversion | 0.381 ms (24.8 GB/s) | 3.388 ms (9.8 GB/s) | 12.665 ms (9.1 GB/s) |
| AOS-to-SOA conversion | 0.611 ms (15.4 GB/s) | 4.016 ms (8.2 GB/s) | 12.949 ms (8.9 GB/s) |
| Parallel GPU(VRAM) + CPU(host) | 0.441 ms wall | 1.192 ms wall | 3.382 ms wall |

#### Key Findings

1. **GPU zero-copy bandwidth is PCIe-limited at ~11.3 GB/s regardless of layout.** SOA vs AOS makes no measurable difference over PCIe. The bus is the bottleneck, not the access pattern.

2. **CPU multi-threaded (18 threads) is 1.6-2.6x faster than GPU zero-copy for all tensor sizes.** At 18.6-29.8 GB/s, the CPU saturates DDR5-5600 bandwidth and avoids the PCIe bottleneck entirely.

3. **SOA layout has minimal CPU penalty.** SOA is only 1.0-1.1x slower than AOS on CPU, and for 33 MB tensors (FFN-sized) SOA is actually 2x faster due to better cache-line utilization of the dequant hot fields.

4. **SOA-to-AOS conversion is prohibitively expensive.** A single conversion pass costs 100-125% of a GPU zero-copy read -- converting layout on eviction or promotion would eliminate any benefit from CPU compute.

5. **Parallel GPU(VRAM) + CPU(host) gives 1.9-3.0x wall-time improvement over GPU-only zero-copy.** When GPU processes VRAM-resident layers and CPU processes host-resident layers concurrently, the two memory buses operate independently.

#### Design Decision

Based on these results, the host-resident weight access strategy is:

- **Store SOA layout in host-pinned memory** (same layout as VRAM). No layout conversion on eviction or promotion. A single SOA copy serves both tiers.
- **Use CPU compute for host-resident weights** (not GPU zero-copy). The CPU path is strictly faster and does not contend for PCIe bandwidth.
- **Pipeline GPU and CPU work in parallel.** GPU processes VRAM-resident layers while CPU processes host-resident layers. The two paths use independent memory buses (VRAM vs DDR5) and independent compute units.
- **This aligns with the existing HOST_COMPUTE architecture** (`GGML_SYCL_HOST_COMPUTE=1`), which already dispatches host-resident operations to CPU via `sycl::host_task` and a multi-threaded CPU thread pool.

### 3.6 Micro-Benchmark: MoE Expert Placement Strategy

**Test system**: Arc B580 (12 GB VRAM, PCIe 4.0 x8), Arrow Lake Core Ultra 7 265K (20 cores, DDR5-5600), Level Zero, `ONEAPI_DEVICE_SELECTOR=level_zero:0`

**Benchmark source**: `tests/test-moe-expert-placement.cpp`

This micro-benchmark measures whether routed MoE experts should be computed on GPU (requiring H2D DMA transfer of host-resident weights) or on CPU directly. For each expert size, we measure: GPU VRAM compute (baseline), CPU 18-thread compute, H2D DMA transfer, and the combined DMA+GPU latency.

#### Results

| Expert Size | GPU VRAM | CPU 18T | H2D DMA | DMA+GPU | Break-even reuses |
|-------------|----------|---------|---------|---------|-------------------|
| 19 MB (DeepSeek-V3) | 0.08 ms (249 GB/s) | 0.36 ms (56 GB/s) | 1.47 ms | 1.58 ms | 6 |
| 40 MB (GPT-OSS-120B) | 0.15 ms (285 GB/s) | 1.19 ms (35 GB/s) | 3.12 ms | 3.42 ms | 3 |
| 99 MB (Mixtral) | 1.13 ms (92 GB/s) | 3.33 ms (31 GB/s) | 7.70 ms | 8.94 ms | 4 |
| 128 MB (large) | 1.61 ms (83 GB/s) | 4.60 ms (29 GB/s) | 9.97 ms | 11.62 ms | 4 |

#### Key Findings

1. **DMA+GPU is 2.5-4.4x SLOWER than CPU-only for single-use experts.** The H2D transfer dominates: DMA cost alone is 217-414% of CPU compute time. Even with the GPU's superior compute throughput, the PCIe transfer overhead makes GPU execution strictly worse for one-shot expert evaluation.

2. **Break-even requires 3-6 consecutive reuses**, meaning the same expert must be routed 3-6 tokens in a row before GPU placement pays off. Research on MoE routing persistence shows typical values of 1.8-3x consecutive reuse, which falls below the break-even threshold for most expert sizes.

3. **GPU VRAM bandwidth advantage is irrelevant without residency.** GPU compute is 7-16x faster than CPU when weights are already in VRAM (249-285 GB/s vs 29-56 GB/s), but this advantage is entirely negated by the DMA transfer cost for non-resident experts.

4. **KTransformers approach confirmed.** The data validates the KTransformers design: routed experts should be computed on CPU with host-pinned weights, not transferred to GPU on demand.

#### Design Decision

Based on these results combined with Unsloth's GPT-OSS offloading guide, the MoE expert placement strategy is **static, priority-ordered fill at load time**:

1. **Fill VRAM with highest-priority weights first.** Dense layers (attention, shared experts) always go in VRAM (SOA). They are accessed every token and benefit from VRAM residency.
2. **Fill remaining VRAM with MoE experts in Unsloth priority order.** When VRAM budget allows, experts that fit should stay — GPU VRAM compute is 2.9-4.5x faster than CPU. The priority order (from Unsloth's `-ot` tiering, most impactful first):
   - Down projections first (produce MoE output, most impactful per Unsloth)
   - Up projections next
   - Gate projections last (least impactful, first to overflow to host)
   - Within each category: earlier layers before later layers
3. **Overflow experts go to host-pinned (AOS) for CPU compute.** Never shuttle experts via DMA during inference — the 2.5-4.4x DMA+GPU penalty makes on-demand transfer strictly worse than CPU-local compute.
4. **Expert prefetch cache merges into unified cache.** The legacy shadow pool (`vram_pool_` in `expert-prefetch.cpp`) is eliminated. The prefetcher becomes a scheduling/prediction layer only; all memory management flows through the unified cache's placement plan.

### 3.7 Micro-Benchmark: KV Cache Co-Location

**Test system**: Arc B580 (12 GB VRAM, PCIe 4.0 x8), Arrow Lake Core Ultra 7 265K (20 cores, DDR5-5600), Level Zero, `ONEAPI_DEVICE_SELECTOR=level_zero:0`

**Benchmark source**: `tests/test-kv-colocation.cpp`

This micro-benchmark measures the cost of attention's KV cache access across different placement strategies. When a model is split between GPU and CPU layers, should KV cache be co-located with the layer's weights, or centralized in one tier? We measure flash-attention-style KV read bandwidth for Mistral 7B parameters (8 KV heads, 128 d_head, FP16).

#### Results

| Scenario | seq=512 | seq=2048 | seq=4096 | seq=8192 |
|----------|---------|----------|----------|----------|
| GPU + VRAM KV (co-located) | 0.032 ms (66 GB/s) | 0.041 ms (205 GB/s) | 0.063 ms (266 GB/s) | 0.118 ms (284 GB/s) |
| GPU + host KV (cross-tier) | 0.188 ms (11 GB/s) | 0.735 ms (11 GB/s) | 1.469 ms (11 GB/s) | 2.936 ms (11 GB/s) |
| CPU + host KV (co-located) | 0.351 ms (6 GB/s) | 0.256 ms (33 GB/s) | 0.582 ms (29 GB/s) | 1.173 ms (29 GB/s) |
| CPU + VRAM KV (DMA required) | 0.511 ms | 0.856 ms | 1.768 ms | 3.535 ms |

#### Key Findings

1. **HARD CONSTRAINT: `sycl::malloc_device` is NOT CPU-accessible.** CPU layers cannot read VRAM KV cache directly -- a DMA transfer (D2H) is required, adding latency that grows linearly with sequence length. This is not a performance preference but a hardware limitation.

2. **GPU reading host-pinned KV is 6-25x slower than VRAM KV** and the penalty grows with sequence length. At seq=512, cross-tier is 5.9x slower; at seq=8192, it is 24.9x slower. For long-context inference, cross-tier KV access becomes the dominant bottleneck.

3. **Split model co-located KV outperforms centralized VRAM KV by 2.5x.** For a 24 GPU + 8 CPU layer split, co-located KV (GPU layers use VRAM KV, CPU layers use host KV) totals 3.1 ms. Centralizing all KV in VRAM forces 8 CPU layers to DMA-read, totaling 7.9 ms -- 2.5x worse.

4. **At seq=8192, cross-tier penalty for 8 CPU layers is +18.9 ms per token (153% overhead).** This is the difference between 12.3 ms (co-located) and 31.2 ms (all-VRAM KV), making co-location mandatory for acceptable long-context performance.

#### Design Decision

Based on these results, the KV cache placement strategy is:

- **KV cache MUST be co-located with weights.** GPU layers allocate KV in VRAM. CPU layers allocate KV in host-pinned memory. This is decided at context creation time and fixed during inference.
- **No cross-tier KV access in the hot path.** Each layer reads KV from its local memory tier only.
- **VRAM arena reserves space for GPU-layer KV only**, not for the full model's KV cache. This frees VRAM for weight caching.

**VRAM arena layout** (updated):
```
VRAM Arena:
[Compute Scratch] [KV for GPU layers ONLY] [oneDNN scratch] [<-- Weight Cache]

Host-Pinned Arena:
[KV for CPU layers] [Host-resident weights (AOS)] [Staging]
```

---

## 4. Current State Analysis

### 4.1 Unified Cache Class Structure

The `unified_cache` class (unified-cache.hpp/cpp, ~1400 lines header, ~9100 lines implementation) is the central memory manager for the SYCL backend. It manages:

**Weight caching** (primary purpose):
- `entries_` hashmap: `unified_cache_key -> unified_cache_entry` mapping cache keys to device pointers
- `id_to_key_` reverse map for fast lookup
- Layout-aware: supports AOS, SOA, COALESCED, ONEDNN_PACKED layouts per entry
- LRU+LFU hybrid eviction via `compute_score()` (access_count * exp(-decay * age))
- Tier-based eviction ordering via `eviction_tier()` with priority boost from `alloc_category_priority()`

**Sub-allocators owned by the cache**:
- `layout_pool_` (`sycl_device_pool`): Bump allocator with 256MB chunks for weight entries
- `compute_arena_ptr_`: Single pre-reserved VRAM block for compute scratch
- `scratch_pool_ptr_`: Pre-allocated VRAM for per-op temporaries
- `onednn_weights_scratch_` / `onednn_activations_scratch_`: Fixed buffers for oneDNN FP16 path
- `reorder_temp_buffer_`: Single buffer for GPU-side AOS->SOA reorder
- `persistent_scratches_`: Named scratch buffers for persistent TG kernel

**Host-side pools**:
- `host_cache`: Parallel host-side cache with pinned_chunk_pool for host-pinned memory
- `staging_` buffer + `copy_stage_slots_`: Pre-allocated pinned staging for DMA

**Budget tracking**:
- `budget_` = `base_budget_` - `reserved_` (reserved tracks runtime non-weight allocations)
- `used_` tracks bytes consumed by weight entries + arena + scratch pool
- `update_reserved_bytes()` adjusts budget for KV cache, compute buffers, etc.
- `available_device()` queries Level Zero for actual free VRAM as safety check

**Allocation APIs** (multiple layers):
- `ensure_cached()` / `ensure_cached_layout()`: Weight-specific, includes DMA fill
- `allocate_slot()` / `register_ready()` / `lookup()`: Decomposed cache operations
- `allocate()` / `deallocate()`: Unified allocation with host fallback (Phase 3)
- `unified_cache_allocate()` / `unified_cache_deallocate()`: Simplified global API
- `unified_alloc()` / `unified_free()`: Low-level alloc with intent/constraints
- `arena_alloc()` / `get_scratch()`: Pre-reserved pool sub-allocators

### 4.2 Pool Leg (ggml_sycl_pool_leg)

The `ggml_sycl_pool_leg` in ggml-sycl.cpp is a best-fit buffer pool (similar to CUDA's `ggml_cuda_pool_leg`). It caches up to 1024 freed buffers and reuses them by best-fit matching. When no cached buffer fits, it calls into the unified cache allocation path:

1. First tries `unified_cache_arena_alloc()` (bump from pre-reserved arena)
2. Falls back to `unified_cache_allocate()` (budget-tracked device/host allocation)

This is already partially integrated but not fully. The pool_leg still holds device pointers that the cache does not know about (between alloc and free from pool_leg's perspective).

### 4.3 KV Cache Allocation

KV cache is allocated in `ggml-sycl.cpp` via `unified_cache_allocate()` with `alloc_category::KV_CACHE`. Per-layer allocation is supported for hot/cold KV tiering. The cache tracks KV bytes via `update_reserved_bytes()` so weight eviction accounts for KV pressure.

### 4.4 Resolve Weight Pattern

`ggml_sycl_resolve_weight()` in common.hpp is the location-transparent weight lookup:
1. Non-weight tensors: return raw data pointer
2. Weight tensors: query unified cache for the best available entry
3. Returns `{ptr, layout, on_device}` so callers can dispatch to appropriate kernel

This pattern already exists and works correctly. It needs to be extended to non-weight allocations.

---

## 5. Gap Analysis

### 5.1 What Already Exists (Strengths)

| Capability | Status | Implementation |
|-----------|--------|----------------|
| Weight caching with layout support | COMPLETE | ensure_cached_layout, 4 layout modes |
| LRU+LFU eviction with tier ordering | COMPLETE | evict_one, compute_score, eviction_tier |
| Host fallback for weights | COMPLETE | host_cache, host_resident entries |
| Budget tracking (weights + runtime) | COMPLETE | used_, reserved_, available() |
| Pre-allocated compute arena | COMPLETE | reserve_compute_arena, arena_alloc |
| Pre-allocated scratch pool | COMPLETE | reserve_scratch_pool, get_scratch |
| Chunk-based device pool | COMPLETE | sycl_device_pool (layout_pool_) |
| Chunk-based pinned host pool | COMPLETE | pinned_chunk_pool |
| Unified allocation API | COMPLETE | unified_cache_allocate, unified_alloc |
| Location-transparent weight lookup | COMPLETE | resolve_weight, weight_ptr_result |
| Alloc registry (pointer type tracking) | COMPLETE | alloc_registry with binary search |
| Expert group atomic staging/eviction | COMPLETE | stage_expert_group, evict_expert_group |
| Deferred free (safe async release) | COMPLETE | enqueue_deferred_free, process_deferred_frees |
| Graph replay pin/unpin coordination | COMPLETE | pin, unpin, unpin_on_event |
| Async layer prefetch | COMPLETE | queue_layer_prefetch, prefetch_worker_loop |

### 5.2 What is Missing (Gaps)

| Gap | Current State | Target State | Difficulty |
|-----|---------------|--------------|------------|
| **G1: Single VRAM pre-allocation** | Per-entry malloc_device + 256MB pool chunks | One malloc_device for entire VRAM budget at init | Medium |
| **G2: Single host-pinned pre-allocation** | pinned_chunk_pool grows on demand (8GB chunks) | All host-pinned memory pre-allocated at init | Low |
| **G3: Zero runtime malloc** | pool_leg still calls malloc_device as fallback | All paths use arena/pool sub-allocation only | High |
| **G4: Priority-based eviction** | Tier-based scoring (3 tiers, MoE frequency) | Full priority ordering: scratch > KV > hot weights > cold weights > experts | Medium |
| **G5: KV cache through unified pool** | KV allocated via unified_cache_allocate (separate) | KV sub-allocated from same VRAM arena | High |
| **G6: pool_leg replaced by arena** | pool_leg caches freed buffers independently | pool_leg is a thin wrapper around arena_alloc | Medium |
| **G7: Location-transparent non-weight access** | Only weights have resolve_weight | All allocations queryable for current location | Medium |
| **G8: Async DMA eviction** | Eviction is synchronous (deferred free) | Evict-to-host via async DMA with event tracking | High |
| **G9: Promotion (host-to-device)** | Manual re-cache via ensure_cached | Automatic promotion when VRAM becomes available | Low priority |

### 5.3 What Must NOT Change (Invariants)

1. **Graph replay compatibility**: Pinned entries must not move during graph execution
2. **Arena reset semantics**: Compute scratch must be ephemeral (reset between tokens)
3. **Event-based lifecycle**: Deferred frees must wait for in-flight kernels
4. **Thread safety**: Shared mutex for readers, exclusive for writers
5. **Zero regression**: GPU-only path (PP512, TG128) must not slow down

---

## 6. Target Architecture

### 6.1 Memory Hierarchy

```
                    +----------------------------------+
                    |        UNIFIED MEMORY MANAGER     |
                    |     (unified_cache singleton)     |
                    +----------------------------------+
                    |                                    |
          +---------+---------+              +-----------+----------+
          |   VRAM Arena      |              |   Host-Pinned Arena  |
          |  (single alloc)   |              |  (pre-alloc chunks)  |
          +-------------------+              +----------------------+
          |                   |              |                      |
    +-----+------+  +--------+-------+  +---+--------+  +---------+--+
    | Compute    |  | Weight         |  | Evicted    |  | Staging    |
    | Scratch    |  | Cache          |  | Weights    |  | Buffers    |
    | Zone       |  | Zone           |  | Zone       |  | Zone       |
    | (bump)     |  | (bump+reclaim) |  | (LRU)      |  | (ring)     |
    +------------+  +----------------+  +------------+  +------------+
    | KV Cache   |  | Dense weights  |
    | Zone       |  | Expert weights |
    | (bump)     |  | Partial rows   |
    +------------+  +----------------+
```

### 6.2 VRAM Arena Layout

The entire VRAM budget is allocated as a single `sycl::malloc_device` block (or 2-3 blocks if exceeding per-allocation limits). The arena is divided into zones:

```
VRAM Arena (e.g., 10.7 GB on Arc B580 at 90% budget):
+--------+----------+-------------+----------------------------+
| Compute| KV Cache | oneDNN      |       Weight Cache         |
| Scratch| (grows   | scratch     |    (grows <-- from end)    |
| (fixed)| -->)     | (fixed)     |                            |
+--------+----------+-------------+----------------------------+
^        ^          ^             ^                            ^
0      256MB     256MB+KV      +scratch              arena_end
```

- **Compute Scratch Zone** (fixed, left side): Same as current compute_arena. Reset between tokens. Size determined at context creation.
- **KV Cache Zone** (growable, left-to-right): Bump-allocated from after compute scratch. Grows as context fills.
- **OneDNN Scratch** (fixed): Dequant weight + activation buffers for FP16 PP path.
- **Weight Cache Zone** (growable, right-to-left): Weights loaded from right end of arena. Can shrink if KV cache grows (by evicting cold weights to host).

### 6.3 Priority Tiers

```
Priority 0 (highest, never evicted from VRAM):
  - Compute scratch arena (ephemeral, reset each token)
  - OneDNN scratch buffers (reserved at init)
  - Persistent scratch (TG kernel state)

Priority 1 (evictable only under extreme pressure):
  - KV cache (hot tokens) -- latency-critical
  - Attention weights (used every token)

Priority 2 (evictable under moderate pressure):
  - FFN dense weights
  - Shared MoE expert weights

Priority 3 (readily evictable):
  - Cold MoE expert weights (LFU-ordered)
  - Staging buffers

Priority 4 (always host-eligible):
  - Rarely-used expert weights
  - Debug/profiling allocations
```

Eviction scans from highest priority number (coldest) to lowest. Within a tier, existing LRU+LFU scoring determines victim selection.

### 6.4 Core API (Target)

```cpp
// === Arena Management (init time) ===

// Pre-allocate entire VRAM as a single block. Called once at cache init.
// Replaces individual malloc_device calls throughout inference.
bool reserve_vram_arena(size_t total_bytes);

// === Zone Sub-allocation (runtime, lock-free) ===

// Bump-allocate from compute scratch zone. Reset between tokens.
void * scratch_alloc(size_t size);    // existing arena_alloc
void   scratch_reset();               // existing arena_reset

// Bump-allocate from KV zone. Never freed during inference.
void * kv_alloc(size_t size);

// Allocate from weight zone (right-to-left bump, or reclaim evicted space).
void * weight_alloc(size_t size);

// === Priority-Aware Eviction ===

// Evict entries below the given priority tier until bytes_needed is freed.
// Returns bytes freed. Evicted data is DMA'd to host arena asynchronously.
size_t evict_below_priority(int max_priority, size_t bytes_needed);

// === Location Query (runtime, lock-free) ===

struct memory_location {
    void *           ptr;          // Current pointer (may be device or host)
    alloc_tier       tier;         // DEVICE_VRAM, HOST_PINNED, HOST_MMAP
    ggml_layout_mode layout;       // Data layout at current location
    sycl::event      ready_event;  // Valid if data is in-flight (DMA)
    bool             has_event;
};

// Query current location of a cached allocation.
// MUST be called before every use -- data may have been evicted since last query.
memory_location query_location(const unified_cache_key & key);
```

### 6.5 Kernel Dispatch Adaptation

The existing `resolve_weight` pattern extends to all allocations:

```
For each tensor operation:
  1. location = cache.query_location(weight_key)
  2. if location.tier == DEVICE_VRAM:
       -> dispatch GPU kernel (current fast path)
  3. if location.tier == HOST_PINNED:
       -> dispatch zero-copy kernel (PCIe read, ~8 GB/s)
       -> OR dispatch CPU kernel if compute-bound
  4. if location.has_event:
       -> queue.depends_on(location.ready_event)
```

This is essentially what `resolve_weight` + the dispatch logic in `unified-kernel.cpp` already does. The change is extending it from weights-only to all memory types.

---

## 7. Phased Implementation Plan

### Phase 1: VRAM Arena Pre-allocation (4-6 days)

**Goal**: Replace per-entry `sycl::malloc_device` with sub-allocation from a single pre-allocated VRAM block. Zero functional change -- pure allocation path refactor.

**Changes**:

| File | Function | Change |
|------|----------|--------|
| `unified-cache.hpp` | `unified_cache` class | Add `vram_arena_ptr_`, `vram_arena_size_`, `vram_arena_weight_off_` (right-to-left bump for weights) |
| `unified-cache.cpp` | Constructor | Call `reserve_vram_arena(budget_bytes)` to pre-allocate single block |
| `unified-cache.cpp` | `reserve_vram_arena()` | NEW: Single `sycl::malloc_device(budget_bytes, queue_)`. On BMG (B580), maxMemAllocSize covers 100% VRAM so a single allocation works. On older hardware (Alchemist), split into 2-3 chunks if needed. Track in `vram_arena_chunks_[]` |
| `unified-cache.cpp` | `allocate_slot()` | Replace `layout_pool_->allocate()` and `ggml_sycl_malloc_device_raw()` with sub-allocation from vram_arena |
| `unified-cache.cpp` | `reserve_compute_arena()` | Sub-allocate from vram_arena instead of separate malloc_device |
| `unified-cache.cpp` | `reserve_scratch_pool()` | Sub-allocate from vram_arena instead of separate malloc_device |
| `unified-cache.cpp` | `reserve_onednn_scratch()` | Sub-allocate from vram_arena |
| `unified-cache.cpp` | `reserve_reorder_temp()` | Sub-allocate from vram_arena |
| `unified-cache.cpp` | `evict_one()` | No change to eviction logic. Freed space stays in arena (mark as reclaimable, do NOT sycl::free) |
| `device-pool.hpp` | `sycl_device_pool` | Becomes thin wrapper: sub-allocates from vram_arena instead of calling malloc_device for new chunks |

**Sub-allocator design** (within VRAM arena):

```
struct vram_zone {
    size_t start;    // Offset from arena base
    size_t size;     // Zone capacity
    size_t used;     // Bump pointer (atomic for lock-free alloc)
};

struct vram_arena {
    void *    base;           // Single malloc_device result
    size_t    total;          // Total arena size
    vram_zone compute;        // Left side, bump right
    vram_zone kv;             // After compute, bump right
    vram_zone onednn;         // Fixed block after KV
    vram_zone weights;        // Right side, bump LEFT (grows toward KV)
    // Free list for reclaimed weight space:
    struct free_block { size_t offset; size_t size; };
    std::vector<free_block> weight_free_list;
};
```

**Free list for weight zone**: When a weight is evicted, its space is added to a sorted free list. New weight allocations first try the free list (best-fit), then extend the bump pointer. This avoids fragmentation without runtime sycl::malloc/free.

**Fallback**: If the single pre-allocation fails (e.g., driver refuses to allocate 90% of VRAM), fall back to 2-chunk strategy (50%+40%), then to current per-entry allocation as last resort.

**Migration strategy**:
- Gate behind `GGML_SYCL_VRAM_ARENA=1` env var initially (default OFF)
- Validate with: `llama-bench` PP512+TG128 Mistral 7B Q4_0
- Measure: allocation count reduction, time-to-first-token improvement
- Flip default to ON after validation

**Risks**:
- Max single allocation limit on older hardware (Alchemist ~75% VRAM; BMG has no such limit). Mitigation: chunked arena with virtual addressing.
- Fragmentation in weight free list after heavy eviction/reload cycles. Mitigation: compaction sweep at safe sync points (between graph_compute calls).
- Pre-allocation failure on low-VRAM devices. Mitigation: graceful fallback to per-entry allocation.

### Phase 2: Host-Pinned Arena Consolidation (2-3 days)

**Goal**: Ensure all host-pinned memory is pre-allocated at init. No `sycl::malloc_host` calls during inference.

**Changes**:

| File | Function | Change |
|------|----------|--------|
| `pinned-pool.hpp/cpp` | `pinned_chunk_pool` | Add `pre_allocate_all(model_weight_bytes)`: allocate enough chunks to hold the full model in host memory |
| `unified-cache.cpp` | Constructor | Call `host_cache_.pre_allocate_pinned(total_model_bytes)` at init |
| `unified-cache.cpp` | `copy_to_device_async` | Remove fallback to `sycl::malloc_host` -- staging slots are pre-allocated |
| `unified-cache.cpp` | `host_cache::ensure_cached_alloc` | All allocations from pinned_chunk_pool (already partially true) |

**Current state**: `pinned_chunk_pool` already has `pre_allocate()` and grows in 8GB chunks. The gap is that model size is not always known at cache init time.

**Solution**: Two-phase init:
1. Cache created with estimated budget
2. After model header is parsed (tensor count + sizes known), call `pre_allocate_pinned(model_weight_total_bytes)` to ensure sufficient host capacity

**Risks**: Low. The pinned pool already supports this pattern. Main risk is over-allocation on systems with limited RAM.

### Phase 3: pool_leg Integration (3-4 days)

**Goal**: Replace `ggml_sycl_pool_leg`'s independent buffer caching with pure arena sub-allocation. Eliminate the 1024-buffer free list.

**Changes**:

| File | Function | Change |
|------|----------|--------|
| `ggml-sycl.cpp` | `ggml_sycl_pool_leg::alloc()` | Remove buffer_pool search. Always call `arena_alloc()`. If arena full, return nullptr (let caller handle) |
| `ggml-sycl.cpp` | `ggml_sycl_pool_leg::free()` | No-op. Arena memory is reclaimed by `arena_reset()` between tokens |
| `ggml-sycl.cpp` | `ggml_sycl_pool_leg` destructor | No individual frees needed. Arena freed with cache |
| `unified-cache.cpp` | `arena_alloc()` | Already correct -- atomic bump allocator |
| `unified-cache.cpp` | `arena_reset()` | Already correct -- resets bump pointer |

**Key insight**: The current pool_leg is a **best-fit reuse pool** -- it caches freed buffers and reuses them. But with the arena, compute scratch is bump-allocated and reset each token. There is no need for a free list because all scratch is ephemeral.

**Complication**: Some pool_leg allocations are not ephemeral (e.g., quantization buffers that persist across multiple ops within a graph_compute). The arena handles this because all such allocations occur within a single graph_compute call and are freed at reset.

**Verification**: Track allocation patterns in pool_leg under `GGML_SYCL_DEBUG=1` to confirm all allocations are within graph_compute scope.

**Migration strategy**:
- First: Add arena-first path alongside pool_leg (try arena, fall back to pool_leg)
- Then: Remove pool_leg buffer caching
- Gate: `GGML_SYCL_ARENA_ONLY=1` for testing

**Risks**:
- Arena may be too small for peak scratch usage. Mitigation: size arena based on worst-case scan during first graph_compute, then re-reserve with margin.
- Lifetime mismatch if any pool_leg allocation outlives graph_compute. Mitigation: audit all alloc/free pairs; any persistent ones go through `allocate()` not pool.

### Phase 4: Priority-Based Eviction (3-4 days)

**Goal**: Replace the current tier-based scoring with explicit priority ordering. Higher-priority allocations can force-evict lower-priority ones.

**Changes**:

| File | Function | Change |
|------|----------|--------|
| `unified-cache.hpp` | `unified_cache_entry` | Add `alloc_priority priority` field (0=highest, 4=lowest) |
| `unified-cache.hpp` | New enum | `enum class alloc_priority : uint8_t { COMPUTE=0, KV_HOT=1, WEIGHT_ATTN=2, WEIGHT_FFN=3, EXPERT_HOT=3, EXPERT_COLD=4, STAGING=4 }` |
| `unified-cache.cpp` | `evict_one()` | Change victim selection: scan for lowest priority entry first, then LRU within that priority |
| `unified-cache.cpp` | `evict_below_priority()` | NEW: Evict only entries with priority > threshold. Used when specific priority class needs space |
| `unified-cache.cpp` | `ensure_cached_layout()` | Set priority based on entry type and layer position |
| `unified-cache.cpp` | `allocate()` | Accept priority parameter. If VRAM full, evict entries with lower priority |
| `unified-cache.cpp` | `compute_score()` | Factor in priority: `score = priority_weight * (access_count * decay)` |

**Current state**: `eviction_tier()` already assigns tiers (0=cold MoE, 1=hot MoE, 2=dense weight, 3=hot dense). `alloc_category_priority()` returns 0-4 for different allocation categories. The gap is that eviction does not consider what is requesting the space -- it just evicts the globally coldest entry.

**Target**: When KV cache needs to grow and VRAM is full, it specifically evicts cold weights (priority 3-4), not other KV entries (priority 1). When compute scratch overflows the arena, it can reclaim from cold experts.

**Migration**: Additive change. New priority field defaults to current tier value. Existing behavior is preserved. New callers can specify higher priority to get preferential treatment.

**Risks**: Priority inversion (high-priority data evicted for low-priority requestor due to race). Mitigation: evict_below_priority takes explicit threshold, never evicts above it.

### Phase 5: KV Cache Arena Integration (4-5 days)

**Goal**: KV cache allocates from the VRAM arena instead of through separate `unified_cache_allocate()` calls.

**Changes**:

| File | Function | Change |
|------|----------|--------|
| `unified-cache.hpp` | `unified_cache` class | Add `kv_zone_` within vram_arena (bump allocator for KV) |
| `unified-cache.cpp` | `kv_alloc()` | NEW: Bump-allocate from KV zone within arena |
| `unified-cache.cpp` | `kv_free()` | Mark KV space as reclaimable (for context reset / sequence eviction) |
| `ggml-sycl.cpp` | KV buffer allocation | Replace `unified_cache_allocate(KV_CACHE)` with `cache->kv_alloc()` |
| `kv-cache-manager.cpp` | Per-layer KV | Allocate from KV zone instead of unified_cache_allocate |
| `kv-offload.cpp` | KV host offload | When KV evicted to host, mark arena space reclaimable |

**KV zone sizing**: At context creation, reserve KV zone based on `n_ctx * n_layers * kv_bytes_per_token`. This is already computed in the existing code.

**Dynamic growth**: If KV zone exhausts its initial allocation, it can grow into the weight zone by evicting cold weights. This is the key interaction between KV and weight priorities.

**Risks**:
- KV cache lifetime is different from weights (per-context, not per-model). Arena reset on context destruction must handle partial arena reclamation.
- Per-layer KV hot/cold tiering complicates zone management. Mitigation: hot layers use arena KV zone, cold layers use host-pinned pool (existing kv-offload.cpp pattern).

### Phase 6: Location-Transparent Non-Weight Access (2-3 days)

**Goal**: Extend `resolve_weight` pattern to all allocation types. Any allocation can be queried for current location.

**Changes**:

| File | Function | Change |
|------|----------|--------|
| `unified-cache.hpp` | `memory_location` struct | NEW: Generic location descriptor (ptr, tier, layout, event) |
| `unified-cache.hpp` | `query_location()` | NEW: Look up any allocation by key. Returns current tier and pointer |
| `common.hpp` | `resolve_allocation()` | NEW: Generic version of resolve_weight for any tensor/buffer |
| `unified-kernel.cpp` | Kernel dispatch | Use `query_location()` result to select GPU vs zero-copy vs CPU kernel |
| `ggml-sycl.cpp` | `ggml_sycl_compute_forward()` | Resolve all source tensors via location query before dispatch |

**Current state**: `resolve_weight` handles this for weights. The `alloc_registry` tracks pointer types. The `cache_ptr_view` struct already has `location` field. The gap is that non-weight allocations (KV, scratch) do not have cache keys and cannot be queried.

**Solution**: Allocations from `kv_alloc()` register a synthetic cache key based on (context_id, layer_id, kv_type). Scratch allocations do not need location queries (always in VRAM arena).

**Risks**: Low. This is mostly API surface extension. The underlying mechanism (alloc_registry lookup + cache query) already works.

### Phase 7: Async DMA Eviction (3-4 days, stretch goal)

**Goal**: When evicting a weight from VRAM to host, perform the DMA asynchronously so inference can continue on other data.

**Changes**:

| File | Function | Change |
|------|----------|--------|
| `unified-cache.cpp` | `evict_one()` | Instead of deferred-free, issue async D2H copy to host arena. Mark entry as EVICTING with event |
| `unified-cache.cpp` | `query_location()` | If entry is EVICTING, return host pointer + event (caller can wait or use alternative) |
| `unified-cache.cpp` | `promote_to_device()` | NEW: Async H2D copy to bring evicted data back to VRAM |

**Current state**: Eviction just frees the device pointer (deferred). The host_cache holds a separate copy if the weight was cached there. There is no async D2H during eviction -- the weight must be re-loaded from the original source (mmap) if needed again.

**Improvement**: Before freeing VRAM, copy the current device data to host arena. This preserves transformed layouts (SOA, COALESCED) that are expensive to regenerate.

**Risks**: 
- BCS queue contention during eviction DMA. Mitigation: use separate in-order queue for eviction DMA.
- Host arena capacity for evicted data. Mitigation: evict oldest-evicted entries from host when host arena is full (double eviction: VRAM -> host -> discard).

---

## 8. Risks and Mitigations

### 8.1 Critical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Single large malloc_device fails | Cannot pre-allocate arena | Medium (device-specific limits) | Fall back to 2-3 chunk strategy, then per-entry allocation |
| Fragmentation in weight zone free list | Weights cannot be loaded despite available total space | Low (weights are similarly-sized) | Compaction at safe sync points; defrag sweep between inference batches |
| Arena sizing wrong (too small for peak) | OOM during inference | Medium | Conservative initial sizing + dynamic re-reserve after first graph profiling |
| Priority inversion race | High-priority data evicted unexpectedly | Low | Explicit priority thresholds, no cross-priority eviction |
| Regression in GPU-only path performance | Unacceptable for production | Must be zero | Gate behind env var; A/B test; performance CI gate |

### 8.2 Compatibility Risks

| Risk | Mitigation |
|------|------------|
| Multi-device (tensor split): each device needs separate arena | Already PER_DEVICE mode in cache. Each device's cache gets its own arena |
| Graph replay: arena pointers change on re-reserve | Arena is reserved BEFORE graph recording. Arena_reset only moves bump pointer, not base |
| Host compute mode: arena_alloc returns VRAM, but HOST_COMPUTE needs host-pinned | HOST_COMPUTE allocations go through host arena, not VRAM arena. Check mode before routing |
| Persistent TG kernel: bakes pointers at init | Persistent scratch is in Priority 0 (never evicted). Pointers stable for kernel lifetime |

### 8.3 Performance Targets

All phases must maintain:
- PP512 >= 1336 tok/s (Mistral 7B Q4_0, Arc B580)
- TG128 >= 72 tok/s
- Zero additional synchronization points during inference
- Arena_alloc must be < 100ns (atomic bump, no mutex)

---

## 9. Sources

### Production Inference Engines
- [TensorRT-LLM Memory Usage Reference](https://nvidia.github.io/TensorRT-LLM/reference/memory.html)
- [TensorRT-LLM Optimization Guide](https://www.mintlify.com/NVIDIA/TensorRT-LLM/performance/optimization-guide)
- [NVIDIA KV Cache Early Reuse Blog](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/)
- [NVIDIA CPU-GPU Memory Sharing for KV Cache Offload](https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/)
- [vLLM PagedAttention Design](https://docs.vllm.ai/en/stable/design/paged_attention/)
- [PagedAttention Architecture Overview (Medium)](https://medium.com/@mandeep0405/the-architecture-behind-vllm-how-pagedattention-improves-memory-utilization-2f9b25272110)
- [Efficient Memory Management for LLM Serving with PagedAttention (paper)](https://arxiv.org/pdf/2309.06180)
- [vLLM: Efficient Inference Engine (Woosuk Kwon, UC Berkeley)](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf)
- [vAttention: Dynamic Memory Management without PagedAttention](https://arxiv.org/abs/2405.04437)
- [FlexGen: High-Throughput Generative Inference on Single GPU](https://arxiv.org/abs/2303.06865)
- [ZeRO-Inference: Democratizing Massive Model Inference (DeepSpeed)](https://www.deepspeed.ai/2022/09/09/zero-inference.html)
- [KTransformers: CPU/GPU Hybrid Inference for MoE (SOSP 2025)](https://dl.acm.org/doi/10.1145/3731569.3764843)
- [KTransformers GitHub](https://github.com/kvcache-ai/ktransformers/)
- [ORCA: Distributed Serving for Transformer Models](https://www.semanticscholar.org/paper/Orca:-A-Distributed-Serving-System-for-Generative-Yu-Jeong/9d7a75601e0e50dd68d40cfb8ef0e891dad797a6)
- [Sarathi-Serve: Taming Throughput-Latency Tradeoff](https://www.cse.iitd.ac.in/~rijurekha/col851/scheduling_oct23_oct25/2024_osdi-sarathiserve.pdf)

### SYCL / Intel oneAPI
- [Intel oneAPI USM Allocations Optimization Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-2/unified-shared-memory-allocations.html)
- [Intel Memory Allocation Optimization](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-0/memory-allocation.html)
- [Intel Level Zero Optimization Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/level-zero.html)
- [Intel USM Host/Device Memory Guide (2025)](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/host-device-memory-buffer-and-usm.html)
- [Level Zero Memory Allocation Limits](https://jjfumero.github.io/posts/2022/04/understanding-memory-allocation-size-limitations-with-levelzero/)
- [Intel UMF API Documentation](https://oneapi-src.github.io/unified-memory-framework/api.html)

### llama.cpp
- [llama.cpp CUDA Pool Discussion](https://github.com/ggml-org/llama.cpp/discussions/6324)
- [llama.cpp Memory Allocation Discussion](https://github.com/ggml-org/llama.cpp/discussions/9936)
