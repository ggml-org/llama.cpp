# 120B MoE Inference Optimization Plan — VTune-Driven

## VTune Profile Results (120s capture, 120B MXFP4, B580)

### Summary
- **GPU utilization: 22.6%** — GPU mostly idle
- **XVE Stalled/Idle: 42.5%** when GPU is busy
- **GPU active compute: ~0s** — GPU kernels barely fire
- **Investigation confirmed: 432 expert tensor accesses per token** (4 × 36 layers × 3 tensors)

### Top CPU Hotspots

| Function | CPU Time | Idle Time | What It Does |
|----------|----------|-----------|-------------|
| **`func@0x1fa760` (libze_intel_gpu)** | **82s** | **226s** | L0 driver: GGTT page table ops for staging alloc/free |
| **`func@0x1fa450` (libze_intel_gpu)** | **10s** | **28s** | L0 driver: related GGTT operations |
| **`__intel_avx_rep_memcpy`** | **8s** | **22s** | Staging memcpy (host → staging → device) |
| **`__alloc_pages_noprof`** | **5s** | **14s** | Kernel page faults from staging buffer pinning |
| **`vmf_insert_pfn_prot`** | **3s** | **9s** | GGTT mapping for pinned staging buffers |
| **`ggml_sycl_reorder_weight_cpu`** | **1.4s** | **3.8s** | CPU-side AOS→SOA reorder (single-threaded) |

**75% of CPU time is in Level Zero driver** — all from per-expert staging buffer alloc/free.

---

## Root Cause

Each of 432 expert cache misses per token does:
1. `sycl::malloc_device` for VRAM SOA slot → L0 GGTT mapping
2. `sycl::malloc_host` for staging buffer → L0 GGTT mapping + page pin
3. CPU AOS→SOA reorder into staging buffer → single-threaded CPU work
4. `queue.memcpy(device, staging, size)` → DMA
5. `sycl::free(staging)` → L0 GGTT unmap

Steps 2 + 5 (staging alloc/free) dominate. Each is ~0.5s of L0 driver time.

---

## Multi-GPU Budget

| Device | VRAM for Expert Cache | Expert Slots (~4.3 MB each) |
|--------|----------------------|----------------------------|
| B580 | ~9.6 GB | ~2,230 |
| B50 | ~5.7 GB | ~1,330 |
| **Total** | **~15.3 GB** | **~3,560** |

With 432 unique expert tensors per token, the cache holds ~8 tokens worth.
After 2-3 tokens, cache hit rate approaches **95-98%**.

---

## Architecture: Dual-Path Expert Dispatch

### Cache Hit (hot path — 95%+ after warmup)
```
Expert SOA already in VRAM → pure MMVQ kernel → maximum GPU throughput
```
No staging, no reorder, no L0 driver calls. Just compute.

### Cache Miss (cold path — fused reorder + compute)
```
GPU reads AOS from host-pinned via PCIe zero-copy
  → GPU reorders to SOA in registers on-the-fly during matmul
  → Simultaneously: async DMA uploads SOA copy to VRAM cache for future hits
```
**Key insight**: On a cache miss, compute starts IMMEDIATELY via fused kernel.
No waiting for CPU reorder + staging. The cache fill happens in parallel.

This means:
- **First token**: slightly slower (PCIe-bound fused kernel) but NOT blocked
- **Every subsequent access**: VRAM SOA hit → full speed
- **No cold-start penalty** — the model produces output from the first token

### Unified Cache Tracks Everything

All VRAM allocations — expert SOA slots, staging buffers, DMA transfer buffers —
go through the unified cache budget. The cache decides:
- Which experts are hot (keep in VRAM SOA)
- Which are warm (on B50 VRAM SOA)
- Which are cold (host-pinned AOS, use fused kernel on miss)
- When to evict (LFU + staleness, gate > up > down priority)

No separate staging pool or shadow allocations outside the cache's knowledge.

---

## Optimization Phases

### Phase 1: Staging Buffer Pool [OPT-1] — Eliminate L0 churn

**Problem:** 432 `sycl::malloc_host` + `sycl::free` pairs per token for staging.

**Fix:** Pre-allocate reusable staging buffer(s) at init time. Round-robin for
concurrent expert uploads. Never alloc/free during inference.

**Impact:** Eliminates 75% of CPU time (L0 driver GGTT ops).
**Effort:** Small — wire `staging_buffer_pool` into `copy_to_device_async`.

### Phase 2: GPU-Side Fused Reorder + Compute [NEW] — Zero cold-start penalty

**Problem:** CPU SOA reorder is single-threaded (~1.4s/token) and blocks compute.

**Fix:** On cache miss, GPU kernel reads AOS from host-pinned via PCIe and
reorders to SOA in registers during matmul. Simultaneously, background DMA
uploads SOA to VRAM cache for future hits.

Two MMVQ kernel variants:
- `mmvq_soa_cached` — reads SOA from VRAM (existing, fast)
- `mmvq_aos_fused` — reads AOS from host, reorders on-the-fly (new, PCIe-bound)

Dispatch selects based on cache hit/miss per expert.

**Impact:** Eliminates CPU reorder entirely. Compute starts immediately on miss.
**Effort:** Medium — new kernel variant + dispatch logic.

### Phase 3: Pre-Cache During Warmup [OPT-3] — Maximize hit rate from token 1

**Problem:** First few tokens have 0% cache hit rate.

**Fix:** After warmup profiling (8 tokens), pre-stage popular experts to VRAM
SOA cache on BOTH GPUs before real inference begins.

- B580: hottest ~2,230 expert tensors
- B50: next ~1,330 warm expert tensors
- Priority: gate > up > down (S5 already implements this)

**Impact:** ~95% cache hit rate from first real token.
**Effort:** Small — fix S1 prestage path (partially broken).

### Phase 4: Skip SOA for CPU Experts [OPT-4] — Reduce cache pressure

**Problem:** Experts routed to CPU don't need SOA conversion.

**Fix:** MoE dispatch triage checks routing BEFORE calling `ensure_cached_layout`.
CPU-routed experts use `host_src_ptr` directly (AOS). Only GPU-routed experts
get SOA in VRAM.

**Impact:** Reduces VRAM cache pressure. More room for hot experts.
**Effort:** Small — conditional in dispatch path.

### Phase 5: Multi-GPU Concurrent Dispatch — Use both GPUs

**Problem:** B50 has 5.7 GB of expert cache but may not be utilized.

**Fix:** 3-way concurrent dispatch (already implemented in S5/S6):
- B580: cached SOA experts via MMVQ
- B50: warm SOA experts via secondary dispatch
- CPU: cold experts via AVX-VNNI (or fused GPU kernel from Phase 2)

Verify multi-GPU dispatch works with `ONEAPI_DEVICE_SELECTOR="level_zero:0,1"`.

**Impact:** +74% expert cache capacity. Near-100% hit rate after warmup.
**Effort:** Verification only — dispatch code exists.

---

## Priority Order

```
Phase 1 (staging pool)     → Immediate 5-10x speedup on cold start
Phase 3 (pre-cache warmup) → Eliminate cold start entirely
Phase 4 (skip CPU SOA)     → Quick win, reduces pressure
Phase 5 (multi-GPU verify) → +74% cache, verification only
Phase 2 (fused kernel)     → Zero-penalty cache misses (medium effort)
```

## Expected Performance

| State | TG (tok/s) | Why |
|-------|-----------|-----|
| Current (no optimization) | ~1 | L0 driver churn on every expert access |
| After Phase 1 (staging pool) | ~3-5 | No alloc/free, still CPU reorder |
| After Phase 1+3 (pool + pre-cache) | ~8-12 | 95% cache hits, minimal reorder |
| After Phase 1+3+5 (+ multi-GPU) | ~10-15 | 98% hits across 15.3 GB cache |
| After Phase 1+2+3+5 (+ fused kernel) | ~15-20 | Zero cold penalty, GPU-only compute |

---

## Fixed Bugs (Prerequisites)

| Bug | Fix | Commit |
|-----|-----|--------|
| 120B inference deadlock | Deferred free in layout switch (was `queue_.wait()`) | `ac5393b8a` |
| GPU lockup after kill | SIGTERM/SIGINT/SIGHUP signal handlers free GGTT mappings | `b30e1e37b` |
| `sycl::malloc_host` hang | Cap at iGPU's 4 GB `max_mem_alloc_size`, ggml auto-chunks | `03dd78bab` |
| Host buffer staging segfault | Skip staging for host-pinned USM (direct DMA) | `78c51ffc1` |
