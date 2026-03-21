# 120B Phase 2 Optimization — Eliminate Layout Churn + Pipeline Multi-GPU

## VTune Profile (Post-Phase 1 Optimizations)

| Function | CPU Time | Idle Time | What |
|----------|----------|-----------|------|
| `func@0x1fa760` (L0 driver) | 78.6s | 227s | ensure_cached_layout SOA conversion + H2D |
| `[vmlinux]` | 19.5s | 56s | Page faults from SOA staging |
| `__intel_avx_rep_memcpy` | 7.5s | 21.7s | Staging memcpy |
| `ggml_sycl_reorder_weight_cpu` | 4.8s | 13.9s | CPU AOS→SOA reorder during prestage |
| GPU utilization | — | — | **10.5%** (down from 22.6%) |

## Root Causes

### 1. Warmup prestage does CPU SOA reorder (13.9s)

`moe_prestage_popular_experts` Phase 1 converts 2485 popular experts AOS→SOA
using `ggml_sycl_reorder_weight_cpu` — **single-threaded CPU** at ~1.9 ms/expert.
This runs during warmup, before any tokens are generated.

The prestage explicitly sets `fill_fn = ggml_sycl_fill_reordered_host` at line 913,
which calls `ggml_sycl_reorder_weight_cpu` internally.

### 2. S1-PRELOAD fills VRAM with AOS, prestage must evict+replace

S1-PRELOAD uploads 291 dense weights as AOS (2.2 GB). Then prestage evicts
AOS entries to make room for SOA experts. This is VRAM thrash — two layout
versions of the same data competing for space.

### 3. Layout policy forces AOS for large MoE

`ggml_sycl_select_moe_graph_layout` at line 7174 returns `GGML_LAYOUT_AOS`
for models with >64 experts in S1 mode. This was a workaround for the old
staging alloc/free problem (now fixed by OPT-1). The override blocks SOA
dispatch for MoE experts.

### 4. No GPU0-B50 pipelining

Each MoE layer waits for all expert results before proceeding. GPU0 stalls
while B50 processes secondary experts. No overlap between layers.

---

## Fix Plan

### Phase 2A: SOA-only preload (eliminate AOS from VRAM entirely)

**Problem:** VRAM has AOS from S1-PRELOAD, then prestage evicts+replaces with SOA.

**Fix:**
1. Change S1-PRELOAD to upload dense weights as **SOA** (not AOS)
   - In `graph_preload_weights`, change `is_decode ? GGML_LAYOUT_SOA : GGML_LAYOUT_AOS`
     to always use SOA
   - The PP oneDNN path can dequantize from SOA (fixed in OPT-FUSED commit)
2. Remove `GGML_LAYOUT_AOS` override in `ggml_sycl_select_moe_graph_layout` for S1 mode
   - Let the layout policy select SOA for MXFP4 MoE
3. Prestage popular experts as SOA directly — no AOS→SOA conversion needed

**Impact:** Eliminates VRAM layout churn. One layout (SOA) everywhere.

### Phase 2B: GPU-side SOA reorder for prestage

**Problem:** `moe_prestage_popular_experts` uses CPU for SOA reorder (13.9s for 2485 experts).

**Fix:** Use the GPU to do AOS→SOA reorder:
1. Upload AOS expert data from host-pinned to VRAM temp buffer via DMA
2. GPU kernel reorders AOS→SOA in VRAM (device bandwidth ~276 GB/s vs CPU DDR5 ~70 GB/s)
3. Result is SOA in VRAM cache — no CPU involvement

This is ~4x faster than CPU reorder AND doesn't block the CPU for inference prep.

Alternative: Multi-threaded CPU reorder using TBB parallel_for (simpler, ~4x speedup from 18 cores).

### Phase 2C: Pipeline multi-GPU dispatch across layers

**Problem:** GPU0 stalls while B50 processes experts. No layer overlap.

**Fix:** Double-buffer the expert dispatch:
- While B50 computes layer N's experts, GPU0 starts layer N+1's attention
- B50's results for layer N arrive via DMA and merge with GPU0's MoE output
- GPU0 never stalls waiting for B50

This requires decoupling the MoE output merge from the attention compute:
```
Layer N:  GPU0[attn] GPU0[experts_cached] B50[experts_warm]  CPU[experts_cold]
Layer N+1: GPU0[attn]←──merge(N)───────↗  B50[experts_warm]  CPU[experts_cold]
```

### Phase 2D: Batch expert kernel launches

**Problem:** 432 separate kernel launches per token (36 layers × 3 tensors × 4 experts).

**Fix:** The existing fused gate+up path already batches gate and up into one dispatch.
Extend to batch all 4 experts per tensor type per layer:
- 1 kernel launch per (layer, tensor_type) instead of 4 separate launches
- Reduces kernel launch overhead from 432 to 108 per token

---

## Priority

```
Phase 2A (SOA-only preload): Quick, eliminates layout churn → big impact
Phase 2B (GPU SOA reorder):  Medium, eliminates CPU reorder bottleneck
Phase 2C (pipeline dispatch): Complex, overlaps GPU0+B50 → throughput increase
Phase 2D (batch kernels):    Medium, reduces launch overhead → latency decrease
```

## Expected Performance

| State | TG (tok/s) |
|-------|-----------|
| Current (post Phase 1) | <0.5 |
| After 2A (SOA-only) | ~2-3 |
| After 2A+2B (GPU reorder) | ~5-8 |
| After 2A+2B+2C (pipeline) | ~10-15 |
| After all (+ batch) | ~15-20 |
