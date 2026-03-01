# fit_params vs Unified Cache: Architectural Options

## The Overlap Problem

| What | fit_params does | Unified cache does |
|------|----------------|-------------------|
| Weights don't fit VRAM | Moves layers to **CPU backend** | Evicts to **pinned host** (same SYCL backend) |
| KV cache too big | **Reduces context** (32K→4K) | **KV host offload** (GGML_SYCL_KV_HOST=1) |
| Budget awareness | Probes `ggml_backend_dev_memory()` | Tracks internally via budget system |
| Compute dispatch | CPU backend computes offloaded layers | GPU streams from host + computes on GPU |

They're solving the **same problem** with **incompatible strategies**. fit_params splits the model across two backends. Unified cache keeps everything on one backend with tiered storage. Mixing them causes conflicts (the hang we just fixed, the OOM during real load with reduced n_gpu_layers + streaming activation).

## Options

### Option A: Bypass fit_params entirely — auto-enable KV host offload

When the model exceeds VRAM budget, skip fit_params and let unified cache handle everything:
- **Weights**: Unified cache tiers them (VRAM -> pinned host -> mmap)
- **KV cache**: Auto-enable `KV_HOST=1` when budget pressure detected
- **Context**: Stays at full size (32K) since KV is on host
- **Compute**: All layers dispatched to SYCL, weights streamed from host as needed

**Pros**: Cleanest architecture. Zero overlap. Already mostly works with `-fit off`. Full context preserved.
**Cons**: GPU streaming for every excess layer = slower than CPU compute for those layers. At 30% budget: 2.8 tok/s (streaming) vs ~14 tok/s (fit_params + CPU).

### Option B: Bypass fit_params, auto-size context from remaining VRAM

Same as A but size the KV cache to fit in remaining VRAM after weights load:
- Unified cache loads weights (tiered)
- Query `unified_cache_available_for_compute()` -> allocate KV to fit
- No probing, no CPU backend layers

**Pros**: KV stays on device (faster), no streaming overhead for KV access.
**Cons**: Context gets reduced (maybe severely under heavy budget pressure). Still need streaming for excess weights.

### Option C: Hybrid — unified cache for weights, fit_params only for context sizing

Strip fit_params down to just context sizing. Remove its layer offloading entirely:
- All layers always on SYCL backend (unified cache tiers them)
- fit_params only queries: "given this model + weight tiers, how much VRAM is left for KV?"
- Adjusts `n_ctx` accordingly
- No `n_gpu_layers` manipulation

**Pros**: Best of both — weights handled by cache, context intelligently sized.
**Cons**: Still need a probe step (lighter weight though — no model loading, just budget query).

### Option D: Let the user control it — disable fit_params on SYCL, document env vars

Simplest approach:
- Auto-detect SYCL unified cache -> skip fit_params entirely
- User controls behavior via `GGML_SYCL_VRAM_BUDGET_PCT`, `GGML_SYCL_KV_HOST`, etc.
- Unified cache handles everything

**Pros**: Zero code overlap. Immediate fix.
**Cons**: Loses the "just works" automatic sizing. User needs to know env vars.

## Recommendation

**Option A** is the cleanest architecturally and aligns with the "unified cache owns all memory" principle. The 2.8 vs 14 tok/s gap at extreme pressure (30% budget) is real, but:
1. That's an extreme case (30% budget = artificially constrained)
2. Layer prefetch (double-buffered async streaming) would close much of that gap
3. Full 32K context is preserved (huge UX win)
4. Zero architectural conflict

The immediate implementation would be: detect SYCL unified cache -> skip fit_params -> auto-enable KV host when budget tight.

## Performance Reference (Mistral 7B Q4_0, Arc B580)

| Scenario | PP512 | TG128 | Notes |
|----------|-------|-------|-------|
| Default (all VRAM) | ~1242 | ~70.5 | Baseline, no budget pressure |
| 30% budget, fit_params + CPU | ~269 | ~14 | fit_params offloads 17 layers to CPU |
| 30% budget, streaming (-fit off) | N/A | ~2.8 | All layers on SYCL, excess streamed |
| 30% budget, forced streaming | N/A | ~7.0 | FORCE_STREAMING=1 |

## Current State (Feb 10, 2026)

- **Fixed**: fit_params no longer hangs with VRAM_BUDGET_PCT < 100 (no_alloc guard)
- **New issue**: fit_params + reduced n_gpu_layers + streaming activation = OOM/hang during real load
  - Root cause: inventory counts ALL tensors (GPU+CPU), triggers streaming for CPU-bound layers
  - Partial fix applied: filter inventory to SYCL buffer types only
- **Beads**: llama.cpp-io0q (in_progress)
