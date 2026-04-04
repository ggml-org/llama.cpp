# Smart Handle Architecture for SYCL Unified Memory Manager

**Status**: Design Complete — Ready for Implementation
**Date**: 2026-04-04

## Summary

Replace raw `void*` pointers held by external systems with `mem_handle` — a lookup handle that caches pointer resolution and auto-detects staleness via a global generation counter. The cache can transparently move data between VRAM and host without invalidating handles.

## Key Design Decisions

1. **No ownership/refcount** — handles are lookup-only. Pointer stability is guaranteed by existing `graph_compute_active` flag and `pin_model_weights()`.
2. **Global generation counter** — single atomic, bumped on eviction/promotion (~0-3 times per inference run). Hot path: compare + return cached ptr (~3 ns).
3. **Two handle kinds**: WEIGHT (cache-managed, gen-checked) and DIRECT (raw ptr, never stale — for scratch/KV).
4. **Per-thread handles** — no sharing, no mutex needed.

## Hot Path Performance

~2-4 ns (gen compare + cached ptr return). Same order as current `data_device[dev]` access. Zero measurable TG impact (<0.02% of token time).

## Migration Phases

- **P0**: Infrastructure (new files + generation bump points) — zero behavior change
- **P1**: Persistent TG kernel (`layer_weight_pointers` → `mem_handle`) — contained subsystem
- **P2**: `extra_gpu::data_device[]` → `data_handle[]` — ~50 call sites, incremental
- **P3**: Unified resolve function — replace 3 separate resolve chains
- **P4**: Graph replay verification — no changes needed (handles are transparent)

## Full Design

See the implementation details in the team discussion. Class definition, integration plan, edge cases, and performance analysis are comprehensive.
