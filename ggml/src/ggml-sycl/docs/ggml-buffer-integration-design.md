# ggml Buffer Integration Design — Cache-Backed SYCL Buffers

**Status**: Design Complete — Ready for Implementation
**Date**: 2026-04-04
**Epic**: llama.cpp-4ceif

## Summary

Replace the SYCL backend's buffer implementation so the unified cache IS the ggml buffer. `alloc_buffer` sub-allocates from the arena's weight zone. Zero changes to shared ggml code.

## Key Constraint: Contiguity Invariant

ggml's `ggml_tallocr_alloc` (ggml-alloc.c:90) computes `tensor->data = (char*)get_base(buffer) + offset`. The buffer MUST provide a contiguous address range. The arena's weight zone satisfies this — it's a sub-range of a single `sycl::malloc_device` allocation.

## Buffer Method Mapping

| ggml Method | Cache-Backed Implementation |
|-------------|---------------------------|
| alloc_buffer(size) | arena.zone_alloc(WEIGHT, size) → contiguous VRAM |
| get_base() | Returns arena sub-allocation pointer |
| init_tensor() | Creates extras, registers with cache |
| set_tensor(data) | Staged H2D memcpy to arena location |
| get_tensor(data) | D2H memcpy from arena or layout pool |
| free_buffer() | weight_reclaim() → arena free-list |
| get_max_size() | Returns arena weight zone capacity |
| clear(value) | stream->memset on arena region |

## Phases

1. **P1**: Arena-backed alloc_buffer (5 sub-tasks, ~80 lines)
2. **P2**: Arena reservation timing (2 sub-tasks, ~20 lines)
3. **P3**: S1-PRELOAD adaptation (3 sub-tasks, ~40 lines)
4. **P4**: Integration testing

## Critical Risk: 2x VRAM

AOS data in arena + SOA in layout pool = 2x VRAM per weight. Mitigated by reclaiming arena AOS space after layout conversion completes.
