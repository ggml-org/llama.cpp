# Unified Tiered Weight Architecture (Mar 2026)

## Decision
ALL weights (dense + expert) flow through the unified cache as single source of truth.
ExpertCache (expert-cache.cpp) to be eliminated — its responsibilities absorbed by unified cache.

## Tiers
- **Tier 0: HOST-PINNED** (canonical store) — `sycl::malloc_host` via `pinned_chunk_pool` (8GB chunks)
  - Layout: AOS (CPU-optimal, native ggml block format)
  - Budget: 90% system RAM (~115 GB on 128 GB)
  - CPU dispatch reads directly from this tier
  - Fallback: mmap alias when pinned budget exhausted
- **Tier 1: GPU VRAM** (hot cache) — `sycl::malloc_device`
  - Layout: SOA (GPU-optimal, coalesced access)
  - Contents: all dense weights + predicted hot experts
  - Eviction: LFU+staleness, dense pinned, experts by popularity

## Why AOS for CPU, SOA for GPU
- CPU vec_dot: sequential block access, scale+quants in same 64B cache line
- AVX-VNNI `_mm256_dpbssd_epi32` works directly on contiguous 32-byte blocks
- GPU: 32+ threads read same field from different blocks simultaneously (SOA = coalesced)
- Industry standard: KTransformers, MoE-Infinity, Fiddler all use dual-layout storage

## Current State (gaps)
- Dense weights: already go through unified cache host_cache → device cache ✓
- Expert weights: bypass unified cache entirely, use separate ExpertCache with mmap pointers ✗
  - ExpertCache registers mmap ptrs (expert-cache.cpp:187)
  - CPU dispatch reads raw mmap (ggml-sycl.cpp:24254)
  - ExpertCache H2D from mmap (expert-cache.cpp:296)

## Existing Infrastructure
- `host_cache::ensure_cached_alloc()` — already supports expert_id tracking (unified-cache.hpp:342)
- `pinned_chunk_pool` — 8GB chunks avoid Level Zero ~1005 allocation limit
- `cache_entry_type::MOE_EXPERT` — already defined
- `prestage_routed_experts()` — partially wired (unified-cache.cpp:6135)
- Fallback chain: pinned → unpinned → mmap alias → nullptr (unified-cache.cpp:1526-1577)

## Pinned Memory Limits (Linux)
- Requires: `memlock unlimited` in /etc/security/limits.conf
- Safe budget: 60-80 GB on 128 GB system
- `vm.max_map_count=524288` for large models
- Pinned vs mmap CPU bandwidth: SAME (both in DRAM). Advantage: no page faults, 3x GPU DMA speed.

## Plan doc
See: docs/plans/2026-03-01-moe-expert-distribution.md
