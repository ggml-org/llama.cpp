# Task 7: ExpertCache Removal and Code Cleanup

## Status: STEP 1 COMPLETE (awaiting Step 2 approval)

## High-Level Goal
Remove ExpertCache class entirely, replace with placement-table-based expert management via unified cache.

## Two-Step Implementation Plan

### Step 1: Add ifdef Guards (✅ COMPLETED)
**Objective**: Validate that ExpertCache can be safely removed by building without it.

**Completed Changes**:
- Line 223-225: Wrapped `#include "ggml-sycl/expert-cache.hpp"` in ifdef guard
- Lines 235-243: Wrapped global variables (g_expert_caches[], ggml_sycl_get_expert_cache())
- Lines 24717-24722: Wrapped expert_cache variable declarations with nullptr fallback
- Lines 24740-24747: Wrapped expert_cache->lookup() for B580 cache check
- Lines 24755-24759: Wrapped expert_cache1->lookup() for B50 cache check  
- Lines 24771-24783: Wrapped all expert_cache->update_score() calls
- Lines 25040-25044: Wrapped expert_cache_rec->record_access_batch() call

**Files to modify**:
- `ggml-sycl.cpp`: Wrap ExpertCache include and all references in `#ifdef GGML_SYCL_LEGACY_EXPERT_CACHE`
- `expert-prefetch.hpp`: Wrap ExpertCache forward declarations and member
- `expert-prefetch.cpp`: Wrap ExpertCache method implementations

**Locations in ggml-sycl.cpp**:
- Line 223: `#include "ggml-sycl/expert-cache.hpp"`
- Line 233: `static ggml_sycl::ExpertCache g_expert_caches[GGML_SYCL_MAX_DEVICES]`
- Line 234-240: Functions like `ggml_sycl_get_expert_cache()`, `ggml_sycl_init_expert_cache()`
- Line 25062-25063: Variables like `auto * expert_cache` and `auto * expert_cache1`
- Other ExpertCache method calls (use grep to find all)

**Build Command**:
```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build
```
Must compile without errors. If it fails, we know ExpertCache is still being used somewhere.

**Commit**: `git commit -m "sycl: ifdef-guard ExpertCache code for removal validation"`

### Step 2: Delete and Refactor
**Objective**: Remove ExpertCache entirely and migrate ExpertPrefetcher to use unified cache.

**Files to delete**:
- `ggml/src/ggml-sycl/expert-cache.cpp`
- `ggml/src/ggml-sycl/expert-cache.hpp`

**Files to modify**:
1. **extract-first**: Extract PinnedBufferPool (expert-cache.hpp lines 316-362) to new file `pinned-buffer-pool.hpp`
2. **ggml-sycl.cpp**: Remove all `#ifdef GGML_SYCL_LEGACY_EXPERT_CACHE` blocks
3. **expert-prefetch.hpp**: 
   - Remove ExpertCache * cache_ member
   - Add reference to unified_cache (or global singleton accessor)
4. **expert-prefetch.cpp**:
   - `init()`: Instead of `cache_->init()`, nothing needed (unified cache auto-initializes)
   - `hint()`: Replace `cache_->prefetch_async()` with `unified_cache::ensure_cached_layout()`
   - `await()`: Replace `ExpertCache::event` with `cache_layout_result.event`

**ExpertPredictor**: LEAVE UNCHANGED — already independent of ExpertCache

**Commit**: `git commit -m "sycl: remove ExpertCache, refactor ExpertPrefetcher to unified cache"`

## Critical Notes
- PinnedBufferPool is used by `dispatch_cpu_and_scatter()` for activation/output staging
- ExpertPredictor uses only internal state (last_experts_, freq_table_, gate_weight_ptrs_) — NO ExpertCache dependency
- CpuExpertPool is already in separate file (cpu-expert-pool.hpp) — no extraction needed
- CMakeLists.txt uses glob pattern `file(GLOB ...)` — no edit needed when .cpp file deleted

## Acceptance Criteria
- [ ] expert-cache.cpp and expert-cache.hpp deleted
- [ ] ExpertPrefetcher uses unified cache ensure_cached_layout()
- [ ] ExpertPredictor unchanged
- [ ] All g_expert_caches[] references removed
- [ ] No compile/link errors
- [ ] Correctness verified with deterministic test (llama-completion on GPT-OSS 120B)
- [ ] No performance regression on Mistral 7B (non-MoE)

## Blocked Tasks
- Task 8: Integration tests (depends on Task 7 completion)

## Pending Decisions
1. PinnedBufferPool location: new file vs inline in ggml-sycl.cpp?
2. ExpertPrefetcher cache reference: global singleton vs direct?
3. Build environment: proceed with edits despite bash issues?
