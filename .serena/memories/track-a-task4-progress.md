# Task 4: Unified Cache Expert Registration - Progress

## Status: Phase 1 Complete (Foundation)

### Phase 1: ExpertPlacementTable Initialization & Population
- **Location**: ggml-sycl.cpp, lines 394-437
- **What works**:
  - ExpertPlacementTable initialized with n_moe_layers, n_experts_per_layer
  - All experts registered with valid host_ptr from expert_list
  - All experts start as CPU-only (device_id = -1)
  - popularity_rank unranked initially (-1)
  - Backward compatible with ExpertCache (ifdef guarded)
  
### Phase 1 Acceptance Criteria Met:
- [x] ExpertPlacementTable fully populated after moe_hybrid_init_once()
- [x] All experts have valid host_ptr
- [x] Debug logging added for placement summary
- [x] Dense Mistral 7B unaffected (no non-MoE changes)

### Phase 2: Hot Expert Upload to VRAM (Deferred)
Requires:
1. Unified cache `ensure_cached_layout()` for SOA upload
2. VRAM budget querying via dpct device API
3. Slot capacity calculation: (available_vram - reserved) / expert_bytes
4. Upload loop: populate device_ptr for hot experts
5. Feedback from team-lead on approach

## Code Changes Summary

**File: ggml/src/ggml-sycl/ggml-sycl.cpp**
- Replaced lines 394-404 (ExpertCache only path)
- Added ExpertPlacementTable init & population
- Wrapped ExpertCache in #ifdef GGML_SYCL_LEGACY_EXPERT_CACHE
- Added logging for both modes

## Unblocked Tasks:
- Task 2: Multi-device dispatch (can query placement table now)
- Task 6: Fused kernel fix (can check placement.device_ptr)
- Task 5: Activation shipping (depends on Task 3, not Task 4)

## Next Steps:
1. Wait for team-lead guidance on Phase 2 scope
2. If Phase 2 approved: Implement hot expert uploading
3. Commit Phase 1 independently if Phase 2 deferred
4. Verify with MoE model testing
