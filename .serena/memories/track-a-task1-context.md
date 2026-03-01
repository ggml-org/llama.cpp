# Track A - Task 1: ExpertPlacement Table Context

## Current MoE Infrastructure
1. **ExpertCache** (expert-cache.hpp/cpp): Manages expert VRAM caching at granular level
   - Uses `expert_key` struct (layer, expert_id pair)
   - ExpertSlot stores VRAM placement info
   - Methods: register_expert(), lookup(), ensure_cached(), prefetch_async()
   
2. **moe_hybrid_init_once()** (ggml-sycl.cpp:289)
   - Called once per device via std::call_once
   - Scans MUL_MAT_ID nodes to discover MoE architecture
   - Registers all experts with ExpertCache
   - Initializes ExpertPrefetcher and ExpertPredictor

3. **Multi-GPU MoE Dispatch** (ggml-sycl.cpp:257-269)
   - g_moe_multi_gpu_active flag
   - g_gpu1_staging_buffer for B50 outputs (malloc_host)
   - g_gpu1_queue for B50 compute submissions

## Task 1 Status: IMPLEMENTATION COMPLETE

### ExpertPlacement Struct (unified-cache.hpp, after line 282)
```cpp
struct ExpertPlacement {
    int    device_id      = -1;      // 0..n_gpu-1 for GPU, -1 = CPU-only
    void * device_ptr     = nullptr; // SOA device pointer (nullptr if CPU-only)
    void * host_ptr       = nullptr; // AOS host-pinned pointer (always valid after init)
    size_t weight_bytes   = 0;       // Per-expert weight size in bytes
    int    popularity_rank = -1;     // 0 = most popular, -1 = unranked
    bool   is_valid() const { return host_ptr != nullptr; }
};
```

### ExpertPlacementTable Class (unified-cache.hpp, after line 282)
- Methods: init(), set(), get(), set_device_ptr(), set_popularity(), get_layer_experts()
- Thread-safe with shared_mutex for concurrent reads (O(1) lookup)
- 64-bit key: (layer_id << 32) | expert_id
- Global accessor: get_expert_placement_table()

### Implementation (unified-cache.cpp, before "OneDNN FP16 Scratch Buffer" section)
- ~60 lines of method implementations
- Proper mutex locking patterns
- get_layer_experts() returns sorted by popularity_rank

Files modified:
- ggml/src/ggml-sycl/unified-cache.hpp (added types + global accessor)
- ggml/src/ggml-sycl/unified-cache.cpp (added implementations ~60 lines)

Awaiting build verification before commit.
