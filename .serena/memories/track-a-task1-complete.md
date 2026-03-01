# Task 1: ExpertPlacement Table - COMPLETE

## What Task 1 Provides

### Data Structure: ExpertPlacement Struct
- device_id: int (-1 for CPU-only, 0+ for GPU device ordinal)
- device_ptr: void* (SOA device pointer, nullptr if CPU-only)
- host_ptr: void* (AOS host-pinned pointer, always valid after init)
- weight_bytes: size_t (per-expert weight size)
- popularity_rank: int (0=most popular, -1=unranked)
- is_valid(): bool (checks host_ptr != nullptr)

### API: ExpertPlacementTable Class
- init(n_layers, n_experts_per_layer) - initialize dimensions
- set(layer_id, expert_id, placement) - register expert placement
- get(layer_id, expert_id) - O(1) concurrent lookup (shared_lock)
- set_device_ptr(layer_id, expert_id, device_id, ptr) - update device pointer
- set_popularity(layer_id, expert_id, rank) - update popularity rank
- get_layer_experts(layer_id) - bulk read sorted by popularity
- n_layers(), n_experts(), is_initialized() - metadata queries

### Thread Safety
- Uses std::shared_mutex for TG-performance-critical concurrent reads
- Shared_lock for get() (hot path during dispatch)
- Unique_lock for set/update (initialization and profiling)

### Key Implementation Details
- 64-bit key: (layer_id << 32) | expert_id (handles FNV-hash layer_id)
- Global singleton: get_expert_placement_table() returns static instance
- O(1) lookups via std::unordered_map<int64_t, ExpertPlacement>

## Files Modified
- ggml/src/ggml-sycl/unified-cache.hpp: Types + global accessor
- ggml/src/ggml-sycl/unified-cache.cpp: Method implementations

## Downstream Consumers
- **Task 2**: MUL_MAT_ID dispatch reads placement.device_id to partition work
- **Task 4**: moe_hybrid_init_once() populates placement table during model load
- **Task 6**: Fused kernel reads placement.device_ptr for weight lookup
- **Task 5**: dispatch_experts_secondary_gpu() reads placement per expert
