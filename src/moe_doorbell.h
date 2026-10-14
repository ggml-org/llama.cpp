#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// 64-byte cache-line aligned doorbell for lockless GPU <-> Host synchronization
struct alignas(64) LayerDoorbell {
    volatile uint32_t state;        // 0 = IDLE, 1 = MISS_SIGNALED, 2 = SERVICED
    volatile uint32_t gpu_ready;    // 0 = WAITING, 1 = DMA_COMPLETE / UNBLOCK
    uint32_t miss_count;            // Number of cold misses (0..10)
    int32_t  miss_experts[10];      // Expert IDs that missed VRAM
    int32_t  assigned_slots[10];    // VRAM slots assigned by host pager
};

// Initialize doorbell system (allocates 48 doorbells in pinned mapped memory)
bool moe_doorbell_init(int n_layer, int n_expert, int s_per_layer);

// Release doorbell system
void moe_doorbell_free();

// Check if doorbell system is active
bool moe_doorbell_is_active();

// Get host-accessible pointer to doorbells
LayerDoorbell * moe_doorbell_get_host_ptr();

// Get device-accessible pointer to doorbells
LayerDoorbell * moe_doorbell_get_dev_ptr();

// Resolve slots on device with zero-split doorbell fallback
void moe_doorbell_resolve_slots(
    int layer,
    const int32_t * d_topk_experts,
    int32_t * d_out_slot_ids,
    int32_t * d_layer_slot_map,
    int n_used,
    int sentinel_slot,
    void * stream
);

#ifdef __cplusplus
}
#endif
