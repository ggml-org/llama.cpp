#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

struct alignas(64) LayerDoorbell {
    volatile uint32_t need_host;
    volatile uint32_t gpu_ready;
    uint32_t miss_count;
    uint32_t k;
    int32_t  miss_experts[16];
    int32_t  miss_pos[16];
    int32_t  topk_all[16];
    int32_t  assigned_slots[16];
};

bool moe_doorbell_init(int n_layer, int n_expert, int s_per_layer);
void moe_doorbell_free();
bool moe_doorbell_is_active();
LayerDoorbell * moe_doorbell_get_host_ptr();
LayerDoorbell * moe_doorbell_get_dev_ptr();

#ifdef __cplusplus
}
#endif
