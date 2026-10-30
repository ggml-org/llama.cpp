#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

struct alignas(64) LayerDoorbell {
    volatile uint32_t state;        // 0 = IDLE, 1 = MISS_SIGNALED, 2 = SERVICED
    volatile uint32_t gpu_ready;    // 0 = WAITING, 1 = DMA_COMPLETE / UNBLOCK
    uint32_t miss_count;            // Number of cold misses
    uint32_t pad;
    int32_t  miss_experts[32];      // Expert IDs that missed VRAM
    int32_t  assigned_slots[32];    // VRAM slots assigned by host pager
    int32_t  evicted_experts[32];   // Expert IDs evicted from VRAM slots
    uint64_t selected;
    uint64_t misses;
    uint64_t decode_layers;
    uint64_t decode_selected;
    uint64_t decode_misses;
};

bool moe_doorbell_init(int n_layer, int n_expert, int s_per_layer);

struct alignas(64) DoorbellPrekick {
    volatile uint32_t state;
    uint32_t generation;
    uint32_t n;
    uint32_t compared;
    uint32_t overlap;
    int32_t experts[8];
};

DoorbellPrekick * moe_doorbell_prekick_host(int il);
void moe_doorbell_free();
bool moe_doorbell_is_active();
LayerDoorbell * moe_doorbell_get_host_ptr();
LayerDoorbell * moe_doorbell_get_dev_ptr();
void moe_doorbell_log_stats();
void moe_doorbell_signal_unblock(int il, void * stream);
bool moe_doorbell_dma_enabled();
bool moe_doorbell_dma_submit(int il, int copy_index);
const uint64_t * moe_doorbell_lru_host(int il);


void moe_doorbell_resolve_launch(
    void * stream,
    const int32_t * topk,
    const int32_t * table,
    int32_t * out,
    int64_t nk,
    int64_t nt,
    size_t topk_nb0,
    size_t topk_nb1,
    size_t table_nb0,
    size_t out_nb0,
    size_t out_nb1,
    const int32_t * full,
    int64_t fnk,
    int64_t fnt,
    size_t full_nb1,
    const int32_t * look,
    int64_t lnk,
    int32_t il,
    int32_t n_exp,
    int32_t sentinel,
    int32_t op
);

#ifdef __cplusplus
}
#endif
