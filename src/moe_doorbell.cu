#include "moe_doorbell.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static LayerDoorbell * g_doorbells_host = nullptr;
static LayerDoorbell * g_doorbells_dev  = nullptr;
static int g_n_layer = 0;
static int g_n_expert = 0;
static int g_s_per_layer = 0;
static bool g_active = false;

__global__ void moe_resolve_slots_kernel(
    int layer,
    const int32_t * __restrict__ topk_experts,  // [n_used]
    int32_t       * __restrict__ out_slot_ids,  // [n_used]
    int32_t       * __restrict__ slot_map,      // [n_expert]
    LayerDoorbell * __restrict__ doorbells,     // [n_layer]
    int n_used,
    int sentinel_slot
) {
    const int tid = threadIdx.x; // [0..n_used-1]
    if (tid >= n_used) return;

    LayerDoorbell & db = doorbells[layer];
    const int32_t exp_id = topk_experts[tid];

    int32_t slot = -1;
    if (exp_id >= 0 && exp_id < 512) {
        slot = slot_map[exp_id];
    }

    __shared__ int s_miss_idx[10];
    __shared__ int s_miss_exp[10];
    __shared__ int s_miss_count;

    if (tid == 0) {
        s_miss_count = 0;
    }
    __syncthreads();

    // Check for miss
    if (slot < 0) {
        int m_pos = atomicAdd(&s_miss_count, 1);
        if (m_pos < 10) {
            s_miss_idx[m_pos] = tid;
            s_miss_exp[m_pos] = exp_id;
        }
    }
    __syncthreads();

    // If no misses, all experts are resident! Pure 2 us on-device resolution!
    if (s_miss_count == 0) {
        out_slot_ids[tid] = (slot >= 0) ? slot : sentinel_slot;
        return;
    }

    // Cold miss detected: signal doorbell to host pager thread
    if (tid == 0) {
        db.miss_count = s_miss_count;
        for (int i = 0; i < s_miss_count && i < 10; ++i) {
            db.miss_experts[i] = s_miss_exp[i];
        }
        __threadfence_system();
        db.state = 1; // Signal MISS_SIGNALED to host

          // Spin on nanosleep until host DMA completes and writes gpu_ready = 1
          while (db.gpu_ready == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
              asm volatile("nanosleep.u32 %0;" :: "r"(250));
#endif
          }
        __threadfence_system();
    }
    __syncthreads();

    // Host has serviced the miss and populated db.assigned_slots
    if (tid < s_miss_count && tid < 10) {
        int target_exp = s_miss_exp[tid];
        int new_slot = db.assigned_slots[tid];
        if (target_exp >= 0 && target_exp < 512) {
            slot_map[target_exp] = new_slot;
        }
    }
    __syncthreads();

    // Re-read slot and assign to output
    if (slot < 0) {
        out_slot_ids[tid] = (exp_id >= 0 && exp_id < 512) ? slot_map[exp_id] : sentinel_slot;
    } else {
        out_slot_ids[tid] = slot;
    }

    // Reset doorbell for next pass
    if (tid == 0) {
        db.gpu_ready = 0;
        db.state = 0;
        __threadfence_system();
    }
}

bool moe_doorbell_init(int n_layer, int n_expert, int s_per_layer) {
    if (g_active) return true;

    g_n_layer = n_layer;
    g_n_expert = n_expert;
    g_s_per_layer = s_per_layer;

    size_t total_bytes = sizeof(LayerDoorbell) * (size_t) n_layer;
    cudaError_t err = cudaHostAlloc(&g_doorbells_host, total_bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: cudaHostAlloc failed for doorbells: %s\n", __func__, cudaGetErrorString(err));
        return false;
    }

    err = cudaHostGetDevicePointer(&g_doorbells_dev, g_doorbells_host, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: cudaHostGetDevicePointer failed: %s\n", __func__, cudaGetErrorString(err));
        cudaFreeHost(g_doorbells_host);
        g_doorbells_host = nullptr;
        return false;
    }

    memset(g_doorbells_host, 0, total_bytes);
    g_active = true;
    fprintf(stderr, "%s: Stream-Ordered Doorbell initialized (%d layers, 64B cacheline-aligned)\n",
            __func__, n_layer);
    return true;
}

void moe_doorbell_free() {
    if (!g_active) return;
    if (g_doorbells_host) {
        cudaFreeHost(g_doorbells_host);
        g_doorbells_host = nullptr;
        g_doorbells_dev = nullptr;
    }
    g_active = false;
}

bool moe_doorbell_is_active() {
    return g_active;
}

LayerDoorbell * moe_doorbell_get_host_ptr() {
    return g_doorbells_host;
}

LayerDoorbell * moe_doorbell_get_dev_ptr() {
    return g_doorbells_dev;
}

void moe_doorbell_resolve_slots(
    int layer,
    const int32_t * d_topk_experts,
    int32_t * d_out_slot_ids,
    int32_t * d_layer_slot_map,
    int n_used,
    int sentinel_slot,
    void * stream
) {
    if (!g_active || !d_topk_experts || !d_out_slot_ids || !d_layer_slot_map || !g_doorbells_dev) {
        return;
    }
    cudaStream_t s = (cudaStream_t) stream;
    moe_resolve_slots_kernel<<<1, n_used, 0, s>>>(
        layer, d_topk_experts, d_out_slot_ids, d_layer_slot_map, g_doorbells_dev, n_used, sentinel_slot
    );
}
