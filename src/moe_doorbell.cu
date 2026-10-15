#include "moe_doorbell.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static LayerDoorbell * g_doorbells_host = nullptr;
static LayerDoorbell * g_doorbells_dev  = nullptr;
static int g_n_layer = 0;
static bool g_active = false;

__global__ void moe_slot_resolve_kernel(
    const int32_t * __restrict__ topk,
    const int32_t * __restrict__ map,
    int32_t       * __restrict__ out,
    int64_t nk,
    int64_t nt,
    size_t topk_nb0,
    size_t topk_nb1,
    size_t map_nb0,
    size_t out_nb0,
    size_t out_nb1,
    int il,
    int n_exp,
    int sentinel,
    LayerDoorbell * __restrict__ dbs) {
    const int64_t total = nk * nt;
    const int tid = (int)(threadIdx.x + blockDim.x * blockIdx.x);
    __shared__ uint32_t s_count;
    __shared__ int32_t  s_mexp[16];
    __shared__ int32_t  s_mpos[16];
    if (threadIdx.x == 0) {
        s_count = 0;
    }
    __syncthreads();
    if ((int64_t) tid < total && total <= 16) {
        const int64_t k = tid % nk;
        const int64_t t = tid / nk;
        const int32_t exp = *(const int32_t *) ((const char *) topk + t * topk_nb1 + k * topk_nb0);
        int32_t slot = sentinel;
        if (exp >= 0 && exp < n_exp) {
            slot = *(const int32_t *) ((const char *) map + (size_t) exp * map_nb0);
        }
        if (slot == sentinel) {
            const uint32_t pos = atomicAdd(&s_count, 1u);
            if (pos < 16) {
                s_mexp[pos] = exp;
                s_mpos[pos] = tid;
            }
            *(int32_t *) ((char *) out + t * out_nb1 + k * out_nb0) = sentinel;
        } else {
            *(int32_t *) ((char *) out + t * out_nb1 + k * out_nb0) = slot;
        }
    }
    __syncthreads();
    if (total > 16) {
        if ((int64_t) tid < total) {
            const int64_t k = tid % nk;
            const int64_t t = tid / nk;
            const int32_t exp = *(const int32_t *) ((const char *) topk + t * topk_nb1 + k * topk_nb0);
            int32_t slot = sentinel;
            if (exp >= 0 && exp < n_exp) {
                slot = *(const int32_t *) ((const char *) map + (size_t) exp * map_nb0);
            }
            *(int32_t *) ((char *) out + t * out_nb1 + k * out_nb0) = slot;
        }
        return;
    }
    if (s_count == 0) {
        return;
    }
    if (threadIdx.x == 0) {
        LayerDoorbell * db = &dbs[il];
        db->miss_count = s_count;
        db->k = (uint32_t) total;
        for (uint32_t i = 0; i < s_count; i++) {
            db->miss_experts[i] = s_mexp[i];
        }
        for (int64_t i = 0; i < total; i++) {
            const int64_t k = i % nk;
            const int64_t t = i / nk;
            db->topk_all[i] = *(const int32_t *) ((const char *) topk + t * topk_nb1 + k * topk_nb0);
        }
        __threadfence_system();
        if (s_count > 0) {
            db->need_host = 1;
        }
    }
}

void moe_slot_resolve_launch(
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
    int32_t il,
    int32_t n_exp,
    int32_t sentinel) {
    if (!g_active || !g_doorbells_dev) {
        return;
    }
    moe_slot_resolve_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(
        topk, table, out, nk, nt, topk_nb0, topk_nb1, table_nb0,
        out_nb0, out_nb1, il, n_exp, sentinel, g_doorbells_dev);
}

bool moe_doorbell_init(int n_layer, int n_expert, int s_per_layer) {
    (void) n_expert;
    (void) s_per_layer;
    if (g_active) return true;
    g_n_layer = n_layer;
    const size_t total_bytes = sizeof(LayerDoorbell) * (size_t) n_layer;
    cudaError_t err = cudaHostAlloc(&g_doorbells_host, total_bytes, cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        return false;
    }
    err = cudaHostGetDevicePointer(&g_doorbells_dev, g_doorbells_host, 0);
    if (err != cudaSuccess) {
        cudaFreeHost(g_doorbells_host);
        g_doorbells_host = nullptr;
        return false;
    }
    memset(g_doorbells_host, 0, total_bytes);
    ggml_cuda_moe_slot_resolve_register(moe_slot_resolve_launch);
    g_active = true;
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
