#include "moe_doorbell.h"
#include <atomic>
#include "ggml-cuda.h"

extern "C" {
    void ggml_cuda_moe_slot_resolve_register(ggml_cuda_moe_slot_resolve_fn_t fn);
}
#include <cuda_runtime.h>
#include <cuda.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Per-layer batch plan. It is host-mapped so the kernel can publish the full
// top-k list and the host pager can answer without a graph split.
// Keep this layout in sync with the copy in llama-context.cpp.
#define MOE_DOORBELL_PLAN_MAX 256

struct DoorbellPlan {
    volatile uint32_t state;
    volatile uint32_t gpu_ready;
    uint32_t n;
    uint32_t n_miss;
    uint32_t n_copy;
    uint32_t stage_stride;
    uint32_t gate_bytes;
    uint32_t up_bytes;
    uint32_t down_bytes;
    uint32_t s_pl;
    uint32_t pad0;
    uint32_t pad1;
    int32_t  sel[MOE_DOORBELL_PLAN_MAX];
    int32_t  assigned[MOE_DOORBELL_PLAN_MAX];
    int32_t  evicted[MOE_DOORBELL_PLAN_MAX];
    int32_t  copy_slot[MOE_DOORBELL_PLAN_MAX];
    uint32_t copy_pre[MOE_DOORBELL_PLAN_MAX];
    volatile uint32_t copy_ready[MOE_DOORBELL_PLAN_MAX];
    uint64_t dst_gate;
    uint64_t dst_up;
    uint64_t dst_down;
    uint64_t stage;
};

static LayerDoorbell * g_doorbells_host = nullptr;
static LayerDoorbell * g_doorbells_dev  = nullptr;
static DoorbellPlan  * g_plans_host     = nullptr;
static DoorbellPlan  * g_plans_dev      = nullptr;
static DoorbellPrekick * g_prekick_host = nullptr;
static DoorbellPrekick * g_prekick_dev = nullptr;
static char * g_stage_host = nullptr;
static char * g_stage_dev  = nullptr;
static size_t g_stage_stride = 0;
static int g_stage_cap = 0;
static int g_n_layer = 0;
static bool g_active = false;
static bool g_count_enabled = false;
static uint64_t g_logged_selected = 0;
static cudaStream_t g_stream_h2d = nullptr;
static uint32_t * g_dma_done = nullptr;
static uint32_t * g_dma_one = nullptr;
static int g_device = 0;
static int g_n_expert = 0;
static uint64_t * g_lru_host = nullptr;
static uint64_t * g_lru_dev = nullptr;
static uint64_t * g_lru_epoch = nullptr;

__device__ __forceinline__ uint32_t pci_ld_u32(const volatile uint32_t * addr) {
    uint32_t val;
    asm volatile("ld.sys.relaxed.u32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
}

__device__ __forceinline__ uint64_t pci_ld_u64(const volatile uint64_t * addr) {
    uint64_t val;
    asm volatile("ld.sys.relaxed.u64 %0, [%1];" : "=l"(val) : "l"(addr));
    return val;
}

__device__ __forceinline__ void pci_st_u32(volatile uint32_t * addr, uint32_t val) {
    asm volatile("st.sys.relaxed.u32 [%0], %1;" :: "l"(addr), "r"(val));
}

__device__ __forceinline__ void doorbell_spin() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("nanosleep.u32 %0;" :: "r"(250));
#endif
}

// Copy one staged expert slice from pinned host memory to the pool. The worker
// already read the bytes; this only moves them, so it needs no CUDA API on the
// host and cannot deadlock against the launch queue.
__device__ __forceinline__ void doorbell_copy(char * dst, const char * src, size_t n, int tid, int nthr) {
    size_t i;
    if ((((uintptr_t) dst | (uintptr_t) src) & 15u) == 0) {
        const size_t step = (size_t) nthr * 16;
        i = (size_t) tid * 16;
        #pragma unroll 4
        for (; i + 16 <= n; i += step) {
            *(uint4 *) (dst + i) = *(const uint4 *) (src + i);
        }
    } else if ((((uintptr_t) dst | (uintptr_t) src) & 3u) == 0) {
        const size_t step = (size_t) nthr * 4;
        i = (size_t) tid * 4;
        for (; i + 4 <= n; i += step) {
            *(uint32_t *) (dst + i) = *(const uint32_t *) (src + i);
        }
    } else {
        i = (size_t) tid;
    }
    for (size_t j = i; j < n; j += (size_t) nthr) {
        dst[j] = src[j];
    }
}

// Resolve top-k experts to pool slots on the compute stream. On a hit this is
// pure device work. On a cold miss the kernel publishes the whole batch plan,
// rings the host doorbell and spins until the pager filled the missing slots.
__global__ void moe_resolve_slots_kernel(
    int layer,
    const char * __restrict__ topk,
    char       * __restrict__ out,
    int32_t    * __restrict__ slot_map,
    LayerDoorbell * __restrict__ doorbells,
    DoorbellPlan  * __restrict__ plans,
    DoorbellPrekick * __restrict__ prekick_plans,
    int nk,
    int nt,
    size_t topk_nb0,
    size_t topk_nb1,
    size_t out_nb0,
    size_t out_nb1,
    int n_exp,
    int sentinel,
    int total,
    int op,
    bool count_enabled,
    uint32_t * dma_done,
    bool use_dma,
    uint64_t * lru_touch,
    uint64_t * lru_epoch
) {
    const int tid = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    LayerDoorbell & db = doorbells[layer];
    DoorbellPlan  & pl = plans[layer];

    __shared__ int s_miss;
    __shared__ int s_ready;
    __shared__ int32_t s_cslot;
    __shared__ uint32_t s_cpre;
    __shared__ uint32_t s_ncopy, s_stride, s_gb, s_ub, s_db, s_spl;
    __shared__ uint64_t s_stage, s_dgate, s_dup, s_ddown;
    if (tid == 0) {
        s_miss = 0;
    }
    __syncthreads();

    int32_t exp  = -1;
    int32_t slot = sentinel;
    if (tid < total) {
        const int k = tid % nk;
        const int t = tid / nk;
        exp = *(const int32_t *) (topk + (size_t) t * topk_nb1 + (size_t) k * topk_nb0);
        if (exp >= 0 && exp < n_exp) {
            slot = slot_map[exp];
        }
        if (slot < 0 || slot >= sentinel) {
            slot = sentinel;
            atomicAdd(&s_miss, 1);
        }
    }
    __syncthreads();

    if (tid == 0 && lru_touch && op != GGML_OP_MOE_SLOT_RESOLVE_WAIT) {
        const uint64_t epoch = ++lru_epoch[layer];
        for (int selected = 0; selected < total; ++selected) {
            const int selected_rank = selected % nk;
            const int selected_token = selected / nk;
            const int32_t expert = *(const int32_t *) (topk + (size_t) selected_token * topk_nb1 + (size_t) selected_rank * topk_nb0);
            if (expert >= 0 && expert < n_exp) {
                uint64_t * address = lru_touch + (size_t) layer * n_exp + expert;
                asm volatile("st.sys.relaxed.u64 [%0], %1;" :: "l"(address), "l"(epoch));
            }
        }
        __threadfence_system();
    }

    if (tid == 0 && op != GGML_OP_MOE_SLOT_RESOLVE_WAIT) pl.pad0 = use_dma ? 1 : 0;
    if (tid == 0 && count_enabled && op != GGML_OP_MOE_SLOT_RESOLVE_WAIT) {
        db.selected += (uint64_t) total;
        db.misses += (uint64_t) s_miss;
        if (nt == 1) {
            db.decode_layers++;
            db.decode_selected += (uint64_t) total;
            db.decode_misses += (uint64_t) s_miss;
        }
    }

    if (tid == 0 && op == GGML_OP_MOE_SLOT_RESOLVE_KICK) {
        DoorbellPrekick & pre = prekick_plans[layer];
        if (pre.n > 0) {
            for (uint32_t pred = 0; pred < pre.n; ++pred) {
                for (int selected = 0; selected < total; ++selected) {
                    const int32_t actual = *(const int32_t *) (topk + (size_t) selected * topk_nb0);
                    if (pre.experts[pred] == actual) {
                        pre.overlap++;
                        break;
                    }
                }
            }
            pre.compared += pre.n;
            pci_st_u32(&pre.state, 0);
        }
    }

    if (s_miss == 0) {
        if (tid < total) {
            const int k = tid % nk;
            const int t = tid / nk;
            *(int32_t *) (out + (size_t) t * out_nb1 + (size_t) k * out_nb0) = slot;
        }
        return;
    }

    if (tid == 0 && op != GGML_OP_MOE_SLOT_RESOLVE_WAIT) {
        pci_st_u32(&db.gpu_ready, 0);
        db.miss_count = (uint32_t) s_miss;
        pl.n      = (uint32_t) total;
        pl.n_miss = (uint32_t) s_miss;
        pl.n_copy = 0;
        for (int i = 0; i < total; ++i) {
            const int k = i % nk;
            const int t = i / nk;
            pl.sel[i] = *(const int32_t *) (topk + (size_t) t * topk_nb1 + (size_t) k * topk_nb0);
            pci_st_u32(&pl.copy_ready[i], 0);
            if (dma_done) dma_done[layer * MOE_DOORBELL_PLAN_MAX + i] = 0;
        }
        __threadfence_system();
        pci_st_u32(&db.state, 1);
        __threadfence_system();
    }

    if (op == GGML_OP_MOE_SLOT_RESOLVE_KICK) {
        if (tid < total) {
            const int selected = tid % nk;
            const int token = tid / nk;
            *(int32_t *) (out + (size_t) token * out_nb1 + (size_t) selected * out_nb0) = slot;
        }
        return;
    }

    // Every thread spins so the block stays together. The host stages the
    // missing expert bytes in pinned memory and sets gpu_ready. Each expert is
    // copied as soon as its read lands, so the copy overlaps the remaining I/O.
    while (pci_ld_u32(&db.gpu_ready) == 0) {
        doorbell_spin();
    }
    __threadfence_system();
    __syncthreads();

    if (tid == 0) {
        s_ncopy  = pci_ld_u32(&pl.n_copy);
        s_stride = pci_ld_u32(&pl.stage_stride);
        s_gb     = pci_ld_u32(&pl.gate_bytes);
        s_ub     = pci_ld_u32(&pl.up_bytes);
        s_db     = pci_ld_u32(&pl.down_bytes);
        s_spl    = pci_ld_u32(&pl.s_pl);
        s_stage  = pci_ld_u64(&pl.stage);
        s_dgate  = pci_ld_u64(&pl.dst_gate);
        s_dup    = pci_ld_u64(&pl.dst_up);
        s_ddown  = pci_ld_u64(&pl.dst_down);
    }
    __syncthreads();

    const int nthr = (int) blockDim.x;
    for (uint32_t ci = 0; ci < s_ncopy; ++ci) {
        if (tid == 0) {
            uint32_t r;
            do {
                r = pci_ld_u32(&pl.copy_ready[ci]);
            } while (r == 0);
            s_ready = (int) r;
            if (r == 1) {
                s_cslot = (int32_t) pci_ld_u32((const volatile uint32_t *) &pl.copy_slot[ci]);
                s_cpre  = pci_ld_u32(&pl.copy_pre[ci]);
            }
        }
        __syncthreads();
        const int32_t cslot = s_cslot;
        if (s_ready == 1 && cslot >= 0 && (uint32_t) cslot < s_spl) {
            if (use_dma) {
                while (pci_ld_u32(&dma_done[layer * MOE_DOORBELL_PLAN_MAX + ci]) == 0) doorbell_spin();
                __threadfence_system();
            } else {
                const char * src = (const char *) (uintptr_t) s_stage + (size_t) ci * s_stride + s_cpre;
                char * g = (char *) (uintptr_t) s_dgate + (size_t) cslot * s_gb;
                char * u = (char *) (uintptr_t) s_dup   + (size_t) cslot * s_ub;
                char * d = (char *) (uintptr_t) s_ddown + (size_t) cslot * s_db;
                doorbell_copy(g, src,             s_gb, tid, nthr);
                doorbell_copy(u, src + s_gb,      s_ub, tid, nthr);
                doorbell_copy(d, src + s_gb + s_ub, s_db, tid, nthr);
            }
        }
        __syncthreads();
    }

    if (tid < total) {
        const int32_t new_slot = (int32_t) pci_ld_u32((const volatile uint32_t *) &pl.assigned[tid]);
        const int32_t old_exp  = (int32_t) pci_ld_u32((const volatile uint32_t *) &pl.evicted[tid]);
        if (old_exp >= 0 && old_exp < n_exp && old_exp != exp) {
            slot_map[old_exp] = sentinel;
        }
        const int32_t res = (new_slot >= 0 && new_slot < sentinel) ? new_slot : sentinel;
        if (exp >= 0 && exp < n_exp) {
            slot_map[exp] = res;
        }
        const int k = tid % nk;
        const int t = tid / nk;
        *(int32_t *) (out + (size_t) t * out_nb1 + (size_t) k * out_nb0) = res;
    }
    __syncthreads();

    if (tid == 0) {
        pci_st_u32(&db.gpu_ready, 0);
        pci_st_u32(&db.state, 0);
        __threadfence_system();
    }
}

static __global__ void moe_prekick_kernel(
        const char * topk, char * out, DoorbellPrekick * plans,
        int layer, int count, size_t topk_nb0, size_t out_nb0) {
    if (threadIdx.x != 0) return;
    DoorbellPrekick & plan = plans[layer];
    plan.generation++;
    plan.n = (uint32_t) count;
    for (int selected = 0; selected < count; ++selected) {
        const int32_t expert = *(const int32_t *) (topk + (size_t) selected * topk_nb0);
        plan.experts[selected] = expert;
        *(int32_t *) (out + (size_t) selected * out_nb0) = expert;
    }
    __threadfence_system();
    pci_st_u32(&plan.state, 1);
    __threadfence_system();
}

DoorbellPrekick * moe_doorbell_prekick_host(int il) {
    return g_prekick_host && il >= 0 && il < g_n_layer ? &g_prekick_host[il] : nullptr;
}

// Fallback for batches wider than the doorbell plan. It only reads the table;
// the caller must have published every mapping before the node runs.
static __global__ void moe_prefill_resolve_kernel(
    const int32_t * __restrict__ topk,
    int32_t       * __restrict__ out,
    const int32_t * __restrict__ slot_map,
    int total,
    int n_exp,
    int sentinel
) {
    const int tid = (int) (threadIdx.x + blockDim.x * blockIdx.x);
    if (tid >= total) return;
    const int32_t exp_id = topk[tid];
    int32_t slot = sentinel;
    if (exp_id >= 0 && exp_id < n_exp) {
        slot = slot_map[exp_id];
    }
    out[tid] = (slot >= 0 && slot < sentinel) ? slot : sentinel;
}

extern "C" void * moe_doorbell_plan_host(int il) {
    if (!g_plans_host || il < 0 || il >= g_n_layer) return nullptr;
    return &g_plans_host[il];
}

extern "C" void * moe_doorbell_plan_dev(int il) {
    if (!g_plans_dev || il < 0 || il >= g_n_layer) return nullptr;
    return &g_plans_dev[il];
}

extern "C" int moe_doorbell_plan_cap() {
    return MOE_DOORBELL_PLAN_MAX;
}

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
) {
    (void) table_nb0;
    (void) full; (void) fnk; (void) fnt; (void) full_nb1; (void) look; (void) lnk;

    if (!g_active || !g_doorbells_dev || !g_plans_dev) return;
    if (g_lru_dev) GGML_ASSERT(n_exp == g_n_expert);
    const int64_t total64 = nk * nt;
    if (total64 <= 0) return;

    cudaStream_t s = (cudaStream_t) stream;
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    if (g_dma_done) GGML_ASSERT(cudaStreamIsCapturing(s, &capture_status) == cudaSuccess);
    const bool use_dma = g_dma_done && capture_status == cudaStreamCaptureStatusActive;
    if (op == GGML_OP_MOE_SLOT_PREKICK) {
        moe_prekick_kernel<<<1, 32, 0, s>>>(
            (const char *) topk, (char *) out, g_prekick_dev, il, (int) nk, topk_nb0, out_nb0);
        return;
    }
    if (!g_stage_dev) {
        // no pinned stage: resolve from the table only, never spin
        const int total = (int) total64;
        const int block = 256;
        const int grid = (total + block - 1) / block;
        moe_prefill_resolve_kernel<<<grid, block, 0, s>>>(
            topk, out, table, total, n_exp, sentinel
        );
        return;
    }
    if (total64 <= MOE_DOORBELL_PLAN_MAX) {
        const int total = (int) total64;
        moe_resolve_slots_kernel<<<1, 1024, 0, s>>>(
            il, (const char *) topk, (char *) out, const_cast<int32_t *>(table),
            g_doorbells_dev, g_plans_dev, g_prekick_dev,
            (int) nk, (int) nt, topk_nb0, topk_nb1, out_nb0, out_nb1,
            n_exp, sentinel, total, op, g_count_enabled, g_dma_done, use_dma, g_lru_dev, g_lru_epoch
        );
    } else {
        const int total = (int) total64;
        const int block = 256;
        const int grid = (total + block - 1) / block;
        moe_prefill_resolve_kernel<<<grid, block, 0, s>>>(
            topk, out, table, total, n_exp, sentinel
        );
    }
}

void moe_doorbell_signal_unblock(int il, void * stream) {
    (void) stream;
    if (!g_active || il < 0 || il >= g_n_layer) return;
    if (!g_doorbells_host) return;

    if (g_plans_host) {
        DoorbellPlan * p = &g_plans_host[il];
#if defined(__x86_64__) || defined(_M_X64)
        const uintptr_t begin = ((uintptr_t) p + offsetof(DoorbellPlan, n)) & ~(uintptr_t) 63;
        const uintptr_t end   = (uintptr_t) p + sizeof(DoorbellPlan);
        for (uintptr_t a = begin; a < end; a += 64) {
            _mm_clflush((const void *) a);
        }
        _mm_sfence();
#endif
    }

    g_doorbells_host[il].gpu_ready = 1;
#if defined(__x86_64__) || defined(_M_X64)
    _mm_clflush((void *) &g_doorbells_host[il].gpu_ready);
    _mm_sfence();
#endif
}

static bool moe_doorbell_dma_init() {
    const char * dma_env = getenv("MOE_DMA");
    if ((dma_env && atoi(dma_env) == 0) || g_dma_done || !g_stage_host || g_n_layer <= 0) return true;
    cudaError_t error = cudaGetDevice(&g_device);
    if (error == cudaSuccess) error = cudaStreamCreateWithFlags(&g_stream_h2d, cudaStreamNonBlocking);
    if (error == cudaSuccess) error = cudaMalloc((void **) &g_dma_done, (size_t) g_n_layer * MOE_DOORBELL_PLAN_MAX * sizeof(uint32_t));
    if (error == cudaSuccess) error = cudaMemset(g_dma_done, 0, (size_t) g_n_layer * MOE_DOORBELL_PLAN_MAX * sizeof(uint32_t));
    if (error == cudaSuccess) error = cudaHostAlloc((void **) &g_dma_one, sizeof(uint32_t), cudaHostAllocPortable);
    if (error != cudaSuccess) GGML_ABORT("moe_doorbell_dma_init failed: %s", cudaGetErrorString(error));
    *g_dma_one = 1;
    fprintf(stderr, "[moe_dma] enabled, dedicated nonblocking H2D stream\n");
    return true;
}

extern "C" bool moe_doorbell_stage_init(size_t stride, int n_slots) {
    if (g_stage_host) return true;
    if (stride == 0 || n_slots <= 0 || (size_t) n_slots > SIZE_MAX / stride) return false;
    const size_t bytes = stride * (size_t) n_slots;
    const char * wc_env = getenv("MOE_STAGE_WC");
    const bool write_combined = !wc_env || atoi(wc_env) != 0;
    const unsigned int flags = cudaHostAllocMapped | cudaHostAllocPortable | (write_combined ? cudaHostAllocWriteCombined : 0);
    cudaError_t err = cudaHostAlloc((void **) &g_stage_host, bytes, flags);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe_doorbell_stage_init: cudaHostAlloc(%zu) failed: %s (%d)\n",
                bytes, cudaGetErrorString(err), (int) err);
        return false;
    }
    err = cudaHostGetDevicePointer((void **) &g_stage_dev, g_stage_host, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe_doorbell_stage_init: cudaHostGetDevicePointer failed: %s (%d)\n",
                cudaGetErrorString(err), (int) err);
        cudaFreeHost(g_stage_host);
        g_stage_host = nullptr;
        return false;
    }
    g_stage_stride = stride;
    g_stage_cap = n_slots;
    moe_doorbell_dma_init();
    fprintf(stderr, "moe_doorbell_stage_init: %zu bytes (%d x %zu), dev=%p\n",
            bytes, n_slots, stride, (void *) g_stage_dev);
    return true;
}

extern "C" void * moe_doorbell_stage_host() { return g_stage_host; }
extern "C" void * moe_doorbell_stage_dev()  { return g_stage_dev; }
extern "C" size_t moe_doorbell_stage_stride() { return g_stage_stride; }
extern "C" int moe_doorbell_stage_cap() { return g_stage_cap; }

bool moe_doorbell_dma_enabled() { return g_dma_done != nullptr; }

const uint64_t * moe_doorbell_lru_host(int il) {
    return g_lru_host && il >= 0 && il < g_n_layer ? g_lru_host + (size_t) il * g_n_expert : nullptr;
}

bool moe_doorbell_dma_submit_from(int il, int copy_index, const char * source) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!g_dma_done || il < 0 || il >= g_n_layer || copy_index < 0 ||
        copy_index >= g_stage_cap || copy_index >= MOE_DOORBELL_PLAN_MAX) return false;
    DoorbellPlan & plan = g_plans_host[il];
    if (!plan.pad0) return true;
    const int slot = plan.copy_slot[copy_index];
    if (slot < 0 || (uint32_t) slot >= plan.s_pl) return false;
    if (cudaSetDevice(g_device) != cudaSuccess) return false;
    if (!source) return false;
    const uint64_t destinations[3] = {plan.dst_gate, plan.dst_up, plan.dst_down};
    const size_t lengths[3] = {plan.gate_bytes, plan.up_bytes, plan.down_bytes};
    size_t offset = 0;
    cudaError_t error = cudaSuccess;
    for (int role = 0; role < 3 && error == cudaSuccess; ++role) {
        char * destination = (char *) (uintptr_t) destinations[role] + (size_t) slot * lengths[role];
        error = cudaMemcpyAsync(destination, source + offset, lengths[role], cudaMemcpyHostToDevice, g_stream_h2d);
        offset += lengths[role];
    }
    if (error == cudaSuccess) error = cudaMemcpyAsync(g_dma_done + il * MOE_DOORBELL_PLAN_MAX + copy_index,
        g_dma_one, sizeof(uint32_t), cudaMemcpyHostToDevice, g_stream_h2d);
    if (error != cudaSuccess) fprintf(stderr, "[moe_dma] submission failed: %s\n", cudaGetErrorString(error));
    return error == cudaSuccess;
}

bool moe_doorbell_dma_submit(int il, int copy_index) {
    if (il < 0 || il >= g_n_layer || copy_index < 0 || copy_index >= g_stage_cap || copy_index >= MOE_DOORBELL_PLAN_MAX) return false;
    const char * source = g_stage_host + (size_t) copy_index * g_stage_stride + g_plans_host[il].copy_pre[copy_index];
    return moe_doorbell_dma_submit_from(il, copy_index, source);
}

bool moe_doorbell_dma_store(int il, int slot, char * destination) {
    if (!g_dma_done || !g_stream_h2d || il < 0 || il >= g_n_layer || slot < 0 || !destination) return false;
    const DoorbellPlan & plan = g_plans_host[il];
    if (!plan.pad0 || (uint32_t) slot >= plan.s_pl) return false;
    if (cudaSetDevice(g_device) != cudaSuccess) return false;
    const uint64_t sources[3] = {plan.dst_gate, plan.dst_up, plan.dst_down};
    const size_t lengths[3] = {plan.gate_bytes, plan.up_bytes, plan.down_bytes};
    size_t offset = 0;
    cudaError_t error = cudaSuccess;
    for (int role = 0; role < 3 && error == cudaSuccess; ++role) {
        const char * source = (const char *) (uintptr_t) sources[role] + (size_t) slot * lengths[role];
        error = cudaMemcpyAsync(destination + offset, source, lengths[role], cudaMemcpyDeviceToHost, g_stream_h2d);
        offset += lengths[role];
    }
    if (error != cudaSuccess) fprintf(stderr, "[moe_dma] cache store failed: %s\n", cudaGetErrorString(error));
    return error == cudaSuccess;
}

bool moe_doorbell_init(int n_layer, int n_expert, int s_per_layer) {
    (void) s_per_layer;
    if (g_active) return true;
    if (n_layer <= 0 || n_expert <= 0 || (size_t) n_layer > SIZE_MAX / sizeof(uint64_t) / (size_t) n_expert) return false;
    g_n_layer = n_layer;
    g_n_expert = n_expert;
    const char * count_env = getenv("MOE_IO_DIAG");
    g_count_enabled = count_env && atoi(count_env) != 0;
    g_logged_selected = 0;

    const size_t db_bytes    = sizeof(LayerDoorbell) * (size_t) n_layer;
    const size_t plan_bytes  = sizeof(DoorbellPlan)  * (size_t) n_layer;

    cudaError_t err = cudaHostAlloc((void **) &g_doorbells_host, db_bytes,
            cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe_doorbell_init: cudaHostAlloc failed: %s (%d)\n",
                cudaGetErrorString(err), (int) err);
        return false;
    }
    err = cudaHostGetDevicePointer((void **) &g_doorbells_dev, g_doorbells_host, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe_doorbell_init: cudaHostGetDevicePointer failed: %s (%d)\n",
                cudaGetErrorString(err), (int) err);
        cudaFreeHost(g_doorbells_host);
        g_doorbells_host = nullptr;
        return false;
    }

    err = cudaHostAlloc((void **) &g_plans_host, plan_bytes,
            cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe_doorbell_init: plan cudaHostAlloc failed: %s (%d)\n",
                cudaGetErrorString(err), (int) err);
        cudaFreeHost(g_doorbells_host);
        g_doorbells_host = nullptr;
        g_doorbells_dev = nullptr;
        return false;
    }
    err = cudaHostGetDevicePointer((void **) &g_plans_dev, g_plans_host, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "moe_doorbell_init: plan cudaHostGetDevicePointer failed: %s (%d)\n",
                cudaGetErrorString(err), (int) err);
        cudaFreeHost(g_plans_host);
        cudaFreeHost(g_doorbells_host);
        g_plans_host = nullptr;
        g_doorbells_host = nullptr;
        g_doorbells_dev = nullptr;
        return false;
    }

    memset(g_doorbells_host, 0, db_bytes);
    memset(g_plans_host, 0, plan_bytes);

    err = cudaHostAlloc((void **) &g_prekick_host, sizeof(DoorbellPrekick) * (size_t) n_layer,
            cudaHostAllocMapped | cudaHostAllocPortable);
    if (err == cudaSuccess) {
        err = cudaHostGetDevicePointer((void **) &g_prekick_dev, g_prekick_host, 0);
    }
    if (err != cudaSuccess) {
        if (g_prekick_host) cudaFreeHost(g_prekick_host);
        cudaFreeHost(g_plans_host);
        cudaFreeHost(g_doorbells_host);
        g_prekick_host = g_prekick_dev = nullptr;
        g_plans_host = g_plans_dev = nullptr;
        g_doorbells_host = g_doorbells_dev = nullptr;
        return false;
    }
    memset(g_prekick_host, 0, sizeof(DoorbellPrekick) * (size_t) n_layer);

    const char * lru_env = getenv("MOE_GPU_LRU");
    if (!lru_env || atoi(lru_env) != 0) {
        const size_t touch_bytes = (size_t) n_layer * n_expert * sizeof(uint64_t);
        err = cudaHostAlloc((void **) &g_lru_host, touch_bytes, cudaHostAllocMapped | cudaHostAllocPortable);
        if (err == cudaSuccess) err = cudaHostGetDevicePointer((void **) &g_lru_dev, g_lru_host, 0);
        if (err == cudaSuccess) err = cudaMalloc((void **) &g_lru_epoch, (size_t) n_layer * sizeof(uint64_t));
        if (err == cudaSuccess) err = cudaMemset(g_lru_epoch, 0, (size_t) n_layer * sizeof(uint64_t));
        if (err != cudaSuccess) GGML_ABORT("doorbell LRU allocation failed: %s", cudaGetErrorString(err));
        memset(g_lru_host, 0, touch_bytes);
        fprintf(stderr, "[moe_gpu_lru] tracking all expert selections, %zu bytes\n", touch_bytes);
    }

    ggml_cuda_moe_slot_resolve_register(moe_doorbell_resolve_launch);
    g_active = true;
    moe_doorbell_dma_init();
    fprintf(stderr, "moe_doorbell_init: SUCCESS, %zu + %zu bytes for %d layers (plan cap %d)\n",
            db_bytes, plan_bytes, n_layer, MOE_DOORBELL_PLAN_MAX);
    return true;
}

void moe_doorbell_free() {
    if (g_stream_h2d) {
        cudaStreamSynchronize(g_stream_h2d);
        cudaStreamDestroy(g_stream_h2d);
        g_stream_h2d = nullptr;
    }
    if (g_dma_done) cudaFree(g_dma_done);
    if (g_dma_one) cudaFreeHost(g_dma_one);
    g_dma_done = g_dma_one = nullptr;
    if (g_lru_epoch) cudaFree(g_lru_epoch);
    if (g_lru_host) cudaFreeHost(g_lru_host);
    g_lru_host = g_lru_dev = g_lru_epoch = nullptr;
    g_n_expert = 0;
    if (g_stage_host) {
        cudaFreeHost(g_stage_host);
        g_stage_host = nullptr;
        g_stage_dev = nullptr;
        g_stage_stride = 0;
        g_stage_cap = 0;
    }
    if (!g_active) {
        g_n_layer = 0;
        return;
    }
    if (g_prekick_host) {
        cudaFreeHost(g_prekick_host);
        g_prekick_host = g_prekick_dev = nullptr;
    }
    if (g_plans_host) {
        cudaFreeHost(g_plans_host);
        g_plans_host = nullptr;
        g_plans_dev = nullptr;
    }
    if (g_doorbells_host) {
        cudaFreeHost(g_doorbells_host);
        g_doorbells_host = nullptr;
        g_doorbells_dev = nullptr;
    }
    g_active = false;
    g_n_layer = 0;
}

bool moe_doorbell_is_active() { return g_active; }
LayerDoorbell * moe_doorbell_get_host_ptr() { return g_doorbells_host; }
LayerDoorbell * moe_doorbell_get_dev_ptr() { return g_doorbells_dev; }

void moe_doorbell_log_stats() {
    if (!g_active || !g_count_enabled) return;
    uint64_t selected = 0, misses = 0, decode_layers = 0, decode_selected = 0, decode_misses = 0;
    for (int layer = 0; layer < g_n_layer; ++layer) {
        const LayerDoorbell & db = g_doorbells_host[layer];
        selected += db.selected;
        misses += db.misses;
        decode_layers += db.decode_layers;
        decode_selected += db.decode_selected;
        decode_misses += db.decode_misses;
    }
    if (selected == g_logged_selected) return;
    g_logged_selected = selected;
    fprintf(stderr, "[moe_slots] selected=%llu misses=%llu decode_layers=%llu decode_selected=%llu decode_misses=%llu hit=%.3f%%\n",
            (unsigned long long) selected, (unsigned long long) misses, (unsigned long long) decode_layers,
            (unsigned long long) decode_selected, (unsigned long long) decode_misses,
            selected ? 100.0 * (double) (selected - misses) / selected : 0.0);
}
