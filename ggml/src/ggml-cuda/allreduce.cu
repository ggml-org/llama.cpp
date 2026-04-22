#include "allreduce.cuh"
#include "ggml-impl.h"

#include <cstring>

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory: 0 = not arrived,
// 1 = arrived.  There is exactly one writer (the owning GPU) and one reader
// (the peer), so we don't need atomics.  A volatile store paired with
// __threadfence_system() provides the release ordering that makes the D2H
// writes visible system-wide before the arrival flag is observed.
//
// atomicAdd_system() (mechanism 0 in the prototype) is broken on RTX 5090
// (hostNativeAtomicSupported = 0), so we use the volatile path throughout.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p) {
    *(volatile int *)p = 1;
    __threadfence_system();
}

static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
    return *(const volatile int *)p;
}

// ---------------------------------------------------------------------------
// Single-phase AllReduce kernel — float32, 2 GPUs
//
// Both GPUs run this kernel simultaneously in independent streams.  Each GPU:
//
//   Phase 1 (all threads): copy sendbuf → host_mine via float4 loads.
//                          __threadfence_system() commits writes to host.
//   Phase 2 (thread 0):   set arrival_mine = 1; spin on arrival_other == 1.
//   Phase 3 (all threads): reduce: recvbuf[i] = sendbuf[i] + host_other[i].
//
// The single-block configuration means __syncthreads() is sufficient for
// intra-block coordination and we can use the cheaper non-cooperative launch.
// 256 threads gives good occupancy while keeping register pressure low.
// ---------------------------------------------------------------------------
static __global__ void ggml_cuda_ar_f32_kernel(
        const float * __restrict__ sendbuf,
        float       * __restrict__ recvbuf,
        float       * __restrict__ host_mine,
        const float * __restrict__ host_other,
        int                        count,
        int *                      arrival_mine,
        int *                      arrival_other) {

    const int tid    = threadIdx.x;
    const int nt     = blockDim.x;
    const int count4 = count >> 2;
    const int tail   = count4 << 2;

    // Phase 1: vectorised D2H copy using float4 (16 bytes per load/store).
    {
        const float4 * s4 = reinterpret_cast<const float4 *>(sendbuf);
        float4       * d4 = reinterpret_cast<float4 *>(host_mine);
        for (int i = tid; i < count4; i += nt) {
            d4[i] = s4[i];
        }
        // Scalar tail if count is not a multiple of 4.
        if (tid < count - tail) {
            host_mine[tail + tid] = sendbuf[tail + tid];
        }
    }

    // Commit all host writes before signalling; __syncthreads() ensures
    // every thread's stores are in flight before thread 0 writes the flag.
    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 signals this GPU's arrival, then spins until the
    // peer signals back.  The spin uses __nanosleep to yield the SM to
    // other work rather than burning cycles in a hot loop.
    if (tid == 0) {
        ggml_cuda_ar_signal_set(arrival_mine);
        while (ggml_cuda_ar_signal_get(arrival_other) == 0) {
            __nanosleep(100);
        }
    }

    // Broadcast "peer has arrived" and acquire the peer's host_other writes.
    __syncthreads();
    __threadfence_system();

    // Phase 3: reduce — each thread handles its slice of the output.
    {
        const float4 * s4 = reinterpret_cast<const float4 *>(sendbuf);
        const float4 * o4 = reinterpret_cast<const float4 *>(host_other);
        float4       * r4 = reinterpret_cast<float4 *>(recvbuf);
        for (int i = tid; i < count4; i += nt) {
            float4 a = s4[i];
            float4 b = o4[i];
            r4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        }
        if (tid < count - tail) {
            recvbuf[tail + tid] = sendbuf[tail + tid] + host_other[tail + tid];
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  128 is well above the actual
// in-flight depth (single digits in practice) while keeping init cost low.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 128;

// Byte spacing between adjacent arrival ints.  Two cache lines (128 bytes)
// ensures the arrival slots for the two GPUs never share a cache line,
// preventing false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 128;

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    uint64_t call_count;

    // Per-device resources.
    float *               host_buf[GGML_CUDA_MAX_DEVICES];  // pinned staging
    cudaStream_t          streams[GGML_CUDA_MAX_DEVICES];   // non-blocking kernel streams
    ggml_cuda_ar_event_slot * ev_pool[GGML_CUDA_MAX_DEVICES]; // [device][slot]

    // Arrival ring: pinned, ARRIVAL_STRIDE bytes between adjacent ints.
    // Index helper: use ggml_cuda_ar_arrival_ptr().
    char * arrival;
};

// Return a pointer to the arrival int for (slot, rank).
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival + offset);
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
        const int * devices, int n_devices, size_t max_bytes) {
    GGML_ASSERT(n_devices >= 2 && n_devices <= GGML_CUDA_MAX_DEVICES);

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices  = n_devices;
    p->buf_bytes  = 0;
    p->call_count = 0;
    p->arrival    = nullptr;
    for (int i = 0; i < n_devices; ++i) {
        p->devices[i]  = devices[i];
        p->host_buf[i] = nullptr;
        p->streams[i]  = nullptr;
        p->ev_pool[i]  = nullptr;
    }

    // Per-device streams and event pools.
    for (int i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        if (cudaStreamCreateWithFlags(&p->streams[i], cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }

        p->ev_pool[i] = new ggml_cuda_ar_event_slot[GGML_CUDA_AR_POOL_SIZE]();
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            const bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }
    }

    // Arrival ring: cache-line padded so each GPU's int is on its own line.
    const size_t arrival_bytes =
        (size_t)GGML_CUDA_AR_POOL_SIZE * n_devices * GGML_CUDA_AR_ARRIVAL_STRIDE;
    if (cudaHostAlloc(reinterpret_cast<void **>(&p->arrival), arrival_bytes,
                      cudaHostAllocPortable) != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaHostAlloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
    memset(p->arrival, 0, arrival_bytes);

    // Per-device pinned staging buffers.
    p->buf_bytes = max_bytes;
    for (int i = 0; i < n_devices; ++i) {
        if (cudaHostAlloc(reinterpret_cast<void **>(&p->host_buf[i]), max_bytes,
                          cudaHostAllocPortable) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaHostAlloc for staging failed (%zu bytes)\n",
                           __func__, max_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        memset(p->host_buf[i], 0, max_bytes);
    }

    // Warmup: run the kernel N times at the expected tensor size to pay the
    // first-use driver / PCIe / page-mapping cost during model load rather
    // than during the first inference step, and to encourage the GPU clock
    // governor to boost before timing begins.
    // Currently limited to the 2-GPU case.
    if (n_devices == 2) {
        constexpr int    WARMUP_ITERS = 64;
        constexpr size_t WARMUP_COUNT = 8192;  // 32 KB of fp32
        constexpr size_t WARMUP_BYTES = WARMUP_COUNT * sizeof(float);

        float * dev_buf[2] = {};
        bool warmup_ok = true;
        for (int i = 0; i < 2; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            if (cudaMalloc(reinterpret_cast<void **>(&dev_buf[i]), WARMUP_BYTES) != cudaSuccess) {
                GGML_LOG_WARN("%s: warmup alloc failed for device %d, skipping\n",
                              __func__, p->devices[i]);
                warmup_ok = false;
                break;
            }
        }

        if (warmup_ok) {
            // Reuse slot 0 for every iteration, resetting arrival before each.
            for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
                for (int r = 0; r < 2; ++r) {
                    *ggml_cuda_ar_arrival_ptr(p, /*slot=*/0, r) = 0;
                }
                for (int r = 0; r < 2; ++r) {
                    ggml_cuda_set_device(p->devices[r]);
                    ggml_cuda_ar_f32_kernel<<<dim3(1), dim3(256), 0, p->streams[r]>>>(
                        dev_buf[r], dev_buf[r],
                        p->host_buf[r],
                        p->host_buf[1 - r],
                        static_cast<int>(WARMUP_COUNT),
                        ggml_cuda_ar_arrival_ptr(p, /*slot=*/0, r),
                        ggml_cuda_ar_arrival_ptr(p, /*slot=*/0, 1 - r));
                }
            }
            for (int i = 0; i < 2; ++i) {
                ggml_cuda_set_device(p->devices[i]);
                cudaStreamSynchronize(p->streams[i]);
            }
            GGML_LOG_DEBUG("%s: warmup complete (%d iters x %zu KB)\n",
                           __func__, WARMUP_ITERS, WARMUP_BYTES >> 10);
        }

        for (int i = 0; i < 2; ++i) {
            if (dev_buf[i]) {
                ggml_cuda_set_device(p->devices[i]);
                cudaFree(dev_buf[i]);
            }
        }
    }

    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %d GPUs, "
                  "%zu KB staging per GPU\n",
                  __func__, n_devices, max_bytes >> 10);
    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->host_buf[i]) {
            cudaFreeHost(p->host_buf[i]);
        }
        if (p->ev_pool[i]) {
            ggml_cuda_set_device(p->devices[i]);
            for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
                if (p->ev_pool[i][s].app) { cudaEventDestroy(p->ev_pool[i][s].app); }
                if (p->ev_pool[i][s].ker) { cudaEventDestroy(p->ev_pool[i][s].ker); }
            }
            delete[] p->ev_pool[i];
        }
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamDestroy(p->streams[i]);
        }
    }
    if (p->arrival) {
        cudaFreeHost(p->arrival);
    }
    delete p;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;

    // Only the 2-GPU path is implemented; fall back for larger communicators.
    if (n != 2) {
        return false;
    }

    // Only FP32 tensors are handled by the kernel; other types need a
    // separate implementation.
    if (tensors[0]->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t ne    = ggml_nelements(tensors[0]);
    const size_t  bytes = (size_t)ne * sizeof(float);

    if (ne == 0) {
        return true;
    }

    if (bytes > p->buf_bytes) {
        // Staging buffers too small; the caller should fall back.
        // TODO: reallocate or chunk for larger tensors.
        return false;
    }

    // Cycle through the event pool.  On the second pass through the ring,
    // synchronise on the slot's ker event before touching arrival ints —
    // the event and arrival pools wrap in lock-step so this guarantees that
    // the kernels which last used this slot have finished.
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    if (pool_lapped) {
        for (int i = 0; i < n; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    // Reset the arrival ints for this slot before any kernel can read them.
    for (int i = 0; i < n; ++i) {
        *ggml_cuda_ar_arrival_ptr(p, slot, i) = 0;
    }

    // Insert the kernel into each GPU's existing compute stream via events:
    //   record(app, compute_stream)        — capture "upstream done" point
    //   wait(internal_stream, app)         — internal stream defers until then
    //   launch kernel on internal_stream
    //   record(ker, internal_stream)       — capture "kernel done" point
    //   wait(compute_stream, ker)          — compute stream resumes after kernel
    for (int i = 0; i < n; ++i) {
        const int peer = 1 - i;  // valid for n == 2 only
        ggml_cuda_set_device(p->devices[i]);
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        ggml_cuda_ar_event_slot & ev = p->ev_pool[i][slot];

        CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
        CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], ev.app));

        ggml_cuda_ar_f32_kernel<<<dim3(1), dim3(256), 0, p->streams[i]>>>(
            static_cast<const float *>(tensors[i]->data),
            static_cast<float *>(tensors[i]->data),
            p->host_buf[i],
            p->host_buf[peer],
            static_cast<int>(ne),
            ggml_cuda_ar_arrival_ptr(p, slot, i),
            ggml_cuda_ar_arrival_ptr(p, slot, peer));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(ev.ker, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), ev.ker));
    }

    return true;
}
