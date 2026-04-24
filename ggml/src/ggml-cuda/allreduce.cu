#include "allreduce.cuh"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>

// Set to 1 to enable the AllReduce spin-limit watchdog (development only).
// When enabled, the debug kernel bails out after GGML_CUDA_AR_MAX_SPIN
// iterations and writes a record to a per-GPU ring buffer that the
// background watchdog thread prints.
#define GGML_CUDA_AR_WATCHDOG 0

#if GGML_CUDA_AR_WATCHDOG
#include <atomic>
#include <chrono>
#include <thread>
#endif

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory: 0 = not arrived,
// 1 = arrived.  There is exactly one writer (the owning GPU) and one reader
// (the peer), so we don't need atomics.  A volatile store paired with
// __threadfence_system() provides the release ordering that makes the D2H
// writes visible system-wide before the arrival flag is observed.
//
// atomicAdd_system() is broken on RTX 5090 (hostNativeAtomicSupported = 0),
// so we use the volatile path throughout.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p) {
    *(volatile int *)p = 1;
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
    return *(const volatile int *)p;
}

// ---------------------------------------------------------------------------
// Single-kernel AllReduce — 2 GPUs, supports float, half, and bfloat16.
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
//
// When GGML_CUDA_AR_WATCHDOG is enabled, Phase 2 has a spin limit
// (max_spin).  If the limit is reached the kernel writes a debug record to
// a per-GPU ring buffer in pinned host memory, then bails out — all threads
// exit the kernel immediately (Phase 3 is skipped).
// ---------------------------------------------------------------------------

#if GGML_CUDA_AR_WATCHDOG
// One debug record written by the kernel on spin-limit bailout.
struct ggml_cuda_ar_debug_record {
    int rank;            // GPU rank (0 or 1)
    int slot;            // AllReduce pool slot
    int spin_count;      // spins before bailout
    int arrival_mine;    // readback of own arrival flag after signal_set
    int arrival_other;   // last value of peer's arrival flag
    int count;           // element count of the AllReduce call
    int complete;        // 1 = record fully written (set last, after fence)
};

static constexpr int GGML_CUDA_AR_RING_SIZE = 64;

// Per-GPU ring buffer in pinned host memory.  head is incremented by the
// GPU via atomicAdd; records[] is written by the GPU and read by the host.
struct ggml_cuda_ar_debug_ring {
    int                          head;  // next slot to write (GPU atomicAdd)
    ggml_cuda_ar_debug_record    records[GGML_CUDA_AR_RING_SIZE];
};
#endif // GGML_CUDA_AR_WATCHDOG

// ---------------------------------------------------------------------------
// Vectorised add helpers for Phase 3 reduction.  All types use float4
// (16 bytes) as the vector load unit for maximum PCIe throughput.
// ---------------------------------------------------------------------------
template <typename T>
static __device__ __forceinline__ float4 ggml_cuda_ar_vec_add(float4 a, float4 b);

template <>
__device__ __forceinline__ float4 ggml_cuda_ar_vec_add<float>(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <>
__device__ __forceinline__ float4 ggml_cuda_ar_vec_add<half>(float4 a, float4 b) {
    float4 r;
    half2 * ha = reinterpret_cast<half2 *>(&a);
    half2 * hb = reinterpret_cast<half2 *>(&b);
    half2 * hr = reinterpret_cast<half2 *>(&r);
    #pragma unroll
    for (int k = 0; k < 4; ++k) { hr[k] = ha[k] + hb[k]; }
    return r;
}

template <>
__device__ __forceinline__ float4 ggml_cuda_ar_vec_add<__nv_bfloat16>(float4 a, float4 b) {
    float4 r;
    __nv_bfloat162 * ba = reinterpret_cast<__nv_bfloat162 *>(&a);
    __nv_bfloat162 * bb = reinterpret_cast<__nv_bfloat162 *>(&b);
    __nv_bfloat162 * br = reinterpret_cast<__nv_bfloat162 *>(&r);
    #pragma unroll
    for (int k = 0; k < 4; ++k) { br[k] = ba[k] + bb[k]; }
    return r;
}

template <typename T>
static __global__ void ggml_cuda_ar_kernel(
        const T * __restrict__ sendbuf,
        T       * __restrict__ recvbuf,
        T       * __restrict__ host_mine,
        const T * __restrict__ host_other,
        int                    count,
        int *                  arrival_mine,
        int *                  arrival_other
#if GGML_CUDA_AR_WATCHDOG
       ,ggml_cuda_ar_debug_ring * ring,
        int                    max_spin,
        int                    rank,
        int                    ar_slot
#endif
        ) {

    // Number of elements of T per float4 vector (16 bytes).
    constexpr int ELEMS_PER_VEC = 16 / sizeof(T);

#if GGML_CUDA_AR_WATCHDOG
    __shared__ int bail;
#endif

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

#if GGML_CUDA_AR_WATCHDOG
    if (tid == 0) { bail = 0; }
    __syncthreads();
#endif

    // Phase 1: vectorised D2H copy using float4 (16 bytes per load/store).
    {
        const float4 * s4 = reinterpret_cast<const float4 *>(sendbuf);
        float4       * d4 = reinterpret_cast<float4 *>(host_mine);
        for (int i = tid; i < count_vec; i += nt) {
            d4[i] = s4[i];
        }
        if (tid < count - tail) {
            host_mine[tail + tid] = sendbuf[tail + tid];
        }
    }

    // Commit all host writes before signalling.
    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 signals arrival, then spins for the peer.
    if (tid == 0) {
        ggml_cuda_ar_signal_set(arrival_mine);

        __threadfence_system(); // ensure the signal itself is visible across all GPUs

#if GGML_CUDA_AR_WATCHDOG
        int writeback = ggml_cuda_ar_signal_get(arrival_mine);
        int spin = 0;
        int last = 0;
        while ((last = ggml_cuda_ar_signal_get(arrival_other)) == 0) {
            ++spin;
            if (max_spin > 0 && spin >= max_spin) {
                int ri = atomicAdd(&ring->head, 1) % GGML_CUDA_AR_RING_SIZE;
                ggml_cuda_ar_debug_record * rec = &ring->records[ri];

                rec->rank          = rank;
                rec->slot          = ar_slot;
                rec->spin_count    = spin;
                rec->arrival_mine  = writeback;
                rec->arrival_other = last;
                rec->count         = count;

                __threadfence_system();
                rec->complete = 1;
                __threadfence_system();

                bail = 1;
                break;
            }
            __nanosleep(100);
        }
#else
        while (ggml_cuda_ar_signal_get(arrival_other) == 0) {
            __nanosleep(100);
        }
#endif
    }

    __syncthreads();
#if GGML_CUDA_AR_WATCHDOG
    if (bail) {
        return;
    }
#endif

    // Broadcast "peer has arrived" and acquire peer's host_other writes.
    __threadfence_system();

    // Phase 3: reduce.
    {
        const float4 * s4 = reinterpret_cast<const float4 *>(sendbuf);
        const float4 * o4 = reinterpret_cast<const float4 *>(host_other);
        float4       * r4 = reinterpret_cast<float4 *>(recvbuf);
        for (int i = tid; i < count_vec; i += nt) {
            r4[i] = ggml_cuda_ar_vec_add<T>(s4[i], o4[i]);
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

// Maximum chunk size (bytes per GPU) handled by one internal kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 256 * 1024; // 256 KB

// Byte spacing between adjacent arrival ints.  128 bytes (two cache lines)
// ensures the arrival slots for the two GPUs never share a cache line,
// preventing false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 128;

#if GGML_CUDA_AR_WATCHDOG
// Watchdog poll interval in milliseconds.
static constexpr int GGML_CUDA_AR_WDOG_POLL_MS = 1;
#endif

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
    char *                   host_buf[GGML_CUDA_MAX_DEVICES];  // pinned staging
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    ggml_cuda_ar_event_slot *ev_pool[GGML_CUDA_MAX_DEVICES];   // [device][slot]

    // Arrival ring: pinned, ARRIVAL_STRIDE bytes between adjacent ints.
    // Use ggml_cuda_ar_arrival_ptr() to index.
    char * arrival;

#if GGML_CUDA_AR_WATCHDOG
    // Per-GPU debug ring buffers in pinned host memory.  Written by the debug
    // kernel on spin-limit bailout, read by the background watchdog thread.
    ggml_cuda_ar_debug_ring * debug_ring[GGML_CUDA_MAX_DEVICES];
    int                       wdog_max_spin;    // 0 = no limit (env: GGML_CUDA_AR_MAX_SPIN)
    std::atomic<bool>         wdog_stop{false};
    std::thread               wdog_thr;
#endif
};

// Return a pointer to the arrival int for (slot, rank).
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival + offset);
}

static int ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    if (pool_lapped) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        *ggml_cuda_ar_arrival_ptr(p, slot, i) = 0;
    }

    return slot;
}

static void ggml_cuda_ar_wait_for_compute(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(p->streams[rank], ev.app));
}

static void ggml_cuda_ar_record_chunk_done(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot, bool last_chunk) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.ker, p->streams[rank]));
    if (last_chunk) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), ev.ker));
    }
}

// ---------------------------------------------------------------------------
// Background watchdog thread — monitors per-GPU debug ring buffers for new
// bailout records.  The kernel writes a record when it hits the spin limit;
// this thread polls the ring head counters every 1ms and prints any new
// complete records.  Zero overhead on the dispatch path (no queue, no events).
// ---------------------------------------------------------------------------
#if GGML_CUDA_AR_WATCHDOG
static void ggml_cuda_ar_wdog_thread(ggml_cuda_ar_pipeline * p) {
    int last_seen[GGML_CUDA_MAX_DEVICES] = {};

    while (!p->wdog_stop.load(std::memory_order_relaxed)) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_ar_debug_ring * ring = p->debug_ring[i];
            if (!ring) { continue; }

            int head = *(volatile int *)&ring->head;
            while (last_seen[i] < head) {
                int ri = last_seen[i] % GGML_CUDA_AR_RING_SIZE;
                const ggml_cuda_ar_debug_record * rec = &ring->records[ri];

                // Wait for the completion flag (kernel writes it last after fence).
                if (*(volatile int *)&rec->complete) {
                    GGML_LOG_WARN("ggml_cuda_ar BAILOUT: gpu%d rank=%d slot=%d "
                                  "spins=%d arrival_mine=%d arrival_other=%d count=%d\n",
                                  p->devices[i], rec->rank, rec->slot,
                                  rec->spin_count, rec->arrival_mine,
                                  rec->arrival_other, rec->count);
                    last_seen[i]++;
                } else {
                    break;  // record not yet complete — check again next poll
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(GGML_CUDA_AR_WDOG_POLL_MS));
    }
}

static bool ggml_cuda_ar_wdog_init(ggml_cuda_ar_pipeline * p) {
    for (int i = 0; i < p->n_devices; ++i) {
        if (cudaHostAlloc(reinterpret_cast<void **>(&p->debug_ring[i]),
                          sizeof(ggml_cuda_ar_debug_ring),
                          cudaHostAllocPortable) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaHostAlloc for debug ring failed on device %d\n",
                           __func__, p->devices[i]);
            return false;
        }
        memset(p->debug_ring[i], 0, sizeof(ggml_cuda_ar_debug_ring));
    }

    const char * spin_env = getenv("GGML_CUDA_AR_MAX_SPIN");
    p->wdog_max_spin = (spin_env && spin_env[0]) ? atoi(spin_env) : 0;
    GGML_LOG_INFO("%s: AR watchdog enabled — max_spin=%d "
                  "(set GGML_CUDA_AR_MAX_SPIN=<n> to adjust)\n",
                  __func__, p->wdog_max_spin);

    p->wdog_stop.store(false);
    p->wdog_thr = std::thread(ggml_cuda_ar_wdog_thread, p);
    return true;
}

static void ggml_cuda_ar_wdog_stop(ggml_cuda_ar_pipeline * p) {
    p->wdog_stop.store(true);
    if (p->wdog_thr.joinable()) {
        p->wdog_thr.join();
    }
}

static void ggml_cuda_ar_wdog_free(ggml_cuda_ar_pipeline * p) {
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->debug_ring[i]) {
            cudaFreeHost(p->debug_ring[i]);
        }
    }
}
#endif // GGML_CUDA_AR_WATCHDOG

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if ((n_devices != 2) || (n_devices > GGML_CUDA_MAX_DEVICES)) {
        return nullptr;
    }

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
#if GGML_CUDA_AR_WATCHDOG
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        p->debug_ring[i] = nullptr;
    }
    p->wdog_max_spin = 0;
#endif

    // Per-device streams and event pools.
    for (int i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        p->streams[i] = stream;

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
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    for (int i = 0; i < n_devices; ++i) {
        if (cudaHostAlloc(&p->host_buf[i], p->buf_bytes, cudaHostAllocPortable) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaHostAlloc for staging failed (%zu bytes)\n",
                           __func__, p->buf_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        memset(p->host_buf[i], 0, p->buf_bytes);
    }

#if GGML_CUDA_AR_WATCHDOG
    if (!ggml_cuda_ar_wdog_init(p)) {
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
#endif

    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %d GPUs, "
                  "%zu KB staging per GPU\n",
                  __func__, n_devices, p->buf_bytes >> 10);

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

#if GGML_CUDA_AR_WATCHDOG
    // Stop the watchdog thread first — it only reads pinned host memory,
    // no GPU resources, so this is safe and returns within ~1ms.
    ggml_cuda_ar_wdog_stop(p);
#endif

    // Drain all in-flight kernels before tearing down resources.
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamSynchronize(p->streams[i]);
        }
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
#if GGML_CUDA_AR_WATCHDOG
    ggml_cuda_ar_wdog_free(p);
#endif
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
    GGML_ASSERT(n == 2);

    const ggml_type type      = tensors[0]->type;
    const size_t    type_size = ggml_type_size(type);
    GGML_ASSERT(type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);
    GGML_ASSERT(p->buf_bytes >= type_size);

    const size_t max_chunk_elems = p->buf_bytes / type_size;
    GGML_ASSERT(max_chunk_elems > 0);

    // Insert chunked kernels into each GPU's existing compute stream via events:
    //   record(app, compute_stream)       — capture "upstream done"
    //   wait(internal_stream, app)        — internal stream defers until then
    //   launch one or more chunk kernels on internal_stream
    //   record(ker, internal_stream)      — capture "final chunk done"
    //   wait(compute_stream, ker)         — compute stream resumes after reduce
    for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
        const size_t remaining_elems = (size_t) (ne - chunk_start);
        const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
        const size_t chunk_bytes = chunk_elems * type_size;

        const int slot = ggml_cuda_ar_acquire_slot(p);
        const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;

        for (int i = 0; i < n; ++i) {
            const int peer = 1 - i;  // valid for n == 2 only
            ggml_cuda_set_device(p->devices[i]);
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            const bool compute = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;

            if (chunk_start == 0) {
                ggml_cuda_ar_wait_for_compute(p, cuda_ctx, i, slot);
            }

            char * data = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) type_size;

            // Match the NCCL and meta-backend semantics: inactive shards
            // contribute zeros to the reduction.
            if (!compute) {
                CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_bytes, p->streams[i]));
            }

#if GGML_CUDA_AR_WATCHDOG
#define GGML_CUDA_AR_WDOG_EXTRA_ARGS , p->debug_ring[i], p->wdog_max_spin, i, slot
#else
#define GGML_CUDA_AR_WDOG_EXTRA_ARGS
#endif

#define LAUNCH_AR_KERNEL(T) \
            ggml_cuda_ar_kernel<T><<<dim3(1), dim3(256), 0, p->streams[i]>>>( \
                reinterpret_cast<const T *>(data), \
                reinterpret_cast<T *>(data), \
                reinterpret_cast<T *>(p->host_buf[i]), \
                reinterpret_cast<const T *>(p->host_buf[peer]), \
                static_cast<int>(chunk_elems), \
                ggml_cuda_ar_arrival_ptr(p, slot, i), \
                ggml_cuda_ar_arrival_ptr(p, slot, peer) \
                GGML_CUDA_AR_WDOG_EXTRA_ARGS)

            switch (type) {
                case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float);           break;
                case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half);            break;
                case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(__nv_bfloat16);   break;
                default: GGML_ASSERT(false);
            }

#undef LAUNCH_AR_KERNEL
#undef GGML_CUDA_AR_WDOG_EXTRA_ARGS
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_ar_record_chunk_done(p, cuda_ctx, i, slot, last_chunk);
        }
    }

    return true;
}
