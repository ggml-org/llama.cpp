#include "allreduce.cuh"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>

#ifdef GGML_CUDA_AR_WATCHDOG
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
    __threadfence_system();   // ensure all prior writes (D2H data) are globally visible
    *(volatile int *)p = 1;
    __threadfence_system();   // ensure the signal itself is globally visible
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
    return *(const volatile int *)p;
}

// ---------------------------------------------------------------------------
// Single-kernel AllReduce — float32, 2 GPUs (production)
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
        
        // ensure all GPUs have access to the arrival signal
        __threadfence_system();
        
        while (ggml_cuda_ar_signal_get(arrival_other) == 0) {
            //__threadfence_system();
            __nanosleep(100);
        }
    }

    // Broadcast "peer has arrived" and acquire peer's host_other writes.
    __syncthreads();
    __threadfence_system();

    // Phase 3: reduce.
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
// Watchdog debug variant — compiled only when GGML_CUDA_AR_WATCHDOG is defined.
//
// Identical to the production kernel except Phase 2 has a spin limit
// (max_spin).  If the limit is reached the kernel writes a debug record to
// a per-GPU ring buffer in pinned host memory, then bails out — all threads
// exit the kernel immediately (Phase 3 is skipped).
//
// The ring slot is claimed with atomicAdd on the ring head counter.  Host
// memory atomics work for a single GPU on RTX 5090 (just not cross-GPU).
// After writing the record fields the kernel issues __threadfence_system()
// and then sets the completion flag so the host watchdog thread can safely
// read the record.
// ---------------------------------------------------------------------------
#ifdef GGML_CUDA_AR_WATCHDOG

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

static __global__ void ggml_cuda_ar_f32_kernel_dbg(
        const float * __restrict__ sendbuf,
        float       * __restrict__ recvbuf,
        float       * __restrict__ host_mine,
        const float * __restrict__ host_other,
        int                        count,
        int *                      arrival_mine,
        int *                      arrival_other,
        ggml_cuda_ar_debug_ring *  ring,
        int                        max_spin,
        int                        rank,
        int                        ar_slot) {

    __shared__ int bail;

    const int tid    = threadIdx.x;
    const int nt     = blockDim.x;
    const int count4 = count >> 2;
    const int tail   = count4 << 2;

    if (tid == 0) { bail = 0; }
    __syncthreads();

    // Phase 1: D2H copy (identical to production kernel).
    {
        const float4 * s4 = reinterpret_cast<const float4 *>(sendbuf);
        float4       * d4 = reinterpret_cast<float4 *>(host_mine);
        for (int i = tid; i < count4; i += nt) {
            d4[i] = s4[i];
        }
        if (tid < count - tail) {
            host_mine[tail + tid] = sendbuf[tail + tid];
        }
    }

    __threadfence_system();
    __syncthreads();

    // Phase 2: signal + instrumented spin.
    if (tid == 0) {
        ggml_cuda_ar_signal_set(arrival_mine);

        int writeback = ggml_cuda_ar_signal_get(arrival_mine);

        int spin = 0;
        int last = 0;
        while ((last = ggml_cuda_ar_signal_get(arrival_other)) == 0) {
            ++spin;
            if (max_spin > 0 && spin >= max_spin) {
                // Acquire a ring slot via atomicAdd (single-GPU host atomics OK).
                int ri = atomicAdd(&ring->head, 1) % GGML_CUDA_AR_RING_SIZE;
                ggml_cuda_ar_debug_record * rec = &ring->records[ri];

                rec->rank          = rank;
                rec->slot          = ar_slot;
                rec->spin_count    = spin;
                rec->arrival_mine  = writeback;
                rec->arrival_other = last;
                rec->count         = count;

                __threadfence_system();  // ensure fields visible before completion flag
                rec->complete = 1;
                __threadfence_system();  // ensure completion flag visible to host

                bail = 1;
                break;
            }
            __nanosleep(100);
        }
    }

    __syncthreads();
    if (bail) {
        return;  // all threads exit — skip Phase 3
    }

    // Broadcast "peer has arrived" and acquire peer's host_other writes.
    __threadfence_system();

    // Phase 3: reduce.
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
#endif // GGML_CUDA_AR_WATCHDOG

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  128 is well above the actual
// in-flight depth (single digits in practice) while keeping init cost low.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 128;

// Byte spacing between adjacent arrival ints.  128 bytes (two cache lines)
// ensures the arrival slots for the two GPUs never share a cache line,
// preventing false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 128;

#ifdef GGML_CUDA_AR_WATCHDOG
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
    float *                  host_buf[GGML_CUDA_MAX_DEVICES];  // pinned staging
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    ggml_cuda_ar_event_slot *ev_pool[GGML_CUDA_MAX_DEVICES];   // [device][slot]

    // Arrival ring: pinned, ARRIVAL_STRIDE bytes between adjacent ints.
    // Use ggml_cuda_ar_arrival_ptr() to index.
    char * arrival;

#ifdef GGML_CUDA_AR_WATCHDOG
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

// ---------------------------------------------------------------------------
// Background watchdog thread — monitors per-GPU debug ring buffers for new
// bailout records.  The kernel writes a record when it hits the spin limit;
// this thread polls the ring head counters every 1ms and prints any new
// complete records.  Zero overhead on the dispatch path (no queue, no events).
// ---------------------------------------------------------------------------
#ifdef GGML_CUDA_AR_WATCHDOG
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
#endif // GGML_CUDA_AR_WATCHDOG

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
#ifdef GGML_CUDA_AR_WATCHDOG
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        p->debug_ring[i] = nullptr;
    }
    p->wdog_max_spin = 0;
#endif

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

#ifdef GGML_CUDA_AR_WATCHDOG
    // Per-GPU debug ring buffers: written by the kernel on spin-limit bailout,
    // polled by the background watchdog thread.  Each ring is pinned host
    // memory accessed only by its owning GPU (single-GPU host atomics OK).
    {
        for (int i = 0; i < n_devices; ++i) {
            if (cudaHostAlloc(reinterpret_cast<void **>(&p->debug_ring[i]),
                              sizeof(ggml_cuda_ar_debug_ring),
                              cudaHostAllocPortable) != cudaSuccess) {
                GGML_LOG_ERROR("%s: cudaHostAlloc for debug ring failed on device %d\n",
                               __func__, p->devices[i]);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
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
    }
#endif

#if 0
    // Warmup: run the kernel N times to pay first-use driver / PCIe /
    // page-mapping costs during model load and encourage the GPU clock
    // governor to boost before inference begins.
    if (n_devices == 2) {
        printf("ggml_cuda_ar_pipeline_init warmup\n");
        
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
            // Warmup always uses the production kernel (no debug overhead).
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
        
        printf("ggml_cuda_ar_pipeline_init warmup finished\n");
    }
#endif
    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %d GPUs, "
                  "%zu KB staging per GPU\n",
                  __func__, n_devices, max_bytes >> 10);
                  
    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

#ifdef GGML_CUDA_AR_WATCHDOG
    // Stop the watchdog thread first — it only reads pinned host memory,
    // no GPU resources, so this is safe and returns within ~1ms.
    p->wdog_stop.store(true);
    if (p->wdog_thr.joinable()) {
        p->wdog_thr.join();
    }
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
#ifdef GGML_CUDA_AR_WATCHDOG
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->debug_ring[i]) {
            cudaFreeHost(p->debug_ring[i]);
        }
    }
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
    //printf("ggml_cuda_ar_allreduce\n");
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;

    // Only the 2-GPU path is implemented; fall back for larger communicators.
    if (n != 2) {
        return false;
    }

    // Only FP32 tensors are handled by the kernel.
    if (tensors[0]->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t ne    = ggml_nelements(tensors[0]);
    const size_t  bytes = (size_t)ne * sizeof(float);

    if (ne == 0) {
        return true;
    }

    if (bytes > p->buf_bytes) {
        return false;
    }

    // Cycle through the event pool.  On the second pass through the ring,
    // synchronise on the slot's ker event before touching arrival ints —
    // the event and arrival pools wrap in lock-step so this guarantees the
    // kernels which last used this slot have finished.
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
    //   record(app, compute_stream)       — capture "upstream done"
    //   wait(internal_stream, app)        — internal stream defers until then
    //   launch kernel on internal_stream
    //   record(ker, internal_stream)      — capture "kernel done"
    //   wait(compute_stream, ker)         — compute stream resumes after kernel
    for (int i = 0; i < n; ++i) {
        const int peer = 1 - i;  // valid for n == 2 only
        ggml_cuda_set_device(p->devices[i]);
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        ggml_cuda_ar_event_slot & ev = p->ev_pool[i][slot];

        CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
        CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], ev.app));

#ifdef GGML_CUDA_AR_WATCHDOG
        ggml_cuda_ar_f32_kernel_dbg<<<dim3(1), dim3(256), 0, p->streams[i]>>>(
            static_cast<const float *>(tensors[i]->data),
            static_cast<float *>(tensors[i]->data),
            p->host_buf[i],
            p->host_buf[peer],
            static_cast<int>(ne),
            ggml_cuda_ar_arrival_ptr(p, slot, i),
            ggml_cuda_ar_arrival_ptr(p, slot, peer),
            p->debug_ring[i],
            p->wdog_max_spin,
            i,
            slot);
#else
        ggml_cuda_ar_f32_kernel<<<dim3(1), dim3(256), 0, p->streams[i]>>>(
            static_cast<const float *>(tensors[i]->data),
            static_cast<float *>(tensors[i]->data),
            p->host_buf[i],
            p->host_buf[peer],
            static_cast<int>(ne),
            ggml_cuda_ar_arrival_ptr(p, slot, i),
            ggml_cuda_ar_arrival_ptr(p, slot, peer));
#endif
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(ev.ker, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), ev.ker));

    }

    return true;
}
