#include "allreduce.cuh"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>

#ifdef GGML_CUDA_AR_WATCHDOG
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
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
        while (ggml_cuda_ar_signal_get(arrival_other) == 0) {
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
// Adds three extra parameters to the kernel and instruments Phase 2:
//
//   debug[0]  spin iteration count, updated every ~4096 iterations
//             (-1 written on max_spin bailout as a sentinel)
//   debug[1]  last value of arrival_other observed during the spin
//   debug[2]  readback of arrival_mine immediately after signal_set;
//             should always be 1 — if 0, the write did not reach host memory
//   debug[3]  reserved (always 0)
//
//   max_spin  if > 0, the spin bails out after this many iterations and the
//             kernel logs via printf before proceeding to Phase 3 with stale
//             data.  Output will be numerically wrong, but the kernel exits
//             rather than hanging, which is useful for post-mortem logging.
//
//   rank      this GPU's rank within the communicator (for printf output)
// ---------------------------------------------------------------------------
#ifdef GGML_CUDA_AR_WATCHDOG
static __global__ void ggml_cuda_ar_f32_kernel_dbg(
        const float * __restrict__ sendbuf,
        float       * __restrict__ recvbuf,
        float       * __restrict__ host_mine,
        const float * __restrict__ host_other,
        int                        count,
        int *                      arrival_mine,
        int *                      arrival_other,
        int *                      debug,
        int                        max_spin,
        int                        rank) {

    const int tid    = threadIdx.x;
    const int nt     = blockDim.x;
    const int count4 = count >> 2;
    const int tail   = count4 << 2;

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

        // Readback: can this GPU see its own signal?  If writeback == 0 the
        // write did not reach host-visible memory, which would explain why
        // the peer never observes arrival.
        int writeback = ggml_cuda_ar_signal_get(arrival_mine);
        debug[2] = writeback;

        int spin = 0;
        int last = 0;
        while ((last = ggml_cuda_ar_signal_get(arrival_other)) == 0) {
            ++spin;
            // Periodically expose progress so the host watchdog can read it.
            if ((spin & 0xFFF) == 0) {
                debug[0] = spin;
                debug[1] = last;
            }
            if (max_spin > 0 && spin >= max_spin) {
                debug[0] = -1;  // bailout sentinel
                debug[1] = last;
                printf("ggml_cuda_ar: rank=%d BAILOUT after %d spins; "
                       "writeback_mine=%d arrival_other=%d "
                       "(pm=%p po=%p)\n",
                       rank, spin, writeback, last,
                       (void *)arrival_mine, (void *)arrival_other);
                break;
            }
            __nanosleep(100);
        }
        // Write final spin count, unless we already wrote the bailout sentinel.
        if (debug[0] != -1) {
            debug[0] = spin;
            debug[1] = last;
        }
    }

    // Phase 3: reduce (proceeds even after bailout; output will be wrong).
    __syncthreads();
    __threadfence_system();

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
// Ints per device in the debug buffer.  Layout (index → meaning):
//   0  spin count, updated every ~4096 iterations (-1 = bailed out)
//   1  last arrival_other value observed during the spin
//   2  readback of arrival_mine after signal_set (should always be 1)
//   3  reserved
static constexpr int GGML_CUDA_AR_DEBUG_INTS = 4;

// Background-thread poll interval.  This has no effect on dispatch latency
// because the poll runs in a dedicated thread.
static constexpr int GGML_CUDA_AR_WDOG_POLL_MS = 100;

// One work item posted to the background watchdog thread per dispatch.
struct ggml_cuda_ar_wdog_item {
    int         slot;
    int         n;
    int         devices[GGML_CUDA_MAX_DEVICES];
    cudaEvent_t ker_events[GGML_CUDA_MAX_DEVICES];
};
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
    // Pinned debug buffer written by the debug kernel, read by the background
    // watchdog thread.  Layout: debug_buf[rank * GGML_CUDA_AR_DEBUG_INTS + field].
    int * debug_buf;
    int   wdog_timeout_ms;  // 0 = disabled (env: GGML_CUDA_AR_WATCHDOG)
    int   wdog_max_spin;    // 0 = no limit (env: GGML_CUDA_AR_MAX_SPIN)

    // Background watchdog thread: polls kernel events without blocking dispatch.
    std::mutex                          wdog_mtx;
    std::condition_variable             wdog_cv;
    std::deque<ggml_cuda_ar_wdog_item>  wdog_queue;
    bool                                wdog_stop = false;
    std::thread                         wdog_thr;
#endif
};

// Return a pointer to the arrival int for (slot, rank).
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival + offset);
}

// ---------------------------------------------------------------------------
// Background watchdog thread — polls kernel events and debug state without
// blocking the dispatch path.  One work item is posted per dispatch; the
// thread logs any slot that doesn't complete within wdog_timeout_ms.
//
// Output is sparse on the fast path: nothing is logged when a slot completes
// before the first poll tick.  On slow or hanging slots every tick is logged,
// including the final "done" tick, to capture the full timeline.
// ---------------------------------------------------------------------------
#ifdef GGML_CUDA_AR_WATCHDOG
static void ggml_cuda_ar_wdog_thread(ggml_cuda_ar_pipeline * p) {
    while (true) {
        ggml_cuda_ar_wdog_item item;
        {
            std::unique_lock<std::mutex> lk(p->wdog_mtx);
            p->wdog_cv.wait(lk, [p] { return !p->wdog_queue.empty() || p->wdog_stop; });
            if (p->wdog_stop && p->wdog_queue.empty()) {
                break;
            }
            item = p->wdog_queue.front();
            p->wdog_queue.pop_front();
        }

        if (p->wdog_timeout_ms <= 0) {
            continue;  // watchdog disabled — drain queue, do nothing
        }

        int  elapsed_ms    = 0;
        bool observed_busy = false;

        while (elapsed_ms <= p->wdog_timeout_ms) {
            bool        all_done = true;
            cudaError_t qstat[GGML_CUDA_MAX_DEVICES];
            for (int i = 0; i < item.n; ++i) {
                ggml_cuda_set_device(item.devices[i]);
                qstat[i] = cudaEventQuery(item.ker_events[i]);
                if (qstat[i] != cudaSuccess) {
                    all_done = false;
                }
            }

            if (!all_done) {
                observed_busy = true;
            }

            if (!all_done || observed_busy) {
                char msg[512];
                int  pos = 0;
                pos += snprintf(msg + pos, sizeof(msg) - pos,
                                "ggml_cuda_ar watchdog +%dms slot=%d:",
                                elapsed_ms, item.slot);
                for (int i = 0; i < item.n; ++i) {
                    const int * dbg = p->debug_buf + i * GGML_CUDA_AR_DEBUG_INTS;
                    int arr  = *(volatile int *)ggml_cuda_ar_arrival_ptr(p, item.slot, i);
                    int spin = *(volatile int *)&dbg[0];
                    int last = *(volatile int *)&dbg[1];
                    int wb   = *(volatile int *)&dbg[2];
                    pos += snprintf(msg + pos, sizeof(msg) - pos,
                                    " gpu%d[%s arr=%d spin=%d lastOther=%d wbMine=%d]",
                                    item.devices[i],
                                    qstat[i] == cudaSuccess ? "done" : "busy",
                                    arr, spin, last, wb);
                }
                GGML_LOG_WARN("%s\n", msg);
            }

            if (all_done) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(GGML_CUDA_AR_WDOG_POLL_MS));
            elapsed_ms += GGML_CUDA_AR_WDOG_POLL_MS;
        }
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
    p->debug_buf      = nullptr;
    p->wdog_timeout_ms = 0;
    p->wdog_max_spin   = 0;
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
    // Debug buffer: written by the instrumented kernel, polled by the host.
    {
        const size_t dbg_bytes = (size_t)n_devices * GGML_CUDA_AR_DEBUG_INTS * sizeof(int);
        if (cudaHostAlloc(reinterpret_cast<void **>(&p->debug_buf), dbg_bytes,
                          cudaHostAllocPortable) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaHostAlloc for debug buffer failed (%zu bytes)\n",
                           __func__, dbg_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        memset(p->debug_buf, 0, dbg_bytes);

        const char * wdog_env = getenv("GGML_CUDA_AR_WATCHDOG");
        const char * spin_env = getenv("GGML_CUDA_AR_MAX_SPIN");
        p->wdog_timeout_ms = (wdog_env && wdog_env[0]) ? atoi(wdog_env) : 0;
        p->wdog_max_spin   = (spin_env && spin_env[0]) ? atoi(spin_env) : 0;
        GGML_LOG_INFO("%s: AR watchdog enabled — timeout=%dms max_spin=%d "
                      "(set GGML_CUDA_AR_WATCHDOG=<ms> / GGML_CUDA_AR_MAX_SPIN=<n> to adjust)\n",
                      __func__, p->wdog_timeout_ms, p->wdog_max_spin);

        // Start the background polling thread.
        p->wdog_thr = std::thread(ggml_cuda_ar_wdog_thread, p);
    }
#endif

    // Warmup: run the kernel N times to pay first-use driver / PCIe /
    // page-mapping costs during model load and encourage the GPU clock
    // governor to boost before inference begins.
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
#ifdef GGML_CUDA_AR_WATCHDOG
    if (p->wdog_thr.joinable()) {
        {
            std::lock_guard<std::mutex> lk(p->wdog_mtx);
            p->wdog_stop = true;
        }
        p->wdog_cv.notify_one();
        p->wdog_thr.join();
    }
    if (p->debug_buf) {
        cudaFreeHost(p->debug_buf);
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

#ifdef GGML_CUDA_AR_WATCHDOG
    // Clear per-device debug state so the watchdog reads reflect only this call.
    memset(p->debug_buf, 0, (size_t)n * GGML_CUDA_AR_DEBUG_INTS * sizeof(int));

    // Collect ker events for the watchdog poll after the launch loop.
    cudaEvent_t ker_events[GGML_CUDA_MAX_DEVICES];
#endif

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
            p->debug_buf + i * GGML_CUDA_AR_DEBUG_INTS,
            p->wdog_max_spin,
            i);
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

#ifdef GGML_CUDA_AR_WATCHDOG
        ker_events[i] = ev.ker;
#endif
    }

#ifdef GGML_CUDA_AR_WATCHDOG
    // Post this slot to the background watchdog thread — non-blocking.
    {
        ggml_cuda_ar_wdog_item item;
        item.slot = slot;
        item.n    = n;
        for (int i = 0; i < n; ++i) {
            item.devices[i]    = p->devices[i];
            item.ker_events[i] = ker_events[i];
        }
        std::lock_guard<std::mutex> lk(p->wdog_mtx);
        p->wdog_queue.push_back(item);
    }
    p->wdog_cv.notify_one();
#endif

    return true;
}
