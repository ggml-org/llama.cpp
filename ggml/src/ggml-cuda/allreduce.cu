#include "allreduce.cuh"
#include "convert.cuh"
#include "ggml-impl.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

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

// Combined chunked-kernel AllReduce.  sendbuf/recvbuf live in Tdst (the
// caller's tensor type); host_mine/host_other carry data in Twire (the
// on-wire type, possibly narrower than Tdst).
//
// Phase 1 reads Tdst from sendbuf, casts each element to Twire, and packs
// 16-byte vectors into host_mine — for Tdst=F32, Twire=BF16 this halves the
// host bytes written.
//
// Phase 3 reads 16-byte Twire vectors from host_other, casts each element to
// Tdst, then sums with the local sendbuf value (also rounded through Twire
// for bit-equivalence between GPUs since both sides truncate).  When
// Tdst == Twire the casts are no-ops and behaviour matches the original
// homogeneous kernel.
template <typename Tdst, typename Twire>
static __global__ void ggml_cuda_ar_kernel(
        const Tdst  * __restrict__ sendbuf,
        Tdst        * __restrict__ recvbuf,
        Twire       * __restrict__ host_mine,
        const Twire * __restrict__ host_other,
        int                        count,
        int *                      arrival_mine,
        int *                      arrival_other
#if GGML_CUDA_AR_WATCHDOG
       ,ggml_cuda_ar_debug_ring * ring,
        int                       max_spin,
        int                       rank,
        int                       ar_slot
#endif
        ) {

    // 16-byte vector unit for the wire type.  Each phase-1 iter writes one
    // vector to host memory; each phase-3 iter reads one and produces
    // ELEMS_PER_VEC sums.
    constexpr int ELEMS_PER_VEC = 16 / sizeof(Twire);

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

    // Phase 1: cast sendbuf (Tdst) -> host_mine (Twire) and store as 16-byte vectors.
    {
        for (int i = tid; i < count_vec; i += nt) {
            const int off = i * ELEMS_PER_VEC;
            Twire wire[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = static_cast<Twire>(sendbuf[off + k]);
            }
            *reinterpret_cast<float4 *>(&host_mine[off]) =
                *reinterpret_cast<const float4 *>(wire);
        }
        if (tid < count - tail) {
            host_mine[tail + tid] = static_cast<Twire>(sendbuf[tail + tid]);
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

    // Phase 3: read peer's Twire vector, cast both sides through Twire for
    // bit-equivalence, sum in Tdst precision, and write back to recvbuf.
    {
        for (int i = tid; i < count_vec; i += nt) {
            const int off = i * ELEMS_PER_VEC;
            Twire wire[ELEMS_PER_VEC];
            *reinterpret_cast<float4 *>(wire) =
                *reinterpret_cast<const float4 *>(&host_other[off]);
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                const Twire d_low = static_cast<Twire>(sendbuf[off + k]);
                recvbuf[off + k] = static_cast<Tdst>(d_low) + static_cast<Tdst>(wire[k]);
            }
        }
        if (tid < count - tail) {
            const Twire d_low = static_cast<Twire>(sendbuf[tail + tid]);
            recvbuf[tail + tid] =
                static_cast<Tdst>(d_low) + static_cast<Tdst>(host_other[tail + tid]);
        }
    }
}

// Combined load-convert-add kernel.  The peer's contribution arrives as Tsrc
// (which may be a lower-precision type than Tdst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through Tsrc's precision via a static_cast — peer already truncated its own
// value the same way before sending — so both sides perform identical
// arithmetic.  When Tdst == Tsrc the round-trip cast is a no-op.
template <typename Tdst, typename Tsrc>
static __global__ void ggml_cuda_ar_add_kernel(
        Tdst       * __restrict__ dst,
        const Tsrc * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const Tsrc d_low = static_cast<Tsrc>(dst[i]);
        dst[i] = static_cast<Tdst>(d_low) + static_cast<Tdst>(src[i]);
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  128 is well above the actual
// in-flight depth (single digits in practice) while keeping init cost low.
// Two-slot ring is sufficient: lockstep guarantees the two GPUs are at most
// one AR (or chunk) apart, so slot[N%2] is always safe to reuse — peer has
// already consumed slot[N%2] from AR N-2 by the time we get to AR N.  The
// slot wraparound's cudaEventSynchronize on ev.ker covers the host-side
// arrival reset against the prior AR's kernel.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 2;

// Maximum chunk size (bytes per GPU) handled by one internal kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB

// Prototype copy-engine path for large F32 reductions.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_DEFAULT = 2 * 1024 * 1024; // 2 MB
// Minimum chunk size the env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB → up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
static constexpr int GGML_CUDA_AR_COPY_MAX_CHUNKS =
    static_cast<int>((GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) /
                    GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

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
    cudaEvent_t cpy[GGML_CUDA_AR_COPY_MAX_CHUNKS] = {};  // copy-engine D2H chunks complete
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in host_large[] / dev_tmp[]
    size_t   copy_threshold;
    size_t   copy_chunk_bytes;
    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    uint64_t call_count;
    uint64_t reduce_count;

    // Per-device resources.
    char *                   host_buf[GGML_CUDA_MAX_DEVICES];  // pinned staging
    char *                   host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging for copy-engine path
    char *                   dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    ggml_cuda_ar_event_slot *ev_pool[GGML_CUDA_MAX_DEVICES];   // [device][slot]

    // Arrival ring: pinned, ARRIVAL_STRIDE bytes between adjacent ints.
    // Use ggml_cuda_ar_arrival_ptr() to index.
    char * arrival;

    // Temporary host-side tracing for prefill AllReduce analysis.
    bool     trace_enabled;
    bool     trace_chunks;
    uint64_t trace_limit;
    uint64_t trace_chunk_limit;
    uint64_t trace_chunk_count;

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

static bool ggml_cuda_ar_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0 &&
           strcmp(value, "false") != 0 && strcmp(value, "FALSE") != 0;
}

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

static void ggml_cuda_ar_trace_call(
        const ggml_cuda_ar_pipeline * p,
        uint64_t reduce_id,
        const char * path,
        ggml_tensor ** tensors,
        ggml_type type,
        int64_t ne,
        size_t nbytes,
        size_t max_chunk_elems,
        size_t chunks) {
    if (!p->trace_enabled || reduce_id >= p->trace_limit) {
        return;
    }

    fprintf(stdout,
            "GGML_CUDA_AR_TRACE call=%" PRIu64
            " path=%s name=\"%s\" type=%s ne=%" PRId64 " nbytes=%zu chunks=%zu"
            " max_chunk_elems=%zu max_chunk_bytes=%zu"
            " flags=[0x%x,0x%x] compute=[%d,%d]"
            " data=[%p,%p]"
            " ne0=[%" PRId64 ",%" PRId64 "] ne1=[%" PRId64 ",%" PRId64 "]"
            " ne2=[%" PRId64 ",%" PRId64 "] ne3=[%" PRId64 ",%" PRId64 "]"
            " nb0=[%zu,%zu] nb1=[%zu,%zu] nb2=[%zu,%zu] nb3=[%zu,%zu]\n",
            reduce_id,
            path,
            tensors[0]->name,
            ggml_type_name(type),
            ne,
            nbytes,
            chunks,
            max_chunk_elems,
            max_chunk_elems * ggml_type_size(type),
            tensors[0]->flags,
            tensors[1]->flags,
            (tensors[0]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0,
            (tensors[1]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0,
            tensors[0]->data,
            tensors[1]->data,
            tensors[0]->ne[0], tensors[1]->ne[0],
            tensors[0]->ne[1], tensors[1]->ne[1],
            tensors[0]->ne[2], tensors[1]->ne[2],
            tensors[0]->ne[3], tensors[1]->ne[3],
            (size_t) tensors[0]->nb[0], (size_t) tensors[1]->nb[0],
            (size_t) tensors[0]->nb[1], (size_t) tensors[1]->nb[1],
            (size_t) tensors[0]->nb[2], (size_t) tensors[1]->nb[2],
            (size_t) tensors[0]->nb[3], (size_t) tensors[1]->nb[3]);
    fflush(stdout);
}

static void ggml_cuda_ar_trace_chunk(
        ggml_cuda_ar_pipeline * p,
        uint64_t reduce_id,
        const char * path,
        size_t chunk_index,
        int slot,
        int64_t chunk_start,
        size_t chunk_elems,
        size_t chunk_bytes,
        bool last_chunk) {
    if (!p->trace_enabled || !p->trace_chunks ||
            reduce_id >= p->trace_limit ||
            p->trace_chunk_count >= p->trace_chunk_limit) {
        return;
    }

    p->trace_chunk_count++;
    fprintf(stdout,
            "GGML_CUDA_AR_TRACE_CHUNK call=%" PRIu64
            " path=%s chunk=%zu slot=%d start=%" PRId64
            " elems=%zu bytes=%zu launches=%d last=%d\n",
            reduce_id,
            path,
            chunk_index,
            slot,
            chunk_start,
            chunk_elems,
            chunk_bytes,
            p->n_devices,
            last_chunk);
    fflush(stdout);
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
    p->n_devices         = n_devices;
    p->buf_bytes         = 0;
    p->copy_bytes        = GGML_CUDA_AR_COPY_MAX_BYTES;
    p->copy_threshold    = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
    p->copy_chunk_bytes  = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_CHUNK_BYTES", GGML_CUDA_AR_COPY_CHUNK_BYTES_DEFAULT);
    if (p->copy_chunk_bytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n",
                      __func__, p->copy_chunk_bytes, GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
        p->copy_chunk_bytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
    }
    p->bf16_threshold    = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 128 * 1024); // 128 KB default
    p->call_count        = 0;
    p->reduce_count      = 0;
    p->arrival           = nullptr;
    p->trace_enabled     = ggml_cuda_ar_env_u64("GGML_CUDA_AR_TRACE", 0) != 0 &&
                           !ggml_cuda_ar_env_enabled("GGML_CUDA_AR_TRACE_DISABLE");
    p->trace_chunks      = ggml_cuda_ar_env_u64("GGML_CUDA_AR_TRACE_CHUNKS", 1) != 0;
    p->trace_limit       = ggml_cuda_ar_env_u64("GGML_CUDA_AR_TRACE_LIMIT", 2048);
    p->trace_chunk_limit = ggml_cuda_ar_env_u64("GGML_CUDA_AR_TRACE_CHUNK_LIMIT", 8192);
    p->trace_chunk_count = 0;
    for (int i = 0; i < n_devices; ++i) {
        p->devices[i]  = devices[i];
        p->host_buf[i] = nullptr;
        p->host_large[i] = nullptr;
        p->dev_tmp[i]  = nullptr;
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
            bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            for (int c = 0; ok && c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                ok = cudaEventCreateWithFlags(&p->ev_pool[i][s].cpy[c], cudaEventDisableTiming) == cudaSuccess;
            }
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

    // Prototype copy-engine path resources. Keep these deliberately large for
    // now; memory footprint can be reduced after the bandwidth experiment.
    for (int i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (cudaHostAlloc(&p->host_large[i], p->copy_bytes, cudaHostAllocPortable) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaHostAlloc for large staging failed (%zu bytes)\n",
                           __func__, p->copy_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                           __func__, p->copy_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
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
    if (p->trace_enabled) {
        fprintf(stdout,
                "GGML_CUDA_AR_TRACE_INIT devices=%d staging_bytes=%zu pool=%d chunks=%d"
                " copy_bytes=%zu copy_threshold=%zu copy_chunk_bytes=%zu"
                " trace_limit=%" PRIu64 " chunk_limit=%" PRIu64 "\n",
                p->n_devices,
                p->buf_bytes,
                GGML_CUDA_AR_POOL_SIZE,
                p->trace_chunks,
                p->copy_bytes,
                p->copy_threshold,
                p->copy_chunk_bytes,
                p->trace_limit,
                p->trace_chunk_limit);
        fflush(stdout);
    }

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
        if (p->host_large[i]) {
            cudaFreeHost(p->host_large[i]);
        }
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaFree(p->dev_tmp[i]);
        }
        if (p->ev_pool[i]) {
            ggml_cuda_set_device(p->devices[i]);
            for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
                if (p->ev_pool[i][s].app) { cudaEventDestroy(p->ev_pool[i][s].app); }
                for (int c = 0; c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                    if (p->ev_pool[i][s].cpy[c]) { cudaEventDestroy(p->ev_pool[i][s].cpy[c]); }
                }
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

// Asymmetric copy_impl: data sent over PCIe in Tsrc precision (one element of
// nbytes per ne element); accumulated locally into a Tdst buffer.  When
// Tsrc == Tdst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through Tsrc
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename Tsrc, typename Tdst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        Tsrc * const            src_buf[GGML_CUDA_MAX_DEVICES],
        Tdst * const            dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        uint64_t                reduce_id,
        int64_t                 ne,
        size_t                  nbytes,
        const char *            trace_label) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());
    GGML_ASSERT(p->copy_chunk_bytes > 0);

    const int slot = ggml_cuda_ar_acquire_slot(p);
    const size_t copy_chunks = (nbytes + p->copy_chunk_bytes - 1) / p->copy_chunk_bytes;
    GGML_ASSERT(copy_chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    // Stage 1: both GPUs copy their local contribution to pinned host memory.
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);

        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * p->copy_chunk_bytes;
            const size_t chunk_bytes = (nbytes - offset) < p->copy_chunk_bytes ?
                (nbytes - offset) : p->copy_chunk_bytes;

            if (i == 0) {
                ggml_cuda_ar_trace_chunk(
                    p, reduce_id, trace_label, c, slot, offset / sizeof(Tsrc),
                    chunk_bytes / sizeof(Tsrc), chunk_bytes, c + 1 == copy_chunks);
            }

            CUDA_CHECK(cudaMemcpyAsync(
                p->host_large[i] + offset, reinterpret_cast<char *>(src_buf[i]) + offset, chunk_bytes,
                cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }

    // Stage 2: each GPU waits for each peer D2H chunk, pulls that chunk back to
    // local scratch, then performs one device-local add over the assembled peer tensor.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * p->copy_chunk_bytes;
            const size_t chunk_bytes = (nbytes - offset) < p->copy_chunk_bytes ?
                (nbytes - offset) : p->copy_chunk_bytes;

            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i] + offset, p->host_large[peer] + offset, chunk_bytes,
                cudaMemcpyHostToDevice, p->streams[i]));
        }

        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_add_kernel<Tdst, Tsrc><<<n_blocks, block_size, 0, p->streams[i]>>>(
            dst_buf[i],
            reinterpret_cast<const Tsrc *>(p->dev_tmp[i]),
            (int) ne);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_ar_record_chunk_done(p, cuda_ctx[i], i, slot, true);
    }

    return true;
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const uint64_t reduce_id = p->reduce_count++;
    const size_t   input_nbytes = ggml_nbytes(tensors[0]);

    // BF16 round-trip: F32 inputs >= bf16_threshold are converted to BF16 for
    // the reduction (chunked or copy-engine), halving on-wire bytes. Matches
    // NCCL's behaviour. The pre-conversion zeroes inactive shards so the
    // inner paths see them as already-prepared compute tensors.
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;

    const ggml_type kernel_type = use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t    type_size   = ggml_type_size(kernel_type);
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes      = (size_t) ne * type_size;

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    // Decide between copy-engine and chunked-kernel paths based on the working
    // type's actual byte count.
    const bool use_copy_engine =
        p->copy_threshold > 0 &&
        nbytes >= p->copy_threshold &&
        nbytes <= p->copy_bytes;

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // (chunked-kernel path) and the combined add kernel (copy_engine path)
    // both accumulate into the F32 tensor data directly, so an inactive
    // shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked-kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            CUDA_CHECK(cudaGetLastError());
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    bool ok = true;
    if (use_copy_engine) {
        const char * label = use_bf16 ? "copy_engine-bf16" : "copy_engine";
        const size_t copy_chunk_elems = p->copy_chunk_bytes / type_size;
        const size_t copy_chunks = (nbytes + p->copy_chunk_bytes - 1) / p->copy_chunk_bytes;
        ggml_cuda_ar_trace_call(p, reduce_id, label, tensors, kernel_type, ne, nbytes, copy_chunk_elems, copy_chunks);

        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
        // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
        // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
        // through BF16 for bit-equivalence and writes F32 directly, so no
        // post-conversion is needed.  Otherwise src == dst (same native type).
        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            __nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES];
            float         * dst[GGML_CUDA_MAX_DEVICES];
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<__nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_impl<__nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, reduce_id, ne, nbytes, label);
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES];
                    for (int i = 0; i < n; ++i) buf[i] = static_cast<float *>(tensors[i]->data);
                    ok = ggml_cuda_ar_allreduce_copy_impl<float, float>(
                        p, backends, buf, buf, inner_compute, reduce_id, ne, nbytes, label);
                    break;
                }
                case GGML_TYPE_BF16: {
                    __nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES];
                    for (int i = 0; i < n; ++i) buf[i] = static_cast<__nv_bfloat16 *>(tensors[i]->data);
                    ok = ggml_cuda_ar_allreduce_copy_impl<__nv_bfloat16, __nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, reduce_id, ne, nbytes, label);
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        const char * label = use_bf16 ? "kernel-bf16" : "kernel";
        // host_buf carries Twire-typed data; max_chunk_elems is the count that
        // fits in one host_buf at the wire size.
        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t chunks = ((size_t) ne + max_chunk_elems - 1) / max_chunk_elems;
        const size_t input_type_size = ggml_type_size(input_type);
        ggml_cuda_ar_trace_call(p, reduce_id, label, tensors, kernel_type, ne, nbytes, max_chunk_elems, chunks);

        // Chunked-kernel path runs entirely on the caller's compute stream:
        // since AR is a barrier here, same-stream ordering replaces the
        // wait_for_compute / record_chunk_done event pairs and skips the
        // cross-stream scheduling overhead that was hurting the small-tensor
        // (tg) latency on the AR-stream variant.  Only ev.ker is still
        // recorded at end-of-AR for acquire_slot's pool-wraparound check.
        size_t chunk_index = 0;
        for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems, ++chunk_index) {
            const size_t remaining_elems = (size_t) (ne - chunk_start);
            const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
            const size_t chunk_wire_bytes = chunk_elems * type_size;
            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const int slot = ggml_cuda_ar_acquire_slot(p);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;
            ggml_cuda_ar_trace_chunk(p, reduce_id, label, chunk_index, slot, chunk_start, chunk_elems, chunk_wire_bytes, last_chunk);

            for (int i = 0; i < n; ++i) {
                const int peer = 1 - i;  // valid for n == 2 only
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                cudaStream_t stream = cuda_ctx->stream();

                char * data = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) input_type_size;

                // Match NCCL/meta-backend semantics: inactive shards contribute
                // zeros.  On the BF16 path the F32 tensor data was already
                // zeroed up-front (above), so per-chunk zeroing isn't needed.
                if (!compute_flag[i] && !use_bf16) {
                    CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_dst_bytes, stream));
                }

#if GGML_CUDA_AR_WATCHDOG
#define GGML_CUDA_AR_WDOG_EXTRA_ARGS , p->debug_ring[i], p->wdog_max_spin, i, slot
#else
#define GGML_CUDA_AR_WDOG_EXTRA_ARGS
#endif

#define LAUNCH_AR_KERNEL(Tdst, Twire) \
                ggml_cuda_ar_kernel<Tdst, Twire><<<dim3(1), dim3(256), 0, stream>>>( \
                    reinterpret_cast<const Tdst *>(data), \
                    reinterpret_cast<Tdst *>(data), \
                    reinterpret_cast<Twire *>(p->host_buf[i]), \
                    reinterpret_cast<const Twire *>(p->host_buf[peer]), \
                    static_cast<int>(chunk_elems), \
                    ggml_cuda_ar_arrival_ptr(p, slot, i), \
                    ggml_cuda_ar_arrival_ptr(p, slot, peer) \
                    GGML_CUDA_AR_WDOG_EXTRA_ARGS)

                if (use_bf16) {
                    GGML_ASSERT(input_type == GGML_TYPE_F32);
                    LAUNCH_AR_KERNEL(float, __nv_bfloat16);
                } else {
                    switch (input_type) {
                        case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float,         float);         break;
                        case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half,          half);          break;
                        case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(__nv_bfloat16, __nv_bfloat16); break;
                        default: GGML_ASSERT(false);
                    }
                }

#undef LAUNCH_AR_KERNEL
#undef GGML_CUDA_AR_WDOG_EXTRA_ARGS
                CUDA_CHECK(cudaGetLastError());

                if (last_chunk) {
                    CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
                }
            }
        }
    }

    return ok;
}
