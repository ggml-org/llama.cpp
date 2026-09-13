#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "kv-stream-partial.cuh"
#include "common.cuh"
#include "../ggml-backend-impl.h"
#include <memory>
#include <chrono>

namespace {
struct measurement {
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<cudaEvent_t> start, end;
    std::vector<size_t> pending;
    std::vector<uint64_t> tickets;
    uint64_t ticket = 0;
    size_t occupied = 0;
    bool enabled = false;
    ggml_kv_stream_copy_feedback result;
    std::chrono::steady_clock::time_point begin;
    explicit measurement(size_t slots) : start(slots), end(slots), pending(slots), tickets(slots) {}
    // The queue drains and selects its device before these diagnostic resources are destroyed.
    ~measurement() {
        for (auto e : start) if (e) CUDA_CHECK(cudaEventDestroy(e));
        for (auto e : end) if (e) CUDA_CHECK(cudaEventDestroy(e));
        ggml_backend_buffer_free(buffer);
    }
    uint64_t * data() const { return static_cast<uint64_t *>(ggml_backend_buffer_get_base(buffer)); }
    // Harvest only completed samples; a busy timing slot skips new sampling instead of blocking copies.
    void collect(size_t slot) {
        if (!pending[slot]) return;
        const auto status = cudaEventQuery(end[slot]);
        if (status == cudaErrorNotReady) return;
        CUDA_CHECK(status);
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms,start[slot],end[slot]));
        result.copy_ms += ms; result.timed_bytes += pending[slot]; pending[slot] = 0;
    }
};

// Tickets avoid resetting a reused flag after the next consumer has already checked it.
static __global__ void publish_ready(uint64_t * flags, size_t first, size_t count, uint64_t ticket) {
    for (size_t i = threadIdx.x; i < count; i += blockDim.x)
        atomicExch(reinterpret_cast<unsigned long long *>(flags+first+i),static_cast<unsigned long long>(ticket));
}

// Sample on the consuming stream, not when the host submits the future dependency.
static __global__ void sample_deadline(uint64_t * flags, size_t slots, size_t first, size_t count, uint64_t ticket) {
    bool missing = false;
    for (size_t i = 0; i < count; ++i)
        missing |= atomicAdd(reinterpret_cast<unsigned long long *>(flags+first+i),0ULL) != ticket;
    ++flags[slots]; flags[slots+1] += missing;
}

struct copy_queue {
    ggml_backend_cuda_context * context;
    ggml_backend_buffer_t device = nullptr, host = nullptr;
    ggml_kv_stream_layout page, ring;
    ggml_kv_stream_copy_state state;
    ggml_kv_stream_copy_stats statistics;
    cudaStream_t stream = nullptr;
    cudaEvent_t producer = nullptr;
    std::vector<cudaEvent_t> ready, consumed;
    std::vector<uint8_t> completed;
    std::unique_ptr<measurement> timing;
    copy_queue(ggml_backend_cuda_context * context, size_t slots) : context(context), state(slots), ready(slots), consumed(slots), completed(slots,false) {}
    // Retire all pointer users before destroying events or releasing pinned/device backing.
    ~copy_queue() {
        ggml_cuda_set_device(context->device);
        if (stream) { CUDA_CHECK(cudaStreamSynchronize(stream)); CUDA_CHECK(cudaStreamSynchronize(context->stream())); }
        timing.reset();
        for (auto e : ready) if (e) CUDA_CHECK(cudaEventDestroy(e));
        for (auto e : consumed) if (e) CUDA_CHECK(cudaEventDestroy(e));
        if (producer) CUDA_CHECK(cudaEventDestroy(producer));
        if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
        ggml_backend_buffer_free(host); ggml_backend_buffer_free(device);
    }
};

// Allocate O(ring slots) diagnostics through the explicit device-local buffer type, never the KV region.
static bool measure(void * handle, bool enable) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (q.state.running()) return false;
    ggml_cuda_set_device(q.context->device);
    if (!enable) { q.timing.reset(); return true; }
    if (enable && !q.timing) {
        try {
            const size_t slots = q.ready.size();
            if (slots > SIZE_MAX/sizeof(uint64_t)-2) return false;
            auto m = std::make_unique<measurement>(slots);
            // Lazy kernel loading can synchronize the context. Resolve kernels before any producer/consumer gate.
            cudaFuncAttributes attributes;
            if (cudaFuncGetAttributes(&attributes,publish_ready) != cudaSuccess ||
                    cudaFuncGetAttributes(&attributes,sample_deadline) != cudaSuccess) {
                (void) cudaGetLastError(); return false;
            }
            m->buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_device_buffer_type(q.context->device),(slots+2)*sizeof(uint64_t));
            if (!m->buffer) return false;
            for (size_t i = 0; i < slots; ++i) {
                if (cudaEventCreate(&m->start[i]) != cudaSuccess || cudaEventCreate(&m->end[i]) != cudaSuccess) {
                    (void) cudaGetLastError(); return false;
                }
            }
            q.timing = std::move(m);
        } catch (const std::bad_alloc &) { return false; }
    }
    if (q.timing) { q.timing->enabled = enable; q.timing->result = {}; }
    return true;
}

// Source spans must remain inside the retained pinned allocation; zero or wrapping ranges are invalid.
static bool source_range(const copy_queue & q, const void * pointer, size_t bytes) {
    const auto base = uintptr_t(ggml_backend_buffer_get_base(q.host)), p = uintptr_t(pointer);
    const size_t capacity = ggml_backend_buffer_get_size(q.host);
    return pointer && bytes && p >= base && p-base <= capacity && bytes <= capacity-(p-base) && p <= UINTPTR_MAX-bytes;
}

// Allocate only stream/event bookkeeping. KV bytes already belong to the caller's device and host buffers.
static void * create(ggml_backend_t backend, ggml_backend_buffer_t device, ggml_backend_buffer_t host,
        const ggml_kv_stream_shape & shape, size_t slots) {
    if (!backend || !ggml_backend_is_cuda(backend) || !device || !host || !slots || shape.page_tokens <= 0 ||
            uint64_t(shape.page_tokens) > SIZE_MAX || slots > SIZE_MAX/size_t(shape.page_tokens) || !ggml_backend_buffer_is_host(host)) return nullptr;
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (ggml_backend_buffer_get_type(device) != ggml_backend_cuda_device_buffer_type(ctx->device)) return nullptr;
    ggml_kv_stream_layout page, ring;
    if (ggml_kv_stream_layout_make(shape,size_t(shape.page_tokens),page).status != ggml_kv_stream_status::success ||
            ggml_kv_stream_layout_make(shape,slots*size_t(shape.page_tokens),ring).status != ggml_kv_stream_status::success ||
            ring.bytes > ggml_backend_buffer_get_size(device) || !page.bytes || page.v_offset != page.k_bytes ||
            page.k_bytes%shape.alignment || page.v_bytes%shape.alignment) return nullptr;
    const auto base = uintptr_t(ggml_backend_buffer_get_base(device));
    if (!base || base%shape.alignment || base > UINTPTR_MAX-ring.bytes) return nullptr;
    ggml_cuda_set_device(ctx->device);
    cudaPointerAttributes attributes = {};
    if (cudaPointerGetAttributes(&attributes,ggml_backend_buffer_get_base(host)) != cudaSuccess) {
        (void) cudaGetLastError(); return nullptr;
    }
    if (attributes.type != cudaMemoryTypeHost) return nullptr;
    try {
        auto q = std::make_unique<copy_queue>(ctx,slots);
        q->page = page; q->ring = ring;
        q->device = ggml_backend_buffer_retain(device); q->host = ggml_backend_buffer_retain(host);
        if (cudaStreamCreateWithFlags(&q->stream,cudaStreamNonBlocking) != cudaSuccess ||
                cudaEventCreateWithFlags(&q->producer,cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError(); return nullptr;
        }
        for (size_t i = 0; i < slots; ++i) {
            if (cudaEventCreateWithFlags(&q->ready[i],cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&q->consumed[i],cudaEventDisableTiming) != cudaSuccess) {
                (void) cudaGetLastError(); return nullptr;
            }
        }
        return q.release();
    } catch (const std::bad_alloc &) { return nullptr; }
}

// Fence earlier compute producers before any ring writes; active CUDA capture is unsupported.
static bool begin(void * handle) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    ggml_cuda_set_device(q.context->device);
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(q.context->stream(),&status));
    if (status != cudaStreamCaptureStatusNone || !q.state.begin()) return false;
    q.statistics = {};
    if (q.timing && q.timing->enabled) {
        auto & m = *q.timing;
        m.result = {}; m.ticket = 0; m.occupied = 0;
        std::fill(m.pending.begin(),m.pending.end(),0);
        m.begin = std::chrono::steady_clock::now();
        m.result.instrumentation_bytes = ggml_backend_buffer_get_size(m.buffer);
        CUDA_CHECK(cudaMemsetAsync(m.data(),0,(q.ready.size()+2)*sizeof(uint64_t),q.context->stream()));
    }
    CUDA_CHECK(cudaEventRecord(q.producer,q.context->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(q.stream,q.producer,0));
    return true;
}

// Queue independent K/V copies and finite tail padding after the previous final consumer retires.
static bool enqueue_span(void * handle, size_t slot, const void * k, const void * v, size_t live, size_t padded) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!live || live > padded || padded > q.ring.tokens) return false;
    const size_t count = padded/q.page.tokens+(padded%q.page.tokens != 0);
    if (!q.state.can_queue_span(slot,count) ||
            !source_range(q,k,live*q.page.k_token_bytes) || !source_range(q,v,live*q.page.v_token_bytes)) return false;
    auto * m = q.timing && q.timing->enabled ? q.timing.get() : nullptr;
    if (m && m->ticket == UINT64_MAX) return false;
    ggml_cuda_set_device(q.context->device);
    for (size_t i = 0; i < count; ++i)
        if (q.state.recycled(slot+i) && !q.completed[slot+i]) CUDA_CHECK(cudaStreamWaitEvent(q.stream,q.consumed[slot+i],0));
    auto * base = static_cast<char *>(ggml_backend_buffer_get_base(q.device));
    bool timed = false;
    if (m) {
        m->collect(slot); timed = !m->pending[slot];
        if (timed) CUDA_CHECK(cudaEventRecord(m->start[slot],q.stream));
    }
    for (int value = 0; value < 2; ++value) {
        const size_t stride = value ? q.page.v_token_bytes : q.page.k_token_bytes;
        const size_t offset = value ? q.ring.v_offset+slot*q.page.v_bytes : slot*q.page.k_bytes;
        CUDA_CHECK(cudaMemcpyAsync(base+offset,value ? v : k,live*stride,cudaMemcpyHostToDevice,q.stream));
        q.statistics.bytes += live*stride; ++q.statistics.calls;
        if (padded > live) CUDA_CHECK(cudaMemsetAsync(base+offset+live*stride,0,(padded-live)*stride,q.stream));
    }
    if (m) {
        if (timed) {
            CUDA_CHECK(cudaEventRecord(m->end[slot],q.stream));
            m->pending[slot] = live*(q.page.k_token_bytes+q.page.v_token_bytes);
        }
        const uint64_t ticket = ++m->ticket;
        for (size_t i = 0; i < count; ++i) m->tickets[slot+i] = ticket;
        publish_ready<<<1,32,0,q.stream>>>(m->data(),slot,count,ticket);
        CUDA_CHECK(cudaGetLastError());
        m->occupied += count;
        m->result.peak_slots = std::max(m->result.peak_slots,m->occupied);
    }
    for (size_t i = 0; i < count; ++i) CUDA_CHECK(cudaEventRecord(q.ready[slot+i],q.stream));
    return q.state.queue_span(slot,count);
}

// Keep v1's one-slot admission boundary for existing callers.
static bool enqueue(void * handle, size_t slot, const void * k, const void * v, size_t live, size_t padded) {
    if (!handle || padded > static_cast<copy_queue *>(handle)->page.tokens) return false;
    return enqueue_span(handle,slot,k,v,live,padded);
}

// Report actual DMA submissions, not an inferred count from the consumer's plan.
static ggml_kv_stream_copy_stats stats(void * handle) {
    return handle ? static_cast<copy_queue *>(handle)->statistics : ggml_kv_stream_copy_stats{};
}

// Observation does not transfer ownership or replace the compute stream's mandatory event wait.
static bool ready(void * handle, size_t slot) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!q.state.waiting(slot) && !q.state.held(slot)) return false;
    ggml_cuda_set_device(q.context->device);
    const auto result = cudaEventQuery(q.ready[slot]);
    if (result == cudaErrorNotReady) return false;
    CUDA_CHECK(result); return true;
}

// Queue the consumer dependency without blocking the host on transfer completion.
static bool acquire_span(void * handle, size_t slot, size_t count) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!count || slot >= q.ready.size() || count > q.ready.size()-slot) return false;
    for (size_t i = 0; i < count; ++i) if (!q.state.waiting(slot+i)) return false;
    auto * m = q.timing && q.timing->enabled ? q.timing.get() : nullptr;
    // One consumed span must come from a single enqueue_span call.
    if (m) for (size_t i = 1; i < count; ++i) if (m->tickets[slot+i] != m->tickets[slot]) return false;
    ggml_cuda_set_device(q.context->device);
    if (m) {
        sample_deadline<<<1,1,0,q.context->stream()>>>(m->data(),q.ready.size(),slot,count,m->tickets[slot]);
        CUDA_CHECK(cudaGetLastError());
    }
    for (size_t i = 0; i < count; ++i) {
        CUDA_CHECK(cudaStreamWaitEvent(q.context->stream(),q.ready[slot+i],0));
        GGML_ASSERT(q.state.acquire(slot+i));
    }
    return true;
}

// Preserve the one-page interface while using the same deadline sampling boundary.
static bool acquire(void * handle, size_t slot) { return acquire_span(handle,slot,1); }

// Every encoded-slot read must be submitted before recording this final-consumer event.
static bool release(void * handle, size_t slot) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!q.state.held(slot)) return false;
    ggml_cuda_set_device(q.context->device);
    CUDA_CHECK(cudaEventRecord(q.consumed[slot],q.context->stream()));
    q.completed[slot] = false;
    if (q.timing && q.timing->enabled) --q.timing->occupied;
    return q.state.release(slot);
}

// The caller already completed every reader; retain the queued-fence path for asynchronous consumers.
static bool release_completed(void * handle, size_t slot) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!q.state.held(slot)) return false;
    q.completed[slot] = true;
    if (q.timing && q.timing->enabled) --q.timing->occupied;
    return q.state.release(slot);
}

// Cancellation drops logical ownership only after both streams complete outstanding work.
static void drain(void * handle) {
    if (!handle) return;
    auto & q = *static_cast<copy_queue *>(handle);
    ggml_cuda_set_device(q.context->device);
    CUDA_CHECK(cudaStreamSynchronize(q.stream)); CUDA_CHECK(cudaStreamSynchronize(q.context->stream()));
    if (q.state.running() && q.timing && q.timing->enabled) {
        auto & m = *q.timing;
        for (size_t i = 0; i < q.ready.size(); ++i) m.collect(i);
        uint64_t counters[2];
        CUDA_CHECK(cudaMemcpy(counters,m.data()+q.ready.size(),sizeof(counters),cudaMemcpyDeviceToHost));
        m.result.samples = counters[0]; m.result.misses = counters[1];
        m.result.bytes = q.statistics.bytes;
        m.result.elapsed_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-m.begin).count();
        m.result.available = true;
    }
    q.state.drained();
}

// Do not expose partial GPU counters or incompletely harvested timing events.
static ggml_kv_stream_copy_feedback feedback(void * handle) {
    if (!handle) return {};
    const auto & q = *static_cast<copy_queue *>(handle);
    return q.timing && q.timing->enabled && !q.state.running() ? q.timing->result : ggml_kv_stream_copy_feedback{};
}

// Destruction also drains, so no retained allocation can outlive its final GPU use.
static void destroy(void * handle) { delete static_cast<copy_queue *>(handle); }
} // namespace

// Expose the CUDA adapter through an opaque, backend-neutral ownership contract.
const ggml_kv_stream_copy_ops * ggml_cuda_kv_stream_copy_ops() {
    static const ggml_kv_stream_copy_ops ops{4,create,begin,enqueue,ready,acquire,release,drain,destroy,enqueue_span,stats,release_completed,measure,acquire_span,feedback};
    return &ops;
}
#endif
