#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "kv-stream-partial.cuh"
#include "common.cuh"
#include "../ggml-backend-impl.h"
#include <memory>

namespace {
struct copy_queue {
    ggml_backend_cuda_context * context;
    ggml_backend_buffer_t device = nullptr, host = nullptr;
    ggml_kv_stream_layout page, ring;
    ggml_kv_stream_copy_state state;
    cudaStream_t stream = nullptr;
    cudaEvent_t producer = nullptr;
    std::vector<cudaEvent_t> ready, consumed;
    copy_queue(ggml_backend_cuda_context * context, size_t slots) : context(context), state(slots), ready(slots), consumed(slots) {}
    // Retire all pointer users before destroying events or releasing pinned/device backing.
    ~copy_queue() {
        ggml_cuda_set_device(context->device);
        if (stream) { CUDA_CHECK(cudaStreamSynchronize(stream)); CUDA_CHECK(cudaStreamSynchronize(context->stream())); }
        for (auto e : ready) if (e) CUDA_CHECK(cudaEventDestroy(e));
        for (auto e : consumed) if (e) CUDA_CHECK(cudaEventDestroy(e));
        if (producer) CUDA_CHECK(cudaEventDestroy(producer));
        if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
        ggml_backend_buffer_free(host); ggml_backend_buffer_free(device);
    }
};

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
    CUDA_CHECK(cudaEventRecord(q.producer,q.context->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(q.stream,q.producer,0));
    return true;
}

// Queue independent K/V copies and finite tail padding after the previous final consumer retires.
static bool enqueue(void * handle, size_t slot, const void * k, const void * v, size_t live, size_t padded) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!q.state.can_queue(slot) || !live || live > padded || padded > q.page.tokens ||
            !source_range(q,k,live*q.page.k_token_bytes) || !source_range(q,v,live*q.page.v_token_bytes)) return false;
    ggml_cuda_set_device(q.context->device);
    if (q.state.recycled(slot)) CUDA_CHECK(cudaStreamWaitEvent(q.stream,q.consumed[slot],0));
    auto * base = static_cast<char *>(ggml_backend_buffer_get_base(q.device));
    for (int value = 0; value < 2; ++value) {
        const size_t stride = value ? q.page.v_token_bytes : q.page.k_token_bytes;
        const size_t offset = value ? q.ring.v_offset+slot*q.page.v_bytes : slot*q.page.k_bytes;
        CUDA_CHECK(cudaMemcpyAsync(base+offset,value ? v : k,live*stride,cudaMemcpyHostToDevice,q.stream));
        if (padded > live) CUDA_CHECK(cudaMemsetAsync(base+offset+live*stride,0,(padded-live)*stride,q.stream));
    }
    CUDA_CHECK(cudaEventRecord(q.ready[slot],q.stream));
    return q.state.queue(slot);
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
static bool acquire(void * handle, size_t slot) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!q.state.waiting(slot)) return false;
    ggml_cuda_set_device(q.context->device);
    CUDA_CHECK(cudaStreamWaitEvent(q.context->stream(),q.ready[slot],0));
    return q.state.acquire(slot);
}

// Every encoded-slot read must be submitted before recording this final-consumer event.
static bool release(void * handle, size_t slot) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!q.state.held(slot)) return false;
    ggml_cuda_set_device(q.context->device);
    CUDA_CHECK(cudaEventRecord(q.consumed[slot],q.context->stream()));
    return q.state.release(slot);
}

// Cancellation drops logical ownership only after both streams complete outstanding work.
static void drain(void * handle) {
    if (!handle) return;
    auto & q = *static_cast<copy_queue *>(handle);
    ggml_cuda_set_device(q.context->device);
    CUDA_CHECK(cudaStreamSynchronize(q.stream)); CUDA_CHECK(cudaStreamSynchronize(q.context->stream()));
    q.state.drained();
}

// Destruction also drains, so no retained allocation can outlive its final GPU use.
static void destroy(void * handle) { delete static_cast<copy_queue *>(handle); }
} // namespace

// Expose the CUDA adapter through an opaque, backend-neutral ownership contract.
const ggml_kv_stream_copy_ops * ggml_cuda_kv_stream_copy_ops() {
    static const ggml_kv_stream_copy_ops ops{1,create,begin,enqueue,ready,acquire,release,drain,destroy};
    return &ops;
}
#endif
