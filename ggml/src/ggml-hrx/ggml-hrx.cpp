#include "ggml-hrx.h"

#include "backend-buffer-binding.h"
#include "backend-context.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "graph/op-params.h"
#include "hrx_runtime.h"
#include "kernel-corpus/kernel-corpus.h"
#include "loom-jit.h"
#include "runtime/graph-executor.h"
#include "runtime/graph-program-cache.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/prepared-command-program-cache.h"
#include "runtime/transient-arena.h"

#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

static constexpr size_t      GGML_HRX_ALIGNMENT     = 256;
// GGML represents tensor locations as host pointers and derives view/arena offsets with ordinary pointer arithmetic.
// Device-local HRX buffers have no host address to return, so expose a non-null sentinel base as an offset coordinate.
static constexpr uintptr_t   GGML_HRX_FAKE_PTR_BASE = 0x1000;
static std::atomic<uint64_t> g_allocation_generation{ 1 };

static bool environment_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static void log_hrx_device_event(void *, const hrx_device_event_t * event) {
    if (event == nullptr || event->type != HRX_DEVICE_EVENT_TYPE_ASAN_REPORT || event->payload.data == nullptr ||
        event->payload.data_length < sizeof(hrx_device_asan_report_t)) {
        return;
    }
    hrx_device_asan_report_t report;
    std::memcpy(&report, event->payload.data, sizeof(report));
    GGML_LOG_ERROR("HRX ASAN: executable=%" PRIu64 " export=%u site=%" PRIu64 " access=%u address=0x%016" PRIx64
                   " length=%" PRIu64 " workgroup=(%u,%u,%u) workitem=(%u,%u,%u) shadow=0x%016" PRIx64
                   " value=0x%016" PRIx64 " dispatch=0x%016" PRIx64 "\n",
                   event->source.executable_id, event->source.export_ordinal, report.site_id, report.access_kind,
                   report.fault_address, report.access_length, report.workgroup_id[0], report.workgroup_id[1],
                   report.workgroup_id[2], report.workitem_id[0], report.workitem_id[1], report.workitem_id[2],
                   report.shadow_address, report.shadow_value, report.source_dispatch_ptr);
}

static bool hrx_check(hrx_status_t status, const char * expression, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }
    char * message = nullptr;
    size_t length  = 0;
    hrx_status_to_string(status, &message, &length);
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expression,
                   message != nullptr ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

#define HRX_CHECK(expression) hrx_check((expression), #expression, __FILE__, __LINE__)

}  // namespace

ggml_backend_hrx_reg_context::~ggml_backend_hrx_reg_context() {
    for (auto & context : device_contexts) {
        if (context->buffer_stream != nullptr) {
            hrx_stream_release(context->buffer_stream);
        }
        if (context->device != nullptr) {
            hrx_device_release(context->device);
        }
    }
    if (initialized) {
        hrx_status_t status = hrx_gpu_shutdown();
        if (!hrx_status_is_ok(status)) {
            hrx_status_ignore(status);
        }
    }
}

namespace {

static std::optional<std::string> device_string_property(hrx_device_t          device,
                                                         hrx_device_property_t property,
                                                         const char *          property_name) {
    std::vector<char> buffer(64);
    while (buffer.size() <= 4096) {
        hrx_status_t status = hrx_device_get_property(device, property, buffer.data(), buffer.size());
        if (hrx_status_is_ok(status)) {
            return std::string(buffer.data());
        }
        if (hrx_status_code(status) != HRX_STATUS_OUT_OF_RANGE) {
            hrx_check(status, property_name, __FILE__, __LINE__);
            return std::nullopt;
        }
        hrx_status_ignore(status);
        buffer.resize(buffer.size() * 2);
    }
    GGML_LOG_ERROR("%s exceeds the maximum supported property string length\n", property_name);
    return std::nullopt;
}

static ggml_guid_t ggml_backend_hrx_guid() {
    static ggml_guid guid = {
        0xd2, 0x3d, 0x72, 0x83, 0xb2, 0x82, 0x4d, 0xe0, 0x8a, 0x3e, 0x21, 0x1d, 0x68, 0x87, 0x2f, 0x4b,
    };
    return &guid;
}

static ggml_backend_hrx_device_context * device_context(ggml_backend_dev_t device) {
    return static_cast<ggml_backend_hrx_device_context *>(device->context);
}

static ggml_backend_hrx_buffer_context * buffer_context(ggml_backend_buffer_t buffer) {
    return ggml_backend_hrx_buffer_context_from_buffer(buffer);
}

static size_t tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor) {
    return ggml_backend_hrx_tensor_offset(context, tensor);
}

static const char * buffer_type_name(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context)->name.c_str();
}

static bool buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context)->host_visible;
}

static bool buffer_submit_and_wait(ggml_backend_hrx_device_context * device,
                                   hrx_status_t (*submit)(hrx_stream_t, void *),
                                   void * user_data) {
    std::lock_guard<std::mutex> lock(device->buffer_stream_mutex);
    if (!HRX_CHECK(submit(device->buffer_stream, user_data))) {
        return false;
    }
    return HRX_CHECK(hrx_stream_synchronize(device->buffer_stream));
}

struct FillBufferArgs {
    hrx_buffer_t buffer;
    size_t       offset;
    size_t       size;
    uint8_t      value;
};

static hrx_status_t submit_fill_buffer(hrx_stream_t stream, void * user_data) {
    auto * args = static_cast<FillBufferArgs *>(user_data);
    return hrx_stream_fill_buffer(stream, args->buffer, args->offset, args->size, &args->value, sizeof(args->value));
}

struct CopyBufferArgs {
    hrx_buffer_t source;
    size_t       source_offset;
    hrx_buffer_t destination;
    size_t       destination_offset;
    size_t       size;
};

static hrx_status_t submit_copy_buffer(hrx_stream_t stream, void * user_data) {
    auto * args = static_cast<CopyBufferArgs *>(user_data);
    return hrx_stream_copy_buffer(stream, args->source, args->source_offset, args->destination,
                                  args->destination_offset, args->size);
}

static void buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = buffer_context(buffer);
    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
        context->device->host_buffers.remove(context->buffer);
    }
    if (context->buffer != nullptr) {
        hrx_buffer_release(context->buffer);
    }
    delete context;
}

static void buffer_memset(ggml_backend_buffer_t buffer,
                          ggml_tensor *         tensor,
                          uint8_t               value,
                          size_t                offset,
                          size_t                size) {
    if (size == 0) {
        return;
    }
    auto *       context            = buffer_context(buffer);
    const size_t destination_offset = tensor_offset(context, tensor) + offset;
    GGML_ASSERT(destination_offset <= buffer->size && size <= buffer->size - destination_offset);
    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
        std::memset(context->base + destination_offset, value, size);
        return;
    }
    FillBufferArgs args{ context->buffer, destination_offset, size, value };
    if (!buffer_submit_and_wait(context->device, submit_fill_buffer, &args)) {
        GGML_LOG_ERROR("%s: HRX buffer fill failed\n", __func__);
    }
}

static void buffer_set(ggml_backend_buffer_t buffer,
                       ggml_tensor *         tensor,
                       const void *          data,
                       size_t                offset,
                       size_t                size) {
    if (size == 0) {
        return;
    }
    auto *       context            = buffer_context(buffer);
    const size_t destination_offset = tensor_offset(context, tensor) + offset;
    GGML_ASSERT(destination_offset <= buffer->size && size <= buffer->size - destination_offset);
    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
        std::memcpy(context->base + destination_offset, data, size);
        return;
    }
    if (!HRX_CHECK(hrx_synchronous_h2d(context->device->device, data, context->buffer, destination_offset, size))) {
        GGML_LOG_ERROR("%s: HRX buffer upload failed\n", __func__);
    }
}

static void buffer_get(ggml_backend_buffer_t buffer,
                       const ggml_tensor *   tensor,
                       void *                data,
                       size_t                offset,
                       size_t                size) {
    if (size == 0) {
        return;
    }
    auto *       context       = buffer_context(buffer);
    const size_t source_offset = tensor_offset(context, tensor) + offset;
    GGML_ASSERT(source_offset <= buffer->size && size <= buffer->size - source_offset);
    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
        std::memcpy(data, context->base + source_offset, size);
        return;
    }
    if (!HRX_CHECK(hrx_synchronous_d2h(context->device->device, context->buffer, source_offset, data, size))) {
        GGML_LOG_ERROR("%s: HRX buffer download failed\n", __func__);
    }
}

static bool buffer_copy(ggml_backend_buffer_t buffer, const ggml_tensor * source, ggml_tensor * destination) {
    ggml_backend_buffer_t source_buffer = source->view_src != nullptr ? source->view_src->buffer : source->buffer;
    if (source_buffer == nullptr || source_buffer->iface.get_base != ggml_backend_hrx_buffer_base) {
        return false;
    }
    auto * source_context      = buffer_context(source_buffer);
    auto * destination_context = buffer_context(buffer);
    if (source_context->device != destination_context->device) {
        return false;
    }
    const size_t source_offset      = tensor_offset(source_context, source);
    const size_t destination_offset = tensor_offset(destination_context, destination);
    const size_t size               = ggml_nbytes(source);
    if (source_offset > source_buffer->size || size > source_buffer->size - source_offset ||
        destination_offset > buffer->size || size > buffer->size - destination_offset) {
        return false;
    }
    CopyBufferArgs args{ source_context->buffer, source_offset, destination_context->buffer, destination_offset, size };
    return buffer_submit_and_wait(destination_context->device, submit_copy_buffer, &args);
}

static void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (buffer->size == 0) {
        return;
    }
    auto * context = buffer_context(buffer);
    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
        std::memset(context->base, value, buffer->size);
        return;
    }
    FillBufferArgs args{ context->buffer, 0, buffer->size, value };
    if (!buffer_submit_and_wait(context->device, submit_fill_buffer, &args)) {
        GGML_LOG_ERROR("%s: HRX buffer clear failed\n", __func__);
    }
}

static const ggml_backend_buffer_i buffer_i = {
    buffer_free, ggml_backend_hrx_buffer_base,
    nullptr,     buffer_memset,
    buffer_set,  buffer_get,
    nullptr,     nullptr,
    buffer_copy, buffer_clear,
    nullptr,
};

static ggml_backend_buffer_t buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto *            type_context        = static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context);
    const bool        host_visible        = type_context->host_visible;
    const bool        direct_host_binding = host_visible && type_context->device->use_direct_host_bindings;
    hrx_memory_type_t memory_type         = HRX_MEMORY_TYPE_DEVICE_LOCAL;
    // Direct command-program bindings require coherent CPU/GPU visibility. Otherwise HRX host buffers are pinned
    // transfer memory: DEVICE_VISIBLE permits handle-based stream copies without implying direct device access.
    if (host_visible) {
        memory_type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
        if (direct_host_binding) {
            memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT;
        }
    }
    hrx_buffer_params_t params = {
        memory_type,
        HRX_MEMORY_ACCESS_ALL,
        host_visible ?
            HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED | HRX_BUFFER_USAGE_MAPPING_PERSISTENT :
            HRX_BUFFER_USAGE_DEFAULT,
        0,
    };
    hrx_buffer_t allocation = nullptr;
    if (size > 0 && !HRX_CHECK(hrx_allocator_allocate_buffer(hrx_device_allocator(type_context->device->device), params,
                                                             size, &allocation))) {
        return nullptr;
    }
    uint8_t * base = reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE);
    if (host_visible && size > 0) {
        void * mapped = nullptr;
        if (!HRX_CHECK(hrx_buffer_map(allocation, HRX_MAP_READ | HRX_MAP_WRITE, 0, size, &mapped))) {
            hrx_buffer_release(allocation);
            return nullptr;
        }
        base = static_cast<uint8_t *>(mapped);
    }
    const uint64_t generation = g_allocation_generation.fetch_add(1);
    auto *         context    = new (std::nothrow) ggml_backend_hrx_buffer_context{
        type_context->device, allocation, base, generation, generation, direct_host_binding,
    };
    if (context == nullptr) {
        if (allocation != nullptr) {
            hrx_buffer_release(allocation);
        }
        return nullptr;
    }
    if (host_visible && allocation != nullptr) {
        type_context->device->host_buffers.add(allocation, base, size);
    }
    return ggml_backend_buffer_init(buft, buffer_i, context, size);
}

static size_t buffer_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX_ALIGNMENT;
}

static size_t buffer_max_size(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context)->device->memory_total;
}

static const ggml_backend_buffer_type_i buffer_type_i = {
    buffer_type_name, buffer_alloc, buffer_alignment, buffer_max_size, nullptr, buffer_type_is_host,
};

static const char * backend_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_hrx_context *>(backend->context)->name.c_str();
}

static void backend_synchronize(ggml_backend_t backend);

static void backend_free(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    backend_synchronize(backend);
    context->prepared_programs.clear();
    context->graph_programs.clear();
    context->kernel_executables.clear();
    context->transient_arena.clear();
    context->host_weights.clear();
    context->host_transfers.clear();
    hrx_stream_release(context->stream);
    delete context;
    delete backend;
}

static bool synchronous_upload_fallback(ggml_backend_hrx_context * backend,
                                        const void *               source,
                                        hrx_buffer_t               destination,
                                        size_t                     destination_offset,
                                        size_t                     size) {
    const uint64_t fallback = backend->device->synchronous_upload_fallbacks.fetch_add(1, std::memory_order_relaxed);
    if (fallback == 0) {
        GGML_LOG_WARN(
            "ggml_hrx: synchronous upload fallback for an unregistered host pointer; use the HRX host "
            "buffer type for asynchronous transfers\n");
    }
    // Compatibility path for arbitrary GGML pointers. Keep the synchronization explicit until a bounded staging ring
    // with transfer retirement is available.
    const ggml::hrx::Status status =
        backend->host_transfers.upload_synchronous(backend->stream, source, destination, destination_offset, size);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status.errors().front().c_str());
        return false;
    }
    return true;
}

static bool synchronous_download_fallback(ggml_backend_hrx_context * backend,
                                          hrx_buffer_t               source,
                                          size_t                     source_offset,
                                          void *                     destination,
                                          size_t                     size) {
    const uint64_t fallback = backend->device->synchronous_download_fallbacks.fetch_add(1, std::memory_order_relaxed);
    if (fallback == 0) {
        GGML_LOG_WARN(
            "ggml_hrx: synchronous download fallback for an unregistered host pointer; use the HRX host "
            "buffer type for asynchronous transfers\n");
    }
    // Compatibility path for arbitrary GGML pointers. Keep the synchronization explicit until a bounded staging ring
    // with transfer retirement is available.
    const ggml::hrx::Status status =
        backend->host_transfers.download_synchronous(backend->stream, source, source_offset, destination, size);
    if (!status.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status.errors().front().c_str());
        return false;
    }
    return true;
}

static void backend_set_tensor_async(ggml_backend_t backend,
                                     ggml_tensor *  tensor,
                                     const void *   data,
                                     size_t         offset,
                                     size_t         size) {
    auto *                            backend_context = static_cast<ggml_backend_hrx_context *>(backend->context);
    ggml_backend_hrx_buffer_context * context         = nullptr;
    size_t                            tensor_base     = 0;
    if (!ggml_backend_hrx_tensor_binding(tensor, &context, &tensor_base) || offset > ggml_nbytes(tensor) ||
        size > ggml_nbytes(tensor) - offset) {
        GGML_LOG_ERROR("%s: invalid HRX tensor upload\n", __func__);
        return;
    }
    ggml::hrx::HostBufferRef source = backend_context->device->host_buffers.find(data, size);
    if (source.valid()) {
        // Registered host buffers can participate directly in the stream command buffer.
        HRX_CHECK(hrx_stream_copy_buffer(backend_context->stream, source.buffer(), source.offset(), context->buffer,
                                         tensor_base + offset, size));
    } else {
        synchronous_upload_fallback(backend_context, data, context->buffer, tensor_base + offset, size);
    }
}

static void backend_get_tensor_async(ggml_backend_t      backend,
                                     const ggml_tensor * tensor,
                                     void *              data,
                                     size_t              offset,
                                     size_t              size) {
    auto *                            backend_context = static_cast<ggml_backend_hrx_context *>(backend->context);
    ggml_backend_hrx_buffer_context * context         = nullptr;
    size_t                            tensor_base     = 0;
    if (!ggml_backend_hrx_tensor_binding(tensor, &context, &tensor_base) || offset > ggml_nbytes(tensor) ||
        size > ggml_nbytes(tensor) - offset) {
        GGML_LOG_ERROR("%s: invalid HRX tensor download\n", __func__);
        return;
    }
    ggml::hrx::HostBufferRef destination = backend_context->device->host_buffers.find(data, size);
    if (destination.valid()) {
        // Registered host buffers can participate directly in the stream command buffer.
        HRX_CHECK(hrx_stream_copy_buffer(backend_context->stream, context->buffer, tensor_base + offset,
                                         destination.buffer(), destination.offset(), size));
    } else {
        synchronous_download_fallback(backend_context, context->buffer, tensor_base + offset, data, size);
    }
}

static bool backend_copy_tensor_async(ggml_backend_t      backend_src,
                                      ggml_backend_t      backend_dst,
                                      const ggml_tensor * source,
                                      ggml_tensor *       destination) {
    GGML_UNUSED(backend_src);
    auto * destination_backend = static_cast<ggml_backend_hrx_context *>(backend_dst->context);
    ggml_backend_hrx_buffer_context * destination_context = nullptr;
    size_t                            destination_offset  = 0;
    if (!ggml_backend_hrx_tensor_binding(destination, &destination_context, &destination_offset)) {
        return false;
    }
    ggml_backend_hrx_buffer_context * source_context = nullptr;
    size_t                            source_offset  = 0;
    const size_t                      size           = ggml_nbytes(source);
    if (ggml_backend_hrx_tensor_binding(source, &source_context, &source_offset)) {
        if (source_context->device != destination_context->device) {
            return false;
        }
        return HRX_CHECK(hrx_stream_copy_buffer(destination_backend->stream, source_context->buffer, source_offset,
                                                destination_context->buffer, destination_offset, size));
    }
    ggml_backend_buffer_t source_buffer = source->view_src != nullptr ? source->view_src->buffer : source->buffer;
    if (source_buffer != nullptr && ggml_backend_buffer_is_host(source_buffer)) {
        return synchronous_upload_fallback(destination_backend, source->data, destination_context->buffer,
                                           destination_offset, size);
    }
    return false;
}

static void backend_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (HRX_CHECK(hrx_stream_synchronize(context->stream))) {
        context->graph_replay_state.mark_stream_synchronized();
    }
}

static const char * status_first_error(const ggml::hrx::Status & status) {
    return status.errors().empty() ? "" : status.errors().front().c_str();
}

static enum ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto *                                context  = static_cast<ggml_backend_hrx_context *>(backend->context);
    const ggml::hrx::GraphExecutor        executor = ggml::hrx::GraphExecutor(*context);
    const ggml::hrx::GraphExecutionResult result   = executor.execute(*graph);
    if (!result.success()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, status_first_error(result.status));
    }
    return result.code;
}

static const ggml_backend_i backend_i = {
    backend_name,
    backend_free,
    backend_set_tensor_async,
    backend_get_tensor_async,
    nullptr,
    nullptr,
    backend_copy_tensor_async,
    backend_synchronize,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    graph_compute,
    nullptr,
    nullptr,
    nullptr,
};

static const char * device_name(ggml_backend_dev_t device) {
    return device_context(device)->name.c_str();
}

static const char * device_description(ggml_backend_dev_t device) {
    return device_context(device)->description.c_str();
}

static void device_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    *free  = device_context(device)->memory_total;
    *total = device_context(device)->memory_total;
}

static enum ggml_backend_dev_type device_type(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void device_props(ggml_backend_dev_t device, ggml_backend_dev_props * props) {
    props->name        = device_name(device);
    props->description = device_description(device);
    device_memory(device, &props->memory_free, &props->memory_total);
    props->type      = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = nullptr;
    props->caps      = { true, true, false, false };
}

static ggml_backend_t device_init(ggml_backend_dev_t device, const char * parameters) {
    GGML_UNUSED(parameters);
    auto *       device_ctx = device_context(device);
    hrx_stream_t stream     = nullptr;
    if (!HRX_CHECK(hrx_stream_create(device_ctx->device, 0, &stream))) {
        return nullptr;
    }
    auto * context = new (std::nothrow) ggml_backend_hrx_context;
    if (context != nullptr) {
        context->device = device_ctx;
        context->stream = stream;
        context->name   = device_ctx->name;
    }
    auto * backend = context != nullptr ? new (std::nothrow)
                                              ggml_backend{ ggml_backend_hrx_guid(), backend_i, device, context } :
                                          nullptr;
    if (backend == nullptr) {
        delete context;
        hrx_stream_release(stream);
    }
    return backend;
}

static ggml_backend_buffer_type_t device_buffer_type(ggml_backend_dev_t device) {
    return &device_context(device)->buft;
}

static ggml_backend_buffer_type_t device_host_buffer_type(ggml_backend_dev_t device) {
    return &device_context(device)->host_buft;
}

static bool eager_capability_declared(enum ggml_op op) {
    switch (op) {
        // The scheduler probes preallocated weight tensors as NONE operations when deciding whether their buffer type is
        // usable by this backend. Fused ops are declared here so graph-claim can validate the full dispatch pattern.
        // TODO: split this into placement capability and exact graph execution capability once graph claiming owns the
        // full decision.
        case GGML_OP_NONE:
        case GGML_OP_ADD:
        case GGML_OP_ARGSORT:
        case GGML_OP_CLAMP:
        case GGML_OP_CONCAT:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DIV:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_GET_ROWS:
        case GGML_OP_GLU:
        case GGML_OP_L2_NORM:
        case GGML_OP_MUL:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_PERMUTE:
        case GGML_OP_RESHAPE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE:
        case GGML_OP_SCALE:
        case GGML_OP_SET_ROWS:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_UNARY:
        case GGML_OP_VIEW:
            return true;
        default:
            return false;
    }
}

static bool zero_output_elision_safe_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_DUP:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_CUMSUM:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_CONCAT:
        case GGML_OP_SILU_BACK:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
        case GGML_OP_SCALE:
        case GGML_OP_CONT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_CLAMP:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_COL2IM_1D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
        case GGML_OP_ADD_REL_POS:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_LIGHTNING_INDEXER:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
        case GGML_OP_UNARY:
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_GLU:
            return true;
        default:
            return false;
    }
}

static bool zero_output_elision_supported(const ggml_tensor * op) {
    return op != nullptr && (ggml_nelements(op) == 0 || ggml_nbytes(op) == 0) && zero_output_elision_safe_op(op->op);
}

struct TensorStorageRange {
    const ggml_tensor * root   = nullptr;
    size_t              offset = 0;
    size_t              size   = 0;
    bool                valid  = false;
};

static TensorStorageRange tensor_storage_range(const ggml_tensor * tensor) {
    TensorStorageRange range;
    if (tensor == nullptr) {
        return range;
    }
    range.size = ggml_nbytes(tensor);
    while (tensor->view_src != nullptr) {
        if (range.offset > std::numeric_limits<size_t>::max() - tensor->view_offs) {
            return {};
        }
        range.offset += tensor->view_offs;
        tensor = tensor->view_src;
    }
    range.root  = tensor;
    range.valid = true;
    return range;
}

static TensorStorageRange tensor_storage_range(const ggml_tensor * tensor, size_t size) {
    TensorStorageRange range = tensor_storage_range(tensor);
    if (range.valid) {
        range.size = size;
    }
    return range;
}

static bool tensor_storage_exactly_overlaps(const TensorStorageRange & lhs, const TensorStorageRange & rhs) {
    return lhs.root == rhs.root && lhs.offset == rhs.offset && lhs.size == rhs.size;
}

static bool tensor_storage_ranges_disjoint(const TensorStorageRange & lhs, const TensorStorageRange & rhs) {
    if (lhs.root != rhs.root) {
        return true;
    }
    if (lhs.offset > std::numeric_limits<size_t>::max() - lhs.size ||
        rhs.offset > std::numeric_limits<size_t>::max() - rhs.size) {
        return false;
    }
    return lhs.offset + lhs.size <= rhs.offset || rhs.offset + rhs.size <= lhs.offset;
}

static bool scale_storage_is_safe(const ggml_tensor * input, const ggml_tensor * output) {
    const TensorStorageRange input_range  = tensor_storage_range(input);
    const TensorStorageRange output_range = tensor_storage_range(output);
    if (!input_range.valid || !output_range.valid) {
        return false;
    }
    return tensor_storage_exactly_overlaps(input_range, output_range) ||
           tensor_storage_ranges_disjoint(input_range, output_range);
}

static bool binary_output_storage_is_safe(const ggml_tensor * lhs, const ggml_tensor * rhs, const ggml_tensor * output) {
    const TensorStorageRange lhs_range    = tensor_storage_range(lhs);
    const TensorStorageRange rhs_range    = tensor_storage_range(rhs);
    const TensorStorageRange output_range = tensor_storage_range(output);
    if (!lhs_range.valid || !rhs_range.valid || !output_range.valid) {
        return false;
    }
    return tensor_storage_ranges_disjoint(lhs_range, output_range) &&
           tensor_storage_ranges_disjoint(rhs_range, output_range);
}

static bool binary_output_storage_is_safe(const ggml_tensor * lhs,
                                          size_t              lhs_byte_count,
                                          const ggml_tensor * rhs,
                                          size_t              rhs_byte_count,
                                          const ggml_tensor * output) {
    const TensorStorageRange lhs_range    = tensor_storage_range(lhs, lhs_byte_count);
    const TensorStorageRange rhs_range    = tensor_storage_range(rhs, rhs_byte_count);
    const TensorStorageRange output_range = tensor_storage_range(output);
    if (!lhs_range.valid || !rhs_range.valid || !output_range.valid) {
        return false;
    }
    return tensor_storage_ranges_disjoint(lhs_range, output_range) &&
           tensor_storage_ranges_disjoint(rhs_range, output_range);
}

static bool tensor_has_positive_shape(const ggml_tensor * tensor) {
    if (tensor == nullptr || ggml_nelements(tensor) <= 0) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return false;
        }
    }
    return true;
}

static bool tensor_has_packed_f32_layout(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    size_t expected_stride = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->nb[i] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<size_t>(tensor->ne[i]);
    }
    return true;
}

static bool tensor_has_strided_f32_layout(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->nb[0] != sizeof(float)) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (tensor->nb[i] % sizeof(float) != 0) {
            return false;
        }
    }
    return true;
}

static bool tensor_storage_span_bytes(const ggml_tensor * tensor, size_t & byte_count) {
    if (!tensor_has_strided_f32_layout(tensor)) {
        return false;
    }
    size_t max_offset = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return false;
        }
        const size_t extent = static_cast<size_t>(tensor->ne[i] - 1);
        if (extent != 0 && tensor->nb[i] > std::numeric_limits<size_t>::max() / extent) {
            return false;
        }
        const size_t dim_offset = extent * tensor->nb[i];
        if (max_offset > std::numeric_limits<size_t>::max() - dim_offset) {
            return false;
        }
        max_offset += dim_offset;
    }
    if (max_offset > std::numeric_limits<size_t>::max() - sizeof(float)) {
        return false;
    }
    byte_count = max_offset + sizeof(float);
    return true;
}

static bool tensor_broadcastable_to(const ggml_tensor * source, const ggml_tensor * output) {
    if (source == nullptr || output == nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (source->ne[i] != output->ne[i] && source->ne[i] != 1) {
            return false;
        }
    }
    return true;
}

static bool binary_kind_allows_broadcast(ggml::hrx::BinaryKind kind,
                                         const ggml_tensor *   lhs,
                                         const ggml_tensor *   rhs,
                                         const ggml_tensor *   output) {
    const bool lhs_full = ggml_are_same_shape(lhs, output);
    const bool rhs_full = ggml_are_same_shape(rhs, output);
    if (!tensor_broadcastable_to(lhs, output) || !tensor_broadcastable_to(rhs, output) || (!lhs_full && !rhs_full)) {
        return false;
    }

    switch (kind) {
        case ggml::hrx::BinaryKind::Add:
        case ggml::hrx::BinaryKind::Mul:
            return true;
        case ggml::hrx::BinaryKind::Sub:
        case ggml::hrx::BinaryKind::Div:
            return lhs_full;
        case ggml::hrx::BinaryKind::SwiGLU:
        case ggml::hrx::BinaryKind::GeGLU:
        case ggml::hrx::BinaryKind::RegLU:
        case ggml::hrx::BinaryKind::GeGLUErf:
        case ggml::hrx::BinaryKind::GeGLUQuick:
            return lhs_full && rhs_full;
    }
    return false;
}

static bool supported_binary_f32_tensor(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    if (!tensor_has_positive_shape(op)) {
        return false;
    }
    if (!ggml_is_contiguous(op) || !tensor_has_packed_f32_layout(op) || op->view_src != nullptr) {
        return false;
    }
    size_t lhs_byte_count = 0;
    size_t rhs_byte_count = 0;
    if (!tensor_storage_span_bytes(op->src[0], lhs_byte_count) ||
        !tensor_storage_span_bytes(op->src[1], rhs_byte_count)) {
        return false;
    }
    if (!binary_output_storage_is_safe(op->src[0], lhs_byte_count, op->src[1], rhs_byte_count, op)) {
        return false;
    }

    ggml::hrx::BinaryKind binary_kind;
    if (!ggml::hrx::import_binary_kind(*op, binary_kind)) {
        return false;
    }

    if (!ggml::hrx::binary_kind_supported(binary_kind)) {
        return false;
    }

    if ((!ggml_are_same_shape(op->src[0], op) || !ggml_are_same_shape(op->src[1], op)) &&
        (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1]) ||
         !tensor_has_packed_f32_layout(op->src[0]) || !tensor_has_packed_f32_layout(op->src[1]))) {
        return false;
    }

    if (!binary_kind_allows_broadcast(binary_kind, op->src[0], op->src[1], op)) {
        return false;
    }

    return true;
}

static bool supported_scale_f32_tensor(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_SCALE && op->src[0] != nullptr && op->type == GGML_TYPE_F32 &&
           op->src[0]->type == GGML_TYPE_F32 && ggml_are_same_shape(op, op->src[0]) && tensor_has_positive_shape(op) &&
           ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && tensor_has_packed_f32_layout(op) &&
           tensor_has_packed_f32_layout(op->src[0]) && scale_storage_is_safe(op->src[0], op) &&
           static_cast<uint64_t>(ggml_nelements(op)) <= std::numeric_limits<uint32_t>::max();
}

static bool supported_qwen_attention_projection_get_rows_tensor(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_GET_ROWS || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->src[0]->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        op->src[1]->type != GGML_TYPE_I32 || !ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) ||
        !ggml_is_contiguous(op->src[1]) || op->ne[0] != op->src[0]->ne[0] || op->ne[1] <= 0 ||
        op->src[1]->ne[0] != op->ne[1] || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    return true;
}

static bool supported_qwen_attention_residual_add_tensor(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ADD || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1]) ||
        !ggml_are_same_shape(op, op->src[0]) || !ggml_are_same_shape(op, op->src[1])) {
        return false;
    }

    return op->src[0]->op == GGML_OP_GET_ROWS || op->src[1]->op == GGML_OP_GET_ROWS ||
           op->src[0]->op == GGML_OP_MUL_MAT || op->src[1]->op == GGML_OP_MUL_MAT;
}

static bool supported_qwen_routed_ffn_reduce_add_tensor(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ADD || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(op) || !ggml_are_same_shape(op, op->src[0]) || !ggml_are_same_shape(op, op->src[1]) ||
        op->ne[0] != 2048 || op->ne[3] != 1 ||
        !((op->ne[1] > 0 && op->ne[2] == 1) || (op->ne[1] == 1 && op->ne[2] > 0))) {
        return false;
    }

    return op->src[0]->op == GGML_OP_VIEW || op->src[1]->op == GGML_OP_VIEW || op->src[0]->op == GGML_OP_ADD ||
           op->src[1]->op == GGML_OP_ADD;
}

static bool supported_unary_f32_tensor(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr || op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(op, op->src[0]) || !ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
        return false;
    }

    ggml::hrx::UnaryKind unary_kind;
    if (!ggml::hrx::import_unary_kind(*op, unary_kind)) {
        return false;
    }

    return ggml::hrx::unary_kind_supported(unary_kind);
}

static bool device_supports_op(ggml_backend_dev_t device, const ggml_tensor * op) {
    GGML_UNUSED(device);
    if (op == nullptr) {
        return false;
    }
    if (zero_output_elision_supported(op)) {
        return true;
    }
    const bool supported_binary = supported_binary_f32_tensor(op);
    if (supported_binary) {
        return true;
    }
    if (supported_scale_f32_tensor(op)) {
        return true;
    }
    if (supported_qwen_attention_projection_get_rows_tensor(op)) {
        return true;
    }
    if (supported_qwen_attention_residual_add_tensor(op)) {
        return true;
    }
    if (supported_qwen_routed_ffn_reduce_add_tensor(op)) {
        return true;
    }
    const bool supported_unary = supported_unary_f32_tensor(op);
    if (supported_unary) {
        return true;
    }
    if (op->op == GGML_OP_GET_ROWS && op->src[0] != nullptr &&
        (op->src[0]->type == GGML_TYPE_IQ3_S || op->src[0]->type == GGML_TYPE_IQ4_NL)) {
        return false;
    }
    return eager_capability_declared(op->op);
}

static bool device_supports_buffer_type(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    auto * context = device_context(device);
    return buft == &context->buft || buft == &context->host_buft || ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i device_i = {
    device_name,
    device_description,
    device_memory,
    device_type,
    device_props,
    device_init,
    device_buffer_type,
    device_host_buffer_type,
    nullptr,
    device_supports_op,
    device_supports_buffer_type,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

static const char * registry_name(ggml_backend_reg_t registry) {
    GGML_UNUSED(registry);
    return "HRX";
}

static size_t registry_device_count(ggml_backend_reg_t registry) {
    return static_cast<ggml_backend_hrx_reg_context *>(registry->context)->devices.size();
}

static ggml_backend_dev_t registry_device(ggml_backend_reg_t registry, size_t index) {
    auto * context = static_cast<ggml_backend_hrx_reg_context *>(registry->context);
    GGML_ASSERT(index < context->devices.size());
    return &context->devices[index];
}

static void * registry_proc(ggml_backend_reg_t registry, const char * name) {
    GGML_UNUSED(registry);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i registry_i = { registry_name, registry_device_count, registry_device, registry_proc };

static std::unique_ptr<ggml_backend_hrx_reg_context> create_registry_context() {
    auto         context = std::make_unique<ggml_backend_hrx_reg_context>();
    if (environment_flag_enabled("GGML_HRX_LOG_DEVICE_EVENTS")) {
        hrx_device_event_sink_t sink = { log_hrx_device_event, nullptr };
        if (!HRX_CHECK(hrx_runtime_set_device_event_sink(sink))) {
            return context;
        }
    }
    hrx_status_t status  = hrx_gpu_initialize(0);
    if (hrx_status_is_ok(status)) {
        context->initialized = true;
    } else if (hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
        hrx_status_ignore(status);
    } else {
        hrx_status_ignore(status);
        return context;
    }
    int count = 0;
    if (!HRX_CHECK(hrx_gpu_device_count(&count))) {
        return context;
    }
    context->device_contexts.reserve(count);
    context->devices.reserve(count);
    for (int i = 0; i < count; ++i) {
        hrx_device_t hrx_device = nullptr;
        if (!HRX_CHECK(hrx_gpu_device_get(i, &hrx_device)) || hrx_device == nullptr) {
            continue;
        }
        hrx_device_retain(hrx_device);
        auto device_ctx                      = std::make_unique<ggml_backend_hrx_device_context>();
        device_ctx->device                   = hrx_device;
        device_ctx->name                     = "HRX" + std::to_string(i);
        device_ctx->use_direct_host_bindings = environment_flag_enabled("GGML_HRX_USE_UNIFIED_MEMORY");
        if (device_ctx->use_direct_host_bindings) {
            GGML_LOG_INFO("ggml_hrx: direct coherent host bindings enabled by GGML_HRX_USE_UNIFIED_MEMORY\n");
        }
        const std::optional<std::string> name =
            device_string_property(hrx_device, HRX_DEVICE_PROPERTY_NAME, "query HRX device name");
        const std::optional<std::string> architecture =
            device_string_property(hrx_device, HRX_DEVICE_PROPERTY_ARCHITECTURE, "query HRX device architecture");
        if (!name || !architecture) {
            hrx_device_release(hrx_device);
            continue;
        }
        uint64_t memory = 0;
        if (!HRX_CHECK(
                hrx_device_get_property(hrx_device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &memory, sizeof(memory)))) {
            hrx_device_release(hrx_device);
            continue;
        }
        if (!HRX_CHECK(hrx_stream_create(hrx_device, 0, &device_ctx->buffer_stream))) {
            hrx_device_release(hrx_device);
            continue;
        }
        device_ctx->memory_total      = static_cast<size_t>(memory);
        device_ctx->description       = *name + " (" + *architecture + ")";
        device_ctx->architecture      = *architecture;
        device_ctx->buft_context      = { device_ctx.get(), device_ctx->name, false };
        device_ctx->buft              = { buffer_type_i, nullptr, &device_ctx->buft_context };
        device_ctx->host_buft_context = { device_ctx.get(), device_ctx->name + "_HOST", true };
        device_ctx->host_buft         = { buffer_type_i, nullptr, &device_ctx->host_buft_context };
        context->device_contexts.emplace_back(std::move(device_ctx));
        context->devices.push_back({ device_i, nullptr, context->device_contexts.back().get() });
        context->device_contexts.back()->buft.device      = &context->devices.back();
        context->device_contexts.back()->host_buft.device = &context->devices.back();
    }
    return context;
}

}  // namespace

ggml_backend_reg_t ggml_backend_hrx_reg() {
    static std::unique_ptr<ggml_backend_hrx_reg_context> context = create_registry_context();
    static ggml_backend_reg registry = { GGML_BACKEND_API_VERSION, registry_i, context.get() };
    for (auto & device : context->devices) {
        device.reg = &registry;
    }
    return &registry;
}

ggml_backend_t ggml_backend_hrx_init(size_t device) {
    ggml_backend_reg_t registry = ggml_backend_hrx_reg();
    if (device >= ggml_backend_reg_dev_count(registry)) {
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, device), nullptr);
}

bool ggml_backend_is_hrx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hrx_guid());
}

bool ggml_backend_hrx_get_cache_stats(ggml_backend_t backend, ggml_backend_hrx_cache_stats * stats) {
    if (!ggml_backend_is_hrx(backend) || stats == nullptr) {
        return false;
    }
    auto *                                  context     = static_cast<ggml_backend_hrx_context *>(backend->context);
    const ggml::hrx::GraphProgramCacheStats graph_stats = context->graph_programs.stats();
    const ggml::hrx::PreparedCommandProgramCacheStats prepared_stats = context->prepared_programs.stats();
    stats->graph_program_builds                                      = graph_stats.builds;
    stats->graph_program_hits                                        = graph_stats.hits;
    stats->prepared_program_builds = graph_stats.prepared_program_builds + prepared_stats.builds;
    stats->prepared_program_hits   = graph_stats.prepared_program_hits + prepared_stats.hits;
    return true;
}

int ggml_backend_hrx_get_device_count() {
    return static_cast<int>(ggml_backend_reg_dev_count(ggml_backend_hrx_reg()));
}

ggml_backend_buffer_type_t ggml_backend_hrx_buffer_type(size_t device) {
    ggml_backend_reg_t registry = ggml_backend_hrx_reg();
    return device < ggml_backend_reg_dev_count(registry) ?
               ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(registry, device)) :
               nullptr;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx_reg)
