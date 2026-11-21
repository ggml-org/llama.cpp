#include "ggml-hrx.h"

#include "ggml-hrx-common.h"
#include "ggml-hrx-ops.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <optional>

static constexpr size_t GGML_HRX_ALIGNMENT = 256;

// Device memory has no host address. Tensor data pointers are offsets from this fake base. The Vulkan backend uses the same method.
static uint8_t * const GGML_HRX_BASE = (uint8_t *) (uintptr_t) 0x1000;

bool ggml_hrx_check(hrx_status_t status, const char * expr, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }
    char * message = nullptr;
    size_t length  = 0;
    hrx_status_t format_status = hrx_status_to_string(status, &message, &length);
    if (!hrx_status_is_ok(format_status)) {
        hrx_status_ignore(format_status);
    }
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expr, message != nullptr ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

struct ggml_hrx_buffer_context {
    ggml_hrx_device_context * dev    = nullptr;
    hrx_buffer_t              buffer = nullptr;
};

static void * ggml_backend_hrx_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return GGML_HRX_BASE;
}

static size_t ggml_hrx_tensor_offset(const ggml_tensor * t) {
    return (size_t) ((const uint8_t *) t->data - GGML_HRX_BASE);
}

static ggml_backend_buffer_t ggml_hrx_tensor_buffer(const ggml_tensor * t) {
    return t->view_src != nullptr ? t->view_src->buffer : t->buffer;
}

static bool ggml_hrx_tensor_on_device(const ggml_tensor * t) {
    ggml_backend_buffer_t buffer = ggml_hrx_tensor_buffer(t);
    return buffer != nullptr && buffer->iface.get_base == ggml_backend_hrx_buffer_get_base;
}

// buffer_stream records uploads up to this size. They run before the next graph or read.
static constexpr size_t GGML_HRX_MAX_RECORDED_UPLOAD = 1024 * 1024;
// This is the HAL limit for one update_buffer command.
static constexpr size_t GGML_HRX_UPDATE_CHUNK = 64 * 1024;

// Hold buffer_mutex when you call this function.
static bool ggml_hrx_sync_buffer_stream(ggml_hrx_device_context * dev) {
    dev->upload_pending = false;
    return GGML_HRX_CHECK(hrx_stream_synchronize(dev->buffer_stream));
}

static bool ggml_hrx_finish_uploads(ggml_hrx_device_context * dev) {
    std::lock_guard<std::mutex> lock(dev->buffer_mutex);
    return !dev->upload_pending || ggml_hrx_sync_buffer_stream(dev);
}

static void ggml_backend_hrx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_hrx_buffer_context *) buffer->context;
    if (ctx->buffer != nullptr) {
        ggml_hrx_finish_uploads(ctx->dev);
        hrx_buffer_release(ctx->buffer);
    }
    delete ctx;
}

struct ggml_hrx_fill_args {
    hrx_buffer_t buffer;
    size_t       offset;
    size_t       size;
    uint8_t      value;
};

static bool ggml_hrx_fill(ggml_hrx_device_context * dev, const ggml_hrx_fill_args & args) {
    std::lock_guard<std::mutex> lock(dev->buffer_mutex);
    if (!GGML_HRX_CHECK(hrx_stream_fill_buffer(dev->buffer_stream, args.buffer, args.offset, args.size, &args.value, sizeof(args.value)))) {
        return false;
    }
    return ggml_hrx_sync_buffer_stream(dev);
}

static void ggml_backend_hrx_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                  size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    auto *       ctx = (ggml_hrx_buffer_context *) buffer->context;
    const size_t dst = ggml_hrx_tensor_offset(tensor) + offset;
    GGML_ASSERT(dst <= buffer->size && size <= buffer->size - dst);
    if (!ggml_hrx_fill(ctx->dev, { ctx->buffer, dst, size, value })) {
        GGML_ABORT("%s: HRX buffer fill failed", __func__);
    }
}

static void ggml_backend_hrx_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                               size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    auto *       ctx = (ggml_hrx_buffer_context *) buffer->context;
    const size_t dst = ggml_hrx_tensor_offset(tensor) + offset;
    GGML_ASSERT(dst <= buffer->size && size <= buffer->size - dst);
    ggml_hrx_device_context * dev = ctx->dev;
    if (size <= GGML_HRX_MAX_RECORDED_UPLOAD) {
        std::lock_guard<std::mutex> lock(dev->buffer_mutex);
        // The stream copies the data now. The caller can reuse it.
        for (size_t i = 0; i < size; i += GGML_HRX_UPDATE_CHUNK) {
            const size_t n = std::min(GGML_HRX_UPDATE_CHUNK, size - i);
            if (!GGML_HRX_CHECK(hrx_stream_update_buffer(dev->buffer_stream, (const uint8_t *) data + i, n, ctx->buffer, dst + i))) {
                GGML_ABORT("%s: HRX upload failed", __func__);
            }
        }
        dev->upload_pending = true;
        return;
    }
    if (!ggml_hrx_finish_uploads(dev) || !GGML_HRX_CHECK(hrx_synchronous_h2d(dev->transfer_queue, data, ctx->buffer, dst, size))) {
        GGML_ABORT("%s: HRX upload failed", __func__);
    }
}

static void ggml_backend_hrx_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                               size_t offset, size_t size) {
    if (size == 0) {
        return;
    }
    auto *       ctx = (ggml_hrx_buffer_context *) buffer->context;
    const size_t src = ggml_hrx_tensor_offset(tensor) + offset;
    GGML_ASSERT(src <= buffer->size && size <= buffer->size - src);
    if (!ggml_hrx_finish_uploads(ctx->dev) ||
        !GGML_HRX_CHECK(hrx_synchronous_d2h(ctx->dev->transfer_queue, ctx->buffer, src, data, size))) {
        GGML_ABORT("%s: HRX download failed", __func__);
    }
}

static bool ggml_backend_hrx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_hrx_tensor_on_device(src)) {
        return false;
    }
    auto * src_ctx = (ggml_hrx_buffer_context *) ggml_hrx_tensor_buffer(src)->context;
    auto * dst_ctx = (ggml_hrx_buffer_context *) buffer->context;
    if (src_ctx->dev != dst_ctx->dev) {
        return false;
    }
    const size_t size = ggml_nbytes(src);
    if (size == 0) {
        return true;
    }
    ggml_hrx_device_context * dev = dst_ctx->dev;
    std::lock_guard<std::mutex> lock(dev->buffer_mutex);
    if (!GGML_HRX_CHECK(hrx_stream_copy_buffer(dev->buffer_stream, src_ctx->buffer, ggml_hrx_tensor_offset(src),
                                               dst_ctx->buffer, ggml_hrx_tensor_offset(dst), size))) {
        return false;
    }
    return ggml_hrx_sync_buffer_stream(dev);
}

static void ggml_backend_hrx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_hrx_buffer_context *) buffer->context;
    if (buffer->size == 0 || ctx->buffer == nullptr) {
        return;
    }
    if (!ggml_hrx_fill(ctx->dev, { ctx->buffer, 0, buffer->size, value })) {
        GGML_ABORT("%s: HRX buffer clear failed", __func__);
    }
}

static const ggml_backend_buffer_i ggml_backend_hrx_buffer_interface = {
    /* .free_buffer   = */ ggml_backend_hrx_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_hrx_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_hrx_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_hrx_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_hrx_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_hrx_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_hrx_buffer_clear,
    /* .reset         = */ nullptr,
    /* .set_tensor_n  = */ nullptr,
    /* .get_tensor_n  = */ nullptr,
    /* .cpy_tensor_n  = */ nullptr,
};

static const char * ggml_backend_hrx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ((ggml_hrx_device_context *) buft->context)->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_hrx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * dev = (ggml_hrx_device_context *) buft->context;

    hrx_buffer_t allocation = nullptr;
    if (size > 0) {
        const hrx_buffer_params_t params = {
            HRX_MEMORY_TYPE_DEVICE_LOCAL,
            HRX_MEMORY_ACCESS_ALL,
            HRX_BUFFER_USAGE_DEFAULT,
        };
        if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(hrx_device_allocator(dev->device), params, size, &allocation))) {
            GGML_LOG_ERROR("%s: failed to allocate %.2f MiB on %s\n", __func__, size / 1024.0 / 1024.0, dev->name.c_str());
            return nullptr;
        }
    }

    auto * ctx   = new ggml_hrx_buffer_context;
    ctx->dev     = dev;
    ctx->buffer  = allocation;
    return ggml_backend_buffer_init(buft, ggml_backend_hrx_buffer_interface, ctx, size);
}

static size_t ggml_backend_hrx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX_ALIGNMENT;
}

static size_t ggml_backend_hrx_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return ((ggml_hrx_device_context *) buft->context)->memory_total;
}

static const ggml_backend_buffer_type_i ggml_backend_hrx_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_hrx_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_hrx_buffer_type_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ ggml_backend_hrx_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_hrx_buffer_type_get_max_size,
    /* .get_alloc_size   = */ nullptr,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ nullptr,
};

ggml_hrx_kernel::~ggml_hrx_kernel() {
    ggml_hrx_loom_launch_program_free(launch);
    if (executable != nullptr) {
        hrx_executable_release(executable);
    }
}

static const ggml_hrx_kernel_source * ggml_hrx_find_source(const char * path) {
    for (size_t i = 0; i < ggml_hrx_kernel_source_count; ++i) {
        if (std::strcmp(ggml_hrx_kernel_sources[i].path, path) == 0) {
            return &ggml_hrx_kernel_sources[i];
        }
    }
    GGML_LOG_ERROR("%s: kernel source %s is not embedded\n", __func__, path);
    return nullptr;
}

static std::unique_ptr<ggml_hrx_kernel> ggml_hrx_build_kernel(ggml_hrx_device_context * dev, const ggml_hrx_kernel_desc & desc) {
    if (dev->jit == nullptr && !GGML_HRX_CHECK(ggml_hrx_loom_jit_create(dev->arch->name, dev->arch->loom_target, &dev->jit))) {
        return nullptr;
    }

    const ggml_hrx_kernel_source * source = ggml_hrx_find_source(desc.file);
    if (source == nullptr) {
        return nullptr;
    }
    std::vector<ggml_hrx_loom_source> deps;
    for (const char * lib : desc.libs) {
        const ggml_hrx_kernel_source * dep = ggml_hrx_find_source(lib);
        if (dep == nullptr) {
            return nullptr;
        }
        deps.push_back({ dep->data, dep->size, dep->path });
    }
    std::vector<ggml_hrx_loom_config_binding> config;
    for (const auto & kv : desc.config) {
        config.push_back({ kv.first.c_str(), kv.second.c_str() });
    }

    ggml_hrx_loom_compile_options options;
    options.source   = { source->data, source->size, source->path };
    options.deps     = deps.data();
    options.n_deps   = deps.size();
    options.config   = config.data();
    options.n_config = config.size();
    options.symbol   = desc.symbol;

    ggml_hrx_loom_compile_result compiled;
    if (!GGML_HRX_CHECK(ggml_hrx_loom_compile(dev->jit, &options, &compiled))) {
        GGML_LOG_ERROR("%s: failed to compile %s from %s\n", __func__, desc.symbol, desc.file);
        return nullptr;
    }

    auto kernel    = std::make_unique<ggml_hrx_kernel>();
    kernel->launch = compiled.launch;
    if (!GGML_HRX_CHECK(hrx_executable_load_data(dev->device, compiled.hsaco.data(), compiled.hsaco.size(), "amdgpu",
                                                 dev->arch->name, &kernel->executable)) ||
        !GGML_HRX_CHECK(hrx_executable_lookup_export_by_name(kernel->executable, desc.symbol, &kernel->export_ordinal)) ||
        !GGML_HRX_CHECK(hrx_executable_export_info(kernel->executable, kernel->export_ordinal, &kernel->export_info))) {
        return nullptr;
    }

    const hrx_executable_export_info_t & info = kernel->export_info;
    if (info.parameter_count < info.binding_count) {
        GGML_LOG_ERROR("%s: %s has an invalid parameter count\n", __func__, desc.symbol);
        return nullptr;
    }
    kernel->n_scalars = info.parameter_count - info.binding_count;
    if (kernel->n_scalars == 0) {
        if (info.constant_byte_length != 0) {
            GGML_LOG_ERROR("%s: %s has constants but no scalar parameters\n", __func__, desc.symbol);
            return nullptr;
        }
    } else {
        kernel->scalar_size = info.constant_byte_length / kernel->n_scalars;
        if (info.constant_byte_length % kernel->n_scalars != 0 || (kernel->scalar_size != 4 && kernel->scalar_size != 8)) {
            GGML_LOG_ERROR("%s: %s has an unsupported scalar ABI\n", __func__, desc.symbol);
            return nullptr;
        }
    }
    return kernel;
}

ggml_hrx_kernel * ggml_hrx_get_kernel(ggml_hrx_device_context * dev, const ggml_hrx_kernel_desc & desc) {
    std::string key = std::string(desc.file) + "|" + desc.symbol;
    for (const auto & kv : desc.config) {
        key += "|" + kv.first + "=" + kv.second;
    }

    std::lock_guard<std::mutex> lock(dev->kernel_mutex);
    auto it = dev->kernels.find(key);
    if (it != dev->kernels.end()) {
        return it->second.get();
    }
    // Cache a failed build as nullptr. This prevents a retry for each node.
    auto & slot = dev->kernels[key];
    slot = ggml_hrx_build_kernel(dev, desc);
    return slot.get();
}

bool ggml_hrx_dispatch(ggml_backend_hrx_context * ctx, ggml_hrx_kernel * kernel, const std::vector<int64_t> & workload,
                       const std::vector<int64_t> & scalars, const std::vector<const ggml_tensor *> & buffers) {
    if (kernel == nullptr) {
        return false;
    }
    if (scalars.size() != kernel->n_scalars || buffers.size() != kernel->export_info.binding_count) {
        GGML_LOG_ERROR("%s: argument count mismatch: %zu scalars, %zu buffers\n", __func__, scalars.size(), buffers.size());
        return false;
    }

    ggml_hrx_loom_launch_config launch;
    if (!GGML_HRX_CHECK(ggml_hrx_loom_launch_program_eval(kernel->launch, workload.data(), workload.size(), &launch))) {
        return false;
    }
    const uint32_t max_yz = ctx->device->arch->max_workgroups_yz;
    if (launch.workgroup_count[1] > max_yz || launch.workgroup_count[2] > max_yz) {
        GGML_LOG_ERROR("%s: workgroup count y or z is above %u\n", __func__, max_yz);
        return false;
    }

    std::vector<uint8_t> constants(kernel->n_scalars * kernel->scalar_size);
    for (size_t i = 0; i < scalars.size(); ++i) {
        const int64_t value = scalars[i];
        if (value < 0 || (kernel->scalar_size == 4 && (uint64_t) value > std::numeric_limits<uint32_t>::max())) {
            GGML_LOG_ERROR("%s: launch scalar %zu out of range: %" PRId64 "\n", __func__, i, value);
            return false;
        }
        if (kernel->scalar_size == 4) {
            const uint32_t v = (uint32_t) value;
            std::memcpy(constants.data() + i * 4, &v, 4);
        } else {
            const uint64_t v = (uint64_t) value;
            std::memcpy(constants.data() + i * 8, &v, 8);
        }
    }

    std::vector<hrx_buffer_ref_t> refs;
    refs.reserve(buffers.size());
    for (const ggml_tensor * t : buffers) {
        if (t == nullptr || !ggml_hrx_tensor_on_device(t)) {
            GGML_LOG_ERROR("%s: tensor %s is not in an HRX buffer\n", __func__, t != nullptr ? t->name : "(null)");
            return false;
        }
        auto * bctx = (ggml_hrx_buffer_context *) ggml_hrx_tensor_buffer(t)->context;
        refs.push_back({ bctx->buffer, ggml_hrx_tensor_offset(t), ggml_nbytes(t) });
    }

    const hrx_dispatch_config_t config = {
        { launch.workgroup_count[0], launch.workgroup_count[1], launch.workgroup_count[2] },
        { launch.workgroup_size[0],  launch.workgroup_size[1],  launch.workgroup_size[2] },
        launch.subgroup_size,
    };
    return GGML_HRX_CHECK(hrx_stream_dispatch(ctx->stream, kernel->executable, kernel->export_ordinal, &config,
                                              constants.data(), constants.size(), refs.data(), refs.size(), 0));
}

static ggml_guid_t ggml_backend_hrx_guid() {
    static ggml_guid guid = { 0xd2, 0x3d, 0x72, 0x83, 0xb2, 0x82, 0x4d, 0xe0, 0x8a, 0x3e, 0x21, 0x1d, 0x68, 0x87, 0x2f, 0x4b };
    return &guid;
}

static const char * ggml_backend_hrx_get_name(ggml_backend_t backend) {
    return ((ggml_backend_hrx_context *) backend->context)->name.c_str();
}

static void ggml_hrx_readback_release(ggml_backend_hrx_context * ctx) {
    if (ctx->readback != nullptr) {
        GGML_HRX_CHECK(hrx_buffer_unmap(ctx->readback));
        hrx_buffer_release(ctx->readback);
    }
    ctx->readback      = nullptr;
    ctx->readback_host = nullptr;
    ctx->readback_size = 0;
}

// Returns false if the readback buffer cannot take size more bytes now.
static bool ggml_hrx_readback_reserve(ggml_backend_hrx_context * ctx, size_t size) {
    if (ctx->readback_used + size <= ctx->readback_size) {
        return true;
    }
    // Pending copies use the current buffer.
    if (ctx->readback_failed || !ctx->readback_copies.empty()) {
        return false;
    }
    size = std::max(size, 2 * ctx->readback_size);
    ggml_hrx_readback_release(ctx);
    const hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_HOST_COHERENT,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
    };
    void * host = nullptr;
    if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(hrx_device_allocator(ctx->device->device), params, size, &ctx->readback)) ||
        !GGML_HRX_CHECK(hrx_buffer_map(ctx->readback, HRX_MAP_READ, 0, size, &host))) {
        if (ctx->readback != nullptr) {
            hrx_buffer_release(ctx->readback);
            ctx->readback = nullptr;
        }
        ctx->readback_failed = true;
        return false;
    }
    ctx->readback_host = (const uint8_t *) host;
    ctx->readback_size = size;
    return true;
}

static bool ggml_hrx_stream_finish(ggml_backend_hrx_context * ctx) {
    if (!GGML_HRX_CHECK(hrx_stream_synchronize(ctx->stream))) {
        return false;
    }
    for (const auto & c : ctx->readback_copies) {
        std::memcpy(c.dst, ctx->readback_host + c.offset, c.size);
    }
    ctx->readback_copies.clear();
    ctx->readback_used = 0;
    return true;
}

static void ggml_backend_hrx_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_hrx_context *) backend->context;
    GGML_HRX_CHECK(hrx_stream_synchronize(ctx->stream));
    hrx_stream_release(ctx->stream);
    ggml_hrx_readback_release(ctx);
    delete ctx;
    delete backend;
}

static void ggml_backend_hrx_synchronize(ggml_backend_t backend) {
    ggml_hrx_stream_finish((ggml_backend_hrx_context *) backend->context);
}

// Make the backend stream wait for the uploads recorded in buffer_stream.
static bool ggml_hrx_wait_uploads(ggml_backend_hrx_context * ctx) {
    ggml_hrx_device_context * dev = ctx->device;
    hrx_timeline_point_t      point;
    {
        std::lock_guard<std::mutex> lock(dev->buffer_mutex);
        if (dev->upload_pending && !GGML_HRX_CHECK(hrx_stream_flush(dev->buffer_stream))) {
            return false;
        }
        if (!GGML_HRX_CHECK(hrx_stream_get_timeline_position(dev->buffer_stream, &point))) {
            return false;
        }
    }
    if (point.value <= ctx->upload_wait) {
        return true;
    }
    ctx->upload_wait = point.value;
    return GGML_HRX_CHECK(hrx_stream_wait_on(ctx->stream, point));
}

static void ggml_backend_hrx_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
                                              size_t offset, size_t size) {
    auto * ctx = (ggml_backend_hrx_context *) backend->context;
    if (size == 0) {
        return;
    }
    GGML_ASSERT(ggml_hrx_tensor_on_device(tensor));
    auto *       bctx = (ggml_hrx_buffer_context *) ggml_hrx_tensor_buffer(tensor)->context;
    const size_t dst  = ggml_hrx_tensor_offset(tensor) + offset;
    const bool   ok   = ggml_hrx_wait_uploads(ctx) &&
                    (size <= GGML_HRX_UPDATE_CHUNK ?
                         GGML_HRX_CHECK(hrx_stream_update_buffer(ctx->stream, data, size, bctx->buffer, dst)) :
                         GGML_HRX_CHECK(hrx_stream_copy_h2d(ctx->stream, data, bctx->buffer, dst, size)));
    if (!ok) {
        GGML_ABORT("%s: HRX upload failed", __func__);
    }
}

static void ggml_backend_hrx_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
                                              size_t offset, size_t size) {
    auto * ctx = (ggml_backend_hrx_context *) backend->context;
    if (size == 0) {
        return;
    }
    GGML_ASSERT(ggml_hrx_tensor_on_device(tensor));
    auto *       bctx = (ggml_hrx_buffer_context *) ggml_hrx_tensor_buffer(tensor)->context;
    const size_t src  = ggml_hrx_tensor_offset(tensor) + offset;
    if (!ggml_hrx_wait_uploads(ctx)) {
        GGML_ABORT("%s: HRX download failed", __func__);
    }
    if (ctx->device->tune->mapped_readback && ggml_hrx_readback_reserve(ctx, size)) {
        const size_t dst = ctx->readback_used;
        if (!GGML_HRX_CHECK(hrx_stream_copy_buffer(ctx->stream, bctx->buffer, src, ctx->readback, dst, size))) {
            GGML_ABORT("%s: HRX download failed", __func__);
        }
        ctx->readback_copies.push_back({ data, dst, size });
        ctx->readback_used = dst + GGML_PAD(size, GGML_HRX_ALIGNMENT);
        return;
    }
    if (!GGML_HRX_CHECK(hrx_stream_copy_d2h(ctx->stream, bctx->buffer, src, data, size))) {
        GGML_ABORT("%s: HRX download failed", __func__);
    }
}

static void ggml_backend_hrx_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ctx = (ggml_backend_hrx_context *) backend->context;
    // The host copies of earlier get_tensor_async calls must be done when the event completes.
    if (!ctx->readback_copies.empty()) {
        ggml_hrx_stream_finish(ctx);
    }
    GGML_HRX_CHECK(hrx_event_record((hrx_event_t) event->context, ctx->stream));
}

static void ggml_backend_hrx_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ctx = (ggml_backend_hrx_context *) backend->context;
    GGML_HRX_CHECK(hrx_stream_wait_event(ctx->stream, (hrx_event_t) event->context));
}

static bool ggml_hrx_compute_node(ggml_backend_hrx_context * ctx, const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_SCALE:    return ggml_hrx_op_scale(ctx, node);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:     return ggml_hrx_op_cpy(ctx, node);
        case GGML_OP_SET_ROWS: return ggml_hrx_op_set_rows(ctx, node);
        case GGML_OP_MUL_MAT:  return ggml_hrx_op_mul_mat(ctx, node);
        default:               return false;
    }
}

static enum ggml_status ggml_backend_hrx_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto *                ctx       = (ggml_backend_hrx_context *) backend->context;
    const ggml_hrx_tune * tune      = ctx->device->tune;
    int                   n_pending = 0;
    // The GPU starts work while the host records the next nodes. A short first interval starts a small graph sooner.
    int                   n_submit  = tune->nodes_first_submit;

    if (!ggml_hrx_wait_uploads(ctx)) {
        return GGML_STATUS_FAILED;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || ggml_op_is_empty(node->op) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (!ggml_hrx_compute_node(ctx, node)) {
            GGML_LOG_ERROR("%s: failed to run node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }
        if (++n_pending >= n_submit) {
            n_pending = 0;
            n_submit  = std::min(2 * n_submit, tune->nodes_per_submit);
            if (!GGML_HRX_CHECK(hrx_stream_flush(ctx->stream))) {
                return GGML_STATUS_FAILED;
            }
        }
    }

    // synchronize waits for the graph.
    if (!GGML_HRX_CHECK(hrx_stream_flush(ctx->stream))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_hrx_interface = {
    /* .get_name            = */ ggml_backend_hrx_get_name,
    /* .free                = */ ggml_backend_hrx_free,
    /* .set_tensor_async    = */ ggml_backend_hrx_set_tensor_async,
    /* .get_tensor_async    = */ ggml_backend_hrx_get_tensor_async,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ ggml_backend_hrx_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ ggml_backend_hrx_graph_compute,
    /* .event_record        = */ ggml_backend_hrx_event_record,
    /* .event_wait          = */ ggml_backend_hrx_event_wait,
    /* .graph_optimize      = */ nullptr,
};

static ggml_hrx_device_context * ggml_hrx_dev_ctx(ggml_backend_dev_t dev) {
    return (ggml_hrx_device_context *) dev->context;
}

static const char * ggml_backend_hrx_device_get_name(ggml_backend_dev_t dev) {
    return ggml_hrx_dev_ctx(dev)->name.c_str();
}

static const char * ggml_backend_hrx_device_get_description(ggml_backend_dev_t dev) {
    return ggml_hrx_dev_ctx(dev)->description.c_str();
}

static void ggml_backend_hrx_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // HRX does not report free memory yet.
    *free  = ggml_hrx_dev_ctx(dev)->memory_total;
    *total = ggml_hrx_dev_ctx(dev)->memory_total;
}

static enum ggml_backend_dev_type ggml_backend_hrx_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_hrx_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hrx_device_get_name(dev);
    props->description = ggml_backend_hrx_device_get_description(dev);
    props->type        = ggml_backend_hrx_device_get_type(dev);
    props->device_id   = nullptr;
    ggml_backend_hrx_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ true,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_t ggml_backend_hrx_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_hrx_device_context * dev_ctx = ggml_hrx_dev_ctx(dev);

    hrx_stream_t stream = nullptr;
    if (!GGML_HRX_CHECK(hrx_stream_create(dev_ctx->stream_queue, 0, &stream))) {
        return nullptr;
    }
    auto * ctx   = new ggml_backend_hrx_context;
    ctx->device  = dev_ctx;
    ctx->stream  = stream;
    ctx->name    = dev_ctx->name;
    return new ggml_backend{ ggml_backend_hrx_guid(), ggml_backend_hrx_interface, dev, ctx };
}

static ggml_backend_buffer_type_t ggml_backend_hrx_device_get_buffer_type(ggml_backend_dev_t dev) {
    return &ggml_hrx_dev_ctx(dev)->buft;
}

static bool ggml_backend_hrx_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const ggml_hrx_device_context * ctx = ggml_hrx_dev_ctx(dev);
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE: return true;
        case GGML_OP_SCALE:     return ggml_hrx_supports_scale(ctx, op);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:      return ggml_hrx_supports_cpy(ctx, op);
        case GGML_OP_SET_ROWS:  return ggml_hrx_supports_set_rows(ctx, op);
        case GGML_OP_MUL_MAT:   return ggml_hrx_supports_mul_mat(ctx, op);
        default:                return false;
    }
}

static bool ggml_backend_hrx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == &ggml_hrx_dev_ctx(dev)->buft;
}

static ggml_backend_event_t ggml_backend_hrx_device_event_new(ggml_backend_dev_t dev) {
    hrx_event_t event = nullptr;
    if (!GGML_HRX_CHECK(hrx_event_create(ggml_hrx_dev_ctx(dev)->device, HRX_EVENT_FLAG_DISABLE_TIMING, &event))) {
        return nullptr;
    }
    return new ggml_backend_event{ dev, event };
}

static void ggml_backend_hrx_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    hrx_event_release((hrx_event_t) event->context);
    delete event;
}

static void ggml_backend_hrx_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    GGML_HRX_CHECK(hrx_event_synchronize((hrx_event_t) event->context));
}

static const ggml_backend_device_i ggml_backend_hrx_device_interface = {
    /* .get_name             = */ ggml_backend_hrx_device_get_name,
    /* .get_description      = */ ggml_backend_hrx_device_get_description,
    /* .get_memory           = */ ggml_backend_hrx_device_get_memory,
    /* .get_type             = */ ggml_backend_hrx_device_get_type,
    /* .get_props            = */ ggml_backend_hrx_device_get_props,
    /* .init_backend         = */ ggml_backend_hrx_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_hrx_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_hrx_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hrx_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ ggml_backend_hrx_device_event_new,
    /* .event_free           = */ ggml_backend_hrx_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hrx_device_event_synchronize,
};

struct ggml_hrx_reg_context {
    hrx_runtime_t                   runtime = nullptr;
    std::vector<hrx_device_group_t> groups;

    std::vector<std::unique_ptr<ggml_hrx_device_context>> contexts;
    std::vector<ggml_backend_device>                      devices;

    ~ggml_hrx_reg_context() {
        for (auto & dev : contexts) {
            dev->kernels.clear();
            ggml_hrx_loom_jit_free(dev->jit);
            if (dev->buffer_stream != nullptr) {
                hrx_stream_release(dev->buffer_stream);
            }
            hrx_device_release(dev->device);
        }
        for (hrx_device_group_t group : groups) {
            hrx_device_group_release(group);
        }
        if (runtime != nullptr) {
            hrx_runtime_release(runtime);
        }
    }
};

static std::optional<std::string> ggml_hrx_device_string(hrx_device_t device, hrx_device_property_t property) {
    std::vector<char> buffer(64);
    while (buffer.size() <= 4096) {
        hrx_status_t status = hrx_device_get_property(device, property, buffer.data(), buffer.size());
        if (hrx_status_is_ok(status)) {
            return std::string(buffer.data());
        }
        if (hrx_status_code(status) != HRX_STATUS_OUT_OF_RANGE) {
            GGML_HRX_CHECK(status);
            return std::nullopt;
        }
        hrx_status_ignore(status);
        buffer.resize(buffer.size() * 2);
    }
    return std::nullopt;
}

static bool ggml_hrx_find_queue(hrx_device_t device, hrx_queue_role_flags_t roles, uint32_t index, hrx_queue_t * queue) {
    *queue = nullptr;
    const size_t n_families = hrx_device_queue_family_count(device);
    for (size_t i = 0; i < n_families; ++i) {
        hrx_queue_family_info_t family = {};
        if (!GGML_HRX_CHECK(hrx_device_query_queue_family(device, (uint32_t) i, &family))) {
            return false;
        }
        if ((family.roles & roles) == roles && family.provisioned_queue_count > index) {
            *queue = hrx_device_get_queue(device, (uint32_t) i, index);
            if (*queue != nullptr) {
                return true;
            }
        }
    }
    return false;
}

// Returns nullptr if the device is not usable. In that case, it releases all that it took.
static std::unique_ptr<ggml_hrx_device_context> ggml_hrx_init_device(hrx_device_t device, int index) {
    const std::optional<std::string> name = ggml_hrx_device_string(device, HRX_DEVICE_PROPERTY_NAME);
    const std::optional<std::string> arch = ggml_hrx_device_string(device, HRX_DEVICE_PROPERTY_ARCHITECTURE);
    if (!name || !arch) {
        return nullptr;
    }
    const ggml_hrx_arch * info = ggml_hrx_arch_find(arch->c_str());
    if (info == nullptr) {
        GGML_LOG_INFO("ggml_hrx: skipping %s (%s): architecture is not supported\n", name->c_str(), arch->c_str());
        return nullptr;
    }

    auto ctx    = std::make_unique<ggml_hrx_device_context>();
    ctx->device = device;
    ctx->arch   = info;
    ctx->tune   = info->tune;
    ctx->name   = GGML_HRX_NAME + std::to_string(index);
    ctx->index  = index;
    ctx->description = *name + " (" + *arch + ")";

    uint64_t memory = 0;
    if (!GGML_HRX_CHECK(hrx_device_get_property(device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY, &memory, sizeof(memory))) ||
        !GGML_HRX_CHECK(hrx_device_get_property(device, HRX_DEVICE_PROPERTY_COMPUTE_UNITS, &ctx->n_cu, sizeof(ctx->n_cu)))) {
        return nullptr;
    }
    ctx->memory_total = (size_t) memory;

    // Synchronous transfers use a second queue if there is one. Then they do not wait for queued graphs.
    if ((!ggml_hrx_find_queue(device, HRX_QUEUE_ROLE_TRANSFER, 1, &ctx->transfer_queue) &&
         !ggml_hrx_find_queue(device, HRX_QUEUE_ROLE_TRANSFER, 0, &ctx->transfer_queue)) ||
        !ggml_hrx_find_queue(device, HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER, 0, &ctx->stream_queue)) {
        GGML_LOG_ERROR("ggml_hrx: %s has no usable queues\n", name->c_str());
        return nullptr;
    }
    if (!GGML_HRX_CHECK(hrx_stream_create(ctx->stream_queue, 0, &ctx->buffer_stream))) {
        return nullptr;
    }
    return ctx;
}

static void ggml_hrx_discover(ggml_hrx_reg_context * reg) {
    hrx_runtime_options_t      options   = {};
    const hrx_host_allocator_t allocator = hrx_host_allocator_system();
    if (!GGML_HRX_CHECK(hrx_runtime_create(&options, allocator, &reg->runtime))) {
        return;
    }

    size_t              count = 0;
    hrx_device_info_t * infos = nullptr;
    if (!GGML_HRX_CHECK(hrx_runtime_query_available_devices(reg->runtime, "amdgpu", allocator, &count, &infos))) {
        return;
    }
    // One group per device.
    for (size_t i = 0; i < count; ++i) {
        const hrx_device_create_params_t params = { infos[i].uri, HRX_WAIT_FLAG_NONE };
        hrx_device_group_t group = nullptr;
        if (!GGML_HRX_CHECK(hrx_device_group_create(reg->runtime, 1, &params, &group))) {
            continue;
        }
        hrx_device_t device = hrx_device_group_get_device(group, 0);
        if (device == nullptr) {
            hrx_device_group_release(group);
            continue;
        }
        hrx_device_retain(device);
        auto ctx = ggml_hrx_init_device(device, (int) reg->contexts.size());
        if (!ctx) {
            hrx_device_release(device);
            hrx_device_group_release(group);
            continue;
        }
        reg->groups.push_back(group);
        reg->contexts.push_back(std::move(ctx));
    }
    hrx_host_allocator_free(allocator, infos);
}

static const char * ggml_backend_hrx_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_HRX_NAME;
}

static size_t ggml_backend_hrx_reg_get_device_count(ggml_backend_reg_t reg) {
    return ((ggml_hrx_reg_context *) reg->context)->devices.size();
}

static ggml_backend_dev_t ggml_backend_hrx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = (ggml_hrx_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return &ctx->devices[index];
}

static const ggml_backend_reg_i ggml_backend_hrx_reg_interface = {
    /* .get_name         = */ ggml_backend_hrx_reg_get_name,
    /* .get_device_count = */ ggml_backend_hrx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hrx_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_hrx_reg(void) {
    static ggml_hrx_reg_context reg_ctx;
    static ggml_backend_reg     reg = { GGML_BACKEND_API_VERSION, ggml_backend_hrx_reg_interface, &reg_ctx };

    static std::once_flag once;
    std::call_once(once, [] {
        ggml_hrx_discover(&reg_ctx);
        reg_ctx.devices.reserve(reg_ctx.contexts.size());
        for (auto & ctx : reg_ctx.contexts) {
            reg_ctx.devices.push_back({ ggml_backend_hrx_device_interface, &reg, ctx.get() });
        }
        for (size_t i = 0; i < reg_ctx.contexts.size(); ++i) {
            ggml_hrx_device_context * ctx = reg_ctx.contexts[i].get();
            ctx->buft = { ggml_backend_hrx_buffer_type_interface, &reg_ctx.devices[i], ctx };
        }
    });
    return &reg;
}

ggml_backend_t ggml_backend_hrx_init(size_t device) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, device), nullptr);
}

bool ggml_backend_is_hrx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hrx_guid());
}

int ggml_backend_hrx_get_device_count(void) {
    return (int) ggml_backend_reg_dev_count(ggml_backend_hrx_reg());
}

ggml_backend_buffer_type_t ggml_backend_hrx_buffer_type(size_t device) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, device));
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx_reg)
