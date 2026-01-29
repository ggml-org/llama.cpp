#include "backend-ops.hpp"
#include "common.hpp"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "logger.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace {

qnn::ggml_backend_qnn_device_context * get_device_context(ggml_backend_dev_t dev) {
    return reinterpret_cast<qnn::ggml_backend_qnn_device_context *>(dev->context);
}

qnn::qnn_buffer_interface * get_buffer_context(ggml_backend_buffer_t buffer) {
    return reinterpret_cast<qnn::qnn_buffer_interface *>(buffer->context);
}

/*
 * -----------------------------------------------------------------------------------------------
 * qnn backend buffer object
 * -----------------------------------------------------------------------------------------------
 */
void ggml_backend_qnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = get_buffer_context(buffer);
    delete ctx;
}

void * ggml_backend_qnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = get_buffer_context(buffer);
    return ctx->get_buffer();
}

ggml_status ggml_backend_qnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_qnn_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                        ggml_tensor *         tensor,
                                        const void *          data,
                                        size_t                offset,
                                        size_t                size) {
    GGML_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

void ggml_backend_qnn_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                        const ggml_tensor *   tensor,
                                        void *                data,
                                        size_t                offset,
                                        size_t                size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

bool ggml_backend_qnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}

void ggml_backend_qnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = get_buffer_context(buffer);
    memset(ctx->get_buffer(), value, ctx->get_size());
}

constexpr const ggml_backend_buffer_i ggml_backend_qnn_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_qnn_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_qnn_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_qnn_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_qnn_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_qnn_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_qnn_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_qnn_buffer_clear,
    /* .reset           = */ nullptr,
};

/*
 * -----------------------------------------------------------------------------------------------
 * qnn backend object
 * -----------------------------------------------------------------------------------------------
 */
const char * ggml_backend_qnn_buffer_type_name(ggml_backend_buffer_type_t buft) {
    auto * dev_ctx = get_device_context(buft->device);
    return qnn::get_backend_name(dev_ctx->device);
}

ggml_backend_buffer_t ggml_backend_qnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    qnn::qnn_buffer_interface * ctx = new qnn::qnn_mem_buffer(size);
    if (!ctx->is_valid()) {
        return nullptr;
    }

    QNN_LOG_DEBUG("[%s]alloc buffer: %p, size: %ld\n", qnn::get_backend_name(get_device_context(buft->device)->device),
                  (void *) ctx->get_buffer(), (long) size);
    return ggml_backend_buffer_init(buft, ggml_backend_qnn_buffer_interface, ctx, size);
}

size_t ggml_backend_qnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // TODO: fix this
    return 32;
}

size_t ggml_backend_qnn_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // TODO: get the max size from device
    return 1024L * 1024 * 1024;
}

bool ggml_backend_qnn_buffer_is_host(ggml_backend_buffer_type_t buft) {
    // TODO: fix this
    GGML_UNUSED(buft);
    return true;
}

const char * ggml_backend_qnn_name(ggml_backend_t backend) {
    auto * device_ctx = get_device_context(backend->device);
    return device_ctx->name.c_str();
}

void ggml_backend_qnn_free(ggml_backend_t backend) {
    auto * device_ctx = get_device_context(backend->device);
    QNN_LOG_INFO("idx %d, name:%s\n", device_ctx->device, device_ctx->name.c_str());

    auto & instance = device_ctx->instance;
    if (instance) {
        device_ctx->qnn_graph_cache.clear();
        device_ctx->qnn_interface.reset();
        instance->qnn_finalize();
        instance.reset();
    }

    delete backend;
}

ggml_guid_t ggml_backend_qnn_guid() {
    static ggml_guid guid = { 0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81,
                              0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09 };
    return &guid;
}

bool ggml_backend_is_qnn(ggml_backend_t backend) {
    return ggml_guid_matches(backend->guid, ggml_backend_qnn_guid());
}

bool ggml_backend_qnn_cpy_tensor_async(ggml_backend_t      backend_src,
                                       ggml_backend_t      backend_dst,
                                       const ggml_tensor * src,
                                       ggml_tensor *       dst) {
    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);

    QNN_LOG_DEBUG("opy form %s to %s, src_is_qnn: %d, dst_is_qnn: %d\n", ggml_get_name(src), ggml_get_name(dst),
                  (int) ggml_backend_is_qnn(backend_src), (int) ggml_backend_is_qnn(backend_dst));
    return false;
}

ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type ggml_backend_qnn_buffer_types[QNN_BACKEND_COUNT];
    auto *                          dev_ctx = get_device_context(dev);
    if (!ggml_backend_qnn_buffer_types[dev_ctx->device].device) {
        ggml_backend_qnn_buffer_types[dev_ctx->device] = {
            /* .iface   = */ {
                              /* .get_name         = */ ggml_backend_qnn_buffer_type_name,
                              /* .alloc_buffer     = */
                ggml_backend_qnn_buffer_type_alloc_buffer,  /* .get_alignment    = */
                ggml_backend_qnn_buffer_type_get_alignment, /* .get_max_size     = */
                ggml_backend_qnn_buffer_type_get_max_size, /* .get_alloc_size   = */ nullptr,          // defaults to ggml_nbytes
                /* .is_host          = */ ggml_backend_qnn_buffer_is_host,
                              },
            /* .device */
            dev,
            /* .context = */ nullptr,
        };
    } else {
        GGML_ASSERT(ggml_backend_qnn_buffer_types[dev_ctx->device].device == dev);
    }

    return &ggml_backend_qnn_buffer_types[dev_ctx->device];
}

ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    return qnn::device_compute_graph(get_device_context(backend->device), cgraph) ? GGML_STATUS_SUCCESS :
                                                                                    GGML_STATUS_FAILED;
}

constexpr const ggml_backend_i ggml_backend_qnn_interface = {
    /* .get_name                = */ ggml_backend_qnn_name,
    /* .free                    = */ ggml_backend_qnn_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ ggml_backend_qnn_cpy_tensor_async,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_qnn_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

/*
 * -----------------------------------------------------------------------------------------------
 * qnn backend device object
 * -----------------------------------------------------------------------------------------------
 */
const char * ggml_backend_qnn_device_get_name(ggml_backend_dev_t dev) {
    auto * dev_ctx = get_device_context(dev);
    return qnn::get_backend_name(dev_ctx->device);
}

const char * ggml_backend_qnn_device_get_description(ggml_backend_dev_t dev) {
    auto * dev_ctx = get_device_context(dev);
    return dev_ctx->description.empty() ? qnn::get_backend_desc(dev_ctx->device) : dev_ctx->description.c_str();
}

void ggml_backend_qnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free  = common::get_system_free_memory_in_bytes();
    *total = common::get_system_total_memory_in_bytes();
    QNN_LOG_DEBUG("free memory: %ldMB, total memory: %ldMB\n", (*free / 1048576), (*total) / 1048576);
}

enum ggml_backend_dev_type ggml_backend_qnn_device_get_type(ggml_backend_dev_t dev) {
    return qnn::get_device_caps(get_device_context(dev)->device).type;
}

void ggml_backend_qnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_qnn_device_get_name(dev);
    props->description = ggml_backend_qnn_device_get_description(dev);
    props->type        = ggml_backend_qnn_device_get_type(dev);
    ggml_backend_qnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* async                */ false,
        /* host_buffer          */ false,
        /* buffer_from_host_ptr */ false,
        /* events               */ false,
    };
}

ggml_backend_t ggml_backend_qnn_init_with_device_context(ggml_backend_dev_t dev, const char * extend_lib_search_path) {
    if (!extend_lib_search_path) {
        extend_lib_search_path = GGML_QNN_DEFAULT_LIB_SEARCH_PATH;
        QNN_LOG_WARN(
            "extend_lib_search_path is nullptr, will "
            "use " GGML_QNN_DEFAULT_LIB_SEARCH_PATH " as default\n");
    }

    auto *     dev_ctx = get_device_context(dev);
    const auto device  = dev_ctx->device;
    QNN_LOG_DEBUG("device %s\n", qnn::get_backend_name(device));
    QNN_LOG_DEBUG("extend_lib_search_path %s\n", extend_lib_search_path);
    auto instance = std::make_shared<qnn::qnn_instance>(extend_lib_search_path, device);
    if (!instance->qnn_init(nullptr)) {
        QNN_LOG_WARN("failed to init qnn backend %s\n", qnn::get_backend_name(device));
        return nullptr;
    }
    auto qnn_interface = instance->get_qnn_interface();
    if (!qnn_interface) {
        QNN_LOG_WARN("qnn subsystem failure\n");
        return nullptr;
    }

    std::string device_name = qnn::get_backend_name(device);
    QNN_LOG_INFO("qnn device name %s\n", device_name.c_str());
    const auto & device_caps          = qnn::get_device_caps(device);
    dev_ctx->instance                 = instance;
    dev_ctx->qnn_interface            = qnn_interface;
    dev_ctx->socinfo                  = instance->get_soc_info();
    dev_ctx->supported_types          = device_caps.supported_types;
    dev_ctx->cpu_preprocess_types     = device_caps.cpu_preprocess_types;
    dev_ctx->max_tensor_size_in_bytes = device_caps.max_tensor_size_in_bytes;
    {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "%s(%s)", qnn::get_chipset_desc(dev_ctx->socinfo.soc_model),
                 qnn::get_backend_desc(dev_ctx->device));
        dev_ctx->description = buffer;
    }

#ifdef GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS
    // TODO: remove npu from here if hardware quantization is supported
    dev_ctx->enable_cpu_dequantize = device == QNN_BACKEND_CPU;
#endif

    ggml_backend_t qnn_backend = new ggml_backend{
        /* .guid      = */ ggml_backend_qnn_guid(),
        /* .iface     = */ ggml_backend_qnn_interface,
        /* .device    = */ dev,
        /* .context   = */ nullptr,
    };

    return qnn_backend;
}

ggml_backend_t ggml_backend_qnn_device_init(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_qnn_init_with_device_context(dev, params);
}

ggml_backend_buffer_type_t ggml_backend_qnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_qnn_buffer_type(dev);
}

ggml_backend_buffer_t ggml_backend_qnn_device_buffer_from_ptr(ggml_backend_dev_t dev,
                                                              void *             ptr,
                                                              size_t             size,
                                                              size_t             max_tensor_size) {
    // TODO
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

bool ggml_backend_qnn_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // Note that this function could be called before the device context is initialized
    auto * device_ctx = get_device_context(dev);
    return qnn::device_supports_op(device_ctx, op);
}

bool ggml_backend_qnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

bool ggml_backend_qnn_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
#ifdef NDEBUG
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
#else
    auto * device_ctx = get_device_context(dev);
    QNN_LOG_DEBUG("[%s][%s]offload op\n", qnn::get_backend_name(device_ctx->device), ggml_op_name(op->op));
#endif
    return false;
}

constexpr const ggml_backend_device_i ggml_backend_qnn_device_interface = {
    /* .get_name             = */ ggml_backend_qnn_device_get_name,
    /* .get_description      = */ ggml_backend_qnn_device_get_description,
    /* .get_memory           = */ ggml_backend_qnn_device_get_memory,
    /* .get_type             = */ ggml_backend_qnn_device_get_type,
    /* .get_props            = */ ggml_backend_qnn_device_get_props,
    /* .init_backend         = */ ggml_backend_qnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_qnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_qnn_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_qnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_qnn_device_supports_buft,
    /* .offload_op           = */ ggml_backend_qnn_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

class qnn_device_proxy : public backend_device_proxy {
  public:
    explicit qnn_device_proxy(backend_index_type device) {
        const auto & device_caps = qnn::get_device_caps(device);
        _device_context          = std::make_unique<qnn::ggml_backend_qnn_device_context>(
            /* .device   = */ device,  // init from the last device, i.e. NPU
            /* .threads  = */ 1,       // TODO: fix this
            /* .name     = */ qnn::get_backend_name(device),
            /* .supported_types = */ device_caps.supported_types);
    }

    const ggml_backend_device_i & get_iface() const { return ggml_backend_qnn_device_interface; }

    void * get_context() { return _device_context.get(); }

  private:
    std::unique_ptr<qnn::ggml_backend_qnn_device_context> _device_context;
};

}  // namespace

backend_device_proxy_ptr create_qnn_backend_context(backend_index_type device) {
    if (device >= QNN_BACKEND_COUNT) {
        QNN_LOG_ERROR("[qnn]invalid device %d\n", device);
        return backend_device_proxy_ptr();
    }

#ifndef GGML_QNN_ENABLE_CPU_BACKEND
    if (device == QNN_BACKEND_CPU) {
        /*
                     * here we skip the initialization of CPU device,
                     *   cause it'll block unsupported ops fallback to ggml cpu backend
                     */
        GGML_LOG_DEBUG("qnn backend registry skip CPU device\n");
        return backend_device_proxy_ptr();
    }
#endif

    return std::make_unique<qnn_device_proxy>(device);
}
