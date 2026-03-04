#include "ggml-cxl-impl.h"

static const char * ggml_backend_cxl_device_get_name(ggml_backend_dev_t dev) {
    return CXL_DEV_CTX(dev)->name.c_str();
}

static const char * ggml_backend_cxl_device_get_description(ggml_backend_dev_t dev) {
    return CXL_DEV_CTX(dev)->description.c_str();
}

static void ggml_backend_cxl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cxl_device_context * ctx = CXL_DEV_CTX(dev);

    std::lock_guard<std::mutex> lock(ctx->cmd_mutex);
    cxl_device_get_memory(&ctx->cxl_dev, free, total);
}

static enum ggml_backend_dev_type ggml_backend_cxl_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cxl_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    ggml_backend_cxl_device_context * ctx = CXL_DEV_CTX(dev);

    props->name        = ctx->name.c_str();
    props->description = ctx->description.c_str();
    props->type        = GGML_BACKEND_DEVICE_TYPE_GPU;

    std::lock_guard<std::mutex> lock(ctx->cmd_mutex);
    cxl_device_get_memory(&ctx->cxl_dev, &props->memory_free, &props->memory_total);

    props->caps.async              = false;
    props->caps.host_buffer        = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events             = false;
}

static bool ggml_backend_cxl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    // The CXL device forwards all ops to the host GPU backend,
    // which supports all standard GGML operations
    return true;
}

static bool ggml_backend_cxl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->device == dev;
}

static bool ggml_backend_cxl_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    // Offload compute-intensive operations
    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            return true;
        default:
            return false;
    }
}

const ggml_backend_device_i ggml_backend_cxl_device_interface = {
    /* .get_name             = */ ggml_backend_cxl_device_get_name,
    /* .get_description      = */ ggml_backend_cxl_device_get_description,
    /* .get_memory           = */ ggml_backend_cxl_device_get_memory,
    /* .get_type             = */ ggml_backend_cxl_device_get_type,
    /* .get_props            = */ ggml_backend_cxl_device_get_props,
    /* .init_backend         = */ ggml_backend_cxl_device_init,
    /* .get_buffer_type      = */ ggml_backend_cxl_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_cxl_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cxl_device_supports_buft,
    /* .offload_op           = */ ggml_backend_cxl_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// Buffer type instances (one per device, lazily initialized)
static ggml_backend_buffer_type cxl_buffer_types[CXL_MAX_DEVICES];
static std::atomic<bool> cxl_buffer_types_initialized[CXL_MAX_DEVICES] = {};

ggml_backend_buffer_type_t ggml_backend_cxl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cxl_device_context * ctx = CXL_DEV_CTX(dev);
    int idx = ctx->index;

    if (!cxl_buffer_types_initialized[idx]) {
        cxl_buffer_types[idx] = {
            /* .iface   = */ ggml_backend_cxl_buffer_type_interface,
            /* .device  = */ dev,
            /* .context = */ ctx,
        };
        cxl_buffer_types_initialized[idx] = true;
    }

    return &cxl_buffer_types[idx];
}
