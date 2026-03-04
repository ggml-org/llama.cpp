#include "ggml-cxl-impl.h"

#include <cstring>

static const char * ggml_backend_cxl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cxl_device_context * ctx = CXL_DEV_CTX(buft->device);
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_cxl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cxl_device_context * dev_ctx = CXL_DEV_CTX(buft->device);

    std::lock_guard<std::mutex> lock(dev_ctx->cmd_mutex);

    uint64_t dev_ptr = cxl_device_alloc(&dev_ctx->cxl_dev, size);
    if (dev_ptr == 0) {
        GGML_LOG_ERROR(GGML_CXL_LOG "%s: failed to allocate %zu bytes on %s\n",
                       __func__, size, dev_ctx->name.c_str());
        return nullptr;
    }

    ggml_backend_cxl_buffer_context * buf_ctx = new ggml_backend_cxl_buffer_context;
    buf_ctx->dev_ctx = dev_ctx;
    buf_ctx->dev_ptr = dev_ptr;
    buf_ctx->size    = size;

    return ggml_backend_buffer_init(buft, ggml_backend_cxl_buffer_interface, buf_ctx, size);
}

static size_t ggml_backend_cxl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128; // same as CUDA
}

static size_t ggml_backend_cxl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_cxl_device_context * ctx = CXL_DEV_CTX(buft->device);
    return ctx->cxl_dev.total_memory;
}

static size_t ggml_backend_cxl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    // For quantized types, may need padding like CUDA does
    size_t size = ggml_nbytes(tensor);
    // Pad to 512 bytes for quantized types (matching CUDA pattern)
    if (ggml_is_quantized(tensor->type)) {
        size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        size_t row_size_padded = (row_size + 511) & ~511;
        int64_t n_rows = ggml_nrows(tensor);
        size = n_rows * row_size_padded;
    }
    return size;
}

const ggml_backend_buffer_type_i ggml_backend_cxl_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_cxl_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_cxl_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_cxl_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_cxl_buffer_type_get_max_size,
    /* .get_alloc_size = */ ggml_backend_cxl_buffer_type_get_alloc_size,
    /* .is_host        = */ NULL,
};
