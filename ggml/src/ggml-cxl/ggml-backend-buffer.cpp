#include "ggml-cxl-impl.h"

#include <cstring>

static void ggml_backend_cxl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cxl_buffer_context * ctx = CXL_BUF_CTX(buffer);
    if (ctx->dev_ptr != 0) {
        std::lock_guard<std::mutex> lock(ctx->dev_ctx->cmd_mutex);
        cxl_device_free(&ctx->dev_ctx->cxl_dev, ctx->dev_ptr);
    }
    delete ctx;
}

static void * ggml_backend_cxl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cxl_buffer_context * ctx = CXL_BUF_CTX(buffer);
    // Return the device pointer cast as a host pointer.
    // This is an opaque handle -- actual data access goes through set/get_tensor.
    return (void *)(uintptr_t)ctx->dev_ptr;
}

static void ggml_backend_cxl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor * tensor,
                                                const void * data,
                                                size_t offset,
                                                size_t size) {
    ggml_backend_cxl_buffer_context * ctx = CXL_BUF_CTX(buffer);
    uint64_t dev_dst = (uint64_t)(uintptr_t)tensor->data + offset;

    std::lock_guard<std::mutex> lock(ctx->dev_ctx->cmd_mutex);
    int ret = cxl_device_htod(&ctx->dev_ctx->cxl_dev, dev_dst, data, size);
    if (ret != 0) {
        GGML_LOG_ERROR(GGML_CXL_LOG "%s: htod failed for tensor %s\n", __func__, tensor->name);
    }
}

static void ggml_backend_cxl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor * tensor,
                                                void * data,
                                                size_t offset,
                                                size_t size) {
    ggml_backend_cxl_buffer_context * ctx = CXL_BUF_CTX(buffer);
    uint64_t dev_src = (uint64_t)(uintptr_t)tensor->data + offset;

    std::lock_guard<std::mutex> lock(ctx->dev_ctx->cmd_mutex);
    int ret = cxl_device_dtoh(&ctx->dev_ctx->cxl_dev, data, dev_src, size);
    if (ret != 0) {
        GGML_LOG_ERROR(GGML_CXL_LOG "%s: dtoh failed for tensor %s\n", __func__, tensor->name);
    }
}

static bool ggml_backend_cxl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor * src,
                                                ggml_tensor * dst) {
    // Only support copy within the same CXL device
    if (src->buffer == nullptr || dst->buffer == nullptr) {
        return false;
    }

    ggml_backend_cxl_buffer_context * src_ctx = CXL_BUF_CTX(src->buffer);
    ggml_backend_cxl_buffer_context * dst_ctx = CXL_BUF_CTX(buffer);

    if (src_ctx->dev_ctx != dst_ctx->dev_ctx) {
        // Cross-device copy: bounce through host
        size_t size = ggml_nbytes(src);
        std::vector<uint8_t> tmp(size);

        {
            std::lock_guard<std::mutex> lock(src_ctx->dev_ctx->cmd_mutex);
            cxl_device_dtoh(&src_ctx->dev_ctx->cxl_dev, tmp.data(),
                            (uint64_t)(uintptr_t)src->data, size);
        }
        {
            std::lock_guard<std::mutex> lock(dst_ctx->dev_ctx->cmd_mutex);
            cxl_device_htod(&dst_ctx->dev_ctx->cxl_dev,
                            (uint64_t)(uintptr_t)dst->data, tmp.data(), size);
        }
        return true;
    }

    // Same device: use device-to-device copy via DTOH + HTOD
    // (The host-side QEMU handler could implement true D2D, but for now bounce through BAR)
    size_t size = ggml_nbytes(src);
    std::vector<uint8_t> tmp(size);

    std::lock_guard<std::mutex> lock(dst_ctx->dev_ctx->cmd_mutex);
    cxl_device_dtoh(&dst_ctx->dev_ctx->cxl_dev, tmp.data(),
                    (uint64_t)(uintptr_t)src->data, size);
    cxl_device_htod(&dst_ctx->dev_ctx->cxl_dev,
                    (uint64_t)(uintptr_t)dst->data, tmp.data(), size);
    return true;
}

static void ggml_backend_cxl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cxl_buffer_context * ctx = CXL_BUF_CTX(buffer);

    std::lock_guard<std::mutex> lock(ctx->dev_ctx->cmd_mutex);
    cxl_device_memset(&ctx->dev_ctx->cxl_dev, ctx->dev_ptr, value, ctx->size);
}

static void ggml_backend_cxl_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor * tensor,
                                                   uint8_t value,
                                                   size_t offset,
                                                   size_t size) {
    ggml_backend_cxl_buffer_context * ctx = CXL_BUF_CTX(buffer);
    uint64_t dev_ptr = (uint64_t)(uintptr_t)tensor->data + offset;

    std::lock_guard<std::mutex> lock(ctx->dev_ctx->cmd_mutex);
    cxl_device_memset(&ctx->dev_ctx->cxl_dev, dev_ptr, value, size);
}

const ggml_backend_buffer_i ggml_backend_cxl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cxl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cxl_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_cxl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cxl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cxl_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cxl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cxl_buffer_clear,
    /* .reset           = */ NULL,
};
