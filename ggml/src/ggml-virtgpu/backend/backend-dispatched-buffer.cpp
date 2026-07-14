#include "backend-dispatched.h"
#include "backend-virgl-apir.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cstdint>

static uint32_t validate_buffer_operation(size_t offset, size_t size, const char * operation) {
    // Only check for critical integer overflow - no arbitrary size limits
    if (offset > SIZE_MAX - size) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Integer overflow in offset+size: %zu + %zu\n", operation, offset, size);
        return 1;
    }

    return 0;  // Valid
}

uint32_t backend_buffer_get_base(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    if (!buffer || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        return 1;
    }

    uintptr_t base = (uintptr_t) buffer->iface.get_base(buffer);
    apir_encode_uintptr_t(enc, &base);

    return 0;
}

uint32_t backend_buffer_set_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    if (!buffer || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        return 1;
    }

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t offset;
    apir_decode_size_t(dec, &offset);

    size_t size;
    apir_decode_size_t(dec, &size);

    if (validate_buffer_operation(offset, size, __func__) != 0) {
        return 1;
    }

    void * shmem_data = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);

    if (!shmem_data) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        return 1;
    }

    buffer->iface.set_tensor(buffer, tensor, shmem_data, offset, size);

    return 0;
}

uint32_t backend_buffer_get_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    if (!buffer || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        return 1;
    }

    const ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t offset;
    apir_decode_size_t(dec, &offset);

    size_t size;
    apir_decode_size_t(dec, &size);

    if (validate_buffer_operation(offset, size, __func__) != 0) {
        return 1;
    }

    void * shmem_data = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    if (!shmem_data) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        return 1;
    }

    buffer->iface.get_tensor(buffer, tensor, shmem_data, offset, size);

    return 0;
}

uint32_t backend_buffer_cpy_tensor(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    if (!buffer || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        return 1;
    }

    const ggml_tensor * src;
    // safe to remove the const qualifier here
    src               = apir_decode_ggml_tensor(dec);
    ggml_tensor * dst = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    bool ret = buffer->iface.cpy_tensor(buffer, src, (ggml_tensor *) dst);

    apir_encode_bool_t(enc, &ret);

    return 0;
}

uint32_t backend_buffer_clear(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    if (!buffer || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        return 1;
    }

    uint8_t value;
    apir_decode_uint8_t(dec, &value);

    buffer->iface.clear(buffer, value);

    return 0;
}

uint32_t backend_buffer_free_buffer(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(enc);

    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    if (!buffer || apir_decoder_get_fatal(dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid buffer handle from guest\n", __func__);
        return 1;
    }

    if (!apir_untrack_backend_buffer(buffer)) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: unknown buffer %p\n", __func__, (void *) buffer);
        return 1;
    }

    buffer->iface.free_buffer(buffer);

    return 0;
}
