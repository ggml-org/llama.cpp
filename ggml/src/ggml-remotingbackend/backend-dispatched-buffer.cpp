#include "backend-dispatched.h"
#include "backend-internal.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cstdint>

struct timer_data get_tensor_timer = { 0, 0, 0, "get_tensor" };
struct timer_data set_tensor_timer = { 0, 0, 0, "set_tensor" };
struct timer_data cpy_tensor_timer = { 0, 0, 0, "set_tensor" };

uint32_t backend_buffer_get_base(struct apir_encoder *       enc,
                                 struct apir_decoder *       dec,
                                 struct virgl_apir_context * ctx) {
    UNUSED(ctx);
    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    uintptr_t base = (uintptr_t) buffer->iface.get_base(buffer);
    apir_encode_uintptr_t(enc, &base);

    return 0;
}

uint32_t backend_buffer_set_tensor(struct apir_encoder *       enc,
                                   struct apir_decoder *       dec,
                                   struct virgl_apir_context * ctx) {
    UNUSED(ctx);
    UNUSED(enc);

    start_timer(&set_tensor_timer);

    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t offset;
    apir_decode_size_t(dec, &offset);

    size_t size;
    apir_decode_size_t(dec, &size);

    void * shmem_data = ctx->iface.get_shmem_ptr(ctx->virgl_ctx, shmem_res_id);

    if (!shmem_data) {
        FATAL("Couldn't get the shmem addr from virgl :/");
    }

#if 0
  INFO("Calling (%p)->set_tensor(tensor=%p, data=%p, offset=%lu, size=%lu",
       buffer, tensor, shmem_data, offset, size);
#endif
#if 0
  void **addr = (void **)(uintptr_t) shmem_data;
  for (int i = 0; i <= 10; i++) {
    INFO("%s: %p | %llx", __func__, addr, *addr);
    addr++;
  }
  INFO("\n");
#endif

    buffer->iface.set_tensor(buffer, tensor, shmem_data, offset, size);

    stop_timer(&set_tensor_timer);

    return 0;
}

uint32_t backend_buffer_get_tensor(struct apir_encoder *       enc,
                                   struct apir_decoder *       dec,
                                   struct virgl_apir_context * ctx) {
    UNUSED(ctx);
    UNUSED(enc);

    start_timer(&get_tensor_timer);

    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    const ggml_tensor * tensor;
    // safe to remove the const qualifier here
    tensor = apir_decode_ggml_tensor(dec);

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    size_t offset;
    apir_decode_size_t(dec, &offset);

    size_t size;
    apir_decode_size_t(dec, &size);

    void * shmem_data = ctx->iface.get_shmem_ptr(ctx->virgl_ctx, shmem_res_id);
    if (!shmem_data) {
        FATAL("Couldn't get the shmem addr from virgl :/");
    }

    buffer->iface.get_tensor(buffer, tensor, shmem_data, offset, size);

    stop_timer(&get_tensor_timer);

    return 0;
}

uint32_t backend_buffer_cpy_tensor(struct apir_encoder *       enc,
                                   struct apir_decoder *       dec,
                                   struct virgl_apir_context * ctx) {
    UNUSED(ctx);

    start_timer(&cpy_tensor_timer);

    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);
    INFO("%s <---->", __func__);
    const ggml_tensor * src;
    // safe to remove the const qualifier here
    src               = apir_decode_ggml_tensor(dec);
    ggml_tensor * dst = (ggml_tensor *) (uintptr_t) apir_decode_ggml_tensor(dec);

    bool ret = buffer->iface.cpy_tensor(buffer, src, (ggml_tensor *) dst);

    apir_encode_bool_t(enc, &ret);

    stop_timer(&cpy_tensor_timer);

    return 0;
}

uint32_t backend_buffer_clear(struct apir_encoder * enc, struct apir_decoder * dec, struct virgl_apir_context * ctx) {
    UNUSED(ctx);
    UNUSED(enc);

    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    uint8_t value;
    apir_decode_uint8_t(dec, &value);

    buffer->iface.clear(buffer, value);

    return 0;
}

uint32_t backend_buffer_free_buffer(struct apir_encoder *       enc,
                                    struct apir_decoder *       dec,
                                    struct virgl_apir_context * ctx) {
    UNUSED(ctx);
    UNUSED(enc);

    ggml_backend_buffer_t buffer;
    buffer = apir_decode_ggml_buffer(dec);

    if (!apir_untrack_backend_buffer(buffer)) {
        WARNING("%s: unknown buffer %p", (void *) buffer);
        return 1;
    }

    buffer->iface.free_buffer(buffer);

    return 0;
}
