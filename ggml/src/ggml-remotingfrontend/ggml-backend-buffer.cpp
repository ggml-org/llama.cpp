#include "ggml-remoting.h"

#define BUFFER_TO_GPU(name)                                             \
    ((struct ggml_backend_remoting_buffer_context *) (name)->context)->gpu

struct timer_data get_tensor_timer = {0, 0, 0, "get_tensor"};
struct timer_data set_tensor_timer = {0, 0, 0, "set_tensor"};
struct timer_data cpy_tensor_timer = {0, 0, 0, "cpy_tensor"};

struct timer_data get_tensor_from_ptr_timer = {0, 0, 0, "get_tensor_from_ptr"};
struct timer_data set_tensor_from_ptr_timer = {0, 0, 0, "set_tensor_from_ptr"};

static void * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
    struct ggml_backend_remoting_buffer_context *context = (struct ggml_backend_remoting_buffer_context *) buffer->context;
    if (context->base) {
        return context->base;
    }

    context->base = apir_buffer_get_base(BUFFER_TO_GPU(buffer),
                                         BUFFER_TO_APIR_CONTEXT(buffer));

    return context->base;
}

static void ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    struct virtgpu *gpu = BUFFER_TO_GPU(buffer);

    struct ggml_backend_remoting_buffer_context *context = BUFFER_TO_GGML_CONTEXT(buffer);
    if (context->is_from_ptr) {
        memcpy((char *)tensor->data + offset, data, size);
    } else {
        apir_buffer_set_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }

    return;
}

static void ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    struct virtgpu *gpu = BUFFER_TO_GPU(buffer);
    struct ggml_backend_remoting_buffer_context *context = BUFFER_TO_GGML_CONTEXT(buffer);
    if (context->is_from_ptr) {
        memcpy(data, (const char *)tensor->data + offset, size);
    } else {
        apir_buffer_get_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }
}

static void ggml_backend_remoting_buffer_set_tensor_from_ptr(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    UNUSED(buffer);

    memcpy((char *)tensor->data + offset, data, size);

    return;
}

static void ggml_backend_remoting_buffer_get_tensor_from_ptr(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    UNUSED(buffer);

    memcpy(data, (const char *)tensor->data + offset, size);
}

static bool ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    struct virtgpu *gpu = BUFFER_TO_GPU(buffer);

    bool ret = apir_buffer_cpy_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), src, dst);

    return ret;
}

static void ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    struct virtgpu *gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_clear(gpu, BUFFER_TO_APIR_CONTEXT(buffer), value);

    return;
}

static void ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    struct virtgpu *gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_free_buffer(gpu, BUFFER_TO_APIR_CONTEXT(buffer));

    struct ggml_backend_remoting_buffer_context *context = BUFFER_TO_GGML_CONTEXT(buffer);
    free(context);
    buffer->context = NULL;
}

const ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ NULL,
};

const ggml_backend_buffer_i ggml_backend_remoting_buffer_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor_from_ptr,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ NULL,
};
