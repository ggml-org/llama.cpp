#include "ggml-remoting.h"

static ggml_backend_buffer_t ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(sizeof(*context));
    if (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    bool async__unused, host_buffer__unused, events__unused;
    bool buffer_from_host_ptr;
    apir_device_get_props(gpu, &async__unused, &host_buffer__unused, &buffer_from_host_ptr, &events__unused);

    if (buffer_from_host_ptr) {
        context->apir_context = apir_device_buffer_from_ptr(gpu, size, size);
        context->base         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = true;
    } else {
        context->apir_context = apir_buffer_type_alloc_buffer(gpu, gpu->cached_buffer_type.host_handle, size);
        context->is_from_ptr  = false;
        context->base         = NULL;
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, (void *) context, size);

    return buffer;
}

static const char * ggml_backend_remoting_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    return gpu->cached_buffer_type.name;
}

static size_t ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    return gpu->cached_buffer_type.alignment;
}

static size_t ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    return gpu->cached_buffer_type.max_size;
}

static size_t ggml_backend_remoting_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               const ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    if (tensor->buffer == NULL
        || !tensor->buffer->context
        || !buft->device->iface.supports_buft(buft->device, tensor->buffer->buft)) {
        return ggml_nbytes(tensor);
    }

    return apir_buffer_type_get_alloc_size(gpu, gpu->cached_buffer_type.host_handle, tensor);
}

const ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_remoting_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

const ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface = {
    /* .get_name         = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer     = */ NULL,
    /* .get_alignment    = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};
