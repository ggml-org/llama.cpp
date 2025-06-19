#include "ggml-remoting.h"

#define BUFT_TO_GPU(name) \
  ((struct ggml_backend_remoting_device_context *) (name)->device->context)->gpu

static ggml_backend_buffer_t
ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
  IMPLEMENTED_ONCE;
  struct virtgpu *gpu = BUFT_TO_GPU(buft);

  struct ggml_backend_remoting_buffer_context *context = (struct ggml_backend_remoting_buffer_context *) malloc(sizeof(*context));
  if (!context) {
    FATAL("Couldn't allocate the buffer context ...");
  }

  context->gpu = gpu;

  const int USE_FROM_PTR = true;

  if (USE_FROM_PTR) {
    context->apir_context = apir_device_buffer_from_ptr(gpu, size, size);
    context->base = context->apir_context.shmem->mmap_ptr;
    context->is_from_ptr = true;
  } else {
    context->apir_context = apir_buffer_type_alloc_buffer(gpu, buft, size);
    context->is_from_ptr = false;
    context->base = NULL;
  }
  context->is_host_buffer = false;

  ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, (void *) context, size);

  return buffer;
}

static const char *
ggml_backend_remoting_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
  IMPLEMENTED_ONCE;

  struct virtgpu *gpu = BUFT_TO_GPU(buft);

  return apir_buffer_type_get_name(gpu, buft);
}

static size_t
ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
  IMPLEMENTED_ONCE;
  struct virtgpu *gpu = BUFT_TO_GPU(buft);

  static size_t align = 0;

  if (align == 0) {
    align = apir_buffer_type_get_alignment(gpu, buft);
  }

  return align;
}

static size_t
ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
  IMPLEMENTED_ONCE;
  struct virtgpu *gpu = BUFT_TO_GPU(buft);

  static size_t max_size = 0;
  if (max_size == 0) {
    max_size = apir_buffer_type_get_max_size(gpu, buft);
  }

  return max_size;
}

static bool
ggml_backend_remoting_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
  IMPLEMENTED;
  struct virtgpu *gpu = BUFT_TO_GPU(buft);

  return apir_buffer_type_is_host(gpu, buft);
}

const ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface = {
  /* .get_name         = */ ggml_backend_remoting_buffer_type_get_name,
  /* .alloc_buffer     = */ ggml_backend_remoting_buffer_type_alloc_buffer,
  /* .get_alignment    = */ ggml_backend_remoting_buffer_type_get_alignment,
  /* .get_max_size     = */ ggml_backend_remoting_buffer_type_get_max_size,
  /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
  /* .is_host          = */ NULL,
};

const ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface = {
  /* .get_name         = */ ggml_backend_remoting_buffer_type_get_name,
  /* .alloc_buffer     = */ NULL,
  /* .get_alignment    = */ ggml_backend_remoting_buffer_type_get_alignment,
  /* .get_max_size     = */ ggml_backend_remoting_buffer_type_get_max_size,
  /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
  /* .is_host          = */ NULL,
};

/****************************************************************************************/
