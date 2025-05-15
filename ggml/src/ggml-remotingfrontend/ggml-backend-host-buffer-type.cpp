#include "ggml-remoting.h"

#define BUFT_TO_GPU(name) \
  ((struct ggml_backend_remoting_device_context *) (name)->device->context)->gpu

extern const ggml_backend_buffer_i ggml_backend_remoting_buffer_interface;

static ggml_backend_buffer_t
ggml_backend_remoting_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
  BEING_IMPLEMENTED;
  struct virtgpu *gpu = BUFT_TO_GPU(buft);
  UNUSED(gpu);

  void *ctx = NULL;

  NOT_IMPLEMENTED;

  STOP_HERE;
  return ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, ctx, size);
}

static const char *
ggml_backend_remoting_host_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
  UNUSED(buft);

  IMPLEMENTED;

  return "GUEST host buffer";
}

static size_t
ggml_backend_remoting_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
  UNUSED(buft);

  NOT_IMPLEMENTED;

  return 4096;
}

static bool
ggml_backend_remoting_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
  UNUSED(buft);

  NOT_IMPLEMENTED;

  return true;
}

const ggml_backend_buffer_type_i ggml_backend_remoting_host_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_remoting_host_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_remoting_host_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_remoting_host_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
    /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
  };
