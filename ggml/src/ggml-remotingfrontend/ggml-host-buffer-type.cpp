#include "ggml-remoting.h"

// host buffer type

static const char * ggml_backend_remoting_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
  UNUSED(buft);

  NOT_IMPLEMENTED;

  return GGML_REMOTING_FRONTEND_NAME "_Host";
}

static void ggml_backend_remoting_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
  UNUSED(buffer);
  NOT_IMPLEMENTED;

# if 0
  ggml_remoting_host_free(remoting_instance.devices[0], buffer->context);
#endif
}

static ggml_backend_buffer_t ggml_backend_remoting_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
  UNUSED(buft);

  NOT_IMPLEMENTED;

  void *ptr = nullptr;
  ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
  buffer->buft = buft;
  buffer->iface.free_buffer = ggml_backend_remoting_host_buffer_free_buffer;

  return buffer;
}

static size_t ggml_backend_remoting_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
  UNUSED(buft);

  NOT_IMPLEMENTED;
  return 4096;
}

// Should be changed to return device-specific host buffer type
// but that probably requires changes in llama.cpp
ggml_backend_buffer_type_t ggml_backend_remoting_host_buffer_type() {
  static struct ggml_backend_buffer_type ggml_backend_remoting_buffer_type_host = {
    /* .iface    = */ {
      /* .get_name         = */ ggml_backend_remoting_host_buffer_type_name,
      /* .alloc_buffer     = */ ggml_backend_remoting_host_buffer_type_alloc_buffer,
      /* .get_alignment    = */ ggml_backend_remoting_host_buffer_type_get_alignment,
      /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
      /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
      /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
    },
    /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_remoting_frontend_reg(), 0),
    /* .context  = */ nullptr,
  };

  // Make sure device 0 is initialized
  //ggml_remoting_instance_init();
  //ggml_remoting_get_device(0);

  return &ggml_backend_remoting_buffer_type_host;
}
