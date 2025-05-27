#include "ggml-remoting.h"

#define BUFFER_TO_GPU(name) \
  ((struct ggml_backend_remoting_buffer_context *) (name)->context)->gpu

static void * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
  IMPLEMENTED_ONCE;

  struct ggml_backend_remoting_buffer_context *context = (struct ggml_backend_remoting_buffer_context *) buffer->context;
  if (context->base) {
    return context->base;
  }

  context->base = apir_buffer_get_base(BUFFER_TO_GPU(buffer),
				       BUFFER_TO_APIR_CONTEXT(buffer));

  return context->base;
}

static void ggml_backend_remoting_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
  NOT_IMPLEMENTED;

  STOP_HERE;

  UNUSED(buffer);
  UNUSED(tensor);
  UNUSED(value);
  UNUSED(offset);
  UNUSED(size);
}

static void ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
  IMPLEMENTED_ONCE;

  struct virtgpu *gpu = BUFFER_TO_GPU(buffer);
#if 0
  INFO("%s: data=%p, offset=%lu, size=%lu\n", __func__, data, offset, size);
#endif
#if 0
  void **addr = (void **)(uintptr_t)data;
  for (int i = 0; i <= 10; i++) {
    INFO("%s: %p | %llx", __func__, addr, *addr);
    addr++;
  }
  INFO("\n");
#endif
  apir_buffer_set_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);

  return;
}

static void ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
  IMPLEMENTED_ONCE;
  struct virtgpu *gpu = BUFFER_TO_GPU(buffer);

  apir_buffer_get_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
}


static bool ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
  NOT_IMPLEMENTED;

  STOP_HERE;

  return true;

  UNUSED(buffer);
  UNUSED(src);
  UNUSED(dst);
}

static void ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
  IMPLEMENTED_ONCE;

  struct virtgpu *gpu = BUFFER_TO_GPU(buffer);

  apir_buffer_clear(gpu, BUFFER_TO_APIR_CONTEXT(buffer), value);

  return;
}

static void ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
  UNUSED(buffer);

  IMPLEMENTED_ONCE;

  struct virtgpu *gpu = BUFFER_TO_GPU(buffer);

  apir_buffer_free_buffer(gpu, BUFFER_TO_APIR_CONTEXT(buffer));
}

const ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
  /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
  /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
  /* .init_tensor     = */ NULL,
  /* .memset_tensor   = */ ggml_backend_remoting_buffer_memset_tensor,
  /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
  /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
  /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
  /* .clear           = */ ggml_backend_remoting_buffer_clear,
  /* .reset           = */ NULL,
};
