#include "ggml-remoting.h"

#define BUFFER_TO_GPU(name) \
  ((struct ggml_backend_remoting_buffer_context *) (name)->context)->gpu

static enum ggml_status ggml_backend_remoting_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
  UNUSED(buffer);
  UNUSED(tensor);

  NEXT;
  NOT_IMPLEMENTED;
  STOP_HERE;
  return GGML_STATUS_SUCCESS;
}

static void * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
  UNUSED(buffer);
  IMPLEMENTED;

  struct virtgpu *gpu = BUFFER_TO_GPU(buffer);

  return apir_buffer_get_base(gpu, ((struct ggml_backend_remoting_buffer_context *) buffer->context)->handle);
}

static void ggml_backend_remoting_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
  NOT_IMPLEMENTED;

  UNUSED(buffer);
  UNUSED(tensor);
  UNUSED(value);
  UNUSED(offset);
  UNUSED(size);
}


static void ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {

  NOT_IMPLEMENTED;

  UNUSED(buffer);
  UNUSED(tensor);
  UNUSED(data);
  UNUSED(offset);
  UNUSED(size);
}

static void ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
  NOT_IMPLEMENTED;

  UNUSED(buffer);
  UNUSED(tensor);
  UNUSED(data);
  UNUSED(offset);
  UNUSED(size);
}


static bool ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
  NOT_IMPLEMENTED;

  return true;

  UNUSED(buffer);
  UNUSED(src);
  UNUSED(dst);
}

static void ggml_remoting_buffer_memset(remoting_buffer& dst, size_t offset, uint32_t c, size_t size) {
  NOT_IMPLEMENTED;

  UNUSED(dst);
  UNUSED(c);
  UNUSED(size);
  UNUSED(offset);
}

static void ggml_remoting_buffer_memset_async(remoting_context& ctx, remoting_buffer& dst, size_t offset, uint32_t c, size_t size) {
  NOT_IMPLEMENTED;

  UNUSED(ctx);
  UNUSED(dst);
  UNUSED(c);
  UNUSED(size);
  UNUSED(offset);
}

static void ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
  UNUSED(buffer);
  UNUSED(value);

  NOT_IMPLEMENTED;
}

static void ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
  UNUSED(buffer);

  NOT_IMPLEMENTED;
}

const ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
  /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
  /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
  /* .init_tensor     = */ ggml_backend_remoting_buffer_init_tensor,
  /* .memset_tensor   = */ ggml_backend_remoting_buffer_memset_tensor,
  /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
  /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
  /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
  /* .clear           = */ ggml_backend_remoting_buffer_clear,
  /* .reset           = */ NULL,
};
