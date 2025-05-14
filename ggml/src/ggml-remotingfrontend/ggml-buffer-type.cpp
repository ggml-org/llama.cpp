#include "ggml-remoting.h"

#define BUFT_TO_GPU(name) \
  ((struct ggml_backend_remoting_device_context *) (name)->device->context)->gpu

extern const ggml_backend_buffer_i ggml_backend_remoting_buffer_interface;

static ggml_backend_buffer_t
ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
  BEING_IMPLEMENTED;
  struct virtgpu *gpu = BUFT_TO_GPU(buft);
  UNUSED(gpu);
  /* ... */

  void *ctx = NULL;

  return ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, ctx, size);
}

static const char *
ggml_backend_remoting_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
  BEING_IMPLEMENTED;

  struct virtgpu *gpu = BUFT_TO_GPU(buft);

  return apir_buffer_type_get_name(gpu, buft);
}

static size_t
ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
  IMPLEMENTED;

  struct virtgpu *gpu = BUFT_TO_GPU(buft);

  return apir_buffer_type_get_alignment(gpu, buft);
}

static size_t
ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
  IMPLEMENTED;
  struct virtgpu *gpu = BUFT_TO_GPU(buft);

  return apir_buffer_type_get_max_size(gpu, buft);
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
  /* .is_host          = */ ggml_backend_remoting_buffer_type_is_host,
};

/****************************************************************************************/

static void ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
  ggml_backend_remoting_buffer_context * ctx = (ggml_backend_remoting_buffer_context *)buffer->context;
  NEXT;
  NOT_IMPLEMENTED;

  ggml_remoting_destroy_buffer(ctx->dev_buffer);
  delete ctx;
}

static enum ggml_status ggml_backend_remoting_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
  NEXT;
  NOT_IMPLEMENTED;
  if (tensor->view_src != nullptr) {
    GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
  }
  return GGML_STATUS_SUCCESS;
}

static void * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
  UNUSED(buffer);

  NEXT;
  NOT_IMPLEMENTED;

  return (void *) 4096;
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

#if 0
  ggml_backend_remoting_buffer_context * buf_ctx = (ggml_backend_remoting_buffer_context *)buffer->context;
  remoting_buffer buf = buf_ctx->dev_buffer;

  ggml_remoting_buffer_write(buf, remoting_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
#else
  UNUSED(buffer);
  UNUSED(tensor);
  UNUSED(data);
  UNUSED(offset);
  UNUSED(size);
#endif
}

static void ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
  NOT_IMPLEMENTED;

#if 0
  ggml_backend_remoting_buffer_context * buf_ctx = (ggml_backend_remoting_buffer_context *)buffer->context;

  remoting_buffer buf = buf_ctx->dev_buffer;

  ggml_remoting_buffer_read(buf, remoting_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
#else
  UNUSED(buffer);
  UNUSED(tensor);
  UNUSED(data);
  UNUSED(offset);
  UNUSED(size);
#endif
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
  NOT_IMPLEMENTED;

  ggml_backend_remoting_buffer_context * ctx = (ggml_backend_remoting_buffer_context *)buffer->context;

  ggml_remoting_buffer_memset(ctx->dev_buffer, 0, value, buffer->size);
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
