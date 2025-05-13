#include <memory>

#include "ggml-remoting.h"

void ggml_remoting_destroy_buffer(remoting_buffer& buf) {
  NOT_IMPLEMENTED;

  UNUSED(buf);
}

static void ggml_remoting_buffer_write(remoting_buffer& dst, size_t offset, const void * src, size_t size) {
  NOT_IMPLEMENTED;

  UNUSED(dst);
  UNUSED(offset);
  UNUSED(src);
  UNUSED(size);
}

static void ggml_remoting_buffer_read(remoting_buffer& src, size_t offset, void * dst, size_t size) {
  NOT_IMPLEMENTED;

  UNUSED(src);
  UNUSED(offset);
  UNUSED(dst);
  UNUSED(size);
}

static void ggml_remoting_buffer_copy_async(remoting_context& ctx, remoting_buffer& dst, size_t dst_offset, remoting_buffer& src, size_t src_offset, size_t size) {
  NOT_IMPLEMENTED;

  UNUSED(ctx);
  UNUSED(dst);
  UNUSED(dst_offset);
  UNUSED(src);
  UNUSED(src_offset);
  UNUSED(size);
}

static void * const remoting_ptr_base = (void *)(uintptr_t) 0x1000;  // NOLINT

static uint64_t remoting_tensor_offset(const ggml_tensor * tensor) {
  NOT_IMPLEMENTED;

  if (tensor->view_src) {
    return (uint8_t *) tensor->view_src->data - (uint8_t *) remoting_ptr_base;
  }
  return (uint8_t *) tensor->data - (uint8_t *) remoting_ptr_base;
}
