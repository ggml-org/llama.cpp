#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

uint32_t backend_reg_get_device_count(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(ctx);
  UNUSED(dec);

  int32_t dev_count = reg->iface.get_device_count(reg);
  vn_encode_int32_t(enc, &dev_count);

  return 0;
}

uint32_t backend_device_get_name(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(dec);

  const char *string = dev->iface.get_name(dev);

  const size_t string_size = strlen(string) + 1;
  vn_encode_array_size(enc, string_size);
  vn_encode_char_array(enc, string, string_size);

  return 0;
}

uint32_t
backend_device_get_description(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(dec);

  const char *string = dev->iface.get_description(dev);

  const size_t string_size = strlen(string) + 1;
  vn_encode_array_size(enc, string_size);
  vn_encode_char_array(enc, string, string_size);

  return 0;
}

uint32_t
backend_device_get_type(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(dec);

  uint32_t type = dev->iface.get_type(dev);
  vn_encode_uint32_t(enc, &type);

  return 0;
}

uint32_t
backend_device_get_memory(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(dec);

  size_t free, total;
  dev->iface.get_memory(dev, &free, &total);

  vn_encode_size_t(enc, &free);
  vn_encode_size_t(enc, &total);

  return 0;
}

uint32_t
backend_device_supports_op(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);

  const ggml_tensor *op = vn_decode_ggml_tensor_inplace(dec);

  bool supports_op = dev->iface.supports_op(dev, op);

  vn_encode_bool_t(enc, &supports_op);

  return 0;
}

uint32_t
backend_device_get_buffer_type(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(dec);

  ggml_backend_buffer_type_t bufft = dev->iface.get_buffer_type(dev);

  vn_encode_ggml_buffer_type(enc, bufft);

  return 0;
}

uint32_t
backend_device_get_props(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(dec);

  struct ggml_backend_dev_props props;
  dev->iface.get_props(dev, &props);

  vn_encode_bool_t(enc, &props.caps.async);
  vn_encode_bool_t(enc, &props.caps.host_buffer);
  vn_encode_bool_t(enc, &props.caps.buffer_from_host_ptr);
  vn_encode_bool_t(enc, &props.caps.events);

  return 0;
}

uint32_t
backend_device_buffer_from_ptr(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(dec);

  uint32_t shmem_res_id;
  vn_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

  void *shmem_ptr = ctx->iface.get_shmem_ptr(ctx->virgl_ctx, shmem_res_id);
  if (!shmem_ptr) {
    FATAL("Couldn't get the shmem addr from virgl :/");
  }

  size_t size;
  vn_decode_size_t(dec, &size);
  size_t max_tensor_size;
  vn_decode_size_t(dec, &max_tensor_size);

  ggml_backend_buffer_t buffer;
  buffer = dev->iface.buffer_from_host_ptr(dev, shmem_ptr, size, max_tensor_size);

  vn_encode_ggml_buffer(enc, buffer);
  vn_encode_ggml_buffer_type(enc, buffer->buft);

  if (buffer) {
    track_backend_buffer(buffer);
  }

  return 0;
}
