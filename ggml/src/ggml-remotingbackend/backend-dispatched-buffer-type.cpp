#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

uint32_t
backend_buffer_type_get_name(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  ggml_backend_buffer_type_t buft;
  buft = vn_decode_ggml_buffer_type(dec);

  const char *string = buft->iface.get_name(buft);

  const size_t string_size = strlen(string) + 1;
  vn_encode_array_size(enc, string_size);
  vn_encode_char_array(enc, string, string_size);

  return 0;
}

uint32_t
backend_buffer_type_get_alignment(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  ggml_backend_buffer_type_t buft;
  buft = vn_decode_ggml_buffer_type(dec);

  size_t value = buft->iface.get_alignment(buft);
  vn_encode_size_t(enc, &value);

  return 0;
}

uint32_t
backend_buffer_type_get_max_size(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  ggml_backend_buffer_type_t buft;
  buft = vn_decode_ggml_buffer_type(dec);

  size_t value = buft->iface.get_max_size(buft);
  vn_encode_size_t(enc, &value);

  return 0;
}

uint32_t
backend_buffer_type_is_host(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  ggml_backend_buffer_type_t buft;
  buft = vn_decode_ggml_buffer_type(dec);

  bool is_host = buft->iface.is_host(buft);
  vn_encode_bool_t(enc, &is_host);

  return 0;
}

uint32_t
backend_buffer_type_alloc_buffer(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);

  ggml_backend_buffer_type_t buft;
  buft = vn_decode_ggml_buffer_type(dec);

  size_t size;
  vn_decode_size_t(dec, &size);

  ggml_backend_buffer_t buffer;

  buffer = buft->iface.alloc_buffer(buft, size);

  vn_encode_ggml_buffer(enc, buffer);

  if (buffer) {
    track_backend_buffer(buffer);
  }

  return 0;
}
