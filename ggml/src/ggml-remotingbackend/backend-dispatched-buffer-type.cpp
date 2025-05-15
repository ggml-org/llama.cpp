#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include "ggml-metal.h"

uint32_t
backend_buffer_type_get_name(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  ggml_backend_buffer_type_t buft;
  buft = vn_decode_ggml_buft(dec);

  const char *string = buft->iface.get_name(buft);

  const size_t string_size = strlen(string) + 1;
  vn_encode_array_size(enc, string_size);
  vn_encode_char_array(enc, string, string_size);

  return 0;
}

uint32_t
backend_buffer_type_get_alignment(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  ggml_backend_buffer_type_t buft;
  buft = vn_decode_ggml_buft(dec);

  size_t value = buft->iface.get_alignment(buft);
  vn_encode_size_t(enc, &value);

  return 0;
}

uint32_t
backend_buffer_type_get_max_size(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  ggml_backend_buffer_type_t buft;
  buft = vn_decode_ggml_buft(dec);

  size_t value = buft->iface.get_max_size(buft);
  vn_encode_size_t(enc, &value);

  return 0;
}

uint32_t
backend_buffer_type_is_host(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  ggml_backend_buffer_type_t buft;
  buft = vn_decode_ggml_buft(dec);

  bool is_host = buft->iface.is_host(buft);
  vn_encode_bool_t(enc, &is_host);

  return 0;
}

uint32_t
backend_buffer_type_alloc_buffer(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  ggml_backend_buffer_type_t buft;
  buft = vn_decode_ggml_buft(dec);

  size_t size;
  vn_decode_size_t(dec, &size);

  ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, size);
  apir_buffer_handle_t *buffer_handle = (apir_buffer_handle_t *) buffer;
  vn_encode_ggml_buffer_handle(enc, buffer_handle);

  return 0;
}

uint32_t
backend_buffer_get_base(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  ggml_backend_buffer_t buffer;
  buffer = vn_decode_ggml_buffer(dec);

  uintptr_t base = (uintptr_t) buffer->iface.get_base(buffer);
  vn_encode_uintptr_t(enc, &base);

  INFO("%s: send base %p\n", __func__,  (void *) base);

  return 0;
}
