#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-remoting-backend.h"

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
