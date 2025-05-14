#include "virtgpu-forward-impl.h"

// buffer_type_alloc_buffer
const char *
apir_buffer_type_get_name(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME);

  vn_encode_ggml_buft(encoder, buft);

  REMOTE_CALL(gpu, encoder, decoder);

  const size_t string_size = vn_decode_array_size_unchecked(decoder);
  char *string = (char *) vn_cs_decoder_alloc_array(decoder, sizeof(char), string_size);
  if (!string) {
    FATAL("%s: Could not allocate the device name buffer", __func__);
  }
  vn_decode_char_array(decoder, string, string_size);

  INFO("%s: Forward BUFT NAME --> %s", __func__, string);

  /* *** */

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  return string;
}

size_t
apir_buffer_type_get_alignment(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT);

  vn_encode_ggml_buft(encoder, buft);

  REMOTE_CALL(gpu, encoder, decoder);

  size_t alignment;
  vn_decode_size_t(decoder, &alignment);

  INFO("%s: Forward BUFT ALIGNMENT --> %zu ", __func__, alignment);

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  return alignment;
}

size_t
apir_buffer_type_get_max_size(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE);

  vn_encode_ggml_buft(encoder, buft);

  REMOTE_CALL(gpu, encoder, decoder);

  size_t max_size;
  vn_decode_size_t(decoder, &max_size);

  INFO("%s: Forward BUFT MAX SIZE --> %zu ", __func__, max_size);

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  return max_size;
}

bool
apir_buffer_type_is_host(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST);

  vn_encode_ggml_buft(encoder, buft);

  REMOTE_CALL(gpu, encoder, decoder);

  bool is_host;
  vn_decode_bool_t(decoder, &is_host);

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  return is_host;
}
