#include "virtgpu-forward-impl.h"

const char *
apir_buffer_type_get_name(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME);

  apir_buffer_type_handle_t handle = (apir_buffer_type_handle_t) buft->context;
  vn_encode_apir_buffer_handle_t(encoder, &handle);

  REMOTE_CALL(gpu, encoder, decoder);

  const size_t string_size = vn_decode_array_size_unchecked(decoder);
  char *string = (char *) vn_cs_decoder_alloc_array(decoder, sizeof(char), string_size);
  if (!string) {
    FATAL("%s: Could not allocate the device name buffer", __func__);
  }
  vn_decode_char_array(decoder, string, string_size);

  //INFO("%s: Forward BUFT NAME --> %s", __func__, string);

  /* *** */

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  return string;
}

size_t
apir_buffer_type_get_alignment(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT);

  apir_buffer_type_handle_t handle = (apir_buffer_type_handle_t) buft->context;
  vn_encode_apir_buffer_handle_t(encoder, &handle);

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

  apir_buffer_type_handle_t handle = (apir_buffer_type_handle_t) buft->context;
  vn_encode_apir_buffer_handle_t(encoder, &handle);

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

  apir_buffer_type_handle_t handle = (apir_buffer_type_handle_t) buft->context;
  vn_encode_apir_buffer_handle_t(encoder, &handle);

  REMOTE_CALL(gpu, encoder, decoder);

  bool is_host;
  vn_decode_bool_t(decoder, &is_host);

  INFO("%s: buffer is host? %d", __func__, is_host);

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  return is_host;
}

apir_buffer_handle_t
apir_buffer_type_alloc_buffer(struct virtgpu *gpu, ggml_backend_buffer_type_t buft, size_t size) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  INFO("%s: allocate device memory (%lu)", __func__,  size);

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER);

  apir_buffer_type_handle_t buft_handle = (apir_buffer_type_handle_t) buft->context;
  vn_encode_apir_buffer_handle_t(encoder, &buft_handle);

  vn_encode_size_t(encoder, &size);

  REMOTE_CALL(gpu, encoder, decoder);

  apir_buffer_handle_t buffer_handle;
  vn_decode_apir_buffer_handle_t(decoder, &buffer_handle);

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  return buffer_handle;
}
