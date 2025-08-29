#include "virtgpu-forward-impl.h"

const char *
apir_buffer_type_get_name(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME);

  vn_encode_ggml_buffer_type(encoder, buft);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  const size_t string_size = vn_decode_array_size_unchecked(decoder);
  char *string = (char *) vn_cs_decoder_alloc_array(decoder, sizeof(char), string_size);
  if (!string) {
    FATAL("%s: Could not allocate the device name buffer", __func__);
  }
  vn_decode_char_array(decoder, string, string_size);

  //INFO("%s: Forward BUFT NAME --> %s", __func__, string);

  /* *** */

  remote_call_finish(gpu, encoder, decoder);

  return string;
}

size_t
apir_buffer_type_get_alignment(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT);

  vn_encode_ggml_buffer_type(encoder, buft);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  size_t alignment;
  vn_decode_size_t(decoder, &alignment);

  INFO("%s: Forward BUFT ALIGNMENT --> %zu ", __func__, alignment);

  remote_call_finish(gpu, encoder, decoder);

  return alignment;
}

size_t
apir_buffer_type_get_max_size(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE);

  vn_encode_ggml_buffer_type(encoder, buft);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  size_t max_size;
  vn_decode_size_t(decoder, &max_size);

  INFO("%s: Forward BUFT MAX SIZE --> %zu ", __func__, max_size);

  remote_call_finish(gpu, encoder, decoder);

  return max_size;
}

bool
apir_buffer_type_is_host(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST);

  vn_encode_ggml_buffer_type(encoder, buft);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  bool is_host;
  vn_decode_bool_t(decoder, &is_host);

  INFO("%s: buffer is host? %d", __func__, is_host);

  remote_call_finish(gpu, encoder, decoder);

  return is_host;
}

apir_buffer_context_t
apir_buffer_type_alloc_buffer(struct virtgpu *gpu, ggml_backend_buffer_type_t buft, size_t size) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  apir_buffer_context_t buffer_context;
  INFO("%s: allocate device memory (%lu)", __func__,  size);

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_TYPE_ALLOC_BUFFER);

  vn_encode_ggml_buffer_type(encoder, buft);

  vn_encode_size_t(encoder, &size);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  vn_decode_apir_buffer_host_handle_t(decoder, &buffer_context.host_handle);

  remote_call_finish(gpu, encoder, decoder);

  return buffer_context;
}
