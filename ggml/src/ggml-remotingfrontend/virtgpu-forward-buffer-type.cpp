#include "ggml-backend-impl.h"
#include "virtgpu.h"
#include "/Users/kevinpouget/remoting/llama_cpp/src/ggml/src/ggml-remotingbackend/shared/apir_backend.h"
#include "/Users/kevinpouget/remoting/llama_cpp/src/ggml/src/ggml-remotingbackend/shared/venus_cs_ggml.h"

#define CACHED
//  printf("INFO: ### found response in the cache %s\n", __func__)



// buffer_type_alloc_buffer
const char *
apir_buffer_type_get_name(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_BUFFER_TYPE_GET_NAME;

  struct vn_cs_encoder *encoder = remote_call_prepare(gpu, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }

  vn_encode_ggml_buft(encoder, buft);

  struct vn_cs_decoder *decoder = remote_call(gpu, encoder);
  if (!decoder) {
    FATAL("%s: failed to kick the remote call :/", __func__);
  }

  const size_t string_size = vn_decode_array_size_unchecked(decoder);
  char *string = (char *) vn_cs_decoder_alloc_array(decoder, sizeof(char), string_size);
  if (!string) {
    FATAL("%s: Could not allocate the device name buffer", __func__);
  }
  vn_decode_char_array(decoder, string, string_size);

  INFO("%s: Forward BUFT NAME --> %s", __func__, string);

  /* *** */

  int32_t ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret);
  }

  return string;
}

size_t
apir_buffer_type_get_alignment(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  static int32_t dev_count = -1;
  if (dev_count != -1) {
    CACHED;
    return dev_count;
  }
  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_BUFFER_TYPE_GET_ALIGNMENT;
  struct vn_cs_encoder *encoder = remote_call_prepare(gpu, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }

  vn_encode_ggml_buft(encoder, buft);

  struct vn_cs_decoder *decoder = remote_call(gpu, encoder);
  if (!decoder) {
    FATAL("%s: failed to kick the remote call :/", __func__);
  }

  size_t alignment;
  vn_decode_size_t(decoder, &alignment);

  INFO("%s: Forward BUFT ALIGNMENT --> %zu ", __func__, alignment);

  int32_t ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret);
  }

  return alignment;
}

size_t
apir_buffer_type_get_max_size(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_BUFFER_TYPE_GET_MAX_SIZE;
  struct vn_cs_encoder *encoder = remote_call_prepare(gpu, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }

  vn_encode_ggml_buft(encoder, buft);

  struct vn_cs_decoder *decoder = remote_call(gpu, encoder);
  if (!decoder) {
    FATAL("%s: failed to kick the remote call :/", __func__);
  }

  size_t max_size;
  vn_decode_size_t(decoder, &max_size);

  INFO("%s: Forward BUFT MAX SIZE --> %zu ", __func__, max_size);

  int32_t ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret);
  }

  return max_size;
}

bool
apir_buffer_type_is_host(struct virtgpu *gpu, ggml_backend_buffer_type_t buft) {
  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_BUFFER_TYPE_IS_HOST;

  struct vn_cs_encoder *encoder = remote_call_prepare(gpu, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }

  vn_encode_ggml_buft(encoder, buft);

  struct vn_cs_decoder *decoder = remote_call(gpu, encoder);
  if (!decoder) {
    FATAL("%s: failed to kick the remote call :/", __func__);
  }

  bool is_host;
  vn_decode_bool_t(decoder, &is_host);

  /* *** */

  int32_t ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret);
  }

  return is_host;
}
