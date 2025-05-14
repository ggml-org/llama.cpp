#include "ggml-backend-impl.h"
#include "virtgpu.h"
#include "/Users/kevinpouget/remoting/llama_cpp/src/ggml/src/ggml-remotingbackend/shared/apir_backend.h"
#include "/Users/kevinpouget/remoting/llama_cpp/src/ggml/src/ggml-remotingbackend/shared/venus_cs_ggml.h"

#define CACHED
//  printf("INFO: ### found response in the cache %s\n", __func__)

int
apir_device_get_count(struct virtgpu *gpu) {
  static int32_t dev_count = -1;
  if (dev_count != -1) {
    CACHED;
    return dev_count;
  }
  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_DEVICE_GET_COUNT;
  struct vn_cs_encoder *encoder = remote_call_prepare(gpu, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }

  struct vn_cs_decoder *decoder = remote_call(gpu, encoder);
  if (!decoder) {
    FATAL("%s: failed to kick the remote call :/", __func__);
  }

  vn_decode_int32_t(decoder, &dev_count);

  INFO("%s: Forward DEV COUNT --> %d ", __func__, dev_count);

  int32_t ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret);
  }

  return dev_count;
}

const char *
apir_device_get_name(struct virtgpu *gpu) {
  static int32_t dev_count = -1;
  if (dev_count != -1) {
    CACHED;
    return "Nothing";
  }

  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_DEVICE_GET_NAME;
  struct vn_cs_encoder *encoder = remote_call_prepare(gpu, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }

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

  INFO("%s: Forward DEV NAME --> %s", __func__, string);

  int32_t ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret);
  }

  return string;
}

const char *
apir_device_get_description(struct virtgpu *gpu) {
  static int32_t dev_count = -1;
  if (dev_count != -1) {
    CACHED;
    return "Nothing";
  }
  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION;
  struct vn_cs_encoder *encoder = remote_call_prepare(gpu, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }

  struct vn_cs_decoder *decoder = remote_call(gpu, encoder);
  if (!decoder) {
    FATAL("%s: failed to kick the remote call :/", __func__);
  }

  const size_t string_size = vn_decode_array_size_unchecked(decoder);
  char *string = (char *) vn_cs_decoder_alloc_array(decoder, sizeof(char), string_size);
  if (!string) {
    FATAL("%s: Could not allocate the device description buffer", __func__);
  }
  vn_decode_char_array(decoder, string, string_size);

  INFO("%s: Forward DEV DESCR --> %s", __func__, string);

  int32_t ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret);
  }

  return string;
}

uint32_t
apir_device_get_type(struct virtgpu *gpu) {
  static uint32_t dev_type = 255;
  if (dev_type != 255) {
    CACHED;
    return dev_type;
  }
  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_DEVICE_GET_TYPE;

  struct vn_cs_encoder *encoder = remote_call_prepare(gpu, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }

  struct vn_cs_decoder *decoder = remote_call(gpu, encoder);
  if (!decoder) {
    FATAL("%s: failed to kick the remote call :/", __func__);
  }

  vn_decode_uint32_t(decoder, &dev_type);

  INFO("%s: Forward DEV TYPE --> %d ", __func__, dev_type);

  int32_t ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret);
  }

  return dev_type;
}

void
apir_device_get_memory(struct virtgpu *gpu, size_t *free, size_t *total) {
  static size_t dev_free = 0;
  static size_t dev_total = 0;
  /*
  if (dev_total != 0) {
    WARNING("Not sure if llama.cpp expects fresh information for the free memory ...");
    *free = dev_free;
    *total = dev_total;

    CACHED;
    return;
  }
  */
  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_DEVICE_GET_MEMORY;

  struct vn_cs_encoder *encoder = remote_call_prepare(gpu, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }

  struct vn_cs_decoder *decoder = remote_call(gpu, encoder);
  if (!decoder) {
    FATAL("%s: failed to kick the remote call :/", __func__);
  }

  vn_decode_size_t(decoder, &dev_free);
  vn_decode_size_t(decoder, &dev_total);

  *free = dev_free;
  *total = dev_total;

  INFO("%s: Forward DEV FREE  mem --> %zu MB", __func__, dev_free / 1024 / 1024);
  INFO("%s: Forward DEV TOTAL mem --> %zu MB", __func__, dev_total / 1024 / 1024);

  int32_t ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret);
  }

  return;
}

bool
apir_device_supports_op(struct virtgpu *gpu, const ggml_tensor *op) {
  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP;

  struct vn_cs_encoder *encoder = remote_call_prepare(gpu, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }

  vn_encode_ggml_tensor(encoder, op);

  struct vn_cs_decoder *decoder = remote_call(gpu, encoder);
  if (!decoder) {
    FATAL("%s: failed to kick the remote call :/", __func__);
  }

  bool supports_op;
  vn_decode_bool_t(decoder, &supports_op);

  /* *** */

  int32_t ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret);
  }

  return supports_op;
}

apir_buffer_type_context_t
apir_device_get_buffer_type(struct virtgpu *gpu) {
  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE;

  struct vn_cs_encoder *encoder = remote_call_prepare(gpu, VIRGL_APIR_COMMAND_TYPE_Forward, forward_flag);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }

  struct vn_cs_decoder *decoder = remote_call(gpu, encoder);
  if (!decoder) {
    FATAL("%s: failed to kick the remote call :/", __func__);
  }

  apir_buffer_type_context_t buffer_type_ctx;
  vn_decode_apir_buffer_type_context_t(decoder, &buffer_type_ctx);

  /* *** */

  int32_t ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to forward the API call (code=%d):/", __func__, ret);
  }

  return buffer_type_ctx;
}
