#include "virtgpu-forward-impl.h"

int
apir_device_get_count(struct virtgpu *gpu) {
  static int32_t dev_count = -1;
  if (dev_count != -1) {
    CACHED;
    return dev_count;
  }

  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_COUNT);
  REMOTE_CALL(gpu, encoder, decoder, ret);

  vn_decode_int32_t(decoder, &dev_count);

  INFO("%s: Forward DEV COUNT --> %d ", __func__, dev_count);

  remote_call_finish(gpu, encoder, decoder);

  return dev_count;
}

const char *
apir_device_get_name(struct virtgpu *gpu) {
  static char *string = nullptr;
  if (string) {
    CACHED;
    return string;
  }
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
  REMOTE_CALL(gpu, encoder, decoder, ret);

  const size_t string_size = vn_decode_array_size_unchecked(decoder);
  string = (char *) vn_cs_decoder_alloc_array(decoder, sizeof(char), string_size);
  if (!string) {
    FATAL("%s: Could not allocate the device name buffer", __func__);
  }
  vn_decode_char_array(decoder, string, string_size);

  INFO("%s: Forward DEV NAME --> %s", __func__, string);

  remote_call_finish(gpu, encoder, decoder);

  return string;
}

const char *
apir_device_get_description(struct virtgpu *gpu) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  const size_t string_size = vn_decode_array_size_unchecked(decoder);
  char *string = (char *) vn_cs_decoder_alloc_array(decoder, sizeof(char), string_size);
  if (!string) {
    FATAL("%s: Could not allocate the device description buffer", __func__);
  }
  vn_decode_char_array(decoder, string, string_size);

  //INFO("%s: Forward DEV DESCR --> %s", __func__, string);

  remote_call_finish(gpu, encoder, decoder);

  return string;
}

uint32_t
apir_device_get_type(struct virtgpu *gpu) {
  static uint32_t dev_type = 255;
  if (dev_type != 255) {
    CACHED;
    return dev_type;
  }

  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  vn_decode_uint32_t(decoder, &dev_type);

  INFO("%s: Forward DEV TYPE --> %d ", __func__, dev_type);

  remote_call_finish(gpu, encoder, decoder);

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
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_MEMORY);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  vn_decode_size_t(decoder, &dev_free);
  vn_decode_size_t(decoder, &dev_total);

  *free = dev_free;
  *total = dev_total;
  
  //INFO("%s: Forward DEV FREE  mem --> %zu MB", __func__, dev_free / 1024 / 1024);
  //INFO("%s: Forward DEV TOTAL mem --> %zu MB", __func__, dev_total / 1024 / 1024);

  remote_call_finish(gpu, encoder, decoder);

  return;
}

bool
apir_device_supports_op(struct virtgpu *gpu, const ggml_tensor *op) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

  vn_encode_ggml_tensor_inline(encoder, op);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  bool supports_op;
  vn_decode_bool_t(decoder, &supports_op);

  remote_call_finish(gpu, encoder, decoder);

  return supports_op;
}

apir_buffer_type_host_handle_t
apir_device_get_buffer_type(struct virtgpu *gpu) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  apir_buffer_type_host_handle_t buft_handle;
  vn_decode_apir_buffer_type_host_handle_t(decoder, &buft_handle);

  remote_call_finish(gpu, encoder, decoder);

  return buft_handle;
}

void
apir_device_get_props(struct virtgpu *gpu,
		      bool *async,
		      bool *host_buffer,
		      bool *buffer_from_host_ptr,
		      bool *events) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_PROPS);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  vn_decode_bool_t(decoder, async);
  vn_decode_bool_t(decoder, host_buffer);
  vn_decode_bool_t(decoder, buffer_from_host_ptr);
  vn_decode_bool_t(decoder, events);

  /* *** */
  remote_call_finish(gpu, encoder, decoder);

  return;
}

apir_buffer_context_t
apir_device_buffer_from_ptr(struct virtgpu *gpu,
			    size_t size,
			    size_t max_tensor_size) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  apir_buffer_context_t buffer_context;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR);

  /* *** */

  buffer_context.shmem = virtgpu_shmem_create(gpu, size);
  if (!buffer_context.shmem) {
    FATAL("Couldn't allocate the guest-host shared buffer :/");
  }

  vn_encode_virtgpu_shmem_res_id(encoder, buffer_context.shmem->res_id);

  vn_encode_size_t(encoder, &size);
  vn_encode_size_t(encoder, &max_tensor_size);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  vn_decode_apir_buffer_host_handle_t(decoder, &buffer_context.host_handle);
  buffer_context.buft_host_handle = vn_decode_apir_buffer_type_host_handle(decoder);

  /* *** */

  remote_call_finish(gpu, encoder, decoder);

  return buffer_context;
}
