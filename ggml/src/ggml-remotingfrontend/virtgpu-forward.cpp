#include "virtgpu.h"
#include "/Users/kevinpouget/remoting/llama_cpp/src/ggml/src/ggml-remotingbackend/shared/apir_backend.h"

#define CACHED \
  printf("INFO: ### found response in the cache %s\n", __func__)

int
apir_get_device_count(struct virtgpu *gpu) {
  static int32_t dev_count = -1;
  if (dev_count != -1) {
    CACHED;
    return dev_count;
  }
  int32_t forward_flag = (int32_t) APIR_COMMAND_TYPE_GET_DEVICE_COUNT;
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
