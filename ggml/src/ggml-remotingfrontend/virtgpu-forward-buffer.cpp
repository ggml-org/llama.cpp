#include "virtgpu-forward-impl.h"

void *
apir_buffer_get_base(struct virtgpu *gpu, apir_buffer_handle_t buffer_handle) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_GET_BASE);

  vn_encode_apir_buffer_handle_t(encoder, &buffer_handle);

  REMOTE_CALL(gpu, encoder, decoder);

  uintptr_t base;
  vn_decode_uintptr_t(decoder, &base);

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  //INFO("%s: received base %p\n", __func__,  (void *) base);

  return (void *) base;
}

void
apir_buffer_set_tensor(struct virtgpu *gpu, apir_buffer_handle_t buffer_handle,
		       ggml_tensor *tensor, const void *data, size_t offset, size_t size) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
#if 0
  INFO("Calling (%p)->set_tensor(tensor=%p, data=%p, offset=%lu, size=%lu",
    buffer_handle, tensor, data, offset, size);
#endif
  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_SET_TENSOR);

  vn_encode_apir_buffer_handle_t(encoder, &buffer_handle);
  vn_encode_ggml_tensor(encoder, tensor, TENSOR_MAX_DEPTH_BUFFER_SET_TENSOR);

  struct vn_renderer_shmem *shmem = virtgpu_shmem_create(gpu, size);
  if (!shmem) {
    FATAL("Couldn't allocate the guest-host shared buffer :/");
  }

  memcpy(shmem->mmap_ptr, data, size);
  vn_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

  vn_encode_size_t(encoder, &offset);
  vn_encode_size_t(encoder, &size);

  REMOTE_CALL(gpu, encoder, decoder);

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  virtgpu_shmem_destroy(gpu, shmem->shmem);

  return;
}

void
apir_buffer_get_tensor(struct virtgpu *gpu, apir_buffer_handle_t buffer_handle,
		       const ggml_tensor *tensor, void *data, size_t offset, size_t size) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_GET_TENSOR);

  vn_encode_apir_buffer_handle_t(encoder, &buffer_handle);
  vn_encode_ggml_tensor(encoder, tensor, TENSOR_MAX_DEPTH_BUFFER_GET_TENSOR);
  struct vn_renderer_shmem *shmem = virtgpu_shmem_create(gpu, size);
  if (!shmem) {
    FATAL("Couldn't allocate the guest-host shared buffer :/");
  }
  vn_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
  vn_encode_size_t(encoder, &offset);
  vn_encode_size_t(encoder, &size);

  REMOTE_CALL(gpu, encoder, decoder);

  memcpy(data, shmem->mmap_ptr, size);

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  virtgpu_shmem_destroy(gpu, shmem->shmem);
}

void
apir_buffer_clear(struct virtgpu *gpu, apir_buffer_handle_t buffer_handle,
		  uint8_t value) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_CLEAR);

  vn_encode_apir_buffer_handle_t(encoder, &buffer_handle);
  vn_encode_uint8_t(encoder, &value);

  REMOTE_CALL(gpu, encoder, decoder);

  REMOTE_CALL_FINISH(gpu, encoder, decoder);
}
