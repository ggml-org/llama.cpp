#include "virtgpu-forward-impl.h"

void *
apir_buffer_get_base(struct virtgpu *gpu, apir_buffer_context_t *buffer_context) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_GET_BASE);

  vn_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  uintptr_t base;
  vn_decode_uintptr_t(decoder, &base);

  remote_call_finish(gpu, encoder, decoder);

  return (void *) base;
}

void
apir_buffer_set_tensor(struct virtgpu *gpu, apir_buffer_context_t *buffer_context,
		       ggml_tensor *tensor, const void *data, size_t offset, size_t size) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

#if 0
  INFO("Calling (%p)->set_tensor(tensor=%p, data=%p, offset=%lu, size=%lu",
       buffer_context->host_handle, tensor, data, offset, size);
#endif
  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_SET_TENSOR);

  vn_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);
  vn_encode_ggml_tensor(encoder, tensor);

  struct vn_renderer_shmem *shmem;
  if (size > gpu->data_shmem->mmap_size) {
    shmem = virtgpu_shmem_create(gpu, size);
    //WARNING("%s: 0x%lx | %dkB | %dMB", __func__, size, (int)size/1024, (int)size/1024/1024);
    if (!shmem) {
      FATAL("Couldn't allocate the guest-host shared buffer :/");
    }
  } else {
    shmem = gpu->data_shmem;
  }

  memcpy(shmem->mmap_ptr, data, size);
  vn_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

  vn_encode_size_t(encoder, &offset);
  vn_encode_size_t(encoder, &size);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  remote_call_finish(gpu, encoder, decoder);

  if (shmem != gpu->data_shmem) {
    virtgpu_shmem_destroy(gpu, shmem->shmem);
  }

  return;
}

#if false
void
apir_buffer_get_tensor(struct virtgpu *gpu, apir_buffer_context_t *buffer_context,
		       const ggml_tensor *tensor, void *data, size_t offset, size_t size) {
  UNUSED(gpu);
  UNUSED(tensor);
  char *buffer_base_addr = (char *) buffer_context->shmem->mmap_ptr;

  memcpy(data, buffer_base_addr+offset, size);
}
#else
void
apir_buffer_get_tensor(struct virtgpu *gpu, apir_buffer_context_t *buffer_context,
		       const ggml_tensor *tensor, void *data, size_t offset, size_t size) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_GET_TENSOR);

  vn_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);
  vn_encode_ggml_tensor(encoder, tensor);

  struct vn_renderer_shmem *shmem;
  if (size > gpu->data_shmem->mmap_size) {
    shmem = virtgpu_shmem_create(gpu, size);
    WARNING("%s: 0x%lx | %dkB | %dMB", __func__, size, (int)size/1024, (int)size/1024/1024);
    if (!shmem) {
      FATAL("Couldn't allocate the guest-host shared buffer :/");
    }
  } else {
    shmem = gpu->data_shmem;
  }

  vn_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
  vn_encode_size_t(encoder, &offset);
  vn_encode_size_t(encoder, &size);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  memcpy(data, shmem->mmap_ptr, size);

  remote_call_finish(gpu, encoder, decoder);

  if (shmem != gpu->data_shmem) {
    virtgpu_shmem_destroy(gpu, shmem->shmem);
  }
}
#endif

void
apir_buffer_clear(struct virtgpu *gpu, apir_buffer_context_t *buffer_context,
		  uint8_t value) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_CLEAR);

  vn_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);
  vn_encode_uint8_t(encoder, &value);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  remote_call_finish(gpu, encoder, decoder);
}


void
apir_buffer_free_buffer(struct virtgpu *gpu, apir_buffer_context_t *buffer_context) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER);

  vn_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  remote_call_finish(gpu, encoder, decoder);
}
