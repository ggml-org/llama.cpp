#include "virtgpu-forward-impl.h"

ggml_status
apir_backend_graph_compute(struct virtgpu *gpu, ggml_cgraph *cgraph) {
  UNUSED(cgraph);

  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE);

  size_t size = vn_encode_sizeof_ggml_cgraph_data(cgraph);
  struct vn_renderer_shmem *shmem = virtgpu_shmem_create(gpu, size);
  if (!shmem) {
    FATAL("Couldn't allocate the guest-host shared buffer :/");
  }
  INFO("Send shmem ID %d", shmem->res_id);
  vn_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
  INFO("Send shmem size %lu", size);
  vn_encode_size_t(encoder, &size);

  char *shmem_data = (char *) shmem->mmap_ptr;
  struct vn_cs_encoder secondary_enc = vn_cs_new_encoder(shmem_data, size);

  vn_encode_ggml_cgraph(encoder, cgraph, &secondary_enc);

  REMOTE_CALL(gpu, encoder, decoder);

  ggml_status status = GGML_STATUS_ABORTED;
  vn_decode_ggml_status(decoder, &status);
  INFO("Received status %u", status);

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  virtgpu_shmem_destroy(gpu, shmem->shmem);

  return status;
}
