#include "virtgpu-forward-impl.h"

static long long current_time_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
  return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_status
apir_backend_graph_compute(struct virtgpu *gpu, ggml_cgraph *cgraph) {
  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE);

  std::vector<uint8_t> cgraph_data;
  size_t cgraph_size = vn_serialize_ggml_cgraph(cgraph, cgraph_data);

  struct vn_renderer_shmem *shmem;
  if (cgraph_size > gpu->data_shmem->mmap_size) {
    shmem = virtgpu_shmem_create(gpu, cgraph_size);
    WARNING("%s: 0x%lx | %dkB | %dMB", __func__, cgraph_size, (int)cgraph_size/1024, (int)cgraph_size/1024/1024);
    if (!shmem) {
      FATAL("Couldn't allocate the guest-host shared buffer :/");
    }
  } else {
    shmem = gpu->data_shmem;
  }

  //INFO("Send shmem ID %d", shmem->res_id);
  vn_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
  //INFO("Send shmem size %lu", cgraph_size);
  vn_encode_size_t(encoder, &cgraph_size);

  char *shmem_data = (char *) shmem->mmap_ptr;
  struct vn_cs_encoder secondary_enc = vn_cs_new_encoder(shmem_data, cgraph_size);

  vn_encode_cgraph_data(&secondary_enc, cgraph_data);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  ggml_status status = GGML_STATUS_ABORTED;
  vn_decode_ggml_status(decoder, &status);
  //INFO("Received status %u", status);

  remote_call_finish(gpu, encoder, decoder);

  if (shmem != gpu->data_shmem) {
    virtgpu_shmem_destroy(gpu, shmem->shmem);
  }

  return status;
}
