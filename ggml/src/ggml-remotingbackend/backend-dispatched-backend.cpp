#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

uint32_t
backend_graph_compute(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(enc);

  uint32_t shmem_res_id;
  vn_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

  const void *shmem_data = ctx->iface.get_shmem_ptr(ctx->virgl_ctx, shmem_res_id);
  if (!shmem_data) {
    FATAL("Couldn't get the shmem addr from virgl :/");
  }
  size_t shmem_size;
  vn_decode_size_t(dec, &shmem_size);

  struct vn_cs_decoder secondary_dec = vn_cs_new_decoder((const char *) shmem_data, shmem_size);

  ggml_cgraph *cgraph = vn_decode_ggml_cgraph(dec, &secondary_dec);

  ggml_status status = bck->iface.graph_compute(bck, cgraph);

  vn_encode_ggml_status(enc, &status);

  return 0;
}
