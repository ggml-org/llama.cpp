#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include "shared/apir_backend.h"

struct timer_data graph_compute_timer = {0, 0, 0, "compute_timer"};

uint32_t
backend_graph_compute(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec, struct virgl_apir_context *ctx) {
  UNUSED(ctx);
  UNUSED(enc);

  start_timer(&graph_compute_timer);

  uint32_t shmem_res_id;
  vn_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

  const void *shmem_data = ctx->iface.get_shmem_ptr(ctx->virgl_ctx, shmem_res_id);
  if (!shmem_data) {
    FATAL("Couldn't get the shmem addr from virgl :/");
  }
  size_t cgraph_size;
  vn_decode_size_t(dec, &cgraph_size);

  struct vn_cs_decoder secondary_dec = vn_cs_new_decoder((const char *) shmem_data, cgraph_size);

  ggml_cgraph *cgraph = vn_decode_ggml_cgraph(&secondary_dec, cgraph_size);

  ggml_status status;
  status = bck->iface.graph_compute(bck, cgraph);

  vn_encode_ggml_status(enc, &status);

  stop_timer(&graph_compute_timer);

  return 0;
}
