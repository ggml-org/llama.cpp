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

  ggml_cgraph *cgraph = vn_decode_ggml_cgraph(dec);

  ggml_status status = bck->iface.graph_compute(bck, cgraph);

  vn_encode_ggml_status(enc, &status);

  return 0;
}
