#include "virtgpu-forward-impl.h"

ggml_status
apir_backend_graph_compute(struct virtgpu *gpu, ggml_cgraph *cgraph) {
  ggml_status status;
  UNUSED(cgraph);

  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE);

  vn_encode_ggml_cgraph(encoder, cgraph);

  REMOTE_CALL(gpu, encoder, decoder);

  vn_decode_ggml_status(decoder, &status);

  REMOTE_CALL_FINISH(gpu, encoder, decoder);

  return status;
}
