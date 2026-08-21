#ifndef GGML_SYCL_FATTN_ONEDNN_HPP
#define GGML_SYCL_FATTN_ONEDNN_HPP

#include "common.hpp"

// Static-only check: fused-XMX oneDNN Graph SDPA path==flash-attn op
// (f16 KV, no softcap/ALiBi, single stream, tuned head_dim, prefill-sized q.)
bool ggml_sycl_flash_attn_ext_onednn_supported(const ggml_tensor * dst);

// Run flash attention through oneDNN's fused xmx SDPA
// execute the cached SDPA partition, write the f32 dst. Falls back to the TILE kernel on any failure.
void ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_ONEDNN_HPP
