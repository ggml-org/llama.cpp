#ifndef GGML_SYCL_FATTN_XMX_HPP
#define GGML_SYCL_FATTN_XMX_HPP

#include "common.hpp"

// XMX (Intel Xe Matrix Extensions / DPAS) flash-attention path.
//
// Status: first ggml-sycl integration cut, correct but not yet performance
// competitive (benched slower than VEC -- see docs/research), so OFF unless
// GGML_SYCL_FA_XMX is set. The FA math (fused online softmax + QK^T/PV on DPAS
// at sub-group 8) is validated standalone (docs/research/xmx_fa_*.cpp) and via
// the CPU-vs-SYCL oracle harness.
//
// Supported: KV kinds f16 / q8_0 / turbo2_0 / turbo3_0 / turbo4_0 (K and V must
// share a kind), head dim D in {128, 256}, decode and prefill (tiled over query
// blocks), an optional additive f16 mask (ne[2] == 1). NOT supported (routed to
// VEC/TILE by the fattn.cpp gate): D=512 (exceeds 64KB SLM), ALiBi (max_bias),
// logit soft-capping, attention sinks (src[4]), and multi-sequence ne[3] > 1.
//
// Key constraint: DG2 XMX requires sub-group 8 (not the backend's WARP_SIZE 16);
// forcing 16 triggers an IGC ICE (see docs/backend/SYCL.md Known Issues).
void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_XMX_HPP
