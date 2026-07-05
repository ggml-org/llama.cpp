#ifndef GGML_SYCL_FATTN_XMX_HPP
#define GGML_SYCL_FATTN_XMX_HPP

#include "common.hpp"

// XMX (Intel Xe Matrix Extensions / DPAS) flash-attention path.
//
// Status: SCAFFOLD. The XMX FA math (fused online softmax + QK^T/PV on DPAS at
// sub-group 8) is validated standalone (see docs/research/xmx_fa_*.cpp). This is
// the first ggml-sycl integration cut: it handles the f16 K/V, D=128, causal
// (no explicit mask), GQA-packed decode case and GGML_ABORTs otherwise. It is
// OFF unless GGML_SYCL_FA_XMX is set. q8_0 / turbo dequant-into-tile, explicit
// masks, ALiBi, D in {256,512}, and prefill are follow-ups.
//
// Key constraint: DG2 XMX requires sub-group 8 (not the backend's WARP_SIZE 16);
// forcing 16 triggers an IGC ICE (see docs/backend/SYCL.md Known Issues).
void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_XMX_HPP
