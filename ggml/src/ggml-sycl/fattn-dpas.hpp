#ifndef GGML_SYCL_FATTN_DPAS_HPP
#define GGML_SYCL_FATTN_DPAS_HPP

#include "common.hpp"

// q4_0 / q8_0 K/V flash attention for decode and small batches on Xe2 XMX (s8 x s4 / s8 x s8 DPAS)
bool ggml_sycl_flash_attn_ext_dpas_supported(int device, const ggml_tensor * dst);
void ggml_sycl_flash_attn_ext_dpas(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_DPAS_HPP
