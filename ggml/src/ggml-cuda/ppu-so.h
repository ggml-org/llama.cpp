#pragma once
// dlopen wrapper around the external kernel .so files (libppu_fa.so + libppu_moe.so).
//
// Why dlopen and not link-time: llama.cpp must build with ZERO cutlass / flash-attention / DeepGEMM / torch headers
// or link deps. The heavy kernels are built separately from the submodules under thirdparty/ into standalone .so
// files that hide cutlass inside them; only the .so binaries (+ ppu-fa-so.h / ppu-moe-so.h) are shipped to users,
// never the submodule source. At runtime we dlopen the .so and dlsym the C ABI; if a .so is absent or has no kernel
// for the requested shape, the caller transparently falls back to the inline ggml path.
//
// .so locations are resolved (first hit wins):
//   1. env  GGML_PPU_FA_SO  / GGML_PPU_MOE_SO   (absolute path to the .so)
//   2. bare soname on the loader search path:  libppu_fa.so / libppu_moe.so
//
// Everything here is a no-op (available()==false) unless the build defines GGML_PPU_SO.
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

// ---- FlashAttention (mirror of ppu-fa-so.h's ppu_flash_attn_fwd) ----
bool ggml_ppu_so_fa_available(void);
int  ggml_ppu_so_flash_attn_fwd(
    const void * q, const void * k, const void * v, void * o,
    int batch, int seqlen_q, int seqlen_k, int n_heads_q, int n_heads_kv, int head_dim,
    long long q_batch_stride, long long q_head_stride, long long q_row_stride,
    long long k_batch_stride, long long k_head_stride, long long k_row_stride,
    long long v_batch_stride, long long v_head_stride, long long v_row_stride,
    long long o_batch_stride, long long o_head_stride, long long o_row_stride,
    float scale, float logit_softcap, int is_causal, int dtype, void * stream);

// ---- MoE grouped-GEMM (mirror of ppu-moe-so.h's ppu_moe_grouped_gemm_bf16_nopad) ----
bool ggml_ppu_so_moe_available(void);
// Per-expert row alignment the compact A buffer must satisfy; 0 if the .so is absent.
int  ggml_ppu_so_moe_row_alignment(void);
int  ggml_ppu_so_moe_grouped_gemm_bf16_nopad(
    const void * A, const void * B, void * out, const int * m_indices,
    int total_rows, int N, int K, int n_experts, int expected_m, void * stream);

// ---- gated delta net, recurrent path (mirror of ppu-gdn-so.h's ppu_gdn_recurrent) ----
bool ggml_ppu_so_gdn_available(void);
int  ggml_ppu_so_gdn_recurrent(
    const float * q, const float * k, const float * v, const float * g, const float * beta,
    const float * h0, float * o, float * ht,
    int n_seqs, int T, int H, int HV, int S, float scale, void * stream);
bool ggml_ppu_so_gdn_chunked_available(void);
int  ggml_ppu_so_gdn_chunked(
    const float * q, const float * k, const float * v, const float * g_raw, const float * beta,
    const float * h0, float * o, float * ht,
    int n_seqs, int T, int H, int HV, int S, float scale, void * stream);

#ifdef __cplusplus
}
#endif
