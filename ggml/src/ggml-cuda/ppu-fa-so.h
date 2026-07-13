#pragma once
// Thin C ABI for the external FlashAttention .so (wraps the flash-attention repo's run_mha_fwd_; cutlass/torch
// stay inside the .so, llama.cpp sees only this declaration -- the cuBLAS model, same as ppu-moe-so.h).
//
// Build libppu_fa.so from flash-attention/csrc/flash_attn/src/*.cu + ppu_so/fa/flash_api_c.cpp: a torch-free shim
// that fills Flash_fwd_params from raw pointers and dispatches run_mha_fwd_<T, headdim, Is_causal>. The .so links
// ONLY the CUDA runtime, NOT libtorch.
//
// LAYOUT via strides (not packed): q/k/v/o all have a CONTIGUOUS head_dim (last axis, stride 1); the batch / head /
// row(=seqlen) strides are passed explicitly in ELEMENTS. This lets the .so consume ggml's native tensor layout
// (ggml Q ne={d,seqlen,head,batch} -> physical [batch,head,seqlen,d], i.e. head/seqlen transposed vs a packed FA
// tensor) with zero repacking. q/k/v/o are half or bf16 (see `dtype`); o has the same dtype as q.
//   q : heads=n_heads_q,  rows=seqlen_q
//   k : heads=n_heads_kv, rows=seqlen_k
//   v : heads=n_heads_kv, rows=seqlen_k
//   o : heads=n_heads_q,  rows=seqlen_q
// GQA: n_heads_q must be a multiple of n_heads_kv.
//
// MASK: FA2's sm80 forward has NO additive-mask input -- attention shape is expressed only via is_causal. The caller
// MUST only engage this path for full (is_causal=0, no mask) or pure-causal (is_causal=1) attention; arbitrary
// additive masks (sliding window, padding, ALiBi, cross-sequence) are NOT representable here -> use the inline path.
//
// Returns 0 on success; non-zero if the .so has no kernel for this (head_dim, dtype) -> caller falls back to inline.
#ifdef __cplusplus
extern "C" {
#endif

// dtype: 0 = fp16, 1 = bf16 (applies to q/k/v/o).
int ppu_flash_attn_fwd(
    const void * q, const void * k, const void * v, void * o,
    int batch, int seqlen_q, int seqlen_k, int n_heads_q, int n_heads_kv, int head_dim,
    // element strides (contiguous head_dim assumed, stride 1):
    long long q_batch_stride, long long q_head_stride, long long q_row_stride,
    long long k_batch_stride, long long k_head_stride, long long k_row_stride,
    long long v_batch_stride, long long v_head_stride, long long v_row_stride,
    long long o_batch_stride, long long o_head_stride, long long o_row_stride,
    float scale, float logit_softcap, int is_causal, int dtype,
    void * stream);

#ifdef __cplusplus
}
#endif
