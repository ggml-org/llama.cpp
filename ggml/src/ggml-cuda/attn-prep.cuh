#include "common.cuh"

// Inkling 2, one token: the per-head norms between the attention projections and flash attention as one kernel,
// a warp per head row,
//   MUL(q, wq) -> L2_NORM [-> MUL(., tau)]                          clipped Q
//   L2_NORM(k)                                                     clipped K
//   RMS_NORM(v)                                                    normalized V
//   SET_ROWS(cache_k, K, idx), SET_ROWS(cache_v, V, idx)           both stored in the KV cache (F16)
//   MUL(r, wr) -> L2_NORM -> SCALE [-> MUL(., tau)]                clipped R
// in this graph order (views may sit between them). Returns the index of the last node, or -1.
int  ggml_cuda_match_attn_prep(const ggml_cgraph * cgraph, int i, bool check_memory = true);
void ggml_cuda_op_attn_prep(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last);
