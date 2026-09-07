#pragma once


#ifdef __cplusplus
extern "C" {
#endif

// per-thread workspace size (0 when tiled is disabled)
size_t ggml_tiled_ws_size(void);

// tiled K-quant matmul; returns true if the op was computed here, 
// false to fall through to the stock path
bool ggml_compute_forward_mul_mat_tiled(const struct ggml_compute_params * params,
                                        struct ggml_tensor * dst);

// MUL_MAT_ID (MoE) path: node level gate (per expert eligibility is decided with
// ggml_tiled_mul_mat_id_min_batch), the batch floor and the per expert entry point
bool ggml_tiled_matmul_id_supported(const struct ggml_tensor * dst);

bool ggml_tiled_mul_mat_id_min_batch(int64_t cne1);

// one expert: expert_rows points at its row of the matrix_rows table of (expert slot, batch
// row) int32 pairs; scratch is the per-thread staging region reserved in wdata
// (see ggml_tiled_mul_mat_id_extra_wdata_len)
void ggml_compute_forward_mul_mat_id_tiled(const struct ggml_compute_params * params,
                                           struct ggml_tensor *               dst,
                                           int64_t                            cur_a,
                                           int64_t                            cne1,
                                           const int32_t *                    expert_rows,
                                           char *                             scratch);

// wdata reservation size for the MUL_MAT_ID tiled path, if any: per thread, a ring of
// TILED_TILE_ROWS gathered q8_K rows plus the VNNI interleave region. 0 when disabled
size_t ggml_tiled_mul_mat_id_extra_wdata_len(int64_t ne10, int64_t n_tasks);

#ifdef __cplusplus
}
#endif
