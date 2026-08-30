#pragma once


#ifdef __cplusplus
extern "C" {
#endif

// Extra wdata reservation size required, if any (see VNNI impl)
size_t ggml_tiled_extra_wdata_len(int64_t ne10, int64_t nr1);

// tiled K-quant matmul; returns true if the op was computed here, 
// false to fall through to the stock path
bool ggml_compute_forward_mul_mat_tiled(const struct ggml_compute_params * params,
                                        struct ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
