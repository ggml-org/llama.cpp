#pragma once

#include "ggml-cpu-impl.h"
#include "ggml.h"

// GGML internal header

// batched mul_mat path for the grid based IQ types: decode 8 src0 rows at a time into per thread scratch
// (block_iqp_x8, see iqp.cpp) and run an integer gemm over them against all src1 columns

#ifdef __cplusplus
extern "C" {
#endif

// whether cne1 rows of src1 are enough for the decode to pay for itself, per expert, for MUL_MAT_ID
bool ggml_cpu_iqp_mul_mat_id_min_batch(int64_t cne1);

bool ggml_cpu_iqp_supports_mul_mat(const struct ggml_tensor * dst);

// node level test only - per expert eligibility is decided with ggml_cpu_iqp_mul_mat_id_min_batch
bool ggml_cpu_iqp_supports_mul_mat_id(const struct ggml_tensor * dst);

// per thread panel scratch bytes, padded
size_t ggml_cpu_iqp_scratch_size(const struct ggml_tensor * dst);

// must be called after src1 has been converted to q8_K into params->wdata and the threads have synchronized on it
void ggml_compute_forward_mul_mat_iqp(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// one expert: expert_rows points at its row of the matrix_rows table of (i1, i2) int32 pairs, panels at the base of the per thread panel scratches
void ggml_compute_forward_mul_mat_id_iqp(const struct ggml_compute_params * params,
                                         struct ggml_tensor *               dst,
                                         int64_t                            cur_a,
                                         int64_t                            cne1,
                                         const int32_t *                    expert_rows,
                                         void *                             panels);

#ifdef __cplusplus
}
#endif
