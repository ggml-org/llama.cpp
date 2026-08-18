#pragma once

#include "ggml-cpu-impl.h"
#include "ggml.h"

// GGML internal header

// Transient "Q8 panel" path for the grid based IQ quants (iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s,
// iq3_xxs, iq3_s, iq4_xs). During a large batch MUL_MAT each thread decodes 8 src0 rows at a time
// into a small per-thread scratch panel (block_iqp_x8, see repack.h) and runs an integer gemm over
// it against all src1 columns, instead of re-running the grid table decode inside
// ggml_vec_dot_iq*_q8_K once per (row, column). The weights stay compressed in the model buffer, so
// there is no resident memory overhead and small batches keep using the ordinary vec_dot path.
//
// Implemented in repack.cpp (next to the panel decode and the gemm kernels).

#ifdef __cplusplus
extern "C" {
#endif

// Smallest src1 batch for which the per-mul_mat decode pays for itself: the crossover vs the
// vec_dot path is N = 20..32 depending on type, 32 is conservative for all of them.
#define GGML_IQP_MIN_BATCH 32

// same idea for MUL_MAT_ID, but per expert: the routed row count of one expert, not the token
// count of the node. Must be >= 1 - the gather layout relies on zero-row experts being skipped.
#define GGML_IQP_MIN_BATCH_ID 16

// is one expert worth the panel path? the gather in repack.cpp and the per expert dispatch in
// ggml-cpu.c must agree on this, the packed gather layout depends on it
static inline bool ggml_cpu_iqp_expert_eligible(int64_t cne1) {
    return cne1 >= GGML_IQP_MIN_BATCH_ID;
}

// is the panel path eligible for this MUL_MAT node?
bool ggml_cpu_iqp_supported_mul_mat(const struct ggml_tensor * dst);

// is the panel path eligible for this MUL_MAT_ID node? this is the node level test only - whether an
// individual expert is worth it is decided per expert with GGML_IQP_MIN_BATCH_ID
bool ggml_cpu_iqp_supported_mul_mat_id(const struct ggml_tensor * dst);

// bytes of src1 converted to the vec_dot type, i.e. the live part of the work buffer the panel
// scratch has to start behind. Same formula ggml_graph_plan and the conversion writer use.
size_t ggml_cpu_iqp_src1_conv_size(const struct ggml_tensor * dst);

// offset of the panel scratch area inside the work buffer (past the q8_K conversion of src1)
size_t ggml_cpu_iqp_scratch_offset(const struct ggml_tensor * dst);

// per thread panel scratch bytes, padded
size_t ggml_cpu_iqp_scratch_size(const struct ggml_tensor * dst);

// must be called after src1 has been converted to q8_K into params->wdata and the threads have
// synchronized on it
void ggml_compute_forward_mul_mat_iqp(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// bytes of the MUL_MAT_ID gather area, where the q8_K rows routed to each eligible expert are
// copied into one contiguous run: worst case every routed pair, i.e. ids->ne[0]*ids->ne[1] rows
size_t ggml_cpu_iqp_id_gather_size(const struct ggml_tensor * dst);

// copy the q8_K rows of every eligible expert into `gathered`, experts in ascending order, packed.
// Call after the row grouping and the q8_K conversion, and synchronize the threads on it before
// the first ggml_compute_forward_mul_mat_id_iqp.
// matrix_rows is the [n_as][ids->ne[0]*ids->ne[1]] table of (i1, i2) int32 pairs built by the caller.
void ggml_cpu_iqp_gather_mul_mat_id(const struct ggml_compute_params * params,
                                    const struct ggml_tensor *         dst,
                                    const int64_t *                    matrix_row_counts,
                                    const int32_t *                    matrix_rows,
                                    void *                             gathered);

// one expert. `gathered` points at that expert's run inside the gather area, `expert_rows` at its
// row of the matrix_rows table, `panels` at the base of the nth per thread panel scratches.
void ggml_compute_forward_mul_mat_id_iqp(const struct ggml_compute_params * params,
                                         struct ggml_tensor *               dst,
                                         int64_t                            cur_a,
                                         int64_t                            cne1,
                                         const int32_t *                    expert_rows,
                                         const void *                       gathered,
                                         void *                             panels);

#ifdef __cplusplus
}
#endif
