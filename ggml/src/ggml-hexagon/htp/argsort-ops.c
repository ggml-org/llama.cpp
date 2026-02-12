#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <HAP_farf.h>
#include <HAP_perf.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

#include "hvx-utils.h"
#include "hex-dma.h"

#include "htp-ctx.h"
#include "htp-msg.h"
#include "htp-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

struct htp_argsort_context {
    struct htp_ops_context * octx;
    uint32_t                 nrows_per_thread;
};

static inline bool all_greater_f32(HVX_Vector x, HVX_Vector y)
{
    const HVX_Vector one  = Q6_V_vsplat_R(1);
    const HVX_Vector zero = Q6_V_vzero();

    HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(x, y);
    HVX_Vector matches = Q6_V_vmux_QVV(pred, one, zero);
    HVX_Vector sum = hvx_vec_reduce_sum_i32(matches);
    return hvx_vec_get_i32(sum) == 32;
}

// Sorts values and mirrors swaps to indices.
static void quicksort_values_indices_asc(float * values, int32_t * indices, int left, int right) {
    if (left >= right) return;

    int pivot_idx = (left + right) / 2;
    float pivot = values[pivot_idx];
    int i = left;
    int j = right;

    HVX_Vector pivot_vec = hvx_vec_splat_f32(pivot);
    while (i <= j) {
        // Vectorized scan for i
        while (i <= j) {
            // Check if we have at least one full vector
            if (i + 32 <= j) {
                HVX_Vector vals_vec = *(HVX_UVector *)(values + i);
                if (all_greater_f32(pivot_vec, vals_vec)) {
                    // If all elements are < pivot, we can skip this whole block
                    i += 32;
                    continue;
                }
            }

            // Scalar fallback / cleanup
            if (values[i] < pivot) {
                i++;
            } else {
                break;
            }
        }

        // Vectorized scan for j
        while (i <= j) {
            if (j - 32 >= i) {
                // Load 32 elements ending at j.
                // Since we want `values[j] > pivot`, let's load from j-31 to j.
                HVX_Vector vals_vec = *(HVX_UVector *)(values + j - 31);
                if (all_greater_f32(vals_vec, pivot_vec)) {
                    j -= 32;
                    continue;
                }
            }

            if (values[j] > pivot) {
                j--;
            } else {
                break;
            }
        }

        if (i <= j) {
            float tmp_val = values[i];
            values[i] = values[j];
            values[j] = tmp_val;

            int32_t tmp_idx = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp_idx;
            i++;
            j--;
        }
    }

    if (left < j) quicksort_values_indices_asc(values, indices, left, j);
    if (i < right) quicksort_values_indices_asc(values, indices, i, right);
}

static void quicksort_values_indices_desc(float * values, int32_t * indices, int left, int right) {
    if (left >= right) return;

    int pivot_idx = (left + right) / 2;
    float pivot = values[pivot_idx];
    int i = left;
    int j = right;

    HVX_Vector pivot_vec = hvx_vec_splat_f32(pivot);

    while (i <= j) {
        // Vectorized scan for i (values[i] > pivot)
        while (i <= j) {
            if (i + 32 <= j) {
                HVX_Vector vals_vec = *(HVX_UVector *)(values + i);
                if (all_greater_f32(vals_vec, pivot_vec)) {
                    i += 32;
                    continue;
                }
            }

            if (values[i] > pivot) {
                i++;
            } else {
                break;
            }
        }

        // Vectorized scan for j (values[j] < pivot)
        while (i <= j) {
            if (j - 32 >= i) {
                HVX_Vector vals_vec = *(HVX_UVector *)(values + j - 31);
                if (all_greater_f32(pivot_vec, vals_vec)) {
                    j -= 32;
                    continue;
                }
            }

            if (values[j] < pivot) {
                j--;
            } else {
                break;
            }
        }

        if (i <= j) {
            float tmp_val = values[i];
            values[i] = values[j];
            values[j] = tmp_val;

            int32_t tmp_idx = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp_idx;
            i++;
            j--;
        }
    }

    if (left < j) quicksort_values_indices_desc(values, indices, left, j);
    if (i < right) quicksort_values_indices_desc(values, indices, i, right);
}

static void htp_argsort_f32(unsigned int n, unsigned int i, void * data) {
    struct htp_argsort_context * actx = (struct htp_argsort_context *)data;
    struct htp_ops_context * octx = actx->octx;

    // Unpack context
    const struct htp_tensor * src0 = &octx->src0;
    const struct htp_tensor * dst = &octx->dst;

    // Scratchpad memory
    uint8_t * spad = octx->src0_spad.data + octx->src0_spad.size_per_thread * i;

    // Dimensions
    uint32_t ne00 = src0->ne[0];
    uint32_t ne01 = src0->ne[1];
    uint32_t ne02 = src0->ne[2];
    uint32_t ne03 = src0->ne[3];

    uint32_t nb01 = src0->nb[1];
    //uint32_t nb02 = src0->nb[2];
    //uint32_t nb03 = src0->nb[3];

    uint32_t nb1 = dst->nb[1];
    //uint32_t nb2 = dst->nb[2];
    //uint32_t nb3 = dst->nb[3];

    // Sort order
    enum ggml_sort_order order = (enum ggml_sort_order) octx->op_params[0];

    // Rows to process
    uint32_t total_rows = ne01 * ne02 * ne03;
    uint32_t rows_per_thread = actx->nrows_per_thread;
    uint32_t start_row = rows_per_thread * i;
    uint32_t end_row = MIN(start_row + rows_per_thread, total_rows);

    // Scratchpad layout:
    // We need space for one row of float data (values) and one row of int32 indices.
    // values: ne00 * sizeof(float)
    // indices: ne00 * sizeof(int32_t)
    // Padded to 128 bytes.

    size_t values_size = hex_round_up(ne00 * sizeof(float), 128);
    float * values_buf = (float *) spad;
    int32_t * indices_buf = (int32_t *) (spad + values_size);

    for (uint32_t r = start_row; r < end_row; r++) {
        uint32_t src_offset = r * nb01;
        uint32_t dst_offset = r * nb1;

        uint8_t * src_ptr = (uint8_t *) src0->data + src_offset;
        uint8_t * dst_ptr = (uint8_t *) dst->data  + dst_offset;

        hex_l2fetch(src_ptr, ne00 * sizeof(float), ne00 * sizeof(float), 1);
        hvx_copy_f32_au((uint8_t*)values_buf, src_ptr, ne00);

        // Initialize indices
        for (uint32_t j = 0; j < ne00; j++) {
            indices_buf[j] = j;
        }

        // Sort values and mirror swaps to indices
        if (order == GGML_SORT_ORDER_ASC) {
            quicksort_values_indices_asc(values_buf, indices_buf, 0, ne00 - 1);
        } else {
            quicksort_values_indices_desc(values_buf, indices_buf, 0, ne00 - 1);
        }

        // Copy indices back to DDR
        hvx_copy_f32_ua(dst_ptr, (const uint8_t *) indices_buf, ne00);
    }
}

int op_argsort(struct htp_ops_context * octx) {
    // Check supported types
    if (octx->src0.type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // Allocate scratchpad
    // We need 1 row of float + 1 row of int32 per thread.
    uint32_t ne00 = octx->src0.ne[0];
    size_t values_size  = hex_round_up(ne00 * sizeof(float), 128);
    size_t indices_size = hex_round_up(ne00 * sizeof(int32_t), 128);
    size_t spad_per_thread = values_size + indices_size;

    // Make sure we round up to 256 for alignment requirements
    spad_per_thread = hex_round_up(spad_per_thread, 256);

    size_t total_spad_size = spad_per_thread * octx->n_threads;

    if (octx->ctx->vtcm_size < total_spad_size) {
        FARF(ERROR, "argsort: VTCM size too small. Needed %zu, have %zu", total_spad_size, octx->ctx->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->src0_spad.size = total_spad_size;
    octx->src0_spad.size_per_thread = spad_per_thread;

    FARF(HIGH, "argsort: %ux%ux%ux%u -> %ux%ux%ux%u (0x%x, 0x%x)",
         octx->src0.ne[0], octx->src0.ne[1], octx->src0.ne[2], octx->src0.ne[3],
         octx->dst.ne[0], octx->dst.ne[1], octx->dst.ne[2], octx->dst.ne[3],
         octx->src0.data, octx->dst.data);

    uint32_t total_rows = octx->src0.ne[1] * octx->src0.ne[2] * octx->src0.ne[3];
    uint32_t n_jobs = MIN(total_rows, octx->n_threads);

    struct htp_argsort_context actx;
    actx.octx = octx;
    actx.nrows_per_thread = (total_rows + n_jobs - 1) / n_jobs;

    // Run jobs
    worker_pool_run_func(octx->ctx->worker_pool, htp_argsort_f32, &actx, n_jobs);

    return HTP_STATUS_OK;
}
