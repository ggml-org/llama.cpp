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
#include "htp-ops.h"
#include "htp-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

struct htp_argsort_context {
    struct htp_ops_context * octx;
    uint32_t                 nrows_per_thread;
    uint8_t *                vtcm_base;
    size_t                   vtcm_per_thread;
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

// LUT for ramp initialization of argsort output (first 32 members)
int32_t argosrt_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

__attribute__((always_inline))
static inline void vec_cas(HVX_Vector * X_val, HVX_Vector * X_idx, HVX_Vector * Y_val, HVX_Vector * Y_idx, bool asc) {
    HVX_VectorPred pred = asc ? Q6_Q_vcmp_gt_VsfVsf(*X_val, *Y_val)
                              : Q6_Q_vcmp_gt_VsfVsf(*Y_val, *X_val);
    HVX_Vector next_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    HVX_Vector next_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    HVX_Vector next_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    HVX_Vector Y_tmp_idx  = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = next_X_val;
    *Y_val = next_Y_val;
    *X_idx = next_X_idx;
    *Y_idx = Y_tmp_idx;
}

__attribute__((always_inline))
static inline void bitonic_cas_32(HVX_Vector * V, HVX_Vector * I, int d, HVX_VectorPred dir_mask, HVX_Vector idx_vec, HVX_Vector zero_vec) {
    HVX_VectorPred mask_left;
    HVX_Vector V_rot_left, V_rot_right;
    HVX_Vector I_rot_left, I_rot_right;

    if (d == 1) {
        mask_left = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_left = Q6_V_vror_VR(*V, 4);
        V_rot_right = Q6_V_vror_VR(*V, 124);
        I_rot_left = Q6_V_vror_VR(*I, 4);
        I_rot_right = Q6_V_vror_VR(*I, 124);
    } else if (d == 2) {
        mask_left = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_left = Q6_V_vror_VR(*V, 8);
        V_rot_right = Q6_V_vror_VR(*V, 120);
        I_rot_left = Q6_V_vror_VR(*I, 8);
        I_rot_right = Q6_V_vror_VR(*I, 120);
    } else if (d == 4) {
        mask_left = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_left = Q6_V_vror_VR(*V, 16);
        V_rot_right = Q6_V_vror_VR(*V, 112);
        I_rot_left = Q6_V_vror_VR(*I, 16);
        I_rot_right = Q6_V_vror_VR(*I, 112);
    } else if (d == 8) {
        mask_left = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_left = Q6_V_vror_VR(*V, 32);
        V_rot_right = Q6_V_vror_VR(*V, 96);
        I_rot_left = Q6_V_vror_VR(*I, 32);
        I_rot_right = Q6_V_vror_VR(*I, 96);
    } else { // d == 16
        mask_left = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_left = Q6_V_vror_VR(*V, 64);
        V_rot_right = Q6_V_vror_VR(*V, 64);
        I_rot_left = Q6_V_vror_VR(*I, 64);
        I_rot_right = Q6_V_vror_VR(*I, 64);
    }

    HVX_Vector V_paired = Q6_V_vmux_QVV(mask_left, V_rot_left, V_rot_right);
    HVX_Vector I_paired = Q6_V_vmux_QVV(mask_left, I_rot_left, I_rot_right);

    HVX_VectorPred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_VectorPred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_VectorPred mask_right = Q6_Q_not_Q(mask_left);
    HVX_VectorPred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(mask_left, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, mask_right)
    );
    HVX_VectorPred Q_swap = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_mask, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_mask), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
}

__attribute__((always_inline))
static inline void bitonic_sort_generic_hvx(uint8_t * values, uint8_t * indices, int K, bool asc_order) {
    HVX_Vector V[32];
    HVX_Vector I[32];

    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(HVX_Vector *)argosrt_ramp_lut;

    // Load values and initialize indices
    for (int v = 0; v < K; v++) {
        V[v] = *(HVX_Vector *)(values + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    int M = 5;
    while ((1 << (M - 5)) < K) M++;

    for (int s = 1; s <= M; s++) {
        for (int stage_d = s - 1; stage_d >= 0; stage_d--) {
            int d = 1 << stage_d;
            if (d >= 32) {
                int v_dist = d / 32;
                for (int v1 = 0; v1 < K; v1++) {
                    if ((v1 & v_dist) == 0) {
                        int v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_order;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } else {
                if (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    for (int v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                } else {
                    for (int v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_order;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    // Write back sorted values and indices
    for (int v = 0; v < K; v++) {
        *(HVX_Vector *)(values + v * 128)  = V[v];
        *(HVX_Vector *)(indices + v * 128) = I[v];
    }
}

__attribute__((always_inline))
static inline void sort32_f32_hvx(uint8_t * values, uint8_t * indices, enum ggml_sort_order order) {
    bitonic_sort_generic_hvx(values, indices, 1, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
static inline void sort64_f32_hvx(uint8_t * values, uint8_t * indices, enum ggml_sort_order order) {
    bitonic_sort_generic_hvx(values, indices, 2, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
static inline void sort128_f32_hvx(uint8_t * values, uint8_t * indices, enum ggml_sort_order order) {
    bitonic_sort_generic_hvx(values, indices, 4, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
static inline void sort256_f32_hvx(uint8_t * values, uint8_t * indices, enum ggml_sort_order order) {
    bitonic_sort_generic_hvx(values, indices, 8, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
static inline void sort512_f32_hvx(uint8_t * values, uint8_t * indices, enum ggml_sort_order order) {
    bitonic_sort_generic_hvx(values, indices, 16, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
static inline void sort1024_f32_hvx(uint8_t * values, uint8_t * indices, enum ggml_sort_order order) {
    bitonic_sort_generic_hvx(values, indices, 32, order == GGML_SORT_ORDER_ASC);
}

#define HTP_ARGSORT_FN(ne00, order_name, order_enum, sort_fn)                                                  \
static void htp_argsort_f32_##ne00##_##order_name(unsigned int n, unsigned int i, void * data) {               \
    struct htp_argsort_context * actx = (struct htp_argsort_context *)data;                                    \
    struct htp_ops_context * octx = actx->octx;                                                                \
    const struct htp_tensor * src0 = octx->src[0];                                                             \
    const struct htp_tensor * dst = octx->dst;                                                                 \
    uint8_t * spad = actx->vtcm_base + actx->vtcm_per_thread * i;                                              \
    uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];                                             \
    uint32_t rows_per_thread = actx->nrows_per_thread;                                                         \
    uint32_t start_row = rows_per_thread * i;                                                                  \
    uint32_t end_row = MIN(start_row + rows_per_thread, total_rows);                                           \
    size_t values_size = hex_round_up(ne00 * sizeof(float), 128);                                              \
    float * values_buf = (float *) spad;                                                                       \
    int32_t * indices_buf = (int32_t *) (spad + values_size);                                                  \
    uint32_t nb01 = src0->nb[1];                                                                               \
    uint32_t nb1 = dst->nb[1];                                                                                 \
    struct htp_thread_trace * tr = &octx->ctx->trace[i];                                                       \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, start_row);                                              \
    for (uint32_t r = start_row; r < end_row; r++) {                                                           \
        uint32_t src_offset = r * nb01;                                                                        \
        uint32_t dst_offset = r * nb1;                                                                         \
        uint8_t * src_ptr = (uint8_t *) src0->data + src_offset;                                               \
        uint8_t * dst_ptr = (uint8_t *) dst->data  + dst_offset;                                               \
        hex_l2fetch(src_ptr, ne00 * sizeof(float), ne00 * sizeof(float), 1);                                   \
        hvx_copy_f32_au((uint8_t*)values_buf, src_ptr, ne00);                                                  \
        sort_fn((uint8_t*)values_buf, (uint8_t*)indices_buf, order_enum);                                      \
        hvx_copy_f32_ua(dst_ptr, (const uint8_t *) indices_buf, ne00);                                         \
    }                                                                                                          \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, start_row);                                               \
}

HTP_ARGSORT_FN(32,   asc, GGML_SORT_ORDER_ASC,  sort32_f32_hvx)
HTP_ARGSORT_FN(32,   dsc, GGML_SORT_ORDER_DESC, sort32_f32_hvx)
HTP_ARGSORT_FN(64,   asc, GGML_SORT_ORDER_ASC,  sort64_f32_hvx)
HTP_ARGSORT_FN(64,   dsc, GGML_SORT_ORDER_DESC, sort64_f32_hvx)
HTP_ARGSORT_FN(128,  asc, GGML_SORT_ORDER_ASC,  sort128_f32_hvx)
HTP_ARGSORT_FN(128,  dsc, GGML_SORT_ORDER_DESC, sort128_f32_hvx)
HTP_ARGSORT_FN(256,  asc, GGML_SORT_ORDER_ASC,  sort256_f32_hvx)
HTP_ARGSORT_FN(256,  dsc, GGML_SORT_ORDER_DESC, sort256_f32_hvx)
HTP_ARGSORT_FN(512,  asc, GGML_SORT_ORDER_ASC,  sort512_f32_hvx)
HTP_ARGSORT_FN(512,  dsc, GGML_SORT_ORDER_DESC, sort512_f32_hvx)
HTP_ARGSORT_FN(1024, asc, GGML_SORT_ORDER_ASC,  sort1024_f32_hvx)
HTP_ARGSORT_FN(1024, dsc, GGML_SORT_ORDER_DESC, sort1024_f32_hvx)

static void htp_argsort_f32_fallback(unsigned int n, unsigned int i, void * data) {
    struct htp_argsort_context * actx = (struct htp_argsort_context *)data;
    struct htp_ops_context * octx = actx->octx;

    // Unpack context
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    // Scratchpad memory
    uint8_t * spad = actx->vtcm_base + actx->vtcm_per_thread * i;

    // Dimensions
    uint32_t ne00 = src0->ne[0];
    uint32_t ne01 = src0->ne[1];
    uint32_t ne02 = src0->ne[2];
    uint32_t ne03 = src0->ne[3];

    uint32_t nb01 = src0->nb[1];

    uint32_t nb1 = dst->nb[1];

    // Sort order
    enum ggml_sort_order order = (enum ggml_sort_order) octx->op_params[0];

    // Rows to process
    uint32_t total_rows = ne01 * ne02 * ne03;
    uint32_t rows_per_thread = actx->nrows_per_thread;
    uint32_t start_row = rows_per_thread * i;
    uint32_t end_row = MIN(start_row + rows_per_thread, total_rows);

    size_t values_size = hex_round_up(ne00 * sizeof(float), 128);
    uint32_t num_vec_ind_values = hmx_ceil_div(ne00, VLEN/(sizeof(int32_t)));
    float * values_buf = (float *) spad;
    int32_t * indices_buf = (int32_t *) (spad + values_size);
    HVX_Vector * indices_buf_vec = (HVX_Vector *) (spad + values_size);
    const HVX_Vector ind_init_vec = *(HVX_Vector *)argosrt_ramp_lut;
    const HVX_Vector ind_diff_vec = Q6_V_vsplat_R(32);

    struct htp_thread_trace * tr = &octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, start_row);

    for (uint32_t r = start_row; r < end_row; r++) {
        uint32_t src_offset = r * nb01;
        uint32_t dst_offset = r * nb1;

        uint8_t * src_ptr = (uint8_t *) src0->data + src_offset;
        uint8_t * dst_ptr = (uint8_t *) dst->data  + dst_offset;

        hex_l2fetch(src_ptr, ne00 * sizeof(float), ne00 * sizeof(float), 1);
        hvx_copy_f32_au((uint8_t*)values_buf, src_ptr, ne00);

        // Initialize indices - Start with values 0..31, add 32 for additional vec iterations
        HVX_Vector curr_ind_vec = ind_init_vec;
        for (uint32_t j_vec = 0; j_vec < num_vec_ind_values; j_vec++) {
            indices_buf_vec[j_vec] = curr_ind_vec;
            curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
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

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, start_row);
}

int op_argsort(struct htp_ops_context * octx) {
    // Check supported types
    if (octx->src[0]->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t total_rows = octx->src[0]->ne[1] * octx->src[0]->ne[2] * octx->src[0]->ne[3];
    const uint32_t n_threads = MIN(total_rows, octx->n_threads);

    // Allocate scratchpad
    // We need 1 row of float + 1 row of int32 per thread.
    uint32_t ne00 = octx->src[0]->ne[0];
    size_t values_size  = hex_round_up(ne00 * sizeof(float), 128);
    size_t indices_size = hex_round_up(ne00 * sizeof(int32_t), 128);
    size_t spad_per_thread = values_size + indices_size;

    // Make sure we round up to 256 for alignment requirements
    spad_per_thread = hex_round_up(spad_per_thread, 256);

    size_t total_spad_size = spad_per_thread * n_threads;

    if (octx->ctx->vtcm_size < total_spad_size) {
        FARF(ERROR, "argsort: VTCM size too small. Needed %zu, have %zu", total_spad_size, octx->ctx->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    FARF(HIGH, "argsort: %ux%ux%ux%u -> %ux%ux%ux%u (0x%x, 0x%x)",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         octx->src[0]->data, octx->dst->data);

    struct htp_argsort_context actx;
    actx.octx = octx;
    actx.nrows_per_thread = (total_rows + n_threads - 1) / n_threads;
    actx.vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    actx.vtcm_per_thread = spad_per_thread;

    enum ggml_sort_order order = (enum ggml_sort_order) octx->op_params[0];
    worker_callback_t job_func = htp_argsort_f32_fallback;

    if (order == GGML_SORT_ORDER_ASC) {
        switch (ne00) {
            case 1024: job_func = htp_argsort_f32_1024_asc; break;
            case 512:  job_func = htp_argsort_f32_512_asc;  break;
            case 256:  job_func = htp_argsort_f32_256_asc;  break;
            case 128:  job_func = htp_argsort_f32_128_asc;  break;
            case 64:   job_func = htp_argsort_f32_64_asc;   break;
            case 32:   job_func = htp_argsort_f32_32_asc;   break;
            default:   job_func = htp_argsort_f32_fallback; break;
        }
    } else {
        switch (ne00) {
            case 1024: job_func = htp_argsort_f32_1024_dsc; break;
            case 512:  job_func = htp_argsort_f32_512_dsc;  break;
            case 256:  job_func = htp_argsort_f32_256_dsc;  break;
            case 128:  job_func = htp_argsort_f32_128_dsc;  break;
            case 64:   job_func = htp_argsort_f32_64_dsc;   break;
            case 32:   job_func = htp_argsort_f32_32_dsc;   break;
            default:   job_func = htp_argsort_f32_fallback; break;
        }
    }

    // Run jobs
    worker_pool_run_func(octx->ctx->worker_pool, job_func, &actx, n_threads);

    return HTP_STATUS_OK;
}
