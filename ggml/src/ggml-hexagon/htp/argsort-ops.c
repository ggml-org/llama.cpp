#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <HAP_farf.h>
#include <HAP_perf.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

#include "hvx-utils.h"
#include "dma-queue.h"

#include "hex-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

struct htp_argsort_context {
    struct htp_ops_context * octx;
    uint32_t                 nrows_per_thread;
    uint32_t                 total_rows;
    uint32_t                 row_start;
    uint32_t                 row_end;
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

static uint32_t top_k_max_value_index(const float * values, uint32_t n, float * value) {
    uint32_t index = 0;
    float max_value = values[0];

    for (uint32_t i = 1; i < n; i++) {
        if (values[i] > max_value) {
            max_value = values[i];
            index = i;
        }
    }

    *value = max_value;
    return index;
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


// Sorts descending; values pre-padded with -INFINITY. init_indices=true resets
// indices to a fresh ramp (normal full-row sort); false leaves caller-supplied
// indices in place and only permutes them (used when merging candidates, to
// preserve their original global index).

static void bitonic_sort_vtcm_desc(uint8_t * values, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(HVX_Vector *)argosrt_ramp_lut;

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    if (init_indices) {
        // Initialize indices ramp (values are already populated by the caller)
        for (uint32_t v = 0; v < n_vec; v++) {
            HVX_Vector idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_Vector *)(indices + v * 128) = idx;
        }
    }

    int M = 5;
    while ((1u << (M - 5)) < n_vec) M++;

    for (int s = 1; s <= M; s++) {
        for (int stage_d = s - 1; stage_d >= 0; stage_d--) {
            int d = 1 << stage_d;
            if (d >= 32) {
                uint32_t v_dist = d / 32;
                for (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    if ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_Vector Vv1 = *(HVX_Vector *)(values + v1 * 128);
                        HVX_Vector Iv1 = *(HVX_Vector *)(indices + v1 * 128);
                        HVX_Vector Vv2 = *(HVX_Vector *)(values + v2 * 128);
                        HVX_Vector Iv2 = *(HVX_Vector *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_Vector *)(values + v1 * 128)  = Vv1;
                        *(HVX_Vector *)(indices + v1 * 128) = Iv1;
                        *(HVX_Vector *)(values + v2 * 128)  = Vv2;
                        *(HVX_Vector *)(indices + v2 * 128) = Iv2;
                    }
                }
            } else {
                if (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    for (uint32_t v = 0; v < n_vec; v++) {
                        HVX_Vector Vv = *(HVX_Vector *)(values + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(values + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                } else {
                    for (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;

                        HVX_Vector Vv = *(HVX_Vector *)(values + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(values + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

static void top_k_select_tiled(const uint8_t * src, uint32_t n, uint32_t k,
                               float * values_buf, int32_t * indices_buf,
                               float * top_values, int32_t * top_indices) {
    const uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_count    = n_tiles * k;
    uint32_t merge_n_vec        = hmx_ceil_div(candidate_count, 32);
    uint32_t merge_n_vec_pow2   = 1;
    while (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
    uint32_t merge_elems        = merge_n_vec_pow2 * 32;
    float * candidate_values    = values_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    for (uint32_t offset = 0; offset < n; offset += tile_elems) {
        uint32_t tile_count = MIN(tile_elems, n - offset);
        hvx_copy_f32_au((uint8_t *) values_buf, src + offset * sizeof(float), tile_count);
        if (tile_count < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (values_buf + tile_count), -INFINITY, tile_elems - tile_count);
        }

        bitonic_sort_vtcm_desc((uint8_t *) values_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_count);
        for (uint32_t j = 0; j < tile_k; j++) {
            candidate_values[candidate_pos] = values_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) offset;
            candidate_pos++;
        }
    }

    if (merge_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_values + candidate_pos), -INFINITY, merge_elems - candidate_pos);
        for (uint32_t j = candidate_pos; j < merge_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_sort_vtcm_desc((uint8_t *) candidate_values, (uint8_t *) candidate_indices, merge_n_vec_pow2, false);

    for (uint32_t j = 0; j < k; j++) {
        top_values[j] = candidate_values[j];
        top_indices[j] = candidate_indices[j];
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
    uint32_t rows_per_thread = actx->nrows_per_thread;                                                         \
    uint32_t start_row = actx->row_start + rows_per_thread * i;                                                \
    uint32_t end_row = MIN(start_row + rows_per_thread, actx->row_end);                                        \
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

    uint32_t nb01 = src0->nb[1];

    uint32_t nb1 = dst->nb[1];

    // Sort order
    enum ggml_sort_order order = (enum ggml_sort_order) octx->op_params[0];

    // Rows to process
    uint32_t rows_per_thread = actx->nrows_per_thread;
    uint32_t start_row = actx->row_start + rows_per_thread * i;
    uint32_t end_row = MIN(start_row + rows_per_thread, actx->row_end);

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

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t total_rows  = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const size_t dst_row_size  = dst->ne[0]  * sizeof(int32_t);

    uint32_t row_start = 0;
    uint32_t row_end   = total_rows;
    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, sizeof(int32_t), (uint32_t) dst_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        row_end   = range.start + range.count;
    }

    const uint32_t nrows = row_end - row_start;
    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = octx->n_threads;

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
    actx.nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    actx.total_rows       = nrows;
    actx.row_start        = row_start;
    actx.row_end          = row_end;
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
    work_queue_run(octx->ctx->work_queue, job_func, &actx, n_threads);

    return HTP_STATUS_OK;
}

// ggml_compute_forward_top_k
//
// Reuses ARGSORT's sort kernels. Only the first `k` indices are copied
// to dst, and there's no asc/desc param -- always largest-first.

struct htp_top_k_context {
    struct htp_ops_context * octx;
    uint32_t                 nrows_per_thread;
    uint32_t                 row_start;
    uint32_t                 row_end;
    uint8_t *                vtcm_base;
    size_t                   vtcm_per_thread;
    uint32_t                 k;
};

#define HTP_TOP_K_FN(ne00, sort_fn)                                                                            \
static void htp_top_k_f32_##ne00(unsigned int n, unsigned int i, void * data) {                                \
    struct htp_top_k_context * actx = (struct htp_top_k_context *)data;                                        \
    struct htp_ops_context * octx = actx->octx;                                                                \
    const struct htp_tensor * src0 = octx->src[0];                                                             \
    const struct htp_tensor * dst = octx->dst;                                                                 \
    uint8_t * spad = actx->vtcm_base + actx->vtcm_per_thread * i;                                              \
    uint32_t row_start = actx->row_start;                                                                      \
    uint32_t row_end = actx->row_end;                                                                          \
    uint32_t rows_per_thread = actx->nrows_per_thread;                                                         \
    uint32_t start_row = row_start + rows_per_thread * i;                                                      \
    uint32_t end_row = MIN(start_row + rows_per_thread, row_end);                                              \
    size_t values_size = hex_round_up(ne00 * sizeof(float), 128);                                              \
    float * values_buf = (float *) spad;                                                                       \
    int32_t * indices_buf = (int32_t *) (spad + values_size);                                                  \
    uint32_t nb01 = src0->nb[1];                                                                               \
    uint32_t nb1 = dst->nb[1];                                                                                 \
    uint32_t k = actx->k;                                                                                      \
    struct htp_thread_trace * tr = &octx->ctx->trace[i];                                                       \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, start_row);                                              \
    for (uint32_t r = start_row; r < end_row; r++) {                                                           \
        uint32_t src_offset = r * nb01;                                                                        \
        uint32_t dst_offset = r * nb1;                                                                         \
        uint8_t * src_ptr = (uint8_t *) src0->data + src_offset;                                               \
        uint8_t * dst_ptr = (uint8_t *) dst->data  + dst_offset;                                               \
        hex_l2fetch(src_ptr, ne00 * sizeof(float), ne00 * sizeof(float), 1);                                   \
        hvx_copy_f32_au((uint8_t*)values_buf, src_ptr, ne00);                                                  \
        sort_fn((uint8_t*)values_buf, (uint8_t*)indices_buf, GGML_SORT_ORDER_DESC);                            \
        hvx_copy_f32_ua(dst_ptr, (const uint8_t *) indices_buf, k);                                            \
    }                                                                                                          \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, start_row);                                               \
}

HTP_TOP_K_FN(32,   sort32_f32_hvx)
HTP_TOP_K_FN(64,   sort64_f32_hvx)
HTP_TOP_K_FN(128,  sort128_f32_hvx)
HTP_TOP_K_FN(256,  sort256_f32_hvx)
HTP_TOP_K_FN(512,  sort512_f32_hvx)
HTP_TOP_K_FN(1024, sort1024_f32_hvx)

static void htp_top_k_f32_fallback(unsigned int n, unsigned int i, void * data) {
    struct htp_top_k_context * actx = (struct htp_top_k_context *)data;
    struct htp_ops_context * octx = actx->octx;

    // Unpack context
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    // Scratchpad memory
    uint8_t * spad = actx->vtcm_base + actx->vtcm_per_thread * i;

    // Dimensions
    uint32_t ne00 = src0->ne[0];
    uint32_t nb01 = src0->nb[1];

    uint32_t nb1 = dst->nb[1];

    uint32_t k = actx->k;

    // Rows to process
    uint32_t row_start = actx->row_start;
    uint32_t row_end = actx->row_end;
    uint32_t rows_per_thread = actx->nrows_per_thread;
    uint32_t start_row = row_start + rows_per_thread * i;
    uint32_t end_row = MIN(start_row + rows_per_thread, row_end);

    // Pad ne00 to n_vec*32 (n_vec a power of 2) for the bitonic network;
    // pad with -INFINITY so it never lands in the top-k.
    uint32_t n_vec = hmx_ceil_div(ne00, 32);
    uint32_t n_vec_pow2 = 1;
    while (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
    uint32_t ne00_padded = n_vec_pow2 * 32;

    size_t values_size = hex_round_up(ne00_padded * sizeof(float), 128);
    float * values_buf = (float *) spad;
    int32_t * indices_buf = (int32_t *) (spad + values_size);

    struct htp_thread_trace * tr = &octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, start_row);

    for (uint32_t r = start_row; r < end_row; r++) {
        uint32_t src_offset = r * nb01;
        uint32_t dst_offset = r * nb1;

        uint8_t * src_ptr = (uint8_t *) src0->data + src_offset;
        uint8_t * dst_ptr = (uint8_t *) dst->data  + dst_offset;

        hex_l2fetch(src_ptr, ne00 * sizeof(float), ne00 * sizeof(float), 1);

        if (k <= 64 && ne00 > 1024) {
            float top_values[64];
            int32_t top_indices[64];
            top_k_select_tiled(src_ptr, ne00, k, values_buf, indices_buf, top_values, top_indices);
            memcpy(dst_ptr, top_indices, k * sizeof(int32_t));
            continue;
        }

        hvx_copy_f32_au((uint8_t*)values_buf, src_ptr, ne00);

        // Fills the indices ramp itself, so no init needed here.
        if (ne00_padded > ne00) {
            hvx_splat_f32_u((uint8_t *)(values_buf + ne00), -INFINITY, ne00_padded - ne00);
        }
        bitonic_sort_vtcm_desc((uint8_t*)values_buf, (uint8_t*)indices_buf, n_vec_pow2, true);

        // Copy top-k indices back to DDR
        hvx_copy_f32_ua(dst_ptr, (const uint8_t *) indices_buf, k);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, start_row);
}

// Single row (ne01=ne02=ne03=1) + large ne00 would otherwise run on one
// HVX thread while the rest sit idle. Split the row into n_chunks
// power-of-two chunks, sort each in parallel with bitonic_sort_vtcm_desc,
// then merge the n_chunks*local_k winners with one more sort that carries
// the global index through instead of re-deriving it.
struct htp_top_k_chunk_ctx {
    struct htp_ops_context * octx;
    uint8_t *                vtcm_base;
    size_t                    phase1_slot_size;
    uint32_t                  ne00;
    uint32_t                  chunk_elems;
    uint32_t                  local_k;
    size_t                    merge_values_off;
    size_t                    merge_indices_off;
};

static void htp_top_k_chunk_job(unsigned int n, unsigned int i, void * data) {
    struct htp_top_k_chunk_ctx * cctx = (struct htp_top_k_chunk_ctx *) data;
    struct htp_ops_context * octx = cctx->octx;
    const struct htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = cctx->chunk_elems;
    uint32_t ne00        = cctx->ne00;
    uint32_t local_k     = cctx->local_k;
    uint32_t chunk_base  = i * chunk_elems;

    uint8_t * spad = cctx->vtcm_base + cctx->phase1_slot_size * i;
    size_t values_size = hex_round_up(chunk_elems * sizeof(float), 128);
    float *   values_buf  = (float *) spad;
    int32_t * indices_buf = (int32_t *) (spad + values_size);

    uint32_t real_count = (chunk_base < ne00) ? MIN(chunk_elems, ne00 - chunk_base) : 0;

    if (real_count == 0) {
        float *   merge_values  = (float *) (cctx->vtcm_base + cctx->merge_values_off);
        int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + cctx->merge_indices_off);
        for (uint32_t j = 0; j < local_k; j++) {
            merge_values[i * local_k + j] = -INFINITY;
            merge_indices[i * local_k + j] = 0;
        }
        return;
    }

    if (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
        uint8_t * src_ptr = (uint8_t *) src0->data + (size_t) chunk_base * sizeof(float);
        float top_values[64];
        int32_t top_indices[64];
        float *   merge_values  = (float *) (cctx->vtcm_base + cctx->merge_values_off);
        int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + cctx->merge_indices_off);
        for (uint32_t j = 0; j < local_k; j++) {
            top_values[j] = -INFINITY;
            top_indices[j] = 0;
        }
        top_k_select_tiled(src_ptr, real_count, local_k, values_buf, indices_buf, top_values, top_indices);

        for (uint32_t j = 0; j < local_k; j++) {
            merge_values[i * local_k + j] = top_values[j];
            merge_indices[i * local_k + j] = top_indices[j] + (int32_t) chunk_base;
        }
        return;
    }

    if (real_count > 0) {
        uint8_t * src_ptr = (uint8_t *) src0->data + (size_t) chunk_base * sizeof(float);
        hex_l2fetch(src_ptr, real_count * sizeof(float), real_count * sizeof(float), 1);
        hvx_copy_f32_au((uint8_t *) values_buf, src_ptr, real_count);
    }
    if (chunk_elems > real_count) {
        hvx_splat_f32_u((uint8_t *) (values_buf + real_count), -INFINITY, chunk_elems - real_count);
    }

    if (local_k == 1 && cctx->ne00 >= 128*1024) {
        float max_value;
        uint32_t max_index = top_k_max_value_index(values_buf, real_count, &max_value);
        float *   merge_values  = (float *) (cctx->vtcm_base + cctx->merge_values_off);
        int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + cctx->merge_indices_off);
        merge_values[i] = max_value;
        merge_indices[i] = (int32_t) (max_index + chunk_base);
        return;
    }

    // chunk_elems is always a power-of-two multiple of 32
    bitonic_sort_vtcm_desc((uint8_t *) values_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

    float *   merge_values  = (float *)   (cctx->vtcm_base + cctx->merge_values_off);
    int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + cctx->merge_indices_off);

    for (uint32_t j = 0; j < local_k; j++) {
        merge_values[i * local_k + j]  = values_buf[j];
        merge_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_base;
    }
}

struct htp_top_k_merge_ctx {
    struct htp_ops_context * octx;
    uint8_t *                vtcm_base;
    size_t                    merge_values_off;
    size_t                    merge_indices_off;
    uint32_t                  merge_elems;
    uint32_t                  total_candidates;
    uint32_t                  k;
};

static void htp_top_k_merge_job(unsigned int n, unsigned int i, void * data) {
    struct htp_top_k_merge_ctx * mctx = (struct htp_top_k_merge_ctx *) data;
    struct htp_ops_context * octx = mctx->octx;
    const struct htp_tensor * dst = octx->dst;

    float *   merge_values  = (float *)   (mctx->vtcm_base + mctx->merge_values_off);
    int32_t * merge_indices = (int32_t *) (mctx->vtcm_base + mctx->merge_indices_off);

    if (mctx->merge_elems > mctx->total_candidates) {
        uint32_t pad = mctx->merge_elems - mctx->total_candidates;
        hvx_splat_f32_u((uint8_t *) (merge_values + mctx->total_candidates), -INFINITY, pad);
        for (uint32_t j = mctx->total_candidates; j < mctx->merge_elems; j++) {
            merge_indices[j] = 0;
        }
    }

    // Preserve the global indices computed in phase 1 -- init_indices=false
    // so they aren't overwritten with a local ramp.
    bitonic_sort_vtcm_desc((uint8_t *) merge_values, (uint8_t *) merge_indices, mctx->merge_elems / 32, false);

    hvx_copy_f32_ua((uint8_t *) dst->data, (const uint8_t *) merge_indices, mctx->k);
}

static int op_top_k_single_row_threaded(struct htp_ops_context * octx, uint32_t ne00, uint32_t k) {
    uint32_t n_threads_avail = octx->n_threads;

    uint32_t n_vec = hmx_ceil_div(ne00, 32);
    uint32_t n_vec_pow2 = 1;
    while (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;

    // Largest power-of-two chunk count that both fits the available
    // threads and evenly divides n_vec_pow2
    uint32_t n_chunks = 1;
    while (n_chunks * 2 <= n_threads_avail && n_chunks * 2 <= n_vec_pow2) {
        n_chunks *= 2;
    }

    uint32_t chunk_n_vec = n_vec_pow2 / n_chunks;
    uint32_t chunk_elems = chunk_n_vec * 32;
    uint32_t local_k     = MIN(k, chunk_elems);

    uint32_t total_candidates = n_chunks * local_k;
    uint32_t merge_n_vec = hmx_ceil_div(total_candidates, 32);
    uint32_t merge_n_vec_pow2 = 1;
    while (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
    uint32_t merge_elems = merge_n_vec_pow2 * 32;

    size_t phase1_values_size  = hex_round_up(chunk_elems * sizeof(float), 128);
    size_t phase1_indices_size = hex_round_up(chunk_elems * sizeof(int32_t), 128);
    size_t phase1_slot_size    = hex_round_up(phase1_values_size + phase1_indices_size, 256);
    size_t phase1_total_size   = phase1_slot_size * n_chunks;

    size_t merge_values_size  = hex_round_up(merge_elems * sizeof(float), 128);
    size_t merge_indices_size = hex_round_up(merge_elems * sizeof(int32_t), 128);
    size_t merge_values_off   = phase1_total_size;
    size_t merge_indices_off  = merge_values_off + merge_values_size;

    size_t total_vtcm = phase1_total_size + merge_values_size + merge_indices_size;
    if (octx->ctx->vtcm_size < total_vtcm) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    struct htp_top_k_chunk_ctx cctx;
    cctx.octx              = octx;
    cctx.vtcm_base         = vtcm_base;
    cctx.phase1_slot_size  = phase1_slot_size;
    cctx.ne00              = ne00;
    cctx.chunk_elems       = chunk_elems;
    cctx.local_k           = local_k;
    cctx.merge_values_off  = merge_values_off;
    cctx.merge_indices_off = merge_indices_off;

    work_queue_run(octx->ctx->work_queue, htp_top_k_chunk_job, &cctx, n_chunks);

    struct htp_top_k_merge_ctx mctx;
    mctx.octx              = octx;
    mctx.vtcm_base         = vtcm_base;
    mctx.merge_values_off  = merge_values_off;
    mctx.merge_indices_off = merge_indices_off;
    mctx.merge_elems       = merge_elems;
    mctx.total_candidates  = total_candidates;
    mctx.k                 = k;

    work_queue_run(octx->ctx->work_queue, htp_top_k_merge_job, &mctx, 1);

    return HTP_STATUS_OK;
}

int op_top_k(struct htp_ops_context * octx) {
    // Check supported types
    if (octx->src[0]->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const size_t dst_row_size = dst->ne[0] * sizeof(int32_t);

    uint32_t row_start = 0;
    uint32_t row_end   = total_rows;
    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, sizeof(int32_t), (uint32_t) dst_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk,
            octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        row_end   = range.start + range.count;
    }

    const uint32_t nrows = row_end - row_start;
    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    // Single row + large ne00: the per-row dispatch below would run on one
    // HVX thread while the rest sit idle. Split the row across threads.
    if (total_rows == 1 && ne00 > 1024) {
        int status = op_top_k_single_row_threaded(octx, ne00, k);
        if (status != HTP_STATUS_VTCM_TOO_SMALL) {
            return status;
        }
        // else: fall through to the single-thread path below.
    }

    const uint32_t n_threads = MIN(nrows, octx->n_threads);

    // Scratchpad layout: values + indices
    // For bitonic: need padding to power-of-2 size
    // Allocate for worst case (bitonic with padding)
    uint32_t n_vec = hmx_ceil_div(ne00, 32);
    uint32_t n_vec_pow2 = 1;
    while (n_vec_pow2 < n_vec) n_vec_pow2 <<= 1;
    uint32_t ne00_padded = n_vec_pow2 * 32;

    size_t values_size  = hex_round_up(ne00_padded * sizeof(float), 128);
    size_t indices_size = hex_round_up(ne00_padded * sizeof(int32_t), 128);
    size_t spad_per_thread = values_size + indices_size;

    // Make sure we round up to 256 for alignment requirements
    spad_per_thread = hex_round_up(spad_per_thread, 256);

    size_t total_spad_size = spad_per_thread * n_threads;

    if (octx->ctx->vtcm_size < total_spad_size) {
        FARF(ERROR, "top_k: VTCM size too small. Needed %zu, have %zu", total_spad_size, octx->ctx->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    FARF(HIGH, "top_k: %ux%ux%ux%u -> %ux%ux%ux%u (0x%x, 0x%x)",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         octx->src[0]->data, octx->dst->data);

    struct htp_top_k_context actx;
    const struct fastdiv_values n_threads_div = init_fastdiv_values(n_threads);
    actx.octx             = octx;
    actx.nrows_per_thread = fastdiv(nrows + n_threads - 1, &n_threads_div);
    actx.row_start        = row_start;
    actx.row_end          = row_end;
    actx.vtcm_base        = (uint8_t *) octx->ctx->vtcm_base;
    actx.vtcm_per_thread  = spad_per_thread;
    actx.k                = k;

    worker_callback_t job_func = htp_top_k_f32_fallback;
    switch (ne00) {
        case 1024: job_func = htp_top_k_f32_1024; break;
        case 512:  job_func = htp_top_k_f32_512;  break;
        case 256:  job_func = htp_top_k_f32_256;  break;
        case 128:  job_func = htp_top_k_f32_128;  break;
        case 64:   job_func = htp_top_k_f32_64;   break;
        case 32:   job_func = htp_top_k_f32_32;   break;
        default:   job_func = htp_top_k_f32_fallback; break;
    }

    // Run jobs
    work_queue_run(octx->ctx->work_queue, job_func, &actx, n_threads);

    return HTP_STATUS_OK;
}
