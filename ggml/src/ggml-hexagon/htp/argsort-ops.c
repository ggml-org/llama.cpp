#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <HAP_farf.h>
#include <HAP_perf.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

#include "hvx-utils.h"
#include "hvx-copy.h"
#include "hvx-reduce.h"
#include "dma-queue.h"

#include "hex-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "argsort-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif


static void quicksort_values_indices_asc(float * values, int32_t * indices, int left, int right) {
    while (left < right) {
        float pivot = values[left + (right - left) / 2];
        int i = left;
        int j = right;

        while (i <= j) {
            while (values[i] < pivot) i++;
            while (values[j] > pivot) j--;

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

        // Tail-recursion elimination to bound stack depth
        if (j - left < right - i) {
            if (left < j) quicksort_values_indices_asc(values, indices, left, j);
            left = i;
        } else {
            if (i < right) quicksort_values_indices_asc(values, indices, i, right);
            right = j;
        }
    }
}

static void quicksort_values_indices_desc(float * values, int32_t * indices, int left, int right) {
    while (left < right) {
        float pivot = values[left + (right - left) / 2];
        int i = left;
        int j = right;

        while (i <= j) {
            while (values[i] > pivot) i++;
            while (values[j] < pivot) j--;

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

        // Tail-recursion elimination to bound stack depth
        if (j - left < right - i) {
            if (left < j) quicksort_values_indices_desc(values, indices, left, j);
            left = i;
        } else {
            if (i < right) quicksort_values_indices_desc(values, indices, i, right);
            right = j;
        }
    }
}

static const int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

static const uint32_t top_k_bit_lut[32] __attribute__((aligned(VLEN))) = {
    1u <<  0, 1u <<  1, 1u <<  2, 1u <<  3, 1u <<  4, 1u <<  5, 1u <<  6, 1u <<  7,
    1u <<  8, 1u <<  9, 1u << 10, 1u << 11, 1u << 12, 1u << 13, 1u << 14, 1u << 15,
    1u << 16, 1u << 17, 1u << 18, 1u << 19, 1u << 20, 1u << 21, 1u << 22, 1u << 23,
    1u << 24, 1u << 25, 1u << 26, 1u << 27, 1u << 28, 1u << 29, 1u << 30, 1u << 31
};

static inline void init_indices_ramp(int32_t * indices_buf, uint32_t count, uint32_t base) {
    const HVX_Vector ind_init_vec = Q6_Vw_vadd_VwVw(*(const HVX_Vector *) argsort_ramp_lut, Q6_V_vsplat_R(base));
    const HVX_Vector ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_Vector * indices_buf_vec = (HVX_Vector *) indices_buf;
    uint32_t num_vecs = count >> 5;
    HVX_Vector curr_ind_vec = ind_init_vec;
    for (uint32_t j = 0; j < num_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = count & 31u;
    if (rem > 0) {
        hvx_vec_store_a((void *) &indices_buf_vec[num_vecs], rem * sizeof(int32_t), curr_ind_vec);
    }
}

// Map IEEE-754 float lanes to signed int32 keys preserving +INF > finite > -INF (0x807fffff) > NaN (0x80000001) > pad (0x80000000).
__attribute__((always_inline))
static inline HVX_Vector float_to_ord_key_vec(HVX_Vector vf) {
    const HVX_Vector v_abs     = Q6_V_vand_VV(vf, Q6_V_vsplat_R(0x7fffffff));
    const HVX_VectorPred q_nan = Q6_Q_vcmp_gt_VwVw(v_abs, Q6_V_vsplat_R(0x7f800000));
    const HVX_Vector v_sign    = Q6_Vw_vasr_VwR(vf, 31);
    const HVX_Vector v_flip    = Q6_Vuw_vlsr_VuwR(v_sign, 1);
    const HVX_Vector v_key     = Q6_V_vxor_VV(vf, v_flip);
    return Q6_V_vmux_QVV(q_nan, Q6_V_vsplat_R((int32_t) 0x80000001u), v_key);
}

static inline float ord_key_to_float(int32_t key) {
    const uint32_t u = ((uint32_t) key == 0x80000001u)
                       ? 0x7fc00000u
                       : ((key < 0) ? ((uint32_t) key ^ 0x7fffffffu) : (uint32_t) key);
    union { uint32_t u; float f; } conv = { .u = u };
    return conv.f;
}

__attribute__((always_inline))
static inline uint32_t hvx_pred_i32_to_bits(HVX_VectorPred q, HVX_Vector v_bit_lut) {
    HVX_Vector m = Q6_V_vand_QV(q, v_bit_lut);
    m = Q6_V_vor_VV(m, Q6_V_vror_VR(m, 64));
    m = Q6_V_vor_VV(m, Q6_V_vror_VR(m, 32));
    m = Q6_V_vor_VV(m, Q6_V_vror_VR(m, 16));
    m = Q6_V_vor_VV(m, Q6_V_vror_VR(m, 8));
    m = Q6_V_vor_VV(m, Q6_V_vror_VR(m, 4));
    return (uint32_t) Q6_R_vextract_VR(m, 0);
}

__attribute__((noinline))
static int32_t top_k_argmax_hvx(const float * restrict vals, uint32_t n, uint32_t idx_base, float * restrict out_val) {
    const HVX_Vector * restrict vvals = (const HVX_Vector *) vals;
    const uint32_t n_full = n >> 5;
    const uint32_t rem    = n & 31u;
    const HVX_Vector v_step = Q6_V_vsplat_R(32);
    const HVX_Vector v_pad  = Q6_V_vsplat_R((int32_t) 0x80000000u);

    HVX_Vector v_idx      = Q6_Vw_vadd_VwVw(*(const HVX_Vector *) argsort_ramp_lut, Q6_V_vsplat_R((int32_t) idx_base));
    HVX_Vector v_best_key = v_pad;
    HVX_Vector v_best_idx = v_idx;

    for (uint32_t v = 0; v < n_full; v++) {
        const HVX_Vector vk = float_to_ord_key_vec(vvals[v]);
        const HVX_VectorPred q = Q6_Q_vcmp_gt_VwVw(vk, v_best_key);
        v_best_key = Q6_V_vmux_QVV(q, vk, v_best_key);
        v_best_idx = Q6_V_vmux_QVV(q, v_idx, v_best_idx);
        v_idx      = Q6_Vw_vadd_VwVw(v_idx, v_step);
    }
    if (rem > 0) {
        const HVX_VectorPred q_tail = Q6_Q_vsetq2_R(rem * sizeof(int32_t));
        const HVX_Vector vk = Q6_V_vmux_QVV(q_tail, float_to_ord_key_vec(vvals[n_full]), v_pad);
        const HVX_VectorPred q = Q6_Q_vcmp_gt_VwVw(vk, v_best_key);
        v_best_key = Q6_V_vmux_QVV(q, vk, v_best_key);
        v_best_idx = Q6_V_vmux_QVV(q, v_idx, v_best_idx);
    }

    #pragma clang loop unroll(full)
    for (int shift = 64; shift >= 4; shift >>= 1) {
        const HVX_Vector rk = Q6_V_vror_VR(v_best_key, shift);
        const HVX_Vector ri = Q6_V_vror_VR(v_best_idx, shift);
        const HVX_VectorPred q_gt  = Q6_Q_vcmp_gt_VwVw(rk, v_best_key);
        const HVX_VectorPred q_eq  = Q6_Q_vcmp_eq_VwVw(rk, v_best_key);
        const HVX_VectorPred q_idx = Q6_Q_vcmp_gt_VwVw(v_best_idx, ri);
        const HVX_VectorPred q_win = Q6_Q_or_QQ(q_gt, Q6_Q_and_QQ(q_eq, q_idx));
        v_best_key = Q6_V_vmux_QVV(q_win, rk, v_best_key);
        v_best_idx = Q6_V_vmux_QVV(q_win, ri, v_best_idx);
    }

    if (out_val) {
        *out_val = ord_key_to_float((int32_t) Q6_R_vextract_VR(v_best_key, 0));
    }
    return (int32_t) Q6_R_vextract_VR(v_best_idx, 0);
}

__attribute__((always_inline))
static inline void vec_cas(HVX_Vector * X_val, HVX_Vector * X_idx, HVX_Vector * Y_val, HVX_Vector * Y_idx, bool asc) {
    asm volatile("" : "+v"(*X_val), "+v"(*X_idx), "+v"(*Y_val), "+v"(*Y_idx));
    const HVX_Vector lhs        = asc ? *X_val : *Y_val;
    const HVX_Vector rhs        = asc ? *Y_val : *X_val;
    const HVX_VectorPred pred   = Q6_Q_vcmp_gt_VsfVsf(lhs, rhs);
    const HVX_Vector next_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    const HVX_Vector next_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    const HVX_Vector next_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    const HVX_Vector next_Y_idx = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = next_X_val;
    *Y_val = next_Y_val;
    *X_idx = next_X_idx;
    *Y_idx = next_Y_idx;
    asm volatile("" : "+v"(*X_val), "+v"(*X_idx), "+v"(*Y_val), "+v"(*Y_idx));
}

__attribute__((always_inline))
static inline void bitonic_cas_32(HVX_Vector * V, HVX_Vector * I, int d, int s, int stage_d, bool asc, HVX_Vector idx_vec, HVX_Vector zero_vec) {
    HVX_Vector idx = idx_vec;
    int flip = asc ? 0 : d;
    asm volatile("" : "+v"(*V), "+v"(*I), "+v"(idx), "+r"(d), "+r"(flip));

    const HVX_Vector v_ctrl   = Q6_Vb_vsplat_R(d << 2);
    const HVX_Vector V_paired = Q6_V_vdelta_VV(*V, v_ctrl);
    const HVX_Vector I_paired = Q6_V_vdelta_VV(*I, v_ctrl);

    const HVX_Vector v_rhs    = (s < 5) ? Q6_Vuw_vlsr_VuwR(idx, s - stage_d) : Q6_V_vsplat_R(flip);
    const HVX_Vector v_lane_x = Q6_V_vxor_VV(idx, v_rhs);
    const HVX_VectorPred want_max = Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(v_lane_x, Q6_V_vsplat_R(d)), zero_vec);
    const HVX_VectorPred V_gt_P   = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    const HVX_VectorPred P_gt_V   = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    const HVX_VectorPred Q_swap   = Q6_Q_or_QQ(
        Q6_Q_and_QQ(want_max, P_gt_V),
        Q6_Q_and_QQn(V_gt_P, want_max)
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
    asm volatile("" : "+v"(*V), "+v"(*I));
}

__attribute__((always_inline))
static inline void bitonic_sort_generic_hvx(uint8_t * values, uint8_t * indices, int K, bool asc_order) {
    HVX_Vector V[8];
    HVX_Vector I[8];

    const HVX_Vector zero_vec = Q6_V_vzero();
    const HVX_Vector idx_vec  = *(const HVX_Vector *) argsort_ramp_lut;

    #pragma clang loop unroll(full)
    for (int v = 0; v < K; v++) {
        V[v] = *(HVX_Vector *)(values + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    int M = 5;
    while ((1 << (M - 5)) < K) {
        M++;
    }

    #pragma clang loop unroll(full)
    for (int s = 1; s <= M; s++) {
        #pragma clang loop unroll(full)
        for (int stage_d = s - 1; stage_d >= 0; stage_d--) {
            const int d = 1 << stage_d;
            if (d >= 32) {
                const int v_dist = d >> 5;
                #pragma clang loop unroll(full)
                for (int v1 = 0; v1 < K; v1++) {
                    if ((v1 & v_dist) == 0) {
                        const int v2 = v1 + v_dist;
                        const bool asc = (s < M) ? ((((v1 * 32) >> s) & 1) == 0) : asc_order;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } else {
                #pragma clang loop unroll(full)
                for (int v = 0; v < K; v++) {
                    const bool asc = (s < M) ? ((((v * 32) >> s) & 1) == 0) : asc_order;
                    bitonic_cas_32(&V[v], &I[v], d, s, stage_d, asc, idx_vec, zero_vec);
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (int v = 0; v < K; v++) {
        *(HVX_Vector *)(values + v * 128)  = V[v];
        *(HVX_Vector *)(indices + v * 128) = I[v];
    }
}

__attribute__((always_inline))
static inline void vec_cas_i32(HVX_Vector * X_val, HVX_Vector * X_idx, HVX_Vector * Y_val, HVX_Vector * Y_idx, bool asc) {
    asm volatile("" : "+v"(*X_val), "+v"(*X_idx), "+v"(*Y_val), "+v"(*Y_idx));
    const HVX_Vector lhs        = asc ? *X_val : *Y_val;
    const HVX_Vector rhs        = asc ? *Y_val : *X_val;
    const HVX_VectorPred pred   = Q6_Q_vcmp_gt_VwVw(lhs, rhs);
    const HVX_Vector next_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    const HVX_Vector next_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    const HVX_Vector next_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    const HVX_Vector next_Y_idx = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = next_X_val;
    *Y_val = next_Y_val;
    *X_idx = next_X_idx;
    *Y_idx = next_Y_idx;
    asm volatile("" : "+v"(*X_val), "+v"(*X_idx), "+v"(*Y_val), "+v"(*Y_idx));
}

__attribute__((always_inline))
static inline void bitonic_cas_32_i32(HVX_Vector * V, HVX_Vector * I, int d, int s, int stage_d, bool asc, HVX_Vector idx_vec, HVX_Vector zero_vec) {
    HVX_Vector idx = idx_vec;
    int flip = asc ? 0 : d;
    asm volatile("" : "+v"(*V), "+v"(*I), "+v"(idx), "+r"(d), "+r"(flip));

    const HVX_Vector v_ctrl   = Q6_Vb_vsplat_R(d << 2);
    const HVX_Vector V_paired = Q6_V_vdelta_VV(*V, v_ctrl);
    const HVX_Vector I_paired = Q6_V_vdelta_VV(*I, v_ctrl);

    const HVX_Vector v_rhs    = (s < 5) ? Q6_Vuw_vlsr_VuwR(idx, s - stage_d) : Q6_V_vsplat_R(flip);
    const HVX_Vector v_lane_x = Q6_V_vxor_VV(idx, v_rhs);
    const HVX_VectorPred want_max = Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(v_lane_x, Q6_V_vsplat_R(d)), zero_vec);
    const HVX_VectorPred V_gt_P   = Q6_Q_vcmp_gt_VwVw(*V, V_paired);
    const HVX_VectorPred P_gt_V   = Q6_Q_vcmp_gt_VwVw(V_paired, *V);
    const HVX_VectorPred Q_swap   = Q6_Q_or_QQ(
        Q6_Q_and_QQ(want_max, P_gt_V),
        Q6_Q_and_QQn(V_gt_P, want_max)
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
    asm volatile("" : "+v"(*V), "+v"(*I));
}

__attribute__((always_inline))
static inline void bitonic_sort_top_k_hvx(const float * restrict values, int32_t * restrict indices, uint32_t n, uint32_t k, int K) {
    HVX_Vector V[8];
    HVX_Vector I[8];

    const HVX_Vector * restrict vvals = (const HVX_Vector *) values;
    const HVX_Vector zero_vec = Q6_V_vzero();
    const HVX_Vector idx_vec  = *(const HVX_Vector *) argsort_ramp_lut;
    const HVX_Vector v_pad    = Q6_V_vsplat_R((int32_t) 0x80000000u);
    const uint32_t   n_full   = n >> 5;
    const uint32_t   rem      = n & 31u;

    #pragma clang loop unroll(full)
    for (int v = 0; v < K; v++) {
        if ((uint32_t) v < n_full) {
            V[v] = float_to_ord_key_vec(vvals[v]);
        } else if ((uint32_t) v == n_full && rem > 0u) {
            const HVX_VectorPred q_tail = Q6_Q_vsetq2_R(rem * sizeof(int32_t));
            V[v] = Q6_V_vmux_QVV(q_tail, float_to_ord_key_vec(vvals[v]), v_pad);
        } else {
            V[v] = v_pad;
        }
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    int M = 5;
    while ((1 << (M - 5)) < K) {
        M++;
    }

    #pragma clang loop unroll(full)
    for (int s = 1; s <= M; s++) {
        #pragma clang loop unroll(full)
        for (int stage_d = s - 1; stage_d >= 0; stage_d--) {
            const int d = 1 << stage_d;
            if (d >= 32) {
                const int v_dist = d >> 5;
                #pragma clang loop unroll(full)
                for (int v1 = 0; v1 < K; v1++) {
                    if ((v1 & v_dist) == 0) {
                        const int v2 = v1 + v_dist;
                        const bool asc = (s < M) ? ((((v1 * 32) >> s) & 1) == 0) : false;
                        vec_cas_i32(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } else {
                #pragma clang loop unroll(full)
                for (int v = 0; v < K; v++) {
                    const bool asc = (s < M) ? ((((v * 32) >> s) & 1) == 0) : false;
                    bitonic_cas_32_i32(&V[v], &I[v], d, s, stage_d, asc, idx_vec, zero_vec);
                }
            }
        }
    }

    HVX_Vector * restrict vout = (HVX_Vector *) indices;
    const uint32_t k_vecs = (k + 31u) >> 5;
    #pragma clang loop unroll(full)
    for (int v = 0; v < K; v++) {
        if ((uint32_t) v < k_vecs) {
            vout[v] = I[v];
        }
    }
}

__attribute__((noinline))
static void sort32_top_k_hvx(const float * restrict values, int32_t * restrict indices, uint32_t n, uint32_t k) {
    bitonic_sort_top_k_hvx(values, indices, n, k, 1);
}

__attribute__((noinline))
static void sort64_top_k_hvx(const float * restrict values, int32_t * restrict indices, uint32_t n, uint32_t k) {
    bitonic_sort_top_k_hvx(values, indices, n, k, 2);
}

__attribute__((noinline))
static void sort128_top_k_hvx(const float * restrict values, int32_t * restrict indices, uint32_t n, uint32_t k) {
    bitonic_sort_top_k_hvx(values, indices, n, k, 4);
}

__attribute__((noinline))
static void sort256_top_k_hvx(const float * restrict values, int32_t * restrict indices, uint32_t n, uint32_t k) {
    bitonic_sort_top_k_hvx(values, indices, n, k, 8);
}

__attribute__((noinline))
static void sort64_i32_desc(int32_t * restrict keys, int32_t * restrict indices) {
    HVX_Vector * restrict vk = (HVX_Vector *) keys;
    HVX_Vector * restrict vi = (HVX_Vector *) indices;
    HVX_Vector V[2] = { vk[0], vk[1] };
    HVX_Vector I[2] = { vi[0], vi[1] };

    const HVX_Vector zero_vec = Q6_V_vzero();
    const HVX_Vector idx_vec  = *(const HVX_Vector *) argsort_ramp_lut;

    #pragma clang loop unroll(full)
    for (int s = 1; s <= 6; s++) {
        #pragma clang loop unroll(full)
        for (int stage_d = s - 1; stage_d >= 0; stage_d--) {
            const int d = 1 << stage_d;
            if (d >= 32) {
                vec_cas_i32(&V[0], &I[0], &V[1], &I[1], false);
            } else {
                #pragma clang loop unroll(full)
                for (int v = 0; v < 2; v++) {
                    const bool asc = (s < 6) ? (v == 0) : false;
                    bitonic_cas_32_i32(&V[v], &I[v], d, s, stage_d, asc, idx_vec, zero_vec);
                }
            }
        }
    }

    vk[0] = V[0];
    vk[1] = V[1];
    vi[0] = I[0];
    vi[1] = I[1];
}

__attribute__((noinline))
static void bitonic_sort_vtcm(uint8_t * restrict values, uint8_t * restrict indices, uint32_t n_vec, bool init_indices, bool asc_order) {
    const HVX_Vector zero_vec = Q6_V_vzero();
    const HVX_Vector idx_vec  = *(const HVX_Vector *) argsort_ramp_lut;

    if (init_indices) {
        for (uint32_t v = 0; v < n_vec; v++) {
            *(HVX_Vector *)(indices + v * 128) = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
        }
    }

    int M = 5;
    while ((1u << (M - 5)) < n_vec) {
        M++;
    }

    for (int s = 1; s <= M; s++) {
        for (int stage_d = s - 1; stage_d >= 5; stage_d--) {
            const uint32_t v_dist = 1u << (stage_d - 5);
            for (uint32_t v1 = 0; v1 < n_vec; v1++) {
                if ((v1 & v_dist) == 0u) {
                    const uint32_t v2 = v1 + v_dist;
                    const bool asc = (s < M) ? ((((v1 * 32u) >> s) & 1u) == 0u) : asc_order;

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
        }

        // Fuse intra-vector butterfly stages d = 16, 8, 4, 2, 1 into one VTCM load/store per vector.
        for (uint32_t v = 0; v < n_vec; v++) {
            HVX_Vector Vv = *(HVX_Vector *)(values + v * 128);
            HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);
            const bool asc = (s < M) ? ((((v * 32u) >> s) & 1u) == 0u) : asc_order;

            if (s >= 5) {
                bitonic_cas_32(&Vv, &Iv, 16, s, 4, asc, idx_vec, zero_vec);
            }
            if (s >= 4) {
                bitonic_cas_32(&Vv, &Iv, 8, s, 3, asc, idx_vec, zero_vec);
            }
            if (s >= 3) {
                bitonic_cas_32(&Vv, &Iv, 4, s, 2, asc, idx_vec, zero_vec);
            }
            if (s >= 2) {
                bitonic_cas_32(&Vv, &Iv, 2, s, 1, asc, idx_vec, zero_vec);
            }
            bitonic_cas_32(&Vv, &Iv, 1, s, 0, asc, idx_vec, zero_vec);

            *(HVX_Vector *)(values + v * 128)  = Vv;
            *(HVX_Vector *)(indices + v * 128) = Iv;
        }
    }
}

__attribute__((noinline))
static void top_k_scan(const float * restrict vals, uint32_t n, uint32_t k, float * restrict out_v, int32_t * restrict out_i) {
    float    best_v[64];
    int32_t  best_i[64];
    uint32_t cnt = 0;
    float    thr = -INFINITY;

    const HVX_Vector v_one     = Q6_V_vsplat_R(1);
    const HVX_Vector v_bit_lut = *(const HVX_Vector *) top_k_bit_lut;
    const HVX_Vector v_ninf    = hvx_vec_splat_f32(-INFINITY);

    const uint32_t blk = 256u;
    for (uint32_t off = 0; off < n; off += blk) {
        const uint32_t bn = MIN(blk, n - off);
        const HVX_Vector v_thr_blk = hvx_vec_splat_f32(thr);
        HVX_Vector v_any = Q6_V_vzero();

        for (uint32_t vi = 0; vi < bn; vi += 32u) {
            const uint32_t vn = MIN(32u, bn - vi);
            HVX_Vector vv = *(const HVX_Vector *)(vals + off + vi);
            if (vn < 32u) {
                const HVX_VectorPred q_tail = Q6_Q_vsetq2_R(vn * sizeof(float));
                vv = Q6_V_vmux_QVV(q_tail, vv, v_ninf);
            }
            const HVX_VectorPred q_gt = Q6_Q_vcmp_gt_VsfVsf(vv, v_thr_blk);
            v_any = Q6_V_vor_VV(v_any, Q6_V_vand_QV(q_gt, v_one));
        }
        v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 64));
        v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 32));
        v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 16));
        v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 8));
        v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 4));
        if (Q6_R_vextract_VR(v_any, 0) == 0) {
            continue;
        }

        for (uint32_t vi = 0; vi < bn; vi += 32u) {
            const uint32_t vn = MIN(32u, bn - vi);
            HVX_Vector vv = *(const HVX_Vector *)(vals + off + vi);
            if (vn < 32u) {
                const HVX_VectorPred q_tail = Q6_Q_vsetq2_R(vn * sizeof(float));
                vv = Q6_V_vmux_QVV(q_tail, vv, v_ninf);
            }
            const HVX_VectorPred q_gt = Q6_Q_vcmp_gt_VsfVsf(vv, hvx_vec_splat_f32(thr));
            uint32_t mask = hvx_pred_i32_to_bits(q_gt, v_bit_lut);
            while (mask != 0u) {
                const uint32_t bit = (uint32_t) __builtin_ctz(mask);
                mask &= mask - 1u;
                union { int32_t i; float f; } u = { .i = Q6_R_vextract_VR(vv, (int) (bit * 4u)) };
                const float v = u.f;
                if (!(v > thr)) {
                    continue;
                }
                uint32_t pos = cnt < k ? cnt++ : k - 1u;
                while (pos > 0u && best_v[pos - 1u] < v) {
                    best_v[pos] = best_v[pos - 1u];
                    best_i[pos] = best_i[pos - 1u];
                    pos--;
                }
                best_v[pos] = v;
                best_i[pos] = (int32_t) (off + vi + bit);
                if (cnt == k) {
                    thr = best_v[k - 1u];
                }
            }
        }
    }

    if (cnt < k) {
        for (uint32_t off = 0; off < n && cnt < k; off += 32u) {
            const uint32_t vn = MIN(32u, n - off);
            const HVX_Vector vv = *(const HVX_Vector *)(vals + off);
            for (uint32_t j = 0; j < vn && cnt < k; j++) {
                union { int32_t i; float f; } u = { .i = Q6_R_vextract_VR(vv, (int) (j * 4u)) };
                if (u.f == -INFINITY) {
                    best_v[cnt] = -INFINITY;
                    best_i[cnt] = (int32_t) (off + j);
                    cnt++;
                }
            }
        }
        for (uint32_t off = 0; off < n && cnt < k; off += 32u) {
            const uint32_t vn = MIN(32u, n - off);
            const HVX_Vector vv = *(const HVX_Vector *)(vals + off);
            for (uint32_t j = 0; j < vn && cnt < k; j++) {
                union { int32_t i; float f; } u = { .i = Q6_R_vextract_VR(vv, (int) (j * 4u)) };
                if (isnan(u.f)) {
                    best_v[cnt] = u.f;
                    best_i[cnt] = (int32_t) (off + j);
                    cnt++;
                }
            }
        }
    }
    for (; cnt < k; cnt++) {
        best_v[cnt] = -INFINITY;
        best_i[cnt] = 0;
    }

    for (uint32_t j = 0; j < k; j++) {
        if (out_v) {
            out_v[j] = best_v[j];
        }
        out_i[j] = best_i[j];
    }
}

// O(N) HVX radix-narrowed top-k selection in-place over VTCM float buffer.
__attribute__((noinline))
static void top_k_select_hvx(
    float * restrict   buf,
    int32_t * restrict out_idx,
    float * restrict   out_val,
    int32_t * restrict cand_scratch,
    uint32_t           n,
    uint32_t           k,
    uint32_t           idx_base
) {
    const int32_t KEY_PAD  = (int32_t) 0x80000000u;
    const int32_t KEY_NINF = (int32_t) 0x807fffffu;
    const int32_t KEY_MAX  = (int32_t) 0x7fffffffu;

    HVX_Vector * restrict vbuf = (HVX_Vector *) buf;
    const uint32_t n_full = n >> 5;
    const uint32_t rem    = n & 31u;
    const uint32_t n_vec  = (n + 31u) >> 5;

    const HVX_Vector v_pad  = Q6_V_vsplat_R(KEY_PAD);
    const HVX_Vector v_ninf = Q6_V_vsplat_R(KEY_NINF);
    const HVX_Vector v_max  = Q6_V_vsplat_R(KEY_MAX);
    const HVX_Vector v_one  = Q6_V_vsplat_R(1);

    HVX_Vector v_min_all = v_max;
    HVX_Vector v_max_all = v_pad;
    HVX_Vector v_min_fin = v_max;
    HVX_Vector v_cnt_fin = Q6_V_vzero();

    for (uint32_t v = 0; v < n_full; v++) {
        const HVX_Vector vk = float_to_ord_key_vec(vbuf[v]);
        vbuf[v] = vk;
        v_min_all = Q6_Vw_vmin_VwVw(v_min_all, vk);
        v_max_all = Q6_Vw_vmax_VwVw(v_max_all, vk);
        const HVX_VectorPred q_fin = Q6_Q_vcmp_gt_VwVw(vk, v_ninf);
        v_cnt_fin = Q6_Vw_vadd_VwVw(v_cnt_fin, Q6_V_vand_QV(q_fin, v_one));
        v_min_fin = Q6_Vw_vmin_VwVw(v_min_fin, Q6_V_vmux_QVV(q_fin, vk, v_max));
    }
    if (rem > 0u) {
        const HVX_VectorPred q_tail = Q6_Q_vsetq2_R(rem * sizeof(int32_t));
        const HVX_Vector vk_raw = float_to_ord_key_vec(vbuf[n_full]);
        const HVX_Vector vk     = Q6_V_vmux_QVV(q_tail, vk_raw, v_pad);
        vbuf[n_full] = vk;
        v_min_all = Q6_Vw_vmin_VwVw(v_min_all, Q6_V_vmux_QVV(q_tail, vk_raw, v_max));
        v_max_all = Q6_Vw_vmax_VwVw(v_max_all, vk);
        const HVX_VectorPred q_fin = Q6_Q_vcmp_gt_VwVw(vk, v_ninf);
        v_cnt_fin = Q6_Vw_vadd_VwVw(v_cnt_fin, Q6_V_vand_QV(q_fin, v_one));
        v_min_fin = Q6_Vw_vmin_VwVw(v_min_fin, Q6_V_vmux_QVV(q_fin, vk, v_max));
    }

    #pragma clang loop unroll(full)
    for (int shift = 64; shift >= 4; shift >>= 1) {
        v_min_all = Q6_Vw_vmin_VwVw(v_min_all, Q6_V_vror_VR(v_min_all, shift));
        v_max_all = Q6_Vw_vmax_VwVw(v_max_all, Q6_V_vror_VR(v_max_all, shift));
        v_min_fin = Q6_Vw_vmin_VwVw(v_min_fin, Q6_V_vror_VR(v_min_fin, shift));
        v_cnt_fin = Q6_Vw_vadd_VwVw(v_cnt_fin, Q6_V_vror_VR(v_cnt_fin, shift));
    }

    const int32_t  min_key     = (int32_t)  Q6_R_vextract_VR(v_min_all, 0);
    const int32_t  max_key     = (int32_t)  Q6_R_vextract_VR(v_max_all, 0);
    const int32_t  min_fin_key = (int32_t)  Q6_R_vextract_VR(v_min_fin, 0);
    const uint32_t n_gt_ninf   = (uint32_t) Q6_R_vextract_VR(v_cnt_fin, 0);

    if (min_key == max_key) {
        init_indices_ramp(out_idx, k, idx_base);
        if (out_val) {
            const float fv = ord_key_to_float(max_key);
            for (uint32_t j = 0; j < k; j++) {
                out_val[j] = fv;
            }
        }
        return;
    }

    int32_t  lo     = min_key - 1;
    uint32_t cnt_lo = n;
    int32_t  hi     = max_key;
    uint32_t cnt_hi = 0;

    if (n_gt_ninf > k) {
        lo     = min_fin_key - 1;
        cnt_lo = n_gt_ninf;
    } else {
        hi     = KEY_NINF;
        cnt_hi = n_gt_ninf;
        if (min_key >= KEY_NINF) {
            lo = KEY_NINF - 1;
        }
    }

    while ((uint32_t) hi - (uint32_t) lo > 1u && (cnt_lo - cnt_hi) > 64u && cnt_hi < k) {
        const uint32_t span = (uint32_t) hi - (uint32_t) lo;
        int32_t p[7];
        #pragma clang loop unroll(full)
        for (int m = 0; m < 7; m++) {
            uint32_t step = (uint32_t) (((uint64_t) span * (uint32_t) (m + 1)) >> 3);
            if (step == 0u) {
                step = 1u;
            } else if (step >= span) {
                step = span - 1u;
            }
            p[m] = (int32_t) ((uint32_t) lo + step);
        }

        const HVX_Vector vp0 = Q6_V_vsplat_R(p[0]);
        const HVX_Vector vp1 = Q6_V_vsplat_R(p[1]);
        const HVX_Vector vp2 = Q6_V_vsplat_R(p[2]);
        const HVX_Vector vp3 = Q6_V_vsplat_R(p[3]);
        const HVX_Vector vp4 = Q6_V_vsplat_R(p[4]);
        const HVX_Vector vp5 = Q6_V_vsplat_R(p[5]);
        const HVX_Vector vp6 = Q6_V_vsplat_R(p[6]);

        HVX_Vector vc0 = Q6_V_vzero();
        HVX_Vector vc1 = Q6_V_vzero();
        HVX_Vector vc2 = Q6_V_vzero();
        HVX_Vector vc3 = Q6_V_vzero();
        HVX_Vector vc4 = Q6_V_vzero();
        HVX_Vector vc5 = Q6_V_vzero();
        HVX_Vector vc6 = Q6_V_vzero();

        for (uint32_t v = 0; v < n_vec; v++) {
            const HVX_Vector vk = vbuf[v];
            vc0 = Q6_Vw_vadd_VwVw(vc0, Q6_V_vand_QV(Q6_Q_vcmp_gt_VwVw(vk, vp0), v_one));
            vc1 = Q6_Vw_vadd_VwVw(vc1, Q6_V_vand_QV(Q6_Q_vcmp_gt_VwVw(vk, vp1), v_one));
            vc2 = Q6_Vw_vadd_VwVw(vc2, Q6_V_vand_QV(Q6_Q_vcmp_gt_VwVw(vk, vp2), v_one));
            vc3 = Q6_Vw_vadd_VwVw(vc3, Q6_V_vand_QV(Q6_Q_vcmp_gt_VwVw(vk, vp3), v_one));
            vc4 = Q6_Vw_vadd_VwVw(vc4, Q6_V_vand_QV(Q6_Q_vcmp_gt_VwVw(vk, vp4), v_one));
            vc5 = Q6_Vw_vadd_VwVw(vc5, Q6_V_vand_QV(Q6_Q_vcmp_gt_VwVw(vk, vp5), v_one));
            vc6 = Q6_Vw_vadd_VwVw(vc6, Q6_V_vand_QV(Q6_Q_vcmp_gt_VwVw(vk, vp6), v_one));
        }

        #pragma clang loop unroll(full)
        for (int shift = 64; shift >= 4; shift >>= 1) {
            vc0 = Q6_Vw_vadd_VwVw(vc0, Q6_V_vror_VR(vc0, shift));
            vc1 = Q6_Vw_vadd_VwVw(vc1, Q6_V_vror_VR(vc1, shift));
            vc2 = Q6_Vw_vadd_VwVw(vc2, Q6_V_vror_VR(vc2, shift));
            vc3 = Q6_Vw_vadd_VwVw(vc3, Q6_V_vror_VR(vc3, shift));
            vc4 = Q6_Vw_vadd_VwVw(vc4, Q6_V_vror_VR(vc4, shift));
            vc5 = Q6_Vw_vadd_VwVw(vc5, Q6_V_vror_VR(vc5, shift));
            vc6 = Q6_Vw_vadd_VwVw(vc6, Q6_V_vror_VR(vc6, shift));
        }

        const uint32_t c[7] = {
            (uint32_t) Q6_R_vextract_VR(vc0, 0),
            (uint32_t) Q6_R_vextract_VR(vc1, 0),
            (uint32_t) Q6_R_vextract_VR(vc2, 0),
            (uint32_t) Q6_R_vextract_VR(vc3, 0),
            (uint32_t) Q6_R_vextract_VR(vc4, 0),
            (uint32_t) Q6_R_vextract_VR(vc5, 0),
            (uint32_t) Q6_R_vextract_VR(vc6, 0),
        };

        #pragma clang loop unroll(full)
        for (int m = 0; m < 7; m++) {
            if (c[m] >= k) {
                lo     = p[m];
                cnt_lo = c[m];
            } else if (p[m] < hi) {
                hi     = p[m];
                cnt_hi = c[m];
            }
        }
    }

    const HVX_Vector v_bit_lut = *(const HVX_Vector *) top_k_bit_lut;
    const HVX_Vector v_hi      = Q6_V_vsplat_R(hi);
    uint32_t out_cnt = 0;
    uint32_t rem_k   = k - cnt_hi;

    if (rem_k == 0u) {
        for (uint32_t v = 0; v < n_vec && out_cnt < k; v++) {
            const HVX_Vector vk = vbuf[v];
            uint32_t mask = hvx_pred_i32_to_bits(Q6_Q_vcmp_gt_VwVw(vk, v_hi), v_bit_lut);
            const uint32_t base = idx_base + (v << 5);
            while (mask != 0u && out_cnt < k) {
                const uint32_t bit = (uint32_t) __builtin_ctz(mask);
                mask &= mask - 1u;
                if (out_val) {
                    out_val[out_cnt] = ord_key_to_float((int32_t) Q6_R_vextract_VR(vk, (int) (bit * 4u)));
                }
                out_idx[out_cnt] = (int32_t) (base + bit);
                out_cnt++;
            }
        }
        return;
    }

    if ((uint32_t) hi - (uint32_t) lo == 1u) {
        const float hi_f = out_val ? ord_key_to_float(hi) : 0.0f;
        for (uint32_t v = 0; v < n_vec && out_cnt < k; v++) {
            const HVX_Vector vk = vbuf[v];
            const HVX_VectorPred q_gt = Q6_Q_vcmp_gt_VwVw(vk, v_hi);
            const HVX_VectorPred q_eq = Q6_Q_vcmp_eq_VwVw(vk, v_hi);
            uint32_t mask_gt = hvx_pred_i32_to_bits(q_gt, v_bit_lut);
            uint32_t mask_eq = (rem_k > 0u) ? hvx_pred_i32_to_bits(q_eq, v_bit_lut) : 0u;
            const uint32_t base = idx_base + (v << 5);
            while (mask_gt != 0u) {
                const uint32_t bit = (uint32_t) __builtin_ctz(mask_gt);
                mask_gt &= mask_gt - 1u;
                if (out_val) {
                    out_val[out_cnt] = ord_key_to_float((int32_t) Q6_R_vextract_VR(vk, (int) (bit * 4u)));
                }
                out_idx[out_cnt] = (int32_t) (base + bit);
                out_cnt++;
            }
            while (mask_eq != 0u && rem_k > 0u) {
                const uint32_t bit = (uint32_t) __builtin_ctz(mask_eq);
                mask_eq &= mask_eq - 1u;
                if (out_val) {
                    out_val[out_cnt] = hi_f;
                }
                out_idx[out_cnt] = (int32_t) (base + bit);
                out_cnt++;
                rem_k--;
            }
        }
        return;
    }

    const HVX_Vector v_lo = Q6_V_vsplat_R(lo);
    int32_t * restrict cand_k = cand_scratch;
    int32_t * restrict cand_i = cand_scratch + 64;
    uint32_t m_cnt = 0;

    for (uint32_t v = 0; v < n_vec; v++) {
        const HVX_Vector vk = vbuf[v];
        const HVX_VectorPred q_gt_hi = Q6_Q_vcmp_gt_VwVw(vk, v_hi);
        const HVX_VectorPred q_gt_lo = Q6_Q_vcmp_gt_VwVw(vk, v_lo);
        const HVX_VectorPred q_mid   = Q6_Q_and_QQn(q_gt_lo, q_gt_hi);
        uint32_t mask_gt  = hvx_pred_i32_to_bits(q_gt_hi, v_bit_lut);
        uint32_t mask_mid = hvx_pred_i32_to_bits(q_mid, v_bit_lut);
        const uint32_t base = idx_base + (v << 5);
        while (mask_gt != 0u) {
            const uint32_t bit = (uint32_t) __builtin_ctz(mask_gt);
            mask_gt &= mask_gt - 1u;
            if (out_val) {
                out_val[out_cnt] = ord_key_to_float((int32_t) Q6_R_vextract_VR(vk, (int) (bit * 4u)));
            }
            out_idx[out_cnt] = (int32_t) (base + bit);
            out_cnt++;
        }
        while (mask_mid != 0u && m_cnt < 64u) {
            const uint32_t bit = (uint32_t) __builtin_ctz(mask_mid);
            mask_mid &= mask_mid - 1u;
            cand_k[m_cnt] = (int32_t) Q6_R_vextract_VR(vk, (int) (bit * 4u));
            cand_i[m_cnt] = (int32_t) (base + bit);
            m_cnt++;
        }
    }

    for (uint32_t j = m_cnt; j < 64u; j++) {
        cand_k[j] = KEY_PAD;
        cand_i[j] = 0;
    }

    sort64_i32_desc(cand_k, cand_i);

    for (uint32_t j = 0; j < rem_k && out_cnt < k; j++) {
        if (out_val) {
            out_val[out_cnt] = ord_key_to_float(cand_k[j]);
        }
        out_idx[out_cnt] = cand_i[j];
        out_cnt++;
    }
}

__attribute__((noinline))
static void sort32_f32_hvx(uint8_t * values, uint8_t * indices, enum ggml_sort_order order) {
    bitonic_sort_generic_hvx(values, indices, 1, order == GGML_SORT_ORDER_ASC);
}

__attribute__((noinline))
static void sort64_f32_hvx(uint8_t * values, uint8_t * indices, enum ggml_sort_order order) {
    bitonic_sort_generic_hvx(values, indices, 2, order == GGML_SORT_ORDER_ASC);
}

__attribute__((noinline))
static void sort128_f32_hvx(uint8_t * values, uint8_t * indices, enum ggml_sort_order order) {
    bitonic_sort_generic_hvx(values, indices, 4, order == GGML_SORT_ORDER_ASC);
}

__attribute__((noinline))
static void sort256_f32_hvx(uint8_t * values, uint8_t * indices, enum ggml_sort_order order) {
    bitonic_sort_generic_hvx(values, indices, 8, order == GGML_SORT_ORDER_ASC);
}

static void merge_runs(
    const float * restrict in_val0, const int32_t * restrict in_idx0, uint32_t n0,
    const float * restrict in_val1, const int32_t * restrict in_idx1, uint32_t n1,
    float * restrict out_val, int32_t * restrict out_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t out = 0;

    if (asc) {
        while (i0 < n0 && i1 < n1) {
            if (in_val0[i0] <= in_val1[i1]) {
                if (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } else {
                if (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    } else {
        while (i0 < n0 && i1 < n1) {
            if (in_val0[i0] >= in_val1[i1]) {
                if (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } else {
                if (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    }

    while (i0 < n0) {
        if (out_val) out_val[out] = in_val0[i0];
        out_idx[out] = in_idx0[i0];
        i0++;
        out++;
    }

    while (i1 < n1) {
        if (out_val) out_val[out] = in_val1[i1];
        out_idx[out] = in_idx1[i1];
        i1++;
        out++;
    }
}

// Expand fp16 values in-place to fp32 backwards from the last vector so lower fp16 vectors are read before overwrite.
__attribute__((noinline))
static void sort_unpack_f16_to_f32_inplace(float * restrict values, uint32_t count) {
    const HVX_Vector * vsrc = (const HVX_Vector *) values;
    HVX_Vector *       vdst = (HVX_Vector *) values;
    const uint32_t n_vec_f32 = (count + 31) / 32;
    const uint32_t n_vec_f16 = (n_vec_f32 + 1) / 2;

    for (int32_t v = (int32_t) n_vec_f16 - 1; v >= 0; v--) {
        const HVX_VectorPair p = hvx_vec_f16_to_f32(vsrc[v]);
        if ((uint32_t) (2 * v + 1) < n_vec_f32) {
            vdst[2 * v + 1] = Q6_V_hi_W(p);
        }
        vdst[2 * v] = Q6_V_lo_W(p);
    }
}

struct htp_sort_chunk_ctx {
    struct htp_ops_context *              octx;
    const struct htp_sort_kernel_params * kparams;
    uint8_t *                             phase1_base;
    size_t                                phase1_slot_size;
    size_t                                off_cand;
    float *                               merge_values;
    int32_t *                             merge_indices;
    uint32_t                              real_count[HTP_OP_MAX_NHVX];
    uint32_t                              cand_count[HTP_OP_MAX_NHVX];
};

static void htp_sort_chunk_job(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    struct htp_sort_chunk_ctx * cctx = (struct htp_sort_chunk_ctx *) data;
    struct htp_ops_context * octx = cctx->octx;
    const struct htp_sort_kernel_params * kparams = cctx->kparams;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = src0->ne[0];
    uint32_t chunk_base  = i * chunk_elems;

    uint32_t real_count = (chunk_base < ne00) ? MIN(chunk_elems, ne00 - chunk_base) : 0;
    cctx->real_count[i] = real_count;

    uint8_t * spad = cctx->phase1_base + cctx->phase1_slot_size * i;
    float *   values_buf  = (float *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * sizeof(float));

    float *   merge_values  = cctx->merge_values;
    int32_t * merge_indices = cctx->merge_indices;

    struct htp_thread_trace * tr = th->trace;
    const bool src0_f16 = (src0->type == HTP_TYPE_F16);
    const size_t src_elem_size = src0_f16 ? sizeof(__fp16) : sizeof(float);
    const dma_addr_t src_addr = src0->data + (size_t) chunk_base * src_elem_size;

    if (kparams->is_top_k) {
        const uint32_t k       = dst->ne[0];
        const uint32_t local_k = MIN(k, chunk_elems);
        const uint32_t take_k  = MIN(local_k, real_count);
        cctx->cand_count[i]    = take_k;

        if (take_k == 0u) {
            return;
        }

        float *   out_val      = &merge_values[i * local_k];
        int32_t * out_idx      = &merge_indices[i * local_k];
        int32_t * cand_scratch = (int32_t *) (spad + cctx->off_cand);

        dma_sync_read(th->dma_q, values_buf, src_addr, real_count * src_elem_size);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        if (src0_f16) {
            sort_unpack_f16_to_f32_inplace(values_buf, real_count);
        }

        if (take_k == real_count) {
            for (uint32_t j = 0; j < take_k; j++) {
                out_val[j] = values_buf[j];
                out_idx[j] = (int32_t) (chunk_base + j);
            }
        } else if (take_k == 1u) {
            out_idx[0] = top_k_argmax_hvx(values_buf, real_count, chunk_base, &out_val[0]);
        } else if (take_k <= 64u && chunk_elems > 1024u) {
            top_k_scan(values_buf, real_count, take_k, out_val, out_idx);
            for (uint32_t j = 0; j < take_k; j++) {
                out_idx[j] += (int32_t) chunk_base;
            }
        } else {
            top_k_select_hvx(values_buf, out_idx, out_val, cand_scratch, real_count, take_k, chunk_base);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } else {
        if (real_count > 0) {
            dma_sync_read(th->dma_q, values_buf, src_addr, real_count * src_elem_size);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            if (src0_f16) {
                sort_unpack_f16_to_f32_inplace(values_buf, real_count);
            }
            init_indices_ramp(indices_buf, real_count, chunk_base);

            if (kparams->order == GGML_SORT_ORDER_ASC) {
                quicksort_values_indices_asc(values_buf, indices_buf, 0, real_count - 1);
            } else {
                quicksort_values_indices_desc(values_buf, indices_buf, 0, real_count - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

struct htp_argsort_merge_l1_ctx {
    struct htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

static void htp_argsort_merge_l1_job(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    struct htp_argsort_merge_l1_ctx * mctx = (struct htp_argsort_merge_l1_ctx *) data;
    struct htp_sort_chunk_ctx * cctx = mctx->cctx;
    const struct htp_sort_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_count[c0];
    uint32_t count1 = cctx->real_count[c1];

    size_t slot_size = cctx->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    const float *   val0 = (const float *)   (cctx->phase1_base + slot_size * c0);
    const int32_t * idx0 = (const int32_t *) (cctx->phase1_base + slot_size * c0 + chunk_elems * sizeof(float));
    const float *   val1 = (const float *)   (cctx->phase1_base + slot_size * c1);
    const int32_t * idx1 = (const int32_t *) (cctx->phase1_base + slot_size * c1 + chunk_elems * sizeof(float));

    uint32_t out_offset = (2 * i) * chunk_elems;
    float *   out_val = cctx->merge_values  + out_offset;
    int32_t * out_idx = cctx->merge_indices + out_offset;

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

struct htp_argsort_merge_l2_ctx {
    struct htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

static void htp_argsort_merge_l2_job(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    struct htp_argsort_merge_l2_ctx * mctx = (struct htp_argsort_merge_l2_ctx *) data;
    struct htp_sort_chunk_ctx * cctx = mctx->cctx;
    const struct htp_sort_kernel_params * kparams = cctx->kparams;

    size_t slot_size = cctx->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    const float *   val0 = cctx->merge_values  + off0;
    const int32_t * idx0 = cctx->merge_indices + off0;
    const float *   val1 = cctx->merge_values  + off1;
    const int32_t * idx1 = cctx->merge_indices + off1;

    uint32_t count0 = cctx->real_count[4*i + 0] + cctx->real_count[4*i + 1];
    uint32_t count1 = cctx->real_count[4*i + 2] + cctx->real_count[4*i + 3];

    uint8_t * slot_base = cctx->phase1_base + (4 * i) * slot_size;
    float *   out_val   = (float *) slot_base;
    int32_t * out_idx   = (int32_t *) (slot_base + 4 * chunk_elems * sizeof(float));

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

struct htp_sort_multi_row_ctx {
    struct htp_ops_context *              octx;
    const struct htp_sort_kernel_params * kparams;
    uint8_t *                             phase1_base;
    size_t                                phase1_slot_size;
    size_t                                row_values_size;
    size_t                                row_indices_size;
    size_t                                off_cand;
    uint32_t                              batch_rows;
    uint32_t                              row_start;
    uint32_t                              row_end;
    uint32_t                              nrows_per_hvx;
};

struct htp_tensor_row_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

static inline struct htp_tensor_row_coord init_row_coord(
    uint32_t r,
    uint32_t ne01,
    uint32_t ne02,
    const struct fastdiv_values * div_ne01,
    const struct fastdiv_values * div_ne01_ne02
) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? fastdiv(r, div_ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r - i03 * ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? fastdiv(rem, div_ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem - i02 * ne01) : 0;
    return (struct htp_tensor_row_coord){ i01, i02, i03 };
}

static inline void step_row_coord(struct htp_tensor_row_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    if (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        if (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

static inline size_t coord_offset(struct htp_tensor_row_coord c, size_t nb1, size_t nb2, size_t nb3) {
    return (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

static inline void compute_row_sort(
    float * restrict     cur_values,
    int32_t * restrict   cur_indices,
    int32_t * restrict   cand_scratch,
    uint32_t             ne00,
    uint32_t             chunk_elems,
    bool                 src0_f16,
    bool                 is_top_k,
    enum ggml_sort_order order,
    uint32_t             k
) {
    if (src0_f16) {
        sort_unpack_f16_to_f32_inplace(cur_values, ne00);
    }
    if (is_top_k) {
        if (k == 1u) {
            cur_indices[0] = top_k_argmax_hvx(cur_values, ne00, 0, NULL);
        } else if (ne00 <= 256u) {
            switch (chunk_elems) {
                case 32:  sort32_top_k_hvx(cur_values, cur_indices, ne00, k);  break;
                case 64:  sort64_top_k_hvx(cur_values, cur_indices, ne00, k);  break;
                case 128: sort128_top_k_hvx(cur_values, cur_indices, ne00, k); break;
                default:  sort256_top_k_hvx(cur_values, cur_indices, ne00, k); break;
            }
        } else if (k <= 64u && ne00 > 1024u) {
            top_k_scan(cur_values, ne00, k, NULL, cur_indices);
        } else {
            top_k_select_hvx(cur_values, cur_indices, NULL, cand_scratch, ne00, k, 0);
        }
    } else {
        if (ne00 == 32u || ne00 == 64u || ne00 == 128u || ne00 == 256u) {
            switch (ne00) {
                case 32:  sort32_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, order);  break;
                case 64:  sort64_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, order);  break;
                case 128: sort128_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, order); break;
                case 256: sort256_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, order); break;
            }
        } else if (ne00 == 512u || ne00 == 1024u) {
            bitonic_sort_vtcm((uint8_t *) cur_values, (uint8_t *) cur_indices, ne00 >> 5, true, order == GGML_SORT_ORDER_ASC);
        } else {
            init_indices_ramp(cur_indices, ne00, 0);
            if (order == GGML_SORT_ORDER_ASC) {
                quicksort_values_indices_asc(cur_values, cur_indices, 0, (int) ne00 - 1);
            } else {
                quicksort_values_indices_desc(cur_values, cur_indices, 0, (int) ne00 - 1);
            }
        }
    }
}

static void htp_sort_multi_row_job(const htp_thread_t * th, unsigned int n, unsigned int ith, void * data) {
    struct htp_sort_multi_row_ctx * mctx = (struct htp_sort_multi_row_ctx *) data;
    struct htp_ops_context * octx = mctx->octx;
    const struct htp_sort_kernel_params * kparams = mctx->kparams;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    uint32_t ne00 = src0->ne[0];
    uint32_t ne01 = src0->ne[1];
    uint32_t ne02 = src0->ne[2];

    uint32_t nb01 = src0->nb[1];
    uint32_t nb02 = src0->nb[2];
    uint32_t nb03 = src0->nb[3];

    uint32_t nb1 = dst->nb[1];
    uint32_t nb2 = dst->nb[2];
    uint32_t nb3 = dst->nb[3];

    uint32_t row_start = mctx->row_start;
    uint32_t row_end   = mctx->row_end;
    uint32_t r_start   = row_start + ith * mctx->nrows_per_hvx;
    uint32_t r_end     = MIN(r_start + mctx->nrows_per_hvx, row_end);

    if (r_start >= r_end) {
        return;
    }

    const size_t   slot_size        = mctx->phase1_slot_size;
    const size_t   row_values_size  = mctx->row_values_size;
    const size_t   row_indices_size = mctx->row_indices_size;
    const size_t   off_cand         = mctx->off_cand;
    const uint32_t batch_rows       = mctx->batch_rows;
    const size_t   values_area_size = row_values_size * (size_t) batch_rows;

    const size_t spad_offset = ith * slot_size * (size_t) kparams->n_slots;
    uint8_t * hvx_spad = mctx->phase1_base + spad_offset;

    const uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    dma_queue * dma_q = th->dma_q;
    struct htp_thread_trace * tr = th->trace;

    const bool src0_f16 = (src0->type == HTP_TYPE_F16);
    const bool is_top_k = kparams->is_top_k != 0;
    const uint32_t k = dst->ne[0];
    const enum ggml_sort_order order = (enum ggml_sort_order) kparams->order;

    const uint32_t src_bytes = ne00 * (src0_f16 ? sizeof(__fp16) : sizeof(float));
    const uint32_t dst_bytes = (is_top_k ? k : ne00) * sizeof(int32_t);

    if (kparams->n_slots == 2) {
        struct htp_tensor_row_coord next_coord = init_row_coord(r_start, ne01, ne02, &kparams->div_ne01, &kparams->div_ne01_ne02);
        struct htp_tensor_row_coord slot_coord[2];
        uint32_t slot_cnt[2] = { 0, 0 };
        uint32_t slot_r[2]   = { 0, 0 };
        uint32_t next_r      = r_start;

        for (uint32_t s = 0; s < 2 && next_r < r_end; s++) {
            const uint32_t b_cnt = MIN(batch_rows, MIN(r_end - next_r, ne01 - next_coord.i01));
            uint8_t * cur_spad         = hvx_spad + s * slot_size;
            float *   cur_values_base  = (float *) cur_spad;
            int32_t * cur_indices_base = (int32_t *) (cur_spad + values_area_size);

            const dma_addr_t dst_addr = dst->data + coord_offset(next_coord, nb1, nb2, nb3);
            const dma_addr_t src_addr = src0->data + coord_offset(next_coord, nb01, nb02, nb03);

            dma_queue_push(dma_q, dma_make_data(dst_addr, cur_indices_base),
                           nb1, row_indices_size, dst_bytes, 0);
            dma_queue_push(dma_q, dma_make_data(cur_values_base, src_addr),
                           row_values_size, nb01, src_bytes, b_cnt);

            slot_cnt[s]   = b_cnt;
            slot_r[s]     = next_r;
            slot_coord[s] = next_coord;

            next_r += b_cnt;
            for (uint32_t b = 0; b < b_cnt; b++) {
                step_row_coord(&next_coord, ne01, ne02);
            }
        }

        uint32_t cur_slot = 0;
        while (slot_cnt[cur_slot] > 0) {
            const uint32_t b_cnt       = slot_cnt[cur_slot];
            const uint32_t base_r      = slot_r[cur_slot];
            uint8_t * cur_spad         = hvx_spad + cur_slot * slot_size;
            float *   cur_values_base  = (float *) cur_spad;
            int32_t * cur_indices_base = (int32_t *) (cur_spad + values_area_size);
            int32_t * cand_scratch     = (int32_t *) (cur_spad + off_cand);

            dma_queue_pop(dma_q);
            dma_queue_pop(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) base_r);
            for (uint32_t b = 0; b < b_cnt; b++) {
                float *   row_val = (float *)   (cur_spad + (size_t) b * row_values_size);
                int32_t * row_idx = (int32_t *) (cur_spad + values_area_size + (size_t) b * row_indices_size);
                compute_row_sort(row_val, row_idx, cand_scratch, ne00, chunk_elems, src0_f16, is_top_k, order, k);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) base_r);

            const dma_addr_t dst_addr = dst->data + coord_offset(slot_coord[cur_slot], nb1, nb2, nb3);
            dma_queue_push(dma_q, dma_make_data(dst_addr, cur_indices_base),
                           nb1, row_indices_size, dst_bytes, b_cnt);

            if (next_r < r_end) {
                const uint32_t next_b = MIN(batch_rows, MIN(r_end - next_r, ne01 - next_coord.i01));
                const dma_addr_t next_src_addr = src0->data + coord_offset(next_coord, nb01, nb02, nb03);
                dma_queue_push(dma_q, dma_make_data(cur_values_base, next_src_addr),
                               row_values_size, nb01, src_bytes, next_b);
                slot_cnt[cur_slot]   = next_b;
                slot_r[cur_slot]     = next_r;
                slot_coord[cur_slot] = next_coord;

                next_r += next_b;
                for (uint32_t b = 0; b < next_b; b++) {
                    step_row_coord(&next_coord, ne01, ne02);
                }
            } else {
                slot_cnt[cur_slot] = 0;
            }

            cur_slot ^= 1u;
        }

        dma_queue_flush(dma_q);
    } else {
        struct htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02, &kparams->div_ne01, &kparams->div_ne01_ne02);
        int32_t * cand_scratch = (int32_t *) (hvx_spad + off_cand);
        for (uint32_t r = r_start; r < r_end; ) {
            const uint32_t b_cnt = MIN(batch_rows, MIN(r_end - r, ne01 - coord.i01));
            const dma_addr_t src_addr = src0->data + coord_offset(coord, nb01, nb02, nb03);
            const dma_addr_t dst_addr = dst->data + coord_offset(coord, nb1, nb2, nb3);

            dma_queue_push(dma_q, dma_make_data(hvx_spad, src_addr),
                           row_values_size, nb01, src_bytes, b_cnt);
            dma_queue_flush(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            for (uint32_t b = 0; b < b_cnt; b++) {
                float *   row_val = (float *)   (hvx_spad + (size_t) b * row_values_size);
                int32_t * row_idx = (int32_t *) (hvx_spad + values_area_size + (size_t) b * row_indices_size);
                compute_row_sort(row_val, row_idx, cand_scratch, ne00, chunk_elems, src0_f16, is_top_k, order, k);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_queue_push(dma_q, dma_make_data(dst_addr, hvx_spad + values_area_size),
                           nb1, row_indices_size, dst_bytes, b_cnt);
            dma_queue_flush(dma_q);

            r += b_cnt;
            for (uint32_t b = 0; b < b_cnt; b++) {
                step_row_coord(&coord, ne01, ne02);
            }
        }
    }
}

struct htp_top_k_identity_ctx {
    struct htp_ops_context *              octx;
    const struct htp_sort_kernel_params * kparams;
    uint8_t *                             phase1_base;
    size_t                                row_indices_size;
    uint32_t                              batch_rows;
    uint32_t                              row_start;
    uint32_t                              row_end;
    uint32_t                              nrows_per_hvx;
};

static void htp_top_k_identity_job(const htp_thread_t * th, unsigned int n, unsigned int ith, void * data) {
    const struct htp_top_k_identity_ctx * ictx = (const struct htp_top_k_identity_ctx *) data;
    const struct htp_ops_context * octx = ictx->octx;
    const struct htp_sort_kernel_params * kparams = ictx->kparams;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t ne01 = src0->ne[1];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t nb1  = dst->nb[1];
    const uint32_t nb2  = dst->nb[2];
    const uint32_t nb3  = dst->nb[3];

    const uint32_t r_start = ictx->row_start + ith * ictx->nrows_per_hvx;
    const uint32_t r_end   = MIN(r_start + ictx->nrows_per_hvx, ictx->row_end);
    if (r_start >= r_end) {
        return;
    }

    const uint32_t dst_bytes  = dst->ne[0] * sizeof(int32_t);
    const size_t   row_stride = ictx->row_indices_size;
    const uint32_t batch_rows = ictx->batch_rows;
    dma_queue * dma_q = th->dma_q;

    struct htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02, &kparams->div_ne01, &kparams->div_ne01_ne02);
    for (uint32_t r = r_start; r < r_end; ) {
        const uint32_t b_cnt = MIN(batch_rows, MIN(r_end - r, ne01 - coord.i01));
        const dma_addr_t dst_addr = dst->data + coord_offset(coord, nb1, nb2, nb3);

        if (dma_ring_depth(dma_q->ring0) >= 8u) {
            dma_queue_pop(dma_q);
        }
        dma_queue_push(dma_q, dma_make_data(dst_addr, ictx->phase1_base),
                       nb1, row_stride, dst_bytes, b_cnt);

        r += b_cnt;
        for (uint32_t b = 0; b < b_cnt; b++) {
            step_row_coord(&coord, ne01, ne02);
        }
    }
    dma_queue_flush(dma_q);
}

static int op_sort_common(struct htp_ops_context * octx, bool is_top_k) {
    if (octx->src[0]->type != HTP_TYPE_F32 && octx->src[0]->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    uint32_t row_start = 0;
    uint32_t row_end   = total_rows;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        row_end   = range.start + range.count;
    }

    const uint32_t nrows = row_end - row_start;
    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    const struct htp_sort_kernel_params * kparams = (const struct htp_sort_kernel_params *) octx->kernel_params;

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    struct htp_sort_vtcm_layout layout;
    htp_sort_vtcm_layout_build(&layout,
                               ne00,
                               total_rows,
                               (uint32_t) kparams->chunk_elems,
                               (uint32_t) kparams->n_chunks,
                               (uint32_t) kparams->merge_elems,
                               (uint32_t) kparams->n_hvx,
                               (uint32_t) kparams->n_slots,
                               k,
                               is_top_k);

    if (octx->ctx->vtcm_size < layout.total_bytes) {
        FARF(ERROR, "sort: VTCM size too small. Needed %zu, have %zu", layout.total_bytes, octx->ctx->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base     = (uint8_t *) octx->ctx->vtcm_base;
    uint8_t * phase1_base   = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, layout.off_phase1);
    float *   merge_values  = VTCM_LAYOUT_PTR(float,   vtcm_base, layout.off_merge_values);
    int32_t * merge_indices = VTCM_LAYOUT_PTR(int32_t, vtcm_base, layout.off_merge_indices);

    if (is_top_k && k >= ne00) {
        init_indices_ramp((int32_t *) phase1_base, k, 0);
        const uint32_t n_idx_vecs = (uint32_t) (layout.row_indices_size >> 7);
        const HVX_Vector * restrict vsrc0 = (const HVX_Vector *) phase1_base;
        for (uint32_t b = 1; b < layout.batch_rows; b++) {
            HVX_Vector * restrict vdst_b = (HVX_Vector *) (phase1_base + (size_t) b * layout.row_indices_size);
            for (uint32_t v = 0; v < n_idx_vecs; v++) {
                vdst_b[v] = vsrc0[v];
            }
        }

        const uint32_t dr = fastdiv(nrows + octx->n_hvx - 1, &octx->n_hvx_div);

        struct htp_top_k_identity_ctx ictx;
        ictx.octx             = octx;
        ictx.kparams          = kparams;
        ictx.phase1_base      = phase1_base;
        ictx.row_indices_size = layout.row_indices_size;
        ictx.batch_rows       = layout.batch_rows;
        ictx.row_start        = row_start;
        ictx.row_end          = row_end;
        ictx.nrows_per_hvx    = dr;

        work_queue_run(octx->ctx->hvx_queue, htp_top_k_identity_job, &ictx, octx->n_hvx);
        return HTP_STATUS_OK;
    }

    if (total_rows == 1 && kparams->n_chunks > 1) {
        struct htp_sort_chunk_ctx cctx;
        cctx.octx             = octx;
        cctx.kparams          = kparams;
        cctx.phase1_base      = phase1_base;
        cctx.phase1_slot_size = layout.phase1_slot_size;
        cctx.off_cand         = layout.off_cand;
        cctx.merge_values     = merge_values;
        cctx.merge_indices    = merge_indices;
        memset(cctx.real_count, 0, sizeof(cctx.real_count));
        memset(cctx.cand_count, 0, sizeof(cctx.cand_count));

        work_queue_run(octx->ctx->hvx_queue, htp_sort_chunk_job, &cctx, (uint32_t) kparams->n_chunks);

        if (is_top_k) {
            const uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            const uint32_t local_k     = MIN(k, chunk_elems);

            struct htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);

            if (k == 1u) {
                int32_t best_key = (int32_t) 0x80000000u;
                int32_t best_idx = 0;
                for (uint32_t c = 0; c < (uint32_t) kparams->n_chunks; c++) {
                    if (cctx.cand_count[c] == 0u) {
                        continue;
                    }
                    union { float f; uint32_t u; int32_t i; } u = { .f = merge_values[c * local_k] };
                    const int32_t key = ((u.u & 0x7fffffffu) > 0x7f800000u)
                                        ? (int32_t) 0x80000001u
                                        : ((u.i < 0) ? (int32_t) (u.u ^ 0x7fffffffu) : u.i);
                    if (key > best_key) {
                        best_key = key;
                        best_idx = merge_indices[c * local_k];
                    }
                }
                merge_indices[0] = best_idx;
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, merge_indices, sizeof(int32_t));
                return HTP_STATUS_OK;
            }

            uint32_t total_candidates = 0;
            for (uint32_t c = 0; c < (uint32_t) kparams->n_chunks; c++) {
                const uint32_t cnt  = cctx.cand_count[c];
                const uint32_t base = c * local_k;
                if (total_candidates != base) {
                    for (uint32_t j = 0; j < cnt; j++) {
                        merge_values[total_candidates + j]  = merge_values[base + j];
                        merge_indices[total_candidates + j] = merge_indices[base + j];
                    }
                }
                total_candidates += cnt;
            }

            int32_t * pos_idx      = (int32_t *) phase1_base;
            const size_t pos_bytes = ((size_t) k * sizeof(int32_t) + 127u) & ~(size_t) 127u;
            int32_t * cand_scratch = (int32_t *) (phase1_base + pos_bytes);

            top_k_select_hvx(merge_values, pos_idx, NULL, cand_scratch, total_candidates, k, 0);

            int32_t * final_idx = (int32_t *) merge_values;
            for (uint32_t t = 0; t < k; t++) {
                final_idx[t] = merge_indices[pos_idx[t]];
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, k * sizeof(int32_t));
        } else {
            bool asc = kparams->order == GGML_SORT_ORDER_ASC;
            if (kparams->n_chunks == 8) {
                struct htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->hvx_queue, htp_argsort_merge_l1_job, &l1ctx, 4);

                struct htp_argsort_merge_l2_ctx l2ctx = { &cctx, asc };
                work_queue_run(octx->ctx->hvx_queue, htp_argsort_merge_l2_job, &l2ctx, 2);

                size_t slot_size = layout.phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                const float *   val0 = (const float *) phase1_base;
                const int32_t * idx0 = (const int32_t *) (phase1_base + 4 * chunk_elems * sizeof(float));
                const float *   val1 = (const float *) (phase1_base + 4 * slot_size);
                const int32_t * idx1 = (const int32_t *) (phase1_base + 4 * slot_size + 4 * chunk_elems * sizeof(float));

                uint32_t n01 = cctx.real_count[0] + cctx.real_count[1] + cctx.real_count[2] + cctx.real_count[3];
                uint32_t n23 = cctx.real_count[4] + cctx.real_count[5] + cctx.real_count[6] + cctx.real_count[7];

                int32_t * final_idx = merge_indices;
                struct htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, n01, val1, idx1, n23, NULL, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * sizeof(int32_t));
            } else if (kparams->n_chunks == 4) {
                struct htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->hvx_queue, htp_argsort_merge_l1_job, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_count[0] + cctx.real_count[1];
                uint32_t n23 = cctx.real_count[2] + cctx.real_count[3];

                const float *   val01 = merge_values;
                const int32_t * idx01 = merge_indices;
                const float *   val23 = merge_values  + 2 * chunk_elems;
                const int32_t * idx23 = merge_indices + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) phase1_base;
                struct htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val01, idx01, n01, val23, idx23, n23, NULL, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * sizeof(int32_t));
            } else if (kparams->n_chunks == 2) {
                size_t slot_size = layout.phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                const float *   val0 = (const float *)   (phase1_base);
                const int32_t * idx0 = (const int32_t *) (phase1_base + chunk_elems * sizeof(float));
                const float *   val1 = (const float *)   (phase1_base + slot_size);
                const int32_t * idx1 = (const int32_t *) (phase1_base + slot_size + chunk_elems * sizeof(float));

                int32_t * final_idx = merge_indices;
                struct htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, cctx.real_count[0], val1, idx1, cctx.real_count[1], NULL, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * sizeof(int32_t));
            } else {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (phase1_base + chunk_elems * sizeof(float));
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * sizeof(int32_t));
            }
        }
        return HTP_STATUS_OK;
    }

    struct htp_sort_multi_row_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.phase1_base      = phase1_base;
    mctx.phase1_slot_size = layout.phase1_slot_size;
    mctx.row_values_size  = layout.row_values_size;
    mctx.row_indices_size = layout.row_indices_size;
    mctx.off_cand         = layout.off_cand;
    mctx.batch_rows       = layout.batch_rows;
    mctx.row_start        = row_start;
    mctx.row_end          = row_end;

    mctx.nrows_per_hvx = fastdiv(nrows + octx->n_hvx - 1, &octx->n_hvx_div);

    work_queue_run(octx->ctx->hvx_queue, htp_sort_multi_row_job, &mctx, octx->n_hvx);

    return HTP_STATUS_OK;
}

int op_argsort(struct htp_ops_context * octx) {
    return op_sort_common(octx, false);
}

int op_top_k(struct htp_ops_context * octx) {
    return op_sort_common(octx, true);
}
