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

static inline uint32_t top_k_max_value_index(const float * values, uint32_t n, float * value) {
    int32_t max_idx = 0;
    hvx_argmax_f32(values, n, 0, value, &max_idx);
    return (uint32_t) max_idx;
}

static const int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

static inline void init_indices_ramp(int32_t * indices_buf, uint32_t count, uint32_t base) {
    const HVX_Vector ind_init_vec = Q6_Vw_vadd_VwVw(*(const HVX_Vector *) argsort_ramp_lut, Q6_V_vsplat_R(base));
    const HVX_Vector ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_Vector * indices_buf_vec = (HVX_Vector *) indices_buf;
    uint32_t num_vecs = count / 32;
    HVX_Vector curr_ind_vec = ind_init_vec;
    for (uint32_t j = 0; j < num_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = count % 32;
    if (rem > 0) {
        hvx_vec_store_a((void *) &indices_buf_vec[num_vecs], rem * sizeof(int32_t), curr_ind_vec);
    }
}

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
    } else {
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
    HVX_Vector idx_vec = *(const HVX_Vector *) argsort_ramp_lut;

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

    for (int v = 0; v < K; v++) {
        *(HVX_Vector *)(values + v * 128)  = V[v];
        *(HVX_Vector *)(indices + v * 128) = I[v];
    }
}

static void bitonic_sort_vtcm_desc(uint8_t * values, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(const HVX_Vector *) argsort_ramp_lut;

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    if (init_indices) {
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

static void top_k_select_tiled(dma_queue * dma_q, struct htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               float * values_buf, int32_t * indices_buf,
                               float * out_values, int32_t * out_indices,
                               uint32_t chunk_base) {
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
        dma_sync_read(dma_q, values_buf, src_addr + (size_t) offset * sizeof(float), tile_count * sizeof(float));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
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
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    if (merge_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_values + candidate_pos), -INFINITY, merge_elems - candidate_pos);
        for (uint32_t j = candidate_pos; j < merge_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_sort_vtcm_desc((uint8_t *) candidate_values, (uint8_t *) candidate_indices, merge_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    for (uint32_t j = 0; j < k; j++) {
        out_values[j]  = candidate_values[j];
        out_indices[j] = candidate_indices[j] + (int32_t) chunk_base;
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

struct htp_sort_chunk_ctx {
    struct htp_ops_context *              octx;
    const struct htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              real_count[HTP_MAX_NTHREADS];
};

static void htp_sort_chunk_job(unsigned int n, unsigned int i, void * data) {
    struct htp_sort_chunk_ctx * cctx = (struct htp_sort_chunk_ctx *) data;
    struct htp_ops_context * octx = cctx->octx;
    const struct htp_sort_kernel_params * kparams = cctx->kparams;
    const struct htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_base  = i * chunk_elems;

    uint32_t real_count = (chunk_base < ne00) ? MIN(chunk_elems, ne00 - chunk_base) : 0;
    cctx->real_count[i] = real_count;

    uint8_t * spad = cctx->vtcm_base + (size_t) kparams->phase1_slot_size * i;
    float *   values_buf  = (float *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * sizeof(float));

    float *   merge_values  = (float *)   (cctx->vtcm_base + kparams->merge_values_off);
    int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off);

    struct htp_thread_trace * tr = &octx->ctx->trace[i];
    const dma_addr_t src_addr = src0->data + (size_t) chunk_base * sizeof(float);

    if (kparams->is_top_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        if (real_count == 0) {
            for (uint32_t j = 0; j < local_k; j++) {
                merge_values[i * local_k + j]  = -INFINITY;
                merge_indices[i * local_k + j] = 0;
            }
            return;
        }

        if (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], values_buf, src_addr, real_count * sizeof(float));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            float max_value;
            uint32_t max_index = top_k_max_value_index(values_buf, real_count, &max_value);
            merge_values[i] = max_value;
            merge_indices[i] = (int32_t) (max_index + chunk_base);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            return;
        }

        if (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            top_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_count, local_k,
                               values_buf, indices_buf,
                               &merge_values[i * local_k], &merge_indices[i * local_k],
                               chunk_base);
            return;
        }

        dma_sync_read(octx->ctx->dma[i], values_buf, src_addr, real_count * sizeof(float));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        if (chunk_elems > real_count) {
            hvx_splat_f32_u((uint8_t *)(values_buf + real_count), -INFINITY, chunk_elems - real_count);
        }

        bitonic_sort_vtcm_desc((uint8_t *) values_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        for (uint32_t j = 0; j < local_k; j++) {
            merge_values[i * local_k + j]  = values_buf[j];
            merge_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_base;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } else {
        if (real_count > 0) {
            dma_sync_read(octx->ctx->dma[i], values_buf, src_addr, real_count * sizeof(float));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
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

static void htp_argsort_merge_l1_job(unsigned int n, unsigned int i, void * data) {
    struct htp_argsort_merge_l1_ctx * mctx = (struct htp_argsort_merge_l1_ctx *) data;
    struct htp_sort_chunk_ctx * cctx = mctx->cctx;
    const struct htp_sort_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_count[c0];
    uint32_t count1 = cctx->real_count[c1];

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    const float *   val0 = (const float *)   (cctx->vtcm_base + slot_size * c0);
    const int32_t * idx0 = (const int32_t *) (cctx->vtcm_base + slot_size * c0 + chunk_elems * sizeof(float));
    const float *   val1 = (const float *)   (cctx->vtcm_base + slot_size * c1);
    const int32_t * idx1 = (const int32_t *) (cctx->vtcm_base + slot_size * c1 + chunk_elems * sizeof(float));

    uint32_t out_offset = (2 * i) * chunk_elems;
    float *   out_val = (float *)   (cctx->vtcm_base + kparams->merge_values_off)  + out_offset;
    int32_t * out_idx = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + out_offset;

    struct htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

struct htp_argsort_merge_l2_ctx {
    struct htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

static void htp_argsort_merge_l2_job(unsigned int n, unsigned int i, void * data) {
    struct htp_argsort_merge_l2_ctx * mctx = (struct htp_argsort_merge_l2_ctx *) data;
    struct htp_sort_chunk_ctx * cctx = mctx->cctx;
    const struct htp_sort_kernel_params * kparams = cctx->kparams;

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    const float *   val0 = (const float *)   (cctx->vtcm_base + kparams->merge_values_off)  + off0;
    const int32_t * idx0 = (const int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off0;
    const float *   val1 = (const float *)   (cctx->vtcm_base + kparams->merge_values_off)  + off1;
    const int32_t * idx1 = (const int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off1;

    uint32_t count0 = cctx->real_count[4*i + 0] + cctx->real_count[4*i + 1];
    uint32_t count1 = cctx->real_count[4*i + 2] + cctx->real_count[4*i + 3];

    uint8_t * slot_base = cctx->vtcm_base + (4 * i) * slot_size;
    float *   out_val   = (float *) slot_base;
    int32_t * out_idx   = (int32_t *) (slot_base + 4 * chunk_elems * sizeof(float));

    struct htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

struct htp_sort_multi_row_ctx {
    struct htp_ops_context *              octx;
    const struct htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              nrows_per_thread;
};

struct htp_tensor_row_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

static inline struct htp_tensor_row_coord init_row_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
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
    float * cur_values,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_top_k,
    enum ggml_sort_order order
) {
    if (is_top_k) {
        if (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            switch (ne00) {
                case 32:   sort32_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); break;
                case 64:   sort64_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); break;
                case 128:  sort128_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); break;
                case 256:  sort256_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); break;
                case 512:  sort512_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); break;
                case 1024: sort1024_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); break;
            }
        } else {
            if (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_values + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_sort_vtcm_desc((uint8_t *) cur_values, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } else {
        if (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            switch (ne00) {
                case 32:   sort32_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, order); break;
                case 64:   sort64_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, order); break;
                case 128:  sort128_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, order); break;
                case 256:  sort256_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, order); break;
                case 512:  sort512_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, order); break;
                case 1024: sort1024_f32_hvx((uint8_t *) cur_values, (uint8_t *) cur_indices, order); break;
            }
        } else {
            init_indices_ramp(cur_indices, ne00, 0);
            if (order == GGML_SORT_ORDER_ASC) {
                quicksort_values_indices_asc(cur_values, cur_indices, 0, ne00 - 1);
            } else {
                quicksort_values_indices_desc(cur_values, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

static void htp_sort_multi_row_job(unsigned int n, unsigned int ith, void * data) {
    struct htp_sort_multi_row_ctx * mctx = (struct htp_sort_multi_row_ctx *) data;
    struct htp_ops_context * octx = mctx->octx;
    const struct htp_sort_kernel_params * kparams = mctx->kparams;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    uint32_t ne00 = (uint32_t) kparams->ne00;
    uint32_t ne01 = src0->ne[1];
    uint32_t ne02 = src0->ne[2];

    uint32_t nb01 = src0->nb[1];
    uint32_t nb02 = src0->nb[2];
    uint32_t nb03 = src0->nb[3];

    uint32_t nb1 = dst->nb[1];
    uint32_t nb2 = dst->nb[2];
    uint32_t nb3 = dst->nb[3];

    uint32_t row_start = (uint32_t) kparams->row_start;
    uint32_t row_end   = (uint32_t) kparams->row_end;
    uint32_t r_start   = row_start + ith * mctx->nrows_per_thread;
    uint32_t r_end     = MIN(r_start + mctx->nrows_per_thread, row_end);

    if (r_start >= r_end) return;

    size_t spad_offset = ith * (size_t) (kparams->phase1_slot_size * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_base + spad_offset;
    size_t slot_size = (size_t) kparams->phase1_slot_size;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    float *   values_buf  = (float *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * sizeof(float));

    dma_queue * dma_q = octx->ctx->dma[ith];
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_top_k = kparams->is_top_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    enum ggml_sort_order order = (enum ggml_sort_order) kparams->order;

    const uint32_t src_bytes = ne00 * sizeof(float);
    const uint32_t dst_bytes = (is_top_k ? k : ne00) * sizeof(int32_t);

    if (kparams->n_slots == 2) {
        struct htp_tensor_row_coord prime_coord = init_row_coord(r_start, ne01, ne02);
        for (uint32_t r = r_start, spad_idx = 0; r < r_end && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * slot_size;
            float *   cur_values  = (float *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * sizeof(float));

            dma_addr_t dst_addr = dst->data + coord_offset(prime_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->data + coord_offset(prime_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_queue_push(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 0);

            // Prefetch input row
            dma_queue_push(dma_q, dma_make_data(cur_values, src_addr),
                           src_bytes, src_bytes, src_bytes, 1);

            step_row_coord(&prime_coord, ne01, ne02);
        }

        struct htp_tensor_row_coord dst_coord = init_row_coord(r_start, ne01, ne02);
        struct htp_tensor_row_coord next_src_coord = init_row_coord(r_start + 2, ne01, ne02);

        for (uint32_t r = r_start; r < r_end; r++) {
            uint32_t cur_slot = (r - r_start) & 1;
            uint8_t * cur_spad = thread_spad + cur_slot * slot_size;
            float *   cur_values  = (float *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * sizeof(float));

            // Wait for previous writeback of this slot to complete
            dma_queue_pop(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_queue_pop(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(cur_values, cur_indices, ne00, chunk_elems, is_top_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->data + coord_offset(dst_coord, nb1, nb2, nb3);
            dma_queue_push(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 1);
            step_row_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            const uint32_t next_row = r + 2;
            if (next_row < r_end) {
                dma_addr_t next_src_addr = src0->data + coord_offset(next_src_coord, nb01, nb02, nb03);
                dma_queue_push(dma_q, dma_make_data(cur_values, next_src_addr),
                               src_bytes, src_bytes, src_bytes, 1);
                step_row_coord(&next_src_coord, ne01, ne02);
            }
        }

        dma_queue_flush(dma_q);
    } else {
        struct htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02);
        for (uint32_t r = r_start; r < r_end; r++) {
            dma_addr_t src_addr = src0->data + coord_offset(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->data + coord_offset(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, values_buf, src_addr, src_bytes);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(values_buf, indices_buf, ne00, chunk_elems, is_top_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_bytes);
            step_row_coord(&coord, ne01, ne02);
        }
    }
}

static int op_sort_common(struct htp_ops_context * octx, bool is_top_k) {
    if (octx->src[0]->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const size_t dst_row_size = dst->ne[0] * sizeof(int32_t);

    uint32_t row_start = 0;
    uint32_t row_end   = total_rows;
    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, sizeof(int32_t), (uint32_t) dst_row_size, &rows_per_chunk);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        row_end   = range.start + range.count;
    }

    const uint32_t nrows = row_end - row_start;
    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    struct htp_sort_kernel_params kparams_local;
    const struct htp_sort_kernel_params * kparams = (const struct htp_sort_kernel_params *) octx->kernel_params;

    if (kparams->n_threads == 0) {
        struct htp_sort_vtcm_layout layout;
        if (!htp_sort_solve_layout(&layout, ne00, total_rows, k, octx->n_threads, octx->ctx->vtcm_size, is_top_k)) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, sizeof(kparams_local));
        kparams_local.n_threads         = (int32_t) layout.n_threads;
        kparams_local.total_rows        = (int32_t) total_rows;
        kparams_local.row_start         = (int32_t) row_start;
        kparams_local.row_end           = (int32_t) row_end;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.order             = is_top_k ? GGML_SORT_ORDER_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_top_k          = is_top_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_size         = (int32_t) layout.total_bytes;
        kparams_local.phase1_slot_size  = (int32_t) layout.phase1_slot_size;
        kparams_local.merge_values_off  = (int32_t) layout.merge_values_off;
        kparams_local.merge_indices_off = (int32_t) layout.merge_indices_off;
        kparams_local.merge_elems       = (int32_t) layout.merge_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } else if (octx->ctx->mdev.count > 1) {
        memcpy(&kparams_local, kparams, sizeof(kparams_local));
        kparams_local.row_start = (int32_t) row_start;
        kparams_local.row_end   = (int32_t) row_end;
        kparams = &kparams_local;
    }

    if (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_size, octx->ctx->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    if (total_rows == 1 && kparams->n_chunks > 1) {
        struct htp_sort_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_base = vtcm_base;
        memset(cctx.real_count, 0, sizeof(cctx.real_count));

        work_queue_run(octx->ctx->work_queue, htp_sort_chunk_job, &cctx, (uint32_t) kparams->n_chunks);

        if (is_top_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            if (k == 1 && ne00 >= 128*1024) {
                float *   cand_vals = (float *)   (vtcm_base + kparams->merge_values_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                float best_val = cand_vals[0];
                int32_t best_idx = cand_idxs[0];
                for (int32_t c = 1; c < kparams->n_chunks; c++) {
                    if (cand_vals[c] > best_val) {
                        best_val = cand_vals[c];
                        best_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->data;
                dst_ptr[0] = best_idx;
                return HTP_STATUS_OK;
            }

            uint32_t total_candidates = (uint32_t) kparams->n_chunks * local_k;
            float *   cand_vals = (float *)   (vtcm_base + kparams->merge_values_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
            uint32_t merge_elems = (uint32_t) kparams->merge_elems;

            if (merge_elems > total_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + total_candidates), -INFINITY, merge_elems - total_candidates);
                for (uint32_t j = total_candidates; j < merge_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            struct htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_sort_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, merge_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->data, cand_idxs, k * sizeof(int32_t));
        } else {
            bool asc = kparams->order == GGML_SORT_ORDER_ASC;
            if (kparams->n_chunks == 8) {
                struct htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 4);

                struct htp_argsort_merge_l2_ctx l2ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l2_job, &l2ctx, 2);

                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                const float *   val0 = (const float *) vtcm_base;
                const int32_t * idx0 = (const int32_t *) (vtcm_base + 4 * chunk_elems * sizeof(float));
                const float *   val1 = (const float *) (vtcm_base + 4 * slot_size);
                const int32_t * idx1 = (const int32_t *) (vtcm_base + 4 * slot_size + 4 * chunk_elems * sizeof(float));

                uint32_t n01 = cctx.real_count[0] + cctx.real_count[1] + cctx.real_count[2] + cctx.real_count[3];
                uint32_t n23 = cctx.real_count[4] + cctx.real_count[5] + cctx.real_count[6] + cctx.real_count[7];

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                struct htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, n01, val1, idx1, n23, NULL, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * sizeof(int32_t));
            } else if (kparams->n_chunks == 4) {
                struct htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_count[0] + cctx.real_count[1];
                uint32_t n23 = cctx.real_count[2] + cctx.real_count[3];

                const float *   val01 = (const float *)   (vtcm_base + kparams->merge_values_off);
                const int32_t * idx01 = (const int32_t *) (vtcm_base + kparams->merge_indices_off);
                const float *   val23 = (const float *)   (vtcm_base + kparams->merge_values_off)  + 2 * chunk_elems;
                const int32_t * idx23 = (const int32_t *) (vtcm_base + kparams->merge_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_base;
                struct htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val01, idx01, n01, val23, idx23, n23, NULL, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * sizeof(int32_t));
            } else if (kparams->n_chunks == 2) {
                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                const float *   val0 = (const float *)   (vtcm_base);
                const int32_t * idx0 = (const int32_t *) (vtcm_base + chunk_elems * sizeof(float));
                const float *   val1 = (const float *)   (vtcm_base + slot_size);
                const int32_t * idx1 = (const int32_t *) (vtcm_base + slot_size + chunk_elems * sizeof(float));

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                struct htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, cctx.real_count[0], val1, idx1, cctx.real_count[1], NULL, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * sizeof(int32_t));
            } else {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_base + chunk_elems * sizeof(float));
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * sizeof(int32_t));
            }
        }
        return HTP_STATUS_OK;
    }

    struct htp_sort_multi_row_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_base        = vtcm_base;

    uint32_t dst_row_bytes = (uint32_t) dst_row_size;
    uint32_t rows_per_chunk = 0;
    htp_tensor_mdev_rows_per_chunk(dst, sizeof(int32_t), dst_row_bytes, &rows_per_chunk);
    if (rows_per_chunk == 0) {
        rows_per_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_threads - 1) / (uint32_t) kparams->n_threads;
    if (rows_per_chunk > 1) {
        dr = hex_round_up(dr, rows_per_chunk);
    }
    mctx.nrows_per_thread = dr;

    work_queue_run(octx->ctx->work_queue, htp_sort_multi_row_job, &mctx, (uint32_t) kparams->n_threads);

    return HTP_STATUS_OK;
}

int op_argsort(struct htp_ops_context * octx) {
    return op_sort_common(octx, false);
}

int op_top_k(struct htp_ops_context * octx) {
    return op_sort_common(octx, true);
}
