#include "ggml-et-uberkernel-common.h"
#include "ggml-et-uberkernel-kernel-map.h"
#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_glu_params;
struct ggml_et_unary_params;
struct ggml_et_rope_params;
struct ggml_et_rms_norm_params;
struct ggml_et_rms_norm_mul_params;
struct ggml_et_softmax_params;
struct ggml_et_set_rows_params;
struct ggml_et_get_rows_params;
struct ggml_et_cont_params;
struct ggml_et_concat_params;
struct ggml_et_cumsum_params;
struct ggml_et_diag_params;
struct ggml_et_fill_params;
struct ggml_et_flash_attn_ext_params;
struct ggml_et_gated_delta_net_params;
struct ggml_et_group_norm_params;
struct ggml_et_im2col_params;
struct ggml_et_l2_norm_params;
struct ggml_et_mul_mat_id_params;
struct ggml_et_norm_params;
struct ggml_et_pad_params;
struct ggml_et_repeat_params;
struct ggml_et_rwkv_wkv6_params;
struct ggml_et_rwkv_wkv7_params;
struct ggml_et_scale_params;
struct ggml_et_set_params;
struct ggml_et_solve_tri_params;
struct ggml_et_sqr_params;
struct ggml_et_ssm_conv_params;
struct ggml_et_ssm_scan_params;
struct ggml_et_sum_rows_params;
struct ggml_et_tri_params;

extern int el_map_f32_entry(struct ggml_et_binary_params *, void *);
extern int glu_f32_entry(struct ggml_et_glu_params *, void *);
extern int unary_f32_entry(struct ggml_et_unary_params *, void *);
extern int rope_f32_entry(struct ggml_et_rope_params *, void *);
extern int rms_norm_f32_entry(struct ggml_et_rms_norm_params *, void *);
extern int rms_norm_mul_f32_entry(struct ggml_et_rms_norm_mul_params *, void *);
extern int softmax_f32_entry(struct ggml_et_softmax_params *, void *);
extern int set_rows_f32_entry(struct ggml_et_set_rows_params *, void *);
extern int get_rows_f32_entry(struct ggml_et_get_rows_params *, void *);
extern int cont_f32_entry(struct ggml_et_cont_params *, void *);
extern int cont_f16_entry(struct ggml_et_cont_params *, void *);
extern int cpy_f32_f16_entry(struct ggml_et_cont_params *, void *);
extern int concat_f32_entry(struct ggml_et_concat_params *, void *);
extern int cumsum_f32_entry(struct ggml_et_cumsum_params *, void *);
extern int diag_f32_entry(struct ggml_et_diag_params *, void *);
extern int fill_f32_entry(struct ggml_et_fill_params *, void *);
extern int flash_attn_ext_f32_entry(struct ggml_et_flash_attn_ext_params *, void *);
extern int flash_attn_ext_f16_me_entry(struct ggml_et_flash_attn_ext_params *, void *);
extern int gated_delta_net_f32_entry(struct ggml_et_gated_delta_net_params *, void *);
extern int group_norm_f32_entry(struct ggml_et_group_norm_params *, void *);
extern int im2col_entry(struct ggml_et_im2col_params *, void *);
extern int l2_norm_f32_entry(struct ggml_et_l2_norm_params *, void *);
extern int mul_mat_id_f32_entry(struct ggml_et_mul_mat_id_params *, void *);
extern int norm_f32_entry(struct ggml_et_norm_params *, void *);
extern int pad_f32_entry(struct ggml_et_pad_params *, void *);
extern int repeat_f32_entry(struct ggml_et_repeat_params *, void *);
extern int rwkv_wkv6_f32_entry(struct ggml_et_rwkv_wkv6_params *, void *);
extern int rwkv_wkv7_f32_entry(struct ggml_et_rwkv_wkv7_params *, void *);
extern int scale_f32_entry(struct ggml_et_scale_params *, void *);
extern int set_f32_entry(struct ggml_et_set_params *, void *);
extern int solve_tri_f32_entry(struct ggml_et_solve_tri_params *, void *);
extern int sqr_f32_entry(struct ggml_et_sqr_params *, void *);
extern int ssm_conv_f32_entry(struct ggml_et_ssm_conv_params *, void *);
extern int ssm_scan_f32_entry(struct ggml_et_ssm_scan_params *, void *);
extern int sum_rows_f32_entry(struct ggml_et_sum_rows_params *, void *);
extern int tri_f32_entry(struct ggml_et_tri_params *, void *);
extern int mul_mat_f16_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_f16_matrix_engine_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_f32_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_f32_matrix_engine_entry(struct ggml_et_binary_params *, void *);
extern int mul_mat_Q8_0_entry(struct ggml_et_mm_q8_params *, void *);
extern int mul_mat_Q4_0_entry(struct ggml_et_binary_params *, void *);

static inline size_t tensor_bytes(const struct ggml_tensor * t) {
    return (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

struct uber_glu_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
    // trailing scalars omitted — not needed for eviction
};

struct uber_unary_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct uber_rope_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
};

struct uber_rms_norm_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct uber_rms_norm_mul_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_softmax_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
};

struct uber_set_rows_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_get_rows_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_cont_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

// src0 + src1 + dst (no trailing scalars needed for eviction)
struct uber_concat_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_ssm_conv_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_solve_tri_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct uber_mul_mat_id_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
};

// flash_attn_ext: Q=src0, K=src1, V=src2, mask=src3, dst (mask optional)
struct uber_flash_attn_ext_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor mask;
    struct ggml_tensor dst;
};

// ssm_scan: 7 source tensors + dst
struct uber_ssm_scan_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor src3;
    struct ggml_tensor src4;
    struct ggml_tensor src5;
    struct ggml_tensor src6;
    struct ggml_tensor dst;
};

// gated_delta_net: q,k,v,g,beta,state_in,dst
struct uber_gated_delta_net_params {
    struct ggml_tensor q;
    struct ggml_tensor k;
    struct ggml_tensor v;
    struct ggml_tensor g;
    struct ggml_tensor beta;
    struct ggml_tensor state_in;
    struct ggml_tensor dst;
};

static void copy_f32_to_f16_row(uint16_t * dst, const float * src, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

static void copy_f32_row(float * dst, const float * src, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = src[i];
    }
}

static void evict_region_past_l2_local(const void * addr, size_t bytes) {
    if (!addr || bytes == 0) {
        return;
    }

    const uint64_t CL     = 64;
    uint64_t       base   = (uint64_t) addr & ~(CL - 1);
    uint64_t       end    = ((uint64_t) addr + bytes + CL - 1) & ~(CL - 1);
    uint64_t       nlines = (end - base) / CL;
    cache_ops_priv_evict_sw(0, /*to_L2*/ 3, 0, 0, CL);
}

int entry_point(struct ggml_et_uberkernel_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    if (!kernel_env || !params) {
        return -1;
    }

    struct ggml_et_uberkernel_inst * insts       = (struct ggml_et_uberkernel_inst *) (uintptr_t) params->insts;
    uint8_t *                        params_blob = (uint8_t *) (uintptr_t) params->params_blob;

    if (!insts || !params_blob || params->inst_stride < sizeof(struct ggml_et_uberkernel_inst)) {
        return -1;
    }

    for (uint32_t i = 0; i < params->num_insts; ++i) {
        struct ggml_et_uberkernel_inst * inst =
            (struct ggml_et_uberkernel_inst *) ((uint8_t *) insts + (i * params->inst_stride));
        void * inst_params = params_blob + inst->params_offset;
        int    rc          = -1;

        et_barrier_global(32ULL);

        switch (inst->kernel_id) {
            case GGML_ET_UBERKERNEL_KERNEL_EL_MAP_F32:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    rc                               = el_map_f32_entry(p, env);
                    break;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_UNARY_F32: {
            //     // struct uber_unary_params *p = (struct uber_unary_params *) inst_params;
            //     // et_barrier(ET_BARRIER_GLOBAL);
            //     rc = unary_f32_entry((struct ggml_et_unary_params *) inst_params, env);
            //     break;
            // }
            // case GGML_ET_UBERKERNEL_KERNEL_CPY_F32_F16: {
            //     struct uber_unary_params *p = (struct uber_unary_params *) inst_params;
            //     // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
            //     rc = cpy_f32_f16_entry((struct ggml_et_cont_params *) inst_params, env);
            //     break;
            // }
            // case GGML_ET_UBERKERNEL_KERNEL_GET_ROWS_F32: {
            //     struct uber_get_rows_params *p = (struct uber_get_rows_params *) inst_params;
            //     rc = get_rows_f32_entry((struct ggml_et_get_rows_params *) inst_params, env);
            //     break;
            // }
            // case GGML_ET_UBERKERNEL_KERNEL_CONT_F32: {
            //     struct uber_cont_params *p = (struct uber_cont_params *) inst_params;
            //     // evict_region_past_l2_local(p->src0.data, tensor_bytes(&p->src0));
            //     // evict_region_past_l2(p->dst.data, tensor_bytes(&p->dst));
            //     rc = cont_f32_entry((struct ggml_et_cont_params *) inst_params, env);
            //     break;
            // }
            case GGML_ET_UBERKERNEL_KERNEL_GLU_F32:
                {
                    rc = glu_f32_entry((struct ggml_et_glu_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_ROPE_F32:
                {
                    rc = rope_f32_entry((struct ggml_et_rope_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_F32:
                {
                    // struct ggml_et_rms_norm_params *p = (struct ggml_et_rms_norm_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc = rms_norm_f32_entry((struct ggml_et_rms_norm_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_RMS_NORM_MUL_F32:
                {
                    struct uber_rms_norm_mul_params * p = (struct uber_rms_norm_mul_params *) inst_params;
                    evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    evict_region_past_l2(p->src1.data, tensor_bytes(&p->src1));
                    rc = rms_norm_mul_f32_entry((struct ggml_et_rms_norm_mul_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SOFTMAX_F32:
                {
                    rc = softmax_f32_entry((struct ggml_et_softmax_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SET_ROWS_F32:
                {
                    rc = set_rows_f32_entry((struct ggml_et_set_rows_params *) inst_params, env);
                    break;
                }

            // Single-source ops (src0 → dst)
            case GGML_ET_UBERKERNEL_KERNEL_SQR_F32:
                {
                    rc = sqr_f32_entry((struct ggml_et_sqr_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SCALE_F32:
                {
                    rc = scale_f32_entry((struct ggml_et_scale_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SUM_ROWS_F32:
                {
                    rc = sum_rows_f32_entry((struct ggml_et_sum_rows_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_CUMSUM_F32:
                {
                    rc = cumsum_f32_entry((struct ggml_et_cumsum_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_NORM_F32:
                {
                    rc = norm_f32_entry((struct ggml_et_norm_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_L2_NORM_F32:
                {
                    rc = l2_norm_f32_entry((struct ggml_et_l2_norm_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_GROUP_NORM_F32:
                {
                    rc = group_norm_f32_entry((struct ggml_et_group_norm_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_REPEAT_F32:
                {
                    rc = repeat_f32_entry((struct ggml_et_repeat_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_DIAG_F32:
                {
                    rc = diag_f32_entry((struct ggml_et_diag_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_TRI_F32:
                {
                    rc = tri_f32_entry((struct ggml_et_tri_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_PAD_F32:
                {
                    rc = pad_f32_entry((struct ggml_et_pad_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_CONT_F16:
                {
                    rc = cont_f16_entry((struct ggml_et_cont_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_FILL_F32:
                {
                    rc = fill_f32_entry((struct ggml_et_fill_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SET_F32:
                {
                    rc = set_f32_entry((struct ggml_et_set_params *) inst_params, env);
                    break;
                }

            // Two-source ops
            case GGML_ET_UBERKERNEL_KERNEL_CONCAT_F32:
                {
                    rc = concat_f32_entry((struct ggml_et_concat_params *) inst_params, env);
                    break;
                }
            // case GGML_ET_UBERKERNEL_KERNEL_SSM_CONV_F32: {
            //     rc = ssm_conv_f32_entry((struct ggml_et_ssm_conv_params *) inst_params, env);
            //     break;
            // }
            case GGML_ET_UBERKERNEL_KERNEL_SOLVE_TRI_F32:
                {
                    rc = solve_tri_f32_entry((struct ggml_et_solve_tri_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_IM2COL:
                {
                    rc = im2col_entry((struct ggml_et_im2col_params *) inst_params, env);
                    break;
                }

            // Three-source ops
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_ID_F32:
                {
                    rc = mul_mat_id_f32_entry((struct ggml_et_mul_mat_id_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F32:
                {
                    rc = flash_attn_ext_f32_entry((struct ggml_et_flash_attn_ext_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_FLASH_ATTN_EXT_F16_ME:
                {
                    rc = flash_attn_ext_f16_me_entry((struct ggml_et_flash_attn_ext_params *) inst_params, env);
                    break;
                }

            case GGML_ET_UBERKERNEL_KERNEL_GATED_DELTA_NET_F32:
                {
                    rc = gated_delta_net_f32_entry((struct ggml_et_gated_delta_net_params *) inst_params, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_SSM_SCAN_F32:
                {
                    rc = ssm_scan_f32_entry((struct ggml_et_ssm_scan_params *) inst_params, env);
                    break;
                }
            // rwkv: raw float* params, no ggml_tensor fields to evict via
            case GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV6_F32:
                {
                    rc = rwkv_wkv6_f32_entry((struct ggml_et_rwkv_wkv6_params *) inst_params, env);
                    break;
                }

            case GGML_ET_UBERKERNEL_KERNEL_RWKV_WKV7_F32:
                {
                    rc = rwkv_wkv7_f32_entry((struct ggml_et_rwkv_wkv7_params *) inst_params, env);
                    break;
                }

            // MUL_MAT: evict src1 (activations); src0=weights is
            //  read-only so never stale from a prior uberkernel op
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    rc                               = mul_mat_f16_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F16_MATRIX_ENGINE:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    rc                               = mul_mat_f16_matrix_engine_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    rc                               = mul_mat_f32_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_F32_MATRIX_ENGINE:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    rc                               = mul_mat_f32_matrix_engine_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q8_0:
                {
                    struct ggml_et_mm_q8_params * p = (struct ggml_et_mm_q8_params *) inst_params;
                    // evict_region_past_l2(p->src0.data, tensor_bytes(&p->src0));
                    rc                              = mul_mat_Q8_0_entry(p, env);
                    break;
                }
            case GGML_ET_UBERKERNEL_KERNEL_MUL_MAT_Q4_0:
                {
                    struct ggml_et_binary_params * p = (struct ggml_et_binary_params *) inst_params;
                    rc                               = mul_mat_Q4_0_entry(p, env);
                    break;
                }

            default:
                return -1;
        }

        if (rc != 0) {
            return rc;
        }
    }

    return 0;
}
