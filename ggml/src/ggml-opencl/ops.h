#pragma once

#include "ggml-backend.h"

#include <CL/cl.h>

struct ggml_backend_opencl_context;

struct ggml_opencl_add_kernels {
    cl_kernel kernel_add;
    cl_kernel kernel_add_row;
    cl_kernel kernel_add_f16;
    cl_kernel kernel_add_row_f16;
};

struct ggml_opencl_add_id_kernels {
    cl_kernel kernel_add_id;
};

struct ggml_opencl_mul_kernels {
    cl_kernel kernel_mul;
    cl_kernel kernel_mul_row;
    cl_kernel kernel_mul_f16;
    cl_kernel kernel_mul_row_f16;
};

struct ggml_opencl_div_kernels {
    cl_kernel kernel_div;
    cl_kernel kernel_div_row;
    cl_kernel kernel_div_f16;
    cl_kernel kernel_div_row_f16;
};

struct ggml_opencl_sub_kernels {
    cl_kernel kernel_sub;
    cl_kernel kernel_sub_row;
    cl_kernel kernel_sub_f16;
    cl_kernel kernel_sub_row_f16;
};

struct ggml_opencl_scale_kernels {
    cl_kernel kernel_scale_f32;
    cl_kernel kernel_scale_f32_4;
};

struct ggml_opencl_sqr_kernels {
    cl_kernel kernel_sqr_cont_f32;
    cl_kernel kernel_sqr_cont_f32_4;
    cl_kernel kernel_sqr_cont_f16;
    cl_kernel kernel_sqr_cont_f16_4;
};

struct ggml_opencl_sqrt_kernels {
    cl_kernel kernel_sqrt_cont_f32;
    cl_kernel kernel_sqrt_cont_f32_4;
    cl_kernel kernel_sqrt_cont_f16;
    cl_kernel kernel_sqrt_cont_f16_4;
};

struct ggml_opencl_mean_kernels {
    cl_kernel kernel_mean_f32;
    cl_kernel kernel_mean_f32_4;
};

struct ggml_opencl_silu_kernels {
    cl_kernel kernel_silu;
    cl_kernel kernel_silu_4;
};

struct ggml_opencl_gelu_kernels {
    cl_kernel kernel_gelu;
    cl_kernel kernel_gelu_4;
    cl_kernel kernel_gelu_erf;
    cl_kernel kernel_gelu_erf_4;
    cl_kernel kernel_gelu_quick;
    cl_kernel kernel_gelu_quick_4;
};

struct ggml_opencl_relu_kernels {
    cl_kernel kernel_relu;
};

struct ggml_opencl_sigmoid_kernels {
    cl_kernel kernel_sigmoid_f32;
    cl_kernel kernel_sigmoid_f16;
};

struct ggml_opencl_tri_kernels {
    cl_kernel kernel_tri;
};

struct ggml_opencl_fill_kernels {
    cl_kernel kernel_fill;
};

struct ggml_opencl_clamp_kernels {
    cl_kernel kernel_clamp;
};

struct ggml_opencl_glu_kernels {
    cl_kernel kernel_geglu;
    cl_kernel kernel_reglu;
    cl_kernel kernel_swiglu;
    cl_kernel kernel_swiglu_oai;
    cl_kernel kernel_swiglu_clamp;
    cl_kernel kernel_geglu_erf;
    cl_kernel kernel_geglu_quick;
    cl_kernel kernel_geglu_f16;
    cl_kernel kernel_reglu_f16;
    cl_kernel kernel_swiglu_f16;
    cl_kernel kernel_swiglu_clamp_f16;
    cl_kernel kernel_geglu_erf_f16;
    cl_kernel kernel_geglu_quick_f16;
};

struct ggml_opencl_norm_kernels {
    cl_kernel kernel_norm;
    cl_kernel kernel_norm_mul_add;
};

struct ggml_opencl_rms_norm_kernels {
    cl_kernel kernel_rms_norm;
    cl_kernel kernel_rms_norm_mul;
    cl_kernel kernel_rms_norm_mul_add;
};

struct ggml_opencl_l2_norm_kernels {
    cl_kernel kernel_l2_norm_f32;
};

struct ggml_opencl_group_norm_kernels {
    cl_kernel kernel_group_norm;
    cl_kernel kernel_group_norm_mul_add;
};

struct ggml_opencl_diag_mask_inf_kernels {
    cl_kernel kernel_diag_mask_inf;
    cl_kernel kernel_diag_mask_inf_8;
};

struct ggml_opencl_diag_kernels {
    cl_kernel kernel_diag_f32;
};

struct ggml_opencl_soft_max_kernels {
    cl_kernel kernel_soft_max;
    cl_kernel kernel_soft_max_4;
    cl_kernel kernel_soft_max_f16;
    cl_kernel kernel_soft_max_4_f16;
};

struct ggml_opencl_get_rows_kernels {
    cl_kernel kernel_get_rows_f32;
    cl_kernel kernel_get_rows_f16;
    cl_kernel kernel_get_rows_q4_0;
};

struct ggml_opencl_set_rows_kernels {
    cl_kernel kernel_set_rows_f32_i64;
    cl_kernel kernel_set_rows_f32_i32;
    cl_kernel kernel_set_rows_f16_i64;
    cl_kernel kernel_set_rows_f16_i32;
    cl_kernel kernel_set_rows_q8_0_i64;
    cl_kernel kernel_set_rows_q8_0_i32;
    cl_kernel kernel_set_rows_q8_0_soa_i64;
    cl_kernel kernel_set_rows_q8_0_soa_i32;
    cl_kernel kernel_set_rows_q4_0_i64;
    cl_kernel kernel_set_rows_q4_0_i32;
    cl_kernel kernel_set_rows_q4_0_soa_i64;
    cl_kernel kernel_set_rows_q4_0_soa_i32;
};

struct ggml_opencl_rope_kernels {
    cl_kernel kernel_rope_norm_f32;
    cl_kernel kernel_rope_norm_f16;
    cl_kernel kernel_rope_neox_f32;
    cl_kernel kernel_rope_neox_f16;
    cl_kernel kernel_rope_multi_f32;
    cl_kernel kernel_rope_multi_f16;
    cl_kernel kernel_rope_vision_f32;
    cl_kernel kernel_rope_vision_f16;
};

struct ggml_opencl_cpy_kernels {
    cl_kernel kernel_cpy_f16_f16;
    cl_kernel kernel_cpy_f16_f32;
    cl_kernel kernel_cpy_f32_f16;
    cl_kernel kernel_cpy_f32_f32;
    cl_kernel kernel_cpy_f32_f32_pack;
    cl_kernel kernel_cpy_i32_i32;
    cl_kernel kernel_cpy_f32_f32_flat;
};

struct ggml_opencl_solve_tri_kernels {
    cl_kernel kernel_solve_tri_f32;
};

struct ggml_opencl_im2col_kernels {
    cl_kernel kernel_im2col_f32;
    cl_kernel kernel_im2col_f16;
};

struct ggml_opencl_argsort_kernels {
    cl_kernel kernel_argsort_f32_i32;
};

struct ggml_opencl_sum_rows_kernels {
    cl_kernel kernel_sum_rows_f32;
    cl_kernel kernel_sum_rows_f32_4;
};

struct ggml_opencl_cumsum_kernels {
    cl_kernel kernel_cumsum_blk;
    cl_kernel kernel_cumsum_add;
};

struct ggml_opencl_repeat_kernels {
    cl_kernel kernel_repeat_f32;
};

struct ggml_opencl_pad_kernels {
    cl_kernel kernel_pad;
};

struct ggml_opencl_tanh_kernels {
    cl_kernel kernel_tanh_f32;
    cl_kernel kernel_tanh_f32_4;
    cl_kernel kernel_tanh_f32_nc;
    cl_kernel kernel_tanh_f16;
    cl_kernel kernel_tanh_f16_4;
    cl_kernel kernel_tanh_f16_nc;
};

struct ggml_opencl_neg_kernels {
    cl_kernel kernel_neg_f32;
    cl_kernel kernel_neg_f32_4;
    cl_kernel kernel_neg_f32_nc;
    cl_kernel kernel_neg_f16;
    cl_kernel kernel_neg_f16_4;
    cl_kernel kernel_neg_f16_nc;
};

struct ggml_opencl_exp_kernels {
    cl_kernel kernel_exp_f32;
    cl_kernel kernel_exp_f32_4;
    cl_kernel kernel_exp_f32_nc;
    cl_kernel kernel_exp_f16;
    cl_kernel kernel_exp_f16_4;
    cl_kernel kernel_exp_f16_nc;
};

struct ggml_opencl_expm1_kernels {
    cl_kernel kernel_expm1_f32;
    cl_kernel kernel_expm1_f32_4;
    cl_kernel kernel_expm1_f32_nc;
    cl_kernel kernel_expm1_f16;
    cl_kernel kernel_expm1_f16_4;
    cl_kernel kernel_expm1_f16_nc;
};

struct ggml_opencl_abs_kernels {
    cl_kernel kernel_abs_f32;
    cl_kernel kernel_abs_f32_4;
    cl_kernel kernel_abs_f32_nc;
    cl_kernel kernel_abs_f16;
    cl_kernel kernel_abs_f16_4;
    cl_kernel kernel_abs_f16_nc;
};

struct ggml_opencl_unary_ext_kernels {
    cl_kernel kernel_sgn_f32;
    cl_kernel kernel_sgn_f32_4;
    cl_kernel kernel_sgn_f32_nc;
    cl_kernel kernel_sgn_f16;
    cl_kernel kernel_sgn_f16_4;
    cl_kernel kernel_sgn_f16_nc;
    cl_kernel kernel_step_f32;
    cl_kernel kernel_step_f32_4;
    cl_kernel kernel_step_f32_nc;
    cl_kernel kernel_step_f16;
    cl_kernel kernel_step_f16_4;
    cl_kernel kernel_step_f16_nc;
    cl_kernel kernel_elu_f32;
    cl_kernel kernel_elu_f32_4;
    cl_kernel kernel_elu_f32_nc;
    cl_kernel kernel_elu_f16;
    cl_kernel kernel_elu_f16_4;
    cl_kernel kernel_elu_f16_nc;
    cl_kernel kernel_hardswish_f32;
    cl_kernel kernel_hardswish_f32_4;
    cl_kernel kernel_hardswish_f32_nc;
    cl_kernel kernel_hardswish_f16;
    cl_kernel kernel_hardswish_f16_4;
    cl_kernel kernel_hardswish_f16_nc;
    cl_kernel kernel_hardsigmoid_f32;
    cl_kernel kernel_hardsigmoid_f32_4;
    cl_kernel kernel_hardsigmoid_f32_nc;
    cl_kernel kernel_hardsigmoid_f16;
    cl_kernel kernel_hardsigmoid_f16_4;
    cl_kernel kernel_hardsigmoid_f16_nc;
    cl_kernel kernel_floor_f32;
    cl_kernel kernel_floor_f32_4;
    cl_kernel kernel_floor_f32_nc;
    cl_kernel kernel_floor_f16;
    cl_kernel kernel_floor_f16_4;
    cl_kernel kernel_floor_f16_nc;
    cl_kernel kernel_ceil_f32;
    cl_kernel kernel_ceil_f32_4;
    cl_kernel kernel_ceil_f32_nc;
    cl_kernel kernel_ceil_f16;
    cl_kernel kernel_ceil_f16_4;
    cl_kernel kernel_ceil_f16_nc;
    cl_kernel kernel_round_f32;
    cl_kernel kernel_round_f32_4;
    cl_kernel kernel_round_f32_nc;
    cl_kernel kernel_round_f16;
    cl_kernel kernel_round_f16_4;
    cl_kernel kernel_round_f16_nc;
    cl_kernel kernel_trunc_f32;
    cl_kernel kernel_trunc_f32_4;
    cl_kernel kernel_trunc_f32_nc;
    cl_kernel kernel_trunc_f16;
    cl_kernel kernel_trunc_f16_4;
    cl_kernel kernel_trunc_f16_nc;
};

struct ggml_opencl_softplus_kernels {
    cl_kernel kernel_softplus_f32;
    cl_kernel kernel_softplus_f32_4;
    cl_kernel kernel_softplus_f32_nc;
    cl_kernel kernel_softplus_f16;
    cl_kernel kernel_softplus_f16_4;
    cl_kernel kernel_softplus_f16_nc;
};

struct ggml_opencl_upscale_kernels {
    cl_kernel kernel_upscale;
    cl_kernel kernel_upscale_bilinear;
};

struct ggml_opencl_concat_kernels {
    cl_kernel kernel_concat_b1;
    cl_kernel kernel_concat_b2;
    cl_kernel kernel_concat_b4;
    cl_kernel kernel_concat_b8;
    cl_kernel kernel_concat_b4_pack;
};

struct ggml_opencl_conv_2d_kernels {
    cl_kernel kernel_conv_2d_f16;
    cl_kernel kernel_conv_2d_f32;
    cl_kernel kernel_conv_2d_f16_f32;
};

struct ggml_opencl_ssm_conv_kernels {
    cl_kernel kernel_ssm_conv_f32_f32;
    cl_kernel kernel_ssm_conv_f32_f32_4;
};

struct ggml_opencl_gated_delta_net_kernels {
    cl_kernel kernel_gated_delta_net_f32[4][2][2];
};

struct ggml_opencl_ssm_scan_kernels {
    cl_kernel kernel_ssm_scan_f32;
    cl_kernel kernel_ssm_scan_f32_mamba2_d128;
    cl_kernel kernel_ssm_scan_f32_mamba2_d256;
};

struct ggml_opencl_timestep_embedding_kernels {
    cl_kernel kernel_timestep_embedding;
};

struct ggml_opencl_repack_kernels {
    cl_kernel kernel_convert_block_q1_0;
    cl_kernel kernel_restore_block_q1_0;
    cl_kernel kernel_convert_block_q4_0;
    cl_kernel kernel_restore_block_q4_0;
    cl_kernel kernel_convert_block_q4_0_trans4_ns;
    cl_kernel kernel_restore_block_q4_0_trans4_ns;
    cl_kernel kernel_convert_block_q4_1;
    cl_kernel kernel_restore_block_q4_1;
    cl_kernel kernel_convert_block_q4_1_trans4_ns;
    cl_kernel kernel_restore_block_q4_1_trans4_ns;
    cl_kernel kernel_convert_block_q5_0;
    cl_kernel kernel_restore_block_q5_0;
    cl_kernel kernel_convert_block_q5_0_trans4_ns;
    cl_kernel kernel_restore_block_q5_0_trans4_ns;
    cl_kernel kernel_convert_block_q5_1;
    cl_kernel kernel_restore_block_q5_1;
    cl_kernel kernel_convert_block_q5_1_trans4_ns;
    cl_kernel kernel_restore_block_q5_1_trans4_ns;
    cl_kernel kernel_convert_block_q4_k_trans4_ns;
    cl_kernel kernel_restore_block_q4_k_trans4_ns;
    cl_kernel kernel_convert_block_q5_k_trans4_ns;
    cl_kernel kernel_restore_block_q5_k_trans4_ns;
    cl_kernel kernel_convert_block_q6_k_trans4_ns;
    cl_kernel kernel_restore_block_q6_k_trans4_ns;
    cl_kernel kernel_convert_block_mxfp4;
    cl_kernel kernel_convert_block_mxfp4_trans;
    cl_kernel kernel_restore_block_mxfp4;
    cl_kernel kernel_restore_block_mxfp4_trans;
    cl_kernel kernel_convert_block_mxfp4_trans4_ns;
    cl_kernel kernel_restore_block_mxfp4_trans4_ns;
    cl_kernel kernel_convert_block_q8_0;
    cl_kernel kernel_restore_block_q8_0;
    cl_kernel kernel_restore_block_q8_0_trans;
    cl_kernel kernel_dequant_q8_0_f16_view_aos;
    cl_kernel kernel_dequant_q8_0_f32_view_aos;
    cl_kernel kernel_dequant_q4_0_f16_view_aos;
    cl_kernel kernel_dequant_q4_0_f32_view_aos;
    cl_kernel kernel_convert_block_q6_K_noshuffle;
    cl_kernel kernel_restore_block_q6_K_noshuffle;
    cl_kernel kernel_convert_bf16_to_f16;
    cl_kernel kernel_convert_f16_to_bf16;
    cl_kernel kernel_convert_block_q4_0_noshuffle;
    cl_kernel kernel_restore_block_q4_0_noshuffle;
    cl_kernel kernel_convert_block_q4_1_noshuffle;
    cl_kernel kernel_restore_block_q4_1_noshuffle;
    cl_kernel kernel_convert_block_q5_0_noshuffle;
    cl_kernel kernel_restore_block_q5_0_noshuffle;
    cl_kernel kernel_convert_block_q5_1_noshuffle;
    cl_kernel kernel_restore_block_q5_1_noshuffle;
    cl_kernel kernel_convert_block_q4_K_noshuffle;
    cl_kernel kernel_restore_block_q4_K_noshuffle;
    cl_kernel kernel_convert_block_q4_K;
    cl_kernel kernel_restore_block_q4_K;
    cl_kernel kernel_convert_block_q5_K;
    cl_kernel kernel_restore_block_q5_K;
    cl_kernel kernel_convert_block_q5_K_noshuffle;
    cl_kernel kernel_restore_block_q5_K_noshuffle;
    cl_kernel kernel_convert_block_q6_K;
    cl_kernel kernel_restore_block_q6_K;
    cl_kernel kernel_convert_block_iq4_nl;
    cl_kernel kernel_restore_block_iq4_nl;
    cl_kernel kernel_convert_block_iq4_nl_noshuffle;
    cl_kernel kernel_restore_block_iq4_nl_noshuffle;
    cl_kernel kernel_moe_expand_scale_q8_0;
    cl_kernel kernel_moe_expand_scale_q5_0;
    cl_kernel kernel_moe_expand_scale_q5_K;
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    cl_kernel kernel_convert_block_q4_k_tiled_ns;
    cl_kernel kernel_convert_block_q6_k_tiled_ns;
#endif
};

struct ggml_opencl_mul_mat_kernels {
    cl_kernel kernel_mul_mat_f32_f32;
    cl_kernel kernel_mul_mat_f16_f16;
    cl_kernel kernel_mul_mat_f16_f32_1row;
    cl_kernel kernel_mul_mat_f16_f32_mrow;
    cl_kernel kernel_mul_mat_f16_f32_mrow_r2;
    cl_kernel kernel_mul_mat_f16_f32_mrow_r4;
    cl_kernel kernel_mul_mat_f16_f32_mrow_h8;
    cl_kernel kernel_mul_mat_f16_f32_mrow_h8r2;
    cl_kernel kernel_mul_mat_f16_f32;
    cl_kernel kernel_mul_mat_f16_f32_l4;
    cl_kernel kernel_mul_mat_f16_f32_l4_dr;
    cl_kernel kernel_mul_mat_f16_f32_l4_dr_ls;
    cl_kernel kernel_mul_mat_f16_f32_l4_dr_lq;
    cl_kernel kernel_mul_mat_f16_f32_l4_x8;
    cl_kernel kernel_mul_mat_f16_f32_l4_x8_pair;
    cl_kernel kernel_mul_mat_f16_f32_l4_x8_gqa4;
    cl_kernel kernel_mul_mat_f16_f32_l4_x8_gqa4_img;
    cl_kernel kernel_mul_mat_f16_f32_l4_x8_gqa_r4_img;
    cl_kernel kernel_mul_mat_f16_f32_l4_x8_gqa_r2_dk256_img;
    cl_kernel kernel_mul_mat_f16_f32_l4_y8;
    cl_kernel kernel_mul_mat_f16_f32_l4_y8_gqa;
    cl_kernel kernel_mul_mat_f16_f32_l4_y8_gqa_img;
    cl_kernel kernel_mul_mat_f16_f32_tiled;
    cl_kernel kernel_adreno_xmem_pack_src_f32;
    cl_kernel kernel_adreno_xmem_prepack_weight_f16;
    cl_kernel kernel_gemm_xmem_f16_f32_os8;
    cl_kernel kernel_adreno_xmem_store_dst_f32;
    cl_kernel kernel_mul_mm_f16_f32_kqv;
    cl_kernel kernel_mul_mm_f16_f32_kq;
    cl_kernel kernel_mul_mat_q4_0_f32;
    cl_kernel kernel_mul_mat_q4_0_f32_v;
    cl_kernel kernel_mul_mat_q4_0_f32_8x_flat;
    cl_kernel kernel_mul_mv_q1_0_f32;
    cl_kernel kernel_mul_mv_q1_0_f32_flat;
    cl_kernel kernel_mul_mat_q4_0_f32_1d_8x_flat;
    cl_kernel kernel_mul_mat_q4_0_f32_1d_16x_flat;
    cl_kernel kernel_mul_mv_q4_1_f32;
    cl_kernel kernel_mul_mv_q4_1_f32_flat;
    cl_kernel kernel_mul_mv_q5_0_f32;
    cl_kernel kernel_mul_mv_q5_0_f32_flat;
    cl_kernel kernel_mul_mv_q5_1_f32;
    cl_kernel kernel_mul_mv_q5_1_f32_flat;
    cl_kernel kernel_mul_mv_q4_K_f32;
    cl_kernel kernel_mul_mv_q4_K_f32_flat;
    cl_kernel kernel_mul_mv_q5_K_f32;
    cl_kernel kernel_mul_mv_q5_K_f32_flat;
    cl_kernel kernel_mul_mv_q6_K_f32;
    cl_kernel kernel_mul_mv_q6_K_f32_flat;
    cl_kernel kernel_mul_mv_mxfp4_f32;
    cl_kernel kernel_mul_mv_mxfp4_f32_flat;
    cl_kernel kernel_mul_mv_q8_0_f32;
    cl_kernel kernel_mul_mv_q8_0_f32_flat;
    cl_kernel kernel_mul_mv_iq4_nl_f32;
    cl_kernel kernel_mul_mv_iq4_nl_f32_flat;
    cl_kernel kernel_mul_mm_f32_f32_l4_lm;
    cl_kernel kernel_gemv_f32_f32_mc;
    cl_kernel kernel_mul_mm_f16_f32_l4_lm;
    cl_kernel kernel_mul_mm_q1_0_f32_l4_lm;
    cl_kernel kernel_mul_mm_q4_0_f32_l4_lm;
    cl_kernel kernel_mul_mm_q4_1_f32_l4_lm;
    cl_kernel kernel_mul_mm_q5_0_f32_l4_lm;
    cl_kernel kernel_mul_mm_q5_1_f32_l4_lm;
    cl_kernel kernel_mul_mm_q8_0_f32_l4_lm;
    cl_kernel kernel_mul_mm_q4_k_f32_l4_lm;
    cl_kernel kernel_mul_mm_q5_k_f32_l4_lm;
    cl_kernel kernel_mul_mm_q6_k_f32_l4_lm;
    cl_kernel kernel_mul_mm_iq4_nl_f32_l4_lm;
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    cl_kernel kernel_transpose_32;
    cl_kernel kernel_transpose_32_16;
    cl_kernel kernel_transpose_16;
    cl_kernel kernel_transpose_8_buf;
    cl_kernel kernel_transpose_16_buf;
    cl_kernel kernel_transpose_32_buf;
    cl_kernel kernel_transpose_16_4x1;
    cl_kernel kernel_gemm_noshuffle_q4_0_f32;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32_mc3;
    cl_kernel kernel_gemm_noshuffle_q4_0_f32_32b_trans_ila_a8_bin;
    cl_kernel kernel_gemm_noshuffle_q4_0_q8_1_dp4a_ila_a8_bin;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32_32b_trans;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32_4096_1_11008;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32_4096_1_4096;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32_11008_1_4096;
    cl_kernel kernel_gemv_noshuffle_q4_0_f32_32000_1_4096;
    cl_kernel kernel_gemv_noshuffle_q4_1_f32;
    cl_kernel kernel_gemv_noshuffle_q4_1_f32_mc3;
    cl_kernel kernel_gemm_noshuffle_q4_1_f32;
    cl_kernel kernel_gemm_noshuffle_q8_0_f32;
    cl_kernel kernel_gemm_noshuffle_q8_0_f32_bin;
    cl_kernel kernel_gemm_noshuffle_q8_0_q8_1_dp4a;
    cl_kernel kernel_gemm_noshuffle_q8_0_q8_1_dp4a_wimg;
    cl_kernel kernel_gemv_noshuffle_q8_0_f32;
    cl_kernel kernel_gemv_noshuffle_q8_0_f32_splitk;
    cl_kernel kernel_gemm_noshuffle_q1_0_f32;
    cl_kernel kernel_gemv_noshuffle_q1_0_f32;
    cl_kernel kernel_gemv_noshuffle_q4_k_f32;
    cl_kernel kernel_gemv_noshuffle_q4_k_f32_o4;
    cl_kernel kernel_gemv_noshuffle_q4_k_f32_tiled;
    cl_kernel kernel_gemv_noshuffle_q4_k_f32_splitk;
    cl_kernel kernel_gemv_splitk_reduce_f32;
    cl_kernel kernel_gemv_noshuffle_q4_k_f32_glu;
    cl_kernel kernel_gemv_noshuffle_q4_k_f32_mc3;
    cl_kernel kernel_gemm_noshuffle_q4_k_f32;
    cl_kernel kernel_gemm_noshuffle_q4_k_f32_32b_trans_ila_a8_bin;
    cl_kernel kernel_gemm_noshuffle_q4_k_q8_1_dp4a_ila_a8_bin;
    cl_kernel kernel_gemv_noshuffle_q4_k_f32_32b_trans;
    cl_kernel kernel_gemm_noshuffle_q4_k_q8_1_dp4a;
    cl_kernel kernel_gemm_noshuffle_q4_k_q8_1_dp4a_wimg;
    cl_kernel kernel_gemm_noshuffle_q5_k_q8_1_dp4a;
    cl_kernel kernel_gemm_noshuffle_q6_k_q8_1_dp4a;
    cl_kernel kernel_quant_a_q8_1;
    cl_kernel kernel_gemm_noshuffle_q4_k_f32_r1;
    cl_kernel kernel_gemm_noshuffle_q4_k_f32_kimg;
    cl_kernel kernel_gemm_noshuffle_q4_k_f32_cok;
    cl_kernel kernel_gemv_noshuffle_q6_K_f32;
    cl_kernel kernel_gemv_noshuffle_q6_K_f32_o4;
    cl_kernel kernel_gemv_noshuffle_q6_K_f32_o4_global;
    cl_kernel kernel_gemv_noshuffle_q6_K_f32_tiled;
    cl_kernel kernel_gemv_noshuffle_q6_K_f32_tiled_mc3;
    cl_kernel kernel_gemm_noshuffle_q6_K_f32_tiled;
    cl_kernel kernel_gemv_noshuffle_q6_K_f32_mc3;
    cl_kernel kernel_gemm_noshuffle_q6_K_f32;
    cl_kernel kernel_gemm_noshuffle_q6_K_f32_cok;
    cl_kernel kernel_gemm_noshuffle_q6_k_f32_32b_trans_ila_a8_bin;
    cl_kernel kernel_gemm_noshuffle_q6_k_q8_1_dp4a_ila_a8_bin;
    cl_kernel kernel_gemv_noshuffle_q6_k_f32_32b_trans;
    cl_kernel kernel_gemv_noshuffle_q5_k_f32;
    cl_kernel kernel_gemv_noshuffle_q5_k_f32_mc3;
    cl_kernel kernel_gemm_noshuffle_q5_k_f32;
    cl_kernel kernel_gemv_noshuffle_q5_0_f32;
    cl_kernel kernel_gemm_noshuffle_q5_0_f32;
    cl_kernel kernel_gemm_noshuffle_q5_0_q8_1_dp4a;
    cl_kernel kernel_gemm_noshuffle_q5_0_q8_1_dp4a_wimg;
    cl_kernel kernel_gemv_noshuffle_q5_1_f32;
    cl_kernel kernel_gemm_noshuffle_q5_1_f32;
    cl_kernel kernel_gemv_noshuffle_iq4_nl_f32;
    cl_kernel kernel_gemm_noshuffle_iq4_nl_f32;
    cl_kernel kernel_gemm_noshuffle_iq4_nl_q8_1_dp4a;
    cl_kernel kernel_gemm_noshuffle_q4_0_q8_1_dp4a;
#endif
};

struct ggml_opencl_mul_mat_id_kernels {
    cl_kernel kernel_add_id_add_id_swiglu_oai;
    cl_kernel kernel_gemv_moe_q4_0_f32_ns;
    cl_kernel kernel_gemm_moe_q4_0_f32_ns;
    cl_kernel kernel_gemm_moe_q4_0_f32_ns_bin;
    cl_kernel kernel_gemm_moe_q8_0_f32_ns;
    cl_kernel kernel_gemv_moe_q4_1_f32_ns;
    cl_kernel kernel_gemm_moe_q4_1_f32_ns;
    cl_kernel kernel_gemm_moe_q4_1_f32_ns_bin;
    cl_kernel kernel_gemv_moe_q5_0_f32_ns;
    cl_kernel kernel_gemm_moe_q5_0_f32_ns;
    cl_kernel kernel_gemv_moe_q5_1_f32_ns;
    cl_kernel kernel_gemm_moe_q5_1_f32_ns;
    cl_kernel kernel_gemv_moe_q4_k_f32_ns;
    cl_kernel kernel_gemm_moe_q4_k_f32_ns;
    cl_kernel kernel_gemm_moe_q4_k_f32_ns_bin;
    cl_kernel kernel_gemv_moe_q4_k_f32_ns_wimg;
    cl_kernel kernel_gemm_moe_q4_k_q8_1_dp4a;
    cl_kernel kernel_moe_reorder_quant_a_q8_1;
    cl_kernel kernel_gemm_moe_q8_1_dp4a_q80;
    cl_kernel kernel_gemm_moe_q8_1_dp4a_q50;
    cl_kernel kernel_gemm_moe_q8_1_dp4a_q5k;
    cl_kernel kernel_gemv_moe_q5_k_f32_ns;
    cl_kernel kernel_gemm_moe_q5_k_f32_ns;
    cl_kernel kernel_gemv_moe_q6_k_f32_ns;
    cl_kernel kernel_gemm_moe_q6_k_f32_ns;
    cl_kernel kernel_gemm_moe_q6_k_f32_ns_bin;
    cl_kernel kernel_gemm_moe_q6_k_q8_1_dp4a;
    cl_kernel kernel_gemv_moe_mxfp4_f32;
    cl_kernel kernel_gemm_moe_mxfp4_f32;
    cl_kernel kernel_gemv_moe_mxfp4_f32_ns;
    cl_kernel kernel_gemm_moe_mxfp4_f32_ns;
    cl_kernel kernel_gemm_moe_mxfp4_f32_ns_bin;
    cl_kernel kernel_gemv_moe_mxfp4_f32_ns_wimg;
    cl_kernel kernel_gemm_moe_mxfp4_q8_1_dp4a;
    cl_kernel kernel_gemm_moe_q4_0_q8_1_dp4a;
    cl_kernel kernel_gemm_moe_mxfp4_q8_1_dp4a_bin;
    cl_kernel kernel_gemm_moe_q4_0_q8_1_dp4a_bin;
    cl_kernel kernel_moe_reorder_b;
    cl_kernel kernel_moe_histogram;
    cl_kernel kernel_moe_scan;
    cl_kernel kernel_moe_fill;
    cl_kernel kernel_moe_scatter;
    cl_kernel kernel_moe_scatter_stable;
    cl_kernel kernel_moe_combine_f32;
    cl_kernel kernel_moe_combine_bias_f32;
    cl_kernel kernel_mul_mv_id_q4_0_f32_8x_flat;
    cl_kernel kernel_mul_mv_id_q8_0_f32;
    cl_kernel kernel_mul_mv_id_q8_0_f32_flat;
    cl_kernel kernel_mul_mv_id_mxfp4_f32;
    cl_kernel kernel_mul_mv_id_mxfp4_f32_flat;
};

void ggml_cl_init_fa_dims_table();
bool ggml_opencl_ensure_fa_f32_f16_prefill_512(
        ggml_backend_opencl_context * backend_ctx, bool split);
bool use_fa_bin_kernels_prefill(
        const ggml_backend_opencl_context * backend_ctx,
        const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v);

void ggml_cl_nop(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_get_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_set_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_add(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_add_id(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_div(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_sub(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_sqr(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_sqrt(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mean(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_ssm_conv(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_ssm_scan(ggml_backend_t backend, ggml_tensor * dst);
void ggml_cl_gelu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_gelu_erf(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_gelu_quick(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_silu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_relu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_sigmoid(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_tri(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_fill(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_clamp(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_rms_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_group_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_l2_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_tanh(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_neg(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_exp(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_expm1(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_abs(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_sgn(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_step(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_elu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_hardswish(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_hardsigmoid(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_floor(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_ceil(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_round(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_trunc(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_softplus(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_repeat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_concat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_conv_2d(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat_id(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cl_scale(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_cpy(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_dup(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_cont(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_reshape(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_view(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_permute(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_transpose(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_set(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_diag_mask_inf(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_diag(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_soft_max(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_rope(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_solve_tri(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_im2col(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_argsort(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_sum_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_cumsum(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_glu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_pad(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst);
void ggml_cl_upscale(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst);
void ggml_cl_timestep_embedding(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst);
void ggml_cl_flash_attn(ggml_backend_t backend, const ggml_tensor * q, const ggml_tensor * k, ggml_tensor * dst);
void ggml_cl_gated_delta_net(ggml_backend_t backend, ggml_tensor * dst);

// adreno mul_mat
bool ggml_cl_can_use_adreno_xmem_gemm_f16_f32(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cl_mul_mat_f16_f32_adreno_xmem(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_cl_mul_mat_f16_f32_attn_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat_kq_kqv_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool is_kq);
void ggml_cl_mul_mat_q1_0_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat_q4_0_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat_q4_1_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat_q5_0_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat_q5_1_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat_iq4_nl_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat_q8_0_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat_q4_K_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat_q6_K_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cl_mul_mat_q5_K_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// fused ops
void ggml_opencl_op_rms_norm_fused(ggml_backend_t backend, ggml_tensor * rms_norm_tensor, ggml_tensor * mul_tensor);
void ggml_opencl_op_rms_norm_mul_add_fused(ggml_backend_t backend, ggml_tensor * rms_norm_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor);
void ggml_opencl_op_norm_fused(ggml_backend_t backend, ggml_tensor * norm_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor);
void ggml_opencl_op_group_norm_fused(ggml_backend_t backend, ggml_tensor * gn_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor);
void ggml_cl_moe_combine_fused(ggml_backend_t backend, const ggml_tensor * mul, const ggml_tensor * dst);
void ggml_cl_moe_bias_glu_fused(ggml_backend_t backend, ggml_tensor * gate_mm, const ggml_tensor * gate_add,
                                ggml_tensor * up_mm, const ggml_tensor * up_add, const ggml_tensor * glu);
void ggml_cl_moe_bias_combine_fused(ggml_backend_t backend, const ggml_tensor * add,
                                    const ggml_tensor * mul, const ggml_tensor * dst);
void ggml_cl_mul_mat_q4_K_glu_fused(ggml_backend_t backend, ggml_tensor * gate_tensor, ggml_tensor * up_tensor, ggml_tensor * glu_tensor);

// kernel loaders
void ggml_cl_load_kernels_nop(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_unary_ext(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_get_rows(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_set_rows(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_add(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_add_id(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_mul(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_div(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_sub(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_sqr(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_sqrt(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_mean(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_ssm_conv(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_ssm_scan(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_gelu(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_silu(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_relu(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_sigmoid(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_tri(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_fill(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_clamp(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_norm(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_rms_norm(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_group_norm(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_l2_norm(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_tanh(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_neg(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_exp(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_expm1(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_abs(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_softplus(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_repeat(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_pad(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_upscale(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_concat(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_timestep_embedding(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_flash_attn(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_mul_mat(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_mul_mat_adreno(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_conv_2d(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_mul_mat_id(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_scale(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_cpy(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_dup(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_cont(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_reshape(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_view(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_permute(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_transpose(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_set(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_diag_mask_inf(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_diag(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_soft_max(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_rope(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_solve_tri(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_im2col(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_argsort(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_sum_rows(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_cumsum(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_glu(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_gated_delta_net(ggml_backend_opencl_context * backend_ctx);
