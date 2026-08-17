#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

void ggml_cuda_op_ssm_gate_beta(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0_alpha, const ggml_tensor * src0_beta, const ggml_tensor * src1,
    const ggml_tensor * dt, const ggml_tensor * ssm_a,
    ggml_tensor * dst_gate, ggml_tensor * dst_beta);

void ggml_cuda_op_ssm_conv_l2_gatebeta(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * conv_input, const ggml_tensor * conv_w,
    const ggml_tensor * src0_alpha, const ggml_tensor * src0_beta, const ggml_tensor * src1,
    const ggml_tensor * dt, const ggml_tensor * ssm_a,
    ggml_tensor * q_norm, ggml_tensor * k_norm, ggml_tensor * v_raw,
    ggml_tensor * dst_gate, ggml_tensor * dst_beta,
    const int head_k_dim, const int n_qk_heads,
    const int head_v_dim, const int n_v_heads, const float eps);

void ggml_cuda_op_shexp_down_gate(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * w_down, const ggml_tensor * swiglu,
    const ggml_tensor * w_gate, const ggml_tensor * x_gate,
    const ggml_tensor * moe_out, const ggml_tensor * ffn_residual,
    ggml_tensor * dst);
