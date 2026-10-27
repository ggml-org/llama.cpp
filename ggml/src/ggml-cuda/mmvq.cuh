#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

void ggml_cuda_spark_qkv_gate(ggml_backend_cuda_context & ctx,
    const ggml_tensor * qkv, const ggml_tensor * gate, const ggml_tensor * input,
    ggml_tensor * qkv_dst, ggml_tensor * gate_dst, const void * quantized = nullptr);

void ggml_cuda_spark_qkv_gate_rope(ggml_backend_cuda_context & ctx,
    const ggml_tensor * qkv, const ggml_tensor * gate, const ggml_tensor * input,
    ggml_tensor * q_rope, ggml_tensor * k_rows, ggml_tensor * v_rows,
    ggml_tensor * gate_dst, const void * quantized);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr,
    const void * src1_quantized = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

void ggml_cuda_spark_ffn_up_q8(ggml_backend_cuda_context & ctx,
    const ggml_tensor * up, const ggml_tensor * gate, const void * input_q8,
    void * output_q8);

bool ggml_cuda_spark_ffn_norm_up_q8(ggml_backend_cuda_context & ctx,
    const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * input,
    const ggml_tensor * norm_weight, float eps, void * output_q8);
