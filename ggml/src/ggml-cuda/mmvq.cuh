#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// RDNA4: the columns up to which gate/up + GLU run as one fused MMVQ (elsewhere 1). The other types' fused kernels
// spill registers past 2 columns.
static constexpr __host__ __device__ int mmvq_fusion_max_ncols_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q4_K:
            return 4;
        default:
            return 1;
    }
}

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

// gate = softplus(w_alpha*x + dt) * a and beta = sigmoid(w_beta*x) for a gated delta net, Q8_0 weights, 2 to
// MMVQ_MAX_BATCH_SIZE columns of x; bit-identical to the matmuls run as mmvq and the unfused element-wise ops
void ggml_cuda_op_gdn_gate_beta(
    ggml_backend_cuda_context & ctx, const ggml_tensor * w_alpha, const ggml_tensor * w_beta, const ggml_tensor * x,
    const ggml_tensor * dt, const ggml_tensor * a, ggml_tensor * gate, ggml_tensor * beta);
