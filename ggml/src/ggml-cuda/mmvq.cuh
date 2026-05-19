#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// Returns true if a quantized matmul of shape (..., ne11) on a device with
// compute capability `cc` should take the MMVQ (per-row GEMV) path.
// Returning false sends it to the MMQ path (batched GEMM, MFMA-tiled on CDNA).
//
// On AMD MFMA hardware (CDNA) the optimal batch threshold is quant-dependent:
// K-quants have a heavier per-row GEMV (block scales + super-block decode), so
// MFMA-tiled MMQ overtakes MMVQ at a smaller batch; legacy and IQ quants have
// lean GEMV kernels that stay ahead until the batch nearly fills an MFMA tile.
// Thresholds calibrated on MI250X with Llama-3.2-3B (pp512, ubatch 1..8) — see
// the PR description for the full sweep.
//
// Set GGML_CUDA_FORCE_MMVQ=1 to restore the original global threshold.
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
