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

// MoE expert cache: one batched matvec over slot-pool experts selected by a
// device-side index array. Computes, for c in [0, n_hits):
//   dst[c*n_out .. ] = W[ids[c]] (n_out x n_in) . act_q8[c % act_rows]
// where W[i] starts at (char *)pool + i*slot_stride_bytes (slot_stride_bytes
// must be a multiple of the type's block size, i.e. the original tensor's
// nb[2]). act_q8 holds act_rows quantized activation rows in the standard
// padded q8_1 layout produced by quantize_row_q8_1_cuda.
// gate_pool: optional parallel slab of gate weights (same slot stride); when
// non-null the kernel computes dst = up_dot * glu(gate_dot) on-chip (glu_op).
void ggml_cuda_moe_cache_mmv(
    const void * pool, ggml_type type0, const char * act_q8, const int32_t * ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_slots,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows, cudaStream_t stream,
    const void * gate_pool, int glu_op);
