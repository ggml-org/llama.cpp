#pragma once

#include "common.cuh"

// MUL_MAT_ID with few tokens on CDNA, RDNA3 and RDNA4 for Q4_K, Q5_K, Q5_1 and Q8_0, see mmvq-moe-cdna.cu.

// the type, shape and alignment of src0, without the fusion and dst checks of ggml_cuda_mmvq_moe_cdna_supported
bool ggml_cuda_mmvq_moe_cdna_src0_supported(const ggml_tensor * src0, int cc);

bool ggml_cuda_mmvq_moe_cdna_supported(const ggml_tensor * src0, const ggml_tensor * dst, int cc, const ggml_cuda_mm_fusion_args_device & fusion);

void ggml_cuda_mmvq_moe_cdna(
        const ggml_tensor * src0, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        int64_t ncols_dst, int64_t nchannels_dst, int64_t nchannels_y, int64_t stride_col_y, int64_t stride_channel_y,
        int64_t stride_col_dst, int64_t stride_channel_dst, int64_t ids_stride, cudaStream_t stream);
