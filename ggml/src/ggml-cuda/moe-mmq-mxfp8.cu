#include "moe-mmq-mxfp8.cuh"

#if CUDART_VERSION >= 12080

static __device__ __forceinline__ uint8_t moe_mxfp8_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    constexpr float e4m3_max = 448.0f;
    const int       exponent = __float2int_ru(log2f(amax / e4m3_max));
    return (uint8_t) max(0, min(254, exponent + 127));
}

static __global__ void moe_quantize_scatter_mxfp8(const float * __restrict__ src,
                                                  const int32_t * __restrict__ ids_src1_inv,
                                                  block_mxfp8_mmq * __restrict__ dst,
                                                  int64_t ne00,
                                                  int64_t stride_token,
                                                  int64_t n_rows,
                                                  int     n_expert_used) {
    const int64_t token   = blockIdx.x;
    const int64_t k_block = blockIdx.y;
    const int     lane    = threadIdx.x;
    const int64_t k       = k_block * 32 + lane;
    const float   value   = k < ne00 ? src[token * stride_token + k] : 0.0f;

    float amax = fabsf(value);
#    pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
    }

    const uint8_t       scale = moe_mxfp8_scale(amax);
    const float         inv   = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
    const __nv_fp8_e4m3 quantized(value * inv);

    for (int slot = 0; slot < n_expert_used; ++slot) {
        const int64_t     row   = ids_src1_inv[token * n_expert_used + slot];
        block_mxfp8_mmq * block = dst + k_block * n_rows + row;
        block->qs[lane]         = quantized.__x;
        if (lane == 0) {
            block->scale = scale;
        }
    }
}

#endif

size_t ggml_cuda_moe_mxfp8_size(int64_t n_rows, int64_t ne0) {
    GGML_ASSERT(ne0 % 32 == 0);
    return n_rows * (ne0 / 32) * sizeof(block_mxfp8_mmq);
}

void ggml_cuda_moe_quantize_scatter_mxfp8(const float *   src,
                                          const int32_t * ids_src1_inv,
                                          void *          dst,
                                          int64_t         ne00,
                                          int64_t         stride_token,
                                          int64_t         ne0,
                                          int64_t         n_tokens,
                                          int64_t         n_rows,
                                          int             n_expert_used,
                                          cudaStream_t    stream) {
#if CUDART_VERSION >= 12080
    GGML_ASSERT(ne0 % 32 == 0);
    const dim3 blocks(n_tokens, ne0 / 32, 1);
    moe_quantize_scatter_mxfp8<<<blocks, WARP_SIZE, 0, stream>>>(src, ids_src1_inv, (block_mxfp8_mmq *) dst, ne00,
                                                                 stride_token, n_rows, n_expert_used);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(src, ids_src1_inv, dst, ne00, stride_token, ne0, n_tokens, n_rows, n_expert_used, stream);
    GGML_ABORT("MXFP8 MoE activations require CUDA 12.8 or newer");
#endif
}
