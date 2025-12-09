/**
 * CUDA Repeat Kernel for MLA Tensor Shapes
 * 
 * This file implements optimized CUDA kernels for the repeat operation
 * specifically designed for MLA (Multi-head Latent Attention) tensor patterns.
 * 
 * MLA uses repeat to broadcast k_pe from [qk_rope_head_dim, 1, n_tokens]
 * to [qk_rope_head_dim, n_head, n_tokens].
 * 
 * **Feature: mla-flash-attention-fix**
 * **Validates: Requirements 2.1, 2.2**
 */

#include "repeat.cuh"

// Maximum grid dimension for CUDA
#define MAX_GRIDDIM_X 0x7FFFFFFF

/**
 * CUDA kernel for float32 repeat operation optimized for MLA patterns.
 * 
 * This kernel handles the specific case of repeating along dimension 1:
 * [ne0, 1, ne2, ne3] -> [ne0, nr1, ne2, ne3]
 * 
 * Memory access pattern is optimized for coalescing:
 * - Threads in a warp access consecutive elements in ne0 dimension
 * - Each thread block handles a tile of the output tensor
 * 
 * @param src Source tensor data
 * @param dst Destination tensor data
 * @param ne0 Size of dimension 0 (innermost, e.g., qk_rope_head_dim)
 * @param ne1 Size of dimension 1 in output (e.g., n_head)
 * @param ne2 Size of dimension 2 (e.g., n_tokens)
 * @param ne3 Size of dimension 3
 * @param ne10 Size of dimension 0 in source (same as ne0)
 * @param ne11 Size of dimension 1 in source (typically 1 for MLA)
 * @param ne12 Size of dimension 2 in source (same as ne2)
 * @param ne13 Size of dimension 3 in source (same as ne3)
 */
static __global__ void repeat_f32_mla_kernel(
    const float * __restrict__ src,
    float * __restrict__ dst,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    const int64_t ne3,
    const int64_t ne10,
    const int64_t ne11,
    const int64_t ne12,
    const int64_t ne13,
    const int64_t nb1,
    const int64_t nb2,
    const int64_t nb3,
    const int64_t nb11,
    const int64_t nb12,
    const int64_t nb13
) {
    // Calculate global thread index
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elements = ne0 * ne1 * ne2 * ne3;
    
    if (idx >= total_elements) {
        return;
    }
    
    // Decompose linear index into multi-dimensional indices
    // Using fast integer division for better performance
    const int64_t i0 = idx % ne0;
    const int64_t i1 = (idx / ne0) % ne1;
    const int64_t i2 = (idx / (ne0 * ne1)) % ne2;
    const int64_t i3 = idx / (ne0 * ne1 * ne2);
    
    // Map output indices to source indices (with modulo for repeat)
    const int64_t i10 = i0 % ne10;
    const int64_t i11 = i1 % ne11;
    const int64_t i12 = i2 % ne12;
    const int64_t i13 = i3 % ne13;
    
    // Calculate source and destination offsets
    // Source offset uses source strides
    const int64_t src_offset = i10 + i11 * (nb11 / sizeof(float)) + 
                               i12 * (nb12 / sizeof(float)) + 
                               i13 * (nb13 / sizeof(float));
    
    // Destination offset uses destination strides
    const int64_t dst_offset = i0 + i1 * (nb1 / sizeof(float)) + 
                               i2 * (nb2 / sizeof(float)) + 
                               i3 * (nb3 / sizeof(float));
    
    dst[dst_offset] = src[src_offset];
}

/**
 * CUDA kernel for float16 repeat operation optimized for MLA patterns.
 */
static __global__ void repeat_f16_mla_kernel(
    const half * __restrict__ src,
    half * __restrict__ dst,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    const int64_t ne3,
    const int64_t ne10,
    const int64_t ne11,
    const int64_t ne12,
    const int64_t ne13,
    const int64_t nb1,
    const int64_t nb2,
    const int64_t nb3,
    const int64_t nb11,
    const int64_t nb12,
    const int64_t nb13
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elements = ne0 * ne1 * ne2 * ne3;
    
    if (idx >= total_elements) {
        return;
    }
    
    const int64_t i0 = idx % ne0;
    const int64_t i1 = (idx / ne0) % ne1;
    const int64_t i2 = (idx / (ne0 * ne1)) % ne2;
    const int64_t i3 = idx / (ne0 * ne1 * ne2);
    
    const int64_t i10 = i0 % ne10;
    const int64_t i11 = i1 % ne11;
    const int64_t i12 = i2 % ne12;
    const int64_t i13 = i3 % ne13;
    
    const int64_t src_offset = i10 + i11 * (nb11 / sizeof(half)) + 
                               i12 * (nb12 / sizeof(half)) + 
                               i13 * (nb13 / sizeof(half));
    
    const int64_t dst_offset = i0 + i1 * (nb1 / sizeof(half)) + 
                               i2 * (nb2 / sizeof(half)) + 
                               i3 * (nb3 / sizeof(half));
    
    dst[dst_offset] = src[src_offset];
}

/**
 * Launch the float32 repeat kernel for MLA tensor shapes.
 */
static void repeat_f32_mla_cuda(
    const float * src,
    float * dst,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    const int64_t ne3,
    const int64_t ne10,
    const int64_t ne11,
    const int64_t ne12,
    const int64_t ne13,
    const int64_t nb1,
    const int64_t nb2,
    const int64_t nb3,
    const int64_t nb11,
    const int64_t nb12,
    const int64_t nb13,
    cudaStream_t stream
) {
    const int64_t total_elements = ne0 * ne1 * ne2 * ne3;
    const int64_t num_blocks = (total_elements + CUDA_REPEAT_BLOCK_SIZE - 1) / CUDA_REPEAT_BLOCK_SIZE;
    
    repeat_f32_mla_kernel<<<MIN(MAX_GRIDDIM_X, num_blocks), CUDA_REPEAT_BLOCK_SIZE, 0, stream>>>(
        src, dst, ne0, ne1, ne2, ne3, ne10, ne11, ne12, ne13, nb1, nb2, nb3, nb11, nb12, nb13
    );
}

/**
 * Launch the float16 repeat kernel for MLA tensor shapes.
 */
static void repeat_f16_mla_cuda(
    const half * src,
    half * dst,
    const int64_t ne0,
    const int64_t ne1,
    const int64_t ne2,
    const int64_t ne3,
    const int64_t ne10,
    const int64_t ne11,
    const int64_t ne12,
    const int64_t ne13,
    const int64_t nb1,
    const int64_t nb2,
    const int64_t nb3,
    const int64_t nb11,
    const int64_t nb12,
    const int64_t nb13,
    cudaStream_t stream
) {
    const int64_t total_elements = ne0 * ne1 * ne2 * ne3;
    const int64_t num_blocks = (total_elements + CUDA_REPEAT_BLOCK_SIZE - 1) / CUDA_REPEAT_BLOCK_SIZE;
    
    repeat_f16_mla_kernel<<<MIN(MAX_GRIDDIM_X, num_blocks), CUDA_REPEAT_BLOCK_SIZE, 0, stream>>>(
        src, dst, ne0, ne1, ne2, ne3, ne10, ne11, ne12, ne13, nb1, nb2, nb3, nb11, nb12, nb13
    );
}

bool ggml_cuda_repeat_mla_supported(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src = dst->src[0];
    
    // Check data type support
    if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) {
        return false;
    }
    
    if (dst->type != src->type) {
        return false;
    }
    
    // Check for MLA-specific pattern: repeat along dimension 1
    // Source shape: [ne0, 1, ne2, ne3] or similar
    // This is the k_pe broadcast pattern
    const bool is_mla_pattern = (src->ne[1] == 1 && dst->ne[1] > 1) ||
                                 (src->ne[2] == 1 && dst->ne[2] > 1) ||
                                 (src->ne[3] == 1 && dst->ne[3] > 1);
    
    // Also support general repeat patterns
    // Check that dimensions are compatible (dst dimensions are multiples of src)
    const bool dims_compatible = 
        (dst->ne[0] % src->ne[0] == 0) &&
        (dst->ne[1] % src->ne[1] == 0) &&
        (dst->ne[2] % src->ne[2] == 0) &&
        (dst->ne[3] % src->ne[3] == 0);
    
    return dims_compatible && (is_mla_pattern || 
        (src->ne[0] == dst->ne[0] && src->ne[1] == dst->ne[1] && 
         src->ne[2] == dst->ne[2] && src->ne[3] == dst->ne[3]));
}

void ggml_cuda_op_repeat_mla(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    
    GGML_ASSERT(src->type == dst->type);
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    
    cudaStream_t stream = ctx.stream();
    
    // Get tensor dimensions
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];
    
    const int64_t ne10 = src->ne[0];
    const int64_t ne11 = src->ne[1];
    const int64_t ne12 = src->ne[2];
    const int64_t ne13 = src->ne[3];
    
    // Get tensor strides (in bytes)
    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];
    
    const int64_t nb11 = src->nb[1];
    const int64_t nb12 = src->nb[2];
    const int64_t nb13 = src->nb[3];
    
    if (src->type == GGML_TYPE_F32) {
        const float * src_d = (const float *)src->data;
        float * dst_d = (float *)dst->data;
        
        repeat_f32_mla_cuda(
            src_d, dst_d,
            ne0, ne1, ne2, ne3,
            ne10, ne11, ne12, ne13,
            nb1, nb2, nb3,
            nb11, nb12, nb13,
            stream
        );
    } else if (src->type == GGML_TYPE_F16) {
        const half * src_d = (const half *)src->data;
        half * dst_d = (half *)dst->data;
        
        repeat_f16_mla_cuda(
            src_d, dst_d,
            ne0, ne1, ne2, ne3,
            ne10, ne11, ne12, ne13,
            nb1, nb2, nb3,
            nb11, nb12, nb13,
            stream
        );
    }
}
