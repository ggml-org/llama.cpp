/**
 * CUDA Concat Kernel for MLA Tensor Shapes
 * 
 * This file implements optimized CUDA kernels for the concat operation
 * specifically designed for MLA (Multi-head Latent Attention) tensor patterns.
 * 
 * MLA uses concat to combine k_nope and k_pe_repeated:
 * - k_nope: [qk_nope_head_dim, n_head, n_tokens] = [128, 64, n_tokens]
 * - k_pe_repeated: [qk_rope_head_dim, n_head, n_tokens] = [64, 64, n_tokens]
 * - K (output): [n_embd_head_k_mla, n_head, n_tokens] = [192, 64, n_tokens]
 * 
 * **Feature: mla-flash-attention-fix**
 * **Validates: Requirements 3.1, 3.2**
 */

#include "concat.cuh"

// =============================================================================
// Float32 Contiguous Kernels
// =============================================================================

static __global__ void concat_f32_dim0(const float * x, const float * y, float * dst, const int ne0, const int ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim1(const float * x, const float * y, float * dst, const int ne0, const int ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < (unsigned)ne01) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim2(const float * x, const float * y, float * dst, const int ne0, const int ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < (unsigned)ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 *  gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_cuda(const float * x, const float * y, float * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2);
    if (dim == 0) {
        concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00);
        return;
    }
    if (dim == 1) {
        concat_f32_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne01);
        return;
    }
    concat_f32_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
}

// =============================================================================
// Float16 Contiguous Kernels
// =============================================================================

static __global__ void concat_f16_dim0(const half * x, const half * y, half * dst, const int ne0, const int ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f16_dim1(const half * x, const half * y, half * dst, const int ne0, const int ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < (unsigned)ne01) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f16_dim2(const half * x, const half * y, half * dst, const int ne0, const int ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < (unsigned)ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 *  gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f16_cuda(const half * x, const half * y, half * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2);
    if (dim == 0) {
        concat_f16_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00);
        return;
    }
    if (dim == 1) {
        concat_f16_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne01);
        return;
    }
    concat_f16_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
}

// =============================================================================
// Non-contiguous Kernels (slow)
// =============================================================================

// non-contiguous kernel for float32 (slow)
template <int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_f32_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const float * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const float *) (src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const float *) (src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        float * y = (float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

// non-contiguous kernel for float16 (slow)
template <int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_f16_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const half * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const half *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const half *) (src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const half *) (src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const half *) (src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const half *) (src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        half * y = (half *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


// =============================================================================
// Support Check Function
// =============================================================================

/**
 * Check if the concat operation is supported for MLA tensor shapes on CUDA.
 * 
 * MLA-specific concat patterns:
 * - Concatenating k_nope [128, n_head, n_tokens] with k_pe_repeated [64, n_head, n_tokens]
 * - Result: K [192, n_head, n_tokens]
 * - Concat along dimension 0
 * 
 * Supported data types:
 * - GGML_TYPE_F32 (float32)
 * - GGML_TYPE_F16 (float16)
 * 
 * **Feature: mla-flash-attention-fix**
 * **Validates: Requirements 3.1, 3.4**
 */
bool ggml_cuda_concat_mla_supported(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    
    // Check data type support - F32 and F16 are supported
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        return false;
    }
    
    // Both sources must have the same type
    if (src0->type != src1->type) {
        return false;
    }
    
    // Destination must have the same type
    if (dst->type != src0->type) {
        return false;
    }
    
    // Get concat dimension
    const int32_t dim = ((int32_t *) dst->op_params)[0];
    
    // Check for valid dimension (0-3)
    if (dim < 0 || dim > 3) {
        return false;
    }
    
    // Check for MLA-specific pattern: concat along dimension 0
    // k_nope: [qk_nope_head_dim, n_head, n_tokens]
    // k_pe_repeated: [qk_rope_head_dim, n_head, n_tokens]
    // Result: [qk_nope_head_dim + qk_rope_head_dim, n_head, n_tokens]
    const bool is_mla_pattern = (dim == 0) &&
                                 (src0->ne[1] == src1->ne[1]) &&  // Same n_head
                                 (src0->ne[2] == src1->ne[2]) &&  // Same n_tokens
                                 (src0->ne[3] == src1->ne[3]);    // Same batch
    
    // Also support general concat patterns
    // For dim=0: ne[1], ne[2], ne[3] must match
    // For dim=1: ne[0], ne[2], ne[3] must match
    // For dim=2: ne[0], ne[1], ne[3] must match
    // For dim=3: ne[0], ne[1], ne[2] must match
    bool dims_compatible = true;
    for (int i = 0; i < 4; i++) {
        if (i != dim && src0->ne[i] != src1->ne[i]) {
            dims_compatible = false;
            break;
        }
    }
    
    return dims_compatible || is_mla_pattern;
}

// =============================================================================
// Main Concat Operation
// =============================================================================

/**
 * Execute the CUDA concat operation.
 * 
 * Supports both F32 and F16 data types, with optimized kernels for
 * contiguous tensors and a general kernel for non-contiguous tensors.
 * 
 * **Feature: mla-flash-attention-fix**
 * **Validates: Requirements 3.1, 3.2, 3.3**
 */
void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    // Support both F32 and F16
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == src0->type);
    GGML_ASSERT(dst->type  == src0->type);

    const bool is_f16 = (src0->type == GGML_TYPE_F16);
    const size_t type_size = is_f16 ? sizeof(half) : sizeof(float);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        if (is_f16) {
            // Float16 path
            const half * src0_d = (const half *)src0->data;
            const half * src1_d = (const half *)src1->data;
            half * dst_d = (half *)dst->data;

            if (dim != 3) {
                for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                    concat_f16_cuda(
                            src0_d + i3 * (src0->nb[3] / type_size),
                            src1_d + i3 * (src1->nb[3] / type_size),
                            dst_d + i3 * ( dst->nb[3] / type_size),
                            src0->ne[0], src0->ne[1], src0->ne[2],
                            dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
                }
            } else {
                const size_t size0 = ggml_nbytes(src0);
                const size_t size1 = ggml_nbytes(src1);

                CUDA_CHECK(cudaMemcpyAsync(dst_d,                        src0_d, size0, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(dst_d + size0/type_size, src1_d, size1, cudaMemcpyDeviceToDevice, stream));
            }
        } else {
            // Float32 path (original implementation)
            const float * src0_d = (const float *)src0->data;
            const float * src1_d = (const float *)src1->data;
            float * dst_d = (float *)dst->data;

            if (dim != 3) {
                for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                    concat_f32_cuda(
                            src0_d + i3 * (src0->nb[3] / type_size),
                            src1_d + i3 * (src1->nb[3] / type_size),
                            dst_d + i3 * ( dst->nb[3] / type_size),
                            src0->ne[0], src0->ne[1], src0->ne[2],
                            dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
                }
            } else {
                const size_t size0 = ggml_nbytes(src0);
                const size_t size1 = ggml_nbytes(src1);

                CUDA_CHECK(cudaMemcpyAsync(dst_d,                        src0_d, size0, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(dst_d + size0/type_size, src1_d, size1, cudaMemcpyDeviceToDevice, stream));
            }
        }
    } else {
        // Non-contiguous path
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        
        if (is_f16) {
            // Float16 non-contiguous
            auto launch_kernel_f16 = [&](auto dim_val) {
                concat_f16_non_cont<dim_val><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                    (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            };
            switch (dim) {
                case 0:
                    launch_kernel_f16(std::integral_constant<int, 0>{});
                    break;
                case 1:
                    launch_kernel_f16(std::integral_constant<int, 1>{});
                    break;
                case 2:
                    launch_kernel_f16(std::integral_constant<int, 2>{});
                    break;
                case 3:
                    launch_kernel_f16(std::integral_constant<int, 3>{});
                    break;
                default:
                    GGML_ABORT("Invalid dim: %d", dim);
                    break;
            }
        } else {
            // Float32 non-contiguous (original implementation)
            auto launch_kernel_f32 = [&](auto dim_val) {
                concat_f32_non_cont<dim_val><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                    (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            };
            switch (dim) {
                case 0:
                    launch_kernel_f32(std::integral_constant<int, 0>{});
                    break;
                case 1:
                    launch_kernel_f32(std::integral_constant<int, 1>{});
                    break;
                case 2:
                    launch_kernel_f32(std::integral_constant<int, 2>{});
                    break;
                case 3:
                    launch_kernel_f32(std::integral_constant<int, 3>{});
                    break;
                default:
                    GGML_ABORT("Invalid dim: %d", dim);
                    break;
            }
        }
    }
}
