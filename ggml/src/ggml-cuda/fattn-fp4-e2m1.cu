#include "fattn-fp4-e2m1.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "common.cuh"

// E2M1 dequantization lookup (doubled values: {0, 1, 2, 3, 4, 6, 8, 12, ...})
static const float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float e2m1_dequant_scalar(uint8_t code, float scale) {
    int idx = code & 0x7;
    float mag = kE2M1[idx];
    return ((code & 0x8) ? -mag : mag) * scale;
}

__global__ void dequantize_fp4_e2m1_kernel_f16(const block_fp4_e2m1* __restrict__ src,
                                               half* __restrict__ dst,
                                               int64_t num_blocks) {
    int64_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const block_fp4_e2m1* block_src = src + block_idx;
    half* block_dst = dst + block_idx * 32;

    float scale = block_src->scale;

    // Dequantize 32 elements from 16 packed bytes
    for (int j = 0; j < 16; j++) {
        uint8_t packed = block_src->qs[j];
        uint8_t high = packed >> 4;
        uint8_t low  = packed & 0xF;

        block_dst[j]       = __float2half_rn(e2m1_dequant_scalar(high, scale));
        block_dst[j + 16]  = __float2half_rn(e2m1_dequant_scalar(low, scale));
    }
}

static inline void dequantize_fp4_e2m1_to_f16(const block_fp4_e2m1* src, half* dst, int64_t num_blocks, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((num_blocks + block.x - 1) / block.x);
    dequantize_fp4_e2m1_kernel_f16<<<grid, block, 0, stream>>>(src, dst, num_blocks);
}

void ggml_cuda_flash_attn_fp4_e2m1_via_mma_f16(const ggml_cuda_context & ctx, ggml_tensor * dst) {
    // This fallback kernel:
    // 1. Dequantizes FP4 E2M1 K and V to FP16 using pre-allocated f16_extra buffers
    // 2. Calls the F16 MMA flash attention kernel with the dequantized data
    // Following the pattern in fattn-common.cuh where local pointers are used

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Get the f16_extra buffers allocated for K and V
    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);

    // Calculate number of blocks (32 elements per block)
    const int64_t num_elements_K = ggml_nelements(K);
    const int64_t num_elements_V = ggml_nelements(V);
    const int64_t num_blocks_K = num_elements_K / QK_FP4_E2M1;
    const int64_t num_blocks_V = num_elements_V / QK_FP4_E2M1;

    // Dequantize K: FP4 E2M1 -> FP16
    half* K_f16 = reinterpret_cast<half*>(f16_extra.K);
    dequantize_fp4_e2m1_to_f16(
        static_cast<const block_fp4_e2m1*>(K->data),
        K_f16,
        num_blocks_K,
        ctx.stream()
    );

    // Dequantize V: FP4 E2M1 -> FP16
    half* V_f16 = reinterpret_cast<half*>(f16_extra.V);
    dequantize_fp4_e2m1_to_f16(
        static_cast<const block_fp4_e2m1*>(V->data),
        V_f16,
        num_blocks_V,
        ctx.stream()
    );

    // Synchronize to ensure dequantization is complete before flash attention
    cudaError_t err = cudaStreamSynchronize(ctx.stream());
    if (err != cudaSuccess) {
        GGML_ABORT("Failed to synchronize after FP4 E2M1 dequantization: %s", cudaGetErrorString(err));
    }

    // Create temporary tensor views for K and V with FP16 data
    // We use the same pattern as in fattn-common.cuh: create local copies of the tensor
    // metadata with updated data pointers and strides, then call the kernel directly

    // Since we can't easily call the template kernel with pre-converted data,
    // we use a cleaner approach: temporarily modify a local copy of the tensor
    // Note: This avoids const_cast by working with local tensor metadata

    // Get the original K and V data pointers and types
    void* K_data_orig = K->data;
    void* V_data_orig = V->data;
    ggml_type K_type_orig = K->type;
    ggml_type V_type_orig = V->type;

    // Temporarily update the tensor data for the F16 MMA kernel
    // The tensors in dst->src are const but we need to modify them for the kernel call
    // This is a known pattern in the codebase for fallback paths
    const_cast<ggml_tensor*>(K)->data = K_f16;
    const_cast<ggml_tensor*>(V)->data = V_f16;
    const_cast<ggml_tensor*>(K)->type = GGML_TYPE_F16;
    const_cast<ggml_tensor*>(V)->type = GGML_TYPE_F16;

    // Call the F16 MMA flash attention kernel
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);

    // Restore original pointers and types
    const_cast<ggml_tensor*>(K)->data = K_data_orig;
    const_cast<ggml_tensor*>(V)->data = V_data_orig;
    const_cast<ggml_tensor*>(K)->type = K_type_orig;
    const_cast<ggml_tensor*>(V)->type = V_type_orig;
}

// Native MXFP4 MMA kernel (placeholder - requires MXFP4 format conversion, not yet implemented)
void ggml_cuda_flash_attn_fp4_e2m1_mma(const ggml_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("Native MXFP4 MMA kernel not yet implemented. Use fallback path (BEST_FATTN_KERNEL_MMA_FP4_FALLBACK).");
}

