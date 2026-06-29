#include "col2im_1d.cuh"




// 1. Kernel body
template <typename T>
static __global__ void col2im_1d_kernel(
        const T * src, T * dst,
        int64_t T_out, int64_t T_in, int64_t K, int64_t OC, int64_t K_OC,
        int s0, int p0) {
    
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= T_out * OC) {
        return; // Guard out-of-bounds threads
    }

    // Decompose 1D thread ID into 2D output coordinates
    // dst layout: [T_out, OC] -> flat index = oc * T_out + t_out
    const int64_t oc = i / T_out;
    const int64_t t_out = i % T_out;

    const int64_t t_abs = t_out + p0;

    // Establish bounds using the solved equation
    int64_t t_in_min = (t_abs - K + 1 + s0 - 1) / s0; 
    if (t_in_min < 0) t_in_min = 0;
    
    int64_t t_in_max = t_abs / s0;
    if (t_in_max >= T_in) t_in_max = T_in - 1;

    float sum = 0.0f; // Accumulate in fp32
    for (int64_t t_in = t_in_min; t_in <= t_in_max; t_in++) {
        int64_t k = t_abs - t_in * s0;
        
        // src layout: [K_OC, T_in] -> element (oc * K + k) at column t_in
        // flat offset = t_in * K_OC + (oc * K + k)
        sum += (float)src[t_in * K_OC + (oc * K + k)];
    }

    // Write directly to current thread's mapped location
    dst[i] = (T)sum;
}


// 2. Host function wrapper
void ggml_cuda_op_col2im_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    cudaStream_t stream = ctx.stream();

    // Strict type pairing and format enforcement
    GGML_ASSERT(src->type == dst->type);
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_BF16);

    // Extract structural parameters
    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t OC = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];

    const int64_t K_OC  = src->ne[0];
    const int64_t T_in  = src->ne[1];
    const int64_t K     = K_OC / OC;
    const int64_t T_out = dst->ne[0];

    // Grid configuration
    const int64_t total_elements = T_out * OC;
    const int block_size = 256; 
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // Type dispatching
    if (src->type == GGML_TYPE_F32) {
        col2im_1d_kernel<float><<<num_blocks, block_size, 0, stream>>>(
            (const float *)src->data, (float *)dst->data, T_out, T_in, K, OC, K_OC, s0, p0);
    } else if (src->type == GGML_TYPE_F16) {
        col2im_1d_kernel<half><<<num_blocks, block_size, 0, stream>>>(
            (const half *)src->data, (half *)dst->data, T_out, T_in, K, OC, K_OC, s0, p0);
    } else if (src->type == GGML_TYPE_BF16) {
        col2im_1d_kernel<nv_bfloat16><<<num_blocks, block_size, 0, stream>>>(
            (const nv_bfloat16 *)src->data, (nv_bfloat16 *)dst->data, T_out, T_in, K, OC, K_OC, s0, p0);
    } else {
        GGML_ABORT("col2im_1d: unsupported type %d", src->type);
    }
}
