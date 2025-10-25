#include "set.cuh"
__global__ static void set_f32(const float * x, float * dst, const int ne, 
                               const size_t offset) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= ne) return;
    dst[offset + i] = x[i];
}

void ggml_cuda_op_set(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // קבלת המקורות
    const ggml_tensor * src0 = dst->src[0]; // הבסיס
    const ggml_tensor * src1 = dst->src[1]; // הערכים החדשים
    
    // בדיקות סוג
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    

    const size_t nb1 = ((const int32_t *) dst->op_params)[0];
    const size_t nb2 = ((const int32_t *) dst->op_params)[1]; 
    const size_t nb3 = ((const int32_t *) dst->op_params)[2];
    const size_t offset = ((const int32_t *) dst->op_params)[3];


    // קבלת pointers לזיכרון
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    
    // העתקת src0 ל-dst
    cudaMemcpyAsync(dst_d, src0_d, ggml_nbytes(dst), 
                    cudaMemcpyDeviceToDevice, ctx.stream());
    
    // קריאה ל-kernel לעדכון הערכים
    const int ne = ggml_nelements(src1);
    const int num_blocks = (ne + CUDA_SET_BLOCK_SIZE - 1) / CUDA_SET_BLOCK_SIZE;
    set_f32<<<num_blocks, CUDA_SET_BLOCK_SIZE, 0, ctx.stream()>>>(
        src1_d, dst_d, ne, offset / sizeof(float));
} 