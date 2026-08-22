#include "reduce_rows.cuh"
#include "sumrows.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    }
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    if (ggml_is_contiguous(src0)) {
        sum_rows_f32_cuda(src0_d, dst_d, ncols, nrows, stream);
        return;
    }

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_dims((nrows / nsm) < 2 ? 512 : (ncols < 1024 ? 32 : 128), 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    const char * src0_d_bytes = (const char *) src0->data;
    ggml_cuda_kernel_launch(reduce_rows_f32_strided</*norm=*/false>, launch_params, src0_d_bytes, dst_d, ncols,
            src0->ne[1], src0->ne[2], src0->nb[1], src0->nb[2], src0->nb[3]);
}
