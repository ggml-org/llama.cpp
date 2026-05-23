#include "common.cuh"
#include "fwht.cuh"

template <size_t N> __global__ void fwht_cuda(const float * src, float * dst, const int64_t n_rows, const float scale) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr size_t el_w = N / WARP_SIZE;
    float                   reg[el_w];
    const int               lane = threadIdx.x;

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * WARP_SIZE + lane] * scale;
    }

    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    for (int h = WARP_SIZE; h < N; h *= 2) {
        const int step = h / WARP_SIZE;
        for (int j = 0; j < el_w; j += 2 * step) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

void ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const float * src_d = (const float *) src->data;
    float *       dst_d = (float *) dst->data;

    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(WARP_SIZE, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            {
                ggml_cuda_kernel_launch(fwht_cuda<64>, launch_params, src_d, dst_d, rows, scale);
                break;
            }
        case 128:
            {
                ggml_cuda_kernel_launch(fwht_cuda<128>, launch_params, src_d, dst_d, rows, scale);
                break;
            }
        case 256:
            {
                ggml_cuda_kernel_launch(fwht_cuda<256>, launch_params, src_d, dst_d, rows, scale);
                break;
            }
        case 512:
            {
                ggml_cuda_kernel_launch(fwht_cuda<512>, launch_params, src_d, dst_d, rows, scale);
                break;
            }
        default:
            GGML_ABORT("fatal error");
    }
}
