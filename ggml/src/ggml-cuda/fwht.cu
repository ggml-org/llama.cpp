#include "common.cuh"
#include "fwht.cuh"

template <int N>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const float * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * warp_size + lane] * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
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
        dst[i * warp_size + lane] = reg[i];
    }
}

template <int N>
__global__ void regular_hadamard_cuda(const float * src, float * dst, const int64_t n_groups, const float scale) {
    const int64_t group = blockIdx.x;
    const int tid       = threadIdx.x;
    if (group >= n_groups) {
        return;
    }

    __shared__ float values[N];
    src += group * N;
    dst += group * N;

    values[tid] = src[tid] * scale;
    __syncthreads();

#pragma unroll
    for (int stride = 1; stride < N; stride *= 4) {
        if (tid < N / 4) {
            const int base = (tid / stride) * 4 * stride + tid % stride;
            const int i0   = base;
            const int i1   = i0 + stride;
            const int i2   = i1 + stride;
            const int i3   = i2 + stride;
            const float a  = values[i0];
            const float b  = values[i1];
            const float c  = values[i2];
            const float d  = values[i3];
            values[i0]     =  a + b + c - d;
            values[i1]     =  a + b - c + d;
            values[i2]     =  a - b + c + d;
            values[i3]     = -a + b + c + d;
        }
        __syncthreads();
    }

    dst[tid] = values[tid];
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const float * src_d = (const float *) src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 128:
            ggml_cuda_kernel_launch(fwht_cuda<128>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 256:
            ggml_cuda_kernel_launch(fwht_cuda<256>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512>, launch_params, src_d, dst_d, rows, scale);
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_op_regular_hadamard(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst, int group_size) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(src) || !ggml_is_contiguous(dst) ||
        group_size <= 0 || ggml_nelements(src) % group_size != 0) {
        return false;
    }

    const int64_t n_groups = ggml_nelements(src) / group_size;
    const float scale      = 1.0f / sqrtf((float) group_size);
    const float * src_d    = (const float *) src->data;
    float * dst_d          = (float *) dst->data;
    cudaStream_t stream    = ctx.stream();

    switch (group_size) {
        case 4:
            regular_hadamard_cuda<4><<<n_groups, 4, 0, stream>>>(src_d, dst_d, n_groups, scale);
            return true;
        case 16:
            regular_hadamard_cuda<16><<<n_groups, 16, 0, stream>>>(src_d, dst_d, n_groups, scale);
            return true;
        case 64:
            regular_hadamard_cuda<64><<<n_groups, 64, 0, stream>>>(src_d, dst_d, n_groups, scale);
            return true;
        case 256:
            regular_hadamard_cuda<256><<<n_groups, 256, 0, stream>>>(src_d, dst_d, n_groups, scale);
            return true;
        default:
            return false;
    }
}
