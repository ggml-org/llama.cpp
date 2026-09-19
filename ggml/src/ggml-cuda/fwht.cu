#include "common.cuh"
#include "fwht.cuh"

#define P 1.0f
#define N -1.0f

// Constant Hadamard matrices via Paley I construction.
static __device__ __constant__ float fwht_h12[12][12] = {
    { P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, P, P, P, N, N, N, P, N },
    { P, N, N, P, N, P, P, P, N, N, N, P },
    { P, P, N, N, P, N, P, P, P, N, N, N },
    { P, N, P, N, N, P, N, P, P, P, N, N },
    { P, N, N, P, N, N, P, N, P, P, P, N },
    { P, N, N, N, P, N, N, P, N, P, P, P },
    { P, P, N, N, N, P, N, N, P, N, P, P },
    { P, P, P, N, N, N, P, N, N, P, N, P },
    { P, P, P, P, N, N, N, P, N, N, P, N },
    { P, N, P, P, P, N, N, N, P, N, N, P },
    { P, P, N, P, P, P, N, N, N, P, N, N },
};

static __device__ __constant__ float fwht_h20[20][20] = {
    { P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N },
    { P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P },
    { P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P },
    { P, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N },
    { P, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N },
    { P, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N },
    { P, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N },
    { P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P },
    { P, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N },
    { P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P },
    { P, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N },
    { P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P },
    { P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P },
    { P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P },
    { P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P },
    { P, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N },
    { P, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N },
    { P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P },
    { P, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N },
};

#undef P
#undef N

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

template <int N, int M>
__launch_bounds__(4 * ggml_cuda_get_physical_warp_size(), 1)
__global__ void kronecker_cuda(const float * src, float * dst, const int64_t n_rows, const float scale) {
    static_assert(M == 12 || M == 20, "block size has to be 12 or 20");

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int blocks_per_group = N / M;
    constexpr int el_w = blocks_per_group / warp_size;

    static_assert(el_w >= 1, "at least one block per lane is required");
    static_assert(blocks_per_group % warp_size == 0, "blocks_per_group must be a multiple of warp_size");

    float reg[el_w * M];

    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int b_idx = i * warp_size + lane;

#pragma unroll
        for (int j = 0; j < M; ++j) {
            reg[i * M + j] = src[b_idx * M + j] * scale;
        }
    }

#pragma unroll
    for (int b = 0; b < el_w; ++b) {
        float z[M] = { 0.0f };

#pragma unroll
        for (int i = 0; i < M; ++i) {
#pragma unroll
            for (int j = 0; j < M; ++j) {
                const float h = M == 12 ? fwht_h12[j][i] : fwht_h20[j][i];
                z[i] += reg[b * M + j] * h;
            }
        }

#pragma unroll
        for (int i = 0; i < M; ++i) {
            reg[b * M + i] = z[i];
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
#pragma unroll
            for (int k = 0; k < M; ++k) {
                const float val  = reg[j * M + k];
                const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

                reg[j * M + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    for (int h = warp_size; h < blocks_per_group; h *= 2) {
        const int step = h / warp_size;

#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int s = 0; s < step; ++s) {
#pragma unroll
                for (int k = 0; k < M; ++k) {
                    const float x = reg[(j + s) * M + k];
                    const float y = reg[(j + s + step) * M + k];

                    reg[(j + s) * M + k]        = x + y;
                    reg[(j + s + step) * M + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int b_idx = i * warp_size + lane;

#pragma unroll
        for (int k = 0; k < M; ++k) {
            dst[b_idx * M + k] = reg[i * M + k];
        }
    }
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
        case 384:
            ggml_cuda_kernel_launch(kronecker_cuda<384, 12>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 640:
            ggml_cuda_kernel_launch(kronecker_cuda<640, 20>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 768:
            ggml_cuda_kernel_launch(kronecker_cuda<768, 12>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 1280:
            ggml_cuda_kernel_launch(kronecker_cuda<1280, 20>, launch_params, src_d, dst_d, rows, scale);
            return true;
        default:
            return false;
    }
}
