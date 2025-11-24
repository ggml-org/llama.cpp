#include "common.cuh"
#include "ggml.h"
#include "solve_tri.cuh"

#define MAX_N_FAST 64
#define MAX_K_FAST 32

// ======================
// Fast Kernel (n <= 64, k <= 32) - Warp-based parallel reduction
// ======================
template <int N, int K>
static __global__ void solve_tri_f32_fast(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ X,
    const uint3 ne02,
    const size_t nb02, const size_t nb03,
    const size_t nb12, const size_t nb13,
    const size_t nb2, const size_t nb3,
    int n, int k) {
    const int batch_idx = blockIdx.x;
    const int lane      = threadIdx.x;
    const int col_idx   = threadIdx.y;

    if constexpr (K == 0) {
        if (col_idx >= k) {
            return;
        }
    } else {
        if (col_idx >= K) {
            return;
        }
    }

    const uint2 i02_i03 = fast_div_modulo(batch_idx, ne02);
    const int64_t i02 = i02_i03.y;
    const int64_t i03 = i02_i03.x;

    const float* const A_batch = (const float*)((const char *)A + i02 * nb02 + i03 * nb03);
    const float* const B_batch = (const float*)((const char *)B + i02 * nb12 + i03 * nb13);
    float*             X_batch = (float*)      ((char *)X + i02 * nb2  + i03 * nb3);


    __shared__ float sA[MAX_N_FAST * MAX_N_FAST];
    __shared__ float sX[MAX_N_FAST * MAX_K_FAST];

    const int offset = threadIdx.x + threadIdx.y * blockDim.x;

    if constexpr (K == 0) {
        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < n * n; i += blockDim.x * blockDim.y) {
            sA[i] = A_batch[i];
        }

        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < n * k; i += blockDim.x * blockDim.y) {
            sX[i] = B_batch[i];
        }
    } else {
#pragma unroll
        for (int i = 0; i < N * N; i += K * WARP_SIZE) {
            int i0 = i + offset;
            sA[i0] = A_batch[i0];
        }

#pragma unroll
        for (int i = 0; i < N * K; i += K * WARP_SIZE) {
            int i0 = i + threadIdx.x + threadIdx.y * blockDim.x;
            sX[i0] = B_batch[i0];
        }
    }

    __syncthreads();

    for (int row = 0; row < max(n, N); ++row) {
        float sum = 0.0f;

        if constexpr (K == 0) {
            for (int j = lane; j < row; j += WARP_SIZE) {
                sum += sA[row * n + j] * sX[j * k + col_idx];
            }
        } else {
            for (int j = lane; j < row; j += WARP_SIZE) {
                sum += sA[row * N + j] * sX[j * K + col_idx];
            }
        }

        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            if constexpr (K == 0) {
                const float b_val = sX[row * k + col_idx]; // Value from B
                const float a_diag = sA[row * n + row];
                if (a_diag != 0.0f) {
                    sX[row * k + col_idx] = (b_val - sum) / a_diag;
                } else {
                    sX[row * k + col_idx] = 0.0f; // Avoid division by zero
                }
            } else {
                const float b_val = sX[row * K + col_idx]; // Value from B
                const float a_diag = sA[row * N + row];
                if (a_diag != 0.0f) {
                    sX[row * K + col_idx] = (b_val - sum) / a_diag;
                } else {
                    sX[row * K + col_idx] = 0.0f; // Avoid division by zero
                }
            }
        }
        __syncthreads();
    }

    if constexpr (K == 0) {
        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < n * k; i += blockDim.x * blockDim.y) {
            X_batch[i] = sX[i];
        }
    } else {
#pragma unroll
        for (int i = 0; i < N * K; i += K * WARP_SIZE) {
            const int i0 = i + threadIdx.x + threadIdx.y*blockDim.x;
            X_batch[i0] = sX[i0];
        }
    }
}

// Launcher
static void solve_tri_f32_cuda(
    const float* A, const float* B, float* X,
    int n, int k,
    int64_t ne02, int64_t ne03,
    size_t nb02, size_t nb03,
    size_t nb12, size_t nb13,
    size_t nb2, size_t nb3,
    cudaStream_t stream)
{
    // n <= 64, k <= 32
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    dim3 threads(WARP_SIZE, k);
    dim3 grid(ne02 * ne03);
    if (n == 64) {
        if (k == 32) {
            solve_tri_f32_fast<64, 32><<<grid, threads, 0, stream>>>(
                A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
        } else if (k == 16) {
            solve_tri_f32_fast<64, 16><<<grid, threads, 0, stream>>>(
                A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
        } else if (k == 14) {
            solve_tri_f32_fast<64, 14><<<grid, threads, 0, stream>>>(
                A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
        } else if (k == 12) {
            solve_tri_f32_fast<64, 12><<<grid, threads, 0, stream>>>(
                A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
        } else if (k == 10) {
            solve_tri_f32_fast<64, 10><<<grid, threads, 0, stream>>>(
                A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
        } else if (k == 8) {
            solve_tri_f32_fast<64, 8><<<grid, threads, 0, stream>>>(
                A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
        } else if (k == 6) {
            solve_tri_f32_fast<64, 6><<<grid, threads, 0, stream>>>(
                A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
        } else if (k == 4) {
            solve_tri_f32_fast<64, 4><<<grid, threads, 0, stream>>>(
                A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
        } else if (k == 2) {
            solve_tri_f32_fast<64, 2><<<grid, threads, 0, stream>>>(
                A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
        } else if (k == 1) {
            solve_tri_f32_fast<64, 1><<<grid, threads, 0, stream>>>(
                A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3,0, 0);
        } else {
            solve_tri_f32_fast<0, 0><<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, k);
        }
    } else { // run general case
        solve_tri_f32_fast<0, 0><<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, k);
    }
}

void ggml_cuda_op_solve_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor* src0 = dst->src[0]; // A
    const ggml_tensor* src1 = dst->src[1]; // B

    ggml_is_contiguous(src0);
    ggml_is_contiguous(src1);

    const int64_t n = src0->ne[0];
    const int64_t k = src1->ne[0];

    GGML_ASSERT(n <= 64);
    GGML_ASSERT(k <= 32);

    solve_tri_f32_cuda(
        (const float*)src0->data, (const float*)src1->data, (float*)dst->data,
        n, k,
        src0->ne[2], src0->ne[3],
        src0->nb[2], src0->nb[3],
        src1->nb[2], src1->nb[3],
        dst->nb[2], dst->nb[3],
        ctx.stream()
    );
}
