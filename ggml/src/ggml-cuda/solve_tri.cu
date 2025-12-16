#include "common.cuh"
#include "ggml.h"
#include "solve_tri.cuh"
#include "ggml-cuda.h"
#include <cublas_v2.h>

#define MAX_N_FAST 64
#define MAX_K_FAST 64

// Kernel to set up pointer arrays for batched cuBLAS TRSM
// This avoids host-device copy during CUDA graph capture
static __global__ void setup_trsm_batch_pointers(
    const float * A,
    float * X,
    const float ** A_ptrs,
    float ** X_ptrs,
    const int64_t ne02,
    const int64_t total_batches,
    const size_t nb02,  // stride for A dim 2 (in floats)
    const size_t nb03,  // stride for A dim 3 (in floats)
    const size_t nb2,   // stride for X dim 2 (in floats)
    const size_t nb3    // stride for X dim 3 (in floats)
) {
    const int64_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= total_batches) return;

    // Decompose batch_idx into i02, i03
    const int64_t i02 = batch_idx % ne02;
    const int64_t i03 = batch_idx / ne02;

    A_ptrs[batch_idx] = A + i02 * nb02 + i03 * nb03;
    X_ptrs[batch_idx] = X + i02 * nb2  + i03 * nb3;
}

// Latency-optimized kernel for n=64, k=64 (single-token generation)
static __global__ void solve_tri_f32_64x64_latency(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ X,
    const uint3  ne02,
    const size_t nb02,
    const size_t nb03,
    const size_t nb12,
    const size_t nb13,
    const size_t nb2,
    const size_t nb3)
{
    const int batch_idx = blockIdx.x;
    const int lane      = threadIdx.x;
    const int warp_id   = threadIdx.y;

    const uint2   i02_i03 = fast_div_modulo(batch_idx, ne02);
    const int64_t i02     = i02_i03.y;
    const int64_t i03     = i02_i03.x;

    const float * const A_batch = (const float *) (A + i02 * nb02 + i03 * nb03);
    const float * const B_batch = (const float *) (B + i02 * nb12 + i03 * nb13);
    float *             X_batch = (float *) (X + i02 * nb2 + i03 * nb3);

    // Shared memory: A is 64x64, X is 64x65 (padded for bank conflicts)
    __shared__ float sA[64 * 64];
    __shared__ float sX[64 * 65];
    __shared__ float sDiagInv[64];  // Precomputed 1/diagonal

    const int tid = lane + warp_id * WARP_SIZE;

    // Cooperative load of A matrix (4096 elements / 512 threads = 8 per thread)
    #pragma unroll 8
    for (int i = tid; i < 64 * 64; i += 512) {
        sA[i] = A_batch[i];
    }

    // Cooperative load of B matrix into sX with padding
    #pragma unroll 8
    for (int i = tid; i < 64 * 64; i += 512) {
        const int row = i / 64;
        const int col = i % 64;
        sX[row * 65 + col] = B_batch[i];
    }

    __syncthreads();

    // Precompute diagonal inverses (first 2 warps handle this)
    if (warp_id == 0) {
        if (lane < 32) {
            sDiagInv[lane] = 1.0f / sA[lane * 64 + lane];
        }
    }
    if (warp_id == 1) {
        if (lane < 32) {
            sDiagInv[32 + lane] = 1.0f / sA[(32 + lane) * 64 + (32 + lane)];
        }
    }

    __syncthreads();

    // Each warp handles 4 columns: cols = warp_id*4 to warp_id*4+3
    const int col_base = warp_id * 4;

    #pragma unroll 1
    for (int row = 0; row < 64; ++row) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        if (row > 0) {
            for (int j = lane; j < row; j += WARP_SIZE) {
                const float a_val = sA[row * 64 + j];
                sum0 += a_val * sX[j * 65 + col_base + 0];
                sum1 += a_val * sX[j * 65 + col_base + 1];
                sum2 += a_val * sX[j * 65 + col_base + 2];
                sum3 += a_val * sX[j * 65 + col_base + 3];
            }
        }

        sum0 = warp_reduce_sum(sum0);
        sum1 = warp_reduce_sum(sum1);
        sum2 = warp_reduce_sum(sum2);
        sum3 = warp_reduce_sum(sum3);

        if (lane == 0) {
            const float inv_diag = sDiagInv[row];
            sX[row * 65 + col_base + 0] = (sX[row * 65 + col_base + 0] - sum0) * inv_diag;
            sX[row * 65 + col_base + 1] = (sX[row * 65 + col_base + 1] - sum1) * inv_diag;
            sX[row * 65 + col_base + 2] = (sX[row * 65 + col_base + 2] - sum2) * inv_diag;
            sX[row * 65 + col_base + 3] = (sX[row * 65 + col_base + 3] - sum3) * inv_diag;
        }

        __syncthreads();
    }

    // Cooperative write results back
    #pragma unroll 8
    for (int i = tid; i < 64 * 64; i += 512) {
        const int row = i / 64;
        const int col = i % 64;
        X_batch[i] = sX[row * 65 + col];
    }
}

static __global__ void solve_tri_f32_64x64_opt(const float * __restrict__ A,
                                               const float * __restrict__ B,
                                               float * __restrict__ X,
                                               const uint3  ne02,
                                               const size_t nb02,
                                               const size_t nb03,
                                               const size_t nb12,
                                               const size_t nb13,
                                               const size_t nb2,
                                               const size_t nb3) {
    const int batch_idx = blockIdx.x;
    const int lane      = threadIdx.x;
    const int warp_id   = threadIdx.y;

    const uint2   i02_i03 = fast_div_modulo(batch_idx, ne02);
    const int64_t i02     = i02_i03.y;
    const int64_t i03     = i02_i03.x;

    const float * const A_batch = (const float *) (A + i02 * nb02 + i03 * nb03);
    const float * const B_batch = (const float *) (B + i02 * nb12 + i03 * nb13);
    float *             X_batch = (float *) (X + i02 * nb2 + i03 * nb3);

    // Shared memory: A is 64x64, sXt is 64x65 (padded)
    __shared__ float sA[64 * 64];
    __shared__ float sXt[64 * 65];

    const int tid = lane + warp_id * WARP_SIZE;

    // Cooperative load of A matrix (4096 elements / 1024 threads = 4 per thread)
    #pragma unroll 4
    for (int i = tid; i < 64 * 64; i += 1024) {
        sA[i] = A_batch[i];
    }

    // Cooperative load of B matrix transposed into sXt
    // sXt[col * 65 + row] = B[row * 64 + col]
    #pragma unroll 4
    for (int i = tid; i < 64 * 64; i += 1024) {
        const int row = i / 64;
        const int col = i % 64;
        sXt[col * 65 + row] = B_batch[row * 64 + col];
    }

    __syncthreads();

    // Each warp handles 2 columns: col0 = warp_id*2, col1 = warp_id*2 + 1
    const int col0 = warp_id * 2;
    const int col1 = warp_id * 2 + 1;

    // Forward substitution with all columns processed in parallel
    // Each row depends on previous rows, but different columns are independent
    #pragma unroll 1
    for (int row = 0; row < 64; ++row) {
        // Each lane computes partial sum for indices it handles
        float sum0 = 0.0f;
        float sum1 = 0.0f;

        // Sum over j < row
        // For row <= 32: each lane handles at most 1 element
        // For row > 32: each lane handles at most 2 elements
        if (lane < row) {
            const float a_val = sA[row * 64 + lane];
            sum0 = a_val * sXt[col0 * 65 + lane];
            sum1 = a_val * sXt[col1 * 65 + lane];
        }
        if (row > WARP_SIZE) {
            const int j2 = lane + WARP_SIZE;
            if (j2 < row) {
                const float a_val2 = sA[row * 64 + j2];
                sum0 += a_val2 * sXt[col0 * 65 + j2];
                sum1 += a_val2 * sXt[col1 * 65 + j2];
            }
        }

        // Warp-level reduction
        sum0 = warp_reduce_sum(sum0);
        sum1 = warp_reduce_sum(sum1);

        // Lane 0 computes and stores the result
        if (lane == 0) {
            const float a_diag = sA[row * 64 + row];
            const float inv_diag = 1.0f / a_diag;
            sXt[col0 * 65 + row] = (sXt[col0 * 65 + row] - sum0) * inv_diag;
            sXt[col1 * 65 + row] = (sXt[col1 * 65 + row] - sum1) * inv_diag;
        }

        // Sync within warp to ensure writes are visible before next row reads
        __syncwarp();
    }

    __syncthreads();

    // Cooperative write of results back (transpose sXt to X)
    #pragma unroll 4
    for (int i = tid; i < 64 * 64; i += 1024) {
        const int row = i / 64;
        const int col = i % 64;
        X_batch[row * 64 + col] = sXt[col * 65 + row];
    }
}

static __global__ void solve_tri_f32_128x128_opt(const float * __restrict__ A,
                                                  const float * __restrict__ B,
                                                  float * __restrict__ X,
                                                  const uint3  ne02,
                                                  const size_t nb02,
                                                  const size_t nb03,
                                                  const size_t nb12,
                                                  const size_t nb13,
                                                  const size_t nb2,
                                                  const size_t nb3,
                                                  const int n,
                                                  const int k) {
    const int batch_idx = blockIdx.x;
    const int lane      = threadIdx.x;
    const int warp_id   = threadIdx.y;

    const uint2   i02_i03 = fast_div_modulo(batch_idx, ne02);
    const int64_t i02     = i02_i03.y;
    const int64_t i03     = i02_i03.x;

    const float * const A_batch = (const float *) (A + i02 * nb02 + i03 * nb03);
    const float * const B_batch = (const float *) (B + i02 * nb12 + i03 * nb13);
    float *             X_batch = (float *) (X + i02 * nb2 + i03 * nb3);

    // Shared memory with padding to avoid bank conflicts
    // Layout: sA[128][128] + sXt[128][129]
    extern __shared__ char smem_raw[];
    float * sA = (float *)smem_raw;              // 128×128 (zero-initialized for unused parts)
    float * sXt = sA + 128 * 128;                // 128×129 (padded)

    const int tid = lane + warp_id * WARP_SIZE;

    // Zero-initialize shared memory first (important for variable n, k)
    #pragma unroll 16
    for (int i = tid; i < 128 * 128; i += 1024) {
        sA[i] = 0.0f;
    }
    #pragma unroll 16
    for (int i = tid; i < 128 * 129; i += 1024) {
        sXt[i] = 0.0f;
    }
    __syncthreads();

    // Cooperative load of A matrix (n×n elements)
    for (int i = tid; i < n * n; i += 1024) {
        const int row = i / n;
        const int col = i % n;
        sA[row * 128 + col] = A_batch[row * n + col];
    }

    // Cooperative load of B matrix transposed into sXt
    // sXt[col * 129 + row] = B[row * k + col]
    for (int i = tid; i < n * k; i += 1024) {
        const int row = i / k;
        const int col = i % k;
        sXt[col * 129 + row] = B_batch[row * k + col];
    }

    __syncthreads();

    // Each warp handles columns: col_base to col_base+3
    // But only process if col < k
    const int col_base = warp_id * 4;

    // Forward substitution with all columns processed in parallel
    for (int row = 0; row < n; ++row) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        // Sum over j < row - each lane handles multiple elements
        for (int j = lane; j < row; j += WARP_SIZE) {
            const float a_val = sA[row * 128 + j];
            if (col_base + 0 < k) sum0 += a_val * sXt[(col_base + 0) * 129 + j];
            if (col_base + 1 < k) sum1 += a_val * sXt[(col_base + 1) * 129 + j];
            if (col_base + 2 < k) sum2 += a_val * sXt[(col_base + 2) * 129 + j];
            if (col_base + 3 < k) sum3 += a_val * sXt[(col_base + 3) * 129 + j];
        }

        // Warp-level reduction
        sum0 = warp_reduce_sum(sum0);
        sum1 = warp_reduce_sum(sum1);
        sum2 = warp_reduce_sum(sum2);
        sum3 = warp_reduce_sum(sum3);

        // Lane 0 computes and stores the result
        if (lane == 0) {
            const float inv_diag = 1.0f / sA[row * 128 + row];
            if (col_base + 0 < k) {
                sXt[(col_base + 0) * 129 + row] = (sXt[(col_base + 0) * 129 + row] - sum0) * inv_diag;
            }
            if (col_base + 1 < k) {
                sXt[(col_base + 1) * 129 + row] = (sXt[(col_base + 1) * 129 + row] - sum1) * inv_diag;
            }
            if (col_base + 2 < k) {
                sXt[(col_base + 2) * 129 + row] = (sXt[(col_base + 2) * 129 + row] - sum2) * inv_diag;
            }
            if (col_base + 3 < k) {
                sXt[(col_base + 3) * 129 + row] = (sXt[(col_base + 3) * 129 + row] - sum3) * inv_diag;
            }
        }

        __syncwarp();
    }

    __syncthreads();

    // Cooperative write of results back (transpose sXt to X)
    for (int i = tid; i < n * k; i += 1024) {
        const int row = i / k;
        const int col = i % k;
        X_batch[row * k + col] = sXt[col * 129 + row];
    }
}

static __global__ void solve_tri_f32_256x256_tiled(const float * __restrict__ A,
                                                    const float * __restrict__ B,
                                                    float * __restrict__ X,
                                                    const uint3  ne02,
                                                    const size_t nb02,
                                                    const size_t nb03,
                                                    const size_t nb12,
                                                    const size_t nb13,
                                                    const size_t nb2,
                                                    const size_t nb3,
                                                    const int n,
                                                    const int k) {
    const int batch_idx = blockIdx.x;
    const int lane      = threadIdx.x;
    const int warp_id   = threadIdx.y;

    const uint2   i02_i03 = fast_div_modulo(batch_idx, ne02);
    const int64_t i02     = i02_i03.y;
    const int64_t i03     = i02_i03.x;

    const float * const A_batch = (const float *) (A + i02 * nb02 + i03 * nb03);
    const float * const B_batch = (const float *) (B + i02 * nb12 + i03 * nb13);
    float *             X_batch = (float *) (X + i02 * nb2 + i03 * nb3);

    // Tiled approach using 64×64 tiles to fit in shared memory
    constexpr int TILE_SIZE = 64;

    extern __shared__ char smem_raw[];
    float * sA_tile = (float *)smem_raw;                    // 64×64 = 16KB
    float * sXt_tile = sA_tile + TILE_SIZE * TILE_SIZE;     // 64×65 = 16.25KB (padded)
    float * sA_off = sXt_tile + TILE_SIZE * (TILE_SIZE+1);  // 64×64 = 16KB (for off-diagonal blocks)

    const int tid = lane + warp_id * WARP_SIZE;

    // Initialize X = B (we'll solve in-place conceptually, using global memory)
    for (int i = tid; i < n * k; i += 1024) {
        X_batch[i] = B_batch[i];
    }
    __syncthreads();

    // Process tile-by-tile along the diagonal
    for (int tile_row = 0; tile_row < n; tile_row += TILE_SIZE) {
        const int tile_n = min(TILE_SIZE, n - tile_row);  // Actual rows in this tile

        // Zero-init and load diagonal tile of A
        for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += 1024) {
            sA_tile[i] = 0.0f;
        }
        __syncthreads();

        for (int i = tid; i < tile_n * tile_n; i += 1024) {
            int local_row = i / tile_n;
            int local_col = i % tile_n;
            sA_tile[local_row * TILE_SIZE + local_col] = A_batch[(tile_row + local_row) * n + tile_row + local_col];
        }
        __syncthreads();

        // For each column tile of X
        for (int tile_col = 0; tile_col < k; tile_col += TILE_SIZE) {
            const int tile_k = min(TILE_SIZE, k - tile_col);  // Actual columns in this tile

            // Zero-init and load X tile transposed
            for (int i = tid; i < TILE_SIZE * (TILE_SIZE+1); i += 1024) {
                sXt_tile[i] = 0.0f;
            }
            __syncthreads();

            for (int i = tid; i < tile_n * tile_k; i += 1024) {
                int local_row = i / tile_k;
                int local_col = i % tile_k;
                sXt_tile[local_col * (TILE_SIZE+1) + local_row] =
                    X_batch[(tile_row + local_row) * k + tile_col + local_col];
            }
            __syncthreads();

            // Apply updates from previous tile rows
            for (int prev_tile = 0; prev_tile < tile_row; prev_tile += TILE_SIZE) {
                const int prev_n = min(TILE_SIZE, n - prev_tile);

                // Zero-init and load off-diagonal block
                for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += 1024) {
                    sA_off[i] = 0.0f;
                }
                __syncthreads();

                for (int i = tid; i < tile_n * prev_n; i += 1024) {
                    int local_row = i / prev_n;
                    int local_col = i % prev_n;
                    sA_off[local_row * TILE_SIZE + local_col] = A_batch[(tile_row + local_row) * n + prev_tile + local_col];
                }
                __syncthreads();

                // Update: X_tile -= A_off @ X_prev
                int col0 = warp_id * 2;
                int col1 = warp_id * 2 + 1;

                for (int row = 0; row < tile_n; row++) {
                    float sum0 = 0.0f, sum1 = 0.0f;

                    for (int j = lane; j < prev_n; j += WARP_SIZE) {
                        float a_val = sA_off[row * TILE_SIZE + j];
                        if (col0 < tile_k) {
                            float x_prev0 = X_batch[(prev_tile + j) * k + tile_col + col0];
                            sum0 += a_val * x_prev0;
                        }
                        if (col1 < tile_k) {
                            float x_prev1 = X_batch[(prev_tile + j) * k + tile_col + col1];
                            sum1 += a_val * x_prev1;
                        }
                    }

                    sum0 = warp_reduce_sum(sum0);
                    sum1 = warp_reduce_sum(sum1);

                    if (lane == 0) {
                        if (col0 < tile_k) {
                            sXt_tile[col0 * (TILE_SIZE+1) + row] -= sum0;
                        }
                        if (col1 < tile_k) {
                            sXt_tile[col1 * (TILE_SIZE+1) + row] -= sum1;
                        }
                    }
                    __syncwarp();
                }
                __syncthreads();
            }

            // Solve the diagonal tile
            int col0 = warp_id * 2;
            int col1 = warp_id * 2 + 1;

            for (int row = 0; row < tile_n; ++row) {
                float sum0 = 0.0f, sum1 = 0.0f;

                if (lane < row) {
                    float a_val = sA_tile[row * TILE_SIZE + lane];
                    if (col0 < tile_k) sum0 = a_val * sXt_tile[col0 * (TILE_SIZE+1) + lane];
                    if (col1 < tile_k) sum1 = a_val * sXt_tile[col1 * (TILE_SIZE+1) + lane];
                }
                if (row > WARP_SIZE) {
                    int j2 = lane + WARP_SIZE;
                    if (j2 < row) {
                        float a_val2 = sA_tile[row * TILE_SIZE + j2];
                        if (col0 < tile_k) sum0 += a_val2 * sXt_tile[col0 * (TILE_SIZE+1) + j2];
                        if (col1 < tile_k) sum1 += a_val2 * sXt_tile[col1 * (TILE_SIZE+1) + j2];
                    }
                }

                sum0 = warp_reduce_sum(sum0);
                sum1 = warp_reduce_sum(sum1);

                if (lane == 0) {
                    float inv_diag = 1.0f / sA_tile[row * TILE_SIZE + row];
                    if (col0 < tile_k) {
                        sXt_tile[col0 * (TILE_SIZE+1) + row] =
                            (sXt_tile[col0 * (TILE_SIZE+1) + row] - sum0) * inv_diag;
                    }
                    if (col1 < tile_k) {
                        sXt_tile[col1 * (TILE_SIZE+1) + row] =
                            (sXt_tile[col1 * (TILE_SIZE+1) + row] - sum1) * inv_diag;
                    }
                }
                __syncwarp();
            }
            __syncthreads();

            // Write solved tile back to global memory
            for (int i = tid; i < tile_n * tile_k; i += 1024) {
                int local_row = i / tile_k;
                int local_col = i % tile_k;
                X_batch[(tile_row + local_row) * k + tile_col + local_col] =
                    sXt_tile[local_col * (TILE_SIZE+1) + local_row];
            }
            __syncthreads();
        }
    }
}

// When ncols_template == 0 the bounds for the loops in this function are not
// known and can't be unrolled. As we want to keep pragma unroll for all other
// cases we supress the clang transformation warning here.
#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wpass-failed"
#endif  // __clang__
// Template parameters: n_template/k_template are the matrix dimensions when known at compile time (0 = runtime)
// threads_y_template is the number of threads in y dimension (max 32 to stay within 1024 thread limit)
template <int n_template, int k_template, int threads_y_template>
static __global__ void solve_tri_f32_fast(const float * __restrict__ A,
                                          const float * __restrict__ B,
                                          float * __restrict__ X,
                                          const uint3  ne02,
                                          const size_t nb02,
                                          const size_t nb03,
                                          const size_t nb12,
                                          const size_t nb13,
                                          const size_t nb2,
                                          const size_t nb3,
                                          const int    n_arg,
                                          const int    k_arg) {
    const int n = n_template == 0 ? n_arg : n_template;
    const int k = k_template == 0 ? k_arg : k_template;
    const int threads_y = threads_y_template == 0 ? blockDim.y : threads_y_template;

    const int batch_idx = blockIdx.x;
    const int lane      = threadIdx.x;

    const uint2   i02_i03 = fast_div_modulo(batch_idx, ne02);
    const int64_t i02     = i02_i03.y;
    const int64_t i03     = i02_i03.x;

    const float * const A_batch = (const float *) (A + i02 * nb02 + i03 * nb03);
    const float * const B_batch = (const float *) (B + i02 * nb12 + i03 * nb13);
    float *             X_batch = (float *) (X + i02 * nb2 + i03 * nb3);

    __shared__ float sA[MAX_N_FAST * MAX_N_FAST];
    __shared__ float sXt[MAX_N_FAST * (MAX_K_FAST + 1)];

    const int offset = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_threads = blockDim.x * blockDim.y;

    // Load A matrix into shared memory
#pragma unroll
    for (int i = 0; i < n * n; i += block_threads) {
        int i0 = i + offset;
        if (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    const int rows_per_warp = (n + WARP_SIZE - 1) / WARP_SIZE;
    const int cols_per_thread = (k + threads_y - 1) / threads_y;

    // Load B matrix into shared memory (transposed as sXt)
    // Each thread handles multiple columns when k > threads_y
    for (int c = 0; c < cols_per_thread; c++) {
        const int col_idx = threadIdx.y + c * threads_y;
        if (col_idx < k) {
#pragma unroll
            for (int i = 0; i < rows_per_warp; i++) {
                const int i0 = lane + i * WARP_SIZE;
                if (i0 < n) {
                    sXt[col_idx * n + i0] = B_batch[i0 * k + col_idx];
                }
            }
        }
    }

    __syncthreads();

    // Solve for each column this thread handles
    for (int c = 0; c < cols_per_thread; c++) {
        const int col_idx = threadIdx.y + c * threads_y;
        if (col_idx >= k) {
            continue;
        }

#pragma unroll
        for (int row = 0; row < n; ++row) {
            float sum = 0.0f;

            {
                int j = lane;
                if (j < row) {
                    sum += sA[row * n + j] * sXt[col_idx * n + j];
                }
            }
            if (row >= WARP_SIZE) {
                int j = WARP_SIZE + lane;
                if (j < row) {
                    sum += sA[row * n + j] * sXt[col_idx * n + j];
                }
            }

            sum = warp_reduce_sum(sum);

            if (lane == 0) {
                const float b_val      = sXt[col_idx * n + row];
                const float a_diag     = sA[row * n + row];
                // no safeguards for division by zero because that indicates corrupt
                // data anyway
                sXt[col_idx * n + row] = (b_val - sum) / a_diag;
            }
        }

        // Sync between columns to ensure writes are visible
        if (c + 1 < cols_per_thread) {
            __syncwarp();
        }
    }

    __syncthreads();

    // Write results back
    for (int c = 0; c < cols_per_thread; c++) {
        const int col_idx = threadIdx.y + c * threads_y;
        if (col_idx < k) {
#pragma unroll
            for (int i = 0; i < rows_per_warp; i++) {
                const int i0 = lane + i * WARP_SIZE;
                if (i0 < n) {
                    X_batch[i0 * k + col_idx] = sXt[col_idx * n + i0];
                }
            }
        }
    }
}
#ifdef __clang__
#    pragma clang diagnostic pop
#endif  // __clang__

// cuBLAS batched TRSM fallback for larger matrices or as robust path
// Solves A * X = B where A is lower triangular
// This function modifies X in-place (X should be initialized with B)
static void solve_tri_f32_cublas(
    ggml_backend_cuda_context & ctx,
    const float * A,
    float * X,  // Input: B, Output: solution X (in-place)
    int n,
    int k,
    int64_t ne02,
    int64_t ne03,
    size_t nb02,
    size_t nb03,
    size_t nb2,
    size_t nb3,
    cudaStream_t stream
) {
    const int64_t total_batches = ne02 * ne03;

    // Allocate pointer arrays on device
    ggml_cuda_pool_alloc<const float *> A_ptrs(ctx.pool(), total_batches);
    ggml_cuda_pool_alloc<float *> X_ptrs(ctx.pool(), total_batches);

    // Set up pointer arrays on device (CUDA graph compatible)
    {
        const int block_size = 256;
        const int grid_size = (total_batches + block_size - 1) / block_size;
        setup_trsm_batch_pointers<<<grid_size, block_size, 0, stream>>>(
            A, X,
            A_ptrs.get(), X_ptrs.get(),
            ne02, total_batches,
            nb02, nb03, nb2, nb3
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Get cuBLAS handle and set stream
    cublasHandle_t handle = ctx.cublas_handle();
    cublasSetStream(handle, stream);

    // Save current math mode and set to default for accuracy
    // (TF32 can cause numerical issues with triangular solves)
    cublasMath_t prev_math_mode;
    cublasGetMathMode(handle, &prev_math_mode);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    const float alpha = 1.0f;

    cublasStatus_t status = cublasStrsmBatched(
        handle,
        CUBLAS_SIDE_RIGHT,       // A is on the right: X * A = B
        CUBLAS_FILL_MODE_UPPER,  // A^T is upper (since A is lower in row-major)
        CUBLAS_OP_N,             // No additional transpose
        CUBLAS_DIAG_NON_UNIT,    // Diagonal is not assumed to be 1
        k,                       // m: rows of X^T (columns of X)
        n,                       // n: columns of X^T (rows of X) = size of A
        &alpha,
        (const float **)A_ptrs.get(), n,  // lda = n (leading dimension)
        (float **)X_ptrs.get(), k,        // ldb = k (leading dimension of X^T)
        total_batches
    );

    // Restore previous math mode
    cublasSetMathMode(handle, prev_math_mode);

    if (status != CUBLAS_STATUS_SUCCESS) {
        GGML_LOG_ERROR("cuBLAS batched TRSM failed: %d\n", (int)status);
    }
}

static void solve_tri_f32_cuda(const float * A,
                               const float * B,
                               float *       X,
                               int           n,
                               int           k,
                               int64_t       ne02,
                               int64_t       ne03,
                               size_t        nb02,
                               size_t        nb03,
                               size_t        nb12,
                               size_t        nb13,
                               size_t        nb2,
                               size_t        nb3,
                               cudaStream_t  stream) {
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    dim3 grid(ne02 * ne03);

    // Handle large matrices first (256×256 and 65-128 range)

    // Route sizes 65-256 to the tiled kernel
    if (n > 64 || k > 64) {
        // Use the tiled kernel which works for any size up to 256
        // and only requires ~48KB shared memory (within standard limits)
        dim3 threads_256(WARP_SIZE, 32);  // 1024 threads
        // Shared memory: 64×64 + 64×65 + 64×64 = 16KB + 16.25KB + 16KB = ~48KB
        const size_t smem_size = (64 * 64 + 64 * 65 + 64 * 64) * sizeof(float);

        // Configure extended shared memory for this kernel
        static bool smem_configured_tiled = false;
        if (!smem_configured_tiled) {
            cudaFuncSetAttribute(solve_tri_f32_256x256_tiled,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            smem_configured_tiled = true;
        }

        solve_tri_f32_256x256_tiled<<<grid, threads_256, smem_size, stream>>>(
            A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, k);
        return;
    }

    // Limit threads_y to 32 to ensure we don't exceed 1024 threads per block (32 * 32 = 1024)
    const int threads_y = k <= 32 ? k : 32;
    dim3      threads(WARP_SIZE, threads_y);

    if (n == 64) {
        switch (k) {
            case 64:
                {
                    // Use optimized kernel for n=64, k=64 case (common in Qwen3 Next DeltaNet)
                    // Block config: 32x32 = 1024 threads (32 warps)
                    dim3 threads_64x64(WARP_SIZE, 32);
                    solve_tri_f32_64x64_opt
                        <<<grid, threads_64x64, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3);
                }
                break;
            case 48:
                // k=48 needs 2 columns per thread (threads_y=32, some threads handle 1, some 2)
                solve_tri_f32_fast<64, 48, 32>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 40:
                // k=40 needs 2 columns per thread (threads_y=32, some threads handle 1, some 2)
                solve_tri_f32_fast<64, 40, 32>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 32:
                solve_tri_f32_fast<64, 32, 32>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 16:
                solve_tri_f32_fast<64, 16, 16>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 14:
                solve_tri_f32_fast<64, 14, 14>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 12:
                solve_tri_f32_fast<64, 12, 12>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 10:
                solve_tri_f32_fast<64, 10, 10>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 8:
                solve_tri_f32_fast<64, 8, 8>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 6:
                solve_tri_f32_fast<64, 6, 6>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 4:
                solve_tri_f32_fast<64, 4, 4>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 2:
                solve_tri_f32_fast<64, 2, 2>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 1:
                solve_tri_f32_fast<64, 1, 1>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            default:
                solve_tri_f32_fast<0, 0, 0>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, k);
        }
    } else {  // run general case
        solve_tri_f32_fast<0, 0, 0>
            <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, k);
    }
}

void ggml_cuda_op_solve_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // A (triangular n x n matrix)
    const ggml_tensor * src1 = dst->src[1];  // B (right hand side of n x k equation columns)

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int64_t n = src0->ne[0];
    const int64_t k = src1->ne[0];

    const int64_t total_batches = src0->ne[2] * src0->ne[3];
    const size_t X_size = n * k * total_batches * sizeof(float);

    // Copy B to X (cuBLAS solves in-place)
    CUDA_CHECK(cudaMemcpyAsync(
        dst->data, src1->data, X_size,
        cudaMemcpyDeviceToDevice, ctx.stream()
    ));

    solve_tri_f32_cublas(
        ctx,
        (const float *) src0->data,
        (float *) dst->data,
        n, k,
        src0->ne[2], src0->ne[3],
        src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
        dst->nb[2] / sizeof(float), dst->nb[3] / sizeof(float),
        ctx.stream()
    );
}
