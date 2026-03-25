/**
 * TurboQuant TQ4_0 CUDA kernel for SET_ROWS.
 *
 * Applies random orthogonal rotation + Q4_0 block quantization per head vector.
 * Each CUDA thread block processes one head_dim-sized vector.
 *
 * The rotation matrix is stored in device memory, initialized once by tq_init().
 */

#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include <math.h>

// Global rotation matrix in device memory
static float * d_rotation_t = nullptr;  // Pi^T on device, head_dim x head_dim
static int     d_head_dim   = 0;

extern "C" {
    // Called from tq_init() to upload rotation matrix to GPU
    void tq_cuda_set_rotation(const float * rotation_t, int head_dim) {
        if (d_rotation_t) cudaFree(d_rotation_t);
        d_head_dim = head_dim;
        cudaMalloc(&d_rotation_t, head_dim * head_dim * sizeof(float));
        cudaMemcpy(d_rotation_t, rotation_t, head_dim * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    }
}

/**
 * CUDA kernel: rotate + quantize one head vector.
 *
 * Each block handles one head_dim vector:
 * 1. Read head_dim floats from src
 * 2. Compute norm, normalize
 * 3. Rotate: y = Pi^T @ x_unit (shared memory matmul)
 * 4. Quantize each 32-value sub-block as Q4_0
 * 5. Write to dst
 */
__global__ void k_tq4_0_set_rows(
    const float * __restrict__ src,      // Source FP32 data
    const int64_t * __restrict__ indices, // Row indices
    block_q4_0 * __restrict__ dst,       // Destination TQ4_0 blocks
    const float * __restrict__ rot_t,    // Rotation matrix Pi^T (head_dim x head_dim)
    const int head_dim,
    const int n_blocks_per_head,         // head_dim / 32
    const int64_t ne00,                  // elements per row in src
    const int64_t ne01,                  // rows per channel
    const int64_t s01,                   // stride for rows (in floats)
    const int64_t s1,                    // dst stride
    const int64_t s10                    // index stride
) {
    // Each block handles one head vector
    const int vec_idx = blockIdx.x;
    const int head_idx = vec_idx % (ne00 / head_dim);
    const int row_idx = vec_idx / (ne00 / head_dim);

    // Source data for this head vector
    const int64_t dst_row = indices[row_idx * s10];
    const float * src_vec = src + row_idx * s01 + head_idx * head_dim;
    block_q4_0 * dst_blocks = dst + (dst_row * s1) / sizeof(block_q4_0) + head_idx * n_blocks_per_head;

    // Shared memory for the rotated vector
    extern __shared__ float smem[];
    float * y_rot = smem;  // head_dim floats

    // Step 1: Compute norm (parallel reduction)
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = src_vec[i];
        local_sq += v * v;
    }

    // Warp reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_sq += __shfl_down_sync(0xffffffff, local_sq, offset);
    }

    // Block reduction via shared memory
    __shared__ float shared_norm;
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&shared_norm, local_sq);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        shared_norm = sqrtf(shared_norm + 1e-10f);
    }
    __syncthreads();
    float norm = shared_norm;
    float inv_norm = 1.0f / norm;

    // Step 2: Rotate (y = Pi^T @ (x / norm))
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            sum += rot_t[i * head_dim + j] * src_vec[j] * inv_norm;
        }
        y_rot[i] = sum;
    }
    __syncthreads();

    // Step 3: Quantize each 32-value block (standard Q4_0)
    for (int b = threadIdx.x; b < n_blocks_per_head; b += blockDim.x) {
        const float * bv = y_rot + b * QK4_0;

        // Find max absolute value in block
        float amax = 0.0f;
        for (int i = 0; i < QK4_0; i++) {
            float a = fabsf(bv[i]);
            if (a > amax) amax = a;
        }

        // Scale (bake norm back in)
        float d = (amax / 8.0f) * norm;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;

        // Store scale as FP16
        dst_blocks[b].d = __float2half(d);

        // Quantize to 4-bit
        for (int i = 0; i < QK4_0 / 2; i++) {
            int q0 = __float2int_rn(bv[2*i]     * norm * id) + 8;
            int q1 = __float2int_rn(bv[2*i + 1] * norm * id) + 8;
            q0 = max(0, min(15, q0));
            q1 = max(0, min(15, q1));
            dst_blocks[b].qs[i] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}
