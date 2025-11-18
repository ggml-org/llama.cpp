/**
 * Quantized GEMM Kernel (Q4_0 format)
 *
 * Demonstrates on-the-fly dequantization during matrix multiply
 * Based on llama.cpp mmq.cu implementation
 *
 * Compile: nvcc -o qgemm 02_quantized_gemm.cu -arch=sm_80
 * Run: ./qgemm
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#define QK4_0 32  // Block size for Q4_0
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/**
 * Q4_0 quantization block
 * - 32 values packed into 16 bytes (4 bits each)
 * - 1 FP16 scale (delta)
 * Total: 18 bytes per 32 values = 4.5 bits/weight
 */
typedef struct {
    half d;           // Delta (scale)
    uint8_t qs[16];   // Quantized values (4-bit each, packed)
} block_q4_0;

/**
 * Dequantize Q4_0 block
 */
__device__ void dequantize_q4_0(const block_q4_0* block, float* out) {
    const float d = __half2float(block->d);

    for (int i = 0; i < 16; i++) {
        const uint8_t packed = block->qs[i];

        // Extract two 4-bit values
        const int q0 = (packed & 0x0F) - 8;  // Lower 4 bits, signed
        const int q1 = (packed >> 4) - 8;    // Upper 4 bits, signed

        out[i*2 + 0] = q0 * d;
        out[i*2 + 1] = q1 * d;
    }
}

/**
 * Quantized Matrix Multiply: C = A × B
 *
 * A: [M, K] quantized (Q4_0 format)
 * B: [K, N] float32
 * C: [M, N] float32
 */
__global__ void mul_mat_q4_0(
    const void* __restrict__ vA,   // Q4_0 weights
    const float* __restrict__ B,   // FP32 activations
    float* __restrict__ C,         // FP32 output
    int M, int N, int K
) {
    const block_q4_0* A = (const block_q4_0*)vA;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Number of Q4_0 blocks per row
    const int num_blocks = K / QK4_0;

    float sum = 0.0f;

    for (int b = 0; b < num_blocks; b++) {
        // Get Q4_0 block
        const block_q4_0* block = &A[row * num_blocks + b];

        // Dequantize block
        float dequant[QK4_0];
        dequantize_q4_0(block, dequant);

        // Dot product with B
        for (int k = 0; k < QK4_0; k++) {
            const int k_global = b * QK4_0 + k;
            sum += dequant[k] * B[col * K + k_global];
        }
    }

    C[row * N + col] = sum;
}

/**
 * Optimized version using shared memory and vectorized loads
 */
__global__ void mul_mat_q4_0_optimized(
    const void* __restrict__ vA,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const block_q4_0* A = (const block_q4_0*)vA;

    // Tile size
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 64;  // 2 Q4_0 blocks

    // Shared memory for tiles
    __shared__ float tileA[TILE_M][TILE_K];
    __shared__ float tileB[TILE_K][TILE_N];

    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;

    const int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; t++) {
        // Load A tile (dequantize)
        if (row < M && (t * TILE_K + threadIdx.x) < K) {
            const int block_idx = (t * TILE_K + threadIdx.x) / QK4_0;
            const int block_offset = (t * TILE_K + threadIdx.x) % QK4_0;

            if (block_offset == 0 && (t * TILE_K + threadIdx.x + QK4_0) <= K) {
                // Dequantize entire block
                const block_q4_0* block = &A[row * (K/QK4_0) + block_idx];
                float temp[QK4_0];
                dequantize_q4_0(block, temp);

                // Store in shared memory
                for (int k = 0; k < QK4_0 && (threadIdx.x + k) < TILE_K; k++) {
                    tileA[threadIdx.y][threadIdx.x + k] = temp[k];
                }
            }
        }

        // Load B tile
        if ((t * TILE_K + threadIdx.y) < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[col * K + t * TILE_K + threadIdx.y];
        }

        __syncthreads();

        // Compute partial product
        if (row < M && col < N) {
            for (int k = 0; k < TILE_K; k++) {
                sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Quantize FP32 array to Q4_0 format
 */
void quantize_q4_0(const float* src, block_q4_0* dst, int n) {
    const int num_blocks = n / QK4_0;

    for (int b = 0; b < num_blocks; b++) {
        const float* block_src = src + b * QK4_0;

        // Find max absolute value
        float max_abs = 0.0f;
        for (int i = 0; i < QK4_0; i++) {
            max_abs = fmaxf(max_abs, fabsf(block_src[i]));
        }

        // Calculate scale
        const float d = max_abs / 7.0f;  // 4-bit signed: -8 to 7, use -7 to 7
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

        dst[b].d = __float2half(d);

        // Quantize values
        for (int i = 0; i < 16; i++) {
            int q0 = (int)roundf(block_src[i*2 + 0] * id) + 8;
            int q1 = (int)roundf(block_src[i*2 + 1] * id) + 8;

            // Clamp to 4-bit range
            q0 = (q0 < 0) ? 0 : ((q0 > 15) ? 15 : q0);
            q1 = (q1 < 0) ? 0 : ((q1 > 15) ? 15 : q1);

            dst[b].qs[i] = (q1 << 4) | q0;
        }
    }
}

int main() {
    // Matrix dimensions
    const int M = 4096;  // Rows of A (and C)
    const int K = 4096;  // Cols of A, Rows of B
    const int N = 128;   // Cols of B (and C) - typical for 1 token

    printf("Quantized GEMM Example\n");
    printf("======================\n");
    printf("Matrix C[%d,%d] = A[%d,%d] × B[%d,%d]\n", M, N, M, K, K, N);
    printf("A is quantized (Q4_0 format)\n\n");

    // Allocate host memory
    const int size_A_fp32 = M * K;
    const int size_A_q4 = M * K / QK4_0;
    const int size_B = K * N;
    const int size_C = M * N;

    float* h_A_fp32 = (float*)malloc(size_A_fp32 * sizeof(float));
    block_q4_0* h_A_q4 = (block_q4_0*)malloc(size_A_q4 * sizeof(block_q4_0));
    float* h_B = (float*)malloc(size_B * sizeof(float));
    float* h_C = (float*)malloc(size_C * sizeof(float));

    // Initialize with random data
    for (int i = 0; i < size_A_fp32; i++) {
        h_A_fp32[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < size_B; i++) {
        h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Quantize A
    printf("Quantizing matrix A...\n");
    quantize_q4_0(h_A_fp32, h_A_q4, size_A_fp32);

    float original_size = size_A_fp32 * sizeof(float);
    float quantized_size = size_A_q4 * sizeof(block_q4_0);
    printf("Original size: %.2f MB\n", original_size / 1e6);
    printf("Quantized size: %.2f MB\n", quantized_size / 1e6);
    printf("Compression ratio: %.2fx\n\n", original_size / quantized_size);

    // Allocate device memory
    block_q4_0* d_A;
    float *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A_q4 * sizeof(block_q4_0)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A_q4, size_A_q4 * sizeof(block_q4_0), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel (optimized version)
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    mul_mat_q4_0_optimized<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    printf("Kernel executed in: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Throughput: %.2f GB/s\n",
           (quantized_size + size_B * sizeof(float) + size_C * sizeof(float)) /
           (milliseconds * 1e6));

    // Verify (sample outputs)
    printf("\nSample outputs (first 5):\n");
    for (int i = 0; i < 5; i++) {
        printf("C[%d] = %.6f\n", i, h_C[i]);
    }

    // Cleanup
    free(h_A_fp32); free(h_A_q4); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("\nDone!\n");
    return 0;
}
