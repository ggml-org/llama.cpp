/**
 * Simplified Flash Attention Kernel
 *
 * Demonstrates key concepts from llama.cpp's fattn.cu
 * - Tiled computation
 * - Online softmax
 * - Shared memory usage
 *
 * Compile: nvcc -o attention 01_attention_kernel.cu -arch=sm_80
 * Run: ./attention
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>

#define TILE_SIZE 32
#define HEAD_DIM 64
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/**
 * Simplified Flash Attention Kernel
 *
 * Q: [batch, heads, seq_len, head_dim]
 * K: [batch, heads, seq_len, head_dim]
 * V: [batch, heads, seq_len, head_dim]
 * O: [batch, heads, seq_len, head_dim] (output)
 *
 * Computes: O = softmax(Q·K^T / √d) · V
 */
__global__ void flash_attention_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int seq_len,
    int head_dim,
    float scale
) {
    // Block processes one query token
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_idx = blockIdx.x;  // Query index

    // Shared memory for tiles
    __shared__ half sQ[TILE_SIZE][HEAD_DIM];
    __shared__ half sK[TILE_SIZE][HEAD_DIM];
    __shared__ half sV[TILE_SIZE][HEAD_DIM];
    __shared__ half sQK[TILE_SIZE][TILE_SIZE];  // Q·K^T scores

    const int tid = threadIdx.x;

    // Load Q tile (this query)
    if (q_idx < seq_len && tid < head_dim) {
        int q_offset = ((batch_idx * gridDim.y + head_idx) * seq_len + q_idx) * head_dim;
        sQ[0][tid] = Q[q_offset + tid];
    }

    // Initialize output accumulator
    float acc[HEAD_DIM] = {0.0f};
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Loop over K/V tiles
    const int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        const int k_idx = tile_idx * TILE_SIZE + threadIdx.y;

        // Load K tile
        if (k_idx < seq_len && tid < head_dim) {
            int k_offset = ((batch_idx * gridDim.y + head_idx) * seq_len + k_idx) * head_dim;
            sK[threadIdx.y][tid] = K[k_offset + tid];
        }

        // Load V tile
        if (k_idx < seq_len && tid < head_dim) {
            int v_offset = ((batch_idx * gridDim.y + head_idx) * seq_len + k_idx) * head_dim;
            sV[threadIdx.y][tid] = V[v_offset + tid];
        }

        __syncthreads();

        // Compute Q·K^T for this tile
        if (threadIdx.y < TILE_SIZE && tid < TILE_SIZE) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += __half2float(sQ[0][d]) * __half2float(sK[tid][d]);
            }
            sQK[0][tid] = __float2half(score * scale);
        }

        __syncthreads();

        // Online softmax update
        if (tid < TILE_SIZE) {
            float score = __half2float(sQK[0][tid]);

            // Update running max and sum (online softmax)
            float old_max = max_score;
            max_score = fmaxf(max_score, score);

            // Correction factor for previous accumulator
            float exp_correction = expf(old_max - max_score);
            sum_exp = sum_exp * exp_correction;

            // Add current tile contribution
            float exp_score = expf(score - max_score);
            sum_exp += exp_score;

            // Update accumulator (corrected and new contribution)
            for (int d = 0; d < head_dim; d++) {
                acc[d] = acc[d] * exp_correction +
                         exp_score * __half2float(sV[tid][d]);
            }
        }

        __syncthreads();
    }

    // Final normalization and write output
    if (q_idx < seq_len && tid < head_dim) {
        int o_offset = ((batch_idx * gridDim.y + head_idx) * seq_len + q_idx) * head_dim;
        O[o_offset + tid] = __float2half(acc[tid] / sum_exp);
    }
}

// Helper function to initialize random data
void init_random_fp16(half* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = __float2half((float)rand() / RAND_MAX);
    }
}

int main() {
    // Problem size
    const int batch_size = 1;
    const int num_heads = 8;
    const int seq_len = 512;
    const int head_dim = 64;
    const float scale = 1.0f / sqrtf(head_dim);

    const int qkv_size = batch_size * num_heads * seq_len * head_dim;

    printf("Flash Attention Example\n");
    printf("=======================\n");
    printf("Batch: %d, Heads: %d, Seq: %d, Dim: %d\n\n",
           batch_size, num_heads, seq_len, head_dim);

    // Allocate host memory
    half *h_Q = (half*)malloc(qkv_size * sizeof(half));
    half *h_K = (half*)malloc(qkv_size * sizeof(half));
    half *h_V = (half*)malloc(qkv_size * sizeof(half));
    half *h_O = (half*)malloc(qkv_size * sizeof(half));

    // Initialize with random data
    init_random_fp16(h_Q, qkv_size);
    init_random_fp16(h_K, qkv_size);
    init_random_fp16(h_V, qkv_size);

    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O, qkv_size * sizeof(half)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_size * sizeof(half), cudaMemcpyHostToDevice));

    // Launch configuration
    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(head_dim, TILE_SIZE);

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    flash_attention_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, head_dim, scale);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_O, d_O, qkv_size * sizeof(half), cudaMemcpyDeviceToHost));

    printf("Kernel executed in: %.3f ms\n", milliseconds);
    printf("Throughput: %.2f GFLOPS\n",
           (2.0f * batch_size * num_heads * seq_len * seq_len * head_dim) / (milliseconds * 1e6));

    // Verify output (simple sanity check)
    printf("\nSample outputs (first 5 of head 0):\n");
    for (int i = 0; i < 5; i++) {
        printf("O[%d] = %.4f\n", i, __half2float(h_O[i]));
    }

    // Cleanup
    free(h_Q); free(h_K); free(h_V); free(h_O);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("\nDone!\n");
    return 0;
}
