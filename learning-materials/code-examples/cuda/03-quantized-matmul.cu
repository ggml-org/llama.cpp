/*
 * CUDA Example 3: Quantized Matrix Multiplication
 * ================================================
 *
 * This example demonstrates:
 * - INT4/INT8 quantized operations
 * - Dequantization techniques
 * - Mixed-precision computation
 * - Per-channel quantization
 * - Performance vs. accuracy tradeoffs
 *
 * Learning Objectives:
 * - Understand quantization for neural networks
 * - Learn how to work with low-precision data types
 * - Master dequantization strategies
 * - Optimize memory bandwidth with quantization
 *
 * Relevance to LLM Inference:
 * Modern LLMs use quantization to reduce memory footprint and increase speed:
 * - GPT models: 16-bit (FP16/BF16) and 8-bit (INT8) quantization
 * - Llama models: Support for 4-bit quantization (GPTQ, GGUF)
 * - Memory savings: 4-bit is 8x smaller than FP32
 * - Speed: Lower precision = higher throughput
 *
 * Quantization Basics:
 * - Original weights: 32-bit float (FP32) - 4 bytes per value
 * - Quantized weights: 8-bit int (INT8) - 1 byte per value (4x smaller!)
 * - Quantized weights: 4-bit int (INT4) - 0.5 bytes per value (8x smaller!)
 *
 * Mathematical Formula:
 *   quantized_value = round((float_value - zero_point) / scale)
 *   float_value = quantized_value * scale + zero_point
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define TILE_SIZE 16

/*
 * QUANTIZATION PARAMETERS STRUCTURE
 * ==================================
 *
 * Per-channel quantization: Each row/column has its own scale and zero-point
 * This provides better accuracy than per-tensor quantization
 */
struct QuantParams {
    float scale;
    int8_t zero_point;
};

/*
 * HOST QUANTIZATION FUNCTIONS
 * ============================
 */

// Quantize a single float value to INT8
int8_t quantize_fp32_to_int8(float value, float scale, int8_t zero_point) {
    float scaled = value / scale;
    int32_t quantized = (int32_t)roundf(scaled) + zero_point;

    // Clamp to INT8 range [-128, 127]
    if (quantized < -128) quantized = -128;
    if (quantized > 127) quantized = 127;

    return (int8_t)quantized;
}

// Dequantize INT8 to float
float dequantize_int8_to_fp32(int8_t value, float scale, int8_t zero_point) {
    return (float)(value - zero_point) * scale;
}

// Compute quantization parameters for a vector (row or column)
QuantParams compute_quant_params(const float* data, int size) {
    // Find min and max values
    float min_val = data[0];
    float max_val = data[0];

    for (int i = 1; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    // Symmetric quantization (zero_point = 0)
    // This is simpler and often used in practice
    float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));

    QuantParams params;
    params.scale = abs_max / 127.0f;  // Map [-abs_max, abs_max] to [-127, 127]
    params.zero_point = 0;

    // Handle edge case: all zeros
    if (params.scale == 0.0f) {
        params.scale = 1.0f;
    }

    return params;
}

// Quantize entire matrix with per-row quantization
void quantize_matrix(const float* input, int8_t* output, QuantParams* params,
                     int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        // Compute quantization parameters for this row
        params[i] = compute_quant_params(&input[i * cols], cols);

        // Quantize all elements in the row
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            output[idx] = quantize_fp32_to_int8(input[idx], params[i].scale, params[i].zero_point);
        }
    }
}

/*
 * DEVICE FUNCTION: Dequantize on GPU
 * ===================================
 */
__device__ float dequantize_int8(int8_t value, float scale, int8_t zero_point) {
    return (float)(value - zero_point) * scale;
}

/*
 * BASELINE: FP32 Matrix Multiplication
 * =====================================
 * Standard floating-point matrix multiplication for comparison
 */
__global__ void matmul_fp32(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tiles into shared memory
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/*
 * QUANTIZED MATRIX MULTIPLICATION KERNEL
 * =======================================
 *
 * Strategy: Compute in INT32, dequantize final result
 * - Load INT8 values into shared memory
 * - Perform integer multiplication and accumulation
 * - Dequantize final result using scale factors
 *
 * Benefits:
 * - 4x less memory bandwidth (INT8 vs FP32)
 * - Integer arithmetic can be faster
 * - Larger models fit in GPU memory
 */
__global__ void matmul_int8(const int8_t* A, const int8_t* B, float* C,
                            const QuantParams* paramsA, const QuantParams* paramsB,
                            int M, int N, int K) {
    /*
     * Shared memory for INT8 tiles
     * Less memory usage than FP32!
     */
    __shared__ int8_t tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t tileB[TILE_SIZE][TILE_SIZE];

    // Also cache scale factors in shared memory
    __shared__ float scalesA[TILE_SIZE];
    __shared__ float scalesB[TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    /*
     * Use INT32 accumulator to avoid overflow
     * INT8 * INT8 = up to 16-bit result
     * Accumulating many of these requires 32-bit
     */
    int32_t sum = 0;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load quantized tiles
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0;

        // Load scale factors (one thread per row/column)
        if (threadIdx.x == 0 && row < M) {
            scalesA[threadIdx.y] = paramsA[row].scale;
        }
        if (threadIdx.y == 0 && col < N) {
            scalesB[threadIdx.x] = paramsB[col].scale;
        }

        __syncthreads();

        /*
         * Integer multiplication and accumulation
         * Much faster than FP32 on many architectures
         */
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += (int32_t)tileA[threadIdx.y][k] * (int32_t)tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    /*
     * Dequantize final result
     * =======================
     * result_fp32 = (sum_int32) * scaleA * scaleB
     *
     * This is where we convert back to floating point
     * Only done once per output element!
     */
    if (row < M && col < N) {
        float scaleA = scalesA[threadIdx.y];
        float scaleB = scalesB[threadIdx.x];
        C[row * N + col] = (float)sum * scaleA * scaleB;
    }
}

/*
 * HYBRID KERNEL: INT8 weights, FP32 activations
 * ==============================================
 *
 * Common pattern in LLM inference:
 * - Weights (B matrix): Quantized to INT8 (stored on disk/memory)
 * - Activations (A matrix): Keep in FP32 (computed on-the-fly)
 *
 * This is the most practical approach for inference:
 * - Weights are large and constant (good for quantization)
 * - Activations are small and dynamic (keep precision)
 */
__global__ void matmul_hybrid(const float* A, const int8_t* B, float* C,
                              const QuantParams* paramsB,
                              int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t tileB[TILE_SIZE][TILE_SIZE];
    __shared__ float scalesB[TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // Load FP32 activations
        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        // Load INT8 weights
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0;

        // Load scale for weight column
        if (threadIdx.y == 0 && col < N) {
            scalesB[threadIdx.x] = paramsB[col].scale;
        }

        __syncthreads();

        // Mixed-precision computation
        for (int k = 0; k < TILE_SIZE; k++) {
            float a_val = tileA[threadIdx.y][k];
            float b_val = (float)tileB[k][threadIdx.x] * scalesB[threadIdx.x];
            sum += a_val * b_val;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/*
 * HOST UTILITY FUNCTIONS
 * ======================
 */

void initializeMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

void matmulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool verifyResults(const float* reference, const float* test, int size, float tolerance) {
    int errors = 0;
    float max_error = 0.0f;

    for (int i = 0; i < size; i++) {
        float error = fabsf(reference[i] - test[i]);
        if (error > max_error) max_error = error;

        if (error > tolerance) {
            if (errors < 5) {
                printf("  Error at %d: ref=%.6f, test=%.6f, diff=%.6f\n",
                       i, reference[i], test[i], error);
            }
            errors++;
        }
    }

    printf("  Max error: %.6f\n", max_error);
    printf("  Total errors: %d / %d (%.2f%%)\n", errors, size, 100.0f * errors / size);

    return errors == 0;
}

int main() {
    printf("=== CUDA Quantized Matrix Multiplication Example ===\n\n");

    // Matrix dimensions
    const int M = 512;
    const int K = 512;
    const int N = 512;

    printf("Matrix dimensions:\n");
    printf("  A: %d x %d\n", M, K);
    printf("  B: %d x %d\n", K, N);
    printf("  C: %d x %d\n\n", M, N);

    // Memory sizes
    size_t bytes_fp32_A = M * K * sizeof(float);
    size_t bytes_fp32_B = K * N * sizeof(float);
    size_t bytes_fp32_C = M * N * sizeof(float);
    size_t bytes_int8_A = M * K * sizeof(int8_t);
    size_t bytes_int8_B = K * N * sizeof(int8_t);

    printf("Memory comparison (FP32 vs INT8):\n");
    printf("  Matrix A: %.2f MB (FP32) vs %.2f MB (INT8) - %.1fx reduction\n",
           bytes_fp32_A / (1024.0f * 1024.0f),
           bytes_int8_A / (1024.0f * 1024.0f),
           (float)bytes_fp32_A / bytes_int8_A);
    printf("  Matrix B: %.2f MB (FP32) vs %.2f MB (INT8) - %.1fx reduction\n\n",
           bytes_fp32_B / (1024.0f * 1024.0f),
           bytes_int8_B / (1024.0f * 1024.0f),
           (float)bytes_fp32_B / bytes_int8_B);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes_fp32_A);
    float *h_B = (float*)malloc(bytes_fp32_B);
    float *h_C_fp32 = (float*)malloc(bytes_fp32_C);
    float *h_C_int8 = (float*)malloc(bytes_fp32_C);
    float *h_C_hybrid = (float*)malloc(bytes_fp32_C);
    float *h_C_reference = (float*)malloc(bytes_fp32_C);

    int8_t *h_A_quant = (int8_t*)malloc(bytes_int8_A);
    int8_t *h_B_quant = (int8_t*)malloc(bytes_int8_B);

    QuantParams *h_paramsA = (QuantParams*)malloc(M * sizeof(QuantParams));
    QuantParams *h_paramsB = (QuantParams*)malloc(N * sizeof(QuantParams));

    // Initialize matrices
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    printf("Quantizing matrices...\n");

    // Quantize A (row-wise)
    quantize_matrix(h_A, h_A_quant, h_paramsA, M, K);

    // Quantize B (column-wise for better cache behavior)
    // Note: We need to transpose B for column-wise quantization
    float *h_B_T = (float*)malloc(bytes_fp32_B);
    int8_t *h_B_T_quant = (int8_t*)malloc(bytes_int8_B);
    QuantParams *h_paramsB_T = (QuantParams*)malloc(K * sizeof(QuantParams));

    // Transpose B
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B_T[j * K + i] = h_B[i * N + j];
        }
    }

    // Quantize B column-wise (which is row-wise in transposed form)
    quantize_matrix(h_B_T, h_B_T_quant, h_paramsB_T, N, K);

    // Transpose back for kernel
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B_quant[i * N + j] = h_B_T_quant[j * K + i];
            if (i == 0) h_paramsB[j] = h_paramsB_T[j];
        }
    }

    printf("Quantization complete!\n\n");

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    int8_t *d_A_quant, *d_B_quant;
    QuantParams *d_paramsA, *d_paramsB;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_fp32_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_fp32_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_fp32_C));
    CUDA_CHECK(cudaMalloc(&d_A_quant, bytes_int8_A));
    CUDA_CHECK(cudaMalloc(&d_B_quant, bytes_int8_B));
    CUDA_CHECK(cudaMalloc(&d_paramsA, M * sizeof(QuantParams)));
    CUDA_CHECK(cudaMalloc(&d_paramsB, N * sizeof(QuantParams)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_fp32_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_fp32_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_quant, h_A_quant, bytes_int8_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_quant, h_B_quant, bytes_int8_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_paramsA, h_paramsA, M * sizeof(QuantParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_paramsB, h_paramsB, N * sizeof(QuantParams), cudaMemcpyHostToDevice));

    // Kernel configuration
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /*
     * Run FP32 baseline
     */
    printf("Running FP32 baseline...\n");
    CUDA_CHECK(cudaEventRecord(start));
    matmul_fp32<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_fp32;
    CUDA_CHECK(cudaEventElapsedTime(&time_fp32, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_fp32, d_C, bytes_fp32_C, cudaMemcpyDeviceToHost));
    printf("  Time: %.3f ms\n", time_fp32);

    /*
     * Run INT8 quantized
     */
    printf("Running INT8 quantized...\n");
    CUDA_CHECK(cudaEventRecord(start));
    matmul_int8<<<gridDim, blockDim>>>(d_A_quant, d_B_quant, d_C, d_paramsA, d_paramsB, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_int8;
    CUDA_CHECK(cudaEventElapsedTime(&time_int8, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_int8, d_C, bytes_fp32_C, cudaMemcpyDeviceToHost));
    printf("  Time: %.3f ms\n", time_int8);

    /*
     * Run hybrid (FP32 activations, INT8 weights)
     */
    printf("Running hybrid (FP32 x INT8)...\n");
    CUDA_CHECK(cudaEventRecord(start));
    matmul_hybrid<<<gridDim, blockDim>>>(d_A, d_B_quant, d_C, d_paramsB, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_hybrid;
    CUDA_CHECK(cudaEventElapsedTime(&time_hybrid, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_hybrid, d_C, bytes_fp32_C, cudaMemcpyDeviceToHost));
    printf("  Time: %.3f ms\n\n", time_hybrid);

    /*
     * Performance summary
     */
    printf("Performance Summary:\n");
    printf("  FP32:    %.3f ms (baseline)\n", time_fp32);
    printf("  INT8:    %.3f ms (%.2fx speedup)\n", time_int8, time_fp32 / time_int8);
    printf("  Hybrid:  %.3f ms (%.2fx speedup)\n\n", time_hybrid, time_fp32 / time_hybrid);

    // Calculate GFLOPS
    double gflops = (2.0 * M * N * K) / 1e9;
    printf("  FP32:    %.2f GFLOPS\n", gflops / (time_fp32 / 1000.0));
    printf("  INT8:    %.2f GFLOPS\n", gflops / (time_int8 / 1000.0));
    printf("  Hybrid:  %.2f GFLOPS\n\n", gflops / (time_hybrid / 1000.0));

    /*
     * Accuracy verification
     */
    printf("Computing CPU reference (FP32)...\n");
    matmulCPU(h_A, h_B, h_C_reference, M, N, K);

    printf("\nAccuracy Analysis:\n");
    printf("INT8 vs FP32 reference:\n");
    verifyResults(h_C_reference, h_C_int8, M * N, 1.0f);  // Relaxed tolerance

    printf("\nHybrid vs FP32 reference:\n");
    verifyResults(h_C_reference, h_C_hybrid, M * N, 1.0f);

    printf("\nFP32 GPU vs FP32 CPU reference:\n");
    verifyResults(h_C_reference, h_C_fp32, M * N, 1e-3f);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A_quant));
    CUDA_CHECK(cudaFree(d_B_quant));
    CUDA_CHECK(cudaFree(d_paramsA));
    CUDA_CHECK(cudaFree(d_paramsB));

    free(h_A);
    free(h_B);
    free(h_C_fp32);
    free(h_C_int8);
    free(h_C_hybrid);
    free(h_C_reference);
    free(h_A_quant);
    free(h_B_quant);
    free(h_B_T);
    free(h_B_T_quant);
    free(h_paramsA);
    free(h_paramsB);
    free(h_paramsB_T);

    printf("\n=== Example Complete ===\n");
    return EXIT_SUCCESS;
}

/*
 * QUANTIZATION INSIGHTS FOR LLM INFERENCE
 * ========================================
 *
 * 1. Why Quantization Matters:
 *    - LLaMA-70B: ~140 GB in FP32, ~35 GB in INT8, ~18 GB in INT4
 *    - Fits larger models in GPU memory
 *    - Reduces memory bandwidth bottleneck
 *    - Enables faster inference
 *
 * 2. Quantization Strategies:
 *    a) Post-Training Quantization (PTQ):
 *       - Quantize pre-trained model
 *       - No retraining needed
 *       - Examples: GPTQ, AWQ
 *
 *    b) Quantization-Aware Training (QAT):
 *       - Train with quantization in mind
 *       - Better accuracy
 *       - More expensive
 *
 * 3. Granularity Levels:
 *    - Per-tensor: One scale for entire matrix (simple, less accurate)
 *    - Per-channel: Scale per row/column (shown here, good balance)
 *    - Per-group: Scale for groups of values (GPTQ uses this)
 *    - Per-token: Dynamic scales (best accuracy, more overhead)
 *
 * 4. INT4 Quantization:
 *    - Even more memory savings (8x vs FP32)
 *    - Requires special handling (pack 2 values per byte)
 *    - Used in GGML/GGUF format (llama.cpp)
 *    - Some accuracy loss but often acceptable
 *
 * 5. Mixed Precision Strategies:
 *    - Keep sensitive layers in FP16 (e.g., first/last layers)
 *    - Quantize middle layers more aggressively
 *    - Activation tensors often stay in higher precision
 *
 * 6. Hardware Acceleration:
 *    - Modern GPUs have INT8 tensor cores
 *    - Can be faster than FP16 on some architectures
 *    - NVIDIA: Turing, Ampere, Ada have INT8/INT4 support
 *
 * BUILD INSTRUCTIONS
 * ==================
 *
 * Compile:
 *   nvcc -o quantized_matmul 03-quantized-matmul.cu
 *
 * With optimization:
 *   nvcc -O3 -o quantized_matmul 03-quantized-matmul.cu
 *
 * For Ampere architecture (supports faster INT8):
 *   nvcc -O3 -arch=sm_86 -o quantized_matmul 03-quantized-matmul.cu
 *
 * EXERCISES
 * =========
 *
 * 1. Implement INT4 quantization
 *    - Pack two 4-bit values into one byte
 *    - Implement unpacking in kernel
 *
 * 2. Try different quantization schemes:
 *    - Asymmetric quantization (non-zero zero_point)
 *    - Per-group quantization (groups of 32-128 values)
 *
 * 3. Measure accuracy vs. speedup tradeoff:
 *    - Plot accuracy loss vs. compression ratio
 *    - Find optimal bit-width for your use case
 *
 * 4. Implement dynamic quantization:
 *    - Compute scales on-the-fly
 *    - Useful for activation quantization
 *
 * 5. Use Tensor Cores for INT8 operations:
 *    - Look into wmma (warp matrix multiply-accumulate)
 *    - Can be significantly faster on modern GPUs
 *
 * 6. Compare with real quantization libraries:
 *    - TensorRT INT8 quantization
 *    - llama.cpp GGUF format
 *    - bitsandbytes library
 *
 * REFERENCES
 * ==========
 *
 * - GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
 * - AWQ: Activation-aware Weight Quantization for LLM Compression
 * - GGML: Tensor library for machine learning (used in llama.cpp)
 * - NVIDIA TensorRT: High-performance inference library
 */
