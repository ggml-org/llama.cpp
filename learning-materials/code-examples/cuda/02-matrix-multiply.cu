/*
 * CUDA Example 2: Matrix Multiplication with Shared Memory
 * =========================================================
 *
 * This example demonstrates:
 * - 2D thread blocks and grids
 * - Shared memory usage for optimization
 * - Tiled matrix multiplication algorithm
 * - Thread synchronization
 * - Performance comparison (naive vs. optimized)
 *
 * Learning Objectives:
 * - Understand CUDA memory hierarchy
 * - Learn shared memory and __syncthreads()
 * - Master 2D indexing in CUDA
 * - Optimize memory access patterns
 *
 * Relevance to LLM Inference:
 * Matrix multiplication (GEMM) is the core operation in neural networks.
 * All transformer attention and feed-forward layers rely heavily on matmul.
 * Understanding these optimizations is crucial for efficient LLM inference.
 *
 * Mathematical Operation: C = A * B
 * - A: M x K matrix
 * - B: K x N matrix
 * - C: M x N matrix (result)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Tile size for shared memory (16x16 = 256 threads per block)
#define TILE_SIZE 16

/*
 * NAIVE MATRIX MULTIPLICATION KERNEL
 * ===================================
 *
 * Each thread computes one element of the result matrix.
 * This version accesses global memory for every multiplication - SLOW!
 *
 * Performance Issue:
 * - Global memory latency: ~400-800 cycles
 * - Each element requires K memory accesses
 * - No data reuse between threads
 */
__global__ void matmulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    // Calculate row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Dot product of row from A and column from B
        for (int k = 0; k < K; k++) {
            // A is row-major: A[row][k] = A[row * K + k]
            // B is row-major: B[k][col] = B[k * N + col]
            sum += A[row * K + k] * B[k * N + col];
        }

        // Write result: C[row][col]
        C[row * N + col] = sum;
    }
}

/*
 * OPTIMIZED MATRIX MULTIPLICATION KERNEL WITH SHARED MEMORY
 * ==========================================================
 *
 * Key Optimization: Tiled Algorithm
 * - Load data into shared memory (on-chip, very fast)
 * - Reuse data across threads in the same block
 * - Reduce global memory accesses by ~TILE_SIZE times
 *
 * Memory Hierarchy Performance (approximate):
 * - Global Memory: 400-800 cycles latency, ~750 GB/s bandwidth
 * - Shared Memory: ~1-2 cycles latency, ~10+ TB/s bandwidth
 * - Registers: 1 cycle, highest bandwidth
 *
 * Shared Memory Benefits:
 * - 100x lower latency than global memory
 * - Data loaded once, used by all threads in block
 * - Explicit programmer control
 */
__global__ void matmulTiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    /*
     * Shared memory allocation
     * ========================
     * __shared__ keyword: allocate in shared memory (per-block)
     * All threads in a block can access this memory
     * Much faster than global memory!
     */
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Calculate global row and column
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    /*
     * Tiled computation loop
     * ======================
     * Process matrix in TILE_SIZE x TILE_SIZE tiles
     * Number of tiles needed: ceil(K / TILE_SIZE)
     */
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        /*
         * Step 1: Load tile from A into shared memory
         * ============================================
         * Each thread loads one element
         * Coalesced memory access pattern (good for performance)
         */
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;  // Padding for boundary
        }

        /*
         * Step 2: Load tile from B into shared memory
         * ============================================
         */
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;  // Padding for boundary
        }

        /*
         * Step 3: Synchronize threads
         * ===========================
         * __syncthreads() is a barrier
         * - Ensures all threads have loaded their data
         * - No thread proceeds until all threads reach this point
         * - Critical for correctness when using shared memory!
         */
        __syncthreads();

        /*
         * Step 4: Compute partial dot product using shared memory
         * ========================================================
         * Now all data is in fast shared memory!
         * Each thread computes partial sum for its output element
         */
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        /*
         * Step 5: Synchronize before loading next tile
         * =============================================
         * Prevent overwriting shared memory before all threads finish using it
         */
        __syncthreads();
    }

    // Write final result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/*
 * HOST UTILITY FUNCTIONS
 * ======================
 */

// Initialize matrix with random values
void initializeMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Range: [-1, 1]
    }
}

// CPU matrix multiplication for verification
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

// Verify GPU results against CPU
bool verifyResults(const float* cpu, const float* gpu, int size) {
    const float epsilon = 1e-3f;  // Relaxed tolerance for floating-point
    int errors = 0;
    const int maxErrors = 10;

    for (int i = 0; i < size; i++) {
        float diff = fabs(cpu[i] - gpu[i]);
        if (diff > epsilon) {
            if (errors < maxErrors) {
                printf("Mismatch at index %d: CPU=%.6f, GPU=%.6f, diff=%.6f\n",
                       i, cpu[i], gpu[i], diff);
            }
            errors++;
        }
    }

    if (errors > 0) {
        printf("Total errors: %d out of %d elements\n", errors, size);
        return false;
    }
    return true;
}

// CUDA timing helper
float measureKernelTime(void (*kernel)(const float*, const float*, float*, int, int, int),
                        const float* d_A, const float* d_B, float* d_C,
                        int M, int N, int K, dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up run
    if (kernel == matmulNaive) {
        matmulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    } else {
        matmulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs (average of 3)
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 3; i++) {
        if (kernel == matmulNaive) {
            matmulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        } else {
            matmulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return milliseconds / 3.0f;  // Average time
}

int main() {
    printf("=== CUDA Matrix Multiplication Example ===\n\n");

    // Matrix dimensions (can be adjusted)
    const int M = 1024;  // Rows of A and C
    const int K = 1024;  // Cols of A, Rows of B
    const int N = 1024;  // Cols of B and C

    printf("Matrix dimensions:\n");
    printf("  A: %d x %d\n", M, K);
    printf("  B: %d x %d\n", K, N);
    printf("  C: %d x %d\n\n", M, N);

    // Calculate memory sizes
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    printf("Memory requirements:\n");
    printf("  A: %.2f MB\n", bytesA / (1024.0f * 1024.0f));
    printf("  B: %.2f MB\n", bytesB / (1024.0f * 1024.0f));
    printf("  C: %.2f MB\n\n", bytesC / (1024.0f * 1024.0f));

    // Allocate host memory
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C_naive = (float*)malloc(bytesC);
    float *h_C_tiled = (float*)malloc(bytesC);
    float *h_C_cpu = (float*)malloc(bytesC);

    // Initialize matrices
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    /*
     * Configure kernel launch parameters
     * ===================================
     * dim3: 3D dimensions for blocks and grids
     * For 2D matrices, we use x and y dimensions
     */
    dim3 blockDim(TILE_SIZE, TILE_SIZE);  // 16x16 = 256 threads per block
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("Kernel configuration:\n");
    printf("  Block: %d x %d threads\n", blockDim.x, blockDim.y);
    printf("  Grid: %d x %d blocks\n\n", gridDim.x, gridDim.y);

    /*
     * Run naive kernel
     * ================
     */
    printf("Running naive kernel...\n");
    float timeNaive = measureKernelTime(matmulNaive, d_A, d_B, d_C, M, N, K, gridDim, blockDim);
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytesC, cudaMemcpyDeviceToHost));
    printf("  Time: %.3f ms\n", timeNaive);

    /*
     * Run tiled kernel
     * ================
     */
    printf("Running tiled (optimized) kernel...\n");
    float timeTiled = measureKernelTime(matmulTiled, d_A, d_B, d_C, M, N, K, gridDim, blockDim);
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytesC, cudaMemcpyDeviceToHost));
    printf("  Time: %.3f ms\n\n", timeTiled);

    /*
     * Performance analysis
     * ====================
     */
    printf("Performance Analysis:\n");
    printf("  Speedup (tiled vs naive): %.2fx\n", timeNaive / timeTiled);

    // Calculate GFLOPS (Giga Floating-Point Operations Per Second)
    // Matrix multiplication requires 2*M*N*K operations (multiply-add per element)
    double gflops = (2.0 * M * N * K) / 1e9;
    printf("  Naive performance: %.2f GFLOPS\n", gflops / (timeNaive / 1000.0));
    printf("  Tiled performance: %.2f GFLOPS\n\n", gflops / (timeTiled / 1000.0));

    /*
     * Verify results
     * ==============
     */
    printf("Computing CPU reference...\n");
    matmulCPU(h_A, h_B, h_C_cpu, M, N, K);

    printf("Verifying naive kernel results... ");
    if (verifyResults(h_C_cpu, h_C_naive, M * N)) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
    }

    printf("Verifying tiled kernel results... ");
    if (verifyResults(h_C_cpu, h_C_tiled, M * N)) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    free(h_C_cpu);

    printf("\n=== Example Complete ===\n");
    return EXIT_SUCCESS;
}

/*
 * PERFORMANCE NOTES
 * =================
 *
 * 1. Why Shared Memory Helps:
 *    - Each element of A is reused N times (row dot products)
 *    - Each element of B is reused M times (column dot products)
 *    - Shared memory enables this reuse within a thread block
 *
 * 2. Memory Access Patterns:
 *    - Coalesced access: consecutive threads access consecutive memory
 *    - Naive kernel: B matrix access is strided (poor performance)
 *    - Tiled kernel: Both A and B loaded with coalesced access
 *
 * 3. Occupancy Considerations:
 *    - Shared memory per block: 2 * TILE_SIZE * TILE_SIZE * sizeof(float)
 *    - For TILE_SIZE=16: 2 * 16 * 16 * 4 = 2 KB per block
 *    - Modern GPUs have 48-96 KB shared memory per SM
 *    - Can fit many blocks concurrently
 *
 * 4. Further Optimizations (not shown here):
 *    - Use cuBLAS library (highly optimized by NVIDIA)
 *    - Tensor Cores for mixed-precision (FP16/INT8)
 *    - Larger tile sizes (e.g., 32x32)
 *    - Register tiling
 *    - Double buffering
 *
 * 5. Relevance to LLMs:
 *    - Transformer attention: Q @ K^T and (scores @ V)
 *    - Feed-forward layers: W1 @ hidden, W2 @ intermediate
 *    - These operations dominate inference time
 *    - cuBLAS and cutlass provide production-quality implementations
 *
 * BUILD INSTRUCTIONS
 * ==================
 *
 * Compile:
 *   nvcc -o matmul 02-matrix-multiply.cu
 *
 * With optimization:
 *   nvcc -O3 -o matmul 02-matrix-multiply.cu
 *
 * For specific architecture:
 *   nvcc -O3 -arch=sm_86 -o matmul 02-matrix-multiply.cu
 *
 * EXERCISES
 * =========
 *
 * 1. Experiment with different TILE_SIZE values (8, 32, 64)
 *    - Observe impact on performance and shared memory usage
 *
 * 2. Add timing for memory transfers
 *    - Compare compute time vs. transfer time
 *
 * 3. Try different matrix sizes
 *    - Small (256x256), Medium (1024x1024), Large (4096x4096)
 *
 * 4. Implement a transpose operation
 *    - B_transposed = B^T, then optimize access pattern
 *
 * 5. Profile with nsys (Nsight Systems)
 *    - nsys profile ./matmul
 *    - Analyze memory bandwidth utilization
 *
 * 6. Compare with cuBLAS
 *    - Use cublasSgemm() for reference
 *    - See how close your implementation gets to the optimized library
 */
