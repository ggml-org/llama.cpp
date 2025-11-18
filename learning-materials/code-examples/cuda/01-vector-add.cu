/*
 * CUDA Example 1: Vector Addition
 * ================================
 *
 * This is the "Hello World" of CUDA programming. It demonstrates:
 * - Basic CUDA kernel syntax
 * - Memory allocation (host and device)
 * - Data transfer between host and device
 * - Kernel launch configuration
 * - Error checking
 *
 * Learning Objectives:
 * - Understand the CUDA execution model
 * - Learn memory management in CUDA
 * - Master kernel launch syntax <<<blocks, threads>>>
 * - Implement proper error handling
 *
 * Relevance to LLM Inference:
 * Vector addition is the foundation for understanding how GPUs process
 * parallel operations, which is essential for matrix operations in neural networks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro - always check CUDA calls!
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/*
 * DEVICE CODE (runs on GPU)
 * ==========================
 *
 * __global__ keyword indicates this is a CUDA kernel that:
 * - Runs on the device (GPU)
 * - Is callable from the host (CPU)
 * - Returns void
 *
 * Each thread computes one element of the output array.
 * This is called "element-wise" parallelism.
 */
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    /*
     * Calculate global thread index:
     * - blockIdx.x: which block this thread belongs to
     * - blockDim.x: number of threads per block
     * - threadIdx.x: thread index within the block
     *
     * Example: If blockDim.x = 256, and blockIdx.x = 2, threadIdx.x = 10
     * Then idx = 2 * 256 + 10 = 522
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * Boundary check: Not all problems divide evenly by block size
     * Extra threads should do nothing
     */
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/*
 * HOST CODE (runs on CPU)
 * =======================
 */

// Initialize vector with random values
void initializeVector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Verify results
bool verifyResults(const float* a, const float* b, const float* c, int n) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < n; i++) {
        float expected = a[i] + b[i];
        if (fabs(c[i] - expected) > epsilon) {
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected, c[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== CUDA Vector Addition Example ===\n\n");

    // Problem size
    const int N = 1 << 20;  // 1 Million elements (2^20)
    const size_t bytes = N * sizeof(float);

    printf("Vector size: %d elements (%.2f MB)\n", N, bytes / (1024.0f * 1024.0f));

    /*
     * Step 1: Allocate host memory
     * =============================
     * Regular CPU memory allocation
     */
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input vectors
    initializeVector(h_a, N);
    initializeVector(h_b, N);

    /*
     * Step 2: Allocate device memory
     * ===============================
     * cudaMalloc is like malloc but allocates GPU memory
     */
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    /*
     * Step 3: Copy data from host to device
     * ======================================
     * cudaMemcpyHostToDevice: CPU -> GPU
     * This is often a performance bottleneck!
     */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /*
     * Step 4: Configure kernel launch parameters
     * ===========================================
     * threads_per_block: Number of threads in each block (usually 128, 256, or 512)
     * num_blocks: Number of blocks needed to cover all elements
     *
     * Why 256? It's a good balance:
     * - Multiple of 32 (warp size)
     * - Not too large (max is 1024)
     * - Allows good occupancy
     */
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;  // Ceiling division

    printf("\nLaunch configuration:\n");
    printf("  Threads per block: %d\n", threads_per_block);
    printf("  Number of blocks: %d\n", num_blocks);
    printf("  Total threads: %d\n", num_blocks * threads_per_block);

    /*
     * Step 5: Launch kernel
     * =====================
     * Syntax: kernelName<<<num_blocks, threads_per_block>>>(args...)
     *
     * This launches num_blocks * threads_per_block threads in parallel!
     * Each thread executes the same kernel code but with different data.
     */
    vectorAddKernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    /*
     * Step 6: Copy result back to host
     * =================================
     * cudaMemcpyDeviceToHost: GPU -> CPU
     */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /*
     * Step 7: Verify results
     * ======================
     */
    printf("\nVerifying results...\n");
    if (verifyResults(h_a, h_b, h_c, N)) {
        printf("SUCCESS! All results correct.\n");
    } else {
        printf("FAILURE! Results incorrect.\n");
    }

    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("  %.3f + %.3f = %.3f\n", h_a[i], h_b[i], h_c[i]);
    }

    /*
     * Step 8: Cleanup
     * ===============
     * Always free allocated memory!
     */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c);

    printf("\n=== Example Complete ===\n");

    return EXIT_SUCCESS;
}

/*
 * PERFORMANCE NOTES
 * =================
 *
 * 1. Memory Transfer Bottleneck:
 *    - PCIe bandwidth is limited (~16 GB/s for PCIe 3.0 x16)
 *    - Computation is much faster than data transfer
 *    - For real applications, minimize host-device transfers
 *
 * 2. Kernel Optimization:
 *    - This kernel is memory-bound (limited by memory bandwidth)
 *    - Each thread does minimal computation (one addition)
 *    - GPU global memory bandwidth: ~750 GB/s (RTX 3090)
 *    - Theoretical peak: We read 2 floats and write 1 float per operation
 *
 * 3. Thread Block Size:
 *    - Must be multiple of 32 (warp size)
 *    - 256 is a good default for simple kernels
 *    - Larger blocks can reduce overhead but may limit occupancy
 *
 * 4. Occupancy:
 *    - Ratio of active warps to maximum possible warps
 *    - Higher occupancy can hide memory latency
 *    - Use CUDA Occupancy Calculator or nvprof to analyze
 *
 * BUILD INSTRUCTIONS
 * ==================
 *
 * Compile with:
 *   nvcc -o vector_add 01-vector-add.cu
 *
 * Run:
 *   ./vector_add
 *
 * With optimization:
 *   nvcc -O3 -o vector_add 01-vector-add.cu
 *
 * For specific GPU architecture (e.g., Ampere):
 *   nvcc -O3 -arch=sm_86 -o vector_add 01-vector-add.cu
 *
 * EXERCISES
 * =========
 *
 * 1. Modify to use different thread block sizes (128, 512, 1024)
 *    - Measure performance differences
 *
 * 2. Implement subtraction, multiplication, division kernels
 *
 * 3. Add timing code to measure kernel execution time
 *    Hint: Use cudaEvent_t for accurate GPU timing
 *
 * 4. Try different vector sizes (small, large)
 *    - Observe when computation becomes worthwhile vs. CPU
 *
 * 5. Implement a CPU version and compare performance
 */
