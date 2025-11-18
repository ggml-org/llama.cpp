/**
 * GPU Profiling Example
 *
 * Demonstrates:
 * - CUDA events for timing
 * - Memory bandwidth measurement
 * - Occupancy checking
 * - NVTX markers for Nsight
 *
 * Compile: nvcc -o profiling 03_profiling_example.cu -lnvToolsExt -arch=sm_80
 * Run: ./profiling
 * Profile: ncu --set full ./profiling
 *          nsys profile --trace=cuda,nvtx ./profiling
 */

#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Simple vector add kernel (memory-bound)
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Compute-intensive kernel (compute-bound)
__global__ void compute_heavy(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // Expensive computation
        for (int i = 0; i < 100; i++) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
        }
        out[idx] = val;
    }
}

// Matrix multiply (to test occupancy)
__global__ void matrix_multiply(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + 31) / 32; t++) {
        // Load tiles
        if (row < M && (t * 32 + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * 32 + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * 32 + threadIdx.y) < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * 32 + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute
        for (int k = 0; k < 32; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Helper: Measure bandwidth
void measure_bandwidth(const char* name, size_t bytes, float milliseconds) {
    double bandwidth_gb = (bytes / 1e9) / (milliseconds / 1000.0);
    printf("  %s: %.2f GB/s\n", name, bandwidth_gb);
}

// Helper: Measure GFLOPS
void measure_gflops(const char* name, size_t flops, float milliseconds) {
    double gflops = (flops / 1e9) / (milliseconds / 1000.0);
    printf("  %s: %.2f GFLOPS\n", name, gflops);
}

int main() {
    printf("GPU Profiling Example\n");
    printf("=====================\n\n");

    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("Memory Clock: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d-bit\n", prop.memoryBusWidth);

    double peak_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    printf("Peak Memory Bandwidth: %.2f GB/s\n", peak_bandwidth);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("\n");

    // Problem sizes
    const int N = 16 * 1024 * 1024;  // 16M elements
    const int M = 1024, K = 1024;     // Matrix size

    size_t bytes_vector = N * sizeof(float);
    size_t bytes_matrix = M * K * sizeof(float);

    // Allocate memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes_vector));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_vector));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_vector));

    float *d_mat_a, *d_mat_b, *d_mat_c;
    CUDA_CHECK(cudaMalloc(&d_mat_a, bytes_matrix));
    CUDA_CHECK(cudaMalloc(&d_mat_b, bytes_matrix));
    CUDA_CHECK(cudaMalloc(&d_mat_c, bytes_matrix));

    // Initialize data
    CUDA_CHECK(cudaMemset(d_a, 1, bytes_vector));
    CUDA_CHECK(cudaMemset(d_b, 1, bytes_vector));
    CUDA_CHECK(cudaMemset(d_mat_a, 1, bytes_matrix));
    CUDA_CHECK(cudaMemset(d_mat_b, 1, bytes_matrix));

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    //
    // Test 1: Memory-Bound Kernel (Vector Add)
    //
    printf("Test 1: Memory-Bound Kernel (Vector Add)\n");
    printf("-----------------------------------------\n");

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    nvtxRangePush("Vector Add");
    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("  Time: %.3f ms\n", ms);
    measure_bandwidth("Effective Bandwidth", 3 * bytes_vector, ms);
    printf("  Efficiency: %.1f%% of peak\n",
           (3 * bytes_vector / 1e9) / (ms / 1000.0) / peak_bandwidth * 100);
    printf("\n");

    //
    // Test 2: Compute-Bound Kernel
    //
    printf("Test 2: Compute-Bound Kernel\n");
    printf("-----------------------------\n");

    nvtxRangePush("Compute Heavy");
    CUDA_CHECK(cudaEventRecord(start));
    compute_heavy<<<blocks, threads>>>(d_a, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Approximate FLOP count (sin, cos, sqrt, mul, add per iteration × 100 iterations)
    size_t flops = (size_t)N * 100 * 5;

    printf("  Time: %.3f ms\n", ms);
    measure_gflops("Compute Performance", flops, ms);
    printf("\n");

    //
    // Test 3: Matrix Multiply with Occupancy Check
    //
    printf("Test 3: Matrix Multiply (Occupancy Check)\n");
    printf("------------------------------------------\n");

    dim3 block_mm(32, 32);
    dim3 grid_mm((K + 31) / 32, (M + 31) / 32);

    // Calculate theoretical occupancy
    int numBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, matrix_multiply, block_mm.x * block_mm.y, 32*32*sizeof(float)));

    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int maxBlocksPerSM = maxThreadsPerSM / (block_mm.x * block_mm.y);

    printf("  Block size: %dx%d = %d threads\n", block_mm.x, block_mm.y,
           block_mm.x * block_mm.y);
    printf("  Shared memory per block: %zu bytes\n", 2 * 32 * 32 * sizeof(float));
    printf("  Theoretical Occupancy: %.1f%% (%d/%d blocks per SM)\n",
           (float)numBlocks / maxBlocksPerSM * 100, numBlocks, maxBlocksPerSM);

    nvtxRangePush("Matrix Multiply");
    CUDA_CHECK(cudaEventRecord(start));
    matrix_multiply<<<grid_mm, block_mm>>>(d_mat_a, d_mat_b, d_mat_c, M, K, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    size_t matmul_flops = (size_t)2 * M * K * K;
    printf("  Time: %.3f ms\n", ms);
    measure_gflops("Performance", matmul_flops, ms);
    printf("\n");

    //
    // Test 4: Kernel Launch Overhead
    //
    printf("Test 4: Kernel Launch Overhead\n");
    printf("-------------------------------\n");

    const int num_launches = 1000;

    nvtxRangePush("Launch Overhead Test");
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_launches; i++) {
        vector_add<<<1, 1>>>(d_a, d_b, d_c, 1);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("  Total time for %d launches: %.3f ms\n", num_launches, ms);
    printf("  Average launch overhead: %.3f μs\n", ms * 1000.0 / num_launches);
    printf("\n");

    //
    // Test 5: Memory Transfer Timing
    //
    printf("Test 5: Memory Transfer Performance\n");
    printf("------------------------------------\n");

    float* h_data = (float*)malloc(bytes_vector);

    // Host to Device
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_a, h_data, bytes_vector, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    measure_bandwidth("H2D Transfer", bytes_vector, ms);

    // Device to Host
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_data, d_a, bytes_vector, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    measure_bandwidth("D2H Transfer", bytes_vector, ms);

    // Device to Device
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_c, d_a, bytes_vector, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    measure_bandwidth("D2D Transfer", bytes_vector, ms);

    printf("\n");

    printf("Profiling Tips:\n");
    printf("  1. Run with Nsight Compute: ncu --set full ./profiling\n");
    printf("  2. Run with Nsight Systems: nsys profile --trace=cuda,nvtx ./profiling\n");
    printf("  3. Check roofline: ncu --set roofline --kernel-name vector_add ./profiling\n");
    printf("\n");

    // Cleanup
    free(h_data);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_mat_a); cudaFree(d_mat_b); cudaFree(d_mat_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("Done!\n");
    return 0;
}
