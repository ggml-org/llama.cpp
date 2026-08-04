#define GGML_COMMON_DECL_CUDA
#include "ggml/src/ggml-common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void cuda_check(cudaError_t result, const char * expression) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", expression, cudaGetErrorString(result));
        std::exit(1);
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)

enum class weight_kind {
    w13,
    w2,
};

struct weight_shape {
    const char * name;
    weight_kind kind;
    int src_k;
    int src_n;
    int dst_k;
    int dst_n;
};

static __device__ int source_row(weight_kind kind, int dst_row, int src_n, int dst_n) {
    if (kind == weight_kind::w2) {
        return dst_row < src_n ? dst_row : -1;
    }

    const int src_ff = src_n / 2;
    const int dst_ff = dst_n / 2;
    const int part = dst_row >= dst_ff;
    const int row = dst_row - part * dst_ff;
    return row < src_ff ? part * src_ff + row : -1;
}

static __global__ void repack_padded_aos(
        const block_mxfp4 * src, block_mxfp4 * dst, weight_kind kind,
        int src_k_blocks, int src_n, int dst_k_blocks, int dst_n, int n_experts) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = (int64_t) n_experts * dst_n * dst_k_blocks;
    if (i >= n) {
        return;
    }

    const int k_block = i % dst_k_blocks;
    const int64_t row_index = i / dst_k_blocks;
    const int dst_row = row_index % dst_n;
    const int expert = row_index / dst_n;
    const int src_row = source_row(kind, dst_row, src_n, dst_n);

    block_mxfp4 value = {};
    if (src_row >= 0 && k_block < src_k_blocks) {
        value = src[((int64_t) expert * src_n + src_row) * src_k_blocks + k_block];
    }
    dst[i] = value;
}

static __global__ void repack_padded_split_scale(
        const block_mxfp4 * src, uint8_t * qs, uint8_t * scales, weight_kind kind,
        int src_k_blocks, int src_n, int dst_k_blocks, int dst_n, int n_experts) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = (int64_t) n_experts * dst_n * dst_k_blocks;
    if (i >= n) {
        return;
    }

    const int k_block = i % dst_k_blocks;
    const int64_t row_index = i / dst_k_blocks;
    const int dst_row = row_index % dst_n;
    const int expert = row_index / dst_n;
    const int src_row = source_row(kind, dst_row, src_n, dst_n);
    const block_mxfp4 * value = nullptr;
    if (src_row >= 0 && k_block < src_k_blocks) {
        value = src + ((int64_t) expert * src_n + src_row) * src_k_blocks + k_block;
    }

    scales[i] = value ? value->e : 0;
#pragma unroll
    for (int j = 0; j < QK_MXFP4 / 2; ++j) {
        qs[i * (QK_MXFP4 / 2) + j] = value ? value->qs[j] : 0;
    }
}

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float result = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&result, start, stop));
    return result;
}

static void print_result(
        const weight_shape & shape, const char * layout, int n_experts, int iterations,
        size_t src_bytes, size_t dst_bytes, float total_ms) {
    const double average_ms = total_ms / iterations;
    const double traffic_gb = (double) (src_bytes + dst_bytes) / 1.0e9;
    std::printf(
        "{\"weight\":\"%s\",\"layout\":\"%s\",\"experts\":%d,"
        "\"src_bytes\":%zu,\"dst_bytes\":%zu,\"avg_ms\":%.6f,\"traffic_gbps\":%.3f}\n",
        shape.name, layout, n_experts, src_bytes, dst_bytes, average_ms, traffic_gb / (average_ms / 1.0e3));
}

static void run_shape(const weight_shape & shape, int n_experts, int iterations) {
    const int src_k_blocks = shape.src_k / QK_MXFP4;
    const int dst_k_blocks = shape.dst_k / QK_MXFP4;
    const int64_t src_blocks = (int64_t) n_experts * shape.src_n * src_k_blocks;
    const int64_t dst_blocks = (int64_t) n_experts * shape.dst_n * dst_k_blocks;
    const size_t src_bytes = src_blocks * sizeof(block_mxfp4);
    const size_t dst_bytes = dst_blocks * sizeof(block_mxfp4);
    constexpr int threads = 256;
    const int blocks = (int) ((dst_blocks + threads - 1) / threads);

    block_mxfp4 * src = nullptr;
    block_mxfp4 * dst = nullptr;
    uint8_t * qs = nullptr;
    uint8_t * scales = nullptr;
    CUDA_CHECK(cudaMalloc(&src, src_bytes));
    CUDA_CHECK(cudaMalloc(&dst, dst_bytes));
    CUDA_CHECK(cudaMalloc(&qs, dst_blocks * (QK_MXFP4 / 2)));
    CUDA_CHECK(cudaMalloc(&scales, dst_blocks));
    CUDA_CHECK(cudaMemset(src, 0x5a, src_bytes));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    repack_padded_aos<<<blocks, threads>>>(
        src, dst, shape.kind, src_k_blocks, shape.src_n, dst_k_blocks, shape.dst_n, n_experts);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        repack_padded_aos<<<blocks, threads>>>(
            src, dst, shape.kind, src_k_blocks, shape.src_n, dst_k_blocks, shape.dst_n, n_experts);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    print_result(shape, "padded-aos", n_experts, iterations, src_bytes, dst_bytes, elapsed_ms(start, stop));

    repack_padded_split_scale<<<blocks, threads>>>(
        src, qs, scales, shape.kind, src_k_blocks, shape.src_n, dst_k_blocks, shape.dst_n, n_experts);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        repack_padded_split_scale<<<blocks, threads>>>(
            src, qs, scales, shape.kind, src_k_blocks, shape.src_n, dst_k_blocks, shape.dst_n, n_experts);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    print_result(shape, "padded-split-scale", n_experts, iterations, src_bytes, dst_bytes, elapsed_ms(start, stop));

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(scales));
    CUDA_CHECK(cudaFree(qs));
    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFree(src));
}

int main(int argc, char ** argv) {
    int n_experts = 128;
    int iterations = 10;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--experts") == 0 && i + 1 < argc) {
            n_experts = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "usage: %s [--experts N] [--iterations N]\n", argv[0]);
            return 2;
        }
    }
    if (n_experts <= 0 || iterations <= 0) {
        std::fprintf(stderr, "experts and iterations must be positive\n");
        return 2;
    }

    const weight_shape shapes[] = {
        {"w13", weight_kind::w13, 2880, 5760, 2944, 5888},
        {"w2", weight_kind::w2, 2880, 2880, 2944, 2944},
    };
    for (const weight_shape & shape : shapes) {
        run_shape(shape, n_experts, iterations);
    }
    return 0;
}
