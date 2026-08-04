#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                                                              \
    do {                                                                                                              \
        const cudaError_t err_ = (expr);                                                                              \
        if (err_ != cudaSuccess) {                                                                                     \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));          \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                             \
    } while (0)

template <bool use_mxfp8>
__device__ __forceinline__ void mma(float & d0, float & d1, float & d2, float & d3,
                                    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1, uint32_t scale_a, uint32_t scale_b) {
    if constexpr (use_mxfp8) {
        asm volatile(
            "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
    } else {
        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
    }
}

__device__ __forceinline__ void mma_mxfp4_mxfp8(float & d0, float & d1, float & d2, float & d3,
                                                 uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                 uint32_t b0, uint32_t b1, uint32_t scale_a, uint32_t scale_b) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
}

__global__ void mma_correctness_probe(float * output, uint32_t a, uint32_t b, uint32_t scale_a, uint32_t scale_b) {
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    mma_mxfp4_mxfp8(d0, d1, d2, d3,
                     a, a, a, a, b, b, scale_a, scale_b);

    const int offset = threadIdx.x * 4;
    output[offset + 0] = d0;
    output[offset + 1] = d1;
    output[offset + 2] = d2;
    output[offset + 3] = d3;
}

__host__ __device__ static int8_t e2m1_unpacked(uint8_t code) {
    const int magnitude = code & 7;
    int value;
    if (magnitude <= 4) {
        value = magnitude;
    } else if (magnitude <= 6) {
        value = 2 * magnitude - 4;
    } else {
        value = 12;
    }
    return (code & 8) != 0 ? -value : value;
}

__host__ __device__ static uint8_t e2m1_mma_byte(uint8_t code) {
    return uint8_t(code << 2);
}

__host__ __device__ static uint8_t matrix_a_code(int row, int col) {
    return uint8_t((row * 5 + col * 3) & 15);
}

__host__ __device__ static uint8_t matrix_b_code(int row, int col) {
    const uint8_t magnitude = uint8_t(0x30 + ((row + 2 * col) % 3) * 8);
    return ((row + col) & 4) != 0 ? uint8_t(magnitude | 0x80) : magnitude;
}

__host__ __device__ static float matrix_b_value(int row, int col) {
    const uint8_t code = matrix_b_code(row, col);
    float value = ldexpf(1.0f, (code & 0x7F) / 8 - 7);
    return (code & 0x80) != 0 ? -value : value;
}

__global__ void mma_matrix_probe(float * output) {
    const int lane   = threadIdx.x;
    const int group  = lane / 4;
    const int thread = lane % 4;

    uint32_t a[4] = {};
    for (int element = 0; element < 16; ++element) {
        const int row = group + (((element >= 4 && element < 8) || element >= 12) ? 8 : 0);
        const int col = thread * 4 + (element & 3) + (element >= 8 ? 16 : 0);
        const uint8_t value = e2m1_mma_byte(matrix_a_code(row, col));
        a[element / 4] |= uint32_t(value) << (8 * (element & 3));
    }

    uint32_t b[2] = {};
    for (int element = 0; element < 8; ++element) {
        const int row = thread * 4 + (element & 3) + (element >= 4 ? 16 : 0);
        b[element / 4] |= uint32_t(matrix_b_code(row, group)) << (8 * (element & 3));
    }

    float d[4] = {};
    mma_mxfp4_mxfp8(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0], b[1], 0x7Fu, 0x7Fu);
    for (int element = 0; element < 4; ++element) {
        const int row = group + (element >= 2 ? 8 : 0);
        const int col = thread * 2 + (element & 1);
        output[row * 8 + col] = d[element];
    }
}

static float matrix_correctness() {
    constexpr int values = 16 * 8;
    float * output = nullptr;
    CUDA_CHECK(cudaMalloc(&output, values * sizeof(float)));
    mma_matrix_probe<<<1, 32>>>(output);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> host(values);
    CUDA_CHECK(cudaMemcpy(host.data(), output, values * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(output));

    float max_error = 0.0f;
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 8; ++col) {
            float expected = 0.0f;
            for (int k = 0; k < 32; ++k) {
                expected += 0.5f * float(e2m1_unpacked(matrix_a_code(row, k))) * matrix_b_value(k, col);
            }
            max_error = std::max(max_error, std::abs(host[row * 8 + col] - expected));
        }
    }
    return max_error;
}

static float constant_error(uint8_t a, uint8_t b, uint32_t scale_a, uint32_t scale_b, float expected) {
    constexpr int values = 32 * 4;
    float * output = nullptr;
    CUDA_CHECK(cudaMalloc(&output, values * sizeof(float)));
    const uint32_t a4 = uint32_t(a) * 0x01010101u;
    const uint32_t b4 = uint32_t(b) * 0x01010101u;
    mma_correctness_probe<<<1, 32>>>(output, a4, b4, scale_a, scale_b);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> host(values);
    CUDA_CHECK(cudaMemcpy(host.data(), output, values * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(output));

    float max_error = 0.0f;
    for (float value : host) {
        max_error = std::max(max_error, std::abs(value - expected));
    }
    return max_error;
}

template <bool use_mxfp8, int iterations>
__global__ void mma_probe(float * output) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    const uint32_t a0 = 0x38383838u ^ lane;
    const uint32_t a1 = 0x38383838u ^ (lane << 1);
    const uint32_t a2 = 0x38383838u ^ (lane << 2);
    const uint32_t a3 = 0x38383838u ^ (lane << 3);
    const uint32_t b0 = 0x22222222u ^ lane;
    const uint32_t b1 = 0x22222222u ^ (lane << 1);
    const uint32_t scale_a = 0x7f7f7f7fu;
    const uint32_t scale_b = 0x7f7f7f7fu;

    float d00 = 0.0f;
    float d01 = 0.0f;
    float d02 = 0.0f;
    float d03 = 0.0f;
    float d10 = 0.0f;
    float d11 = 0.0f;
    float d12 = 0.0f;
    float d13 = 0.0f;
    float d20 = 0.0f;
    float d21 = 0.0f;
    float d22 = 0.0f;
    float d23 = 0.0f;
    float d30 = 0.0f;
    float d31 = 0.0f;
    float d32 = 0.0f;
    float d33 = 0.0f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        mma<use_mxfp8>(d00, d01, d02, d03, a0, a1, a2, a3, b0, b1, scale_a, scale_b);
        mma<use_mxfp8>(d10, d11, d12, d13, a0, a1, a2, a3, b0, b1, scale_a, scale_b);
        mma<use_mxfp8>(d20, d21, d22, d23, a0, a1, a2, a3, b0, b1, scale_a, scale_b);
        mma<use_mxfp8>(d30, d31, d32, d33, a0, a1, a2, a3, b0, b1, scale_a, scale_b);
        if constexpr (use_mxfp8) {
            mma<use_mxfp8>(d00, d01, d02, d03, a0, a1, a2, a3, b0, b1, scale_a, scale_b);
            mma<use_mxfp8>(d10, d11, d12, d13, a0, a1, a2, a3, b0, b1, scale_a, scale_b);
            mma<use_mxfp8>(d20, d21, d22, d23, a0, a1, a2, a3, b0, b1, scale_a, scale_b);
            mma<use_mxfp8>(d30, d31, d32, d33, a0, a1, a2, a3, b0, b1, scale_a, scale_b);
        }
    }

    if (lane == 0) {
        output[warp] =
            d00 + d01 + d02 + d03 + d10 + d11 + d12 + d13 +
            d20 + d21 + d22 + d23 + d30 + d31 + d32 + d33;
    }
}

template <bool use_mxfp8, int iterations>
static float benchmark(int blocks, int threads, int repetitions, float * output) {
    for (int i = 0; i < 3; ++i) {
        mma_probe<use_mxfp8, iterations><<<blocks, threads>>>(output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times(repetitions);
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < repetitions; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        mma_probe<use_mxfp8, iterations><<<blocks, threads>>>(output);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main(int argc, char ** argv) {
    constexpr int iterations = 4096;
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 512;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 128;
    const int repetitions = argc > 3 ? std::atoi(argv[3]) : 9;

    if (blocks <= 0 || threads <= 0 || threads % 32 != 0 || repetitions <= 0) {
        std::fprintf(stderr, "usage: %s [blocks] [threads] [repetitions]\n", argv[0]);
        return EXIT_FAILURE;
    }

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    const float unit_scale_error = constant_error(8, 0x38u, 0x7Fu, 0x7Fu, 32.0f);
    const float inverse_scale_error = constant_error(8, 0x38u, 0x80u, 0x7Eu, 32.0f);
    const float matrix_error = matrix_correctness();
    if (unit_scale_error != 0.0f || inverse_scale_error != 0.0f || matrix_error != 0.0f) {
        std::fprintf(stderr, "mxfp4_x_mxfp8 correctness failed: unit=%g inverse=%g matrix=%g\n",
                     unit_scale_error, inverse_scale_error, matrix_error);
        return EXIT_FAILURE;
    }

    const int warps = blocks * threads / 32;
    float * output = nullptr;
    CUDA_CHECK(cudaMalloc(&output, size_t(warps) * sizeof(float)));

    const float fp4_ms = benchmark<false, iterations>(blocks, threads, repetitions, output);
    const float fp8_ms = benchmark<true, iterations>(blocks, threads, repetitions, output);
    CUDA_CHECK(cudaFree(output));

    const double flops = double(warps) * iterations * 4.0 * 2.0 * 16.0 * 8.0 * 64.0;
    const double fp4_tflops = flops / (double(fp4_ms) * 1.0e9);
    const double fp8_tflops = flops / (double(fp8_ms) * 1.0e9);

    std::printf("gpu=%s blocks=%d threads=%d warps=%d iterations=%d\n",
                props.name, blocks, threads, warps, iterations);
    std::printf("mxfp4_x_mxfp8_max_error=%.9g\n", std::max({unit_scale_error, inverse_scale_error, matrix_error}));
    std::printf("mxfp4_x_mxfp4_ms=%.6f tflops=%.3f\n", fp4_ms, fp4_tflops);
    std::printf("mxfp8_x_mxfp4_ms=%.6f tflops=%.3f\n", fp8_ms, fp8_tflops);
    std::printf("mxfp8_over_mxfp4=%.6f\n", fp4_ms / fp8_ms);
    return EXIT_SUCCESS;
}
