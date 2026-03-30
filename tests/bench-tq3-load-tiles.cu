// Microbenchmark: measure the TQ3_0 -> q8_0 MMQ tile transform with CUDA events.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#define QK 32
#define QI8_0 8

typedef struct { __half d; uint8_t qs[12]; } block_tq3_0;

static float cpu_sign(int i) {
    return ((((unsigned) i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

static const float B[7] = {-1.7479f, -1.0500f, -0.5005f, 0.0f, 0.5005f, 1.0500f, 1.7479f};

static void cpu_quantize_tq3(const float * x, block_tq3_0 * blk) {
    static const float C[8] = {-2.1519f, -1.3439f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3439f, 2.1519f};
    (void) C;

    float sq = 0.0f;
    for (int i = 0; i < QK; ++i) sq += x[i] * x[i];
    float rms = std::sqrt(sq / QK);
    if (rms < 1e-10f) rms = 1.0f;

    blk->d = __float2half(rms);

    float buf[QK];
    for (int i = 0; i < QK; ++i) buf[i] = x[i] / rms * cpu_sign(i);
    for (int s = 1; s < QK; s <<= 1) {
        for (int i = 0; i < QK; i += s * 2) {
            for (int j = i; j < i + s; ++j) {
                const float a = buf[j];
                const float b = buf[j + s];
                buf[j] = a + b;
                buf[j + s] = a - b;
            }
        }
    }
    for (int i = 0; i < QK; ++i) buf[i] /= std::sqrt(32.0f);

    uint8_t idx[QK];
    for (int i = 0; i < QK; ++i) {
        idx[i] = 0;
        for (int b = 0; b < 7; ++b) if (buf[i] > B[b]) idx[i] = b + 1;
    }

    for (int g = 0; g < 4; ++g) {
        uint8_t * q = blk->qs + g * 3;
        uint8_t * d = idx + g * 8;
        q[0] = d[0] | (d[1] << 3) | (d[2] << 6);
        q[1] = (d[2] >> 2) | (d[3] << 1) | (d[4] << 4) | (d[5] << 7);
        q[2] = (d[5] >> 1) | (d[6] << 2) | (d[7] << 5);
    }
}

__constant__ float GPU_C[8] = {-2.1519f, -1.3439f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3439f, 2.1519f};

__device__ float gpu_sign(int i) {
    return ((((unsigned) i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

__global__ void tq3_to_q8_mmq_tile_kernel(const block_tq3_0 * tq3, uint32_t * out_qs, float * out_df) {
    const int lane = threadIdx.x;
    const int blk = blockIdx.x;

    const block_tq3_0 * bxi = tq3 + blk;
    const int g = lane / 8;
    const int r = lane % 8;
    const int leader = g * 8;
    float rms = 0.0f;
    uint32_t packed = 0;
    if (r == 0) {
        rms = __half2float(bxi->d);
        const uint8_t * qp = bxi->qs + g * 3;
        packed = (uint32_t) qp[0] | ((uint32_t) qp[1] << 8) | ((uint32_t) qp[2] << 16);
    }
    rms = __shfl_sync(0xFFFFFFFF, rms, leader);
    packed = __shfl_sync(0xFFFFFFFF, packed, leader);
    const uint8_t idx = (packed >> (3 * r)) & 7;

    float val = GPU_C[idx];
    #pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        val = (lane & step) ? (other - val) : (other + val);
    }

    const float xf = val / sqrtf(32.0f) * gpu_sign(lane) * rms;
    float a = fabsf(xf);
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, m));
    }

    const float d  = __shfl_sync(0xFFFFFFFF, a / 127.0f, 0);
    const float id = __shfl_sync(0xFFFFFFFF, a > 0.0f ? 127.0f / a : 0.0f, 0);
    const int q = max(-127, min(127, (int) roundf(xf * id)));

    const int slot = lane % QI8_0;
    const int q1 = __shfl_sync(0xFFFFFFFF, q, 4 * slot + 0);
    const int q2 = __shfl_sync(0xFFFFFFFF, q, 4 * slot + 1);
    const int q3 = __shfl_sync(0xFFFFFFFF, q, 4 * slot + 2);
    const int q4 = __shfl_sync(0xFFFFFFFF, q, 4 * slot + 3);
    if (lane < QI8_0) {
        out_qs[blk * QI8_0 + lane] = (uint8_t) q1 | ((uint8_t) q2 << 8) | ((uint8_t) q3 << 16) | ((uint8_t) q4 << 24);
        if (lane == 0) out_df[blk] = d;
    }
}

int main() {
    constexpr int nblocks = 8192;
    constexpr int niters = 1000;

    block_tq3_0 * h_tq3 = new block_tq3_0[nblocks];
    for (int b = 0; b < nblocks; ++b) {
        float x[QK];
        for (int i = 0; i < QK; ++i) {
            x[i] = sinf(0.1f * (b * QK + i)) * cosf(0.013f * (b + i));
        }
        cpu_quantize_tq3(x, &h_tq3[b]);
    }

    block_tq3_0 * d_tq3 = nullptr;
    uint32_t * d_qs = nullptr;
    float * d_df = nullptr;
    cudaMalloc(&d_tq3, nblocks * sizeof(block_tq3_0));
    cudaMalloc(&d_qs, nblocks * QI8_0 * sizeof(uint32_t));
    cudaMalloc(&d_df, nblocks * sizeof(float));
    cudaMemcpy(d_tq3, h_tq3, nblocks * sizeof(block_tq3_0), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 20; ++i) {
        tq3_to_q8_mmq_tile_kernel<<<nblocks, 32>>>(d_tq3, d_qs, d_df);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < niters; ++i) {
        tq3_to_q8_mmq_tile_kernel<<<nblocks, 32>>>(d_tq3, d_qs, d_df);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    const double blocks_per_s = (double) nblocks * niters / (ms / 1000.0);
    const double values_per_s = blocks_per_s * QK;

    printf("tq3_to_q8_mmq_tile_kernel: %.3f ms total, %.3f us/launch\n", ms, 1000.0 * ms / niters);
    printf("throughput: %.3f Mblocks/s, %.3f Gvalues/s\n", blocks_per_s / 1e6, values_per_s / 1e9);

    cudaFree(d_tq3);
    cudaFree(d_qs);
    cudaFree(d_df);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_tq3;
    return 0;
}
