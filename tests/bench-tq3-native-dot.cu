// Microbenchmark: compare direct native TQ3_0 x q8_0 block dot against
// a dequantize-then-dot baseline at the single-block contract level.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#define QK_TQ3_0 32
typedef struct { __half d; uint8_t qs[12]; } block_tq3_0;
typedef struct { __half d; int8_t qs[32]; } block_q8_0;

#include "../ggml/src/ggml-cuda/tq3-native.cuh"

static float cpu_sign(int i) {
    return ((((unsigned) i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

static const float CPU_CENTROIDS[8] = {-2.1519f, -1.3439f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3439f, 2.1519f};
static const float CPU_BOUNDARIES[7] = {-1.7479f, -1.0500f, -0.5005f, 0.0f, 0.5005f, 1.0500f, 1.7479f};

__device__ float gpu_sign(int i) {
    return ((((unsigned) i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

__constant__ float GPU_CENTROIDS[8] = {-2.1519f, -1.3439f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3439f, 2.1519f};

static void cpu_wht(float *x, int n) {
    for (int s = 1; s < n; s <<= 1) {
        for (int i = 0; i < n; i += s * 2) {
            for (int j = i; j < i + s; ++j) {
                const float a = x[j];
                const float b = x[j + s];
                x[j] = a + b;
                x[j + s] = a - b;
            }
        }
    }
}

static void cpu_quantize_tq3(const float *x, block_tq3_0 *blk) {
    float sq = 0.0f;
    for (int i = 0; i < QK_TQ3_0; ++i) sq += x[i] * x[i];
    float rms = std::sqrt(sq / QK_TQ3_0);
    if (rms < 1e-10f) rms = 1.0f;
    blk->d = __float2half(rms);

    float buf[QK_TQ3_0];
    for (int i = 0; i < QK_TQ3_0; ++i) buf[i] = x[i] / rms * cpu_sign(i);
    cpu_wht(buf, QK_TQ3_0);
    for (int i = 0; i < QK_TQ3_0; ++i) buf[i] /= std::sqrt(32.0f);

    uint8_t idx[QK_TQ3_0];
    for (int i = 0; i < QK_TQ3_0; ++i) {
        idx[i] = 0;
        for (int b = 0; b < 7; ++b) if (buf[i] > CPU_BOUNDARIES[b]) idx[i] = b + 1;
    }

    for (int g = 0; g < 4; ++g) {
        uint8_t *q = blk->qs + g * 3;
        uint8_t *d = idx + g * 8;
        q[0] = d[0] | (d[1] << 3) | (d[2] << 6);
        q[1] = (d[2] >> 2) | (d[3] << 1) | (d[4] << 4) | (d[5] << 7);
        q[2] = (d[5] >> 1) | (d[6] << 2) | (d[7] << 5);
    }
}

static void cpu_quantize_q8_0(const float *x, block_q8_0 *blk) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[i]));
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    blk->d = __float2half(d);
    for (int i = 0; i < 32; ++i) blk->qs[i] = (int8_t) roundf(x[i] * id);
}

__global__ void k_native_dot(const block_tq3_0 *x, const block_q8_0 *y, float *out) {
    const int blk = blockIdx.x;
    const float sum = vec_dot_tq3_0_q8_0_native_block(x + blk, y + blk);
    if (threadIdx.x == 0) {
        out[blk] = sum;
    }
}

__global__ void k_dequant_dot(const block_tq3_0 *x, const block_q8_0 *y, float *out) {
    const int blk = blockIdx.x;
    const int lane = threadIdx.x;
    const block_tq3_0 *bq = x + blk;
    const block_q8_0  *bq8 = y + blk;

    const int g = lane / 8;
    const int r = lane % 8;
    const uint8_t *qp = bq->qs + g * 3;
    uint8_t idx;
    switch (r) {
        case 0: idx =  qp[0]       & 7; break;
        case 1: idx = (qp[0] >> 3) & 7; break;
        case 2: idx = ((qp[0] >> 6) | (qp[1] << 2)) & 7; break;
        case 3: idx = (qp[1] >> 1) & 7; break;
        case 4: idx = (qp[1] >> 4) & 7; break;
        case 5: idx = ((qp[1] >> 7) | (qp[2] << 1)) & 7; break;
        case 6: idx = (qp[2] >> 2) & 7; break;
        default: idx = (qp[2] >> 5) & 7; break;
    }

    float val = GPU_CENTROIDS[idx];
    for (int step = 1; step < 32; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        val = (lane & step) ? (other - val) : (other + val);
    }

    const float x_dq = val * gpu_sign(lane) * (__half2float(bq->d) / sqrtf(32.0f));
    float contrib = x_dq * ((float) bq8->qs[lane] * __half2float(bq8->d));
    for (int step = 16; step > 0; step >>= 1) {
        contrib += __shfl_xor_sync(0xFFFFFFFF, contrib, step);
    }
    if (lane == 0) {
        out[blk] = contrib;
    }
}

int main() {
    constexpr int nblocks = 16384;
    constexpr int niters = 2000;

    block_tq3_0 *h_tq3 = new block_tq3_0[nblocks];
    block_q8_0 *h_q8 = new block_q8_0[nblocks];
    for (int b = 0; b < nblocks; ++b) {
        float x[32], y[32];
        for (int i = 0; i < 32; ++i) {
            x[i] = sinf(0.017f * (b * 32 + i)) * cosf(0.031f * (b + i));
            y[i] = cosf(0.013f * (b * 32 + i)) * sinf(0.023f * (b + 2 * i));
        }
        cpu_quantize_tq3(x, &h_tq3[b]);
        cpu_quantize_q8_0(y, &h_q8[b]);
    }

    block_tq3_0 *d_tq3 = nullptr;
    block_q8_0 *d_q8 = nullptr;
    float *d_native = nullptr;
    float *d_dequant = nullptr;
    cudaMalloc(&d_tq3, nblocks * sizeof(block_tq3_0));
    cudaMalloc(&d_q8, nblocks * sizeof(block_q8_0));
    cudaMalloc(&d_native, nblocks * sizeof(float));
    cudaMalloc(&d_dequant, nblocks * sizeof(float));
    cudaMemcpy(d_tq3, h_tq3, nblocks * sizeof(block_tq3_0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q8, h_q8, nblocks * sizeof(block_q8_0), cudaMemcpyHostToDevice);

    for (int i = 0; i < 20; ++i) {
        k_native_dot<<<nblocks, 32>>>(d_tq3, d_q8, d_native);
        k_dequant_dot<<<nblocks, 32>>>(d_tq3, d_q8, d_dequant);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < niters; ++i) {
        k_native_dot<<<nblocks, 32>>>(d_tq3, d_q8, d_native);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float native_ms = 0.0f;
    cudaEventElapsedTime(&native_ms, start, stop);

    cudaEventRecord(start);
    for (int i = 0; i < niters; ++i) {
        k_dequant_dot<<<nblocks, 32>>>(d_tq3, d_q8, d_dequant);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float dequant_ms = 0.0f;
    cudaEventElapsedTime(&dequant_ms, start, stop);

    float h_native = 0.0f, h_dequant = 0.0f;
    cudaMemcpy(&h_native, d_native, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_dequant, d_dequant, sizeof(float), cudaMemcpyDeviceToHost);

    printf("native_dot:  %.3f ms total, %.3f us/launch\n", native_ms, 1000.0f * native_ms / niters);
    printf("dequant_dot: %.3f ms total, %.3f us/launch\n", dequant_ms, 1000.0f * dequant_ms / niters);
    printf("speedup:     %.3fx\n", dequant_ms / native_ms);
    printf("sample:      native=%f dequant=%f diff=%e\n", h_native, h_dequant, fabsf(h_native - h_dequant));

    cudaFree(d_tq3);
    cudaFree(d_q8);
    cudaFree(d_native);
    cudaFree(d_dequant);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_tq3;
    delete[] h_q8;
    return 0;
}
