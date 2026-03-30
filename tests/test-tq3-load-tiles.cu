// Unit test: TQ3_0 -> exact q8_0 MMQ tile layout
// Verifies the real load_tiles_tq3_0 contract for one DP4A tile row:
// 4 TQ3_0 blocks -> packed q8_0 ints in x_qs + exact scales in x_df.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define QK 32
#define QI8_0 8
#define MMQ_TILE_NE_K 32

typedef struct { __half d; uint8_t qs[12]; } block_tq3_0;

static float cpu_sign(int i) {
    return ((((unsigned) i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

static const float C[8] = {-2.1519f, -1.3439f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3439f, 2.1519f};
static const float B[7] = {-1.7479f, -1.0500f, -0.5005f, 0.0f, 0.5005f, 1.0500f, 1.7479f};

static void cpu_quantize_tq3(const float * x, block_tq3_0 * blk) {
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
        for (int b = 0; b < 7; ++b) {
            if (buf[i] > B[b]) idx[i] = b + 1;
        }
    }

    for (int g = 0; g < 4; ++g) {
        uint8_t * q = blk->qs + g * 3;
        uint8_t * d = idx + g * 8;
        q[0] = d[0] | (d[1] << 3) | (d[2] << 6);
        q[1] = (d[2] >> 2) | (d[3] << 1) | (d[4] << 4) | (d[5] << 7);
        q[2] = (d[5] >> 1) | (d[6] << 2) | (d[7] << 5);
    }
}

static void cpu_dequant_tq3(const block_tq3_0 * blk, float * out) {
    const float rms = __half2float(blk->d);
    float v[QK];

    for (int g = 0; g < 4; ++g) {
        const uint8_t * q = blk->qs + g * 3;
        const int b = g * 8;
        v[b + 0] = C[q[0] & 7];
        v[b + 1] = C[(q[0] >> 3) & 7];
        v[b + 2] = C[((q[0] >> 6) | (q[1] << 2)) & 7];
        v[b + 3] = C[(q[1] >> 1) & 7];
        v[b + 4] = C[(q[1] >> 4) & 7];
        v[b + 5] = C[((q[1] >> 7) | (q[2] << 1)) & 7];
        v[b + 6] = C[(q[2] >> 2) & 7];
        v[b + 7] = C[(q[2] >> 5) & 7];
    }

    for (int s = 1; s < QK; s <<= 1) {
        for (int i = 0; i < QK; i += s * 2) {
            for (int j = i; j < i + s; ++j) {
                const float a = v[j];
                const float b = v[j + s];
                v[j] = a + b;
                v[j + s] = a - b;
            }
        }
    }

    for (int i = 0; i < QK; ++i) out[i] = v[i] / std::sqrt(32.0f) * cpu_sign(i) * rms;
}

static void cpu_quantize_q8_0(const float * x, float * d_out, int8_t * qs) {
    float amax = 0.0f;
    for (int i = 0; i < QK; ++i) amax = fmaxf(amax, fabsf(x[i]));
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    *d_out = d;
    for (int i = 0; i < QK; ++i) qs[i] = (int8_t) roundf(x[i] * id);
}

static uint32_t pack_q8_pair(const int8_t * qs, int i) {
    const int j = 4 * i;
    return (uint8_t) qs[j + 0] | ((uint8_t) qs[j + 1] << 8) | ((uint8_t) qs[j + 2] << 16) | ((uint8_t) qs[j + 3] << 24);
}

__constant__ float GPU_C[8] = {-2.1519f, -1.3439f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3439f, 2.1519f};

__device__ float gpu_sign(int i) {
    return ((((unsigned) i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

// Mirrors the DP4A tile contract used by load_tiles_tq3_0 for one row with 4 TQ3 blocks.
__global__ void tq3_to_q8_mmq_tile_kernel(const block_tq3_0 * tq3, uint32_t * out_qs, float * out_df) {
    const int lane = threadIdx.x; // 0..31
    const int blk = blockIdx.x;   // 0..3

    const block_tq3_0 * bxi = tq3 + blk;
    const float rms = __half2float(bxi->d);

    const int g = lane / 8;
    const int r = lane % 8;
    const uint8_t * qp = bxi->qs + g * 3;
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

    float val = GPU_C[idx];
    for (int step = 1; step < QK; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        val = (lane & step) ? (other - val) : (other + val);
    }

    const float xf = val / sqrtf(32.0f) * gpu_sign(lane) * rms;

    float a = fabsf(xf);
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
    printf("=== TQ3_0 -> q8_0 MMQ tile layout test ===\n\n");

    float inputs[4][QK];
    for (int b = 0; b < 4; ++b) {
        for (int i = 0; i < QK; ++i) {
            inputs[b][i] = sinf(b * 100 + i * 0.3f + 1.0f) * (0.5f + b * 0.3f);
        }
    }

    block_tq3_0 cpu_tq3[4];
    float ref_d[4];
    int8_t ref_qs[4][QK];
    uint32_t ref_tile_qs[4 * QI8_0];
    for (int b = 0; b < 4; ++b) {
        cpu_quantize_tq3(inputs[b], &cpu_tq3[b]);
        float dq[QK];
        cpu_dequant_tq3(&cpu_tq3[b], dq);
        cpu_quantize_q8_0(dq, &ref_d[b], ref_qs[b]);
        for (int i = 0; i < QI8_0; ++i) {
            ref_tile_qs[b * QI8_0 + i] = pack_q8_pair(ref_qs[b], i);
        }
    }

    block_tq3_0 * d_tq3 = nullptr;
    uint32_t * d_tile_qs = nullptr;
    float * d_tile_df = nullptr;
    cudaMalloc(&d_tq3, sizeof(cpu_tq3));
    cudaMalloc(&d_tile_qs, sizeof(ref_tile_qs));
    cudaMalloc(&d_tile_df, sizeof(ref_d));
    cudaMemcpy(d_tq3, cpu_tq3, sizeof(cpu_tq3), cudaMemcpyHostToDevice);

    tq3_to_q8_mmq_tile_kernel<<<4, 32>>>(d_tq3, d_tile_qs, d_tile_df);
    cudaDeviceSynchronize();

    uint32_t gpu_tile_qs[4 * QI8_0];
    float gpu_tile_df[4];
    cudaMemcpy(gpu_tile_qs, d_tile_qs, sizeof(gpu_tile_qs), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_tile_df, d_tile_df, sizeof(gpu_tile_df), cudaMemcpyDeviceToHost);

    int pass = 0;
    int fail = 0;
    for (int b = 0; b < 4; ++b) {
        int q_diff = 0;
        for (int i = 0; i < QI8_0; ++i) {
            if (gpu_tile_qs[b * QI8_0 + i] != ref_tile_qs[b * QI8_0 + i]) {
                ++q_diff;
                if (b == 0) {
                    printf("  slot %d ref=%08x gpu=%08x\n", i, ref_tile_qs[b * QI8_0 + i], gpu_tile_qs[b * QI8_0 + i]);
                }
            }
        }
        const bool d_ok = fabsf(gpu_tile_df[b] - ref_d[b]) < 1e-5f;
        const bool ok = d_ok && q_diff == 0;
        printf("Block %d: d cpu=%.6f gpu=%.6f %s | packed_q mismatches=%d %s\n",
                b, ref_d[b], gpu_tile_df[b], d_ok ? "OK" : "FAIL", q_diff, q_diff == 0 ? "OK" : "FAIL");
        ok ? ++pass : ++fail;
    }

    printf("\n%d passed, %d failed\n", pass, fail);

    cudaFree(d_tq3);
    cudaFree(d_tile_qs);
    cudaFree(d_tile_df);
    return fail > 0 ? 1 : 0;
}
