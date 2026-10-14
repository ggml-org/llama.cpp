#include "moe_predictor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define N_LAYER_MAX 48
#define IN_DIM_DEF 2048
#define H_DIM_DEF 256
#define OUT_DIM_DEF 512
#define K_HIST_DEF 4
#define N_USED_DEF 10

// Fused 31M Router Predictor Kernel (512 threads/block, coalesced warps)
__global__ void moe_predictor_fused_fast512(
    const uint16_t * __restrict__ history_k4, // [4, n_layer, n_used]
    const half     * __restrict__ W1,         // [n_layer, in_dim, h_dim]
    const half     * __restrict__ b1,         // [n_layer, h_dim]
    const half     * __restrict__ W2,         // [n_layer, h_dim, out_dim] (row-major)
    const half     * __restrict__ b2,         // [n_layer, out_dim]
    uint16_t       * __restrict__ out_top16   // [n_layer, 16]
) {
    const int l = blockIdx.x;
    const int j = threadIdx.x; // [0..511]

    __shared__ half  h_shared[H_DIM_DEF];
    __shared__ float s_val[OUT_DIM_DEF];
    __shared__ uint16_t s_idx[OUT_DIM_DEF];

    const half * w1_l = W1 + (size_t) l * IN_DIM_DEF * H_DIM_DEF;
    const half * w2_l = W2 + (size_t) l * H_DIM_DEF * OUT_DIM_DEF;

    // 1. Layer 1: First 256 threads compute Hidden (256)
    if (j < H_DIM_DEF) {
        float sum1 = 0.0f;
        #pragma unroll
        for (int k = 0; k < K_HIST_DEF; ++k) {
            const int hist_offset = (k * N_LAYER_MAX + l) * N_USED_DEF;
            #pragma unroll
            for (int u = 0; u < N_USED_DEF; ++u) {
                uint16_t exp_id = history_k4[hist_offset + u];
                if (exp_id < OUT_DIM_DEF) {
                    int feat_idx = k * OUT_DIM_DEF + exp_id;
                    sum1 += __half2float(w1_l[(size_t) feat_idx * H_DIM_DEF + j]);
                }
            }
        }
        sum1 += __half2float(b1[l * H_DIM_DEF + j]);
        h_shared[j] = __float2half(fmaxf(sum1, 0.0f));
    }
    __syncthreads();

    // 2. Layer 2: All 512 threads compute 1 output logit each
    float sum2 = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < H_DIM_DEF; ++k) {
        sum2 += __half2float(h_shared[k]) * __half2float(w2_l[k * OUT_DIM_DEF + j]);
    }
    sum2 += __half2float(b2[l * OUT_DIM_DEF + j]);

    s_val[j] = sum2;
    s_idx[j] = (uint16_t) j;
    __syncthreads();

    // 3. Fast Parallel Top-16 Selection
    #pragma unroll
    for (int sel = 0; sel < 16; ++sel) {
        if (j < 256) {
            if (s_val[j + 256] > s_val[j]) {
                s_val[j] = s_val[j + 256];
                s_idx[j] = s_idx[j + 256];
            }
        }
        __syncthreads();

        if (j < 128) {
            if (s_val[j + 128] > s_val[j]) {
                s_val[j] = s_val[j + 128];
                s_idx[j] = s_idx[j + 128];
            }
        }
        __syncthreads();

        if (j < 64) {
            if (s_val[j + 64] > s_val[j]) {
                s_val[j] = s_val[j + 64];
                s_idx[j] = s_idx[j + 64];
            }
        }
        __syncthreads();

        if (j < 32) {
            if (s_val[j + 32] > s_val[j]) {
                s_val[j] = s_val[j + 32];
                s_idx[j] = s_idx[j + 32];
            }
            float v = s_val[j];
            uint16_t idx = s_idx[j];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other_v = __shfl_down_sync(0xffffffff, v, offset);
                uint16_t other_i = (uint16_t)__shfl_down_sync(0xffffffff, (unsigned)idx, offset);
                if (other_v > v) {
                    v = other_v;
                    idx = other_i;
                }
            }
            if (j == 0) {
                out_top16[l * 16 + sel] = idx;
                s_val[idx] = -1e9f; // Mask out selected element
            }
        }
        __syncthreads();
    }
}

// Global predictor state
static struct {
    bool ready = false;
    int n_layer = 48;
    int in_dim = 2048;
    int h_dim = 256;
    int out_dim = 512;

    half * d_W1 = nullptr;
    half * d_b1 = nullptr;
    half * d_W2 = nullptr;
    half * d_b2 = nullptr;

    uint16_t * d_hist = nullptr;
    uint16_t * d_top16 = nullptr;
    uint16_t * h_top16_pinned = nullptr;

    cudaStream_t stream = nullptr;
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;
} g_pred;

bool moe_predictor_init(const char * bin_path, int n_layer, int in_dim, int h_dim, int out_dim) {
    if (g_pred.ready) {
        return true;
    }

    FILE * f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "%s: failed to open %s\n", __func__, bin_path);
        return false;
    }

    g_pred.n_layer = n_layer;
    g_pred.in_dim = in_dim;
    g_pred.h_dim = h_dim;
    g_pred.out_dim = out_dim;

    size_t w1_bytes = (size_t) n_layer * in_dim * h_dim * sizeof(half);
    size_t b1_bytes = (size_t) n_layer * h_dim * sizeof(half);
    size_t w2_bytes = (size_t) n_layer * h_dim * out_dim * sizeof(half);
    size_t b2_bytes = (size_t) n_layer * out_dim * sizeof(half);
    size_t total_w_bytes = w1_bytes + b1_bytes + w2_bytes + b2_bytes;

    std::vector<char> buf(total_w_bytes);
    size_t rd = fread(buf.data(), 1, total_w_bytes, f);
    fclose(f);

    if (rd != total_w_bytes) {
        fprintf(stderr, "%s: weight size mismatch, got %zu, expected %zu\n", __func__, rd, total_w_bytes);
        return false;
    }

    cudaError_t err;
    err = cudaMalloc(&g_pred.d_W1, w1_bytes);
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&g_pred.d_b1, b1_bytes);
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&g_pred.d_W2, w2_bytes);
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&g_pred.d_b2, b2_bytes);
    if (err != cudaSuccess) return false;

    size_t hist_bytes = (size_t) K_HIST_DEF * n_layer * N_USED_DEF * sizeof(uint16_t);
    size_t top16_bytes = (size_t) n_layer * 16 * sizeof(uint16_t);

    err = cudaMalloc(&g_pred.d_hist, hist_bytes);
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&g_pred.d_top16, top16_bytes);
    if (err != cudaSuccess) return false;
    err = cudaHostAlloc(&g_pred.h_top16_pinned, top16_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) return false;

    err = cudaStreamCreateWithFlags(&g_pred.stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) return false;
    cudaEventCreate(&g_pred.start_evt);
    cudaEventCreate(&g_pred.stop_evt);

    // Upload weights to GPU
    char * p = buf.data();
    cudaMemcpyAsync(g_pred.d_W1, p, w1_bytes, cudaMemcpyHostToDevice, g_pred.stream); p += w1_bytes;
    cudaMemcpyAsync(g_pred.d_b1, p, b1_bytes, cudaMemcpyHostToDevice, g_pred.stream); p += b1_bytes;
    cudaMemcpyAsync(g_pred.d_W2, p, w2_bytes, cudaMemcpyHostToDevice, g_pred.stream); p += w2_bytes;
    cudaMemcpyAsync(g_pred.d_b2, p, b2_bytes, cudaMemcpyHostToDevice, g_pred.stream);
    cudaStreamSynchronize(g_pred.stream);

    g_pred.ready = true;
    fprintf(stderr, "%s: 31M router predictor initialized on GPU (%.2f MB VRAM)\n",
            __func__, (total_w_bytes + hist_bytes + top16_bytes) / (1024.0 * 1024.0));
    return true;
}

void moe_predictor_free() {
    if (!g_pred.ready) return;
    if (g_pred.d_W1) cudaFree(g_pred.d_W1);
    if (g_pred.d_b1) cudaFree(g_pred.d_b1);
    if (g_pred.d_W2) cudaFree(g_pred.d_W2);
    if (g_pred.d_b2) cudaFree(g_pred.d_b2);
    if (g_pred.d_hist) cudaFree(g_pred.d_hist);
    if (g_pred.d_top16) cudaFree(g_pred.d_top16);
    if (g_pred.h_top16_pinned) cudaFreeHost(g_pred.h_top16_pinned);
    if (g_pred.stream) cudaStreamDestroy(g_pred.stream);
    if (g_pred.start_evt) cudaEventDestroy(g_pred.start_evt);
    if (g_pred.stop_evt) cudaEventDestroy(g_pred.stop_evt);
    memset(&g_pred, 0, sizeof(g_pred));
}

bool moe_predictor_is_ready() {
    return g_pred.ready;
}

float moe_predictor_predict(const uint16_t * history_k4, uint16_t * out_top16) {
    if (!g_pred.ready || !history_k4 || !out_top16) {
        return 0.0f;
    }

    size_t hist_bytes = (size_t) K_HIST_DEF * g_pred.n_layer * N_USED_DEF * sizeof(uint16_t);
    size_t top16_bytes = (size_t) g_pred.n_layer * 16 * sizeof(uint16_t);

    // Upload history (3,840 bytes) to device stream
    cudaMemcpyAsync(g_pred.d_hist, history_k4, hist_bytes, cudaMemcpyHostToDevice, g_pred.stream);

    cudaEventRecord(g_pred.start_evt, g_pred.stream);
    moe_predictor_fused_fast512<<<g_pred.n_layer, OUT_DIM_DEF, 0, g_pred.stream>>>(
        g_pred.d_hist, g_pred.d_W1, g_pred.d_b1, g_pred.d_W2, g_pred.d_b2, g_pred.d_top16
    );
    cudaEventRecord(g_pred.stop_evt, g_pred.stream);

    // Download top16 (1,536 bytes) to pinned host buffer, then copy to out
    cudaMemcpyAsync(g_pred.h_top16_pinned, g_pred.d_top16, top16_bytes, cudaMemcpyDeviceToHost, g_pred.stream);
    cudaStreamSynchronize(g_pred.stream);

    memcpy(out_top16, g_pred.h_top16_pinned, top16_bytes);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, g_pred.start_evt, g_pred.stop_evt);
    return elapsed_ms;
}
