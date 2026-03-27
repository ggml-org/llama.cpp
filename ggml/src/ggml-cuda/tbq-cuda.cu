#include "common.cuh"
#include "../ggml-common.h"
#include "../../include/ggml-turbo-quant.h"

/* Device global symbols (declared extern in tbq-cuda.cuh) */
__device__ float* g_tbq_rotation = nullptr;
__device__ float* g_tbq_rotation_t = nullptr;

/* Host: set device pointers */
void tbq_cuda_set_rotation(float* d_rotation, float* d_rotation_t) {
    cudaMemcpyToSymbol(g_tbq_rotation, &d_rotation, sizeof(float*));
    cudaMemcpyToSymbol(g_tbq_rotation_t, &d_rotation_t, sizeof(float*));
}

/* Lazy GPU init: upload rotation matrix on first use */
static float* s_d_rotation = nullptr;
static float* s_d_rotation_t = nullptr;
static half*  s_d_rotation_t_fp16 = nullptr;  /* FP16 version for optimized dequant */

half* tbq_cuda_get_rotation_t_fp16(void) { return s_d_rotation_t_fp16; }

/* CUDA kernel: convert FP32 rotation to FP16 on device */
static __global__ void k_f32_to_f16(const float* __restrict__ src, half* __restrict__ dst, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void tbq_cuda_ensure_rotation(void) {
    if (s_d_rotation) return; /* already uploaded */

    turbo_quant_ctx* ctx = turbo_quant_get_ctx();
    if (!ctx || !ctx->rotation) return;

    int dim = ctx->dim;
    int n = dim * dim;
    size_t bytes_f32 = (size_t)n * sizeof(float);
    size_t bytes_f16 = (size_t)n * sizeof(half);

    cudaMalloc(&s_d_rotation, bytes_f32);
    cudaMalloc(&s_d_rotation_t, bytes_f32);
    cudaMalloc(&s_d_rotation_t_fp16, bytes_f16);

    cudaMemcpy(s_d_rotation, ctx->rotation, bytes_f32, cudaMemcpyHostToDevice);
    cudaMemcpy(s_d_rotation_t, ctx->rotation_t, bytes_f32, cudaMemcpyHostToDevice);
    tbq_cuda_set_rotation(s_d_rotation, s_d_rotation_t);

    /* Convert rotation_t to FP16 on device */
    k_f32_to_f16<<<(n + 255) / 256, 256>>>(s_d_rotation_t, s_d_rotation_t_fp16, n);
    cudaDeviceSynchronize();

    GGML_LOG_INFO("TurboQuant: rotation uploaded to GPU (FP32: %zu KB + FP16: %zu KB)\n",
            bytes_f32 / 1024, bytes_f16 / 1024);
}
