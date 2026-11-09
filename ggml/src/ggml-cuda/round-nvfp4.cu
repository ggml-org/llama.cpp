#include "round-nvfp4.cuh"

#if CUDART_VERSION >= 12080 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && \
        defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#define GGML_CUDA_ROUND_NVFP4_NATIVE_AVAILABLE
#endif

// from Ada Lovelace on, cuda_fp8.h converts to and from E4M3 with the cvt instructions of the GPU
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE
#define GGML_CUDA_ROUND_E4M3_NATIVE_AVAILABLE
#endif

// nearest E4M3 value of x > 0, ties to even, saturated at 448
static __device__ __forceinline__ float round_e4m3(float x) {
#if defined(GGML_CUDA_ROUND_E4M3_NATIVE_AVAILABLE)
    return static_cast<float>(__nv_fp8_e4m3(x));
#else
    int e;
    frexpf(x, &e);
    // spacing of the 3 mantissa bits, subnormals below 2^-6 share the spacing 2^-9
    const int s = max(e - 4, -9);
    return fminf(ldexpf(rintf(ldexpf(x, -s)), s), 448.0f);
#endif
}

#if !defined(GGML_CUDA_ROUND_NVFP4_NATIVE_AVAILABLE)
// nearest E2M1 value of x, ties to even, saturated at 6
static __device__ __forceinline__ float round_e2m1(float x) {
    const float ax = fabsf(x);
    const float r = ax <= 0.25f ? 0.0f : ax <  0.75f ? 0.5f : ax <= 1.25f ? 1.0f : ax <  1.75f ? 1.5f :
                    ax <= 2.5f  ? 2.0f : ax <  3.5f  ? 3.0f : ax <= 5.0f  ? 4.0f : 6.0f;
    return copysignf(r, x);
}
#endif

// one thread per group of 16 values
static __global__ void round_nvfp4_kernel(
        const char * src, char * dst, const int64_t n_groups_row, const int64_t ne1, const int64_t ne2, const int64_t n_groups,
        const size_t nb01, const size_t nb02, const size_t nb03, const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t ig = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (ig >= n_groups) {
        return;
    }

    const int64_t i0 = (ig % n_groups_row) * 16;
    const int64_t ir = ig / n_groups_row;
    const int64_t i1 = ir % ne1;
    const int64_t i2 = (ir / ne1) % ne2;
    const int64_t i3 = ir / (ne1*ne2);

    const float * x = (const float *) (src + i1*nb01 + i2*nb02 + i3*nb03) + i0;
    float       * y = (float       *) (dst + i1*nb1  + i2*nb2  + i3*nb3)  + i0;

    float v[16];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        v[j] = x[j];
        amax = fmaxf(amax, fabsf(v[j]));
    }

    // IEEE divisions: fast math would move values across the rounding ties
    const float scale = round_e4m3(fmaxf(__fdiv_rn(amax, 6.0f), 0.001953125f));
#if defined(GGML_CUDA_ROUND_NVFP4_NATIVE_AVAILABLE)
#pragma unroll
    for (int j = 0; j < 16; j += 4) {
        const __nv_fp4x4_e2m1 q(make_float4(
            __fdiv_rn(v[j + 0], scale),
            __fdiv_rn(v[j + 1], scale),
            __fdiv_rn(v[j + 2], scale),
            __fdiv_rn(v[j + 3], scale)));
        const __nv_fp4x2_storage_t q_lo = static_cast<__nv_fp4x2_storage_t>(q.__x);
        const __nv_fp4x2_storage_t q_hi = static_cast<__nv_fp4x2_storage_t>(q.__x >> 8U);
        const float2 y_lo = __half22float2(static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(q_lo, __NV_E2M1)));
        const float2 y_hi = __half22float2(static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(q_hi, __NV_E2M1)));
        y[j + 0] = copysignf(y_lo.x * scale, v[j + 0]);
        y[j + 1] = copysignf(y_lo.y * scale, v[j + 1]);
        y[j + 2] = copysignf(y_hi.x * scale, v[j + 2]);
        y[j + 3] = copysignf(y_hi.y * scale, v[j + 3]);
    }
#else
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        y[j] = round_e2m1(__fdiv_rn(v[j], scale)) * scale;
    }
#endif
}

#undef GGML_CUDA_ROUND_NVFP4_NATIVE_AVAILABLE
#undef GGML_CUDA_ROUND_E4M3_NATIVE_AVAILABLE

void ggml_cuda_op_round_nvfp4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));
    GGML_ASSERT(src0->ne[0] % 16 == 0);

    const int64_t n_groups_row = src0->ne[0] / 16;
    const int64_t n_groups     = n_groups_row * ggml_nrows(src0);
    if (n_groups == 0) {
        return;
    }

    const int block_size = 256;
    const int64_t n_blocks = (n_groups + block_size - 1) / block_size;

    round_nvfp4_kernel<<<n_blocks, block_size, 0, ctx.stream()>>>(
        (const char *) src0->data, (char *) dst->data, n_groups_row, src0->ne[1], src0->ne[2], n_groups,
        src0->nb[1], src0->nb[2], src0->nb[3], dst->nb[1], dst->nb[2], dst->nb[3]);
}
