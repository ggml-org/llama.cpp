#pragma once
#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * x, T * y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<nv_bfloat16> to_bf16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

// TODO more general support for non-contiguous inputs

template<typename T>
using to_t_nc_cuda_t = void (*)(const void * x, T * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);

struct ue4m3 {
    uint8_t x;
};

template<typename dst_t, typename src_t>
 __host__ __device__ inline dst_t ggml_cuda_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;
    } else if constexpr(std::is_same_v<dst_t, nv_bfloat16>) {
        return __float2bfloat16(float(x));
    } else if constexpr(std::is_same_v<src_t, nv_bfloat16>) {
        return __bfloat162float(x);
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, half2>) {
        return __float22half2_rn(x);
    } else if constexpr(std::is_same_v<src_t, nv_bfloat162> && std::is_same_v<dst_t, float2>) {
#ifdef GGML_USE_HIP
        return make_float2(__bfloat162float(__low2bfloat16(x)), __bfloat162float(__high2bfloat16(x)));
#else
#if __CUDA_ARCH__ >= 800
        return __bfloat1622float2(x);
#else
        return make_float2(__bfloat162float(x.x), __bfloat162float(x.y));
#endif // __CUDA_ARCH__ >= 800
#endif // GGML_USE_HIP
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, nv_bfloat162>) {
        // bypass compile error on cuda 12.0.1
#ifdef GGML_USE_HIP
        return __float22bfloat162_rn(x);
#else
        return {x.x, x.y};
#endif // GGML_USE_HIP
    } else if constexpr (std::is_same_v<src_t, ue4m3>) {
#if defined(__CUDA_ARCH__)
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12050 // This matches cuda_fp8.h's version gate.
        // This uses the same fp8 conversion that __nv_fp8_e4m3 uses internally.
        __half h = __half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) x.x, __NV_E4M3));
        unsigned short hb = __half_as_ushort(h) - 0x0400u; // Built in 0.5f op.
        float f = __half2float(__ushort_as_half(hb));

        // __nv_fp8_e4m3 is signed but UE4M3
        if (x.x == 0u || x.x == 0x7Fu) { // force 0x7F and 0x00 to 0.0f
            f = 0.0f;
        }

        return f;
#else
        if (x.x == 0u || x.x == 0x7Fu) {
            return 0.0f;
        }

        const uint32_t exp  = x.x >> 3;
        const uint32_t mant = x.x & 0x7u;
        uint32_t bits;

        if (exp != 0u) {
            bits = ((exp + 119u) << 23) | (mant << 20);
        } else {
            uint32_t p;
            if (mant < 2u) {
                p = 0u;
            } else if (mant < 4u) {
                p = 1u;
            } else {
                p = 2u;
            }

            const uint32_t r = mant - (1u << p);
            const uint32_t exp32  = p + 117u;
            const uint32_t mant32 = r << (23u - p);
            bits = (exp32 << 23) | mant32;
        }

        float f;
        memcpy(&f, &bits, sizeof(f)); // 0.5f is baked in.
        return f;
#endif
#else
        return ggml_ue4m3_to_fp32(x.x);
#endif
    } else if constexpr(std::is_same_v<dst_t, int32_t>) {
        return int32_t(x);
    } else {
        return float(x);
    }
}
