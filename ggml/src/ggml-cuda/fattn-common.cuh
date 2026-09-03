#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

struct ggml_cuda_flash_attn_ext_f16_extra_data {
    uintptr_t K;
    uintptr_t V;
    uintptr_t end;
    int64_t   kv_per_chunk; // 0 == no chunking
};

// Max size in bytes for the scratch buffers holding the K/V data converted to f16.
// If exceeded, it is consumed in chunks (serial split-KV, see the chunk loop in launch_fattn).
static int64_t ggml_cuda_flash_attn_ext_convert_bytes_max() {
    static const int64_t convert_bytes_max = []() {
        const char * env = getenv("GGML_CUDA_FATTN_CONVERT_BYTES");
        return env ? std::max<int64_t>(atoll(env), 0) : 64ll*1024*1024;
    }();
    return convert_bytes_max;
}

// KV cache types for which scratch buffer chunking is enabled.
// KV quantized q8_0 and below is where VRAM saving matters
static bool ggml_cuda_fattn_type_chunk_enabled(ggml_type type) {
    return ggml_is_quantized(type);
}

static int64_t ggml_cuda_flash_attn_ext_kv_per_chunk(
        const ggml_tensor * K, const ggml_tensor * V,
        const bool convert_K, const bool convert_V, const bool V_is_K_view, const bool kernel_supports_chunking) {
    if (!kernel_supports_chunking) {
        return 0;
    }

    if ((convert_K && !ggml_cuda_fattn_type_chunk_enabled(K->type)) ||
        (convert_V && !ggml_cuda_fattn_type_chunk_enabled(V->type))) {
        return 0;
    }

    int64_t bytes_per_kv = 0; // f16 bytes per KV cache position, summed over the converted tensors.
    if (convert_K) {
        bytes_per_kv += K->ne[0]*K->ne[2]*K->ne[3]*ggml_type_size(GGML_TYPE_F16);
    }
    if (convert_V && !V_is_K_view) {
        bytes_per_kv += V->ne[0]*V->ne[2]*V->ne[3]*ggml_type_size(GGML_TYPE_F16);
    }

    const int64_t convert_bytes_max = ggml_cuda_flash_attn_ext_convert_bytes_max();
    if (bytes_per_kv*K->ne[1] <= convert_bytes_max) {
        return 0;
    }

    const int64_t rows_max = std::max((int64_t) FATTN_KQ_STRIDE, (convert_bytes_max/bytes_per_kv/FATTN_KQ_STRIDE)*FATTN_KQ_STRIDE);
    return rows_max < K->ne[1] ? rows_max : 0;
}

static inline ggml_cuda_flash_attn_ext_f16_extra_data ggml_cuda_flash_attn_ext_get_f16_extra_data(
        const ggml_tensor * dst, const bool need_f16_K, const bool need_f16_V, const bool kernel_supports_chunking) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const bool convert_K = need_f16_K && K->type != GGML_TYPE_F16;
    const bool convert_V = need_f16_V && V->type != GGML_TYPE_F16;

    ggml_cuda_flash_attn_ext_f16_extra_data data = {};
    data.end          = (uintptr_t) dst->data + ggml_nbytes(dst);
    data.kv_per_chunk = ggml_cuda_flash_attn_ext_kv_per_chunk(K, V, convert_K, convert_V, V_is_K_view, kernel_supports_chunking);

    if (convert_K) {
        data.end = GGML_PAD(data.end, 128);
        data.K   = data.end;
        const int64_t ne1 = data.kv_per_chunk > 0 ? data.kv_per_chunk : K->ne[1];
        data.end += K->ne[0]*ne1*K->ne[2]*K->ne[3]*ggml_type_size(GGML_TYPE_F16);
    }

    if (convert_V) {
        if (V_is_K_view) {
            data.V = data.K;
        } else {
            data.end = GGML_PAD(data.end, 128);
            data.V   = data.end;
            const int64_t ne1 = data.kv_per_chunk > 0 ? data.kv_per_chunk : V->ne[1];
            data.end += V->ne[0]*ne1*V->ne[2]*V->ne[3]*ggml_type_size(GGML_TYPE_F16);
        }
    }

    return data;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * mask_ptr, int * KV_max_ptr, const int ne30, const int64_t s31, const int64_t s33) {
    const half2 * GGML_CUDA_RESTRICT mask   = mask_ptr;
    int         * GGML_CUDA_RESTRICT KV_max = KV_max_ptr;

    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    ggml_cuda_pdl_sync();
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        float2 * dst_meta_partial_ptr,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;
    ggml_cuda_pdl_lc();
    float        * GGML_CUDA_RESTRICT dst              = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup        = dst_fixup_ptr;
    float2       * GGML_CUDA_RESTRICT dst_meta_partial = dst_meta_partial_ptr; // != nullptr when the result is an unnormalized partial.

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    ggml_cuda_pdl_sync();
    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    if (dst_meta_partial == nullptr) {
        *dst = dst_val / rowsum;
        return;
    }

    // Unnormalized partial with (max, sum) per output row, for flash_attn_combine_results
    *dst = dst_val;
    if (tid == 0) {
        dst_meta_partial[(sequence*ne01 + jt*ncols1 + j)*ne02 + zt_Q + c] = make_float2(max_val, rowsum);
    }
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        float2 * dst_meta_partial_ptr,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    float        * GGML_CUDA_RESTRICT dst              = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup        = dst_fixup_ptr;
    float2       * GGML_CUDA_RESTRICT dst_meta_partial = dst_meta_partial_ptr;
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    ggml_cuda_pdl_sync();
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (dst_meta_partial == nullptr) {
        *dst = dst_val / rowsum;
        return;
    }

    // Unnormalized partial with (max, sum) per output row, for flash_attn_combine_results
    *dst = dst_val;
    if (tid == 0) {
        dst_meta_partial[(sequence*ne01 + jt*ncols1 + j)*ne02 + zt_Q + c] = make_float2(max_val, rowsum);
    }
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * VKQ_parts_ptr,
        const float2 * VKQ_meta_ptr,
        float        * dst_ptr,
        float        * acc_O_ptr,
        float2       * acc_meta_ptr,
        const float  * sinks_ptr,
        const int      parallel_blocks,
        const bool     first) {
    ggml_cuda_pdl_lc();
    const float  * GGML_CUDA_RESTRICT VKQ_parts = VKQ_parts_ptr;
    const float2 * GGML_CUDA_RESTRICT VKQ_meta  = VKQ_meta_ptr;
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    float        * GGML_CUDA_RESTRICT acc_O     = acc_O_ptr;
    float2       * GGML_CUDA_RESTRICT acc_meta  = acc_meta_ptr;
    const float  * GGML_CUDA_RESTRICT sinks     = sinks_ptr;
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    // Seed with the sink logit, applied exactly once - chunks are launched without sinks
    const float2 acc = first ? make_float2(sinks ? sinks[head] : -FLT_MAX/2.0f, sinks ? 1.0f : 0.0f) : acc_meta[j_dst_unrolled];

    float kqmax = acc.x;
    for (int l = 0; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    const float acc_scale = expf(acc.x - kqmax);

    float VKQ_numerator   = first ? 0.0f : acc_scale * acc_O[j_dst_unrolled*D + tid];
    float VKQ_denominator = acc_scale * acc.y;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    if (dst) {
        dst[j_dst_unrolled*D + tid] = VKQ_numerator / VKQ_denominator;
        return;
    }

    acc_O[j_dst_unrolled*D + tid] = VKQ_numerator;
    if (tid == 0) {
        acc_meta[j_dst_unrolled] = make_float2(kqmax, VKQ_denominator);
    }
}

template<int D> // D == head size
static void launch_flash_attn_combine_results(
        const float * VKQ_parts, const float2 * VKQ_meta, float * dst,
        float * acc_O, float2 * acc_meta, const float * sinks, const int parallel_blocks, const bool first,
        const int ne01, const int ne02, const int ne03, cudaStream_t stream) {
    const dim3 block_dim_combine(D, 1, 1);
    const dim3 blocks_num_combine(ne01, ne02, ne03);
    const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, stream);
    ggml_cuda_kernel_launch(flash_attn_combine_results<D>, launch_params,
        VKQ_parts, VKQ_meta, dst, acc_O, acc_meta, sinks, parallel_blocks, first);
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool kernel_supports_chunking,
    const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(KQV, need_f16_K, need_f16_V, kernel_supports_chunking);

    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    // Buffers for the running accumulator if the KV cache is consumed in chunks
    ggml_cuda_pool_alloc<float>  acc_O(pool);
    ggml_cuda_pool_alloc<float2> acc_meta(pool);

    const int64_t kv_per_chunk = f16_extra.kv_per_chunk;
    const bool    chunked      = kv_per_chunk > 0;
    const int     n_chunks     = chunked ? (K->ne[1] + kv_per_chunk - 1) / kv_per_chunk : 1;

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (!chunked && mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, launch_params,
            (const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);

    auto calc_parallel_blocks = [&](const int64_t kv_len) {
        // parallel_blocks must not be larger than what the tensor size allows:
        const int ntiles_KV = (kv_len + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.
        int parallel_blocks = std::min(max_blocks_per_sm, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        return parallel_blocks;
    };

    if (chunked) {
        // Serial split-KV: the KV dimension is split into chunks that are converted and consumed
        // one after another, analogous to serial split-K for GEMM but across kernel launches.
        // Chunks are launched without normalization and with (max, sum) per output row,
        // using the same layout as the buffers of flash_attn_combine_results.
        // flash_attn_combine_results folds each chunk into the running accumulator acc_O/acc_meta.
        int parallel_blocks_max = 1;
        if (!stream_k) {
            const int64_t kv_len_last = K->ne[1] - (n_chunks - 1)*kv_per_chunk;
            // Chunks always launch as split-K (>= 2 blocks), see the chunk loop.
            parallel_blocks_max = std::max(2, std::max(calc_parallel_blocks(kv_per_chunk), calc_parallel_blocks(kv_len_last)));
        }
        dst_tmp.alloc(parallel_blocks_max*ggml_nelements(KQV));
        if (stream_k) {
            // The per-row meta data goes to the start of the buffer, the stream-k fixup data behind it.
            dst_tmp_meta.alloc(ggml_nrows(KQV) + size_t(max_blocks_per_sm)*nsm * ncols * (2 + DV/2));
        } else {
            dst_tmp_meta.alloc(parallel_blocks_max*ggml_nrows(KQV));
        }
        acc_O.alloc(ggml_nelements(KQV));
        acc_meta.alloc(ggml_nrows(KQV));
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int64_t kv_off = chunk*kv_per_chunk; // == 0 if not chunked
        const int64_t kv_len = chunked ? std::min(kv_per_chunk, K->ne[1] - kv_off) : K->ne[1];

        const char * K_data = (const char *) K->data;
        size_t nb11 = K->nb[1];
        size_t nb12 = K->nb[2];
        size_t nb13 = K->nb[3];

        const char * V_data = (const char *) V->data;
        size_t nb21 = V->nb[1];
        size_t nb22 = V->nb[2];
        size_t nb23 = V->nb[3];

        // Convert [kv_off, kv_off + kv_len) of a K/V tensor to f16, updating the data pointer and strides
        const auto convert_to_f16 = [&](const ggml_tensor * t, const uintptr_t buf, const char * & data, size_t & nb1, size_t & nb2, size_t & nb3) {
            const size_t bs = ggml_blck_size(t->type);
            const size_t ts = ggml_type_size(t->type);

            GGML_ASSERT(buf != 0);
            half * f16 = (half *) buf;
            // A row-chunk of a contiguously allocated tensor is itself a contiguous slice
            // if the row dimension is outermost in memory; convert it with the (much faster)
            // contiguous kernel instead of the generic strided one.
            const bool cont_full  = !chunked && ggml_is_contiguously_allocated(t);
            const bool cont_slice =  chunked && ggml_is_contiguously_allocated(t) && t->ne[3] == 1 && nb1 >= nb2;
            if (cont_full || cont_slice) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(t->type);
                to_fp16(data + kv_off*nb1, f16, cont_slice ? t->ne[0]*t->ne[2]*kv_len : ggml_nelements(t), main_stream);

                nb1 = nb1*bs*sizeof(half)/ts;
                nb2 = nb2*bs*sizeof(half)/ts;
                nb3 = nb3*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(t->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(t->type);
                const int64_t s01 = nb1 / ts;
                const int64_t s02 = nb2 / ts;
                const int64_t s03 = nb3 / ts;
                to_fp16(data + kv_off*nb1, f16, t->ne[0], kv_len, t->ne[2], t->ne[3], s01, s02, s03, main_stream);

                nb1 = t->ne[0] * sizeof(half);
                nb2 = kv_len   * nb1;
                nb3 = t->ne[2] * nb2;
            }
            data = (const char *) f16;
        };

        if (need_f16_K && K->type != GGML_TYPE_F16) {
            convert_to_f16(K, f16_extra.K, K_data, nb11, nb12, nb13);
        } else if (kv_off > 0) {
            K_data += kv_off*nb11;
        }

        if (need_f16_V && V->type != GGML_TYPE_F16) {
            if (V_is_K_view) {
                V_data = K_data;
                nb21   = nb11;
                nb22   = nb12;
                nb23   = nb13;
            } else {
                convert_to_f16(V, f16_extra.V, V_data, nb21, nb22, nb23);
            }
        } else if (kv_off > 0) {
            V_data += kv_off*nb21;
        }

        dim3 blocks_num;
        int parallel_blocks = 1;
        if (stream_k) {
            // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
            const int max_blocks = max_blocks_per_sm*nsm;
            const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
            const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

            const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

            blocks_num.x = ntiles_dst;
            blocks_num.y = 1;
            blocks_num.z = 1;

            if(use_stream_k) {
                const int ntiles_KV = (kv_len + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.
                const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
                // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
                // Only do this if the occupancy loss from rounding is acceptable.
                const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
                const int max_efficiency_loss_percent = 5;
                const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                    ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                    : 100;
                const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                    ? nblocks_stream_k_rounded
                    : nblocks_stream_k_raw;

                blocks_num.x = nblocks_stream_k;
            }

            if (!chunked && ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
                dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
            }
        } else {
            parallel_blocks = calc_parallel_blocks(kv_len);
            if (chunked) {
                // The tile kernel emits an unnormalized partial with (max, sum) per row iff gridDim.y != 1, so chunks
                // always launch as split-K. A block without KV tiles of its own (short last chunk) contributes an
                // empty partial (max == -FLT_MAX/2, sum == 0) that the combine weighs with 0.
                parallel_blocks = std::max(parallel_blocks, 2);
            }

            blocks_num.x = ntiles_x;
            blocks_num.y = parallel_blocks;
            blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

            if (!chunked && parallel_blocks > 1) {
                dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
                dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
            }
        }

        // Chunks emit an unnormalized partial with per-row (max, sum) at the start of dst_tmp_meta, for stream-k
        // launches the fixup data goes behind it. The kernel receives the base pointer: the write_partial MMA
        // variant derives the fixup area itself and the split-K tile path writes its partials at the base.
        // The fixup kernels receive both pointers explicitly.
        float2 * const dst_meta         = chunked && stream_k ? dst_tmp_meta.ptr + ggml_nrows(KQV) : dst_tmp_meta.ptr;
        float2 * const dst_meta_partial = chunked             ? dst_tmp_meta.ptr                    : nullptr;

        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
        ggml_cuda_kernel_launch(fattn_kernel, launch_params,
            (const char *) Q->data,
            K_data,
            V_data,
            mask  ?            ((const char *) mask->data) + kv_off*sizeof(half) : nullptr,
            sinks && !chunked ? ((const char *) sinks->data)                     : nullptr, // For chunks the sink is applied by the combine instead
            KV_max.ptr,
            chunked || (!stream_k && parallel_blocks > 1) ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K->ne[0], kv_len,   K->ne[2], K->ne[3], nb11, nb12, nb13,
            nb21, nb22, nb23,
            mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
            mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0
        );
        CUDA_CHECK(cudaGetLastError());

        if (stream_k) {
            // For chunks the fixup emits an unnormalized partial into the chunk buffers instead of KQV
            float * const fixup_dst = chunked ? dst_tmp.ptr : (float *) KQV->data;

            if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
                // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
                const int nblocks_sk  = (int)blocks_num.x;
                const int bpt         = nblocks_sk / ntiles_dst;

                const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
                const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
                const uint3 fd2 = init_fastdiv_values(ntiles_x);

                const dim3 block_dim_combine(DV, 1, 1);
                const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
                ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, launch_params,
                    fixup_dst, dst_meta, dst_meta_partial,
                     Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                     gqa_ratio, bpt, fd0, fd1, fd2);
            } else if (ntiles_dst % blocks_num.x != 0) {
                // General fixup for the cases where nblocks_stream_k < ntiles_dst.
                const int ntiles_KV = (kv_len + nbatch_fa - 1) / nbatch_fa;
                const int total_work = ntiles_KV * ntiles_dst;

                const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
                const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
                const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
                const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

                const dim3 block_dim_combine(DV, 1, 1);
                const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
                ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                    fixup_dst, dst_meta, dst_meta_partial,
                     Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                     fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
            }
        } else if (!chunked && parallel_blocks > 1) {
            launch_flash_attn_combine_results<DV>(dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data,
                nullptr, nullptr, nullptr, parallel_blocks, true,
                Q->ne[1], Q->ne[2], Q->ne[3], main_stream);
        }

        if (chunked) {
            launch_flash_attn_combine_results<DV>(dst_tmp.ptr, dst_tmp_meta.ptr,
                chunk == n_chunks - 1 ? (float *) KQV->data : nullptr,
                acc_O.ptr, acc_meta.ptr, sinks ? (const float *) sinks->data : nullptr,
                stream_k ? 1 : parallel_blocks, chunk == 0,
                Q->ne[1], Q->ne[2], Q->ne[3], main_stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}
