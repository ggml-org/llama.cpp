#include "common.cuh"
#include "fattn-common.cuh"

#ifdef __cplusplus
// Template definition must come before macro usage, and must NOT be in extern "C"
template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_bf16_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_bf16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif

// Device exponential for bfloat16 with clamping to avoid overflow/underflow
#ifndef BF16_MAX
#define BF16_MAX 3.38953139e+38f
#endif
#ifndef BF16_MIN
#define BF16_MIN -3.38953139e+38f
#endif

__device__ __forceinline__ __nv_bfloat16 hexp_bf16(float x) {
    float val = expf(x);
    val = fminf(fmaxf(val, BF16_MIN), BF16_MAX);
    return __float2bfloat16(val);
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
#ifndef GGML_USE_HIP
__launch_bounds__(D, 1)
#endif // GGML_USE_HIP
static __global__ void flash_attn_vec_ext_bf16(
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
        const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ncols > 1) {
        NO_DEVICE_CODE;
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.


    constexpr vec_dot_KQ_bf16_t vec_dot_KQ = get_vec_dot_KQ_bf16<D>(type_K);
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_BF16;
    constexpr dequantize_1_bf16_t dequantize_1_v = get_dequantize_1_bf16(type_V);

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const __nv_bfloat16  * maskh  = (const __nv_bfloat16  *) (mask + nb33*(sequence % ne33) + nb31*ic0);
    const float * sinksf = (const float *) (sinks);

    const float slopef = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
    const __nv_bfloat16  slopeh = __float2bfloat16(slopef);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = D / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ __nv_bfloat16 KQ[ncols*D];
    __nv_bfloat162 * KQ2 = (__nv_bfloat162 *) KQ;

    __nv_bfloat16 kqmax[ncols];
    __nv_bfloat16 kqsum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -BF16_MAX;
        kqsum[j] = 0.0f;
    }

    __shared__ __nv_bfloat16 kqmax_shared[ncols][WARP_SIZE];
    __shared__ __nv_bfloat16 kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -BF16_MAX;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __shared__ __nv_bfloat16 maskh_shared[ncols*D];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        maskh_shared[j*D + tid] = 0.0f;
    }

    __syncthreads();

    // Convert Q to __nv_bfloat162 (bf16 K) or q8_1 (quantized K) and store in registers:
    __nv_bfloat162  Q_h2[ncols][D/(2*WARP_SIZE)];
    int   Q_i32[ncols][D/(sizeof(int)*QK8_1) == 0 ? 1 : D/(sizeof(int)*QK8_1)];
    __nv_bfloat162  Q_ds[ncols][D/QK8_1 == 0 ? 1 : D/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int   * tmp_q_i32 = (int   *) &KQ[j*D];
            __nv_bfloat162 * tmp_q_ds  = (__nv_bfloat162 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_bfloat162(0.0f, 0.0f);
                }
                continue;
            }

            const float * Q_f = (const float *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<__nv_bfloat162>(Q_f + 4*i0, scale, tmp_q_i32, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int   * tmp_q_i32 = (int   *) &KQ[j*D];
            __nv_bfloat162 * tmp_q_ds  = (__nv_bfloat162 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0/WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_f2_j = (const float2 *) (Q + j*nb01);

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float2 tmp = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_h2[j][i0/WARP_SIZE] = make_bfloat162(scale, scale) * make_bfloat162(tmp.x, tmp.y);
            }
        }
    }


    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*D + tid] = -BF16_MAX;
    }
    __syncthreads();

    __nv_bfloat162 VKQ[ncols] = {make_bfloat162(0.0f, 0.0f)};

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*D * nb11;
    V     += blockIdx.y*D * nb21;
    maskh += blockIdx.y*D;
    for (int k_VKQ_0 = blockIdx.y*D; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*D,
             /* Increment pointers after each loop: */
             K += gridDim.y*D*nb11, V += gridDim.y*D*nb21, maskh += gridDim.y*D) {

        // Calculate KQ tile and keep track of new maximum KQ values:

        if (mask) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                maskh_shared[j*D + tid] = slopeh*maskh[j*ne11 + tid];
            }
            __syncthreads();
        }

        // For unknown reasons using a half array of size 1 for kqmax_new causes a performance regression,
        // see https://github.com/ggerganov/llama.cpp/pull/7061 .
        // Therefore this variable is defined twice but only used once (so that the compiler can optimize out the unused variable).
    __nv_bfloat16 kqmax_new = kqmax[0];
    __nv_bfloat16 kqmax_new_arr[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax_new_arr[j] = kqmax[j];
    }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                __nv_bfloat16 sum = vec_dot_KQ(K + i_KQ*nb11, Q_h2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum((float)sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                sum += maskh_shared[j*D + i_KQ];

                if (ncols == 1) {
                    kqmax_new        = ggml_cuda_bf16max(kqmax_new,        sum);
                } else {
                    kqmax_new_arr[j] = ggml_cuda_bf16max(kqmax_new_arr[j], sum);
                }

                if (threadIdx.x == 0) {
                    KQ[j*D + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            __nv_bfloat16 kqmax_new_j = ncols == 1 ? kqmax_new : kqmax_new_arr[j];

            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            __nv_bfloat16 kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const __nv_bfloat16 KQ_max_scale = hexp_bf16(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const __nv_bfloat16 val = hexp_bf16(KQ[j*D + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*D + tid] = val;

            VKQ[j] *= __bfloat162bfloat162(KQ_max_scale);
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < D; k0 += 2) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k0 >= ne11) {
                break;
            }

            __nv_bfloat162 V_k;
            reinterpret_cast<__nv_bfloat16&>(V_k.x) = dequantize_1_v(V + (k0 + 0)*nb21, tid);
            reinterpret_cast<__nv_bfloat16&>(V_k.y) = dequantize_1_v(V + (k0 + 1)*nb21, tid);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_k*KQ2[j*(D/2) + k0/2];
            }
        }

        __syncthreads();
    }

    if (sinksf && blockIdx.y == 0) {
        const __nv_bfloat16 sink = __float2bfloat16(sinksf[head]);

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = fmaxf((float)kqmax[j], (float)sink);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            __nv_bfloat16 kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const __nv_bfloat16 KQ_max_scale = hexp_bf16(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const __nv_bfloat16 val = hexp_bf16(sink - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale;

            if (tid == 0) {
                kqsum[j] += val;
            }

            VKQ[j] *= __bfloat162bfloat162(KQ_max_scale);
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum((float)kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum((float)kqsum[j_VKQ]);

        float dst_val = (__low2bfloat16(VKQ[j_VKQ]) + __high2bfloat16(VKQ[j_VKQ]));
        if (gridDim.y == 1) {
            const float inv_kqsum = 1.0f / (float)kqsum[j_VKQ];
            dst_val *= inv_kqsum;
        }
        dst[(((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + tid] = dst_val;
    }

    if (gridDim.y != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[((sequence*ne01 + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2((float)kqmax[tid], (float)kqsum[tid]);
    }
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask); GGML_UNUSED(sinks);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta);
    GGML_UNUSED(scale); GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03);
    GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
    GGML_UNUSED(ne10); GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13);
    GGML_UNUSED(nb11); GGML_UNUSED(nb12); GGML_UNUSED(nb13);
    GGML_UNUSED(nb21); GGML_UNUSED(nb22); GGML_UNUSED(nb23);
    GGML_UNUSED(ne31); GGML_UNUSED(ne32); GGML_UNUSED(ne33);
    GGML_UNUSED(nb31); GGML_UNUSED(nb32); GGML_UNUSED(nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(FP16_AVAILABLE)
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_bf16_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = D/WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_vec_ext_bf16<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    GGML_ASSERT(fattn_kernel != nullptr);
    constexpr bool need_f16_K = false;
    constexpr bool need_f16_V = false;
    constexpr size_t nbytes_shared = 0;

    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_bf16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    GGML_ASSERT(K->type == type_K);
    GGML_ASSERT(V->type == type_V);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (Q->ne[1] == 1 || GGML_CUDA_CC_IS_NVIDIA(cc)) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_bf16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_bf16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block = 2;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_bf16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_bf16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_bf16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_bf16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 8;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_bf16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_bf16_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_BF16_CASE(D, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_bf16_case                     \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

DECL_FATTN_VEC_BF16_CASE( 64, GGML_TYPE_BF16, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_BF16_CASE( 64, GGML_TYPE_BF16, GGML_TYPE_Q4_1);
DECL_FATTN_VEC_BF16_CASE( 64, GGML_TYPE_BF16, GGML_TYPE_Q5_0);
DECL_FATTN_VEC_BF16_CASE( 64, GGML_TYPE_BF16, GGML_TYPE_Q5_1);
DECL_FATTN_VEC_BF16_CASE( 64, GGML_TYPE_BF16, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_BF16_CASE( 64, GGML_TYPE_BF16, GGML_TYPE_BF16);

DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_BF16,  GGML_TYPE_Q4_0);

DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_BF16,  GGML_TYPE_Q4_1);

DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_BF16,  GGML_TYPE_Q5_0);

DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_BF16,  GGML_TYPE_Q5_1);

DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_BF16,  GGML_TYPE_Q8_0);

DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_BF16);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_BF16);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_BF16);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_BF16);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_BF16);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_BF16,  GGML_TYPE_BF16);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_BF16);
DECL_FATTN_VEC_BF16_CASE(128, GGML_TYPE_BF16, GGML_TYPE_F16);

DECL_FATTN_VEC_BF16_CASE(256, GGML_TYPE_BF16, GGML_TYPE_BF16);
