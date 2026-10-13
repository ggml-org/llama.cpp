#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "kv-stream-attention-dispatch.h"
#include "kv-stream-dispatch.cuh"
#include "../ggml-kv-stream-device.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// one list per group of ncols1 queries: a column is selected if any query of the group can see it
template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            bool selected = false;
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    // the last group of queries is partial only if ncols1 does not divide n_queries
    GGML_ASSERT(ncols1 == 1 || ncols1 == 8);
    const auto kernel = ncols1 == 1       ? flash_attn_mask_to_sparse_indices<1, false> :
                        n_queries % 8 != 0 ? flash_attn_mask_to_sparse_indices<8, true>  :
                                             flash_attn_mask_to_sparse_indices<8, false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    // the dense kernel handles up to 64/ncols2 queries per K/V pass, the single-query gather has to beat that
    const int64_t n_gather = (ncols1 == 1 ? std::min<int64_t>(Q->ne[1], 64/ncols2) : ncols1) * (int64_t) n_kv_max;

    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2*n_gather);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        // a sparse variant at the full tile width gathers the union of its queries once, prefer it for large batches
        constexpr bool has_wide_sparse = ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 64/ncols2, ncols2);
        if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, ncols2)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20 && GGML_CUDA_CC_IS_NVIDIA(cc)) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        const bool type_K_okay = type_K == GGML_TYPE_##type_K_case || (type_K == GGML_TYPE_F32 && GGML_TYPE_##type_K_case == GGML_TYPE_F16); \
        const bool type_V_okay = type_V == GGML_TYPE_##type_V_case || (type_V == GGML_TYPE_F32 && GGML_TYPE_##type_V_case == GGML_TYPE_F16); \
        if (head_size == (D) && type_K_okay && type_V_okay) {                                                                      \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K, const ggml_type type_V) {
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}


#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
namespace {
struct mma_span_launch {
    dim3 blocks;
    int ncols1 = 0;
    int ncols2 = 0;
    int ntiles_x = 0;
    int ntiles_z_gqa = 0;
    int ntiles_dst = 0;
    int ntiles_kv = 0;
    int gqa_ratio = 0;
    size_t shared_bytes = 0;
    size_t span_shared_offset = 0;
    size_t descriptor_bytes = 0;
    size_t fixup_bytes = 0;
    size_t state_offset = 0;
    size_t block_map_offset = 0;
    size_t state_map_offset = 0;
    size_t state_map_entries = 0;
    bool state_per_block = false;
    size_t output_offset = 0;
    bool resumable = false;
    size_t bytes = 0;
};

// Reproduce stock Stream-K planning while accounting for descriptors in caller-owned scratch.
static bool mma_span_launch_make(
        ggml_backend_cuda_context & ctx, const ggml_tensor * dst,
        size_t span_count, mma_span_launch & output, ggml_cuda_kv_stream_plan_status * failure = nullptr,
        bool resumable = false) {
    using status = ggml_cuda_kv_stream_plan_status;
    if (failure) *failure = status::unsupported_geometry;
    if (!dst || dst->op != GGML_OP_FLASH_ATTN_EXT || !span_count || span_count > 3 ||
            ggml_get_op_params_i32(dst,4) != 0) return false;
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    if (!q || !k || !v || !mask || dst->src[4] ||
            q->type != GGML_TYPE_F32 ||
            !ggml_cuda_kv_stream_kernel(k->type,v->type) ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            dst->ne[0] != 256 || dst->ne[1] != q->ne[2] || dst->ne[2] != q->ne[1] || dst->ne[3] != 1 ||
            q->ne[0] != 256 || v->ne[0] != 256 ||
            k->ne[0] != 256 || q->ne[1] < 1 || q->ne[1] > INT32_MAX ||
            k->ne[2] <= 0 || v->ne[2] != k->ne[2] || q->ne[2] <= 0 || q->ne[2] > 65535 ||
            q->ne[2]%k->ne[2] ||
            (q->ne[2]/k->ne[2] != 2 && q->ne[2]/k->ne[2] != 6 && q->ne[2]/k->ne[2] != 8) ||
            q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
            k->ne[1] != v->ne[1] || k->ne[1] <= 0 || k->ne[1] > INT_MAX ||
            mask->type != GGML_TYPE_F16 || mask->ne[0] < k->ne[1] ||
            mask->ne[1] < q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1) return false;
    if (q->ne[1] > INT32_MAX/512/q->ne[2]) return false;
    float params[3];
    memcpy(params,dst->op_params,sizeof(params));
    if (!std::isfinite(params[0]) || params[0] <= 0 || params[1] != 0 || params[2] != 0) return false;
    if (k->ne[1]%FATTN_KQ_STRIDE) return false;
    for (const auto * tensor : {q,k,v,mask}) {
        if (tensor->nb[1] > INT32_MAX || tensor->nb[2] > INT32_MAX || tensor->nb[3] > INT64_MAX) return false;
    }
    if (q->nb[3] > INT32_MAX) return false;
    for (const auto * tensor : {q,k,v,mask}) if (!ggml_is_quantized(tensor->type))
        for (int i=1;i<4;++i) if (tensor->nb[i]%16) return false;
    if (ggml_cuda_flash_attn_ext_kernel_family(ctx.device,dst) != ggml_cuda_kv_stream_kernel_family::mma) {
        if (failure) *failure=status::stock_unavailable;
        return false;
    }

    constexpr int DKQ = 256, DV = 256;
    const int gqa_ratio = int(q->ne[2]/k->ne[2]);
    // Stock Blackwell uses eight columns for ratio six and masks the two padded heads.
    const int ncols2 = gqa_ratio > 4 ? 8 : 2;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!turing_mma_available(cc)) {
        if (failure) *failure = status::unsupported_device_features;
        return false;
    }
    const bool turing_only = ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING;
    const int ncols1 = ggml_cuda_kv_stream_mma_ncols1(uint32_t(q->ne[1]), gqa_ratio, turing_only);
    if (!ncols1) return false;
    const int ncols = ncols1*ncols2;
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ,DV,ncols,cc);
    const int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ,DV,ncols,cc);
    const int nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ,DV,ncols,cc);
    const int nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ,DV,ncols,cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ,DV,ncols,cc);
    const bool q_in_reg = ggml_cuda_fattn_mma_get_Q_in_reg(DKQ,DV,ncols,cc);
    const int cols_per_warp = std::min(ncols,get_cols_per_warp(cc));
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps = nthreads/warp_size;
    const int nstages = ggml_cuda_fattn_mma_get_nstages(DKQ,DV,ncols1,ncols2,cc);
    const bool swizzled = ggml_cuda_fattn_mma_get_swizzled(DKQ,DV,ncols,cc);
    const int stride_k = nbatch_K2+(swizzled ? 0 : 4), stride_v = nbatch_V2+(swizzled ? 0 : 4);
    const size_t shared_kv_one = size_t(nbatch_fa)*std::max(stride_k,stride_v)*sizeof(half2);
    const size_t shared_kv_two = size_t(nbatch_fa)*(stride_k+stride_v)*sizeof(half2);
    const size_t shared_kv = nstages <= 1 ? shared_kv_one : shared_kv_two;
    const size_t shared_q = size_t(ncols)*(DKQ/2+4)*sizeof(half2);
    const size_t shared_mask = size_t(ncols1)*(nbatch_fa/2+4)*sizeof(half2);
    const size_t shared_combine = size_t(nwarps)*cols_per_warp*(nbatch_combine+4)*sizeof(half2);

    mma_span_launch next;
    next.resumable=resumable;
    next.ncols1 = ncols1;
    next.ncols2 = ncols2;
    next.shared_bytes = std::max(shared_combine,q_in_reg ?
        std::max(shared_q,shared_kv+shared_mask) : shared_q+shared_kv+shared_mask);
    const size_t native_shared_bytes=next.shared_bytes;
    if (span_count <= 3) {
        next.span_shared_offset = (next.shared_bytes+15)/16*16;
        next.shared_bytes = next.span_shared_offset+3*sizeof(ggml_cuda_kv_span);
    }
    next.ntiles_x = ggml_cuda_kv_stream_mma_query_tiles(uint32_t(q->ne[1]), gqa_ratio, turing_only);
    next.gqa_ratio = int(q->ne[2]/k->ne[2]);
    next.ntiles_z_gqa = (next.gqa_ratio+ncols2-1)/ncols2;
    const int64_t ntiles_dst = int64_t(next.ntiles_x)*next.ntiles_z_gqa*k->ne[2];
    if (ntiles_dst > INT_MAX) {
        if (failure) *failure = status::overflow;
        return false;
    }
    next.ntiles_dst = int(ntiles_dst);
    next.ntiles_kv = int((int64_t(k->ne[1])+nbatch_fa-1)/nbatch_fa);
    if (int64_t(next.ntiles_kv)*next.ntiles_dst > INT_MAX) {
        if (failure) *failure = status::overflow;
        return false;
    }

    fattn_kernel_t ordinary = ncols1 == 1 ?
        flash_attn_ext_f16<DKQ,DV,1,8,false,false,false> : ncols1 == 2 ?
        flash_attn_ext_f16<DKQ,DV,2,8,false,false,false> :
        ncols1 == 4 ? (ncols2 == 8 ? flash_attn_ext_f16<DKQ,DV,4,8,false,false,false> :
            flash_attn_ext_f16<DKQ,DV,4,2,false,false,false>) :
        ncols2 == 8 ? flash_attn_ext_f16<DKQ,DV,8,8,false,false,false> :
            flash_attn_ext_f16<DKQ,DV,8,2,false,false,false>;
    auto spanned = ggml_cuda_kv_stream_mma_kernel(k->type,v->type,uint32_t(ncols1),uint32_t(ncols2),resumable);
    if (!spanned) {
        if (failure) *failure=status::missing_streamed_implementation;
        return false;
    }
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes,spanned) != cudaSuccess) {
        if (failure) *failure=status::unsupported_launch_resources;
        (void) cudaGetLastError(); return false;
    }
    const auto limit=ggml_cuda_info().devices[ctx.device].smpbo;
    if (size_t(attributes.sharedSizeBytes) > limit || next.shared_bytes > limit-size_t(attributes.sharedSizeBytes) ||
            nthreads > attributes.maxThreadsPerBlock) {
        if (failure) *failure=status::unsupported_launch_resources;
        return false;
    }
    if (cudaFuncSetAttribute(spanned,cudaFuncAttributeMaxDynamicSharedMemorySize,int(next.shared_bytes)) != cudaSuccess) {
        if (failure) *failure=status::unsupported_launch_resources;
        (void) cudaGetLastError(); return false;
    }
    int span_occupancy=0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&span_occupancy,spanned,nthreads,next.shared_bytes) != cudaSuccess || !span_occupancy) {
        if (failure) *failure=status::unsupported_launch_resources;
        (void) cudaGetLastError(); return false;
    }
    // Extra descriptor bytes may reduce streamed occupancy, but must not change stock Stream-K arithmetic.
    if (cudaFuncSetAttribute(ordinary,cudaFuncAttributeMaxDynamicSharedMemorySize,int(native_shared_bytes)) != cudaSuccess) {
        if (failure) *failure = status::unsupported_launch_resources;
        (void) cudaGetLastError(); return false;
    }
    int max_blocks_per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,ordinary,nthreads,native_shared_bytes) != cudaSuccess) {
        if (failure) *failure = status::unsupported_launch_resources;
        (void) cudaGetLastError(); return false;
    }
    if (!max_blocks_per_sm) {
        if (failure) *failure = status::unsupported_launch_resources;
        return false;
    }
    const int max_blocks = max_blocks_per_sm*ggml_cuda_info().devices[ctx.device].nsm;
    ggml_cuda_fattn_stream_k_plan stock_plan;
    if (!ggml_cuda_fattn_stream_k_plan_make(next.ntiles_dst,next.ntiles_kv,max_blocks,
            cc >= GGML_CUDA_CC_ADA_LOVELACE,false,stock_plan)) return false;
    const int blocks = stock_plan.blocks;
    next.blocks = {unsigned(blocks),1,1};
    if (span_count > (SIZE_MAX-127)/sizeof(ggml_cuda_kv_span)) return false;
    next.descriptor_bytes = (span_count*sizeof(ggml_cuda_kv_span)+127)/128*128;
    if (next.ntiles_dst%blocks != 0) {
        size_t entries = size_t(blocks);
        for (size_t factor : {size_t(ncols),size_t(2+DV/2)}) {
            if (entries > SIZE_MAX/factor) {
                if (failure) *failure = status::overflow;
                return false;
            }
            entries *= factor;
        }
        if (entries > (SIZE_MAX-next.descriptor_bytes)/sizeof(float2)) {
            if (failure) *failure = status::overflow;
            return false;
        }
        next.fixup_bytes = entries*sizeof(float2);
    }
    next.bytes = next.descriptor_bytes+next.fixup_bytes;
    if (resumable) {
        ggml_cuda_fattn_mma_resume_layout state;
        if (!ggml_cuda_fattn_mma_resume_layout_make(next.ntiles_dst,blocks,nthreads,ncols1,ncols2,state,next.ntiles_kv,true) ||
                next.bytes > SIZE_MAX-127) {
            if (failure) *failure=status::overflow;
            return false;
        }
        next.block_map_offset=(next.bytes+127)/128*128;
        const size_t map_bytes=(size_t(blocks)*sizeof(uint32_t)+127)/128*128;
        if (map_bytes > SIZE_MAX-next.block_map_offset) return false;
        next.state_map_offset=next.block_map_offset+map_bytes;
        next.state_map_entries=state.map_entries;
        next.state_per_block=state.block_indexed;
        const size_t state_map_bytes=(state.map_entries*sizeof(uint32_t)+127)/128*128;
        if (state_map_bytes > SIZE_MAX-next.state_map_offset) return false;
        next.state_offset=next.state_map_offset+state_map_bytes;
        if (state.bytes > SIZE_MAX-next.state_offset) {
            if (failure) *failure=status::overflow;
            return false;
        }
        next.bytes=next.state_offset+state.bytes;
        if (state.bytes) {
            next.output_offset=next.bytes;
            const size_t payload=ggml_nbytes(dst);
            if (payload > SIZE_MAX-next.output_offset) {
                if (failure) *failure=status::overflow;
                return false;
            }
            next.bytes=next.output_offset+payload;
        }
        // Singleton slices write only fixup fragments until the last K tile, so no private output is needed.
    }
    output = next;
    if (failure) *failure = status::success;
    return true;
}
} // namespace

size_t ggml_cuda_flash_attn_ext_mma_f16_spans_workspace(
        ggml_backend_cuda_context & ctx, const ggml_tensor * dst, size_t spans) {
    mma_span_launch plan;
    return mma_span_launch_make(ctx,dst,spans,plan) ? plan.bytes : 0;
}

// Report the same caller-owned layout and shared-memory usage used by the unchanged span launcher.
bool ggml_cuda_flash_attn_ext_mma_f16_spans_requirements(
        ggml_backend_cuda_context & ctx, const ggml_tensor * dst, size_t spans,
        ggml_cuda_kv_stream_workspace_requirements & output, ggml_cuda_kv_stream_plan_status * failure) {
    mma_span_launch plan;
    if (!mma_span_launch_make(ctx, dst, spans, plan, failure)) return false;
    output = {true, 0, plan.bytes, 128, plan.shared_bytes, true, true};
    return true;
}

template<int ncols1, int ncols2>
static bool ggml_cuda_flash_attn_ext_mma_f16_spans_case(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cuda_kv_span * spans, size_t count,
        void * workspace, const mma_span_launch & plan,
        ggml_cuda_fattn_mma_resume_control control = {}) {
    constexpr int DKQ = 256, DV = 256;
    auto kernel=ggml_cuda_kv_stream_mma_kernel(dst->src[1]->type,dst->src[2]->type,ncols1,ncols2,plan.resumable);
    if (!kernel) return false;
    auto * device_spans = static_cast<ggml_cuda_kv_span *>(workspace);
    CUDA_CHECK(cudaMemcpyAsync(
        device_spans,spans,count*sizeof(*spans),cudaMemcpyHostToDevice,ctx.stream()));
    auto * meta = reinterpret_cast<float2 *>(static_cast<char *>(workspace)+plan.descriptor_bytes);
    auto * destination=plan.resumable && plan.output_offset ? reinterpret_cast<float *>(static_cast<char *>(workspace)+plan.output_offset) :
        static_cast<float *>(dst->data);

    auto * q = dst->src[0];
    auto * k = dst->src[1];
    auto * v = dst->src[2];
    auto * mask = dst->src[3];
    float scale;
    memcpy(&scale,dst->op_params,sizeof(scale));
    const uint3 ne01 = init_fastdiv_values(q->ne[1]);
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ,DV,ncols1*ncols2,ggml_cuda_info().devices[ctx.device].cc);
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const ggml_cuda_kernel_launch_params launch(
        plan.blocks,{unsigned(warp_size),unsigned(nthreads/warp_size),1},
        plan.shared_bytes,ctx.stream());
    auto selected_launch=launch;
    std::vector<uint32_t> active_blocks;
    std::vector<uint32_t> state_ids;
    if (plan.resumable) {
        control.state_per_block=plan.state_per_block;
        const int tile=ggml_cuda_fattn_mma_get_nbatch_fa(DKQ,DV,ncols1*ncols2,ggml_cuda_info().devices[ctx.device].cc);
        try {
            active_blocks.reserve(plan.blocks.x);
            for (uint32_t b=0;b<plan.blocks.x;++b)
                if (ggml_cuda_fattn_mma_resume_block_active(plan.ntiles_dst,plan.ntiles_kv,plan.blocks.x,b,
                        control.first/tile,(control.end+tile-1)/tile)) active_blocks.push_back(b);
        } catch (const std::bad_alloc &) { return false; }
        if (active_blocks.empty()) return false;
        if (plan.state_map_entries) {
            auto * ids=reinterpret_cast<uint32_t *>(static_cast<char *>(workspace)+plan.state_map_offset);
            // Like the checkpoints, this immutable table belongs to the ordered operation, not each wave.
            // The first wave initializes it; later waves retain the same caller-owned scratch until fixup.
            if (!control.first) {
                try { state_ids.resize(plan.state_map_entries); }
                catch (const std::bad_alloc &) { return false; }
                if (!ggml_cuda_fattn_mma_resume_state_map(plan.ntiles_dst,plan.blocks.x,plan.ntiles_kv,
                        state_ids.data(),state_ids.size())) return false;
                CUDA_CHECK(cudaMemcpyAsync(ids,state_ids.data(),state_ids.size()*sizeof(uint32_t),cudaMemcpyHostToDevice,ctx.stream()));
            }
            control.state_ids=ids;
        }
        control.native_blocks=plan.blocks.x;
        if (active_blocks.size() != plan.blocks.x) {
            auto * ids=reinterpret_cast<uint32_t *>(static_cast<char *>(workspace)+plan.block_map_offset);
            CUDA_CHECK(cudaMemcpyAsync(ids,active_blocks.data(),active_blocks.size()*sizeof(uint32_t),cudaMemcpyHostToDevice,ctx.stream()));
            control.block_ids=ids;
            selected_launch=ggml_cuda_kernel_launch_params(
                {unsigned(active_blocks.size()),1,1},{unsigned(warp_size),unsigned(nthreads/warp_size),1},plan.shared_bytes,ctx.stream());
        }
    }
    ggml_cuda_kernel_launch(kernel,selected_launch,
        static_cast<const char *>(q->data),spans[0].k,spans[0].v,
        static_cast<const char *>(mask->data),(const char *)nullptr,(const int *)nullptr,
        destination,meta,
        scale,0.0f,1.0f,1.0f,uint32_t(1),0.0f,
        int32_t(q->ne[0]),ne01,int32_t(q->ne[2]),int32_t(q->ne[3]),
        int32_t(q->nb[1]),int32_t(q->nb[2]),int32_t(q->nb[3]),
        int32_t(k->ne[0]),int32_t(k->ne[1]),int32_t(k->ne[2]),int32_t(k->ne[3]),
        int32_t(k->nb[1]),int32_t(k->nb[2]),int64_t(k->nb[3]),
        int32_t(v->nb[1]),int32_t(v->nb[2]),int64_t(v->nb[3]),
        int32_t(mask->ne[1]),int32_t(mask->ne[2]),int32_t(mask->ne[3]),
        int32_t(mask->nb[1]),int32_t(mask->nb[2]),int64_t(mask->nb[3]),
        device_spans,int(count),int(plan.span_shared_offset),
        uint32_t((ggml_is_contiguous(k) ? 1 : 0)|(ggml_is_contiguous(v) ? 2 : 0)),control);
    CUDA_CHECK(cudaGetLastError());

    if (plan.resumable && control.end < k->ne[1]) return true;

    const int blocks = int(plan.blocks.x);
    if (blocks%plan.ntiles_dst == 0 && blocks > plan.ntiles_dst) {
        const int bpt = blocks/plan.ntiles_dst;
        const uint3 fd0 = init_fastdiv_values(plan.ntiles_x*plan.ntiles_z_gqa*k->ne[2]);
        const uint3 fd1 = init_fastdiv_values(plan.ntiles_x*plan.ntiles_z_gqa);
        const uint3 fd2 = init_fastdiv_values(plan.ntiles_x);
        const ggml_cuda_kernel_launch_params fixup(
            {unsigned(plan.ntiles_dst),ncols1,ncols2},{DV,1,1},0,ctx.stream());
        ggml_cuda_kernel_launch(
            flash_attn_stream_k_fixup_uniform<DV,ncols1,ncols2>,fixup,
            destination,meta,
            q->ne[1],q->ne[2],k->ne[2],blocks,
            plan.gqa_ratio,bpt,fd0,fd1,fd2);
    } else if (plan.ntiles_dst%blocks != 0) {
        const int total_work = plan.ntiles_kv*plan.ntiles_dst;
        const uint3 fd0 = init_fastdiv_values(
            plan.ntiles_kv*plan.ntiles_x*plan.ntiles_z_gqa*k->ne[2]);
        const uint3 fd1 = init_fastdiv_values(
            plan.ntiles_kv*plan.ntiles_x*plan.ntiles_z_gqa);
        const uint3 fd2 = init_fastdiv_values(plan.ntiles_kv*plan.ntiles_x);
        const uint3 fd3 = init_fastdiv_values(plan.ntiles_kv);
        const ggml_cuda_kernel_launch_params fixup(
            {unsigned(blocks),ncols1,ncols2},{DV,1,1},0,ctx.stream());
        ggml_cuda_kernel_launch(
            flash_attn_stream_k_fixup_general<DV,ncols1,ncols2>,fixup,
            destination,meta,
            q->ne[1],q->ne[2],plan.gqa_ratio,total_work,fd0,fd1,fd2,fd3);
    }
    CUDA_CHECK(cudaGetLastError());
    if (plan.resumable && plan.output_offset) CUDA_CHECK(cudaMemcpyAsync(dst->data,destination,ggml_nbytes(dst),cudaMemcpyDeviceToDevice,ctx.stream()));
    return true;
}

// Reserve raw native fragments and a private publication plane in the same caller-owned grant.
bool ggml_cuda_flash_attn_ext_mma_f16_resume_requirements(
        ggml_backend_cuda_context & ctx, const ggml_tensor * dst,
        ggml_cuda_kv_stream_workspace_requirements & output, ggml_cuda_kv_stream_plan_status * failure) {
    mma_span_launch plan;
    if (!mma_span_launch_make(ctx,dst,3,plan,failure,true)) return false;
    output={true,0,plan.bytes,128,plan.shared_bytes,true,true};
    return true;
}

// Quote the selected native topology and opaque configuration without allocating scratch.
bool ggml_cuda_flash_attn_ext_mma_f16_resume_plan(
        ggml_backend_cuda_context & ctx, const ggml_tensor * dst, ggml_kv_stream_resume_plan & output) {
    mma_span_launch plan;
    if (!mma_span_launch_make(ctx,dst,3,plan,nullptr,true)) return false;
    ggml_kv_stream_resume_plan next;
    next.heads=uint32_t(dst->src[0]->ne[2]); next.queries=uint32_t(dst->src[0]->ne[1]);
    next.splits=plan.blocks.x; next.tokens=size_t(dst->src[1]->ne[1]); next.bytes=plan.bytes;
    next.state_bytes=plan.output_offset ? plan.output_offset-plan.state_offset : 0;
    next.partial_offset=plan.output_offset; next.meta_offset=plan.descriptor_bytes;
    next.kernel_config[0]=2; next.kernel_config[1]=uint32_t(dst->src[1]->ne[2]);
    next.kernel_config[2]=uint32_t(ggml_cuda_fattn_mma_get_nbatch_fa(256,256,plan.ncols1*plan.ncols2,
        ggml_cuda_info().devices[ctx.device].cc));
    next.kernel_config[3]=uint32_t(plan.ncols1); next.kernel_config[4]=uint32_t(plan.ncols2);
    next.resume_resident=true;
    next.span_resume=true;
    output=next;
    return true;
}

// Consume one ordered complete-tile wave; only the final wave runs stock fixup and publishes output.
bool ggml_cuda_flash_attn_ext_mma_f16_resume(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cuda_kv_span * spans, size_t count, void * workspace, size_t workspace_bytes,
        size_t first, size_t end, size_t active) {
    mma_span_launch plan;
    if (!spans || !count || count > 3 || !workspace || !mma_span_launch_make(ctx,dst,3,plan,nullptr,true) ||
            workspace_bytes < plan.bytes || uintptr_t(workspace)%128 ||
            plan.bytes > UINTPTR_MAX-uintptr_t(workspace)) return false;
    const auto * k=dst->src[1], * v=dst->src[2];
    const auto cc=ggml_cuda_info().devices[ctx.device].cc;
    const size_t tile=ggml_cuda_fattn_mma_get_nbatch_fa(256,256,plan.ncols1*plan.ncols2,cc);
    if (active < size_t(dst->src[0]->ne[1]) || !ggml_cuda_fattn_mma_resume_wave_validate(
            spans,count,active,size_t(k->ne[1]),tile,first,end,k->ne[2],
            ggml_row_size(k->type,256),ggml_row_size(v->type,256))) return false;
    const ggml_cuda_fattn_mma_resume_control control{static_cast<char *>(workspace)+plan.state_offset,int32_t(first),int32_t(end)};
    if (plan.ncols1 == 1) return ggml_cuda_flash_attn_ext_mma_f16_spans_case<1,8>(ctx,dst,spans,count,workspace,plan,control);
    if (plan.ncols1 == 2) return ggml_cuda_flash_attn_ext_mma_f16_spans_case<2,8>(ctx,dst,spans,count,workspace,plan,control);
    if (plan.ncols1 == 8) return plan.ncols2 == 8 ?
        ggml_cuda_flash_attn_ext_mma_f16_spans_case<8,8>(ctx,dst,spans,count,workspace,plan,control) :
        ggml_cuda_flash_attn_ext_mma_f16_spans_case<8,2>(ctx,dst,spans,count,workspace,plan,control);
    return plan.ncols2 == 8 ?
        ggml_cuda_flash_attn_ext_mma_f16_spans_case<4,8>(ctx,dst,spans,count,workspace,plan,control) :
        ggml_cuda_flash_attn_ext_mma_f16_spans_case<4,2>(ctx,dst,spans,count,workspace,plan,control);
}

bool ggml_cuda_flash_attn_ext_mma_f16_spans(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cuda_kv_span * spans, size_t count,
        void * workspace, size_t workspace_bytes) {
    mma_span_launch plan;
    if (!spans || !workspace || !mma_span_launch_make(ctx,dst,count,plan) ||
            workspace_bytes < plan.bytes || uintptr_t(workspace)%128) return false;
    const auto * k = dst->src[1], * v = dst->src[2];
    const int64_t active = int64_t(spans[count-1].first)+spans[count-1].tokens;
    if (active < dst->src[0]->ne[1] || active > k->ne[1] ||
            (active != k->ne[1] && (active+255)/256*256 != k->ne[1]) ||
            !ggml_cuda_kv_spans_validate(spans,count,active,k->ne[2],
            ggml_row_size(k->type,k->ne[0]),ggml_row_size(v->type,v->ne[0]))) return false;
    if (plan.ncols1 == 1) return ggml_cuda_flash_attn_ext_mma_f16_spans_case<1,8>(ctx,dst,spans,count,workspace,plan);
    if (plan.ncols1 == 2) return ggml_cuda_flash_attn_ext_mma_f16_spans_case<2,8>(ctx,dst,spans,count,workspace,plan);
    if (plan.ncols1 == 8) return plan.ncols2 == 8 ?
        ggml_cuda_flash_attn_ext_mma_f16_spans_case<8,8>(ctx,dst,spans,count,workspace,plan) :
        ggml_cuda_flash_attn_ext_mma_f16_spans_case<8,2>(ctx,dst,spans,count,workspace,plan);
    return plan.ncols2 == 8 ?
        ggml_cuda_flash_attn_ext_mma_f16_spans_case<4,8>(ctx,dst,spans,count,workspace,plan) :
        ggml_cuda_flash_attn_ext_mma_f16_spans_case<4,2>(ctx,dst,spans,count,workspace,plan);
}

template<ggml_type type>
__global__ void flash_attn_ext_mma_convert_rows_kernel(
        const char * source, half * destination, int64_t source_stride, int64_t destination_stride, bool contiguous) {
    const int row = blockIdx.x;
    const int first = 8*threadIdx.x;
    auto * output = reinterpret_cast<half2 *>(reinterpret_cast<char *>(destination)+int64_t(row)*destination_stride);
    for (int i = 0; i < 8; i += 2)
        output[(first+i)/2] = ggml_cuda_fattn_decode_pair<type>(source+int64_t(row)*source_stride,first+i,contiguous);
}

bool ggml_cuda_flash_attn_ext_mma_convert_rows(
        ggml_backend_cuda_context & ctx, ggml_type type,
        const char * source, half * destination, int rows, int64_t source_stride, int64_t destination_stride,
        bool contiguous) {
    if (!source || !destination || rows <= 0 || source_stride <= 0 || destination_stride < 256*int64_t(sizeof(half))) return false;
    const ggml_cuda_kernel_launch_params launch({unsigned(rows),1,1},{32,1,1},0,ctx.stream());
    if (type == GGML_TYPE_Q8_0) {
        ggml_cuda_kernel_launch(flash_attn_ext_mma_convert_rows_kernel<GGML_TYPE_Q8_0>,launch,
            source,destination,source_stride,destination_stride,contiguous);
    } else if (type == GGML_TYPE_Q4_0) {
        ggml_cuda_kernel_launch(flash_attn_ext_mma_convert_rows_kernel<GGML_TYPE_Q4_0>,launch,
            source,destination,source_stride,destination_stride,contiguous);
    } else return false;
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (max_bias == 0.0f && gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (Q->ne[0] > 256 && gqa_opt_applies && Q->ne[1] * gqa_ratio_eff > 128) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

ggml_cuda_kv_stream_kernel_family ggml_cuda_flash_attn_ext_kernel_family(int device, const ggml_tensor * dst) {
    using family = ggml_cuda_kv_stream_kernel_family;
    switch (ggml_cuda_get_best_fattn_kernel(device, dst)) {
        case BEST_FATTN_KERNEL_VEC: return family::vector;
        case BEST_FATTN_KERNEL_TILE: return family::tile;
        case BEST_FATTN_KERNEL_MMA_F16: return family::mma;
        case BEST_FATTN_KERNEL_NONE: return family::none;
    }
    return family::none;
}

size_t ggml_cuda_flash_attn_ext_selected_alloc_size(ggml_cuda_kv_stream_kernel_family kernel, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case ggml_cuda_kv_stream_kernel_family::tile:
        case ggml_cuda_kv_stream_kernel_family::mma:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case ggml_cuda_kv_stream_kernel_family::vector: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case ggml_cuda_kv_stream_kernel_family::none:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    return ggml_cuda_flash_attn_ext_selected_alloc_size(ggml_cuda_flash_attn_ext_kernel_family(device, dst), dst);
}

void ggml_cuda_flash_attn_ext_selected(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_kv_stream_kernel_family family) {
    ggml_cuda_set_device(ctx.device);
    switch (family) {
        case ggml_cuda_kv_stream_kernel_family::none:
            GGML_ABORT("fatal error");
        case ggml_cuda_kv_stream_kernel_family::tile:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case ggml_cuda_kv_stream_kernel_family::vector:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case ggml_cuda_kv_stream_kernel_family::mma:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    ggml_cuda_flash_attn_ext_selected(ctx, dst, ggml_cuda_flash_attn_ext_kernel_family(ctx.device, dst));
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
