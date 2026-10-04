#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "kv-stream-attention-dispatch.h"

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

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
            if (gqa_ratio == 20) { // GLM 4.7 Flash
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

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}


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
    size_t bytes = 0;
};

// Reproduce stock Stream-K planning while accounting for descriptors in caller-owned scratch.
static bool mma_span_launch_make(
        ggml_backend_cuda_context & ctx, const ggml_tensor * dst,
        size_t span_count, mma_span_launch & output, ggml_cuda_kv_stream_plan_status * failure = nullptr) {
    using status = ggml_cuda_kv_stream_plan_status;
    if (failure) *failure = status::unsupported_geometry;
    if (!dst || dst->op != GGML_OP_FLASH_ATTN_EXT || !span_count || span_count > INT_MAX) return false;
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    if (!q || !k || !v || !mask || dst->src[4] ||
            q->type != GGML_TYPE_F32 ||
            !((k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16) ||
              (k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q4_0)) ||
            dst->type != GGML_TYPE_F32 || q->ne[0] != 256 || v->ne[0] != 256 ||
            k->ne[0] != 256 || (q->ne[1] != 2 && q->ne[1] != 3 && q->ne[1] != 4) ||
            k->ne[2] <= 0 || v->ne[2] != k->ne[2] || q->ne[2] <= 0 ||
            q->ne[2]%k->ne[2] ||
            (q->ne[2]/k->ne[2] != 2 && q->ne[2]/k->ne[2] != 6 && q->ne[2]/k->ne[2] != 8) ||
            q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
            k->ne[1] != v->ne[1] || k->ne[1] <= 0 || k->ne[1] > INT_MAX ||
            mask->type != GGML_TYPE_F16 || mask->ne[0] < k->ne[1] ||
            mask->ne[1] < q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1) return false;
    float params[3];
    memcpy(params,dst->op_params,sizeof(params));
    if (!std::isfinite(params[0]) || params[0] <= 0 || params[1] != 0 || params[2] != 0) return false;

    constexpr int DKQ = 256, DV = 256;
    const int gqa_ratio = int(q->ne[2]/k->ne[2]);
    // Stock Blackwell uses eight columns for ratio six and masks the two padded heads.
    const int ncols1 = ggml_cuda_kv_stream_mma_ncols1(uint32_t(q->ne[1]),gqa_ratio);
    const int ncols2 = gqa_ratio > 4 ? 8 : 2, ncols = ncols1*ncols2;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!turing_mma_available(cc)) {
        if (failure) *failure = status::unsupported_device_features;
        return false;
    }
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
    const size_t shared_kv_one = size_t(nbatch_fa)*std::max(nbatch_K2+4,nbatch_V2+4)*sizeof(half2);
    const size_t shared_kv_two = size_t(nbatch_fa)*(nbatch_K2+4+nbatch_V2+4)*sizeof(half2);
    const size_t shared_kv = nstages <= 1 ? shared_kv_one : shared_kv_two;
    const size_t shared_q = size_t(ncols)*(DKQ/2+4)*sizeof(half2);
    const size_t shared_mask = size_t(ncols1)*(nbatch_fa/2+4)*sizeof(half2);
    const size_t shared_combine = size_t(nwarps)*cols_per_warp*(nbatch_combine+4)*sizeof(half2);

    mma_span_launch next;
    next.ncols1 = ncols1;
    next.ncols2 = ncols2;
    next.shared_bytes = std::max(shared_combine,q_in_reg ?
        std::max(shared_q,shared_kv+shared_mask) : shared_q+shared_kv+shared_mask);
    if (span_count <= 3) {
        next.span_shared_offset = (next.shared_bytes+15)/16*16;
        next.shared_bytes = next.span_shared_offset+3*sizeof(ggml_cuda_kv_span);
    }
    next.ntiles_x = 1;
    next.gqa_ratio = int(q->ne[2]/k->ne[2]);
    next.ntiles_z_gqa = (next.gqa_ratio+ncols2-1)/ncols2;
    next.ntiles_dst = next.ntiles_x*next.ntiles_z_gqa*int(k->ne[2])*int(q->ne[3]);
    next.ntiles_kv = int((int64_t(k->ne[1])+nbatch_fa-1)/nbatch_fa);
    if (int64_t(next.ntiles_kv)*next.ntiles_dst > INT_MAX) return false;

    fattn_kernel_t ordinary = ncols1 == 2 ?
        flash_attn_ext_f16<DKQ,DV,2,8,false,false> :
        ncols2 == 8 ? flash_attn_ext_f16<DKQ,DV,4,8,false,false> :
        flash_attn_ext_f16<DKQ,DV,4,2,false,false>;
    if (cudaFuncSetAttribute(ordinary,cudaFuncAttributeMaxDynamicSharedMemorySize,int(next.shared_bytes)) != cudaSuccess) {
        if (failure) *failure = status::unsupported_launch_resources;
        (void) cudaGetLastError(); return false;
    }
    int max_blocks_per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,ordinary,nthreads,next.shared_bytes) != cudaSuccess) {
        if (failure) *failure = status::unsupported_launch_resources;
        (void) cudaGetLastError(); return false;
    }
    if (!max_blocks_per_sm) {
        if (failure) *failure = status::unsupported_launch_resources;
        return false;
    }
    const int max_blocks = max_blocks_per_sm*ggml_cuda_info().devices[ctx.device].nsm;
    const int tiles_nwaves = (next.ntiles_dst+max_blocks-1)/max_blocks;
    const int tiles_efficiency = 100*next.ntiles_dst/(max_blocks*tiles_nwaves);
    const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || tiles_efficiency < 75;
    int blocks = next.ntiles_dst;
    if (use_stream_k) {
        const int raw = std::min(max_blocks,next.ntiles_kv*next.ntiles_dst);
        const int rounded = (raw/next.ntiles_dst)*next.ntiles_dst;
        const int loss = rounded > 0 ? 100*(raw-rounded)/raw : 100;
        blocks = loss <= 5 ? rounded : raw;
    }
    if (blocks <= 0) return false;
    next.blocks = {unsigned(blocks),1,1};
    if (span_count > (SIZE_MAX-127)/sizeof(ggml_cuda_kv_span)) return false;
    next.descriptor_bytes = (span_count*sizeof(ggml_cuda_kv_span)+127)/128*128;
    if (next.ntiles_dst%blocks != 0) {
        const size_t entries = size_t(blocks)*ncols*(2+DV/2);
        if (entries > (SIZE_MAX-next.descriptor_bytes)/sizeof(float2)) return false;
        next.fixup_bytes = entries*sizeof(float2);
    }
    next.bytes = next.descriptor_bytes+next.fixup_bytes;
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
        void * workspace, const mma_span_launch & plan) {
    constexpr int DKQ = 256, DV = 256;
    constexpr bool softcap = false, v_is_k = false;
    auto kernel = dst->src[1]->type == GGML_TYPE_Q8_0 ?
        flash_attn_ext_f16_spans<DKQ,DV,ncols1,ncols2,softcap,v_is_k,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0> :
        flash_attn_ext_f16_spans<DKQ,DV,ncols1,ncols2,softcap,v_is_k,GGML_TYPE_F16,GGML_TYPE_F16>;
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(plan.shared_bytes)));
    auto * device_spans = static_cast<ggml_cuda_kv_span *>(workspace);
    CUDA_CHECK(cudaMemcpyAsync(
        device_spans,spans,count*sizeof(*spans),cudaMemcpyHostToDevice,ctx.stream()));
    auto * meta = reinterpret_cast<float2 *>(static_cast<char *>(workspace)+plan.descriptor_bytes);

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
    ggml_cuda_kernel_launch(kernel,launch,
        static_cast<const char *>(q->data),spans[0].k,spans[0].v,
        static_cast<const char *>(mask->data),(const char *)nullptr,(const int *)nullptr,
        static_cast<float *>(dst->data),meta,
        scale,0.0f,1.0f,1.0f,uint32_t(1),0.0f,
        int32_t(q->ne[0]),ne01,int32_t(q->ne[2]),int32_t(q->ne[3]),
        int32_t(q->nb[1]),int32_t(q->nb[2]),int32_t(q->nb[3]),
        int32_t(k->ne[0]),int32_t(k->ne[1]),int32_t(k->ne[2]),int32_t(k->ne[3]),
        int32_t(k->nb[1]),int32_t(k->nb[2]),int64_t(k->nb[3]),
        int32_t(v->nb[1]),int32_t(v->nb[2]),int64_t(v->nb[3]),
        int32_t(mask->ne[1]),int32_t(mask->ne[2]),int32_t(mask->ne[3]),
        int32_t(mask->nb[1]),int32_t(mask->nb[2]),int64_t(mask->nb[3]),
        device_spans,int(count),int(plan.span_shared_offset));
    CUDA_CHECK(cudaGetLastError());

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
            static_cast<float *>(dst->data),meta,
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
            static_cast<float *>(dst->data),meta,
            q->ne[1],q->ne[2],plan.gqa_ratio,total_work,fd0,fd1,fd2,fd3);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_flash_attn_ext_mma_f16_spans(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cuda_kv_span * spans, size_t count,
        void * workspace, size_t workspace_bytes) {
    mma_span_launch plan;
    if (!spans || !workspace || !mma_span_launch_make(ctx,dst,count,plan) ||
            workspace_bytes < plan.bytes || uintptr_t(workspace)%128) return false;
    if (plan.ncols1 == 2) return ggml_cuda_flash_attn_ext_mma_f16_spans_case<2,8>(ctx,dst,spans,count,workspace,plan);
    return plan.ncols2 == 8 ?
        ggml_cuda_flash_attn_ext_mma_f16_spans_case<4,8>(ctx,dst,spans,count,workspace,plan) :
        ggml_cuda_flash_attn_ext_mma_f16_spans_case<4,2>(ctx,dst,spans,count,workspace,plan);
}

template<ggml_type type>
__global__ void flash_attn_ext_mma_convert_rows_kernel(
        const char * source, half * destination, int64_t source_stride, int64_t destination_stride) {
    const int row = blockIdx.x;
    const int first = 8*threadIdx.x;
    flash_attn_ext_f16_dequantize_8<type>(
        source+int64_t(row)*source_stride,
        reinterpret_cast<half2 *>(reinterpret_cast<char *>(destination)+int64_t(row)*destination_stride)+first/2,
        first);
}

bool ggml_cuda_flash_attn_ext_mma_convert_rows(
        ggml_backend_cuda_context & ctx, ggml_type type,
        const char * source, half * destination, int rows, int64_t source_stride, int64_t destination_stride) {
    if (!source || !destination || rows <= 0 || source_stride <= 0 || destination_stride < 256*int64_t(sizeof(half))) return false;
    const ggml_cuda_kernel_launch_params launch({unsigned(rows),1,1},{32,1,1},0,ctx.stream());
    if (type == GGML_TYPE_Q8_0) {
        ggml_cuda_kernel_launch(flash_attn_ext_mma_convert_rows_kernel<GGML_TYPE_Q8_0>,launch,
            source,destination,source_stride,destination_stride);
    } else if (type == GGML_TYPE_Q4_0) {
        ggml_cuda_kernel_launch(flash_attn_ext_mma_convert_rows_kernel<GGML_TYPE_Q4_0>,launch,
            source,destination,source_stride,destination_stride);
    } else return false;
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
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

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

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
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
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
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
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
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
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
        case ggml_cuda_kv_stream_kernel_family::vector:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
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
