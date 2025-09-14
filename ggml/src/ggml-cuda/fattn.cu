#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec-f16.cuh"
#include "fattn-vec-f32.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

#include <cstdlib>
static inline bool env_flag_true(const char *name) {
    const char *v = std::getenv(name);
    if (!v) return false;
    return v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

// want_fp16 is intentionally unused: vec availability for the supported instances does not
// differ by accumulation precision for our deterministic paths.
static bool det_vec_supported(const ggml_tensor * dst, bool want_fp16) {
    (void) want_fp16; // intentionally unused
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const int D  = (int) Q->ne[0];
    const ggml_type tK = K->type;
    const ggml_type tV = V->type;

    // Minimal, robust support set that matches compiled vec instances across builds.
    // F16/F16 at D in {64,128,256} always exists.
    if (tK == GGML_TYPE_F16 && tV == GGML_TYPE_F16) {
        if (D == 64 || D == 128 || D == 256) return true;
        return false;
    }

    // Quantized: keep to conservative pairs that exist without GGML_CUDA_FA_ALL_QUANTS
    // and that are exercised by tests: D=128 with q4_0/q4_0 or q8_0/q8_0.
    if (D == 128 && tK == tV && (tK == GGML_TYPE_Q4_0 || tK == GGML_TYPE_Q8_0)) {
        // Both vec-f16 and vec-f32 cases are instantiated for these pairs.
        return true;
    }

    // Expanded coverage when full quant vec instances are compiled.
#ifdef GGML_CUDA_FA_ALL_QUANTS
    // D=128: allow any mix of F16 and {q4_0,q4_1,q5_0,q5_1,q8_0} except both F16, which was handled above.
    if (D == 128) {
        auto is_q = [](ggml_type t) {
            switch (t) {
                case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0: return true;
                default: return false;
            }
        };
        if ((is_q(tK) || tK == GGML_TYPE_F16) && (is_q(tV) || tV == GGML_TYPE_F16) && !(tK == GGML_TYPE_F16 && tV == GGML_TYPE_F16)) {
            return true;
        }
    }
    // D=64: K must be F16; V can be quantized.
    if (D == 64 && tK == GGML_TYPE_F16) {
        switch (tV) {
            case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
                return true;
            default: break;
        }
    }
#endif

    // Otherwise, not guaranteed available deterministically.
    return false;
}


template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
        return;
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    const bool use_gqa_opt = mask && max_bias == 0.0f;

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

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

    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
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
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
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
            GGML_ASSERT(gqa_ratio % 16 == 0);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_F16_CASE(D, type_K, type_V)                               \
    if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f16_case<D, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }                                                                       \

static void ggml_cuda_flash_attn_ext_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16 )

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16)

    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#else
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

#define FATTN_VEC_F32_CASE(D, type_K, type_V)                               \
    if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f32_case<D, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }                                                                       \

static void ggml_cuda_flash_attn_ext_vec_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16)

    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#else
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC_F32  = 100,
    BEST_FATTN_KERNEL_VEC_F16  = 110,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

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

    const int cc = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const enum ggml_prec prec = ggml_flash_attn_ext_get_prec(KQV);

    switch (K->ne[0]) {
        case  64:
        case 128:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case  80:
        case  96:
        case 112:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!fp16_mma_available(cc) && !turing_mma_available(cc)) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!turing_mma_available(cc) || gqa_ratio % 16 != 0) {
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

    switch (K->type) {
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
#ifdef GGML_CUDA_FA_ALL_QUANTS
            if (K->ne[0] != 128 && K->ne[0] != 64) {
                return BEST_FATTN_KERNEL_NONE;
            }
#else
            if (K->ne[0] != 128) {
                return BEST_FATTN_KERNEL_NONE;
            }
#endif // GGML_CUDA_FA_ALL_QUANTS
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    switch (V->type) {
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            if (K->ne[0] != 128) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % (2*warp_size) == 0;

    // If Turing tensor cores available, use them except for some cases with batch size 1:
    if (turing_mma_available(cc)) {
        const bool gqa_opt_applies = gqa_ratio % 2 == 0 && mask; // The mma-based kernels have GQA-specific optimizations
        const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
        const bool mma_faster_for_rtx4000 = Q->ne[3] > 1 || (gqa_ratio > 4 && K->ne[1] >= 8192);
        const bool mma_faster_for_bs1 = gqa_opt_applies && !mma_needs_data_conversion &&
            (cc < GGML_CUDA_CC_ADA_LOVELACE || mma_faster_for_rtx4000);
        if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1) {
            if (prec == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
                return BEST_FATTN_KERNEL_VEC_F16;
            }
            return BEST_FATTN_KERNEL_VEC_F32;
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use kernels specializes for small batch sizes if possible:
    if (Q->ne[1] <= 8 && can_use_vector_kernel) {
        if (prec == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            return BEST_FATTN_KERNEL_VEC_F16;
        }
        return BEST_FATTN_KERNEL_VEC_F32;
    }

    // For large batch sizes, use the WMMA kernel if possible:
    if (fp16_mma_available(cc)) {
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    // If there is no suitable kernel for tensor cores or small batch sizes, use the generic kernel for large batch sizes:
    return BEST_FATTN_KERNEL_TILE;
}

static bool det_mma_supported(const ggml_tensor * dst) {
    const int device = ggml_cuda_get_device();
    switch (ggml_cuda_get_best_fattn_kernel(device, dst)) {
        case BEST_FATTN_KERNEL_MMA_F16: return true;
        default: return false;
    }
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    // Deterministic mode: bypass heuristic kernel picker and route to
    // stable, batch-invariant paths. Kernel math stays unchanged; we
    // rely on launch_fattn() to enforce single-block accumulation.
    if (ggml_is_deterministic()) {
        // Helpers
        static bool logged_tile_once = false;
        auto log_tile_once = [&]() {
            if (!logged_tile_once) {
                GGML_LOG_INFO("[det] attention falling back to tile kernel; expect lower throughput.\n");
                logged_tile_once = true;
            }
        };

        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * V = dst->src[2];

        const enum ggml_prec prec = ggml_flash_attn_ext_get_prec(dst);
        const int D  = (int)Q->ne[0];
        const int DV = (int)V->ne[0];

        const bool kv_is_f16 = (K && K->type == GGML_TYPE_F16) && (V && V->type == GGML_TYPE_F16);
        const auto is_quant = [](ggml_type t) {
            switch (t) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return true;
                default: return false;
            }
        };
        const bool kv_both_quant = is_quant(K->type) && is_quant(V->type);

        const bool force_vec   = env_flag_true("GGML_DETERMINISTIC_ATTENTION_FORCE_VEC");
        const bool force_tile  = env_flag_true("GGML_DETERMINISTIC_ATTENTION_FORCE_TILE");
        const bool allow_mma   = env_flag_true("GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA");

        // 1) Special head sizes (80/96/112/576): attempt MMA only if explicitly allowed and supported; otherwise
        // fall back to vec if available, else F16 tile, else abort with instructions.
        if (kv_is_f16 && (D == 80 || D == 96 || D == 112 || D == 576) && !force_vec) {
            if (allow_mma && det_mma_supported(dst)) {
                ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
                return;
            }
            // Prefer vec (if compiled), otherwise deterministic tile for F16.
            if (det_vec_supported(dst, /*want_fp16=*/(ggml_flash_attn_ext_get_prec(dst) == GGML_PREC_DEFAULT))) {
                if (ggml_flash_attn_ext_get_prec(dst) == GGML_PREC_DEFAULT) ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
                else                                                      ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
                return;
            }
            log_tile_once();
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            return;
        }

        // 2) Quantized K/V: supported via vec kernels for selected shapes only
        if (kv_both_quant) {
            // Probe exact vec availability for the pair
            const bool want_fp16 = (prec == GGML_PREC_DEFAULT);
            if (force_tile) {
                GGML_ABORT("deterministic attention: FORCE_TILE requested but tile path does not support quantized K/V. Use F16 K/V. (KV must be multiple of 256; mask padded to GGML_KQ_MASK_PAD)");
            }
            if (det_vec_supported(dst, want_fp16)) {
                if (want_fp16) ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
                else           ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
                return;
            }
            GGML_ABORT("deterministic attention: quantized K/V unsupported in det mode for this shape (D=%d, DV=%d, K=%s, V=%s). Use F16 K/V or D=128 with q4_0/q4_0 or q8_0/q8_0. (KV must be multiple of 256; mask padded to GGML_KQ_MASK_PAD)",
                       D, DV, ggml_type_name(K->type), ggml_type_name(V->type));
        }

        // 3) F16 K/V path
        if (kv_is_f16) {
            if (force_tile) {
                log_tile_once();
                ggml_cuda_flash_attn_ext_tile(ctx, dst);
                return;
            }

            const bool want_fp16 = (prec == GGML_PREC_DEFAULT) || force_vec;
            if (det_vec_supported(dst, want_fp16)) {
                if (want_fp16) ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
                else           ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
                return;
            }
            // vec not available for this F16 case -> deterministic tile fallback
            log_tile_once();
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            return;
        }

        // 4) Any other combination: not supported deterministically
        GGML_ABORT("deterministic attention: unsupported K/V types in det mode (K=%s, V=%s). Use F16 or supported quantized pairs. (KV must be multiple of 256; mask padded to GGML_KQ_MASK_PAD)",
                   ggml_type_name(K->type), ggml_type_name(V->type));
    }

    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC_F32:
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC_F16:
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
