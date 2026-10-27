#include "common.cuh"
#include "fattn-tile.cuh"

#define FATTN_TILE_QUANT_CASE(D, type_K_case, type_V_case)                                                                         \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        if (head_size == (D) && type_K == GGML_TYPE_##type_K_case && type_V == GGML_TYPE_##type_V_case) {                          \
            return ggml_cuda_get_fattn_tile_quant_kernel<D, D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>(ncols1, ncols2);  \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_TILE_QUANT_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_TILE_QUANT_CASE( 64, type_K_case, type_V_case)       \
    FATTN_TILE_QUANT_CASE(128, type_K_case, type_V_case)       \
    FATTN_TILE_QUANT_CASE(256, type_K_case, type_V_case)       \

fattn_kernel_t ggml_cuda_get_fattn_tile_quant_case_kernel(
        const int64_t head_size, const int ncols1, const int ncols2,
        const ggml_type type_K_in, const ggml_type type_V_in) {
    const ggml_type type_K = type_K_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_K_in;
    const ggml_type type_V = type_V_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_V_in;

    FATTN_TILE_QUANT_CASES_ALL_D(Q4_0, F16)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_1, F16)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_0, F16)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_1, F16)
    FATTN_TILE_QUANT_CASES_ALL_D(Q8_0, F16)
    FATTN_TILE_QUANT_CASES_ALL_D(BF16, F16)

    FATTN_TILE_QUANT_CASES_ALL_D(F16,  Q4_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_TILE_QUANT_CASES_ALL_D(BF16, Q4_0)

    FATTN_TILE_QUANT_CASES_ALL_D(F16,  Q4_1)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_TILE_QUANT_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_TILE_QUANT_CASES_ALL_D(BF16, Q4_1)

    FATTN_TILE_QUANT_CASES_ALL_D(F16,  Q5_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_TILE_QUANT_CASES_ALL_D(BF16, Q5_0)

    FATTN_TILE_QUANT_CASES_ALL_D(F16,  Q5_1)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_TILE_QUANT_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_TILE_QUANT_CASES_ALL_D(BF16, Q5_1)

    FATTN_TILE_QUANT_CASES_ALL_D(F16,  Q8_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_TILE_QUANT_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_TILE_QUANT_CASES_ALL_D(BF16, Q8_0)

    FATTN_TILE_QUANT_CASES_ALL_D(F16,  BF16)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_0, BF16)
    FATTN_TILE_QUANT_CASES_ALL_D(Q4_1, BF16)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_0, BF16)
    FATTN_TILE_QUANT_CASES_ALL_D(Q5_1, BF16)
    FATTN_TILE_QUANT_CASES_ALL_D(Q8_0, BF16)
    FATTN_TILE_QUANT_CASES_ALL_D(BF16, BF16)

    return nullptr;
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } break;
        case 192: {
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_tile_case<192, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } break;
        case 320: {
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_tile_case<320, 256>(ctx, dst);
        } break;
        case 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<512, 512>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } break;
        default:
            GGML_ABORT("Unsupported head size");
    }
}
