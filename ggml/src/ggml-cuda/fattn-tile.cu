#include "common.cuh"
#include "fattn-tile.cuh"

// Non-f16 instances are only compiled for HIP, see ggml_cuda_fattn_instances. Without one the caller uses the f16 case.
#define FATTN_TILE_CASE(DKQ, DV, type_K_case, type_V_case)                                                  \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case && (GGML_CUDA_FA_TILE_NATIVE_KV ||             \
            (GGML_TYPE_##type_K_case == GGML_TYPE_F16 && GGML_TYPE_##type_V_case == GGML_TYPE_F16))) {      \
        if (head_size_kq == (DKQ) && head_size_v == (DV) &&                                                 \
                type_K == GGML_TYPE_##type_K_case && type_V == GGML_TYPE_##type_V_case) {                   \
            return ggml_cuda_flash_attn_ext_tile_case<DKQ, DV, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>; \
        }                                                                                                   \
    }                                                                                                       \

#define FATTN_TILE_CASES_FP(type_K_case, type_V_case)        \
    FATTN_TILE_CASE( 40,  40, type_K_case, type_V_case)      \
    FATTN_TILE_CASE( 64,  64, type_K_case, type_V_case)      \
    FATTN_TILE_CASE( 72,  72, type_K_case, type_V_case)      \
    FATTN_TILE_CASE( 80,  80, type_K_case, type_V_case)      \
    FATTN_TILE_CASE( 96,  96, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(112, 112, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(128, 128, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(192, 128, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(256, 256, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(320, 256, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(512, 512, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(576, 512, type_K_case, type_V_case)      \

#define FATTN_TILE_CASES_QUANT(type_K_case, type_V_case)     \
    FATTN_TILE_CASE( 64,  64, type_K_case, type_V_case)      \
    FATTN_TILE_CASE( 96,  96, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(128, 128, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(192, 128, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(256, 256, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(320, 256, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(512, 512, type_K_case, type_V_case)      \
    FATTN_TILE_CASE(576, 512, type_K_case, type_V_case)      \

fattn_tile_case_t ggml_cuda_get_fattn_tile_case(
        const int64_t head_size_kq, const int64_t head_size_v,
        const ggml_type type_K_in, const ggml_type type_V_in) {
    const ggml_type type_K = type_K_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_K_in;
    const ggml_type type_V = type_V_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_V_in;

    FATTN_TILE_CASES_FP   (F16,  F16)
    FATTN_TILE_CASES_QUANT(Q4_0, F16)
    FATTN_TILE_CASES_QUANT(Q4_1, F16)
    FATTN_TILE_CASES_QUANT(Q5_0, F16)
    FATTN_TILE_CASES_QUANT(Q5_1, F16)
    FATTN_TILE_CASES_QUANT(Q8_0, F16)
    FATTN_TILE_CASES_FP   (BF16, F16)

    FATTN_TILE_CASES_QUANT(F16,  Q4_0)
    FATTN_TILE_CASES_QUANT(Q4_0, Q4_0)
    FATTN_TILE_CASES_QUANT(Q4_1, Q4_0)
    FATTN_TILE_CASES_QUANT(Q5_0, Q4_0)
    FATTN_TILE_CASES_QUANT(Q5_1, Q4_0)
    FATTN_TILE_CASES_QUANT(Q8_0, Q4_0)
    FATTN_TILE_CASES_QUANT(BF16, Q4_0)

    FATTN_TILE_CASES_QUANT(F16,  Q4_1)
    FATTN_TILE_CASES_QUANT(Q4_0, Q4_1)
    FATTN_TILE_CASES_QUANT(Q4_1, Q4_1)
    FATTN_TILE_CASES_QUANT(Q5_0, Q4_1)
    FATTN_TILE_CASES_QUANT(Q5_1, Q4_1)
    FATTN_TILE_CASES_QUANT(Q8_0, Q4_1)
    FATTN_TILE_CASES_QUANT(BF16, Q4_1)

    FATTN_TILE_CASES_QUANT(F16,  Q5_0)
    FATTN_TILE_CASES_QUANT(Q4_0, Q5_0)
    FATTN_TILE_CASES_QUANT(Q4_1, Q5_0)
    FATTN_TILE_CASES_QUANT(Q5_0, Q5_0)
    FATTN_TILE_CASES_QUANT(Q5_1, Q5_0)
    FATTN_TILE_CASES_QUANT(Q8_0, Q5_0)
    FATTN_TILE_CASES_QUANT(BF16, Q5_0)

    FATTN_TILE_CASES_QUANT(F16,  Q5_1)
    FATTN_TILE_CASES_QUANT(Q4_0, Q5_1)
    FATTN_TILE_CASES_QUANT(Q4_1, Q5_1)
    FATTN_TILE_CASES_QUANT(Q5_0, Q5_1)
    FATTN_TILE_CASES_QUANT(Q5_1, Q5_1)
    FATTN_TILE_CASES_QUANT(Q8_0, Q5_1)
    FATTN_TILE_CASES_QUANT(BF16, Q5_1)

    FATTN_TILE_CASES_QUANT(F16,  Q8_0)
    FATTN_TILE_CASES_QUANT(Q4_0, Q8_0)
    FATTN_TILE_CASES_QUANT(Q4_1, Q8_0)
    FATTN_TILE_CASES_QUANT(Q5_0, Q8_0)
    FATTN_TILE_CASES_QUANT(Q5_1, Q8_0)
    FATTN_TILE_CASES_QUANT(Q8_0, Q8_0)
    FATTN_TILE_CASES_QUANT(BF16, Q8_0)

    FATTN_TILE_CASES_FP   (F16,  BF16)
    FATTN_TILE_CASES_QUANT(Q4_0, BF16)
    FATTN_TILE_CASES_QUANT(Q4_1, BF16)
    FATTN_TILE_CASES_QUANT(Q5_0, BF16)
    FATTN_TILE_CASES_QUANT(Q5_1, BF16)
    FATTN_TILE_CASES_QUANT(Q8_0, BF16)
    FATTN_TILE_CASES_FP   (BF16, BF16)

    return nullptr;
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_tile_case_t tile_case = ggml_cuda_fattn_tile_use_native_kv(cc, dst) ?
        ggml_cuda_get_fattn_tile_case(K->ne[0], V->ne[0], K->type, V->type) : nullptr;
    if (tile_case == nullptr) {
        tile_case = ggml_cuda_get_fattn_tile_case(K->ne[0], V->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(tile_case != nullptr);
    tile_case(ctx, dst);
}
