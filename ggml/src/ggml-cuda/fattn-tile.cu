#include "common.cuh"
#include "fattn-tile.cuh"

template <ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_tile_switch_head_size(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 40,  40, type_K, type_V>(ctx, dst);
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 64,  64, type_K, type_V>(ctx, dst);
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 72,  72, type_K, type_V>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 80,  80, type_K, type_V>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case< 96,  96, type_K, type_V>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<112, 112, type_K, type_V>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<128, 128, type_K, type_V>(ctx, dst);
        } break;
        case 192: {
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_tile_case<192, 128, type_K, type_V>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<256, 256, type_K, type_V>(ctx, dst);
        } break;
        case 320: {
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_tile_case<320, 256, type_K, type_V>(ctx, dst);
        } break;
        case 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case<512, 512, type_K, type_V>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<576, 512, type_K, type_V>(ctx, dst);
        } break;
        default:
            GGML_ABORT("Unsupported head size");
    }
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // q8_0 K/V are dequantized in-kernel during the global->shared copy, no f16 scratch is needed.
    // The raw readers produce half2, so pre-Volta NVIDIA (fp32 accumulation) keeps the f16 scratch path.
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool fast_fp16 = fast_fp16_available(cc);

    if constexpr (GGML_CUDA_FA_Q8_0_Q8_0) {
        if (fast_fp16 && ggml_cuda_fattn_kv_native_q8_0(K, V)) {
            ggml_cuda_flash_attn_ext_tile_switch_head_size<GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst);
            return;
        }
    }

    if constexpr (GGML_CUDA_FA_F16_Q8_0) {
        if (fast_fp16 && ggml_cuda_fattn_kv_native_v_q8_0(K, V)) {
            ggml_cuda_flash_attn_ext_tile_switch_head_size<GGML_TYPE_F16, GGML_TYPE_Q8_0>(ctx, dst);
            return;
        }
    }

    if constexpr (GGML_CUDA_FA_F16_F16) {
        if (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
            ggml_cuda_flash_attn_ext_tile_switch_head_size<GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst);
            return;
        }
    }

    // Fallback: convert K and V to F16.
    ggml_cuda_flash_attn_ext_tile_switch_head_size<GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst);
}
