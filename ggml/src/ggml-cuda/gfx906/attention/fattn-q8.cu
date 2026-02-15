// Q8 Flash Attention dispatch: head sizes 64, 96, 128, 256, 576

#include "../../common.cuh"
#include "fattn-q8.cuh"

void ggml_cuda_flash_attn_ext_tile_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_q8_case< 64,  64>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_q8_case< 96,  96>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_q8_case<128, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_q8_case<256, 256>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_q8_case<576, 512>(ctx, dst);
        } break;
        default: {
            GGML_ABORT("Unsupported head size for Q8 tile kernel");
        } break;
    }
}
