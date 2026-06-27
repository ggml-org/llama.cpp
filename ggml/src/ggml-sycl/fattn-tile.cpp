#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile.hpp"
#include <cmath>
#include <float.h>
namespace syclex = sycl::ext::oneapi::experimental;

void ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K->type != GGML_TYPE_TURBO2_0 && K->type != GGML_TYPE_TURBO3_0 && K->type != GGML_TYPE_TURBO4_0 &&
                V->type != GGML_TYPE_TURBO2_0 && V->type != GGML_TYPE_TURBO3_0 && V->type != GGML_TYPE_TURBO4_0 &&
                "turbo KV must route to VEC");

    const int type_K = K->type;

    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            if (type_K != GGML_TYPE_F16) {
                 GGML_ABORT("TurboQuant not supported for head size 40");
            } else {
                 ggml_sycl_flash_attn_ext_tile_case< 40,  40, GGML_TYPE_F16>(ctx, dst);
            }
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            if (type_K == GGML_TYPE_TURBO2_0) {
                 ggml_sycl_flash_attn_ext_tile_case< 64,  64, GGML_TYPE_TURBO2_0>(ctx, dst);
            } else if (type_K == GGML_TYPE_TURBO3_0) {
                 ggml_sycl_flash_attn_ext_tile_case< 64,  64, GGML_TYPE_TURBO3_0>(ctx, dst);
            } else if (type_K == GGML_TYPE_TURBO4_0) {
                 ggml_sycl_flash_attn_ext_tile_case< 64,  64, GGML_TYPE_TURBO4_0>(ctx, dst);
            } else {
                 ggml_sycl_flash_attn_ext_tile_case< 64,  64, GGML_TYPE_F16>(ctx, dst);
            }
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 72,  72, GGML_TYPE_F16>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 80,  80, GGML_TYPE_F16>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case< 96,  96, GGML_TYPE_F16>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_sycl_flash_attn_ext_tile_case<112, 112, GGML_TYPE_F16>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            if (type_K == GGML_TYPE_TURBO2_0) {
                 ggml_sycl_flash_attn_ext_tile_case<128, 128, GGML_TYPE_TURBO2_0>(ctx, dst);
            } else if (type_K == GGML_TYPE_TURBO3_0) {
                 ggml_sycl_flash_attn_ext_tile_case<128, 128, GGML_TYPE_TURBO3_0>(ctx, dst);
            } else if (type_K == GGML_TYPE_TURBO4_0) {
                 ggml_sycl_flash_attn_ext_tile_case<128, 128, GGML_TYPE_TURBO4_0>(ctx, dst);
            } else {
                 ggml_sycl_flash_attn_ext_tile_case<128, 128, GGML_TYPE_F16>(ctx, dst);
            }
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            if (type_K == GGML_TYPE_TURBO2_0) {
                 ggml_sycl_flash_attn_ext_tile_case<256, 256, GGML_TYPE_TURBO2_0>(ctx, dst);
            } else if (type_K == GGML_TYPE_TURBO3_0) {
                 ggml_sycl_flash_attn_ext_tile_case<256, 256, GGML_TYPE_TURBO3_0>(ctx, dst);
            } else if (type_K == GGML_TYPE_TURBO4_0) {
                 ggml_sycl_flash_attn_ext_tile_case<256, 256, GGML_TYPE_TURBO4_0>(ctx, dst);
            } else {
                 ggml_sycl_flash_attn_ext_tile_case<256, 256, GGML_TYPE_F16>(ctx, dst);
            }
        } break;
        case 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            if (type_K == GGML_TYPE_TURBO2_0) {
                 ggml_sycl_flash_attn_ext_tile_case<512, 512, GGML_TYPE_TURBO2_0>(ctx, dst);
            } else if (type_K == GGML_TYPE_TURBO3_0) {
                 ggml_sycl_flash_attn_ext_tile_case<512, 512, GGML_TYPE_TURBO3_0>(ctx, dst);
            } else if (type_K == GGML_TYPE_TURBO4_0) {
                 ggml_sycl_flash_attn_ext_tile_case<512, 512, GGML_TYPE_TURBO4_0>(ctx, dst);
            } else {
                 ggml_sycl_flash_attn_ext_tile_case<512, 512, GGML_TYPE_F16>(ctx, dst);
            }
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_flash_attn_ext_tile_case<576, 512, GGML_TYPE_F16>(ctx, dst);
        } break;
        default: {
            GGML_ABORT("Unsupported head size");
        } break;
    }
}
