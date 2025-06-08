#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile-f16.cuh"
#include "fattn-tile-f32.cuh"
#include "fattn-vec-f16.cuh"
#include "fattn-vec-f32.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

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

    on_no_fattn_vec_case(Q->ne[0]);
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

    on_no_fattn_vec_case(Q->ne[0]);
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst; // KQV is a convention where dst itself might hold op_params like scale, bias
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_set_device(ctx.device);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const enum ggml_prec prec = ggml_flash_attn_ext_get_prec(KQV);

    // Check for paged attention flag in op_params
    // Let's define GGML_FLASH_ATTN_EXT_OP_PARAMS_IS_PAGED_IDX as 3 (0: scale, 1: max_bias, 2: logit_softcap)
    // This index should be centrally defined in ggml.h or similar eventually.
    const int GGML_FLASH_ATTN_EXT_OP_PARAMS_IS_PAGED_IDX = 3; // Or another available index
    bool is_paged_call = false;
    if (KQV->op_params[GGML_FLASH_ATTN_EXT_OP_PARAMS_IS_PAGED_IDX] != 0) {
        // Assuming a non-zero value (e.g., 1) indicates a paged call.
        // The specific value might be an enum or flags in the future.
        // For now, let's assume op_params is float[4] and we use the last float as a flag.
        // A more robust way would be to ensure op_params is large enough and the index is defined.
        // For this change, we'll assume op_params[3] (if it's float) being non-zero means paged.
        // A safer way is to check if the value is exactly a specific flag, e.g. 1.0f
        float paged_flag_val;
        memcpy(&paged_flag_val, (const float *) KQV->op_params + GGML_FLASH_ATTN_EXT_OP_PARAMS_IS_PAGED_IDX, sizeof(float));
        if (paged_flag_val == 1.0f) { // Example: 1.0f indicates paged
            is_paged_call = true;
        }
    }


    if (is_paged_call) {
        GGML_LOG_DEBUG("%s: Paged Flash Attention path selected.\n", __func__);
        const paged_kv_sequence_view_host_for_gpu* k_view_host = (const paged_kv_sequence_view_host_for_gpu*)K->extra;
        const paged_kv_sequence_view_host_for_gpu* v_view_host = (const paged_kv_sequence_view_host_for_gpu*)V->extra;

        if (k_view_host == nullptr || v_view_host == nullptr) {
            GGML_LOG_ERROR("%s: K or V tensor extra data is null for paged attention call.\n", __func__);
            GGML_ABORT("fatal error: K/V extra data missing in paged attention");
            return;
        }
        if (k_view_host->token_mappings_gpu_ptr == nullptr || k_view_host->page_pool_gpu_ptr == nullptr ||
            v_view_host->token_mappings_gpu_ptr == nullptr || v_view_host->page_pool_gpu_ptr == nullptr) {
            if (k_view_host->num_tokens_in_logical_sequence > 0) { // only error if sequence is not empty
                 GGML_LOG_ERROR("%s: K or V view internal GPU pointers are null for paged attention call with non-empty sequence.\n", __func__);
                 GGML_ABORT("fatal error: K/V view GPU pointers missing in paged attention");
                 return;
            }
        }

        paged_kv_sequence_view_gpu k_view_gpu_kernel_arg;
        paged_kv_sequence_view_gpu v_view_gpu_kernel_arg;

        // Populate kernel args from host views
        k_view_gpu_kernel_arg.token_mappings = (const paged_kv_token_mapping_gpu*)k_view_host->token_mappings_gpu_ptr;
        k_view_gpu_kernel_arg.page_pool_gpu = (const void**)k_view_host->page_pool_gpu_ptr;
        k_view_gpu_kernel_arg.num_tokens_in_logical_sequence = k_view_host->num_tokens_in_logical_sequence;
        k_view_gpu_kernel_arg.dtype = k_view_host->dtype;
        k_view_gpu_kernel_arg.k_head_size_elements = k_view_host->k_head_size_elements;
        k_view_gpu_kernel_arg.v_head_size_elements = k_view_host->v_head_size_elements;
        k_view_gpu_kernel_arg.num_k_heads_total = k_view_host->num_k_heads_total;
        k_view_gpu_kernel_arg.num_v_heads_total = k_view_host->num_v_heads_total;
        k_view_gpu_kernel_arg.element_size_bytes = k_view_host->element_size_bytes;
        k_view_gpu_kernel_arg.page_size_bytes = k_view_host->page_size_bytes;
        k_view_gpu_kernel_arg.v_block_start_offset_bytes = k_view_host->v_block_start_offset_bytes;

        v_view_gpu_kernel_arg.token_mappings = (const paged_kv_token_mapping_gpu*)v_view_host->token_mappings_gpu_ptr;
        v_view_gpu_kernel_arg.page_pool_gpu = (const void**)v_view_host->page_pool_gpu_ptr;
        v_view_gpu_kernel_arg.num_tokens_in_logical_sequence = v_view_host->num_tokens_in_logical_sequence;
        v_view_gpu_kernel_arg.dtype = v_view_host->dtype;
        v_view_gpu_kernel_arg.k_head_size_elements = v_view_host->k_head_size_elements;
        v_view_gpu_kernel_arg.v_head_size_elements = v_view_host->v_head_size_elements;
        v_view_gpu_kernel_arg.num_k_heads_total = v_view_host->num_k_heads_total;
        v_view_gpu_kernel_arg.num_v_heads_total = v_view_host->num_v_heads_total;
        v_view_gpu_kernel_arg.element_size_bytes = v_view_host->element_size_bytes;
        v_view_gpu_kernel_arg.page_size_bytes = v_view_host->page_size_bytes;
        v_view_gpu_kernel_arg.v_block_start_offset_bytes = v_view_host->v_block_start_offset_bytes;

        paged_kv_sequence_view_gpu* d_k_view_gpu_kernel_arg = nullptr;
        paged_kv_sequence_view_gpu* d_v_view_gpu_kernel_arg = nullptr;

        CUDA_CHECK(cudaMalloc((void**)&d_k_view_gpu_kernel_arg, sizeof(paged_kv_sequence_view_gpu)));
        CUDA_CHECK(cudaMalloc((void**)&d_v_view_gpu_kernel_arg, sizeof(paged_kv_sequence_view_gpu)));

        cudaStream_t stream = ctx.stream();
        CUDA_CHECK(cudaMemcpyAsync(d_k_view_gpu_kernel_arg, &k_view_gpu_kernel_arg, sizeof(paged_kv_sequence_view_gpu), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_v_view_gpu_kernel_arg, &v_view_gpu_kernel_arg, sizeof(paged_kv_sequence_view_gpu), cudaMemcpyHostToDevice, stream));
        // No cudaStreamSynchronize here, let the kernel launch wait on the stream if needed.

        ggml_cuda_flash_attn_ext_paged(ctx, dst, d_k_view_gpu_kernel_arg, d_v_view_gpu_kernel_arg);

        // Synchronize necessary for safe free if kernel is also async
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_k_view_gpu_kernel_arg));
        CUDA_CHECK(cudaFree(d_v_view_gpu_kernel_arg));
        return;
    }

    // Original non-paged dispatch logic
    if (GGML_CUDA_CC_IS_AMD(cc)) {
#if defined(GGML_HIP_ROCWMMA_FATTN)
        if (fp16_mma_available(cc)) {
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            return;
        }
#endif // defined(GGML_HIP_ROCWMMA_FATTN)

        // On AMD the tile kernels perform poorly, use the vec kernel instead:
        if (prec == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
        }
        return;
    }

    if (!fp16_mma_available(cc)) {
        if (prec == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            }
        }
        return;
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < GGML_CUDA_CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % (2*warp_size) == 0; // Check if head dim is suitable for vector kernels
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1) { // If batch size 1 and vector kernel is suitable and MMA is not clearly faster
        if (prec == GGML_PREC_DEFAULT) { // Prefer F16 for default precision if available
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else { // Otherwise use F32 vector kernel
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    // The MMA implementation needs Turing or newer, use the old WMMA code for Volta:
    if (fp16_mma_available(cc) && !new_mma_available(cc)) { // If only WMMA is available (e.g., Volta)
        ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
        return;
    }
    // Default to MMA-based kernels for newer architectures
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
}

// PAGED KV CACHE IMPLEMENTATION STARTS HERE

// Placeholder for page mapping information (conceptual)
// These structures would be populated by the host and their data copied to the GPU.
struct paged_kv_token_mapping_gpu {
    int page_idx;               // Index of the page in the page_pool_gpu array
    int offset_in_page_elements; // Offset in terms of elements (e.g., fp16) from the start of the page
    // int V_page_idx;            // Separate page index for V if K and V are in different page pools
    // int V_offset_in_page_elements;
};

struct paged_kv_sequence_view_gpu {
    const paged_kv_token_mapping_gpu* token_mappings; // GPU pointer to an array of mappings for each token in the logical sequence. [max_seq_len]
    const void** page_pool_gpu;                      // GPU pointer to an array of base pointers for each physical page. [num_physical_pages]
                                                     // For K and V, this pool would contain pointers to half* or float* depending on type.
    // const void** V_page_pool_gpu;                 // Separate pool for V if needed.
    int32_t num_tokens_in_logical_sequence;          // Current number of tokens in this specific sequence (n_past + n_seq_curr for this call)
    ggml_type dtype;                                 // Data type of K/V cache (e.g. GGML_TYPE_F16)
};

// Forward declarations for paged versions of dispatch functions
// (mirroring the structure of the non-paged versions)

template <int DKQ, int DV, int NCOLS1, int NCOLS2>
static void ggml_cuda_flash_attn_ext_mma_f16_case_paged(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    const paged_kv_sequence_view_gpu * k_paged_view,
    const paged_kv_sequence_view_gpu * v_paged_view);

template <int DKQ, int DV, int NCOLS1>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1_paged(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    const paged_kv_sequence_view_gpu * k_paged_view,
    const paged_kv_sequence_view_gpu * v_paged_view) {

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    // NCOLS2 is the GQA ratio for K heads per Q head group, essentially.
    // This dispatch logic seems to select kernel variants based on Q length and NCOLS2 (GQA factor).
    if constexpr (NCOLS1 <= 8) { // NCOLS1 appears to be related to Q sequence length processing blocks
        if (Q->ne[1] <= 8/NCOLS1) {
            ggml_cuda_flash_attn_ext_mma_f16_case_paged<DKQ, DV, 8/NCOLS1, NCOLS1>(ctx, dst, k_paged_view, v_paged_view);
            return;
        }
    }

    if (Q->ne[1] <= 16/NCOLS1) {
        ggml_cuda_flash_attn_ext_mma_f16_case_paged<DKQ, DV, 16/NCOLS1, NCOLS1>(ctx, dst, k_paged_view, v_paged_view);
        return;
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || Q->ne[1] <= 32/NCOLS1) {
        ggml_cuda_flash_attn_ext_mma_f16_case_paged<DKQ, DV, 32/NCOLS1, NCOLS1>(ctx, dst, k_paged_view, v_paged_view);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case_paged<DKQ, DV, 64/NCOLS1, NCOLS1>(ctx, dst, k_paged_view, v_paged_view);
}


template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2_paged(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    const paged_kv_sequence_view_gpu * k_paged_view,
    const paged_kv_sequence_view_gpu * v_paged_view) {

    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K_tensor_metadata = dst->src[1]; // Original K tensor for metadata, not data
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    const bool use_gqa_opt = mask && max_bias == 0.0f;

    GGML_ASSERT(Q->ne[2] % K_tensor_metadata->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K_tensor_metadata->ne[2];

    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1_paged<DKQ, DV, 8>(ctx, dst, k_paged_view, v_paged_view);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1_paged<DKQ, DV, 4>(ctx, dst, k_paged_view, v_paged_view);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1_paged<DKQ, DV, 2>(ctx, dst, k_paged_view, v_paged_view);
        return;
    }
    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1_paged<DKQ, DV, 1>(ctx, dst, k_paged_view, v_paged_view);
}

static void ggml_cuda_flash_attn_ext_mma_f16_paged(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    const paged_kv_sequence_view_gpu * k_paged_view,
    const paged_kv_sequence_view_gpu * v_paged_view) {

    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K_tensor_metadata = dst->src[1]; // Original K tensor for metadata
    const ggml_tensor * V_tensor_metadata = dst->src[2]; // Original V tensor for metadata
    const ggml_tensor * mask = dst->src[3];

    // Dispatch based on Q head dimension (DKQ) and V head dimension (DV)
    // This logic is identical to the original ggml_cuda_flash_attn_ext_mma_f16
    // It just passes k_paged_view and v_paged_view along.
    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V_tensor_metadata->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2_paged<64, 64>(ctx, dst, k_paged_view, v_paged_view);
            break;
        case 80:
            GGML_ASSERT(V_tensor_metadata->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2_paged<80, 80>(ctx, dst, k_paged_view, v_paged_view);
            break;
        // ... (other cases from original function) ...
        case 128:
            GGML_ASSERT(V_tensor_metadata->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2_paged<128, 128>(ctx, dst, k_paged_view, v_paged_view);
            break;
        case 256:
            GGML_ASSERT(V_tensor_metadata->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2_paged<256, 256>(ctx, dst, k_paged_view, v_paged_view);
            break;
        // TODO: Add all cases from the original function.
        // For brevity in this example, only a few are included.
        case 576: {
            GGML_ASSERT(V_tensor_metadata->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K_tensor_metadata->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K_tensor_metadata->ne[2];
            GGML_ASSERT(gqa_ratio % 16 == 0);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1_paged<576, 512, 16>(ctx, dst, k_paged_view, v_paged_view);
        } break;
        default:
            fprintf(stderr, "%s: Head dimension %" PRId64 " not supported for paged MMA F16 FA\n", __func__, Q->ne[0]);
            GGML_ABORT("fatal error");
            break;
    }
}

// Placeholder for the actual paged kernel cases.
// These would call the __global__ kernels with paged parameters.
template <int DKQ, int DV, int NCOLS1, int NCOLS2>
static void ggml_cuda_flash_attn_ext_mma_f16_case_paged(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    const paged_kv_sequence_view_gpu * k_paged_view,
    const paged_kv_sequence_view_gpu * v_paged_view) {
    // In a real implementation, this function would be similar to
    // ggml_cuda_flash_attn_ext_mma_f16_case, but it would:
    // 1. Extract Q, K_metadata, V_metadata, mask, scale, bias from dst.
    // 2. Launch a new templated __global__ kernel (e.g., flash_attention_mma_f16_paged_kernel)
    // 3. Pass Q->data, K_metadata, V_metadata as before for dimensions, strides etc.
    // 4. Critically, it passes k_paged_view and v_paged_view to the kernel.
    // This function will now call the __global__ paged kernel using launch_fattn_paged.
    // Original non-paged: ggml_cuda_flash_attn_ext_mma_f16_case
    // Paged version: ggml_cuda_flash_attn_ext_mma_f16_case_paged

    const ggml_tensor * KQV_tensor = dst; // Output tensor, also holds op_params
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    typedef fattn_mma_f16_config<DKQ, DV> config; // Kernel specific configurations

    const int nstages = cp_async_available(cc) ? config::nstages_target : 0; // For shared memory calculation if needed by kernel

    constexpr int ncols = NCOLS1 * NCOLS2;
    constexpr int ntiles = ncols <= 8 ? 1 : 2; // Number of tiles per warp, from original logic
    constexpr int nwarps_max_x  = ncols / (ntiles * tile_B::I); // tile_B::I is likely 16 (elements in tile width)
    constexpr int nwarps_max_y  = config::nbatch_fa / tile_A::I; // tile_A::I is likely 16 (elements in tile height)
    constexpr int nwarps_kernel = (nwarps_max_x * nwarps_max_y <= config::nwarps_max) ? (nwarps_max_x * nwarps_max_y) : config::nwarps_max;

    constexpr bool mla = (DKQ == 576 && DV == 512); // Example, specific to certain head dims from original kernel

    // Calculate shared memory (example, needs to match what the paged kernel expects)
    // This is complex and depends on the kernel's internal structure.
    // The original kernel calculates it based on tile_Q, tile_K, tile_V, tile_mask sizes.
    // For paged, tile_K and tile_V might be smaller if data is processed in sub-batches due to gather.
    // For this sketch, we'll use a placeholder or simplified shared memory calculation.
    // A more accurate calculation would be:
    // size_t nbytes_shared_Q = (config::Q_in_reg ? 0 : ncols * (DKQ/2 + 4)) * sizeof(half2);
    // size_t nbytes_shared_K_tile = config::nbatch_fa * (config::get_nbatch_K2_device(ncols) + 4) * sizeof(half2);
    // size_t nbytes_shared_V_tile = config::nbatch_fa * (config::get_nbatch_V2_device(ncols) + 4) * sizeof(half2);
    // ... and so on for mask, combine buffers.
    // This is highly dependent on the paged kernel's specific shared memory strategy.
    // For now, let's reuse part of the logic from the original launcher.
    const size_t nbatch_K2_sh = config::get_nbatch_K2_host(cc, ncols); // Or _device version if used by kernel
    const size_t nbatch_V2_sh = config::get_nbatch_V2_host(cc, ncols);
    const size_t nbatch_combine_sh = config::get_nbatch_combine_host(cc, ncols);

    const size_t nbytes_shared_KV_1stage = config::nbatch_fa * std::max(nbatch_K2_sh + 4,  nbatch_V2_sh + 4) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = config::nbatch_fa * (nbatch_K2_sh + 4 + nbatch_V2_sh + 4) * sizeof(half2);
    const size_t nbytes_shared_Q_sh      = ncols * (DKQ/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_mask_sh   = NCOLS1 * (config::nbatch_fa/2 + 4) * sizeof(half2);
    const size_t nbytes_shared_combine_sh= nwarps_kernel * (ntiles * tile_B::I) * (nbatch_combine_sh + 4) * sizeof(half2);

    const size_t nbytes_shared_KV_eff = (nstages <= 1) ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;
    size_t nbytes_shared_total = std::max(nbytes_shared_combine_sh, (config::Q_in_reg ? 0 : nbytes_shared_Q_sh) + nbytes_shared_KV_eff + nbytes_shared_mask_sh);
    // This calculation is illustrative and needs to exactly match the __shared__ memory usage of the paged kernel.

    float logit_softcap_param;
    memcpy(&logit_softcap_param, (const float *) KQV_tensor->op_params + 2, sizeof(float));

    fattn_paged_kernel_t kernel_ptr;
    if (logit_softcap_param == 0.0f) {
        constexpr bool use_logit_softcap_template = false;
        kernel_ptr = flash_attn_ext_f16_paged<DKQ, DV, NCOLS1, NCOLS2, nwarps_kernel, ntiles, use_logit_softcap_template, mla>;
    } else {
        constexpr bool use_logit_softcap_template = true;
        kernel_ptr = flash_attn_ext_f16_paged<DKQ, DV, NCOLS1, NCOLS2, nwarps_kernel, ntiles, use_logit_softcap_template, mla>;
    }

    // The stream_k parameter in launch_fattn determines grid size and fixup logic.
    // This needs careful consideration for paged attention.
    // Assuming stream_k = true for now (simpler grid calculation, may need fixup kernel later)
    // The KQ_row_granularity for paged is FATTN_KQ_STRIDE (max tokens processed per block before fixup/normalization)
    launch_fattn_paged<DV, NCOLS1, NCOLS2>(
        ctx, dst,
        *k_paged_view, *v_paged_view,
        kernel_ptr,
        nwarps_kernel, nbytes_shared_total,
        FATTN_KQ_STRIDE, // KQ_row_granularity
        true // stream_k (influences grid calculation and fixup)
    );
}


// TODO: Similarly define paged versions for _vec_f32, _tile_f16, _tile_f32, _wmma_f16

// Example for Vector F16 paged case
template <int D, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_vec_f16_case_paged(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    const paged_kv_sequence_view_gpu * k_paged_view,
    const paged_kv_sequence_view_gpu * v_paged_view) {

    const ggml_tensor * KQV_tensor = dst;
    const ggml_tensor * Q_tensor = dst->src[0];
    const ggml_tensor * K_meta_tensor = dst->src[1];
    // const ggml_tensor * V_meta_tensor = dst->src[2]; // For V metadata if needed by launcher

    float logit_softcap_param;
    memcpy(&logit_softcap_param, (const float *) KQV_tensor->op_params + 2, sizeof(float));

    // Determine ncols based on Q_tensor->ne[1] (q_seq_len) for vector kernels
    // This logic is from the original non-paged ggml_cuda_flash_attn_ext_vec_f16_case
    int ncols_param = 1; // Default for Q_seq_len == 1
    if (Q_tensor->ne[1] == 2) {
        ncols_param = 2;
    } else if (Q_tensor->ne[1] <= 4) {
        ncols_param = 4;
    } else if (Q_tensor->ne[1] <=8) { // Matches original logic more closely
        ncols_param = 8;
    }
    // If Q_tensor->ne[1] > 8, original uses ncols=8. This means a single kernel invocation processes
    // at most 8 Q elements (tokens) against the K/V cache. If Q_tensor->ne[1] is larger,
    // the host code (ggml_metal_flash_attn_ext) iterates, slicing Q.
    // The `launch_fattn_paged` will need to be aware of this if ncols_param is passed to it.

    fattn_paged_kernel_t kernel_ptr;
    if (logit_softcap_param == 0.0f) {
        constexpr bool use_logit_softcap_template = false;
        // The actual kernel selected here depends on D, type_K_dummy, type_V_dummy, and ncols_param from template instantiation
        // This is a placeholder for the correct template instantiation for the paged vector kernel.
        // Example: kernel_ptr = flash_attn_vec_ext_f16_paged<D, ncols_param, K_dummy, V_dummy, use_logit_softcap_template>;
        // Since type_K_dummy and type_V_dummy are part of the template, and k_paged_view/v_paged_view now carry type info,
        // we might need a switch on k_paged_view->dtype / v_paged_view->dtype here if kernels are specialized by type,
        // or the paged kernel itself handles type dispatch internally (less likely for perf).
        // For now, assume template parameters D and ncols are sufficient for a generic F16 paged vector kernel.
        // The dummy types in the kernel template will be ignored.
        kernel_ptr = flash_attn_vec_ext_f16_paged<D, 1 /*ncols_placeholder*/, GGML_TYPE_COUNT, GGML_TYPE_COUNT, use_logit_softcap_template>;
         if (ncols_param == 2) kernel_ptr = flash_attn_vec_ext_f16_paged<D, 2, GGML_TYPE_COUNT, GGML_TYPE_COUNT, use_logit_softcap_template>;
         if (ncols_param == 4) kernel_ptr = flash_attn_vec_ext_f16_paged<D, 4, GGML_TYPE_COUNT, GGML_TYPE_COUNT, use_logit_softcap_template>;
         if (ncols_param == 8) kernel_ptr = flash_attn_vec_ext_f16_paged<D, 8, GGML_TYPE_COUNT, GGML_TYPE_COUNT, use_logit_softcap_template>;

    } else {
        constexpr bool use_logit_softcap_template = true;
        kernel_ptr = flash_attn_vec_ext_f16_paged<D, 1 /*ncols_placeholder*/, GGML_TYPE_COUNT, GGML_TYPE_COUNT, use_logit_softcap_template>;
        if (ncols_param == 2) kernel_ptr = flash_attn_vec_ext_f16_paged<D, 2, GGML_TYPE_COUNT, GGML_TYPE_COUNT, use_logit_softcap_template>;
        if (ncols_param == 4) kernel_ptr = flash_attn_vec_ext_f16_paged<D, 4, GGML_TYPE_COUNT, GGML_TYPE_COUNT, use_logit_softcap_template>;
        if (ncols_param == 8) kernel_ptr = flash_attn_vec_ext_f16_paged<D, 8, GGML_TYPE_COUNT, GGML_TYPE_COUNT, use_logit_softcap_template>;
    }

    GGML_LOG_INFO("%s: Launching STUB Paged Vector F16 kernel (D=%d, K_type=%d, V_type=%d, ncols_param=%d)\n", __func__, D, (int)k_paged_view->dtype, (int)v_paged_view->dtype, ncols_param);

    launch_fattn_paged<D, 1 /*NCOLS1 for launch_fattn_paged, effectively q_tile_size for vector */, 1 /*NCOLS2 for launch_fattn_paged*/>(
        ctx, dst,
        *k_paged_view, *v_paged_view,
        kernel_ptr,
        D / WARP_SIZE, // nwarps for vector kernel is typically head_dim / warp_size
        0,             // shared memory for vector kernel is often 0 or minimal
        D,             // KQ_row_granularity for vector kernel (processes one Q against D K/V elements)
        false          // stream_k (vector kernels usually don't use the same stream_k fixup as MMA)
    );
}

// Definition for Tile F16 paged case
template <int D_kernel_template, ggml_type type_K_dummy, ggml_type type_V_dummy>
static void ggml_cuda_flash_attn_ext_tile_f16_case_paged(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst,
    const paged_kv_sequence_view_gpu * k_view, const paged_kv_sequence_view_gpu * v_view) {

    const ggml_tensor * Q = dst->src[0];
    GGML_ASSERT(Q->ne[0] == D_kernel_template); // Ensure D_kernel_template matches actual Q head dim
    GGML_UNUSED(type_K_dummy); // Not used directly, type comes from k_view
    GGML_UNUSED(type_V_dummy); // Not used directly, type comes from v_view

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // Determine cols_per_block based on Q->ne[1] (n_q)
    // This matches the logic in the non-paged version ggml_cuda_flash_attn_ext_tile_f16
    if (Q->ne[1] <= 16) {
        constexpr int cols_per_block = 16; // This is NCOLS1 for launch_fattn_paged
        constexpr int nwarps_kernel = 8;
        constexpr size_t shared_mem = 0;
        constexpr bool stream_k_flag = false; // Assuming stream_k is false for paged tile for now
        constexpr int kq_granularity = FATTN_KQ_STRIDE_TILE_F16;

        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap_kernel = false;
            launch_fattn_paged<D_kernel_template, cols_per_block, 1>( // DV = D_kernel_template, NCOLS2 = 1
                ctx, dst, *k_view, *v_view,
                flash_attn_tile_ext_f16_paged<D_kernel_template, cols_per_block, nwarps_kernel, use_logit_softcap_kernel>,
                nwarps_kernel, shared_mem, kq_granularity, stream_k_flag);
        } else { // logit_softcap != 0.0f
            constexpr bool use_logit_softcap_kernel = true;
            launch_fattn_paged<D_kernel_template, cols_per_block, 1>(
                ctx, dst, *k_view, *v_view,
                flash_attn_tile_ext_f16_paged<D_kernel_template, cols_per_block, nwarps_kernel, use_logit_softcap_kernel>,
                nwarps_kernel, shared_mem, kq_granularity, stream_k_flag);
        }
    } else { // Q->ne[1] > 16
        constexpr int cols_per_block = 32; // This is NCOLS1 for launch_fattn_paged
        constexpr int nwarps_kernel = 8;
        constexpr size_t shared_mem = 0;
        constexpr bool stream_k_flag = false;
        constexpr int kq_granularity = FATTN_KQ_STRIDE_TILE_F16;

        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap_kernel = false;
            launch_fattn_paged<D_kernel_template, cols_per_block, 1>(
                ctx, dst, *k_view, *v_view,
                flash_attn_tile_ext_f16_paged<D_kernel_template, cols_per_block, nwarps_kernel, use_logit_softcap_kernel>,
                nwarps_kernel, shared_mem, kq_granularity, stream_k_flag);
        } else { // logit_softcap != 0.0f
            constexpr bool use_logit_softcap_kernel = true;
            launch_fattn_paged<D_kernel_template, cols_per_block, 1>(
                ctx, dst, *k_view, *v_view,
                flash_attn_tile_ext_f16_paged<D_kernel_template, cols_per_block, nwarps_kernel, use_logit_softcap_kernel>,
                nwarps_kernel, shared_mem, kq_granularity, stream_k_flag);
        }
    }
}


// Main entry point for paged flash attention
void ggml_cuda_flash_attn_ext_paged(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    const paged_kv_sequence_view_gpu * k_paged_view,
    const paged_kv_sequence_view_gpu * v_paged_view) {

    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K_meta = dst->src[1]; // K tensor for metadata
    const ggml_tensor * V_meta = dst->src[2]; // V tensor for metadata
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_set_device(ctx.device);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const enum ggml_prec prec = ggml_flash_attn_ext_get_prec(KQV);

    GGML_ASSERT(k_paged_view != nullptr && k_paged_view->token_mappings != nullptr && k_paged_view->page_pool_gpu != nullptr);
    GGML_ASSERT(v_paged_view != nullptr && v_paged_view->token_mappings != nullptr && v_paged_view->page_pool_gpu != nullptr);
    // Type check against view's dtype, not K_meta/V_meta->type, as K_meta/V_meta are just for shape/stride metadata
    // GGML_ASSERT(K_meta->type == k_paged_view->dtype); // This might be wrong if K_meta is dummy type
    // GGML_ASSERT(V_meta->type == v_paged_view->dtype);
    GGML_ASSERT(k_paged_view->element_size_bytes == ggml_type_size(k_paged_view->dtype));
    GGML_ASSERT(v_paged_view->element_size_bytes == ggml_type_size(v_paged_view->dtype));


    // --- This dispatch logic is a clone of ggml_cuda_flash_attn_ext ---
    // --- It needs to call _paged versions of the specific implementations ---

    if (GGML_CUDA_CC_IS_AMD(cc)) {
#if defined(GGML_HIP_ROCWMMA_FATTN)
        if (fp16_mma_available(cc)) {
            // ggml_cuda_flash_attn_ext_wmma_f16_paged(ctx, dst, k_paged_view, v_paged_view); // TODO
            GGML_LOG_WARN("Paged WMMA F16 for AMD not implemented, falling back or aborting.\n");
            GGML_ABORT("Paged AMD WMMA F16 not implemented");
            return;
        }
#endif
        // Paged Vec path for AMD
        if (prec == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            // Dispatch to paged F16 vec kernels based on Q->ne[0] (head_dim) and K/V types from view
            // Example for D=64, K/V from view (e.g. F16)
            // if (Q->ne[0] == 64 && k_paged_view->dtype == GGML_TYPE_F16 && v_paged_view->dtype == GGML_TYPE_F16) {
            //    ggml_cuda_flash_attn_ext_vec_f16_case_paged<64, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst, k_paged_view, v_paged_view);
            // } else // ... other cases ...
            // else { GGML_ABORT("Paged AMD Vec F16 not implemented for this config"); }
            GGML_LOG_WARN("Paged Vec F16 for AMD not fully implemented in dispatch, aborting.\n");
            GGML_ABORT("Paged AMD Vec F16 dispatch incomplete");
        } else {
            // ggml_cuda_flash_attn_ext_vec_f32_paged(ctx, dst, k_paged_view, v_paged_view); // TODO
            GGML_LOG_WARN("Paged Vec F32 for AMD not implemented, falling back or aborting.\n");
            GGML_ABORT("Paged AMD Vec F32 not implemented");
        }
        return;
    }

    if (!fast_fp16_available(cc)) { // Architectures without fast FP16 support
        // ggml_cuda_flash_attn_ext_tile_f32_paged or vec_f32_paged
        GGML_LOG_WARN("Paged Tile/Vec F32 for older NVIDIA not implemented.\n");
        GGML_ABORT("Paged Tile/Vec F32 for older NVIDIA not implemented");
        return;
    }

    // Architectures with FP16 support but no tensor cores (MMA)
    if (!fp16_mma_available(cc)) {
        if (prec == GGML_PREC_DEFAULT) {
            // Dispatch to appropriate paged F16 vector or tile kernel
            // Example for D=128, K/V F16 from view
            // if (Q->ne[0] == 128 && k_paged_view->dtype == GGML_TYPE_F16 && v_paged_view->dtype == GGML_TYPE_F16) {
            //    ggml_cuda_flash_attn_ext_vec_f16_case_paged<128, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst, k_paged_view, v_paged_view);
            // } // ... other cases ...
            // else { GGML_ABORT("Paged F16 for NVIDIA non-MMA not implemented for this config"); }
            GGML_LOG_WARN("Paged Tile/Vec F16 for NVIDIA without MMA not fully implemented in dispatch, aborting.\n");
            GGML_ABORT("Paged F16 non-MMA dispatch incomplete");
        } else { // Higher precision requested
            // ggml_cuda_flash_attn_ext_tile_f32_paged or vec_f32_paged
            GGML_LOG_WARN("Paged Tile/Vec F32 for NVIDIA without MMA not implemented.\n");
            GGML_ABORT("Paged Tile/Vec F32 for NVIDIA without MMA not implemented");
        }
        return;
    }

    // Architectures with tensor cores (MMA)
    const bool gqa_opt_applies = K_meta && V_meta && mask && ((Q->ne[2] / K_meta->ne[2]) % 2 == 0) ; // Grouped-Query Attention optimization check
    const bool mma_needs_data_conversion = k_paged_view->dtype != GGML_TYPE_F16 || v_paged_view->dtype != GGML_TYPE_F16;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < GGML_CUDA_CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % (2*warp_size) == 0;

    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1) {
        if (prec == GGML_PREC_DEFAULT) {
            // Example: Choose based on Q head dim and K/V types from view
            if (Q->ne[0] == 128 && k_paged_view->dtype == GGML_TYPE_F16 && v_paged_view->dtype == GGML_TYPE_F16) {
                 ggml_cuda_flash_attn_ext_vec_f16_case_paged<128, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst, k_paged_view, v_paged_view);
            } else if (Q->ne[0] == 64 && k_paged_view->dtype == GGML_TYPE_F16 && v_paged_view->dtype == GGML_TYPE_F16) {
                 ggml_cuda_flash_attn_ext_vec_f16_case_paged<64, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst, k_paged_view, v_paged_view);
            } // ... other vector cases for F16 ...
            else {
                GGML_LOG_WARN("Paged Vec F16 for BS1 NVIDIA with MMA not implemented for this specific config D=%lld, Ktype=%d, Vtype=%d.\n", Q->ne[0], (int)k_paged_view->dtype, (int)v_paged_view->dtype);
                GGML_ABORT("Paged Vec F16 BS1 dispatch incomplete");
            }
        } else {
            // ggml_cuda_flash_attn_ext_vec_f32_paged(ctx, dst, k_paged_view, v_paged_view); // TODO
            GGML_LOG_WARN("Paged Vec F32 for BS1 NVIDIA with MMA not implemented.\n");
            GGML_ABORT("Paged Vec F32 BS1 not implemented");
        }
        return;
    }

    // Paged Tile Path (Example for non-MMA FP16 capable GPUs, or specific configs)
    // This condition needs to be aligned with when original non-paged version chooses tile kernels.
    // Original logic: !fp16_mma_available(cc) && fast_fp16_available(cc) && prec == GGML_PREC_DEFAULT
    // AND Q->ne[1] > 8 && Q->ne[0] is 64 or 128
    bool use_tile_kernel_path = !fp16_mma_available(cc) && fast_fp16_available(cc) && prec == GGML_PREC_DEFAULT &&
                               (Q->ne[0] == 64 || Q->ne[0] == 128) && Q->ne[1] > 8; // Simplified condition

    if (use_tile_kernel_path) {
        if (k_paged_view->dtype == GGML_TYPE_F16 && v_paged_view->dtype == GGML_TYPE_F16) { // Only F16 K/V for tile F16 kernel
            if (Q->ne[0] == 64) {
                ggml_cuda_flash_attn_ext_tile_f16_case_paged<64, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst, k_paged_view, v_paged_view);
            } else if (Q->ne[0] == 128) {
                ggml_cuda_flash_attn_ext_tile_f16_case_paged<128, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst, k_paged_view, v_paged_view);
            } else {
                GGML_LOG_WARN("Paged Tile F16 not implemented for this head dim D=%lld.\n", Q->ne[0]);
                GGML_ABORT("Paged Tile F16 dispatch incomplete for head dim");
            }
        } else {
            GGML_LOG_WARN("Paged Tile F16 requires F16 K/V types. Got Ktype=%d, Vtype=%d.\n", (int)k_paged_view->dtype, (int)v_paged_view->dtype);
            GGML_ABORT("Paged Tile F16 type mismatch");
        }
        return;
    }

    // MMA path (Turing+)
    if (fp16_mma_available(cc) && !new_mma_available(cc)) { // Volta (WMMA)
        // ggml_cuda_flash_attn_ext_wmma_f16_paged(ctx, dst, k_paged_view, v_paged_view); // TODO
        GGML_LOG_WARN("Paged WMMA F16 for Volta not implemented.\n");
        GGML_ABORT("Paged WMMA F16 for Volta not implemented");
        return;
    }

    // Default to MMA-based kernels for Turing and newer
    ggml_cuda_flash_attn_ext_mma_f16_paged(ctx, dst, k_paged_view, v_paged_view);
}

// PAGED KV CACHE IMPLEMENTATION ENDS HERE
