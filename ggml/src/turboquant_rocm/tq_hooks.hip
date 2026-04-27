// Forward declaration from ggml-cuda.cu - needed for HIP compilation
struct ggml_backend_cuda_context;

#include "turboquant.h"
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../ggml-cuda/common.cuh"

// These are the core hook logic functions, now isolated
extern "C" bool tq_on_set_rows(ggml_backend_cuda_context & cuda_ctx, const ggml_tensor * dst) {
    if (!dst || !dst->src[0]) {
        return false;
    }

    if (dst->src[0]->type == GGML_TYPE_IQ1_S) {
        struct llama_tq_context * tq = llama_tq_get_global();
        if (tq) {
            ggml_tensor * target = dst->src[0]->view_src ? dst->src[0]->view_src : dst->src[0];
            if (target && target->name[0] != '\0' && dst->src[1]) {
                const int * d_indices = dst->src[2] ? (const int *)dst->src[2]->data : nullptr;
                if (strncmp(target->name, "cache_k_l", 9) == 0) {
                    llama_tq_store_k(tq, std::atoi(target->name + 9), 0, dst->src[1]->data, dst->src[1]->ne[1], d_indices);
                } else if (strncmp(target->name, "cache_v_l", 9) == 0) {
                    llama_tq_store_v(tq, std::atoi(target->name + 9), 0, dst->src[1]->data, dst->src[1]->ne[1], d_indices);
                }
            }
        }
        return true; // Always skip original for IQ1_S
    }
    return false;
}

extern "C" bool tq_on_flash_attn(ggml_backend_cuda_context & cuda_ctx, const ggml_tensor * dst) {
    ggml_tensor * k_tensor = (dst && dst->src[1]) ? dst->src[1] : nullptr;
    ggml_tensor * target = (k_tensor) ? (k_tensor->view_src ? k_tensor->view_src : k_tensor) : nullptr;

    if (target && target->type == GGML_TYPE_IQ1_S) {
        struct llama_tq_context * tq = llama_tq_get_global();
        if (tq && dst->src[0] && dst->src[0]->data && dst->data &&
            target->name[0] != '\0' && strncmp(target->name, "cache_k_l", 9) == 0) {

            int il = std::atoi(target->name + 9);
            int n_heads_q = dst->src[0]->ne[2];
            int n_queries = dst->src[0]->ne[1];
            int n_tokens_kv = dst->src[1]->ne[1];

            llama_tq_attn(tq, il, (const void *)dst->src[0]->data, nullptr, (float *)dst->data,
                          n_heads_q, n_queries, n_tokens_kv);
        } else if (dst && dst->data) {
            // Safe fallback: zero output if we can't process
            // ctx.stream() is actually cuda_ctx.stream()
            hipMemsetAsync(dst->data, 0, ggml_nbytes(dst), cuda_ctx.stream());
        }
        return true; // Always skip original for IQ1_S
    }
    return false;
}
