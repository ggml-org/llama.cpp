// SYCL tensor usage inference test.
// Validates ggml_sycl_get_tensor_usage classification by tensor name.

#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif

#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

static bool check_usage(ggml_tensor * tensor, tensor_usage expected, const char * label) {
    const tensor_usage got = ggml_sycl_get_tensor_usage(tensor);
    if (got != expected) {
        fprintf(stderr, "FAIL: %s expected=%d got=%d\n", label, (int) expected, (int) got);
        return false;
    }
    return true;
}

int main() {
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "FAIL: CPU backend unavailable\n");
        return 1;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "FAIL: ggml_init failed\n");
        ggml_backend_free(cpu_backend);
        return 1;
    }

    ggml_tensor * t_attn = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_attn, "blk.0.attn_q.weight");
    ggml_tensor * t_ffn = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_ffn, "blk.0.ffn_up.weight");
    ggml_tensor * t_moe_exp = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_moe_exp, "blk.0.ffn_gate_exps.weight");
    ggml_tensor * t_moe_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_moe_gate, "blk.0.ffn_gate_inp.weight");
    ggml_tensor * t_embed = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_embed, "token_embd.weight");
    ggml_tensor * t_norm = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_norm, "blk.0.attn_norm.weight");
    ggml_tensor * t_unknown = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_unknown, "mystery.weight");

    bool ok = true;
    ok = ok && check_usage(t_attn, tensor_usage::ATTENTION_WEIGHT, "attn");
    ok = ok && check_usage(t_ffn, tensor_usage::FFN_WEIGHT, "ffn");
    ok = ok && check_usage(t_moe_exp, tensor_usage::MOE_EXPERT_WEIGHT, "moe_expert");
    ok = ok && check_usage(t_moe_gate, tensor_usage::MOE_GATE, "moe_gate");
    ok = ok && check_usage(t_embed, tensor_usage::EMBEDDING, "embedding");
    ok = ok && check_usage(t_norm, tensor_usage::NORM, "norm");
    ok = ok && check_usage(t_unknown, tensor_usage::UNKNOWN, "unknown");

    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    printf("\nTensor usage test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
