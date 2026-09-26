// Upstream API compatibility shims.
//
// The fork extends two upstream signatures for GDN ingredient replay:
//   - ggml_gated_delta_net gained a trailing emit_mode (fork: ggml_gated_delta_net_ext)
//   - llama_memory_recurrent gained a bool gdn_replay_req constructor argument
// Upstream call sites must keep compiling unchanged on every ggml-org sync, so the
// upstream-shaped entry points stay available and mean "emit_mode 0" / "no replay".
// This test fails to compile when either shim is missing and fails at runtime when
// the wrapper stops encoding op_params the way the CPU/Vulkan kernels expect.

#include "ggml.h"
#include "llama-memory-recurrent.h"

#include <cstdio>
#include <type_traits>

// Upstream constructor shape (ggml-org src/llama-memory-recurrent.h): no gdn_replay_req.
static_assert(std::is_constructible_v<
        llama_memory_recurrent,
        const llama_model &, ggml_type, ggml_type, bool, uint32_t, uint32_t, uint32_t,
        const llama_memory_i::layer_filter_cb &>,
    "llama_memory_recurrent must keep the upstream constructor signature");

// Fork constructor shape stays available for the replay-aware callers.
static_assert(std::is_constructible_v<
        llama_memory_recurrent,
        const llama_model &, ggml_type, ggml_type, bool, uint32_t, uint32_t, uint32_t, bool,
        const llama_memory_i::layer_filter_cb &>,
    "llama_memory_recurrent must keep the fork constructor signature");

static int check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    return ok ? 0 : 1;
}

int main(void) {
    const int64_t S = 16, H = 2, n_tokens = 4, n_seqs = 1;

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n_tokens, n_seqs);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n_tokens, n_seqs);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n_tokens, n_seqs);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, n_seqs);
    ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, n_seqs);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, n_seqs);

    int failures = 0;

    // Upstream signature: K only. Must encode op_params[0] = K and op_params[1] = 0.
    ggml_tensor * up = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, /*K=*/2);
    failures += check(up->op == GGML_OP_GATED_DELTA_NET, "upstream ggml_gated_delta_net builds GATED_DELTA_NET");
    failures += check(up->op_params[0] == 2, "upstream form stores K in op_params[0]");
    failures += check(up->op_params[1] == 0, "upstream form leaves emit_mode 0 in op_params[1]");

    // Fork signature: explicit emit_mode. Output shape and params must match what the
    // ingredient-replay callers rely on (see ggml_gated_delta_net_ext in ggml.c).
    ggml_tensor * ext0 = ggml_gated_delta_net_ext(ctx, q, k, v, g, beta, state, /*K=*/2, /*emit_mode=*/0);
    failures += check(ext0->ne[1] == up->ne[1], "ext emit_mode 0 has the same output rows as the upstream form");

    ggml_tensor * ext1 = ggml_gated_delta_net_ext(ctx, q, k, v, g, beta, state, /*K=*/2, /*emit_mode=*/1);
    failures += check(ext1->op_params[1] == 1, "ext emit_mode 1 stores emit_mode in op_params[1]");
    // emit_mode 1 with n_tokens > K: K*4 ingredient rows + final state + checkpoint state per seq.
    const int64_t expect_rows = n_tokens * n_seqs + (2 * 4 * n_seqs + S * n_seqs + S * n_seqs);
    failures += check(ext1->ne[1] == expect_rows, "ext emit_mode 1 output rows include ingredients and both state blocks");

    ggml_free(ctx);

    printf("%s: %s\n", "test-upstream-api-shims", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
