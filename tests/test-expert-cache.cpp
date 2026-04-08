// SPDX-License-Identifier: MIT
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void test_expert_cache_enable_disable() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu);

    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, 512, false, false);
    GGML_ASSERT(sched);

    // Stats should be zero before enabling
    int64_t hits = -1, misses = -1, fate = -1, saved = -1, copied = -1;
    ggml_backend_sched_get_expert_cache_stats(sched, &hits, &misses, &fate, &saved, &copied);
    GGML_ASSERT(hits == 0);
    GGML_ASSERT(misses == 0);
    GGML_ASSERT(fate == 0);
    GGML_ASSERT(saved == 0);
    GGML_ASSERT(copied == 0);

    ggml_backend_sched_set_expert_cache(sched, true);
    ggml_backend_sched_set_expert_cache(sched, false);

    ggml_backend_sched_get_expert_cache_stats(sched, &hits, &misses, &fate, &saved, &copied);
    GGML_ASSERT(hits == 0);
    GGML_ASSERT(misses == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    printf("  PASS: test_expert_cache_enable_disable\n");
}

static void test_expert_cache_reset_survives() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, 512, false, false);

    ggml_backend_sched_set_expert_cache(sched, true);

    for (int i = 0; i < 10; i++) {
        ggml_backend_sched_reset(sched);
    }

    int64_t h, m, f, s, c;
    ggml_backend_sched_get_expert_cache_stats(sched, &h, &m, &f, &s, &c);
    GGML_ASSERT(h == 0 && m == 0 && f == 0 && s == 0 && c == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    printf("  PASS: test_expert_cache_reset_survives\n");
}

static void test_expert_cache_toggle() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, 512, false, false);

    ggml_backend_sched_set_expert_cache(sched, false);
    ggml_backend_sched_set_expert_cache(sched, true);
    ggml_backend_sched_set_expert_cache(sched, false);

    int64_t h, m, f, s, c;
    ggml_backend_sched_get_expert_cache_stats(sched, &h, &m, &f, &s, &c);
    GGML_ASSERT(h == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    printf("  PASS: test_expert_cache_toggle\n");
}

static void test_expert_cache_null_stats() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, 512, false, false);

    ggml_backend_sched_set_expert_cache(sched, true);
    ggml_backend_sched_get_expert_cache_stats(sched, nullptr, nullptr, nullptr, nullptr, nullptr);

    int64_t hits;
    ggml_backend_sched_get_expert_cache_stats(sched, &hits, nullptr, nullptr, nullptr, nullptr);
    GGML_ASSERT(hits == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    printf("  PASS: test_expert_cache_null_stats\n");
}

static void test_expert_cache_disabled_no_crash() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, 512, false, false);

    // Never enable -- sched must work normally
    ggml_backend_sched_reset(sched);

    int64_t h, m, f, s, c;
    ggml_backend_sched_get_expert_cache_stats(sched, &h, &m, &f, &s, &c);
    GGML_ASSERT(h == 0 && m == 0);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);

    printf("  PASS: test_expert_cache_disabled_no_crash\n");
}

// Integration smoke test: exercises the actual cache copy logic via a minimal
// MUL_MAT_ID compute graph with CPU expert weights and a GPU copy target.
// On the second token with identical expert IDs the cache must record hits and
// bytes_saved > 0.  The test is skipped when no GPU backend is present; full
// behavioural correctness is validated in Task 5 (perplexity-parity run).
static void test_expert_cache_hit_on_second_token() {
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) {
        printf("  SKIP: test_expert_cache_hit_on_second_token (no GPU backend)\n");
        return;
    }

    const int n_expert      = 8;
    const int n_ff          = 16;
    const int n_embd        = 8;
    const int n_expert_used = 2;
    // n_tokens must be >= GGML_OP_OFFLOAD_MIN_BATCH (default 32) for MUL_MAT_ID
    // to be offloaded to GPU, which is required to trigger the cross-backend copy path
    const int n_tokens      = 32;

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu);

    ggml_backend_t             backends[2] = { gpu, cpu };
    ggml_backend_buffer_type_t buft_gpu    = ggml_backend_get_default_buffer_type(gpu);
    ggml_backend_buffer_type_t buft_cpu    = ggml_backend_get_default_buffer_type(cpu);
    ggml_backend_buffer_type_t bufts[2]    = { buft_gpu, buft_cpu };

    // op_offload=true is required for weights on the last (CPU) backend to be
    // offloaded to GPU, which triggers the cross-backend expert copy path
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 8192, false, true);
    GGML_ASSERT(sched);
    ggml_backend_sched_set_expert_cache(sched, true);

    struct ggml_init_params iparams = { 1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(iparams);
    GGML_ASSERT(ctx);

    // Expert weight tensor: [n_embd, n_ff, n_expert] on CPU (host weights)
    struct ggml_tensor * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_set_name(experts, "ffn_up_exps");

    // Allocate CPU buffer sized correctly for this tensor (accounts for alignment)
    ggml_backend_buffer_t cpu_buf = ggml_backend_buft_alloc_buffer(buft_cpu,
        ggml_backend_buft_get_alloc_size(buft_cpu, experts));
    GGML_ASSERT(cpu_buf);
    ggml_backend_buffer_set_usage(cpu_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    struct ggml_tallocr tallocr = ggml_tallocr_new(cpu_buf);
    ggml_tallocr_alloc(&tallocr, experts);

    // Zero-fill the weight data
    memset(experts->data, 0, ggml_nbytes(experts));

    // Input: [n_embd, n_expert_used, n_tokens]
    struct ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_expert_used, n_tokens);

    // IDs: [n_expert_used, n_tokens] i32 — selects experts 0 and 1
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);

    struct ggml_tensor * out = ggml_mul_mat_id(ctx, experts, input, ids);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_sched_alloc_graph(sched, graph);

    // IDs: experts 0 and 1 interleaved across all tokens (same on every run)
    const int n_id_vals = n_expert_used * n_tokens;
    int32_t * id_vals = (int32_t *)malloc(n_id_vals * sizeof(int32_t));
    GGML_ASSERT(id_vals);
    for (int i = 0; i < n_id_vals; i++) { id_vals[i] = i % n_expert_used; }
    ggml_backend_tensor_set(ids, id_vals, 0, n_id_vals * sizeof(int32_t));
    free(id_vals);

    // Zero-fill input activations
    const size_t input_bytes = (size_t)n_embd * n_expert_used * n_tokens * sizeof(float);
    float * indata = (float *)calloc(n_embd * n_expert_used * n_tokens, sizeof(float));
    GGML_ASSERT(indata);
    ggml_backend_tensor_set(input, indata, 0, input_bytes);
    free(indata);

    // Token 1: cold cache — expect misses >= n_expert_used
    ggml_backend_sched_graph_compute(sched, graph);

    int64_t h1, m1, f1, s1, c1;
    ggml_backend_sched_get_expert_cache_stats(sched, &h1, &m1, &f1, &s1, &c1);
    printf("  After token 1: hits=%lld misses=%lld fate=%lld saved=%lld\n",
        (long long)h1, (long long)m1, (long long)f1, (long long)s1);
    GGML_ASSERT(m1 >= n_expert_used);

    // Token 2: same IDs — cache should hit and save bytes (ids already set above)
    ggml_backend_sched_graph_compute(sched, graph);

    int64_t h2, m2, f2, s2, c2;
    ggml_backend_sched_get_expert_cache_stats(sched, &h2, &m2, &f2, &s2, &c2);
    printf("  After token 2: hits=%lld misses=%lld fate=%lld saved=%lld\n",
        (long long)h2, (long long)m2, (long long)f2, (long long)s2);
    // Either direct hits or FATE pre-population should fire on the second token
    GGML_ASSERT(h2 > h1 || f2 > f1);
    GGML_ASSERT(s2 > 0);

    ggml_free(ctx);
    ggml_backend_buffer_free(cpu_buf);
    ggml_backend_sched_free(sched);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);

    printf("  PASS: test_expert_cache_hit_on_second_token\n");
}

int main() {
    printf("test-expert-cache:\n");
    test_expert_cache_enable_disable();
    test_expert_cache_reset_survives();
    test_expert_cache_toggle();
    test_expert_cache_null_stats();
    test_expert_cache_disabled_no_crash();
    test_expert_cache_hit_on_second_token();
    printf("ALL TESTS PASSED\n");
    return 0;
}
