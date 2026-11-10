// Reproducer for small-M FlashAttention row/batch invariance on CUDA.
//
// Builds the same flash_attn_ext computation twice: once with a single
// query row (M=1, the decode shape) and once with M>1 query rows (the
// small-batch verification shape), over identical K/V contents, and
// compares the outputs bitwise. With GGML_CUDA_FATTN_ROW_FAITHFUL=1 the
// small-batch attention is routed to the same vector kernel family used
// for single-row decode, and the outputs must be bit-identical.
//
// Set GGML_CUDA_FATTN_ROW_FAITHFUL=1 in the environment before running.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static uint32_t lcg_state = 42;
static float frand() {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return ((lcg_state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
}

struct fa_graph {
    ggml_context * ctx = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
};

// M = number of query rows; nh_kv KV heads with GQA factor gqa.
static bool build_fa_graph(fa_graph & g, ggml_backend_t backend, int64_t M,
        int64_t hsk, int64_t hsv, int64_t kv, int64_t nh_kv, int64_t gqa) {
    static const size_t tensor_count = 16;
    struct ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*tensor_count + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    g.ctx = ggml_init(ip);
    if (!g.ctx) { return false; }

    g.q = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, hsk, M, nh_kv*gqa, 1);
    g.k = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, hsk, kv, nh_kv, 1);
    g.v = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, hsv, kv, nh_kv, 1);
    ggml_set_name(g.q, "q");
    ggml_set_name(g.k, "k");
    ggml_set_name(g.v, "v");

    // Deterministic fill (identical K/V across both graphs; identical
    // query rows so row 0 of both outputs must match bitwise).
    lcg_state = 42;
    std::vector<float> q_data(ggml_nelements(g.q));
    for (size_t i = 0; i < q_data.size(); ++i) { q_data[i] = frand(); }
    std::vector<float> k_data(ggml_nelements(g.k));
    for (size_t i = 0; i < k_data.size(); ++i) { k_data[i] = frand(); }
    std::vector<float> v_data(ggml_nelements(g.v));
    for (size_t i = 0; i < v_data.size(); ++i) { v_data[i] = frand(); }

    // Place the inputs in the backend buffer.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(g.ctx,
        ggml_backend_get_default_buffer_type(backend));
    if (!buf) { return false; }
    ggml_backend_tensor_set(g.q, q_data.data(), 0, ggml_nbytes(g.q));
    ggml_backend_tensor_set(g.k, k_data.data(), 0, ggml_nbytes(g.k));
    ggml_backend_tensor_set(g.v, v_data.data(), 0, ggml_nbytes(g.v));

    g.out = ggml_flash_attn_ext(g.ctx, g.q, g.k, g.v, /*mask=*/nullptr,
        1.0f/(float)hsk, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    ggml_set_name(g.out, "out");

    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(g.galloc, g.gf)) { return false; }
    return ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS;
}

static void free_fa_graph(fa_graph & g) {
    if (g.galloc) { ggml_gallocr_free(g.galloc); }
    if (g.ctx) { ggml_free(g.ctx); }
    g = {};
}

int main() {
    // The flag must be set before the first FlashAttention dispatch (the
    // routing decision is cached in a function-local static).
#ifdef _WIN32
    _putenv("GGML_CUDA_FATTN_ROW_FAITHFUL=1");
#else
    setenv("GGML_CUDA_FATTN_ROW_FAITHFUL", "1", 1);
#endif

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }

    const int64_t hsk = 128, hsv = 128, kv = 96, nh_kv = 1, gqa = 8;
    const int64_t M = 5; // the smallest small-batch shape past the M=1 decode case

    fa_graph g1, gk;
    if (!build_fa_graph(g1, backend, 1, hsk, hsv, kv, nh_kv, gqa)) {
        fprintf(stderr, "M=1 graph failed\n"); return 1;
    }
    if (!build_fa_graph(gk, backend, M, hsk, hsv, kv, nh_kv, gqa)) {
        fprintf(stderr, "M=%lld graph failed\n", (long long) M); return 1;
    }

    std::vector<float> out1(ggml_nelements(g1.out));
    std::vector<float> outk(ggml_nelements(gk.out));
    ggml_backend_tensor_get(g1.out, out1.data(), 0, ggml_nbytes(g1.out));
    ggml_backend_tensor_get(gk.out, outk.data(), 0, ggml_nbytes(gk.out));

    // Row 0 of the M-row output must be bit-identical to the M=1 output
    // (identical query content, identical K/V, identical kernel family).
    const size_t row = (size_t) hsv;
    const bool identical = memcmp(out1.data(), outk.data(), row*sizeof(float)) == 0;

    free_fa_graph(g1);
    free_fa_graph(gk);
    ggml_backend_free(backend);

    if (!identical) {
        fprintf(stderr, "FAIL: M=1 and M=%lld FlashAttention row 0 differ bitwise\n",
            (long long) M);
        return 1;
    }
    printf("PASS: M=1 and M=%lld FlashAttention row 0 are bit-identical "
           "(GGML_CUDA_FATTN_ROW_FAITHFUL=1)\n", (long long) M);
    return 0;
}
