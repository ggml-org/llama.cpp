// Op-level A/B for this model's hot matmuls on the CPU backend: plain buffer vs repack buffer.
// Shapes: Ornith/Qwen3.5-35B-A3B. Reports time per op and effective GB/s of weights read, and
// checks the repack result against the plain result.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "repack.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <string>

static int g_threads = 16;

struct Case { const char * name; ggml_type wtype; int K, N, n_expert, n_used, M; bool moe; };

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void fill_ids(std::vector<int32_t> & ids, int n_used, int M, int n_expert, std::mt19937 & rng) {
    ids.resize((size_t) n_used * M);
    std::uniform_int_distribution<int> U(0, n_expert - 1);
    for (int t = 0; t < M; t++) {
        // distinct experts per token
        std::vector<int> pick;
        while ((int) pick.size() < n_used) { int e = U(rng); bool dup = false; for (int p : pick) if (p == e) dup = true; if (!dup) pick.push_back(e); }
        for (int j = 0; j < n_used; j++) ids[(size_t) t * n_used + j] = pick[j];
    }
}

// returns max |diff| relative to max |ref|
static double run_case(const Case & c, ggml_backend_buffer_type_t buft, const char * label, std::vector<float> & out, double & gbps, double & ms) {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int n_mat = c.moe ? c.n_expert : 1;

    // weights
    ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * W = c.moe ? ggml_new_tensor_3d(wctx, c.wtype, c.K, c.N, c.n_expert) : ggml_new_tensor_2d(wctx, c.wtype, c.K, c.N);
    ggml_set_name(W, "W");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
    {
        // deterministic quantized weights: quantize random f32 rows
        const size_t row_el = c.K;
        std::vector<float> rowf(row_el * 64);
        std::vector<uint8_t> rowq(ggml_row_size(c.wtype, c.K) * 64);
        const size_t nrows = (size_t) c.N * n_mat;
        std::vector<uint8_t> all(ggml_row_size(c.wtype, c.K) * nrows);
        std::mt19937 wr(7);
        for (size_t r = 0; r < nrows; r += 64) {
            size_t cnt = std::min<size_t>(64, nrows - r);
            for (size_t i = 0; i < cnt * row_el; i++) rowf[i] = nd(wr);
            ggml_quantize_chunk(c.wtype, rowf.data(), rowq.data(), 0, cnt, c.K, nullptr);
            memcpy(all.data() + r * ggml_row_size(c.wtype, c.K), rowq.data(), cnt * ggml_row_size(c.wtype, c.K));
        }
        ggml_backend_tensor_set(W, all.data(), 0, all.size());
    }

    // activations, ids, graph
    const size_t mem = 256u * 1024 * 1024 + (size_t) c.K * c.n_used * c.M * 4 * 2 + (size_t) c.N * c.n_used * c.M * 4 * 2;
    ggml_init_params gp = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * X; ggml_tensor * ids = nullptr; ggml_tensor * Y;
    if (c.moe) {
        // gate/up style: X is [K, 1, M]; down style would be [K, n_used, M]; use [K,1,M] when K==2048 else [K,n_used,M]
        const bool per_slot = (c.K != 2048);
        X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, per_slot ? c.n_used : 1, c.M);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.M);
        std::vector<int32_t> idv; fill_ids(idv, c.n_used, c.M, c.n_expert, rng);
        memcpy(ids->data, idv.data(), idv.size() * 4);
        Y = ggml_mul_mat_id(ctx, W, X, ids);
    } else {
        X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.M);
        Y = ggml_mul_mat(ctx, W, X);
    }
    { float * xd = (float *) X->data; std::uniform_real_distribution<float> U01(0.f, 1.f);
      const char * tails = getenv("BENCH_TAILS");
      for (int64_t i = 0; i < ggml_nelements(X); i++) { float v = nd(rng); if (tails && U01(rng) < 0.01f) v *= 60.0f; xd[i] = v; } }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, g_threads);

    // warmup + timing
    ggml_backend_graph_compute(be, gf);
    const int reps = c.M == 1 ? 20 : 4;
    double t0 = now();
    for (int r = 0; r < reps; r++) ggml_backend_graph_compute(be, gf);
    double t = (now() - t0) / reps;
    ms = t * 1000.0;
    // bytes of weights touched: moe -> n_used experts per token (upper bound: distinct experts)
    const double row_bytes = (double) ggml_row_size(c.wtype, c.K);
    double bytes;
    if (c.moe) {
        // distinct experts touched
        std::vector<char> seen(c.n_expert, 0); int distinct = 0;
        const int32_t * idv = (const int32_t *) ids->data;
        for (int64_t i = 0; i < ggml_nelements(ids); i++) if (!seen[idv[i]]) { seen[idv[i]] = 1; distinct++; }
        bytes = row_bytes * c.N * distinct;
    } else bytes = row_bytes * c.N;
    gbps = bytes / t / 1e9;

    out.resize(ggml_nelements(Y));
    memcpy(out.data(), Y->data, out.size() * 4);

    ggml_backend_free(be);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    (void) label;
    return 0.0;
}

int main(int argc, char ** argv) {
    if (argc > 1) g_threads = atoi(argv[1]);
    ggml_cpu_init();
    const Case cases[] = {
        { "experts gate/up  Q4_K  K2048 N512  x256 top8  M=1",   GGML_TYPE_Q4_K, 2048,  512, 256, 8,   1, true },
        { "experts down     Q6_K  K512  N2048 x256 top8  M=1",   GGML_TYPE_Q6_K,  512, 2048, 256, 8,   1, true },
        { "experts gate/up  Q4_K  K2048 N512  x256 top8  M=512", GGML_TYPE_Q4_K, 2048,  512, 256, 8, 512, true },
        { "experts down     Q6_K  K512  N2048 x256 top8  M=512", GGML_TYPE_Q6_K,  512, 2048, 256, 8, 512, true },
        { "dense attn_qkv   Q6_K  K2048 N8192              M=1",   GGML_TYPE_Q6_K, 2048, 8192,   1, 1,   1, false },
        { "dense attn_qkv   Q6_K  K2048 N8192              M=512", GGML_TYPE_Q6_K, 2048, 8192,   1, 1, 512, false },
        { "dense lm_head    Q6_K  K2048 N248320            M=1",   GGML_TYPE_Q6_K, 2048, 248320, 1, 1,  1, false },
        { "dense ssm_out    Q4_K  K4096 N2048              M=1",   GGML_TYPE_Q4_K, 4096, 2048,   1, 1,   1, false },
        { "dense lm_head    Q6_K  K2048 N248320            M=512", GGML_TYPE_Q6_K, 2048, 248320, 1, 1, 512, false },
        { "experts gate/up  Q4_K  K2048 N512  x256 top8  M=2048", GGML_TYPE_Q4_K, 2048,  512, 256, 8, 2048, true },
        { "experts down     Q6_K  K512  N2048 x256 top8  M=2048", GGML_TYPE_Q6_K,  512, 2048, 256, 8, 2048, true },
        { "dense attn_qkv   Q6_K  K2048 N8192              M=2048", GGML_TYPE_Q6_K, 2048, 8192,   1, 1, 2048, false },
        { "dense lm_head    Q6_K  K2048 N248320            M=20",  GGML_TYPE_Q6_K, 2048, 248320, 1, 1,  20, false },
    };
    printf("threads=%d\n", g_threads);
    printf("%-52s %12s %12s %12s %12s %10s\n", "case", "plain ms", "plain GB/s", "repack ms", "repack GB/s", "max rel d");
    const char * filt = argc > 2 ? argv[2] : nullptr;
    for (const Case & c : cases) {
        if (filt && !strstr(c.name, filt)) continue;
        std::vector<float> o1, o2; double g1, g2, m1, m2;
        run_case(c, ggml_backend_cpu_buffer_type(), "plain", o1, g1, m1);
        run_case(c, ggml_backend_cpu_repack_buffer_type(), "repack", o2, g2, m2);
        double maxd = 0, maxr = 0;
        for (size_t i = 0; i < o1.size(); i++) { maxd = std::max(maxd, (double) fabs(o1[i] - o2[i])); maxr = std::max(maxr, (double) fabs(o1[i])); }
        printf("%-52s %12.3f %12.1f %12.3f %12.1f %10.2e\n", c.name, m1, g1, m2, g2, maxr > 0 ? maxd / maxr : maxd);
    }
    return 0;
}
