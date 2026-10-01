// Byte-for-byte check of ggml_gated_delta_net_inplace against the gathered path that the
// graph otherwise emits for the recurrent state of a Gated Delta Net layer (K == 1):
//
//   A (gathered): ggml_get_rows(cache, rows) -> ggml_gated_delta_net(K = 1)
//                 -> ggml_cpy of the final state into the cache rows [head, head + n_seqs)
//                 (llm_graph_context::build_rs + llm_build_delta_net_base::build_recurrent_attn)
//   B (in-place): ggml_gated_delta_net_inplace(view of the cache rows [head, head + n_seqs))
//                 -- the state never leaves the cache
//
// For every case the attention scores of B must equal A byte for byte and the WHOLE cache
// tensor after B must equal the cache after A byte for byte (updated rows identical, the
// other rows untouched).
//
// Matrix: T in {1, 3}, n_seqs in {1, 4, 8}, H_k in {H_v, H_v/2}, scalar and vector (KDA)
// gate, 1 and 4 threads. CPU only, tiny synthetic shapes (S_v = 32, H_v = 4), no model needed.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct gdn_case {
    int64_t S_v;
    int64_t H_v;
    int64_t H_k;
    int64_t T;
    int64_t n_seqs;
    int64_t mem_size;
    int64_t head;
    bool    kda;      // vector gate [S_v, H_v, T, n_seqs] instead of scalar [1, H_v, T, n_seqs]
    int     n_threads;
};

// deterministic input set shared by the two paths
struct gdn_inputs {
    std::vector<float>   q, k, v, g, beta, cache;
    std::vector<int32_t> rows;
};

static gdn_inputs make_inputs(const gdn_case & c, uint32_t seed) {
    std::mt19937 rng(seed);
    auto fill = [&](std::vector<float> & dst, size_t n, float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        dst.resize(n);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = d(rng);
        }
    };

    const int64_t D  = c.S_v * c.S_v * c.H_v;
    const int64_t g0 = c.kda ? c.S_v : 1;

    gdn_inputs in;
    fill(in.q,    c.S_v * c.H_k * c.T * c.n_seqs, -1.0f, 1.0f);
    fill(in.k,    c.S_v * c.H_k * c.T * c.n_seqs, -1.0f, 1.0f);
    fill(in.v,    c.S_v * c.H_v * c.T * c.n_seqs, -1.0f, 1.0f);
    fill(in.g,    g0    * c.H_v * c.T * c.n_seqs, -2.0f, 0.0f); // log decay
    fill(in.beta,         c.H_v * c.T * c.n_seqs,  0.0f, 1.0f);
    // every row of the cache is random so that a wrong row read/write shows up
    fill(in.cache, D * c.mem_size, -1.0f, 1.0f);

    // identity mapping: sequence i reads and writes cache row head + i
    in.rows.resize(c.n_seqs);
    for (int64_t i = 0; i < c.n_seqs; ++i) {
        in.rows[i] = (int32_t) (c.head + i);
    }

    return in;
}

struct gdn_tensors {
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * g;
    ggml_tensor * beta;
    ggml_tensor * cache; // [D, mem_size]
    ggml_tensor * rows;  // I32 [n_seqs]
};

static gdn_tensors make_tensors(ggml_context * ctx, const gdn_case & c, const gdn_inputs & in) {
    const int64_t D  = c.S_v * c.S_v * c.H_v;
    const int64_t g0 = c.kda ? c.S_v : 1;

    gdn_tensors t;
    t.q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H_k, c.T, c.n_seqs);
    t.k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H_k, c.T, c.n_seqs);
    t.v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H_v, c.T, c.n_seqs);
    t.g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g0,    c.H_v, c.T, c.n_seqs);
    t.beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,     c.H_v, c.T, c.n_seqs);
    t.cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, c.mem_size);
    t.rows  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_seqs);

    memcpy(t.q->data,     in.q.data(),     ggml_nbytes(t.q));
    memcpy(t.k->data,     in.k.data(),     ggml_nbytes(t.k));
    memcpy(t.v->data,     in.v.data(),     ggml_nbytes(t.v));
    memcpy(t.g->data,     in.g.data(),     ggml_nbytes(t.g));
    memcpy(t.beta->data,  in.beta.data(),  ggml_nbytes(t.beta));
    memcpy(t.cache->data, in.cache.data(), ggml_nbytes(t.cache));
    memcpy(t.rows->data,  in.rows.data(),  ggml_nbytes(t.rows));

    return t;
}

static ggml_context * new_ctx() {
    ggml_init_params params = {
        /*.mem_size   =*/ 64u*1024u*1024u,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ggml_init failed\n");
        exit(1);
    }
    return ctx;
}

struct path_result {
    std::vector<uint8_t> attn;  // attention scores, S_v*H_v*T*n_seqs floats
    std::vector<uint8_t> cache; // the whole cache tensor after the graph ran
};

static void snapshot(path_result & r, const ggml_tensor * attn_src, size_t attn_bytes, const ggml_tensor * cache) {
    r.attn.resize(attn_bytes);
    memcpy(r.attn.data(), attn_src->data, attn_bytes);
    r.cache.resize(ggml_nbytes(cache));
    memcpy(r.cache.data(), cache->data, ggml_nbytes(cache));
}

// A: the gathered path, as build_rs + build_recurrent_attn (n_rs_seq == 0) emit it
static path_result run_gathered(const gdn_case & c, const gdn_inputs & in) {
    ggml_context * ctx = new_ctx();
    gdn_tensors t = make_tensors(ctx, c, in);

    const int64_t D = c.S_v * c.S_v * c.H_v;
    const int64_t attn_score_elems = c.S_v * c.H_v * c.T * c.n_seqs;

    ggml_cgraph * gf = ggml_new_graph(ctx);

    // build_rs: gather the state rows of the sequences
    ggml_tensor * gathered = ggml_get_rows(ctx, t.cache, t.rows);
    ggml_tensor * s        = ggml_reshape_4d(ctx, gathered, c.S_v, c.S_v, c.H_v, c.n_seqs);

    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx, t.q, t.k, t.v, t.g, t.beta, s, /*K=*/1);

    // build_delta_net_fused new_state view + build_recurrent_attn write-back into rows [head, head + n_seqs)
    ggml_tensor * new_state = ggml_view_4d(ctx, gdn_out,
            c.S_v, c.S_v, c.H_v, c.n_seqs,
            ggml_row_size(gdn_out->type, c.S_v),
            ggml_row_size(gdn_out->type, c.S_v * c.S_v),
            ggml_row_size(gdn_out->type, c.S_v * c.S_v * c.H_v),
            ggml_row_size(gdn_out->type, attn_score_elems));

    ggml_build_forward_expand(gf,
            ggml_cpy(ctx, new_state,
                ggml_view_2d(ctx, t.cache, D, c.n_seqs, t.cache->nb[1], c.head * t.cache->nb[1])));

    if (ggml_graph_compute_with_ctx(ctx, gf, c.n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "gathered graph compute failed\n");
        exit(1);
    }

    path_result r;
    snapshot(r, gdn_out, attn_score_elems * sizeof(float), t.cache);
    ggml_free(ctx);
    return r;
}

// B: the in-place op on a view of the cache rows
static path_result run_inplace(const gdn_case & c, const gdn_inputs & in) {
    ggml_context * ctx = new_ctx();
    gdn_tensors t = make_tensors(ctx, c, in);

    const int64_t attn_score_elems = c.S_v * c.H_v * c.T * c.n_seqs;

    ggml_cgraph * gf = ggml_new_graph(ctx);

    // the cache rows [head, head + n_seqs) as [S_v, S_v, H_v, n_seqs]
    ggml_tensor * s = ggml_view_4d(ctx, t.cache,
            c.S_v, c.S_v, c.H_v, c.n_seqs,
            ggml_row_size(t.cache->type, c.S_v),
            ggml_row_size(t.cache->type, c.S_v * c.S_v),
            t.cache->nb[1],
            c.head * t.cache->nb[1]);

    ggml_tensor * gdn_out = ggml_gated_delta_net_inplace(ctx, t.q, t.k, t.v, t.g, t.beta, s);

    if (ggml_nelements(gdn_out) != attn_score_elems) {
        fprintf(stderr, "in-place result has %lld elements, expected the attention scores only (%lld)\n",
                (long long) ggml_nelements(gdn_out), (long long) attn_score_elems);
        exit(1);
    }

    ggml_build_forward_expand(gf, gdn_out);

    if (ggml_graph_compute_with_ctx(ctx, gf, c.n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "in-place graph compute failed\n");
        exit(1);
    }

    path_result r;
    snapshot(r, gdn_out, attn_score_elems * sizeof(float), t.cache);
    ggml_free(ctx);
    return r;
}

static size_t count_diff(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    if (a.size() != b.size()) {
        return std::max(a.size(), b.size());
    }
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += a[i] != b[i];
    }
    return n;
}

int main() {
    const int64_t S_v = 32;
    const int64_t H_v = 4;

    int n_cases  = 0;
    int n_failed = 0;

    uint32_t seed = 1234;

    for (int64_t T : { (int64_t) 1, (int64_t) 3 }) {
    for (int64_t n_seqs : { (int64_t) 1, (int64_t) 4, (int64_t) 8 }) {
    for (int64_t H_k : { H_v, H_v / 2 }) {
    for (int kda = 0; kda < 2; ++kda) {
    for (int n_threads : { 1, 4 }) {
        gdn_case c;
        c.S_v       = S_v;
        c.H_v       = H_v;
        c.H_k       = H_k;
        c.T         = T;
        c.n_seqs    = n_seqs;
        c.mem_size  = n_seqs + 3; // the rows before and after the sequences must stay untouched
        c.head      = 2;
        c.kda       = kda != 0;
        c.n_threads = n_threads;

        const gdn_inputs in = make_inputs(c, seed++);

        const path_result A = run_gathered(c, in);
        const path_result B = run_inplace(c, in);

        const size_t d_attn  = count_diff(B.attn,  A.attn);
        const size_t d_cache = count_diff(B.cache, A.cache);

        // the check is not vacuous: the in-place op must have changed the state rows
        const std::vector<uint8_t> cache0((const uint8_t *) in.cache.data(),
                                          (const uint8_t *) in.cache.data() + in.cache.size() * sizeof(float));
        const bool touched = count_diff(B.cache, cache0) != 0;

        const bool ok = d_attn == 0 && d_cache == 0 && touched;

        printf("T=%lld n_seqs=%lld H_k=%lld kda=%d nth=%d : attn diff = %zu bytes, cache diff = %zu bytes, cache touched = %d -> %s\n",
               (long long) T, (long long) n_seqs, (long long) H_k, kda, n_threads,
               d_attn, d_cache, (int) touched, ok ? "OK" : "FAIL");

        n_cases  += 1;
        n_failed += !ok;
    }}}}}

    printf("%d cases, %d failed\n", n_cases, n_failed);

    return n_failed == 0 ? 0 : 1;
}
