#include "fa-vec.h"
#include "bench.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

// GQA spec-decode shape: enough query heads to keep the GPU busy so the Q>1 K/V-reuse
// benefit is visible. nh KV heads, nr2 query heads each, nr3 batches.
static const int FA_NH   = 4;
static const int FA_NR2  = 8;
static const int FA_NR3  = 1;

struct fa_shape {
    int       dk;
    int       dv;
    int       ne01;  // query rows
    int       ne11;  // KV length
    ggml_type type_kv;
};

// mirrors test_flash_attn_ext::build_graph for the subset this tuner sweeps
// (mask=true, sinks=false, prec=F32, type_K==type_V, no permute)
static ggml_tensor * fa_build_graph(ggml_context * ctx, const fa_shape & s) {
    const int64_t dk_padded = GGML_PAD(s.dk, ggml_blck_size(s.type_kv));
    const int64_t dv_padded = GGML_PAD(s.dv, ggml_blck_size(s.type_kv));

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dk_padded, s.ne01, FA_NH*FA_NR2, FA_NR3);
    ggml_set_name(q, "q");

    // K/V are views of a 2x-tall parent, as they are of the KV cache in production
    ggml_tensor * k0 = ggml_new_tensor_4d(ctx, s.type_kv, dk_padded, 2*s.ne11, FA_NH, FA_NR3);
    ggml_tensor * k  = ggml_view_4d(ctx, k0, dk_padded, s.ne11, FA_NH, FA_NR3,
                                    k0->nb[1], k0->nb[2], k0->nb[3], 0);
    ggml_set_name(k, "k");

    ggml_tensor * v = nullptr;
    if (dk_padded == 576 && dv_padded == 512) {
        // MLA: the V cache is a sub-view of the K cache
        v = ggml_view_4d(ctx, k, dv_padded, s.ne11, FA_NH, FA_NR3, k->nb[1], k->nb[2], k->nb[3], 0);
    } else {
        ggml_tensor * v0 = ggml_new_tensor_4d(ctx, s.type_kv, dv_padded, 2*s.ne11, FA_NH, FA_NR3);
        v = ggml_view_4d(ctx, v0, dv_padded, s.ne11, FA_NH, FA_NR3,
                         v0->nb[1], v0->nb[2], v0->nb[3], 0);
    }
    ggml_set_name(v, "v");

    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s.ne11, s.ne01, 1, FA_NR3);
    ggml_set_name(m, "m");

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float) s.dk), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_set_name(out, "out");

    return out;
}

static uint64_t fa_op_flops(const fa_shape & s) {
    // Q*K^T is ne01 x dk x ne11, P*V is ne01 x ne11 x dv, per head
    return (uint64_t) 2*FA_NH*FA_NR2*s.ne01*(s.dk + s.dv)*s.ne11*FA_NR3;
}

// mirrors init_tensor_uniform: uniform f32 data, quantized in place for quantized types
static void fa_init_uniform(ggml_tensor * t, std::mt19937 & rng, float min, float max) {
    const size_t nels = ggml_nelements(t);

    std::vector<float> data(nels);
    std::uniform_real_distribution<float> dist(min, max);
    for (size_t i = 0; i < nels; i++) {
        data[i] = dist(rng);
    }

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, data.data(), 0, nels*sizeof(float));
        return;
    }

    GGML_ASSERT(ggml_is_quantized(t->type) || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16);
    GGML_ASSERT(nels % ggml_blck_size(t->type) == 0);

    std::vector<float> imatrix(t->ne[0], 1.0f);
    const float * im = imatrix.data();
    if (!ggml_quantize_requires_imatrix(t->type)) {
        // when the imatrix is optional, exercise both paths; pick via one of the random numbers
        if (data[0] > 0.5f*(min + max)) {
            im = nullptr;
        }
    }

    const size_t blck_size = ggml_blck_size(t->type);
    const size_t n_blocks  = nels / blck_size;

    std::vector<uint8_t> dataq(ggml_row_size(t->type, nels));
    ggml_quantize_chunk(t->type, data.data(), dataq.data(), 0, n_blocks, blck_size, im);

    ggml_backend_tensor_set(t, dataq.data(), 0, dataq.size());
}

// mirrors init_tensor_kq_mask: f16 mask with ~20% of its blocks set to -INF or zero.
// the -INF blocks are what drives the kernel's skip-INF path, so this pattern is
// load-bearing for the timings, not just for numerics.
static void fa_init_kq_mask(ggml_tensor * t, std::mt19937 & rng, float min, float max) {
    GGML_ASSERT(t->type == GGML_TYPE_F16);

    const int32_t ne0 = (int32_t) t->ne[0];
    const int32_t ne1 = (int32_t) t->ne[1];
    const int32_t ne2 = (int32_t) t->ne[2];
    const int32_t ne3 = (int32_t) t->ne[3];

    std::vector<float>       data_f32(size_t(ne0)*ne1*ne2*ne3);
    std::vector<ggml_fp16_t> data_f16(size_t(ne0)*ne1*ne2*ne3);

    std::uniform_real_distribution<float> dis(min, max);
    for (size_t i = 0; i < data_f32.size(); i++) {
        data_f32[i] = dis(rng);
    }

    const int blck0 = 128;
    const int blck1 = 64;

    const int n_inf_zero_blocks = 0.2*(ne0*ne1*ne2*ne3)/(blck0*blck1);

    for (int b = 0; b < n_inf_zero_blocks; b++) {
        const int p3 = (int) (rng() % ne3);
        const int p2 = (int) (rng() % ne2);
        const int p1 = (int) (rng() % ne1);
        const int p0 = (int) (rng() % ne0);

        const bool inf = rng() & 1;

        for (int i1 = 0; i1 < blck1 && p1 + i1 < ne1; i1++) {
            const int idx = p3*ne2*ne1*ne0 + p2*ne1*ne0 + (p1 + i1)*ne0 + p0;

            for (int i0 = 0; i0 < blck0 && p0 + i0 < ne0; i0++) {
                data_f32[idx + i0] = inf ? -INFINITY : 0.0f;
            }
        }
    }

    ggml_fp32_to_fp16_row(data_f32.data(), data_f16.data(), ne0*ne1*ne2*ne3);

    ggml_backend_tensor_set(t, data_f16.data(), 0, data_f16.size()*sizeof(ggml_fp16_t));
}

// per-cell deterministic seed: the shape decides it, so a cell is reproducible
// regardless of what else the sweep visited before it
static unsigned fa_cell_seed(const fa_shape & s, unsigned base) {
    unsigned h = base;
    for (int v : { s.dk, s.dv, s.ne01, s.ne11, (int) s.type_kv }) {
        h = h*1000003u + (unsigned) v;
    }
    return h;
}

static void fa_init_tensors(ggml_context * ctx, const fa_shape & s, unsigned base_seed) {
    std::mt19937 rng(fa_cell_seed(s, base_seed));

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src != NULL) {
            continue;  // views share their parent's data
        }
        if (strcmp(t->name, "m") == 0) {
            fa_init_kq_mask(t, rng, -1.0f, 1.0f);
        } else {
            fa_init_uniform(t, rng, -1.0f, 1.0f);
        }
    }
}

// legal NE for a (dk,dv): NL = 32/NE, require (dk/4)%NL==0 && (dv/4)%NL==0
static std::vector<int> fa_legal_ne(int dk, int dv) {
    std::vector<int> r;
    for (int ne : { 1, 2, 4 }) {
        const int nl = 32 / ne;
        if ((dk/4) % nl == 0 && (dv/4) % nl == 0) {
            r.push_back(ne);
        }
    }
    return r;
}

bool tuner_fa_vec_run(ggml_backend_t backend, ggml_backend_dev_t dev, const tuner_opts & opts) {
    fprintf(stderr, "fa-vec tuner: sweep not implemented yet\n");

    (void) backend;
    (void) dev;
    (void) opts;

    return true;
}
