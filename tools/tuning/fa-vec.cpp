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

using set_override_t   = void (*)(int, int);
using clear_override_t = void (*)(void);
using bucket_t         = int  (*)(int64_t);
using baseline_ne_t    = int  (*)(int, int);
using device_token_t   = const char * (*)(ggml_backend_dev_t);

struct fa_procs {
    set_override_t   set_ov      = nullptr;
    clear_override_t clr_ov      = nullptr;
    bucket_t         ne11_bucket = nullptr;
    bucket_t         ne01_bucket = nullptr;
    baseline_ne_t    baseline_ne = nullptr;
    device_token_t   dev_token   = nullptr;

    bool ok() const {
        return set_ov && clr_ov && ne11_bucket && ne01_bucket && baseline_ne && dev_token;
    }
};

static fa_procs fa_resolve_procs(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

    fa_procs p;
    p.set_ov      = (set_override_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_set_fa_vec_override");
    p.clr_ov      = (clear_override_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_clear_fa_vec_override");
    p.ne11_bucket = (bucket_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_fa_vec_ne11_bucket");
    p.ne01_bucket = (bucket_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_fa_vec_ne01_bucket");
    p.baseline_ne = (baseline_ne_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_fa_vec_baseline_ne");
    p.dev_token   = (device_token_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_device_token");

    return p;
}

static const char * fa_type_token(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: return "GGML_TYPE_Q4_0";
        case GGML_TYPE_Q4_1: return "GGML_TYPE_Q4_1";
        case GGML_TYPE_Q5_0: return "GGML_TYPE_Q5_0";
        case GGML_TYPE_Q5_1: return "GGML_TYPE_Q5_1";
        case GGML_TYPE_Q8_0: return "GGML_TYPE_Q8_0";
        default:             return "GGML_TYPE_F16";
    }
}

// "f16,q4_0" -> does it contain ggml_type_name(t)?  null filter accepts everything
static bool fa_filter_has(const char * filter, const char * name) {
    if (!filter) {
        return true;
    }

    const std::string f = std::string(",") + filter + ",";

    return f.find(std::string(",") + name + ",") != std::string::npos;
}

struct fa_cand { int Q, NE; };

struct fa_point {                   // one swept grid point with its candidate times
    int                 dk, dv, ne11, ne01;
    std::vector<double> t;          // indexed like the shape's candidate list
};

// candidate list for one shape, identical for every grid point of it. base_i is the index of
// the (Q=1, baseline NE) candidate: the anchor config, and what the tuning gates compare to.
static std::vector<fa_cand> fa_build_cands(const fa_procs & procs, int dk, int dv, int & base_i) {
    const int base_ne = procs.baseline_ne(dk, dv);

    std::vector<fa_cand> cands;
    base_i = -1;
    for (int ne : fa_legal_ne(dk, dv)) {
        for (int Q : { 1, 2, 4 }) {
            if (Q == 1 && ne == base_ne) {
                base_i = (int) cands.size();
            }
            cands.push_back({ Q, ne });
        }
    }
    GGML_ASSERT(base_i >= 0);

    return cands;
}

bool tuner_fa_vec_run(ggml_backend_t backend, ggml_backend_dev_t dev, const tuner_opts & opts) {
    const fa_procs procs = fa_resolve_procs(dev);
    if (!procs.ok()) {
        fprintf(stderr, "error: metal fa_vec tuning procs unavailable\n");
        return false;
    }

    const char * dev_token = procs.dev_token(dev);

    struct shape_t { int dk, dv; };
    const shape_t shapes[] = { { 32, 32 }, { 64, 64 }, { 96, 96 }, { 128, 128 }, { 192, 192 },
                               { 192, 128 }, { 256, 256 }, { 320, 256 }, { 512, 512 }, { 576, 512 } };
    const int ne11_rep[] = { 512, 2048, 8192, 32768 };      // ne11 bucket representatives
    const int ne01_rep[] = { 1, 2, 3, 4, 5, 6, 7, 8, 16 };  // point buckets (1-4) + tail mod-4 cycle + anchor
    const ggml_type types[] = { GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                                GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0 };

    const double TUNE_TAU   = 0.05;  // max POINTWISE regret to ride a domain default
    const double TUNE_THETA = 1.05;  // min AGGREGATE bucket speedup vs baseline to tune at all

    cooldown_opts cool;
    cool.enabled   = opts.cooldown;
    cool.drift     = opts.cool_drift;
    cool.eps       = opts.cool_eps;
    cool.max_wait  = opts.cool_max_wait;
    cool.max_retry = opts.cool_max_retry;

    fprintf(stderr, "seed=%u reps=%d cooldown=%s (drift=%.2f eps=%.2f max_wait=%ds max_retry=%d)\n",
            opts.seed, opts.reps, cool.enabled ? "on" : "off",
            cool.drift, cool.eps, cool.max_wait, cool.max_retry);
    fprintf(stderr, "device token: %s\n", dev_token);

    int n_untrusted = 0;

    printf("// ==== BEGIN fa_vec_tuned_table rows (%s) ====\n", dev_token);

    for (ggml_type type_kv : types) {
        if (!fa_filter_has(opts.dtype_filter, ggml_type_name(type_kv))) {
            continue;
        }

        fprintf(stderr, "\n### dtype=%s\n", ggml_type_name(type_kv));

        std::vector<fa_point> pts;

        for (auto s : shapes) {
            if (!fa_filter_has(opts.dk_filter, std::to_string(s.dk).c_str())) {
                continue;
            }

            int base_i = 0;
            std::vector<fa_cand> cands = fa_build_cands(procs, s.dk, s.dv, base_i);

            for (int ne11 : ne11_rep) {
                for (int ne01 : ne01_rep) {
                    const fa_shape sh = { s.dk, s.dv, ne01, ne11, type_kv };

                    perf_cell cell = build_perf_cell(backend,
                        [&](ggml_context * ctx) { return fa_build_graph(ctx, sh); },
                        [&](ggml_context * ctx) { fa_init_tensors(ctx, sh, opts.seed); },
                        [&](ggml_tensor *) { return fa_op_flops(sh); });

                    if (!cell.ok) {
                        continue;
                    }

                    // randomize candidate order to decorrelate thermal drift across the cell
                    std::vector<int> order((size_t) cands.size());
                    for (size_t i = 0; i < order.size(); ++i) {
                        order[i] = (int) i;
                    }
                    std::shuffle(order.begin(), order.end(), std::mt19937(opts.seed));

                    char label[128];
                    snprintf(label, sizeof(label), "dk=%d ne11=%d", s.dk, ne11);

                    cell_result r = measure_cell(backend, cell, opts.reps,
                        (int) cands.size(), order,
                        [&](int i) { procs.set_ov(cands[i].Q, cands[i].NE); },
                        [&]()      { procs.clr_ov(); },
                        base_i, cool, label);

                    // per-cell noise floor: spread of the repeated same-config anchor
                    if (r.anchor_min > 0.0) {
                        fprintf(stderr, "# noise dk=%d dv=%d ne11=%d ne01=%d spread=%.1f%%\n",
                                s.dk, s.dv, ne11, ne01, 100.0*(r.anchor_max - r.anchor_min)/r.anchor_min);
                    }

                    if (!r.trusted) {
                        n_untrusted++;
                        fprintf(stderr, "# DROP untrusted cell dk=%d dv=%d ne11=%d ne01=%d\n",
                                s.dk, s.dv, ne11, ne01);
                        continue;
                    }

                    int best_i = -1;
                    for (size_t i = 0; i < cands.size(); ++i) {
                        if (r.t[i] > 0.0 && (best_i < 0 || r.t[i] < r.t[best_i])) {
                            best_i = (int) i;
                        }
                    }
                    const double base_t = r.t[base_i];
                    const bool   keep   = best_i >= 0 && base_t > 0.0 && r.t[best_i] < base_t*0.98;

                    fprintf(stderr, "# dtype=%s dk=%d dv=%d ne11=%d ne01=%d:",
                            ggml_type_name(type_kv), s.dk, s.dv, ne11, ne01);
                    for (size_t i = 0; i < cands.size(); ++i) {
                        fprintf(stderr, "  Q%dNE%d=%.1f%s", cands[i].Q, cands[i].NE, r.t[i],
                                (int) i == best_i ? "*" : "");
                    }
                    if (keep) {
                        fprintf(stderr, "  => Q%d,NE%d  %.2fx\n",
                                cands[best_i].Q, cands[best_i].NE, base_t/r.t[best_i]);
                    } else {
                        fprintf(stderr, "  => baseline\n");
                    }

                    pts.push_back({ s.dk, s.dv, ne11, ne01, r.t });
                }
            }
        }

        // compress into pasteable rows. per (dk,dv) and ne01 domain {decode==1, batch>=2},
        // emit one ne11-collapsed default cfg (ne11_b=-1) plus a per-bucket exception wherever
        // the default's pointwise regret vs the bucket target, or its aggregate slowdown vs
        // baseline, exceeds TUNE_TAU.
        std::vector<std::string> rows_out;
        char rbuf[192];

        for (auto s : shapes) {
            if (!fa_filter_has(opts.dk_filter, std::to_string(s.dk).c_str())) {
                continue;
            }

            int base_i = 0;
            std::vector<fa_cand> cands = fa_build_cands(procs, s.dk, s.dv, base_i);

            struct bkt_t {
                int b11, b01, Ti;
                std::vector<double>          agg;
                double                       base_agg;
                std::vector<const fa_point *> bp;
            };

            // bucket the grid points with the runtime's bucketers, so keys match fa_vec_pick.
            // short-KV points (ne11 bucket 0) are dropped: the runtime serves those from baseline.
            std::set<std::pair<int, int>> seen;
            for (const auto & p : pts) {
                if (p.dk != s.dk || p.dv != s.dv) {
                    continue;
                }
                const int b11 = procs.ne11_bucket(p.ne11);
                if (b11 == 0) {
                    continue;
                }
                seen.insert({ b11, procs.ne01_bucket(p.ne01) });
            }

            std::vector<bkt_t> bks;
            for (const auto & bb : seen) {
                const int b11 = bb.first, b01 = bb.second;

                std::vector<const fa_point *> bp;
                for (const auto & p : pts) {
                    if (p.dk == s.dk && p.dv == s.dv &&
                        procs.ne11_bucket(p.ne11) == b11 && procs.ne01_bucket(p.ne01) == b01) {
                        bp.push_back(&p);
                    }
                }

                std::vector<double> agg(cands.size(), 0.0), worst(cands.size(), 0.0);
                for (const auto * p : bp) {
                    double bestt = 0.0;
                    for (size_t i = 0; i < cands.size(); ++i) {
                        if (p->t[i] > 0.0 && (bestt == 0.0 || p->t[i] < bestt)) {
                            bestt = p->t[i];
                        }
                    }
                    for (size_t i = 0; i < cands.size(); ++i) {
                        agg[i] += p->t[i];
                        if (p->t[i] > 0.0 && bestt > 0.0) {
                            worst[i] = std::max(worst[i], p->t[i]/bestt);
                        }
                    }
                }

                int robust = 0;
                for (size_t i = 1; i < cands.size(); ++i) {
                    if (worst[i] < worst[robust] ||
                        (worst[i] == worst[robust] && (cands[i].Q < cands[robust].Q ||
                        (cands[i].Q == cands[robust].Q && cands[i].NE < cands[robust].NE)))) {
                        robust = (int) i;
                    }
                }

                const bool tune = robust != base_i && agg[base_i] > 0.0 && agg[robust] > 0.0 &&
                                  agg[base_i]/agg[robust] >= TUNE_THETA;

                bks.push_back({ b11, b01, tune ? robust : base_i, agg, agg[base_i], bp });
            }

            // bucket coverage: a hardcoded sampling grid can't produce a wrong key, only miss
            // a bucket, so report what each bucket actually got
            for (const auto & b : bks) {
                fprintf(stderr, "# bucket dk=%d dv=%d ne11_b=%d ne01_b=%d samples=%zu\n",
                        s.dk, s.dv, b.b11, b.b01, b.bp.size());
                if (b.bp.empty()) {
                    fprintf(stderr, "# WARN empty bucket dk=%d dv=%d ne11_b=%d ne01_b=%d\n",
                            s.dk, s.dv, b.b11, b.b01);
                }
            }

            // pointwise regret of default cfg d vs the bucket target: a ratio-of-sums lets a
            // default that wins on aligned ne01 hide a large penalty on a misaligned point
            auto reg_pointwise = [&](const bkt_t * b, int d) {
                double r = 0.0;
                for (const auto * p : b->bp) {
                    const double td = p->t[d], tT = p->t[b->Ti];
                    if (td > 0.0 && tT > 0.0) {
                        r = std::max(r, td/tT - 1.0);
                    }
                }
                return r;
            };

            for (int dom = 0; dom <= 1; ++dom) {  // 0 = decode (ne01==1), 1 = batch (ne01>=2)
                std::vector<const bkt_t *> db;
                for (const auto & b : bks) {
                    if ((dom == 0) == (b.b01 == 0)) {
                        db.push_back(&b);
                    }
                }
                if (db.empty()) {
                    continue;
                }

                // default cfg = the one minimizing (#rows, total achieved time, Q, NE)
                int bestD = -1, bestRows = 1 << 30;
                double bestTot = 0.0;
                for (size_t d = 0; d < cands.size(); ++d) {
                    int rows = ((int) d != base_i) ? 1 : 0;
                    double tot = 0.0;
                    for (const auto * b : db) {
                        const double reg  = reg_pointwise(b, (int) d);
                        const double slow = b->base_agg > 0.0 ? b->agg[d]/b->base_agg - 1.0 : 0.0;
                        if (reg > TUNE_TAU || slow > TUNE_TAU) {
                            rows++;
                            tot += b->agg[b->Ti];
                        } else {
                            tot += b->agg[d];
                        }
                    }
                    const bool better = bestD < 0 || rows < bestRows ||
                        (rows == bestRows && (tot < bestTot ||
                        (tot == bestTot && (cands[d].Q < cands[bestD].Q ||
                        (cands[d].Q == cands[bestD].Q && cands[d].NE < cands[bestD].NE)))));
                    if (better) {
                        bestD = (int) d;
                        bestRows = rows;
                        bestTot = tot;
                    }
                }

                const int dom_id = (dom == 0) ? 0 : 1;  // FA_VEC_DOMAIN_DECODE / FA_VEC_DOMAIN_BATCH
                if (bestD != base_i) {
                    snprintf(rbuf, sizeof(rbuf), "    { { %s, %s, %d, %d, -1, %d }, { %d, %d } },",
                             dev_token, fa_type_token(type_kv), s.dk, s.dv, dom_id,
                             cands[bestD].Q, cands[bestD].NE);
                    rows_out.emplace_back(rbuf);
                }
                for (const auto * b : db) {
                    const double reg  = reg_pointwise(b, bestD);
                    const double slow = b->base_agg > 0.0 ? b->agg[bestD]/b->base_agg - 1.0 : 0.0;
                    if (reg <= TUNE_TAU && slow <= TUNE_TAU) {
                        continue;  // rides the default / baseline
                    }
                    snprintf(rbuf, sizeof(rbuf), "    { { %s, %s, %d, %d, %d, %d }, { %d, %d } },",
                             dev_token, fa_type_token(type_kv), s.dk, s.dv, b->b11, b->b01,
                             cands[b->Ti].Q, cands[b->Ti].NE);
                    rows_out.emplace_back(rbuf);
                }
            }
        }

        printf("\n    // ---- %s: %zu rows ----\n", ggml_type_name(type_kv), rows_out.size());
        for (const auto & r : rows_out) {
            printf("%s\n", r.c_str());
        }
        fflush(stdout);
    }

    printf("// ==== END fa_vec_tuned_table rows (%s) ====\n", dev_token);

    if (n_untrusted > 0) {
        fprintf(stderr, "\n%d cells excluded as untrusted (see DROP lines above)\n", n_untrusted);
    }

    return true;
}
