#include "fa-vec.h"

#include "bench.h"
#include "ggml-backend.h"
#include "ggml-metal-tuning.h"
#include "ggml.h"

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
static const int FA_NH  = 4;
static const int FA_NR2 = 8;
static const int FA_NR3 = 1;

struct fa_shape {
    int       dk;
    int       dv;
    int       ne01;  // query rows
    int       ne11;  // KV length
    ggml_type type_kv;
    int       nh = FA_NH;  // KV heads, FA_NR2 query heads each
};

// mirrors test_flash_attn_ext::build_graph for the subset this tuner sweeps
// (mask=true, sinks=false, prec=F32, type_K==type_V, no permute)
static ggml_tensor * fa_build_graph(ggml_context * ctx, const fa_shape & s) {
    const int64_t dk_padded = GGML_PAD(s.dk, ggml_blck_size(s.type_kv));
    const int64_t dv_padded = GGML_PAD(s.dv, ggml_blck_size(s.type_kv));

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dk_padded, s.ne01, s.nh * FA_NR2, FA_NR3);
    ggml_set_name(q, "q");

    // K/V are views of a 2x-tall parent, as they are of the KV cache in production
    ggml_tensor * k0 = ggml_new_tensor_4d(ctx, s.type_kv, dk_padded, 2 * s.ne11, s.nh, FA_NR3);
    ggml_tensor * k  = ggml_view_4d(ctx, k0, dk_padded, s.ne11, s.nh, FA_NR3, k0->nb[1], k0->nb[2], k0->nb[3], 0);
    ggml_set_name(k, "k");

    ggml_tensor * v = nullptr;
    if (dk_padded == 576 && dv_padded == 512) {
        // MLA: the V cache is a sub-view of the K cache
        v = ggml_view_4d(ctx, k, dv_padded, s.ne11, s.nh, FA_NR3, k->nb[1], k->nb[2], k->nb[3], 0);
    } else {
        ggml_tensor * v0 = ggml_new_tensor_4d(ctx, s.type_kv, dv_padded, 2 * s.ne11, s.nh, FA_NR3);
        v                = ggml_view_4d(ctx, v0, dv_padded, s.ne11, s.nh, FA_NR3, v0->nb[1], v0->nb[2], v0->nb[3], 0);
    }
    ggml_set_name(v, "v");

    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s.ne11, s.ne01, 1, FA_NR3);
    ggml_set_name(m, "m");

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf((float) s.dk), 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_set_name(out, "out");

    return out;
}

static uint64_t fa_op_flops(const fa_shape & s) {
    // Q*K^T is ne01 x dk x ne11, P*V is ne01 x ne11 x dv, per head
    return (uint64_t) 2 * s.nh * FA_NR2 * s.ne01 * (s.dk + s.dv) * s.ne11 * FA_NR3;
}

static void fa_init_uniform(ggml_tensor * t, std::mt19937 & rng, float min, float max) {
    const size_t nels = ggml_nelements(t);

    std::vector<float>                    data(nels);
    std::uniform_real_distribution<float> dist(min, max);
    for (size_t i = 0; i < nels; i++) {
        data[i] = dist(rng);
    }

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(float));
        return;
    }

    GGML_ASSERT(ggml_is_quantized(t->type) || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16);
    GGML_ASSERT(nels % ggml_blck_size(t->type) == 0);

    std::vector<float> imatrix(t->ne[0], 1.0f);
    const float *      im = imatrix.data();
    if (!ggml_quantize_requires_imatrix(t->type)) {
        // when the imatrix is optional, exercise both paths; pick via one of the random numbers
        if (data[0] > 0.5f * (min + max)) {
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

    std::vector<float>       data_f32(size_t(ne0) * ne1 * ne2 * ne3);
    std::vector<ggml_fp16_t> data_f16(size_t(ne0) * ne1 * ne2 * ne3);

    std::uniform_real_distribution<float> dis(min, max);
    for (size_t i = 0; i < data_f32.size(); i++) {
        data_f32[i] = dis(rng);
    }

    const int blck0 = 128;
    const int blck1 = 64;

    const int n_inf_zero_blocks = 0.2 * (ne0 * ne1 * ne2 * ne3) / (blck0 * blck1);

    for (int b = 0; b < n_inf_zero_blocks; b++) {
        const int p3 = (int) (rng() % ne3);
        const int p2 = (int) (rng() % ne2);
        const int p1 = (int) (rng() % ne1);
        const int p0 = (int) (rng() % ne0);

        const bool inf = rng() & 1;

        for (int i1 = 0; i1 < blck1 && p1 + i1 < ne1; i1++) {
            const int idx = p3 * ne2 * ne1 * ne0 + p2 * ne1 * ne0 + (p1 + i1) * ne0 + p0;

            for (int i0 = 0; i0 < blck0 && p0 + i0 < ne0; i0++) {
                data_f32[idx + i0] = inf ? -INFINITY : 0.0f;
            }
        }
    }

    ggml_fp32_to_fp16_row(data_f32.data(), data_f16.data(), ne0 * ne1 * ne2 * ne3);

    ggml_backend_tensor_set(t, data_f16.data(), 0, data_f16.size() * sizeof(ggml_fp16_t));
}

static unsigned fa_cell_seed(const fa_shape & s, unsigned base) {
    unsigned h = base;
    for (int v : { s.dk, s.dv, s.ne01, s.ne11, (int) s.type_kv }) {
        h = h * 1000003u + (unsigned) v;
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

using set_override_t   = void (*)(int, int);
using clear_override_t = void (*)(void);
using bucket_t         = int (*)(int64_t);
using baseline_ne_t    = int (*)(int, int);
using device_token_t   = const char * (*) (ggml_backend_dev_t);

struct fa_procs {
    set_override_t   set_ov      = nullptr;
    clear_override_t clr_ov      = nullptr;
    bucket_t         ne11_bucket = nullptr;
    bucket_t         ne01_bucket = nullptr;
    baseline_ne_t    baseline_ne = nullptr;
    device_token_t   dev_token   = nullptr;

    bool ok() const { return set_ov && clr_ov && ne11_bucket && ne01_bucket && baseline_ne && dev_token; }
};

static fa_procs fa_resolve_procs(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

    fa_procs p;
    p.set_ov = (set_override_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_set_fa_vec_override");
    p.clr_ov =
        (clear_override_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_clear_fa_vec_override");
    p.ne11_bucket = (bucket_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_fa_vec_ne11_bucket");
    p.ne01_bucket = (bucket_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_fa_vec_ne01_bucket");
    p.baseline_ne =
        (baseline_ne_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_fa_vec_baseline_ne");
    p.dev_token = (device_token_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_device_token");

    return p;
}

static bool fa_filter_has(const char * filter, const char * name) {
    if (!filter) {
        return true;
    }

    const std::string f = std::string(",") + filter + ",";

    return f.find(std::string(",") + name + ",") != std::string::npos;
}

struct fa_cand {
    int Q, NE;
};

struct fa_point {
    int                 dk, dv, ne11, ne01;
    std::vector<double> t;
};

// base_i identifies the (Q=1, baseline NE) anchor configuration.
static std::vector<fa_cand> fa_build_cands(const fa_procs & procs, int dk, int dv, int & base_i) {
    const int base_ne = procs.baseline_ne(dk, dv);

    std::vector<fa_cand> cands;
    base_i = -1;
    for (int ne : ggml_metal_tuning::fa_vec_legal_ne(dk, dv)) {
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

    struct shape_t {
        int dk, dv;
    };

    const shape_t shapes[] = {
        { 32,  32  },
        { 64,  64  },
        { 96,  96  },
        { 128, 128 },
        { 192, 192 },
        { 192, 128 },
        { 256, 256 },
        { 320, 256 },
        { 512, 512 },
        { 576, 512 }
    };
    // nsg is a pipeline specialization constant (1 up to ne11=2048, 2 up to 4096, 4 above), so ne11
    // bucket 1 takes two samples to cover both of its regimes. Bucket 0 is not sampled at all: the
    // runtime leaves short KV at baseline, so no measurement there can reach the table.
    const int ne11_rep[] = { 2048, 3072, 8192, 32768 };
    const int ne01_rep[] = { 1, 2, 3, 4, 5, 6, 7, 8, 16 };  // point buckets (1-4) + tail mod-4 cycle + anchor

    struct dtype_t {
        ggml_type    type;
        const char * token;
    };

    const dtype_t dtypes[] = {
        { GGML_TYPE_F16,  "GGML_TYPE_F16"  },
        { GGML_TYPE_Q4_0, "GGML_TYPE_Q4_0" },
        { GGML_TYPE_Q4_1, "GGML_TYPE_Q4_1" },
        { GGML_TYPE_Q5_0, "GGML_TYPE_Q5_0" },
        { GGML_TYPE_Q5_1, "GGML_TYPE_Q5_1" },
        { GGML_TYPE_Q8_0, "GGML_TYPE_Q8_0" },
    };

    const double TUNE_TAU   = 0.05;  // max POINTWISE regret to ride a domain default
    const double TUNE_THETA = 1.05;  // min AGGREGATE bucket speedup vs baseline to tune at all

    const cooldown_opts cool = {
        opts.cooldown, opts.cool_drift, opts.cool_eps, opts.cool_max_wait, opts.cool_max_retry,
    };

    fprintf(stderr, "seed=%u reps=%d cooldown=%s (drift=%.2f eps=%.2f max_wait=%ds max_retry=%d)\n", opts.seed,
            opts.reps, cool.enabled ? "on" : "off", cool.drift, cool.eps, cool.max_wait, cool.max_retry);
    fprintf(stderr, "device token: %s\n", dev_token);

    int n_untrusted = 0;

    // stdout carries nothing but table rows, so the whole stream pastes into fa_vec_tuned_table
    for (const auto & dtype : dtypes) {
        const ggml_type type_kv = dtype.type;
        if (!fa_filter_has(opts.dtype_filter, ggml_type_name(type_kv))) {
            continue;
        }

        fprintf(stderr, "\n### dtype=%s\n", ggml_type_name(type_kv));

        std::vector<fa_point> pts;

        for (auto s : shapes) {
            if (!fa_filter_has(opts.dk_filter, std::to_string(s.dk).c_str())) {
                continue;
            }

            int                  base_i = 0;
            std::vector<fa_cand> cands  = fa_build_cands(procs, s.dk, s.dv, base_i);

            for (int ne11 : ne11_rep) {
                for (int ne01 : ne01_rep) {
                    const fa_shape sh = { s.dk, s.dv, ne01, ne11, type_kv };

                    perf_cell cell = build_perf_cell(
                        backend, [&](ggml_context * ctx) { return fa_build_graph(ctx, sh); },
                        [&](ggml_context * ctx) { fa_init_tensors(ctx, sh, opts.seed); },
                        [&](ggml_tensor *) { return fa_op_flops(sh); });

                    if (cell.gf == nullptr) {
                        continue;
                    }

                    // randomize candidate order to decorrelate thermal drift across the cell
                    std::vector<int> order((size_t) cands.size());
                    for (size_t i = 0; i < order.size(); ++i) {
                        order[i] = (int) i;
                    }
                    std::shuffle(order.begin(), order.end(), std::mt19937(fa_cell_seed(sh, opts.seed)));

                    char label[128];
                    snprintf(label, sizeof(label), "dk=%d ne11=%d", s.dk, ne11);

                    cell_result r = measure_cell(
                        backend, cell, opts.reps, order, [&](int i) { procs.set_ov(cands[i].Q, cands[i].NE); },
                        [&]() { procs.clr_ov(); }, base_i, cool, label);

                    if (r.anchor_min > 0.0) {
                        fprintf(stderr, "# noise dk=%d dv=%d ne11=%d ne01=%d spread=%.1f%%\n", s.dk, s.dv, ne11, ne01,
                                100.0 * (r.anchor_max - r.anchor_min) / r.anchor_min);
                    }

                    if (!r.trusted) {
                        n_untrusted++;
                        fprintf(stderr, "# DROP untrusted cell dk=%d dv=%d ne11=%d ne01=%d\n", s.dk, s.dv, ne11, ne01);
                        continue;
                    }

                    int best_i = -1;
                    for (size_t i = 0; i < cands.size(); ++i) {
                        if (r.t[i] > 0.0 && (best_i < 0 || r.t[i] < r.t[best_i])) {
                            best_i = (int) i;
                        }
                    }
                    const double base_t = r.t[base_i];
                    const bool   keep   = best_i >= 0 && base_t > 0.0 && r.t[best_i] < base_t * 0.98;

                    fprintf(stderr, "# dtype=%s dk=%d dv=%d ne11=%d ne01=%d:", ggml_type_name(type_kv), s.dk, s.dv,
                            ne11, ne01);
                    for (size_t i = 0; i < cands.size(); ++i) {
                        fprintf(stderr, "  Q%dNE%d=%.1f%s", cands[i].Q, cands[i].NE, r.t[i],
                                (int) i == best_i ? "*" : "");
                    }
                    if (keep) {
                        fprintf(stderr, "  => Q%d,NE%d  %.2fx\n", cands[best_i].Q, cands[best_i].NE,
                                base_t / r.t[best_i]);
                    } else {
                        fprintf(stderr, "  => baseline\n");
                    }

                    pts.push_back({ s.dk, s.dv, ne11, ne01, r.t });
                }
            }
        }

        // compress into pasteable rows. per (dk,dv) and ne01 domain {decode==1, batch>=2},
        // emit one ne11-collapsed default cfg (ne11_b=-1) plus a per-bucket exception wherever the
        // default's pointwise regret vs the bucket target exceeds TUNE_TAU, or the default is not
        // admissible for that bucket (see never_slower / admissible below).
        std::vector<std::string> rows_out;
        char                     rbuf[192];

        for (auto s : shapes) {
            if (!fa_filter_has(opts.dk_filter, std::to_string(s.dk).c_str())) {
                continue;
            }

            int                  base_i = 0;
            std::vector<fa_cand> cands  = fa_build_cands(procs, s.dk, s.dv, base_i);

            struct bkt_t {
                int                           b11, b01, Ti;
                std::vector<double>           agg;
                std::vector<const fa_point *> bp;
            };

            // A config may represent a bucket only if it is no slower than baseline at every point that
            // bucket covers. The aggregate gate below sums absolute times, so it can pass on the aligned
            // and deep points while a misaligned ne01 pays the mod-Q padding. Nothing measured, nothing
            // proven: a bucket with no surviving sample admits baseline only.
            auto never_slower = [&](const std::vector<const fa_point *> & bp, int i) {
                if (i == base_i) {
                    return true;
                }
                if (bp.empty()) {
                    return false;
                }
                for (const auto * p : bp) {
                    if (p->t[i] <= 0.0 || p->t[base_i] <= 0.0 || p->t[i] > p->t[base_i]) {
                        return false;
                    }
                }
                return true;
            };

            // The padded-row waste ceil(n/Q)*Q/n is largest at the smallest ne01 of each residue class
            // mod Q, so one of a bucket's first Q values carries the worst padding it can ever see, and
            // that value has to be sampled. Otherwise the bucket bounds nothing: a config picked on the
            // aligned ne01=8,16 says nothing about ne01=9. This covers the padding term only - the
            // per-row cost varies with ne01 too - so it is a floor on the evidence, not a proof.
            auto admissible = [&](const std::vector<const fa_point *> & bp, int b01, int i) {
                if (!never_slower(bp, i)) {
                    return false;
                }
                const int Q = cands[i].Q;
                if (Q == 1) {
                    return true;  // one row per threadgroup, no padding to witness
                }
                int lo = bp[0]->ne01;
                for (const auto * p : bp) {
                    lo = std::min(lo, p->ne01);
                }
                while (lo > 1 && procs.ne01_bucket(lo - 1) == b01) {
                    lo--;  // walk down to where this bucket's runtime domain starts
                }
                int    wit  = lo;
                double wmax = 0.0;
                for (int n = lo; n < lo + Q && procs.ne01_bucket(n) == b01; ++n) {
                    const int    padded = ((n + Q - 1) / Q) * Q;
                    const double w      = (double) padded / n;
                    if (w > wmax) {
                        wmax = w;
                        wit  = n;
                    }
                }
                for (const auto * p : bp) {
                    if (p->ne01 == wit) {
                        return true;
                    }
                }
                return false;
            };

            std::set<std::pair<int, int>> buckets;
            for (int ne11 : ne11_rep) {
                const int b11 = procs.ne11_bucket(ne11);
                if (b11 == 0) {
                    continue;
                }
                for (int ne01 : ne01_rep) {
                    buckets.insert({ b11, procs.ne01_bucket(ne01) });
                }
            }

            std::vector<bkt_t> bks;
            for (const auto & bb : buckets) {
                const int b11 = bb.first, b01 = bb.second;

                std::vector<const fa_point *> bp;
                for (const auto & p : pts) {
                    if (p.dk == s.dk && p.dv == s.dv && procs.ne11_bucket(p.ne11) == b11 &&
                        procs.ne01_bucket(p.ne01) == b01) {
                        bp.push_back(&p);
                    }
                }

                fprintf(stderr, "# bucket dk=%d dv=%d ne11_b=%d ne01_b=%d samples=%zu\n", s.dk, s.dv, b11, b01,
                        bp.size());
                if (bp.empty()) {
                    // nothing to check a config against, so pin the bucket to baseline instead of
                    // letting the ne11-collapsed domain default ride in unmeasured
                    fprintf(stderr, "# WARN empty bucket dk=%d dv=%d ne11_b=%d ne01_b=%d -> baseline\n", s.dk, s.dv,
                            b11, b01);
                    bks.push_back({ b11, b01, base_i, std::vector<double>(cands.size(), 0.0), {} });
                    continue;
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
                            worst[i] = std::max(worst[i], p->t[i] / bestt);
                        }
                    }
                }

                int robust = -1, oracle_pick = -1;
                for (size_t i = 0; i < cands.size(); ++i) {
                    auto tighter = [&](int j) {
                        return j < 0 || worst[i] < worst[j] ||
                               (worst[i] == worst[j] && (cands[i].Q < cands[j].Q ||
                                                         (cands[i].Q == cands[j].Q && cands[i].NE < cands[j].NE)));
                    };
                    if (tighter(oracle_pick)) {
                        oracle_pick = (int) i;
                    }
                    if (admissible(bp, b01, (int) i) && tighter(robust)) {
                        robust = (int) i;
                    }
                }

                const bool tune = robust != base_i && agg[base_i] > 0.0 && agg[robust] > 0.0 &&
                                  agg[base_i] / agg[robust] >= TUNE_THETA;

                // report what the no-harm rule cost this bucket, but only when it changed the outcome:
                // a sweep on another machine then shows where the winner loses, instead of just
                // emitting a smaller table
                const bool refused = oracle_pick != robust && oracle_pick != base_i && agg[base_i] > 0.0 &&
                                     agg[oracle_pick] > 0.0 && agg[base_i] / agg[oracle_pick] >= TUNE_THETA;
                if (refused) {
                    double over = 0.0;
                    int    at11 = 0, at01 = 0;
                    for (const auto * p : bp) {
                        if (p->t[base_i] > 0.0 && p->t[oracle_pick] / p->t[base_i] - 1.0 > over) {
                            over = p->t[oracle_pick] / p->t[base_i] - 1.0;
                            at11 = p->ne11;
                            at01 = p->ne01;
                        }
                    }
                    if (over > 0.0) {
                        fprintf(stderr,
                                "# reject dk=%d dv=%d ne11_b=%d ne01_b=%d Q%dNE%d: +%.2f%% vs baseline at "
                                "ne11=%d ne01=%d\n",
                                s.dk, s.dv, b11, b01, cands[oracle_pick].Q, cands[oracle_pick].NE, 100.0 * over, at11,
                                at01);
                    } else {
                        fprintf(stderr, "# reject dk=%d dv=%d ne11_b=%d ne01_b=%d Q%dNE%d: no padding witness\n", s.dk,
                                s.dv, b11, b01, cands[oracle_pick].Q, cands[oracle_pick].NE);
                    }
                }

                bks.push_back({ b11, b01, tune ? robust : base_i, agg, bp });
            }

            // pointwise regret of default cfg d vs the bucket target: a ratio-of-sums lets a
            // default that wins on aligned ne01 hide a large penalty on a misaligned point
            auto reg_pointwise = [&](const bkt_t * b, int d) {
                double r = 0.0;
                for (const auto * p : b->bp) {
                    const double td = p->t[d], tT = p->t[b->Ti];
                    if (td > 0.0 && tT > 0.0) {
                        r = std::max(r, td / tT - 1.0);
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
                int    bestD = -1, bestRows = 1 << 30;
                double bestTot = 0.0;
                for (size_t d = 0; d < cands.size(); ++d) {
                    int    rows = ((int) d != base_i) ? 1 : 0;
                    double tot  = 0.0;
                    for (const auto * b : db) {
                        if (reg_pointwise(b, (int) d) > TUNE_TAU || !admissible(b->bp, b->b01, (int) d)) {
                            rows++;
                            tot += b->agg[b->Ti];
                        } else {
                            tot += b->agg[d];
                        }
                    }
                    const bool better =
                        bestD < 0 || rows < bestRows ||
                        (rows == bestRows &&
                         (tot < bestTot ||
                          (tot == bestTot && (cands[d].Q < cands[bestD].Q ||
                                              (cands[d].Q == cands[bestD].Q && cands[d].NE < cands[bestD].NE)))));
                    if (better) {
                        bestD    = (int) d;
                        bestRows = rows;
                        bestTot  = tot;
                    }
                }

                if (bestD != base_i) {
                    snprintf(rbuf, sizeof(rbuf), "    { { %s, %s, %d, %d, -1, %d }, { %d, %d } },", dev_token,
                             dtype.token, s.dk, s.dv, dom, cands[bestD].Q, cands[bestD].NE);
                    rows_out.emplace_back(rbuf);
                }
                for (const auto * b : db) {
                    if (reg_pointwise(b, bestD) <= TUNE_TAU && admissible(b->bp, b->b01, bestD)) {
                        continue;
                    }
                    snprintf(rbuf, sizeof(rbuf), "    { { %s, %s, %d, %d, %d, %d }, { %d, %d } },", dev_token,
                             dtype.token, s.dk, s.dv, b->b11, b->b01, cands[b->Ti].Q, cands[b->Ti].NE);
                    rows_out.emplace_back(rbuf);
                }
            }
        }

        for (const auto & r : rows_out) {
            printf("%s\n", r.c_str());
        }
        fflush(stdout);
    }

    if (n_untrusted > 0) {
        fprintf(stderr, "\n%d cells excluded as untrusted (see DROP lines above)\n", n_untrusted);
    }

    return true;
}

// ---- FA (non-vec) (Q, NSG): baseline tile vs the wide tile (Q = 16) with 4 or 8 simdgroups ----

bool tuner_fa_run(ggml_backend_t backend, ggml_backend_dev_t dev, const tuner_opts & opts) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

    auto set_ov    = (set_override_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_set_fa_override");
    auto clr_ov    = (clear_override_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_clear_fa_override");
    auto ne11_b    = (bucket_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_fa_ne11_bucket");
    auto dev_token = (device_token_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_device_token");
    if (!set_ov || !clr_ov || !ne11_b || !dev_token) {
        fprintf(stderr, "error: metal fa tuning procs unavailable\n");
        return false;
    }

    // head sizes that fit the wide tile in the threadgroup memory (see ggml_metal_op_flash_attn_ext_cfg)
    struct shape_t {
        int dk, dv;
    };

    const shape_t shapes[] = {
        { 32,  32  },
        { 40,  40  },
        { 48,  48  },
        { 64,  64  },
        { 72,  72  },
        { 80,  80  },
        { 96,  96  },
        { 96,  64  },
        { 112, 112 },
        { 128, 128 },
        { 192, 192 },
        { 192, 128 },
        { 256, 256 },
    };

    // KV lengths per ne11 bucket, and per KV length: the smallest batch that uses the wide tile, the smallest one
    // that may pad its last tile with more rows than baseline, and two typical physical batch sizes
    const std::vector<std::vector<int>> ne11_rep = { { 512, 1024, 2048 }, { 4096 }, { 8192 }, { 16384 }, { 32768 }, { 65536 } };
    const int ne01_rep[] = { ggml_metal_tuning::FA_NE01_MIN, 8*ggml_metal_tuning::FA_NE01_MIN_PARTIAL_TILES + 8, 512, 2048 };

    // 8 query heads, 32..1024 wide tiles
    const int ne01_rep_small[] = { 64, 128, 256, 512, 1024, 2048 };

    const int TILES_LARGE = 1024;  // a row must win from this launch size up

    // cands[0] is the baseline
    const ggml_metal_tuning::fa_cfg_t cands[] = {
        { ggml_metal_tuning::FA_Q_BASELINE, 0 },
        { ggml_metal_tuning::FA_Q_WIDE,     4 },
        { ggml_metal_tuning::FA_Q_WIDE,     8 },
    };
    const size_t n_cands = std::size(cands);

    const double TUNE_THETA = 1.02;   // min speedup vs baseline at large launches to emit a row
                                      // (FA_NE01_MIN_PARTIAL_TILES is sized against this)
    const double TUNE_TIE   = 1.03;   // min speedup vs an earlier candidate to replace it
    const double TUNE_EPS   = 0.015;  // a win must clear this, and a loss must exceed it
    const double TUNE_BAND  = 0.03;   // a cell this close to baseline is re-measured, so that decisions
                                      // either side of TUNE_EPS rest on the same number of measurements
    const int    TUNE_RETRY = 2;      // extra measurements of such a cell, always all of them, all count

    const cooldown_opts cool = {
        opts.cooldown, opts.cool_drift, opts.cool_eps, opts.cool_max_wait, opts.cool_max_retry,
    };

    fprintf(stderr, "seed=%u reps=%d cooldown=%s\n", opts.seed, opts.reps, cool.enabled ? "on" : "off");
    fprintf(stderr, "device token: %s\n", dev_token(dev));

    for (auto s : shapes) {
        if (!fa_filter_has(opts.dk_filter, std::to_string(s.dk).c_str())) {
            continue;
        }

        // per ne11 bucket: the fastest config at large launches (or 0) and its tiles_min
        std::vector<int> best;
        std::vector<int> best_tiles;
        std::set<int>    tiles_all = { TILES_LARGE };

        for (const auto & ne11_pts : ne11_rep) {
            std::vector<double> t_large(n_cands, 0.0);
            std::vector<bool>   ok(n_cands, true);
            std::vector<int>    tiles_bad(n_cands, 0);

            // the host rejects the wide tile when the padded V head does not split over the simdgroups
            for (size_t i = 1; i < n_cands; ++i) {
                ok[i] = GGML_PAD(s.dv, 64) % (16*cands[i].NSG) == 0;
            }

            struct pt_t {
                int ne11, ne01, nh;
            };

            std::vector<pt_t> pts;
            for (int ne11 : ne11_pts) {
                for (int ne01 : ne01_rep) {
                    if (ne01 <= ne11) {
                        pts.push_back({ ne11, ne01, FA_NH });
                    }
                }
                for (int ne01 : ne01_rep_small) {
                    if (ne01 <= ne11) {
                        pts.push_back({ ne11, ne01, 1 });
                    }
                }
            }

            for (const auto & [ne11, ne01, nh] : pts) {
                const fa_shape sh = { s.dk, s.dv, ne01, ne11, GGML_TYPE_F16, nh };

                const int tiles = ((ne01 + ggml_metal_tuning::FA_Q_WIDE - 1)/ggml_metal_tuning::FA_Q_WIDE)*nh*FA_NR2;

                perf_cell cell = build_perf_cell(
                    backend, [&](ggml_context * ctx) { return fa_build_graph(ctx, sh); },
                    [&](ggml_context * ctx) { fa_init_tensors(ctx, sh, opts.seed); },
                    [&](ggml_tensor *) { return fa_op_flops(sh); });

                char label[128];
                snprintf(label, sizeof(label), "dk=%d dv=%d ne11=%d ne01=%d tiles=%d", s.dk, s.dv, ne11, ne01, tiles);

                // a cell whose first measurement is within TUNE_BAND of baseline is measured TUNE_RETRY more times;
                // every measurement is kept and the decision uses the median ratio over all of them
                std::vector<cell_result> rs;
                for (int retry = 0; cell.gf != nullptr && retry <= TUNE_RETRY; ++retry) {
                    cell_result r1 = measure_cell(
                        backend, cell, opts.reps, { 2, 1, 0 }, [&](int i) { set_ov(cands[i].Q, cands[i].NSG); },
                        [&]() { clr_ov(); }, 0, cool, label);
                    if (!r1.trusted || r1.t[0] <= 0.0) {
                        rs.clear();
                        break;
                    }
                    rs.push_back(r1);
                    if (retry == 0) {
                        bool close = false;
                        for (size_t i = 1; i < n_cands; ++i) {
                            close = close || (ok[i] && r1.t[i] > 0.0 && std::fabs(r1.t[0]/r1.t[i] - 1.0) <= TUNE_BAND);
                        }
                        if (!close) {
                            break;
                        }
                        fprintf(stderr, "# RETRY %s (within %.1f%% of baseline)\n", label, 100.0*TUNE_BAND);
                    }
                    fprintf(stderr, "#   attempt %d: Q8=%.1f", retry, r1.t[0]);
                    for (size_t i = 1; i < n_cands; ++i) {
                        fprintf(stderr, "  Q%dNSG%d=%.1f", cands[i].Q, cands[i].NSG, r1.t[i]);
                    }
                    fprintf(stderr, "\n");
                }

                cell_result r;
                if (!rs.empty()) {
                    // per candidate: the median time and the median baseline/candidate ratio across the measurements
                    r.t.assign(n_cands, 0.0);
                    for (size_t i = 0; i < n_cands; ++i) {
                        std::vector<double> v;
                        for (const auto & r1 : rs) {
                            v.push_back(i == 0 ? r1.t[0] : r1.t[0]/r1.t[i]);
                        }
                        std::sort(v.begin(), v.end());
                        r.t[i] = i == 0 ? v[v.size()/2] : r.t[0]/v[v.size()/2];
                    }
                } else {
                    r.trusted = false;
                }

                if (cell.gf == nullptr || !r.trusted) {
                    fprintf(stderr, "# DROP untrusted cell %s\n", label);
                    ok.assign(n_cands, false);
                    continue;
                }

                // a small launch of full tiles moves tiles_min, any other loss drops the config
                const bool small = tiles < TILES_LARGE && ne01 % ggml_metal_tuning::FA_Q_WIDE == 0;
                if (small) {
                    tiles_all.insert(tiles);
                }

                fprintf(stderr, "# %s: Q8=%.1f", label, r.t[0]);
                for (size_t i = 1; i < n_cands; ++i) {
                    if (GGML_PAD(s.dv, 64) % (16*cands[i].NSG) != 0) {
                        fprintf(stderr, "  Q%dNSG%d=n/a", cands[i].Q, cands[i].NSG);
                        continue;
                    }
                    fprintf(stderr, "  Q%dNSG%d=%.1f (%.3fx)", cands[i].Q, cands[i].NSG, r.t[i], r.t[0] / r.t[i]);

                    // a small launch moves tiles_min instead of dropping the config: a full tile of queries
                    // on anything short of a win, a partial one only on a clear loss, as it also pays for the
                    // rows the wide tile pads. A clear loss at a large launch drops the config
                    if (!(r.t[i] > 0.0)) {
                        ok[i] = false;
                        continue;
                    }
                    const bool win  = r.t[i] * (1.0 + TUNE_EPS) <= r.t[0];
                    const bool loss = r.t[0] * (1.0 + TUNE_EPS) <= r.t[i];
                    if (tiles < TILES_LARGE && (loss || (small && !win))) {
                        tiles_bad[i] = std::max(tiles_bad[i], tiles);
                    } else if (loss) {
                        ok[i] = false;
                    }
                }
                fprintf(stderr, "\n");

                if (tiles >= TILES_LARGE) {
                    for (size_t i = 0; i < n_cands; ++i) {
                        t_large[i] += r.t[i];
                    }
                }
            }

            int best_i = 0;
            for (size_t i = 1; i < n_cands; ++i) {
                if (ok[i] && ok[0] && t_large[i] * TUNE_THETA <= t_large[0] && (best_i == 0 || t_large[i] * TUNE_TIE <= t_large[best_i])) {
                    best_i = (int) i;
                }
            }

            // tiles_min: the first sampled launch size with no loss at or above it
            const int tiles_min = best_i ? *tiles_all.upper_bound(tiles_bad[best_i]) : 0;

            fprintf(stderr, "# dk=%d dv=%d ne11=%d => Q%d,NSG%d tiles_min=%d\n", s.dk, s.dv, ne11_pts.back(), cands[best_i].Q, cands[best_i].NSG, tiles_min);
            best.push_back(best_i);
            best_tiles.push_back(tiles_min);
        }

        // the same row in every bucket -> one ne11-collapsed default row
        bool collapse = true;
        for (size_t b = 1; b < best.size(); ++b) {
            collapse = collapse && best[b] == best[0] && best_tiles[b] == best_tiles[0];
        }

        for (size_t b = 0; b < best.size(); ++b) {
            if (best[b] == 0 || (collapse && b > 0)) {
                continue;
            }
            printf("    { { %s, %d, %d, %d }, { %d, %d }, %d },\n", dev_token(dev),
                   collapse ? (int) ggml_metal_tuning::FA_NE11_DEFAULT : ne11_b(ne11_rep[b].back()), s.dk, s.dv,
                   cands[best[b]].Q, cands[best[b]].NSG, best_tiles[b]);
        }
        fflush(stdout);
    }

    return true;
}
