#include "mul-mm.h"

#include "bench.h"
#include "ggml-backend.h"
#include "ggml-metal-tuning.h"
#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// --- Metal proc bridges (reached via ggml_backend_reg_get_proc_address) ---
using set_override_t   = void (*)(int, int);
using clear_override_t = void (*)(void);
using set_ne11_min_t   = void (*)(int);
using bucket_t         = int (*)(int64_t);
using device_token_t   = const char * (*) (ggml_backend_dev_t);

struct mm_procs {
    set_override_t   set_ov     = nullptr;
    clear_override_t clr_ov     = nullptr;
    set_ne11_min_t   set_sw     = nullptr;  // ne11_mm_min override: 1 forces mm, 8 forces mv_ext, -1 clears
    bucket_t         N0_bucket  = nullptr;
    bucket_t         tok_bucket = nullptr;
    device_token_t   dev_token  = nullptr;

    bool ok() const { return set_ov && clr_ov && set_sw && N0_bucket && tok_bucket && dev_token; }
};

static mm_procs mm_resolve_procs(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

    mm_procs p;
    p.set_ov     = (set_override_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_set_mm_tile_override");
    p.clr_ov     = (clear_override_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_clear_mm_tile_override");
    p.set_sw     = (set_ne11_min_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_set_mm_tile_ne11_mm_min_override");
    p.N0_bucket  = (bucket_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_mm_tile_N0_bucket");
    p.tok_bucket = (bucket_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_mm_tile_token_bucket");
    p.dev_token  = (device_token_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_device_token");

    return p;
}

// "q4_K" -> "GGML_TYPE_Q4_K". The emitted rows are C++ source and need the enum spelling,
// not the display name; deriving it keeps a newly swept type from carrying a stale token.
static std::string mm_dtype_token(ggml_type dt) {
    std::string s = "GGML_TYPE_";
    for (const char * p = ggml_type_name(dt); *p; ++p) {
        s += (char) toupper((unsigned char) *p);
    }
    return s;
}

static bool mm_filter_has(const char * filter, const char * name) {
    if (!filter) {
        return true;
    }
    const std::string f = std::string(",") + filter + ",";
    return f.find(std::string(",") + name + ",") != std::string::npos;
}

// C^T = A * B^T: A(K,M) quantized weight, B(K,tokens) f32 activations -> C(M,tokens).
// M is the output feature count (ne01), tokens the batch width (ne11).
static ggml_tensor * mm_build_graph(ggml_context * ctx, ggml_type dt, int M, int K, int tokens) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, dt,             K, M);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, tokens);
    ggml_set_name(b, "b");
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_set_name(out, "out");
    return out;
}

static uint64_t mm_op_flops(int M, int K, int tokens) {
    return (uint64_t) 2 * M * tokens * K;
}

static unsigned mm_cell_seed(ggml_type dt, int K, int N0, int tokens, unsigned base) {
    unsigned h = base;
    for (int v : { K, N0, tokens, (int) dt }) {
        h = h * 1000003u + (unsigned) v;
    }
    return h;
}

static void mm_init_uniform(ggml_tensor * t, std::mt19937 & rng, float min, float max) {
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

static void mm_init_tensors(ggml_context * ctx, ggml_type dt, int K, int N0, int tokens, unsigned base_seed) {
    std::mt19937 rng(mm_cell_seed(dt, K, N0, tokens, base_seed));

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src != NULL || t->op != GGML_OP_NONE) {
            continue;  // views share data; op results are computed
        }
        mm_init_uniform(t, rng, -1.0f, 1.0f);
    }
}

bool tuner_mul_mm_run(ggml_backend_t backend, ggml_backend_dev_t dev, const tuner_opts & opts) {
    const mm_procs procs = mm_resolve_procs(dev);
    if (!procs.ok()) {
        fprintf(stderr, "error: metal mm_tile tuning procs unavailable\n");
        return false;
    }

    const char * dev_token = procs.dev_token(dev);

    // sweep grid (shape-zoo of real model (K, N0) pairs sampled against a token ladder).
    // (K, N0) is sampled in PAIRS, not a cartesian grid: real weights only occupy a few
    // diagonals of the (K, N0) plane (proj / FFN-up / FFN-down / vocab). K is not a table
    // key (it scales all candidates equally and never flips the ranking), but deep-K
    // shapes stay in the zoo so the N0/token aggregation measures them instead of
    // extrapolating.
    struct shape_t { int K, N0; };
    const shape_t zoo[] = {
        // proj (q/k/v/o): K=hidden, N0=hidden or kv-dim
        { 1536, 1536 }, { 3584, 3584 }, { 4096, 4096 }, { 5120, 5120 },
        { 4096, 1024 }, { 8192, 8192 }, { 8192, 1024 },
        // FFN up (gate/up): K=hidden, N0=intermediate. 2048/8192 samples the shallow-K side
        // of the [8192,30000) N0 bucket (1B/3B-class gate/up).
        { 4096, 14336 }, { 3584, 18944 }, { 8192, 28672 }, { 5120, 27648 },
        { 2048, 8192 },
        // FFN down: K=intermediate, N0=hidden. 5120/4608 N0 samples the [4608,8192) bucket
        // so 13B/14B/32B/Gemma-27B down_proj gets measured, not extrapolated.
        { 14336, 4096 }, { 18944, 3584 }, { 28672, 8192 },
        { 17408, 5120 }, { 13824, 5120 }, { 36864, 4608 },
        // vocab (lm_head): K=hidden, N0=vocab (huge outlier)
        { 4096, 128256 }, { 3584, 152064 },
        // exotic: DeepSeek-V3 tall-thin kv_b/q_b + Qwen3-MoE expert small-K
        { 32768, 512 }, { 24576, 1536 }, { 2048, 768 },
    };
    // token ladder ends at 192: tokens >= 256 is short-circuited to baseline by mm_tile_pick
    // (the baseline dispatch saturates the GPU there). When tuning a NEW device, temporarily
    // re-add {256, 512} once to confirm the short-circuit holds. The low end [2,8] is the
    // range a lowered switch point can route into mm (measured with the mm path forced); it
    // is token bucket 0 and feeds both that bucket's tile and the switch point.
    const int tokens_full[] = { 2, 3, 4, 5, 6, 7, 8, 9, 16, 24, 32, 48, 64, 96, 128, 192 };

    const double TUNE_TAU   = 0.05;
    const double TUNE_THETA = 1.05;
    // A tile is emitted only if it never loses to baseline by >TUNE_FLOOR at a real prefill
    // token. Off-ladder tokens (9/24/48/96/192) still feed min-max-regret but do not authorize
    // a tile: baseline's period-32 tail-waste lets a tile win at t=48 yet lose at the t=32/64
    // the runtime visits.
    const double TUNE_FLOOR = 0.02;
    // Conservative margin for the routing decision: route a small batch into mm only if the
    // tok0 tile beats mv_ext by this much at every shape (a near-tie stays mv_ext).
    const double TUNE_SWITCH_MARGIN = 0.05;
    auto is_real_token = [](int t) {
        return (t >= 2 && t <= 8) || t == 16 || t == 32 || t == 64 || t == 128;
    };
    // Occupancy-saturation threshold, used only for the emit-time sanity report below;
    // mirrors MM_TILE_C_SAT_M4_MAX.
    const int64_t C_SAT = ggml_metal_tuning::MM_TILE_C_SAT_M4_MAX;
    const int     NE11_MM_MIN_DEFAULT = ggml_metal_tuning::MM_TILE_NE11_MM_MIN_DEFAULT;
    auto n_tg_base = [](int64_t N0, int64_t tokens) {
        return ((N0 + 63) / 64) * ((tokens + 31) / 32);
    };

    // candidates = the sweep-budget geometry list + the baseline anchor (read from the
    // runtime header, so the sweep can never drift from what the runtime can serve).
    struct mm_cand { int nr0, nr1; };
    std::vector<mm_cand> cands;
    for (const auto & f : ggml_metal_tuning::MM_TILE_SWEEP_CANDIDATES) {
        cands.push_back({ f.nr0, f.nr1 });
    }
    const int base_i = (int) cands.size();
    cands.push_back({ ggml_metal_tuning::MM_TILE_BASELINE_CFG.nr0, ggml_metal_tuning::MM_TILE_BASELINE_CFG.nr1 });

    // Every src0 type the Metal mul_mm kernel is instantiated for. All of them are
    // tile-eligible, so all of them are swept; --dtype narrows this for a split run.
    const ggml_type dtypes[] = {
        GGML_TYPE_F32,     GGML_TYPE_F16,     GGML_TYPE_BF16,
        GGML_TYPE_Q1_0,    GGML_TYPE_Q2_0,    GGML_TYPE_Q4_0,   GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,    GGML_TYPE_Q5_1,    GGML_TYPE_Q8_0,   GGML_TYPE_MXFP4,
        GGML_TYPE_Q2_K,    GGML_TYPE_Q3_K,    GGML_TYPE_Q4_K,   GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S,   GGML_TYPE_IQ2_S,   GGML_TYPE_IQ1_S,  GGML_TYPE_IQ1_M,
        GGML_TYPE_IQ4_NL,  GGML_TYPE_IQ4_XS,  GGML_TYPE_TQ2_0,
    };

    // mul_mv_ext is a routing rival, not a tile: it stays out of the tile group_split and
    // only feeds the switch-point analysis. The sweep forces the mm path (set_sw(1)) to time
    // the tile at t<=8, and forces the upstream bound (set_sw(8)) to time mv_ext.
    // Mirror of the dispatch condition in ggml-metal-ops.cpp: only these src0 types have an
    // mv_ext kernel, and the K-series one needs ne11 >= 4. A type absent from both lists has
    // no mv_ext rival to measure, so it gets no switch row and keeps the upstream dispatch.
    auto mv_ext_k_series = [](ggml_type dt) {
        return dt == GGML_TYPE_Q2_K || dt == GGML_TYPE_Q3_K || dt == GGML_TYPE_Q4_K ||
               dt == GGML_TYPE_Q5_K || dt == GGML_TYPE_Q6_K;
    };
    auto mv_ext_applies = [&](ggml_type dt, int t) {
        if (t < 2 || t > 8) { return false; }
        if (mv_ext_k_series(dt)) { return t >= 4; }
        switch (dt) {
            case GGML_TYPE_F32:  case GGML_TYPE_F16:  case GGML_TYPE_BF16:
            case GGML_TYPE_Q1_0: case GGML_TYPE_Q2_0: case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0: case GGML_TYPE_MXFP4: case GGML_TYPE_IQ4_NL:
                return true;
            default:
                return false;
        }
    };

    const cooldown_opts cool = {
        opts.cooldown, opts.cool_drift, opts.cool_eps, opts.cool_max_wait, opts.cool_max_retry,
    };

    fprintf(stderr, "seed=%u reps=%d cooldown=%s (drift=%.2f eps=%.2f max_wait=%ds max_retry=%d)\n", opts.seed,
            opts.reps, cool.enabled ? "on" : "off", cool.drift, cool.eps, cool.max_wait, cool.max_retry);
    fprintf(stderr, "device token: %s\n", dev_token);

    int n_untrusted = 0;

    struct pt_t { int K, N0, tokens; std::vector<double> ts; double base_t; double mv_ext_t; };

    // accumulate pasteable rows for the two tables; stdout stays rows-only.
    std::vector<std::string> tile_rows;
    std::vector<std::string> switch_rows;
    char rbuf[256];

    for (const ggml_type dt : dtypes) {
        if (!mm_filter_has(opts.dtype_filter, ggml_type_name(dt))) {
            continue;
        }
        const std::string dtype_token = mm_dtype_token(dt);

        fprintf(stderr, "\n### dtype=%s\n", ggml_type_name(dt));

        std::vector<pt_t> pts;

        for (const shape_t & sh : zoo) {
            if (sh.K % ggml_blck_size(dt) != 0) { continue; }  // K not aligned to the dtype block

            for (int tokens : tokens_full) {
                perf_cell cell = build_perf_cell(
                    backend,
                    [&](ggml_context * ctx) { return mm_build_graph(ctx, dt, sh.N0, sh.K, tokens); },
                    [&](ggml_context * ctx) { mm_init_tensors(ctx, dt, sh.K, sh.N0, tokens, opts.seed); },
                    [&](ggml_tensor *)      { return mm_op_flops(sh.N0, sh.K, tokens); });
                if (cell.gf == nullptr) { continue; }

                std::vector<int> order((size_t) cands.size());
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), std::mt19937(mm_cell_seed(dt, sh.K, sh.N0, tokens, opts.seed)));

                char label[128];
                snprintf(label, sizeof(label), "K=%d N0=%d tokens=%d", sh.K, sh.N0, tokens);

                procs.set_sw(1);  // force the mm path so the tile candidates run at t<=8 too
                cell_result r = measure_cell(
                    backend, cell, opts.reps, order,
                    [&](int i) { procs.set_ov(cands[i].nr0, cands[i].nr1); },
                    [&]()      { procs.clr_ov(); },
                    base_i, cool, label);

                // mv_ext rival for the small-batch range: force the upstream switch point so the
                // dispatch selects mul_mv_ext, then time the same cell.
                double mv_ext_t = -1.0;
                if (r.trusted && mv_ext_applies(dt, tokens)) {
                    procs.set_sw(8);
                    mv_ext_t = time_cell_median(backend, cell, opts.reps);
                }
                procs.set_sw(-1);  // restore the per-dtype switch-point table between cells

                if (r.anchor_min > 0.0) {
                    fprintf(stderr, "# noise K=%d N0=%d tokens=%d spread=%.1f%%\n", sh.K, sh.N0, tokens,
                            100.0 * (r.anchor_max - r.anchor_min) / r.anchor_min);
                }
                if (!r.trusted) {
                    n_untrusted++;
                    fprintf(stderr, "# DROP untrusted cell K=%d N0=%d tokens=%d\n", sh.K, sh.N0, tokens);
                    continue;
                }

                int best = -1;
                for (int i = 0; i < (int) cands.size(); ++i) {
                    if (r.t[i] > 0.0 && (best < 0 || r.t[i] < r.t[best])) { best = i; }
                }
                const double base_t = r.t[base_i];
                fprintf(stderr, "# dt=%s K=%d N0=%d tokens=%d:", ggml_type_name(dt), sh.K, sh.N0, tokens);
                for (int i = 0; i < (int) cands.size(); ++i) {
                    fprintf(stderr, "  %dx%d=%.1f%s", cands[i].nr0, cands[i].nr1, r.t[i], (i == best) ? "*" : "");
                }
                if (mv_ext_t > 0.0) {
                    fprintf(stderr, "  mv_ext=%.1f", mv_ext_t);
                    if (best >= 0 && r.t[best] > 0.0) { fprintf(stderr, " (mm %.2fx mv_ext)", mv_ext_t / r.t[best]); }
                }
                const bool keep = best >= 0 && base_t > 0.0 && r.t[best] < base_t * 0.98;
                if (keep) { fprintf(stderr, "  => %dx%d  %.2fx\n", cands[best].nr0, cands[best].nr1, base_t / r.t[best]); }
                else      { fprintf(stderr, "  => baseline\n"); }

                pts.push_back({ sh.K, sh.N0, tokens, r.t, base_t, mv_ext_t });
            }
        }

        // group_split: tokens is the strongest signal and never folds, so each token bucket is
        // its own group. Within a token bucket, points aggregate per N0 bucket ACROSS K.
        // group_split finds one L2 default cfg (ANY, tok_b) plus exact-row exceptions where that
        // default is insufficient; both are gated by the real-token floor.
        std::vector<std::string> dtype_tile_rows;
        int                          tok0_L2 = base_i;  // token-bucket-0 default cfg index
        std::unordered_map<int, int> tok0_L1;           // N0_b -> exact cfg index

        std::set<int> token_buckets;
        for (const auto & p : pts) { token_buckets.insert(procs.tok_bucket(p.tokens)); }

        for (int tb : token_buckets) {
            std::vector<const pt_t *> db;
            for (const auto & p : pts) {
                if (procs.tok_bucket(p.tokens) == tb) { db.push_back(&p); }
            }
            if (db.empty()) { continue; }

            std::set<int> seen;
            for (const auto * p : db) { seen.insert(procs.N0_bucket(p->N0)); }

            struct bkt_t { int bN0; int Ti; std::vector<double> agg; double base_agg;
                           std::vector<const pt_t *> bp; };
            std::vector<bkt_t> bks;
            for (int bN0 : seen) {
                std::vector<const pt_t *> bp;
                for (const auto * p : db) {
                    if (procs.N0_bucket(p->N0) == bN0) { bp.push_back(p); }
                }
                std::vector<double> agg(cands.size(), 0.0), worst(cands.size(), 0.0);
                for (const auto * p : bp) {
                    double bestt = 0.0;
                    for (int i = 0; i < (int) cands.size(); ++i) {
                        if (p->ts[i] > 0.0 && (bestt == 0.0 || p->ts[i] < bestt)) { bestt = p->ts[i]; }
                    }
                    for (int i = 0; i < (int) cands.size(); ++i) {
                        agg[i] += p->ts[i];
                        if (p->ts[i] > 0.0 && bestt > 0.0) { worst[i] = std::max(worst[i], p->ts[i] / bestt); }
                    }
                }
                int robust = 0;
                for (int i = 1; i < (int) cands.size(); ++i) {
                    if (worst[i] < worst[robust] ||
                        (worst[i] == worst[robust] &&
                         std::make_pair(cands[i].nr0, cands[i].nr1) < std::make_pair(cands[robust].nr0, cands[robust].nr1))) {
                        robust = i;
                    }
                }
                // theta gate on the per-point geomean of base/robust (equal weight per shape).
                double theta = 0.0;
                {
                    double lsum = 0.0; int ln = 0;
                    for (const auto * p : bp) {
                        double tb2 = p->ts[base_i], tr = p->ts[robust];
                        if (tb2 > 0.0 && tr > 0.0) { lsum += std::log(tb2 / tr); ln++; }
                    }
                    if (ln > 0) { theta = std::exp(lsum / ln); }
                }
                bool tune = robust != base_i && theta >= TUNE_THETA;
                if (tune) {
                    double wr = 0.0;
                    for (const auto * p : bp) {
                        if (!is_real_token(p->tokens)) { continue; }
                        double td = p->ts[robust], tb2 = p->ts[base_i];
                        if (td > 0.0 && tb2 > 0.0) { wr = std::max(wr, td / tb2); }
                    }
                    if (wr > 1.0 + TUNE_FLOOR) { tune = false; }
                }
                bks.push_back({ bN0, tune ? robust : base_i, agg, agg[base_i], bp });
            }

            auto reg_pw = [&](const bkt_t * b, int d) {
                double r2 = 0.0;
                for (const auto * p : b->bp) {
                    double td = p->ts[d], tT = p->ts[b->Ti];
                    if (td > 0.0 && tT > 0.0) { r2 = std::max(r2, td / tT - 1.0); }
                }
                return r2;
            };

            // emit-time sanity report for a non-baseline row, cross-checked against the occupancy
            // rule that backs the runtime veto (only samples the veto can fire on: real token AND
            // t>=32). An all-saturated row is unreachable; a disputed row marks an
            // extrapolation-fragile cell.
            auto sanity_row = [&](const std::vector<const pt_t *> & bp, int d, const char * kind) {
                if (d == base_i) { return; }
                int n_vet = 0, n_sat = 0;
                double worst_sat = 0.0;
                for (const auto * p : bp) {
                    if (!is_real_token(p->tokens) || p->tokens < 32) { continue; }
                    n_vet++;
                    if (n_tg_base(p->N0, p->tokens) >= C_SAT) {
                        n_sat++;
                        double td = p->ts[d], tb2 = p->ts[base_i];
                        if (td > 0.0 && tb2 > 0.0) { worst_sat = std::max(worst_sat, td / tb2); }
                    }
                }
                if (n_vet > 0 && n_sat == n_vet) {
                    fprintf(stderr, "# SANITY %s row unreachable: all %d vetoable real-token samples sit at"
                            " n_tg>=%lld, the runtime veto overrides this tile\n", kind, n_vet, (long long) C_SAT);
                } else if (worst_sat > 1.05) {
                    fprintf(stderr, "# SANITY %s row disputed: tile loses %.0f%% to baseline at a saturated"
                            " real-token sample; consider adding sibling shapes to the zoo\n",
                            kind, 100.0 * (worst_sat - 1.0));
                }
            };

            int bestD = -1, bestRows = 1 << 30; double bestTot = 0.0;
            for (int d = 0; d < (int) cands.size(); ++d) {
                int rows = (d != base_i) ? 1 : 0; double tot = 0.0;
                for (const auto & _b : bks) {
                    const bkt_t * b = &_b;
                    double reg  = reg_pw(b, d);
                    double slow = b->agg[d] / b->base_agg - 1.0;
                    if (reg > TUNE_TAU || slow > TUNE_TAU) { rows++; tot += b->agg[b->Ti]; }
                    else                                   {         tot += b->agg[d];      }
                }
                bool better = bestD < 0 || rows < bestRows ||
                    (rows == bestRows && (tot < bestTot ||
                     (tot == bestTot &&
                      std::make_pair(cands[d].nr0, cands[d].nr1) < std::make_pair(cands[bestD].nr0, cands[bestD].nr1))));
                if (better) { bestD = d; bestRows = rows; bestTot = tot; }
            }
            // same floor on the L2 default: it is served to unsampled cells, so it must not lose
            // to baseline at any real token across the bucket.
            if (bestD != base_i) {
                double wr = 0.0;
                for (const auto & _b : bks) {
                    for (const auto * p : _b.bp) {
                        if (!is_real_token(p->tokens)) { continue; }
                        double td = p->ts[bestD], tb2 = p->ts[base_i];
                        if (td > 0.0 && tb2 > 0.0) { wr = std::max(wr, td / tb2); }
                    }
                }
                if (wr > 1.0 + TUNE_FLOOR) { bestD = base_i; }
            }
            if (bestD != base_i) {
                std::vector<const pt_t *> all_bp;
                for (const auto & _b : bks) { all_bp.insert(all_bp.end(), _b.bp.begin(), _b.bp.end()); }
                sanity_row(all_bp, bestD, "L2");
                snprintf(rbuf, sizeof(rbuf),
                    "    { { %s, %s, -1, %d }, { %d, %d } },",
                    dev_token, dtype_token.c_str(), tb, cands[bestD].nr0, cands[bestD].nr1);
                dtype_tile_rows.emplace_back(rbuf);
            }
            if (tb == 0) { tok0_L2 = bestD; }
            for (const auto & _b : bks) {
                const bkt_t * b = &_b;
                if (reg_pw(b, bestD) <= TUNE_TAU && b->agg[bestD] / b->base_agg - 1.0 <= TUNE_TAU) { continue; }
                sanity_row(b->bp, b->Ti, "L1");
                if (tb == 0) { tok0_L1[b->bN0] = b->Ti; }
                snprintf(rbuf, sizeof(rbuf),
                    "    { { %s, %s, %d, %d }, { %d, %d } },",
                    dev_token, dtype_token.c_str(), b->bN0, tb, cands[b->Ti].nr0, cands[b->Ti].nr1);
                dtype_tile_rows.emplace_back(rbuf);
            }
        }

        snprintf(rbuf, sizeof(rbuf), "    // ---- %s: %zu rows ----", ggml_type_name(dt), dtype_tile_rows.size());
        tile_rows.emplace_back(rbuf);
        tile_rows.insert(tile_rows.end(), dtype_tile_rows.begin(), dtype_tile_rows.end());

        // ---- per-(dtype, N0-bucket) mv_ext->mm switch point ----
        // The tile for token bucket 0 was just emitted from the same [2,8] points; here we pick
        // which small batches to route into it. Per (N0 bucket, t), take the worst-case (min over
        // shapes) mv_ext_t / tok0_tile_t (tile = production pick, L1->L2->baseline) and route only
        // if it clears TUNE_SWITCH_MARGIN at every shape. Switch point = bottom of the contiguous
        // safe run from t=8.
        auto tok0_tile_index = [&](int N0_b) {
            auto it = tok0_L1.find(N0_b);
            return it != tok0_L1.end() ? it->second : tok0_L2;
        };
        {
            std::set<int> n0bs;
            for (const auto & p : pts) { n0bs.insert(procs.N0_bucket(p.N0)); }
            std::set<int>                sw_rows_b;
            std::unordered_map<int, int> sw_by_b;
            for (int b : n0bs) {
                const int ti = tok0_tile_index(b);
                fprintf(stderr, "    // %s N0_b=%d tile=%dx%d mv_ext crossover:", ggml_type_name(dt), b,
                        cands[ti].nr0, cands[ti].nr1);
                std::unordered_map<int, bool> safe;
                for (int t = 2; t <= NE11_MM_MIN_DEFAULT; ++t) {
                    double min_ratio = 0.0; int n = 0;
                    for (const auto & p : pts) {
                        if (procs.N0_bucket(p.N0) != b || p.tokens != t || p.mv_ext_t <= 0.0) { continue; }
                        double tt = p.ts[ti];
                        if (tt > 0.0) { double r2 = p.mv_ext_t / tt; min_ratio = (n == 0) ? r2 : std::min(min_ratio, r2); n++; }
                    }
                    if (n == 0) { continue; }
                    safe[t] = min_ratio >= 1.0 + TUNE_SWITCH_MARGIN;
                    fprintf(stderr, " t%d=%.2f%s", t, min_ratio, safe[t] ? "" : "(mv)");
                }
                int sw = NE11_MM_MIN_DEFAULT;
                for (int t = NE11_MM_MIN_DEFAULT; t >= 2; --t) {
                    auto it = safe.find(t);
                    if (it == safe.end()) { break; }  // gap in the ladder ends the run
                    if (it->second) { sw = t - 1; } else { break; }
                }
                // mv_ext starts at ne11 = 4 for the K-series, so a K-series switch point can
                // never authorize routing t < 4 into mm: nothing measured that range.
                GGML_ASSERT(!mv_ext_k_series(dt) || sw >= 3);
                fprintf(stderr, " => ne11_mm_min=%d\n", sw);
                if (sw < NE11_MM_MIN_DEFAULT) { sw_by_b[b] = sw; sw_rows_b.insert(b); }
            }
            for (int b : sw_rows_b) {
                snprintf(rbuf, sizeof(rbuf), "    { %s, %s, %d, %d },",
                         dev_token, dtype_token.c_str(), b, sw_by_b[b]);
                switch_rows.emplace_back(rbuf);
            }
        }
    }

    // stdout carries nothing but pasteable rows, in two labelled table sections.
    printf("// ==== mm_tile_tuned_table (paste into ggml-metal-tuning.cpp) ====\n");
    for (const auto & r : tile_rows) { printf("%s\n", r.c_str()); }
    printf("\n// ==== mm_tile_ne11_mm_min_table ====\n");
    for (const auto & r : switch_rows) { printf("%s\n", r.c_str()); }
    fflush(stdout);

    if (n_untrusted > 0) {
        fprintf(stderr, "\n%d cells excluded as untrusted\n", n_untrusted);
    }

    return true;
}
