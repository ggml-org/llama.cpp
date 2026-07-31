//
// test_operative_routing.cpp
//
// Verifies S3: the regime router is operative. Routing a tensor to an
// expert selects a distinctive parameter profile, and quantizing under
// different expert profiles produces different outputs.
//

#include "tessera-regime.h"
#include "tessera-quant.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// --- 1. routing selects distinct experts for distinct regimes ---

static int test_routing_distinct() {
    struct { const char * name; const char * family; float kurt; float er; ts_expert_id want; } cases[] = {
        { "blk.0.ffn_down.weight", "ffn_down", 15.0f, 0.50f, TS_EXPERT_DARTQUANT },
        { "blk.0.ffn_gate.weight", "ffn_gate",  7.0f, 0.50f, TS_EXPERT_CHAMPQ    },
        { "blk.0.attn_q.weight",   "attn_q",    2.0f, 0.10f, TS_EXPERT_FLRQ      },
        { "blk.1.attn_q.weight",   "attn_q",    2.0f, 0.25f, TS_EXPERT_LRQ       },
        { "blk.0.attn_k.weight",   "attn_k",    2.5f, 0.80f, TS_EXPERT_AWQ       },
    };

    bool seen[TS_EXPERT_COUNT] = { false };
    int n_distinct = 0;

    for (const auto & c : cases) {
        ts_regime_descriptor d = {};
        d.tensor_name = c.name;
        d.family      = c.family;
        d.kurtosis    = c.kurt;
        d.eff_rank    = c.er;

        ts_regime_routing r = ts_regime_classify(&d);
        if (r.expert != c.want) {
            printf("FAIL routing: %s -> %s, expected %s\n",
                   c.name, ts_expert_name(r.expert), ts_expert_name(c.want));
            return 1;
        }
        if (!seen[r.expert]) {
            seen[r.expert] = true;
            n_distinct++;
        }
    }

    if (n_distinct < 4) {
        printf("FAIL routing: only %d distinct experts selected, expected >= 4\n", n_distinct);
        return 1;
    }
    printf("PASS routing: %d distinct experts selected across regimes\n", n_distinct);
    return 0;
}

// --- 2. each expert has a distinctive profile ---

static bool profile_differs(const ts_expert_profile & a, const ts_expert_profile & b) {
    return a.alpha_scale    != b.alpha_scale    ||
           a.clip_scale     != b.clip_scale     ||
           a.use_septq      != b.use_septq      ||
           a.awq_grid       != b.awq_grid       ||
           a.max_outliers   != b.max_outliers   ||
           a.outlier_thresh != b.outlier_thresh;
}

static int test_profiles_distinct() {
    ts_expert_profile base = ts_expert_default_profile(TS_EXPERT_AWQ);

    // AWQ is the identity baseline
    if (base.alpha_scale != 1.0f || base.clip_scale != 1.0f || base.use_septq ||
        base.awq_grid != 20 || base.max_outliers != 0 || base.outlier_thresh != 1.0f) {
        printf("FAIL profile: AWQ is not the identity baseline\n");
        return 1;
    }

    struct { ts_expert_id id; const char * note; } others[] = {
        { TS_EXPERT_LRQ,       "alpha_scale/clip_scale/use_septq" },
        { TS_EXPERT_DARTQUANT, "outlier_thresh/max_outliers"      },
        { TS_EXPERT_CHAMPQ,    "awq_grid/alpha_scale"             },
        { TS_EXPERT_FLRQ,      "use_septq/clip_scale"             },
        { TS_EXPERT_SEPTQ,     "use_septq/awq_grid"               },
    };
    for (const auto & o : others) {
        ts_expert_profile p = ts_expert_default_profile(o.id);
        if (!profile_differs(p, base)) {
            printf("FAIL profile: %s profile identical to AWQ baseline\n", ts_expert_name(o.id));
            return 1;
        }
    }

    // spot-check the documented values
    ts_expert_profile lrq = ts_expert_default_profile(TS_EXPERT_LRQ);
    if (lrq.alpha_scale != 0.9f || lrq.clip_scale != 0.85f || !lrq.use_septq) {
        printf("FAIL profile: LRQ values wrong (alpha=%.2f clip=%.2f septq=%d)\n",
               lrq.alpha_scale, lrq.clip_scale, (int)lrq.use_septq);
        return 1;
    }
    ts_expert_profile dart = ts_expert_default_profile(TS_EXPERT_DARTQUANT);
    if (dart.outlier_thresh != 0.8f || dart.max_outliers != 12) {
        printf("FAIL profile: DartQuant values wrong (thresh=%.2f outliers=%d)\n",
               dart.outlier_thresh, dart.max_outliers);
        return 1;
    }
    ts_expert_profile champ = ts_expert_default_profile(TS_EXPERT_CHAMPQ);
    if (champ.awq_grid != 40 || champ.alpha_scale != 1.1f) {
        printf("FAIL profile: CHAMP-Q values wrong (grid=%d alpha=%.2f)\n",
               champ.awq_grid, champ.alpha_scale);
        return 1;
    }

    printf("PASS profiles: all %d experts distinctive; AWQ is identity baseline\n", TS_EXPERT_COUNT);
    return 0;
}

// --- 3. different expert profiles produce different quantized bytes ---

// mirror the dispatch wiring: apply a profile to a base param set
static ts_quant_params_2d apply_profile(ts_quant_params_2d base, ts_expert_id e) {
    ts_expert_profile p = ts_expert_default_profile(e);
    base.alpha          *= p.alpha_scale;
    base.clip           *= p.clip_scale;
    base.use_septq       = p.use_septq;
    base.awq_grid        = p.awq_grid;
    base.max_outliers    = p.max_outliers;
    base.outlier_thresh *= p.outlier_thresh;
    return base;
}

static bool results_differ(const ts_quant_result_2d & a, const ts_quant_result_2d & b) {
    return a.packed       != b.packed       ||
           a.page_scales  != b.page_scales  ||
           a.lane_scales  != b.lane_scales  ||
           a.outlier_cols != b.outlier_cols ||
           a.outlier_vals != b.outlier_vals ||
           a.act_scale    != b.act_scale;
}

static int test_quant_differs() {
    const int64_t out_dim = 16;
    const int64_t in_dim  = 64;
    const int64_t n       = out_dim * in_dim;

    // deterministic weights with injected outlier columns
    std::vector<float> W((size_t)n);
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t c = 0; c < in_dim; c++) {
            float v = sinf((float)(r + 1) * 0.37f) * cosf((float)(c + 1) * 0.19f)
                    + 0.3f * sinf((float)(r * in_dim + c) * 0.013f);
            if (c % 16 == 0) {
                v *= 6.0f;   // heavy-tailed channels
            }
            W[(size_t)(r * in_dim + c)] = v;
        }
    }

    // varied per-channel activation magnitudes
    std::vector<float> act((size_t)in_dim);
    for (int64_t c = 0; c < in_dim; c++) {
        act[(size_t)c] = 0.5f + 0.5f * fabsf(sinf((float)(c + 1) * 0.31f));
        if (c % 16 == 0) {
            act[(size_t)c] *= 5.0f;
        }
    }

    ts_quant_params_2d base;
    base.alpha          = 0.5f;
    base.clip           = 0.6f;
    base.max_outliers   = 0;
    base.outlier_thresh = 0.05f;
    base.use_imatrix    = false;
    base.use_septq      = false;
    base.awq_grid       = 20;
    base.seed           = 1234;

    auto quant = [&](ts_expert_id e, ts_quant_result_2d * out) -> int {
        ts_quant_params_2d qp = apply_profile(base, e);
        return ts_quantize_2d(W.data(), act.data(), nullptr, nullptr, nullptr,
                              out_dim, in_dim, 0, &qp, out);
    };

    ts_quant_result_2d r_awq, r_other;
    if (quant(TS_EXPERT_AWQ, &r_awq) != 0) {
        printf("FAIL quant: AWQ quantize returned error\n");
        return 1;
    }

    // experts whose profile changes alpha / clip / outlier budget must change bytes
    struct { ts_expert_id id; } varying[] = {
        { TS_EXPERT_LRQ       },
        { TS_EXPERT_CHAMPQ    },
        { TS_EXPERT_FLRQ      },
        { TS_EXPERT_DARTQUANT },
    };
    for (const auto & v : varying) {
        if (quant(v.id, &r_other) != 0) {
            printf("FAIL quant: %s quantize returned error\n", ts_expert_name(v.id));
            return 1;
        }
        if (!results_differ(r_awq, r_other)) {
            printf("FAIL quant: %s output identical to AWQ (routing not operative)\n",
                   ts_expert_name(v.id));
            return 1;
        }
    }

    printf("PASS quant: LRQ/CHAMP-Q/FLRQ/DartQuant each differ from AWQ bytes\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_routing_distinct();
    failures += test_profiles_distinct();
    failures += test_quant_differs();

    if (failures == 0) {
        printf("\nAll operative-routing tests passed.\n");
    } else {
        printf("\n%d test(s) FAILED.\n", failures);
    }
    return failures;
}
