#include "tessera-regime.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int test_family_inference() {
    struct { const char * name; const char * expected; } cases[] = {
        { "blk.5.attn_q.weight",      "attn_q"    },
        { "blk.0.ffn_down.weight",    "ffn_down"   },
        { "blk.12.attn_k.weight",     "attn_k"     },
        { "blk.3.attn_v.weight",      "attn_v"     },
        { "blk.7.attn_output.weight", "attn_out"   },
        { "blk.1.ffn_gate.weight",    "ffn_gate"   },
        { "blk.2.ffn_up.weight",      "ffn_up"     },
        { "token_embd.weight",        "unknown"    },
    };
    for (const auto & c : cases) {
        std::string got = ts_regime_infer_family(c.name);
        if (got != c.expected) {
            printf("FAIL family: \"%s\" -> \"%s\", expected \"%s\"\n",
                   c.name, got.c_str(), c.expected);
            return 1;
        }
    }
    printf("PASS family inference: %d cases\n", (int)(sizeof(cases) / sizeof(cases[0])));
    return 0;
}

static int test_route_high_kurtosis_down() {
    ts_regime_descriptor desc = {};
    desc.tensor_name = "blk.0.ffn_down.weight";
    desc.family      = "ffn_down";
    desc.kurtosis    = 15.0f;
    desc.eff_rank    = 0.5f;

    ts_regime_routing r = ts_regime_classify(&desc);
    if (r.expert != TS_EXPERT_DARTQUANT) {
        printf("FAIL route: kurtosis=15 + ffn_down -> expert %d, expected DARTQUANT\n", r.expert);
        return 1;
    }
    printf("PASS route: high kurtosis + ffn_down -> DARTQUANT (%s)\n", r.reason.c_str());
    return 0;
}

static int test_route_low_eff_rank() {
    ts_regime_descriptor desc = {};
    desc.tensor_name = "blk.0.attn_q.weight";
    desc.family      = "attn_q";
    desc.kurtosis    = 2.0f;
    desc.eff_rank    = 0.2f;

    ts_regime_routing r = ts_regime_classify(&desc);
    if (r.expert != TS_EXPERT_FLRQ && r.expert != TS_EXPERT_LRQ) {
        printf("FAIL route: eff_rank=0.2 -> expert %d, expected FLRQ or LRQ\n", r.expert);
        return 1;
    }
    printf("PASS route: low eff_rank -> %s (%s)\n",
           r.expert == TS_EXPERT_FLRQ ? "FLRQ" : "LRQ", r.reason.c_str());
    return 0;
}

static int test_route_normal() {
    ts_regime_descriptor desc = {};
    desc.tensor_name = "blk.0.attn_k.weight";
    desc.family      = "attn_k";
    desc.kurtosis    = 2.5f;
    desc.eff_rank    = 0.8f;

    ts_regime_routing r = ts_regime_classify(&desc);
    if (r.expert != TS_EXPERT_AWQ) {
        printf("FAIL route: normal stats -> expert %d, expected AWQ\n", r.expert);
        return 1;
    }
    printf("PASS route: normal stats -> AWQ (%s)\n", r.reason.c_str());
    return 0;
}

static int test_route_all_summary() {
    // 5 synthetic tensors with known routing outcomes
    ts_regime_descriptor descs[5] = {};

    // 0: kurtosis=15, ffn_down -> DARTQUANT
    descs[0].tensor_name = "blk.0.ffn_down.weight";
    descs[0].family      = "ffn_down";
    descs[0].kurtosis    = 15.0f;
    descs[0].eff_rank    = 0.5f;

    // 1: kurtosis=7, ffn_gate -> CHAMPQ
    descs[1].tensor_name = "blk.0.ffn_gate.weight";
    descs[1].family      = "ffn_gate";
    descs[1].kurtosis    = 7.0f;
    descs[1].eff_rank    = 0.5f;

    // 2: eff_rank=0.2, low kurtosis -> LRQ
    descs[2].tensor_name = "blk.0.attn_q.weight";
    descs[2].family      = "attn_q";
    descs[2].kurtosis    = 2.0f;
    descs[2].eff_rank    = 0.2f;

    // 3: well-conditioned attn_k -> AWQ
    descs[3].tensor_name = "blk.0.attn_k.weight";
    descs[3].family      = "attn_k";
    descs[3].kurtosis    = 2.0f;
    descs[3].eff_rank    = 0.8f;

    // 4: default regime -> AWQ
    descs[4].tensor_name = "blk.0.ffn_up.weight";
    descs[4].family      = "ffn_up";
    descs[4].kurtosis    = 3.5f;
    descs[4].eff_rank    = 0.5f;

    std::vector<ts_regime_routing> routings = ts_regime_route_all(descs, 5);
    if ((int64_t)routings.size() != 5) {
        printf("FAIL route_all: got %zu routings, expected 5\n", routings.size());
        return 1;
    }

    ts_regime_summary s = ts_regime_summarize(&routings, descs, 5);

    if (s.count_per_expert[TS_EXPERT_DARTQUANT] != 1) {
        printf("FAIL summary: DARTQUANT count %lld, expected 1\n",
               (long long)s.count_per_expert[TS_EXPERT_DARTQUANT]);
        return 1;
    }
    if (s.count_per_expert[TS_EXPERT_CHAMPQ] != 1) {
        printf("FAIL summary: CHAMPQ count %lld, expected 1\n",
               (long long)s.count_per_expert[TS_EXPERT_CHAMPQ]);
        return 1;
    }
    if (s.count_per_expert[TS_EXPERT_LRQ] != 1) {
        printf("FAIL summary: LRQ count %lld, expected 1\n",
               (long long)s.count_per_expert[TS_EXPERT_LRQ]);
        return 1;
    }
    if (s.count_per_expert[TS_EXPERT_AWQ] != 2) {
        printf("FAIL summary: AWQ count %lld, expected 2\n",
               (long long)s.count_per_expert[TS_EXPERT_AWQ]);
        return 1;
    }

    int64_t total = 0;
    for (int i = 0; i < TS_EXPERT_COUNT; i++) {
        total += s.count_per_expert[i];
    }
    if (total != 5) {
        printf("FAIL summary: total %lld, expected 5\n", (long long)total);
        return 1;
    }

    printf("PASS route_all + summary: AWQ=%lld DARTQUANT=%lld CHAMPQ=%lld LRQ=%lld "
           "mean_kurt=%.2f mean_er=%.2f\n",
           (long long)s.count_per_expert[TS_EXPERT_AWQ],
           (long long)s.count_per_expert[TS_EXPERT_DARTQUANT],
           (long long)s.count_per_expert[TS_EXPERT_CHAMPQ],
           (long long)s.count_per_expert[TS_EXPERT_LRQ],
           s.mean_kurtosis, s.mean_eff_rank);
    return 0;
}

static int test_compute_descriptor() {
    // 4x8 weight matrix, uniform
    const int64_t out_dim = 4, in_dim = 8;
    std::vector<float> W(out_dim * in_dim, 0.5f);

    // imatrix: 8 channel magnitudes with one outlier
    float imatrix[8] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 10.0f };

    ts_regime_descriptor desc = ts_regime_compute_descriptor(
        "blk.0.ffn_down.weight", W.data(), out_dim, in_dim, imatrix, 8);

    if (desc.family != "ffn_down") {
        printf("FAIL compute: family \"%s\", expected \"ffn_down\"\n", desc.family.c_str());
        return 1;
    }
    if (desc.out_dim != out_dim || desc.in_dim != in_dim) {
        printf("FAIL compute: dims %lldx%lld, expected %lldx%lld\n",
               (long long)desc.out_dim, (long long)desc.in_dim,
               (long long)out_dim, (long long)in_dim);
        return 1;
    }
    // outlier should push kurtosis above gaussian baseline
    if (desc.kurtosis < 0.0f) {
        printf("FAIL compute: kurtosis %.4f unexpectedly negative\n", desc.kurtosis);
        return 1;
    }
    if (desc.p99 < 5.0f) {
        printf("FAIL compute: p99 %.4f, expected >= 5.0 with outlier at 10\n", desc.p99);
        return 1;
    }
    printf("PASS compute_descriptor: kurt=%.2f eff_rank=%.3f p99=%.2f mean_mag=%.2f\n",
           desc.kurtosis, desc.eff_rank, desc.p99, desc.mean_magnitude);
    return 0;
}

static int test_modality_inference() {
    // M0b: explicit v./a. role prefixes (real mmproj convention from
    // clip.cpp:1831) take precedence over legacy fragment matching. mm.*
    // projector tensors still fall through to the fragment check so
    // hand-written fixtures like "mm.vision_embed.weight" keep working.
    struct { const char * name; int want; } cases[] = {
        // role-prefix pass: vision tower
        { "v.patch_embd.weight",      1 },
        { "v.blk.0.attn_q.weight",    1 },
        { "v.blk.0.ffn_down.weight",  1 },
        // role-prefix pass: audio tower
        { "a.encoder.layers.0.weight", 2 },
        { "a.position_embeddings",    2 },
        // role-prefix pass: text-side projector (mm.* falls through to
        // fragment check; these names have no vision/audio fragment)
        { "mm.0.weight",              0 },
        { "mm.1.bias",                0 },
        // legacy fragment fallback: still classifies old-style names
        { "vision_tower.layer.weight", 1 },
        { "image_encoder.weight",      1 },
        { "audio_encoder.weight",      2 },
        { "speech_proj.weight",        2 },
        // regression: no prefix, no fragment match -> text
        { "attn_q.weight",             0 },
        // guards
        { "",                          0 },
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));
    for (int i = 0; i < n_cases; i++) {
        int got = ts_regime_infer_modality(cases[i].name);
        if (got != cases[i].want) {
            printf("FAIL modality: \"%s\" -> %d, expected %d\n",
                   cases[i].name, got, cases[i].want);
            return 1;
        }
    }
    if (ts_regime_infer_modality(nullptr) != 0) {
        printf("FAIL modality: nullptr -> %d, expected 0\n",
               ts_regime_infer_modality(nullptr));
        return 1;
    }
    n_cases += 1; // nullptr guard
    printf("PASS modality inference: %d cases\n", n_cases);
    return 0;
}

int main() {
    int failures = 0;
    failures += test_family_inference();
    failures += test_route_high_kurtosis_down();
    failures += test_route_low_eff_rank();
    failures += test_route_normal();
    failures += test_route_all_summary();
    failures += test_compute_descriptor();
    failures += test_modality_inference();

    if (failures == 0) {
        printf("\nAll tests passed.\n");
    } else {
        printf("\n%d test(s) FAILED.\n", failures);
    }
    return failures;
}
