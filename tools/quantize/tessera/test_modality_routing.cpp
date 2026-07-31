//
// test_modality_routing.cpp
//
// Verifies S8: modality is an operative regime axis. Routing considers the
// tensor modality, expert profiles carry per-modality adjustments, the
// MAP-Elites archive separates cells by modality, and modality is inferred
// from tensor names.
//

#include "tessera-regime.h"
#include "tessera-search.h"

#include <cmath>
#include <cstdio>
#include <cstring>

static bool feq(float a, float b) {
    return fabsf(a - b) < 1e-5f;
}

static ts_regime_descriptor make_desc(const char * name, const char * family,
                                      float kurtosis, float eff_rank, int modality) {
    ts_regime_descriptor d = {};
    d.tensor_name = name;
    d.family      = family;
    d.kurtosis    = kurtosis;
    d.eff_rank    = eff_rank;
    d.modality    = modality;
    return d;
}

// --- 1. routing produces different experts for different modalities ---

static int test_routing_differs_by_modality() {
    // heavy-tailed ffn_gate: text -> CHAMP-Q, audio -> FLRQ (audio is heavy-tailed)
    ts_regime_descriptor d_gate_text = make_desc("blk.0.ffn_gate.weight", "ffn_gate", 7.0f, 0.5f, 0);
    ts_regime_descriptor d_gate_audio = make_desc("blk.0.ffn_gate.weight", "ffn_gate", 7.0f, 0.5f, 2);
    ts_regime_routing t_gate = ts_regime_classify(&d_gate_text);
    ts_regime_routing a_gate = ts_regime_classify(&d_gate_audio);
    if (t_gate.expert != TS_EXPERT_CHAMPQ) {
        printf("FAIL routing: text ffn_gate kurt=7 -> %s, expected CHAMP-Q\n",
               ts_expert_name(t_gate.expert));
        return 1;
    }
    if (a_gate.expert != TS_EXPERT_FLRQ) {
        printf("FAIL routing: audio ffn_gate kurt=7 -> %s, expected FLRQ\n",
               ts_expert_name(a_gate.expert));
        return 1;
    }
    if (t_gate.expert == a_gate.expert) {
        printf("FAIL routing: text and audio routed to the same expert\n");
        return 1;
    }
    if (a_gate.reason.find("audio") == std::string::npos) {
        printf("FAIL routing: audio reason lacks modality tag: '%s'\n", a_gate.reason.c_str());
        return 1;
    }

    // low-rank attn_q: text -> FLRQ (er < 0.15), image -> LRQ (spatially low-rank)
    ts_regime_descriptor d_q_text  = make_desc("blk.0.attn_q.weight", "attn_q", 2.0f, 0.1f, 0);
    ts_regime_descriptor d_q_image = make_desc("blk.0.attn_q.weight", "attn_q", 2.0f, 0.1f, 1);
    ts_regime_routing t_q = ts_regime_classify(&d_q_text);
    ts_regime_routing i_q = ts_regime_classify(&d_q_image);
    if (t_q.expert != TS_EXPERT_FLRQ) {
        printf("FAIL routing: text attn_q er=0.1 -> %s, expected FLRQ\n",
               ts_expert_name(t_q.expert));
        return 1;
    }
    if (i_q.expert != TS_EXPERT_LRQ) {
        printf("FAIL routing: image attn_q er=0.1 -> %s, expected LRQ\n",
               ts_expert_name(i_q.expert));
        return 1;
    }
    if (i_q.reason.find("image") == std::string::npos) {
        printf("FAIL routing: image reason lacks modality tag: '%s'\n", i_q.reason.c_str());
        return 1;
    }

    printf("PASS routing differs by modality: text/audio ffn_gate=%s/%s, text/image attn_q=%s/%s\n",
           ts_expert_name(t_gate.expert), ts_expert_name(a_gate.expert),
           ts_expert_name(t_q.expert), ts_expert_name(i_q.expert));
    return 0;
}

// --- 2. text routing is unchanged (backward compat) ---

static int test_text_routing_unchanged() {
    struct { const char * family; float kurt; float er; ts_expert_id want; } cases[] = {
        { "ffn_down", 15.0f, 0.50f, TS_EXPERT_DARTQUANT },
        { "ffn_gate",  7.0f, 0.50f, TS_EXPERT_CHAMPQ    },
        { "attn_q",    2.0f, 0.25f, TS_EXPERT_LRQ       },
        { "attn_k",    2.5f, 0.80f, TS_EXPERT_AWQ       },
    };
    for (const auto & c : cases) {
        ts_regime_descriptor d = make_desc("blk.0.x.weight", c.family, c.kurt, c.er, 0);
        ts_regime_routing r = ts_regime_classify(&d);
        if (r.expert != c.want) {
            printf("FAIL text routing: %s kurt=%.1f er=%.2f -> %s, expected %s\n",
                   c.family, c.kurt, c.er, ts_expert_name(r.expert), ts_expert_name(c.want));
            return 1;
        }
    }
    printf("PASS text routing unchanged across %d regimes\n", (int)(sizeof(cases) / sizeof(cases[0])));
    return 0;
}

// --- 3. expert profiles differ by modality ---

static int test_profiles_differ_by_modality() {
    // AWQ baseline: text is identity, audio tightens clip, image widens outliers
    ts_expert_profile text  = ts_expert_default_profile(TS_EXPERT_AWQ, 0);
    ts_expert_profile audio = ts_expert_default_profile(TS_EXPERT_AWQ, 2);
    ts_expert_profile image = ts_expert_default_profile(TS_EXPERT_AWQ, 1);

    if (!feq(text.clip_scale, 1.0f) || text.max_outliers != 0) {
        printf("FAIL profile: text AWQ is not the identity baseline\n");
        return 1;
    }
    if (!(audio.clip_scale < text.clip_scale)) {
        printf("FAIL profile: audio clip_scale %.3f not tighter than text %.3f\n",
               audio.clip_scale, text.clip_scale);
        return 1;
    }
    if (!(image.max_outliers > text.max_outliers)) {
        printf("FAIL profile: image max_outliers %d not wider than text %d\n",
               image.max_outliers, text.max_outliers);
        return 1;
    }

    // the adjustment layers on top of any expert (DartQuant base outliers = 12)
    ts_expert_profile dart_text  = ts_expert_default_profile(TS_EXPERT_DARTQUANT, 0);
    ts_expert_profile dart_image = ts_expert_default_profile(TS_EXPERT_DARTQUANT, 1);
    if (dart_text.max_outliers != 12 || dart_image.max_outliers != 16) {
        printf("FAIL profile: DartQuant outliers text=%d image=%d, expected 12/16\n",
               dart_text.max_outliers, dart_image.max_outliers);
        return 1;
    }

    // default (single-arg) call stays text
    ts_expert_profile def = ts_expert_default_profile(TS_EXPERT_AWQ);
    if (!feq(def.clip_scale, text.clip_scale) || def.max_outliers != text.max_outliers) {
        printf("FAIL profile: default-arg profile differs from explicit text profile\n");
        return 1;
    }

    printf("PASS profiles differ by modality: audio clip=%.2f, image outliers=%d\n",
           audio.clip_scale, image.max_outliers);
    return 0;
}

// --- 4. archive cells are separated by modality ---

static int test_archive_separated_by_modality() {
    ts_map_elites_archive a;
    ts_archive_init(&a, 3, 3, 4, 3);

    // identical regime descriptors, three modalities -> three independent cells
    ts_regime_descriptor text  = make_desc("t.attn_q.weight", "attn_q", 10.0f, 0.5f, 0);
    ts_regime_descriptor image = make_desc("i.attn_q.weight", "attn_q", 10.0f, 0.5f, 1);
    ts_regime_descriptor audio = make_desc("a.attn_q.weight", "attn_q", 10.0f, 0.5f, 2);

    ts_archive_insert(&a, &text,  0.20f, 0.40f, 0.60f, "t.attn_q.weight");
    ts_archive_insert(&a, &image, 0.50f, 0.55f, 0.45f, "i.attn_q.weight");
    ts_archive_insert(&a, &audio, 0.80f, 0.90f, 0.10f, "a.attn_q.weight");

    const ts_archive_cell * ct = ts_archive_query(&a, &text);
    const ts_archive_cell * ci = ts_archive_query(&a, &image);
    const ts_archive_cell * ca = ts_archive_query(&a, &audio);
    if (!ct || !ci || !ca) {
        printf("FAIL archive: expected all three modality cells occupied\n");
        return 1;
    }
    if (!feq(ct->best_fitness, 0.20f) || !feq(ci->best_fitness, 0.50f) ||
        !feq(ca->best_fitness, 0.80f)) {
        printf("FAIL archive: modality cells not independent (t=%.3f i=%.3f a=%.3f)\n",
               ct->best_fitness, ci->best_fitness, ca->best_fitness);
        return 1;
    }
    if (ct->modality_bucket == ci->modality_bucket ||
        ct->modality_bucket == ca->modality_bucket ||
        ci->modality_bucket == ca->modality_bucket) {
        printf("FAIL archive: modality buckets collide (%d %d %d)\n",
               ct->modality_bucket, ci->modality_bucket, ca->modality_bucket);
        return 1;
    }

    ts_archive_summary s = ts_archive_summarize(&a);
    if (s.occupied_cells != 3) {
        printf("FAIL archive: occupied=%d, expected 3\n", s.occupied_cells);
        return 1;
    }

    printf("PASS archive separated by modality: 3 cells, buckets %d/%d/%d\n",
           ct->modality_bucket, ci->modality_bucket, ca->modality_bucket);
    return 0;
}

// --- 5. modality inferred from tensor names ---

static int test_modality_inference() {
    struct { const char * name; int want; } cases[] = {
        { "blk.0.attn_q.weight",            0 },
        { "blk.0.ffn_down.weight",          0 },
        { "mm.vision_embed.weight",         1 },
        { "blk.0.image_proj.weight",        1 },
        { "vision_tower.patch_embed.weight", 1 },
        { "mm.audio_embed.weight",          2 },
        { "audio_encoder.proj.weight",      2 },
    };
    for (const auto & c : cases) {
        int got = ts_regime_infer_modality(c.name);
        if (got != c.want) {
            printf("FAIL infer: \"%s\" -> %d, expected %d\n", c.name, got, c.want);
            return 1;
        }
    }

    // compute_descriptor stamps the inferred modality onto the descriptor
    const int64_t out_dim = 4, in_dim = 8;
    float W[32];
    for (int i = 0; i < 32; i++) W[i] = 0.5f;
    ts_regime_descriptor d = ts_regime_compute_descriptor(
        "mm.vision_embed.weight", W, out_dim, in_dim, nullptr, 0);
    if (d.modality != 1) {
        printf("FAIL infer: compute_descriptor modality=%d, expected 1\n", d.modality);
        return 1;
    }

    printf("PASS modality inference from tensor names\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_routing_differs_by_modality();
    failures += test_text_routing_unchanged();
    failures += test_profiles_differ_by_modality();
    failures += test_archive_separated_by_modality();
    failures += test_modality_inference();

    if (failures == 0) {
        printf("\nAll modality-routing tests passed.\n");
    } else {
        printf("\n%d test(s) FAILED.\n", failures);
    }
    return failures;
}
