//
// test_mm_fitness.cpp
//
// Tests for tessera-mm-fitness.cpp. Verifies the modality-weighted
// composite, weight renormalization when a modality is missing, and
// the per-family breakdown.
//

#include "tessera-mm-fitness.h"

#include <cmath>
#include <cstdio>

static const float TOL = 1e-5f;

static bool feq(float a, float b) {
    return fabsf(a - b) <= TOL;
}

static bool test_composite_all_present() {
    // 3 layers, all modalities present, uniform alpha (nullptr)
    const float t2_text[]  = { 0.10f, 0.20f, 0.30f };  // mean 0.20
    const float t2_image[] = { 0.40f, 0.50f, 0.60f };  // mean 0.50
    const float t2_audio[] = { 0.70f, 0.80f, 0.90f };  // mean 0.80
    const float * t2[3] = { t2_text, t2_image, t2_audio };
    const bool present[3] = { true, true, true };

    ts_mm_fitness_params p = ts_mm_fitness_default_params();

    bool ok = true;
    if (!feq(p.modality_weights[0], 0.5f)) ok = false;
    if (!feq(p.modality_weights[1], 0.3f)) ok = false;
    if (!feq(p.modality_weights[2], 0.2f)) ok = false;

    ts_mm_fitness_score s = ts_mm_fitness_compute(t2, nullptr, present, 3, &p);

    // composite = 0.5*0.20 + 0.3*0.50 + 0.2*0.80 = 0.41
    float expected = 0.5f * 0.20f + 0.3f * 0.50f + 0.2f * 0.80f;
    printf("  composite=%.6f expected=%.6f per=[%.4f,%.4f,%.4f] alpha_w=%.6f\n",
           s.composite, expected, s.per_modality[0], s.per_modality[1],
           s.per_modality[2], s.alpha_weighted);

    if (!feq(s.composite, expected)) ok = false;
    if (!feq(s.composite, 0.41f)) ok = false;
    if (!feq(s.per_modality[0], 0.20f)) ok = false;
    if (!feq(s.per_modality[1], 0.50f)) ok = false;
    if (!feq(s.per_modality[2], 0.80f)) ok = false;

    // uniform alpha=1: alpha_weighted = sum of per-layer weighted t2 = 3 * composite
    if (!feq(s.alpha_weighted, 1.23f)) ok = false;

    return ok;
}

static bool test_renormalize_missing_modality() {
    // audio missing -> weights renormalize over text+image (0.5/0.8, 0.3/0.8)
    const float t2_text[]  = { 0.10f, 0.20f, 0.30f };  // mean 0.20
    const float t2_image[] = { 0.40f, 0.50f, 0.60f };  // mean 0.50
    const float * t2[3] = { t2_text, t2_image, nullptr };
    const bool present[3] = { true, true, false };

    ts_mm_fitness_params p = ts_mm_fitness_default_params();
    ts_mm_fitness_score s = ts_mm_fitness_compute(t2, nullptr, present, 3, &p);

    // 0.625*0.20 + 0.375*0.50 = 0.3125
    float expected = (0.5f / 0.8f) * 0.20f + (0.3f / 0.8f) * 0.50f;
    printf("  composite=%.6f expected=%.6f audio_loss=%.4f\n",
           s.composite, expected, s.per_modality[2]);

    bool ok = true;
    if (!feq(s.composite, expected)) ok = false;
    if (!feq(s.composite, 0.3125f)) ok = false;
    if (!feq(s.per_modality[2], 0.0f)) ok = false;  // absent modality reports 0
    return ok;
}

static bool test_family_breakdown() {
    // 4 tensors across 2 families (attn, ffn), all modalities present
    const char * names[] = { "blk0.attn_v", "blk0.ffn_up", "blk1.attn_v", "blk1.ffn_down" };
    const char * fams[]  = { "attn", "ffn", "attn", "ffn" };

    const float t2_text[]  = { 0.10f, 0.20f, 0.30f, 0.40f };
    const float t2_image[] = { 0.20f, 0.40f, 0.60f, 0.80f };
    const float t2_audio[] = { 0.30f, 0.60f, 0.90f, 1.20f };
    const float * t2[3] = { t2_text, t2_image, t2_audio };
    const bool present[3] = { true, true, true };

    ts_mm_fitness_params p = ts_mm_fitness_default_params();
    auto out = ts_mm_fitness_family_breakdown(names, fams, t2, present, 4, &p);

    if (out.size() != 2) {
        printf("  expected 2 families, got %zu\n", out.size());
        return false;
    }

    bool ok = true;
    for (const auto & fs : out) {
        printf("  family=%-5s n=%ld loss=[%.4f,%.4f,%.4f] composite=%.6f\n",
               fs.family.c_str(), (long)fs.n_tensors,
               fs.loss_per_modality[0], fs.loss_per_modality[1],
               fs.loss_per_modality[2], fs.composite);
    }

    // first-seen order: attn then ffn
    if (out[0].family != "attn") ok = false;
    if (out[1].family != "ffn")  ok = false;

    // attn = tensors {0,2}: text 0.20, image 0.40, audio 0.60
    // composite = 0.5*0.20 + 0.3*0.40 + 0.2*0.60 = 0.34
    if (out[0].n_tensors != 2) ok = false;
    if (!feq(out[0].loss_per_modality[0], 0.20f)) ok = false;
    if (!feq(out[0].loss_per_modality[1], 0.40f)) ok = false;
    if (!feq(out[0].loss_per_modality[2], 0.60f)) ok = false;
    if (!feq(out[0].composite, 0.34f)) ok = false;

    // ffn = tensors {1,3}: text 0.30, image 0.60, audio 0.90
    // composite = 0.5*0.30 + 0.3*0.60 + 0.2*0.90 = 0.51
    if (out[1].n_tensors != 2) ok = false;
    if (!feq(out[1].loss_per_modality[0], 0.30f)) ok = false;
    if (!feq(out[1].loss_per_modality[1], 0.60f)) ok = false;
    if (!feq(out[1].loss_per_modality[2], 0.90f)) ok = false;
    if (!feq(out[1].composite, 0.51f)) ok = false;

    return ok;
}

int main() {
    struct { const char * name; bool (*fn)(); } tests[] = {
        { "composite_all_present",      test_composite_all_present },
        { "renormalize_missing",        test_renormalize_missing_modality },
        { "family_breakdown",           test_family_breakdown },
    };

    bool all = true;
    for (auto & t : tests) {
        bool ok = t.fn();
        printf("[%s] %s\n", ok ? "PASS" : "FAIL", t.name);
        all = all && ok;
    }
    return all ? 0 : 1;
}
