#include "tessera-mm-awq.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

int main() {
    const int64_t out_dim = 4;
    const int64_t in_dim  = 8;

    float weights[32];
    for (int i = 0; i < 32; i++) {
        weights[i] = 0.05f * (i + 1) - 0.3f;
    }

    float act_text[8]  = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float act_image[8] = { 2, 1, 4, 3, 6, 5, 8, 7 };
    float act_audio[8] = { 1, 1, 2, 2, 3, 3, 4, 4 };

    const float * calib[3] = { nullptr, nullptr, nullptr };
    const float * ref[3]   = { nullptr, nullptr, nullptr };
    const int64_t nt[3]    = { 0, 0, 0 };

    // 1. all modalities present: weighted_mse = 0.5*m0 + 0.3*m1 + 0.2*m2
    {
        const float * acts[3] = { act_text, act_image, act_audio };
        ts_mm_awq_params p = ts_mm_awq_default_params();
        ts_mm_awq_result r;
        std::string err;
        int rc = ts_mm_awq_compute(weights, acts, calib, ref, nt, out_dim, in_dim, &p, &r, &err);
        assert(rc == 0);
        assert((int64_t)r.act_scale_text.size()  == in_dim);
        assert((int64_t)r.act_scale_image.size() == in_dim);
        assert((int64_t)r.act_scale_audio.size() == in_dim);

        float expected = 0.5f * r.mse_per_modality[0] +
                         0.3f * r.mse_per_modality[1] +
                         0.2f * r.mse_per_modality[2];
        printf("all-present: weighted_mse=%.6f expected=%.6f alpha=[%.2f,%.2f,%.2f]\n",
               r.weighted_mse, expected,
               r.best_alpha[0], r.best_alpha[1], r.best_alpha[2]);
        assert(fabsf(r.weighted_mse - expected) < 1e-4f * (fabsf(expected) + 1e-6f));
    }

    // 2. missing audio with error_on_missing=true -> -1
    {
        const float * acts[3] = { act_text, act_image, nullptr };
        ts_mm_awq_params p = ts_mm_awq_default_params();
        p.error_on_missing = true;
        ts_mm_awq_result r;
        std::string err;
        int rc = ts_mm_awq_compute(weights, acts, calib, ref, nt, out_dim, in_dim, &p, &r, &err);
        printf("missing+error: rc=%d err='%s'\n", rc, err.c_str());
        assert(rc == -1);
        assert(!err.empty());
    }

    // 3. missing audio with error_on_missing=false -> text fallback
    {
        const float * acts[3] = { act_text, act_image, nullptr };
        ts_mm_awq_params p = ts_mm_awq_default_params();
        p.error_on_missing = false;
        ts_mm_awq_result r;
        std::string err;
        int rc = ts_mm_awq_compute(weights, acts, calib, ref, nt, out_dim, in_dim, &p, &r, &err);
        assert(rc == 0);

        // audio inherits the text alpha and text act_scale
        assert(r.best_alpha[2] == r.best_alpha[0]);
        assert(r.act_scale_audio.size() == r.act_scale_text.size());
        assert(memcmp(r.act_scale_audio.data(), r.act_scale_text.data(),
                      (size_t)in_dim * sizeof(uint16_t)) == 0);

        // weighted over present modalities only, renormalized: (0.5*m0 + 0.3*m1) / 0.8
        float expected = (0.5f * r.mse_per_modality[0] +
                          0.3f * r.mse_per_modality[1]) / 0.8f;
        printf("missing+fallback: weighted_mse=%.6f expected=%.6f alpha2=%.2f (=text %.2f)\n",
               r.weighted_mse, expected, r.best_alpha[2], r.best_alpha[0]);
        assert(fabsf(r.weighted_mse - expected) < 1e-4f * (fabsf(expected) + 1e-6f));
    }

    printf("PASS\n");
    return 0;
}
