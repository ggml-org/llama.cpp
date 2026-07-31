//
// test_w4a4.cpp
//
// Tests for tessera-w4a4.cpp: LLM.int8 outlier detection, per-token dynamic
// scales, decompose/recompose round-trip, the outlier-fraction cap, and the
// activation-aware weight path (W4A4 vs plain Tile640).
//

#include "tessera-w4a4.h"
#include "tessera-quant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            g_failures++;                                    \
        }                                                    \
    } while (0)

int main(void) {
    const int64_t n_tokens = 8;
    const int64_t in_dim   = 128;
    const int64_t n        = n_tokens * in_dim;

    // Synthetic activations: small background in ~[-1, 1] plus two known
    // outlier channels (5 and 42) that exceed the LLM.int8 threshold of 6.0.
    std::vector<float> X((size_t)n);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t c = 0; c < in_dim; c++) {
            X[(size_t)(t * in_dim + c)] = 0.5f * std::sin((float)(t * 7 + c));
        }
    }
    for (int64_t t = 0; t < n_tokens; t++) {
        X[(size_t)(t * in_dim + 5)]  =  9.0f + 0.1f * (float)t;  // outlier, > 6
        X[(size_t)(t * in_dim + 42)] = -8.0f - 0.1f * (float)t;  // outlier, < -6
    }

    ts_w4a4_config cfg = ts_w4a4_default_config();
    cfg.enable         = true;
    cfg.outlier_thresh = 6.0f;
    cfg.outlier_frac   = 0.05f;   // cap = floor(0.05 * 128) = 6 channels

    // --- 1. outlier detection finds exactly the injected channels ---
    ts_w4a4_outliers out;
    ts_w4a4_detect_outliers(X.data(), n_tokens, in_dim, &cfg, &out);
    CHECK(out.channels.size() == 2, "detects exactly 2 outlier channels");
    CHECK(out.channels.size() == 2 && out.channels[0] == 5 && out.channels[1] == 42,
          "outlier channels are {5, 42}, sorted ascending");
    CHECK(out.mask.size() == (size_t)in_dim, "mask sized to in_dim");
    CHECK(out.mask[5] == 1 && out.mask[42] == 1, "mask flags the outlier channels");
    CHECK(out.mask[0] == 0 && out.mask[100] == 0, "mask leaves non-outliers clear");
    CHECK(std::fabs(out.frac - 2.0f / (float)in_dim) < 1e-6f, "outlier fraction correct");
    std::printf("outliers: %zu channels, frac=%g\n", out.channels.size(), out.frac);

    // --- 2. outlier-fraction cap keeps the highest-magnitude channels ---
    {
        std::vector<float> Xc((size_t)n, 0.0f);
        // 10 outlier channels with magnitudes 7, 8, ..., 16
        for (int64_t k = 0; k < 10; k++) {
            int64_t c = 3 + k;
            for (int64_t t = 0; t < n_tokens; t++) {
                Xc[(size_t)(t * in_dim + c)] = 7.0f + (float)k;
            }
        }
        ts_w4a4_config cap_cfg = cfg;
        cap_cfg.outlier_frac = 0.03f;   // cap = floor(0.03 * 128) = 3
        ts_w4a4_outliers capped;
        ts_w4a4_detect_outliers(Xc.data(), n_tokens, in_dim, &cap_cfg, &capped);
        CHECK(capped.channels.size() == 3, "cap trims to 3 channels");
        // top-3 magnitudes are channels with k = 9, 8, 7 -> c = 12, 11, 10
        CHECK(capped.channels.size() == 3 &&
              capped.channels[0] == 10 && capped.channels[1] == 11 && capped.channels[2] == 12,
              "cap keeps the highest-magnitude channels");
        std::printf("cap: kept %zu channels (of 10 above threshold)\n", capped.channels.size());
    }

    // --- 3. per-token dynamic scales = max|row| / qmax (qmax = 7 for 4 bits) ---
    {
        ts_w4a4_act_scales scales;
        ts_w4a4_compute_act_scales(X.data(), n_tokens, in_dim, &cfg, &scales);
        CHECK(scales.mode == TS_W4A4_PER_TOKEN, "scale mode is per-token");
        CHECK(scales.qmax == 7, "qmax == 7 for 4-bit activations");
        CHECK(scales.per_token.size() == (size_t)n_tokens, "one scale per token");
        bool ok = true;
        for (int64_t t = 0; t < n_tokens; t++) {
            float maxabs = 0.0f;
            for (int64_t c = 0; c < in_dim; c++) {
                maxabs = std::max(maxabs, std::fabs(X[(size_t)(t * in_dim + c)]));
            }
            if (std::fabs(scales.per_token[(size_t)t] - maxabs / 7.0f) > 1e-6f) {
                ok = false;
            }
        }
        CHECK(ok, "per-token scale == max|row| / 7");
        // token 0 max is 9.0 (channel 5) -> scale 9/7
        CHECK(std::fabs(scales.per_token[0] - 9.0f / 7.0f) < 1e-6f,
              "token-0 scale reflects the row max");
        std::printf("per-token scales: qmax=%d scale[0]=%g\n", scales.qmax, scales.per_token[0]);

        // W4A8 forward-compat: qmax == 127 for 8 bits
        ts_w4a4_config cfg8 = cfg;
        cfg8.activation_bits = 8;
        ts_w4a4_act_scales s8;
        ts_w4a4_compute_act_scales(X.data(), n_tokens, in_dim, &cfg8, &s8);
        CHECK(s8.qmax == 127, "qmax == 127 for 8-bit activations");
    }

    // --- 4. per-tensor static scale is a single scalar ---
    {
        ts_w4a4_config pt = cfg;
        pt.scale_mode = TS_W4A4_PER_TENSOR;
        ts_w4a4_act_scales scales;
        ts_w4a4_compute_act_scales(X.data(), n_tokens, in_dim, &pt, &scales);
        CHECK(scales.mode == TS_W4A4_PER_TENSOR, "scale mode is per-tensor");
        CHECK(scales.per_token.empty(), "per-tensor mode has no per-token array");
        CHECK(scales.per_tensor > 0.0f && std::isfinite(scales.per_tensor),
              "per-tensor scale is a positive finite scalar");
        std::printf("per-tensor scale: %g\n", scales.per_tensor);
    }

    // --- 5. decompose / recompose round-trip preserves data ---
    {
        ts_w4a4_decomp decomp;
        ts_w4a4_decompose(X.data(), n_tokens, in_dim, &cfg, &out, &decomp);
        CHECK(decomp.outlier_channels.size() == 2, "decomp carries 2 outlier channels");
        CHECK(decomp.quant.size() == (size_t)n, "decomp.quant sized n_tokens x in_dim");
        CHECK(decomp.outlier_vals.size() == (size_t)(n_tokens * 2),
              "decomp.outlier_vals sized n_tokens x n_outliers");

        std::vector<float> recon((size_t)n, 0.0f);
        ts_w4a4_recompose(&decomp, n_tokens, in_dim, recon.data());

        // outlier channels round-trip to f16 precision; non-outlier channels
        // to <= half an INT4 step of the (outlier-aware) per-token scale.
        double sum_err2 = 0.0, sum_x2 = 0.0;
        float max_outlier_err = 0.0f;
        float max_int4_err    = 0.0f;
        for (int64_t t = 0; t < n_tokens; t++) {
            float scale = decomp.scales.per_token[(size_t)t];
            for (int64_t c = 0; c < in_dim; c++) {
                float x = X[(size_t)(t * in_dim + c)];
                float r = recon[(size_t)(t * in_dim + c)];
                float e = std::fabs(x - r);
                sum_err2 += (double)e * e;
                sum_x2   += (double)x * x;
                if (c == 5 || c == 42) {
                    max_outlier_err = std::max(max_outlier_err, e);
                } else {
                    max_int4_err = std::max(max_int4_err, e);
                    CHECK(e <= 0.5f * scale + 1e-4f, "INT4 error within half a step");
                }
            }
        }
        float rel = (float)std::sqrt(sum_err2 / sum_x2);
        CHECK(max_outlier_err < 0.05f, "outlier channels near f16-exact");
        CHECK(rel < 0.05f, "relative Frobenius round-trip error < 5%");
        std::printf("round-trip: rel_frob=%.5f max_int4_err=%g max_outlier_err=%g\n",
                    rel, max_int4_err, max_outlier_err);
    }

    // --- 6. W4A4 weight path differs from plain Tile640 ---
    {
        const int64_t w_out = 4;
        const int64_t w_in  = 640;   // one Tile640 page
        std::vector<float> W((size_t)(w_out * w_in));
        for (int64_t i = 0; i < w_out * w_in; i++) {
            W[(size_t)i] = (float)((i % 100) - 50) / 10.0f;
        }
        // calibration activations matching the weight in_dim
        std::vector<float> Xw((size_t)(n_tokens * w_in));
        for (int64_t t = 0; t < n_tokens; t++) {
            for (int64_t c = 0; c < w_in; c++) {
                Xw[(size_t)(t * w_in + c)] = 0.2f + 0.01f * (float)(c % 40);
            }
        }

        ts_quant_params_2d qp = {};
        qp.alpha          = 0.0f;   // auto-search when activations are present
        qp.clip           = 1.0f;
        qp.max_outliers   = 8;
        qp.outlier_thresh = 2.0f;
        qp.awq_grid       = 5;

        // plain Tile640 (no activations -> no activation-aware scaling)
        ts_quant_result_2d plain;
        int rc = ts_quantize_2d(W.data(), nullptr, nullptr, nullptr, nullptr,
                                w_out, w_in, 0, &qp, &plain);
        CHECK(rc == 0, "plain quantize_2d returns 0");
        CHECK(plain.act_scale.empty(), "plain path has no act_scale");

        // W4A4 activation-aware weight path
        ts_quant_result_2d      wbase;
        ts_w4a4_weight_result   wres;
        rc = ts_w4a4_quantize_weights(W.data(), Xw.data(), w_out, w_in, n_tokens,
                                      &qp, &cfg, &wbase, &wres);
        CHECK(rc == 0, "w4a4 quantize_weights returns 0");
        CHECK(!wbase.act_scale.empty(), "w4a4 path engages activation-aware scaling");
        CHECK(wbase.act_scale.size() == (size_t)w_in, "w4a4 act_scale is per-input-channel");
        CHECK(std::isfinite(wres.effective_bits) && wres.effective_bits > 0.0f &&
              wres.effective_bits <= 8.0f, "effective bits finite and sane");
        CHECK(wres.base == &wbase, "w4a4 result references the base quantization");

        // the activation-aware scaling changes the reconstruction
        bool differs = false;
        for (size_t i = 0; i < plain.recon.size() && i < wbase.recon.size(); i++) {
            if (std::fabs(plain.recon[i] - wbase.recon[i]) > 1e-6f) {
                differs = true;
                break;
            }
        }
        CHECK(differs, "w4a4 reconstruction differs from plain Tile640");
        std::printf("w4a4 weights: eff_bits=%.4f alpha=%g (plain alpha=%g)\n",
                    wres.effective_bits, wbase.best_alpha, plain.best_alpha);
    }

    // --- 7. sidecar JSON carries the w4a4 object ---
    {
        ts_w4a4_sidecar sc;
        sc.enabled           = true;
        sc.activation_bits   = 4;
        sc.scale_mode        = TS_W4A4_PER_TOKEN;
        sc.outlier_frac      = out.frac;
        sc.act_scale_static  = 0.0f;
        sc.outlier_channels  = out.channels;
        std::string json = ts_w4a4_sidecar_json(&sc);
        CHECK(json.find("\"w4a4\"") != std::string::npos, "sidecar has w4a4 object");
        CHECK(json.find("\"scale_mode\": \"per_token\"") != std::string::npos,
              "sidecar records scale_mode");
        CHECK(json.find("\"act_outlier_count\": 2") != std::string::npos,
              "sidecar records outlier count");
        CHECK(json.find("5, 42") != std::string::npos, "sidecar records outlier indices");
        std::printf("sidecar: %s\n", json.c_str());
    }

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}
