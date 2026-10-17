// smoke test for the VibeVoice LM dual-KV audio_token loop, on top of llama_decode
// two contexts share the model: the positive branch has the prompt, the negative (CFG) branch starts at audio_bos
// every frame: last hidden of both -> CFG diffusion sample -> streaming decode -> streaming speech feedback embd fed to both branches
// the frames are forced, the LM is not sampled (see test-vibevoice-sampler for the sampled loop)
// usage: test-vibevoice-lm-loop <model.gguf> [n_frames]

#include "vibevoice-gen.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool all_finite(const std::vector<float> & v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        d = std::max(d, (double) std::fabs(a[i] - b[i]));
    }
    return d;
}

static bool check(const char * name, const std::vector<float> & v, size_t n_expected, bool require_finite = true) {
    bool ok = true;
    if (v.size() != n_expected) {
        fprintf(stderr, "%s: expected %zu values, got %zu\n", name, n_expected, v.size());
        ok = false;
    }

    double sum2 = 0.0;
    for (float x : v) {
        if (!std::isfinite(x)) {
            fprintf(stderr, "%s: non-finite value%s\n", name, require_finite ? "" : " (not checked)");
            ok = !require_finite && ok;
            break;
        }
        sum2 += (double) x * x;
    }

    printf("%s: n = %zu, rms = %.6f, first:", name, v.size(), v.empty() ? 0.0 : sqrt(sum2 / v.size()));
    for (size_t i = 0; i < 4 && i < v.size(); ++i) {
        printf(" %.6f", v[i]);
    }
    printf("\n");

    return ok;
}

static bool run(llama_model * lmodel, int64_t n_frames) {
    vv_generator gen;
    if (!gen.init(lmodel, 256)) {
        return false;
    }

    const auto *  model  = gen.model;
    const int64_t n_embd = gen.n_embd;
    const int64_t n_lat  = model->hparams.vibevoice.diff_latent_size;

    const llama_token audio_bos = llama_model_vibevoice::audio_bos_token_id;

    bool ok = true;

    // the positive branch has the prompt, the negative branch only audio_bos
    ok &= gen.prefill({ 1, 2, 3, 4, 5, audio_bos });
    printf("prefill: pos cache %" PRId64 ", neg cache %" PRId64 "\n", gen.pos.n_cached(), gen.neg.n_cached());
    if (!ok || gen.pos.n_cached() != 6 || gen.neg.n_cached() != 1) {
        fprintf(stderr, "unexpected cache sizes after prefill\n");
        ok = false;
    }

    // restart the negative branch once, as on a new audio_bos from the positive branch
    const int64_t i_reset = n_frames / 2;

    for (int64_t i = 0; i < n_frames && ok; ++i) {
        // CFG checks on the first frame: scale 1 is the cond-only sample, another scale moves away from it
        if (i == 0) {
            const std::vector<float> h_pos = gen.pos.hidden(n_embd);
            const std::vector<float> h_neg = gen.neg.hidden(n_embd);
            ok &= all_finite(h_pos) && all_finite(h_neg);

            const uint32_t seed = gen.seed + gen.n_frames;
            const std::vector<float> lat      = model->sample_speech_latents(gen.backend, h_pos.data(), h_neg.data(), 1, gen.cfg_scale, seed);
            const std::vector<float> lat_cond = model->sample_speech_latents(gen.backend, h_pos.data(), nullptr,      1, 1.0f, seed);
            const std::vector<float> lat_cfg1 = model->sample_speech_latents(gen.backend, h_pos.data(), h_neg.data(), 1, 1.0f, seed);
            const double d_hidden = max_abs_diff(h_pos, h_neg);
            const double d1 = max_abs_diff(lat_cfg1, lat_cond);
            const double d0 = max_abs_diff(lat_cond, std::vector<float>(lat_cond.size(), 0.0f));
            const double ds = max_abs_diff(lat, lat_cfg1);
            printf("pos vs neg hidden max diff = %g\n", d_hidden);
            printf("cfg=1 vs cond-only max diff = %g, cfg=%.1f vs cfg=1 max diff = %g\n", d1, gen.cfg_scale, ds);
            // stacked vs single batch float drift grows with the latent magnitude
            if (d_hidden < 1e-6 || d1 > 1e-6 * std::max(d0, 1000.0) || ds < 1e-6) {
                fprintf(stderr, "unexpected CFG behavior\n");
                ok = false;
            }
        }

        // diffusion frame, streaming decode and feedback, the same embedding goes into both branches
        const size_t n_wav = gen.wav.size();
        ok &= gen.audio_frame();
        ok &= check("frame latents", gen.lat, n_lat);
        ok &= check("feedback", gen.emb, n_embd);
        if (gen.wav.size() != n_wav + gen.hop) {
            fprintf(stderr, "frame %" PRId64 ": %zu new samples, expected %" PRId64 "\n", i, gen.wav.size() - n_wav, gen.hop);
            ok = false;
        }
        printf("frame %" PRId64 ": pos cache %" PRId64 ", neg cache %" PRId64 "\n", i, gen.pos.n_cached(), gen.neg.n_cached());

        if (i == i_reset) {
            // upstream: a new audio_bos goes into the positive branch, the negative branch restarts from it alone
            ok &= gen.audio_bos();
            printf("audio_bos reset: pos cache %" PRId64 ", neg cache %" PRId64 "\n", gen.pos.n_cached(), gen.neg.n_cached());
            if (gen.neg.n_cached() != 1 || gen.neg.pos != 1) {
                fprintf(stderr, "neg cache should hold only audio_bos after the reset\n");
                ok = false;
            }
        }
    }

    // random-weight fixtures overflow the F16 conversions of the decoder
    if (gen.n_nonfinite > 0) {
        printf("decoded wav had %" PRId64 " non-finite samples on this fixture, replaced by 0 (also in the feedback)\n", gen.n_nonfinite);
    }

    const int64_t pos_expected = 6 + n_frames + 1;
    const int64_t neg_expected = n_frames - i_reset;
    printf("final: pos cache %" PRId64 " (expected %" PRId64 "), neg cache %" PRId64 " (expected %" PRId64 ")\n",
            gen.pos.n_cached(), pos_expected, gen.neg.n_cached(), neg_expected);
    if (ok && (int64_t) gen.wav.size() != n_frames * gen.hop) {
        fprintf(stderr, "streamed %zu samples, expected %" PRId64 "\n", gen.wav.size(), n_frames * gen.hop);
        ok = false;
    }
    if (ok && (gen.pos.n_cached() != pos_expected || gen.neg.n_cached() != neg_expected)) {
        fprintf(stderr, "unexpected final cache sizes\n");
        ok = false;
    }

    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_frames]\n", argv[0]);
        return 1;
    }
    const int64_t n_frames = argc > 2 ? std::atoll(argv[2]) : 4;
    if (n_frames < 1) {
        fprintf(stderr, "n_frames must be at least 1\n");
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * lmodel = llama_model_load_from_file(argv[1], mparams);
    if (lmodel == nullptr) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    const bool ok = run(lmodel, n_frames);

    llama_model_free(lmodel);
    llama_backend_free();

    printf("%s\n", ok ? "OK" : "FAIL");

    return ok ? 0 : 1;
}
