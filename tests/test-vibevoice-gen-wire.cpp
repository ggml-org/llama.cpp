// smoke test for VibeVoice generation wiring: LM-hidden cond -> CFG diffusion sample -> speech feedback embeds
// usage: test-vibevoice-gen-wire <model.gguf> [n_frames]
// does not run the LM; cond/uncond are synthetic stand-ins for last_hidden_state rows

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "models/models.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<float> decode(const llama_model_vibevoice & model, ggml_backend_t backend, const std::vector<float> & latents, int64_t n_frames) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.vibevoice.acoustic.vae_dim, n_frames);
    ggml_set_input(inp);

    ggml_tensor * out = model.build_acoustic_dec(ctx, inp);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "failed to allocate the decoder graph\n");
        exit(1);
    }

    ggml_backend_tensor_set(inp, latents.data(), 0, ggml_nbytes(inp));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "failed to compute the decoder graph\n");
        exit(1);
    }

    std::vector<float> wav(ggml_nelements(out));
    ggml_backend_tensor_get(out, wav.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    return wav;
}

static std::vector<float> feedback(
        const llama_model_vibevoice & model,
        ggml_backend_t backend,
        const std::vector<float> & scaled_latents,
        const std::vector<float> & wav,
        int64_t n_frames,
        int64_t n_samples) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * lat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.vibevoice.diff_latent_size, n_frames);
    ggml_set_input(lat);
    ggml_tensor * wav_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_samples, 1);
    ggml_set_input(wav_t);

    ggml_tensor * out = model.build_speech_feedback(ctx, lat, wav_t);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "failed to allocate the speech feedback graph\n");
        exit(1);
    }

    ggml_backend_tensor_set(lat,   scaled_latents.data(), 0, ggml_nbytes(lat));
    ggml_backend_tensor_set(wav_t, wav.data(),            0, ggml_nbytes(wav_t));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "failed to compute the speech feedback graph\n");
        exit(1);
    }

    if (out->ne[0] != model.hparams.n_embd || out->ne[1] != n_frames) {
        fprintf(stderr, "bad feedback shape [%" PRId64 ", %" PRId64 "]\n", out->ne[0], out->ne[1]);
        exit(1);
    }

    std::vector<float> emb(ggml_nelements(out));
    ggml_backend_tensor_get(out, emb.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    return emb;
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
    for (size_t i = 0; i < 8 && i < v.size(); ++i) {
        printf(" %.6f", v[i]);
    }
    printf("\n");

    return ok;
}

static int64_t acoustic_hop(const llama_hparams_vibevoice & vv) {
    int64_t hop = 1;
    for (uint32_t s = 0; s + 1 < vv.acoustic.n_dec_stages; ++s) {
        hop *= vv.acoustic.dec_ratios[s];
    }
    return hop;
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

    const auto * model = dynamic_cast<const llama_model_vibevoice *>(lmodel);
    if (model == nullptr) {
        fprintf(stderr, "%s is not a VibeVoice model\n", argv[1]);
        return 1;
    }

    const auto & vv      = model->hparams.vibevoice;
    const int64_t n_embd = model->hparams.n_embd;
    const int64_t n_lat  = vv.diff_latent_size;
    const int64_t hop    = acoustic_hop(vv);
    const int64_t n_samp = n_frames * hop;

    if (vv.acoustic.vae_dim != vv.diff_latent_size) {
        fprintf(stderr, "latent size %u does not match acoustic vae_dim %u\n", vv.diff_latent_size, vv.acoustic.vae_dim);
        return 1;
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    std::vector<float> noise(n_lat * n_frames);
    for (int64_t f = 0; f < n_frames; ++f) {
        for (int64_t c = 0; c < n_lat; ++c) {
            noise[f * n_lat + c] = 1.5f * sinf(1.37f * c + 2.11f * f + 0.3f);
        }
    }
    std::vector<float> cond(n_embd * n_frames);
    std::vector<float> uncond(n_embd * n_frames);
    for (int64_t f = 0; f < n_frames; ++f) {
        for (int64_t c = 0; c < n_embd; ++c) {
            cond[f * n_embd + c]   = cosf(0.23f * c - 0.71f * f + 0.1f);
            uncond[f * n_embd + c] = sinf(0.19f * c + 0.53f * f - 0.4f);
        }
    }

    bool ok = true;

    // CFG scale 1 must match a cond-only sample; a larger scale must differ from both
    const std::vector<float> lat_cond = model->sample_speech_latents(backend, noise.data(), cond.data(), nullptr,       n_frames, 1.0f);
    const std::vector<float> lat_cfg1 = model->sample_speech_latents(backend, noise.data(), cond.data(), uncond.data(), n_frames, 1.0f);
    const std::vector<float> lat_cfg3 = model->sample_speech_latents(backend, noise.data(), cond.data(), uncond.data(), n_frames, 3.0f);

    ok &= check("latents cond-only", lat_cond, n_lat * n_frames);
    ok &= check("latents cfg=1",     lat_cfg1, n_lat * n_frames);
    ok &= check("latents cfg=3",     lat_cfg3, n_lat * n_frames);

    double max_diff_1 = 0.0;
    double max_diff_3 = 0.0;
    double max_diff_u = 0.0;
    for (size_t i = 0; i < lat_cond.size(); ++i) {
        max_diff_1 = std::max(max_diff_1, (double) std::fabs(lat_cond[i] - lat_cfg1[i]));
        max_diff_3 = std::max(max_diff_3, (double) std::fabs(lat_cond[i] - lat_cfg3[i]));
        max_diff_u = std::max(max_diff_u, (double) std::fabs(lat_cfg1[i] - lat_cfg3[i]));
    }
    printf("cfg=1 vs cond-only max diff = %g\n", max_diff_1);
    printf("cfg=3 vs cond-only max diff = %g\n", max_diff_3);
    printf("cfg=3 vs cfg=1 max diff = %g\n", max_diff_u);
    if (max_diff_1 > 1e-3) {
        fprintf(stderr, "cfg_scale=1 should match a cond-only sample\n");
        ok = false;
    }
    if (max_diff_3 < 1e-3 || max_diff_u < 1e-3) {
        fprintf(stderr, "cfg_scale=3 should differ from cond-only and from cfg=1\n");
        ok = false;
    }

    // Compare changed inputs on a reused graph with the same inputs after a rebuild.
    {
        std::vector<float> noise_alt = noise;
        for (float & x : noise_alt) {
            x = -x;
        }
        const auto reused = model->sample_speech_latents(backend, noise_alt.data(), uncond.data(), cond.data(), n_frames, 1.3f);
        model->sample_speech_latents(backend, noise.data(), cond.data(), nullptr, n_frames, 1.0f);
        const auto rebuilt = model->sample_speech_latents(backend, noise_alt.data(), uncond.data(), cond.data(), n_frames, 1.3f);
        const auto repeated = model->sample_speech_latents(backend, noise.data(), cond.data(), uncond.data(), n_frames, 3.0f);
        ok &= check("latents reused graph", reused, n_lat * n_frames);
        const bool same = reused == rebuilt && repeated == lat_cfg3;
        printf("diffusion graph reuse/rebuild: %s\n", same ? "ok" : "FAIL");
        ok &= same;
    }

    // speech feedback on synthetic scaled latents + wav (always finite on tiny)
    {
        std::vector<float> syn_lat(n_lat * n_frames);
        std::vector<float> syn_wav(n_samp);
        for (size_t i = 0; i < syn_lat.size(); ++i) {
            syn_lat[i] = 0.1f * sinf(0.7f * i + 0.2f);
        }
        for (size_t i = 0; i < syn_wav.size(); ++i) {
            syn_wav[i] = 0.05f * cosf(0.11f * i);
        }
        const std::vector<float> emb = feedback(*model, backend, syn_lat, syn_wav, n_frames, n_samp);
        ok &= check("speech feedback", emb, n_embd * n_frames);
    }

    // full path: sample -> unscale -> decode -> feedback (wav may be non-finite on random tiny weights)
    {
        std::vector<float> scaled = lat_cfg3;
        std::vector<float> unscaled = scaled;
        model->unscale_speech_latents(unscaled.data(), unscaled.size());

        float max_abs = 0.0f;
        for (float x : unscaled) {
            max_abs = std::max(max_abs, std::fabs(x));
        }
        const bool in_range = max_abs < 100.0f;
        if (!in_range) {
            printf("unscaled latents up to %g, out of the trained range, the waveform may not be finite\n", max_abs);
        }

        const std::vector<float> wav = decode(*model, backend, unscaled, n_frames);
        ok &= check("wav", wav, n_samp, in_range);

        bool wav_finite = true;
        for (float x : wav) {
            if (!std::isfinite(x)) {
                wav_finite = false;
                break;
            }
        }
        if (wav_finite) {
            const std::vector<float> emb = feedback(*model, backend, scaled, wav, n_frames, n_samp);
            ok &= check("e2e speech feedback", emb, n_embd * n_frames);
        } else {
            printf("skipping e2e feedback: decoded wav is non-finite on this fixture\n");
        }
    }

    ggml_backend_free(backend);
    llama_model_free(lmodel);
    llama_backend_free();

    printf("%s\n", ok ? "OK" : "FAIL");

    return ok ? 0 : 1;
}
