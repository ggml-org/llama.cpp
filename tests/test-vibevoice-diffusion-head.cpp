// smoke test for the VibeVoice diffusion head graph and its sampler, runs on the CPU with the weights of a VibeVoice GGUF
// the sampled latents are then decoded with the acoustic decoder
// usage: test-vibevoice-diffusion-head <model.gguf> [n_frames]

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

// diffusion head graph with a fixed condition, evaluated once per sampler step
struct head_runner {
    ggml_context * ctx    = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph  * gf     = nullptr;

    ggml_tensor * noisy = nullptr;
    ggml_tensor * c     = nullptr;
    ggml_tensor * t     = nullptr;
    ggml_tensor * out   = nullptr;

    // the allocator may reuse the input buffers, all inputs are set before each compute
    std::vector<float> cond;

    // n_t is the number of timesteps, 1 to share one between all frames
    head_runner(const llama_model_vibevoice & model, ggml_backend_t backend, const std::vector<float> & cond, int64_t n_frames, int64_t n_t) : cond(cond) {
        const auto & hp = model.hparams;

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);

        noisy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.vibevoice.diff_latent_size, n_frames);
        ggml_set_input(noisy);
        c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, n_frames);
        ggml_set_input(c);
        t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_t);
        ggml_set_input(t);

        out = model.build_diffusion_head(ctx, noisy, c, t);
        ggml_set_output(out);

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            fprintf(stderr, "failed to allocate the diffusion head graph\n");
            exit(1);
        }
    }

    ~head_runner() {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }

    std::vector<float> eval(ggml_backend_t backend, const std::vector<float> & x, const std::vector<float> & ts) {
        ggml_backend_tensor_set(noisy, x.data(),    0, ggml_nbytes(noisy));
        ggml_backend_tensor_set(c,     cond.data(), 0, ggml_nbytes(c));
        ggml_backend_tensor_set(t,     ts.data(),   0, ggml_nbytes(t));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "failed to compute the diffusion head graph\n");
            exit(1);
        }

        std::vector<float> v(ggml_nelements(out));
        ggml_backend_tensor_get(out, v.data(), 0, ggml_nbytes(out));
        return v;
    }
};

// decodes latents [vae_dim, n_frames] (vae_dim fastest) into the mono waveform
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

    printf("%s: n = %zu, rms = %.6f, first:", name, v.size(), sqrt(sum2 / v.size()));
    for (size_t i = 0; i < 8 && i < v.size(); ++i) {
        printf(" %.6f", v[i]);
    }
    printf("\n");

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

    const auto * model = dynamic_cast<const llama_model_vibevoice *>(lmodel);
    if (model == nullptr) {
        fprintf(stderr, "%s is not a VibeVoice model\n", argv[1]);
        return 1;
    }

    const auto & vv     = model->hparams.vibevoice;
    const int64_t n_embd = model->hparams.n_embd;
    const int64_t n_lat  = vv.diff_latent_size;

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    // deterministic noise and condition, simple enough to reproduce in a reference implementation
    std::vector<float> noise(n_lat * n_frames);
    for (int64_t f = 0; f < n_frames; ++f) {
        for (int64_t c = 0; c < n_lat; ++c) {
            noise[f * n_lat + c] = 1.5f * sinf(1.37f * c + 2.11f * f + 0.3f);
        }
    }
    std::vector<float> cond(n_embd * n_frames);
    for (int64_t f = 0; f < n_frames; ++f) {
        for (int64_t c = 0; c < n_embd; ++c) {
            cond[f * n_embd + c] = cosf(0.23f * c - 0.71f * f + 0.1f);
        }
    }

    llama_model_vibevoice::diff_sampler sampler(vv.diff_train_steps, vv.diff_inference_steps);

    printf("diffusion steps = %zu, timesteps:", sampler.timesteps.size());
    for (float t : sampler.timesteps) {
        printf(" %.0f", t);
    }
    printf("\n");

    bool ok = true;

    // one evaluation at the first timestep, shared and per frame timesteps must agree
    {
        head_runner head_shared(*model, backend, cond, n_frames, 1);
        head_runner head_frame (*model, backend, cond, n_frames, n_frames);

        const std::vector<float> v = head_shared.eval(backend, noise, { sampler.timesteps[0] });
        ok &= check("head v", v, n_lat * n_frames);

        const std::vector<float> v_frame = head_frame.eval(backend, noise, std::vector<float>(n_frames, sampler.timesteps[0]));

        double max_diff = 0.0;
        for (size_t i = 0; i < v.size() && i < v_frame.size(); ++i) {
            max_diff = std::max(max_diff, (double) std::fabs(v[i] - v_frame[i]));
        }
        printf("shared vs per frame timestep max diff = %g\n", max_diff);
        if (max_diff > 1e-6) {
            fprintf(stderr, "a shared timestep does not match the same timestep per frame\n");
            ok = false;
        }
    }

    // full sampling loop
    std::vector<float> latents = noise;
    {
        head_runner head(*model, backend, cond, n_frames, 1);

        for (float t : sampler.timesteps) {
            const std::vector<float> v = head.eval(backend, latents, { t });
            sampler.step(v.data(), latents.data(), latents.size());
        }
    }
    ok &= check("latents", latents, n_lat * n_frames);

    // undo the speech scaling and decode
    if (vv.acoustic.vae_dim == vv.diff_latent_size) {
        std::vector<float> unscaled(latents.size());
        float max_abs = 0.0f;
        for (size_t i = 0; i < latents.size(); ++i) {
            unscaled[i] = latents[i] / vv.speech_scaling_factor - vv.speech_bias_factor;
            max_abs = std::max(max_abs, std::fabs(unscaled[i]));
        }

        // trained heads give latents of a few units, random-weight fixtures give latents large enough to overflow
        // the F16 conversions in the CPU conv kernels of the decoder, only the length is checked for them
        const bool in_range = max_abs < 100.0f;
        if (!in_range) {
            printf("unscaled latents up to %g, out of the trained range, the waveform may not be finite\n", max_abs);
        }

        int64_t hop = 1;
        for (uint32_t s = 0; s + 1 < vv.acoustic.n_dec_stages; ++s) {
            hop *= vv.acoustic.dec_ratios[s];
        }

        ok &= check("wav", decode(*model, backend, unscaled, n_frames), n_frames * hop, in_range);
    } else {
        fprintf(stderr, "latent size %u does not match the acoustic vae_dim %u\n", vv.diff_latent_size, vv.acoustic.vae_dim);
        ok = false;
    }

    ggml_backend_free(backend);
    llama_model_free(lmodel);
    llama_backend_free();

    printf("%s\n", ok ? "OK" : "FAIL");

    return ok ? 0 : 1;
}
