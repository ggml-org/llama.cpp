// smoke test for the VibeVoice acoustic decoder graph, runs it on the CPU with the weights of a VibeVoice GGUF
// usage: test-vibevoice-acoustic-dec <model.gguf> [n_frames]

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

// decodes latents [vae_dim, n_frames] (vae_dim fastest) into the mono waveform
static std::vector<float> decode(const llama_model_vibevoice & model, ggml_backend_t backend, const std::vector<float> & latents, int64_t n_frames) {
    const int64_t vae_dim = model.hparams.vibevoice.acoustic.vae_dim;

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vae_dim, n_frames);
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

    ggml_backend_tensor_set(inp, latents.data(), 0, vae_dim * n_frames * sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "failed to compute the decoder graph\n");
        exit(1);
    }

    if (out->ne[1] != 1) {
        fprintf(stderr, "expected a mono waveform, got %" PRId64 " channels\n", out->ne[1]);
        exit(1);
    }

    std::vector<float> wav(out->ne[0]);
    ggml_backend_tensor_get(out, wav.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    return wav;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_frames]\n", argv[0]);
        return 1;
    }
    const int64_t n_frames = argc > 2 ? std::atoll(argv[2]) : 16;
    if (n_frames < 2) {
        fprintf(stderr, "n_frames must be at least 2\n");
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

    const auto & hp = model->hparams.vibevoice.acoustic;

    int64_t hop = 1;
    for (uint32_t s = 0; s + 1 < hp.n_dec_stages; ++s) {
        hop *= hp.dec_ratios[s];
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    // deterministic latents, simple enough to reproduce in a reference implementation
    std::vector<float> latents(hp.vae_dim * n_frames);
    for (int64_t t = 0; t < n_frames; ++t) {
        for (uint32_t c = 0; c < hp.vae_dim; ++c) {
            latents[t * hp.vae_dim + c] = sinf(0.37f * t + 0.11f * c + 0.5f);
        }
    }

    const std::vector<float> wav = decode(*model, backend, latents, n_frames);

    bool ok = true;

    if ((int64_t) wav.size() != n_frames * hop) {
        fprintf(stderr, "expected %" PRId64 " samples, got %zu\n", n_frames * hop, wav.size());
        ok = false;
    }

    double sum2 = 0.0;
    for (float v : wav) {
        if (!std::isfinite(v)) {
            fprintf(stderr, "non-finite sample in the output\n");
            ok = false;
            break;
        }
        sum2 += (double) v * v;
    }

    printf("n_frames = %" PRId64 ", hop = %" PRId64 ", n_samples = %zu, rms = %.6f\n", n_frames, hop, wav.size(), sqrt(sum2 / wav.size()));
    printf("first samples:");
    for (size_t i = 0; i < 8 && i < wav.size(); ++i) {
        printf(" %.6f", wav[i]);
    }
    printf("\n");

    // the decoder is causal: decoding a prefix of the latents gives a prefix of the waveform
    const int64_t n_half = n_frames / 2;
    const std::vector<float> latents_half(latents.begin(), latents.begin() + hp.vae_dim * n_half);
    const std::vector<float> wav_half = decode(*model, backend, latents_half, n_half);

    double max_diff = 0.0;
    for (size_t i = 0; i < wav_half.size() && i < wav.size(); ++i) {
        max_diff = std::max(max_diff, (double) std::fabs(wav_half[i] - wav[i]));
    }
    printf("causal prefix max diff = %g\n", max_diff);
    if (max_diff > 1e-5) {
        fprintf(stderr, "decoding a prefix changed the waveform, the decoder is not causal\n");
        ok = false;
    }

    ggml_backend_free(backend);
    llama_model_free(lmodel);
    llama_backend_free();

    printf("%s\n", ok ? "OK" : "FAIL");

    return ok ? 0 : 1;
}
