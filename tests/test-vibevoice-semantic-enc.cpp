// smoke test for the VibeVoice semantic encoder + connector graph
// usage: test-vibevoice-semantic-enc <model.gguf> [n_frames]
// n_frames is the number of latent frames (wav length = n_frames * hop)

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

struct enc_out {
    std::vector<float> latents; // [vae_dim, n_frames], vae_dim fastest
    std::vector<float> embd;    // [n_embd, n_frames]
    int64_t n_frames = 0;
    int64_t vae_dim  = 0;
    int64_t n_embd   = 0;
};

static enc_out encode(
        const llama_model_vibevoice & model,
        ggml_backend_t backend,
        const std::vector<float> & wav,
        int64_t n_samples) {
    const auto & hp = model.hparams.vibevoice.semantic;
    const int64_t n_embd = model.hparams.n_embd;

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_samples, 1);
    ggml_set_input(inp);

    ggml_tensor * latents = model.build_semantic_enc(ctx, inp);
    ggml_set_output(latents);

    ggml_tensor * embd = model.build_connector(ctx, latents, model.semantic_conn);
    ggml_set_output(embd);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, embd);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "failed to allocate the semantic encoder graph\n");
        exit(1);
    }

    ggml_backend_tensor_set(inp, wav.data(), 0, n_samples * sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "failed to compute the semantic encoder graph\n");
        exit(1);
    }

    if (latents->ne[0] != hp.vae_dim) {
        fprintf(stderr, "expected vae_dim %" PRIu32 ", got %" PRId64 "\n", hp.vae_dim, latents->ne[0]);
        exit(1);
    }
    if (embd->ne[0] != n_embd || embd->ne[1] != latents->ne[1]) {
        fprintf(stderr, "bad connector shape [%" PRId64 ", %" PRId64 "]\n", embd->ne[0], embd->ne[1]);
        exit(1);
    }

    enc_out out;
    out.vae_dim  = latents->ne[0];
    out.n_frames = latents->ne[1];
    out.n_embd   = embd->ne[0];
    out.latents.resize(out.vae_dim * out.n_frames);
    out.embd.resize(out.n_embd * out.n_frames);
    ggml_backend_tensor_get(latents, out.latents.data(), 0, ggml_nbytes(latents));
    ggml_backend_tensor_get(embd,    out.embd.data(),    0, ggml_nbytes(embd));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    return out;
}

static int64_t hop_length(const llama_hparams_vv_tokenizer & hp) {
    int64_t hop = 1;
    for (uint32_t i = 0; i + 1 < hp.n_enc_stages; ++i) {
        hop *= hp.enc_ratios[i];
    }
    return hop;
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

    const auto & hp = model->hparams.vibevoice.semantic;
    const int64_t hop = hop_length(hp);
    const int64_t n_samples = n_frames * hop;

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    std::vector<float> wav(n_samples);
    for (int64_t t = 0; t < n_samples; ++t) {
        wav[t] = sinf(0.13f * t + 0.7f);
    }

    const enc_out out = encode(*model, backend, wav, n_samples);

    bool ok = true;

    if (out.n_frames != n_frames) {
        fprintf(stderr, "expected %" PRId64 " frames, got %" PRId64 "\n", n_frames, out.n_frames);
        ok = false;
    }

    double lat_sum2 = 0.0;
    double emb_sum2 = 0.0;
    for (float v : out.latents) {
        if (!std::isfinite(v)) {
            fprintf(stderr, "non-finite latent\n");
            ok = false;
            break;
        }
        lat_sum2 += (double) v * v;
    }
    for (float v : out.embd) {
        if (!std::isfinite(v)) {
            fprintf(stderr, "non-finite connector output\n");
            ok = false;
            break;
        }
        emb_sum2 += (double) v * v;
    }

    printf("n_samples = %" PRId64 ", hop = %" PRId64 ", n_frames = %" PRId64 ", vae_dim = %" PRId64 ", n_embd = %" PRId64 "\n",
            n_samples, hop, out.n_frames, out.vae_dim, out.n_embd);
    printf("latents rms = %.6f, connector rms = %.6f\n",
            sqrt(lat_sum2 / out.latents.size()), sqrt(emb_sum2 / out.embd.size()));
    printf("first latent frame:");
    for (int64_t c = 0; c < std::min<int64_t>(out.vae_dim, 8); ++c) {
        printf(" %.6f", out.latents[c]);
    }
    printf("\n");

    // causal: encoding a prefix of the waveform (aligned to hop) gives a prefix of the latents
    const int64_t n_half = n_frames / 2;
    const std::vector<float> wav_half(wav.begin(), wav.begin() + n_half * hop);
    const enc_out half = encode(*model, backend, wav_half, n_half * hop);

    if (half.n_frames != n_half) {
        fprintf(stderr, "half encode produced %" PRId64 " frames, expected %" PRId64 "\n", half.n_frames, n_half);
        ok = false;
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < half.latents.size() && i < out.latents.size(); ++i) {
        max_diff = std::max(max_diff, (double) std::fabs(half.latents[i] - out.latents[i]));
    }
    printf("causal prefix max diff = %g\n", max_diff);
    if (max_diff > 1e-5) {
        fprintf(stderr, "encoding a prefix changed the latents, the encoder is not causal\n");
        ok = false;
    }

    ggml_backend_free(backend);
    llama_model_free(lmodel);
    llama_backend_free();

    printf("%s\n", ok ? "OK" : "FAIL");

    return ok ? 0 : 1;
}
