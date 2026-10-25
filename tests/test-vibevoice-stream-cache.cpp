// smoke test for the VibeVoice streaming tokenizer: chunk by chunk decode/encode with causal conv caches
// must match the full sequence graphs (acoustic decoder, semantic encoder, speech feedback)
// usage: test-vibevoice-stream-cache <model.gguf> [n_frames]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "models/models.h"
#include "vibevoice-gen.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <functional>
#include <string>
#include <vector>

// full sequence graph with one or two inputs, returns the output
static std::vector<float> run_full(
        ggml_backend_t backend,
        const std::vector<std::pair<std::array<int64_t, 2>, const float *>> & inputs,
        const std::function<ggml_tensor * (ggml_context *, const std::vector<ggml_tensor *> &)> & build) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    std::vector<ggml_tensor *> inp;
    for (const auto & [ne, data] : inputs) {
        inp.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]));
        ggml_set_input(inp.back());
    }

    ggml_tensor * out = build(ctx, inp);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "failed to allocate the graph\n");
        exit(1);
    }

    for (size_t i = 0; i < inp.size(); ++i) {
        ggml_backend_tensor_set(inp[i], inputs[i].second, 0, ggml_nbytes(inp[i]));
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "failed to compute the graph\n");
        exit(1);
    }

    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    return res;
}

static bool all_finite(const std::vector<float> & v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}

// streamed vs full: same length, and when the reference is finite a max abs diff within tol of its max abs
static bool compare(const char * name, const std::vector<float> & full, const std::vector<float> & stream, double tol = 1e-3) {
    if (full.size() != stream.size()) {
        fprintf(stderr, "%s: full has %zu values, stream %zu\n", name, full.size(), stream.size());
        return false;
    }
    if (!all_finite(full)) {
        printf("%s: n = %zu, full output is not finite on this fixture, only the length is checked\n", name, full.size());
        return true;
    }

    double max_ref  = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < full.size(); ++i) {
        max_ref  = std::max(max_ref,  (double) std::fabs(full[i]));
        max_diff = std::max(max_diff, (double) std::fabs(full[i] - stream[i]));
    }
    const double rel = max_ref > 0.0 ? max_diff / max_ref : max_diff;
    const bool ok = all_finite(stream) && rel <= tol;

    printf("%s: n = %zu, max |full| = %.6g, max diff = %.3g, rel = %.3g %s\n", name, full.size(), max_ref, max_diff, rel, ok ? "ok" : "FAIL");
    return ok;
}

static bool check_segment_policy(llama_model * lmodel) {
    vv_generator gen;
    if (!gen.init(lmodel, 256) || !gen.prefill({ 1, 2, 3, 4, 5 })) {
        return false;
    }

    const auto * model = gen.model;
    const int64_t n_lat = model->hparams.vibevoice.diff_latent_size;
    const int64_t n_frames = 3;
    std::vector<float> lat(n_lat * n_frames);
    std::vector<float> wav(gen.hop * n_frames);
    for (size_t i = 0; i < lat.size(); ++i) {
        lat[i] = 0.01f * sinf(0.37f * i + 0.5f);
    }
    for (size_t i = 0; i < wav.size(); ++i) {
        wav[i] = 0.05f * cosf(0.11f * i) + 0.02f * sinf(0.53f * i);
    }

    const auto full_dec = run_full(gen.backend, { { { n_lat, n_frames }, lat.data() } },
            [&](ggml_context * ctx, const std::vector<ggml_tensor *> & inp) { return model->build_acoustic_dec(ctx, inp[0]); });
    const auto full_emb = run_full(gen.backend, { { { n_lat, n_frames }, lat.data() }, { { gen.hop * n_frames, 1 }, wav.data() } },
            [&](ggml_context * ctx, const std::vector<ggml_tensor *> & inp) { return model->build_speech_feedback(ctx, inp[0], inp[1]); });
    if (!all_finite(full_dec) || !all_finite(full_emb)) {
        fprintf(stderr, "segment policy requires finite reference outputs\n");
        return false;
    }

    std::vector<float> stream_emb;
    for (int64_t i = 0; i < n_frames; ++i) {
        if (i == 2) {
            const auto dec_before = gen.dec_cache.convs;
            const auto sem_before = gen.sem_cache.convs;
            const int64_t pos_before = gen.pos.n_cached();
            if (!gen.audio_bos() || gen.pos.n_cached() != pos_before + 1 || gen.neg.n_cached() != 1 || gen.neg.pos != 1 ||
                    gen.dec_cache.convs != dec_before || gen.sem_cache.convs != sem_before) {
                fprintf(stderr, "audio_bos must reset only negative LM KV and keep both stream caches\n");
                return false;
            }
        }

        // finite inputs isolate cache policy from diffusion overflow on the tiny fixture
        gen.lat.assign(lat.begin() + i * n_lat, lat.begin() + (i + 1) * n_lat);
        const auto chunk = model->acoustic_dec_stream(gen.backend, gen.dec_cache, gen.lat.data(), 1);
        gen.emb = model->speech_feedback_stream(gen.backend, gen.sem_cache, gen.lat.data(), wav.data() + i * gen.hop, 1);
        if (!all_finite(chunk) || !all_finite(gen.emb) || (int64_t) chunk.size() != gen.hop || (int64_t) gen.emb.size() != gen.n_embd) {
            fprintf(stderr, "segment policy requires one finite waveform chunk and feedback column per frame\n");
            return false;
        }
        gen.wav.insert(gen.wav.end(), chunk.begin(), chunk.end());
        stream_emb.insert(stream_emb.end(), gen.emb.begin(), gen.emb.end());
        gen.n_frames++;
        if (!gen.pos.decode_embd(gen.emb.data(), gen.n_embd) || !gen.neg.decode_embd(gen.emb.data(), gen.n_embd)) {
            return false;
        }
    }

    bool ok = compare("audio_bos keeps continuous decoder", full_dec, gen.wav);
    ok &= compare("audio_bos keeps continuous feedback", full_emb, stream_emb);

    const auto wav_before = gen.wav;
    const auto lat_before = gen.lat;
    const auto emb_before = gen.emb;
    const auto pos_before = gen.pos.pos;
    const auto neg_before = gen.neg.pos;
    const int64_t pos_cached = gen.pos.n_cached();
    const int64_t neg_cached = gen.neg.n_cached();
    const int64_t nonfinite_before = gen.n_nonfinite;
    gen.reset_speech_stream();
    if (!gen.dec_cache.convs.empty() || !gen.sem_cache.convs.empty() ||
            gen.pos.pos != pos_before || gen.neg.pos != neg_before || gen.pos.n_cached() != pos_cached || gen.neg.n_cached() != neg_cached ||
            gen.wav != wav_before || gen.lat != lat_before || gen.emb != emb_before || gen.n_frames != n_frames || gen.n_nonfinite != nonfinite_before) {
        fprintf(stderr, "reset_speech_stream must clear only tokenizer stream caches\n");
        return false;
    }

    const auto cold_dec = run_full(gen.backend, { { { n_lat, 1 }, gen.lat.data() } },
            [&](ggml_context * ctx, const std::vector<ggml_tensor *> & inp) { return model->build_acoustic_dec(ctx, inp[0]); });
    const auto cold_emb = run_full(gen.backend, { { { n_lat, 1 }, gen.lat.data() }, { { gen.hop, 1 }, wav.data() + 2 * gen.hop } },
            [&](ggml_context * ctx, const std::vector<ggml_tensor *> & inp) { return model->build_speech_feedback(ctx, inp[0], inp[1]); });
    const auto reset_dec = model->acoustic_dec_stream(gen.backend, gen.dec_cache, gen.lat.data(), 1);
    const auto reset_emb = model->speech_feedback_stream(gen.backend, gen.sem_cache, gen.lat.data(), wav.data() + 2 * gen.hop, 1);
    if (!all_finite(cold_dec) || !all_finite(cold_emb)) {
        fprintf(stderr, "segment policy requires finite cold reference outputs\n");
        return false;
    }
    ok &= compare("reset_speech_stream cold decoder", cold_dec, reset_dec);
    ok &= compare("reset_speech_stream cold feedback", cold_emb, reset_emb);
    if (reset_dec == std::vector<float>(gen.wav.end() - gen.hop, gen.wav.end()) || reset_emb == gen.emb) {
        fprintf(stderr, "cold decoder and feedback must differ from the continuation after audio_bos\n");
        ok = false;
    }

    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_frames]\n", argv[0]);
        return 1;
    }
    const int64_t n_frames = argc > 2 ? std::atoll(argv[2]) : 6;
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
    const int64_t n_aco  = vv.acoustic.vae_dim;
    const int64_t n_sem  = vv.semantic.vae_dim;

    int64_t hop = 1;
    for (uint32_t s = 0; s + 1 < vv.acoustic.n_dec_stages; ++s) {
        hop *= vv.acoustic.dec_ratios[s];
    }
    const int64_t n_samp = n_frames * hop;

    if (n_aco != n_lat) {
        fprintf(stderr, "latent size %" PRId64 " does not match acoustic vae_dim %" PRId64 "\n", n_lat, n_aco);
        return 1;
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    // chunkings in frames: one by one, and uneven chunks
    std::vector<std::vector<int64_t>> chunkings = { std::vector<int64_t>(n_frames, 1) };
    {
        std::vector<int64_t> uneven;
        for (int64_t done = 0, k = 2; done < n_frames; k = k % 3 + 1) {
            uneven.push_back(std::min(k, n_frames - done));
            done += uneven.back();
        }
        chunkings.push_back(uneven);
    }

    const auto chunking_name = [](const std::vector<int64_t> & ch) {
        std::string s;
        for (int64_t c : ch) {
            s += (s.empty() ? "" : "+") + std::to_string(c);
        }
        return s;
    };

    bool ok = true;

    // acoustic decoder on small unscaled latents (finite on random fixtures) and on unit ones
    for (float amp : { 0.01f, 1.0f }) {
        std::vector<float> lat(n_aco * n_frames);
        for (size_t i = 0; i < lat.size(); ++i) {
            lat[i] = amp * sinf(0.37f * i + 0.5f);
        }

        const std::vector<float> full = run_full(backend, { { { n_aco, n_frames }, lat.data() } },
                [&](ggml_context * ctx, const std::vector<ggml_tensor *> & inp) { return model->build_acoustic_dec(ctx, inp[0]); });

        for (const auto & ch : chunkings) {
            llama_model_vibevoice::tok_stream_cache cache;
            std::vector<float> stream;
            int64_t done = 0;
            for (int64_t n : ch) {
                const std::vector<float> wav = model->acoustic_dec_stream(backend, cache, lat.data() + done * n_aco, n);
                if ((int64_t) wav.size() != n * hop) {
                    fprintf(stderr, "acoustic_dec_stream: %zu samples for %" PRId64 " frames\n", wav.size(), n);
                    ok = false;
                }
                stream.insert(stream.end(), wav.begin(), wav.end());
                done += n;
            }

            // guard: the last chunk decoded from a cold cache must not match, the left context is really used
            if (ch.size() > 1 && all_finite(full)) {
                llama_model_vibevoice::tok_stream_cache cold;
                const int64_t n = ch.back();
                const std::vector<float> tail = model->acoustic_dec_stream(backend, cold, lat.data() + (n_frames - n) * n_aco, n);
                double d = 0.0;
                for (size_t i = 0; i < tail.size(); ++i) {
                    d = std::max(d, (double) std::fabs(tail[i] - full[full.size() - tail.size() + i]));
                }
                if (!(d > 0.0)) {
                    fprintf(stderr, "a cold cache gives the same last chunk, the cache is not used\n");
                    ok = false;
                }
            }

            const std::string name = "acoustic dec amp " + std::to_string(amp) + " chunks " + chunking_name(ch) + " (" + std::to_string(cache.convs.size()) + " caches)";
            ok &= compare(name.c_str(), full, stream);
        }
    }

    // semantic encoder on a finite synthetic waveform
    std::vector<float> wav(n_samp);
    for (size_t i = 0; i < wav.size(); ++i) {
        wav[i] = 0.05f * cosf(0.11f * i) + 0.02f * sinf(0.53f * i);
    }
    {
        const std::vector<float> full = run_full(backend, { { { n_samp, 1 }, wav.data() } },
                [&](ggml_context * ctx, const std::vector<ggml_tensor *> & inp) { return model->build_semantic_enc(ctx, inp[0]); });
        if ((int64_t) full.size() != n_sem * n_frames) {
            fprintf(stderr, "semantic enc: unexpected full size %zu\n", full.size());
            ok = false;
        }

        for (const auto & ch : chunkings) {
            llama_model_vibevoice::tok_stream_cache cache;
            std::vector<float> stream;
            int64_t done = 0;
            for (int64_t n : ch) {
                const std::vector<float> lat = model->semantic_enc_stream(backend, cache, wav.data() + done * hop, n * hop);
                stream.insert(stream.end(), lat.begin(), lat.end());
                done += n;
            }

            const std::string name = "semantic enc chunks " + chunking_name(ch) + " (" + std::to_string(cache.convs.size()) + " caches)";
            ok &= compare(name.c_str(), full, stream);

            // a cleared cache is a cold start
            cache.clear();
            const std::vector<float> lat0 = model->semantic_enc_stream(backend, cache, wav.data(), ch[0] * hop);
            ok &= compare("  after clear, first chunk", std::vector<float>(stream.begin(), stream.begin() + lat0.size()), lat0, 0.0);
        }
    }

    // speech feedback frame by frame: column i equals column i of the full feedback over all frames
    {
        std::vector<float> scaled(n_lat * n_frames);
        for (size_t i = 0; i < scaled.size(); ++i) {
            scaled[i] = 0.1f * sinf(0.7f * i + 0.2f);
        }

        const std::vector<float> full = run_full(backend, { { { n_lat, n_frames }, scaled.data() }, { { n_samp, 1 }, wav.data() } },
                [&](ggml_context * ctx, const std::vector<ggml_tensor *> & inp) { return model->build_speech_feedback(ctx, inp[0], inp[1]); });

        llama_model_vibevoice::tok_stream_cache cache;
        std::vector<float> stream;
        for (int64_t i = 0; i < n_frames; ++i) {
            const std::vector<float> emb = model->speech_feedback_stream(backend, cache, scaled.data() + i * n_lat, wav.data() + i * hop, 1);
            if ((int64_t) emb.size() != n_embd) {
                fprintf(stderr, "speech_feedback_stream: %zu values for one frame\n", emb.size());
                ok = false;
            }
            stream.insert(stream.end(), emb.begin(), emb.end());
        }
        ok &= compare("speech feedback frame by frame", full, stream);

        // the last streamed frame is the last column of the full feedback (what the generation loop feeds back)
        ok &= compare("speech feedback last frame",
                std::vector<float>(full.end() - n_embd, full.end()), std::vector<float>(stream.end() - n_embd, stream.end()));
    }

    ok &= check_segment_policy(lmodel);

    ggml_backend_free(backend);
    llama_model_free(lmodel);
    llama_backend_free();

    printf("%s\n", ok ? "OK" : "FAIL");

    return ok ? 0 : 1;
}
