// smoke test for the VibeVoice sampled generation loop: pos logits -> sampler -> branch on the token
// random-weight fixtures do not argmax to the audio tokens, so a logit bias sampler picks audio_bos / audio_token for the
// scripted steps, the last steps are plain greedy and must match the argmax of the logits
// usage: test-vibevoice-sampler <model.gguf> [n_frames]

#include "vibevoice-gen.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// logit bias on one token, then greedy
static llama_sampler * init_forced(int32_t n_vocab, llama_token tok) {
    const llama_logit_bias bias = { tok, 1e9f };

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(n_vocab, 1, &bias));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    return smpl;
}

static llama_token argmax_logits(llama_context * ctx, int32_t n_vocab, bool & finite) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        finite = false;
        return LLAMA_TOKEN_NULL;
    }
    llama_token best = 0;
    finite = true;
    for (llama_token i = 0; i < n_vocab; ++i) {
        finite &= std::isfinite(logits[i]);
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

static bool run(llama_model * lmodel, int64_t n_frames) {
    vv_generator gen;
    if (!gen.init(lmodel, 256)) {
        return false;
    }

    const llama_token audio_bos   = llama_model_vibevoice::audio_bos_token_id;
    const llama_token audio_token = llama_model_vibevoice::audio_token_id;
    const int32_t     n_vocab     = llama_vocab_n_tokens(gen.vocab);

    llama_sampler * smpl_greedy = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl_greedy, llama_sampler_init_greedy());
    llama_sampler * smpl_bos   = init_forced(n_vocab, audio_bos);
    llama_sampler * smpl_audio = init_forced(n_vocab, audio_token);

    bool ok = gen.prefill({ 1, 2, 3, 4, 5 });

    // scripted steps: audio_bos, half of the frames, audio_bos again (neg reset), the other frames
    struct expect {
        llama_sampler * smpl;
        llama_token     tok;
    };
    std::vector<expect> script = { { smpl_bos, audio_bos } };
    for (int64_t i = 0; i < n_frames; ++i) {
        if (i == n_frames / 2 && i > 0) {
            script.push_back({ smpl_bos, audio_bos });
        }
        script.push_back({ smpl_audio, audio_token });
    }

    int64_t n_neg_expected = 1;
    for (size_t s = 0; s < script.size() && ok; ++s) {
        const int64_t pos_before = gen.pos.n_cached();
        const int64_t neg_before = gen.neg.n_cached();

        const llama_token tok = gen.step(script[s].smpl);
        if (tok != script[s].tok) {
            fprintf(stderr, "step %zu: sampled %d, expected %d\n", s, tok, script[s].tok);
            ok = false;
            break;
        }

        n_neg_expected = tok == audio_bos ? 1 : neg_before + 1;
        printf("step %zu: %s, pos cache %" PRId64 ", neg cache %" PRId64 ", frames %" PRId64 "\n",
                s, tok == audio_bos ? "audio_bos" : "audio_token", gen.pos.n_cached(), gen.neg.n_cached(), gen.n_frames);
        if (gen.pos.n_cached() != pos_before + 1 || gen.neg.n_cached() != n_neg_expected) {
            fprintf(stderr, "step %zu: unexpected cache sizes\n", s);
            ok = false;
        }
    }

    if (ok && (gen.n_frames != n_frames || (int64_t) gen.wav.size() != n_frames * gen.hop)) {
        fprintf(stderr, "got %" PRId64 " frames and %zu samples, expected %" PRId64 " and %" PRId64 "\n",
                gen.n_frames, gen.wav.size(), n_frames, n_frames * gen.hop);
        ok = false;
    }

    // free greedy steps: the sampled token is the argmax, text tokens only advance pos
    for (int s = 0; s < 4 && ok; ++s) {
        bool finite = false;
        const llama_token best = argmax_logits(gen.pos.ctx, n_vocab, finite);
        if (!finite) {
            fprintf(stderr, "greedy step %d: missing or non-finite logits\n", s);
            ok = false;
            break;
        }

        const int64_t pos_before    = gen.pos.n_cached();
        const int64_t neg_before    = gen.neg.n_cached();
        const int64_t frames_before = gen.n_frames;

        const llama_token tok = gen.step(smpl_greedy);
        printf("greedy step %d: sampled %d (argmax %d), pos cache %" PRId64 ", neg cache %" PRId64 "\n",
                s, tok, best, gen.pos.n_cached(), gen.neg.n_cached());
        if (tok == LLAMA_TOKEN_NULL || tok != best) {
            fprintf(stderr, "greedy step %d: sample does not match the argmax\n", s);
            ok = false;
            break;
        }
        if (gen.is_eog(tok)) {
            printf("eog, stop\n");
            break;
        }

        bool sizes_ok = true;
        if (tok == audio_token) {
            sizes_ok = gen.n_frames == frames_before + 1 && gen.neg.n_cached() == neg_before + 1;
        } else if (tok == audio_bos) {
            sizes_ok = gen.neg.n_cached() == 1;
        } else {
            sizes_ok = gen.neg.n_cached() == neg_before;
        }
        if (!sizes_ok || gen.pos.n_cached() != pos_before + 1) {
            fprintf(stderr, "greedy step %d: unexpected cache sizes\n", s);
            ok = false;
        }
    }

    if (gen.n_nonfinite > 0) {
        printf("decoded wav had %" PRId64 " non-finite samples on this fixture, replaced by 0\n", gen.n_nonfinite);
    }
    printf("total: %" PRId64 " frames, %zu samples\n", gen.n_frames, gen.wav.size());

    llama_sampler_free(smpl_greedy);
    llama_sampler_free(smpl_bos);
    llama_sampler_free(smpl_audio);

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
