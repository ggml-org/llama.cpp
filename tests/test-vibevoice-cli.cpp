// minimal VibeVoice text to speech: greedy LM loop with CFG diffusion frames, writes a mono s16 wav
// random-weight fixtures never sample audio_token, use --force-audio to bias it on every step
// usage: test-vibevoice-cli <model.gguf> [-n max_frames] [-o out.wav] [--cfg scale] [--seed n]
//                           [--tokens 1,2,3 | -p prompt] [--max-tokens n] [--force-audio]

#include "vibevoice-gen.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// upstream processor default, not stored in the GGUF
static const int32_t VV_SAMPLE_RATE = 24000;

static bool write_wav_s16(const std::string & path, const std::vector<float> & pcm, int32_t rate) {
    // RIFF chunk sizes are 32-bit
    if (pcm.size() > ((size_t) UINT32_MAX - 36) / 2) {
        return false;
    }
    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    const uint32_t data_sz = (uint32_t) (pcm.size() * 2);
    const uint32_t riff_sz = 36 + data_sz;
    const uint32_t fmt_sz  = 16, byte_rate = (uint32_t) rate * 2;
    const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    const uint32_t rate32  = (uint32_t) rate;
    auto put = [&](const void * p, size_t n) {
        fwrite(p, 1, n, f);
    };
    put("RIFF", 4); put(&riff_sz, 4); put("WAVE", 4);
    put("fmt ", 4); put(&fmt_sz, 4);
    put(&fmt, 2); put(&ch, 2); put(&rate32, 4);
    put(&byte_rate, 4); put(&align, 2); put(&bits, 2);
    put("data", 4); put(&data_sz, 4);
    for (float v : pcm) {
        int16_t s = (int16_t) (std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
        put(&s, 2);
    }
    return fclose(f) == 0;
}

static std::vector<llama_token> parse_tokens(const char * s) {
    std::vector<llama_token> res;
    while (*s) {
        char * end = nullptr;
        const long v = strtol(s, &end, 10);
        if (end == s) {
            return {};
        }
        res.push_back((llama_token) v);
        s = end;
        while (*s == ',' || *s == ' ') {
            s++;
        }
    }
    return res;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int32_t n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> res(std::max(n, 0));
    if (n <= 0 || llama_tokenize(vocab, text.c_str(), text.size(), res.data(), res.size(), true, true) < 0) {
        return {};
    }
    return res;
}

struct cli_params {
    std::string out_path = "/tmp/vv-out/out.wav";
    std::string prompt;
    std::vector<llama_token> tokens = { 1, 2, 3, 4, 5 };

    int64_t  max_frames  = 32;
    int64_t  max_tokens  = 256;
    float    cfg_scale   = 1.3f;
    uint32_t seed        = 1234;
    bool     force_audio = false;
};

static bool run(llama_model * lmodel, cli_params & params) {
    const llama_vocab * vocab = llama_model_get_vocab(lmodel);
    if (!params.prompt.empty()) {
        params.tokens = tokenize(vocab, params.prompt);
    }
    if (params.tokens.empty()) {
        fprintf(stderr, "empty or invalid prompt\n");
        return false;
    }

    // every step adds at most one position to each branch
    vv_generator gen;
    gen.cfg_scale = params.cfg_scale;
    gen.seed      = params.seed;
    if (!gen.init(lmodel, params.tokens.size() + params.max_tokens + 1) || !gen.prefill(params.tokens)) {
        return false;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const llama_logit_bias bias = { llama_model_vibevoice::audio_token_id, 1e9f };

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.force_audio) {
        llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(n_vocab, 1, &bias));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    printf("prompt: %zu tokens, max %" PRId64 " frames, max %" PRId64 " tokens, cfg %.2f, seed %u%s\n", params.tokens.size(),
            params.max_frames, params.max_tokens, params.cfg_scale, params.seed, params.force_audio ? ", forced audio" : "");

    bool ok  = true;
    bool eog = false;
    int64_t n_steps = 0;
    int64_t n_text  = 0;
    for (; n_steps < params.max_tokens && gen.n_frames < params.max_frames; ++n_steps) {
        const llama_token tok = gen.step(smpl);
        if (tok == LLAMA_TOKEN_NULL) {
            ok = false;
            break;
        }
        if (gen.is_eog(tok)) {
            eog = true;
            break;
        }
        if (tok != llama_model_vibevoice::audio_token_id && tok != llama_model_vibevoice::audio_bos_token_id) {
            n_text++;
        }
    }
    llama_sampler_free(smpl);

    printf("steps %" PRId64 ", frames %" PRId64 ", text tokens %" PRId64 "%s\n", n_steps, gen.n_frames, n_text, eog ? ", eog" : "");
    if (!ok) {
        return false;
    }

    if (!write_wav_s16(params.out_path, gen.wav, VV_SAMPLE_RATE)) {
        fprintf(stderr, "failed to write %s\n", params.out_path.c_str());
        return false;
    }
    printf("wrote %s: %zu samples, %d Hz, %s\n", params.out_path.c_str(), gen.wav.size(), VV_SAMPLE_RATE,
            gen.n_nonfinite > 0 ? "non-finite samples replaced by 0" : "all finite");

    return true;
}

static void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s <model.gguf> [-n max_frames] [-o out.wav] [--cfg scale] [--seed n]\n", argv0);
    fprintf(stderr, "       [--tokens 1,2,3 | -p prompt] [--max-tokens n] [--force-audio]\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    cli_params params;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_val = i + 1 < argc;
        if (arg == "-n" && has_val) {
            params.max_frames = std::atoll(argv[++i]);
        } else if (arg == "-o" && has_val) {
            params.out_path = argv[++i];
        } else if (arg == "--cfg" && has_val) {
            params.cfg_scale = std::atof(argv[++i]);
        } else if (arg == "--seed" && has_val) {
            params.seed = (uint32_t) std::atoll(argv[++i]);
        } else if (arg == "--tokens" && has_val) {
            params.tokens = parse_tokens(argv[++i]);
        } else if (arg == "-p" && has_val) {
            params.prompt = argv[++i];
        } else if (arg == "--max-tokens" && has_val) {
            params.max_tokens = std::atoll(argv[++i]);
        } else if (arg == "--force-audio") {
            params.force_audio = true;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (params.max_frames < 1 || params.max_tokens < 1) {
        fprintf(stderr, "-n and --max-tokens must be at least 1\n");
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

    const bool ok = run(lmodel, params);

    llama_model_free(lmodel);
    llama_backend_free();

    return ok ? 0 : 1;
}
