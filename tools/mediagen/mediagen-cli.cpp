// llama-mediagen: image, video and audio generation with diffusion models, see README.md

#include "mediagen.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static bool write_file(const std::string & path, const uint8_t * data, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LOG_ERR("failed to open %s for writing\n", path.c_str());
        return false;
    }
    const bool ok = fwrite(data, 1, n, f) == n;
    fclose(f);
    return ok;
}

static bool on_progress(int step, int n_steps, void * ud) {
    (void) ud;
    fprintf(stderr, "\rstep %d/%d", step, n_steps);
    if (step == n_steps) {
        fprintf(stderr, "\n");
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.out_file = "output.png";

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MEDIAGEN)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    const auto & mg = params.mediagen;
    if (mg.vae.path.empty() || mg.text_encoder.path.empty()) {
        LOG_ERR("error: --vae and --text-encoder are required (or use -hf with a diffusion model repository)\n");
        return 1;
    }
    if (params.prompt.empty()) {
        LOG_ERR("error: --prompt is required\n");
        return 1;
    }

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * text_model = llama_model_load_from_file(mg.text_encoder.path.c_str(), mparams);
    if (!text_model) {
        LOG_ERR("failed to load text encoder %s\n", mg.text_encoder.path.c_str());
        return 1;
    }

    mediagen_context_params cparams = mediagen_context_params_default();
    cparams.model_path     = params.model.path.c_str();
    cparams.vae_path       = mg.vae.path.c_str();
    cparams.audio_vae_path = mg.audio_vae.path.empty() ? nullptr : mg.audio_vae.path.c_str();
    cparams.text_proj_path = mg.text_proj.path.empty() ? nullptr : mg.text_proj.path.c_str();
    cparams.text_model     = text_model;
    cparams.use_gpu        = true; // -ngl applies to the text encoder only
    cparams.n_threads      = params.cpuparams.n_threads;
    cparams.flash_attn     = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.verbosity      = params.verbosity >= 3 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;

    mediagen_context * ctx = mediagen_init(cparams);
    if (!ctx) {
        LOG_ERR("failed to initialize mediagen\n");
        llama_model_free(text_model);
        return 1;
    }

    mediagen_gen_params gp = mediagen_gen_params_default();
    gp.prompt          = params.prompt.c_str();
    gp.negative_prompt = mg.negative_prompt.c_str();
    gp.width           = mg.width;
    gp.height          = mg.height;
    gp.n_frames        = mg.n_frames;
    gp.fps             = mg.fps;
    gp.n_steps         = mg.steps;
    gp.cfg_scale       = mg.cfg_scale;
    gp.seed            = params.sampling.seed == LLAMA_DEFAULT_SEED ? UINT32_MAX : params.sampling.seed;
    gp.gen_audio       = !mg.audio_vae.path.empty();
    gp.enhance_prompt  = mg.enhance_prompt;
    gp.progress        = on_progress;

    mediagen_result * res = mediagen_generate(ctx, &gp);
    if (!res) {
        LOG_ERR("generation failed\n");
        mediagen_free(ctx);
        llama_model_free(text_model);
        return 1;
    }

    if (gp.enhance_prompt && res->revised_prompt) {
        LOG("enhanced prompt: %s\n", res->revised_prompt);
    }

    int ret = 0;
    const size_t frame_bytes = (size_t) res->width * res->height * 3;
    if (res->n_frames == 1) {
        size_t n = 0;
        uint8_t * png = mediagen_encode_png(res->rgb, res->width, res->height, &n);
        if (!png || !write_file(params.out_file, png, n)) {
            ret = 1;
        } else {
            LOG("saved %s\n", params.out_file.c_str());
        }
        free(png);
    } else if (string_ends_with(params.out_file, ".mp4")) {
        size_t n = 0;
        uint8_t * mp4 = mediagen_encode_mp4(res, &n);
        if (!mp4) {
            LOG_ERR("mp4 output needs the ffmpeg binary in PATH\n");
            ret = 1;
        } else if (!write_file(params.out_file, mp4, n)) {
            ret = 1;
        } else {
            LOG("saved %s (%d frames%s)\n", params.out_file.c_str(), res->n_frames, res->pcm ? ", with audio" : "");
        }
        free(mp4);
    } else {
        std::string base = params.out_file;
        if (string_ends_with(base, ".png")) {
            base.resize(base.size() - 4);
        }
        for (int f = 0; f < res->n_frames && ret == 0; f++) {
            size_t n = 0;
            uint8_t * png = mediagen_encode_png(res->rgb + f * frame_bytes, res->width, res->height, &n);
            char name[1024];
            snprintf(name, sizeof(name), "%s-%04d.png", base.c_str(), f);
            if (!png || !write_file(name, png, n)) {
                ret = 1;
            }
            free(png);
        }
        LOG("saved %d frames to %s-NNNN.png\n", res->n_frames, base.c_str());
        if (res->pcm) {
            size_t n = 0;
            uint8_t * wav = mediagen_encode_wav(res->pcm, res->n_samples, res->n_channels, res->sample_rate, &n);
            if (wav && write_file(base + ".wav", wav, n)) {
                LOG("saved %s.wav\n", base.c_str());
            }
            free(wav);
        }
    }

    mediagen_result_free(res);
    mediagen_free(ctx);
    llama_model_free(text_model);
    llama_backend_free();
    return ret;
}
