#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <cstring>
#include <string>

/**
 * Please note that this is NOT a production-ready binary.
 * It is a playground for trying TTS support in llama.cpp.
 * For contributors: please keep this code simple and easy to understand. Do not add unnecessary complexity. The goal is to have a simple CLI for testing TTS support.
 */

struct tts_timings {
    int64_t t_start_us = ggml_time_us();
    int64_t t_last_us  = t_start_us;

    void report(int n_frames) {
        const int64_t t_now_us = ggml_time_us();
        if (t_now_us - t_last_us < 2000000) {
            return;
        }
        t_last_us = t_now_us;
        const double t_elapsed_s = (t_now_us - t_start_us) / 1e6;
        const double fps = t_elapsed_s > 0 ? n_frames / t_elapsed_s : 0.0;
        LOG_INF("frames generated: %d, speed: %.2f frames/s\n", n_frames, fps);
    }
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m backbone.gguf -mm mmproj.gguf -p \"text to speak\" -o output.wav", argv[0]);
    LOG("\n    %s -hf user/model -p \"text to speak\" -o output.wav\n", argv[0]);
    LOG("\nnote: --tts-lang and --tts-speaker-file may not be supported in all models");
    LOG("\n      use -n to limit the output length");
    LOG("\n      see tts/README.md for per-model usage notes");
    LOG("\n\n");
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }

    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.prompt.empty()) {
        LOG_ERR("no prompt provided, use -p \"text\"\n");
        return 1;
    }
    if (params.mmproj.path.empty()) {
        LOG_ERR("no mmproj provided, use --mmproj\n");
        return 1;
    }

    // important: keep this file as generic as possible
    //            model-specific logic should be in mtmd-helper-gen or mtmd API

    // always enable embd, so that we can pass hidden states to the audio generation helper
    params.embedding = true;

    llama_backend_init();
    llama_numa_init(params.numa);

    //
    // load backbone model and mmproj
    //

    auto llama_init = common_init_from_params(params);
    llama_model    * model = llama_init->model();
    llama_context  * lctx  = llama_init->context();
    common_sampler * smpl  = llama_init->sampler(0);
    if (!model || !lctx) {
        LOG_ERR("failed to init model/context\n");
        return 1;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = params.mmproj_use_gpu;
    mtmd::context_ptr mctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, mtmd_params));
    if (!mctx) {
        LOG_ERR("failed to load mmproj %s\n", params.mmproj.path.c_str());
        return 1;
    }
    if (mtmd_gen_audio_get_info(mctx.get()).type == MTMD_GEN_AUDIO_TYPE_NONE) {
        LOG_ERR("mmproj does not support audio generation\n");
        return 1;
    }

    //
    // stage 0: process speaker reference file, if any
    //

    mtmd::bitmap_ptr speaker_bitmap;
    if (!params.tts_speaker_file.empty()) {
        auto wrapper = mtmd_helper_bitmap_init_from_file(mctx.get(), params.tts_speaker_file.c_str(), false);
        if (!wrapper.bitmap) {
            LOG_ERR("failed to load speaker file %s\n", params.tts_speaker_file.c_str());
            return 1;
        }
        speaker_bitmap.reset(wrapper.bitmap);
    }

    mtmd_helper::gen_audio gen(lctx, mctx.get());
    mtmd_helper_gen_audio_inp inp{};
    inp.seq_id      = 0;
    inp.prompt      = params.prompt.c_str();
    inp.prompt_len  = params.prompt.size();
    inp.speaker_ref = speaker_bitmap.get();
    inp.lang        = params.tts_lang.c_str();
    inp.top_k       = params.sampling.top_k;
    inp.top_p       = params.sampling.top_p;
    inp.seed        = params.sampling.seed;
    inp.out_type    = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;

    //
    // stage 1: process prompt via backbone model, generate semantic representation
    //

    if (gen.set_input(&inp) != 0) {
        LOG_ERR("set_input failed\n");
        return 1;
    }

    const int64_t t_prompt_start_us = ggml_time_us();

    for (;;) {
        int32_t ret = gen.step_prompt(params.n_batch);
        if (ret < 0) {
            LOG_ERR("prompt processing failed\n");
            return 1;
        }
        if (ret == 0) {
            break;
        }
    }

    // note: some pipelines ignore this token and use the hidden state instead
    auto sample_semantic_code = [&]() -> llama_token {
        llama_token t = common_sampler_sample(smpl, lctx, -1);
        common_sampler_accept(smpl, t, true);
        return t;
    };

    const int max_new = params.n_predict > 0 ? params.n_predict : 512;
    int n_frames = 0;
    llama_token sampled = sample_semantic_code();
    const float * h_state = llama_get_embeddings_ith(lctx, -1);

    tts_timings timings;
    const int64_t t_gen_start_us = ggml_time_us();

    bool stop = false;
    while (!stop && n_frames < max_new) {
        const float * h_next = nullptr;

        // stage 2+3: semantic --> acoustic details --> audio waveform
        //            step_gen() runs both stages and returns new h_state for next step
        if (gen.step_gen(sampled, h_state, &h_next, &stop) != 0) {
            LOG_ERR("step_gen failed at frame %d\n", n_frames);
            return 1;
        }
        if (!h_next) {
            break; // stopped without generating a frame
        }

        n_frames++;
        h_state = h_next;
        sampled = sample_semantic_code();
        timings.report(n_frames);
    }
    const double t_gen_s = (ggml_time_us() - t_gen_start_us) / 1e6;

    int32_t      sample_rate = 0;
    const char * data        = nullptr;
    size_t       data_len    = 0;
    int64_t      n_samples   = 0;
    const int64_t t_wav_start_us = ggml_time_us();
    if (gen.get_output(&sample_rate, &data, &data_len, &n_samples) != 0) {
        LOG_ERR("get_output failed\n");
        return 1;
    }
    const double t_wav_s = (ggml_time_us() - t_wav_start_us) / 1e6;

    LOG_INF("generated %d frames, %zu bytes of WAV audio (%d Hz)\n", n_frames, data_len, sample_rate);

    const double t_prompt_s = (t_gen_start_us - t_prompt_start_us) / 1e6;
    const double t_total_s  = t_prompt_s + t_gen_s + t_wav_s;
    const double audio_s    = sample_rate > 0 ? (double) n_samples / sample_rate : 0.0;
    LOG_INF("timings: prompt eval %.2fs + generation %.2fs + vocoder %.2fs = total %.2fs\n",
            t_prompt_s, t_gen_s, t_wav_s, t_total_s);
    LOG_INF("         output audio = %.2fs (audio time = %.2fx process time)\n", audio_s, t_total_s > 0 ? audio_s / t_total_s : 0.0);
    FILE * f = fopen(params.out_file.c_str(), "wb");
    if (!f) {
        LOG_ERR("failed to open %s\n", params.out_file.c_str());
        return 1;
    }
    fwrite(data, 1, data_len, f);
    fclose(f);
    LOG_INF("wrote %s\n", params.out_file.c_str());

    llama_backend_free();
    return 0;
}
