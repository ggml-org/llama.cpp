#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include "whisper-preprocessor.h"

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "TODO\n\n"
        "Usage: %s [options] -m <model> --mmproj <mmproj> --in-file <image> -p <prompt>\n\n",
        argv[0]
    );
}

struct hook_data {
    std::vector<float> embd;
    int n_token_output;
};

// hook to retrieve the embeddings (because we cannot use arbitrary output tensor **shape**)
static bool ggml_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    hook_data * data = (hook_data *) user_data;

    if (t && strcmp(t->name, "result_embd_pooled") == 0) {
        if (ask) return true;
        data->embd.resize(ggml_nelements(t));
        data->n_token_output = t->ne[0];
        ggml_backend_tensor_get(t, data->embd.data(), 0, ggml_nbytes(t));
        printf("%s tensor size: %lld, %lld\n", t->name, t->ne[0], t->ne[1]);
        return true;
    }

    return false;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;
    params.prompt        = "Transcribe the audio";
    params.sampling.temp = 0.2; // lower temp by default for better quality

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_ASR, show_additional_info)) {
        return 1;
    }

    common_init();

    if (params.mmproj.path.empty()) {
        show_additional_info(argc, argv);
        return 1;
    }

    common_init_result ll_result = common_init_from_params(params);
    llama_model   * ll_model = ll_result.model.get();
    llama_context * ll_ctx   = ll_result.context.get();

    if (!ll_model || !ll_ctx) {
        LOG_ERR("Failed to initialize LLM\n");
        return 1;
    }

    common_params params_enc(params); // copy
    params_enc.model.path = params.mmproj.path;
    params_enc.warmup    = false;
    params_enc.n_ubatch  = 1500;
    params_enc.n_batch   = 1500;
    params_enc.embedding = true;

    hook_data hook_data;
    params_enc.cb_eval           = ggml_callback;
    params_enc.cb_eval_user_data = &hook_data;

    common_init_result enc_result = common_init_from_params(params_enc);
    llama_model   * enc_model = enc_result.model.get();
    llama_context * enc_ctx   = enc_result.context.get();

    if (!enc_model || !enc_ctx) {
        LOG_ERR("Failed to initialize audio encoder model\n");
        return 1;
    }

    // load mel_filters
    whisper_preprocessor::whisper_filters mel_filters;
    {
        ggml_context * meta = nullptr;
        gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };
        gguf_context_ptr ctx_gguf(gguf_init_from_file(params_enc.model.path.c_str(), params));

        // read size
        auto mel_filters_tensor = ggml_get_tensor(meta, "whisper.mel_filters");
        mel_filters.n_mel = mel_filters_tensor->ne[1];
        mel_filters.n_fft = mel_filters_tensor->ne[0];
        mel_filters.data.resize(mel_filters.n_mel * mel_filters.n_fft);

        // read data
        auto idx = gguf_find_tensor(ctx_gguf.get(), "whisper.mel_filters");
        auto offset = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), idx);
        auto size = gguf_get_tensor_size(ctx_gguf.get(), idx);
        auto fin = std::ifstream(params_enc.model.path, std::ios::binary);
        fin.seekg(offset, std::ios::beg);
        fin.read(reinterpret_cast<char *>(mel_filters.data.data()), size);
        fin.close();

        printf("mel_filters: n_mel = %d, n_fft = %d\n", mel_filters.n_mel, mel_filters.n_fft);
        ggml_free(meta);
    }

    // read wav file
    std::vector<float> pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM
    auto fname_inp = params.in_files[0];     // TODO: support multiple files
    if (!wav_utils::read_wav(fname_inp, pcmf32, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read WAV file '%s'\n", fname_inp.c_str());
        return 1;
    }

    // mel spectrogram
    whisper_preprocessor::whisper_mel mel;
    whisper_preprocessor::log_mel_spectrogram(
            pcmf32.data(),
            pcmf32.size(),
            WHISPER_SAMPLE_RATE,
            WHISPER_N_FFT,
            WHISPER_HOP_LENGTH,
            mel_filters.n_mel,
            4, // threads
            mel_filters,
            false,
            mel);
    printf("mel.n_len: %d\n", mel.n_len);
    printf("mel.n_mel: %d\n", mel.n_mel);
    printf("mel.size:  %zu\n", mel.data.size());

    // encode audio
    {
        int n_ctx  = llama_model_n_ctx_train(enc_model);
        int n_embd = llama_model_n_embd(enc_model);
        std::vector<float> embd(2*n_ctx * mel.n_mel, 0.0f);
        // set the input
        {
            int mel_offset = 0;

            const int i0 = std::min(mel_offset,           mel.n_len);
            const int i1 = std::min(mel_offset + 2*n_ctx, mel.n_len);

            for (int j = 0; j < mel.n_mel; ++j) {
                for (int i = i0; i < i1; ++i) {
                    embd[j*2*n_ctx + (i - i0)] = mel.data[j*mel.n_len + i];
                }
            }
        }

        // set the input
        GGML_ASSERT((int)embd.size() < 2*n_ctx * n_embd);
        llama_batch batch_embd = llama_batch_init(n_ctx, n_embd, 1);
        batch_embd.n_tokens = n_ctx;
        for (int i = 0; i < batch_embd.n_tokens; i++) {
            batch_embd.pos[i]       = 0; // dummy, unused
            batch_embd.seq_id[i][0] = 0;
            batch_embd.n_seq_id[i]  = 1;
            batch_embd.logits[i]    = false;
        }
        std::memcpy(batch_embd.embd, embd.data(), embd.size() * sizeof(float));

        if (llama_decode(enc_ctx, batch_embd) != 0) {
            LOG_ERR("%s: audio llama_decode() failed\n", __func__);
            return 1;
        }

        // float * embd_out = hook_data.embd.data();
        // print out the first 10 embeddings
        // for (int i = 0; i < 10; i++) {
        //     printf("embd_out[%d] = %f\n", i, embd_out[i]);
        // }

        llama_batch_free(batch_embd);
    }

    // generate text
    {
        llama_batch batch_token = llama_batch_init(llama_n_ctx(ll_ctx), 0, 1);
        llama_batch batch_embd  = llama_batch_init(hook_data.n_token_output, llama_model_n_embd(ll_model), 1);
        int n_past = 0;

        auto eval_text = [&](std::string text, bool add_bos = false) {
            llama_tokens prompt_tokens = common_tokenize(ll_ctx, text, add_bos, true);
            common_batch_clear(batch_token);
            for (auto & token : prompt_tokens) {
                common_batch_add(batch_token, token, n_past++, {0}, false);
            }
            if (!add_bos) {
                // TODO: a bit hacky here
                batch_token.logits[batch_token.n_tokens - 1] = true;
            }
            if (llama_decode(ll_ctx, batch_token) != 0) {
                LOG_ERR("%s: audio llama_decode() failed\n", __func__);
                exit(1);
            }
        };

        auto eval_embd = [&](std::vector<float> & embd, int n_tokens) {
            batch_embd.n_tokens = n_tokens;
            for (int i = 0; i < n_tokens; i++) {
                batch_embd.pos[i]       = n_past++;
                batch_embd.seq_id[i][0] = 0;
                batch_embd.n_seq_id[i]  = 1;
                batch_embd.logits[i]    = false;
            }
            std::memcpy(batch_embd.embd, embd.data(), embd.size() * sizeof(float));
            if (llama_decode(ll_ctx, batch_embd) != 0) {
                LOG_ERR("%s: audio llama_decode() failed\n", __func__);
                exit(1);
            }
        };

        eval_text("<|start_header_id|>user<|end_header_id|>\n\n" + params.prompt + "\n\n", true);
        eval_embd(hook_data.embd, hook_data.n_token_output);
        eval_text("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");

        struct common_sampler * smpl = common_sampler_init(ll_model, params.sampling);

        int n_predict = 50;
        for (int i = 0; i < n_predict; i++) {
            llama_token token_id = common_sampler_sample(smpl, ll_ctx, -1);
            common_sampler_accept(smpl, token_id, true);
    
            if (llama_vocab_is_eog(llama_model_get_vocab(ll_model), token_id)) {
                printf("\n");
                break; // end of generation
            }
    
            printf("%s", common_token_to_piece(ll_ctx, token_id).c_str());
            fflush(stdout);
    
            // eval the token
            common_batch_clear(batch_token);
            common_batch_add(batch_token, token_id, n_past++, {0}, true);
            if (llama_decode(ll_ctx, batch_token)) {
                LOG_ERR("failed to decode token\n");
                return 1;
            }
        }

        common_sampler_free(smpl);
        llama_batch_free(batch_token);
        llama_batch_free(batch_embd);
    }
}
