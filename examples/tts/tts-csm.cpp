#include "ggml.h"
#include "llama.h"
#include "common.h"
#include "log.h"
#include "arg.h"
#include "mimi-model.h"
#include "tts-csm-data.h"

#include <initializer_list>
#include <vector>
#include <regex>
#include <fstream>
#include <float.h>
#include <cstring> // memcpy and strcmp
#include <inttypes.h>

// For more details on how this works, see: https://github.com/ggml-org/llama.cpp/pull/12648

static void print_usage(int, char ** argv) {
    LOG("\nExample usage:\n");
    LOG("\n    By default, model will be downloaded from https://huggingface.co/ggml-org/sesame-csm-1b-GGUF");
    LOG("\n    %s -p \"[0]I have a dream that one day every valley shall be exalted\" -o output.wav", argv[0]);
    LOG("\n");
    LOG("\n    To use a local model, specify the path to the model file:");
    LOG("\n    %s -p ... -m sesame-csm-backbone.gguf -mv kyutai-mimi.gguf -o output.wav", argv[0]);
    LOG("\n");
    LOG("\n    Note: the model need 2 files to run, one ends with '-backbone-<quant>.gguf' and the other ends with '-decoder<quant>.gguf'");
    LOG("\n");
    LOG("\nPrompt format:");
    LOG("\n    Each line must start with speaker ID in square brackets, followed by the text. One turn per line. A full stop is recommended at the end of each turn");
    LOG("\n    Example:");
    LOG("\n        [0]Hey how are you doing.");
    LOG("\n        [1]Pretty good, pretty good.");
    LOG("\n    If you want to enter long text, use -f file.txt to read from file");
    LOG("\n");
}

struct speaker_turn {
    std::string text;
    std::vector<float> audio_embd; // only used for system prompt (speaker reference) processing
    size_t n_embd_tokens = 0;
};

// split text containing "[N]..." into speaker turns
static std::vector<speaker_turn> get_speaker_turns(const std::string & input) {
    if (input.empty()) {
        LOG_ERR("Empty input\n");
        return {};
    }
    if (input[0] != '[') {
        LOG_ERR("Invalid input format: missing speaker ID\n");
        return {};
    }
    std::regex re(R"((\[\d+\][\s\S]*?)(?=\[\d+\]|$))");
    std::smatch match;
    std::vector<speaker_turn> turns;
    std::string::const_iterator searchStart(input.cbegin());
    while (std::regex_search(searchStart, input.cend(), match, re)) {
        std::string turn_text = match[1].str();
        if (turn_text.empty()) {
            continue;
        }
        // clean up newline, the model is quite sensitive to this
        string_replace_all(turn_text, "\n", " ");
        turn_text = string_strip(turn_text);
        // add turn
        speaker_turn turn;
        turn.text = turn_text;
        turns.push_back(turn);
        searchStart = match.suffix().first;
    }
    return turns;
}

static speaker_turn get_ref_speaker_turn(const char * text, std::initializer_list<int> & codes, std::vector<float> & codebook) {
    const size_t n_embd = 2048;
    const size_t n_codes_per_codebook = 2051;
    const size_t n_codebooks = 32;
    GGML_ASSERT(codebook.size() == n_embd * n_codes_per_codebook * n_codebooks);
    GGML_ASSERT(codes.size() % 32 == 0);

    // 1 frame = 32 codes
    size_t n_frames = codes.size() / n_codebooks;
    speaker_turn turn;
    turn.text = text;
    turn.audio_embd.reserve((n_frames+1) * n_embd);
    turn.n_embd_tokens = n_frames+1; // +1 for EOS frame

    for (size_t i_fr = 0; i_fr <= n_frames; i_fr++) {
        std::vector<float> frame_embd_sum(n_embd, 0.0f);

        for (size_t i_cb = 0; i_cb < n_codebooks; i_cb++) {
            const size_t code = i_fr == n_frames
                ? 0 // insert audio EOS for last pseudo-frame
                : codes.begin()[i_fr*n_codebooks + i_cb];
            printf("code %zu: %zu, codebook entry %zu\n", i_cb, code, i_cb*n_codes_per_codebook + code);
            float * entry = codebook.data() + i_cb*n_codes_per_codebook*n_embd + code*n_embd;
            for (size_t i_embd = 0; i_embd < n_embd; i_embd++) {
                frame_embd_sum[i_embd] += entry[i_embd];
            }
        }

        turn.audio_embd.insert(turn.audio_embd.end(), frame_embd_sum.begin(), frame_embd_sum.end());
    }

    GGML_ASSERT(turn.audio_embd.size() == (n_frames+1) * n_embd);
    return turn;
}

// sampling with custom n_vocab
// modified version of llama_sampler_sample()
static llama_token sample_token(struct llama_sampler * smpl, const float * logits, int n_vocab) {
    std::vector<llama_token_data> cur;
    cur.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array cur_p = {
        /* .data       = */ cur.data(),
        /* .size       = */ cur.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };

    llama_sampler_apply(smpl, &cur_p);
    GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size);
    auto token = cur_p.data[cur_p.selected].id;
    llama_sampler_accept(smpl, token);
    return token;
}

struct hook_data {
    std::vector<float> embd;
    std::vector<float> codebook;
};

// hook to retrieve the embeddings
static bool ggml_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    hook_data * data = (hook_data *) user_data;

    // output_csm_proj is the embeddings output from backbone
    // output_audio_embd is the embeddings output from decoder
    if (t && (strcmp(t->name, "output_csm_proj") == 0 || strcmp(t->name, "output_audio_embd") == 0)) {
        if (ask) return true;

        GGML_ASSERT(t->type == GGML_TYPE_F32);
        data->embd.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, data->embd.data(), 0, ggml_nbytes(t));
        // printf("%s tensor size: %lld, %lld\n", t->name, t->ne[0], t->ne[1]);
        return true;
    }

    if (t && strncmp(t->name, "audio_embd.weight", 18) == 0) {
        if (ask) return true;

        printf("%s tensor size: %lld, %lld\n", t->name, t->ne[0], t->ne[1]);
        GGML_ASSERT(t->type == GGML_TYPE_F32);
        GGML_ASSERT(t->ne[0] == 2048); // backbone embd size
        data->codebook.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, data->codebook.data(), 0, ggml_nbytes(t));
        return true;
    }

    return false;
}

// convenience wrapper around llama_batch to handle memory allocation
struct decode_embd_batch {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos     .resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0] = seq_id;
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }
};

int main(int argc, char ** argv) {
    common_params params;

    params.model.path         = "sesame-csm-backbone.gguf";
    params.vocoder.model.path = "kyutai-mimi.gguf";
    params.out_file           = "output.wav";
    params.prompt             = "";
    params.n_predict          = 2048; // CSM's max trained seq length
    params.sampling.top_k     = 50;   // default param from CSM python code
    params.sampling.temp      = 0.9;  // default param from CSM python code

    // HF model (hack: we temporary reuse speculative.model as the decoder model, only to get it downloaded)
    params.model.url              = "https://huggingface.co/ggml-org/sesame-csm-1b-GGUF/resolve/main/sesame-csm-backbone.gguf";
    params.speculative.model.path = "sesame-csm-decoder.gguf";
    params.speculative.model.url  = "https://huggingface.co/ggml-org/sesame-csm-1b-GGUF/resolve/main/sesame-csm-decoder.gguf";
    params.vocoder.model.url      = "https://huggingface.co/ggml-org/sesame-csm-1b-GGUF/resolve/main/kyutai-mimi.gguf";

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    if (params.prompt.empty()) {
        LOG_ERR("prompt is empty\n");
        return 1;
    }

    hook_data cb_data;
    params.cb_eval = ggml_callback;
    params.cb_eval_user_data = &cb_data;

    common_params params_decoder(params); // duplicate the params
    params_decoder.n_ctx = 64; // we never use more than this
    string_replace_all(params_decoder.model.path, "-backbone", "-decoder");
    string_replace_all(params_decoder.model.url,  "-backbone", "-decoder");

    common_init_result llama_backbone = common_init_from_params(params);
    llama_model   * model_bb = llama_backbone.model.get();
    llama_context * ctx_bb   = llama_backbone.context.get();

    common_init_result llama_decoder  = common_init_from_params(params_decoder);
    llama_model   * model_dc = llama_decoder.model.get();
    llama_context * ctx_dc   = llama_decoder.context.get();

    if (model_bb == nullptr || ctx_bb == nullptr) {
        return ENOENT;
    }

    if (model_dc == nullptr || ctx_dc == nullptr) {
        return ENOENT;
    }

    mimi_model mimi(params.vocoder.model.path.c_str(), true);

    // init sampler
    // the python implementation only has top-k and temperature sampling, so we'll use just that
    llama_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(params.sampling.top_k));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(params.sampling.temp));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(params.sampling.seed));

    llama_batch batch_prompt = llama_batch_init(params.n_batch, 0, 1);
    llama_pos n_past_bb = 0;

    // inp_past_embd is the "squashed" embeddings from the decoder
    std::vector<float> inp_past_embd(2048, 0.0f);
    llama_batch batch_past_embd = llama_batch_init(1, inp_past_embd.size(), 1);

    int64_t t_gb_start = ggml_time_ms(); // global start time
    int64_t t_bb       = 0; // backbone time
    int64_t n_bb_gen   = 0; // backbone generation count
    int64_t t_dc       = 0; // decoder time
    int64_t n_dc_gen   = 0; // decoder generation count

    std::vector<int> generated_codes;

    std::vector<speaker_turn> turns;
    // speaker reference
    turns.push_back(get_ref_speaker_turn(default_speaker_a_text, default_speaker_a_codes, cb_data.codebook));
    turns.push_back(get_ref_speaker_turn(default_speaker_b_text, default_speaker_b_codes, cb_data.codebook));

    // user input
    auto custom_turns = get_speaker_turns(params.prompt);
    turns.insert(turns.end(), custom_turns.begin(), custom_turns.end());

    for (speaker_turn & turn : turns) {
        // tokenize the turn
        llama_tokens prompt_tokens;
        {
            printf("\n---\n\nturn: %s\n\n", turn.text.c_str());
            const llama_vocab * vocab = llama_model_get_vocab(model_bb);
            prompt_tokens = common_tokenize(vocab, turn.text, false, true);
            prompt_tokens.insert(prompt_tokens.begin(), llama_vocab_bos(vocab));
            prompt_tokens.insert(prompt_tokens.end(),   llama_vocab_eos(vocab));

            printf("prompt (%zu tokens): \n", prompt_tokens.size());
            for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                printf("%d, ", prompt_tokens[i]);
            }
            printf("\n\n");

            common_batch_clear(batch_prompt);
            for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                common_batch_add(batch_prompt, prompt_tokens[i], n_past_bb++, { 0 }, false);
            }
            batch_prompt.logits[batch_prompt.n_tokens - 1] = true;

            if (llama_decode(ctx_bb, batch_prompt) != 0) {
                LOG_ERR("%s: backbone llama_decode(text) failed\n", __func__);
                return 1;
            }
        }

        // optionally process the system prompt (speaker reference)
        if (turn.n_embd_tokens) {
            decode_embd_batch batch_embd(turn.audio_embd.data(), turn.n_embd_tokens, n_past_bb, 0);
            if (llama_decode(ctx_bb, batch_embd.batch) != 0) {
                LOG_ERR("%s: backbone llama_decode(embeddings) failed\n", __func__);
                return 1;
            }
            LOG_INF("%s: backbone done decoding %zu audio codes\n\n", __func__, turn.n_embd_tokens);
            n_past_bb += turn.n_embd_tokens;
            continue; // no need to generate the audio
        }

        // backbone generation loop
        bool is_end_of_turn = false;
        for (int k = 0; k < params.n_predict; ++k) {
            bool is_first_tok = k == 0;

            if (!is_first_tok) {
                // generate the next RVQ semantic token
                batch_past_embd.n_tokens     = 1;
                batch_past_embd.pos[0]       = n_past_bb++;
                batch_past_embd.seq_id[0][0] = 0;
                batch_past_embd.n_seq_id[0]  = 1;
                batch_past_embd.logits[0]    = true;
                std::memcpy(batch_past_embd.embd, inp_past_embd.data(), inp_past_embd.size() * sizeof(float));

                int64_t t_bb_start = ggml_time_ms();
                if (llama_decode(ctx_bb, batch_past_embd) != 0) {
                    LOG_ERR("%s: backbone llama_decode() failed\n", __func__);
                    return 1;
                }
                n_bb_gen++;
                t_bb += ggml_time_ms() - t_bb_start;
            }

            if (is_end_of_turn) {
                // done decoding audio's EOS token
                break;
            }

            auto vocab_dc = llama_model_get_vocab(model_dc);
            auto logits   = llama_get_logits_ith(ctx_bb, is_first_tok ? (batch_prompt.n_tokens - 1) : 0);
            // for (size_t i = 0; i < 10; ++i) {
            //     printf("%4.2f, ", logits[i]);
            // }
            // printf("\n");

            llama_token semantic_tok = sample_token(sampler.get(), logits, llama_vocab_n_tokens(vocab_dc));
            printf("Sem token %5d : %d,", 1+(int)generated_codes.size()/32, semantic_tok);
            generated_codes.push_back(semantic_tok);

            // for (size_t i = 0; i < 10; ++i) {
            //     printf("%4.2f, ", embd[i]);
            // }
            // printf("\n");


            // decoder generation loop
            inp_past_embd = std::vector<float>(inp_past_embd.size(), 0.0f);
            {
                llama_kv_self_clear(ctx_dc);
                llama_batch batch_embd  = llama_batch_init(1, cb_data.embd.size(), 1);
                llama_batch batch_token = llama_batch_init(1, 0, 1);

                // first "token" is the latent embeddings from backbone
                {
                    batch_embd.n_tokens     = 1;
                    batch_embd.pos[0]       = 0;
                    batch_embd.seq_id[0][0] = 0;
                    batch_embd.n_seq_id[0]  = 1;
                    batch_embd.logits[0]    = false;
                    std::memcpy(batch_embd.embd, cb_data.embd.data(), cb_data.embd.size() * sizeof(float));
                }
                if (llama_decode(ctx_dc, batch_embd) != 0) {
                    LOG_ERR("%s: decoder llama_decode(embd) failed\n", __func__);
                    return 1;
                }

                // then, decode the semantic_tok to generate acoustic tokens
                llama_token tok = semantic_tok;
                int n_codes = 32;
                int sum_codes = semantic_tok; // to check if all codes are 0
                for (int i = 0; i < n_codes; ++i) {
                    common_batch_clear(batch_token);
                    // encoder vocab is further divided into 32 codebooks, each with 2051 entries
                    llama_token inp_tok = tok + 2051*i;
                    common_batch_add(batch_token, inp_tok, i+1, { 0 }, true);

                    int64_t t_bb_start = ggml_time_ms();
                    if (llama_decode(ctx_dc, batch_token) != 0) {
                        LOG_ERR("%s: decoder llama_decode(token) failed\n", __func__);
                        return 1;
                    }
                    n_dc_gen++;
                    t_dc += ggml_time_ms() - t_bb_start;

                    // sample the acoustic token
                    auto logits = llama_get_logits_ith(ctx_dc, 0);
                    llama_token acoustic_tok = sample_token(sampler.get(), logits, llama_vocab_n_tokens(vocab_dc));

                    // discard last code (only for embeddings)
                    if (i < n_codes - 1) {
                        printf("%d,", acoustic_tok);
                        tok = acoustic_tok; // next input token
                        sum_codes += acoustic_tok;
                        generated_codes.push_back(acoustic_tok);
                    }

                    // do progressive hsum of embeddings
                    GGML_ASSERT(inp_past_embd.size() == cb_data.embd.size());
                    for (size_t i = 0; i < inp_past_embd.size(); ++i) {
                        inp_past_embd[i] += cb_data.embd[i];
                    }
                }
                printf("\n");

                llama_batch_free(batch_embd);
                llama_batch_free(batch_token);

                // if all codes are 0, then we are done (got audio EOS token)
                // note: we still need to run backbone decode one more time to decode the audio's EOS token
                is_end_of_turn = sum_codes == 0;
                if (is_end_of_turn) {
                    // remove last 32 codes since they will be all zeros
                    generated_codes.resize(generated_codes.size() - 32);
                }
            }

            // printf("inp_past_embd, n_past_bb = %d\n", n_past_bb);
            // for (size_t i = 0; i < inp_past_embd.size(); ++i) {
            //     printf("%4.4f, ", inp_past_embd[i]);
            //     if (i == 2) {
            //         printf("... ");
            //         i = inp_past_embd.size() - 4;
            //     }
            // }
            // printf("\n");
        }
    }

    // print timing info
    printf("\ntimings:\n");
    printf("  backbone: %" PRId64 " ms, %" PRId64 " generated token (%.2f tok/s)\n", t_bb, n_bb_gen, (float)n_bb_gen*1000/(float)t_bb);
    printf("  decoder:  %" PRId64 " ms, %" PRId64 " generated token (%.2f tok/s)\n", t_dc, n_dc_gen, (float)n_dc_gen*1000/(float)t_dc);
    printf("  total:    %" PRId64 " ms\n\n", ggml_time_ms() - t_gb_start);

    llama_batch_free(batch_prompt);
    llama_batch_free(batch_past_embd);

    printf("decode %zu RVQ tokens into wav...\n", generated_codes.size());
    std::vector<float> wav_data = mimi.decode(generated_codes);

    printf("output wav file: %s\n", params.out_file.c_str());

    if (!save_wav16(params.out_file.c_str(), wav_data, mimi.get_sample_rate())) {
        LOG_ERR("Failed to save wav file\n");
        return 1;
    }

    printf("\n");

    return 0;
}
