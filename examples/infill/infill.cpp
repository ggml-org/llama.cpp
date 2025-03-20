#include "arg.h"
#include "common.h"
#include "console.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;

static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*g_ctx, *g_smpl);

            // make sure all logs are flushed
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    common_params params;
    g_params = &params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_INFILL)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sampling;

    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.logits_all) {
        LOG_ERR("\n************\n");
        LOG_ERR("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.embedding) {
        LOG_ERR("\n************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (!params.interactive_first && (params.input_prefix.empty() && params.input_suffix.empty())) {
        LOG_ERR("\n************\n");
        LOG_ERR("%s: please use '--interactive_first' or specify '--in_prefix' and/or '--in_suffix'\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_INF("%s: llama backend init\n", __func__);
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;

    g_model = &model;
    g_ctx = &ctx;
    g_smpl = &smpl;

    // load the model and apply lora adapter, if any
    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result llama_init = common_init_from_params(params);

    model = llama_init.model.get();
    ctx = llama_init.context.get();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);
    LOG_DBG("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    std::vector<llama_token> embd_inp;
    std::vector<llama_token> embd_end;
    std::vector<llama_token> inp_pfx = common_tokenize(ctx, params.input_prefix, false);
    std::vector<llama_token> inp_sfx = common_tokenize(ctx, params.input_suffix, false);

    GGML_ASSERT(llama_vocab_fim_pre(vocab) >= 0);
    GGML_ASSERT(llama_vocab_fim_suf(vocab) >= 0);

    inp_pfx.insert(inp_pfx.begin(), llama_vocab_fim_pre(vocab));
    inp_sfx.insert(inp_sfx.begin(), llama_vocab_fim_suf(vocab));

    embd_inp = params.spm_infill ? inp_sfx : inp_pfx;
    embd_end = params.spm_infill ? inp_pfx : inp_sfx;
    if (add_bos) {
        embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
    }
    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());

    const llama_token middle_token = llama_vocab_fim_mid(vocab);
    if (middle_token >= 0) {
        embd_inp.push_back(middle_token);
    }

    LOG_DBG("add_bos: %d\n", add_bos);
    LOG_DBG("prefix: \"%s\"\n", params.input_prefix.c_str());
    LOG_DBG("suffix: \"%s\"\n", params.input_suffix.c_str());
    LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_vocab_bos(vocab));
        LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
    }

    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    }

    LOG_INF("inp_pfx: %s\n", string_from(ctx, inp_pfx).c_str());
    LOG_INF("inp_sfx: %s\n", string_from(ctx, inp_sfx).c_str());

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_INF("\n");
        LOG_INF("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > 0) {
        LOG_INF("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_INF("\n");
    }

    if (params.interactive) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

        LOG_INF("%s: interactive mode on.\n", __func__);

        if (params.input_prefix_bos) {
            LOG_INF("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_INF("Input prefix: '%s'\n", params.input_prefix.c_str());
        }

        if (!params.input_suffix.empty()) {
            LOG_INF("Input suffix: '%s'\n", params.input_suffix.c_str());
        }
    }
    smpl = common_sampler_init(model, sparams);

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl));
    LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl).c_str());

    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    LOG_INF("\n");
    LOG_INF("\n#####  Infill mode  #####\n\n");
    if (params.interactive) {
        const char *control_message;
        if (params.multiline_input) {
            control_message = " - To return control to LLaMA, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to LLaMA.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_INF("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_INF(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_INF(       "%s\n", control_message);

        is_interacting = params.interactive_first;
    }

    bool input_echo = true;

    int n_past     = 0;
    int n_remain   = params.n_predict;
    int n_consumed = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;

    // the first thing we will do is to output the prompt, so set color accordingly
    console::set_display(console::prompt);

    std::vector<llama_token> embd;

    while (n_remain != 0 || params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                console::set_display(console::error);
                LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(console::reset);
            }

            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() > n_ctx) {
                if (params.n_predict == -2) {
                    LOG_DBG("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                    break;
                }

                const int n_left    = n_past - params.n_keep - 1;
                const int n_discard = n_left/2;

                LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                    n_past, n_left, n_ctx, params.n_keep, n_discard);

                llama_kv_self_seq_rm (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
                llama_kv_self_seq_add(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

                n_past -= n_discard;

                LOG_DBG("after swap: n_past = %d\n", n_past);

                LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());

            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                    LOG_ERR("%s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                LOG_DBG("n_past = %d\n", n_past);
            }

        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, true);

            // LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG_DBG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id);
                LOG("%s", token_str.c_str());

                if (embd.size() > 1) {
                    input_tokens.push_back(id);
                } else {
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }
        // reset color to default if we there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(console::reset);
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // deal with eot token in infill mode
            if ((common_sampler_last(smpl) == llama_vocab_eot(vocab) || is_interacting) && params.interactive){
                if (is_interacting && !params.interactive_first) {
                    // print an eot token
                    LOG("%s", common_token_to_piece(ctx, llama_vocab_eot(vocab)).c_str());
                }
                LOG("\n");
                console::set_display(console::user_input);
                std::string buffer;
                std::string line;
                bool another_line=true;
                // set a new prefix via stdin
                do {
                    another_line = console::readline(line, params.multiline_input);
                    buffer += line;
                } while (another_line);
                // check if we got an empty line, if so we use the old input
                if (!buffer.empty() && !(buffer.length() == 1 && buffer[0] == '\n')) {
                    params.input_prefix = buffer;
                }
                buffer.clear();
                // set a new suffix via stdin
                do {
                    another_line = console::readline(line, params.multiline_input);
                    buffer += line;
                } while (another_line);
                // check if we got an empty line
                if (!buffer.empty() && !(buffer.length() == 1 && buffer[0] == '\n')) {
                    params.input_suffix = buffer;
                }
                buffer.clear();
                // done taking input, reset color
                console::set_display(console::reset);

                if (params.escape) {
                    //process escape sequences, for the initial prompt this is done in common.cpp when we load the params, but for the interactive mode we need to do it here
                    string_process_escapes(params.input_prefix);
                    string_process_escapes(params.input_suffix);
                }

                // tokenize new prefix and suffix
                std::vector<llama_token> inp_pfx = common_tokenize(ctx, params.input_prefix, false);
                std::vector<llama_token> inp_sfx = common_tokenize(ctx, params.input_suffix, false);

                inp_pfx.insert(inp_pfx.begin(), llama_vocab_fim_pre(vocab));
                inp_sfx.insert(inp_sfx.begin(), llama_vocab_fim_suf(vocab));

                embd_inp = params.spm_infill ? inp_sfx : inp_pfx;
                embd_end = params.spm_infill ? inp_pfx : inp_sfx;
                if (add_bos) {
                    embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
                }
                embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());

                if (middle_token >= 0) {
                    embd_inp.push_back(middle_token);
                }

                embd.clear();
                n_remain = params.n_predict;
                n_past = 0;
                n_consumed = 0;
                is_interacting = false;
            }
            // deal with end of generation tokens in interactive mode
            else if (llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
                LOG_DBG("found EOS token\n");

                if (params.interactive) {

                    is_interacting = true;
                    LOG("\n");
                    console::set_display(console::user_input);
               }
            }

            if (n_past > 0 && is_interacting && !params.interactive) {
                LOG_DBG("waiting for user input\n");

                if (params.input_prefix_bos) {
                    LOG_DBG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_vocab_bos(vocab));
                }

                std::string buffer;
                if (!params.input_prefix.empty()) {
                    LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    buffer += params.input_prefix;
                    LOG("%s", buffer.c_str());
                }

                std::string line;
                bool another_line = true;
                do {
                    another_line = console::readline(line, params.multiline_input);
                    buffer += line;
                } while (another_line);

                // done taking input, reset color
                console::set_display(console::reset);

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty()) {
                        LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        buffer += params.input_suffix;
                        LOG("%s", params.input_suffix.c_str());
                    }

                    LOG_DBG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    const auto line_inp = common_tokenize(ctx, buffer, false);
                    LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        output_tokens.push_back(token);
                        output_ss << common_token_to_piece(ctx, token);
                    }

                    n_remain -= line_inp.size();
                    LOG_DBG("n_remain: %d\n", n_remain);
                } else {
                    LOG_DBG("empty line, passing control back\n");
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;
            }
        }

        // end of generation
        if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !params.interactive) {
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }
    if (!params.interactive && n_remain <= 0) {
        LOG("%s", common_token_to_piece(ctx, llama_vocab_eot(vocab)).c_str());
    }

    LOG("\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);
    llama_backend_free();

    return 0;
}
