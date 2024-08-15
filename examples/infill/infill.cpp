#include "common.h"

#include "console.h"
#include "llama.h"
#include "grammar-parser.h"

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
static gpt_params               * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;

static bool is_interacting = false;

static void write_logfile(
    const llama_context * ctx, const gpt_params & params, const llama_model * model,
    const std::vector<llama_token> & input_tokens, const std::string & output,
    const std::vector<llama_token> & output_tokens
) {
    if (params.logdir.empty()) {
        return;
    }

    const std::string timestamp = string_get_sortable_timestamp();

    const bool success = fs_create_directory_with_parents(params.logdir);
    if (!success) {
        fprintf(stderr, "%s: warning: failed to create logdir %s, cannot write logfile\n",
                __func__, params.logdir.c_str());
        return;
    }

    const std::string logfile_path = params.logdir + timestamp + ".yml";
    FILE * logfile = fopen(logfile_path.c_str(), "w");

    if (logfile == NULL) {
        fprintf(stderr, "%s: failed to open logfile %s\n", __func__, logfile_path.c_str());
        return;
    }

    fprintf(logfile, "binary: infill\n");
    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    yaml_dump_non_result_info(logfile, params, ctx, timestamp, input_tokens, model_desc);

    fprintf(logfile, "\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "# Generation Results #\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "\n");

    yaml_dump_string_multiline(logfile, "output", output.c_str());
    yaml_dump_vector_int(logfile, "output_tokens", output_tokens);

    llama_dump_timing_info_yaml(logfile, ctx);
    fclose(logfile);
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting = true;
        } else {
            console::cleanup();
            printf("\n");
            llama_print_timings(*g_ctx);
            write_logfile(*g_ctx, *g_params, *g_model, *g_input_tokens, g_output_ss->str(), *g_output_tokens);
            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    gpt_params params;
    llama_sampling_params & sparams = params.sparams;
    g_params = &params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("infill", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.logits_all) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }
    if (!params.interactive_first && (params.input_prefix.empty() && params.input_suffix.empty())) {
        printf("\n************\n");
        printf("%s: please use '--interactive_first' or specify '--in_prefix' and/or '--in_suffix'\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_TEE("%s: build = %d (%s)\n",      __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG_TEE("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    LOG_TEE("%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);

    LOG("%s: llama backend init\n", __func__);
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model;
    llama_context * ctx;

    g_model = &model;
    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    llama_init_result llama_init = llama_init_from_gpt_params(params);

    model = llama_init.model;
    ctx = llama_init.context;

    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train) {
        LOG_TEE("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG_TEE("\n");
        LOG_TEE("%s\n", gpt_params_get_system_info(params).c_str());
    }
    const bool add_bos = llama_add_bos_token(model);
    GGML_ASSERT(!llama_add_eos_token(model));
    LOG("add_bos: %d\n", add_bos);

    std::vector<llama_token> embd_inp;
    std::vector<llama_token> embd_end;
    std::vector<llama_token> inp_pfx = ::llama_tokenize(ctx, params.input_prefix, false);
    std::vector<llama_token> inp_sfx = ::llama_tokenize(ctx, params.input_suffix, false);

    GGML_ASSERT(llama_token_prefix(model) >= 0);
    GGML_ASSERT(llama_token_suffix(model) >= 0);

    inp_pfx.insert(inp_pfx.begin(), llama_token_prefix(model));
    inp_sfx.insert(inp_sfx.begin(), llama_token_suffix(model));

    embd_inp = params.spm_infill ? inp_sfx : inp_pfx;
    embd_end = params.spm_infill ? inp_pfx : inp_sfx;
    if (add_bos) {
        embd_inp.insert(embd_inp.begin(), llama_token_bos(model));
    }
    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());

    const llama_token middle_token = llama_token_middle(model);
    if (middle_token >= 0) {
        embd_inp.push_back(middle_token);
    }

    LOG("prefix: \"%s\"\n", log_tostr(params.input_prefix));
    LOG("suffix: \"%s\"\n", log_tostr(params.input_suffix));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(model));
        LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
    }

    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    }

    LOG("inp_pfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_pfx).c_str());
    LOG("inp_sfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_sfx).c_str());


    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_TEE("\n");
        LOG_TEE("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_TEE("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > 0) {
        LOG_TEE("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_TEE("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_TEE("'\n");
        }
        LOG_TEE("\n");
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

        LOG_TEE("%s: interactive mode on.\n", __func__);

        if (params.input_prefix_bos) {
            LOG_TEE("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_TEE("Input prefix: '%s'\n", params.input_prefix.c_str());
        }

        if (!params.input_suffix.empty()) {
            LOG_TEE("Input suffix: '%s'\n", params.input_suffix.c_str());
        }
    }
    LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    LOG_TEE("\n\n");

    LOG_TEE("\n#####  Infill mode  #####\n\n");
    if (params.infill) {
        printf("\n************\n");
        printf("no need to specify '--infill', always running infill\n");
        printf("************\n\n");
    }
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
        LOG_TEE("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_TEE(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_TEE(       "%s\n", control_message);

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

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(sparams);

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
                printf("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(console::reset);
                fflush(stdout);
            }

            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() > n_ctx) {
                if (params.n_predict == -2) {
                    LOG_TEE("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                    break;
                }

                const int n_left    = n_past - params.n_keep - 1;
                const int n_discard = n_left/2;

                LOG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                    n_past, n_left, n_ctx, params.n_keep, n_discard);

                llama_kv_cache_seq_rm (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
                llama_kv_cache_seq_add(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

                n_past -= n_discard;

                LOG("after swap: n_past = %d\n", n_past);

                LOG("embd: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                    LOG_TEE("%s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                LOG("n_past = %d\n", n_past);
            }

        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            const llama_token id = llama_sampling_sample(ctx_sampling, ctx, nullptr);

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());

            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo) {
            for (auto id : embd) {
                const std::string token_str = llama_token_to_piece(ctx, id);
                printf("%s", token_str.c_str());

                if (embd.size() > 1) {
                    input_tokens.push_back(id);
                } else {
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
            fflush(stdout);
        }
        // reset color to default if we there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(console::reset);
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // deal with eot token in infill mode
            if ((llama_sampling_last(ctx_sampling) == llama_token_eot(model) || is_interacting) && params.interactive){
                if (is_interacting && !params.interactive_first) {
                    // print an eot token
                    printf("%s", llama_token_to_piece(ctx, llama_token_eot(model)).c_str());
                }
                fflush(stdout);
                printf("\n");
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
                std::vector<llama_token> inp_pfx = ::llama_tokenize(ctx, params.input_prefix, false);
                std::vector<llama_token> inp_sfx = ::llama_tokenize(ctx, params.input_suffix, false);

                inp_pfx.insert(inp_pfx.begin(), llama_token_prefix(model));
                inp_sfx.insert(inp_sfx.begin(), llama_token_suffix(model));

                embd_inp = params.spm_infill ? inp_sfx : inp_pfx;
                embd_end = params.spm_infill ? inp_pfx : inp_sfx;
                if (add_bos) {
                    embd_inp.insert(embd_inp.begin(), llama_token_bos(model));
                }
                embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());

                if (middle_token >= 0) {
                    embd_inp.push_back(middle_token);
                }

                embd.clear();
                n_remain = params.n_predict;
                n_past = 0;
                n_consumed = 0;
                // LOG_TEE("took new input\n");
                is_interacting = false;
            }
            // deal with end of generation tokens in interactive mode
            else if (llama_token_is_eog(model, llama_sampling_last(ctx_sampling))) {
                LOG("found EOS token\n");

                if (params.interactive) {

                    is_interacting = true;
                    printf("\n");
                    console::set_display(console::user_input);
                    fflush(stdout);
               }
            }

            if (n_past > 0 && is_interacting && !params.interactive) {
                LOG("waiting for user input\n");

                if (params.input_prefix_bos) {
                    LOG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_token_bos(model));
                }

                std::string buffer;
                if (!params.input_prefix.empty()) {
                    LOG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    buffer += params.input_prefix;
                    printf("%s", buffer.c_str());
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
                        LOG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        buffer += params.input_suffix;
                        printf("%s", params.input_suffix.c_str());
                    }

                    LOG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    const auto line_inp = ::llama_tokenize(ctx, buffer, false);
                    LOG("input tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, line_inp).c_str());

                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        output_tokens.push_back(token);
                        output_ss << llama_token_to_piece(ctx, token);
                    }

                    n_remain -= line_inp.size();
                    LOG("n_remain: %d\n", n_remain);
                } else {
                    LOG("empty line, passing control back\n");
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
                if (is_interacting) {
                    llama_sampling_reset(ctx_sampling);
                }
                is_interacting = false;
            }
        }

        // end of generation
        if (!embd.empty() && llama_token_is_eog(model, embd.back()) && !params.interactive) {
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
        printf("%s", llama_token_to_piece(ctx, llama_token_eot(model)).c_str());
        fflush(stdout);
    }

    llama_print_timings(ctx);
    write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);

    llama_free(ctx);
    llama_free_model(model);

    llama_sampling_free(ctx_sampling);
    llama_backend_free();

#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log end\n");
#endif // LOG_DISABLE_LOGS

    return 0;
}
