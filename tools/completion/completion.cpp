#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"
#include "../src/llama-context.h" // For full definition of llama_context C++ class
#include "ggml-backend.h"     // For ggml_backend_sched_t and API functions

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"        // For GPU sampling API
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static bool llama_debug_logits_enabled_cached() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("LLAMA_DEBUG_LOGITS");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

// Debug logits configuration - shared between SYCL and non-SYCL paths
struct debug_logits_config {
    int enabled = -1;
    int topk = 5;
    int row0 = -1;

    void init() {
        if (enabled < 0) {
            const char * env = std::getenv("LLAMA_DEBUG_LOGITS");
            enabled = (env && std::atoi(env) != 0) ? 1 : 0;
            if (enabled) {
                if (const char * topk_env = std::getenv("LLAMA_DEBUG_LOGITS_TOPK")) {
                    const int parsed_topk = std::atoi(topk_env);
                    if (parsed_topk > 0) {
                        topk = parsed_topk;
                    }
                }
                const char * row0_env = std::getenv("LLAMA_DEBUG_LOGITS_ROW0");
                row0 = (row0_env && std::atoi(row0_env) != 0) ? 1 : 0;
            }
        }
    }
};

// Entry for top-k logits tracking
struct logits_top_entry {
    int id;
    float logit;
};

// Compute top-k logits from a logits array
static std::vector<logits_top_entry> compute_topk_logits(const float * logits_ptr, int n_vocab, int k) {
    std::vector<logits_top_entry> top;
    top.reserve(k);
    for (int i = 0; i < n_vocab; ++i) {
        const float v = logits_ptr[i];
        if ((int) top.size() < k) {
            top.push_back({ i, v });
            std::sort(top.begin(), top.end(), [](const logits_top_entry & a, const logits_top_entry & b) {
                return a.logit > b.logit;
            });
            continue;
        }
        int min_idx = 0;
        for (int j = 1; j < k; ++j) {
            if (top[j].logit < top[min_idx].logit) {
                min_idx = j;
            }
        }
        if (v > top[min_idx].logit) {
            top[min_idx] = { i, v };
            std::sort(top.begin(), top.end(), [](const logits_top_entry & a, const logits_top_entry & b) {
                return a.logit > b.logit;
            });
        }
    }
    return top;
}

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
static bool is_interacting  = false;
static bool need_insert_eot = false;

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -sys \"You are a helpful assistant\"\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting  = true;
            need_insert_eot = true;
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

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sampling;

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
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

    std::vector<common_chat_msg> chat_msgs;

    // load the model and apply lora adapter, if any
    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);

    auto llama_init = common_init_from_params(params);

    ctx   = llama_init->context();
    model = llama_init->model();
    smpl  = llama_init->sampler(0);

    if (ctx == NULL) {
        LOG_ERR("%s: error: unable to create context\n", __func__);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // note: the time for chat template initialization is not negligible:
    auto chat_templates = common_chat_templates_init(model, params.chat_template);

    // start measuring performance timings from here
    llama_perf_context_reset(ctx);

    LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, (int) params.cpuparams.n_threads);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        LOG_ERR("%s: no CPU backend found\n", __func__);
        return 1;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    set_process_priority(params.cpuparams.priority);

    struct ggml_threadpool * threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            return 1;
        }

        // start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return 1;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    // auto enable conversation mode if chat template is available
    const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        if (has_chat_template) {
            LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
            params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        } else {
            params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        }
    }

    // in case user force-activate conversation mode (via -cnv) without proper chat template, we show a warning
    if (params.conversation_mode && !has_chat_template) {
        LOG_WRN("%s: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n", __func__);
    }

    // print chat template example in conversation mode
    if (params.conversation_mode) {
        if (params.enable_chat_template) {
            if (!params.prompt.empty() && params.system_prompt.empty()) {
                LOG_WRN("*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?\n");
            }

            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(chat_templates.get(), params.use_jinja, params.default_template_kwargs).c_str());
        } else {
            LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG_INF("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            LOG_INF("%s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(path_session)) {
            LOG_INF("%s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_ERR("%s: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            LOG_INF("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

    LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

    std::vector<llama_token> embd_inp;

    bool waiting_for_first_input = false;
    auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
        common_chat_msg new_msg;
        new_msg.role = role;
        new_msg.content = content;
        auto formatted = common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
        chat_msgs.push_back(new_msg);
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
    };

    std::string prompt;
    {
        if (params.conversation_mode && params.enable_chat_template) {
            if (!params.system_prompt.empty()) {
                // format the system prompt (will use template default if empty)
                chat_add_and_format("system", params.system_prompt);
            }

            if (!params.prompt.empty()) {
                // format and append the user prompt
                chat_add_and_format("user", params.prompt);
            } else {
                waiting_for_first_input = true;
            }

            if (!params.system_prompt.empty() || !params.prompt.empty()) {
                common_chat_templates_inputs inputs;
                inputs.use_jinja = g_params->use_jinja;
                inputs.messages = chat_msgs;
                inputs.add_generation_prompt = !params.prompt.empty();

                prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            }
        } else {
            // otherwise use the prompt as is
            prompt = params.prompt;
        }

        if (params.interactive_first || !prompt.empty() || session_tokens.empty()) {
            LOG_DBG("tokenize the prompt\n");
            embd_inp = common_tokenize(ctx, prompt, true, true);
        } else {
            LOG_DBG("use session tokens\n");
            embd_inp = session_tokens;
        }

        LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
        LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());
    }

    // Should not run without any tokens
    if (!waiting_for_first_input && embd_inp.empty()) {
        if (add_bos) {
            embd_inp.push_back(llama_vocab_bos(vocab));
            LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
        } else {
            LOG_ERR("input is empty\n");
            return -1;
        }
    }

    // Tokenize negative prompt
    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_INF("%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_INF("%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_WRN("%s: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG_INF("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        if (!llama_memory_seq_rm(mem, -1, n_matching_session_tokens, -1)) {
            LOG_INF("%s: unable to resuse common prefix\n", __func__);
            n_matching_session_tokens = 0;
            llama_memory_seq_rm(mem, -1, -1, -1);
        }
    }

    LOG_DBG("recalculate the cached logits (check): embd_inp.size() %zu, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu\n",
         embd_inp.size(), n_matching_session_tokens, embd_inp.size(), session_tokens.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOG_DBG("recalculate the cached logits (do): session_tokens.resize( %zu )\n", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    if (params.conversation_mode) {
        if (params.single_turn && !params.prompt.empty()) {
            params.interactive = false;
            params.interactive_first = false;
        } else {
            params.interactive_first = true;
        }
    }

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_INF("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > add_bos) {
            LOG_INF("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_INF("\n");
    }

    // ctrl+C handling
    {
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
    }

    if (params.interactive) {
        LOG_INF("%s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_INF("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    auto tmp = common_tokenize(ctx, antiprompt, false, true);
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            LOG_INF("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_INF("Input prefix: '%s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_prefix, true, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            LOG_INF("Input suffix: '%s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_suffix, false, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }
    }

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl));
    LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl).c_str());

    // GPU sampler initialization (SYCL only)
    // Note: Sampler is created lazily on first decode to use the correct backend
#ifdef GGML_USE_SYCL
    ggml_sycl_sampler_t gpu_sampler = nullptr;
    bool gpu_sampling_enabled = params.gpu_sampling;
    int gpu_multistep = params.gpu_multistep;
    if (gpu_sampling_enabled) {
        // Enable device logits - skip host copy since GPU sampling reads from device directly
        llama_set_logits_device(ctx, true);
        LOG_DBG("GPU sampling enabled%s\n", gpu_multistep > 1 ? " (multi-step)" : "");
    }
#endif

    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
      //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
      //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_INF("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_INF("\n");

    if (params.interactive) {
        const char * control_message;
        if (params.multiline_input) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to the AI.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_INF("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_INF(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_INF(       "%s", control_message);
        if (params.conversation_mode && params.enable_chat_template && params.system_prompt.empty()) {
            LOG_INF(   " - Not using system message. To change it, set a different value via -sys PROMPT\n");
        }
        LOG_INF("\n");

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool display              = true;
    bool skip_embd_decode     = false;  // Set true after multi-step GPU decode to skip re-decode
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;
    std::ostringstream assistant_ss; // for storing current assistant message, used in conversation mode

    // the first thing we will do is to output the prompt, so set color accordingly
    console::set_display(DISPLAY_TYPE_PROMPT);
    display = params.display_prompt;

    std::vector<llama_token> embd;

    // single-token antiprompts
    std::vector<llama_token> antiprompt_token;

    for (const std::string & antiprompt : params.antiprompt) {
        auto ids = ::common_tokenize(ctx, antiprompt, false, true);
        if (ids.size() == 1) {
            antiprompt_token.push_back(ids[0]);
        }
    }

    if (llama_model_has_encoder(model)) {
        int enc_input_size = embd_inp.size();
        llama_token * enc_input_buf = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }

    auto log_sampled_token_dbg = [&](llama_token token) {
        // Only emits when --log-verbosity is set high enough to include debug logs.
        const std::string piece = common_token_to_piece(ctx, token, /* special = */ true);
        LOG_DBG("sampled token: id=%d piece='%s'\n", token, piece.c_str());
    };

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                console::set_display(DISPLAY_TYPE_ERROR);
                LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(DISPLAY_TYPE_RESET);
            }

            if (ga_n == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

                if (n_past + (int) embd.size() >= n_ctx) {
                    if (!params.ctx_shift){
                        LOG_WRN("\n\n%s: context full and context shift is disabled => stopping\n", __func__);
                        break;
                    }

                    if (params.n_predict == -2) {
                        LOG_WRN("\n\n%s: context full and n_predict == %d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left    = n_past - params.n_keep;
                    const int n_discard = n_left/2;

                    LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                            n_past, n_left, n_ctx, params.n_keep, n_discard);

                    llama_memory_seq_rm (mem, 0, params.n_keep            , params.n_keep + n_discard);
                    llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    LOG_DBG("after swap: n_past = %d\n", n_past);

                    LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());

                    LOG_DBG("clear session path\n");
                    path_session.clear();
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    LOG_DBG("\n");
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    LOG_DBG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_memory_seq_add(mem, 0, ga_i,                n_past,              ib*bd);
                    llama_memory_seq_div(mem, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_memory_seq_add(mem, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    LOG_DBG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            // Skip decode if we just did multi-step GPU decode (tokens already decoded)
            if (skip_embd_decode) {
                LOG_DBG("skipping embd decode (multi-step already decoded)\n");
                skip_embd_decode = false;
            } else {
                for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                    int n_eval = (int) embd.size() - i;
                    if (n_eval > params.n_batch) {
                        n_eval = params.n_batch;
                    }

                    LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

                    llama_batch batch = llama_batch_get_one(&embd[i], n_eval);
                    if (llama_debug_logits_enabled_cached()) {
                        static int dbg_left = 16;
                        if (dbg_left > 0) {
                            LOG_INF("[BATCH] n_past=%d n_eval=%d token_ptr=%p logits_ptr=%p\n",
                                    n_past, n_eval, (void *) batch.token, (void *) batch.logits);
                            if (batch.token && n_eval > 0) {
                                const int last_idx = n_eval - 1;
                                if (batch.logits) {
                                    LOG_INF("[BATCH] last_token=%d last_logit=%d\n",
                                            batch.token[last_idx], batch.logits[last_idx] ? 1 : 0);
                                } else {
                                    LOG_INF("[BATCH] last_token=%d last_logit=(null)\n", batch.token[last_idx]);
                                }
                            }
                            dbg_left--;
                        }
                    }
                    if (llama_decode(ctx, batch)) {
                        LOG_ERR("%s : failed to eval\n", __func__);
                        return 1;
                    }

                    n_past += n_eval;

                    LOG_DBG("n_past = %d\n", n_past);
                    // Display total tokens alongside total time
                    if (params.n_print > 0 && n_past % params.n_print == 0) {
                        LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                    }
                }
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG_DBG("saved session to %s\n", path_session.c_str());
            }

            llama_token id;
#ifdef GGML_USE_SYCL
            // Lazy initialization of GPU sampler - uses the backend where logits reside
            if (gpu_sampling_enabled && !gpu_sampler) {
                ggml_backend_t logits_backend = llama_get_logits_backend(ctx);
                if (logits_backend && ggml_backend_is_sycl(logits_backend)) {
                    const int n_vocab = llama_vocab_n_tokens(vocab);
                    gpu_sampler = ggml_backend_sycl_sampler_create(logits_backend, n_vocab, sparams.seed);
                    if (gpu_sampler) {
                        LOG_DBG("GPU sampler created on '%s' (n_vocab=%d)\n",
                               ggml_backend_name(logits_backend), n_vocab);
                    } else {
                        LOG_WRN("Failed to create GPU sampler, falling back to CPU sampling\n");
                        gpu_sampling_enabled = false;
                    }
                } else {
                    LOG_WRN("Logits backend is not SYCL, falling back to CPU sampling\n");
                    gpu_sampling_enabled = false;
                }
            }
            auto sync_logits_if_needed = [&]() {
                ggml_backend_t logits_backend = llama_get_logits_backend(ctx);
                if (logits_backend && ggml_backend_is_sycl(logits_backend)) {
                    llama_synchronize(ctx);
                }
            };

            static debug_logits_config dbg_cfg;
            dbg_cfg.init();

            auto maybe_debug_logits = [&]() {
                if (!dbg_cfg.enabled) {
                    return;
                }

                sync_logits_if_needed();

                const float * logits = llama_get_logits(ctx);
                if (!logits) {
                    LOG_WRN("[LOGITS] logits pointer is null at n_past=%d\n", n_past);
                    return;
                }

                const int n_vocab = llama_vocab_n_tokens(vocab);
                const int k = std::min(dbg_cfg.topk, n_vocab);
                if (k <= 0) {
                    return;
                }

                const std::vector<logits_top_entry> host_top = compute_topk_logits(logits, n_vocab, k);

                if (dbg_cfg.row0 > 0 && n_past > 1) {
                    const float * logits_row0 = llama_get_logits_ith(ctx, 0);
                    if (logits_row0) {
                        const std::vector<logits_top_entry> host_top_row0 = compute_topk_logits(logits_row0, n_vocab, k);
                        LOG_INF("[LOGITS] n_past=%d host row0 top%d:\n", n_past, k);
                        for (const auto & e : host_top_row0) {
                            const std::string piece = common_token_to_piece(ctx, e.id, true);
                            LOG_INF("[LOGITS]   row0 id=%d logit=%g piece=%s\n", e.id, e.logit, piece.c_str());
                        }
                    } else {
                        LOG_WRN("[LOGITS] logits_row0 pointer is null at n_past=%d\n", n_past);
                    }
                }

                bool have_device_top = false;
                std::vector<logits_top_entry> device_top;
#ifdef GGML_USE_SYCL
                ggml_backend_t logits_backend = llama_get_logits_backend(ctx);
                ggml_tensor * logits_tensor = llama_get_logits_tensor(ctx);
                if (logits_backend && ggml_backend_is_sycl(logits_backend) && logits_tensor) {
                    std::vector<float> device_logits(n_vocab, 0.0f);
                    ggml_backend_tensor_get(logits_tensor, device_logits.data(), 0, n_vocab * sizeof(float));
                    device_top = compute_topk_logits(device_logits.data(), n_vocab, k);
                    have_device_top = true;
                }
#endif

                LOG_INF("[LOGITS] n_past=%d host top%d:\n", n_past, k);
                for (const auto & e : host_top) {
                    const std::string piece = common_token_to_piece(ctx, e.id, true);
                    LOG_INF("[LOGITS]   host id=%d logit=%g piece=%s\n", e.id, e.logit, piece.c_str());
                }
                if (have_device_top) {
                    LOG_INF("[LOGITS] n_past=%d device top%d:\n", n_past, k);
                    for (const auto & e : device_top) {
                        const std::string piece = common_token_to_piece(ctx, e.id, true);
                        LOG_INF("[LOGITS]   dev  id=%d logit=%g piece=%s\n", e.id, e.logit, piece.c_str());
                    }
                    if (!host_top.empty() && !device_top.empty() && host_top[0].id != device_top[0].id) {
                        LOG_WRN("[LOGITS] top-1 mismatch at n_past=%d: host=%d dev=%d\n",
                                n_past, host_top[0].id, device_top[0].id);
                    }
                }
            };
            maybe_debug_logits();

            if (gpu_sampler && gpu_multistep > 1 && n_remain >= gpu_multistep) {
                // Multi-step GPU decode path - generate multiple tokens without CPU sync
                // Uses full top-k/top-p/min-p sampling support
                ggml_tensor * logits_tensor = llama_get_logits_tensor(ctx);
                if (logits_tensor) {
                    // Reset the token buffer
                    ggml_backend_sycl_sampler_reset_buffer(gpu_sampler);

                    // First token - sample to device buffer with full sampling params
                    ggml_backend_sycl_sample_token_to_device_full(gpu_sampler, logits_tensor,
                        sparams.temp, sparams.top_k, sparams.top_p, sparams.min_p);

                    // Generate remaining tokens without CPU sync
                    llama_token dummy_token = 0;  // Placeholder - actual token comes from pending device token
                    for (int step = 1; step < gpu_multistep; step++) {
                        // Set pending device token for this decode step
                        int32_t * token_ptr = ggml_backend_sycl_get_current_token_ptr(gpu_sampler);
                        ggml_backend_sycl_set_pending_device_token(token_ptr, 1);

                        // Decode with pending device token (token value ignored, uses device pointer)
                        if (llama_decode(ctx, llama_batch_get_one(&dummy_token, 1))) {
                            LOG_ERR("%s : failed to eval in multi-step decode\n", __func__);
                            ggml_backend_sycl_clear_pending_device_token();
                            break;
                        }
                        n_past++;

                        // Clear pending device token
                        ggml_backend_sycl_clear_pending_device_token();

                        // Sample next token to device buffer with full sampling params
                        logits_tensor = llama_get_logits_tensor(ctx);
                        if (!logits_tensor) break;
                        ggml_backend_sycl_sample_token_to_device_full(gpu_sampler, logits_tensor,
                            sparams.temp, sparams.top_k, sparams.top_p, sparams.min_p);
                    }

                    // Decode the last sampled token (wasn't decoded in the loop)
                    // This ensures the KV cache has all tokens for the next batch
                    {
                        int32_t * last_token_ptr = ggml_backend_sycl_get_current_token_ptr(gpu_sampler);
                        ggml_backend_sycl_set_pending_device_token(last_token_ptr, 1);
                        llama_token dummy = 0;
                        if (llama_decode(ctx, llama_batch_get_one(&dummy, 1))) {
                            LOG_ERR("%s : failed to decode last multi-step token\n", __func__);
                        }
                        ggml_backend_sycl_clear_pending_device_token();
                        n_past++;
                    }

                    // Sync all tokens back to host
                    int n_generated = ggml_backend_sycl_get_token_count(gpu_sampler);
                    std::vector<int32_t> generated_tokens(n_generated);
                    ggml_backend_sycl_get_sampled_tokens(gpu_sampler, generated_tokens.data(), n_generated);

                    // Process all generated tokens (filter post-EOS, update state)
                    bool hit_eos = false;
                    for (int i = 0; i < n_generated && !hit_eos; i++) {
                        id = generated_tokens[i];

                        // Check for EOS
                        if (llama_vocab_is_eog(vocab, id)) {
                            hit_eos = true;
                        }

                        log_sampled_token_dbg(id);
                        common_sampler_accept(smpl, id, /* accept_grammar= */ true);
                        embd.push_back(id);

                        if (params.conversation_mode && !waiting_for_first_input && !hit_eos) {
                            assistant_ss << common_token_to_piece(ctx, id, false);
                        }

                        --n_remain;
                    }

                    // echo this to console
                    input_echo = true;
                    skip_embd_decode = true;  // Tokens already decoded in multi-step loop
                    LOG_DBG("n_remain: %d (multi-step generated %d tokens)\n", n_remain, n_generated);
                } else {
                    // Fall back to single-step CPU sampling
                    sync_logits_if_needed();
                    id = common_sampler_sample(smpl, ctx, -1);
                    log_sampled_token_dbg(id);
                    common_sampler_accept(smpl, id, /* accept_grammar= */ true);
                    embd.push_back(id);
                    if (params.conversation_mode && !waiting_for_first_input && !llama_vocab_is_eog(vocab, id)) {
                        assistant_ss << common_token_to_piece(ctx, id, false);
                    }
                    input_echo = true;
                    --n_remain;
                    LOG_DBG("n_remain: %d\n", n_remain);
                }
            } else if (gpu_sampler) {
                // Single-step GPU sampling path with full top-k/top-p support
                // Synchronize to ensure decode timing is properly accounted for
                llama_synchronize(ctx);
                ggml_tensor * logits_tensor = llama_get_logits_tensor(ctx);
                if (logits_tensor) {
                    // Use full sampling API with top-k, top-p, min-p support
                    id = ggml_backend_sycl_sample_token_full(gpu_sampler, logits_tensor, 0,
                        sparams.temp, sparams.top_k, sparams.top_p, sparams.min_p);
                } else {
                    sync_logits_if_needed();
                    id = common_sampler_sample(smpl, ctx, -1);
                }
                log_sampled_token_dbg(id);
                common_sampler_accept(smpl, id, /* accept_grammar= */ true);
                embd.push_back(id);
                if (params.conversation_mode && !waiting_for_first_input && !llama_vocab_is_eog(vocab, id)) {
                    assistant_ss << common_token_to_piece(ctx, id, false);
                }
                input_echo = true;
                --n_remain;
                LOG_DBG("n_remain: %d\n", n_remain);
            } else {
                sync_logits_if_needed();
                id = common_sampler_sample(smpl, ctx, -1);
                log_sampled_token_dbg(id);
                common_sampler_accept(smpl, id, /* accept_grammar= */ true);
                embd.push_back(id);
                if (params.conversation_mode && !waiting_for_first_input && !llama_vocab_is_eog(vocab, id)) {
                    assistant_ss << common_token_to_piece(ctx, id, false);
                }
                input_echo = true;
                --n_remain;
                LOG_DBG("n_remain: %d\n", n_remain);
            }
#else
            // Use shared debug logits infrastructure
            static debug_logits_config dbg_cfg;
            dbg_cfg.init();

            if (dbg_cfg.enabled > 0) {
                const float * logits = llama_get_logits(ctx);
                if (!logits) {
                    LOG_WRN("[LOGITS] logits pointer is null at n_past=%d\n", n_past);
                } else {
                    const int n_vocab = llama_vocab_n_tokens(vocab);
                    const int k = std::min(dbg_cfg.topk, n_vocab);
                    if (k > 0) {
                        if (dbg_cfg.row0 > 0 && n_past > 1) {
                            const float * logits_row0 = llama_get_logits_ith(ctx, 0);
                            if (logits_row0) {
                                const std::vector<logits_top_entry> host_top_row0 = compute_topk_logits(logits_row0, n_vocab, k);
                                LOG_INF("[LOGITS] n_past=%d host row0 top%d:\n", n_past, k);
                                for (const auto & e : host_top_row0) {
                                    const std::string piece = common_token_to_piece(ctx, e.id, true);
                                    LOG_INF("[LOGITS]   row0 id=%d logit=%g piece=%s\n", e.id, e.logit, piece.c_str());
                                }
                            } else {
                                LOG_WRN("[LOGITS] logits_row0 pointer is null at n_past=%d\n", n_past);
                            }
                        }

                        const std::vector<logits_top_entry> host_top = compute_topk_logits(logits, n_vocab, k);
                        LOG_INF("[LOGITS] n_past=%d host top%d:\n", n_past, k);
                        for (const auto & e : host_top) {
                            const std::string piece = common_token_to_piece(ctx, e.id, true);
                            LOG_INF("[LOGITS]   host id=%d logit=%g piece=%s\n", e.id, e.logit, piece.c_str());
                        }
                    }
                }
            }

            id = common_sampler_sample(smpl, ctx, -1);
            log_sampled_token_dbg(id);
            common_sampler_accept(smpl, id, /* accept_grammar= */ true);
            embd.push_back(id);
            if (params.conversation_mode && !waiting_for_first_input && !llama_vocab_is_eog(vocab, id)) {
                assistant_ss << common_token_to_piece(ctx, id, false);
            }
            input_echo = true;
            --n_remain;
            LOG_DBG("n_remain: %d\n", n_remain);
#endif
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                // Console/Stream Output
                LOG("%s", token_str.c_str());

                // Record Displayed Tokens To Log
                // Note: Generated tokens are created one by one hence this check
                if (embd.size() > 1) {
                    // Incoming Requested Tokens
                    input_tokens.push_back(id);
                } else {
                    // Outgoing Generated Tokens
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }

        // reset color to default if there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(DISPLAY_TYPE_RESET);
            display = true;
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                // avoid calling common_sampler_last() if last_output is empty
                if (!last_output.empty()) {
                    llama_token last_token = common_sampler_last(smpl);
                    for (auto token : antiprompt_token) {
                        if (token == last_token) {
                            if (params.interactive) {
                                is_interacting = true;
                            }
                            is_antiprompt = true;
                            break;
                        }
                    }
                }

                if (is_antiprompt) {
                    LOG_DBG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of generation tokens in interactive mode
            if (!waiting_for_first_input && llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
                LOG_DBG("found an EOG token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    if (params.enable_chat_template) {
                        chat_add_and_format("assistant", assistant_ss.str());
                    }
                    is_interacting = true;
                    LOG("\n");
                }
            }

            if (params.conversation_mode && !waiting_for_first_input) {
                if (!prompt.empty()) {
                    prompt.clear();
                    is_interacting = false;
                }
            }

            if ((n_past > 0 || waiting_for_first_input) && is_interacting) {
                LOG_DBG("waiting for user input\n");

                if (params.conversation_mode) {
                    LOG("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG_DBG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_vocab_bos(vocab));
                }

                std::string buffer;
                if (!params.input_prefix.empty() && !params.conversation_mode) {
                    LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    LOG("%s", params.input_prefix.c_str());
                }

                // color user input only
                console::set_display(DISPLAY_TYPE_USER_INPUT);
                display = params.display_prompt;

                std::string line;
                bool another_line = true;
                do {
                    another_line = console::readline(line, params.multiline_input);
                    buffer += line;
                } while (another_line);

                // done taking input, reset color
                console::set_display(DISPLAY_TYPE_RESET);
                display = true;

                if (buffer.empty()) { // Ctrl+D on empty line exits
                    LOG("EOF by user\n");
                    break;
                }

                if (buffer.back() == '\n') {
                    // Implement #587:
                    // If the user wants the text to end in a newline,
                    // this should be accomplished by explicitly adding a newline by using \ followed by return,
                    // then returning control by pressing return again.
                    buffer.pop_back();
                }

                if (buffer.empty()) { // Enter key on empty line lets the user pass control back
                    LOG_DBG("empty line, passing control back\n");
                } else { // Add tokens to embd only if the input buffer is non-empty
                    // append input suffix if any
                    if (!params.input_suffix.empty() && !params.conversation_mode) {
                        LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        LOG("%s", params.input_suffix.c_str());
                    }

                    LOG_DBG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    bool format_chat = params.conversation_mode && params.enable_chat_template;
                    std::string user_inp = format_chat
                        ? chat_add_and_format("user", std::move(buffer))
                        : std::move(buffer);
                    // TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                    LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                    // if user stop generation mid-way, we must add EOT to finish model's last response
                    if (need_insert_eot && format_chat) {
                        llama_token eot = llama_vocab_eot(vocab);
                        embd_inp.push_back(eot == LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    if (params.verbose_prompt) {
                        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size() - original_size);
                    }

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        const std::string token_str = common_token_to_piece(ctx, token);
                        output_tokens.push_back(token);
                        output_ss << token_str;

                        if (params.verbose_prompt) {
                            LOG_INF("%6d -> '%s'\n", token, token_str.c_str());
                        }
                    }

                    // reset assistant message
                    assistant_ss.str("");

                    n_remain -= line_inp.size();
                    LOG_DBG("n_remain: %d\n", n_remain);
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0 || waiting_for_first_input) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;

                if (waiting_for_first_input && params.single_turn) {
                    params.interactive = false;
                    params.interactive_first = false;
                }
                waiting_for_first_input = false;
            }
        }

        // end of generation
        if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !(params.interactive)) {
            LOG(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    LOG("\n\n");
    common_perf_print(ctx, smpl);

    // Save MoE expert profile if profiling was enabled
    common_moe_profile_finish(ctx, params);

    // Check if pipeline profiling is enabled and print stats
    ggml_backend_sched_t sched = ctx->get_sched();
    if (sched && ggml_backend_sched_is_profiling_enabled(sched)) {
        ggml_backend_sched_print_pipeline_stats(sched);
    }

    // Note: smpl is owned by llama_init and will be freed when it goes out of scope
    // Do NOT call common_sampler_free(smpl) here as it would cause a double free

#ifdef GGML_USE_SYCL
    if (gpu_sampler) {
        ggml_backend_sycl_sampler_free(gpu_sampler);
    }
#endif

    llama_backend_free();

    ggml_threadpool_free_fn(threadpool);
    ggml_threadpool_free_fn(threadpool_batch);

    return 0;
}
