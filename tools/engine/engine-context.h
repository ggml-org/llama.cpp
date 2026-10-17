#pragma once

#include "engine-common.h"
#include "engine-decision.h"
#include "engine-task.h"
#include "engine-queue.h"

#include "json.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <set>

struct engine_context_impl; // private implementation

constexpr int HTTP_POLLING_SECONDS = 1;

struct engine_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    std::vector<std::string> model_output_modalities; // output modalities for GET /models
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    bool has_inp_video;
    json json_ui_settings;
    int slot_n_ctx;
    enum llama_pooling_type pooling_type;

    // chat params
    engine_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
    std::string model_ftype;
};

enum engine_state {
    ENGINE_STATE_DOWNLOADING,
    ENGINE_STATE_LOADING,
    ENGINE_STATE_READY,
    ENGINE_STATE_SLEEPING,
};

static std::string engine_state_to_str(engine_state state) {
    switch (state) {
        case ENGINE_STATE_DOWNLOADING: return "downloading";
        case ENGINE_STATE_LOADING:     return "loading";
        case ENGINE_STATE_READY:       return "ready";
        case ENGINE_STATE_SLEEPING:    return "sleeping";
        default: GGML_ASSERT(false && "invalid engine_state");
    }
}

static engine_state engine_state_from_str(const std::string & str) {
    if (str == "downloading") return ENGINE_STATE_DOWNLOADING;
    if (str == "loading")     return ENGINE_STATE_LOADING;
    if (str == "ready")       return ENGINE_STATE_READY;
    if (str == "sleeping")    return ENGINE_STATE_SLEEPING;
    GGML_ASSERT(false && "invalid engine_state string");
}

using engine_state_callback_t = std::function<void(engine_state, json /* payload */)>;

struct engine_context {
    std::unique_ptr<engine_context_impl> impl;

    engine_context();
    ~engine_context();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    engine_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    engine_context_meta get_meta() const;

    // note: must be set before load_model() is called
    void set_state_callback(engine_state_callback_t callback);

    // used by the HTTP layer
    // pointers can only be used when not sleeping, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * get_model() const;
    const llama_vocab * get_vocab() const;
    mtmd_context * get_mtmd_context() const;
    const mtmd_helper_init_opt & get_mtmd_init_opt() const;
    const engine_decision_context & get_decision() const;

    engine_queue    & get_queue_tasks();
    engine_response & get_queue_results();

    engine_metrics get_metrics() const;
    void reset_metrics_bucket();
};
