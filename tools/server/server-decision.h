#pragma once

#include "server-common.h"
#include "server-task.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

// typed decision models (TypeSafe /v1/systemone API)
// the model answers each question in one forward pass, no token is generated

enum server_decision_question_type {
    SERVER_DECISION_QUESTION_CHOICE,
    SERVER_DECISION_QUESTION_SCORE,
    SERVER_DECISION_QUESTION_NOUL,
};

struct server_decision_option {
    std::string key;
    json description; // null if not provided
};

struct server_decision_question {
    std::string id;
    server_decision_question_type type;
    json instructions;
    std::vector<server_decision_option> options; // in the order of the model outputs
};

struct server_decision_context {
    common_decision_type type = COMMON_DECISION_TYPE_NONE;

    // read the "<arch>.decision.*" metadata
    // a model without it is a GENERIC one if allow_generic is set, otherwise type stays NONE
    void init(const llama_model * model, bool allow_generic);

    // true if the questions of a request start with the same tokens, and the model can continue from them
    bool can_share_prompt() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_LEV:
            case COMMON_DECISION_TYPE_KEV:
            case COMMON_DECISION_TYPE_GENERIC:
                return true;
            default:
                return false;
        }
    }

    // true if the prompt of the model has a place for images
    bool can_use_images() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_GENERIC:
                return true;
            default:
                return false;
        }
    }

    // true if the prompt of the model has a place for audio
    bool can_use_audio() const {
        switch (type) {
            case COMMON_DECISION_TYPE_GENERIC:
                return true;
            default:
                return false;
        }
    }

    // throw std::invalid_argument on bad input
    std::vector<server_decision_question> parse_questions(const json & body) const;

    // returns the state without its media, they are appended to files in order
    // images come from "images" and from the image_url parts of a state made of chat messages
    // audio comes from the input_audio parts of a state made of chat messages, n_audio is their count
    json parse_state(const json & body, std::vector<raw_buffer> & files, size_t & n_audio) const;

    // number of prompts that are evaluated to answer this question, each one shows the options in a different order
    size_t n_variants(const server_decision_question & question) const;

    // set the prompt of one variant of this question, and where to read its result
    // mctx is only used if there are files, chat_params only by GENERIC
    void fill_task(
            const json & state,
            const server_decision_question & question,
            size_t variant,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            const server_chat_params & chat_params,
            server_task & task) const;

    // scores: the raw model outputs of each variant
    json format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores) const;

private:
    const llama_vocab * vocab = nullptr;
    std::shared_ptr<const common_chat_template> tmpl; // the "systemone" template, the user message for GENERIC

    std::map<std::string, float> temperatures; // "<type>" or "<type>.<n_options bucket>"
    size_t n_options_max   = 0;
    bool   noul_true_first = false; // noul options are [true, false] instead of [false, true]

    // OPENJEV, LEV, GENERIC
    std::vector<llama_token> labels;
    std::vector<std::string> label_texts; // only if the label of an option is given to the template

    // LAYA, KEV
    llama_token token_marker      = LLAMA_TOKEN_NULL;
    llama_token token_sep         = LLAMA_TOKEN_NULL;
    std::string text_marker;
    size_t      max_head_tokens   = 0; // question + options
    size_t      max_option_tokens = 48;

    void init_generic(const llama_model * model);
    std::string render(const json & state, const server_decision_question & question, size_t variant, size_t n_images) const;
    size_t n_outputs(const server_decision_question & question) const;
    void fill_task_laya(llama_tokens & tokens, const server_decision_question & question, server_task & task) const;

    float get_temperature(const server_decision_question & question) const;
};

// group the tasks so that the common prefix of their prompts is evaluated only once
// each group is one parent and its children, it takes at most n_slots slots
// note: the order of the tasks is preserved
std::vector<server_task> server_decision_group_tasks(std::vector<server_task> && tasks, size_t n_slots);
