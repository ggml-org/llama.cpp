#pragma once

#include "engine-common.h"
#include "engine-task.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

// typed decision models (TypeSafe /v1/systemone API)
// the model answers each question in one forward pass, no token is generated

enum engine_decision_question_type {
    ENGINE_DECISION_QUESTION_CHOICE,
    ENGINE_DECISION_QUESTION_SCORE,
    ENGINE_DECISION_QUESTION_NOUL,
};

struct engine_decision_option {
    std::string key;
    json description; // null if not provided
};

struct engine_decision_question {
    std::string id;
    engine_decision_question_type type;
    json instructions;
    std::vector<engine_decision_option> options; // in the order of the model outputs
};

struct engine_decision_context {
    common_decision_type type = COMMON_DECISION_TYPE_NONE;

    // read the "<arch>.decision.*" metadata, type stays NONE if the model has none
    void init(const llama_model * model);

    // true if the questions of a request start with the same tokens, and the model can continue from them
    bool can_share_prompt() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_LEV:
            case COMMON_DECISION_TYPE_KEV:
            case COMMON_DECISION_TYPE_NIMBLE:
            case COMMON_DECISION_TYPE_PPLX_DECIDER:
            case COMMON_DECISION_TYPE_VLLM_DECISION2:
            case COMMON_DECISION_TYPE_LFM2_D1:
                return true;
            default:
                return false;
        }
    }

    // true if all the questions of a request go in one prompt, see fill_task_joint()
    bool is_joint() const {
        return type == COMMON_DECISION_TYPE_CLEF;
    }

    // true if the prompt of the model has a place for images
    bool can_use_images() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_CLEF:
            case COMMON_DECISION_TYPE_PPLX_DECIDER:
            case COMMON_DECISION_TYPE_VLLM_D3:
            case COMMON_DECISION_TYPE_LFM2_D1:
            case COMMON_DECISION_TYPE_LFM2_D1_OMNI:
                return true;
            default:
                return false;
        }
    }

    // throw std::invalid_argument on bad input
    std::vector<engine_decision_question> parse_questions(const json & body) const;

    // returns the state without its images, they are appended to files in order
    // images come from "files" (alias "images") and from the image_url and input_audio parts of a state made of chat messages
    // an image can be an audio clip if the model supports it
    json parse_state(const json & body, std::vector<raw_buffer> & files) const;

    // number of prompts that are evaluated to answer this question, each one shows the options in a different order
    size_t n_variants(const engine_decision_question & question) const;

    // set the prompt of one variant of this question, and where to read its result
    // mctx is only used if there are files
    void fill_task(
            const json & state,
            const std::vector<engine_decision_question> & questions,
            const engine_decision_question & question,
            size_t variant,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            engine_task & task) const;

    // set the prompt of all the questions, the result has the scores of all their options, in order
    // mctx is only used if there are files
    void fill_task_joint(
            const json & state,
            const std::vector<engine_decision_question> & questions,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            engine_task & task) const;

    // scores: the raw model outputs of each variant
    json format_answer(const engine_decision_question & question, const std::vector<std::vector<float>> & scores, bool has_media = false) const;

private:
    const llama_vocab * vocab = nullptr;
    std::shared_ptr<const common_chat_template> tmpl; // the "systemone" template

    std::map<std::string, float> temperatures; // "<type>" or "<type>.<n_options bucket>"
    std::map<size_t, std::vector<float>> score_bias; // added to the scores of a score question, by number of levels
    size_t n_options_max   = 0;
    bool   noul_true_first = false; // noul options are [true, false] instead of [false, true]
    bool   noul_criteria_order = false; // noul options are in the order of the criteria keys, if given
    bool   choice_sorted   = false; // choice options are in the order of their keys
    bool   strict_payload  = false; // state, instructions and descriptions must be text, an object or an array
    bool   canonical_legend = false; // non-string score descriptions are returned as canonical JSON text

    // OPENJEV, LEV, NIMBLE, PPLX_DECIDER, VLLM_D3
    std::vector<llama_token> labels;
    std::vector<std::string> label_texts; // only if the label of an option is given to the template

    // LAYA, KEV
    llama_token token_marker      = LLAMA_TOKEN_NULL;
    llama_token token_sep         = LLAMA_TOKEN_NULL;
    std::string text_marker;
    size_t      max_head_tokens   = 0; // question + options
    size_t      max_option_tokens = 48;

    std::string render(
            const json & state,
            const std::vector<engine_decision_question> & questions,
            const engine_decision_question & question,
            size_t variant,
            size_t n_images,
            bool is_audio = false) const;
    json render_options(const engine_decision_question & question, size_t variant) const;
    size_t n_outputs(const engine_decision_question & question) const;
    // LFM2_D1: label text and tokens of each option
    void d1_labels(const engine_decision_question & question, std::vector<std::string> & texts, std::vector<llama_tokens> & groups) const;
    void fill_task_laya(llama_tokens & tokens, const engine_decision_question & question, engine_task & task) const;
    void fill_task_d1omni(
            const json & state,
            const std::vector<engine_decision_question> & questions,
            const engine_decision_question & question,
            size_t variant,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            engine_task & task) const;

    float get_temperature(const engine_decision_question & question) const;
    double get_decision_confidence(engine_decision_question_type type, const std::vector<double> & probs) const;
};

// group the tasks so that the common prefix of their prompts is evaluated only once
// each group is one parent and its children, it takes at most n_slots slots
// note: the order of the tasks is preserved
std::vector<engine_task> engine_decision_group_tasks(std::vector<engine_task> && tasks, size_t n_slots);

// write "<|name|>" with U+00A6 in place of "|", so text from the request is not parsed as special tokens
std::string engine_decision_escape_special_tokens(const std::string & text);
