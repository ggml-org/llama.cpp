#pragma once

#include "server-common.h"
#include "server-task.h"

#include <algorithm>
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

    // read the "<arch>.decision.*" metadata, type stays NONE if the model has none
    void init(const llama_model * model);

    // true if the questions of a request start with the same tokens, and the model can continue from them
    bool can_share_prompt() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_GRANITE_SWITCH:
            case COMMON_DECISION_TYPE_LEV:
            case COMMON_DECISION_TYPE_KEV:
            case COMMON_DECISION_TYPE_NIMBLE:
            case COMMON_DECISION_TYPE_PPLX_DECIDER:
            case COMMON_DECISION_TYPE_VLLM_DECISION2:
            case COMMON_DECISION_TYPE_LFM2_D1:
            case COMMON_DECISION_TYPE_H2O_LIGHTNING:
                return true;
            default:
                return false;
        }
    }

    // true if all the questions of a request go in one prompt, see fill_task_joint()
    bool is_joint() const {
        return type == COMMON_DECISION_TYPE_CLEF;
    }

    bool is_lfm3_d1() const {
        return type == COMMON_DECISION_TYPE_LFM3_D1 || type == COMMON_DECISION_TYPE_LFM3_D1_OMNI;
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
            case COMMON_DECISION_TYPE_H2O_LIGHTNING:
            case COMMON_DECISION_TYPE_MINICPM_OCR_JEV:
            case COMMON_DECISION_TYPE_LFM3_D1:
            case COMMON_DECISION_TYPE_LFM3_D1_OMNI:
                return true;
            default:
                return false;
        }
    }

    // false for a decision-only model, it serves no text generation
    bool can_generate() const {
        const auto out = server_model_output_modalities(type, true, false);
        return std::find(out.begin(), out.end(), "text") != out.end();
    }

    // throw std::invalid_argument on bad input
    std::vector<server_decision_question> parse_questions(const json & body) const;

    // the state as the model reads it, done once per request; returns the state for most models
    json prepare_state(const json & state) const;

    // returns the state without its images, they are appended to files in order
    // images come from "files" (alias "images") and from the image_url and input_audio parts of a state made of chat messages
    // an image can be an audio clip if the model supports it
    json parse_state(const json & body, std::vector<raw_buffer> & files) const;

    // number of prompts that are evaluated to answer this question, each one shows the options in a different order
    size_t n_variants(const server_decision_question & question) const;

    // set the prompt of one variant of this question, and where to read its result
    // mctx is only used if there are files
    void fill_task(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task) const;

    // set the prompt of all the questions, the result has the scores of all their options, in order
    // mctx is only used if there are files
    void fill_task_joint(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task) const;

    // scores: the raw model outputs of each variant
    json format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores, bool has_media = false) const;

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

    // OPENJEV, LEV, NIMBLE, PPLX_DECIDER, VLLM_D3, H2O_LIGHTNING
    std::vector<llama_token> labels;
    std::vector<std::string> label_texts; // only if the label of an option is given to the template

    // LAYA, KEV
    llama_token token_marker      = LLAMA_TOKEN_NULL;
    llama_token token_sep         = LLAMA_TOKEN_NULL;
    std::string text_marker;
    size_t      max_head_tokens   = 0; // question + options
    size_t      max_option_tokens = 48;

    // H2O_LIGHTNING
    std::vector<std::string> quoted_spellings; // texts of the special tokens, quoted in the input, longest first
    std::string              quoted_first;     // the first bytes of these texts

    std::string render(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            size_t n_images,
            bool is_audio = false) const;
    json render_options(const server_decision_question & question, size_t variant) const;
    size_t n_outputs(const server_decision_question & question) const;
    // LFM2_D1: label text and tokens of each option
    void d1_labels(const server_decision_question & question, std::vector<std::string> & texts, std::vector<llama_tokens> & groups) const;
    // H2O_LIGHTNING: the input of the template, its texts are rendered as the reference shim does
    json render_h2o(const json & state, const server_decision_question & question) const;
    std::string h2o_cut_state(const std::string & text) const;
    std::string h2o_quote(const std::string & text) const;
    // the texts of the tokens with these attributes, quoted_spellings and quoted_first
    void init_quoted_spellings(uint32_t attrs, size_t min_size);
    // LFM3_D1: a zero-width space after the first character of each special token text (d1 prompt.py: plain)
    std::string d1_break(const std::string & text) const;
    void fill_task_laya(llama_tokens & tokens, const server_decision_question & question, server_task & task) const;
    void fill_task_d1omni(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task) const;

    float get_temperature(const server_decision_question & question) const;
    double get_decision_confidence(server_decision_question_type type, const std::vector<double> & probs) const;
};

// group the tasks so that the common prefix of their prompts is evaluated only once
// each group is one parent and its children, it takes at most n_slots slots
// note: the order of the tasks is preserved
std::vector<server_task> server_decision_group_tasks(std::vector<server_task> && tasks, size_t n_slots);

// Keep the returned buffers alive until the request finishes.
std::vector<std::shared_ptr<std::vector<float>>> server_decision_share_images(std::vector<server_task> & tasks);

// write "<|name|>" with U+00A6 in place of "|", so text from the request is not parsed as special tokens
std::string server_decision_escape_special_tokens(const std::string & text);
