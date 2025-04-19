// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

#pragma once

#include "common.h"
#include <string>
#include <vector>

struct common_chat_templates;

struct common_chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;
};

struct common_chat_msg_content_part {
    std::string type;
    std::string text;
};

struct common_chat_msg {
    std::string role;
    std::string content;
    std::vector<common_chat_msg_content_part> content_parts = {};
    std::vector<common_chat_tool_call> tool_calls = {};
    std::string reasoning_content;
    std::string tool_name;
    std::string tool_call_id;
};

struct common_chat_tool {
    std::string name;
    std::string description;
    std::string parameters;
};

enum common_chat_tool_choice {
    COMMON_CHAT_TOOL_CHOICE_AUTO,
    COMMON_CHAT_TOOL_CHOICE_REQUIRED,
    COMMON_CHAT_TOOL_CHOICE_NONE,
};

enum common_chat_format {
    COMMON_CHAT_FORMAT_CONTENT_ONLY,
    COMMON_CHAT_FORMAT_GENERIC,
    COMMON_CHAT_FORMAT_MISTRAL_NEMO,
    COMMON_CHAT_FORMAT_LLAMA_3_X,
    COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
    COMMON_CHAT_FORMAT_DEEPSEEK_R1,
    COMMON_CHAT_FORMAT_DEEPSEEK_R1_EXTRACT_REASONING,
    COMMON_CHAT_FORMAT_FIREFUNCTION_V2,
    COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2,
    COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1,
    COMMON_CHAT_FORMAT_HERMES_2_PRO,
    COMMON_CHAT_FORMAT_HERMES_2_PRO_EXTRACT_REASONING,
    COMMON_CHAT_FORMAT_COMMAND_R7B,
    COMMON_CHAT_FORMAT_COMMAND_R7B_EXTRACT_REASONING,
    COMMON_CHAT_FORMAT_PHI_4,

    COMMON_CHAT_FORMAT_COUNT, // Not a format, just the # formats
};

struct common_chat_templates_inputs {
    std::vector<common_chat_msg> messages;
    std::string grammar;
    std::string json_schema;
    bool add_generation_prompt = true;
    bool use_jinja = true;
    // Parameters below only supported when use_jinja is true
    std::vector<common_chat_tool> tools;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool parallel_tool_calls = false;
    bool extract_reasoning     = true;
};

struct common_chat_params {
    common_chat_format                  format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string                         prompt;
    std::string                         grammar;
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::vector<std::string>            additional_stops;
};

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
bool common_chat_verify_template(const std::string & tmpl, bool use_jinja);

void common_chat_templates_free(struct common_chat_templates * tmpls);

struct common_chat_templates_deleter { void operator()(common_chat_templates * tmpls) { common_chat_templates_free(tmpls); } };

typedef std::unique_ptr<struct common_chat_templates, common_chat_templates_deleter> common_chat_templates_ptr;

common_chat_templates_ptr common_chat_templates_init(
                                    const struct llama_model * model,
                                           const std::string & chat_template_override,
                                           const std::string & bos_token_override = "",
                                           const std::string & eos_token_override = "");

bool         common_chat_templates_was_explicit(const struct common_chat_templates * tmpls);
const char * common_chat_templates_source(const struct common_chat_templates * tmpls, const char * variant = nullptr);


struct common_chat_params      common_chat_templates_apply(
    const struct common_chat_templates * tmpls,
    const struct common_chat_templates_inputs & inputs);

// Format single message, while taking into account the position of that message in chat history
std::string common_chat_format_single(
        const struct common_chat_templates * tmpls,
        const std::vector<common_chat_msg> & past_msg,
        const common_chat_msg & new_msg,
        bool add_ass,
        bool use_jinja);

// Returns an example of formatted chat
std::string common_chat_format_example(
    const struct common_chat_templates * tmpls,
    bool use_jinja);

std::string               common_chat_format_name(common_chat_format format);
common_chat_msg           common_chat_parse(      const std::string & input, common_chat_format format);

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice);

// Parses a JSON array of messages in OpenAI's chat completion API format.
// T can be std::string containing JSON or nlohmann::ordered_json
template <class T> std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const T & messages);
template <class T> T common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text = false);

// Parses a JSON array of tools in OpenAI's chat completion tool call API format.
// T can be std::string containing JSON or nlohmann::ordered_json
template <class T> std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const T & tools);
template <class T> T common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools);
