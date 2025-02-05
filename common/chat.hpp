// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

#pragma once

#include "common.h"
#include <json.hpp>
#include <optional>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct common_chat_inputs {
    json messages;
    json tools;
    json tool_choice;
    json json_schema;
    bool parallel_tool_calls;
    bool stream;
    std::string grammar;
    bool add_generation_prompt = true;
};

enum common_chat_format {
    COMMON_CHAT_FORMAT_CONTENT_ONLY,
    COMMON_CHAT_FORMAT_GENERIC,
    COMMON_CHAT_FORMAT_MISTRAL_NEMO,
    COMMON_CHAT_FORMAT_LLAMA_3_X,
    COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
    COMMON_CHAT_FORMAT_DEEPSEEK_R1,
    COMMON_CHAT_FORMAT_FIREFUNCTION_V2,
    COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2,
    COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1,
    COMMON_CHAT_FORMAT_HERMES_2_PRO,
    COMMON_CHAT_FORMAT_COMMAND_R7B,

    COMMON_CHAT_FORMAT_COUNT, // Not a format, just the # formats
};

struct common_chat_params {
    common_chat_format                  format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    json                                prompt;
    std::string                         grammar;
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::vector<std::string>            additional_stops;
};

struct common_chat_params common_chat_params_init(const common_chat_template & tmpl, const struct common_chat_inputs & params);
std::string               common_chat_format_name(common_chat_format format);
common_chat_msg           common_chat_parse(      const std::string & input, common_chat_format format);
