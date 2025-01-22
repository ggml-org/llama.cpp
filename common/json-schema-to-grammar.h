#pragma once

#include "ggml.h"
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"

std::string json_schema_to_grammar(const nlohmann::ordered_json & schema);

struct llama_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)> add_rule;
    std::function<std::string(const std::string &, const nlohmann::ordered_json &)> add_schema;
    std::function<void(nlohmann::ordered_json &)> resolve_refs;
};

std::string build_grammar(const std::function<void(const llama_grammar_builder &)> & cb);
