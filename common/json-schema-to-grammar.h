#pragma once

#include "json-schema.h"
#include "json.h"

#include <cstddef>
#include <functional>
#include <string>

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf = false);
std::string json_schema_to_grammar(const common_chat_schema_document & schema);

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)>    add_rule;
    std::function<std::string(const std::string &, const common_chat_schema &)> add_schema;
};

// Unordered objects use up to 2^N states; larger objects fall back to ordered properties.
constexpr size_t COMMON_GRAMMAR_MAX_UNORDERED_PROPERTIES = 8;

struct common_grammar_options {
    bool dotall = false;
    // Overrides string length, pattern and format constraints during generation.
    std::string string_rule_name;
    std::string string_delimiter;
    std::string value_rule_name;
    bool bare_keys = false;
    // Applies up to COMMON_GRAMMAR_MAX_UNORDERED_PROPERTIES declared properties per object.
    bool unordered_properties = false;
};

std::string gbnf_format_literal(const std::string & literal);
std::string gbnf_format_rule_name(const std::string & name);

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});
