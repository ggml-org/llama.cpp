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

struct common_grammar_options {
    bool dotall = false;
    // any-order object rules: grammar states per object, and for the whole grammar
    size_t max_object_states = 1024;
    size_t max_total_states  = 4096;
};

std::string gbnf_format_literal(const std::string & literal);
std::string gbnf_format_rule_name(const std::string & name);

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});
