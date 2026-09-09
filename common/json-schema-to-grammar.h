#pragma once

#include "json-schema.h"
#include "json.h"

#include <functional>
#include <string>

// The JSON overload goes through common_schema_parse() first: JSON -> common_schema -> GBNF
std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf = false);
std::string json_schema_to_grammar(const common_schema_document & schema);

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)> add_rule;
    std::function<std::string(const std::string &, const common_json &)> add_schema;
    std::function<void(const common_json &)> resolve_refs;
};

struct common_grammar_options {
    bool dotall = false;
};

std::string gbnf_format_literal(const std::string & literal);

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});
