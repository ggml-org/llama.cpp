#pragma once

#include "json-schema.h"
#include "json.h"

#include <functional>
#include <string>

// The JSON overload goes through common_schema_parse() first: JSON -> common_schema -> GBNF
std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf = false);
std::string json_schema_to_grammar(const common_schema_document & schema);

// Whether a value matching the schema may be a string, through any branch of it.
// Some models emit raw string values rather than JSON-encoded strings for string parameters.
bool common_schema_resolves_to_string(const common_schema & schema);

// Probes the sub-schemas of one JSON schema, e.g. the parameters of a tool
class common_schema_info {
    common_schema_document doc_;

  public:
    // Parses the schema, so that the $refs of its sub-schemas resolve
    void resolve_refs(const common_json & schema);

    // common_schema_resolves_to_string() for a sub-schema of a schema given to resolve_refs(), false when it does not parse
    bool resolves_to_string(const common_json & schema);
};

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
