#pragma once

#include "chat.h"

#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <unordered_map>
#include <optional>
#include <string>
#include <string_view>
#include <functional>
#include <variant>

struct common_grammar_builder;

struct parser_environment {
    common_chat_msg result;

    // Tool call fields for building tool calls
    std::string tool_call_id;
    std::string tool_call_name;
    std::string tool_call_args;

    // Scratch pad for any custom logic
    std::unordered_map<std::string, std::variant<int, double, std::string>> scratchpad;
};

enum parser_result_type {
    PARSER_RESULT_FAIL            = 1 << 0,
    PARSER_RESULT_SUCCESS         = 1 << 1,
    PARSER_RESULT_NEED_MORE_INPUT = 1 << 2,
};

struct parse_cache_key {
    int id;
    size_t start;

    bool operator==(const parse_cache_key & other) const {
        return id == other.id && start == other.start;
    }
};

template <>
struct std::hash<parse_cache_key> {
    std::size_t operator()(const parse_cache_key & k) const {
        return std::hash<size_t>{}(((size_t)k.id << 32) | k.start);
    }
};

struct parser_result {
    parser_result_type type = PARSER_RESULT_FAIL;
    size_t start = 0;
    size_t end = 0;

    parser_result() : type(PARSER_RESULT_FAIL) {}

    parser_result(parser_result_type type, size_t start)
        : type(type), start(start), end(start) {}

    parser_result(parser_result_type type, size_t start, size_t end)
        : type(type), start(start), end(end) {}

    bool is_fail() const { return type == PARSER_RESULT_FAIL; }
    bool is_need_more_input() const { return type == PARSER_RESULT_NEED_MORE_INPUT; }
    bool is_success() const { return type == PARSER_RESULT_SUCCESS; }
};

struct parser_action {
    parser_result & result;
    parser_environment & env;
    std::string_view match;
};

class parse_cache {
    std::unordered_map<parse_cache_key, parser_result> results;

  public:
    parser_result set(int id, size_t start, parser_result result);
    std::optional<parser_result> get(int id, size_t start);
    void clear();
};

struct parser_context {
    std::string_view input;
    parse_cache memo;
    bool input_is_complete;
    parser_environment * env;

    parser_context()
        : memo(), input_is_complete(true), env(nullptr) {}

    parser_context(std::string_view input)
        : input(input), memo(), input_is_complete(true), env(nullptr) {}

    parser_context(std::string_view input, bool complete)
        : input(input), memo(), input_is_complete(complete), env(nullptr) {}

    parser_context(std::string_view input, parse_cache memo, bool complete = true)
        : input(input), memo(std::move(memo)), input_is_complete(complete), env(nullptr) {}

    parser_context(std::string_view input, parser_environment * environment)
        : input(input), memo(), input_is_complete(true), env(environment) {}

    parser_context(std::string_view input, parser_environment * environment, bool complete)
        : input(input), memo(), input_is_complete(complete), env(environment) {}

    parser_context(std::string_view input, parse_cache memo, parser_environment * environment, bool complete = true)
        : input(input), memo(std::move(memo)), input_is_complete(complete), env(environment) {}
};

class parser_base;

class parser {
    std::shared_ptr<parser_base> ptr_;

  public:
    parser();
    parser(std::shared_ptr<parser_base> parser);
    parser(const parser & other) = default;
    parser(const std::string & literal);
    parser(const char * literal);

    parser & operator=(const parser & other) {
        if (this != &other) {
            ptr_ = other.ptr_;
        }
        return *this;
    }

    parser operator~() const;
    parser operator+(const parser & other) const;
    parser operator|(const parser & other) const;
    parser operator<<(const parser & other) const;

    parser_base & operator*() const;
    parser_base * operator->() const;

    std::shared_ptr<parser_base> ptr() const { return ptr_; }

    parser_result parse(parser_context & ctx, size_t start = 0) const;

    std::string dump() const;

    void build_grammar(const common_grammar_builder & builder) const;
};

parser operator+(const char * lhs, const parser & rhs);
parser operator|(const char * lhs, const parser & rhs);
parser operator<<(const char * lhs, const parser & rhs);

class parser_id_counter {
    int next_id_;
  public:
    parser_id_counter(int start) : next_id_(start) {}
    int next() { return next_id_++; }
};

class parser_builder {
    std::shared_ptr<std::unordered_map<std::string, parser>> rules_;
    std::shared_ptr<parser_id_counter> counter_;

  public:
    parser_builder();
    parser_builder(std::shared_ptr<parser_id_counter> counter);

    // Matches an exact literal string.
    //   S -> "hello"
    parser literal(const std::string & literal);

    // Matches a sequence of parsers in order, all must succeed.
    //   S -> A B C
    parser sequence(std::initializer_list<parser> parsers);

    // Matches the first parser that succeeds from a list of alternatives.
    //   S -> A | B | C
    parser choice(std::initializer_list<parser> parsers);

    // Matches one or more repetitions of a parser.
    //   S -> A+
    parser one_or_more(const parser & p);

    // Matches zero or more repetitions of a parser, always succeeds.
    //   S -> A*
    parser zero_or_more(const parser & p);

    // Matches zero or one occurrence of a parser, always succeeds.
    //   S -> A?
    parser optional(const parser & p);

    // Negative lookahead: succeeds if child parser fails, consumes no input.
    //   S -> !A
    parser negate(const parser & p);

    // Matches any single character.
    //   S -> .
    parser any();

    // Matches between min and max repetitions of characters from a character class.
    //   S -> [a-z]{m,n}
    //
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    parser chars(const std::string & classes, int min = 1, int max = -1);

    // Matches a single character from a character class or range.
    //   S -> [a-z] or S -> [^0-9]
    //
    // Equivalent to chars(classes, 1, 1)
    parser one(const std::string & classes);

    // References a named rule for recursive or reusable grammar definitions.
    //   expr -> term | expr "+" term
    parser rule(const std::string & name);

    // Matches zero or more whitespace characters (space, tab, newline).
    //   S -> [ \t\n]*
    parser space();

    // Matches all characters until a delimiter is found (delimiter not consumed).
    //   S -> (!delim .)*
    parser until(const std::string & delimiter);

    // Matches between min and max repetitions of a parser (inclusive).
    //   S -> A{m,n}
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    parser repeat(const parser & p, int min, int max);

    // Matches exactly n repetitions of a parser.
    //   S -> A{n}
    parser repeat(const parser & p, int n);

    // Creates a complete JSON parser supporting objects, arrays, strings, numbers, booleans, and null.
    //   value -> object | array | string | number | true | false | null
    parser json();

    // Specialized single-pass JSON string parser with escape sequence handling
    parser json_string();

    // TODO: improve convenience functions to allow users to build specific JSON fields
    parser json_key(const std::string & name, const parser & p);
    parser json_string(const parser & p);

    // Wraps a parser with JSON schema metadata for grammar generation.
    // Used internally to convert JSON schemas to GBNF grammar rules.
    parser schema(const parser & p, const std::string & name, const nlohmann::ordered_json & schema);

    // Wraps a parser with a semantic action callback.
    // The callback is invoked on successful parse with the result, matched text, and environment.
    //   S -> A [action]
    parser action(const parser & p, std::function<void(const parser_action &)> fn, int when = PARSER_RESULT_SUCCESS);

    // Convenience action wrappers for common patterns

    // Causes a rule to succeed
    parser succeed(const parser & p, int when = PARSER_RESULT_NEED_MORE_INPUT);

    // Appends matched text to env.reasoning_content
    parser append_reasoning(const parser & p);

    // Appends matched text to env.content
    parser append_content(const parser & p);

    // Captures matched text to env.scratchpad[key]
    // If unescape_json is true, the matched text is unescaped as a JSON string
    parser capture(const parser & p, const std::string & key, bool unescape_json = false);

    // Captures matched text to env.tool_call_id
    // If unescape_json is true, the matched text is unescaped as a JSON string
    parser capture_tool_call_id(const parser & p, bool unescape_json = false);

    // Captures matched text to env.tool_call_name
    // If unescape_json is true, the matched text is unescaped as a JSON string
    parser capture_tool_call_name(const parser & p, bool unescape_json = false);

    // Captures matched text to env.tool_call_args
    // If unescape_json is true, the matched text is unescaped as a JSON string
    parser capture_tool_call_args(const parser & p, bool unescape_json = false);

    // Adds a tool call to env.tool_calls using env.tool_call_{id,name,args}
    // Clears the tool call fields after adding
    parser add_tool_call(const parser & p);

    parser add_rule(const std::string & name, const parser & p);

    void assign_ids(parser & p);

    std::shared_ptr<std::unordered_map<std::string, parser>> rules() const { return rules_; }
};

parser build_parser(const std::function<parser(parser_builder&)> & fn);
