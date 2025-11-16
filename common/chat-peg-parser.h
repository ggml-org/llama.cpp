#pragma once

#include "chat.h"

#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <unordered_map>
#include <optional>
#include <string>
#include <string_view>
#include <functional>

struct common_grammar_builder;

struct common_chat_parse_semantics {
    std::string content;
    std::string reasoning_content;
    std::vector<common_chat_tool_call> tool_calls;

    std::unordered_map<std::string, std::string> captures;

    common_chat_msg to_msg() const {
        common_chat_msg msg;
        msg.content = content;
        msg.reasoning_content = reasoning_content;
        msg.tool_calls = tool_calls;
        return msg;
    }
};

enum common_chat_parse_result_type {
    COMMON_CHAT_PARSE_RESULT_FAIL            = 1 << 0,
    COMMON_CHAT_PARSE_RESULT_SUCCESS         = 1 << 1,
    COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT = 1 << 2,
};

const char * common_chat_parse_result_type_name(common_chat_parse_result_type type);

struct common_chat_parse_cache_key {
    int id;
    size_t start;

    bool operator==(const common_chat_parse_cache_key & other) const {
        return id == other.id && start == other.start;
    }
};

template <>
struct std::hash<common_chat_parse_cache_key> {
    std::size_t operator()(const common_chat_parse_cache_key & k) const {
        return std::hash<size_t>{}(((size_t)k.id << 32) | k.start);
    }
};

struct common_chat_parse_result {
    common_chat_parse_result_type type = COMMON_CHAT_PARSE_RESULT_FAIL;
    size_t start = 0;
    size_t end = 0;

    common_chat_parse_result() : type(COMMON_CHAT_PARSE_RESULT_FAIL) {}

    common_chat_parse_result(common_chat_parse_result_type type, size_t start)
        : type(type), start(start), end(start) {}

    common_chat_parse_result(common_chat_parse_result_type type, size_t start, size_t end)
        : type(type), start(start), end(end) {}

    bool fail() const { return type == COMMON_CHAT_PARSE_RESULT_FAIL; }
    bool need_more_input() const { return type == COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT; }
    bool success() const { return type == COMMON_CHAT_PARSE_RESULT_SUCCESS; }
};

enum common_chat_parse_event_type {
    COMMON_CHAT_PARSE_EVENT_NODE_START,
    COMMON_CHAT_PARSE_EVENT_NODE_END,
};

struct common_chat_parse_event {
    common_chat_parse_event_type type;
    std::string rule;
    size_t start;
    size_t end;
    std::string_view text;
    common_chat_parse_result_type status;
    int depth;

    bool starting() const { return type == COMMON_CHAT_PARSE_EVENT_NODE_START; }
    bool ending() const { return type == COMMON_CHAT_PARSE_EVENT_NODE_END; }

    bool success() const { return status == COMMON_CHAT_PARSE_RESULT_SUCCESS; }
    bool need_more_input() const { return status == COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT; }
    bool fail() const { return status == COMMON_CHAT_PARSE_RESULT_FAIL; }
};

using common_chat_parse_event_handler = std::function<void(const common_chat_parse_event &, common_chat_parse_semantics &)>;

class common_chat_parse_cache {
    std::unordered_map<common_chat_parse_cache_key, common_chat_parse_result> results;

  public:
    common_chat_parse_result set(int id, size_t start, common_chat_parse_result result);
    std::optional<common_chat_parse_result> get(int id, size_t start);
    void clear();
};

struct common_chat_parse_context {
    std::string input;
    common_chat_parse_cache cache;
    bool input_is_complete;
    common_chat_parse_semantics * env;
    common_chat_parse_event_handler event_handler;
    int current_depth;

    common_chat_parse_context()
        : cache(), input_is_complete(true), env(nullptr), event_handler(nullptr), current_depth(0) {}

    common_chat_parse_context(const std::string & input)
        : input(input), cache(), input_is_complete(true), env(nullptr), event_handler(nullptr), current_depth(0) {}

    common_chat_parse_context(const std::string & input, bool complete)
        : input(input), cache(), input_is_complete(complete), env(nullptr), event_handler(nullptr), current_depth(0) {}

    common_chat_parse_context(const std::string & input, common_chat_parse_cache memo, bool complete = true)
        : input(input), cache(std::move(memo)), input_is_complete(complete), env(nullptr), event_handler(nullptr), current_depth(0) {}

    common_chat_parse_context(const std::string & input, common_chat_parse_semantics * environment)
        : input(input), cache(), input_is_complete(true), env(environment), event_handler(nullptr), current_depth(0) {}

    common_chat_parse_context(const std::string & input, common_chat_parse_semantics * environment, bool complete)
        : input(input), cache(), input_is_complete(complete), env(environment), event_handler(nullptr), current_depth(0) {}

    common_chat_parse_context(const std::string & input, common_chat_parse_cache memo, common_chat_parse_semantics * environment, bool complete = true)
        : input(input), cache(std::move(memo)), input_is_complete(complete), env(environment), event_handler(nullptr), current_depth(0) {}

    common_chat_parse_context(const std::string & input, common_chat_parse_semantics * environment, common_chat_parse_event_handler handler, bool complete = true)
        : input(input), cache(), input_is_complete(complete), env(environment), event_handler(std::move(handler)), current_depth(0) {}
};

class common_chat_peg_parser_base;

class common_chat_peg_parser {
    std::shared_ptr<common_chat_peg_parser_base> ptr_;

  public:
    common_chat_peg_parser();
    common_chat_peg_parser(std::shared_ptr<common_chat_peg_parser_base> parser);
    common_chat_peg_parser(const common_chat_peg_parser & other) = default;
    common_chat_peg_parser(const std::string & literal);
    common_chat_peg_parser(const char * literal);

    common_chat_peg_parser & operator=(const common_chat_peg_parser & other) {
        if (this != &other) {
            ptr_ = other.ptr_;
        }
        return *this;
    }

    common_chat_peg_parser operator~() const;
    common_chat_peg_parser operator+(const common_chat_peg_parser & other) const;
    common_chat_peg_parser operator|(const common_chat_peg_parser & other) const;
    common_chat_peg_parser operator<<(const common_chat_peg_parser & other) const;

    common_chat_peg_parser_base & operator*() const;
    common_chat_peg_parser_base * operator->() const;

    std::shared_ptr<common_chat_peg_parser_base> ptr() const { return ptr_; }

    common_chat_parse_result parse(common_chat_parse_context & ctx, size_t start = 0) const;

    std::string dump() const;

    void build_grammar(const common_grammar_builder & builder, bool lazy = false) const;
};

common_chat_peg_parser operator+(const char * lhs, const common_chat_peg_parser & rhs);
common_chat_peg_parser operator|(const char * lhs, const common_chat_peg_parser & rhs);
common_chat_peg_parser operator<<(const char * lhs, const common_chat_peg_parser & rhs);

common_chat_peg_parser operator+(const std::string & lhs, const common_chat_peg_parser & rhs);
common_chat_peg_parser operator|(const std::string & lhs, const common_chat_peg_parser & rhs);
common_chat_peg_parser operator<<(const std::string & lhs, const common_chat_peg_parser & rhs);

class common_chat_peg_parser_counter {
    int next_id_;
  public:
    common_chat_peg_parser_counter(int start) : next_id_(start) {}
    int next() { return next_id_++; }
};

class common_chat_peg_parser_builder {
    common_chat_peg_parser root_;
    common_chat_peg_parser_counter counter_;

  public:
    common_chat_peg_parser_builder();

    // Matches the start of the input.
    //   S -> ^
    common_chat_peg_parser start();

    // Matches the end of the input.
    //   S -> $
    common_chat_peg_parser end();

    // Matches an exact literal string.
    //   S -> "hello"
    common_chat_peg_parser literal(const std::string & literal);

    // Matches a sequence of parsers in order, all must succeed.
    //   S -> A B C
    common_chat_peg_parser sequence(const std::vector<common_chat_peg_parser> & parsers);

    // Matches the first parser that succeeds from a list of alternatives.
    //   S -> A | B | C
    common_chat_peg_parser choice(const std::vector<common_chat_peg_parser> & parsers);

    // Matches one or more repetitions of a parser.
    //   S -> A+
    common_chat_peg_parser one_or_more(const common_chat_peg_parser & p);

    // Matches zero or more repetitions of a parser, always succeeds.
    //   S -> A*
    common_chat_peg_parser zero_or_more(const common_chat_peg_parser & p);

    // Matches zero or one occurrence of a parser, always succeeds.
    //   S -> A?
    common_chat_peg_parser optional(const common_chat_peg_parser & p);

    // Negative lookahead: succeeds if child parser fails, consumes no input.
    //   S -> !A
    common_chat_peg_parser peek(const common_chat_peg_parser & p);

    // Negative lookahead: succeeds if child parser fails, consumes no input.
    //   S -> !A
    common_chat_peg_parser negate(const common_chat_peg_parser & p);

    // Matches any single character.
    //   S -> .
    common_chat_peg_parser any();

    // Matches between min and max repetitions of characters from a character class.
    //   S -> [a-z]{m,n}
    //
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_chat_peg_parser chars(const std::string & classes, int min = 1, int max = -1);

    // Matches a single character from a character class or range.
    //   S -> [a-z] or S -> [^0-9]
    //
    // Equivalent to chars(classes, 1, 1)
    common_chat_peg_parser one(const std::string & classes);

    // References a named rule for recursive or reusable grammar definitions.
    //   expr -> term | expr "+" term
    common_chat_peg_parser rule(const std::string & name);

    // Matches zero or more whitespace characters (space, tab, newline).
    //   S -> [ \t\n]*
    common_chat_peg_parser space();

    // Matches all characters until a delimiter is found (delimiter not consumed).
    //   S -> (!delim .)*
    common_chat_peg_parser until(const std::string & delimiter);
    common_chat_peg_parser until_one_of(const std::vector<std::string> & delimiters);

    // Matches between min and max repetitions of a parser (inclusive).
    //   S -> A{m,n}
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_chat_peg_parser repeat(const common_chat_peg_parser & p, int min, int max);

    // Matches exactly n repetitions of a parser.
    //   S -> A{n}
    common_chat_peg_parser repeat(const common_chat_peg_parser & p, int n);

    // Creates a complete JSON parser supporting objects, arrays, strings, numbers, booleans, and null.
    //   value -> object | array | string | number | true | false | null
    common_chat_peg_parser json();
    common_chat_peg_parser json_object();
    common_chat_peg_parser json_string();
    common_chat_peg_parser json_array();
    common_chat_peg_parser json_number();
    common_chat_peg_parser json_bool();
    common_chat_peg_parser json_null();

    // Specialized single-pass JSON string parser with escape sequence handling
    common_chat_peg_parser json_string_unqouted();

    // Wraps a parser with JSON schema metadata for grammar generation.
    // Used internally to convert JSON schemas to GBNF grammar rules.
    common_chat_peg_parser schema(const common_chat_peg_parser & p, const std::string & name, const nlohmann::ordered_json & schema);

    // Captures matched text to semantics.captures[key]
    common_chat_peg_parser capture(const std::string & key, const common_chat_peg_parser & p);

    // Mark a node as a trigger for GBNF grammar generartion. This is used for
    // lazy grammar evaluation by only producing GBNF grammar rules that are
    // reachable from trigger nodes.
    //   S -> Trigger(A)
    common_chat_peg_parser trigger(const common_chat_peg_parser & p);

    // Adds a named rule and returns a rule reference.
    common_chat_peg_parser add_rule(const std::string & name, const common_chat_peg_parser & p);

    // Adds a named rule using a function. This handles recursive grammars by
    // inserting a placeholder rule before invoking the builder, allowing the
    // builder to reference the rule being defined. Use this when the rule
    // definition needs to call back to itself (directly or indirectly).
    //   add_rule("json", [&]() { return json_object() | json_array() | ... })
    common_chat_peg_parser add_rule(const std::string & name, const std::function<common_chat_peg_parser()> & builder);

    void set_root(const common_chat_peg_parser & p);

    // Helper methods for common patterns

    // Adds raw-reasoning for the entire reasoning block plus reasoning-content for the contents, by default thinking tag is "think"
    common_chat_peg_parser reasoning(const std::string & tag = "think");

    // Adds main content block before tool call block, due to the varied nature of tool call openers (not always XML-like) full tag is required
    common_chat_peg_parser content_before_tools(const std::string &tag);

    // Adds a quasi-XML tool call spec without a separate name attribute (Qwen3 style);
    // TODO: accept parameter schemas (required, value types etc.)
    common_chat_peg_parser quasi_xml_no_attr(const std::string &function_name, const std::vector<std::string> &parameters,
        const std::string &function_tag = "function", const std::string &param_tag = "parameter");

    // Adds a quasi-XML tool call spec with a separate name attribute (Minimax-M2 style)
    // TODO: accept parameter schemas (required, value types etc.)
    // common_chat_peg_parser quasi_xml_attr(const std::string &function_name, const std::vector<std::string> &parameters,
    //     const std::string &function_tag = "invoke", const std::string &param_tag = "parameter",
    //     const std::string &name_attr = "name");

    common_chat_peg_parser build();
};

common_chat_peg_parser build_peg_parser(const std::function<common_chat_peg_parser(common_chat_peg_parser_builder&)> & fn);
