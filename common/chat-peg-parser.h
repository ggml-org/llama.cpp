#pragma once

#include "chat.h"

#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <unordered_map>
#include <optional>
#include <string>
#include <string_view>
#include <functional>
#include <vector>
#include <variant>

struct common_grammar_builder;

// Forward declarations
using common_chat_peg_parser_id = size_t;
constexpr common_chat_peg_parser_id COMMON_CHAT_PEG_INVALID_PARSER_ID = static_cast<common_chat_peg_parser_id>(-1);

// Forward declare builder for parser wrapper
class common_chat_peg_parser_builder;

// Lightweight wrapper around common_chat_peg_parser_id that enables operator overloading
// and implicit conversions from strings/literals
class common_chat_peg_parser {
    common_chat_peg_parser_id id_;
    common_chat_peg_parser_builder * builder_;

  public:
    // Construct from common_chat_peg_parser_id
    common_chat_peg_parser(common_chat_peg_parser_id id, common_chat_peg_parser_builder * builder) : id_(id), builder_(builder) {}

    // Implicit conversion to common_chat_peg_parser_id
    operator common_chat_peg_parser_id() const { return id_; }

    // Get the underlying ID
    common_chat_peg_parser_id id() const { return id_; }

    // Get builder (for free function operators)
    common_chat_peg_parser_builder * builder() const { return builder_; }

    // Operator overloads
    common_chat_peg_parser operator+(const common_chat_peg_parser & other) const;
    common_chat_peg_parser operator|(const common_chat_peg_parser & other) const;
    common_chat_peg_parser operator<<(const common_chat_peg_parser & other) const;  // sequence with space

    // Overloads for string literals
    common_chat_peg_parser operator+(const char * str) const;
    common_chat_peg_parser operator+(const std::string & str) const;
    common_chat_peg_parser operator|(const char * str) const;
    common_chat_peg_parser operator|(const std::string & str) const;
    common_chat_peg_parser operator<<(const char * str) const;
    common_chat_peg_parser operator<<(const std::string & str) const;
};

// Free function operators for string + parser
common_chat_peg_parser operator+(const char * str, const common_chat_peg_parser & p);
common_chat_peg_parser operator+(const std::string & str, const common_chat_peg_parser & p);
common_chat_peg_parser operator<<(const char * str, const common_chat_peg_parser & p);
common_chat_peg_parser operator<<(const std::string & str, const common_chat_peg_parser & p);

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
    COMMON_CHAT_PARSE_RESULT_FAIL            = 0,
    COMMON_CHAT_PARSE_RESULT_SUCCESS         = 1,
    COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT = 2,
};

const char * common_chat_parse_result_type_name(common_chat_parse_result_type type);

struct common_chat_parse_cache_key {
    common_chat_peg_parser_id id;
    size_t start;

    bool operator==(const common_chat_parse_cache_key & other) const {
        return id == other.id && start == other.start;
    }
};

template <>
struct std::hash<common_chat_parse_cache_key> {
    std::size_t operator()(const common_chat_parse_cache_key & k) const {
        return std::hash<size_t>{}((k.id << 32) | k.start);
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
    std::string annotation;
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
    common_chat_parse_result set(common_chat_peg_parser_id id, size_t start, common_chat_parse_result result);
    std::optional<common_chat_parse_result> get(common_chat_peg_parser_id id, size_t start);
    void clear();
};

struct common_chat_parse_context {
    std::string input;
    bool input_is_complete;
    common_chat_parse_cache cache;
    common_chat_parse_semantics * semantics;
    common_chat_parse_event_handler event_handler;

    int current_depth;
    int parse_depth;

    common_chat_parse_context()
        : input_is_complete(true), cache(), semantics(nullptr), event_handler(nullptr), current_depth(0), parse_depth(0) {}

    common_chat_parse_context(const std::string & input)
        : input(input), input_is_complete(true), cache(), semantics(nullptr), event_handler(nullptr), current_depth(0), parse_depth(0) {}

    common_chat_parse_context(const std::string & input, bool complete)
        : input(input), input_is_complete(complete), cache(), semantics(nullptr), event_handler(nullptr), current_depth(0), parse_depth(0) {}

    common_chat_parse_context(const std::string & input, common_chat_parse_semantics * semantics)
        : input(input), input_is_complete(true), cache(), semantics(semantics), event_handler(nullptr), current_depth(0), parse_depth(0) {}

    common_chat_parse_context(const std::string & input, common_chat_parse_semantics * semantics, bool complete)
        : input(input), input_is_complete(complete), cache(), semantics(semantics), event_handler(nullptr), current_depth(0), parse_depth(0) {}

    common_chat_parse_context(const std::string & input, common_chat_parse_semantics * semantics, common_chat_parse_event_handler handler, bool complete = true)
        : input(input), input_is_complete(complete), cache(), semantics(semantics), event_handler(std::move(handler)), current_depth(0), parse_depth(0) {}
};

// Forward declaration
class common_chat_peg_arena;

// Parser variant structs (value-based, no inheritance)
struct common_chat_peg_start_parser {};

struct common_chat_peg_end_parser {};

struct common_chat_peg_literal_parser {
    std::string literal;
};

struct common_chat_peg_sequence_parser {
    std::vector<common_chat_peg_parser_id> children;
};

struct common_chat_peg_choice_parser {
    std::vector<common_chat_peg_parser_id> children;
};

struct common_chat_peg_repetition_parser {
    common_chat_peg_parser_id child;
    int min_count;
    int max_count;  // -1 for unbounded
};

struct common_chat_peg_one_or_more_parser {
    common_chat_peg_parser_id child;
};

struct common_chat_peg_zero_or_more_parser {
    common_chat_peg_parser_id child;
};

struct common_chat_peg_optional_parser {
    common_chat_peg_parser_id child;
};

struct common_chat_peg_and_parser {
    common_chat_peg_parser_id child;
};

struct common_chat_peg_not_parser {
    common_chat_peg_parser_id child;
};

struct common_chat_peg_any_parser {};

struct common_chat_peg_space_parser {};

struct common_chat_peg_chars_parser {
    struct char_range {
        uint32_t start;
        uint32_t end;
        bool contains(uint32_t codepoint) const { return codepoint >= start && codepoint <= end; }
    };

    std::string pattern;
    std::vector<char_range> ranges;
    bool negated;
    int min_count;
    int max_count;  // -1 for unbounded
};

struct common_chat_peg_json_string_parser {};

struct common_chat_peg_until_parser {
    std::vector<std::string> delimiters;
};

struct common_chat_peg_schema_parser {
    common_chat_peg_parser_id child;
    std::string name;
    std::shared_ptr<nlohmann::ordered_json> schema;
};

struct common_chat_peg_rule_parser {
    std::string name;
    std::string annotation;
    common_chat_peg_parser_id child;
    bool trigger;
};

struct common_chat_peg_ref_parser {
    std::string name;
};

struct common_chat_peg_capture_parser {
    common_chat_peg_parser_id child;
    std::string key;
};

// Variant holding all parser types
using common_chat_peg_parser_variant = std::variant<
    common_chat_peg_start_parser,
    common_chat_peg_end_parser,
    common_chat_peg_literal_parser,
    common_chat_peg_sequence_parser,
    common_chat_peg_choice_parser,
    common_chat_peg_repetition_parser,
    common_chat_peg_one_or_more_parser,
    common_chat_peg_zero_or_more_parser,
    common_chat_peg_optional_parser,
    common_chat_peg_and_parser,
    common_chat_peg_not_parser,
    common_chat_peg_any_parser,
    common_chat_peg_space_parser,
    common_chat_peg_chars_parser,
    common_chat_peg_json_string_parser,
    common_chat_peg_until_parser,
    common_chat_peg_schema_parser,
    common_chat_peg_rule_parser,
    common_chat_peg_ref_parser,
    common_chat_peg_capture_parser
>;

// Arena owns all parsers
class common_chat_peg_arena {
    std::vector<common_chat_peg_parser_variant> parsers_;
    std::unordered_map<std::string, common_chat_peg_parser_id> rules_;
    common_chat_peg_parser_id root_;

  public:
    common_chat_peg_arena();

    // Access
    const common_chat_peg_parser_variant & get(common_chat_peg_parser_id id) const { return parsers_.at(id); }
    common_chat_peg_parser_variant & get(common_chat_peg_parser_id id) { return parsers_.at(id); }

    size_t size() const { return parsers_.size(); }

    // Rule lookup
    common_chat_peg_parser_id get_rule(const std::string & name) const;
    bool has_rule(const std::string & name) const { return rules_.find(name) != rules_.end(); }

    // Root
    common_chat_peg_parser_id root() const { return root_; }
    void set_root(common_chat_peg_parser_id id) { root_ = id; }

    // Parse
    common_chat_parse_result parse(common_chat_parse_context & ctx, size_t start = 0) const;
    common_chat_parse_result parse(common_chat_peg_parser_id id, common_chat_parse_context & ctx, size_t start) const;

    // Grammar generation
    void build_grammar(const common_grammar_builder & builder, bool lazy = false) const;

    // Dump for debugging
    std::string dump(common_chat_peg_parser_id id) const;

    // Builder access (for adding parsers)
    friend class common_chat_peg_parser_builder;

  private:
    common_chat_peg_parser_id add_parser(common_chat_peg_parser_variant parser);
    void add_rule(const std::string & name, common_chat_peg_parser_id id);
};

// Builder for constructing parsers
class common_chat_peg_parser_builder {
    common_chat_peg_arena arena_;

    // Helper to wrap common_chat_peg_parser_id with this builder
    common_chat_peg_parser wrap(common_chat_peg_parser_id id) { return common_chat_peg_parser(id, this); }

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

    // Implicit conversion: const char* -> parser (literal)
    common_chat_peg_parser operator()(const char * str) { return literal(str); }

    // Implicit conversion: std::string -> parser (literal)
    common_chat_peg_parser operator()(const std::string & str) { return literal(str); }

    // Matches a sequence of parsers in order, all must succeed.
    //   S -> A B C
    common_chat_peg_parser sequence(const std::vector<common_chat_peg_parser_id> & parsers);
    common_chat_peg_parser sequence(const std::vector<common_chat_peg_parser> & parsers);
    common_chat_peg_parser sequence(std::initializer_list<common_chat_peg_parser> parsers);

    // Matches the first parser that succeeds from a list of alternatives.
    //   S -> A | B | C
    common_chat_peg_parser choice(const std::vector<common_chat_peg_parser_id> & parsers);
    common_chat_peg_parser choice(const std::vector<common_chat_peg_parser> & parsers);
    common_chat_peg_parser choice(std::initializer_list<common_chat_peg_parser> parsers);

    // Matches one or more repetitions of a parser.
    //   S -> A+
    common_chat_peg_parser one_or_more(common_chat_peg_parser p);

    // Matches zero or more repetitions of a parser, always succeeds.
    //   S -> A*
    common_chat_peg_parser zero_or_more(common_chat_peg_parser p);

    // Matches zero or one occurrence of a parser, always succeeds.
    //   S -> A?
    common_chat_peg_parser optional(common_chat_peg_parser p);

    // Positive lookahead: succeeds if child parser succeeds, consumes no input.
    //   S -> &A
    common_chat_peg_parser peek(common_chat_peg_parser p);

    // Negative lookahead: succeeds if child parser fails, consumes no input.
    //   S -> !A
    common_chat_peg_parser negate(common_chat_peg_parser p);

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

    // Creates a lightweight reference to a named rule (resolved during build()).
    // Use this for forward references in recursive grammars.
    //   expr_ref -> expr
    common_chat_peg_parser ref(const std::string & name);

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
    common_chat_peg_parser repeat(common_chat_peg_parser p, int min, int max);

    // Matches exactly n repetitions of a parser.
    //   S -> A{n}
    common_chat_peg_parser repeat(common_chat_peg_parser p, int n);

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
    common_chat_peg_parser json_string_content();

    // Wraps a parser with JSON schema metadata for grammar generation.
    // Used internally to convert JSON schemas to GBNF grammar rules.
    common_chat_peg_parser schema(common_chat_peg_parser p, const std::string & name, const nlohmann::ordered_json & schema);

    // Captures matched text to semantics.captures[key]
    common_chat_peg_parser capture(const std::string & key, common_chat_peg_parser p);

    // Creates a named rule, stores it in the grammar, and returns a reference to it.
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", json_obj | json_arr | ...)
    common_chat_peg_parser rule(const std::string & name, common_chat_peg_parser p, bool trigger = false);
    common_chat_peg_parser rule(const std::string & name, const std::string & annotation, common_chat_peg_parser p, bool trigger = false);

    // Creates a named rule using a builder function. This handles recursive grammars by
    // inserting a placeholder rule before invoking the builder, allowing the
    // builder to reference the rule being defined via ref(). Use this when the rule
    // definition needs to call back to itself (directly or indirectly).
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", [&]() { return json_object() | json_array() | ... })
    common_chat_peg_parser rule(const std::string & name, const std::function<common_chat_peg_parser()> & builder, bool trigger = false);
    common_chat_peg_parser rule(const std::string & name, const std::string & annotation, const std::function<common_chat_peg_parser()> & builder, bool trigger = false);

    void set_root(common_chat_peg_parser p);

    common_chat_peg_arena build();
};

// Helper function for building parsers
template<typename F>
common_chat_peg_arena build_peg_parser(F && fn) {
    common_chat_peg_parser_builder builder;
    auto root = fn(builder);
    builder.set_root(root);
    return builder.build();
}
