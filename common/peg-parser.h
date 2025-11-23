#pragma once

#include "chat.h"

#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <unordered_map>
#include <string>
#include <string_view>
#include <functional>
#include <vector>
#include <variant>

struct common_grammar_builder;

// Forward declarations
using common_peg_parser_id = size_t;
constexpr common_peg_parser_id COMMON_PEG_INVALID_PARSER_ID = static_cast<common_peg_parser_id>(-1);

using common_peg_ast_id = size_t;
constexpr common_peg_ast_id COMMON_PEG_INVALID_AST_ID = static_cast<common_peg_ast_id>(-1);

// Forward declare builder for parser wrapper
class common_peg_parser_builder;

// Lightweight wrapper around common_peg_parser_id that enables operator overloading
// and implicit conversions from strings/literals
class common_peg_parser {
    common_peg_parser_id id_;
    common_peg_parser_builder * builder_;

  public:
    // Construct from common_peg_parser_id
    common_peg_parser(common_peg_parser_id id, common_peg_parser_builder * builder) : id_(id), builder_(builder) {}

    // Implicit conversion to common_peg_parser_id
    operator common_peg_parser_id() const { return id_; }

    // Get the underlying ID
    common_peg_parser_id id() const { return id_; }

    // Get builder (for free function operators)
    common_peg_parser_builder * builder() const { return builder_; }

    // Operator overloads
    common_peg_parser operator+(const common_peg_parser & other) const;
    common_peg_parser operator|(const common_peg_parser & other) const;
    common_peg_parser operator<<(const common_peg_parser & other) const;  // sequence with space

    // Overloads for string literals
    common_peg_parser operator+(const char * str) const;
    common_peg_parser operator+(const std::string & str) const;
    common_peg_parser operator|(const char * str) const;
    common_peg_parser operator|(const std::string & str) const;
    common_peg_parser operator<<(const char * str) const;
    common_peg_parser operator<<(const std::string & str) const;
};

// Free function operators for string + parser
common_peg_parser operator+(const char * str, const common_peg_parser & p);
common_peg_parser operator+(const std::string & str, const common_peg_parser & p);
common_peg_parser operator<<(const char * str, const common_peg_parser & p);
common_peg_parser operator<<(const std::string & str, const common_peg_parser & p);

enum common_peg_parse_result_type {
    COMMON_PEG_PARSE_RESULT_FAIL            = 0,
    COMMON_PEG_PARSE_RESULT_SUCCESS         = 1,
    COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT = 2,
};

const char * common_peg_parse_result_type_name(common_peg_parse_result_type type);

struct common_peg_ast_node {
    common_peg_ast_id id;
    std::string rule_name;
    std::string tag;
    size_t start;
    size_t end;
    std::string_view text;
    std::vector<common_peg_ast_id> children;

    bool is_partial = false;
};

struct common_peg_parse_result;

using common_peg_ast_visitor = std::function<void(const common_peg_ast_node & node)>;

class common_peg_ast_arena {
    std::vector<common_peg_ast_node> nodes_;
  public:
    common_peg_ast_id add_node(
        const std::string & rule_name,
        const std::string & tag,
        size_t start,
        size_t end,
        std::string_view text,
        std::vector<common_peg_ast_id> children,
        bool is_partial = false
    ) {
        common_peg_ast_id id = nodes_.size();
        nodes_.push_back({id, rule_name, tag, start, end, text, std::move(children), is_partial});
        return id;
    }

    const common_peg_ast_node & get(common_peg_ast_id id) const { return nodes_.at(id); }

    size_t size() const { return nodes_.size(); }

    void clear() { nodes_.clear(); }

    void visit(common_peg_ast_id id, const common_peg_ast_visitor & visitor) const;
    void visit(const common_peg_parse_result & result, const common_peg_ast_visitor & visitor) const;
};

struct common_peg_parse_result {
    common_peg_parse_result_type type = COMMON_PEG_PARSE_RESULT_FAIL;
    size_t start = 0;
    size_t end = 0;

    std::vector<common_peg_ast_id> nodes;

    common_peg_parse_result() : type(COMMON_PEG_PARSE_RESULT_FAIL) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start)
        : type(type), start(start), end(start) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start, size_t end)
        : type(type), start(start), end(end) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start, size_t end, std::vector<common_peg_ast_id> nodes)
        : type(type), start(start), end(end), nodes(std::move(nodes)) {}

    bool fail() const { return type == COMMON_PEG_PARSE_RESULT_FAIL; }
    bool need_more_input() const { return type == COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT; }
    bool success() const { return type == COMMON_PEG_PARSE_RESULT_SUCCESS; }
};

struct common_peg_parse_context {
    std::string input;
    bool input_is_complete;
    common_peg_ast_arena ast;

    int parse_depth;

    common_peg_parse_context()
        : input_is_complete(true), parse_depth(0) {}

    common_peg_parse_context(const std::string & input)
        : input(input), input_is_complete(true), parse_depth(0) {}

    common_peg_parse_context(const std::string & input, bool complete)
        : input(input), input_is_complete(complete), parse_depth(0) {}
};

// Forward declaration
class common_peg_arena;

// Parser variant structs (value-based, no inheritance)
struct common_peg_epsilon_parser {};

struct common_peg_start_parser {};

struct common_peg_end_parser {};

struct common_peg_literal_parser {
    std::string literal;
};

struct common_peg_sequence_parser {
    std::vector<common_peg_parser_id> children;
};

struct common_peg_choice_parser {
    std::vector<common_peg_parser_id> children;
};

struct common_peg_repetition_parser {
    common_peg_parser_id child;
    int min_count;
    int max_count;  // -1 for unbounded
};

struct common_peg_and_parser {
    common_peg_parser_id child;
};

struct common_peg_not_parser {
    common_peg_parser_id child;
};

struct common_peg_any_parser {};

struct common_peg_space_parser {};

struct common_peg_chars_parser {
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

struct common_peg_json_string_parser {};

struct common_peg_until_parser {
    std::vector<std::string> delimiters;
};

struct common_peg_schema_parser {
    common_peg_parser_id child;
    std::string name;
    std::shared_ptr<nlohmann::ordered_json> schema;
    bool raw;
};

struct common_peg_rule_parser {
    std::string name;
    common_peg_parser_id child;
    bool trigger;
};

struct common_peg_ref_parser {
    std::string name;
};

struct common_peg_capture_parser {
    common_peg_parser_id child;
    std::string key;
};

struct common_peg_atomic_parser {
    common_peg_parser_id child;
};

struct common_peg_tag_parser {
    common_peg_parser_id child;
    std::string tag;
};

// Variant holding all parser types
using common_peg_parser_variant = std::variant<
    common_peg_epsilon_parser,
    common_peg_start_parser,
    common_peg_end_parser,
    common_peg_literal_parser,
    common_peg_sequence_parser,
    common_peg_choice_parser,
    common_peg_repetition_parser,
    common_peg_and_parser,
    common_peg_not_parser,
    common_peg_any_parser,
    common_peg_space_parser,
    common_peg_chars_parser,
    common_peg_json_string_parser,
    common_peg_until_parser,
    common_peg_schema_parser,
    common_peg_rule_parser,
    common_peg_ref_parser,
    common_peg_capture_parser,
    common_peg_atomic_parser,
    common_peg_tag_parser
>;

// Arena owns all parsers
class common_peg_arena {
    std::vector<common_peg_parser_variant> parsers_;
    std::unordered_map<std::string, common_peg_parser_id> rules_;
    common_peg_parser_id root_;

  public:
    common_peg_arena();

    // Access
    const common_peg_parser_variant & get(common_peg_parser_id id) const { return parsers_.at(id); }
    common_peg_parser_variant & get(common_peg_parser_id id) { return parsers_.at(id); }

    size_t size() const { return parsers_.size(); }
    bool empty() const { return parsers_.empty(); }

    // Rule lookup
    common_peg_parser_id get_rule(const std::string & name) const;
    bool has_rule(const std::string & name) const { return rules_.find(name) != rules_.end(); }

    // Root
    common_peg_parser_id root() const { return root_; }
    void set_root(common_peg_parser_id id) { root_ = id; }

    // Parse
    common_peg_parse_result parse(common_peg_parse_context & ctx, size_t start = 0) const;
    common_peg_parse_result parse(common_peg_parser_id id, common_peg_parse_context & ctx, size_t start) const;

    // Resolve all ref parsers to point directly to their corresponding rule parsers
    void resolve_refs();

    // Grammar generation
    void build_grammar(const common_grammar_builder & builder, bool lazy = false) const;

    // Dump for debugging
    std::string dump(common_peg_parser_id id) const;

    // Serialization
    nlohmann::json to_json() const;
    static common_peg_arena from_json(const nlohmann::json & j);

    std::string serialize() const;
    static common_peg_arena deserialize(const std::string & data);

    // Builder access (for adding parsers)
    friend class common_peg_parser_builder;

  private:
    common_peg_parser_id add_parser(common_peg_parser_variant parser);
    void add_rule(const std::string & name, common_peg_parser_id id);

    common_peg_parser_id resolve_ref(common_peg_parser_id id);
};

// Builder for constructing parsers
class common_peg_parser_builder {
    common_peg_arena arena_;

    common_peg_parser wrap(common_peg_parser_id id) { return common_peg_parser(id, this); }
    common_peg_parser add(const common_peg_parser_variant & p) { return wrap(arena_.add_parser(p)); }

  public:
    common_peg_parser_builder();

    // Match nothing, always succeed.
    //   S -> ε
    common_peg_parser eps() { return add(common_peg_epsilon_parser{}); }

    // Matches the start of the input.
    //   S -> ^
    common_peg_parser start() { return add(common_peg_start_parser{}); }

    // Matches the end of the input.
    //   S -> $
    common_peg_parser end() { return add(common_peg_end_parser{}); }

    // Matches an exact literal string.
    //   S -> "hello"
    common_peg_parser literal(const std::string & literal) { return add(common_peg_literal_parser{literal}); }

    // Matches a sequence of parsers in order, all must succeed.
    //   S -> A B C
    common_peg_parser sequence(const std::vector<common_peg_parser_id> & parsers);
    common_peg_parser sequence(const std::vector<common_peg_parser> & parsers);
    common_peg_parser sequence(std::initializer_list<common_peg_parser> parsers);

    // Matches the first parser that succeeds from a list of alternatives.
    //   S -> A | B | C
    common_peg_parser choice(const std::vector<common_peg_parser_id> & parsers);
    common_peg_parser choice(const std::vector<common_peg_parser> & parsers);
    common_peg_parser choice(std::initializer_list<common_peg_parser> parsers);

    // Matches one or more repetitions of a parser.
    //   S -> A+
    common_peg_parser one_or_more(common_peg_parser p) { return repeat(p, 1, -1); }

    // Matches zero or more repetitions of a parser, always succeeds.
    //   S -> A*
    common_peg_parser zero_or_more(common_peg_parser p) { return repeat(p, 0, -1); }

    // Matches zero or one occurrence of a parser, always succeeds.
    //   S -> A?
    common_peg_parser optional(common_peg_parser p) { return repeat(p, 0, 1); }

    // Positive lookahead: succeeds if child parser succeeds, consumes no input.
    //   S -> &A
    common_peg_parser peek(common_peg_parser p) { return add(common_peg_and_parser{p}); }

    // Negative lookahead: succeeds if child parser fails, consumes no input.
    //   S -> !A
    common_peg_parser negate(common_peg_parser p) { return add(common_peg_not_parser{p}); }

    // Matches any single character.
    //   S -> .
    common_peg_parser any() { return add(common_peg_any_parser{}); }

    // Matches between min and max repetitions of characters from a character class.
    //   S -> [a-z]{m,n}
    //
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_peg_parser chars(const std::string & classes, int min = 1, int max = -1);

    // Matches a single character from a character class or range.
    //   S -> [a-z] or S -> [^0-9]
    //
    // Equivalent to chars(classes, 1, 1)
    common_peg_parser one(const std::string & classes) { return chars(classes, 1, 1); }

    // Creates a lightweight reference to a named rule (resolved during build()).
    // Use this for forward references in recursive grammars.
    //   expr_ref -> expr
    common_peg_parser ref(const std::string & name) { return add(common_peg_ref_parser{name}); }

    // Matches zero or more whitespace characters (space, tab, newline).
    //   S -> [ \t\n]*
    common_peg_parser space() { return add(common_peg_space_parser{}); }

    // Matches all characters until a delimiter is found (delimiter not consumed).
    //   S -> (!delim .)*
    common_peg_parser until(const std::string & delimiter) { return add(common_peg_until_parser{{delimiter}}); }
    common_peg_parser until_one_of(const std::vector<std::string> & delimiters) { return add(common_peg_until_parser{delimiters}); }

    // Matches everything
    //   S -> .*
    common_peg_parser rest() { return until_one_of({}); }

    // Matches between min and max repetitions of a parser (inclusive).
    //   S -> A{m,n}
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_peg_parser repeat(common_peg_parser p, int min, int max) { return add(common_peg_repetition_parser{p, min,max}); }

    // Matches exactly n repetitions of a parser.
    //   S -> A{n}
    common_peg_parser repeat(common_peg_parser p, int n) { return repeat(p, n, n); }

    // Creates a complete JSON parser supporting objects, arrays, strings, numbers, booleans, and null.
    //   value -> object | array | string | number | true | false | null
    common_peg_parser json();
    common_peg_parser json_object();
    common_peg_parser json_string();
    common_peg_parser json_array();
    common_peg_parser json_number();
    common_peg_parser json_bool();
    common_peg_parser json_null();

    // Specialized single-pass JSON string parser with escape sequence handling
    common_peg_parser json_string_content();

    // Wraps a parser with JSON schema metadata for grammar generation.
    // Used internally to convert JSON schemas to GBNF grammar rules.
    common_peg_parser schema(common_peg_parser p, const std::string & name, const nlohmann::ordered_json & schema, bool raw = false);

    // Captures matched text to semantics.captures[key]
    common_peg_parser capture(const std::string & key, common_peg_parser p);

    // Creates a named rule, stores it in the grammar, and returns a reference to it.
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", json_obj | json_arr | ...)
    common_peg_parser rule(const std::string & name, common_peg_parser p, bool trigger = false);

    // Creates a named rule using a builder function. This handles recursive grammars by
    // inserting a placeholder rule before invoking the builder, allowing the
    // builder to reference the rule being defined via ref(). Use this when the rule
    // definition needs to call back to itself (directly or indirectly).
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", [&]() { return json_object() | json_array() | ... })
    common_peg_parser rule(const std::string & name, const std::function<common_peg_parser()> & builder, bool trigger = false);

    // Creates a trigger rule. When generating a lazy grammar from the parser,
    // only trigger rules and descendents are emitted.
    common_peg_parser trigger_rule(const std::string & name, common_peg_parser p) { return rule(name, p, true); }
    common_peg_parser trigger_rule(const std::string & name, const std::function<common_peg_parser()> & builder) { return rule(name, builder, true); }

    // Creates an atomic parser. Atomic parsers do not create an AST node if
    // the child results in a partial parse, i.e. NEEDS_MORE_INPUT. This is
    // intended for situations where partial output is undesirable.
    common_peg_parser atomic(common_peg_parser p) { return add(common_peg_atomic_parser{p}); }

    // Tags create nodes in the generated AST for semantic purposes.
    // Unlike rules, you can tag multiple nodes with the same tag.
    common_peg_parser tag(const std::string & tag, common_peg_parser p) { return add(common_peg_tag_parser{p.id(), tag}); }

    void set_root(common_peg_parser p);

    common_peg_arena build();
};

// Helper function for building parsers
template<typename F>
common_peg_arena build_peg_parser(F && fn) {
    common_peg_parser_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}
