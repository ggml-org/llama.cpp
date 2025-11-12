#include "chat-parser-combinator.h"
#include "json-schema-to-grammar.h"
#include "common.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <optional>

enum parser_type {
    PARSER_LITERAL = 0,
    PARSER_SEQUENCE = 1,
    PARSER_CHOICE = 2,
    PARSER_REPETITION = 3,
    PARSER_OPTIONAL = 4,
    PARSER_ZERO_OR_MORE = 5,
    PARSER_ONE_OR_MORE = 6,
    PARSER_NOT = 7,
    PARSER_ANY = 8,
    PARSER_CHAR_CLASS = 9,
    PARSER_GROUP = 10,
    PARSER_RULE = 11,
    PARSER_UNTIL = 12,
    PARSER_SPACE = 13,
    PARSER_SCHEMA = 14,
    PARSER_ROOT = 15,
};

class parser_visitor;

class parser_base {
  protected:
    int id_;

  public:
    parser_base(int id) : id_(id) {}
    virtual ~parser_base() = default;

    int id() const { return id_; }
    void set_id(int id) { id_ = id; }

    virtual parser_type type() const = 0;

    // Template Method: handles caching, delegates to parse_uncached()
    virtual parser_result parse(parser_context & ctx, size_t start = 0) {
        if (id_ == -1) {
            // Don't cache parsers with ID -1 (from operators)
            return parse_uncached(ctx, start);
        }

        // Check cache
        auto cached = ctx.memo.get(id_, start);
        if (cached) {
            return *cached;
        }

        // Execute and cache
        auto result = parse_uncached(ctx, start);
        return ctx.memo.set(id_, start, result);
    }

    // Actual parsing implementation (to be overridden by subclasses)
    virtual parser_result parse_uncached(parser_context & ctx, size_t start = 0) = 0;

    virtual std::string dump() const = 0;
    virtual void accept(parser_visitor & visitor) = 0;
};

// We define our own space function because MSVC's std::isspace()
// crashes for non-printable characters in Debug builds.
static bool is_space(const char c) {
    return (c == ' ' || c == '\t' || c == '\n');
}

// Matches an exact literal string.
//   S -> "hello"
class literal_parser : public parser_base {
    std::string literal_;

  public:
    literal_parser(const std::string & literal, int id) : parser_base(id), literal_(literal) {}

    parser_type type() const override { return PARSER_LITERAL; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto pos = start;
        for (auto i = 0u; i < literal_.size(); ++i) {
            if (pos >= ctx.input.size()) {
                if (ctx.input_is_complete) {
                    return parser_result(PARSER_RESULT_FAIL, start);
                }
                if (i > 0) {
                    return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, pos);
                }
                return parser_result(PARSER_RESULT_FAIL, start);
            }
            if (ctx.input[pos] != literal_[i]) {
                return parser_result(PARSER_RESULT_FAIL, start);
            }
            ++pos;
        }

        return parser_result(PARSER_RESULT_SUCCESS, start, pos);
    }

    std::string dump() const override {
        return "Literal(" + literal_ + ")";
    }

    void accept(parser_visitor & visitor) override;

    const std::string & literal() const { return literal_; }
};

// Matches a sequence of parsers in order, all must succeed.
//   S -> A B C
class sequence_parser : public parser_base {
    std::vector<parser> parsers_;

  public:
    sequence_parser(std::initializer_list<parser> parsers, int id) : parser_base(id) {
        for (const auto & p : parsers) {
            if (p->type() == PARSER_SEQUENCE) {
                // Flatten sequences
                auto seq = std::static_pointer_cast<sequence_parser>(p.ptr());
                for (const auto & embedded : seq->parsers()) {
                    parsers_.push_back(embedded);
                }
            } else {
                parsers_.push_back(p);
            }
        }
    }

    parser_type type() const override { return PARSER_SEQUENCE; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        std::unordered_map<std::string, parser_match_location> groups;

        auto pos = start;
        for (const auto & p : parsers_) {
            auto result = p->parse(ctx, pos);

            // Copy groups
            groups.insert(result.groups.begin(), result.groups.end());

            if (result.is_fail()) {
                if (result.end >= ctx.input.size() && !ctx.input_is_complete) {
                    // If we fail because we don't have enough input, then return success
                    return parser_result(PARSER_RESULT_SUCCESS, start, result.end, groups);
                }
                return parser_result(PARSER_RESULT_FAIL, start, result.end, groups);
            }

            if (result.is_need_more_input()) {
                return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, result.end, groups);
            }

            pos = result.end;
        }

        return parser_result(PARSER_RESULT_SUCCESS, start, pos, groups);
    }

    std::string dump() const override {
        std::vector<std::string> parts;
        parts.reserve(parsers_.size());
        for (const auto & p : parsers_) {
            parts.push_back(p->dump());
        }
        return "Sequence(" + string_join(parts, ", ") + ")";
    }

    void accept(parser_visitor & visitor) override;

    const std::vector<parser> & parsers() const { return parsers_; }
};

// Matches the first parser that succeeds from a list of alternatives.
//   S -> A | B | C
class choice_parser : public parser_base {
    std::vector<parser> parsers_;

  public:
    choice_parser(std::initializer_list<parser> parsers, int id) : parser_base(id) {
        for (const auto & p : parsers) {
            if (p->type() == PARSER_CHOICE) {
                // Flatten choices
                auto choice = std::static_pointer_cast<choice_parser>(p.ptr());
                for (const auto & embedded : choice->parsers()) {
                    parsers_.push_back(embedded);
                }
            } else {
                parsers_.push_back(p);
            }
        }
    }

    parser_type type() const override { return PARSER_CHOICE; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto pos = start;
        for (const auto & p : parsers_) {
            auto result = p->parse(ctx, pos);

            if (result.is_success()) {
                return result;
            }

            if (result.is_need_more_input()) {
                return result;
            }
        }

        return parser_result(PARSER_RESULT_FAIL, start);
    }

    std::string dump() const override {
        std::vector<std::string> parts;
        parts.reserve(parsers_.size());
        for (const auto & p : parsers_) {
            parts.push_back(p->dump());
        }
        return "Choice(" + string_join(parts, ", ") + ")";
    }

    void accept(parser_visitor & visitor) override;

    const std::vector<parser> & parsers() const { return parsers_; }
};

// Matches between min and max repetitions of a parser (inclusive).
//   S -> A{m,n}
// Use -1 for max_count to represent unbounded repetition (equivalent to {m,})
class repetition_parser : public parser_base {
    parser parser_;
    int min_count_;
    int max_count_;

  public:
    repetition_parser(const parser & parser, int min_count, int max_count, int id)
        : parser_base(id), parser_(parser), min_count_(min_count), max_count_(max_count) {}

    parser_type type() const override { return PARSER_REPETITION; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        std::unordered_map<std::string, parser_match_location> groups;
        auto pos = start;
        int match_count = 0;

        // Try to match up to max_count times (or unlimited if max_count is -1)
        while (max_count_ == -1 || match_count < max_count_) {
            auto result = parser_->parse(ctx, pos);
            groups.insert(result.groups.begin(), result.groups.end());

            if (result.is_success()) {
                // Prevent infinite loop on empty matches
                if (result.end == pos) {
                    break;
                }
                pos = result.end;
                match_count++;
                continue;
            }

            if (result.is_need_more_input()) {
                return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, pos, groups);
            }

            // Child failed - stop trying
            break;
        }

        // Check if we got enough matches
        if (match_count < min_count_) {
            return parser_result(PARSER_RESULT_FAIL, start, pos, groups);
        }

        return parser_result(PARSER_RESULT_SUCCESS, start, pos, groups);
    }

    std::string dump() const override {
        if (max_count_ == -1) {
            return "Repetition(" + parser_->dump() + ", " + std::to_string(min_count_) + ", unbounded)";
        }
        return "Repetition(" + parser_->dump() + ", " + std::to_string(min_count_) + ", " + std::to_string(max_count_) + ")";
    }

    void accept(parser_visitor & visitor) override;

    const parser & child() const { return parser_; }

    int min_count() const { return min_count_; }

    int max_count() const { return max_count_; }
};

// Matches one or more repetitions of a parser.
//   S -> A+
class one_or_more_parser : public repetition_parser {
  public:
    one_or_more_parser(const parser & p, int id) : repetition_parser(p, 1, -1, id) {}

    parser_type type() const override { return PARSER_ONE_OR_MORE; }

    std::string dump() const override {
        return "OneOrMore(" + child()->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;
};

// Matches zero or more repetitions of a parser, always succeeds.
//   S -> A*
class zero_or_more_parser : public repetition_parser {
  public:
    zero_or_more_parser(const parser & p, int id) : repetition_parser(p, 0, -1, id) {}

    parser_type type() const override { return PARSER_ZERO_OR_MORE; }

    std::string dump() const override {
        return "ZeroOrMore(" + child()->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;
};

// Matches zero or one occurrence of a parser, always succeeds.
//   S -> A?
class optional_parser : public repetition_parser {
  public:
    optional_parser(const parser & p, int id) : repetition_parser(p, 0, 1, id) {}

    parser_type type() const override { return PARSER_OPTIONAL; }

    std::string dump() const override {
        return "Optional(" + child()->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;
};

// Negative lookahead: succeeds if child parser fails, consumes no input.
//   S -> !A
class not_parser : public parser_base {
    parser parser_;

  public:
    not_parser(const parser & parser, int id) : parser_base(id), parser_(parser) {}

    parser_type type() const override { return PARSER_NOT; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto result = parser_->parse(ctx, start);

        if (result.is_success()) {
            // Fail if the underlying parser matches
            return parser_result(PARSER_RESULT_FAIL, start);
        }

        if (result.is_need_more_input()) {
            // Propagate - need to know what child would match before negating
            return result;
        }

        // Child failed, so negation succeeds
        return parser_result(PARSER_RESULT_SUCCESS, start);
    }

    std::string dump() const override {
        return "Not(" + parser_->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;

    const parser & child() const { return parser_; }
};

// Matches any single character.
//   S -> .
class any_parser : public parser_base {
  public:
    any_parser(int id) : parser_base(id) {}

    parser_type type() const override { return PARSER_ANY; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        if (start >= ctx.input.size()) {
            if (ctx.input_is_complete) {
                return parser_result(PARSER_RESULT_FAIL, start);
            }
            return parser_result(PARSER_RESULT_FAIL, start);
        }

        return parser_result(PARSER_RESULT_SUCCESS, start, start + 1);
    }

    std::string dump() const override {
        return "Any";
    }

    void accept(parser_visitor & visitor) override;
};

// Matches zero or more whitespace characters (space, tab, newline).
//   S -> [ \t\n]*
class space_parser : public parser_base {
  public:
    space_parser(int id) : parser_base(id) {}

    parser_type type() const override { return PARSER_SPACE; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto pos = start;
        while (pos < ctx.input.size()) {
            char c = ctx.input[pos];
            if (is_space(c)) {
                ++pos;
            } else {
                break;
            }
        }

        return parser_result(PARSER_RESULT_SUCCESS, start, pos);
    }

    std::string dump() const override {
        return "Space";
    }

    void accept(parser_visitor & visitor) override;
};

// Matches a single character from a character class or range.
//   S -> [a-z] or S -> [^0-9]
class char_class_parser : public parser_base {
    struct char_range {
        int start;
        int end;

        bool contains(char c) const { return (int)c >= start && int(c) <= end; }
    };

    std::string pattern_;
    std::vector<char_range> ranges_;
    bool negated_;

  public:
    char_class_parser(const std::string & classes, int id) : parser_base(id), pattern_(classes), negated_(false) {
        std::string content = classes;
        if (content.front() == '[') {
            content = content.substr(1);
        }

        if (content.back() == ']') {
            content.pop_back();
        }

        // Check for negation
        if (!content.empty() && content.front() == '^') {
            negated_ = true;
            content = content.substr(1);
        }

        auto parse_char = [&](size_t pos) -> std::pair<char, size_t> {
            if (content[pos] == '\\' && pos + 1 < content.length()) {
                char next = content[pos + 1];
                switch (next) {
                    case 'n':  return {'\n', 2};
                    case 't':  return {'\t', 2};
                    case 'r':  return {'\r', 2};
                    case '\\': return {'\\', 2};
                    case ']':  return {']', 2};
                    case '-':  return {'-', 2};
                    case '[':  return {'[', 2};
                    default:   return {next, 2}; // Treat as literal escaped character
                }
            }
            return {content[pos], 1};
        };

        size_t i = 0;
        while (i < content.length()) {
            auto [start, start_len] = parse_char(i);
            i += start_len;

            if (i + 1 < content.length() && content[i] == '-') {
                // Range detected
                auto [end, end_len] = parse_char(i + 1);
                ranges_.push_back(char_range{start, end});
                i += 1 + end_len;
            } else {
                ranges_.push_back(char_range{start, start});
            }
        }
    }

    parser_type type() const override { return PARSER_CHAR_CLASS; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        if (start >= ctx.input.size()) {
            if (ctx.input_is_complete) {
                return parser_result(PARSER_RESULT_FAIL, start);
            }
            return parser_result(PARSER_RESULT_FAIL, start);
        }

        bool matches = false;
        for (const auto & range : ranges_) {
            if (range.contains(ctx.input[start])) {
                matches = true;
                break;
            }
        }

        // If negated, invert the match result
        if (negated_) {
            matches = !matches;
        }

        if (matches) {
            return parser_result(PARSER_RESULT_SUCCESS, start, start + 1);
        }

        return parser_result(PARSER_RESULT_FAIL, start);
    }

    std::string dump() const override {
        return "Char(" + pattern_ + ")";
    }

    void accept(parser_visitor & visitor) override;

    const std::string & pattern() const { return pattern_; }
};

// Captures the matched text from a parser and stores it with a name.
//   S -> <name:A>
class group_parser : public parser_base {
    std::string name_;
    parser parser_;

  public:
    group_parser(const std::string & name, const parser & parser, int id) : parser_base(id), name_(name), parser_(parser) {}

    parser_type type() const override { return PARSER_GROUP; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto result = parser_->parse(ctx, start);

        // Store result
        result.groups[name_] = parser_match_location{result.start, result.end};
        return result;
    }

    std::string dump() const override {
        return "Group(" + name_ + ", " + parser_->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;

    const parser & child() const { return parser_; }
};

// Matches all characters until a delimiter is found (delimiter not consumed).
//   S -> (!delim .)*
class until_parser : public parser_base {
    std::string delimiter_;
    bool consume_spaces_;

    std::boyer_moore_searcher<std::string::const_iterator> searcher_;

  public:
    until_parser(const std::string & delimiter, bool consume_spaces, int id)
        : parser_base(id), delimiter_(delimiter), consume_spaces_(consume_spaces), searcher_(delimiter_.begin(), delimiter_.end()) {
    }

    parser_type type() const override { return PARSER_UNTIL; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        parser_result result(PARSER_RESULT_SUCCESS, start, ctx.input.size());

        // Search for the delimiter
        const auto * it = std::search(ctx.input.begin(), ctx.input.end(), searcher_);

        if (it != ctx.input.end()) {
            result.type = PARSER_RESULT_SUCCESS;
            result.end = std::distance(ctx.input.begin(), it);
        } else {
            // If not found, check if the input ends with a prefix of the delimiter
            size_t max_overlap = std::min(ctx.input.size(), delimiter_.size() - 1);
            for (size_t overlap = max_overlap; overlap > 0; --overlap) {
                if (std::equal(ctx.input.end() - overlap, ctx.input.end(), delimiter_.begin())) {
                    result.type = (ctx.input_is_complete) ? PARSER_RESULT_FAIL : PARSER_RESULT_NEED_MORE_INPUT;
                    result.end = ctx.input.size() - overlap;
                }
            }
        }

        if (consume_spaces_) {
            // Remove trailing spaces
            while (result.end > start && is_space(ctx.input[result.end - 1])) {
                result.end--;
            }
        }

        return result;
    }

    std::string dump() const override {
        return "Until(" + delimiter_ + ")";
    }

    void accept(parser_visitor & visitor) override;

    const std::string & delimiter() const { return delimiter_; }
};

// Wraps a parser with JSON schema metadata for grammar generation.
// Used internally to convert JSON schemas to GBNF grammar rules.
class schema_parser : public parser_base {
    parser parser_;
    std::string name_;
    nlohmann::ordered_json schema_;

  public:
    schema_parser(const parser & parser, const std::string & name, const nlohmann::ordered_json & schema, int id)
        : parser_base(id), parser_(parser), name_(name), schema_(schema) {}

    parser_type type() const override { return PARSER_SCHEMA; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        return parser_->parse(ctx, start);
    }

    std::string dump() const override {
        return "Schema(" + parser_->dump() + ", " + schema_.dump() + ")";
    }

    void accept(parser_visitor & visitor) override;

    const parser & child() const { return parser_; }

    const std::string & name() const { return name_; }

    const nlohmann::ordered_json & schema() const { return schema_; }
};

// References a named rule for recursive or reusable grammar definitions.
//   expr -> term | expr "+" term
class rule_parser : public parser_base {
    std::string name_;
    std::weak_ptr<std::unordered_map<std::string, parser>> rules_;

  public:
    rule_parser(const std::string & name, const std::shared_ptr<std::unordered_map<std::string, parser>> & rules, int id)
        : parser_base(id), name_(name), rules_(rules) {}

    parser_type type() const override { return PARSER_RULE; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto rules = rules_.lock();
        if (!rules) {
            LOG_ERR("rule_parser::parse called with expired rule registry\n");
            return parser_result(PARSER_RESULT_FAIL, start);
        }

        auto it = rules->find(name_);
        if (it == rules->end()) {
            LOG_ERR("rule_parser::parse rule '%s' not found in registry\n", name_.c_str());
            return parser_result(PARSER_RESULT_FAIL, start);
        }

        return it->second->parse(ctx, start);
    }

    std::string dump() const override {
        return "Rule(" + name_ + ")";
    }

    void accept(parser_visitor & visitor) override;

    const std::string & name() const { return name_; }
};

// Container for the root parser and all named rules in the grammar.
// Manages ownership of rule registry to enable recursive grammar definitions.
class root_parser : public parser_base {
    parser root_;
    std::shared_ptr<std::unordered_map<std::string, parser>> rules_;

    friend class parser_visitor;

  public:
    root_parser(const parser & root, std::shared_ptr<std::unordered_map<std::string, parser>> rules, int id)
        : parser_base(id), root_(root), rules_(std::move(rules)) {}

    parser_type type() const override { return PARSER_ROOT; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        return root_->parse(ctx, start);
    }

    std::string dump() const override {
        return root_->dump();
    }

    void accept(parser_visitor & visitor) override;

    const parser & root() const { return root_; }

    std::shared_ptr<std::unordered_map<std::string, parser>> rules() const { return rules_; }
};

// Base visitor class for parser tree traversal
class parser_visitor {
  public:
    virtual ~parser_visitor() = default;

    virtual void visit(literal_parser & p) = 0;
    virtual void visit(sequence_parser & p) = 0;
    virtual void visit(choice_parser & p) = 0;
    virtual void visit(one_or_more_parser & p) = 0;
    virtual void visit(zero_or_more_parser & p) = 0;
    virtual void visit(optional_parser & p) = 0;
    virtual void visit(repetition_parser & p) = 0;
    virtual void visit(until_parser & p) = 0;
    virtual void visit(not_parser & p) = 0;
    virtual void visit(any_parser & p) = 0;
    virtual void visit(space_parser & p) = 0;
    virtual void visit(char_class_parser & p) = 0;
    virtual void visit(group_parser & p) = 0;
    virtual void visit(schema_parser & p) = 0;
    virtual void visit(rule_parser & p) = 0;
    virtual void visit(root_parser & p) = 0;
};

class gbnf_visitor : public parser_visitor {
    const common_grammar_builder & builder_;
    std::unordered_map<std::string, std::string> rule_name_mapping_;
    std::string current_result_;

  public:
    gbnf_visitor(const common_grammar_builder & builder) : builder_(builder) {}

    const std::string& result() const { return current_result_; }

  private:
    // Escape special characters for GBNF literals
    static std::string escape_literal(const std::string & s) {
        std::string escaped;
        for (char c : s) {
            switch (c) {
                case '\n': escaped += "\\n"; break;
                case '\t': escaped += "\\t"; break;
                case '\r': escaped += "\\r"; break;
                case '\\': escaped += "\\\\"; break;
                case '"':  escaped += "\\\""; break;
                default:   escaped += c; break;
            }
        }
        return escaped;
    }

    // Escape a single character for use in character classes
    static std::string escape_char_class(char c) {
        switch (c) {
            case '\n': return "\\n";
            case '\t': return "\\t";
            case '\r': return "\\r";
            case '\\': return "\\\\";
            case ']':  return "\\]";
            case '-':  return "\\-";
            case '^':  return "\\^";
            default:   return std::string(1, c);
        }
    }

    // Generate pattern for until() that matches prefixes but prevents full delimiter match
    // For "</tag>" generates: ( [^<] | "<" [^/] | "</" [^t] | "</t" [^a] | "</ta" [^g] )*
    static std::string generate_until_pattern(const std::string & delimiter) {
        if (delimiter.empty()) {
            return ".*";  // Match everything if delimiter is empty
        }

        if (delimiter.length() == 1) {
            // Simple case: just negate the single character
            return "[^" + escape_char_class(delimiter[0]) + "]";
        }

        std::vector<std::string> alternatives;

        // First alternative: match any character that's not the start of the delimiter
        alternatives.push_back("[^" + escape_char_class(delimiter[0]) + "]");

        // For each prefix, match the prefix followed by a char that's not the next delimiter char
        for (size_t i = 1; i < delimiter.length(); ++i) {
            std::string prefix = "\"" + escape_literal(delimiter.substr(0, i)) + "\"";
            std::string next_char_negated = "[^" + escape_char_class(delimiter[i]) + "]";
            alternatives.push_back(prefix + " " + next_char_negated);
        }

        // Combine alternatives with |
        std::string result = "(";
        for (size_t i = 0; i < alternatives.size(); ++i) {
            if (i > 0) {
                result += " | ";
            }
            result += alternatives[i];
        }
        result += ")";

        return result;
    }

    // Check if expression needs parentheses
    static bool needs_parens(parser_type type) {
        return type == PARSER_CHOICE || type == PARSER_SEQUENCE;
    }

  public:
    void visit(literal_parser & p) override {
        current_result_ = "\"" + escape_literal(p.literal()) + "\"";
    }

    void visit(sequence_parser & p) override {
        std::string s;
        for (const auto & child : p.parsers()) {
            if (!s.empty()) {
                s += " ";
            }
            child->accept(*this);
            s += current_result_;
        }
        current_result_ = s;
    }

    void visit(choice_parser & p) override {
        std::string s;
        for (const auto & child : p.parsers()) {
            if (!s.empty()) {
                s += " | ";
            }

            child->accept(*this);

            // Parenthesize sequences in choices
            if (child->type() == PARSER_SEQUENCE) {
                s += "(" + current_result_ + ")";
            } else {
                s += current_result_;
            }
        }
        current_result_ = s;
    }

    void visit(one_or_more_parser & p) override {
        p.child()->accept(*this);
        if (needs_parens(p.child()->type())) {
            current_result_ = "(" + current_result_ + ")+";
        } else {
            current_result_ = current_result_ + "+";
        }
    }

    void visit(zero_or_more_parser & p) override {
        p.child()->accept(*this);
        if (needs_parens(p.child()->type())) {
            current_result_ = "(" + current_result_ + ")*";
        } else {
            current_result_ = current_result_ + "*";
        }
    }

    void visit(optional_parser & p) override {
        p.child()->accept(*this);
        if (needs_parens(p.child()->type())) {
            current_result_ = "(" + current_result_ + ")?";
        } else {
            current_result_ = current_result_ + "?";
        }
    }

    void visit(repetition_parser & p) override {
        p.child()->accept(*this);
        std::string child_result = current_result_;

        if (needs_parens(p.child()->type())) {
            child_result = "(" + child_result + ")";
        }

        if (p.max_count() == -1) {
            // Unbounded: {n,}
            current_result_ = child_result + "{" + std::to_string(p.min_count()) + ",}";
        } else {
            // Bounded: {n,m}
            current_result_ = child_result + "{" + std::to_string(p.min_count()) + "," +
                             std::to_string(p.max_count()) + "}";
        }
    }

    void visit(until_parser & p) override {
        // Generate pattern that matches prefixes but prevents full delimiter match
        current_result_ = generate_until_pattern(p.delimiter()) + "*";
    }

    void visit(not_parser &) override {
        // NOT is tricky in GBNF - for now, emit error
        LOG_ERR("NOT operator not directly supported in GBNF generation\n");
        current_result_ = "";
    }

    void visit(any_parser &) override {
        // Match any single character
        current_result_ = "[\\x00-\\x{10FFFF}]";
    }

    void visit(space_parser &) override {
        // Reference the built-in space rule
        current_result_ = "space";
    }

    void visit(char_class_parser & p) override {
        // Return pattern as-is (already in GBNF format)
        current_result_ = p.pattern();
    }

    void visit(group_parser & p) override {
        // Groups are transparent - just visit child
        p.child()->accept(*this);
    }

    void visit(schema_parser & p) override {
        current_result_ = builder_.add_schema(p.name(), p.schema());
    }

    void visit(rule_parser & p) override {
        // Return canonical rule reference
        auto it = rule_name_mapping_.find(p.name());
        if (it != rule_name_mapping_.end()) {
            current_result_ = it->second;
        } else {
            // Fallback to original name if not in mapping (shouldn't happen in valid usage)
            current_result_ = p.name();
        }
    }

    void visit(root_parser & p) override {
        // Generate named rules first
        auto rules = p.rules();
        if (rules) {
            for (const auto & [name, rule] : *rules) {
                rule->accept(*this);
                auto rule_body = current_result_;
                auto canonical_name = builder_.add_rule(name, rule_body);
                rule_name_mapping_[name] = canonical_name;
            }
        }

        // Return root body for composition
        p.root()->accept(*this);
    }
};

// ID assignment visitor for assigning unique IDs to parsers
class id_assignment_visitor : public parser_visitor {
    std::shared_ptr<parser_id_counter> counter_;

  public:
    id_assignment_visitor(const std::shared_ptr<parser_id_counter> & counter) : counter_(counter) {}

    void assign_id(parser_base & p) {
        if (p.id() == -1) {
            p.set_id(counter_->next());
        }
    }

    void visit(literal_parser & p) override {
        assign_id(p);
    }

    void visit(any_parser & p) override {
        assign_id(p);
    }

    void visit(space_parser & p) override {
        assign_id(p);
    }

    void visit(char_class_parser & p) override {
        assign_id(p);
    }

    void visit(schema_parser & p) override {
        assign_id(p);
    }

    void visit(rule_parser & p) override {
        assign_id(p);
    }

    // Composite parsers - assign ID and traverse children
    void visit(sequence_parser & p) override {
        assign_id(p);
        for (const auto & child : p.parsers()) {
            child->accept(*this);
        }
    }

    void visit(choice_parser & p) override {
        assign_id(p);
        for (const auto & child : p.parsers()) {
            child->accept(*this);
        }
    }

    void visit(one_or_more_parser & p) override {
        assign_id(p);
        p.child()->accept(*this);
    }

    void visit(zero_or_more_parser & p) override {
        assign_id(p);
        p.child()->accept(*this);
    }

    void visit(optional_parser & p) override {
        assign_id(p);
        p.child()->accept(*this);
    }

    void visit(repetition_parser & p) override {
        assign_id(p);
        p.child()->accept(*this);
    }

    void visit(until_parser & p) override {
        assign_id(p);
    }

    void visit(not_parser & p) override {
        assign_id(p);
        p.child()->accept(*this);
    }

    void visit(group_parser & p) override {
        assign_id(p);
        p.child()->accept(*this);
    }

    void visit(root_parser & p) override {
        assign_id(p);
        p.root()->accept(*this);
    }
};

// Implement accept() methods for all parser classes
void literal_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void sequence_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void choice_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void one_or_more_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void zero_or_more_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void optional_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void repetition_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void until_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void not_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void any_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void space_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void char_class_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void group_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void schema_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void rule_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void root_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }

std::optional<std::string> parser_result::group(const std::string & name, std::string_view input) const {
    auto it = groups.find(name);
    if (it == groups.end()) {
        return std::nullopt;
    }

    return std::string(it->second.view(input));
}

parser_result parse_cache::set(int id, size_t start, parser_result result) {
    if (id == -1) {
        // Don't cache parsers with ID -1 (from operators and global factory functions)
        return result;
    }
    results[parse_cache_key{id, start}] = result;
    return result;
}

std::optional<parser_result> parse_cache::get(int id, size_t start) {
    if (id == -1) {
        // Don't cache parsers with ID -1 (from operators and global factory functions)
        return std::nullopt;
    }
    auto it = results.find(parse_cache_key{id, start});
    if (it != results.end()) {
        return it->second;
    }
    return std::nullopt;
}

void parse_cache::clear() {
    results.clear();
}

parser::parser() {}

parser::parser(std::shared_ptr<parser_base> parser) : ptr_(std::move(parser)) {}

parser parser::operator~() const {
    return parser(std::make_shared<not_parser>(*this, -1));
}

parser parser::operator+(const parser & other) const {
    return parser(std::make_shared<sequence_parser>(std::initializer_list<parser>{*this, other}, -1));
}

parser parser::operator|(const parser & other) const {
    return parser(std::make_shared<choice_parser>(std::initializer_list<parser>{*this, other}, -1));
}

parser parser::operator<<(const parser & other) const {
    auto ws = parser(std::make_shared<space_parser>(-1));
    return parser(std::make_shared<sequence_parser>(std::initializer_list<parser>{*this, ws, other}, -1));
}

parser_base & parser::operator*() const {
    return *ptr_;
}

parser_base * parser::operator->() const {
    return ptr_.get();
}

parser_result parser::parse(parser_context & ctx, size_t start) const {
    return ptr_->parse(ctx, start);
}

std::string parser::dump() const {
    return ptr_->dump();
}

void parser::build_grammar(const common_grammar_builder & builder) const {
    gbnf_visitor visitor(builder);
    ptr_->accept(visitor);
    auto result = visitor.result();
    if (!result.empty()) {
        builder.add_rule("root", result);
    }
}

parser_builder::parser_builder()
    : rules_(std::make_shared<std::unordered_map<std::string, parser>>())
    , counter_(std::make_shared<parser_id_counter>(0)) {}

parser_builder::parser_builder(std::shared_ptr<parser_id_counter> counter)
    : rules_(std::make_shared<std::unordered_map<std::string, parser>>())
    , counter_(std::move(counter)) {}

parser parser_builder::literal(const std::string & literal) {
    return parser(std::make_shared<literal_parser>(literal, counter_->next()));
}

parser parser_builder::sequence(std::initializer_list<parser> parsers) {
    return parser(std::make_shared<sequence_parser>(parsers, counter_->next()));
}

parser parser_builder::choice(std::initializer_list<parser> parsers) {
    return parser(std::make_shared<choice_parser>(parsers, counter_->next()));
}

parser parser_builder::one_or_more(const parser & p) {
    return parser(std::make_shared<one_or_more_parser>(p, counter_->next()));
}

parser parser_builder::zero_or_more(const parser & p) {
    return parser(std::make_shared<zero_or_more_parser>(p, counter_->next()));
}

parser parser_builder::optional(const parser & p) {
    return parser(std::make_shared<optional_parser>(p, counter_->next()));
}

parser parser_builder::negate(const parser & p) {
    return parser(std::make_shared<not_parser>(p, counter_->next()));
}

parser parser_builder::any() {
    return parser(std::make_shared<any_parser>(counter_->next()));
}

parser parser_builder::char_class(const std::string & classes) {
    return parser(std::make_shared<char_class_parser>(classes, counter_->next()));
}

parser parser_builder::group(const std::string & name, const parser & p) {
    return parser(std::make_shared<group_parser>(name, p, counter_->next()));
}

parser parser_builder::rule(const std::string & name) {
    return parser(std::make_shared<rule_parser>(name, rules_, counter_->next()));
}

parser parser_builder::space() {
    return parser(std::make_shared<space_parser>(counter_->next()));
}

parser parser_builder::until(const std::string & delimiter, bool consume_spaces) {
    return parser(std::make_shared<until_parser>(delimiter, consume_spaces, counter_->next()));
}

parser parser_builder::repeat(const parser & p, int min, int max) {
    return parser(std::make_shared<repetition_parser>(p, min, max, counter_->next()));
}

parser parser_builder::repeat(const parser & p, int n) {
    return repeat(p, n, n);
}

parser parser_builder::schema(const parser & p, const std::string & name, const nlohmann::ordered_json & schema) {
    return parser(std::make_shared<schema_parser>(p, name, schema, counter_->next()));
}

parser parser_builder::json_key(const std::string & name, const parser & p) {
    return literal("\"" + name + "\"") << literal(":") << p;
}

parser parser_builder::json_string(const parser & p) {
    auto quote = literal("\"");
    return quote + p + quote;
}

parser parser_builder::between(const std::string & left, const parser & p, const std::string & right, bool allow_spaces) {
    if (allow_spaces) {
        return literal(left) << p << literal(right);
    }
    return literal(left) + p + literal(right);
}

parser parser_builder::add_rule(const std::string & name, const parser & p) {
    (*rules_)[name] = p;
    return rule(name);
}

void parser_builder::assign_ids(parser & p) {
    if (p.ptr()) {
        id_assignment_visitor visitor(counter_);
        p.ptr()->accept(visitor);
    }
}

parser build_parser(const std::function<parser(parser_builder&)> & fn) {
    parser_builder builder;
    auto root = fn(builder);
    builder.assign_ids(root); // Assign IDs to rules that were created with operators

    // Wrap the root parser in a root_parser to own the rules and break circular references
    auto rules = builder.rules();
    if (rules && !rules->empty()) {
        return parser(std::make_shared<root_parser>(root, rules, -1));
    }
    return root;
}

static parser json_parser(std::shared_ptr<parser_id_counter> counter) {
    parser_builder builder(std::move(counter));

    // Whitespace: space, tab, newline, carriage return
    auto ws = builder.zero_or_more(builder.char_class("[ \\t\\n\\r]"));

    // Number components
    auto digit = builder.char_class("[0-9]");
    auto digit1_9 = builder.char_class("[1-9]");
    auto digits = builder.one_or_more(digit);

    // Integer part: 0 or non-zero digit followed by more digits
    auto int_part = builder.literal("0") | (digit1_9 + builder.zero_or_more(digit));

    // Optional fractional part
    auto frac = builder.literal(".") + digits;

    // Optional exponent part
    auto exp = (builder.literal("e") | builder.literal("E")) + builder.optional(builder.char_class("[+\\-]")) + digits;

    // Complete number
    auto number = builder.optional(builder.literal("-")) + int_part + builder.optional(frac) + builder.optional(exp);

    builder.add_rule("json_number", number);

    // String components
    auto hex = builder.char_class("[0-9a-fA-F]");
    auto unicode_escape = builder.literal("\\u") + hex + hex + hex + hex;
    auto simple_escape = builder.literal("\\") + builder.char_class("[\"\\\\bfnrt/]");
    auto escape = simple_escape | unicode_escape;

    // String character: escape sequence or any char except quote and backslash
    auto string_char = escape | builder.char_class("[^\"\\\\]");
    auto string = builder.literal("\"") + builder.zero_or_more(string_char) + builder.literal("\"");

    builder.add_rule("json_string", string);

    // Literals
    auto true_lit = builder.literal("true");
    auto false_lit = builder.literal("false");
    auto null_lit = builder.literal("null");

    // Object: { "key": value, ... }
    auto member = builder.rule("json_string") + ws + builder.literal(":") + ws + builder.rule("json_value");
    auto members = member + builder.zero_or_more(ws + builder.literal(",") + ws + member);

    // Empty object or object with members
    auto object = (builder.literal("{") + ws + builder.literal("}")) |
                  (builder.literal("{") + ws + members + ws + builder.literal("}"));

    builder.add_rule("json_object", object);

    // Array: [ value, ... ]
    auto elements = builder.rule("json_value") + builder.zero_or_more(ws + builder.literal(",") + ws + builder.rule("json_value"));

    // Empty array or array with elements
    auto array = (builder.literal("[") + ws + builder.literal("]")) |
                 (builder.literal("[") + ws + elements + ws + builder.literal("]"));

    builder.add_rule("json_array", array);

    // Value - uses forward references for recursive structures
    auto root = builder.add_rule("json_value",
        builder.rule("json_object") |
        builder.rule("json_array") |
        builder.rule("json_string") |
        builder.rule("json_number") |
        true_lit |
        false_lit |
        null_lit
    );

    // Wrap in root_parser to own the rules
    return parser(std::make_shared<root_parser>(root, builder.rules(), -1));
}

parser parser_builder::json() {
    return json_parser(counter_);
}
