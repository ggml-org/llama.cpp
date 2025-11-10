#include "chat-parser-combinator.h"
#include "json-schema-to-grammar.h"
#include "common.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <optional>

class gbnf_visitor;

static parser json_parser();

class parser_base {
  protected:
    int id_;

    void set_id(int id) { id_ = id; }

  public:
    parser_base(int id) : id_(id) {}
    virtual ~parser_base() = default;

    virtual parser_type type() const = 0;
    virtual parser_result parse(parser_context & ctx, size_t start = 0) = 0;
    virtual std::string dump() const = 0;
    virtual std::string accept(gbnf_visitor & visitor) const = 0;
    virtual void assign_ids_internal(int& next_id) {
        if (id_ == -1) {
            id_ = next_id++;
        }
    }
};

class literal_parser : public parser_base {
    std::string literal_;

    friend class gbnf_visitor;

  public:
    literal_parser(const std::string & literal, int id) : parser_base(id), literal_(literal) {}

    parser_type type() const override { return PARSER_LITERAL; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        if (cached != std::nullopt) {
            return *cached;
        }

        auto pos = start;
        for (auto i = 0u; i < literal_.size(); ++i) {
            if (pos >= ctx.input.size()) {
                if (ctx.input_is_complete) {
                    return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_FAIL, start));
                }
                if (i > 0) {
                    return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, pos);
                }
                return parser_result(PARSER_RESULT_FAIL, start);
            }
            if (ctx.input[pos] != literal_[i]) {
                return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_FAIL, start));
            }
            ++pos;
        }

        return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_SUCCESS, start, pos));
    }

    std::string dump() const override {
        return "Literal(" + literal_ + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;
};

class sequence_parser : public parser_base {
    std::vector<parser> parsers_;

    friend class gbnf_visitor;

  public:
    sequence_parser(std::initializer_list<parser> parsers, int id) : parser_base(id) {
        for (const auto & p : parsers) {
            if (p.is_sequence()) {
                // Flatten sequences
                for (const auto & embedded : p.to_sequence()->parsers()) {
                    parsers_.push_back(embedded);
                }
            } else {
                parsers_.push_back(p);
            }
        }
    }

    parser_type type() const override { return PARSER_SEQUENCE; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        if (cached != std::nullopt) {
            return *cached;
        }

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
                return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_FAIL, start, result.end, groups));
            }

            if (result.is_need_more_input()) {
                return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, result.end, groups);
            }

            pos = result.end;
        }

        return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_SUCCESS, start, pos, groups));
    }

    std::string dump() const override {
        std::vector<std::string> parts;
        parts.reserve(parsers_.size());
        for (const auto & p : parsers_) {
            parts.push_back(p->dump());
        }
        return "Sequence(" + string_join(parts, ", ") + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;

    const std::vector<parser> & parsers() const { return parsers_; }

    void assign_ids_internal(int& next_id) override {
        if (id_ == -1) {
            id_ = next_id++;
        }
        for (auto & p : parsers_) {
            p->assign_ids_internal(next_id);
        }
    }
};

class choice_parser : public parser_base {
    std::vector<parser> parsers_;

    friend class gbnf_visitor;

  public:
    choice_parser(std::initializer_list<parser> parsers, int id) : parser_base(id) {
        for (const auto & p : parsers) {
            if (p.is_choice()) {
                // Flatten choices
                for (const auto & embedded : p.to_choice()->parsers()) {
                    parsers_.push_back(embedded);
                }
            } else {
                parsers_.push_back(p);
            }
        }
    }

    parser_type type() const override { return PARSER_CHOICE; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        if (cached != std::nullopt) {
            return *cached;
        }

        auto pos = start;
        for (const auto & p : parsers_) {
            auto result = p->parse(ctx, pos);

            if (result.is_success()) {
                return ctx.memo.set(id_, start, result);
            }

            if (result.is_need_more_input()) {
                return result;
            }
        }

        return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_FAIL, start));
    }

    std::string dump() const override {
        std::vector<std::string> parts;
        parts.reserve(parsers_.size());
        for (const auto & p : parsers_) {
            parts.push_back(p->dump());
        }
        return "Choice(" + string_join(parts, ", ") + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;

    const std::vector<parser> & parsers() const { return parsers_; }

    void assign_ids_internal(int& next_id) override {
        if (id_ == -1) {
            id_ = next_id++;
        }
        for (auto & p : parsers_) {
            p->assign_ids_internal(next_id);
        }
    }
};

class one_or_more_parser : public parser_base {
    parser parser_;

    friend class gbnf_visitor;

  public:
    one_or_more_parser(const parser & parser, int id) : parser_base(id), parser_(parser) {}

    parser_type type() const override { return PARSER_ONE_OR_MORE; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        std::unordered_map<std::string, parser_match_location> groups;

        // We can't return back the cached result, since there may be more
        // repetitions since the last parsing attempt. Instead, resume parsing from
        // the last successful repetition found.
        auto pos = start;
        if (cached != std::nullopt) {
            pos = cached->end;
            groups.insert(cached->groups.begin(), cached->groups.end());
        }

        if (pos == start) {
            auto first_result = parser_->parse(ctx, pos);
            if (!first_result.is_success()) {
                return first_result;
            }

            pos = first_result.end;
            groups.insert(first_result.groups.begin(), first_result.groups.end());
        }

        for (;;) {
            auto result = parser_->parse(ctx, pos);
            groups.insert(result.groups.begin(), result.groups.end());

            if (result.is_need_more_input()) {
                return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, pos, groups);
            }

            if (result.is_fail()) {
                // Done with repetitions
                break;
            }

            if (result.end == pos) {
                break; // Prevent an infinite loop
            }

            pos = result.end;
        }

        return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_SUCCESS, start, pos, groups));
    }

    std::string dump() const override {
        return "OneOrMore(" + parser_->dump() + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;

    const parser & child() const { return parser_; }

    void assign_ids_internal(int& next_id) override {
        if (id_ == -1) {
            id_ = next_id++;
        }
        parser_->assign_ids_internal(next_id);
    }
};

class zero_or_more_parser : public parser_base {
    parser parser_;

    friend class gbnf_visitor;

  public:
    zero_or_more_parser(const parser & parser, int id) : parser_base(id), parser_(parser) {}

    parser_type type() const override { return PARSER_ZERO_OR_MORE; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        std::unordered_map<std::string, parser_match_location> groups;

        // We can't return back the cached result, since there may be more
        // repetitions since the last parsing attempt. Instead, resume parsing from
        // the last successful repetition found.
        auto pos = start;
        if (cached != std::nullopt) {
            pos = cached->end;
            groups.insert(cached->groups.begin(), cached->groups.end());
        }

        for (;;) {
            auto result = parser_->parse(ctx, pos);
            groups.insert(result.groups.begin(), result.groups.end());

            if (result.is_need_more_input()) {
                return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, pos, groups);
            }

            if (result.is_fail()) {
                // Done with repetitions (zero or more is always valid)
                break;
            }

            if (result.end == pos) {
                break; // Prevent an infinite loop
            }

            pos = result.end;
        }

        return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_SUCCESS, start, pos, groups));
    }

    std::string dump() const override {
        return "ZeroOrMore(" + parser_->dump() + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;

    const parser & child() const { return parser_; }

    void assign_ids_internal(int& next_id) override {
        if (id_ == -1) {
            id_ = next_id++;
        }
        parser_->assign_ids_internal(next_id);
    }
};

class optional_parser : public parser_base {
    parser parser_;

    friend class gbnf_visitor;

  public:
    optional_parser(const parser & parser, int id) : parser_base(id), parser_(parser) {}

    parser_type type() const override { return PARSER_OPTIONAL; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        if (cached != std::nullopt) {
            return *cached;
        }

        auto result = parser_->parse(ctx, start);

        if (result.is_success()) {
            // Matched successfully
            return ctx.memo.set(id_, start, result);
        }

        if (result.is_need_more_input()) {
            // Propagate - need more input to determine if optional matches
            return result;
        }

        // No match, but optional always succeeds with zero matches
        return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_SUCCESS, start, start));
    }

    std::string dump() const override {
        return "Optional(" + parser_->dump() + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;

    const parser & child() const { return parser_; }

    void assign_ids_internal(int& next_id) override {
        if (id_ == -1) {
            id_ = next_id++;
        }
        parser_->assign_ids_internal(next_id);
    }
};

class until_parser : public parser_base {
    std::string delimiter_;
    parser parser_;

    friend class gbnf_visitor;

  public:
    until_parser(const std::string & delimiter, bool include_spaces, int id, parser_builder & builder)
        : parser_base(id), delimiter_(delimiter) {
        if (include_spaces) {
            auto ws = builder.zero_or_more(builder.char_class("[ \\t\\n\\r]"));
            parser_ = builder.zero_or_more(builder.negate(ws + builder.literal(delimiter)) + builder.any());
        } else {
            parser_ = builder.zero_or_more(builder.negate(builder.literal(delimiter)) + builder.any());
        }
    }

    parser_type type() const override { return PARSER_UNTIL; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        if (cached != std::nullopt) {
            return *cached;
        }

        auto result = parser_->parse(ctx, start);
        return ctx.memo.set(id_, start, result);
    }

    std::string dump() const override {
        return "Until(" + delimiter_ + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;

    void assign_ids_internal(int& next_id) override {
        if (id_ == -1) {
            id_ = next_id++;
        }
        parser_->assign_ids_internal(next_id);
    }
};

class not_parser : public parser_base {
    parser parser_;

    friend class gbnf_visitor;

  public:
    not_parser(const parser & parser, int id) : parser_base(id), parser_(parser) {}

    parser_type type() const override { return PARSER_NOT; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        if (cached != std::nullopt) {
            return *cached;
        }

        auto result = parser_->parse(ctx, start);

        if (result.is_success()) {
            // Fail if the underlying parser matches
            return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_FAIL, start));
        }

        if (result.is_need_more_input()) {
            // Propagate - need to know what child would match before negating
            return result;
        }

        // Child failed, so negation succeeds
        return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_SUCCESS, start));
    }

    std::string dump() const override {
        return "Not(" + parser_->dump() + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;

    const parser & child() const { return parser_; }

    void assign_ids_internal(int& next_id) override {
        if (id_ == -1) {
            id_ = next_id++;
        }
        parser_->assign_ids_internal(next_id);
    }
};

class any_parser : public parser_base {
    friend class gbnf_visitor;

  public:
    any_parser(int id) : parser_base(id) {}

    parser_type type() const override { return PARSER_ANY; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        if (cached != std::nullopt) {
            return *cached;
        }

        if (start >= ctx.input.size()) {
            if (ctx.input_is_complete) {
                return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_FAIL, start));
            }
            return parser_result(PARSER_RESULT_FAIL, start);
        }

        return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_SUCCESS, start, start + 1));
    }

    std::string dump() const override {
        return "Any";
    }

    std::string accept(gbnf_visitor & visitor) const override;
};

class space_parser : public parser_base {
    friend class gbnf_visitor;

  public:
    space_parser(int id) : parser_base(id) {}

    parser_type type() const override { return PARSER_SPACE; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        if (cached != std::nullopt) {
            return *cached;
        }

        auto pos = start;
        while (pos < ctx.input.size()) {
            char c = ctx.input[pos];
            if (c == ' ' || c == '\t' || c == '\n') {
                ++pos;
            } else {
                break;
            }
        }

        return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_SUCCESS, start, pos));
    }

    std::string dump() const override {
        return "Space";
    }

    std::string accept(gbnf_visitor & visitor) const override;
};

class char_class_parser : public parser_base {
    struct char_range {
        int start;
        int end;

        bool contains(char c) const { return (int)c >= start && int(c) <= end; }
    };

    std::string pattern_;
    std::vector<char_range> ranges_;
    bool negated_;

    friend class gbnf_visitor;

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

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        if (cached != std::nullopt) {
            return *cached;
        }

        if (start >= ctx.input.size()) {
            if (ctx.input_is_complete) {
                return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_FAIL, start));
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
            return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_SUCCESS, start, start + 1));
        }

        return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_FAIL, start));
    }

    std::string dump() const override {
        return "Char(" + pattern_ + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;
};

class group_parser : public parser_base {
    std::string name_;
    parser parser_;

    friend class gbnf_visitor;

  public:
    group_parser(const std::string & name, const parser & parser, int id) : parser_base(id), name_(name), parser_(parser) {}

    parser_type type() const override { return PARSER_GROUP; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto result = parser_->parse(ctx, start);

        // Store result
        result.groups[name_] = parser_match_location{result.start, result.end};
        return ctx.memo.set(id_, start, result);
    }

    std::string dump() const override {
        return "Group(" + name_ + ", " + parser_->dump() + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;

    void assign_ids_internal(int& next_id) override {
        if (id_ == -1) {
            id_ = next_id++;
        }
        parser_->assign_ids_internal(next_id);
    }
};

class schema_parser : public parser_base {
    parser parser_;
    std::string name_;
    nlohmann::ordered_json schema_;

    friend class gbnf_visitor;

  public:
    schema_parser(const parser & parser, const std::string & name, const nlohmann::ordered_json & schema, int id)
        : parser_base(id), parser_(parser), name_(name), schema_(schema) {}

    parser_type type() const override { return PARSER_SCHEMA; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        return parser_->parse(ctx, start);
    }

    std::string dump() const override {
        return "Schema(" + parser_->dump() + ", " + schema_.dump() + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;
};

class rule_parser : public parser_base {
    std::string rule_name_;
    std::weak_ptr<std::unordered_map<std::string, parser>> rules_;

    friend class gbnf_visitor;

  public:
    rule_parser(const std::string & name, std::shared_ptr<std::unordered_map<std::string, parser>> rules, int id)
        : parser_base(id), rule_name_(name), rules_(rules) {}

    parser_type type() const override { return PARSER_RULE; }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        auto cached = ctx.memo.get(id_, start);
        if (cached != std::nullopt) {
            return *cached;
        }

        auto rules = rules_.lock();
        if (!rules) {
            LOG_ERR("rule_parser::parse called with expired rule registry\n");
            return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_FAIL, start));
        }

        auto it = rules->find(rule_name_);
        if (it == rules->end()) {
            LOG_ERR("rule_parser::parse rule '%s' not found in registry\n", rule_name_.c_str());
            return ctx.memo.set(id_, start, parser_result(PARSER_RESULT_FAIL, start));
        }

        auto result = it->second->parse(ctx, start);
        return ctx.memo.set(id_, start, result);
    }

    std::string dump() const override {
        return "Rule(" + rule_name_ + ")";
    }

    std::string accept(gbnf_visitor & visitor) const override;
};

class root_parser : public parser_base {
    parser root_;
    std::shared_ptr<std::unordered_map<std::string, parser>> rules_;

    friend class gbnf_visitor;

  public:
    root_parser(const parser & root, std::shared_ptr<std::unordered_map<std::string, parser>> rules, int id)
        : parser_base(id), root_(root), rules_(std::move(rules)) {}

    parser_type type() const override { return root_->type(); }

    parser_result parse(parser_context & ctx, size_t start = 0) override {
        return root_->parse(ctx, start);
    }

    std::string dump() const override {
        return root_->dump();
    }

    std::string accept(gbnf_visitor & visitor) const override;

    void assign_ids_internal(int& next_id) override {
        if (id_ == -1) {
            id_ = next_id++;
        }
        root_->assign_ids_internal(next_id);
    }
};

class gbnf_visitor {
    common_grammar_builder& builder_;
    std::unordered_map<std::string, std::string> rule_name_mapping_;

  public:
    gbnf_visitor(common_grammar_builder& builder) : builder_(builder) {}

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
    std::string visit(const literal_parser & p) {
        return "\"" + escape_literal(p.literal_) + "\"";
    }

    std::string visit(const sequence_parser & p) {
        std::string s;
        for (size_t i = 0; i < p.parsers_.size(); ++i) {
            if (i > 0) s += " ";
            auto child_result = p.parsers_[i]->accept(*this);
            s += child_result;
        }
        return s;
    }

    std::string visit(const choice_parser & p) {
        std::string s;
        for (size_t i = 0; i < p.parsers_.size(); ++i) {
            if (i > 0) {
                s += " | ";
            }

            auto child_type = p.parsers_[i]->type();
            auto child_result = p.parsers_[i]->accept(*this);

            // Parenthesize sequences in choices
            if (child_type == PARSER_SEQUENCE) {
                s += "(" + child_result + ")";
            } else {
                s += child_result;
            }
        }
        return s;
    }

    std::string visit(const one_or_more_parser & p) {
        auto child_type = p.parser_->type();
        auto child_result = p.parser_->accept(*this);
        if (needs_parens(child_type)) {
            return "(" + child_result + ")+";
        }
        return child_result + "+";
    }

    std::string visit(const zero_or_more_parser & p) {
        auto child_type = p.parser_->type();
        auto child_result = p.parser_->accept(*this);
        if (needs_parens(child_type)) {
            return "(" + child_result + ")*";
        }
        return child_result + "*";
    }

    std::string visit(const optional_parser & p) {
        auto child_type = p.parser_->type();
        auto child_result = p.parser_->accept(*this);
        if (needs_parens(child_type)) {
            return "(" + child_result + ")?";
        }
        return child_result + "?";
    }

    std::string visit(const until_parser & p) {
        // Generate pattern that matches prefixes but prevents full delimiter match
        return generate_until_pattern(p.delimiter_) + "*";
    }

    std::string visit(const not_parser &) {
        // NOT is tricky in GBNF - for now, emit error
        LOG_ERR("NOT operator not directly supported in GBNF generation\n");
        return ""; // This will cause compilation errors, which is intended
    }

    std::string visit(const any_parser &) {
        // Match any single character
        return "[\\x00-\\x{10FFFF}]";
    }

    std::string visit(const space_parser &) {
        // Reference the built-in space rule
        return "space";
    }

    std::string visit(const char_class_parser & p) {
        // Return pattern as-is (already in GBNF format)
        return p.pattern_;
    }

    std::string visit(const group_parser & p) {
        // Groups are transparent - just visit child
        return p.parser_->accept(*this);
    }

    std::string visit(const schema_parser & p) {
        return builder_.add_schema(p.name_, p.schema_);
    }

    std::string visit(const rule_parser & p) {
        // Return canonical rule reference
        auto it = rule_name_mapping_.find(p.rule_name_);
        if (it != rule_name_mapping_.end()) {
            return it->second;
        }
        // Fallback to original name if not in mapping (shouldn't happen in valid usage)
        return p.rule_name_;
    }

    std::string visit(const root_parser & p) {
        // Generate named rules first
        if (p.rules_) {
            for (const auto & [name, rule] : *p.rules_) {
                auto rule_body = rule->accept(*this);
                auto canonical_name = builder_.add_rule(name, rule_body);
                rule_name_mapping_[name] = canonical_name;
            }
        }

        // Return root body for composition
        return p.root_->accept(*this);
    }
};

// Implement accept() methods for all parser classes
std::string literal_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string sequence_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string choice_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string one_or_more_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string zero_or_more_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string optional_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string until_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string not_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string any_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string space_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string char_class_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string group_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string schema_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string rule_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

std::string root_parser::accept(gbnf_visitor & visitor) const {
    return visitor.visit(*this);
}

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

parser::parser(std::shared_ptr<parser_base> parser) : ptr(std::move(parser)) {}

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
    return *ptr;
}

parser_base * parser::operator->() const {
    return ptr.get();
}

bool parser::is_sequence() const {
    return ptr->type() == PARSER_SEQUENCE;
}

std::shared_ptr<sequence_parser> parser::to_sequence() const {
    return std::dynamic_pointer_cast<sequence_parser>(ptr);
}

bool parser::is_choice() const {
    return ptr->type() == PARSER_CHOICE;
}

std::shared_ptr<choice_parser> parser::to_choice() const {
    return std::dynamic_pointer_cast<choice_parser>(ptr);
}

parser_type parser::type() const {
    return ptr->type();
}

parser_result parser::parse(parser_context & ctx, size_t start) const {
    return ptr->parse(ctx, start);
}

std::string parser::dump() const {
    return ptr->dump();
}

void parser::build_grammar(common_grammar_builder& builder) const {
    gbnf_visitor visitor(builder);
    auto result = ptr->accept(visitor);
    // The visitor returns the GBNF string for this parser
    // root_parser registers its named rules and returns its root body
    if (!result.empty()) {
        builder.add_rule("root", result);
    }
}

parser_builder::parser_builder()
    : rules_(std::make_shared<std::unordered_map<std::string, parser>>())
    , next_id_(0) {}

parser parser_builder::literal(const std::string & literal) {
    return parser(std::make_shared<literal_parser>(literal, next_id_++));
}

parser parser_builder::sequence(std::initializer_list<parser> parsers) {
    return parser(std::make_shared<sequence_parser>(parsers, next_id_++));
}

parser parser_builder::choice(std::initializer_list<parser> parsers) {
    return parser(std::make_shared<choice_parser>(parsers, next_id_++));
}

parser parser_builder::one_or_more(const parser & p) {
    return parser(std::make_shared<one_or_more_parser>(p, next_id_++));
}

parser parser_builder::zero_or_more(const parser & p) {
    return parser(std::make_shared<zero_or_more_parser>(p, next_id_++));
}

parser parser_builder::optional(const parser & p) {
    return parser(std::make_shared<optional_parser>(p, next_id_++));
}

parser parser_builder::negate(const parser & p) {
    return parser(std::make_shared<not_parser>(p, next_id_++));
}

parser parser_builder::any() {
    return parser(std::make_shared<any_parser>(next_id_++));
}

parser parser_builder::char_class(const std::string & classes) {
    return parser(std::make_shared<char_class_parser>(classes, next_id_++));
}

parser parser_builder::group(const std::string & name, const parser & p) {
    return parser(std::make_shared<group_parser>(name, p, next_id_++));
}

parser parser_builder::rule(const std::string & name) {
    return parser(std::make_shared<rule_parser>(name, rules_, next_id_++));
}

parser parser_builder::space() {
    return parser(std::make_shared<space_parser>(next_id_++));
}

parser parser_builder::until(const std::string & delimiter, bool include_spaces) {
    return parser(std::make_shared<until_parser>(delimiter, include_spaces, next_id_++, *this));
}

parser parser_builder::schema(const parser & p, const std::string & name, const nlohmann::ordered_json & schema) {
    return parser(std::make_shared<schema_parser>(p, name, schema, next_id_++));
}

parser parser_builder::add_rule(const std::string & name, const parser & p) {
    (*rules_)[name] = p;
    return rule(name);
}

void parser_builder::assign_ids(parser & p) {
    if (p.ptr) {
        p.ptr->assign_ids_internal(next_id_);
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

static parser json_parser() {
    parser_builder builder;

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

    // Value - uses forward references for recursive structures
    builder.add_rule("json_value",
        builder.rule("json_object") |
        builder.rule("json_array") |
        builder.rule("json_string") |
        builder.rule("json_number") |
        true_lit |
        false_lit |
        null_lit
    );

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

    // Get the json_value rule as the root
    auto root = builder.rule("json_value");
    builder.assign_ids(root);

    // Wrap in root_parser to own the rules
    return parser(std::make_shared<root_parser>(root, builder.rules(), -1));
}

parser parser_builder::json() {
    return json_parser();
}
