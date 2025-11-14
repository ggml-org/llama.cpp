#include "chat-parser-combinator.h"
#include "json-schema-to-grammar.h"
#include "common.h"
#include "chat.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <deque>
#include <memory>
#include <optional>

enum parser_type {
    PARSER_LITERAL,
    PARSER_SEQUENCE,
    PARSER_CHOICE,
    PARSER_REPETITION,
    PARSER_OPTIONAL,
    PARSER_ZERO_OR_MORE,
    PARSER_ONE_OR_MORE,
    PARSER_NOT,
    PARSER_ANY,
    PARSER_CHARS,
    PARSER_RULE,
    PARSER_UNTIL,
    PARSER_SPACE,
    PARSER_SCHEMA,
    PARSER_ROOT,
    PARSER_JSON_STRING,
    PARSER_ACTION,
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

    virtual void assign_id(std::shared_ptr<parser_id_counter> counter) {
        if (id_ == -1) {
            id_ = counter->next();
        }
    }

    virtual std::string dump() const = 0;
    virtual void accept(parser_visitor & visitor) = 0;
};

// Convenience cast functions
template<typename T>
static std::shared_ptr<T> cast(const std::shared_ptr<parser_base> & p) {
    if (p->type() != T::type_value) {
        return nullptr;
    }
    return std::static_pointer_cast<T>(p);
}

template<typename T>
static std::shared_ptr<T> cast(const parser & p) {
    return cast<T>(p.ptr());
}

// We define our own space function because MSVC's std::isspace()
// crashes for non-printable characters in Debug builds.
static bool is_space(const char c) {
    return (c == ' ' || c == '\t' || c == '\n');
}

static bool is_hex_digit(const char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

// Unescapes a JSON string (without the surrounding quotes)
// Uses nlohmann::json::parse to handle all JSON escape sequences
static std::string unescape_json_string(std::string_view str) {
    try {
        // Wrap in quotes and parse as JSON string
        std::string quoted = "\"" + std::string(str) + "\"";
        auto parsed = nlohmann::json::parse(quoted);
        if (parsed.is_string()) {
            return parsed.get<std::string>();
        }
    } catch (...) {
        // If parsing fails, return original string
    }
    return std::string(str);
}

// Aho-Corasick automation for matching multiple literals.
// This is used in until_parser and to build a GBNF exclusion grammar by
// exploiting its trie structure.
class aho_corasick_matcher {
    struct node {
        size_t fail = 0;
        size_t depth = 0;
        std::map<unsigned char, size_t> children;
        std::vector<size_t> word_lengths;
    };

    std::vector<node> trie;

  public:
    aho_corasick_matcher(const std::vector<std::string> & words) {
      create_node(); // root node
      for (const auto & w : words) {
          insert(w);
      }
      build_fail_links();
    }

    struct search_result {
        size_t pos;
        bool found;
        bool is_partial;
    };

    search_result search(std::string_view sv, size_t start = 0) {
      size_t current = 0;

      for (auto i = start; i < sv.size(); ++i) {
          // Aho-Corasick transition
          while (current != 0 && trie[current].children.find(sv[i]) == trie[current].children.end()) {
              current = trie[current].fail;
          }

          auto it = trie[current].children.find(sv[i]);
          if (it != trie[current].children.end()) {
              current = it->second;
          } else {
              current = 0;
          }

          if (!trie[current].word_lengths.empty()) {
              // Return back the longest word
              size_t pos = sv.size();
              for (const auto & len : trie[current].word_lengths) {
                  pos = std::min(pos, i - len + 1);
              }
              return search_result{pos, true, false};
          }
      }

      if (trie[current].depth > 0) {
          return search_result{sv.size() - trie[current].depth, true, true};
      }
      return search_result{sv.size(), false, false};
    }

    struct prefix_and_next {
        std::string prefix;
        std::string next_chars;
    };

    std::vector<prefix_and_next> collect_prefix_and_next() {
        std::string prefix;
        std::vector<prefix_and_next> result;
        collect_prefix_and_next(0, prefix, result);
        return result;
    }

  private:
    void collect_prefix_and_next(size_t index, std::string & prefix, std::vector<prefix_and_next> & out) {
        if (trie[index].word_lengths.empty()) {
            if (!trie[index].children.empty()) {
                std::string chars;
                chars.reserve(trie[index].children.size());
                for (const auto & p : trie[index].children) {
                    chars.push_back(p.first);
                }
                out.emplace_back(prefix_and_next{prefix, chars});
            }
        }

        for (const auto & p : trie[index].children) {
            unsigned char ch = p.first;
            int child = p.second;
            prefix.push_back(ch);
            collect_prefix_and_next(child, prefix, out);
            prefix.pop_back();
        }
    }

    size_t create_node() {
        size_t index = trie.size();
        trie.emplace_back();
        return index;
    }

    void insert(const std::string & word) {
        size_t current = 0;
        for (unsigned char ch : word) {
            auto it = trie[current].children.find(ch);
            if (it == trie[current].children.end()) {
                size_t child = create_node();
                trie[child].depth = trie[current].depth + 1;
                trie[current].children[ch] = child;
                current = child;
            } else {
                current = it->second;
            }
        }
        trie[current].word_lengths.push_back(word.length());
    }

    void build_fail_links() {
        std::deque<size_t> queue;

        size_t root = 0;
        trie[root].fail = 0;
        for (const auto & it : trie[root].children) {
            size_t child = it.second;
            trie[child].fail = 0;
            queue.push_back(child);
        }

        while (!queue.empty()) {
            size_t current = queue.front();
            queue.pop_front();

            for (const auto & p : trie[current].children) {
                unsigned char ch = p.first;
                size_t child = p.second;
                queue.push_back(child);

                auto fail = trie[current].fail;
                while (fail != 0 && trie[fail].children.find(p.first) == trie[fail].children.end()) {
                    fail = trie[fail].fail;
                }

                auto fail_it = trie[fail].children.find(ch);
                trie[child].fail = fail_it != trie[fail].children.end() ? fail_it->second : 0;
            }
        }
    }
};

// Generate an excluding pattern, with customized escaping
static std::string generic_excluding_pattern(
        const std::vector<std::string> & strings,
        const std::function<std::string(const std::string &)> & literal,
        const std::function<std::string(char c)> & escape_char_class,
        bool pad = false) {

    // Use the aho_corasick_matcher to grab an exhaustive list of prefixes and
    // potential next characters. We can use this to build an exclusion for
    // multiple strings.
    aho_corasick_matcher matcher(strings);
    auto pieces = matcher.collect_prefix_and_next();

    std::string pattern;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (i > 0) {
            pattern += pad ? " | " : "|";
        }

        const auto & pre = pieces[i].prefix;
        const auto & chars = pieces[i].next_chars;

        std::string cls;
        cls.reserve(chars.size());
        for (const auto & ch : chars) {
            cls += escape_char_class(ch);
        }

        if (!pre.empty()) {
            pattern += literal(pre) + (pad ? " [^" : "[^") + cls + "]";
        } else {
            pattern += "[^" + cls + "]";
        }
    }

    return "(" + pattern + ")*";
}

// Escape a single character for use in regex character classes
static std::string regex_escape_char_class(char c) {
    switch (c) {
        case '\\': return "\\\\";
        case ']':  return "\\]";
        case '-':  return "\\-";
        case '^':  return "\\^";
        default:   return std::string(1, c);
    }
}

// Create a regex excluding pattern
static std::string regex_excluding_pattern(const std::vector<std::string> & strings) {
    return generic_excluding_pattern(strings, regex_escape, regex_escape_char_class);
}

// Matches an exact literal string.
//   S -> "hello"
class literal_parser : public parser_base {
    std::string literal_;

  public:
    static constexpr parser_type type_value = PARSER_LITERAL;

    literal_parser(const std::string & literal, int id) : parser_base(id), literal_(literal) {}

    parser_type type() const override { return type_value; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto pos = start;
        for (auto i = 0u; i < literal_.size(); ++i) {
            if (pos >= ctx.input.size()) {
                if (ctx.input_is_complete) {
                    return parser_result(PARSER_RESULT_FAIL, start);
                }
                return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, pos);
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
    static constexpr parser_type type_value = PARSER_SEQUENCE;

    sequence_parser(std::initializer_list<parser> parsers, int id) : parser_base(id) {
        for (const auto & p : parsers) {
            if (auto seq = cast<sequence_parser>(p)) {
                for (const auto & embedded : seq->parsers()) {
                    parsers_.push_back(embedded);
                }
            } else {
                parsers_.push_back(p);
            }
        }
    }

    parser_type type() const override { return type_value; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto pos = start;
        for (const auto & p : parsers_) {
            auto result = p->parse(ctx, pos);
            if (!result.is_success()) {
                return parser_result(result.type, start, result.end);
            }

            pos = result.end;
        }

        return parser_result(PARSER_RESULT_SUCCESS, start, pos);
    }

    void assign_id(std::shared_ptr<parser_id_counter> counter) override {
        parser_base::assign_id(counter);
        for (auto & p : parsers_) {
            p->assign_id(counter);
        }
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
    static constexpr parser_type type_value = PARSER_CHOICE;

    choice_parser(std::initializer_list<parser> parsers, int id) : parser_base(id) {
        for (const auto & p : parsers) {
            if (auto choice = cast<choice_parser>(p)) {
                for (const auto & embedded : choice->parsers()) {
                    parsers_.push_back(embedded);
                }
            } else {
                parsers_.push_back(p);
            }
        }
    }

    parser_type type() const override { return type_value; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto pos = start;
        for (const auto & p : parsers_) {
            auto result = p->parse(ctx, pos);
            if (!result.is_fail()) {
                return result;
            }
        }

        return parser_result(PARSER_RESULT_FAIL, start);
    }

    void assign_id(std::shared_ptr<parser_id_counter> counter) override {
        parser_base::assign_id(counter);
        for (auto & p : parsers_) {
            p->assign_id(counter);
        }
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
    static constexpr parser_type type_value = PARSER_REPETITION;

    repetition_parser(const parser & parser, int min_count, int max_count, int id)
        : parser_base(id), parser_(parser), min_count_(min_count), max_count_(max_count) {}

    parser_type type() const override { return type_value; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto pos = start;
        int match_count = 0;

        // Try to match up to max_count times (or unlimited if max_count is -1)
        while (max_count_ == -1 || match_count < max_count_) {
            if (pos >= ctx.input.size()) {
                break;
            }

            auto result = parser_->parse(ctx, pos);

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
                return parser_result(result.type, start, result.end);
            }

            // Child failed - stop trying
            break;
        }

        // Check if we got enough matches
        if (match_count < min_count_) {
            return parser_result(PARSER_RESULT_FAIL, start, pos);
        }

        return parser_result(PARSER_RESULT_SUCCESS, start, pos);
    }

    void assign_id(std::shared_ptr<parser_id_counter> counter) override {
        parser_base::assign_id(counter);
        parser_->assign_id(counter);
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
    static constexpr parser_type type_value = PARSER_ONE_OR_MORE;

    one_or_more_parser(const parser & p, int id) : repetition_parser(p, 1, -1, id) {}

    parser_type type() const override { return type_value; }

    std::string dump() const override {
        return "OneOrMore(" + child()->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;
};

// Matches zero or more repetitions of a parser, always succeeds.
//   S -> A*
class zero_or_more_parser : public repetition_parser {
  public:
    static constexpr parser_type type_value = PARSER_ZERO_OR_MORE;

    zero_or_more_parser(const parser & p, int id) : repetition_parser(p, 0, -1, id) {}

    parser_type type() const override { return type_value; }

    std::string dump() const override {
        return "ZeroOrMore(" + child()->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;
};

// Matches zero or one occurrence of a parser, always succeeds.
//   S -> A?
class optional_parser : public repetition_parser {
  public:
    static constexpr parser_type type_value = PARSER_OPTIONAL;

    optional_parser(const parser & p, int id) : repetition_parser(p, 0, 1, id) {}

    parser_type type() const override { return type_value; }

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
    static constexpr parser_type type_value = PARSER_NOT;

    not_parser(const parser & parser, int id) : parser_base(id), parser_(parser) {}

    parser_type type() const override { return type_value; }

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

    void assign_id(std::shared_ptr<parser_id_counter> counter) override {
        parser_base::assign_id(counter);
        parser_->assign_id(counter);
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
    static constexpr parser_type type_value = PARSER_ANY;

    any_parser(int id) : parser_base(id) {}

    parser_type type() const override { return type_value; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        if (start >= ctx.input.size()) {
            if (ctx.input_is_complete) {
                return parser_result(PARSER_RESULT_FAIL, start);
            }
            return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start);
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
    static constexpr parser_type type_value = PARSER_SPACE;

    space_parser(int id) : parser_base(id) {}

    parser_type type() const override { return type_value; }

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

// Matches between min and max repetitions of characters from a character class.
//   S -> [a-z]{m,n}
class chars_parser : public parser_base {
    struct char_range {
        int start;
        int end;

        bool contains(char c) const { return (int)c >= start && int(c) <= end; }
    };

    std::string pattern_;
    std::vector<char_range> ranges_;
    bool negated_;
    int min_count_;
    int max_count_;

  public:
    chars_parser(const std::string & classes, int min_count, int max_count, int id)
        : parser_base(id), pattern_(classes), negated_(false), min_count_(min_count), max_count_(max_count) {

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

    static constexpr parser_type type_value = PARSER_CHARS;

    parser_type type() const override { return type_value; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto pos = start;
        int match_count = 0;

        // Try to match up to max_count times (or unlimited if max_count is -1)
        while (max_count_ == -1 || match_count < max_count_) {
            if (pos >= ctx.input.size()) {
                break;
            }

            bool matches = false;
            for (const auto & range : ranges_) {
                if (range.contains(ctx.input[pos])) {
                    matches = true;
                    break;
                }
            }

            // If negated, invert the match result
            if (negated_) {
                matches = !matches;
            }

            if (matches) {
                ++pos;
                ++match_count;
            } else {
                break;
            }
        }

        // Check if we got enough matches
        if (match_count < min_count_) {
            if (pos >= ctx.input.size() && !ctx.input_is_complete) {
                return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, pos);
            }
            return parser_result(PARSER_RESULT_FAIL, start, pos);
        }

        return parser_result(PARSER_RESULT_SUCCESS, start, pos);
    }

    std::string dump() const override {
        if (max_count_ == -1) {
            return "CharRepeat(" + pattern_ + ", " + std::to_string(min_count_) + ", unbounded)";
        }
        return "CharRepeat(" + pattern_ + ", " + std::to_string(min_count_) + ", " + std::to_string(max_count_) + ")";
    }

    void accept(parser_visitor & visitor) override;

    const std::string & pattern() const { return pattern_; }

    int min_count() const { return min_count_; }

    int max_count() const { return max_count_; }
};

// Specialized parser for JSON string content (without quotes).
// Parses the content between quotes with single-pass streaming support.
// Stops before the closing quote (doesn't consume it).
// Handles escape sequences and emits NEED_MORE_INPUT for incomplete input.
//   S -> (regular chars and escape sequences)* until closing "
class json_string_parser : public parser_base {

  public:
    static constexpr parser_type type_value = PARSER_JSON_STRING;

    json_string_parser(int id) : parser_base(id) {}

    parser_type type() const override { return type_value; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto pos = start;

        // Parse string content (without quotes)
        while (pos < ctx.input.size()) {
            char c = ctx.input[pos];

            if (c == '"') {
                // Found closing quote - success (don't consume it)
                return parser_result(PARSER_RESULT_SUCCESS, start, pos);
            }

            if (c == '\\') {
                // Handle escape sequence
                ++pos;
                if (pos >= ctx.input.size()) {
                    // Mid-escape sequence
                    if (ctx.input_is_complete) {
                        return parser_result(PARSER_RESULT_FAIL, start);
                    }
                    return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, pos);
                }

                char escape = ctx.input[pos];
                switch (escape) {
                    case '"':
                    case '\\':
                    case '/':
                    case 'b':
                    case 'f':
                    case 'n':
                    case 'r':
                    case 't':
                        // Valid escape
                        ++pos;
                        break;

                    case 'u':
                        // Unicode escape: must be followed by 4 hex digits
                        ++pos;
                        for (int i = 0; i < 4; ++i) {
                            if (pos >= ctx.input.size()) {
                                // Incomplete unicode escape
                                if (ctx.input_is_complete) {
                                    return parser_result(PARSER_RESULT_FAIL, start);
                                }
                                return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, pos);
                            }
                            if (!is_hex_digit(ctx.input[pos])) {
                                return parser_result(PARSER_RESULT_FAIL, start);
                            }
                            ++pos;
                        }
                        break;

                    default:
                        // Invalid escape sequence
                        return parser_result(PARSER_RESULT_FAIL, start);
                }
            } else {
                // Regular character
                ++pos;
            }
        }

        // Reached end without finding closing quote
        if (ctx.input_is_complete) {
            return parser_result(PARSER_RESULT_FAIL, start, pos);
        }
        return parser_result(PARSER_RESULT_NEED_MORE_INPUT, start, pos);
    }

    std::string dump() const override {
        return "JsonString()";
    }

    void accept(parser_visitor & visitor) override;
};

// Matches all characters until a delimiter is found (delimiter not consumed).
//   S -> (!delim .)*
class until_parser : public parser_base {
    std::vector<std::string> delimiters_;
    aho_corasick_matcher matcher_;

  public:
    static constexpr parser_type type_value = PARSER_UNTIL;

    until_parser(const std::vector<std::string> & delimiters, int id)
        : parser_base(id), delimiters_(delimiters), matcher_(delimiters) {}

    until_parser(const std::string & delimiter, int id)
        : until_parser(std::vector<std::string>{delimiter}, id) {}

    parser_type type() const override { return type_value; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto search_result = matcher_.search(ctx.input, start);
        return parser_result(PARSER_RESULT_SUCCESS, start, search_result.pos);
    }

    std::string dump() const override {
        return "Until(" + string_join(delimiters_, " | ") + ")";
    }

    void accept(parser_visitor & visitor) override;

    std::vector<std::string> delimiters() const { return delimiters_; }
};

// Wraps a parser with JSON schema metadata for grammar generation.
// Used internally to convert JSON schemas to GBNF grammar rules.
class schema_parser : public parser_base {
    parser parser_;
    std::string name_;
    nlohmann::ordered_json schema_;

  public:
    static constexpr parser_type type_value = PARSER_SCHEMA;

    schema_parser(const parser & parser, const std::string & name, const nlohmann::ordered_json & schema, int id)
        : parser_base(id), parser_(parser), name_(name), schema_(schema) {}

    parser_type type() const override { return type_value; }

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
    static constexpr parser_type type_value = PARSER_RULE;

    rule_parser(const std::string & name, const std::shared_ptr<std::unordered_map<std::string, parser>> & rules, int id)
        : parser_base(id), name_(name), rules_(rules) {}

    parser_type type() const override { return type_value; }

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
    static constexpr parser_type type_value = PARSER_ROOT;

    root_parser(const parser & root, std::shared_ptr<std::unordered_map<std::string, parser>> rules, int id)
        : parser_base(id), root_(root), rules_(std::move(rules)) {}

    parser_type type() const override { return type_value; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        return root_->parse(ctx, start);
    }

    void assign_id(std::shared_ptr<parser_id_counter> counter) override {
        parser_base::assign_id(counter);
        root_->assign_id(counter);
    }

    std::string dump() const override {
        return root_->dump();
    }

    void accept(parser_visitor & visitor) override;

    const parser & root() const { return root_; }

    std::shared_ptr<std::unordered_map<std::string, parser>> rules() const { return rules_; }
};

// Wraps a parser with a semantic action callback.
class action_parser : public parser_base {
    parser parser_;
    std::function<void(const parser_action &)> action_;
    int when_;

  public:
    static constexpr parser_type type_value = PARSER_ACTION;

    action_parser(
        const parser & parser,
        std::function<void(const parser_action &)> action,
        int when,
        int id
    ) : parser_base(id), parser_(parser), action_(std::move(action)), when_(when) {}

    parser_type type() const override { return type_value; }

    parser_result parse_uncached(parser_context & ctx, size_t start = 0) override {
        auto result = parser_->parse(ctx, start);

        if ((result.type & when_) && ctx.env && action_) {
            std::string_view matched = ctx.input;
            matched = matched.substr(result.start, result.end - result.start);
            action_({
                result,
                *ctx.env,
                matched,
            });
        }

        return result;
    }

    void assign_id(std::shared_ptr<parser_id_counter> counter) override {
        parser_base::assign_id(counter);
        parser_->assign_id(counter);
    }

    std::string dump() const override {
        return "Action(" + parser_->dump() + ", when=" + std::to_string(when_) +")";
    }

    void accept(parser_visitor & visitor) override;

    const parser & child() const { return parser_; }
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
    virtual void visit(chars_parser & p) = 0;
    virtual void visit(json_string_parser & p) = 0;
    virtual void visit(schema_parser & p) = 0;
    virtual void visit(rule_parser & p) = 0;
    virtual void visit(root_parser & p) = 0;
    virtual void visit(action_parser & p) = 0;
};

// Escape special characters for GBNF literals
static std::string gbnf_literal(const std::string & s) {
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
    return "\"" + escaped + "\"";
}

// Escape a single character for use in gbnf character classes
static std::string gbnf_escape_char_class(char c) {
    switch (c) {
        case '\n': return "\\n";
        case '\t': return "\\t";
        case '\r': return "\\r";
        default:   return regex_escape_char_class(c); // these too
    }
}

// Create a GBNF excluding pattern
static std::string gbnf_excluding_pattern(const std::vector<std::string> & strings) {
    return generic_excluding_pattern(strings, gbnf_literal, gbnf_escape_char_class, true);
}

class gbnf_visitor : public parser_visitor {
    const common_grammar_builder & builder_;
    std::unordered_map<std::string, std::string> rule_name_mapping_;
    std::string current_result_;

  public:
    gbnf_visitor(const common_grammar_builder & builder) : builder_(builder) {}

    const std::string& result() const { return current_result_; }

  private:
    // Check if expression needs parentheses
    static bool needs_parens(parser_type type) {
        return type == PARSER_CHOICE || type == PARSER_SEQUENCE;
    }

  public:
    void visit(literal_parser & p) override {
        current_result_ = gbnf_literal(p.literal());
    }

    void visit(sequence_parser & p) override {
        std::string s;
        for (const auto & child : p.parsers()) {
            if (!s.empty()) {
                s += " ";
            }
            child->accept(*this);

            // Parenthesize choices
            if (needs_parens(child->type())) {
                s += "(" + current_result_ + ")";
            } else {
                s += current_result_;
            }
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

            // Parenthesize choices
            if (child->type() == PARSER_CHOICE) {
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
        current_result_ = gbnf_excluding_pattern(p.delimiters());
    }

    void visit(not_parser &) override {
        // NOT is tricky in GBNF - for now, emit error
        LOG_ERR("NOT operator not directly supported in GBNF generation\n");
        current_result_ = "";
    }

    void visit(any_parser &) override {
        // Match any single character
        current_result_ = ".";
    }

    void visit(space_parser &) override {
        // Reference the built-in space rule
        current_result_ = "space";
    }

    void visit(chars_parser & p) override {
        const std::string & pattern = p.pattern();

        if (p.min_count() == 0 && p.max_count() == -1) {
            // Zero or more: *
            current_result_ = pattern + "*";
        } else if (p.min_count() == 1 && p.max_count() == -1) {
            // One or more: +
            current_result_ = pattern + "+";
        } else if (p.max_count() == -1) {
            // Unbounded: {n,}
            current_result_ = pattern + "{" + std::to_string(p.min_count()) + ",}";
        } else if (p.min_count() == p.max_count()) {
            // Exact count: {n} or just pattern for n=1
            if (p.min_count() == 1) {
                current_result_ = pattern;
            } else {
                current_result_ = pattern + "{" + std::to_string(p.min_count()) + "}";
            }
        } else {
            // Bounded: {n,m}
            current_result_ = pattern + "{" + std::to_string(p.min_count()) + "," +
                             std::to_string(p.max_count()) + "}";
        }
    }

    void visit(json_string_parser &) override {
        // JSON string content (without quotes)
        // Pattern: (any non-quote/backslash OR escape sequences)* until closing quote
        current_result_ = R"(( [^"\\] | "\\" ( ["\\/ bfnrt] | "u" [0-9a-fA-F]{4} ) )*)";
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

    void visit(action_parser & p) override {
        // Actions are transparent for grammar generation - just visit child
        p.child()->accept(*this);
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
void chars_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void json_string_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void schema_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void rule_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void root_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void action_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }

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

parser::parser(const std::string & literal) : ptr_(std::make_shared<literal_parser>(literal, -1)) {}

parser::parser(const char * literal) : ptr_(std::make_shared<literal_parser>(literal, -1)) {}

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

parser operator+(const char * lhs, const parser & rhs) { return parser(lhs) + rhs; }
parser operator|(const char * lhs, const parser & rhs) { return parser(lhs) | rhs; }
parser operator<<(const char * lhs, const parser & rhs) { return parser(lhs) << rhs; }

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

parser parser_builder::chars(const std::string & classes, int min, int max) {
    return parser(std::make_shared<chars_parser>(classes, min, max, counter_->next()));
}

parser parser_builder::one(const std::string & classes) {
    return chars(classes, 1, 1);
}

parser parser_builder::json_string() {
    return parser(std::make_shared<json_string_parser>(counter_->next()));
}

parser parser_builder::rule(const std::string & name) {
    return parser(std::make_shared<rule_parser>(name, rules_, counter_->next()));
}

parser parser_builder::space() {
    return parser(std::make_shared<space_parser>(counter_->next()));
}

parser parser_builder::until(const std::string & delimiter) {
    return parser(std::make_shared<until_parser>(delimiter, counter_->next()));
}

parser parser_builder::until_one_of(const std::vector<std::string> & delimiters) {
    return parser(std::make_shared<until_parser>(delimiters, counter_->next()));
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

parser parser_builder::action(const parser & p, std::function<void(const parser_action &)> fn, int when) {
    return parser(std::make_shared<action_parser>(p, std::move(fn), when, counter_->next()));
}

parser parser_builder::succeed(const parser & p, int when) {
    return action(p, [](const parser_action & act) {
        act.result.type = PARSER_RESULT_SUCCESS;
    }, when);
}

parser parser_builder::append_reasoning(const parser & p) {
    return action(p, [](const parser_action & act) {
        if (!act.env.result.reasoning_content.empty()) {
            act.env.result.reasoning_content += "\n";
        }
        act.env.result.reasoning_content += act.match;
    }, PARSER_RESULT_SUCCESS | PARSER_RESULT_NEED_MORE_INPUT);
}

parser parser_builder::append_content(const parser & p) {
    return action(p, [](const parser_action & act) {
        if (!act.env.result.content.empty()) {
            act.env.result.content += "\n";
        }
        act.env.result.content += act.match;
    }, PARSER_RESULT_SUCCESS | PARSER_RESULT_NEED_MORE_INPUT);
}

parser parser_builder::capture(const parser & p, const std::string & key, bool unescape_json) {
    return action(p, [key, unescape_json](const parser_action & act) {
        std::string value = unescape_json ? unescape_json_string(act.match) : std::string(act.match);
        act.env.scratchpad[key] = std::move(value);
    }, PARSER_RESULT_SUCCESS);
}

parser parser_builder::capture_tool_call_id(const parser & p, bool unescape_json) {
    return action(p, [unescape_json](const parser_action & act) {
        act.env.tool_call_id = unescape_json ? unescape_json_string(act.match) : std::string(act.match);
    }, PARSER_RESULT_SUCCESS);
}

parser parser_builder::capture_tool_call_name(const parser & p, bool unescape_json) {
    return action(p, [unescape_json](const parser_action & act) {
        act.env.tool_call_name = unescape_json ? unescape_json_string(act.match) : std::string(act.match);
    }, PARSER_RESULT_SUCCESS);
}

parser parser_builder::capture_tool_call_args(const parser & p, bool unescape_json) {
    return action(p, [unescape_json](const parser_action & act) {
        act.env.tool_call_args = unescape_json ? unescape_json_string(act.match) : std::string(act.match);
    }, PARSER_RESULT_SUCCESS | PARSER_RESULT_NEED_MORE_INPUT);
}

parser parser_builder::add_tool_call(const parser & p) {
    return action(p, [](const parser_action & act) {
        if (!act.env.tool_call_name.empty() && !act.env.tool_call_args.empty()) {
            auto tool_call = common_chat_tool_call{
                act.env.tool_call_name,
                act.env.tool_call_args,
                act.env.tool_call_id
            };
            act.env.result.tool_calls.push_back(tool_call);
        }

        // Clear the fields to prevent bleeding to next tool call
        act.env.tool_call_id.clear();
        act.env.tool_call_name.clear();
        act.env.tool_call_args.clear();
    }, PARSER_RESULT_SUCCESS | PARSER_RESULT_NEED_MORE_INPUT);
}

parser parser_builder::json_key(const std::string & name, const parser & p) {
    return literal("\"" + name + "\"") << literal(":") << p;
}

parser parser_builder::json_string(const parser & p) {
    auto quote = literal("\"");
    return quote + p + quote;
}

parser parser_builder::add_rule(const std::string & name, const parser & p) {
    (*rules_)[name] = p;
    return rule(name);
}

void parser_builder::assign_ids(parser & p) {
    if (p.ptr()) {
        p.ptr()->assign_id(counter_);
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
    auto ws = builder.space();

    // Number components
    auto digit1_9 = builder.chars("[1-9]", 1, 1);
    auto digits = builder.chars("[0-9]");

    // Integer part: 0 or non-zero digit followed by more digits
    auto int_part = builder.literal("0") | (digit1_9 + builder.chars("[0-9]", 0, -1));

    // Optional fractional part
    auto frac = builder.literal(".") + digits;

    // Optional exponent part
    auto exp = (builder.literal("e") | builder.literal("E")) + builder.optional(builder.chars("[+\\-]", 1, 1)) + digits;

    // Complete number
    auto number = builder.optional(builder.literal("-")) + int_part + builder.optional(frac) + builder.optional(exp);

    builder.add_rule("json_number", number);

    // String: specialized single-pass parser (content only, wrapped with quotes)
    auto string = builder.literal("\"") + builder.json_string() + builder.literal("\"");

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
