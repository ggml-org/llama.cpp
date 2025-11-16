#include "log.h"
#include "common.h"
#include "chat-peg-parser.h"
#include "json-schema-to-grammar.h"
#include "unicode.h"

#include <nlohmann/json.hpp>

#include <deque>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>

enum parser_type {
    START,
    END,
    LITERAL,
    SEQUENCE,
    CHOICE,
    REPETITION,
    OPTIONAL,
    ZERO_OR_MORE,
    ONE_OR_MORE,
    AND,
    NOT,
    ANY,
    CHARS,
    RULE,
    REF,
    UNTIL,
    SPACE,
    SCHEMA,
    ROOT,
    JSON_STRING,
    CAPTURE,
};

const char * common_chat_parse_result_type_name(common_chat_parse_result_type type) {
    switch (type) {
        case COMMON_CHAT_PARSE_RESULT_FAIL:            return "fail";
        case COMMON_CHAT_PARSE_RESULT_SUCCESS:         return "success";
        case COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT: return "need_more_input";
        default:                                       return "unknown";
    }
}

static const char * common_chat_parser_type_name(parser_type type) {
    switch (type) {
        case START:        return "start";
        case END:          return "end";
        case LITERAL:      return "literal";
        case SEQUENCE:     return "sequence";
        case CHOICE:       return "choice";
        case REPETITION:   return "repetition";
        case OPTIONAL:     return "optional";
        case ZERO_OR_MORE: return "zero_or_more";
        case ONE_OR_MORE:  return "one_or_more";
        case AND:          return "and";
        case NOT:          return "not";
        case ANY:          return "any";
        case CHARS:        return "chars";
        case RULE:         return "rule";
        case REF:          return "ref";
        case UNTIL:        return "until";
        case SPACE:        return "space";
        case SCHEMA:       return "schema";
        case ROOT:         return "root";
        case JSON_STRING:  return "json_string";
        case CAPTURE:      return "capture";
        default:           return "unknown";
    }
}

class parser_visitor;

class common_chat_peg_parser_base {
  protected:
    int id_;

  public:
    common_chat_peg_parser_base(int id) : id_(id) {}
    virtual ~common_chat_peg_parser_base() = default;

    int id() const { return id_; }
    void set_id(int id) { id_ = id; }

    virtual parser_type type() const = 0;

    virtual common_chat_parse_result parse(common_chat_parse_context & ctx, size_t start = 0) {
        std::string indent(ctx.parse_depth * 2, ' ');
        ctx.parse_depth++;

        LOG_DBG("%s[CCPP type %d] Rule: %s\n", indent.c_str(), type(), dump().c_str());
        LOG_DBG("%s[CCPP type %d] Trying to parse: %s\n", indent.c_str(), type(), ctx.input.substr(start).c_str());
        if (id_ == -1) {
            // Don't cache parsers with ID -1 (from operators)
            LOG_DBG("%s[CCPP type %d] Parsing uncached due to operator\n", indent.c_str(), type());
            ctx.parse_depth--;
            return parse_uncached(ctx, start);
        }

        auto cached = ctx.cache.get(id_, start);
        if (cached) {
            LOG_DBG("%s[CCPP type %d] Found cached result, returning\n", indent.c_str(), type());
            ctx.parse_depth--;
            return *cached;
        }

        auto result = parse_uncached(ctx, start);
        LOG_DBG("%s[CCPP type %d] Parse result is: %s\n", indent.c_str(), type(), common_chat_parse_result_type_name(result.type));
        ctx.parse_depth--;
        return ctx.cache.set(id_, start, result);
    }

    // Actual parsing implementation (to be overridden by subclasses)
    virtual common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) = 0;

    virtual void assign_id(common_chat_peg_parser_counter & counter) {
        if (id_ == -1) {
            id_ = counter.next();
        }
    }

    virtual std::string dump() const = 0;
    virtual void accept(parser_visitor & visitor) = 0;
};

// Create an internal parser
template <typename Parser, typename... Args>
static std::shared_ptr<common_chat_peg_parser_base> make_parser(int id, Args&&... args) {
    return std::make_shared<Parser>(std::forward<Args>(args)..., id);
}

template <typename Parser, typename... Args>
static std::shared_ptr<common_chat_peg_parser_base> make_parser(common_chat_peg_parser_counter & counter, Args&&... args) {
    return std::make_shared<Parser>(std::forward<Args>(args)..., counter.next());
}

// Convenience cast functions
template<typename T>
static std::shared_ptr<T> cast(const std::shared_ptr<common_chat_peg_parser_base> & p) {
    if (p->type() != T::type_value) {
        return nullptr;
    }
    return std::static_pointer_cast<T>(p);
}

template<typename T>
static std::shared_ptr<T> cast(const common_chat_peg_parser & p) {
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
            auto child = p.second;
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

// Container for the root parser and all named rules in the grammar.
// Manages ownership of rule registry to enable recursive grammar definitions.
class root_parser : public common_chat_peg_parser_base {
    common_chat_peg_parser root_;
    std::unordered_map<std::string, common_chat_peg_parser> rules_;

  public:
    static constexpr parser_type type_value = ROOT;
    parser_type type() const override { return type_value; }

    root_parser(int id) : common_chat_peg_parser_base(id) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        return root_->parse(ctx, start);
    }

    void assign_id(common_chat_peg_parser_counter & counter) override {
        common_chat_peg_parser_base::assign_id(counter);
        for (auto & [name, rule] : rules_) {
            rule->assign_id(counter);
        }
        if (root_.ptr()) {
            root_->assign_id(counter);
        }
    }

    std::string dump() const override {
        return root_->dump();
    }

    void accept(parser_visitor & visitor) override;

    void add_rule(const std::string & name, const common_chat_peg_parser & parser) {
        rules_[name] = parser;
    }

    void set_root(const common_chat_peg_parser & parser) {
        root_ = parser;
    }

    const common_chat_peg_parser & root() const { return root_; }

    std::unordered_map<std::string, common_chat_peg_parser> & rules() { return rules_; }
    const std::unordered_map<std::string, common_chat_peg_parser> & rules() const { return rules_; }
};

// Matches the start of the input
//   S -> ^
class start_parser : public common_chat_peg_parser_base {
  public:
    static constexpr parser_type type_value = START;
    parser_type type() const override { return type_value; }

    start_parser(int id) : common_chat_peg_parser_base(id) {}

    void accept(parser_visitor & visitor) override;
    std::string dump() const override { return "Start"; }

    common_chat_parse_result parse_uncached(common_chat_parse_context & /*ctx*/, size_t start = 0) override {
        return common_chat_parse_result(start == 0 ? COMMON_CHAT_PARSE_RESULT_SUCCESS : COMMON_CHAT_PARSE_RESULT_FAIL, start);
    }
};

// Matches the end of the input
//   S -> $
class end_parser : public common_chat_peg_parser_base {
  public:
    static constexpr parser_type type_value = END;
    parser_type type() const override { return type_value; }

    end_parser(int id) : common_chat_peg_parser_base(id) {}

    void accept(parser_visitor & visitor) override;
    std::string dump() const override { return "End"; }

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        return common_chat_parse_result(start >= ctx.input.size() ? COMMON_CHAT_PARSE_RESULT_SUCCESS : COMMON_CHAT_PARSE_RESULT_FAIL, start);
    }
};

// Matches an exact literal string.
//   S -> "hello"
class literal_parser : public common_chat_peg_parser_base {
    std::string literal_;

  public:
    static constexpr parser_type type_value = LITERAL;
    parser_type type() const override { return type_value; }

    literal_parser(const std::string & literal, int id) : common_chat_peg_parser_base(id), literal_(literal) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        auto pos = start;
        for (auto i = 0u; i < literal_.size(); ++i) {
            if (pos >= ctx.input.size()) {
                if (ctx.input_is_complete) {
                    return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
                }
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
            }
            if (ctx.input[pos] != literal_[i]) {
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
            }
            ++pos;
        }

        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, pos);
    }

    std::string dump() const override {
        return "Literal(" + literal_ + ")";
    }

    void accept(parser_visitor & visitor) override;

    const std::string & literal() const { return literal_; }
};

// Matches a sequence of parsers in order, all must succeed.
//   S -> A B C
class sequence_parser : public common_chat_peg_parser_base {
    std::vector<common_chat_peg_parser> parsers_;

  public:
    static constexpr parser_type type_value = SEQUENCE;
    parser_type type() const override { return type_value; }

    template <typename InputIt>
    sequence_parser(InputIt first, InputIt last, int id) : common_chat_peg_parser_base(id) {
        for (auto it = first; it != last; ++it) {
            if (auto seq = cast<sequence_parser>(*it)) {
                parsers_.insert(parsers_.end(), seq->parsers().begin(), seq->parsers().end());
            } else {
                parsers_.push_back(*it);
            }
        }
    }

    template <typename T>
    sequence_parser(const T & parsers, int id)
        : sequence_parser(std::begin(parsers), std::end(parsers), id) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        auto pos = start;
        for (const auto & p : parsers_) {
            auto result = p->parse(ctx, pos);
            if (!result.success()) {
                return common_chat_parse_result(result.type, start, result.end);
            }

            pos = result.end;
        }

        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, pos);
    }

    void assign_id(common_chat_peg_parser_counter & counter) override {
        common_chat_peg_parser_base::assign_id(counter);
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

    const std::vector<common_chat_peg_parser> & parsers() const { return parsers_; }
};

// Matches the first parser that succeeds from a list of alternatives.
//   S -> A | B | C
class choice_parser : public common_chat_peg_parser_base {
    std::vector<common_chat_peg_parser> parsers_;

  public:
    static constexpr parser_type type_value = CHOICE;
    parser_type type() const override { return type_value; }

    template <typename InputIt>
    choice_parser(InputIt first, InputIt last, int id) : common_chat_peg_parser_base(id) {
        for (auto it = first; it != last; ++it) {
            if (auto choice = cast<choice_parser>(*it)) {
                parsers_.insert(parsers_.end(), choice->parsers().begin(), choice->parsers().end());
            } else {
                parsers_.push_back(*it);
            }
        }
    }

    template <typename T>
    choice_parser(const T & parsers, int id)
        : choice_parser(std::begin(parsers), std::end(parsers), id) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        auto pos = start;
        for (const auto & p : parsers_) {
            auto result = p->parse(ctx, pos);
            if (!result.fail()) {
                return result;
            }
        }

        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
    }

    void assign_id(common_chat_peg_parser_counter & counter) override {
        common_chat_peg_parser_base::assign_id(counter);
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

    const std::vector<common_chat_peg_parser> & parsers() const { return parsers_; }
};

// Matches between min and max repetitions of a parser (inclusive).
//   S -> A{m,n}
// Use -1 for max_count to represent unbounded repetition (equivalent to {m,})
class repetition_parser : public common_chat_peg_parser_base {
    common_chat_peg_parser parser_;
    int min_count_;
    int max_count_;

  public:
    static constexpr parser_type type_value = REPETITION;
    parser_type type() const override { return type_value; }

    repetition_parser(const common_chat_peg_parser & parser, int min_count, int max_count, int id)
        : common_chat_peg_parser_base(id), parser_(parser), min_count_(min_count), max_count_(max_count) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        auto pos = start;
        int match_count = 0;

        // Try to match up to max_count times (or unlimited if max_count is -1)
        while (max_count_ == -1 || match_count < max_count_) {
            if (pos >= ctx.input.size()) {
                break;
            }

            auto result = parser_->parse(ctx, pos);

            if (result.success()) {
                // Prevent infinite loop on empty matches
                if (result.end == pos) {
                    break;
                }
                pos = result.end;
                match_count++;
                continue;
            }

            if (result.need_more_input()) {
                return common_chat_parse_result(result.type, start, result.end);
            }

            // Child failed - stop trying
            break;
        }

        // Check if we got enough matches
        if (min_count_ > 0 && match_count < min_count_) {
            if (pos >= ctx.input.size() && !ctx.input_is_complete) {
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
            }
            return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start, pos);
        }

        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, pos);
    }

    void assign_id(common_chat_peg_parser_counter & counter) override {
        common_chat_peg_parser_base::assign_id(counter);
        parser_->assign_id(counter);
    }

    std::string dump() const override {
        if (max_count_ == -1) {
            return "Repetition(" + parser_->dump() + ", " + std::to_string(min_count_) + ", unbounded)";
        }
        return "Repetition(" + parser_->dump() + ", " + std::to_string(min_count_) + ", " + std::to_string(max_count_) + ")";
    }

    void accept(parser_visitor & visitor) override;

    const common_chat_peg_parser & child() const { return parser_; }

    int min_count() const { return min_count_; }

    int max_count() const { return max_count_; }
};

// Matches one or more repetitions of a parser.
//   S -> A+
class one_or_more_parser : public repetition_parser {
  public:
    static constexpr parser_type type_value = ONE_OR_MORE;
    parser_type type() const override { return type_value; }

    one_or_more_parser(const common_chat_peg_parser & p, int id) : repetition_parser(p, 1, -1, id) {}

    std::string dump() const override {
        return "OneOrMore(" + child()->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;
};

// Matches zero or more repetitions of a parser, always succeeds.
//   S -> A*
class zero_or_more_parser : public repetition_parser {
  public:
    static constexpr parser_type type_value = ZERO_OR_MORE;
    parser_type type() const override { return type_value; }

    zero_or_more_parser(const common_chat_peg_parser & p, int id) : repetition_parser(p, 0, -1, id) {}

    std::string dump() const override {
        return "ZeroOrMore(" + child()->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;
};

// Matches zero or one occurrence of a parser, always succeeds.
//   S -> A?
class optional_parser : public repetition_parser {
  public:
    static constexpr parser_type type_value = OPTIONAL;
    parser_type type() const override { return type_value; }

    optional_parser(const common_chat_peg_parser & p, int id) : repetition_parser(p, 0, 1, id) {}

    std::string dump() const override {
        return "Optional(" + child()->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;
};

// Positive lookahead: succeeds if child parser succeeds, consumes no input.
//   S -> &A
class and_parser : public common_chat_peg_parser_base {
    common_chat_peg_parser parser_;

  public:
    static constexpr parser_type type_value = AND;
    parser_type type() const override { return type_value; }

    and_parser(const common_chat_peg_parser & parser, int id) : common_chat_peg_parser_base(id), parser_(parser) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        auto result = parser_->parse(ctx, start);
        // Pass result but don't consume input
        return common_chat_parse_result(result.type, start);
    }

    void assign_id(common_chat_peg_parser_counter & counter) override {
        common_chat_peg_parser_base::assign_id(counter);
        parser_->assign_id(counter);
    }

    std::string dump() const override {
        return "And(" + parser_->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;

    const common_chat_peg_parser & child() const { return parser_; }
};

// Negative lookahead: succeeds if child parser fails, consumes no input.
//   S -> !A
class not_parser : public common_chat_peg_parser_base {
    common_chat_peg_parser parser_;

  public:
    static constexpr parser_type type_value = NOT;
    parser_type type() const override { return type_value; }

    not_parser(const common_chat_peg_parser & parser, int id) : common_chat_peg_parser_base(id), parser_(parser) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        auto result = parser_->parse(ctx, start);

        if (result.success()) {
            // Fail if the underlying parser matches
            return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
        }

        if (result.need_more_input()) {
            // Propagate - need to know what child would match before negating
            return result;
        }

        // Child failed, so negation succeeds
        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start);
    }

    void assign_id(common_chat_peg_parser_counter & counter) override {
        common_chat_peg_parser_base::assign_id(counter);
        parser_->assign_id(counter);
    }

    std::string dump() const override {
        return "Not(" + parser_->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;

    const common_chat_peg_parser & child() const { return parser_; }
};

// Matches any single character.
//   S -> .
class any_parser : public common_chat_peg_parser_base {
  public:
    static constexpr parser_type type_value = ANY;
    parser_type type() const override { return type_value; }

    any_parser(int id) : common_chat_peg_parser_base(id) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        // Parse a single UTF-8 codepoint (not just a single byte)
        auto result = parse_utf8_codepoint(ctx.input, start);

        if (result.status == utf8_parse_result::INCOMPLETE) {
            if (ctx.input_is_complete) {
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
            }
            return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT, start);
        }
        if (result.status == utf8_parse_result::INVALID) {
            return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
        }
        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, start + result.bytes_consumed);
    }

    std::string dump() const override {
        return "Any";
    }

    void accept(parser_visitor & visitor) override;
};

// Matches zero or more whitespace characters (space, tab, newline).
//   S -> [ \t\n]*
class space_parser : public common_chat_peg_parser_base {
  public:
    static constexpr parser_type type_value = SPACE;
    parser_type type() const override { return type_value; }

    space_parser(int id) : common_chat_peg_parser_base(id) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        auto pos = start;
        while (pos < ctx.input.size()) {
            char c = ctx.input[pos];
            if (is_space(c)) {
                ++pos;
            } else {
                break;
            }
        }

        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, pos);
    }

    std::string dump() const override {
        return "Space";
    }

    void accept(parser_visitor & visitor) override;
};

static std::pair<uint32_t, size_t> parse_hex_escape(const std::string & str, size_t pos, int hex_count) {
    if (pos + hex_count > str.length()) {
        return {0, 0};
    }

    uint32_t value = 0;
    for (int i = 0; i < hex_count; i++) {
        char c = str[pos + i];
        if (!is_hex_digit(c)) {
            return {0, 0};
        }
        value <<= 4;
        if ('a' <= c && c <= 'f') {
            value += c - 'a' + 10;
        } else if ('A' <= c && c <= 'F') {
            value += c - 'A' + 10;
        } else if ('0' <= c && c <= '9') {
            value += c - '0';
        } else {
            break;
        }
    }
    return {value, static_cast<size_t>(hex_count)};
}

static std::pair<uint32_t, size_t> parse_char_class_char(const std::string & content, size_t pos) {
    if (content[pos] == '\\' && pos + 1 < content.length()) {
        switch (content[pos + 1]) {
            case 'x': {
                auto result = parse_hex_escape(content, pos + 2, 2);
                if (result.second > 0) {
                    return {result.first, 2 + result.second};
                }
                // Invalid escape, treat as literal 'x'
                return {static_cast<uint32_t>('x'), 2};
            }
            case 'u': {
                auto result = parse_hex_escape(content, pos + 2, 4);
                if (result.second > 0) {
                    return {result.first, 2 + result.second};
                }
                // Invalid escape, treat as literal 'u'
                return {static_cast<uint32_t>('u'), 2};
            }
            case 'U': {
                auto result = parse_hex_escape(content, pos + 2, 8);
                if (result.second > 0) {
                    return {result.first, 2 + result.second};
                }
                // Invalid escape, treat as literal 'U'
                return {static_cast<uint32_t>('U'), 2};
            }
            case 'n':  return {'\n', 2};
            case 't':  return {'\t', 2};
            case 'r':  return {'\r', 2};
            case '\\': return {'\\', 2};
            case ']':  return {']', 2};
            case '-':  return {'-', 2};
            case '[':  return {'[', 2};
            default:   return {static_cast<uint32_t>(content[pos + 1]), 2};
        }
    }

    // Regular character - return as codepoint
    return {static_cast<uint32_t>(static_cast<unsigned char>(content[pos])), 1};
}

// Matches between min and max repetitions of characters from a character class.
//   S -> [a-z]{m,n}
// Supports Unicode codepoint ranges and escape sequences: \xXX \uXXXX \UXXXXXXXX
class chars_parser : public common_chat_peg_parser_base {
    struct char_range {
        uint32_t start;
        uint32_t end;

        bool contains(uint32_t codepoint) const { return codepoint >= start && codepoint <= end; }
    };

    std::string pattern_;
    std::vector<char_range> ranges_;
    bool negated_;
    int min_count_;
    int max_count_;

  public:
    static constexpr parser_type type_value = CHARS;
    parser_type type() const override { return type_value; }

    chars_parser(const std::string & classes, int min_count, int max_count, int id)
        : common_chat_peg_parser_base(id), pattern_(classes), negated_(false), min_count_(min_count), max_count_(max_count) {

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

        size_t i = 0;
        while (i < content.length()) {
            auto [start, start_len] = parse_char_class_char(content, i);
            i += start_len;

            if (i + 1 < content.length() && content[i] == '-') {
                // Range detected
                auto [end, end_len] = parse_char_class_char(content, i + 1);
                ranges_.push_back(char_range{start, end});
                i += 1 + end_len;
            } else {
                ranges_.push_back(char_range{start, start});
            }
        }
    }

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        auto pos = start;
        int match_count = 0;

        // Try to match up to max_count times (or unlimited if max_count is -1)
        while (max_count_ == -1 || match_count < max_count_) {
            auto result = parse_utf8_codepoint(ctx.input, pos);

            if (result.status == utf8_parse_result::INCOMPLETE) {
                if (match_count >= min_count_) {
                    // We have enough matches, succeed with what we have
                    return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, pos);
                }
                // Not enough matches yet
                if (ctx.input_is_complete) {
                    return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
                }
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
            }

            if (result.status == utf8_parse_result::INVALID) {
                // Malformed UTF-8 in input
                if (match_count >= min_count_) {
                    // We have enough matches, succeed up to here
                    return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, pos);
                }
                // Not enough matches, fail
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
            }

            // Check if this codepoint matches our character class
            bool matches = false;
            for (const auto & range : ranges_) {
                if (range.contains(result.codepoint)) {
                    matches = true;
                    break;
                }
            }

            // If negated, invert the match result
            if (negated_) {
                matches = !matches;
            }

            if (matches) {
                pos += result.bytes_consumed;
                ++match_count;
            } else {
                // Character doesn't match, stop matching
                break;
            }
        }

        // Check if we got enough matches
        if (match_count < min_count_) {
            if (pos >= ctx.input.size() && !ctx.input_is_complete) {
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
            }
            return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start, pos);
        }

        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, pos);
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
class json_string_parser : public common_chat_peg_parser_base {
  public:
    static constexpr parser_type type_value = JSON_STRING;
    parser_type type() const override { return type_value; }

    json_string_parser(int id) : common_chat_peg_parser_base(id) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        auto pos = start;

        // Parse string content (without quotes)
        while (pos < ctx.input.size()) {
            char c = ctx.input[pos];

            if (c == '"') {
                // Found closing quote - success (don't consume it)
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, pos);
            }

            if (c == '\\') {
                auto result = handle_escape_sequence(ctx, start, pos);
                if (!result.success()) {
                    return result;
                }
            } else {
                auto utf8_result = parse_utf8_codepoint(ctx.input, pos);

                if (utf8_result.status == utf8_parse_result::INCOMPLETE) {
                    if (ctx.input_is_complete) {
                        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
                    }
                    return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
                }

                if (utf8_result.status == utf8_parse_result::INVALID) {
                    return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
                }

                pos += utf8_result.bytes_consumed;
            }
        }

        // Reached end without finding closing quote
        if (ctx.input_is_complete) {
            return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start, pos);
        }
        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
    }

    std::string dump() const override {
        return "JsonString()";
    }

    void accept(parser_visitor & visitor) override;

  private:
    static common_chat_parse_result handle_escape_sequence(common_chat_parse_context & ctx, size_t start, size_t & pos) {
        ++pos; // consume '\'
        if (pos >= ctx.input.size()) {
            if (ctx.input_is_complete) {
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
            }
            return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
        }

        switch (ctx.input[pos]) {
            case '"':
            case '\\':
            case '/':
            case 'b':
            case 'f':
            case 'n':
            case 'r':
            case 't':
                ++pos;
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, pos);
            case 'u':
                return handle_unicode_escape(ctx, start, pos);
            default:
                // Invalid escape sequence
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
        }
    }

    static common_chat_parse_result handle_unicode_escape(common_chat_parse_context & ctx, size_t start, size_t & pos) {
        ++pos; // consume 'u'
        for (int i = 0; i < 4; ++i) {
            if (pos >= ctx.input.size()) {
                if (ctx.input_is_complete) {
                    return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
                }
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
            }
            if (!is_hex_digit(ctx.input[pos])) {
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
            }
            ++pos;
        }
        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, pos);
    }
};

// Matches all characters until a delimiter is found (delimiter not consumed).
//   S -> (!delim .)*
class until_parser : public common_chat_peg_parser_base {
    std::vector<std::string> delimiters_;
    aho_corasick_matcher matcher_;

  public:
    static constexpr parser_type type_value = UNTIL;
    parser_type type() const override { return type_value; }

    until_parser(const std::vector<std::string> & delimiters, int id)
        : common_chat_peg_parser_base(id), delimiters_(delimiters), matcher_(delimiters) {}

    until_parser(const std::string & delimiter, int id)
        : until_parser(std::vector<std::string>{delimiter}, id) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        // First pass: byte-based Aho-Corasick search for delimiter
        auto search_result = matcher_.search(ctx.input, start);
        size_t delimiter_pos = search_result.pos;

        // Second pass: validate UTF-8 from start to delimiter_pos
        size_t pos = start;
        size_t last_valid_pos = start;

        while (pos < delimiter_pos) {
            auto utf8_result = parse_utf8_codepoint(ctx.input, pos);

            if (utf8_result.status == utf8_parse_result::INCOMPLETE) {
                // Incomplete UTF-8 sequence before delimiter
                if (ctx.input_is_complete) {
                    // Input is complete but UTF-8 is incomplete = malformed
                    return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
                }
                // Return what we have so far (before incomplete sequence)
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_NEED_MORE_INPUT, start, last_valid_pos);
            }

            if (utf8_result.status == utf8_parse_result::INVALID) {
                // Malformed UTF-8
                return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
            }

            pos += utf8_result.bytes_consumed;
            last_valid_pos = pos;
        }

        // All UTF-8 validated up to delimiter
        return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_SUCCESS, start, last_valid_pos);
    }

    std::string dump() const override {
        return "Until(" + string_join(delimiters_, " | ") + ")";
    }

    void accept(parser_visitor & visitor) override;

    std::vector<std::string> delimiters() const { return delimiters_; }
};

// Wraps a parser with JSON schema metadata for grammar generation.
// Used internally to convert JSON schemas to GBNF grammar rules.
class schema_parser : public common_chat_peg_parser_base {
    common_chat_peg_parser parser_;
    std::string name_;
    nlohmann::ordered_json schema_;

  public:
    static constexpr parser_type type_value = SCHEMA;
    parser_type type() const override { return type_value; }

    schema_parser(const common_chat_peg_parser & parser, const std::string & name, const nlohmann::ordered_json & schema, int id)
        : common_chat_peg_parser_base(id), parser_(parser), name_(name), schema_(schema) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        return parser_->parse(ctx, start);
    }

    std::string dump() const override {
        return "Schema(" + parser_->dump() + ", " + schema_.dump() + ")";
    }

    void accept(parser_visitor & visitor) override;

    const common_chat_peg_parser & child() const { return parser_; }

    const std::string & name() const { return name_; }

    const nlohmann::ordered_json & schema() const { return schema_; }
};

// Defines a named rule for recursive or reusable grammar definitions.
// Owns the implementation and fires NODE_START/END events.
//   expr -> term | expr "+" term
class rule_parser : public common_chat_peg_parser_base, public std::enable_shared_from_this<rule_parser> {
    std::string name_;
    common_chat_peg_parser child_;
    bool trigger_;

  public:
    static constexpr parser_type type_value = RULE;
    parser_type type() const override { return type_value; }

    rule_parser(const std::string & name, const common_chat_peg_parser & child, bool trigger, int id)
        : common_chat_peg_parser_base(id), name_(name), child_(child), trigger_(trigger) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        // Fire NODE_START event
        if (ctx.event_handler && ctx.semantics) {
            ctx.event_handler(common_chat_parse_event{
                COMMON_CHAT_PARSE_EVENT_NODE_START,
                name_,
                start,
                start,
                "",
                COMMON_CHAT_PARSE_RESULT_FAIL,
                ctx.current_depth
            }, *ctx.semantics);
            ctx.current_depth++;
        }

        // Parse the child
        auto result = child_->parse(ctx, start);

        // Fire NODE_END event
        if (ctx.event_handler && ctx.semantics) {
            ctx.current_depth--;
            std::string_view text = ctx.input;
            if (result.start < ctx.input.size()) {
                text = text.substr(result.start, result.end - result.start);
            } else {
                text = "";
            }
            ctx.event_handler(common_chat_parse_event{
                COMMON_CHAT_PARSE_EVENT_NODE_END,
                name_,
                result.start,
                result.end,
                text,
                result.type,
                ctx.current_depth
            }, *ctx.semantics);
        }

        return result;
    }

    void assign_id(common_chat_peg_parser_counter & counter) override {
        common_chat_peg_parser_base::assign_id(counter);
        child_->assign_id(counter);
    }

    std::string dump() const override {
        return "Rule(" + name_ + ", " + child_->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;

    const std::string & name() const { return name_; }
    const common_chat_peg_parser & child() const { return child_; }
    bool is_trigger() const { return trigger_; }
};

// References a named rule (lightweight reference, resolved during resolution phase)
//   expr_ref -> expr
class ref_parser : public common_chat_peg_parser_base, public std::enable_shared_from_this<ref_parser> {
    std::string name_;
    std::weak_ptr<rule_parser> target_;

  public:
    static constexpr parser_type type_value = REF;
    parser_type type() const override { return type_value; }

    ref_parser(const std::string & name, int id)
        : common_chat_peg_parser_base(id), name_(name) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        auto target = target_.lock();
        if (!target) {
            LOG_ERR("ref_parser::parse called with unresolved reference '%s'\n", name_.c_str());
            return common_chat_parse_result(COMMON_CHAT_PARSE_RESULT_FAIL, start);
        }

        // Delegate to the target rule parser
        return target->parse(ctx, start);
    }

    std::string dump() const override {
        return "Ref(" + name_ + ")";
    }

    void accept(parser_visitor & visitor) override;

    const std::string & name() const { return name_; }
    void set_target(const std::weak_ptr<rule_parser> & target) { target_ = target; }
    std::weak_ptr<rule_parser> target() const { return target_; }
};

// Capture content if child parser matches
class capture_parser : public common_chat_peg_parser_base {
    common_chat_peg_parser parser_;
    std::string key_;

  public:
    static constexpr parser_type type_value = CAPTURE;
    parser_type type() const override { return type_value; }

    capture_parser(const common_chat_peg_parser & parser, const std::string & key, int id)
        : common_chat_peg_parser_base(id), parser_(parser), key_(key) {}

    common_chat_parse_result parse_uncached(common_chat_parse_context & ctx, size_t start = 0) override {
        auto result = parser_->parse(ctx, start);

        if (!result.fail() && ctx.semantics) {
            std::string_view matched = ctx.input;
            matched = matched.substr(result.start, result.end - result.start);
            std::string value = std::string(matched);
            ctx.semantics->captures[key_] = std::move(value);
        }

        return result;
    }

    void assign_id(common_chat_peg_parser_counter & counter) override {
        common_chat_peg_parser_base::assign_id(counter);
        parser_->assign_id(counter);
    }

    std::string dump() const override {
        return "Capture(" + key_ + ", " + parser_->dump() + ")";
    }

    void accept(parser_visitor & visitor) override;

    const common_chat_peg_parser & child() const { return parser_; }
};

// Base visitor class for parser tree traversal
class parser_visitor {
  public:
    virtual ~parser_visitor() = default;

    virtual void visit(start_parser & p) = 0;
    virtual void visit(end_parser & p) = 0;
    virtual void visit(literal_parser & p) = 0;
    virtual void visit(sequence_parser & p) = 0;
    virtual void visit(choice_parser & p) = 0;
    virtual void visit(one_or_more_parser & p) = 0;
    virtual void visit(zero_or_more_parser & p) = 0;
    virtual void visit(optional_parser & p) = 0;
    virtual void visit(repetition_parser & p) = 0;
    virtual void visit(until_parser & p) = 0;
    virtual void visit(and_parser & p) = 0;
    virtual void visit(not_parser & p) = 0;
    virtual void visit(any_parser & p) = 0;
    virtual void visit(space_parser & p) = 0;
    virtual void visit(chars_parser & p) = 0;
    virtual void visit(json_string_parser & p) = 0;
    virtual void visit(schema_parser & p) = 0;
    virtual void visit(rule_parser & p) = 0;
    virtual void visit(ref_parser & p) = 0;
    virtual void visit(root_parser & p) = 0;
    virtual void visit(capture_parser & p) = 0;
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
        case '\\': return "\\\\";
        case ']':  return "\\]";
        case '-':  return "\\-";
        case '^':  return "\\^";
        default:   return std::string(1, c);
    }
}

// Create a GBNF excluding pattern
static std::string gbnf_excluding_pattern(const std::vector<std::string> & strings) {
    // Use the aho_corasick_matcher to grab an exhaustive list of prefixes and
    // potential next characters. We can use this to build an exclusion for
    // multiple strings.
    aho_corasick_matcher matcher(strings);
    auto pieces = matcher.collect_prefix_and_next();

    std::string pattern;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (i > 0) {
            pattern += " | ";
        }

        const auto & pre = pieces[i].prefix;
        const auto & chars = pieces[i].next_chars;

        std::string cls;
        cls.reserve(chars.size());
        for (const auto & ch : chars) {
            cls += gbnf_escape_char_class(ch);
        }

        if (!pre.empty()) {
            pattern += gbnf_literal(pre) + " [^" + cls + "]";
        } else {
            pattern += "[^" + cls + "]";
        }
    }

    return "(" + pattern + ")*";
}

// Visitor for resolving rule references and collecting rules
class resolution_visitor : public parser_visitor {
    std::unordered_map<std::string, std::shared_ptr<rule_parser>> rules_;
    std::vector<std::shared_ptr<ref_parser>> refs_;

  public:
    resolution_visitor() = default;

    void visit(start_parser & /* p */) override {}
    void visit(end_parser & /* p */) override {}
    void visit(literal_parser & /* p */) override {}
    void visit(any_parser & /* p */) override {}
    void visit(space_parser & /* p */) override {}
    void visit(json_string_parser & /* p */) override {}
    void visit(chars_parser & /* p */) override {}
    void visit(until_parser & /* p */) override {}
    void visit(and_parser & p) override { p.child()->accept(*this); }
    void visit(not_parser & p) override { p.child()->accept(*this); }

    void visit(sequence_parser & p) override {
        for (const auto & child : p.parsers()) {
            child->accept(*this);
        }
    }

    void visit(choice_parser & p) override {
        for (const auto & child : p.parsers()) {
            child->accept(*this);
        }
    }

    void visit(one_or_more_parser & p) override { p.child()->accept(*this); }
    void visit(zero_or_more_parser & p) override { p.child()->accept(*this); }
    void visit(optional_parser & p) override { p.child()->accept(*this); }
    void visit(repetition_parser & p) override { p.child()->accept(*this); }
    void visit(schema_parser & p) override { p.child()->accept(*this); }
    void visit(capture_parser & p) override { p.child()->accept(*this); }

    void visit(rule_parser & p) override {
        const std::string & name = p.name();

        // Check for duplicate rule names
        if (rules_.find(name) != rules_.end()) {
            throw std::runtime_error("Duplicate rule name: " + name);
        }

        // Collect this rule
        auto rule_ptr = cast<rule_parser>(p.shared_from_this());
        if (rule_ptr) {
            rules_[name] = rule_ptr;
        }

        // Recursively visit the child
        p.child()->accept(*this);
    }

    void visit(ref_parser & p) override {
        // Collect this ref for later resolution
        auto ref_ptr = cast<ref_parser>(p.shared_from_this());
        if (ref_ptr) {
            refs_.push_back(ref_ptr);
        }
    }

    void visit(root_parser & p) override {
        // Visit all rules stored in the map
        for (const auto & [name, rule] : p.rules()) {
            rule->accept(*this);
        }

        // Visit the root tree
        p.root()->accept(*this);
    }

    // Resolve all collected refs
    void resolve() {
        for (const auto & ref : refs_) {
            const std::string & name = ref->name();
            auto it = rules_.find(name);
            if (it == rules_.end()) {
                throw std::runtime_error("Unresolved reference: " + name);
            }
            ref->set_target(it->second);
        }
    }

    const std::unordered_map<std::string, std::shared_ptr<rule_parser>> & rules() const { return rules_; }
};

// Visitor for collecting reachable rules from a subtree
class reachability_visitor : public parser_visitor {
    std::unordered_set<std::string> & reachable_rules_;

  public:
    reachability_visitor(std::unordered_set<std::string> & reachable_rules)
        : reachable_rules_(reachable_rules) {}

    void visit(start_parser & /* p */) override {}
    void visit(end_parser & /* p */) override {}
    void visit(literal_parser & /* p */) override {}
    void visit(any_parser & /* p */) override {}
    void visit(space_parser & /* p */) override {}
    void visit(json_string_parser & /* p */) override {}
    void visit(chars_parser & /* p */) override {}
    void visit(until_parser & /* p */) override {}
    void visit(and_parser & p) override { p.child()->accept(*this); }
    void visit(not_parser & p) override { p.child()->accept(*this); }

    void visit(sequence_parser & p) override {
        for (const auto & child : p.parsers()) {
            child->accept(*this);
        }
    }

    void visit(choice_parser & p) override {
        for (const auto & child : p.parsers()) {
            child->accept(*this);
        }
    }

    void visit(one_or_more_parser & p) override { p.child()->accept(*this); }
    void visit(zero_or_more_parser & p) override { p.child()->accept(*this); }
    void visit(optional_parser & p) override { p.child()->accept(*this); }
    void visit(repetition_parser & p) override { p.child()->accept(*this); }
    void visit(schema_parser & /* p */) override {
        // The schema system will handle rule generation via builder_.add_schema()
    }
    void visit(capture_parser & p) override { p.child()->accept(*this); }

    void visit(rule_parser & p) override {
        const std::string & name = p.name();
        // Check if we arleady reached this rule. Technically we shouldn't,
        // since we don't traverse ref_parser
        if (reachable_rules_.find(name) != reachable_rules_.end()) {
            return;
        }
        reachable_rules_.insert(name);

        // Recursively visit the rule's child
        p.child()->accept(*this);
    }

    void visit(ref_parser & p) override {
        // Mark as reachable
        reachable_rules_.insert(p.name());
    }

    void visit(root_parser & p) override {
        p.root()->accept(*this);
    }
};

class gbnf_visitor : public parser_visitor {
    const common_grammar_builder & builder_;
    std::unordered_map<std::string, std::string> rule_name_mapping_;
    std::string current_result_;
    bool lazy_;
    std::unordered_map<std::string, std::shared_ptr<rule_parser>> all_rules_;
    std::unordered_set<std::string> reachable_rules_;

  public:
    gbnf_visitor(const common_grammar_builder & builder, bool lazy = false)
        : builder_(builder), lazy_(lazy) {}

    const std::string& result() const { return current_result_; }

  private:
    // Check if expression needs parentheses
    static bool needs_parens(parser_type type) {
        return type == CHOICE || type == SEQUENCE;
    }

    // Collect all reachable rules from trigger rules
    void collect_reachable_rules() {
        reachable_rules_.clear();
        reachability_visitor visitor(reachable_rules_);

        // Find all trigger rules and traverse from them
        for (const auto & [name, rule] : all_rules_) {
            if (rule->is_trigger()) {
                rule->accept(visitor);
            }
        }
    }

    // Collect all rules from the tree
    void collect_all_rules(common_chat_peg_parser_base & root) {
        resolution_visitor resolver;
        root.accept(resolver);
        all_rules_ = resolver.rules();
    }

  public:
    void visit(start_parser & /* p */) override {
        current_result_ = "";
    }

    void visit(end_parser & /* p */) override {
        current_result_ = "";
    }

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
            if (child->type() == CHOICE) {
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

    void visit(and_parser & /* p */) override {
        current_result_ = "";
    }

    void visit(not_parser & /* p */) override {
        // NOT is tricky in GBNF - for now, emit error
        LOG_ERR("NOT operator not directly supported in GBNF generation\n");
        current_result_ = "";
    }

    void visit(any_parser & /* p */) override {
        // Match any single character
        current_result_ = ".";
    }

    void visit(space_parser & /* p */) override {
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

    void visit(json_string_parser & /* p */) override {
        // JSON string content (without quotes)
        // Pattern: (any non-quote/backslash OR escape sequences)* until closing quote
        current_result_ = R"(( [^"\\] | "\\" ( ["\\/ bfnrt] | "u" [0-9a-fA-F]{4} ) )*)";
    }

    void visit(schema_parser & p) override {
        current_result_ = builder_.add_schema(p.name(), p.schema());
    }

    void visit(rule_parser & p) override {
        // When visiting a rule, generate its definition
        p.child()->accept(*this);
    }

    void visit(ref_parser & p) override {
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
        // Collect all rules from the tree
        collect_all_rules(p);

        if (!lazy_) {
            // Non-lazy mode: generate all rules eagerly
            for (const auto & [name, rule] : all_rules_) {
                rule->accept(*this);
                auto rule_body = current_result_;
                auto canonical_name = builder_.add_rule(name, rule_body);
                rule_name_mapping_[name] = canonical_name;
            }

            // Return root body for composition
            p.root()->accept(*this);
            return;
        }

        // Lazy mode: only generate rules reachable from triggers

        // Collect all rules reachable from triggers
        collect_reachable_rules();

        // Check if we found any trigger rules
        if (reachable_rules_.empty()) {
            LOG_ERR("Lazy grammar generation enabled but no trigger rules found\n");
            current_result_ = "";
            return;
        }

        // Generate only reachable rules
        for (const auto & [name, rule] : all_rules_) {
            // Skip rules that aren't reachable
            if (reachable_rules_.find(name) == reachable_rules_.end()) {
                continue;
            }

            rule->accept(*this);
            auto rule_body = current_result_;
            auto canonical_name = builder_.add_rule(name, rule_body);
            rule_name_mapping_[name] = canonical_name;
        }

        // Generate root as alternation of trigger rule names
        std::vector<std::string> trigger_names;
        for (const auto & [name, rule] : all_rules_) {
            if (rule->is_trigger()) {
                auto it = rule_name_mapping_.find(name);
                if (it != rule_name_mapping_.end()) {
                    trigger_names.push_back(it->second);
                }
            }
        }
        current_result_ = string_join(trigger_names, " | ");
    }

    void visit(capture_parser & p) override {
        p.child()->accept(*this);
    }
};

// Implement accept() methods for all parser classes
void start_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void end_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void literal_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void sequence_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void choice_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void one_or_more_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void zero_or_more_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void optional_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void repetition_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void until_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void and_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void not_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void any_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void space_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void chars_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void json_string_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void schema_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void rule_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void ref_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void root_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }
void capture_parser::accept(parser_visitor & visitor) { visitor.visit(*this); }

common_chat_parse_result common_chat_parse_cache::set(int id, size_t start, common_chat_parse_result result) {
    if (id == -1) {
        // Don't cache parsers with ID -1 (from operators and global factory functions)
        return result;
    }
    results[common_chat_parse_cache_key{id, start}] = result;
    return result;
}

std::optional<common_chat_parse_result> common_chat_parse_cache::get(int id, size_t start) {
    if (id == -1) {
        // Don't cache parsers with ID -1 (from operators and global factory functions)
        return std::nullopt;
    }
    auto it = results.find(common_chat_parse_cache_key{id, start});
    if (it != results.end()) {
        return it->second;
    }
    return std::nullopt;
}

void common_chat_parse_cache::clear() {
    results.clear();
}

common_chat_peg_parser::common_chat_peg_parser() {}
common_chat_peg_parser::common_chat_peg_parser(std::shared_ptr<common_chat_peg_parser_base> parser) : ptr_(std::move(parser)) {}
common_chat_peg_parser::common_chat_peg_parser(const std::string & literal) : ptr_(make_parser<literal_parser>(-1, literal)) {}
common_chat_peg_parser::common_chat_peg_parser(const char * literal) : ptr_(make_parser<literal_parser>(-1, literal)) {}

common_chat_peg_parser operator~(const common_chat_peg_parser & p) { return make_parser<not_parser>(-1, p); }

common_chat_peg_parser operator+(const common_chat_peg_parser & lhs, const common_chat_peg_parser & rhs) {
    return make_parser<sequence_parser>(-1, std::initializer_list<common_chat_peg_parser>{lhs, rhs});
}

common_chat_peg_parser operator|(const common_chat_peg_parser & lhs, const common_chat_peg_parser & rhs) {
    return make_parser<choice_parser>(-1, std::initializer_list<common_chat_peg_parser>{lhs, rhs});
}

common_chat_peg_parser operator<<(const common_chat_peg_parser & lhs, const common_chat_peg_parser & rhs) {
    auto ws = make_parser<space_parser>(-1);
    return make_parser<sequence_parser>(-1, std::initializer_list<common_chat_peg_parser>{lhs, ws, rhs});
}

common_chat_peg_parser operator+(const char * lhs, const common_chat_peg_parser & rhs) { return common_chat_peg_parser(lhs) + rhs; }
common_chat_peg_parser operator|(const char * lhs, const common_chat_peg_parser & rhs) { return common_chat_peg_parser(lhs) | rhs; }
common_chat_peg_parser operator<<(const char * lhs, const common_chat_peg_parser & rhs) { return common_chat_peg_parser(lhs) << rhs; }

common_chat_peg_parser operator+(const std::string & lhs, const common_chat_peg_parser & rhs) { return common_chat_peg_parser(lhs) + rhs; }
common_chat_peg_parser operator|(const std::string & lhs, const common_chat_peg_parser & rhs) { return common_chat_peg_parser(lhs) | rhs; }
common_chat_peg_parser operator<<(const std::string & lhs, const common_chat_peg_parser & rhs) { return common_chat_peg_parser(lhs) << rhs; }

common_chat_peg_parser_base & common_chat_peg_parser::operator*() const { return *ptr_; }
common_chat_peg_parser_base * common_chat_peg_parser::operator->() const { return ptr_.get(); }

common_chat_parse_result common_chat_peg_parser::parse(common_chat_parse_context & ctx, size_t start) const {
    return ptr_->parse(ctx, start);
}

std::string common_chat_peg_parser::dump() const { return ptr_->dump(); }

void common_chat_peg_parser::build_grammar(const common_grammar_builder & builder, bool lazy) const {
    gbnf_visitor visitor(builder, lazy);
    ptr_->accept(visitor);
    auto result = visitor.result();
    if (!result.empty()) {
        builder.add_rule("root", result);
    }
}

using builder = common_chat_peg_parser_builder;

builder::common_chat_peg_parser_builder() : root_(make_parser<root_parser>(0)) , counter_(1) {}

common_chat_peg_parser builder::start() { return make_parser<start_parser>(counter_); }
common_chat_peg_parser builder::end() { return make_parser<end_parser>(counter_); }
common_chat_peg_parser builder::literal(const std::string & literal) { return make_parser<literal_parser>(counter_, literal); }
common_chat_peg_parser builder::sequence(const std::vector<common_chat_peg_parser> & parsers) { return make_parser<sequence_parser>(counter_, parsers); }
common_chat_peg_parser builder::choice(const std::vector<common_chat_peg_parser> & parsers) { return make_parser<choice_parser>(counter_, parsers); }
common_chat_peg_parser builder::one_or_more(const common_chat_peg_parser & p) { return make_parser<one_or_more_parser>(counter_, p); }
common_chat_peg_parser builder::zero_or_more(const common_chat_peg_parser & p) { return make_parser<zero_or_more_parser>(counter_, p); }
common_chat_peg_parser builder::optional(const common_chat_peg_parser & p) { return make_parser<optional_parser>(counter_, p); }
common_chat_peg_parser builder::peek(const common_chat_peg_parser & p) { return make_parser<and_parser>(counter_, p); }
common_chat_peg_parser builder::negate(const common_chat_peg_parser & p) { return make_parser<not_parser>(counter_, p); }
common_chat_peg_parser builder::any() { return make_parser<any_parser>(counter_); }
common_chat_peg_parser builder::chars(const std::string & classes, int min, int max) { return make_parser<chars_parser>(counter_, classes, min, max); }
common_chat_peg_parser builder::one(const std::string & classes) { return make_parser<chars_parser>(counter_, classes, 1, 1); }
common_chat_peg_parser builder::json_string_content() { return make_parser<json_string_parser>(counter_); }
common_chat_peg_parser builder::space() { return make_parser<space_parser>(counter_); }
common_chat_peg_parser builder::until(const std::string & delimiter) { return make_parser<until_parser>(counter_, delimiter); }
common_chat_peg_parser builder::until_one_of(const std::vector<std::string> & delimiters) { return make_parser<until_parser>(counter_, delimiters); }
common_chat_peg_parser builder::repeat(const common_chat_peg_parser & p, int min, int max) { return make_parser<repetition_parser>(counter_, p, min, max); }
common_chat_peg_parser builder::repeat(const common_chat_peg_parser & p, int n) { return make_parser<repetition_parser>(counter_, p, n, n); }

common_chat_peg_parser builder::ref(const std::string & name) {
    return make_parser<ref_parser>(counter_, name);
}

common_chat_peg_parser builder::schema(const common_chat_peg_parser & p, const std::string & name, const nlohmann::ordered_json & schema) {
    return make_parser<schema_parser>(counter_, p, name, schema);
}

common_chat_peg_parser builder::capture(const std::string & key, const common_chat_peg_parser & p) {
    return make_parser<capture_parser>(counter_, p, key);
}

common_chat_peg_parser builder::rule(const std::string & name, const common_chat_peg_parser & p, bool trigger) {
    auto root_container = cast<root_parser>(root_);
    auto rule_node = make_parser<rule_parser>(counter_, name, p, trigger);
    root_container->add_rule(name, rule_node);
    return make_parser<ref_parser>(counter_, name);
}

common_chat_peg_parser builder::rule(const std::string & name, const std::function<common_chat_peg_parser()> & builder_fn, bool trigger) {
    auto root_container = cast<root_parser>(root_);
    if (root_container->rules().find(name) != root_container->rules().end()) {
        return ref(name);
    }

    // Create placeholder rule to allow recursive references
    auto placeholder = make_parser<rule_parser>(counter_, name, literal(""), trigger);
    root_container->add_rule(name, placeholder);

    // Build the actual parser
    auto parser = builder_fn();

    // Replace placeholder with actual rule
    auto rule_node = make_parser<rule_parser>(counter_, name, parser, trigger);
    root_container->add_rule(name, rule_node);

    return make_parser<ref_parser>(counter_, name);
}

void builder::set_root(const common_chat_peg_parser & p) {
    auto root_container = cast<root_parser>(root_);
    root_container->set_root(p);

    // Recursively issue IDs to reachable nodes
    if (p.ptr()) {
        p.ptr()->assign_id(counter_);
    }
}

common_chat_peg_parser builder::json_number() {
    return rule("json-number", [this]() {
        auto digit1_9 = chars("[1-9]", 1, 1);
        auto digits = chars("[0-9]");
        auto int_part = literal("0") | (digit1_9 + chars("[0-9]", 0, -1));
        auto frac = literal(".") + digits;
        auto exp = (literal("e") | literal("E")) + optional(chars("[+\\-]", 1, 1)) + digits;
        return optional(literal("-")) + int_part + optional(frac) + optional(exp);
    });
}

common_chat_peg_parser builder::json_string() {
    return rule("json-string", [this]() {
        return literal("\"") + json_string_content() + literal("\"");
    });
}

common_chat_peg_parser builder::json_bool() {
    return rule("json-bool", [this]() {
        return literal("true") | literal("false");
    });
}

common_chat_peg_parser builder::json_null() {
    return rule("json-null", [this]() {
        return literal("null");
    });
}

common_chat_peg_parser builder::json_object() {
    return rule("json-object", [this]() {
        auto ws = space();
        auto member = json_string() + ws + literal(":") + ws + json();
        auto members = member + zero_or_more(ws + literal(",") + ws + member);
        return (literal("{") + ws + literal("}")) |
               (literal("{") + ws + members + ws + literal("}"));
    });
}

common_chat_peg_parser builder::json_array() {
    return rule("json-array", [this]() {
        auto ws = space();
        auto elements = json() + zero_or_more(ws + literal(",") + ws + json());
        return (literal("[") + ws + literal("]")) |
               (literal("[") + ws + elements + ws + literal("]"));
    });
}

common_chat_peg_parser builder::json() {
    return rule("json-value", [this]() {
        return json_object() |
               json_array() |
               json_string() |
               json_number() |
               json_bool() |
               json_null();
    });
}

common_chat_peg_parser builder::build() {
    // Resolve all references
    resolution_visitor resolver;
    root_->accept(resolver);
    resolver.resolve();

    return root_;
}

common_chat_peg_parser build_peg_parser(const std::function<common_chat_peg_parser(builder&)> & fn) {
    builder builder;
    auto root = fn(builder);
    builder.set_root(root);
    return builder.build();
}
