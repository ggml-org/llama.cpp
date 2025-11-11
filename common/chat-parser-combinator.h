#pragma once

#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <unordered_map>
#include <optional>
#include <string>
#include <string_view>
#include <functional>

struct common_grammar_builder;

enum parser_type {
    PARSER_LITERAL = 0,
    PARSER_SEQUENCE = 1,
    PARSER_CHOICE = 2,
    PARSER_ZERO_OR_MORE = 3,
    PARSER_ONE_OR_MORE = 4,
    PARSER_NOT = 5,
    PARSER_ANY = 6,
    PARSER_CHAR_CLASS = 7,
    PARSER_GROUP = 8,
    PARSER_RULE = 9,
    PARSER_OPTIONAL = 10,
    PARSER_UNTIL = 11,
    PARSER_SPACE = 12,
    PARSER_SCHEMA = 13,
};

enum parser_result_type {
    PARSER_RESULT_FAIL = 0,
    PARSER_RESULT_NEED_MORE_INPUT = 1,
    PARSER_RESULT_SUCCESS = 2,
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

struct parser_match_location {
    size_t start;
    size_t end;

    size_t length() const { return end - start; }

    std::string_view view(std::string_view sv) const {
        return sv.substr(start, length());
    }
};

struct parser_result {
    parser_result_type type = PARSER_RESULT_FAIL;
    size_t start = 0;
    size_t end = 0;

    std::unordered_map<std::string, parser_match_location> groups;

    parser_result() : type(PARSER_RESULT_FAIL) {}
    parser_result(parser_result_type type, size_t start) : type(type), start(start), end(start) {}
    parser_result(parser_result_type type, size_t start, size_t end) : type(type), start(start), end(end) {}
    parser_result(parser_result_type type, size_t start, size_t end, const std::unordered_map<std::string, parser_match_location> & groups) : type(type), start(start), end(end), groups(groups) {}

    bool is_fail() const { return type == PARSER_RESULT_FAIL; }
    bool is_need_more_input() const { return type == PARSER_RESULT_NEED_MORE_INPUT; }
    bool is_success() const { return type == PARSER_RESULT_SUCCESS; }

    std::optional<std::string> group(const std::string & name, std::string_view input) const;
};

class parse_cache {
    std::unordered_map<parse_cache_key, parser_result> results;

  public:
    parser_result set(int id, size_t start, parser_result result);
    std::optional<parser_result> get(int id, size_t start);
    void clear();
};

class parser;

struct parser_context {
    std::string_view input;
    parse_cache memo;
    bool input_is_complete = true;
};

class parser_base;
class sequence_parser;
class choice_parser;
class parser_builder;
class parser_visitor;

class parser {
    std::shared_ptr<parser_base> ptr;

    friend class parser_builder;

  public:
    parser();
    parser(std::shared_ptr<parser_base> parser);
    parser(const parser & other) = default;
    parser & operator=(const parser & other) {
        if (this != &other) {
            ptr = other.ptr;
        }
        return *this;
    }

    parser operator~() const;
    parser operator+(const parser & other) const;
    parser operator|(const parser & other) const;
    parser operator<<(const parser & other) const;

    parser_base & operator*() const;
    parser_base * operator->() const;

    bool is_sequence() const;
    std::shared_ptr<sequence_parser> to_sequence() const;

    bool is_choice() const;
    std::shared_ptr<choice_parser> to_choice() const;

    parser_type type() const;
    parser_result parse(parser_context & ctx, size_t start = 0) const;
    std::string dump() const;

    void build_grammar(const common_grammar_builder & builder);
};

class parser_builder {
    std::shared_ptr<std::unordered_map<std::string, parser>> rules_;
    int next_id_;

  public:
    parser_builder();

    parser literal(const std::string & literal);
    parser sequence(std::initializer_list<parser> parsers);
    parser choice(std::initializer_list<parser> parsers);
    parser one_or_more(const parser & p);
    parser zero_or_more(const parser & p);
    parser optional(const parser & p);
    parser negate(const parser & p);
    parser any();
    parser char_class(const std::string & classes);
    parser group(const std::string & name, const parser & p);
    parser rule(const std::string & name);
    parser space();
    parser until(const std::string & delimiter, bool include_spaces = true);
    parser json();
    parser schema(const parser & p, const std::string & name, const nlohmann::ordered_json & schema);

    parser add_rule(const std::string & name, const parser & p);

    void assign_ids(parser & p);

    std::shared_ptr<std::unordered_map<std::string, parser>> rules() const { return rules_; }
};

parser build_parser(const std::function<parser(parser_builder&)> & fn);
