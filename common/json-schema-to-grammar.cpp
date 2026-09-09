#include "json-schema-to-grammar.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include <algorithm>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = common_json;

static std::string build_repetition(const std::string & item_rule, int min_items, int max_items, const std::string & separator_rule = "") {
    auto has_max = max_items != std::numeric_limits<int>::max();

    if (max_items == 0) {
        return "";
    }
    if (min_items == 0 && max_items == 1) {
        return item_rule + "?";
    }

    if (separator_rule.empty()) {
        if (min_items == 1 && !has_max) {
            return item_rule + "+";
        }
        if (min_items == 0 && !has_max) {
            return item_rule + "*";
        }
        return item_rule + "{" + std::to_string(min_items) + "," + (has_max ? std::to_string(max_items) : "") + "}";
    }

    auto result = item_rule + " " + build_repetition("(" + separator_rule + " " + item_rule + ")", min_items == 0 ? 0 : min_items - 1, has_max ? max_items - 1 : max_items);
    if (min_items == 0) {
        result = "(" + result + ")?";
    }
    return result;
}

static void build_min_max_int(int64_t min_value, int64_t max_value, std::stringstream & out, int decimals_left = 16, bool top_level = true) {
    auto has_min = min_value != std::numeric_limits<int64_t>::min();
    auto has_max = max_value != std::numeric_limits<int64_t>::max();

    auto digit_range = [&](char from, char to) {
        out << "[";
        if (from == to) {
            out << from;
        } else {
            out << from << "-" << to;
        }
        out << "]";
    };
    auto more_digits = [&](int min_digits, int max_digits) {
        out << "[0-9]";
        if (min_digits == max_digits && min_digits == 1) {
            return;
        }
        out << "{";
        out << min_digits;
        if (max_digits != min_digits) {
            out << ",";
            if (max_digits != std::numeric_limits<int>::max()) {
                out << max_digits;
            }
        }
        out << "}";
    };
    std::function<void(const std::string_view &, const std::string_view &)> uniform_range =
        [&](const std::string_view & from, const std::string_view & to) {
            size_t i = 0;
            while (i < from.length() && i < to.length() && from[i] == to[i]) {
                i++;
            }
            if (i > 0) {
                out << "\"" << from.substr(0, i) << "\"";
            }
            if (i < from.length() && i < to.length()) {
                if (i > 0) {
                    out << " ";
                }
                auto sub_len = from.length() - i - 1;
                if (sub_len > 0) {
                    auto from_sub = from.substr(i + 1);
                    auto to_sub = to.substr(i + 1);
                    auto sub_zeros = string_repeat("0", sub_len);
                    auto sub_nines = string_repeat("9", sub_len);

                    auto to_reached = false;
                    out << "(";
                    if (from_sub == sub_zeros) {
                        digit_range(from[i], to[i] - 1);
                        out << " ";
                        more_digits(sub_len, sub_len);
                    } else {
                        out << "[" << from[i] << "] ";
                        out << "(";
                        uniform_range(from_sub, sub_nines);
                        out << ")";
                        if (from[i] < to[i] - 1) {
                            out << " | ";
                            if (to_sub == sub_nines) {
                                digit_range(from[i] + 1, to[i]);
                                to_reached = true;
                            } else {
                                digit_range(from[i] + 1, to[i] - 1);
                            }
                            out << " ";
                            more_digits(sub_len, sub_len);
                        }
                    }
                    if (!to_reached) {
                        out << " | ";
                        digit_range(to[i], to[i]);
                        out << " ";
                        uniform_range(sub_zeros, to_sub);
                    }
                    out << ")";
                } else {
                    out << "[" << from[i] << "-" << to[i] << "]";
                }
            }
        };

    if (has_min && has_max) {
        if (min_value < 0 && max_value < 0) {
            out << "\"-\" (";
            build_min_max_int(-max_value, -min_value, out, decimals_left, /* top_level= */ true);
            out << ")";
            return;
        }

        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(0, -min_value, out, decimals_left, /* top_level= */ true);
            out << ") | ";
            min_value = 0;
        }

        auto min_s = std::to_string(min_value);
        auto max_s = std::to_string(max_value);
        auto min_digits = min_s.length();
        auto max_digits = max_s.length();

        for (auto digits = min_digits; digits < max_digits; digits++) {
            uniform_range(min_s, string_repeat("9", digits));
            min_s = "1" + string_repeat("0", digits);
            out << " | ";
        }
        uniform_range(min_s, max_s);
        return;
    }

    auto less_decimals = std::max(decimals_left - 1, 1);

    if (has_min) {
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(std::numeric_limits<int64_t>::min(), -min_value, out, decimals_left, /* top_level= */ false);
            out << ") | [0] | [1-9] ";
            more_digits(0, decimals_left - 1);
        } else if (min_value == 0) {
            if (top_level) {
                out << "[0] | [1-9] ";
                more_digits(0, less_decimals);
            } else {
                more_digits(1, decimals_left);
            }
        } else if (min_value <= 9) {
            char c = '0' + min_value;
            auto range_start = top_level ? '1' : '0';
            if (c > range_start) {
                digit_range(range_start, c - 1);
                out << " ";
                more_digits(1, less_decimals);
                out << " | ";
            }
            digit_range(c, '9');
            out << " ";
            more_digits(0, less_decimals);
        } else {
            auto min_s = std::to_string(min_value);
            auto len = min_s.length();
            auto c = min_s[0];

            if (c > '1') {
                digit_range(top_level ? '1' : '0', c - 1);
                out << " ";
                more_digits(len, less_decimals);
                out << " | ";
            }
            digit_range(c, c);
            out << " (";
            build_min_max_int(std::stoll(min_s.substr(1)), std::numeric_limits<int64_t>::max(), out, less_decimals, /* top_level= */ false);
            out << ")";
            if (c < '9') {
                out << " | ";
                digit_range(c + 1, '9');
                out << " ";
                more_digits(len - 1, less_decimals);
            }
        }
        return;
    }

    if (has_max) {
        if (max_value >= 0) {
            if (top_level) {
                out << "\"-\" [1-9] ";
                more_digits(0, less_decimals);
                out << " | ";
            }
            build_min_max_int(0, max_value, out, decimals_left, /* top_level= */ true);
        } else {
            out << "\"-\" (";
            build_min_max_int(-max_value, std::numeric_limits<int64_t>::max(), out, decimals_left, /* top_level= */ false);
            out << ")";
        }
        return;
    }

    throw std::runtime_error("At least one of min_value or max_value must be set");
}

const std::string SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
    std::string content;
    std::vector<std::string> deps;
};

static std::unordered_map<std::string, BuiltinRule> PRIMITIVE_RULES = {
    {"boolean", {"(\"true\" | \"false\")", {}}},
    {"decimal-part", {"[0-9]{1,16}", {}}},
    {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
    {"number", {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", {"integral-part", "decimal-part"}}},
    {"integer", {"(\"-\"? integral-part)", {"integral-part"}}},
    {"value", {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
    {"object", {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"", {"string", "value"}}},
    {"array", {"\"[\" space ( value (\",\" space value)* )? space \"]\"", {"value"}}},
    {"uuid", {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"", {}}},
    {"char",   {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {}}},
    {"string", {"\"\\\"\" char* \"\\\"\"", {"char"}}},
    {"null", {"\"null\"", {}}},
};

static std::unordered_map<std::string, BuiltinRule> STRING_FORMAT_RULES = {
    {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
    {"time", {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", {}}},
    {"date-time", {"date \"T\" time", {"date", "time"}}},
    {"date-string", {"\"\\\"\" date \"\\\"\"", {"date"}}},
    {"time-string", {"\"\\\"\" time \"\\\"\"", {"time"}}},
    {"date-time-string", {"\"\\\"\" date-time \"\\\"\"", {"date-time"}}}
};

static bool is_reserved_name(const std::string & name) {
    static const std::unordered_set<std::string> RESERVED_NAMES = [] {
        std::unordered_set<std::string> s;
        s.insert("root");
        for (const auto & p : PRIMITIVE_RULES) {
            s.insert(p.first);
        }
        for (const auto & p : STRING_FORMAT_RULES) {
            s.insert(p.first);
        }
        return s;
    }();
    return RESERVED_NAMES.find(name) != RESERVED_NAMES.end();
}

static std::regex INVALID_RULE_CHARS_RE("[^a-zA-Z0-9-]+");
static std::regex GRAMMAR_LITERAL_ESCAPE_RE("[\r\n\"\\\\]");
static std::regex GRAMMAR_RANGE_LITERAL_ESCAPE_RE("[\r\n\"\\]\\-\\\\]");
static std::unordered_map<char, std::string> GRAMMAR_LITERAL_ESCAPES = {
    {'\r', "\\r"}, {'\n', "\\n"}, {'"', "\\\""}, {'-', "\\-"}, {']', "\\]"}, {'\\', "\\\\"}
};

static const int MAX_PATTERN_DEPTH = 100;

static std::unordered_set<char> NON_LITERAL_SET = {'|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?', '^', '$'};
static std::unordered_set<char> ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = {'^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?'};

static std::string replacePattern(const std::string & input, const std::regex & regex, const std::function<std::string(const std::smatch  &)> & replacement) {
    std::smatch match;
    std::string result;

    std::string::const_iterator searchStart(input.cbegin());
    std::string::const_iterator searchEnd(input.cend());

    while (std::regex_search(searchStart, searchEnd, match, regex)) {
        result.append(searchStart, searchStart + match.position());
        result.append(replacement(match));
        searchStart = match.suffix().first;
    }

    result.append(searchStart, searchEnd);

    return result;
}

static std::string format_literal(const std::string & literal) {
    std::string escaped = replacePattern(literal, GRAMMAR_LITERAL_ESCAPE_RE, [&](const std::smatch & match) {
        char c = match.str()[0];
        return GRAMMAR_LITERAL_ESCAPES.at(c);
    });
    return "\"" + escaped + "\"";
}

std::string gbnf_format_literal(const std::string & literal) { return format_literal(literal); }

static size_t gbnf_escape_length(const std::string & pattern, size_t pos) {
    if (pos + 1 >= pattern.length() || pattern[pos] != '\\') {
        return 0;
    }
    size_t n_hex = 0;
    switch (pattern[pos + 1]) {
        case 'x': n_hex = 2; break;
        case 'u': n_hex = 4; break;
        case 'U': n_hex = 8; break;
        case 't': case 'r': case 'n': case '\\': case '"': case '[': case ']':
            return 2;
        default:
            return 0;
    }
    if (pos + 2 + n_hex > pattern.length()) {
        return 0;
    }
    for (size_t i = pos + 2; i < pos + 2 + n_hex; i++) {
        char h = pattern[i];
        if (!((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'))) {
            return 0;
        }
    }
    return 2 + n_hex;
}

class common_schema_converter {
private:
    friend std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options);
    bool _dotall;
    std::map<std::string, std::string> _rules;
    std::unordered_set<std::string> _refs_being_resolved;
    std::vector<std::string> _errors;
    std::vector<std::string> _warnings;

    common_schema_document _doc;

    template <typename T>
    static const T & as(const common_schema & node) {
        return static_cast<const T &>(node);
    }

    std::string _add_rule(const std::string & name, const std::string & rule) {
        std::string esc_name = regex_replace(name, INVALID_RULE_CHARS_RE, "-");
        if (_rules.find(esc_name) == _rules.end() || _rules[esc_name] == rule) {
            _rules[esc_name] = rule;
            return esc_name;
        }
        int i = 0;
        while (_rules.find(esc_name + std::to_string(i)) != _rules.end() && _rules[esc_name + std::to_string(i)] != rule) {
            i++;
        }
        std::string key = esc_name + std::to_string(i);
        _rules[key] = rule;
        return key;
    }

    std::string _generate_union_rule(const std::string & name, const std::vector<common_schema_ptr> & alt_schemas) {
        std::vector<std::string> rules;
        rules.reserve(alt_schemas.size());
        for (size_t i = 0; i < alt_schemas.size(); i++) {
            rules.push_back(visit(*alt_schemas[i], name + (name.empty() ? "alternative-" : "-") + std::to_string(i)));
        }
        return string_join(rules, " | ");
    }

    // thrown when the pattern is a valid regex with no grammar equivalent
    struct unsupported_pattern : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // thrown when the pattern is not a valid regex
    struct invalid_pattern : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    std::string _visit_pattern(const std::string & pattern, const std::string & name) {
        auto rules_snapshot = _rules;
        try {
            return _pattern_to_rule(pattern, name);
        } catch (const unsupported_pattern & err) {
            // revert rules
            _rules = std::move(rules_snapshot);
            _warnings.push_back("pattern " + pattern + " is not supported (" + err.what() + "), accepting any string");
            return _add_rule(name, _add_primitive("string", PRIMITIVE_RULES.at("string")));
        } catch (const invalid_pattern & err) {
            _rules = std::move(rules_snapshot);
            _errors.push_back("Invalid pattern " + pattern + ": " + err.what());
            return "";
        }
    }

    std::string _pattern_to_rule(const std::string & pattern, const std::string & name) {
        if (pattern.length() < 2 || pattern.front() != '^' || pattern.back() != '$') {
            throw unsupported_pattern("not anchored with '^' and '$'");
        }
        std::string sub_pattern = pattern.substr(1, pattern.length() - 2);
        std::unordered_map<std::string, std::string> sub_rule_ids;

        size_t i = 0;
        size_t length = sub_pattern.length();
        int paren_depth = 0;

        using literal_or_rule = std::pair<std::string, bool>;
        auto to_rule = [&](const literal_or_rule & ls) {
            auto is_literal = ls.second;
            auto s = ls.first;
            return is_literal ? "\"" + s + "\"" : s;
        };
        std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
            std::vector<literal_or_rule> seq;

            auto get_dot = [&]() {
                std::string rule;
                if (_dotall) {
                    rule = "[\\U00000000-\\U0010FFFF]";
                } else {
                    rule = "[^\\x0A\\x0D]";
                }
                return _add_rule("dot", rule);
            };

            // Joins the sequence, merging consecutive literals together.
            auto join_seq = [&]() {
                std::vector<literal_or_rule> ret;

                std::string literal;
                auto flush_literal = [&]() {
                    if (literal.empty()) {
                        return false;
                    }
                    ret.emplace_back(literal, true);
                    literal.clear();
                    return true;
                };

                for (const auto & item : seq) {
                    auto is_literal = item.second;
                    if (is_literal) {
                        literal += item.first;
                    } else {
                        flush_literal();
                        ret.push_back(item);
                    }
                }
                flush_literal();

                std::vector<std::string> results;
                results.reserve(ret.size());
                for (const auto & item : ret) {
                    results.push_back(to_rule(item));
                }
                return std::make_pair(string_join(results, " "), false);
            };

            while (i < length) {
                char c = sub_pattern[i];
                if (c == '.') {
                    seq.emplace_back(get_dot(), false);
                    i++;
                } else if (c == '(') {
                    i++;
                    if (i < length && sub_pattern[i] == '?') {
                        if (i + 1 < length && sub_pattern[i + 1] == ':') {
                            i += 2; // skip "?:" for non-capturing group, treat as regular group
                        } else {
                            // lookaround, named group, inline flags, ...
                            throw unsupported_pattern("unsupported group syntax");
                        }
                    }
                    paren_depth++;
                    if (paren_depth > MAX_PATTERN_DEPTH) {
                        throw unsupported_pattern("pattern nesting too deep");
                    }
                    seq.emplace_back("(" + to_rule(transform()) + ")", false);
                } else if (c == ')') {
                    i++;
                    if (paren_depth == 0) {
                        throw invalid_pattern("unbalanced parentheses");
                    }
                    paren_depth--;
                    return join_seq();
                } else if (c == '^' || c == '$') {
                    throw unsupported_pattern("anchor inside the pattern");
                } else if (c == '[') {
                    std::string square_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != ']') {
                        if (sub_pattern[i] == '\\') {
                            auto escape_length = gbnf_escape_length(sub_pattern, i);
                            if (escape_length == 0) {
                                throw unsupported_pattern("unsupported escape in character class: " + sub_pattern.substr(i, 2));
                            }
                            square_brackets += sub_pattern.substr(i, escape_length);
                            i += escape_length;
                        } else {
                            square_brackets += sub_pattern[i];
                            i++;
                        }
                    }
                    if (i >= length) {
                        throw invalid_pattern("unterminated character class");
                    }
                    square_brackets += ']';
                    i++;
                    seq.emplace_back(square_brackets, false);
                } else if (c == '|') {
                    seq.emplace_back("|", false);
                    i++;
                } else if (c == '*' || c == '+' || c == '?') {
                    if (seq.empty()) {
                        throw invalid_pattern("nothing to repeat");
                    }
                    seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
                    i++;
                } else if (c == '{') {
                    std::string curly_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != '}') {
                        curly_brackets += sub_pattern[i];
                        i++;
                    }
                    if (i >= length) {
                        throw unsupported_pattern("unterminated curly brackets");
                    }
                    curly_brackets += '}';
                    i++;
                    auto nums = string_split(curly_brackets.substr(1, curly_brackets.length() - 2), ",");
                    int min_times = 0;
                    int max_times = std::numeric_limits<int>::max();
                    if (nums.size() != 1 && nums.size() != 2) {
                        throw unsupported_pattern("wrong number of values in curly brackets");
                    }
                    try {
                        if (nums.size() == 1) {
                            min_times = max_times = std::stoi(nums[0]);
                        } else {
                            if (!nums[0].empty()) {
                                min_times = std::stoi(nums[0]);
                            }
                            if (!nums[1].empty()) {
                                max_times = std::stoi(nums[1]);
                            }
                        }
                    } catch (const std::logic_error &) {
                        throw unsupported_pattern("invalid number in curly brackets");
                    }
                    if (seq.empty()) {
                        throw invalid_pattern("nothing to repeat");
                    }
                    auto &last = seq.back();
                    auto &sub = last.first;
                    auto sub_is_literal = last.second;

                    if (!sub_is_literal) {
                        std::string & sub_id = sub_rule_ids[sub];
                        if (sub_id.empty()) {
                            sub_id = _add_rule(name + "-" + std::to_string(sub_rule_ids.size()), sub);
                        }
                        sub = sub_id;
                    }
                    seq.back().first = build_repetition(
                        sub_is_literal ? "\"" + sub + "\"" : sub,
                        min_times,
                        max_times,
                        ""
                    );
                    seq.back().second = false;
                } else {
                    std::string literal;
                    auto is_non_literal = [&](char c) {
                        return NON_LITERAL_SET.find(c) != NON_LITERAL_SET.end();
                    };
                    while (i < length) {
                        if (sub_pattern[i] == '\\') {
                            if (i == length - 1) {
                                throw invalid_pattern("trailing backslash");
                            }
                            char next = sub_pattern[i + 1];
                            if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.find(next) != ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.end()) {
                                i++;
                                literal += sub_pattern[i];
                                i++;
                            } else {
                                auto escape_length = gbnf_escape_length(sub_pattern, i);
                                if (escape_length == 0) {
                                    throw unsupported_pattern("unsupported escape: " + sub_pattern.substr(i, 2));
                                }
                                literal += sub_pattern.substr(i, escape_length);
                                i += escape_length;
                            }
                        } else if (sub_pattern[i] == '"') {
                            literal += "\\\"";
                            i++;
                        } else if (!is_non_literal(sub_pattern[i]) &&
                                (i == length - 1 || literal.empty() || sub_pattern[i + 1] == '.' || !is_non_literal(sub_pattern[i + 1]))) {
                            literal += sub_pattern[i];
                            i++;
                        } else {
                            break;
                        }
                    }
                    if (literal.empty()) { // nothing was consumed, ex. a stray ']' or '}'
                        throw unsupported_pattern(std::string("unsupported character: ") + c);
                    }
                    seq.emplace_back(literal, true);
                }
            }
            return join_seq();
        };

        auto rule = to_rule(transform());
        if (paren_depth != 0) {
            throw invalid_pattern("unbalanced parentheses");
        }

        return _add_rule(name, "\"\\\"\" (" + rule + ") \"\\\"\"");
    }

    /*
        Returns a rule that matches a JSON string that is none of the provided strings

        not_strings({"a"})
            -> ["] ( [a] char+ | [^"a] char* )? ["]
        not_strings({"and", "also"})
            -> ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["]
    */
    std::string _not_strings(const std::vector<std::string> & strings) {
        common_trie trie(strings);

        std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
        std::ostringstream out;
        out << "[\"] ( ";
        std::function<void(size_t)> visit = [&](size_t idx) {
            const auto & node = trie.nodes[idx];
            std::string rejects;
            auto first = true;
            for (const auto & [cpt, child] : node.children) {
                std::string c = common_unicode_cpt_to_utf8(cpt);
                rejects += c;
                if (first) {
                    first = false;
                } else {
                    out << " | ";
                }
                out << "[" << c << "]";
                if (!trie.nodes[child].children.empty()) {
                    out << " (";
                    visit(child);
                    out << ")";
                } else {
                    out << " " << char_rule << "+";
                }
            }
            if (!node.children.empty()) {
                out << " | [^\"" << rejects << "] " << char_rule << "*";
            }
        };
        visit(0);

        out << " )";
        if (trie.nodes[0].pattern < 0) {
            out << "?";
        }
        out << " [\"]";
        return out.str();
    }

    std::string _resolve_ref(const common_schema_ref & schema) {
        auto it = schema.ref.find('#');
        std::string ref_fragment = it != std::string::npos ? schema.ref.substr(it + 1) : schema.ref;
        static const std::regex nonalphanumeric_regex(R"([^a-zA-Z0-9-]+)");
        std::string ref_name = "ref" + std::regex_replace(ref_fragment, nonalphanumeric_regex, "-");
        if (_rules.find(ref_name) == _rules.end() && _refs_being_resolved.find(schema.ref) == _refs_being_resolved.end()) {
            if (!schema.target) {
                _errors.push_back("Unresolved $ref " + schema.ref);
                return "";
            }
            _refs_being_resolved.insert(schema.ref);
            ref_name = visit(*schema.target, ref_name);
            _refs_being_resolved.erase(schema.ref);
        }
        return ref_name;
    }

    std::string _build_object_rule(
        const std::vector<std::pair<std::string, const common_schema *>> & properties,
        const std::unordered_set<std::string> & required,
        const std::string & name,
        const common_schema * additional_properties)
    {
        std::vector<std::string> required_props;
        std::vector<std::string> optional_props;
        std::unordered_map<std::string, std::string> prop_kv_rule_names;
        std::vector<std::string> prop_names;
        for (const auto & kv : properties) {
            const auto &prop_name = kv.first;
            const auto &prop_schema = kv.second;

            std::string prop_rule_name = visit(*prop_schema, name + (name.empty() ? "" : "-") + prop_name);
            prop_kv_rule_names[prop_name] = _add_rule(
                name + (name.empty() ? "" : "-") + prop_name + "-kv",
                format_literal(json(prop_name).dump()) + " space \":\" space " + prop_rule_name
            );
            if (required.find(prop_name) != required.end()) {
                required_props.push_back(prop_name);
            } else {
                optional_props.push_back(prop_name);
            }
            prop_names.push_back(prop_name);
        }
        if (additional_properties) {
            std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
            std::string value_rule =
                additional_properties->kind() != COMMON_SCHEMA_KIND_ANY ? visit(*additional_properties, sub_name + "-value")
                : _add_primitive("value", PRIMITIVE_RULES.at("value"));

            auto key_rule =
                prop_names.empty() ? _add_primitive("string", PRIMITIVE_RULES.at("string"))
                : _add_rule(sub_name + "-k", _not_strings(prop_names));
            std::string kv_rule = _add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
            prop_kv_rule_names["*"] = kv_rule;
            optional_props.push_back("*");
        }

        if (required_props.empty() && optional_props.empty()) {
            return "\"{\" space \"}\"";
        }

        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); i++) {
            if (i > 0) {
                rule += " \",\" space ";
            }
            rule += prop_kv_rule_names[required_props[i]];
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) {
                rule += " \",\" space ( ";
            }

            std::function<std::string(const std::vector<std::string> &, bool)> get_recursive_refs = [&](const std::vector<std::string> & ks, bool first_is_optional) {
                std::string res;
                if (ks.empty()) {
                    return res;
                }
                const std::string& k = ks[0];
                std::string kv_rule_name = prop_kv_rule_names[k];
                std::string comma_ref = "( \",\" space " + kv_rule_name + " )";
                if (first_is_optional) {
                    res = comma_ref + (k == "*" ? "*" : "?");
                } else {
                    res = kv_rule_name + (k == "*" ? " " + comma_ref + "*" : "");
                }
                if (ks.size() > 1) {
                    res += " " + _add_rule(
                        name + (name.empty() ? "" : "-") + k + "-rest",
                        get_recursive_refs(std::vector<std::string>(ks.begin() + 1, ks.end()), true)
                    );
                }
                return res;
            };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) {
                    rule += " | ";
                }
                rule += get_recursive_refs(std::vector<std::string>(optional_props.begin() + i, optional_props.end()), false);
            }
            if (!required_props.empty()) {
                rule += " )";
            }
            rule += " )?";
        }

        rule += " space \"}\"";

        return rule;
    }

    std::string _add_primitive(const std::string & name, const BuiltinRule & rule) {
        auto n = _add_rule(name, rule.content);
        for (const auto & dep : rule.deps) {
            BuiltinRule dep_rule;
            auto it = PRIMITIVE_RULES.find(dep);
            if (it == PRIMITIVE_RULES.end()) {
                it = STRING_FORMAT_RULES.find(dep);
                if (it == STRING_FORMAT_RULES.end()) {
                    _errors.push_back("Rule " + dep + " not known");
                    continue;
                }
            }
            if (_rules.find(dep) == _rules.end()) {
                _add_primitive(dep, it->second);
            }
        }
        return n;
    }

public:
    explicit common_schema_converter(bool dotall) : _dotall(dotall) {
        _rules["space"] = SPACE_RULE;
    }

    // Parses the document, so that the $refs of the schemas added after it resolve
    void resolve_refs(const common_json & schema) {
        try {
            common_schema_parse(schema, _doc);
        } catch (const std::runtime_error & e) {
            _errors.push_back(e.what());
        }
    }

    std::string add_schema(const std::string & name, const common_json & schema) {
        common_schema_ptr node;
        try {
            node = common_schema_parse(schema, _doc);
        } catch (const std::runtime_error & e) {
            _errors.push_back(e.what());
            return "";
        }
        return visit(*node, name);
    }

    static std::string _generate_constant_rule(const json & value) {
        return format_literal(value.dump());
    }

    // the primitive under its own name, or inlined when it is the root
    std::string _visit_primitive(const std::string & rule_name, const std::string & type) {
        return _add_primitive(rule_name == "root" ? "root" : type, PRIMITIVE_RULES.at(type));
    }

    // An allOf merges its components: the properties of a direct component are required, those of a nested anyOf are optional, enums are intersected.
    std::string _visit_all_of(const common_schema_all_of & schema, const std::string & name, const std::string & rule_name) {
        std::unordered_set<std::string> required;
        std::vector<std::pair<std::string, const common_schema *>> properties;
        std::map<std::string, size_t> enum_values;
        std::function<void(const common_schema &, bool)> add_component = [&](const common_schema & comp, bool is_required) {
            if (comp.kind() == COMMON_SCHEMA_KIND_REF) {
                if (const auto * target = as<common_schema_ref>(comp).target) {
                    add_component(*target, is_required);
                }
            } else if (comp.kind() == COMMON_SCHEMA_KIND_OBJECT) {
                for (const auto & prop : as<common_schema_object>(comp).properties) {
                    properties.emplace_back(prop.name, prop.schema.get());
                    if (is_required) {
                        required.insert(prop.name);
                    }
                }
            } else if (comp.kind() == COMMON_SCHEMA_KIND_ENUM) {
                for (const auto & v : as<common_schema_enum>(comp).values) {
                    enum_values[_generate_constant_rule(v)] += 1;
                }
            }
        };
        for (const auto & child : schema.children) {
            if (child->kind() == COMMON_SCHEMA_KIND_ANY_OF) {
                for (const auto & alt : as<common_schema_any_of>(*child).children) {
                    add_component(*alt, false);
                }
            } else {
                add_component(*child, true);
            }
        }
        if (!enum_values.empty()) {
            std::vector<std::string> enum_intersection;
            for (const auto & p : enum_values) {
                if (p.second == schema.children.size()) {
                    enum_intersection.push_back(p.first);
                }
            }
            if (!enum_intersection.empty()) {
                return _add_rule(rule_name, "(" + string_join(enum_intersection, " | ") + ")");
            }
        }
        return _add_rule(rule_name, _build_object_rule(properties, required, name, nullptr));
    }

    std::string visit(const common_schema & schema, const std::string & name) {
        std::string rule_name = is_reserved_name(name) ? name + "-" : name.empty() ? "root" : name;
        std::string sub_name  = name + (name.empty() ? "" : "-");

        switch (schema.kind()) {
            case COMMON_SCHEMA_KIND_REF:
                return _add_rule(rule_name, _resolve_ref(as<common_schema_ref>(schema)));
            case COMMON_SCHEMA_KIND_ANY_OF:
                return _add_rule(rule_name, _generate_union_rule(name, as<common_schema_any_of>(schema).children));
            case COMMON_SCHEMA_KIND_ALL_OF:
                return _visit_all_of(as<common_schema_all_of>(schema), name, rule_name);
            case COMMON_SCHEMA_KIND_CONST:
                return _add_rule(rule_name, _generate_constant_rule(as<common_schema_const>(schema).value));
            case COMMON_SCHEMA_KIND_ENUM: {
                std::vector<std::string> enum_values;
                for (const auto & v : as<common_schema_enum>(schema).values) {
                    enum_values.push_back(_generate_constant_rule(v));
                }
                return _add_rule(rule_name, "(" + string_join(enum_values, " | ") + ")");
            }
            case COMMON_SCHEMA_KIND_OBJECT: {
                const auto & obj = as<common_schema_object>(schema);
                if (obj.properties.empty() && obj.additional_properties && obj.additional_properties->kind() == COMMON_SCHEMA_KIND_ANY) {
                    return _add_rule(rule_name, _add_primitive("object", PRIMITIVE_RULES.at("object")));
                }
                std::vector<std::pair<std::string, const common_schema *>> properties;
                std::unordered_set<std::string> required;
                for (const auto & prop : obj.properties) {
                    properties.emplace_back(prop.name, prop.schema.get());
                    if (prop.required) {
                        required.insert(prop.name);
                    }
                }
                return _add_rule(rule_name, _build_object_rule(properties, required, name, obj.additional_properties.get()));
            }
            case COMMON_SCHEMA_KIND_TUPLE: {
                const auto & items = as<common_schema_tuple>(schema).items;
                std::string rule = "\"[\" space ";
                for (size_t i = 0; i < items.size(); i++) {
                    if (i > 0) {
                        rule += " \",\" space ";
                    }
                    rule += visit(*items[i], sub_name + "tuple-" + std::to_string(i));
                }
                rule += " space \"]\"";
                return _add_rule(rule_name, rule);
            }
            case COMMON_SCHEMA_KIND_ARRAY: {
                const auto & arr = as<common_schema_array>(schema);
                if (arr.items->kind() == COMMON_SCHEMA_KIND_ANY && arr.min_items == 0 && arr.max_items < 0) {
                    return _visit_primitive(rule_name, "array");
                }
                std::string item_rule_name = visit(*arr.items, sub_name + "item");
                int max_items = arr.max_items < 0 ? std::numeric_limits<int>::max() : arr.max_items;
                return _add_rule(rule_name, "\"[\" space " + build_repetition(item_rule_name, arr.min_items, max_items, "\",\" space") + " space \"]\"");
            }
            case COMMON_SCHEMA_KIND_STRING: {
                const auto & str = as<common_schema_string>(schema);
                if (!str.pattern.empty()) {
                    return _visit_pattern(str.pattern, rule_name);
                }
                if (str.format == COMMON_SCHEMA_FORMAT_UUID) {
                    return _visit_primitive(rule_name, "uuid");
                }
                if (str.format != COMMON_SCHEMA_FORMAT_NONE) {
                    std::string prim_name = std::string(str.format == COMMON_SCHEMA_FORMAT_DATE ? "date" : str.format == COMMON_SCHEMA_FORMAT_TIME ? "time" : "date-time") + "-string";
                    return _add_rule(rule_name, _add_primitive(prim_name, STRING_FORMAT_RULES.at(prim_name)));
                }
                if (str.min_length > 0 || str.max_length >= 0) {
                    std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
                    int max_len = str.max_length < 0 ? std::numeric_limits<int>::max() : str.max_length;
                    return _add_rule(rule_name, "\"\\\"\" " + build_repetition(char_rule, str.min_length, max_len) + " \"\\\"\"");
                }
                return _visit_primitive(rule_name, "string");
            }
            case COMMON_SCHEMA_KIND_INTEGER: {
                const auto & i = as<common_schema_integer>(schema);
                if (i.minimum == std::numeric_limits<int64_t>::min() && i.maximum == std::numeric_limits<int64_t>::max()) {
                    return _visit_primitive(rule_name, "integer");
                }
                std::stringstream out;
                out << "(";
                build_min_max_int(i.minimum, i.maximum, out);
                out << ")";
                return _add_rule(rule_name, out.str());
            }
            case COMMON_SCHEMA_KIND_NUMBER:
                return _visit_primitive(rule_name, "number");
            case COMMON_SCHEMA_KIND_BOOLEAN:
                return _visit_primitive(rule_name, "boolean");
            case COMMON_SCHEMA_KIND_NULL:
                return _visit_primitive(rule_name, "null");
            case COMMON_SCHEMA_KIND_ANY:
                return _add_rule(rule_name, _add_primitive("value", PRIMITIVE_RULES.at("value")));
        }
        return "";
    }

    void check_errors() {
        if (!_errors.empty()) {
            throw std::invalid_argument("JSON schema conversion failed:\n" + string_join(_errors, "\n"));
        }
        if (!_warnings.empty()) {
            fprintf(stderr, "WARNING: JSON schema conversion was incomplete: %s\n", string_join(_warnings, "; ").c_str());
        }
    }

    std::string format_grammar() {
        std::stringstream ss;
        for (const auto & kv : _rules) {
            ss << kv.first << " ::= " << kv.second << '\n';
        }
        return ss.str();
    }
};

bool common_schema_resolves_to_string(const common_schema & schema) {
    std::unordered_set<const common_schema *> visited;

    std::function<bool(const common_schema &)> check = [&](const common_schema & s) -> bool {
        switch (s.kind()) {
            case COMMON_SCHEMA_KIND_STRING:
                return true;
            case COMMON_SCHEMA_KIND_CONST:
                return static_cast<const common_schema_const &>(s).value.is_string();
            case COMMON_SCHEMA_KIND_ENUM:
                for (const auto & v : static_cast<const common_schema_enum &>(s).values) {
                    if (v.is_string()) {
                        return true;
                    }
                }
                return false;
            case COMMON_SCHEMA_KIND_REF: {
                // a cycle is taken as not a string, to be safe
                const auto * target = static_cast<const common_schema_ref &>(s).target;
                return target && visited.insert(target).second && check(*target);
            }
            case COMMON_SCHEMA_KIND_ANY_OF:
                for (const auto & child : static_cast<const common_schema_any_of &>(s).children) {
                    if (check(*child)) {
                        return true;
                    }
                }
                return false;
            case COMMON_SCHEMA_KIND_ALL_OF: {
                // every child must allow a string, an any child constrains nothing
                bool any_string = false;
                for (const auto & child : static_cast<const common_schema_all_of &>(s).children) {
                    if (child->kind() == COMMON_SCHEMA_KIND_ANY) {
                        continue;
                    }
                    if (!check(*child)) {
                        return false;
                    }
                    any_string = true;
                }
                return any_string;
            }
            default:
                return false;
        }
    };

    return check(schema);
}

void common_schema_info::resolve_refs(const common_json & schema) {
    // a schema that does not parse is reported when its grammar is built, here it only answers no
    try {
        common_schema_parse(schema, doc_);
    } catch (const std::runtime_error &) {
    }
}

bool common_schema_info::resolves_to_string(const common_json & schema) {
    try {
        return common_schema_resolves_to_string(*common_schema_parse(schema, doc_));
    } catch (const std::runtime_error &) {
        return false;
    }
}

common_schema_kinds common_schema_info::resolve_kinds(const common_json & schema) {
    try {
        return common_schema_resolve_kinds(*common_schema_parse(schema, doc_));
    } catch (const std::runtime_error &) {
        return {};
    }
}

std::string json_schema_to_grammar(const common_json & schema, bool force_gbnf) {
#ifdef LLAMA_USE_LLGUIDANCE
    if (!force_gbnf) {
        return "%llguidance {}\nstart: %json " + schema.dump();
    }
#else
    (void)force_gbnf;
#endif // LLAMA_USE_LLGUIDANCE
    return build_grammar([&](const common_grammar_builder & callbacks) {
        callbacks.add_schema("", schema);
    });
}

std::string json_schema_to_grammar(const common_schema_document & schema) {
    common_schema_converter converter(false);
    converter.visit(*schema.root, "");
    converter.check_errors();
    return converter.format_grammar();
}

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options) {
    common_schema_converter converter(options.dotall);
    common_grammar_builder builder {
        /* .add_rule = */ [&](const std::string & name, const std::string & rule) {
            return converter._add_rule(name, rule);
        },
        /* .add_schema = */ [&](const std::string & name, const common_json & schema) {
            return converter.add_schema(name == "root" ? "" : name, schema);
        },
        /* .resolve_refs = */ [&](const common_json & schema) {
            converter.resolve_refs(schema);
        }
    };
    cb(builder);
    converter.check_errors();
    return converter.format_grammar();
}
