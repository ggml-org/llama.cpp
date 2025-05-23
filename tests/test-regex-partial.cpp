//  Tests common_regex (esp. its partial final matches support).

#include "common.h"
#include "regex-partial.h"

#include <sstream>
#include <iostream>
#include <optional>

template <class T> static void assert_equals(const T & expected, const T & actual) {
    if (expected != actual) {
        std::cerr << "Expected: " << expected << std::endl;
        std::cerr << "  Actual: " << actual << std::endl;
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

struct test_case {
    std::string pattern;
    struct input_output {
        std::string input;
        common_regex_match output;
    };
    std::vector<input_output> inputs_outputs;
};

static std::string common_regex_match_type_name(common_regex_match_type type) {
    switch (type) {
        case COMMON_REGEX_MATCH_TYPE_NONE:
            return "COMMON_REGEX_MATCH_TYPE_NONE";
        case COMMON_REGEX_MATCH_TYPE_PARTIAL:
            return "COMMON_REGEX_MATCH_TYPE_PARTIAL";
        case COMMON_REGEX_MATCH_TYPE_FULL:
            return "COMMON_REGEX_MATCH_TYPE_FULL";
    }
    return "?";
}

static void test_regex() {
    printf("[%s]\n", __func__);
    auto test = [](const test_case & test_case) {
        common_regex cr(test_case.pattern);
        std::cout << "Testing pattern: /" << test_case.pattern << "/\n";
        // std::cout << "    partial rev: " << cr.reversed_partial_pattern.str() << '\n';
        for (const auto & input_output : test_case.inputs_outputs) {
            std::cout << "  Input: " << input_output.input << '\n';
            auto m = cr.search(input_output.input, 0);
            if (m != input_output.output) {
                auto match_to_str = [&](const std::optional<common_regex_match> & m) {
                    std::ostringstream ss;
                    if (m->type == COMMON_REGEX_MATCH_TYPE_NONE) {
                        ss << "<no match>";
                    } else {
                        GGML_ASSERT(!input_output.output.groups.empty());
                        std::vector<std::string> parts;
                        for (const auto & g : m->groups) {
                            parts.push_back("{" + std::to_string(g.begin) + ", " + std::to_string(g.end) + "}");
                        }
                        ss << "{" << common_regex_match_type_name(m->type) << ", {" << string_join(parts, ", ") << "}}";
                    }
                    return ss.str();
                };
                std::cout << "    Expected: " << match_to_str(input_output.output) << '\n';
                std::cout << "         Got: " << match_to_str(m) << '\n';
                std::cout << " Inverted pattern: /" << regex_to_reversed_partial_regex(test_case.pattern) << "/\n";

                throw std::runtime_error("Test failed");
            }
        }
    };
    test({
        "a",
        {
            {"a", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 1}}}},
            {"b", {COMMON_REGEX_MATCH_TYPE_NONE, {}}},
            {"ab", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 1}}}},
            {"ba", {COMMON_REGEX_MATCH_TYPE_FULL, {{1, 2}}}},
        }
    });
    test({
        "abcd",
        {
            {"abcd", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 4}}}},
            {"abcde", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 4}}}},
            {"abc", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 3}}}},
            {"ab", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 2}}}},
            {"a", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 1}}}},
            {"d", {}},
            {"bcd", {}},
            {"cde", {}},
            {"cd", {}},
            {"yeah ab", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{5, 7}}}},
            {"abbie", {}},
            {"", {}},
        }
    });
    test({
        ".*?ab",
        {
            {"ab", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 2}}}},
            {"abc", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 2}}}},
            {"dab", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 3}}}},
            {"dabc", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 3}}}},
            {"da", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 2}}}},
            {"d", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 1}}}},
        }
    });
    test({
        "a.*?b",
        {
            {"ab", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 2}}}},
            {"abc", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 2}}}},
            {"a b", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 3}}}},
            {"a", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 1}}}},
            {"argh", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 4}}}},
            {"d", {}},
            {"b", {}},
        }
    });
    test({
        "ab(?:cd){2,4}ef",
        {
            // {"ab", {COMMON_REGEX_MATCH_TYPE_PARTIAL, 0, {}}},
            {"ab", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 2}}}},
            {"abcd", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 4}}}},
            {"abcde", {}},
            {"abcdef", {}},
            {"abcdcd", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 6}}}},
            {"abcdcde", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 7}}}},
            {"abcdcdef", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 8}}}},
            {"abcdcdcdcdef", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 12}}}},
            {"abcdcdcdcdcdef", {}},
            {"abcde", {}},
            {"yea", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{2, 3}}}},
        }
    });
    test({
        "a(?:rte| pure )fact",
        {
            {"a", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 1}}}},
            {"art", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 3}}}},
            {"artefa", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 6}}}},
            {"fact", {}},
            {"an arte", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{3, 7}}}},
            {"artefact", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 8}}}},
            {"an artefact", {COMMON_REGEX_MATCH_TYPE_FULL, {{3, 11}}}},
            {"a pure", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 6}}}},
            {"a pure fact", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 11}}}},
            {"it's a pure fact", {COMMON_REGEX_MATCH_TYPE_FULL, {{5, 16}}}},
            {"" , {}},
            {"pure", {}},
            {"pure fact", {}},
        }
    });
    test({
        "abc",
        {
            {" abcc", {COMMON_REGEX_MATCH_TYPE_FULL, {{1, 4}}}},
            {"ab", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 2}}}},
            {"abc", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 3}}}},
            {" ab", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{1, 3}}}},
            {"a", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 1}}}},
            {"b", {}},
            {"c", {}},
            {"", {}},
        }
    });

    test({
        "(?:abc)?\\s*def",
        {
            {"ab", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 2}}}},
            {"abc", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 3}}}},
            {"abc ", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 4}}}},
            {"abc d", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 5}}}},
            {"abc de", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 6}}}},
            {"abc def", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 7}}}},
            {"abc defg", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 7}}}},
            {"abc defgh", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 7}}}},
            {"abcde", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 5}}}},
            {"abcdefgh", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 6}}}},
            {" d", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 2}}}},
            {"def", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 3}}}},
        }
    });

    test({
        "a+b",
        {
            {"aaab", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 4}}}},
            {"aaa", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 3}}}},
            {"ab", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 2}}}},
        }
    });

    test({
        "(?:"
            "(```(?:xml|json)?\\n\\s*)?" // match 1 (block_start)
            "("                          // match 2 (open_tag)
                "<tool_call>"
                "|<function_call>"
                "|<tool>"
                "|<tools>"
                "|<response>"
                "|<json>"
                "|<xml>"
                "|<JSON>"
            ")?"
            "(\\s*\\{\\s*\"name\"\\s*:)" // match 3 (named tool call)
        ")"
        "|<function=([^>]+)>"            // match 4 (function name)
        "|<function name=\"([^\"]+)\">", // match 5 (function name again)
        {
            {"{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 8}, {54, 54}, {54, 54}, {0, 8}, {54, 54}, {54, 54}}}},
            {"<tool_call> {\"name", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 18}}}},
            {"<tool_call>{\"name", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 17}}}},
            {"Let's call something\n<tool_call>{\"name", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{21, 38}}}},
            {"Ok then<tool_call>{\"name", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{7, 24}}}},
            {"{\"name", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{0, 6}}}},
            {"Ok then{\"name", {COMMON_REGEX_MATCH_TYPE_PARTIAL, {{7, 13}}}},
            {"<tool_call> {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 20}, {66, 66}, {0, 11}, {11, 20}, {66, 66}, {66, 66}}}},
            {"<function_call> {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 24}, {70, 70}, {0, 15}, {15, 24}, {70, 70}, {70, 70}}}},
            {"<function name=\"special_function\"> {\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 34}, {89, 89}, {89, 89}, {89, 89}, {89, 89}, {16, 32}}}},
            {"<function=all>", {COMMON_REGEX_MATCH_TYPE_FULL, {{0, 14}, {14, 14}, {14, 14}, {14, 14}, {10, 13}, {14, 14}}}},

        }
    });
}

static void test_regex_to_reversed_partial_regex() {
    printf("[%s]\n", __func__);

    assert_equals<std::string>(
        "((?:(?:c)?b)?a)[\\s\\S]*",
        regex_to_reversed_partial_regex("abc"));

    assert_equals<std::string>(
        "(a+)[\\s\\S]*",
        regex_to_reversed_partial_regex("a+"));

    assert_equals<std::string>(
        "(a*)[\\s\\S]*",
        regex_to_reversed_partial_regex("a*"));

    assert_equals<std::string>(
        "(a?)[\\s\\S]*",
        regex_to_reversed_partial_regex("a?"));

    assert_equals<std::string>(
        "([a-z])[\\s\\S]*",
        regex_to_reversed_partial_regex("[a-z]"));

    assert_equals<std::string>(
        "((?:\\w+)?[a-z])[\\s\\S]*",
        regex_to_reversed_partial_regex("[a-z]\\w+"));

    assert_equals<std::string>(
        "((?:a|b))[\\s\\S]*",
        regex_to_reversed_partial_regex("(?:a|b)"));
    assert_equals<std::string>(
        "((?:(?:(?:d)?c)?b)?a)[\\s\\S]*",
        regex_to_reversed_partial_regex("abcd"));
    assert_equals<std::string>(
        "((?:b)?a*)[\\s\\S]*", // TODO: ((?:b)?a*+).* ??
        regex_to_reversed_partial_regex("a*b"));
    assert_equals<std::string>(
        "((?:(?:b)?a)?.*)[\\s\\S]*",
        regex_to_reversed_partial_regex(".*?ab"));
    assert_equals<std::string>(
        "((?:(?:b)?.*)?a)[\\s\\S]*",
        regex_to_reversed_partial_regex("a.*?b"));
    assert_equals<std::string>(
        "((?:(?:d)?(?:(?:c)?b))?a)[\\s\\S]*",
        regex_to_reversed_partial_regex("a(bc)d"));
    assert_equals<std::string>(
        "((?:(?:(?:c)?b|(?:e)?d))?a)[\\s\\S]*",
        regex_to_reversed_partial_regex("a(bc|de)"));
    assert_equals<std::string>(
        "((?:(?:(?:(?:(?:c)?b?)?b?)?b)?b)?a)[\\s\\S]*",
        regex_to_reversed_partial_regex("ab{2,4}c"));
}

int main() {
    test_regex_to_reversed_partial_regex();
    test_regex();
    std::cout << "All tests passed.\n";
}
