#include "json-schema-to-grammar.h"
#include "tests.h"

test_gbnf_generation::test_gbnf_generation() : compound_test("test_gbnf_generation") {
    // Test literal
    add_test(
        [](test_harness h) {
            auto parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.literal("hello"); });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            h.assert_equals("has_root_hello", true, gbnf.find("root ::= \"hello\"") != std::string::npos);
            h.assert_equals("has_space", true, gbnf.find("space ::=") != std::string::npos);
        },
        "literal grammar generation");

    // Test char class
    add_test(
        [](test_harness h) {
            auto parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.one("[a-z]"); });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            h.assert_equals("has_char_class", true, gbnf.find("root ::= [a-z]") != std::string::npos);
        },
        "char class grammar");

    // Test sequence
    add_test(
        [](test_harness h) {
            auto parser = build_combinator_parser(
                [](common_chat_combinator_parser_builder & p) { return p.literal("hello") + p.literal(" ") + p.literal("world"); });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            h.assert_equals("has_proper_sequence", true,
                            gbnf.find("root ::= \"hello\" \" \" \"world\"") != std::string::npos);
        },
        "sequence grammar");

    // Test choice
    add_test(
        [](test_harness h) {
            auto parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.literal("cat") | p.literal("dog"); });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            h.assert_equals("has_proper_choice", true, gbnf.find("root ::= \"cat\" | \"dog\"") != std::string::npos);
        },
        "choice grammar");

    // Test one_or_more
    add_test(
        [](test_harness h) {
            auto parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.one_or_more(p.one("[0-9]")); });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            h.assert_equals("has_proper_one_or_more", true, gbnf.find("root ::= [0-9]+") != std::string::npos);
        },
        "one_or_more grammar");

    // Test zero_or_more
    add_test(
        [](test_harness h) {
            auto parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.zero_or_more(p.one("[a-z]")); });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            h.assert_equals("has_proper_zero_or_more", true, gbnf.find("root ::= [a-z]*") != std::string::npos);
        },
        "zero_or_more grammar");

    // Test optional
    add_test(
        [](test_harness h) {
            auto parser =
                build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.literal("hello") + p.optional(p.literal(" world")); });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            h.assert_equals("has_proper_optional", true,
                            gbnf.find("root ::= \"hello\" \" world\"?") != std::string::npos);
        },
        "optional grammar");

    // Test until
    add_test(
        [](test_harness h) {
            auto parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.until("</tag>"); });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            // Should generate pattern that prevents matching the full delimiter
            h.assert_equals(
                "has_proper_until", true,
                gbnf.find(
                    "root ::= ([^<] | \"<\" [^/] | \"</\" [^t] | \"</t\" [^a] | \"</ta\" [^g] | \"</tag\" [^>])*") !=
                    std::string::npos);
        },
        "until grammar");

    // Test complex expression with parentheses
    add_test(
        [](test_harness h) {
            auto parser =
                build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.one_or_more(p.literal("a") | p.literal("b")); });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            h.assert_equals("has_proper_complex", true, gbnf.find("root ::= (\"a\" | \"b\")+") != std::string::npos);
        },
        "complex expressions with parentheses");

    // Test rule references
    add_test(
        [](test_harness h) {
            auto parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) {
                auto digit = p.add_rule("digit", p.one("[0-9]"));
                return p.one_or_more(digit);
            });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            // Should have digit rule defined and referenced
            h.assert_equals("has_digit_rule", true, gbnf.find("digit ::= [0-9]") != std::string::npos);
            h.assert_equals("has_root_digit_ref", true, gbnf.find("root ::= digit+") != std::string::npos);
        },
        "rule references");

    // Test escaping in literals
    add_test(
        [](test_harness h) {
            auto parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.literal("hello\nworld\t!"); });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            h.assert_equals("has_escaping", true, gbnf.find("root ::= \"hello\\nworld\\t!\"") != std::string::npos);
        },
        "escaping in literals");

    // Test operator<< (whitespace insertion)
    add_test(
        [](test_harness h) {
            auto parser = build_combinator_parser([](common_chat_combinator_parser_builder & p) { return p.literal("hello") << p.literal("world"); });

            auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

            // Should inline the whitespace pattern
            h.assert_equals("has_inlined_hello", true, gbnf.find("\"hello\"") != std::string::npos);
            h.assert_equals("has_inlined_world", true, gbnf.find("\"world\"") != std::string::npos);
        },
        "operator<< (whitespace insertion)");
}
