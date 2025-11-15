#include "json-schema-to-grammar.h"
#include "tests.h"

void test_gbnf_generation(testing &t) {
    // Test literal
    t.test("literal grammar generation", [](testing &t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.literal("hello"); });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        t.assert_equal("has_root_hello", true, gbnf.find("root ::= \"hello\"") != std::string::npos);
        t.assert_equal("has_space", true, gbnf.find("space ::=") != std::string::npos);
    });

    // Test char class
    t.test("char class grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one("[a-z]"); });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        t.assert_equal("has_char_class", true, gbnf.find("root ::= [a-z]") != std::string::npos);
    });

    // Test sequence
    t.test("sequence grammar", [](testing &t) {
        auto parser = build_peg_parser(
            [](common_chat_peg_parser_builder & p) { return p.literal("hello") + p.literal(" ") + p.literal("world"); });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        t.assert_equal("has_proper_sequence", true,
                        gbnf.find("root ::= \"hello\" \" \" \"world\"") != std::string::npos);
    });

    // Test choice
    t.test("choice grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.literal("cat") | p.literal("dog"); });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        t.assert_equal("has_proper_choice", true, gbnf.find("root ::= \"cat\" | \"dog\"") != std::string::npos);
    });

    // Test one_or_more
    t.test("one_or_more grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one_or_more(p.one("[0-9]")); });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        t.assert_equal("has_proper_one_or_more", true, gbnf.find("root ::= [0-9]+") != std::string::npos);
    });

    // Test zero_or_more
    t.test("zero_or_more grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.zero_or_more(p.one("[a-z]")); });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        t.assert_equal("has_proper_zero_or_more", true, gbnf.find("root ::= [a-z]*") != std::string::npos);
    });

    // Test optional
    t.test("optional grammar", [](testing &t) {
        auto parser =
            build_peg_parser([](common_chat_peg_parser_builder & p) { return p.literal("hello") + p.optional(p.literal(" world")); });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        t.assert_equal("has_proper_optional", true,
                        gbnf.find("root ::= \"hello\" \" world\"?") != std::string::npos);
    });

    // Test until
    t.test("until grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.until("</tag>"); });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        // Should generate pattern that prevents matching the full delimiter
        t.assert_equal("has_proper_until", true,
            gbnf.find(
                "root ::= ([^<] | \"<\" [^/] | \"</\" [^t] | \"</t\" [^a] | \"</ta\" [^g] | \"</tag\" [^>])*") !=
                std::string::npos);
    });

    // Test complex expression with parentheses
    t.test("complex expressions with parentheses", [](testing &t) {
        auto parser =
            build_peg_parser([](common_chat_peg_parser_builder & p) { return p.one_or_more(p.literal("a") | p.literal("b")); });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        t.assert_equal("has_proper_complex", true, gbnf.find("root ::= (\"a\" | \"b\")+") != std::string::npos);
    });

    // Test rule references
    t.test("rule references", [](testing &t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
            auto digit = p.add_rule("digit", p.one("[0-9]"));
            return p.one_or_more(digit);
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        // Should have digit rule defined and referenced
        t.assert_equal("has_digit_rule", true, gbnf.find("digit ::= [0-9]") != std::string::npos);
        t.assert_equal("has_root_digit_ref", true, gbnf.find("root ::= digit+") != std::string::npos);
    });

    // Test escaping in literals
    t.test("escaping in literals", [](testing &t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.literal("hello\nworld\t!"); });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        t.assert_equal("has_escaping", true, gbnf.find("root ::= \"hello\\nworld\\t!\"") != std::string::npos);
    });

    // Test operator<< (whitespace insertion)
    t.test("operator<< (whitespace insertion)", [](testing &t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) { return p.literal("hello") << p.literal("world"); });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });

        // Should inline the whitespace pattern
        t.assert_equal("has_inlined_hello", true, gbnf.find("\"hello\"") != std::string::npos);
        t.assert_equal("has_inlined_world", true, gbnf.find("\"world\"") != std::string::npos);
    });
}
