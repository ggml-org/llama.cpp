#include "tests.h"

#include "json-schema-to-grammar.h"

#include <regex>

static std::string trim_leading_space(const std::string & s) {
    static const std::regex leading_ws_re = std::regex(R"((^|\n)\s+)");
    return std::regex_replace(s, leading_ws_re, "$1");
}

static void assert_gbnf_equal(testing & t, const std::string & expected, const std::string & actual) {
    t.assert_equal("gbnf are equal", trim_leading_space(expected), trim_leading_space(actual));
}

void test_gbnf_generation(testing &t) {
    t.test("literal grammar generation", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("hello");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "hello"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("char class grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.chars("[a-z]", 1, 1);
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= [a-z]
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("sequence grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("hello") + p.literal(" ") + p.literal("world");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "hello" " " "world"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("choice grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("cat") | p.literal("dog");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "cat" | "dog"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("one_or_more grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.one_or_more(p.literal("a"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "a"+
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("zero_or_more grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.zero_or_more(p.literal("a"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "a"*
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("optional grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("hello") + p.optional(p.literal(" world"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "hello" " world"?
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("until grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until("</tag>");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= ([^<] | "<" [^/] | "</" [^t] | "</t" [^a] | "</ta" [^g] | "</tag" [^>])*
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("complex expressions with parentheses", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.one_or_more(p.literal("a") | p.literal("b"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= ("a" | "b")+
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("rule references", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            auto digit = p.rule("digit", p.chars("[0-9]", 1, 1));
            return p.one_or_more(digit);
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            digit ::= [0-9]
            root ::= digit+
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("escaping in literals", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("hello\nworld\n!");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "hello\nworld\n!"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("operator<< (whitespace insertion)", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("hello") << p.literal("world");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "hello" space "world"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("emit only reachable rules", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            p.rule("orphan", p.literal("orphan"));
            return p.literal("hello") + p.rule("child", p.literal(" world"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            child ::= " world"
            root ::= "hello" child
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("emit only trigger rules (and references)", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            auto rule1 = p.rule("rule-1", p.literal("a") + p.ref("rule-2"));
            p.rule("rule-2", p.literal("b") + p.ref("rule-3"), true);
            p.rule("rule-3", p.literal("c") + p.ref("rule-4"));
            p.rule("rule-4", p.literal("d"), true);
            return rule1;
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= rule-1
            rule-1 ::= "a" rule-2
            rule-2 ::= "b" rule-3
            rule-3 ::= "c" rule-4
            rule-4 ::= "d"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);

        auto gbnf_lazy = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, true);
        });

        assert_gbnf_equal(t, R"""(
            root ::= rule-2 | rule-4
            rule-2 ::= "b" rule-3
            rule-3 ::= "c" rule-4
            rule-4 ::= "d"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf_lazy);
    });
}
