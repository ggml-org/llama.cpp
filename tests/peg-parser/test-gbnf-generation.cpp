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
            root ::= until-0
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-0 ::= | [<] until-0-01 | [^<] until-0
            until-0-01 ::= | [<] until-0-01 | [/] until-0-02 | [^/<] until-0
            until-0-02 ::= | [<] until-0-01 | [t] until-0-03 | [^<t] until-0
            until-0-03 ::= | [<] until-0-01 | [a] until-0-04 | [^<a] until-0
            until-0-04 ::= | [<] until-0-01 | [g] until-0-05 | [^<g] until-0
            until-0-05 ::= | [<] until-0-01 | [^<>] until-0
        )""", gbnf);
    });

    t.test("until grammar overlapping delimiter", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until("\n</parameter>\n");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-0
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-0 ::= | [\n] until-0-01 | [^\n] until-0
            until-0-01 ::= | [\n] until-0-01 | [<] until-0-02 | [^\n<] until-0
            until-0-02 ::= | [\n] until-0-01 | [/] until-0-03 | [^\n/] until-0
            until-0-03 ::= | [\n] until-0-01 | [p] until-0-04 | [^\np] until-0
            until-0-04 ::= | [\n] until-0-01 | [a] until-0-05 | [^\na] until-0
            until-0-05 ::= | [\n] until-0-01 | [r] until-0-06 | [^\nr] until-0
            until-0-06 ::= | [\n] until-0-01 | [a] until-0-07 | [^\na] until-0
            until-0-07 ::= | [\n] until-0-01 | [m] until-0-08 | [^\nm] until-0
            until-0-08 ::= | [\n] until-0-01 | [e] until-0-09 | [^\ne] until-0
            until-0-09 ::= | [\n] until-0-01 | [t] until-0-10 | [^\nt] until-0
            until-0-10 ::= | [\n] until-0-01 | [e] until-0-11 | [^\ne] until-0
            until-0-11 ::= | [\n] until-0-01 | [r] until-0-12 | [^\nr] until-0
            until-0-12 ::= | [\n] until-0-01 | [>] until-0-13 | [^\n>] until-0
            until-0-13 ::= | [^\n] until-0
        )""", gbnf);
    });

    // DeepSeek-V3.2 tag prefix. The DSML token (｜DSML｜) embeds U+FF5C,
    // so the delimiter mixes ASCII and multi-byte codepoints.
    t.test("until grammar unicode delimiter", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until("<｜DSML｜");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-0
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-0 ::= | [<] until-0-01 | [^<] until-0
            until-0-01 ::= | [<] until-0-01 | [\uFF5C] until-0-02 | [^<\uFF5C] until-0
            until-0-02 ::= | [<] until-0-01 | [D] until-0-03 | [^<D] until-0
            until-0-03 ::= | [<] until-0-01 | [S] until-0-04 | [^<S] until-0
            until-0-04 ::= | [<] until-0-01 | [M] until-0-05 | [^<M] until-0
            until-0-05 ::= | [<] until-0-01 | [L] until-0-06 | [^<L] until-0
            until-0-06 ::= | [<] until-0-01 | [^<\uFF5C] until-0
        )""", gbnf);
    });

    t.test("until grammar multiple delimiters", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until_one_of({"ab", "cd", "ef"});
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-0
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-0 ::= | [a] until-0-01 | [c] until-0-03 | [e] until-0-05 | [^ace] until-0
            until-0-01 ::= | [a] until-0-01 | [c] until-0-03 | [e] until-0-05 | [^abce] until-0
            until-0-03 ::= | [a] until-0-01 | [c] until-0-03 | [e] until-0-05 | [^acde] until-0
            until-0-05 ::= | [a] until-0-01 | [c] until-0-03 | [e] until-0-05 | [^acef] until-0
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

    t.test("tagged choice inside sequence gets parenthesized", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("a") + p.tag("t", p.literal("b") | p.literal("c"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "a" ("b" | "c")
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("tagged sequence inside choice gets parenthesized", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.tag("t", p.literal("a") + p.literal("b")) | p.literal("c");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "a" "b" | "c"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("atomic choice inside repetition gets parenthesized", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.one_or_more(p.atomic(p.literal("a") | p.literal("b")));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= ("a" | "b")+
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("silent parser emits nothing in gbnf", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("hello") + p.gbnf(p.literal("world"), "");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "hello"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("silent choice inside sequence emits nothing", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("a") + p.gbnf(p.literal("b") | p.literal("c"), "") + p.literal("d");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "a" "d"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("silent wrapped in tag emits nothing", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("a") + p.tag("t", p.gbnf(p.literal("b"), ""));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "a"
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("gbnf parser emits custom grammar", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("a") + p.gbnf(p.literal("b"), "[a-z]+");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "a" [a-z]+
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("nested transparent wrappers get parenthesized", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p) {
            return p.literal("x") + p.tag("outer", p.atomic(p.literal("a") | p.literal("b")));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= "x" ("a" | "b")
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
