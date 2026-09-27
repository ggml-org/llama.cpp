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
            root ::= until-1
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-1 ::= | [<] until-1-01 | [^<] until-1
            until-1-01 ::= | [<] until-1-01 | [/] until-1-02 | [^/<] until-1
            until-1-02 ::= | [<] until-1-01 | [t] until-1-03 | [^<t] until-1
            until-1-03 ::= | [<] until-1-01 | [a] until-1-04 | [^<a] until-1
            until-1-04 ::= | [<] until-1-01 | [g] until-1-05 | [^<g] until-1
            until-1-05 ::= | [<] until-1-01 | [^<>] until-1
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
            root ::= until-1
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-1 ::= | [\n] until-1-01 | [^\n] until-1
            until-1-01 ::= | [\n] until-1-01 | [<] until-1-02 | [^\n<] until-1
            until-1-02 ::= | [\n] until-1-01 | [/] until-1-03 | [^\n/] until-1
            until-1-03 ::= | [\n] until-1-01 | [p] until-1-04 | [^\np] until-1
            until-1-04 ::= | [\n] until-1-01 | [a] until-1-05 | [^\na] until-1
            until-1-05 ::= | [\n] until-1-01 | [r] until-1-06 | [^\nr] until-1
            until-1-06 ::= | [\n] until-1-01 | [a] until-1-07 | [^\na] until-1
            until-1-07 ::= | [\n] until-1-01 | [m] until-1-08 | [^\nm] until-1
            until-1-08 ::= | [\n] until-1-01 | [e] until-1-09 | [^\ne] until-1
            until-1-09 ::= | [\n] until-1-01 | [t] until-1-10 | [^\nt] until-1
            until-1-10 ::= | [\n] until-1-01 | [e] until-1-11 | [^\ne] until-1
            until-1-11 ::= | [\n] until-1-01 | [r] until-1-12 | [^\nr] until-1
            until-1-12 ::= | [\n] until-1-01 | [>] until-1-13 | [^\n>] until-1
            until-1-13 ::= | [^\n] until-1
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
            root ::= until-1
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-1 ::= | [<] until-1-01 | [^<] until-1
            until-1-01 ::= | [<] until-1-01 | [\uFF5C] until-1-02 | [^<\uFF5C] until-1
            until-1-02 ::= | [<] until-1-01 | [D] until-1-03 | [^<D] until-1
            until-1-03 ::= | [<] until-1-01 | [S] until-1-04 | [^<S] until-1
            until-1-04 ::= | [<] until-1-01 | [M] until-1-05 | [^<M] until-1
            until-1-05 ::= | [<] until-1-01 | [L] until-1-06 | [^<L] until-1
            until-1-06 ::= | [<] until-1-01 | [^<\uFF5C] until-1
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
            root ::= until-3
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-3 ::= | [a] until-3-01 | [c] until-3-03 | [e] until-3-05 | [^ace] until-3
            until-3-01 ::= | [a] until-3-01 | [c] until-3-03 | [e] until-3-05 | [^abce] until-3
            until-3-03 ::= | [a] until-3-01 | [c] until-3-03 | [e] until-3-05 | [^acde] until-3
            until-3-05 ::= | [a] until-3-01 | [c] until-3-03 | [e] until-3-05 | [^acef] until-3
        )""", gbnf);
    });

    t.test("until grammar with content", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until(p.until("</tag>"), "</tag>");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-3
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-3 ::= [<] until-3-01 | [^<] until-3
            until-3-01 ::= [<] until-3-01 | [/] until-3-02 | [^/<] until-3
            until-3-02 ::= [<] until-3-01 | [t] until-3-03 | [^<t] until-3
            until-3-03 ::= [<] until-3-01 | [a] until-3-04 | [^<a] until-3
            until-3-04 ::= [<] until-3-01 | [g] until-3-05 | [^<g] until-3
            until-3-05 ::= [>] | [<] until-3-01 | [^<>] until-3
            
        )""", gbnf);
    });

    t.test("until grammar with content terminates at first delimiter", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until(p.until("\n</parameter>\n"), "\n</parameter>\n");
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-3
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-3 ::= [\n] until-3-01 | [^\n] until-3
            until-3-01 ::= [\n] until-3-01 | [<] until-3-02 | [^\n<] until-3
            until-3-02 ::= [\n] until-3-01 | [/] until-3-03 | [^\n/] until-3
            until-3-03 ::= [\n] until-3-01 | [p] until-3-04 | [^\np] until-3
            until-3-04 ::= [\n] until-3-01 | [a] until-3-05 | [^\na] until-3
            until-3-05 ::= [\n] until-3-01 | [r] until-3-06 | [^\nr] until-3
            until-3-06 ::= [\n] until-3-01 | [a] until-3-07 | [^\na] until-3
            until-3-07 ::= [\n] until-3-01 | [m] until-3-08 | [^\nm] until-3
            until-3-08 ::= [\n] until-3-01 | [e] until-3-09 | [^\ne] until-3
            until-3-09 ::= [\n] until-3-01 | [t] until-3-10 | [^\nt] until-3
            until-3-10 ::= [\n] until-3-01 | [e] until-3-11 | [^\ne] until-3
            until-3-11 ::= [\n] until-3-01 | [r] until-3-12 | [^\nr] until-3
            until-3-12 ::= [\n] until-3-01 | [>] until-3-13 | [^\n>] until-3
            until-3-13 ::= [\n] | [^\n] until-3
            
        )""", gbnf);
    });

    t.test("until grammar with content multiple delimiters", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until(p.eps(), { { p.literal("ab") }, { p.literal("cd") }, { p.literal("ef") } });
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-4
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-4 ::= [a] until-4-01 | [c] until-4-03 | [e] until-4-05 | [^ace] until-4
            until-4-01 ::= [b] | [a] until-4-01 | [c] until-4-03 | [e] until-4-05 | [^abce] until-4
            until-4-03 ::= [d] | [a] until-4-01 | [c] until-4-03 | [e] until-4-05 | [^acde] until-4
            until-4-05 ::= [f] | [a] until-4-01 | [c] until-4-03 | [e] until-4-05 | [^acef] until-4
            
        )""", gbnf);
    });

    t.test("until grammar with parser delimiters", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until(p.literal("a") + p.literal("b") | p.literal("cd"));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-5
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-5 ::= | [a] until-5-01 | [c] until-5-03 | [^ac] until-5
            until-5-01 ::= | [a] until-5-01 | [c] until-5-03 | [^abc] until-5
            until-5-03 ::= | [a] until-5-01 | [c] until-5-03 | [^acd] until-5
        )""", gbnf);
    });

    t.test("delimiters see through tags and atomics", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until(p.tag("open", p.literal("a") + p.atomic(p.literal("b"))));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-5
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-5 ::= | [a] until-5-01 | [^a] until-5
            until-5-01 ::= | [a] until-5-01 | [^ab] until-5
        )""", gbnf);
    });

    t.test("delimiters expand a choice in a sequence", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until(p.literal("a") + (p.literal("b") | p.literal("d")));
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-5
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-5 ::= | [a] until-5-01 | [^a] until-5
            until-5-01 ::= | [a] until-5-01 | [^abd] until-5
        )""", gbnf);
    });

    t.test("delimiters reject anything but sequences, choices, literals, and tokens", [](testing &t) {
        bool threw = false;
        try {
            build_peg_parser([](common_peg_parser_builder & p)  {
                return p.until(p.literal("a") + p.space());
            });
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        t.assert_equal("space_rejected", true, threw);
    });

    t.test("until branches reject a shared delimiter", [](testing &t) {
        bool threw = false;
        try {
            build_peg_parser([](common_peg_parser_builder & p)  {
                return p.until(p.eps(), { { p.literal("ab"), p.literal("x") }, { p.literal("ab"), p.literal("y") } });
            });
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        t.assert_equal("duplicate_rejected", true, threw);
    });

    t.test("until grammar branches continue with their rest", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until(p.eps(), { { p.literal("ab"), p.literal("x") }, { p.literal("cd") } });
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-4
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-4 ::= [a] until-4-01 | [c] until-4-03 | [^ac] until-4
            until-4-01 ::= [b] until-4-rest-0 | [a] until-4-01 | [c] until-4-03 | [^abc] until-4
            until-4-03 ::= [d] | [a] until-4-01 | [c] until-4-03 | [^acd] until-4
            until-4-rest-0 ::= "x"
            
        )""", gbnf);
    });

    t.test("until grammar optional may end before a delimiter", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until(p.eps(), { { p.literal("ab"), p.literal("x") } }, true);
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-3
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-3 ::= | [a] until-3-01 | [^a] until-3
            until-3-01 ::= | [b] until-3-rest-0 | [a] until-3-01 | [^ab] until-3
            until-3-rest-0 ::= "x"
            
        )""", gbnf);
    });

    t.test("trigger rule emits its alternatives when not lazy", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            auto ab = p.literal("ab");
            auto cd = p.literal("cd");
            std::vector<common_peg_trigger> triggers = { { ab, p.literal("x") }, { cd, p.literal("y") } };
            return p.until(ab | cd) + p.trigger_rule("tool", triggers);
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, false);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-5 tool
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            tool ::= "ab" "x" | "cd" "y"
            until-5 ::= | [a] until-5-01 | [c] until-5-03 | [^ac] until-5
            until-5-01 ::= | [a] until-5-01 | [c] until-5-03 | [^abc] until-5
            until-5-03 ::= | [a] until-5-01 | [c] until-5-03 | [^acd] until-5
        )""", gbnf);
    });

    t.test("trigger rule makes the lazy grammar scan for its starts", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            auto ab = p.literal("ab");
            auto cd = p.literal("cd");
            std::vector<common_peg_trigger> triggers = { { ab, p.literal("x") }, { cd, p.literal("y") } };
            return p.until(ab | cd) + p.trigger_rule("tool", triggers);
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, true);
        });

        assert_gbnf_equal(t, R"""(
            root ::= trigger
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            tool-rest-0 ::= "x"
            tool-rest-1 ::= "y"
            trigger ::= | [a] trigger-01 | [c] trigger-03 | [^ac] trigger
            trigger-01 ::= | [b] tool-rest-0 | [a] trigger-01 | [c] trigger-03 | [^abc] trigger
            trigger-03 ::= | [d] tool-rest-1 | [a] trigger-01 | [c] trigger-03 | [^acd] trigger
        )""", gbnf);
    });

    t.test("scanning grammar lets other named tokens through", [](testing &t) {
        common_peg_tokens tokens;
        tokens.ids.emplace("<a>", 1);
        tokens.ids.emplace("<b>", 2);
        tokens.tokens.emplace(1, common_peg_token{ "<a>", LLAMA_TOKEN_ATTR_CONTROL });
        tokens.tokens.emplace(2, common_peg_token{ "<b>", LLAMA_TOKEN_ATTR_CONTROL });

        common_peg_parser_builder p(tokens);
        auto start = p.token("<a>") + p.literal("x");
        std::vector<common_peg_trigger> triggers = { { start, p.token("<b>") } };
        p.set_root(p.until(start) + p.trigger_rule("tool", triggers));
        auto parser = p.build();

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, true);
        });

        assert_gbnf_equal(t, R"""(
            root ::= trigger
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            tool-rest-0 ::= <[2]>
            trigger ::= | <[1]> trigger-01 | (. | <[2]>) trigger
            trigger-01 ::= | [x] tool-rest-0 | <[1]> trigger-01 | ([^x] | <[2]>) trigger
        )""", gbnf);
    });

    t.test("token sets", [](testing &t) {
        common_peg_tokens tokens;
        tokens.ids.emplace("<a>", 1);
        tokens.ids.emplace("<b>", 2);
        tokens.tokens.emplace(1, common_peg_token{ "<a>", LLAMA_TOKEN_ATTR_CONTROL });
        tokens.tokens.emplace(2, common_peg_token{ "<b>", LLAMA_TOKEN_ATTR_CONTROL });

        common_peg_parser_builder p(tokens);
        p.set_root(p.token({ "<a>", "<b>" }) + p.token({ "<a>", "<b>" }, true));
        auto parser = p.build();

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= (<[1]> | <[2]>) !<[1-2]>
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )""", gbnf);
    });

    t.test("trigger start choice shares one rest", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            auto start = p.literal("a") + (p.literal("b") | p.literal("d"));
            std::vector<common_peg_trigger> triggers = { { start, p.literal("x") } };
            return p.until(start) + p.trigger_rule("tool", triggers);
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, true);
        });

        assert_gbnf_equal(t, R"""(
            root ::= trigger
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            tool-rest-0 ::= "x"
            trigger ::= | [a] trigger-01 | [^a] trigger
            trigger-01 ::= | [b] tool-rest-0 | [d] tool-rest-0 | [a] trigger-01 | [^abd] trigger
        )""", gbnf);
    });

    t.test("trigger rules reject a shared start", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.trigger_rule("one", { { p.literal("ab"), p.literal("x") } }) |
                   p.trigger_rule("two", { { p.literal("ab"), p.literal("y") } });
        });

        bool threw = false;
        try {
            build_grammar([&](const common_grammar_builder & builder) {
                parser.build_grammar(builder, true);
            });
        } catch (const std::runtime_error &) {
            threw = true;
        }
        t.assert_equal("duplicate_rejected", true, threw);
    });

    t.test("until grammar suffix delimiter offers both rests", [](testing &t) {
        auto parser = build_peg_parser([](common_peg_parser_builder & p)  {
            return p.until(p.eps(), { { p.literal("xab"), p.literal("1") }, { p.literal("ab"), p.literal("2") } });
        });

        auto gbnf = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder);
        });

        assert_gbnf_equal(t, R"""(
            root ::= until-5
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            until-5 ::= [x] until-5-01 | [a] until-5-04 | [^ax] until-5
            until-5-01 ::= [x] until-5-01 | [a] until-5-02 | [^ax] until-5
            until-5-02 ::= [b] (until-5-rest-0 | until-5-rest-1) | [x] until-5-01 | [a] until-5-04 | [^abx] until-5
            until-5-04 ::= [b] until-5-rest-1 | [x] until-5-01 | [a] until-5-04 | [^abx] until-5
            until-5-rest-0 ::= "1"
            until-5-rest-1 ::= "2"
            
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
