#include "tests.h"

class test_partial_parsing : public compound_test {
public:
    test_partial_parsing() : compound_test("test_partial_parsing") {
        // Literals - Basic Success
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("hello");
            });

            parser_context ctx;
            parser_result result;

            ctx = parser_context("hello");
            result = parser.parse(ctx);
            h.assert_equals("literal_success", true, result.is_success());
        }, "literal_success");

        // Char Classes - Basic Lowercase Success
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.one("a-z");
            });

            parser_context ctx;
            parser_result result;

            ctx = parser_context("a");
            result = parser.parse(ctx);
            h.assert_equals("char_class_lowercase_success", true, result.is_success());
        }, "char_class_lowercase_success");

        // Char Classes - Uppercase Fail
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.one("a-z");
            });

            parser_context ctx;
            parser_result result;

            ctx = parser_context("A");
            result = parser.parse(ctx);
            h.assert_equals("char_class_uppercase_fail", true, result.is_fail());
        }, "char_class_uppercase_fail");

        // Char Classes with Dash - Lowercase Success
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.one("a-z-");
            });

            parser_context ctx;
            parser_result result;

            ctx = parser_context("f");
            result = parser.parse(ctx);
            h.assert_equals("char_class_with_dash_lowercase", true, result.is_success());
        }, "char_class_with_dash_lowercase");

        // Char Classes with Dash - Literal Dash Success
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.one("a-z-");
            });

            parser_context ctx;
            parser_result result;

            ctx = parser_context("-");
            result = parser.parse(ctx);
            h.assert_equals("char_class_with_dash_literal_dash", true, result.is_success());
        }, "char_class_with_dash_literal_dash");

        // Char Classes with Dash - Uppercase Fail
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.one("a-z-");
            });

            parser_context ctx;
            parser_result result;

            ctx = parser_context("A");
            result = parser.parse(ctx);
            h.assert_equals("char_class_with_dash_uppercase_fail", true, result.is_fail());
        }, "char_class_with_dash_uppercase_fail");

        // Sequences - Partial Match 1
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("<thi") + p.literal("_end");
            });

            auto ctx = parser_context("thi", false);
            auto result = parser.parse(ctx);
            h.assert_equals("sequence_partial_match_1", true, result.is_need_more_input());
        }, "sequence_partial_match_1");

        // Sequences - Partial Match 2
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("begin") + p.literal("end");
            });

            auto ctx = parser_context("begin", false);
            auto result = parser.parse(ctx);
            h.assert_equals("sequence_partial_match_2", true, result.is_need_more_input());
        }, "sequence_partial_match_2");

        // Sequences - Partial Match 3
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("start") + p.literal("finish");
            });

            auto ctx = parser_context("start/end", false);
            auto result = parser.parse(ctx);
            h.assert_equals("sequence_partial_match_3", true, result.is_need_more_input());
        }, "sequence_partial_match_3");

        // Sequences - Full Match
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("hello") + p.literal("world");
            });

            auto ctx = parser_context("helloworld", true);
            auto result = parser.parse(ctx);
            h.assert_equals("sequence_full_match", true, result.is_success());
        }, "sequence_full_match");

        // Sequences - No Match
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("foo") + p.literal("bar");
            });

            auto ctx = parser_context("foobar", false);
            auto result = parser.parse(ctx);
            h.assert_equals("sequence_no_match", true, result.is_fail());
        }, "sequence_no_match");

        // Choices - Partial Match 1
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("option1") | p.literal("option2");
            });

            auto ctx = parser_context("opt", false);
            auto result = parser.parse(ctx);
            h.assert_equals("choices_partial_match_1", true, result.is_need_more_input());
        }, "choices_partial_match_1");

        // Choices - Partial Match 2
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("choice_a") | p.literal("choice_b");
            });

            auto ctx = parser_context("choice", false);
            auto result = parser.parse(ctx);
            h.assert_equals("choices_partial_match_2", true, result.is_need_more_input());
        }, "choices_partial_match_2");

        // Choices - Full Match 1
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("first") | p.literal("second");
            });

            auto ctx = parser_context("first", true);
            auto result = parser.parse(ctx);
            h.assert_equals("choices_full_match_1", true, result.is_success());
        }, "choices_full_match_1");

        // Choices - Full Match 2
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("alpha") | p.literal("beta");
            });

            auto ctx = parser_context("beta", true);
            auto result = parser.parse(ctx);
            h.assert_equals("choices_full_match_2", true, result.is_success());
        }, "choices_full_match_2");

        // Choices - No Match
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.literal("good") | p.literal("better");
            });

            auto ctx = parser_context("best", true);
            auto result = parser.parse(ctx);
            h.assert_equals("choices_no_match", true, result.is_fail());
        }, "choices_no_match");

        // Zero or More - Partial Match 1
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.zero_or_more(p.literal("ab"));
            });

            auto ctx = parser_context("a", false);
            auto result = parser.parse(ctx);
            h.assert_equals("zero_or_more_partial_match_1", true, result.is_need_more_input());
        }, "zero_or_more_partial_match_1");

        // Zero or More - Partial Match 2
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.zero_or_more(p.literal("xy"));
            });

            auto ctx = parser_context("xyx", false);
            auto result = parser.parse(ctx);
            h.assert_equals("zero_or_more_partial_match_2", true, result.is_need_more_input());
        }, "zero_or_more_partial_match_2");

        // Zero or More - Full Match
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.zero_or_more(p.literal("test"));
            });

            auto ctx = parser_context("test", true);
            auto result = parser.parse(ctx);
            h.assert_equals("zero_or_more_full_match", true, result.is_success());
        }, "zero_or_more_full_match");

        // One or More - Partial Match 1
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.one_or_more(p.literal("repeat"));
            });

            auto ctx = parser_context("rep", false);
            auto result = parser.parse(ctx);
            h.assert_equals("one_or_more_partial_match_1", true, result.is_need_more_input());
        }, "one_or_more_partial_match_1");

        // One or More - Partial Match 2
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.one_or_more(p.literal("again"));
            });

            auto ctx = parser_context("againagain", false);
            auto result = parser.parse(ctx);
            h.assert_equals("one_or_more_partial_match_2", true, result.is_need_more_input());
        }, "one_or_more_partial_match_2");

        // One or More - Full Match
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.one_or_more(p.literal("single"));
            });

            auto ctx = parser_context("single", true);
            auto result = parser.parse(ctx);
            h.assert_equals("one_or_more_full_match", true, result.is_success());
        }, "one_or_more_full_match");

        // One or More - No Match
        add_test([](test_harness h) {
            auto parser = build_parser([](parser_builder& p) {
                return p.one_or_more(p.literal("fail"));
            });

            auto ctx = parser_context("success", true);
            auto result = parser.parse(ctx);
            h.assert_equals("one_or_more_no_match", true, result.is_fail());
        }, "one_or_more_no_match");
    }

    // Provide a convenient way to run all tests
    void run_all_tests() {
        run_all();
        summary();
    }
};