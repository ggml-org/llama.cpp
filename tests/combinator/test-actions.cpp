#include "tests.h"

test_actions::test_actions() : compound_test("test_actions") {
    // Test simple action - append matched text to content
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) {
                auto word = p.chars("[a-z]+");
                return p.action(word,
                                [](const parser_action & act) { act.env.result.content += std::string(act.match); });
            });

            parser_environment env;
            parser_context     ctx("hello", &env);
            auto               result = parser.parse(ctx);

            h.assert_equals("result_is_success", true, result.is_success());
            h.assert_equals("result_is_hello", std::string("hello"), env.result.content);
        },
        "simple action - append matched text to content");

    // Test multiple sequential actions - build a sentence
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) {
                auto greeting = p.action(p.literal("hello"), [](const parser_action & act) {
                    act.env.result.content += std::string(act.match) + " ";
                });

                auto name = p.action(p.chars("[A-Z][a-z]+"), [](const parser_action & act) {
                    act.env.result.content += std::string(act.match);
                    act.env.scratchpad["name"] = std::string(act.match);
                });

                return greeting + p.literal(" ") + name;
            });

            parser_environment env;
            parser_context     ctx("hello Alice", &env);
            auto               result = parser.parse(ctx);

            h.assert_equals("result_is_success", true, result.is_success());
            h.assert_equals("result_content", std::string("hello Alice"), env.result.content);
            h.assert_equals("scratchpad_name", std::string("Alice"), std::get<std::string>(env.scratchpad["name"]));
        },
        "multiple sequential actions - build a sentence");

    // Test using scratchpad for intermediate calculations
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) {
                auto digit = p.action(p.one("[0-9]"), [](const parser_action & act) {
                    auto it          = act.env.scratchpad.find("sum");
                    int  current_sum = it != act.env.scratchpad.end() ? std::get<int>(it->second) : 0;
                    current_sum += (act.match[0] - '0');
                    act.env.scratchpad["sum"] = current_sum;
                });

                return p.one_or_more(digit + p.optional(p.literal("+")));
            });

            parser_environment env;
            parser_context     ctx("1+2+3+4", &env);
            auto               result = parser.parse(ctx);

            h.assert_equals("result_is_success", true, result.is_success());
            h.assert_equals("scratchpad_sum", 10, std::get<int>(env.scratchpad["sum"]));  // 1+2+3+4 = 10
        },
        "using scratchpad for intermediate calculations");

    // Test actions don't run when parse fails
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) {
                return p.action(p.literal("success"),
                                [](const parser_action & act) { act.env.result.content = "action_ran"; });
            });

            parser_environment env;
            parser_context     ctx("failure", &env);
            auto               result = parser.parse(ctx);

            h.assert_equals("result_is_fail", true, result.is_fail());
            h.assert_equals("result_content_empty", std::string(""), env.result.content);  // Action should not have run
        },
        "actions don't run when parse fails");

    // Test Actions work with partial parsing
    add_test(
        [](test_harness h) {
            auto parser = build_parser([](parser_builder & p) {
                auto content = p.action(p.until("<end>"), [](const parser_action & act) {
                    act.env.result.content += std::string(act.match);
                });
                return "<start>" << content << "<end>";
            });

            {
                parser_environment env;
                parser_context     ctx("<start>hello ", &env, false);
                auto               result = parser.parse(ctx);

                h.assert_equals("result_is_need_more_input_1", true, result.is_need_more_input());
                h.assert_equals("result_content_1", std::string("hello "), env.result.content);
            }

            {
                parser_environment env;
                parser_context     ctx("<start>hello world", &env, false);
                auto               result = parser.parse(ctx);

                h.assert_equals("result_is_need_more_input_2", true, result.is_need_more_input());
                h.assert_equals("result_content_2", std::string("hello world"), env.result.content);
            }

            {
                parser_environment env;
                parser_context     ctx("<start>hello world<end>", &env, true);
                auto               result = parser.parse(ctx);

                h.assert_equals("result_is_success", true, result.is_success());
                h.assert_equals("result_content_final", std::string("hello world"), env.result.content);
            }
        },
        "actions work with partial parsing");
}
