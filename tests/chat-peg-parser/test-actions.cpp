#include "tests.h"

test_actions::test_actions() : compound_test("test_actions") {
    // Test simple action - append matched text to content
    add_test(
        [](test_harness h) {
            auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
                auto word = p.chars("[a-z]+");
                return p.action(word,
                                [](const common_chat_parse_action & act) { act.env.result.content += std::string(act.match); });
            });

            common_chat_parse_semantics env;
            common_chat_parse_context     ctx("hello", &env);
            auto               result = parser.parse(ctx);

            h.assert_equals("result_is_success", true, result.success());
            h.assert_equals("result_is_hello", std::string("hello"), env.result.content);
        },
        "simple action - append matched text to content");

    // Test multiple sequential actions - build a sentence
    add_test(
        [](test_harness h) {
            auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
                auto greeting = p.action(p.literal("hello"), [](const common_chat_parse_action & act) {
                    act.env.result.content += std::string(act.match) + " ";
                });

                auto name = p.action(p.chars("[A-Z][a-z]+"), [](const common_chat_parse_action & act) {
                    act.env.result.content += std::string(act.match);
                    act.env.captures["name"] = std::string(act.match);
                });

                return greeting + p.literal(" ") + name;
            });

            common_chat_parse_semantics env;
            common_chat_parse_context     ctx("hello Alice", &env);
            auto               result = parser.parse(ctx);

            h.assert_equals("result_is_success", true, result.success());
            h.assert_equals("result_content", std::string("hello Alice"), env.result.content);
            h.assert_equals("captured_name", std::string("Alice"), env.captures["name"]);
        },
        "multiple sequential actions - build a sentence");

    // Test actions don't run when parse fails
    add_test(
        [](test_harness h) {
            auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
                return p.action(p.literal("success"),
                                [](const common_chat_parse_action & act) { act.env.result.content = "action_ran"; });
            });

            common_chat_parse_semantics env;
            common_chat_parse_context     ctx("failure", &env);
            auto               result = parser.parse(ctx);

            h.assert_equals("result_is_fail", true, result.fail());
            h.assert_equals("result_content_empty", std::string(""), env.result.content);  // Action should not have run
        },
        "actions don't run when parse fails");

    // Test Actions work with partial parsing
    add_test(
        [](test_harness h) {
            auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
                auto content = p.action(p.until("<end>"), [](const common_chat_parse_action & act) {
                    act.env.result.content += std::string(act.match);
                });
                return "<start>" << content << "<end>";
            });

            {
                common_chat_parse_semantics env;
                common_chat_parse_context     ctx("<start>hello ", &env, false);
                auto               result = parser.parse(ctx);

                h.assert_equals("result_is_need_more_input_1", true, result.need_more_input());
                h.assert_equals("result_content_1", std::string("hello "), env.result.content);
            }

            {
                common_chat_parse_semantics env;
                common_chat_parse_context     ctx("<start>hello world", &env, false);
                auto               result = parser.parse(ctx);

                h.assert_equals("result_is_need_more_input_2", true, result.need_more_input());
                h.assert_equals("result_content_2", std::string("hello world"), env.result.content);
            }

            {
                common_chat_parse_semantics env;
                common_chat_parse_context     ctx("<start>hello world<end>", &env, true);
                auto               result = parser.parse(ctx);

                h.assert_equals("result_is_success", true, result.success());
                h.assert_equals("result_content_final", std::string("hello world"), env.result.content);
            }
        },
        "actions work with need_more_input parsing");
}
