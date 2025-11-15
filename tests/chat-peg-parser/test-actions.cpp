#include "tests.h"

void test_actions(testing &t) {
    // Test simple action - append matched text to content
    t.test("simple action - append matched text to content", [](testing &t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
            auto word = p.chars("[a-z]+");
            return p.action(word,
                            [](const common_chat_parse_action & act) { act.env.result.content += std::string(act.match); });
        });

        common_chat_parse_semantics env;
        common_chat_parse_context     ctx("hello", &env);
        auto               result = parser.parse(ctx);

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_is_hello", std::string("hello"), env.result.content);
    });

    // Test multiple sequential actions - build a sentence
    t.test("multiple sequential actions - build a sentence", [](testing &t) {
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

        t.assert_equal("result_is_success", true, result.success());
        t.assert_equal("result_content", std::string("hello Alice"), env.result.content);
        t.assert_equal("captured_name", std::string("Alice"), env.captures["name"]);
    });

    // Test actions don't run when parse fails
    t.test("actions don't run when parse fails", [](testing &t) {
        auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
            return p.action(p.literal("success"),
                            [](const common_chat_parse_action & act) { act.env.result.content = "action_ran"; });
        });

        common_chat_parse_semantics env;
        common_chat_parse_context     ctx("failure", &env);
        auto               result = parser.parse(ctx);

        t.assert_equal("result_is_fail", true, result.fail());
        t.assert_equal("result_content_empty", std::string(""), env.result.content);  // Action should not have run
    });

    // Test Actions work with partial parsing
    t.test("actions work with need_more_input parsing", [](testing &t) {
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

            t.assert_equal("result_is_need_more_input_1", true, result.need_more_input());
            t.assert_equal("result_content_1", std::string("hello "), env.result.content);
        }

        {
            common_chat_parse_semantics env;
            common_chat_parse_context     ctx("<start>hello world", &env, false);
            auto               result = parser.parse(ctx);

            t.assert_equal("result_is_need_more_input_2", true, result.need_more_input());
            t.assert_equal("result_content_2", std::string("hello world"), env.result.content);
        }

        {
            common_chat_parse_semantics env;
            common_chat_parse_context     ctx("<start>hello world<end>", &env, true);
            auto               result = parser.parse(ctx);

            t.assert_equal("result_is_success", true, result.success());
            t.assert_equal("result_content_final", std::string("hello world"), env.result.content);
        }
    });
}
