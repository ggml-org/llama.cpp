#include "common.h"
#include "chat-peg-parser.h"
#include "nlohmann/json.hpp"
#include "tests.h"

#include <numeric>
#include <string>

void test_example_minimax_m2(testing &t) {
    auto helper_parser = build_peg_parser_helper([](common_chat_peg_parser_builder_helper & p) {
        auto thinking = p.reasoning();
        auto content = p.content_before_tools("<minimax:tool_call>");
        auto function = p.quasi_xml_attr("generate_joke",
            std::vector<std::string>({
                "category"
            }));
        auto tool_call = p.rule("tool-call",
            "<minimax:tool_call>" + p.one_or_more(function) + "</minimax:tool_call>", true);

        return thinking + p.optional(p.space() + content) + p.zero_or_more(p.space() + tool_call);
    });


    t.test("minimax_m2_accumulation_test", [&](testing &t) {
        std::string input =
            "<think>"
            "To keep the reply light I’ll fetch a random joke using the `generate_joke` tool."
            "</think>"
            "<minimax:tool_call>"
            "<invoke name=\"generate_joke\">"
            "<parameter name=\"category\">funny</parameter></invoke>"
            "</minimax:tool_call>";

        std::vector<std::string> tokens = simple_tokenize(input);
        // t.log("Tokens: " + string_join(tokens, ", "));

        common_chat_msg prev;
        common_chat_parse_result last_result;
        t.test("helper_builder", [&](testing &t) {
            for (auto it = tokens.begin(); it != tokens.end(); it++) {
                std::string in = std::accumulate(tokens.begin(), it + 1, std::string());
                // t.log("Current input: " + in);

                common_chat_parse_semantics semantics;
                common_chat_parse_context   ctx(in, &semantics, it + 1 == tokens.end());

                common_chat_parse_simple_handler handler;
                ctx.set_event_handler(handler);

                auto result = helper_parser.parse(ctx);
                last_result = result;
                t.assert_equal("not fail", false, result.fail());

                // This shouldn't emit any runtime errors
                auto msg   = semantics.to_msg();
                auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
                prev       = msg;
            }
            // t.log("Last message: " + prev.to_json_oaicompat<nlohmann::ordered_json>().dump());
            t.assert_true("last_result_should_be_success", last_result.success());
        });
    });
}
