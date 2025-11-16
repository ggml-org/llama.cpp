#include "chat-peg-parser.h"
#include "nlohmann/json.hpp"
#include "tests.h"

#include <numeric>
#include <string>

void test_example_minimax_m2(testing &t) {
    auto helper_parser = build_peg_parser_helper([](common_chat_peg_parser_builder_helper & p) {
        auto thinking = p.reasoning();
        auto content = p.content_before_tools("<minimax:tool_call>");
        auto function = p.quasi_xml_attr("get_weather",
            std::vector<std::string>({
                "location", "units"
            }));
        auto tool_call = p.trigger(p.add_rule("tool-call",
            "<minimax:tool_call>" + p.one_or_more(function) + "</seed:minimax>"));

        return thinking + p.optional(p.space() + content) + p.zero_or_more(p.space() + tool_call);
    });


    t.test("minimax_m2_accumulation_test", [&](testing &t) {
        std::string input =
            "<think>"
            "To keep the reply light I’ll fetch a random joke using the `generate_joke` tool."
            "</think>"
            ""
            ""
            "<minimax:tool_call>"
            "<invoke name=\"generate_joke\">"
            "<parameter name=\"category\">funny</parameter></invoke>"
            "</minimax:tool_call>";

        std::vector<std::string> tokens = simple_tokenize(input);

        common_chat_msg prev;
        common_chat_parse_result last_result;
        t.test("helper_builder", [&](testing &t) {
            size_t token_cnt = 0;
            for (auto it = tokens.begin(); it != tokens.end(); it++) {
                token_cnt++;
                std::string in = std::accumulate(tokens.begin(), it, std::string());

                common_chat_parse_semantics semantics;
                common_chat_parse_context   ctx(in, &semantics, it == tokens.end() - 1);

                ctx.event_handler = parser_semantic_handler;

                auto result = helper_parser.parse(ctx);
                if (result.fail()) {
                    break;
                }

                last_result = result;
                // This shouldn't emit any runtime errors
                auto msg   = semantics.to_msg();
                auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
                prev       = msg;
            }
            t.assert_true("last_result_should_be_success", last_result.success());

            std::cout << "Final message:\n" << prev.to_json_oaicompat<nlohmann::ordered_json>().dump();

            t.assert_equal("should_parse_all_tokens_helper", tokens.size(), token_cnt);
        });
    });
}
