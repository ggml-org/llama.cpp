#include "tests.h"

#include <numeric>
#include <string>

void test_example_seed_oss(testing &t) {
    auto helper_parser = build_peg_parser_helper([](common_chat_peg_parser_builder_helper & p) {
        auto thinking = p.reasoning("seed:think");
        auto content = p.content_before_tools("<seed:tool_call>");
        auto function = p.quasi_xml_no_attr("get_weather",
            std::vector<std::string>({
                "location", "units"
            }));
        auto tool_call = p.trigger(p.add_rule("tool-call",
            "<seed:tool_call>" + p.one_or_more(function) + "</seed:tool_call>"));

        return thinking + p.optional(p.space() + content) + p.zero_or_more(p.space() + tool_call);
    });


    t.test("seed_oss_accumulation_test", [&](testing &t) {
        std::string input =
            "<seed:think>Next I need the current weather for Berlin. I'll call the `get_weather` tool.</seed:think><seed:bos>assistant"
            "<seed:tool_call>"
            "<function=get_weather>"
            "<parameter=location>Berlin</parameter>"
            "<parameter=units>metric</parameter>"
            "</function>"
            "</seed:tool_call>";
        std::vector<std::string> tokens = simple_tokenize(input);

        common_chat_msg prev;
        t.test("helper_builder", [&](testing &t) {
            for (auto it = tokens.begin(); it != tokens.end(); it++) {
                std::string in = std::accumulate(tokens.begin(), it + 1, std::string());

                common_chat_parse_semantics semantics;
                common_chat_parse_context   ctx(in, &semantics, it == tokens.end() - 1);

                ctx.event_handler = parser_semantic_handler;

                auto result = helper_parser.parse(ctx);
                t.assert_equal("not fail", false, result.fail());

                // This shouldn't emit any runtime errors
                auto msg   = semantics.to_msg();
                auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
                prev       = msg;
            }
        });
    });
}
