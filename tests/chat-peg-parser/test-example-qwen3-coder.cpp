#include "tests.h"

#include <numeric>
#include <string>

void test_example_qwen3_coder(testing &t) {
    auto parser = build_peg_parser([](common_chat_peg_parser_builder & p) {
        auto thinking = p.reasoning();
        auto content = p.content_before_tools("<tool_call>");
        auto function = p.quasi_xml_no_attr("search_files", 
            std::vector<std::string>({ 
                "path", "pattern", "min_size_mb", "max_depth", "include_hidden", "modified_days_ago",
                "case_sensitive", "sort_by", "filters"
            }));
        auto tool_call = p.trigger(p.add_rule("tool-call",
            "<tool_call>" + p.one_or_more(function) + "</tool_call>"));

        return thinking + p.optional(p.space() + content) + p.zero_or_more(p.space() + tool_call);
    });

    auto handler = [&](const common_chat_parse_event & ev, common_chat_parse_semantics & env) {
        if (ev.rule == "reasoning-content" && ev.ending()) {
            env.result.reasoning_content = ev.text;
        }

        if (ev.rule == "content" && ev.ending()) {
            env.result.content = ev.text;
        }

        if (ev.rule == "function-start" && ev.ending() && ev.success()) {
            env.result.tool_calls.emplace_back();
            auto & tc = env.result.tool_calls.back();
            tc.name = env.captures["tool-name"];
        }

        if (ev.rule == "arg-start" && ev.ending() && ev.success()) {
            auto & tc = env.result.tool_calls.back();
            auto name = env.captures["arg-name"];
            if (tc.arguments.empty()) {
                tc.arguments += "{";
            } else {
                tc.arguments += ", ";
            }
            tc.arguments += "\"" + name + "\": ";
        }

        if (ev.rule == "arg-string-content" && ev.ending() && ev.success()) {
            auto & tc = env.result.tool_calls.back();
            tc.arguments += "\"" + std::string(ev.text);
        }

        if (ev.rule == "arg-string" && ev.ending() && ev.success()) {
            auto & tc = env.result.tool_calls.back();
            tc.arguments += "\"";
        }

        if (ev.rule == "arg-json-content" && ev.ending() && (ev.success() || ev.need_more_input())) {
            auto & tc = env.result.tool_calls.back();
            tc.arguments += std::string(ev.text);
        }
    };

    t.test("accumulation_test", [&](testing &t) {
        std::string input =
            "<think>The user wants to find large log files that haven't been accessed recently. "
            "I should search for files with .log extension, filter by size (over 100MB), "
            "and check access time within the last 30 days. I'll need to use the search_files function.</think>"
            "Based on your requirements, I'll search for log files over 100MB that haven't been "
            "accessed in the last month. This will help identify candidates for cleanup or archival.\n\n"
            "<tool_call>\n"
            "<function=search_files>\n"
            "<parameter=path>/var/log</parameter>\n"
            "<parameter=pattern>*.log</parameter>\n"
            "<parameter=min_size_mb>100</parameter>\n"
            "<parameter=max_depth>5</parameter>\n"
            "<parameter=include_hidden>false</parameter>\n"
            "<parameter=modified_days_ago>30</parameter>\n"
            "<parameter=case_sensitive>true</parameter>\n"
            "<parameter=sort_by>size</parameter>\n"
            "<parameter=filters>{\"exclude_patterns\": [\"*temp*\", \"*cache*\"], \"file_types\": "
            "[\"regular\"]}</parameter>\n"
            "</function>\n"
            "</tool_call>";

        std::vector<std::string> tokens = simple_tokenize(input);

        common_chat_msg prev;
        int token_cnt = 0;
        for (auto it = tokens.begin(); it != tokens.end(); it++) {
            token_cnt++;
            std::string in = std::accumulate(tokens.begin(), it, std::string());

            common_chat_parse_semantics env;
            common_chat_parse_context   ctx(in, &env, it == tokens.end() - 1);

            ctx.event_handler = handler;

            auto result = parser.parse(ctx);
            t.assert_equal(std::string("should_not_fail_token_") + std::to_string(token_cnt), false, result.fail());

            // This shouldn't emit any runtime errors
            auto diffs = common_chat_msg_diff::compute_diffs(prev, env.result);
            prev       = env.result;
        }
    });
}
