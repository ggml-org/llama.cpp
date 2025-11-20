#include "log.h"
#include "tests.h"

#include <numeric>
#include <string>

void test_example_qwen3_coder(testing &t) {
    auto explicit_parser = build_peg_parser([](common_peg_parser_builder & p) {
        auto thinking = p.rule("raw-reasoning",
            "<think>" << p.rule("reasoning-content", p.until("</think>")) << "</think>");

        auto content = p.rule("content", p.until("<tool_call>"));

        auto arg_name = p.rule("arg-start", "<parameter=" + p.capture("arg-name", p.chars("[a-zA-Z0-9_]")) + ">");
        auto arg_end = p.rule("arg-end", "</parameter>" + p.peek(p.literal("<parameter=") | "</function>"));

        auto string_arg_content = p.rule("arg-string-content",
            p.until_one_of({"</parameter><parameter=", "</parameter></function>"}));

        auto string_arg = p.rule("arg-string", "arg-string", arg_name + string_arg_content + arg_end);

        auto json = p.json();

        auto json_arg = p.rule("arg-json", arg_name + p.rule("arg-json-content", json) + arg_end);

        auto function = p.rule("function",
                p.rule("function-start", "<function=" + p.capture("tool-name", p.chars("[a-zA-Z0-9_]")) + ">")
                + p.one_or_more(json_arg | string_arg)
                + "</function>");

        auto tool_call = p.rule("tool-call",
            "<tool_call>" + p.one_or_more(function) + "</tool_call>", true);

        return thinking + p.optional(p.space() + content) + p.zero_or_more(p.space() + tool_call) + p.end();
    });


    auto helper_parser = build_peg_parser_helper([](common_peg_parser_builder_helper & p) {
        auto thinking = p.reasoning();
        auto content = p.content_before_tools("<tool_call>");
        auto function = p.quasi_xml_no_attr("search_files",
            std::vector<std::string>({
                "path", "pattern", "min_size_mb", "max_depth", "include_hidden", "modified_days_ago",
                "case_sensitive", "sort_by", "filters"
            }));
        auto tool_call = p.rule("tool-call",
            "<tool_call>" + p.one_or_more(function) + "</tool_call>", true);

        return thinking + p.optional(p.space() + content) + p.zero_or_more(p.space() + tool_call) + p.end();
    });

    t.test("qwen3_accumulation_test", [&](testing &t) {
        std::string input =
            "<think>The user wants to find large log files that haven't been accessed recently. "
            "I should search for files with .log extension, filter by size (over 100MB), "
            "and check access time within the last 30 days. I'll need to use the search_files function.</think>"
            "Based on your requirements, I'll search for log files over 100MB that haven't been "
            "accessed in the last month. This will help identify candidates for cleanup or archival.\n\n"
            "<tool_call>"
            "<function=search_files>"
            "<parameter=path>/var/log</parameter>"
            "<parameter=pattern>*.log</parameter>"
            "<parameter=min_size_mb>100</parameter>"
            "<parameter=max_depth>5</parameter>"
            "<parameter=include_hidden>false</parameter>"
            "<parameter=modified_days_ago>30</parameter>"
            "<parameter=case_sensitive>true</parameter>"
            "<parameter=sort_by>size</parameter>"
            "<parameter=filters>{\"exclude_patterns\": [\"*temp*\", \"*cache*\"], \"file_types\": "
            "[\"regular\"]}</parameter>"
            "</function>"
            "</tool_call>";

        std::vector<std::string> tokens = simple_tokenize(input);

        t.test("explicit_builder", [&](testing &t) {
            common_chat_msg prev;
            for (auto it = tokens.begin(); it != tokens.end(); it++) {
                std::string in = std::accumulate(tokens.begin(), it + 1, std::string());

                common_peg_parse_semantics semantics;
                common_peg_parse_context   ctx(in, &semantics, it == tokens.end() - 1);

                common_peg_parse_simple_handler handler;
                // handler.log = [&](const std::string & msg) {
                //     t.log(msg);
                // };

                ctx.set_event_handler(handler);

                auto result = explicit_parser.parse(ctx);
                if (!t.assert_equal("not fail", false, result.fail())) {
                    LOG_ERR("%s[failed-->]%s\n", in.substr(0, result.end).c_str(), in.substr(result.end).c_str());
                }

                auto msg = semantics.to_msg();

                try {
                    // This shouldn't emit any runtime errors
                    auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
                } catch(const std::exception & e) {
                    LOG_ERR("%s[failed-->]%s\n", in.substr(0, result.end).c_str(), in.substr(result.end).c_str());
                    t.assert_true(std::string("failed with ") + e.what(), false);
                }

                prev = msg;
            }
        });

        t.test("helper_builder", [&](testing &t) {
            common_chat_msg prev;
            for (auto it = tokens.begin(); it != tokens.end(); it++) {
                std::string in = std::accumulate(tokens.begin(), it + 1, std::string());

                common_peg_parse_semantics semantics;
                common_peg_parse_context   ctx(in, &semantics, it + 1 == tokens.end());

                common_peg_parse_simple_handler handler;
                ctx.set_event_handler(handler);

                auto result = helper_parser.parse(ctx);
                if (!t.assert_equal("not fail", false, result.fail())) {
                    LOG_ERR("%s[failed-->]%s\n", in.substr(0, result.end).c_str(), in.substr(result.end).c_str());
                }

                auto msg = semantics.to_msg();

                try {
                    // This shouldn't emit any runtime errors
                    auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
                } catch(const std::exception & e) {
                    LOG_ERR("%s[failed-->]%s\n", in.substr(0, result.end).c_str(), in.substr(result.end).c_str());
                    t.assert_true(std::string("failed with ") + e.what(), false);
                }

                prev = msg;
            }
        });
    });
}
