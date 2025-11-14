#include "tests.h"

#include <numeric>
#include <string>

test_example_qwen3_coder::test_example_qwen3_coder() : compound_test("sample_qwen3_coder_test") {
    parser = build_parser([](parser_builder & p) {
        auto thinking = p.add_rule("thinking", "<think>" << p.append_reasoning(p.until("</think>")) << "</think>");

        auto content = p.add_rule("content", p.append_content(p.until("<tool_call>")));

        auto arg_start = p.add_rule("arg-start", p.action("<parameter=", [](const parser_action & act) {
            if (act.env.tool_call_args != "{") {
                act.env.tool_call_args += ",";
            }
            act.env.tool_call_args += "\"";
        }) + p.action(p.chars("[a-zA-Z0-9_]"), [](const parser_action & act) {
            act.env.tool_call_args += std::string(act.match);
        }) + p.action(">", [](const parser_action & act) { act.env.tool_call_args += "\":"; }));

        auto arg_end = p.add_rule("arg-end", "</parameter>");

        auto string_arg = p.add_rule("arg-string", p.action(arg_start, [&](const parser_action & act) {
            act.env.tool_call_args += "\"";
        }) << p.action(p.until("</parameter>"), [&](const parser_action & act) {
            // TODO: add a JSON escape helper
            act.env.tool_call_args += std::string(act.match);
        }) << p.action(arg_end, [&](const parser_action & act) { act.env.tool_call_args += "\""; }));

        auto json = p.json();

        auto json_arg = p.add_rule("arg-json", arg_start << p.action(json, [&](const parser_action & act) {
            // JSON should already be properly formatted
            act.env.tool_call_args += std::string(act.match);

            // This can be streamed by passing p.success(json), but we have
            // to be mindful of the potential backtracking--it only works
            // if we only keep the last value...
        }) << arg_end);

        auto function = p.add_rule(
            "function",
            p.add_tool_call(
                "<function=" + p.capture_tool_call_name(p.chars("[a-zA-Z0-9_]")) +
                    p.action(">", [&](const parser_action & act) { act.env.tool_call_args += "{"; }) +
                    p.one_or_more(p.space() + (json_arg | string_arg))
                << p.action("</function>", [&](const parser_action & act) { act.env.tool_call_args += "}"; })));

        auto tool_call = p.add_rule("tool-call", "<tool_call>" << p.one_or_more(function) << "</tool_call>");

        return thinking + p.optional(p.space() + content) + p.zero_or_more(p.space() + tool_call);
    });

    add_test([this](test_harness h) {
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

            parser_environment env;
            parser_context     ctx(in, &env, it == tokens.end() - 1);

            auto result = parser.parse(ctx);
            h.assert_equals(std::string("should_not_fail_token_") + std::to_string(token_cnt), false, result.is_fail());
            /*
    std::cout << "Input:\n" << in << "\n\n";
    std::cout << "Reasoning: " << prev.reasoning_content << "\n";
    std::cout << "Content  : " << prev.content << "\n";
    if (!prev.tool_calls.empty()) {
        std::cout << "\n=== Tool Calls ===\n";
        for (const auto & tc : prev.tool_calls) {
            std::cout << "ID  : " << tc.id << "\n";
            std::cout << "Name: " << tc.name << "\n";
            std::cout << "Args: " << tc.arguments << "\n";
        }
    }
    */

            // This shouldn't emit any runtime errors
            auto diffs = common_chat_msg_diff::compute_diffs(prev, env.result);
            prev       = env.result;

            /*
    std::cout << "----\n";
    std::cout << "Reasoning: " << prev.reasoning_content << "\n";
    std::cout << "Content  : " << prev.content << "\n";
    if (!prev.tool_calls.empty()) {
        std::cout << "\n=== Tool Calls ===\n";
        for (const auto & tc : prev.tool_calls) {
            std::cout << "ID  : " << tc.id << "\n";
            std::cout << "Name: " << tc.name << "\n";
            std::cout << "Args: " << tc.arguments << "\n";
        }
    }
    std::cout << "======================\n";
    */

            /*
    std::cout << "=== Diffs ===\n\n";
    if (!diffs.empty()) {
        for (size_t i = 0; i < diffs.size(); ++i) {
            const auto& diff = diffs[i];

            std::cout << "Diff #" << (i + 1) << "\n";

            if (!diff.reasoning_content_delta.empty()) {
                std::cout << "  [Reasoning Content]: " << diff.reasoning_content_delta << "\n";
            }

            if (!diff.content_delta.empty()) {
                std::cout << "  [Content]: " << diff.content_delta << "\n";
            }

            if (diff.tool_call_index != std::string::npos) {
                std::cout << "  [Tool Call #" << diff.tool_call_index << "]" << "\n";

                if (!diff.tool_call_delta.id.empty()) {
                    std::cout << "    ID: " << diff.tool_call_delta.id << "\n";
                }

                if (!diff.tool_call_delta.name.empty()) {
                    std::cout << "    Name: " << diff.tool_call_delta.name << "\n";
                }

                if (!diff.tool_call_delta.arguments.empty()) {
                    std::cout << "    Arguments: " << diff.tool_call_delta.arguments << "\n";
                }
            }

            std::cout << "\n";
        }
    } else {
        std::cout << "No changes detected.\n";
    }
    */
        }
    }, "accumulation_test");
}
