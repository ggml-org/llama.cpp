#include "test_harness.h"

#include "peg-parser.h"
#include "chat-parser.h"
#include "json-schema-to-grammar.h"

#include <iostream>
#include <type_traits>
#include <vector>
#include <string>
#include <numeric>

static common_peg_arena create_command_r7b_parser() {
    auto parser = build_peg_parser([](common_peg_parser_builder & p) {
        auto thinking = p.rule("thinking",
            "<|START_THINKING|>" << p.rule("reasoning-content", p.until("<|END_THINKING|>")) << "<|END_THINKING|>");

        auto response = p.rule("response",
            "<|START_RESPONSE|>" << p.rule("content", p.until("<|END_RESPONSE|>")) << "<|END_RESPONSE|>");

        auto json = p.rule("json", p.json());

        auto tool_call_id = p.rule("tool-call-id",
            "\"tool_call_id\"" << (":" << p.rule("tool-call-id-value", "\"" + p.json_string_content() + "\"")));

        auto tool_call_name = p.rule("tool-name",
            "\"tool_name\"" << (":" << p.rule("tool-name-value", "\"" + p.json_string_content() + "\"")));

        auto tool_call_args = p.rule("tool-args",
            "\"parameters\"" << (":" << p.rule("tool-args-value", json)));

        auto tool_call_fields = p.rule("tool-call-fields", tool_call_id | tool_call_name | tool_call_args);

        auto tool_call = p.rule("tool-call",
            "{" << tool_call_fields << p.zero_or_more(p.literal(",") << tool_call_fields) << "}");

        auto tool_calls = p.rule("tool-calls",
            "<|START_ACTION|>"
            << ("[" << tool_call << p.zero_or_more(p.literal(",") << tool_call) << "]")
            << "<|END_ACTION|>");

        return p.optional(thinking) << (tool_calls | response);
    });

    // Check if
    build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });
    return parser;
}

static void test_command_r7b_parser(const common_peg_arena & p,
                           const std::string &  input,
                           bool                 need_more_input,
                           bool                 /* print_results */) {
    common_peg_parse_context   ctx(input, !need_more_input);
    p.parse(ctx);
}

static void test_command_r7b_legacy_parser(const std::string & input,
                                  bool                need_more_input,
                                  bool                print_results) {
    // Original common_chat_combinator_parser taken from chat.cpp
    common_chat_msg_parser builder(input,
                                   /* .is_partial = */ need_more_input,
                                   {
                                       /* .format = */ COMMON_CHAT_FORMAT_GENERIC,
                                       /* .reasoning_format = */ COMMON_REASONING_FORMAT_AUTO,
                                       /* .reasoning_in_content = */ false,
                                       /* .thinking_forced_open = */ false,
                                   });

    builder.try_parse_reasoning("<|START_THINKING|>", "<|END_THINKING|>");

    static const common_regex start_action_regex("<\\|START_ACTION\\|>");
    static const common_regex end_action_regex("<\\|END_ACTION\\|>");
    static const common_regex start_response_regex("<\\|START_RESPONSE\\|>");
    static const common_regex end_response_regex("<\\|END_RESPONSE\\|>");

    if (auto res = builder.try_find_regex(start_action_regex)) {
        // If we didn't extract thoughts, prelude includes them.
        auto tool_calls = builder.consume_json_with_dumped_args({ { "parameters" } });
        for (const auto & tool_call : tool_calls.value) {
            std::string name      = tool_call.contains("tool_name") ? tool_call.at("tool_name") : "";
            std::string id        = tool_call.contains("tool_call_id") ? tool_call.at("tool_call_id") : "";
            std::string arguments = tool_call.contains("parameters") ? tool_call.at("parameters") : "";
            if (!builder.add_tool_call(name, id, arguments) || tool_calls.is_partial) {
                throw common_chat_msg_partial_exception("incomplete tool call");
            }
        }
        if (tool_calls.is_partial) {
            throw common_chat_msg_partial_exception("incomplete tool call");
        }
        builder.consume_regex(end_action_regex);
    } else if (auto res = builder.try_find_regex(start_response_regex)) {
        if (!builder.try_find_regex(end_response_regex)) {
            builder.add_content(builder.consume_rest());
            throw common_chat_msg_partial_exception(end_response_regex.str());
        }
    } else {
        builder.add_content(builder.consume_rest());
    }

    if (print_results) {
        std::cout << "== Parsed (legacy) ==\n";
        std::cout << "=== Reasoning ===\n";
        std::cout << builder.result().reasoning_content << "\n";
        std::cout << "\n\n=== Content ===\n";
        std::cout << builder.result().content << "\n";
        std::cout << "\n\n=== Tool Calls ===\n";
        for (const auto & tc : builder.result().tool_calls) {
            std::cout << "id: " << tc.id << "\n";
            std::cout << "name: " << tc.name << "\n";
            std::cout << "args: " << tc.arguments << "\n";
        }
    }
}

static std::vector<std::string> simple_tokenize(const std::string & input) {
    std::vector<std::string> result;
    std::string              current;

    for (size_t i = 0; i < input.size(); i++) {
        switch (input[i]) {
            case ' ':
            case '\n':
            case '\t':
            case '{':
            case '}':
            case ',':
            case '[':
            case '"':
            case ']':
            case '.':
            case '<':
            case '>':
            case '=':
            case '/':
                if (!current.empty()) {
                    result.push_back(current);
                    current.clear();
                }
            default:;
        }
        current += input[i];
    }

    if (!current.empty()) {
        result.push_back(current);
    }

    return result;
}

static void print_ast(const common_peg_ast_arena & arena, common_peg_ast_id id, int indent = 0) {
    const auto & node = arena.get(id);

    // Indentation
    std::string space(indent * 2, ' ');

    // Print Node details
    std::cout << space << "Node [" << id << "] "
              << (node.rule_name.empty() ? "<anon>" : node.rule_name);

    if (node.is_partial) {
        std::cout << "*";
    }

    if (!node.tag.empty()) {
        std::cout << " (" << node.tag << ")";
    }

    // Print text content (truncated if too long for readability)
    std::string_view text = node.text;
    if (text.length() > 20) {
        std::cout << " : \"" << text.substr(0, 17) << "...\"";
    } else {
        std::cout << " : \"" << text << "\"";
    }
    std::cout << "\n";

    // Recursively print children
    for (auto child_id : node.children) {
        print_ast(arena, child_id, indent + 1);
    }
}

int main() {
    testing t;

    auto explicit_parser = build_peg_parser([](common_peg_parser_builder & p) {
        auto thinking = p.tag("raw-reasoning",
            "<think>" << p.tag("reasoning-content", p.until("</think>")) << "</think>");

        auto content = p.tag("content", p.until("<tool_call>"));

        auto arg_open = p.rule("arg-open",
            p.atomic("<parameter=" + p.tag("arg-name", p.chars("[a-zA-Z0-9_]")) + ">")
        );

        auto arg_close = p.rule(
            "arg-close",
            "</parameter>" +
            p.peek(p.literal("<parameter=") | p.literal("</function>"))
        );

        auto string_arg =
            arg_open +
            p.tag("arg-string", p.until_one_of({"</parameter><parameter=", "</parameter></function>"})) +
            p.atomic(p.tag("arg-string-close", arg_close));

        auto json = p.json();

        auto json_arg = arg_open + p.tag("arg-json",json) + arg_close;

        auto function = p.rule("tool",
            p.tag("tool-open", p.atomic("<function=" + p.tag("tool-name", p.chars("[a-zA-Z0-9_]")) + ">")) +
            p.one_or_more(json_arg | string_arg) +
            p.tag("tool-close", p.literal("</function>"))
        );

        auto tool_call = p.rule("tool-call",
            "<tool_call>" +
            p.one_or_more(function) +
            "</tool_call>",
            true);

        return thinking + p.optional(p.space() + content) + p.zero_or_more(p.space() + tool_call) + p.end();
    });

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

    t.test("parse succeeds", [&](testing &t) {
        common_chat_msg prev;
        for (auto it = tokens.begin(); it != tokens.end(); it++) {
            std::string in = std::accumulate(tokens.begin(), it + 1, std::string());

            common_peg_parse_context ctx(in, it == tokens.end() - 1);

            auto result = explicit_parser.parse(ctx);
            if (!t.assert_equal("not fail", false, result.fail())) {
                t.log(in.substr(0, result.end) + "[failed->]" + in.substr(result.end));
            }

            common_chat_msg msg;
            common_chat_tool_call *current;
            int arg_count = 0;

            ctx.ast_arena.visit(result, [&](const common_peg_ast_node & node) {
                bool is_reasoning = node.tag == "reasoning-content";
                bool is_content = node.tag == "content";
                bool is_tool_name = node.tag == "tool-name";
                bool is_tool_close = node.tag == "tool-close";
                bool is_arg_name = node.tag == "arg-name";
                bool is_arg_string = node.tag == "arg-string";
                bool is_arg_string_close = node.tag == "arg-string-close";
                bool is_arg_json = node.tag == "arg-json";

                if (is_reasoning) {
                    msg.reasoning_content = std::string(node.text);
                }

                if (is_content) {
                    msg.content = std::string(node.text);
                }

                if (is_tool_name) {
                    msg.tool_calls.emplace_back();
                    current = &msg.tool_calls.back();
                    arg_count = 0;

                    current->name = std::string(node.text);
                    current->arguments = "{";
                }

                if (is_arg_name) {
                    if (arg_count > 0) {
                        current->arguments += ",";
                    }
                    current->arguments += "\"" + std::string(node.text) + "\":";
                    ++arg_count;
                }

                if (is_arg_string) {
                    current->arguments += "\"" + std::string(node.text);
                }

                if (is_arg_string_close) {
                    current->arguments += "\"";
                }

                if (is_arg_json) {
                    current->arguments += std::string(node.text);
                }

                if (is_tool_close) {
                    current->arguments += "}";
                }
            });

            //t.log("Input: " + input);
            t.log("Reasoning: " + msg.reasoning_content);
            t.log("Content  : " + msg.content);
            for (const auto & tc : msg.tool_calls) {
                t.log("Tool name: " + tc.name);
                t.log("Tool args: " + tc.arguments);
            }

            try {
                // This shouldn't emit any runtime errors
                auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
            } catch(const std::exception & e) {
                t.log(in.substr(0, result.end) + "[failed->]" + in.substr(result.end));
                t.assert_true(std::string("failed with ") + e.what(), false);
            }

            prev = msg;
        }
    });

    // Setup data
    auto parser = create_command_r7b_parser();

    std::string reasoning = "To plan an effective trip to Japan that includes both historical sites and modern attractions within a "
            "budget of $4000 for a two-week stay, we need to:\n\n"
            "1. Identify key historical sites and modern attractions in Japan.\n"
            "2. Find affordable accommodation options that provide a balance between comfort and cost.\n"
            "3. Determine the best modes of transportation for getting around Japan.\n"
            "4. Create a day-by-day itinerary that ensures the user gets to see a variety of attractions without "
            "overspending.\n"
            "5. Provide a detailed cost breakdown that includes accommodation, transportation, meals, and entry fees "
            "to attractions.";

    std::string content = "For a two-week trip to Japan with a $4,000 budget, I recommend planning an itinerary that balances "
            "historical sites with modern attractions. The destination will be Japan, with a duration of 14 days.\n\n"
            "Given your interests in both historical sites and modern attractions, you'll want to focus on cities like "
            "Kyoto for its temples and traditional culture, Tokyo for its cutting-edge technology and entertainment "
            "districts, and possibly Hiroshima or Nara for additional historical significance.\n\n"
            "For accommodation, I suggest looking for affordable options such as budget hotels, hostels, or "
            "guesthouses that offer good value without sacrificing too much comfort. Japan has excellent mid-range "
            "accommodation options that can keep your lodging costs manageable.\n\n"
            "Transportation should prioritize efficiency—consider getting a JR Rail Pass for intercity travel, which "
            "allows unlimited rides on most JR trains including the Shinkansen (bullet train). Within cities, use "
            "local trains and subways, which are both affordable and highly reliable.\n\n"
            "For meals, embrace local cuisine by eating at neighborhood restaurants, ramen shops, and izakayas rather "
            "than touristy establishments. This will give you an authentic experience while keeping costs "
            "reasonable—you can enjoy excellent meals for $10-20 per person at local spots.\n\n";

    std::vector<std::tuple<std::string, std::string, nlohmann::json>> tool_calls = {
        { "call_0", "plan_trip", nlohmann::json::parse(R"({
            "destination": "Japan",
            "duration": 14,
            "budget": 4000,
            "interests": ["historical sites", "modern attractions"],
            "accommodation_preferences": "affordable",
            "transportation_preferences": "efficient",
            "meal_preferences": "local cuisine"
        })") }
    };

    tokens.clear();

    // Build tokens
    if (!reasoning.empty()) {
        auto tokenized = simple_tokenize(reasoning);
        tokens.emplace_back("<|START_THINKING|>");
        tokens.insert(tokens.end(), tokenized.begin(), tokenized.end());
        tokens.emplace_back("<|END_THINKING|>");
    }

    if (!content.empty()) {
        auto tokenized = simple_tokenize(content);
        tokens.emplace_back("<|START_RESPONSE|>");
        tokens.insert(tokens.end(), tokenized.begin(), tokenized.end());
        tokens.emplace_back("<|END_RESPONSE|>");
    }

    if (!tool_calls.empty()) {
        tokens.emplace_back("<|START_ACTION|>");

        auto json = nlohmann::json::array();
        for (const auto & tc : tool_calls) {
            auto tc_json            = nlohmann::json::object();
            tc_json["tool_call_id"] = std::get<0>(tc);
            tc_json["tool_name"]    = std::get<1>(tc);
            tc_json["parameters"]   = std::get<2>(tc);
            json.push_back(tc_json);
        }

        auto tokenized = simple_tokenize(json.dump(-1, ' ', true));
        tokens.insert(tokens.end(), tokenized.begin(), tokenized.end());

        tokens.emplace_back("<|END_ACTION|>");
    }

    input = std::accumulate(tokens.begin(), tokens.end(), std::string());

    // Run tests
    t.test("legacy_parse", [&](testing & t) {
        test_command_r7b_legacy_parser(input, false, false);
    });

    t.test("current_parse", [&](testing & t) {
        test_command_r7b_parser(parser, input, false, false);
    });

    // Run benchmarks
    t.bench("legacy_parse_benchmark", [&]() {
        test_command_r7b_legacy_parser(input, false, false);
    }, 100);

    t.bench("current_parse_benchmark", [&]() {
        test_command_r7b_parser(parser, input, false, false);
    }, 100);

    return t.summary();
}
