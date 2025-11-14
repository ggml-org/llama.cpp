#include "chat-parser.h"
#include "json-schema-to-grammar.h"
#include "tests.h"
#include <iostream>
#include <memory>

class parser test_command7_parser_compare::create_command_r7b_parser() {
    auto parser  = build_parser([](parser_builder & p) {
        auto thinking = p.add_rule(
            "thinking", "<|START_THINKING|>" << p.append_reasoning(p.until("<|END_THINKING|>")) << "<|END_THINKING|>");

        auto response = p.add_rule(
            "response", "<|START_RESPONSE|>" << p.append_content(p.until("<|END_RESPONSE|>")) << "<|END_RESPONSE|>");

        auto json = p.add_rule("json", p.json());

        auto tool_call_id =
            p.add_rule("tool-call-id",
                        p.json_key("tool_call_id",
                                   "\"" + p.capture_tool_call_id(p.json_string(), /* unescape_json = */ true) + "\""));

        auto tool_call_name =
            p.add_rule("tool-name",
                        p.json_key("tool_name",
                                   "\"" + p.capture_tool_call_name(p.json_string(), /* unescape_json = */ true) + "\""));

        auto tool_call_args = p.add_rule("tool-args", p.json_key("parameters", p.capture_tool_call_args(json)));

        auto tool_call_fields = p.add_rule("tool-call-fields", tool_call_id | tool_call_name | tool_call_args);

        auto tool_call = p.add_rule(
            "tool-call",
            "{" << p.add_tool_call(tool_call_fields << p.zero_or_more(p.literal(",") << tool_call_fields)) << "}");

        auto tool_calls = p.add_rule(
            "tool-calls", "<|START_ACTION|>" << ("[" << tool_call << p.zero_or_more(p.literal(",") << tool_call) << "]")
                                             << "<|END_ACTION|>");

        return p.optional(thinking) << p.add_rule("content", tool_calls | response);
    });
    
    // Check if
    build_grammar([&](const common_grammar_builder & builder) { parser.build_grammar(builder); });
    return parser;
}
    
// command7_parser_compare_test implementation
test_command7_parser_compare::test_command7_parser_compare() :
    benchmark_test(std::vector<std::unique_ptr<test_case>>()), 
    parser(create_command_r7b_parser()),
    reasoning("To plan an effective trip to Japan that includes both historical sites and modern attractions within a "
            "budget of $4000 for a two-week stay, we need to:\n\n"
            "1. Identify key historical sites and modern attractions in Japan.\n"
            "2. Find affordable accommodation options that provide a balance between comfort and cost.\n"
            "3. Determine the best modes of transportation for getting around Japan.\n"
            "4. Create a day-by-day itinerary that ensures the user gets to see a variety of attractions without "
            "overspending.\n"
            "5. Provide a detailed cost breakdown that includes accommodation, transportation, meals, and entry fees "
            "to attractions."),
    content("For a two-week trip to Japan with a $4,000 budget, I recommend planning an itinerary that balances "
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
            "reasonable—you can enjoy excellent meals for $10-20 per person at local spots.\n\n"),
    tool_calls({
            { "call_0", "plan_trip", nlohmann::json::parse(R"({
            "destination": "Japan",
            "duration": 14,
            "budget": 4000,
            "interests": ["historical sites", "modern attractions"],
            "accommodation_preferences": "affordable",
            "transportation_preferences": "efficient",
            "meal_preferences": "local cuisine"
        })") }
        })
    {
    // Build response
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
            tc_json["tool_call_id"] = tc.id;
            tc_json["tool_name"]    = tc.name;
            tc_json["parameters"]   = tc.args;
            json.push_back(tc_json);
        }

        auto tokenized = simple_tokenize(json.dump(-1, ' ', true));
        tokens.insert(tokens.end(), tokenized.begin(), tokenized.end());

        tokens.emplace_back("<|END_ACTION|>");
    }

    test_case legacy = test_case([this](test_harness h) {
        bool no_error = true;
        try {
            std::string input = std::accumulate(tokens.begin(), tokens.end(), std::string());
            test_command_r7b_legacy_parser(input, false, false);
        } catch (std::exception &e) {
            no_error = false;
            std::cerr << "Error during legacy run: " << e.what() << "\n";
        }
        h.assert_equals("no_errors", true, no_error);
    }, "legacy_parse");

    test_case current = test_case([this](test_harness h) {
        bool no_error = true;
        try {
            std::string input = std::accumulate(tokens.begin(), tokens.end(), std::string());
            test_command_r7b_parser(parser, input, false, false);
        } catch (std::exception &e) {
            no_error = false;
            std::cerr << "Error during legacy run: " << e.what() << "\n";
        }
        h.assert_equals("no_errors", true, no_error);
    }, "current_parse");
    legacy.set_omit_success_msg(true);
    current.set_omit_success_msg(true);

    cases.push_back(std::make_unique<test_case>(legacy));
    cases.push_back(std::make_unique<test_case>(current));
}

void test_command7_parser_compare::run_comparison(int iterations) {
    long long t1 = run_benchmark(0, iterations);
    long long t2 = run_benchmark(1, iterations);

    std::cout << "=== Command7 parser comparison benchmark (" << iterations << " iterations) ===\n";
    std::cout << "Legacy  parser performance: " << t1 << "ms (" << (float) t1 / iterations << "ms per iteration)\n";
    std::cout << "Current parser performance: " << t2 << "ms (" << (float) t2 / iterations << "ms per iteration)\n";
}

void test_command7_parser_compare::test_command_r7b_parser(const class parser & p,
                                                           const std::string &  input,
                                                           bool                 partial,
                                                           bool                 print_results) {
    parser_environment env;
    parser_context     ctx(input, &env, !partial);
    p.parse(ctx);

    if (print_results) {
        std::cout << "== Parsed (new) ==\n";
        std::cout << "=== Reasoning ===\n";
        std::cout << env.result.reasoning_content << "\n";
        std::cout << "\n\n=== Content ===\n";
        std::cout << env.result.content << "\n";
        std::cout << "\n\n=== Tool Calls ===\n";
        for (const auto & tc : env.result.tool_calls) {
            std::cout << "id: " << tc.id << "\n";
            std::cout << "name: " << tc.name << "\n";
            std::cout << "args: " << tc.arguments << "\n";
        }
    }
}

void test_command7_parser_compare::test_command_r7b_legacy_parser(const std::string & input,
                                                                  bool                partial,
                                                                  bool                print_results) {
    // Original parser taken from chat.cpp
    common_chat_msg_parser builder(input,
                                   /* is_partial= */ partial,
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
