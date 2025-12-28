// Comprehensive test for debugging partial parser behavior
// Specifically testing CohereForAI-style tool calls with reasoning markers

#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "peg-parser.h"

#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

using json = nlohmann::ordered_json;

static void print_result(const char * label, const common_peg_parse_result & result, const std::string & input) {
    const char * type = "UNKNOWN";
    if (result.success()) {
        type = "SUCCESS";
    } else if (result.fail()) {
        type = "FAIL";
    } else if (result.need_more_input()) {
        type = "NEED_MORE_INPUT";
    }

    printf("[%s] type=%s start=%zu end=%zu nodes=%zu\n", label, type, result.start, result.end, result.nodes.size());

    if (result.end < input.size()) {
        size_t show_len = std::min(size_t(40), input.size() - result.end);
        printf("  Remaining: '%s'%s\n", input.substr(result.end, show_len).c_str(),
               show_len < input.size() - result.end ? "..." : "");
    }
}

static void test_until_at_delimiter() {
    printf("\n=== TEST: until() where delimiter is at start ===\n\n");

    std::string delimiter = "<|START_ACTION|>[";
    std::string input     = "<|START_ACTION|>[\n    {\"t";  // delimiter at position 0

    printf("Delimiter: '%s' (len=%zu)\n", delimiter.c_str(), delimiter.size());
    printf("Input: '%s' (len=%zu)\n\n", input.c_str(), input.size());

    // Test 1: until() should match 0 chars when delimiter is at start
    auto until_parser = build_peg_parser([&](common_peg_parser_builder & p) { return p.until(delimiter); });

    printf("--- Test: until at position 0 (delimiter at start) ---\n");
    common_peg_parse_context ctx(input, true);
    auto                     result = until_parser.parse(ctx);
    print_result("until", result, input);

    if (result.success() && result.end == 0) {
        printf("PASS: until correctly matched 0 chars\n\n");
    } else {
        printf("FAIL: expected SUCCESS with end=0\n\n");
    }
}

static void test_simplified_parser() {
    printf("\n=== TEST: Simplified CohereForAI-style parser ===\n\n");

    // Build a parser that mimics the CohereForAI structure - SIMPLIFIED VERSION
    auto parser = build_chat_peg_unified_parser([](common_chat_peg_unified_builder & p) {
        std::string reasoning_start = "<|START_THINKING|>";
        std::string reasoning_end   = "<|END_THINKING|>";
        std::string tool_call_start = "<|START_ACTION|>[";
        std::string tool_call_end   = "]<|END_ACTION|>";

        // reasoning = optional(literal + reasoning + literal)
        auto reasoning =
            p.optional(p.literal(reasoning_start) + p.reasoning(p.until(reasoning_end)) + p.literal(reasoning_end));

        // content_before_tools = content(until(tool_call_start))
        auto content_before_tools = p.content(p.until(tool_call_start));

        // tool_calls = literal(start) + space + until(end) + literal(end)
        auto tool_call = p.sequence({ p.literal(tool_call_start), p.space(), p.tag("tool_args", p.until(tool_call_end)),
                                      p.space(), p.literal(tool_call_end) });
        auto optional_tool_call = p.optional(tool_call);

        // Full sequence: reasoning + content + space + tool_calls + space + end
        return p.sequence({ reasoning, content_before_tools, p.space(), optional_tool_call, p.space(), p.end() });
    });

    // Skip printing simplified parser JSON - to_json() has a serialization bug
    printf("=== Simplified Parser (skipping JSON dump due to serialization bug) ===\n\n");

    // Test partial parsing
    std::string full_input =
        "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
        "<|START_ACTION|>[\n"
        "    {\"tool_call_id\": \"0\", \"tool_name\": \"test\", \"parameters\": {}}\n"
        "]<|END_ACTION|>";

    printf("Full input (%zu chars):\n%s\n\n", full_input.size(), full_input.c_str());

    // Test the specific failing position and complete input
    std::vector<size_t> test_positions = { 46, 63, 70, full_input.size() };

    for (size_t pos : test_positions) {
        if (pos > full_input.size()) {
            continue;
        }

        std::string partial    = full_input.substr(0, pos);
        bool        is_partial = (pos < full_input.size());

        printf("\n--- Position %zu/%zu (is_partial=%d) ---\n", pos, full_input.size(), is_partial);

        common_peg_parse_context ctx(partial, is_partial);
        auto                     result = parser.parse(ctx);

        print_result("simplified", result, partial);
    }
}

static void test_actual_template_parser() {
    printf("\n=== TEST: Actual CohereForAI Template Parser ===\n\n");

    std::string   template_path = "models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja";
    std::ifstream file(template_path);
    if (!file) {
        printf("Could not open template file: %s\n", template_path.c_str());
        printf("Skipping actual template test.\n");
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string template_src = buffer.str();

    printf("Loaded template (%zu chars)\n\n", template_src.size());

    // Initialize templates
    auto tmpls = common_chat_templates_init(nullptr, template_src);

    // Build inputs matching the test at line 3307
    common_chat_templates_inputs inputs;

    // Create user message
    common_chat_msg user_msg;
    user_msg.role    = "user";
    user_msg.content = "Test";
    inputs.messages  = { user_msg };

    // Set reasoning format to DEEPSEEK (matching line 3312)
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    // Create tool definition matching special_function_tool from test-chat.cpp
    common_chat_tool special_function_tool{
        /* .name = */ "special_function",
        /* .description = */ "I'm special",
        /* .parameters = */ R"({
            "type": "object",
            "properties": {
                "arg1": {
                    "type": "integer",
                    "description": "The arg."
                }
            },
            "required": ["arg1"]
        })",
    };
    inputs.tools               = { special_function_tool };
    inputs.tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls = false;

    // Apply template to get parser
    auto params = common_chat_templates_apply(tmpls.get(), inputs);

    printf("Parser format: %s\n", common_chat_format_name(params.format));

    if (params.parser.empty()) {
        printf("ERROR: No PEG parser generated for this template!\n");
        return;
    }

    // Print the actual generated parser JSON
    printf("\n=== Actual Template Parser JSON ===\n");
    printf("%s\n\n", params.parser.c_str());

    // Load parser
    common_peg_arena parser;
    parser.load(params.parser);

    // Test the exact input from line 3308-3311
    std::string test_input =
        "<|START_THINKING|>I'm\nthinking<|END_THINKING|>"
        "<|START_ACTION|>[\n"
        "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n"
        "]<|END_ACTION|>";

    printf("Test input (%zu chars):\n%s\n\n", test_input.size(), test_input.c_str());

    // Test positions including the failing ones
    std::vector<size_t> test_positions = { 45, 46, 47, 62, 63, 64, 70, test_input.size() };

    for (size_t pos : test_positions) {
        if (pos > test_input.size()) {
            continue;
        }

        std::string partial    = test_input.substr(0, pos);
        bool        is_partial = (pos < test_input.size());

        printf("\n--- Position %zu/%zu (is_partial=%d) ---\n", pos, test_input.size(), is_partial);
        printf("Input ends with: '...%s'\n", partial.substr(partial.size() > 25 ? partial.size() - 25 : 0).c_str());

        common_peg_parse_context ctx(partial, is_partial);
        auto                     result = parser.parse(ctx);

        print_result("actual", result, partial);
    }
}

static void compare_parser_structures() {
    printf("\n=== COMPARISON: Simplified vs Actual Parser Root Structure ===\n\n");

    // Build simplified parser
    auto simplified = build_chat_peg_unified_parser([](common_chat_peg_unified_builder & p) {
        std::string reasoning_start = "<|START_THINKING|>";
        std::string reasoning_end   = "<|END_THINKING|>";
        std::string tool_call_start = "<|START_ACTION|>[";
        std::string tool_call_end   = "]<|END_ACTION|>";

        auto reasoning =
            p.optional(p.literal(reasoning_start) + p.reasoning(p.until(reasoning_end)) + p.literal(reasoning_end));
        auto content_before_tools = p.content(p.until(tool_call_start));
        auto tool_call = p.sequence({ p.literal(tool_call_start), p.space(), p.tag("tool_args", p.until(tool_call_end)),
                                      p.space(), p.literal(tool_call_end) });
        auto optional_tool_call = p.optional(tool_call);

        return p.sequence({ reasoning, content_before_tools, p.space(), optional_tool_call, p.space(), p.end() });
    });

    std::string simplified_json   = simplified.to_json();
    json        simplified_parsed = json::parse(simplified_json);

    printf("Simplified parser:\n");
    printf("  Root index: %d\n", simplified_parsed["root"].get<int>());
    auto & simplified_root = simplified_parsed["parsers"][simplified_parsed["root"].get<int>()];
    printf("  Root type: %s\n", simplified_root["type"].get<std::string>().c_str());
    if (simplified_root.contains("children")) {
        printf("  Root children count: %zu\n", simplified_root["children"].size());
        printf("  Root children indices: ");
        for (auto & c : simplified_root["children"]) {
            printf("%d ", c.get<int>());
        }
        printf("\n");
    }

    // Load actual parser from template
    std::string   template_path = "models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja";
    std::ifstream file(template_path);
    if (!file) {
        printf("\nCould not load template for comparison.\n");
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    auto tmpls = common_chat_templates_init(nullptr, buffer.str());

    common_chat_templates_inputs inputs;
    common_chat_msg              user_msg;
    user_msg.role           = "user";
    user_msg.content        = "Test";
    inputs.messages         = { user_msg };
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    common_chat_tool func_tool{
        /* .name = */ "special_function",
        /* .description = */ "I'm special",
        /* .parameters = */ R"({
            "type": "object",
            "properties": {
                "arg1": { "type": "integer" }
            }
        })",
    };
    inputs.tools = { func_tool };

    auto params = common_chat_templates_apply(tmpls.get(), inputs);

    if (!params.parser.empty()) {
        json actual_parsed = json::parse(params.parser);

        printf("\nActual template parser:\n");
        printf("  Root index: %d\n", actual_parsed["root"].get<int>());
        auto & actual_root = actual_parsed["parsers"][actual_parsed["root"].get<int>()];
        printf("  Root type: %s\n", actual_root["type"].get<std::string>().c_str());
        if (actual_root.contains("children")) {
            printf("  Root children count: %zu\n", actual_root["children"].size());
            printf("  Root children indices: ");
            for (auto & c : actual_root["children"]) {
                printf("%d ", c.get<int>());
            }
            printf("\n");

            // Print what each child is
            printf("\n  Root children details:\n");
            for (size_t i = 0; i < actual_root["children"].size(); i++) {
                int    child_idx = actual_root["children"][i].get<int>();
                auto & child     = actual_parsed["parsers"][child_idx];
                printf("    [%zu] index=%d type=%s", i, child_idx, child["type"].get<std::string>().c_str());
                if (child.contains("delimiters")) {
                    printf(" delimiters=[");
                    for (auto & d : child["delimiters"]) {
                        printf("'%s' ", d.get<std::string>().c_str());
                    }
                    printf("]");
                }
                if (child.contains("literal")) {
                    printf(" literal='%s'", child["literal"].get<std::string>().c_str());
                }
                if (child.contains("tag")) {
                    printf(" tag='%s'", child["tag"].get<std::string>().c_str());
                }
                if (child.contains("child")) {
                    printf(" child=%d", child["child"].get<int>());
                }
                printf("\n");
            }
        }
    }
}

int main() {
    printf("==============================================\n");
    printf("Comprehensive Parser Debug Test\n");
    printf("==============================================\n");

    test_until_at_delimiter();
    test_simplified_parser();
    // compare_parser_structures();  // Skip - to_json() has serialization bug
    test_actual_template_parser();

    printf("\n==============================================\n");
    printf("Tests complete\n");
    printf("==============================================\n");

    return 0;
}
