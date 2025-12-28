#include "../src/llama-grammar.h"
#include "chat-auto-parser.h"
#include "common.h"
#include "peg-parser/testing.h"

#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Helper to check if a string matches a grammar
static bool match_string(const std::string & input, const std::string & grammar_str) {
    // Initialize grammar
    struct llama_grammar * grammar =
        llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0);

    if (!grammar) {
        return false;
    }

    if (grammar == nullptr) {
        fprintf(stderr, "%s : failed to initialize grammar\n", __func__);
        return false;
    }

    try {
        llama_grammar_accept_str(*grammar, input);
        bool valid = !grammar->stacks.empty();
        llama_grammar_free_impl(grammar);
        return valid;
    } catch (const std::runtime_error &) {
        llama_grammar_free_impl(grammar);
        return false;
    }
}

static void test_qwen3_tool_call(testing & t) {
    t.log("Testing Qwen3-Coder tool call grammar acceptance");

    // Setup analysis result using unified structures
    TemplateAnalysisResult analysis;

    // Content structure (no reasoning for this test)
    analysis.content.reasoning_mode = ContentStructure::REASONING_NONE;
    analysis.content.content_mode   = ContentStructure::CONTENT_PLAIN;

    // Tool call structure for Qwen3-style XML tags
    analysis.tools.supports_tools     = true;
    analysis.tools.function_format    = ToolCallStructure::FUNC_TAG_WITH_NAME;
    analysis.tools.argument_format    = ToolCallStructure::ARGS_TAGGED;
    analysis.tools.tool_section_start = "<tool_call>";
    analysis.tools.tool_section_end   = "</tool_call>";
    analysis.tools.function_prefix    = "<function=";
    analysis.tools.function_suffix    = ">";
    analysis.tools.function_close     = "</function>";
    analysis.tools.arg_prefix         = "<parameter=";
    analysis.tools.arg_suffix         = ">";
    analysis.tools.arg_close          = "</parameter>";
    analysis.tools.arg_separator      = "\n";

    // Define Tools
    std::vector<common_chat_tool> tools;
    common_chat_tool              tool;
    tool.name        = "get_weather";
    tool.description = "Get weather";
    tool.parameters  = R"({
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    })";
    tools.push_back(tool);

    // Generate Parser
    struct templates_params params = {};
    params.tool_choice             = COMMON_CHAT_TOOL_CHOICE_AUTO;
    params.reasoning_format        = COMMON_REASONING_FORMAT_NONE;
    params.tools                   = nlohmann::json::array();
    for (auto & tool_item : tools) {
        json tj;
        tj["type"]                    = "function";
        tj["function"]["name"]        = tool_item.name;
        tj["function"]["description"] = tool_item.description;
        tj["function"]["parameters"]  = json::parse(tool_item.parameters);
        params.tools.push_back(tj);
    }

    // minja::chat_template constructor takes (source, bos, eos)
    minja::chat_template tmpl(std::string(""), std::string(""), std::string(""));

    auto        result_params = UniversalPEGGenerator::generate_parser(analysis, tmpl, params);
    std::string grammar_str   = result_params.grammar;

    // 3. Test Strings
    // Valid Qwen3 tool call
    std::string valid_call =
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=location>London</parameter>\n"
        "</function>\n"
        "</tool_call>";

    t.assert_true("Matches valid tool call", match_string(valid_call, grammar_str));

    std::string valid_call_2 =
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=location>Paris</parameter>\n"
        "<parameter=unit>celsius</parameter>\n"
        "</function>\n"
        "</tool_call>";
    t.assert_true("Matches valid tool call with 2 args", match_string(valid_call_2, grammar_str));

    // Valid partial?
    std::string partial_call = "<tool_call>\n<function=get_weather>\n";
    t.assert_true("Matches partial call", match_string(partial_call, grammar_str));

    // Invalid (bad tag)
    std::string invalid_call = "<tool_call>\n<function=get_weather>\n<badparam>London</badparam>\n";
    t.assert_equal("Rejects invalid call", false, match_string(invalid_call, grammar_str));
}

static void test_seed_tool_call(testing & t) {
    t.log("Testing Seed-OSS tool call grammar acceptance");

    // Setup analysis result using unified structures
    TemplateAnalysisResult analysis;

    // Content structure (no reasoning for this test)
    analysis.content.reasoning_mode = ContentStructure::REASONING_NONE;
    analysis.content.content_mode   = ContentStructure::CONTENT_PLAIN;

    // Tool call structure for Seed-style XML tags
    analysis.tools.supports_tools     = true;
    analysis.tools.function_format    = ToolCallStructure::FUNC_TAG_WITH_NAME;
    analysis.tools.argument_format    = ToolCallStructure::ARGS_TAGGED;
    analysis.tools.tool_section_start = "<seed:tool_call>";
    analysis.tools.tool_section_end   = "</seed:tool_call>";
    analysis.tools.function_prefix    = "<function=";
    analysis.tools.function_suffix    = ">";
    analysis.tools.function_close     = "</function>";
    analysis.tools.arg_prefix         = "<parameter=";
    analysis.tools.arg_suffix         = ">";
    analysis.tools.arg_close          = "</parameter>";
    analysis.tools.arg_separator      = "\n";

    // Define Tools
    std::vector<common_chat_tool> tools;
    common_chat_tool              tool;
    tool.name        = "get_weather";
    tool.description = "Get weather";
    tool.parameters  = R"({
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    })";
    tools.push_back(tool);

    // Generate Parser
    struct templates_params params = {};
    params.tool_choice             = COMMON_CHAT_TOOL_CHOICE_AUTO;
    params.reasoning_format        = COMMON_REASONING_FORMAT_AUTO;
    params.tools                   = nlohmann::json::array();
    for (auto & tool_item : tools) {
        json tj;
        tj["type"]                    = "function";
        tj["function"]["name"]        = tool_item.name;
        tj["function"]["description"] = tool_item.description;
        tj["function"]["parameters"]  = json::parse(tool_item.parameters);
        params.tools.push_back(tj);
    }

    minja::chat_template tmpl(std::string(""), std::string(""), std::string(""));

    auto        result_params = UniversalPEGGenerator::generate_parser(analysis, tmpl, params);
    std::string grammar_str   = result_params.grammar;

    // Test Valid Seed tool call
    std::string valid_call =
        "<seed:tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=location>Berlin</parameter>\n"
        "</function>\n"
        "</seed:tool_call>";

    t.assert_true("Matches valid tool call", match_string(valid_call, grammar_str));
}

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("qwen3_tool_call", test_qwen3_tool_call);
    t.test("seed_tool_call", test_seed_tool_call);

    return t.summary();
}
