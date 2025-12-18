#include "../src/llama-grammar.h"
#include "chat-auto-parser.h"
#include "chat-parser.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "peg-parser.h"
#include "peg-parser/testing.h"

#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Definition of templates_params to match the one in chat.cpp/test-auto-parser-generation.cpp
struct templates_params {
    json                                  messages;
    json                                  tools;
    common_chat_tool_choice               tool_choice;
    json                                  json_schema;
    bool                                  parallel_tool_calls;
    common_reasoning_format               reasoning_format;
    bool                                  stream;
    std::string                           grammar;
    bool                                  add_generation_prompt;
    bool                                  enable_thinking;
    std::chrono::system_clock::time_point now;
    json                                  extra_context;
    bool                                  add_bos;
    bool                                  add_eos;
    bool                                  is_inference;
};

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
        llama_grammar_accept_str(*grammar, input.c_str());
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

    // Setup Template Pattern
    TemplatePattern qwen3_pattern;
    qwen3_pattern.format                                    = TemplatePattern::XML_CONSTRUCTED;
    qwen3_pattern.special_markers["tool_call_start_marker"] = "<tool_call>";
    qwen3_pattern.special_markers["tool_call_end_marker"]   = "</tool_call>";
    qwen3_pattern.special_markers["function_opener"]        = "<function=";
    qwen3_pattern.special_markers["function_name_suffix"]   = ">";
    qwen3_pattern.special_markers["function_closer"]        = "</function>";
    qwen3_pattern.special_markers["parameter_key_prefix"]   = "<parameter=";
    qwen3_pattern.special_markers["parameter_key_suffix"]   = ">";
    qwen3_pattern.special_markers["parameter_closer"]       = "</parameter>";
    qwen3_pattern.special_markers["argument_separator"]     = "\n";
    qwen3_pattern.special_markers["parameter_opener"]       = "";
    qwen3_pattern.special_markers["reasoning_start_marker"] = "";
    qwen3_pattern.special_markers["reasoning_end_marker"]   = "";

    // Define Tools
    std::vector<common_chat_tool> tools;
    common_chat_tool              tool;
    tool.name        = "get_weather";
    tool.description = "Get weather";
    // parameters is std::string in common_chat_tool
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
    for (auto & t : tools) {
        json tj;
        tj["type"]                    = "function";
        tj["function"]["name"]        = t.name;
        tj["function"]["description"] = t.description;
        tj["function"]["parameters"]  = json::parse(t.parameters);
        params.tools.push_back(tj);
    }

    // minja::chat_template constructor takes (source, bos, eos)
    minja::chat_template tmpl(std::string(""), std::string(""), std::string(""));

    auto        result_params = UniversalPEGGenerator::generate_parser(qwen3_pattern, tmpl, params);
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

    // Setup Template Pattern for Seed
    TemplatePattern seed_pattern;
    seed_pattern.format                                    = TemplatePattern::XML_CONSTRUCTED;
    seed_pattern.special_markers["tool_call_start_marker"] = "<seed:tool_call>";
    seed_pattern.special_markers["tool_call_end_marker"]   = "</seed:tool_call>";
    seed_pattern.special_markers["function_opener"]        = "<function=";
    seed_pattern.special_markers["function_name_suffix"]   = ">";
    seed_pattern.special_markers["function_closer"]        = "</function>";
    seed_pattern.special_markers["parameter_key_prefix"]   = "<parameter=";
    seed_pattern.special_markers["parameter_key_suffix"]   = ">";
    seed_pattern.special_markers["parameter_closer"]       = "</parameter>";
    seed_pattern.special_markers["argument_separator"]     = "\n";
    seed_pattern.special_markers["parameter_opener"]       = "";
    seed_pattern.special_markers["reasoning_start_marker"] = "";
    seed_pattern.special_markers["reasoning_end_marker"]   = "";

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
    for (auto & t : tools) {
        json tj;
        tj["type"]                    = "function";
        tj["function"]["name"]        = t.name;
        tj["function"]["description"] = t.description;
        tj["function"]["parameters"]  = json::parse(t.parameters);
        params.tools.push_back(tj);
    }

    minja::chat_template tmpl(std::string(""), std::string(""), std::string(""));

    auto        result_params = UniversalPEGGenerator::generate_parser(seed_pattern, tmpl, params);
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
