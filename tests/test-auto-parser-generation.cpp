// Tests for automatic parser generation on specific templates
// Tests template analysis and parser generation using the unified two-phase approach

#include "../src/llama-grammar.h"
#include "../src/unicode.h"
#include "chat-auto-parser.h"
#include "chat.h"
#include "log.h"
#include "peg-parser/tests.h"

#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::ordered_json;


static std::string read_file(const std::string & path) {
    std::ifstream fs(path, std::ios_base::binary);
    if (!fs.is_open()) {
        fs = std::ifstream("../" + path, std::ios_base::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }
    }
    fs.seekg(0, std::ios_base::end);
    auto size = fs.tellg();
    fs.seekg(0);
    std::string out;
    out.resize(static_cast<size_t>(size));
    fs.read(out.data(), static_cast<std::streamsize>(size));
    return out;
}

static common_chat_templates_ptr read_templates(const std::string & path) {
    return common_chat_templates_ptr(common_chat_templates_init(/* model= */ nullptr, read_file(path)));
}

static std::ostream & operator<<(std::ostream & os, const common_chat_msg & msg) {
    os << "{ role: " << msg.role << "; ";
    os << "content: " << msg.content << "; ";
    os << "reasoning_content: " << msg.reasoning_content << "; ";
    os << "tool_calls: " << msg.tool_calls.size();
    os << "}";
    return os;
}

template <class T> static bool equals(const T & expected, const T & actual) {
    return expected == actual;
}

static common_chat_msg normalize(const common_chat_msg & msg) {
    common_chat_msg normalized = msg;
    for (auto & tool_call : normalized.tool_calls) {
        try {
            tool_call.arguments = json::parse(tool_call.arguments).dump();
        } catch (const std::exception &) {
            // Do nothing
        }
    }
    return normalized;
}

template <> bool equals(const common_chat_msg & expected, const common_chat_msg & actual) {
    return normalize(expected) == normalize(actual);
}

template <class T> static void assert_equals(const T & expected, const T & actual) {
    if (!equals(expected, actual)) {
        std::cerr << "Expected: " << expected << std::endl;
        std::cerr << "Actual: " << actual << std::endl;
        throw std::runtime_error("Test failed: Objects not equal");
    }
}

static std::unique_ptr<llama_grammar> build_llama_grammar(const std::string & grammar_str) {
    return std::unique_ptr<llama_grammar>(
        llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0));
}

static bool match_string(const std::string & input, llama_grammar * grammar) {
    const auto cpts       = unicode_cpts_from_utf8(input);
    auto &     stacks_cur = llama_grammar_get_stacks(grammar);
    for (const auto & cpt : cpts) {
        llama_grammar_accept(grammar, cpt);
        if (stacks_cur.empty()) {
            printf("Failed at codepoint: %u (char: %c) at index %ld\n", cpt, (char) cpt, &cpt - &cpts[0]);
            return false;
        }
    }
    if (std::any_of(stacks_cur.begin(), stacks_cur.end(), [](const auto & stack) { return stack.empty(); })) {
        return true;
    }
    return false;
}

common_chat_tool special_function_tool{
    "special_function",
    "I'm special",
    R"({
        "type": "object",
        "properties": {
            "arg1": {"type": "string"}
        },
        "required": ["arg1"]
    })"
};

const common_chat_msg message_user{ "user", "Hello, world!", {}, {}, "", "", "" };

static void test_qwen3_coder_template(testing & t) {
    t.log("Testing Qwen3-Coder template analysis and parser generation");

    // Load the Qwen3-Coder template
    auto tmpls = read_templates("models/templates/Qwen3-Coder.jinja");
    t.assert_true("Qwen3-Coder template loaded", tmpls != nullptr);

    // Test template analysis
    auto                 template_source = common_chat_templates_source(tmpls.get());
    minja::chat_template chat_template(template_source, "", "");

    // Use TemplateAnalyzer to detect patterns (unified two-phase analysis)
    TemplateAnalysisResult analysis = TemplateAnalyzer::analyze_template(chat_template);

    // Debug: Print what was detected
    printf("Qwen3-Coder detected:\n");
    printf("  Content structure:\n");
    printf("    reasoning_mode: %d\n", static_cast<int>(analysis.content.reasoning_mode));
    printf("    content_mode: %d\n", static_cast<int>(analysis.content.content_mode));
    printf("  Tool structure:\n");
    printf("    supports_tools: %s\n", analysis.tools.supports_tools ? "true" : "false");
    printf("    function_format: %d\n", static_cast<int>(analysis.tools.function_format));
    printf("    argument_format: %d\n", static_cast<int>(analysis.tools.argument_format));
    printf("    tool_section_start: '%s'\n", analysis.tools.tool_section_start.c_str());
    printf("    function_prefix: '%s'\n", analysis.tools.function_prefix.c_str());
    printf("    function_close: '%s'\n", analysis.tools.function_close.c_str());

    // Verify the detected format is appropriate for Qwen3-Coder (should be tag-based)
    t.assert_true("Qwen3-Coder should support tools", analysis.tools.supports_tools);
    t.assert_equal("Qwen3-Coder should use TAG_WITH_NAME format",
                   static_cast<int>(ToolCallStructure::FUNC_TAG_WITH_NAME),
                   static_cast<int>(analysis.tools.function_format));

    // Verify tool call markers were detected
    t.assert_true("Qwen3-Coder should have tool section start", !analysis.tools.tool_section_start.empty());
    t.assert_true("Qwen3-Coder should have function prefix", !analysis.tools.function_prefix.empty());
    t.assert_true("Qwen3-Coder should have function close", !analysis.tools.function_close.empty());

    // Test parser generation
    templates_params params;
    params.messages            = json::array();
    params.tools               = json::array();
    params.tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
    params.parallel_tool_calls = false;

    // Use UniversalPEGGenerator to generate parser
    try {
        auto parser_data = UniversalPEGGenerator::generate_parser(analysis, chat_template, params);

        // Debug: Print parser format
        printf("Qwen3-Coder generated parser format: %d (PEG_NATIVE=%d)\n",
               static_cast<int>(parser_data.format), static_cast<int>(COMMON_CHAT_FORMAT_PEG_NATIVE));

        // Verify parser was generated successfully (check if not empty)
        t.assert_true("Qwen3-Coder parser should be generated", !parser_data.parser.empty());

        // Verify grammar was generated
        t.assert_true("Qwen3-Coder should have generated grammar", !parser_data.grammar.empty());

        t.log("Qwen3-Coder parser generation successful");
    } catch (const std::exception & e) {
        printf("Qwen3-Coder parser generation failed: %s\n", e.what());
        t.assert_true("Qwen3-Coder parser generation should not throw: " + std::string(e.what()), false);
    }
}

static void test_bytedance_seed_oss_template(testing & t) {
    t.log("Testing ByteDance-Seed-OSS template analysis and parser generation");

    // Load the ByteDance-Seed-OSS template
    auto tmpls = read_templates("models/templates/ByteDance-Seed-OSS.jinja");
    t.assert_true("ByteDance-Seed-OSS template loaded", tmpls != nullptr);

    // Test template analysis
    auto                 template_source = common_chat_templates_source(tmpls.get());
    minja::chat_template chat_template(template_source, "", "");

    // Use TemplateAnalyzer to detect patterns (unified two-phase analysis)
    TemplateAnalysisResult analysis = TemplateAnalyzer::analyze_template(chat_template);

    // Debug: Print what was detected
    printf("ByteDance-Seed-OSS detected:\n");
    printf("  Tool structure:\n");
    printf("    supports_tools: %s\n", analysis.tools.supports_tools ? "true" : "false");
    printf("    function_format: %d\n", static_cast<int>(analysis.tools.function_format));
    printf("    tool_section_start: '%s'\n", analysis.tools.tool_section_start.c_str());

    // Verify tool support detection
    t.assert_true("ByteDance-Seed-OSS should support tools", analysis.tools.supports_tools);

    // Test parser generation
    templates_params params;
    params.messages            = json::array();
    params.tools               = json::array();
    params.tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
    params.parallel_tool_calls = false;

    try {
        auto parser_data = UniversalPEGGenerator::generate_parser(analysis, chat_template, params);

        // Verify parser was generated
        t.assert_true("ByteDance-Seed-OSS parser should be generated", !parser_data.parser.empty());

        t.log("ByteDance-Seed-OSS parser generation successful");
    } catch (const std::exception & e) {
        printf("ByteDance-Seed-OSS parser generation failed: %s\n", e.what());
        t.assert_true("ByteDance-Seed-OSS parser generation should not throw: " + std::string(e.what()), false);
    }
}

static void test_nvidia_nemotron_template(testing & t) {
    t.log("Testing NVIDIA-Nemotron-Nano-v2 template analysis and parser generation");

    // Load the NVIDIA-Nemotron-Nano-v2 template
    auto tmpls = read_templates("models/templates/NVIDIA-Nemotron-Nano-v2.jinja");
    t.assert_true("NVIDIA-Nemotron-Nano-v2 template loaded", tmpls != nullptr);

    // Test template analysis
    auto                 template_source = common_chat_templates_source(tmpls.get());
    minja::chat_template chat_template(template_source, "", "");

    // Use TemplateAnalyzer to detect patterns (unified two-phase analysis)
    TemplateAnalysisResult analysis = TemplateAnalyzer::analyze_template(chat_template);

    // Debug: Print what was detected
    printf("NVIDIA-Nemotron-Nano-v2 detected:\n");
    printf("  Tool structure:\n");
    printf("    supports_tools: %s\n", analysis.tools.supports_tools ? "true" : "false");
    printf("    function_format: %d\n", static_cast<int>(analysis.tools.function_format));

    // Test parser generation
    templates_params params;
    params.messages            = json::array();
    params.tools               = json::array();
    params.tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
    params.parallel_tool_calls = false;

    try {
        auto parser_data = UniversalPEGGenerator::generate_parser(analysis, chat_template, params);

        // Verify parser was generated
        t.assert_true("NVIDIA-Nemotron-Nano-v2 parser should be generated", !parser_data.parser.empty());

        t.log("NVIDIA-Nemotron-Nano-v2 parser generation successful");
    } catch (const std::exception & e) {
        printf("NVIDIA-Nemotron-Nano-v2 parser generation failed: %s\n", e.what());
        t.assert_true("NVIDIA-Nemotron-Nano-v2 parser generation should not throw: " + std::string(e.what()), false);
    }
}

static void test_template_analysis_structure(testing & t) {
    t.log("Testing TemplateAnalysisResult structure initialization");

    // Test that TemplateAnalysisResult structure is properly initialized
    TemplateAnalysisResult analysis;

    // Check initial state - default constructed values
    t.assert_equal("Initial reasoning_mode should be NONE", static_cast<int>(ContentStructure::REASONING_NONE),
                   static_cast<int>(analysis.content.reasoning_mode));
    t.assert_equal("Initial content_mode should be PLAIN", static_cast<int>(ContentStructure::CONTENT_PLAIN),
                   static_cast<int>(analysis.content.content_mode));
    t.assert_true("Initial supports_tools should be false", !analysis.tools.supports_tools);

    // Test with Qwen3-Coder template
    auto tmpls = read_templates("models/templates/Qwen3-Coder.jinja");
    t.assert_true("Qwen3-Coder template loaded for structure test", tmpls != nullptr);

    auto                 template_source = common_chat_templates_source(tmpls.get());
    minja::chat_template chat_template(template_source, "", "");

    analysis = TemplateAnalyzer::analyze_template(chat_template);

    // After analysis, should detect tool support
    t.assert_true("Should detect tool support after analysis", analysis.tools.supports_tools);

    // Tool structure should be populated
    t.assert_true("Should have tool section start", !analysis.tools.tool_section_start.empty());
    t.assert_true("Should have function prefix", !analysis.tools.function_prefix.empty());
    t.assert_true("Should have function close", !analysis.tools.function_close.empty());
}

static void test_universal_peg_generator_edge_cases(testing & t) {
    t.log("Testing UniversalPEGGenerator edge cases");

    // Test with Qwen3-Coder template
    auto tmpls = read_templates("models/templates/Qwen3-Coder.jinja");
    t.assert_true("Qwen3-Coder template loaded for edge case test", tmpls != nullptr);

    auto                 template_source = common_chat_templates_source(tmpls.get());
    minja::chat_template chat_template(template_source, "", "");

    TemplateAnalysisResult analysis = TemplateAnalyzer::analyze_template(chat_template);

    // Test 1: Empty tools array
    {
        templates_params params;
        params.messages            = json::array();
        params.tools               = json::array();
        params.tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
        params.parallel_tool_calls = false;

        try {
            auto parser_data = UniversalPEGGenerator::generate_parser(analysis, chat_template, params);
            t.assert_true("Empty tools should generate parser", !parser_data.parser.empty());
        } catch (const std::exception & e) {
            t.assert_true("Empty tools should not throw: " + std::string(e.what()), false);
        }
    }

    // Test 2: tool_choice = NONE
    {
        templates_params params;
        params.messages            = json::array();
        params.tools               = json::array({ { { "type", "function" }, { "function", { { "name", "test" }, { "parameters", json::object() } } } } });
        params.tool_choice         = COMMON_CHAT_TOOL_CHOICE_NONE;
        params.parallel_tool_calls = false;

        try {
            auto parser_data = UniversalPEGGenerator::generate_parser(analysis, chat_template, params);
            t.assert_true("NONE tool_choice should generate parser", !parser_data.parser.empty());
        } catch (const std::exception & e) {
            t.assert_true("NONE tool_choice should not throw: " + std::string(e.what()), false);
        }
    }

    // Test 3: Parallel tool calls
    {
        templates_params params;
        params.messages            = json::array();
        params.tools               = json::array({ { { "type", "function" }, { "function", { { "name", "test" }, { "parameters", json::object() } } } } });
        params.tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
        params.parallel_tool_calls = true;

        try {
            auto parser_data = UniversalPEGGenerator::generate_parser(analysis, chat_template, params);
            t.assert_true("Parallel tool calls should generate parser", !parser_data.parser.empty());
        } catch (const std::exception & e) {
            t.assert_true("Parallel tool calls should not throw: " + std::string(e.what()), false);
        }
    }
}

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("qwen3_coder_template", test_qwen3_coder_template);
    t.test("bytedance_seed_oss_template", test_bytedance_seed_oss_template);
    t.test("nvidia_nemotron_template", test_nvidia_nemotron_template);
    t.test("template_analysis_structure", test_template_analysis_structure);
    t.test("universal_peg_generator_edge_cases", test_universal_peg_generator_edge_cases);

    return t.summary();
}
