#include "../src/llama-grammar.h"
#include "chat-auto-parser.h"
#include "chat.h"
#include "common.h"
#include "log.h"

#include <fstream>
#include <minja/chat-template.hpp>
#include <minja/minja.hpp>
#include <sstream>
#include <string>

using json = nlohmann::ordered_json;

// Helper to read file
static std::string read_file(const std::string & path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::ostringstream buf;
    buf << fin.rdbuf();
    return buf.str();
}

static const char * reasoning_mode_to_str(ContentStructure::ReasoningMode mode) {
    switch (mode) {
        case ContentStructure::REASONING_NONE:        return "NONE";
        case ContentStructure::REASONING_OPTIONAL:    return "OPTIONAL";
        case ContentStructure::REASONING_FORCED_OPEN: return "FORCED_OPEN";
    }
    return "UNKNOWN";
}

static const char * content_mode_to_str(ContentStructure::ContentMode mode) {
    switch (mode) {
        case ContentStructure::CONTENT_PLAIN:                  return "PLAIN";
        case ContentStructure::CONTENT_ALWAYS_WRAPPED:         return "ALWAYS_WRAPPED";
        case ContentStructure::CONTENT_WRAPPED_WITH_REASONING: return "WRAPPED_WITH_REASONING";
    }
    return "UNKNOWN";
}

static const char * function_format_to_str(ToolCallStructure::FunctionFormat fmt) {
    switch (fmt) {
        case ToolCallStructure::FUNC_JSON_OBJECT:   return "JSON_OBJECT";
        case ToolCallStructure::FUNC_TAG_WITH_NAME: return "TAG_WITH_NAME";
        case ToolCallStructure::FUNC_TAG_NAME_ONLY: return "TAG_NAME_ONLY";
    }
    return "UNKNOWN";
}

static const char * argument_format_to_str(ToolCallStructure::ArgumentFormat fmt) {
    switch (fmt) {
        case ToolCallStructure::ARGS_JSON:   return "JSON";
        case ToolCallStructure::ARGS_TAGGED: return "TAGGED";
    }
    return "UNKNOWN";
}

int main(int argc, char ** argv) {
    // Set log level to most verbose to capture all debug output
    common_log_set_verbosity_thold(99);

    if (argc < 2) {
        LOG_ERR("Usage: %s <template_path> [--no-tools]\n", argv[0]);
        return 1;
    }

    std::string template_path = argv[1];
    bool        with_tools    = true;
    bool        with_deepseek = false;

    // Parse command-line options
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-tools") {
            with_tools = false;
        } else if (arg == "--deepseek") {
            with_deepseek = true;
        } else {
            LOG_ERR("Unknown option: %s\n", arg.c_str());
            LOG_ERR("Usage: %s <template_path> [--no-tools] [--deepseek]\n", argv[0]);
            return 1;
        }
    }
    std::string template_source;
    try {
        template_source = read_file(template_path);
    } catch (const std::exception & e) {
        LOG_ERR("Error reading template: %s\n", e.what());
        return 1;
    }

    LOG_ERR("Analyzing template: %s\n", template_path.c_str());

    try {
        minja::chat_template chat_template(template_source, "", "");
        TemplateAnalysisResult analysis = TemplateAnalyzer::analyze_template(chat_template);

        LOG_ERR("\n=== Analysis Results ===\n");

        LOG_ERR("\n--- Content Structure (Phase 1) ---\n");
        LOG_ERR("reasoning_mode: %s\n", reasoning_mode_to_str(analysis.content.reasoning_mode));
        LOG_ERR("reasoning_start: '%s'\n", analysis.content.reasoning_start.c_str());
        LOG_ERR("reasoning_end: '%s'\n", analysis.content.reasoning_end.c_str());
        LOG_ERR("content_mode: %s\n", content_mode_to_str(analysis.content.content_mode));
        LOG_ERR("content_start: '%s'\n", analysis.content.content_start.c_str());
        LOG_ERR("content_end: '%s'\n", analysis.content.content_end.c_str());

        LOG_ERR("\n--- Tool Structure (Phase 2) ---\n");
        LOG_ERR("supports_tools: %s\n", analysis.tools.supports_tools ? "true" : "false");
        LOG_ERR("function_format: %s\n", function_format_to_str(analysis.tools.function_format));
        LOG_ERR("argument_format: %s\n", argument_format_to_str(analysis.tools.argument_format));
        LOG_ERR("tool_section_start: '%s'\n", analysis.tools.tool_section_start.c_str());
        LOG_ERR("tool_section_end: '%s'\n", analysis.tools.tool_section_end.c_str());
        LOG_ERR("function_prefix: '%s'\n", analysis.tools.function_prefix.c_str());
        LOG_ERR("function_suffix: '%s'\n", analysis.tools.function_suffix.c_str());
        LOG_ERR("function_close: '%s'\n", analysis.tools.function_close.c_str());
        LOG_ERR("arg_prefix: '%s'\n", analysis.tools.arg_prefix.c_str());
        LOG_ERR("arg_suffix: '%s'\n", analysis.tools.arg_suffix.c_str());
        LOG_ERR("arg_close: '%s'\n", analysis.tools.arg_close.c_str());
        LOG_ERR("name_field: '%s'\n", analysis.tools.name_field.c_str());
        LOG_ERR("args_field: '%s'\n", analysis.tools.args_field.c_str());
        LOG_ERR("id_field: '%s'\n", analysis.tools.id_field.c_str());

        // Generate Parser
        templates_params params;
        params.messages         = json::array();
        params.reasoning_format = with_deepseek ? COMMON_REASONING_FORMAT_DEEPSEEK : COMMON_REASONING_FORMAT_NONE;

        if (with_tools) {
            // Create test tool schema properly using json::object()
            json parameters_schema                  = json::object();
            parameters_schema["type"]               = "object";
            parameters_schema["properties"]         = json::object();
            parameters_schema["properties"]["arg1"] = json::object({
                { "type", "string" }
            });
            parameters_schema["properties"]["arg2"] = json::object({
                { "type", "string" }
            });
            parameters_schema["required"]           = json::array({ "arg1", "arg2" });

            json test_tool       = json::array();
            json tool_def        = json::object();
            tool_def["type"]     = "function";
            tool_def["function"] = json::object({
                { "name",        "test_tool"       },
                { "description", "A test tool"     },
                { "parameters",  parameters_schema }
            });
            test_tool.push_back(tool_def);
            params.tools       = test_tool;
            params.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        } else {
            // No tools - set tools to null
            params.tools       = json();
            params.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
        }
        params.parallel_tool_calls = false;

        auto parser_data = UniversalPEGGenerator::generate_parser(analysis, chat_template, params);

        LOG_ERR("\n=== Generated Parser ===\n");
        LOG_ERR("%s\n", parser_data.parser.c_str());

        LOG_ERR("\n=== Generated Grammar ===\n");
        LOG_ERR("%s\n", parser_data.grammar.c_str());

        LOG_ERR("\n=== Generated Lazy Grammar ===\n");
        LOG_ERR("%d\n", parser_data.grammar_lazy);

        LOG_ERR("\n=== Generated Grammar Triggers ===\n");
        for (common_grammar_trigger cgt : parser_data.grammar_triggers) {
            LOG_ERR("Token: %d | Type: %d | Value: %s\n", cgt.token, cgt.type, cgt.value.c_str());
        }

        LOG_ERR("\n=== Verifying created grammar ===\n");
        auto * grammar = llama_grammar_init_impl(nullptr, parser_data.grammar.c_str(), "root", parser_data.grammar_lazy,
                                                 nullptr, 0, nullptr, 0);
        if (grammar != nullptr) {
            LOG_ERR("\n=== Grammar successfully created ===\n");
        }
    } catch (const std::exception & e) {
        LOG_ERR("Analysis failed: %s\n", e.what());
        return 1;
    }

    return 0;
}
