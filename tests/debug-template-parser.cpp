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

// ============================================================================
// Command-line options
// ============================================================================

enum class OutputMode {
    ANALYSIS,  // Only output analysis results (default)
    TEMPLATE,  // Only output rendered template
    BOTH       // Output both
};

enum class InputMessageType {
    NONE,                    // Don't render any message scenarios (only analysis)
    CONTENT_ONLY,            // Simple assistant message with content
    REASONING_CONTENT,       // Message with reasoning_content + content
    TOOL_CALL_ONLY,          // Message with tool_calls only
    CONTENT_TOOL_CALL,       // Message with content + tool_calls
    REASONING_TOOL_CALL,     // Message with reasoning_content + tool_calls
    CONTENT_FAKE_TOOL_CALL,  // Message with content but no actual tool_calls (for testing)
    ALL                      // Render all scenarios
};

struct DebugOptions {
    std::string      template_path;
    bool             with_tools        = true;
    bool             with_deepseek     = false;
    bool             generation_prompt = true;
    bool             enable_reasoning  = true;
    OutputMode       output_mode       = OutputMode::BOTH;
    InputMessageType input_message     = InputMessageType::NONE;
};

// ============================================================================
// Helper functions
// ============================================================================

static std::string read_file(const std::string & path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::ostringstream buf;
    buf << fin.rdbuf();
    return buf.str();
}

static void print_usage(const char * program_name) {
    LOG_ERR("Usage: %s <template_path> [options]\n", program_name);
    LOG_ERR("\nOptions:\n");
    LOG_ERR("  --no-tools              Disable tool definitions\n");
    LOG_ERR("  --deepseek              Use DeepSeek reasoning format\n");
    LOG_ERR("  --generation-prompt=0|1 Set add_generation_prompt (default: 1)\n");
    LOG_ERR("  --enable-reasoning=0|1  Set enable_thinking context (default: 1)\n");
    LOG_ERR("  --output=MODE           Output mode: analysis, template, both (default: both)\n");
    LOG_ERR("  --input-message=TYPE    Message type to render:\n");
    LOG_ERR("                          content_only, reasoning_content, tool_call_only,\n");
    LOG_ERR("                          content_tool_call, reasoning_tool_call,\n");
    LOG_ERR("                          content_fake_tool_call, all\n");
    LOG_ERR("\nExamples:\n");
    LOG_ERR("  %s template.jinja --input-message=all --generation-prompt=1\n", program_name);
    LOG_ERR("  %s template.jinja --output=template --input-message=tool_call_only\n", program_name);
}

static bool parse_bool_option(const std::string & value) {
    return value == "1" || value == "true" || value == "yes";
}

static bool parse_options(int argc, char ** argv, DebugOptions & opts) {
    if (argc < 2) {
        print_usage(argv[0]);
        return false;
    }

    opts.template_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--no-tools") {
            opts.with_tools = false;
        } else if (arg == "--deepseek") {
            opts.with_deepseek = true;
        } else if (arg.rfind("--generation-prompt=", 0) == 0) {
            opts.generation_prompt = parse_bool_option(arg.substr(20));
        } else if (arg.rfind("--enable-reasoning=", 0) == 0) {
            opts.enable_reasoning = parse_bool_option(arg.substr(19));
        } else if (arg.rfind("--output=", 0) == 0) {
            std::string mode = arg.substr(9);
            if (mode == "analysis") {
                opts.output_mode = OutputMode::ANALYSIS;
            } else if (mode == "template") {
                opts.output_mode = OutputMode::TEMPLATE;
            } else if (mode == "both") {
                opts.output_mode = OutputMode::BOTH;
            } else {
                LOG_ERR("Unknown output mode: %s\n", mode.c_str());
                return false;
            }
        } else if (arg.rfind("--input-message=", 0) == 0) {
            std::string type = arg.substr(16);
            if (type == "content_only") {
                opts.input_message = InputMessageType::CONTENT_ONLY;
            } else if (type == "reasoning_content") {
                opts.input_message = InputMessageType::REASONING_CONTENT;
            } else if (type == "tool_call_only") {
                opts.input_message = InputMessageType::TOOL_CALL_ONLY;
            } else if (type == "content_tool_call") {
                opts.input_message = InputMessageType::CONTENT_TOOL_CALL;
            } else if (type == "reasoning_tool_call") {
                opts.input_message = InputMessageType::REASONING_TOOL_CALL;
            } else if (type == "content_fake_tool_call") {
                opts.input_message = InputMessageType::CONTENT_FAKE_TOOL_CALL;
            } else if (type == "all") {
                opts.input_message = InputMessageType::ALL;
            } else {
                LOG_ERR("Unknown input message type: %s\n", type.c_str());
                return false;
            }
        } else {
            LOG_ERR("Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    return true;
}

// ============================================================================
// Test message builders
// ============================================================================

static json build_user_message() {
    return json{
        { "role",    "user"                               },
        { "content", "Hello, please help me with a task." }
    };
}

static json build_content_only_message() {
    return json{
        { "role",    "assistant"                                   },
        { "content", "Hello! I'm here to help you with your task." }
    };
}

static json build_reasoning_content_message() {
    return json{
        { "role",              "assistant"                                                               },
        { "content",           "Hello! I'm here to help you with your task."                             },
        { "reasoning_content", "The user is greeting me and asking for help. I should respond politely." }
    };
}

static json build_tool_call_only_message() {
    return json{
        { "role",       "assistant"      },
        { "content",    nullptr          },
        { "tool_calls",
         json::array({ json{
              { "type", "function" },
              { "function", json{ { "name", "test_function_name" },
                                  { "arguments", json::object({ { "param1", "value1" }, { "param2", "value2" } }) } } },
              { "id", "123456789" } } }) }
    };
}

static json build_content_tool_call_message() {
    return json{
        { "role",       "assistant"                                                                              },
        { "content",    "I'll help you by calling a function."                                                   },
        { "tool_calls",
         json::array({ json{
              { "type", "function" },
              { "function",
                json{ { "name", "test_function_name" },
                      { "arguments", json::object({ { "param1", "value1" }, { "param2", "value2" } }) } } } } }) }
    };
}

static json build_reasoning_tool_call_message() {
    return json{
        { "role",              "assistant"                                                                       },
        { "content",           nullptr                                                                           },
        { "reasoning_content", "I need to call a function to help with this task."                               },
        { "tool_calls",
         json::array({ json{
              { "type", "function" },
              { "function",
                json{ { "name", "test_function_name" },
                      { "arguments", json::object({ { "param1", "value1" }, { "param2", "value2" } }) } } } } }) }
    };
}

static json build_content_fake_tool_call_message() {
    // This message has content but NO tool_calls field
    // It's used to test if a template renders tool definitions but not tool calls
    return json{
        { "role",    "assistant"                            },
        { "content", "I'll help you by calling a function." }
    };
}

static json build_tools_definition() {
    json parameters_schema                    = json::object();
    parameters_schema["type"]                 = "object";
    parameters_schema["properties"]           = json::object();
    parameters_schema["properties"]["param1"] = json::object({
        { "type",        "string"          },
        { "description", "First parameter" }
    });
    parameters_schema["properties"]["param2"] = json::object({
        { "type",        "string"           },
        { "description", "Second parameter" }
    });
    parameters_schema["required"]             = json::array({ "param1", "param2" });

    return json::array({
        json{ { "type", "function" },
             { "function", json{ { "name", "test_function_name" },
                                  { "description", "A test function for debugging" },
                                  { "parameters", parameters_schema } } } }
    });
}

// ============================================================================
// Template rendering
// ============================================================================

static void render_scenario(const minja::chat_template & tmpl,
                            const std::string &          scenario_name,
                            const json &                 messages,
                            const json &                 tools,
                            bool                         add_generation_prompt,
                            bool                         enable_thinking) {
    LOG_ERR("\n=== Scenario: %s ===\n", scenario_name.c_str());
    LOG_ERR("add_generation_prompt: %s, enable_thinking: %s\n", add_generation_prompt ? "true" : "false",
            enable_thinking ? "true" : "false");
    LOG_ERR("Messages:\n%s\n", messages.dump(2).c_str());

    try {
        minja::chat_template_inputs inputs;
        inputs.messages                         = messages;
        inputs.add_generation_prompt            = add_generation_prompt;
        inputs.extra_context["enable_thinking"] = enable_thinking;

        if (!tools.is_null() && tools.is_array() && !tools.empty()) {
            inputs.tools = tools;
        }

        minja::chat_template_options opts;
        opts.apply_polyfills = false;

        std::string output = tmpl.apply(inputs, opts);

        LOG_ERR("\n--- Rendered Output ---\n");
        LOG_ERR("%s\n", output.c_str());
        LOG_ERR("--- End Output (length: %zu) ---\n", output.length());
    } catch (const std::exception & e) {
        LOG_ERR("Rendering failed: %s\n", e.what());
    }
}

static void render_all_scenarios(const minja::chat_template & tmpl,
                                 const json &                 tools,
                                 bool                         add_generation_prompt,
                                 bool                         enable_thinking,
                                 InputMessageType             message_type) {
    json user_msg = build_user_message();

    auto render_if = [&](InputMessageType type, const std::string & name, const json & assistant_msg) {
        if (message_type == InputMessageType::ALL || message_type == type) {
            json messages = json::array({ user_msg, assistant_msg });
            render_scenario(tmpl, name, messages, tools, add_generation_prompt, enable_thinking);
        }
    };

    render_if(InputMessageType::CONTENT_ONLY, "content_only", build_content_only_message());
    render_if(InputMessageType::REASONING_CONTENT, "reasoning_content", build_reasoning_content_message());
    render_if(InputMessageType::TOOL_CALL_ONLY, "tool_call_only", build_tool_call_only_message());
    render_if(InputMessageType::CONTENT_TOOL_CALL, "content_tool_call", build_content_tool_call_message());
    render_if(InputMessageType::REASONING_TOOL_CALL, "reasoning_tool_call", build_reasoning_tool_call_message());
    render_if(InputMessageType::CONTENT_FAKE_TOOL_CALL, "content_fake_tool_call",
              build_content_fake_tool_call_message());

    // Also render with add_generation_prompt=true to show the prompt ending
    if (message_type == InputMessageType::ALL) {
        LOG_ERR("\n\n=== Generation Prompt Scenarios (add_generation_prompt=true) ===\n");

        json prompt_messages = json::array({ user_msg });
        render_scenario(tmpl, "generation_prompt_only", prompt_messages, tools, true, enable_thinking);

        // With enable_thinking toggled
        render_scenario(tmpl, "generation_prompt_thinking_disabled", prompt_messages, tools, true, false);
    }
}

static const char * reasoning_mode_to_str(ContentStructure::ReasoningMode mode) {
    switch (mode) {
        case ContentStructure::REASONING_NONE:
            return "NONE";
        case ContentStructure::REASONING_OPTIONAL:
            return "OPTIONAL";
        case ContentStructure::REASONING_FORCED_OPEN:
            return "FORCED_OPEN";
    }
    return "UNKNOWN";
}

static const char * content_mode_to_str(ContentStructure::ContentMode mode) {
    switch (mode) {
        case ContentStructure::CONTENT_PLAIN:
            return "PLAIN";
        case ContentStructure::CONTENT_ALWAYS_WRAPPED:
            return "ALWAYS_WRAPPED";
        case ContentStructure::CONTENT_WRAPPED_WITH_REASONING:
            return "WRAPPED_WITH_REASONING";
    }
    return "UNKNOWN";
}

static const char * function_format_to_str(ToolCallStructure::FunctionFormat fmt) {
    switch (fmt) {
        case ToolCallStructure::FUNC_JSON_OBJECT:
            return "JSON_OBJECT";
        case ToolCallStructure::FUNC_TAG_WITH_NAME:
            return "TAG_WITH_NAME";
        case ToolCallStructure::FUNC_TAG_NAME_ONLY:
            return "TAG_NAME_ONLY";
        case ToolCallStructure::FUNC_PREFIXED_INDEXED:
            return "PREFIXED_INDEXED";
        case ToolCallStructure::FUNC_NAME_AS_KEY:
            return "NAME_AS_KEY";
    }
    return "UNKNOWN";
}

static const char * argument_format_to_str(ToolCallStructure::ArgumentFormat fmt) {
    switch (fmt) {
        case ToolCallStructure::ARGS_JSON:
            return "JSON";
        case ToolCallStructure::ARGS_TAGGED:
            return "TAGGED";
        case ToolCallStructure::ARGS_KEY_VALUE_TAGS:
            return "KEY_VALUE_TAGS";
    }
    return "UNKNOWN";
}

int main(int argc, char ** argv) {
    // Set log level to most verbose to capture all debug output
    common_log_set_verbosity_thold(99);

    DebugOptions opts;
    if (!parse_options(argc, argv, opts)) {
        return 1;
    }

    std::string template_source;
    try {
        template_source = read_file(opts.template_path);
    } catch (const std::exception & e) {
        LOG_ERR("Error reading template: %s\n", e.what());
        return 1;
    }

    LOG_ERR("Analyzing template: %s\n", opts.template_path.c_str());
    LOG_ERR("Options: with_tools=%s, generation_prompt=%s, enable_reasoning=%s\n", opts.with_tools ? "true" : "false",
            opts.generation_prompt ? "true" : "false", opts.enable_reasoning ? "true" : "false");

    try {
        minja::chat_template chat_template(template_source, "", "");

        // Build tools definition
        json tools = opts.with_tools ? build_tools_definition() : json();

        // Render template scenarios if requested
        if (opts.input_message != InputMessageType::NONE &&
            (opts.output_mode == OutputMode::TEMPLATE || opts.output_mode == OutputMode::BOTH)) {
            LOG_ERR("\n");
            LOG_ERR("================================================================================\n");
            LOG_ERR("                         TEMPLATE RENDERING OUTPUT\n");
            LOG_ERR("================================================================================\n");

            render_all_scenarios(chat_template, tools, opts.generation_prompt, opts.enable_reasoning,
                                 opts.input_message);
        }

        // Output analysis if requested
        if (opts.output_mode == OutputMode::ANALYSIS || opts.output_mode == OutputMode::BOTH) {
            LOG_ERR("\n");
            LOG_ERR("================================================================================\n");
            LOG_ERR("                           TEMPLATE ANALYSIS\n");
            LOG_ERR("================================================================================\n");

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

            // Additional fields for special formats
            if (analysis.tools.function_format == ToolCallStructure::FUNC_PREFIXED_INDEXED) {
                LOG_ERR("\n--- Prefixed-Indexed Format Details ---\n");
                LOG_ERR("per_call_start: '%s'\n", analysis.tools.per_call_start.c_str());
                LOG_ERR("function_namespace: '%s'\n", analysis.tools.function_namespace.c_str());
                LOG_ERR("args_marker: '%s'\n", analysis.tools.args_marker.c_str());
                LOG_ERR("per_call_end: '%s'\n", analysis.tools.per_call_end.c_str());
            }

            // Generate Parser
            templates_params params;
            params.messages = json::array();
            params.reasoning_format =
                opts.with_deepseek ? COMMON_REASONING_FORMAT_DEEPSEEK : COMMON_REASONING_FORMAT_NONE;
            params.enable_thinking       = opts.enable_reasoning;
            params.add_generation_prompt = opts.generation_prompt;

            if (opts.with_tools) {
                params.tools       = tools;
                params.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
            } else {
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
            for (const common_grammar_trigger & cgt : parser_data.grammar_triggers) {
                LOG_ERR("Token: %d | Type: %d | Value: %s\n", cgt.token, cgt.type, cgt.value.c_str());
            }

            LOG_ERR("\n=== Preserved Tokens ===\n");
            for (const std::string & token : parser_data.preserved_tokens) {
                LOG_ERR("  '%s'\n", token.c_str());
            }

            LOG_ERR("\n=== Verifying created grammar ===\n");
            auto * grammar = llama_grammar_init_impl(nullptr, parser_data.grammar.c_str(), "root",
                                                     parser_data.grammar_lazy, nullptr, 0, nullptr, 0);
            if (grammar != nullptr) {
                LOG_ERR("\n=== Grammar successfully created ===\n");
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("Analysis failed: %s\n", e.what());
        return 1;
    }

    return 0;
}
