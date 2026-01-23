#include "../src/llama-grammar.h"
#include "chat-auto-parser.h"
#include "chat-diff-analyzer.h"
#include "chat.h"
#include "common.h"
#include "gguf.h"
#include "jinja/runtime.h"
#include "log.h"

#include <fstream>
#include <numeric>
#include <sstream>
#include <string>

#include "nlohmann/json.hpp"
#include "peg-parser.h"

using json = nlohmann::ordered_json;

enum class output_mode {
    ANALYSIS,  // Only output analysis results (default)
    TEMPLATE,  // Only output rendered template
    BOTH       // Output both
};

enum class input_message_type {
    NONE,                    // Don't render any message scenarios (only analysis)
    CONTENT_ONLY,            // Simple assistant message with content
    REASONING_CONTENT,       // Message with reasoning_content + content
    TOOL_CALL_ONLY,          // Message with tool_calls only
    CONTENT_TOOL_CALL,       // Message with content + tool_calls
    REASONING_TOOL_CALL,     // Message with reasoning_content + tool_calls
    CONTENT_FAKE_TOOL_CALL,  // Message with content but no actual tool_calls (for testing)
    ALL                      // Render all scenarios
};

struct debug_options {
    std::string      template_path;
    bool             with_tools        = true;
    bool             generation_prompt = true;
    bool             enable_reasoning  = true;
    bool             debug_jinja       = false;
    bool             force_tool_call   = false;
    output_mode      mode              = output_mode::BOTH;
    input_message_type input_message   = input_message_type::NONE;
};

static std::string read_file(const std::string & path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::ostringstream buf;
    buf << fin.rdbuf();
    return buf.str();
}

static std::string read_gguf_chat_template(const std::string & path) {
    struct gguf_init_params params = { /*no_alloc =*/true,  // We only need metadata, not tensor data
                                       /*ctx=*/nullptr };

    struct gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (ctx == nullptr) {
        throw std::runtime_error("Could not open GGUF file: " + path);
    }

    const char * key    = "tokenizer.chat_template";
    int64_t      key_id = gguf_find_key(ctx, key);

    if (key_id == -1) {
        gguf_free(ctx);
        throw std::runtime_error("GGUF file does not contain chat template key: " + std::string(key));
    }

    const char * template_str = gguf_get_val_str(ctx, key_id);
    if (template_str == nullptr) {
        gguf_free(ctx);
        throw std::runtime_error("GGUF file contains chat template key but value is null");
    }

    std::string result = template_str;
    gguf_free(ctx);
    return result;
}

static void print_usage(const char * program_name) {
    LOG_ERR("Usage: %s <template_or_gguf_path> [options]\n", program_name);
    LOG_ERR("\nOptions:\n");
    LOG_ERR("  --no-tools              Disable tool definitions\n");
    LOG_ERR("  --force-tool-call       Set tool calls to forced\n");
    LOG_ERR("  --generation-prompt=0|1 Set add_generation_prompt (default: 1)\n");
    LOG_ERR("  --enable-reasoning=0|1  Enable reasoning parsing (default: 1)\n");
    LOG_ERR("  --output=MODE           Output mode: analysis, template, both (default: both)\n");
    LOG_ERR("  --debug-jinja           Enable Jinja fine-grained debug\n");
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

static bool parse_options(int argc, char ** argv, debug_options & opts) {
    if (argc < 2) {
        print_usage(argv[0]);
        return false;
    }

    opts.template_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--force-tool-call") {
            opts.force_tool_call = true;
        } else if (arg == "--debug-jinja") {
            opts.debug_jinja = true;
        } else if (arg == "--no-tools") {
            opts.with_tools = false;
        } else if (arg.rfind("--generation-prompt=", 0) == 0) {
            opts.generation_prompt = parse_bool_option(arg.substr(20));
        } else if (arg.rfind("--enable-reasoning=", 0) == 0) {
            opts.enable_reasoning = parse_bool_option(arg.substr(19));
        } else if (arg.rfind("--output=", 0) == 0) {
            std::string mode = arg.substr(9);
            if (mode == "analysis") {
                opts.mode = output_mode::ANALYSIS;
            } else if (mode == "template") {
                opts.mode = output_mode::TEMPLATE;
            } else if (mode == "both") {
                opts.mode = output_mode::BOTH;
            } else {
                LOG_ERR("Unknown output mode: %s\n", mode.c_str());
                return false;
            }
        } else if (arg.rfind("--input-message=", 0) == 0) {
            std::string type = arg.substr(16);
            if (type == "content_only") {
                opts.input_message = input_message_type::CONTENT_ONLY;
            } else if (type == "reasoning_content") {
                opts.input_message = input_message_type::REASONING_CONTENT;
            } else if (type == "tool_call_only") {
                opts.input_message = input_message_type::TOOL_CALL_ONLY;
            } else if (type == "content_tool_call") {
                opts.input_message = input_message_type::CONTENT_TOOL_CALL;
            } else if (type == "reasoning_tool_call") {
                opts.input_message = input_message_type::REASONING_TOOL_CALL;
            } else if (type == "content_fake_tool_call") {
                opts.input_message = input_message_type::CONTENT_FAKE_TOOL_CALL;
            } else if (type == "all") {
                opts.input_message = input_message_type::ALL;
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
    parameters_schema["required"]             = json::array({ "param1" });

    return json::array({
        json{ { "type", "function" },
             { "function", json{ { "name", "test_function_name" },
                                  { "description", "A test function for debugging" },
                                  { "parameters", parameters_schema } } } }
    });
}

static void render_scenario(const common_chat_template & tmpl,
                            const std::string &          scenario_name,
                            const json &                 messages,
                            const json &                 tools,
                            bool                         add_generation_prompt,
                            bool                         enable_thinking) {
    LOG_ERR("\n=== Scenario: %s ===\n", scenario_name.c_str());
    LOG_ERR("add_generation_prompt: %s, enable_thinking: %s\n", add_generation_prompt ? "true" : "false",
            enable_thinking ? "true" : "false");

    // When add_generation_prompt is true, add a trailing user message to trigger the prompt
    json final_messages = messages;
    if (add_generation_prompt && !messages.empty() && messages.back().value("role", "") == "assistant") {
        final_messages.push_back(json{
            { "role",    "user" },
            { "content", "Now please continue with another response." }
        });
    }

    LOG_ERR("Messages:\n%s\n", final_messages.dump(2).c_str());

    try {
        autoparser::templates_params inputs;
        inputs.messages                         = final_messages;
        inputs.add_generation_prompt            = add_generation_prompt;
        inputs.extra_context["enable_thinking"] = enable_thinking;

        if (!tools.is_null() && tools.is_array() && !tools.empty()) {
            inputs.tools = tools;
        }

        std::string output = common_chat_template_direct_apply(tmpl, inputs);

        LOG_ERR("\n--- Rendered Output ---\n");
        LOG_ERR("%s\n", output.c_str());
        LOG_ERR("--- End Output (length: %zu) ---\n", output.length());
    } catch (const std::exception & e) {
        LOG_ERR("Rendering failed: %s\n", e.what());
    }
}

static void render_all_scenarios(const common_chat_template & tmpl,
                                 const json &                 tools,
                                 bool                         add_generation_prompt,
                                 bool                         enable_thinking,
                                 input_message_type             message_type) {
    json user_msg = build_user_message();

    auto render_if = [&](input_message_type type, const std::string & name, const json & assistant_msg) {
        if (message_type == input_message_type::ALL || message_type == type) {
            json messages = json::array({ user_msg, assistant_msg });
            render_scenario(tmpl, name, messages, tools, add_generation_prompt, enable_thinking);
        }
    };

    render_if(input_message_type::CONTENT_ONLY, "content_only", build_content_only_message());
    render_if(input_message_type::REASONING_CONTENT, "reasoning_content", build_reasoning_content_message());
    render_if(input_message_type::TOOL_CALL_ONLY, "tool_call_only", build_tool_call_only_message());
    render_if(input_message_type::CONTENT_TOOL_CALL, "content_tool_call", build_content_tool_call_message());
    render_if(input_message_type::REASONING_TOOL_CALL, "reasoning_tool_call", build_reasoning_tool_call_message());
    render_if(input_message_type::CONTENT_FAKE_TOOL_CALL, "content_fake_tool_call",
              build_content_fake_tool_call_message());

    // Also render with add_generation_prompt=true to show the prompt ending
    if (message_type == input_message_type::ALL) {
        LOG_ERR("\n\n=== Generation Prompt Scenarios (add_generation_prompt=true) ===\n");

        json prompt_messages = json::array({ user_msg });
        render_scenario(tmpl, "generation_prompt_only", prompt_messages, tools, true, enable_thinking);

        // With enable_thinking toggled
        render_scenario(tmpl, "generation_prompt_thinking_disabled", prompt_messages, tools, true, false);
    }
}

int main(int argc, char ** argv) {
    // Set log level to most verbose to capture all debug output
    common_log_set_verbosity_thold(99);

    debug_options opts;
    if (!parse_options(argc, argv, opts)) {
        return 1;
    }

    if (opts.debug_jinja || std::getenv("LLAMA_DEBUG_JINJA") != nullptr) {
        jinja::enable_debug(true);
    }

    std::string template_source;
    try {
        // Check if the file is a GGUF file
        if (opts.template_path.size() >= 5 &&
            opts.template_path.compare(opts.template_path.size() - 5, 5, ".gguf") == 0) {
            template_source = read_gguf_chat_template(opts.template_path);
        } else {
            template_source = read_file(opts.template_path);
        }
    } catch (const std::exception & e) {
        LOG_ERR("Error reading template: %s\n", e.what());
        return 1;
    }

    LOG_ERR("Analyzing template: %s\n", opts.template_path.c_str());
    LOG_ERR("Options: with_tools=%s, generation_prompt=%s, enable_reasoning=%s\n", opts.with_tools ? "true" : "false",
            opts.generation_prompt ? "true" : "false", opts.enable_reasoning ? "true" : "false");

    try {
        common_chat_template chat_template(template_source, "", "");

        // Build tools definition
        json tools = opts.with_tools ? build_tools_definition() : json();

        // Render template scenarios if requested
        if (opts.input_message != input_message_type::NONE &&
            (opts.mode == output_mode::TEMPLATE || opts.mode == output_mode::BOTH)) {
            LOG_ERR("\n");
            LOG_ERR("================================================================================\n");
            LOG_ERR("                         TEMPLATE RENDERING OUTPUT\n");
            LOG_ERR("================================================================================\n");

            render_all_scenarios(chat_template, tools, opts.generation_prompt, opts.enable_reasoning,
                                 opts.input_message);
        }

        // Output analysis if requested
        if (opts.mode == output_mode::ANALYSIS || opts.mode == output_mode::BOTH) {
            LOG_ERR("\n");
            LOG_ERR("================================================================================\n");
            LOG_ERR("                           TEMPLATE ANALYSIS\n");
            LOG_ERR("================================================================================\n");

            autoparser::analyze_template analysis(chat_template);

            // Generate Parser
            autoparser::templates_params params;
            params.messages = json::array({ build_user_message() });
            params.reasoning_format =
                opts.enable_reasoning ? COMMON_REASONING_FORMAT_DEEPSEEK : COMMON_REASONING_FORMAT_NONE;
            params.enable_thinking       = opts.enable_reasoning;
            params.add_generation_prompt = opts.generation_prompt;

            if (opts.with_tools) {
                params.tools       = tools;
                params.tool_choice = opts.force_tool_call ? COMMON_CHAT_TOOL_CHOICE_REQUIRED : COMMON_CHAT_TOOL_CHOICE_AUTO;
            } else {
                params.tools       = json();
                params.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
            }
            params.parallel_tool_calls = false;

            auto parser_data = autoparser::universal_peg_generator::generate_parser(chat_template, params, analysis);

            LOG_ERR("\n=== Generated Parser ===\n");
            common_peg_arena arena;
            arena.load(parser_data.parser);
            LOG_ERR("%s\n", arena.dump(arena.root()).c_str());

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

            if (!parser_data.grammar.empty()) {
                LOG_ERR("\n=== Verifying created grammar ===\n");
                auto * grammar = llama_grammar_init_impl(nullptr, parser_data.grammar.c_str(), "root",
                                                         parser_data.grammar_lazy, nullptr, 0, nullptr, 0);
                if (grammar != nullptr) {
                    LOG_ERR("\n=== Grammar successfully created ===\n");
                }
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("Analysis failed: %s\n", e.what());
        return 1;
    }

    return 0;
}
