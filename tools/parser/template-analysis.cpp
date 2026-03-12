#include "chat-auto-parser.h"
#include "chat-auto-parser-helpers.h"
#include "chat.h"
#include "log.h"
#include "jinja/caps.h"
#include "jinja/runtime.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

// ANSI color codes - using 256-color palette for brighter colors (all bold)
#define ANSI_RESET       "\033[0m"
#define ANSI_PURPLE      "\033[1m\x1b[38;5;126m"  // Bold bright purple for main headers
#define ANSI_CYAN        "\033[1m\x1b[38;5;81m"   // Bold bright cyan for section headers
#define ANSI_BLUE        "\033[1m\x1b[38;5;12m"   // Bold bright blue for labels
#define ANSI_ORANGE      "\033[1m\x1b[38;5;209m"  // Bold orange for right differences
#define ANSI_GREEN       "\033[1m\x1b[38;5;83m"   // Bold bright green for left differences
#define ANSI_GRAY        "\033[1m\x1b[38;5;240m"  // Bold gray (used for "no variables" message)
#define ANSI_BOLD        "\033[1m"                // Standalone bold
#define ANSI_PREFIX      "\033[1m\x1b[38;5;176m"  // Bold color for common prefix
#define ANSI_SUFFIX      "\033[1m\x1b[38;5;61m"   // Bold color for common suffix

// All template paths extracted from tests/test-chat.cpp
static const std::vector<std::string> ALL_TEMPLATE_PATHS = {
    "models/templates/Apertus-8B-Instruct.jinja",
    "models/templates/Apriel-1.6-15b-Thinker-fixed.jinja",
    "models/templates/ByteDance-Seed-OSS.jinja",
    "models/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja",
    "models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja",
    "models/templates/GLM-4.6.jinja",
    "models/templates/GLM-4.7-Flash.jinja",
    "models/templates/Kimi-K2-Instruct.jinja",
    "models/templates/Kimi-K2-Thinking.jinja",
    "models/templates/MiMo-VL.jinja",
    "models/templates/MiniMax-M2.jinja",
    "models/templates/Mistral-Small-3.2-24B-Instruct-2506.jinja",
    "models/templates/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja",
    "models/templates/NVIDIA-Nemotron-Nano-v2.jinja",
    "models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja",
    "models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja",
    "models/templates/Qwen-QwQ-32B.jinja",
    "models/templates/Qwen-Qwen2.5-7B-Instruct.jinja",
    "models/templates/Qwen3-Coder.jinja",
    "models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja",
    "models/templates/deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja",
    "models/templates/deepseek-ai-DeepSeek-V3.1.jinja",
    "models/templates/fireworks-ai-llama-3-firefunction-v2.jinja",
    "models/templates/google-gemma-2-2b-it.jinja",
    "models/templates/ibm-granite-granite-3.3-2B-Instruct.jinja",
    "models/templates/llama-cpp-deepseek-r1.jinja",
    "models/templates/meetkai-functionary-medium-v3.1.jinja",
    "models/templates/meetkai-functionary-medium-v3.2.jinja",
    "models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja",
    "models/templates/meta-llama-Llama-3.2-3B-Instruct.jinja",
    "models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja",
    "models/templates/mistralai-Ministral-3-14B-Reasoning-2512.jinja",
    "models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja",
    "models/templates/moonshotai-Kimi-K2.jinja",
    "models/templates/openai-gpt-oss-120b.jinja",
    "models/templates/unsloth-Apriel-1.5.jinja",
    "models/templates/unsloth-mistral-Devstral-Small-2507.jinja",
};

struct analysis_options {
    std::vector<std::string> template_paths;
    bool                     analyze_all = false;
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

static void print_usage(const char * program_name) {
    LOG_ERR("Usage: %s [options]\n", program_name);
    LOG_ERR("\nOptions:\n");
    LOG_ERR("  --template <name>       Analyze specific template from test suite (e.g., 'deepseek' or 'DeepSeek-V3.1')\n");
    LOG_ERR("  --template-file <path>  Analyze custom template file\n");
    LOG_ERR("  --all                   Analyze all templates from test suite\n");
    LOG_ERR("\nExamples:\n");
    LOG_ERR("  %s --all\n", program_name);
    LOG_ERR("  %s --template deepseek\n", program_name);
    LOG_ERR("  %s --template-file my-template.jinja\n", program_name);
}

static bool parse_options(int argc, char ** argv, analysis_options & opts) {
    if (argc < 2) {
        print_usage(argv[0]);
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--all") {
            opts.analyze_all = true;
        } else if (arg == "--template") {
            if (i + 1 >= argc) {
                LOG_ERR("--template requires an argument\n");
                return false;
            }
            std::string pattern = argv[++i];
            std::transform(pattern.begin(), pattern.end(), pattern.begin(), ::tolower);

            // Find matching templates
            bool found = false;
            for (const auto & path : ALL_TEMPLATE_PATHS) {
                std::string path_lower = path;
                std::transform(path_lower.begin(), path_lower.end(), path_lower.begin(), ::tolower);
                if (path_lower.find(pattern) != std::string::npos) {
                    opts.template_paths.push_back(path);
                    found = true;
                }
            }

            if (!found) {
                LOG_ERR("No templates found matching: %s\n", pattern.c_str());
                return false;
            }
        } else if (arg == "--template-file") {
            if (i + 1 >= argc) {
                LOG_ERR("--template-file requires an argument\n");
                return false;
            }
            opts.template_paths.push_back(argv[++i]);
        } else {
            LOG_ERR("Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    if (opts.analyze_all) {
        opts.template_paths = ALL_TEMPLATE_PATHS;
    }

    if (opts.template_paths.empty()) {
        LOG_ERR("No templates specified\n");
        print_usage(argv[0]);
        return false;
    }

    return true;
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

// Helper to create a tool call with arguments as JSON object
static json build_tool_call(const std::string & name, const json & args_object, const std::string & id = "call_001") {
    return json{
        {"id", id},
        {"type", "function"},
        {"function", json{
            {"name", name},
            {"arguments", args_object}  // Pass as JSON object, not serialized string
        }}
    };
}

// Helper functions to create repeating message definitions
static json make_user_msg() {
    return json{
        {"role", "user"},
        {"content", "Hello, please help me."}
    };
}

static json make_user_msg2() {
    return json{
        {"role", "user"},
        {"content", "Thank you."}
    };
}

static json make_user_msg2_continue() {
    return json{
        {"role", "user"},
        {"content", "Continue."}
    };
}

static json make_assistant_no_tool() {
    return json{
        {"role", "assistant"},
        {"content", "Let me help you."}
    };
}

static json make_assistant_one_tool() {
    return json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_function_name", json::object({{"param1", "value1"}, {"param2", "value2"}}))
        })}
    };
}

static json make_assistant_two_tools() {
    return json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_function_name", json::object({{"param1", "value1"}, {"param2", "value2"}})),
            build_tool_call("test_function_name", json::object({{"param1", "value3"}, {"param2", "value4"}}), "call_002")
        })}
    };
}

static json make_assistant_no_reasoning() {
    return json{
        {"role", "assistant"},
        {"content", "I can help you with that."}
    };
}

static json make_assistant_with_reasoning() {
    return json{
        {"role", "assistant"},
        {"content", "I can help you with that."},
        {"reasoning_content", "The user is asking for help. I should respond positively."}
    };
}

static json make_assistant_one_tool_with_reasoning() {
    return json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({
            build_tool_call("test_function_name", json::object({{"param1", "value1"}, {"param2", "value2"}}))
        })},
        {"reasoning_content", "I need to call the tool first."}
    };
}

static void print_diff_split(const std::string & title, const diff_split & diff) {
    LOG_ERR("\n%s=== %s ===%s\n", ANSI_CYAN, title.c_str(), ANSI_RESET);
    LOG_ERR("%sCommon Prefix:%s '%s'\n", ANSI_PREFIX, ANSI_RESET, diff.prefix.c_str());
    LOG_ERR("%sCommon Suffix:%s '%s'\n", ANSI_SUFFIX, ANSI_RESET, diff.suffix.c_str());
    LOG_ERR("%sLeft (difference):%s '%s'\n", ANSI_GREEN, ANSI_RESET, diff.left.c_str());
    LOG_ERR("%sRight (difference):%s '%s'\n", ANSI_ORANGE, ANSI_RESET, diff.right.c_str());
}

static void check_reasoning_variables(const common_chat_template & tmpl) {
    LOG_ERR("\n%s=== Checking Reasoning Variables ===%s\n", ANSI_CYAN, ANSI_RESET);

    try {
        // Create a list of candidate reasoning/thinking variable names to probe
        std::vector<std::string> candidate_vars = {
            "enable_reasoning",
            "use_reasoning",
            "reasoning_enabled",
            "has_reasoning",
            "reasoning_mode",
            "reasoning_format",
            "reasoning_active",
            "with_reasoning",
            "use_thinking",
            "thinking_enabled",
            "has_thinking",
            "thinking_mode",
            "thinking_format",
            "thinking_active",
            "with_thinking",
            "enable_reason",
            "reason_enabled",
            "enable_think",
            "think_enabled",
        };

        jinja::context ctx;
        ctx.is_get_stats = true;

        json messages = json::array({
            json{
                {"role", "user"},
                {"content", "Test message"}
            },
            json{
                {"role", "assistant"},
                {"content", "Response"},
                {"reasoning_content", "Some reasoning"}
            }
        });

        // Set up base context
        jinja::global_from_json(ctx, json{
            {"messages", messages},
            {"tools", json::array()},
            {"bos_token", ""},
            {"eos_token", ""},
            {"add_generation_prompt", false},
            {"enable_thinking", true}  // Already passed, so we'll exclude this from results
        }, true);

        // Add candidate variables as undefined to probe which ones are accessed
        for (const auto & var_name : candidate_vars) {
            ctx.set_val(var_name, jinja::mk_val<jinja::value_undefined_t>(var_name));
        }

        try {
            jinja::runtime runtime(ctx);
            runtime.execute(tmpl.prog);
        } catch (const std::exception & e) {
            // Execution may fail, that's okay - we just want to see what variables were accessed
        }

        // Check which candidate variables were accessed (stats.used = true)
        std::vector<std::string> accessed_vars;
        for (const auto & var_name : candidate_vars) {
            auto val = ctx.get_val(var_name);
            if (!val->is_undefined()) {
                // Variable was overwritten, skip it
                continue;
            }
            if (val->stats.used) {
                accessed_vars.push_back(var_name);
            }
        }

        if (accessed_vars.empty()) {
            LOG_ERR("%sNo reasoning/thinking-related variables were queried by the template%s\n", ANSI_GRAY, ANSI_RESET);
        } else {
            LOG_ERR("Template queries the following reasoning/thinking-related variables:\n");
            for (const auto & var : accessed_vars) {
                LOG_ERR("  %s- %s%s\n", ANSI_ORANGE, var.c_str(), ANSI_RESET);
            }
        }

    } catch (const std::exception & e) {
        LOG_ERR("Error checking reasoning variables: %s\n", e.what());
    }
}

static void analyze_template(const std::string & template_path) {
    LOG_ERR("\n");
    LOG_ERR("%s", ANSI_PURPLE);
    LOG_ERR("================================================================================\n");
    LOG_ERR("                    ANALYZING TEMPLATE: %s\n", template_path.c_str());
    LOG_ERR("================================================================================\n");
    LOG_ERR("%s", ANSI_RESET);

    std::string template_source;
    try {
        template_source = read_file(template_path);
    } catch (const std::exception & e) {
        LOG_ERR("Error reading template: %s\n", e.what());
        return;
    }

    try {
        common_chat_template chat_template(template_source, "", "");
        json tools = build_tools_definition();

        // ===== CAPABILITIES ANALYSIS =====
        LOG_ERR("\n%s=== Template Capabilities (from jinja::caps) ===%s\n", ANSI_CYAN, ANSI_RESET);
        auto caps = chat_template.original_caps();
        LOG_ERR("%ssupports_tools:%s %s\n", ANSI_BLUE, ANSI_RESET, caps.supports_tools ? "true" : "false");
        LOG_ERR("%ssupports_tool_calls:%s %s\n", ANSI_BLUE, ANSI_RESET, caps.supports_tool_calls ? "true" : "false");
        LOG_ERR("%ssupports_system_role:%s %s\n", ANSI_BLUE, ANSI_RESET, caps.supports_system_role ? "true" : "false");
        LOG_ERR("%ssupports_parallel_tool_calls:%s %s\n", ANSI_BLUE, ANSI_RESET, caps.supports_parallel_tool_calls ? "true" : "false");
        LOG_ERR("%ssupports_typed_content:%s %s\n", ANSI_BLUE, ANSI_RESET, caps.supports_typed_content ? "true" : "false");
        LOG_ERR("%ssupports_string_content:%s %s\n", ANSI_BLUE, ANSI_RESET, caps.supports_string_content ? "true" : "false");

        // ===== DIFFERENTIAL ANALYSIS =====

        // Test 1: With and without tools (single user message)
        {
            json user_msg = make_user_msg();

            autoparser::templates_params params_no_tools;
            params_no_tools.messages = json::array({ user_msg });
            params_no_tools.add_generation_prompt = false;
            params_no_tools.tools = json::array();

            autoparser::templates_params params_with_tools = params_no_tools;
            params_with_tools.tools = tools;

            std::string output_no_tools = common_chat_template_direct_apply(chat_template, params_no_tools);
            std::string output_with_tools = common_chat_template_direct_apply(chat_template, params_with_tools);

            auto diff = calculate_diff_split(output_no_tools, output_with_tools);
            print_diff_split("Diff: With vs Without Tools (single user message)", diff);
        }

        // Test 2: With and without add_generation_prompt (single user message)
        {
            json user_msg = make_user_msg();

            autoparser::templates_params params_no_prompt;
            params_no_prompt.messages = json::array({ user_msg });
            params_no_prompt.add_generation_prompt = false;
            params_no_prompt.tools = json::array();

            autoparser::templates_params params_with_prompt = params_no_prompt;
            params_with_prompt.add_generation_prompt = true;

            std::string output_no_prompt = common_chat_template_direct_apply(chat_template, params_no_prompt);
            std::string output_with_prompt = common_chat_template_direct_apply(chat_template, params_with_prompt);

            auto diff = calculate_diff_split(output_no_prompt, output_with_prompt);
            print_diff_split("Diff: With vs Without add_generation_prompt (single user message)", diff);
        }

        // Test 3: Assistant with reasoning_content (user, assistant)
        {
            json user_msg = make_user_msg();

            autoparser::templates_params params_no_reasoning;
            params_no_reasoning.messages = json::array({ user_msg, make_assistant_no_reasoning() });
            params_no_reasoning.add_generation_prompt = false;
            params_no_reasoning.enable_thinking = true;

            autoparser::templates_params params_with_reasoning = params_no_reasoning;
            params_with_reasoning.messages = json::array({ user_msg, make_assistant_with_reasoning() });

            std::string output_no_reasoning = common_chat_template_direct_apply(chat_template, params_no_reasoning);
            std::string output_with_reasoning = common_chat_template_direct_apply(chat_template, params_with_reasoning);

            auto diff = calculate_diff_split(output_no_reasoning, output_with_reasoning);
            print_diff_split("Diff: With vs Without reasoning_content (user, assistant)", diff);
        }

        // Test 4: Assistant with reasoning_content (user, assistant, user)
        {
            json user_msg = make_user_msg();
            json user_msg2 = make_user_msg2();

            autoparser::templates_params params_no_reasoning;
            params_no_reasoning.messages = json::array({ user_msg, make_assistant_no_reasoning(), user_msg2 });
            params_no_reasoning.add_generation_prompt = false;
            params_no_reasoning.enable_thinking = true;

            autoparser::templates_params params_with_reasoning = params_no_reasoning;
            params_with_reasoning.messages = json::array({ user_msg, make_assistant_with_reasoning(), user_msg2 });

            std::string output_no_reasoning = common_chat_template_direct_apply(chat_template, params_no_reasoning);
            std::string output_with_reasoning = common_chat_template_direct_apply(chat_template, params_with_reasoning);

            auto diff = calculate_diff_split(output_no_reasoning, output_with_reasoning);
            print_diff_split("Diff: With vs Without reasoning_content (user, assistant, user)", diff);
        }

        // Test 5: Tool call in last assistant message (user, assistant)
        {
            json user_msg = make_user_msg();

            autoparser::templates_params params_no_tool;
            params_no_tool.messages = json::array({ user_msg, make_assistant_no_tool() });
            params_no_tool.add_generation_prompt = false;
            params_no_tool.tools = tools;

            autoparser::templates_params params_with_tool = params_no_tool;
            params_with_tool.messages = json::array({ user_msg, make_assistant_one_tool() });

            std::string output_no_tool = common_chat_template_direct_apply(chat_template, params_no_tool);
            std::string output_with_tool = common_chat_template_direct_apply(chat_template, params_with_tool);

            auto diff = calculate_diff_split(output_no_tool, output_with_tool);
            print_diff_split("Diff: With vs Without tool call (user, assistant)", diff);
        }

        // Test 6: Tool call in last assistant message (user, assistant, user)
        {
            json user_msg = make_user_msg();
            json user_msg2 = make_user_msg2_continue();

            autoparser::templates_params params_no_tool;
            params_no_tool.messages = json::array({ user_msg, make_assistant_no_tool(), user_msg2 });
            params_no_tool.add_generation_prompt = false;
            params_no_tool.tools = tools;

            autoparser::templates_params params_with_tool = params_no_tool;
            params_with_tool.messages = json::array({ user_msg, make_assistant_one_tool(), user_msg2 });

            std::string output_no_tool = common_chat_template_direct_apply(chat_template, params_no_tool);
            std::string output_with_tool = common_chat_template_direct_apply(chat_template, params_with_tool);

            auto diff = calculate_diff_split(output_no_tool, output_with_tool);
            print_diff_split("Diff: With vs Without tool call (user, assistant, user)", diff);
        }

        // Test 7: One vs two tool calls (user, assistant)
        {
            json user_msg = make_user_msg();

            autoparser::templates_params params_one_tool;
            params_one_tool.messages = json::array({ user_msg, make_assistant_one_tool() });
            params_one_tool.add_generation_prompt = false;
            params_one_tool.tools = tools;

            autoparser::templates_params params_two_tools = params_one_tool;
            params_two_tools.messages = json::array({ user_msg, make_assistant_two_tools() });

            std::string output_one_tool = common_chat_template_direct_apply(chat_template, params_one_tool);
            std::string output_two_tools = common_chat_template_direct_apply(chat_template, params_two_tools);

            auto diff = calculate_diff_split(output_one_tool, output_two_tools);
            print_diff_split("Diff: One vs Two tool calls (user, assistant)", diff);
        }

        // Test 8: One vs two tool calls (user, assistant, user)
        {
            json user_msg = make_user_msg();
            json user_msg2 = make_user_msg2_continue();

            autoparser::templates_params params_one_tool;
            params_one_tool.messages = json::array({ user_msg, make_assistant_one_tool(), user_msg2 });
            params_one_tool.add_generation_prompt = false;
            params_one_tool.tools = tools;

            autoparser::templates_params params_two_tools = params_one_tool;
            params_two_tools.messages = json::array({ user_msg, make_assistant_two_tools(), user_msg2 });

            std::string output_one_tool = common_chat_template_direct_apply(chat_template, params_one_tool);
            std::string output_two_tools = common_chat_template_direct_apply(chat_template, params_two_tools);

            auto diff = calculate_diff_split(output_one_tool, output_two_tools);
            print_diff_split("Diff: One vs Two tool calls (user, assistant, user)", diff);
        }

        // Test 9: Tool call with vs without reasoning_content (user, assistant)
        {
            json user_msg = make_user_msg();

            autoparser::templates_params params_no_reasoning;
            params_no_reasoning.messages = json::array({ user_msg, make_assistant_one_tool() });
            params_no_reasoning.add_generation_prompt = false;
            params_no_reasoning.tools = tools;
            params_no_reasoning.enable_thinking = true;

            autoparser::templates_params params_with_reasoning = params_no_reasoning;
            params_with_reasoning.messages = json::array({ user_msg, make_assistant_one_tool_with_reasoning() });

            std::string output_no_reasoning = common_chat_template_direct_apply(chat_template, params_no_reasoning);
            std::string output_with_reasoning = common_chat_template_direct_apply(chat_template, params_with_reasoning);

            auto diff = calculate_diff_split(output_no_reasoning, output_with_reasoning);
            print_diff_split("Diff: Tool call with vs without reasoning_content (user, assistant)", diff);
        }

        // Check reasoning variables
        check_reasoning_variables(chat_template);

    } catch (const std::exception & e) {
        LOG_ERR("Analysis failed: %s\n", e.what());
    }
}

int main(int argc, char ** argv) {
    // Set log level to capture all output
    common_log_set_verbosity_thold(99);

    analysis_options opts;
    if (!parse_options(argc, argv, opts)) {
        return 1;
    }

    LOG_ERR("\n");
    LOG_ERR("%s", ANSI_PURPLE);
    LOG_ERR("================================================================================\n");
    LOG_ERR("                      TEMPLATE ANALYSIS TOOL\n");
    LOG_ERR("================================================================================\n");
    LOG_ERR("%s", ANSI_RESET);
    LOG_ERR("Analyzing %s%zu%s template(s)\n", ANSI_CYAN, opts.template_paths.size(), ANSI_RESET);

    for (const auto & path : opts.template_paths) {
        analyze_template(path);
    }

    LOG_ERR("\n");
    LOG_ERR("%s", ANSI_GREEN);
    LOG_ERR("================================================================================\n");
    LOG_ERR("                      ANALYSIS COMPLETE\n");
    LOG_ERR("================================================================================\n");
    LOG_ERR("%s", ANSI_RESET);

    return 0;
}
