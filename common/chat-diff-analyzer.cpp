#include "chat-diff-analyzer.h"

#include "chat-auto-parser-helpers.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include "peg-parser.h"

#include <algorithm>

#define ANSI_RESET  "\033[0m"
#define ANSI_PURPLE "\033[1m\x1b[38;5;126m"
#define ANSI_ORANGE "\033[1m\x1b[38;5;214m"
#define ANSI_RED    "\033[1m\x1b[38;5;196m"

using json = nlohmann::ordered_json;

namespace autoparser {

static const std::string FUN_FIRST = "FFF_FIRST_FUN_F";
static const std::string FUN_SECOND = "SSS_SECOND_FUN_S";
static const std::string ARG_FIRST = "AA_ARG_FST_AA";
static const std::string ARG_SECOND = "BB_ARG_SND_BB";
static const std::string USER_MSG = "U_USER_MSG Hello END_U";
static const std::string ASSISTANT_MSG = "A_ASST_MSG I can help END_A";
static const std::string THINKING_CONTENT = "REASON_PART I am thinking END_R";

static std::vector<std::function<void(const common_chat_template & tmpl, analyze_template &)>> workarounds(
    { // Old reasoning Qwen templates - they don't really display reasoning content, but we still want to
      // support reasoning on them
      [](const common_chat_template & tmpl, analyze_template & analysis) -> void {
          if (tmpl.src.find("content.split('</think>')") != std::string::npos &&
              analysis.reasoning.mode == reasoning_mode::NONE) {
              analysis.reasoning.mode  = reasoning_mode::FORCED_OPEN;
              analysis.reasoning.start = "<think>";
              analysis.reasoning.end   = "</think>";
              analysis.preserved_tokens.push_back("<think>");
              analysis.preserved_tokens.push_back("</think>");
              LOG_DBG(ANSI_ORANGE "[Patch: old Qwen/Deepseek thinking template]\n" ANSI_RESET);
          }
      },
      // Granite 3.3, with separate reasoning and content markers
      [](const common_chat_template & tmpl, analyze_template & analysis) -> void {
          if (tmpl.src.find("Write your thoughts between <think></think> and write your response between "
                            "<response></response>") != std::string::npos) {
              analysis.reasoning.mode  = reasoning_mode::TAG_BASED;
              analysis.reasoning.start = "<think>";
              analysis.reasoning.end   = "</think>";
              analysis.preserved_tokens.push_back("<think>");
              analysis.preserved_tokens.push_back("</think>");
              analysis.content.mode  = content_mode::WRAPPED_WITH_REASONING;
              analysis.content.start = "<response>";
              analysis.content.end   = "</response>";
              analysis.preserved_tokens.push_back("<response>");
              analysis.preserved_tokens.push_back("</response>");
              LOG_DBG(ANSI_ORANGE "[Patch: Granite 3.3]\n" ANSI_RESET);
          }
      },
      // Cohere Command R+ - content wrapped in <|CHATBOT_TOKEN|>...<|END_OF_TURN_TOKEN|>
      [](const common_chat_template & tmpl, analyze_template & analysis) -> void {
          if (tmpl.src.find("<|CHATBOT_TOKEN|>") != std::string::npos &&
              tmpl.src.find("<|END_OF_TURN_TOKEN|>") != std::string::npos && analysis.content.start.empty()) {
              analysis.content.mode  = content_mode::ALWAYS_WRAPPED;
              analysis.content.start = "<|CHATBOT_TOKEN|>";
              analysis.content.end   = "<|END_OF_TURN_TOKEN|>";
              analysis.preserved_tokens.push_back("<|CHATBOT_TOKEN|>");
              analysis.preserved_tokens.push_back("<|END_OF_TURN_TOKEN|>");
              LOG_DBG(ANSI_ORANGE "[Patch: Cohere Command R+]\n" ANSI_RESET);
          }
      },
      // Functionary - no tool call section delimiter
      [](const common_chat_template & tmpl, analyze_template & analysis) -> void {
          if (tmpl.src.find("set has_code_interpreter = tools | selectattr(\"type\", \"equalto\", "
                            "\"code_interpreter\") | list | length > 0") != std::string::npos) {
              analysis.content.mode                = content_mode::PLAIN;
              analysis.content.end                 = "";
              analysis.tools.function.name_prefix  = "";
              analysis.tools.format.section_start  = "";
              analysis.tools.format.section_end    = "";
              analysis.tools.format.per_call_start = "<function=";
              analysis.tools.format.per_call_end   = "</function>";
              analysis.tools.function.close        = "";
              analysis.preserved_tokens.clear();
              analysis.preserved_tokens.push_back("<|eot_id|>");
              analysis.preserved_tokens.push_back("<|eom_id|>");
              analysis.preserved_tokens.push_back("<function=");
              analysis.preserved_tokens.push_back(">");
              analysis.preserved_tokens.push_back("</function>");
              LOG_DBG(ANSI_ORANGE "[Patch: Functionary 3.1]\n" ANSI_RESET);
          }
      },
      // DeepSeek-R1-Distill-Qwen
      [](const common_chat_template & tmpl, analyze_template & analysis) -> void {
          if (tmpl.src.find(
                  "{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>'") !=
              std::string::npos) {
              analysis.tools.format.section_start  = "<｜tool▁calls▁begin｜>";
              analysis.tools.format.section_end    = "<｜tool▁calls▁end｜>";
              analysis.tools.format.per_call_start = "<｜tool▁call▁begin｜>function";
              analysis.tools.function.name_prefix  = "<｜tool▁sep｜>";
              analysis.tools.format.per_call_end   = "<｜tool▁call▁end｜>";
              analysis.tools.function.close        = "```";
          }
      },
      // Seed - Python dicts
      [](const common_chat_template & tmpl, analyze_template & analysis) -> void {
          if (tmpl.src.find("<seed:bos>") != std::string::npos) {
              analysis.tools.format.uses_python_dicts = true;
          }
      },
    });

// Common JSON structures
static json params_schema = {
    { "type",       "object"                                                           },
    { "properties",
     { { ARG_FIRST, { { "type", "string" }, { "description", "First argument" } } },
        { ARG_SECOND, { { "type", "string" }, { "description", "Second argument" } } } } },
    { "required",   json::array({})                                                    }
};

static json tools = json::array({
    { { "type", "function" },
     { "function",
        json{ { "name", FUN_FIRST }, { "description", "Test function foo" }, { "parameters", params_schema } } } },
    { { "type", "function" },
     { "function",
        json{ { "name", FUN_SECOND }, { "description", "Test function bar" }, { "parameters", params_schema } } } }
});

static json user_msg = json{
    { "role",    "user"  },
    { "content", USER_MSG }
};

static json build_tool_call(const std::string & name, const json & args, const std::string & id = "call00001") {
    return json{
        { "id",       id                                              },
        { "type",     "function"                                      },
        { "function", json{ { "name", name }, { "arguments", args } } }
    };
}

static json first_tool_call_zero_args         = build_tool_call(FUN_FIRST, json::object(), "call00001");
static json first_tool_call_one_arg           = build_tool_call(FUN_FIRST, {{ ARG_FIRST, "XXXX" }}, "call00001");
static json first_tool_call_one_arg_other_val = build_tool_call(FUN_FIRST, {{ ARG_FIRST, "YYYY" }}, "call00001");
static json first_tool_call_other_arg         = build_tool_call(FUN_FIRST, {{ ARG_SECOND, "YYYY" }}, "call00001");

static json first_tool_call =
    build_tool_call(FUN_FIRST, json{{ ARG_FIRST,  "XXXX" }, { ARG_SECOND, "YYYY" }}, "call00001");
static json second_tool_call =
    build_tool_call(FUN_SECOND, json{ { ARG_FIRST,  "XXXX" }, { ARG_SECOND, "YYYY" }}, "call00002");
static json first_tool_call_alt_id =
    build_tool_call(FUN_FIRST, json{{ ARG_FIRST,  "XXXX" }, { ARG_SECOND, "YYYY" }}, "call99999");

template <typename T>
static std::string mode_to_str(T mode) {
    std::ostringstream os;
    os << mode;
    return os.str();
}

analyze_template::analyze_template(const common_chat_template & tmpl)
    : jinja_caps(tmpl.original_caps())
    , reasoning(tmpl, jinja_caps.supports_tool_calls)
    , content(tmpl, reasoning)
    , tools(jinja_caps.supports_tool_calls ? analyze_tools(tmpl, jinja_caps, reasoning) : analyze_tools())
{
    collect_preserved_tokens();

    for (auto & workaround : workarounds) {
        workaround(tmpl, *this);
    }

    LOG_DBG("\n--- Reasoning & Content Structure ---\n");
    LOG_DBG("reasoning_mode: %s\n", mode_to_str(reasoning.mode).c_str());
    LOG_DBG("reasoning_start: '%s'\n", reasoning.start.c_str());
    LOG_DBG("reasoning_end: '%s'\n", reasoning.end.c_str());
    LOG_DBG("content_mode: %s\n", mode_to_str(content.mode).c_str());
    LOG_DBG("content_start: '%s'\n", content.start.c_str());
    LOG_DBG("content_end: '%s'\n", content.end.c_str());

    LOG_DBG("\n--- Tool Call Structure ---\n");
    LOG_DBG("tool_mode: %s\n", mode_to_str(tools.format.mode).c_str());
    LOG_DBG("supports_tools: %s\n", jinja_caps.supports_tools ? "true" : "false");
    LOG_DBG("supports_parallel_calls: %s\n", jinja_caps.supports_parallel_tool_calls ? "true" : "false");
    LOG_DBG("tool_section_start: '%s'\n", tools.format.section_start.c_str());
    LOG_DBG("tool_section_end: '%s'\n", tools.format.section_end.c_str());
    LOG_DBG("per_call_start: '%s'\n", tools.format.per_call_start.c_str());
    LOG_DBG("per_call_end: '%s'\n", tools.format.per_call_end.c_str());
    LOG_DBG("func_name_prefix: '%s'\n", tools.function.name_prefix.c_str());
    LOG_DBG("func_name_suffix: '%s'\n", tools.function.name_suffix.c_str());
    LOG_DBG("func_close: '%s'\n", tools.function.close.c_str());
    LOG_DBG("python_dict_format: %s\n", tools.format.uses_python_dicts ? "true" : "false");
    LOG_DBG("arg_name_prefix: '%s'\n", tools.arguments.name_prefix.c_str());
    LOG_DBG("arg_name_suffix: '%s'\n", tools.arguments.name_suffix.c_str());
    LOG_DBG("arg_value_prefix: '%s'\n", tools.arguments.value_prefix.c_str());
    LOG_DBG("arg_value_suffix: '%s'\n", tools.arguments.value_suffix.c_str());
    LOG_DBG("name_field: '%s'\n", tools.format.name_field.c_str());
    LOG_DBG("args_field: '%s'\n", tools.format.args_field.c_str());
    LOG_DBG("id_field: '%s'\n", tools.format.id_field.c_str());
    LOG_DBG("gen_id_field: '%s'\n", tools.format.gen_id_field.c_str());
    LOG_DBG("parameter_order: '%s'\n", std::accumulate(tools.format.parameter_order.begin(), tools.format.parameter_order.end(),
        std::string(""), [] (const std::string & a, const std::string & b) { return a.empty() ? b : a + ", " + b; }
        ).c_str());

    LOG_DBG(ANSI_PURPLE "=== Differential analysis complete ===\n" ANSI_RESET);
}

void analyze_template::collect_preserved_tokens() {
    auto add_token = [this](const std::string & org_token) {
        std::string token = trim_whitespace(org_token);
        if (!token.empty()) {
            // Avoid duplicates
            if (std::find(preserved_tokens.begin(), preserved_tokens.end(), token) == preserved_tokens.end()) {
                preserved_tokens.push_back(token);
            }
        }
    };

    add_token(reasoning.start);
    add_token(reasoning.end);
    add_token(content.start);
    add_token(content.end);
    add_token(tools.format.section_start);
    add_token(tools.format.section_end);
    add_token(tools.format.per_call_start);
    add_token(tools.format.per_call_end);
    add_token(tools.function.name_prefix);
    add_token(tools.function.name_suffix);
    add_token(tools.function.close);
    add_token(tools.arguments.start);
    add_token(tools.arguments.end);
    add_token(tools.arguments.name_prefix);
    add_token(tools.arguments.name_suffix);
    add_token(tools.arguments.separator);
    add_token(tools.arguments.value_prefix);
    add_token(tools.arguments.value_suffix);
    add_token(tools.call_id.prefix);
    add_token(tools.call_id.suffix);
}

analyze_reasoning::analyze_reasoning(const common_chat_template & tmpl, bool supports_tools)
    : analyze_base(tmpl) {
    LOG_DBG(ANSI_PURPLE "=== Starting differential analysis ===\n" ANSI_RESET);
    LOG_DBG(ANSI_ORANGE "Phase 1: Reasoning analysis\n" ANSI_RESET);

    compare_reasoning_presence();
    compare_thinking_enabled();
    if (supports_tools) {
        compare_reasoning_scope();
    }
}

void analyze_reasoning::compare_reasoning_presence() {
    json user_msg = json{
        { "role",    "user"  },
        { "content", USER_MSG }
    };

    json assistant_no_reasoning = json{
        { "role",    "assistant"   },
        { "content", ASSISTANT_MSG }
    };

    json assistant_with_reasoning = json{
        { "role",              "assistant"                },
        { "content",           ASSISTANT_MSG              },
        { "reasoning_content", THINKING_CONTENT           }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_no_reasoning });
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_reasoning }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed, skipping reasoning detection\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    const std::string reasoning_content = THINKING_CONTENT;

    if (!diff.right.empty() && diff.right.find(reasoning_content) != std::string::npos) {
        auto parser_delimiter = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.literal(reasoning_content) + p.space() + p.optional(p.tag("post", (p.marker() + p.space())) + p.rest());
        });
        auto parser_wrapped = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.marker()) + p.space() + p.literal(reasoning_content) + p.space() + p.tag("post", (p.marker() + p.space())) + p.rest();
        });
        // try the more aggressive parse first, if it fails, fall back to the delimiter one
        auto result = parser_wrapped.parse_anywhere_and_extract(comparison->output_B);
        if (!result.result.success()) {
            result = parser_delimiter.parse_anywhere_and_extract(comparison->output_B);
        }
        if (result.result.success()) {
            if (!result.tags["pre"].empty() && !result.tags["post"].empty()) {
                if (parser_wrapped.parse_anywhere_and_extract(diff.right).result.success()) { // both tags in the diff = no forced close
                    mode = reasoning_mode::TAG_BASED;
                } else {
                    mode = reasoning_mode::FORCED_CLOSED;
                }
                start = trim_whitespace(result.tags["pre"]);
                end   = result.tags["post"];
            } else if (!result.tags["post"].empty()) {
                mode = reasoning_mode::DELIMITER;
                end = result.tags["post"];
            }
        }
    }
}

void analyze_reasoning::compare_thinking_enabled() {
    json user_msg = json{
        { "role",    "user"  },
        { "content", USER_MSG }
    };

    template_params params;
    params.messages              = json::array({ user_msg });
    params.add_generation_prompt = true;
    params.enable_thinking       = false;

    auto comparison = compare_variants(*tmpl, params, [&](template_params & p) { p.enable_thinking = true; });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET , __func__);
        return;
    }

    const auto & diff = comparison->diff;

    std::string left_trimmed = trim_whitespace(diff.left);

    if (left_trimmed.empty() && !diff.right.empty()) {
        std::string right_trimmed = trim_whitespace(diff.right);

        if (!right_trimmed.empty() && string_ends_with(comparison->output_B, right_trimmed)) {
            if (start.empty()) {
                start = right_trimmed;
                mode  = reasoning_mode::FORCED_OPEN;
            }
        }
    }

    if (start.empty() && !end.empty()) {
        mode = reasoning_mode::DELIMITER;
    }

    // Check for FORCED_CLOSED: when enable_thinking=false produces both start and end markers,
    // but enable_thinking=true produces only the start marker
    if (!comparison->output_A.empty() && !comparison->output_B.empty()) {
        auto parser_start = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.literal(start) + p.space() + p.literal(end) + p.rest();
        });
        auto parser_start_end = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.literal(start)) + p.space() + p.negate(p.literal(end)) + p.rest();
        });
        if (!start.empty() && parser_start_end.parse_anywhere_and_extract(comparison->output_A).result.success() &&
            parser_start.parse_anywhere_and_extract(comparison->output_B).result.success()) {
            mode = reasoning_mode::FORCED_CLOSED;
        } else if (!end.empty()) { // we extract the starting marker now since we didn't get it earlier
            auto result = parser_start_end.parse_anywhere_and_extract(comparison->output_A);
            if (result.result.success()) {
                start = result.tags["pre"];
                mode  = reasoning_mode::FORCED_CLOSED;
            }
        }
    }

    if (start.empty() && end.empty()) {  // we might still have the case of "just open" and "just close"
        if (!diff.left.empty() && !diff.right.empty()) {
            auto seg_A = segmentize_markers(trim_trailing_whitespace(diff.left));
            auto seg_B = segmentize_markers(trim_trailing_whitespace(diff.right));
            if (seg_A.size() == 1 && seg_B.size() == 1) {
                mode = reasoning_mode::FORCED_CLOSED;
                start = seg_B[0].value;
                end = seg_A[0].value;
            }
        }
    }
}

void analyze_reasoning::compare_reasoning_scope() {
    json assistant_reasoning_content = json{
        { "role",              "assistant"      },
        { "content",           ASSISTANT_MSG    },
        { "reasoning_content", THINKING_CONTENT }
    };

    json assistant_reasoning_tools = json{
        { "role",              "assistant"                                                                  },
        { "content",           nullptr                                                                      },
        { "reasoning_content", THINKING_CONTENT                                                             },
        { "tool_calls",
         json::array({ build_tool_call(FUN_FIRST, json{ { ARG_FIRST, "VVVV" }, { ARG_SECOND, "XXXX" } }) }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_reasoning_content });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_reasoning_tools }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    std::string reasoning_content = THINKING_CONTENT;

    // Check if reasoning only appears in variant B (with tools)
    bool reasoning_in_A = comparison->output_A.find(reasoning_content) != std::string::npos;
    bool reasoning_in_B = comparison->output_B.find(reasoning_content) != std::string::npos;

    if (!reasoning_in_A && reasoning_in_B) {
        mode = reasoning_mode::TOOLS_ONLY;
        LOG_DBG(ANSI_ORANGE "%s: Detected TOOLS_ONLY reasoning mode\n" ANSI_RESET, __func__);

        auto parser_wrapped = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.marker()) + p.space() + p.literal(reasoning_content) + p.space() + p.tag("post", (p.marker() + p.space()));
        });
        auto result = parser_wrapped.parse_anywhere_and_extract(comparison->output_B);
        if (result.result.success()) {
            start = result.tags["pre"];
            end = result.tags["post"];
        } else {
            auto parser_delimiter = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                return p.literal(reasoning_content) + p.space() + p.optional(p.tag("post", (p.marker() + p.space())));
            });
            result = parser_delimiter.parse_anywhere_and_extract(comparison->output_B);
            if (result.result.success()) {
                end = result.tags["post"];
            } else {
                LOG_DBG(ANSI_ORANGE "%s: Unable to extracft reasoning markers, falling back to reasoning = NONE\n" ANSI_RESET, __func__);
                mode = reasoning_mode::NONE;
            }
        }
    }
}

analyze_content::analyze_content(const common_chat_template & tmpl, const analyze_reasoning & reasoning)
    : analyze_base(tmpl) {
    LOG_DBG(ANSI_ORANGE "Phase 2: Content analysis\n" ANSI_RESET);

    json assistant_content_only = json{
        { "role",    "assistant"     },
        { "content", ASSISTANT_MSG   }
    };

    json assistant_with_tools = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ build_tool_call("test_func", json{ { "arg1", "value1" } }) }) }
    };

    json assistant_with_reasoning = json{
        { "role",              "assistant"      },
        { "content",           ""               },
        { "reasoning_content", THINKING_CONTENT }
    };

    template_params params_content_only;
    params_content_only.messages              = json::array({ user_msg, assistant_content_only });
    params_content_only.add_generation_prompt = false;
    params_content_only.enable_thinking       = true;
    params_content_only.tools                 = tools;

    auto comparison_with_tools = compare_variants(tmpl, params_content_only, [&](template_params & p) {
        p.messages = json::array({ user_msg, assistant_with_tools });
    });

    auto comparison_with_reasoning = compare_variants(tmpl, params_content_only, [&](template_params & p) {
        p.messages = json::array({ user_msg, assistant_with_reasoning });
    });

    if (!comparison_with_tools || !comparison_with_reasoning) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
    }

    const auto & diff_tools     = comparison_with_tools->diff;
    const auto & diff_reasoning = comparison_with_reasoning->diff;

    std::string response = ASSISTANT_MSG;

    bool found_plain_content = false;
    if (trim_whitespace(diff_tools.left) == response) {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.space() + diff_reasoning.left + p.space() + p.optional(p.marker()) + p.space() + p.end();
        });
        if (parser.parse_and_extract(diff_reasoning.left).result.success()) {
            // We only have the content text in the diff (possibly with a stray EOG marker), so no markers
            mode = content_mode::PLAIN;
            found_plain_content = true;
        } else if (reasoning.mode != reasoning_mode::NONE && !reasoning.end.empty()) {
            auto post_reasoning_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.literal(reasoning.end) + p.space() + p.literal(response);
            });
            if (post_reasoning_parser.parse_anywhere_and_extract(diff_reasoning.left).result.success()) {
                mode = content_mode::PLAIN;
                found_plain_content = true;
            }
        }
    }
    if (!found_plain_content) {
        std::string rdiff = diff_reasoning.left;
        if (!reasoning.end.empty() && rdiff.find(reasoning.end) != std::string::npos) {
            rdiff = rdiff.substr(rdiff.find(reasoning.end) + reasoning.end.length());
        }
        // Take the more promising diff
        std::string pure_content = rdiff.length() > diff_tools.left.length() ? rdiff : diff_tools.left;
        auto parser_wrapped = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.marker()) + p.space() + p.literal(response) + p.space() + p.tag("post", (p.marker() + p.space())) + p.rest();
        });
        auto result = parser_wrapped.parse_anywhere_and_extract(pure_content);
        start = result.tags["pre"];
        end = result.tags["post"];
        // TODO: WRAPPED_WITH_REASONING
    }

    // Determine content mode
    if (!start.empty() || !end.empty()) {
        mode = content_mode::ALWAYS_WRAPPED;
        // TODO: END_DELIMITED content mode - delimited at end but not at start?
    }
}

bool analyze_content::is_always_wrapped() const {
    return mode == content_mode::ALWAYS_WRAPPED && !start.empty() && !end.empty();
}

analyze_tools::analyze_tools(const common_chat_template & tmpl,
                             const jinja::caps &          caps,
                             const analyze_reasoning &    reasoning)
    : analyze_base(tmpl) {
    LOG_DBG(ANSI_ORANGE "Phase 3: Tool call analysis\n" ANSI_RESET);

    analyze_tool_calls(reasoning);

    if (format.mode != tool_format::NONE && format.mode != tool_format::JSON_NATIVE) {
        if (caps.supports_parallel_tool_calls) {
            check_per_call_markers();
        }
        extract_function_markers();
        if (format.mode == tool_format::TAG_WITH_TAGGED) {
            analyze_arguments();
        }
        extract_argument_separator();
        extract_args_markers();
        extract_call_id_markers();
    }
}

void analyze_tools::analyze_tool_calls(const analyze_reasoning & reasoning) {
    json assistant_no_tools = json{
        { "role",    "assistant"   },
        { "content", ASSISTANT_MSG }
    };

    json assistant_with_tools = json{
        { "role",       "assistant"                      },
        { "content",    ""                               },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_no_tools });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_tools }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    std::string tool_section = diff.right;

    if (tool_section.empty()) {
        return;
    }

    analyze_tool_call_format(tool_section, FUN_FIRST, ARG_FIRST, reasoning);
}

void analyze_tools::analyze_tool_call_format(const std::string &       haystack,
                                             const std::string &       fun_name_needle,
                                             const std::string &       arg_name_needle,
                                             const analyze_reasoning & reasoning) {
    if (fun_name_needle.empty() || arg_name_needle.empty() || haystack.empty()) {
        return;
    }

    enum class json_quote_style { NONE, DOUBLE_QUOTES, SINGLE_QUOTES };

    auto in_json_haystack = [&haystack](const std::string & needle) -> json_quote_style {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.choice({ p.literal("{"), p.literal(":") }) << p.choice({
                p.tag("sq", p.literal("'") + p.literal(needle) + p.literal("'")),
                p.tag("dq", p.literal("\"") + p.literal(needle) + p.literal("\"")) });
        });
        auto result = parser.parse_anywhere_and_extract(haystack);
        if (!result.result.success()) {
            return json_quote_style::NONE;
        }
        return result.tags.count("sq") && !result.tags["sq"].empty()
            ? json_quote_style::SINGLE_QUOTES
            : json_quote_style::DOUBLE_QUOTES;
    };

    auto fun_quote = in_json_haystack(fun_name_needle);
    auto arg_quote = in_json_haystack(arg_name_needle);

    if (fun_quote != json_quote_style::NONE) {
        // no need to check further, we're in JSON land
        format.mode = tool_format::JSON_NATIVE;
        format.uses_python_dicts = (fun_quote == json_quote_style::SINGLE_QUOTES);
    } else if (arg_quote != json_quote_style::NONE) {
        format.mode = tool_format::TAG_WITH_JSON;
        format.uses_python_dicts = (arg_quote == json_quote_style::SINGLE_QUOTES);
    } else {
        format.mode = tool_format::TAG_WITH_TAGGED;
    }

    // first, remove any reasoning markers
    std::string clean_haystack = haystack;
    if (!reasoning.start.empty()) {
        auto pos = haystack.find(reasoning.start);
        if (pos != std::string::npos) {
            clean_haystack = haystack.substr(0, pos) + haystack.substr(pos + reasoning.start.length());
        }
    }
    if (!reasoning.end.empty()) {
        auto pos = clean_haystack.find(reasoning.end);
        if (pos != std::string::npos) {
            clean_haystack = clean_haystack.substr(0, pos) + clean_haystack.substr(pos + reasoning.end.length());
        }
    }

    if (format.mode == tool_format::JSON_NATIVE) {
        analyze_tool_call_format_json_native(clean_haystack, fun_name_needle, arg_name_needle);
    } else {
        analyze_tool_call_format_non_json(clean_haystack, fun_name_needle);
    }
    // always relax whitespace requirements on ending markers since they don't influence content
    format.section_end  = trim_whitespace(format.section_end);
    format.per_call_end = trim_whitespace(format.per_call_end);
}

void analyze_tools::analyze_tool_call_format_json_native(const std::string & clean_haystack,
                                                         const std::string & fun_name_needle,
                                                         const std::string & arg_name_needle) {
    // we might not have the typical OpenAI tool calling structure
    int  json_start     = clean_haystack.find_first_of('{');
    int  json_end       = clean_haystack.find_last_of('}');
    std::string cut     = clean_haystack.substr(json_start, json_end - json_start + 1);
    json call_struct    = json::parse(cut);
    auto register_field = [&](const std::string & prefix, const nlohmann::detail::iteration_proxy_value<json::iterator> & subel) {
        if (subel.value().is_string() && std::string(subel.value()).find("call0000") != std::string::npos) {
            format.id_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.value().is_string() && std::string(subel.value()) == fun_name_needle) {
            format.name_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.value().dump().find(arg_name_needle) !=
                   std::string::npos) {  // handle both string and JSON obj variants
            format.args_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.key().find("id") != std::string::npos) {
            // heuristics for generated id field
            format.gen_id_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        }
    };
    for (const auto & el : call_struct.items()) {
        if (el.key() == fun_name_needle) {
            format.fun_name_is_key = true;
            // When function name is the key, there's no name field and args are direct
            format.name_field.clear();
            format.args_field.clear();
            // Don't register this element - the function name IS the key, not a field
        } else {
            if (el.value().is_object() &&
                el.value().dump().find(arg_name_needle) == std::string::npos) {  // not the args object
                format.function_field = el.key();
                for (const auto & subel : el.value().items()) {
                    register_field(el.key(), subel);
                }
            }
            // Register this element as a potential field
            register_field("", el);
        }
    }
    auto array_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
        return p.tag("pre", p.literal("[") + p.space()) + p.literal(cut) + p.tag("post", p.space() + p.literal("]"));
    });

    auto ar_parse_res = array_parser.parse_anywhere_and_extract(clean_haystack);
    if (ar_parse_res.result.success()) {
        format.tools_array_wrapped = true;
        json_start -= ar_parse_res.tags["pre"].length();
        json_end += ar_parse_res.tags["post"].length();
    }
    json_end++; // we want to move past the closing char for end marker extraction

    std::vector<std::pair<size_t, std::string>> located_params;
    if (!format.name_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.name_field), format.name_field });
    }
    if (!format.args_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.args_field), format.args_field });
    }
    if (!format.id_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.id_field), format.id_field });
    }
    if (!format.gen_id_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.gen_id_field), format.gen_id_field });
    }
    std::sort(located_params.begin(), located_params.end());
    for (auto & pair : located_params) {
        format.parameter_order.push_back(pair.second);
    }
    // we can immediately extract tool calling markers too
    format.section_start = trim_leading_whitespace(clean_haystack.substr(0, json_start));
    format.section_end   = trim_whitespace(clean_haystack.substr(json_end));
    // When tools_array_wrapped is true, the closing bracket is part of the array structure,
    // not a separate section end marker. Clear tool_section_end to avoid duplicate brackets.
    if (format.tools_array_wrapped && format.section_end == "]") {
        format.section_end.clear();
    }
}

void analyze_tools::analyze_tool_call_format_non_json(const std::string & clean_haystack,
                                                      const std::string & fun_name_needle) {
    // first, let's find out if the function is inside a tag or standalone
    auto fun_marker_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("fun_marker", p.choice({
            p.tag("fun_pre", p.literal("<") + p.until_one_of({ ">", fun_name_needle })) + p.literal(fun_name_needle) +
                p.tag("fun_post", p.negate(p.space() + p.literal("<")) + p.until(">") + p.literal(">")) + p.space(),
            p.tag("fun_pre", p.literal("[") + p.until_one_of({ "]", fun_name_needle })) + p.literal(fun_name_needle) +
                p.tag("fun_post", p.negate(p.space() + p.literal("[") + p.until("]") + p.literal("]")) + p.space()) }));
    });
    auto fun_res = fun_marker_parser.parse_anywhere_and_extract(clean_haystack);
    std::string fun_marker = fun_name_needle;
    if (fun_res.result.success()) {
        fun_marker = fun_res.tags["fun_marker"];
    }
    // now, consume up to two markers, then treat everything up to the function marker as function name prefix
    auto per_tool_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
        return p.tag("sec_start", p.marker() + p.space()) + p.tag("call_start", p.marker() + p.space()) +
            p.tag("fun_pre", p.until(fun_marker)) + fun_marker + p.tag("rest", p.rest());
    });
    auto section_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
        return p.tag("sec_start", p.marker() + p.space()) + fun_marker + p.tag("rest", p.rest());
    });
    auto result = per_tool_parser.parse_anywhere_and_extract(clean_haystack);
    tagged_parse_result result_end;
    if (result.result.success()) {
        auto double_closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("call_end", p.marker() + p.space()) + p.tag("sec_end", p.marker() + p.space()) + p.end();
        });
        result_end = double_closer_parser.parse_anywhere_and_extract(result.tags["rest"]);
        function.name_prefix = fun_res.tags["fun_pre"] + function.name_prefix;
    } else {
        result = section_parser.parse_anywhere_and_extract(clean_haystack);
        auto single_closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("sec_end", p.marker() + p.space()) + p.end();
        });
        result_end = single_closer_parser.parse_anywhere_and_extract(result.tags["rest"]);
    }
    format.per_call_start = result.tags["call_start"];
    format.per_call_end = result_end.tags["call_end"];
    format.section_start = result.tags["sec_start"];
    format.section_end = result_end.tags["sec_end"];
}

void analyze_tools::check_per_call_markers() {
    json assistant_one_tool = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_two_tools = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call, second_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_one_tool });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto one_vs_two = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_two_tools }); });

    if (!one_vs_two) {
        LOG_DBG(ANSI_ORANGE "%s: Generating double tool call comparison failed\n" ANSI_RESET, __func__);
        return;
    }

    diff_split filter_common_call_part = calculate_diff_split(one_vs_two->diff.suffix, one_vs_two->diff.right);

    std::string second_tool_content = trim_leading_whitespace(filter_common_call_part.right);
    if (!format.section_start.empty() &&
        second_tool_content.find(format.section_start) == 0) {
        format.per_call_start = format.section_start;
        format.per_call_end   = format.section_end;
        format.section_start.clear();
        format.section_end.clear();
    }
}

void analyze_tools::extract_function_markers() {
    json assistant_nocall = json{
        { "role",    "assistant"   },
        { "content", ASSISTANT_MSG },
    };

    json assistant_foofoo = json{
        { "role",       "assistant"                      },
        { "content",    ""                               },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_barbar = json{
        { "role",       "assistant"                       },
        { "content",    ""                                },
        { "tool_calls", json::array({ second_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_foofoo });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_barbar }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (diff.left.find(FUN_FIRST) != std::string::npos && diff.right.find(FUN_SECOND) != std::string::npos) {
        std::string prefix_marker;
        if (!format.per_call_start.empty()) {
            prefix_marker = format.per_call_start;
        } else {
            prefix_marker = format.section_start;
        }
        if (!prefix_marker.empty() && diff.prefix.rfind(prefix_marker) != std::string::npos) {
            function.name_prefix =
                diff.prefix.substr(diff.prefix.rfind(prefix_marker) + prefix_marker.size());
        }

        // Extract name prefix/suffix from diff.left (stop at the next marker boundary)
        auto name_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.until(FUN_FIRST)) + p.literal(FUN_FIRST) +
                   p.tag("post", p.zero_or_more(p.negate(p.marker()) + p.any()));
        });
        auto name_result = name_parser.parse_and_extract(diff.left);
        if (name_result.result.success()) {
            function.name_prefix += name_result.tags["pre"];
            function.name_suffix = name_result.tags["post"];
        }

        // Extend name_suffix with content from diff.suffix before args begin
        if (format.mode == tool_format::TAG_WITH_JSON) {
            // For JSON: name_suffix extends to the first non-marker { or [, including any
            // markers along the way. Only applies if there's at least one marker after
            // the JSON content (matching the original "stop < seg_suf.size() - 1" guard).
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                auto non_json = p.marker() | (p.negate(p.literal("{")) + p.negate(p.literal("[")) + p.any());
                auto after_json = p.zero_or_more(p.negate(p.marker()) + p.any()) + p.marker();
                return p.tag("ext", p.zero_or_more(non_json)) + after_json;
            });
            auto suf_result = suffix_parser.parse_and_extract(diff.suffix);
            if (suf_result.result.success()) {
                function.name_suffix += suf_result.tags["ext"];
            }
        } else {
            // For tagged: name_suffix extends to the first marker (arg marker)
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                return p.tag("ext", p.zero_or_more(p.negate(p.marker()) + p.any()));
            });
            auto suf_result = suffix_parser.parse_and_extract(diff.suffix);
            if (suf_result.result.success()) {
                function.name_suffix += suf_result.tags["ext"];
            }
        }

        // Extract the closer (between last arg and call/section end marker)
        std::string suffix_marker;
        if (!format.per_call_end.empty()) {
            suffix_marker = format.per_call_end;
        } else {
            suffix_marker = format.section_end;
        }
        std::string closer_suffix;
        if (suffix_marker.empty()) {
            // we'll have to rely on an extra diff with no-calls version
            auto notool_comp = compare_variants(
                *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_nocall }); });
            auto nt_diff  = notool_comp->diff;
            closer_suffix = nt_diff.left.substr(nt_diff.left.find("YYYY") + 4);
        } else {
            closer_suffix = diff.suffix.substr(0, diff.suffix.find(suffix_marker));
        }
        if (!closer_suffix.empty()) {
            if (format.mode == tool_format::TAG_WITH_TAGGED) {
                // After last arg value, skip the closing arg marker, rest is closer
                auto closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                    return p.until("YYYY") + p.literal("YYYY") + p.space() +
                           p.marker() + p.space() +
                           p.tag("close", p.rest());
                });
                auto close_result = closer_parser.parse_and_extract(closer_suffix);
                if (close_result.result.success()) {
                    function.close = close_result.tags["close"];
                }
            } else if (format.mode == tool_format::TAG_WITH_JSON) {
                // After last arg value, find end of JSON args, rest is closer
                auto closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                    return p.until("YYYY") + p.literal("YYYY") + p.tag("post_val", p.rest());
                });
                auto close_result = closer_parser.parse_and_extract(closer_suffix);
                if (close_result.result.success()) {
                    const auto & post = close_result.tags["post_val"];
                    size_t pos = post.find_last_of("}]");
                    if (pos != std::string::npos && pos < post.size() - 1) {
                        function.close = trim_leading_whitespace(post.substr(pos + 1));
                    }
                }
            }
        }
        function.close = trim_leading_whitespace(function.close);
    }
}

void analyze_tools::analyze_arguments() {
    LOG_DBG(ANSI_ORANGE "Phase 4: Argument analysis\n" ANSI_RESET);

    extract_argument_name_markers();
    extract_argument_value_markers();
}

void analyze_tools::extract_argument_name_markers() {
    json assistant_first_arg = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    json assistant_second_arg = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_other_arg }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_first_arg });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_second_arg }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (!diff.left.empty() && !diff.right.empty()) {
        // Parse both sides to find ARG_FIRST/ARG_SECOND and extract the surrounding structure
        auto left_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.tag("pre", p.until(ARG_FIRST)) + p.literal(ARG_FIRST) +
                   p.tag("suffix", p.until_one_of({"\"", "X"}));
        });
        auto right_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.tag("pre", p.until(ARG_SECOND)) + p.literal(ARG_SECOND) +
                   p.tag("suffix", p.until_one_of({"\"", "Y"}));
        });
        auto left_result  = left_parser.parse_anywhere_and_extract(diff.left);
        auto right_result = right_parser.parse_anywhere_and_extract(diff.right);

        if (left_result.result.success() && right_result.result.success() &&
            !left_result.tags["pre"].empty() &&
            left_result.tags["pre"] == right_result.tags["pre"] &&
            left_result.tags["suffix"] == right_result.tags["suffix"]) {
            // Name is inside a structure (e.g., JSON key): prefix is the shared wrapper
            arguments.name_prefix = trim_whitespace(left_result.tags["pre"]);
            arguments.name_suffix = trim_leading_whitespace(left_result.tags["suffix"]);
        } else if (diff.left.substr(0, ARG_FIRST.length()) == ARG_FIRST && diff.right.substr(0, ARG_SECOND.length()) == ARG_SECOND) {
            // Name is directly in the diff: prefix comes from last marker in diff.prefix
            auto pre_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                auto last_marker = p.marker() + p.zero_or_more(p.negate(p.marker()) + p.any()) + p.end();
                return p.zero_or_more(p.negate(last_marker) + p.any()) + p.tag("name_prefix", last_marker);
            });
            auto pre_result = pre_parser.parse_and_extract(diff.prefix);
            arguments.name_prefix = pre_result.result.success()
                ? pre_result.tags["name_prefix"] : diff.prefix;

            // Suffix extends from after ARG_FIRST to the first marker (+ optional whitespace).
            // The marker could be in diff.left itself or in diff.suffix, so we concatenate.
            std::string after_first = diff.left.substr(ARG_FIRST.length()) + diff.suffix;
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.tag("suffix", p.zero_or_more(p.negate(p.marker()) + p.any()) +
                                       p.marker() + p.space());
            });
            auto suf_result = suffix_parser.parse_anywhere_and_extract(after_first);
            if (suf_result.result.success()) {
                arguments.name_suffix = suf_result.tags["suffix"];
            }
        }
    }
}

void analyze_tools::extract_argument_value_markers() {
    json assistant_val_X = json{
        { "role",       "assistant"                              },
        { "content",    ""                                       },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    json assistant_val_Y = json{
        { "role",       "assistant"                                        },
        { "content",    ""                                                 },
        { "tool_calls", json::array({ first_tool_call_one_arg_other_val }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_val_X });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_val_Y }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (diff.left == "XXXX" && diff.right == "YYYY") {
        std::string arg_name_ending = ARG_FIRST + arguments.name_suffix;
        std::string prefix          = diff.prefix;
        if (prefix.rfind(arg_name_ending) != std::string::npos) {
            prefix = prefix.substr(prefix.rfind(arg_name_ending) + arg_name_ending.size());
        }
        if (!prefix.empty()) {
            // Find the last marker + any trailing non-marker text to end
            auto prefix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                auto last_marker = p.marker() + p.zero_or_more(p.negate(p.marker()) + p.any()) + p.end();
                return p.zero_or_more(p.negate(last_marker) + p.any()) + p.tag("val_prefix", last_marker);
            });
            auto pre_result = prefix_parser.parse_and_extract(prefix);
            arguments.value_prefix = pre_result.result.success() ? pre_result.tags["val_prefix"] : prefix;
        }

        std::string value_suffix = diff.suffix;
        if (!function.close.empty()) {
            size_t func_close_pos = value_suffix.find(function.close);
            if (func_close_pos != std::string::npos) {
                value_suffix = value_suffix.substr(0, func_close_pos);
            }
        } else if (!format.per_call_end.empty() || !format.section_end.empty()) {
            std::string end_marker =
                !format.per_call_end.empty() ? format.per_call_end : format.section_end;
            size_t end_marker_pos = value_suffix.find(end_marker);
            if (end_marker_pos != std::string::npos) {
                value_suffix = value_suffix.substr(0, end_marker_pos);
            }
        }
        value_suffix = trim_leading_whitespace(value_suffix);
        if (!value_suffix.empty()) {
            arguments.value_suffix = value_suffix;
        }
    }
}

void analyze_tools::extract_argument_separator() {
    json assistant_one_arg = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    json assistant_two_args = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_one_arg });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_two_args }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (!diff.right.empty()) {
        std::string separator        = until_common_prefix(diff.right, ARG_FIRST, ARG_SECOND);
        arguments.separator = separator;
    }
}

void analyze_tools::extract_args_markers() {
    json assistant_no_args = json{
        { "role",       "assistant"},
        { "content",    ""         },
        { "tool_calls", json::array({ first_tool_call_zero_args }) }
    };

    json assistant_with_args = json{
        { "role",       "assistant"},
        { "content",    ""         },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_no_args });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_args }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (format.mode != tool_format::JSON_NATIVE) {
        std::string prefix_marker = !format.section_start.empty() ? format.section_start : format.per_call_start;
        std::string suffix_marker = !format.section_end.empty() ? format.section_end : format.per_call_end;
        // these might happen earlier in the tools section as an example or somewhere else, so we need to find the closest ones
        size_t prefix_pos = prefix_marker.empty() ? 0 : diff.prefix.rfind(prefix_marker);
        size_t suffix_pos = suffix_marker.empty() ? diff.suffix.size() : diff.suffix.find(suffix_marker);
        if (prefix_pos == std::string::npos) {
            prefix_pos = 0;
        }
        if (suffix_pos == std::string::npos) {
            suffix_pos = diff.suffix.size();
        }
        std::string prefix_cut = diff.prefix.substr(prefix_pos + prefix_marker.size());
        std::string suffix_cut = diff.suffix.substr(0, suffix_pos);
        std::string args_start = until_common_prefix(prefix_cut, "{}", "{\"first\":");
        std::string args_end   = after_common_suffix(suffix_cut, "{}", "\"XXXX\"}");

        if (!args_start.empty() || !args_end.empty()) {
            size_t find_fun = args_start.find(FUN_FIRST);
            if (find_fun != std::string::npos) {
                args_start = args_start.substr(find_fun + FUN_FIRST.size(), args_start.size() - find_fun - FUN_FIRST.size());
            }
            arguments.start = args_start;
            arguments.end   = args_end;
        }
    }
}

void analyze_tools::extract_call_id_markers() {
    json assistant_id1 = json{
        { "role",       "assistant" },
        { "content",    ""                               },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_id2 = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_alt_id }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_id1 });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_id2 }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed for call_id detection\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (diff.left.empty() && diff.right.empty()) {
        return;
    }

    std::string id_value_1 = "call00001";
    std::string id_value_2 = "call99999";

    size_t common_id_prefix_len = 0;
    for (size_t i = 0; i < std::min(id_value_1.length(), id_value_2.length()); i++) {
        if (id_value_1[i] == id_value_2[i]) {
            common_id_prefix_len++;
        } else {
            break;
        }
    }
    std::string common_id_part = id_value_1.substr(0, common_id_prefix_len);

    // Check if the function name is in the prefix (normal case: BETWEEN_FUNC_AND_ARGS or POST_ARGS)
    // or in the suffix (call_id is PRE_FUNC_NAME)
    std::string func_name           = FUN_FIRST;
    size_t      func_name_in_prefix = diff.prefix.rfind(func_name);
    size_t      func_name_in_suffix = diff.suffix.find(func_name);

    // Helper: find the last marker in a string (returns just the marker, not trailing text)
    auto find_last_marker = [](const std::string & str) -> std::string {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            auto last = p.marker() + p.zero_or_more(p.negate(p.marker()) + p.any()) + p.end();
            return p.zero_or_more(p.negate(last) + p.any()) + p.tag("m", p.marker());
        });
        auto res = parser.parse_anywhere_and_extract(str);
        return res.result.success() ? res.tags["m"] : "";
    };

    // Helper: find the first marker in a string
    auto find_first_marker = [](const std::string & str) -> std::string {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.tag("m", p.marker());
        });
        auto res = parser.parse_anywhere_and_extract(str);
        return res.result.success() ? res.tags["m"] : "";
    };

    if (func_name_in_prefix != std::string::npos && func_name_in_suffix == std::string::npos) {
        // Function name is only in prefix - call_id is BETWEEN_FUNC_AND_ARGS or POST_ARGS
        // Check if args indicator "{" is in prefix or suffix
        size_t args_in_prefix = diff.prefix.find('{', func_name_in_prefix);
        size_t args_in_suffix = diff.suffix.find('{');

        if (args_in_suffix != std::string::npos &&
            (args_in_prefix == std::string::npos || args_in_prefix > diff.prefix.length())) {
            // Args are in suffix, so call_id is BETWEEN_FUNC_AND_ARGS
            call_id.pos = call_id_position::BETWEEN_FUNC_AND_ARGS;

            // Find call_id_prefix: marker immediately preceding common_id_part (no intervening markers)
            std::string after_func = diff.prefix.substr(func_name_in_prefix + func_name.length());
            auto id_prefix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.tag("prefix", p.marker()) +
                       p.zero_or_more(p.negate(p.marker()) + p.negate(p.literal(common_id_part)) + p.any()) +
                       p.literal(common_id_part);
            });
            auto id_res = id_prefix_parser.parse_anywhere_and_extract(after_func);
            if (id_res.result.success()) {
                call_id.prefix = id_res.tags["prefix"];
            } else {
                // Fallback: use the last marker in after_func
                call_id.prefix = find_last_marker(after_func);
            }

            // Extract call_id_suffix: the first marker in the suffix before args "{"
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.zero_or_more(p.negate(p.marker()) + p.negate(p.literal("{")) + p.any()) +
                       p.tag("suffix", p.marker());
            });
            auto suf_res = suffix_parser.parse_anywhere_and_extract(diff.suffix);
            if (suf_res.result.success()) {
                call_id.suffix = suf_res.tags["suffix"];
            }
        } else if (args_in_prefix != std::string::npos) {
            // Args are in prefix, so call_id is POST_ARGS
            call_id.pos = call_id_position::POST_ARGS;

            // Extract last marker between args closing brace and the ID
            std::string after_args    = diff.prefix.substr(args_in_prefix);
            size_t      closing_brace = after_args.rfind('}');
            if (closing_brace != std::string::npos) {
                std::string between_args_and_id = after_args.substr(closing_brace + 1);
                call_id.prefix = find_last_marker(between_args_and_id);
            }

            // call_id_suffix: first marker in diff.suffix
            call_id.suffix = find_first_marker(diff.suffix);
        }
    } else if (func_name_in_suffix != std::string::npos && func_name_in_prefix == std::string::npos) {
        // Function name is only in suffix - call_id is PRE_FUNC_NAME
        call_id.pos = call_id_position::PRE_FUNC_NAME;

        // call_id_prefix: last marker in diff.prefix
        call_id.prefix = find_last_marker(diff.prefix);

        // call_id_suffix: first marker in the portion of diff.suffix before func_name
        std::string before_func = diff.suffix.substr(0, func_name_in_suffix);
        call_id.suffix = find_first_marker(before_func);
    }

    // When call_id is detected, per_call_end may have been incorrectly set to include
    // the call_id_suffix and sample args. Clear it if it starts with call_id_suffix.
    if (call_id.pos != call_id_position::NONE && !call_id.suffix.empty() &&
        format.per_call_end.find(call_id.suffix) == 0) {
        format.per_call_end.clear();
    }
}

}  // namespace autoparser
