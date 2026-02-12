#include "chat-diff-analyzer.h"

#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat.h"
#include "log.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>

#define ANSI_RESET  "\033[0m"
#define ANSI_PURPLE "\033[1m\x1b[38;5;126m"
#define ANSI_ORANGE "\033[1m\x1b[38;5;214m"
#define ANSI_RED    "\033[1m\x1b[38;5;196m"

using json = nlohmann::ordered_json;

static std::vector<std::function<void(const common_chat_template & tmpl, diff_analysis_result &)>> workarounds(
    { // Old reasoning Qwen templates - they don't really display reasoning content, but we still want to
      // support reasoning on them
      [](const common_chat_template & tmpl, diff_analysis_result & analysis) -> void {
          if (tmpl.src.find("content.split('</think>')") != std::string::npos && analysis.reasoning == reasoning_mode::NONE) {
            analysis.reasoning               = reasoning_mode::FORCED_OPEN;
            analysis.markers.reasoning_start = "<think>";
            analysis.markers.reasoning_end   = "</think>";
            analysis.preserved_tokens.push_back("<think>");
            analysis.preserved_tokens.push_back("</think>");
            LOG_DBG(ANSI_ORANGE "[Patch: old Qwen/Deepseek thinking template]\n" ANSI_RESET);
          }
      },
      // Granite 3.3, with separate reasoning and content markers
      [](const common_chat_template & tmpl, diff_analysis_result & analysis) -> void {
          if (tmpl.src.find("Write your thoughts between <think></think> and write your response between "
                            "<response></response>") != std::string::npos) {
            analysis.reasoning               = reasoning_mode::TAG_BASED;
            analysis.markers.reasoning_start = "<think>";
            analysis.markers.reasoning_end   = "</think>";
            analysis.preserved_tokens.push_back("<think>");
            analysis.preserved_tokens.push_back("</think>");
            analysis.content               = content_mode::WRAPPED_WITH_REASONING;
            analysis.markers.content_start = "<response>";
            analysis.markers.content_end   = "</response>";
            analysis.preserved_tokens.push_back("<response>");
            analysis.preserved_tokens.push_back("</response>");
            LOG_DBG(ANSI_ORANGE "[Patch: Granite 3.3]\n" ANSI_RESET);
          }
      },
      // Cohere Command R+ - content wrapped in <|CHATBOT_TOKEN|>...<|END_OF_TURN_TOKEN|>
      [](const common_chat_template & tmpl, diff_analysis_result & analysis) -> void {
          if (tmpl.src.find("<|CHATBOT_TOKEN|>") != std::string::npos &&
              tmpl.src.find("<|END_OF_TURN_TOKEN|>") != std::string::npos &&
              analysis.markers.content_start.empty()) {
            analysis.content               = content_mode::ALWAYS_WRAPPED;
            analysis.markers.content_start = "<|CHATBOT_TOKEN|>";
            analysis.markers.content_end   = "<|END_OF_TURN_TOKEN|>";
            analysis.preserved_tokens.push_back("<|CHATBOT_TOKEN|>");
            analysis.preserved_tokens.push_back("<|END_OF_TURN_TOKEN|>");
            LOG_DBG(ANSI_ORANGE "[Patch: Cohere Command R+]\n" ANSI_RESET);
          }
      },
      // Functionary - no tool call section delimiter
      [](const common_chat_template & tmpl, diff_analysis_result & analysis) -> void {
          if (tmpl.src.find("set has_code_interpreter = tools | selectattr(\"type\", \"equalto\", "
                            "\"code_interpreter\") | list | length > 0") != std::string::npos) {
            analysis.content                    = content_mode::PLAIN;
            analysis.markers.content_end        = "";
            analysis.markers.func_name_prefix   = "";
            analysis.markers.tool_section_start = "";
            analysis.markers.tool_section_end   = "";
            analysis.markers.per_call_start     = "<function=";
            analysis.markers.per_call_end       = "</function>";
            analysis.markers.func_close         = "";
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
      [](const common_chat_template & tmpl, diff_analysis_result & analysis) -> void {
          if (tmpl.src.find("{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>'") !=
              std::string::npos) {
            analysis.markers.tool_section_start = "<｜tool▁calls▁begin｜>";
            analysis.markers.tool_section_end   = "<｜tool▁calls▁end｜>";
            analysis.markers.per_call_start     = "<｜tool▁call▁begin｜>function";
            analysis.markers.func_name_prefix   = "<｜tool▁sep｜>";
            analysis.markers.per_call_end       = "<｜tool▁call▁end｜>";
            analysis.markers.func_close         = "```";
          }
      } });

// Common JSON structures
static json params_schema = {
    { "type",       "object"                                                           },
    { "properties",
     { { "first", { { "type", "string" }, { "description", "First argument" } } },
        { "second", { { "type", "string" }, { "description", "Second argument" } } } } },
    { "required",   json::array({})                                                    }
};

static json tools = json::array({
    { { "type", "function" },
     { "function",
        json{ { "name", "foofoo" }, { "description", "Test function foo" }, { "parameters", params_schema } } } },
    { { "type", "function" },
     { "function",
        json{ { "name", "barbar" }, { "description", "Test function bar" }, { "parameters", params_schema } } } }
});

static json user_msg = json{
    { "role",    "user"  },
    { "content", "Hello" }
};

static json build_tool_call(const std::string & name, const json & args, const std::string & id = "call00001") {
    return json{
        { "id",       id                                              },
        { "type",     "function"                                      },
        { "function", json{ { "name", name }, { "arguments", args } } }
    };
}

static json first_tool_call_zero_args         = build_tool_call("foofoo", json::object(), "call00001");
static json first_tool_call_one_arg           = build_tool_call("foofoo", {{ "first", "XXXX" }}, "call00001");
static json first_tool_call_one_arg_other_val = build_tool_call("foofoo", {{ "first", "YYYY" }}, "call00001");
static json first_tool_call_other_arg         = build_tool_call("foofoo", {{ "second", "YYYY" }}, "call00001");

static json first_tool_call =
    build_tool_call("foofoo", json{{ "first",  "XXXX" }, { "second", "YYYY" }}, "call00001");
static json second_tool_call =
    build_tool_call("barbar", json{ { "first",  "XXXX" }, { "second", "YYYY" }}, "call00002");
static json first_tool_call_alt_id =
    build_tool_call("foofoo", json{{ "first",  "XXXX" }, { "second", "YYYY" }}, "call99999");

std::string differential_analyzer::apply_template(const common_chat_template & tmpl, const template_params & params) {
    templates_params tmpl_params;
    tmpl_params.messages              = params.messages;
    tmpl_params.tools                 = params.tools;
    tmpl_params.add_generation_prompt = params.add_generation_prompt;
    tmpl_params.enable_thinking       = params.enable_thinking;

    if (params.extra_context) {
        tmpl_params.extra_context = *params.extra_context;
    }
    tmpl_params.extra_context["enable_thinking"] = params.enable_thinking;

    try {
        return common_chat_template_direct_apply(tmpl, tmpl_params);
    } catch (const std::exception & e) {
        LOG_DBG("Template application failed: %s\n", e.what());
        return "";
    }
}

std::optional<compare_variants_result> differential_analyzer::compare_variants(
    const common_chat_template &                   tmpl,
    const template_params &                        params_A,
    const std::function<void(template_params &)> & params_modifier) {
    // Create variant B by copying A
    template_params params_B = params_A;

    // Apply modifier to create variant B
    if (params_modifier) {
        params_modifier(params_B);
    }

    // Apply template to both variants
    std::string output_A = apply_template(tmpl, params_A);
    std::string output_B = apply_template(tmpl, params_B);

    // Check for template application failures
    if (output_A.empty() || output_B.empty()) {
        return std::nullopt;
    }

    // Calculate diff and return result with both outputs
    compare_variants_result result;
    result.diff     = calculate_diff_split(output_A, output_B);
    result.output_A = output_A;
    result.output_B = output_B;

    return result;
}

diff_analysis_result differential_analyzer::analyze(const common_chat_template & tmpl) {
    diff_analysis_result result;

    LOG_DBG(ANSI_PURPLE "=== Starting differential analysis ===\n" ANSI_RESET);

    auto caps                      = tmpl.original_caps();
    result.supports_tools          = caps.supports_tools || caps.supports_tool_calls;
    result.supports_parallel_calls = caps.supports_parallel_tool_calls;

    analyze_reasoning(tmpl, result);
    analyze_content(tmpl, result);
    if (result.supports_tools) {
        analyze_tools(tmpl, result);
    }
    collect_preserved_tokens(result);

    for (auto & workaround : workarounds) {
        workaround(tmpl, result);
    }

    LOG_DBG(ANSI_PURPLE "=== Differential analysis complete ===\n" ANSI_RESET);

    return result;
}

void differential_analyzer::analyze_reasoning(const common_chat_template & tmpl, diff_analysis_result & result) {
    LOG_DBG(ANSI_ORANGE "Phase 1: Reasoning analysis\n" ANSI_RESET);

    compare_reasoning_presence(tmpl, result);
    compare_thinking_enabled(tmpl, result);
    if (result.supports_tools) {
        compare_reasoning_scope(tmpl, result);
    }
}

void differential_analyzer::compare_reasoning_presence(const common_chat_template & tmpl,
                                                       diff_analysis_result &       result) {
    json user_msg = json{
        { "role",    "user"  },
        { "content", "Hello" }
    };

    json assistant_no_reasoning = json{
        { "role",    "assistant"   },
        { "content", "I can help." }
    };

    json assistant_with_reasoning = json{
        { "role",              "assistant"                },
        { "content",           "I can help."              },
        { "reasoning_content", "Let me think about this." }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_no_reasoning });
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_reasoning }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "R1: Template application failed, skipping reasoning detection\n" ANSI_RESET);
        return;
    }

    const auto & diff = comparison->diff;

    LOG_DBG(ANSI_ORANGE "R1 diff - suffix: " ANSI_RESET "'%s', " ANSI_ORANGE "left: " ANSI_RESET "'%s', " ANSI_ORANGE
                        "right: " ANSI_ORANGE "'%s'\n" ANSI_RESET,
            diff.suffix.c_str(), diff.left.c_str(), diff.right.c_str());

    const std::string reasoning_content = "Let me think about this.";

    if (!diff.right.empty() && diff.right.find(reasoning_content) != std::string::npos) {
        auto seg = prune_whitespace_segments(segmentize_markers(diff.right));
        if (seg.size() >= 3 && trim_whitespace(seg[1].value) == reasoning_content) {
            // easy one: opening marker - reasoning - closing marker (possibly with trailing whitespace)
            result.reasoning               = reasoning_mode::TAG_BASED;
            result.markers.reasoning_start = trim_whitespace(seg[0].value);
            result.markers.reasoning_end   = trim_leading_whitespace(seg[2].value);
            for (size_t i = 3; i < seg.size(); i++) {
                result.markers.reasoning_end += seg[i].value;
            }
            // we always truncate because this doesn't really influence correctness but model might not always generate newline
            result.markers.reasoning_end = trim_whitespace(result.markers.reasoning_end);
        } else if (seg.size() >= 2 && trim_whitespace(seg[0].value) == reasoning_content) {
            // delimited
            result.reasoning             = reasoning_mode::DELIMITER;
            result.markers.reasoning_end = trim_leading_whitespace(seg[1].value);
            for (size_t i = 2; i < seg.size(); i++) {
                result.markers.reasoning_end += seg[i].value;
            }
            result.markers.reasoning_end = trim_whitespace(result.markers.reasoning_end);
        } else if (seg.size() == 1 && trim_whitespace(seg[0].value) == reasoning_content) {
            // the marker might be in the prefix actually, let's check for case of
            // left: empty
            // right: reasoning_content
            // suffix: <closing marker>content
            // prefix: ...<opening marker>
            auto suf_seg = prune_whitespace_segments(segmentize_markers(diff.suffix));
            if (trim_whitespace(diff.left).empty() && suf_seg.size() >= 2 && suf_seg[0].type == segment_type::MARKER &&
                trim_whitespace(suf_seg[1].value).substr(0, 11) == "I can help.") {
                auto pre_seg = prune_whitespace_segments(segmentize_markers(diff.prefix));
                if (pre_seg[pre_seg.size() - 1].type == segment_type::MARKER ||
                    (pre_seg.size() > 1 && trim_whitespace(pre_seg[pre_seg.size() - 1].value).empty() &&
                     pre_seg[pre_seg.size() - 2].type == segment_type::MARKER)) {
                    auto marker_seg = pre_seg[pre_seg.size() - 1];
                    if (marker_seg.type == segment_type::TEXT) {
                        marker_seg = pre_seg[pre_seg.size() - 2];
                    }
                    result.reasoning               = reasoning_mode::FORCED_CLOSED;
                    result.markers.reasoning_start = trim_whitespace(marker_seg.value);
                    result.markers.reasoning_end   = trim_whitespace(suf_seg[0].value);
                }
            }
        }
    }
}

void differential_analyzer::compare_thinking_enabled(const common_chat_template & tmpl, diff_analysis_result & result) {
    json user_msg = json{
        { "role",    "user"  },
        { "content", "Hello" }
    };

    template_params params;
    params.messages              = json::array({ user_msg });
    params.add_generation_prompt = true;
    params.enable_thinking       = false;

    auto comparison = compare_variants(tmpl, params, [&](template_params & p) { p.enable_thinking = true; });

    if (!comparison) {
        LOG_DBG("R2: Template application failed\n");
        return;
    }

    const auto & diff = comparison->diff;

    LOG_DBG("R2 diff - suffix: '%s', left: '%s', right: '%s'\n", diff.suffix.c_str(), diff.left.c_str(),
            diff.right.c_str());

    std::string left_trimmed = diff.left;
    trim_whitespace(left_trimmed);

    if (left_trimmed.empty() && !diff.right.empty()) {
        std::string right_trimmed = diff.right;
        trim_whitespace(right_trimmed);

        if (!right_trimmed.empty() && string_ends_with(comparison->output_B, right_trimmed)) {
            if (result.markers.reasoning_start.empty()) {
                result.markers.reasoning_start = right_trimmed;
                result.reasoning               = reasoning_mode::FORCED_OPEN;
                LOG_DBG("R2: Detected forced-open reasoning with start marker: '%s'\n", right_trimmed.c_str());
            }
        }
    }

    if (result.markers.reasoning_start.empty() && !result.markers.reasoning_end.empty()) {
        result.reasoning = reasoning_mode::DELIMITER;
        LOG_DBG("R2: Delimiter-based reasoning detected (empty start, end: '%s')\n",
                result.markers.reasoning_end.c_str());
    }

    // Check for FORCED_CLOSED: when enable_thinking=false produces both start and end markers,
    // but enable_thinking=true produces only the start marker
    if (!comparison->output_A.empty() && !comparison->output_B.empty()) {
        std::string output_A = comparison->output_A;  // enable_thinking=false
        std::string output_B = comparison->output_B;  // enable_thinking=true

        // Both should end with the assistant role marker
        // Check if output_A has both reasoning_start and reasoning_end markers
        // while output_B has only reasoning_start
        if (!result.markers.reasoning_start.empty()) {
            // Check if output_A contains both start and end markers
            bool A_has_start = output_A.find(result.markers.reasoning_start) != std::string::npos;
            bool A_has_end   = !result.markers.reasoning_end.empty() &&
                             output_A.find(result.markers.reasoning_end) != std::string::npos;

            // Check if output_B contains only the start marker (and not the end marker)
            bool B_has_start = output_B.find(result.markers.reasoning_start) != std::string::npos;
            bool B_has_end   = !result.markers.reasoning_end.empty() &&
                             output_B.find(result.markers.reasoning_end) != std::string::npos;

            // For FORCED_CLOSED: A should have both, B should have only start
            if (A_has_start && A_has_end && B_has_start && !B_has_end) {
                result.reasoning = reasoning_mode::FORCED_CLOSED;
                LOG_DBG("R2: Detected forced-closed reasoning\n");
            }
        } else if (!result.markers.reasoning_end.empty()) {
            // We might not have detected the reasoning open marker until now,
            // but this is another chance to do so
            auto diff    = comparison->diff;
            auto diff_rt = trim_whitespace(diff.right);
            auto diff_lt = trim_whitespace(diff.left);
            if (diff_rt.empty() && diff_lt == result.markers.reasoning_end) {
                auto seg = segmentize_markers(trim_whitespace(diff.prefix));
                if (!seg.empty() && seg[seg.size() - 1].type == MARKER) {  // this is FORCED_CLOSED
                    result.markers.reasoning_start = seg[seg.size() - 1].value;
                    result.reasoning               = reasoning_mode::FORCED_CLOSED;
                }
            }
        }
    }

    // Check for slash-in-tag pattern: <think> vs </think>
    // diff shows: suffix="think>", left="/", right="" (or vice versa)
    if (result.markers.reasoning_start.empty() && result.markers.reasoning_end.empty()) {
        if (diff.right.empty() && trim_whitespace(diff.left) == "/") {
            auto seg_A = segmentize_markers(trim_trailing_whitespace(comparison->output_A));
            auto seg_B = segmentize_markers(trim_trailing_whitespace(comparison->output_B));
            if (!seg_A.empty() && !seg_B.empty() && seg_A[seg_A.size() - 1].type == segment_type::MARKER &&
                seg_B[seg_B.size() - 1].type == segment_type::MARKER) {
                result.reasoning               = reasoning_mode::FORCED_CLOSED;
                result.markers.reasoning_start = seg_B[seg_B.size() - 1].value;
                result.markers.reasoning_end   = seg_A[seg_A.size() - 1].value;
            }
        }
    }
}

void differential_analyzer::compare_reasoning_scope(const common_chat_template & tmpl, diff_analysis_result & result) {
    json assistant_reasoning_content = json{
        { "role",              "assistant"            },
        { "content",           "Here is my response." },
        { "reasoning_content", "Let me think."        }
    };

    json assistant_reasoning_tools = json{
        { "role",              "assistant"     },
        { "content",           nullptr         },
        { "reasoning_content", "Let me think." },
        { "tool_calls",
            json::array({ build_tool_call("foofoo", json{ { "first", "VVVV" }, { "second", "XXXX" } }) }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_reasoning_content });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_reasoning_tools }); });

    if (!comparison) {
        LOG_DBG("R3: Template application failed\n");
        return;
    }

    const auto & diff = comparison->diff;

    std::string reasoning_content = "Let me think.";

    LOG_DBG("R3 diff - prefix: '%s', suffix: '%s', left: '%s', right: '%s'\n", diff.prefix.c_str(), diff.suffix.c_str(),
            diff.left.c_str(), diff.right.c_str());

    // Check if reasoning only appears in variant B (with tools)
    bool reasoning_in_A = comparison->output_A.find(reasoning_content) != std::string::npos;
    bool reasoning_in_B = comparison->output_B.find(reasoning_content) != std::string::npos;

    if (!reasoning_in_A && reasoning_in_B) {
        result.reasoning = reasoning_mode::TOOLS_ONLY;
        LOG_DBG("R3: Detected TOOLS_ONLY reasoning mode\n");

        // Extract reasoning markers from output_B
        // The reasoning_content is "Let me think."
        size_t reasoning_pos = comparison->output_B.find(reasoning_content);
        if (reasoning_pos != std::string::npos) {
            // Find start marker before reasoning_content
            std::string before_reasoning = comparison->output_B.substr(0, reasoning_pos);
            before_reasoning             = trim_trailing_whitespace(before_reasoning);
            auto segments_before         = segmentize_markers(before_reasoning);
            std::reverse(segments_before.begin(), segments_before.end());

            for (auto & segment : segments_before) {
                if (segment.type == segment_type::MARKER) {
                    result.markers.reasoning_start = segment.value;
                    LOG_DBG("R3: Found reasoning_start: '%s'\n", result.markers.reasoning_start.c_str());
                    break;
                }
            }

            // Find end marker after reasoning_content
            size_t      reasoning_end   = reasoning_pos + reasoning_content.length();
            std::string after_reasoning = comparison->output_B.substr(reasoning_end);
            after_reasoning             = trim_leading_whitespace(after_reasoning);

            if (!after_reasoning.empty()) {
                // Try to find matching end marker
                if (!result.markers.reasoning_start.empty()) {
                    auto segments = segmentize_markers(after_reasoning);
                    for (auto & segment : segments) {
                        if (segment.type == segment_type::MARKER) {
                            result.markers.reasoning_end = segment.value;
                            break;
                        }
                    }
                    if (!result.markers.reasoning_end.empty()) {
                        LOG_DBG("R3: Found reasoning_end (matched): '%s'\n", result.markers.reasoning_end.c_str());
                    }
                }
            }
        }
    }
}

void differential_analyzer::analyze_content(const common_chat_template & tmpl, diff_analysis_result & result) {
    LOG_DBG(ANSI_ORANGE "Phase 2: Content analysis\n" ANSI_RESET);

    compare_content_values(tmpl, result);
}

void differential_analyzer::compare_content_values(const common_chat_template & tmpl, diff_analysis_result & result) {
    json assistant_content_only = json{
        { "role",    "assistant"     },
        { "content", "Response text" }
    };

    json assistant_with_tools = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ build_tool_call("test_func", json{ { "arg1", "value1" } }) }) }
    };

    json assistant_with_reasoning = json{
        { "role",              "assistant"     },
        { "content",           ""              },
        { "reasoning_content", "Need to think" }
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
        LOG_DBG("C1: Template application failed\n");
        return;
    }

    const auto & diff_tools     = comparison_with_tools->diff;
    const auto & diff_reasoning = comparison_with_reasoning->diff;

    std::string response = "Response text";

    bool found_plain_content = false;
    if (trim_whitespace(diff_tools.left) == response) {
        auto segments = segmentize_markers(diff_reasoning.left);
        if (trim_whitespace(diff_reasoning.left) == response ||
            (segments.size() == 2 && trim_whitespace(segments[0].value) == response)) {
            // We only have the content text in the diff (possibly with a stray EOG marker), so no markers
            LOG_DBG("C1: No content markers\n");
            result.content      = content_mode::PLAIN;
            found_plain_content = true;
        } else if (result.reasoning != reasoning_mode::NONE && !result.markers.reasoning_end.empty() &&
                   diff_reasoning.left.find(result.markers.reasoning_end) != std::string::npos) {
            std::string post_closed_reasoning = diff_reasoning.left.substr(
                diff_reasoning.left.find(result.markers.reasoning_end) + result.markers.reasoning_end.length());
            if (trim_whitespace(post_closed_reasoning) == "Response text") {
                LOG_DBG("C1: No content markers after stripping reasoning close marker\n");
                result.content      = content_mode::PLAIN;
                found_plain_content = true;
            }
        }
    }
    if (!found_plain_content) {
        std::string rdiff = diff_reasoning.left;
        if (!result.markers.reasoning_end.empty() && rdiff.find(result.markers.reasoning_end) != std::string::npos) {
            rdiff = rdiff.substr(rdiff.find(result.markers.reasoning_end) + result.markers.reasoning_end.length());
        }
        // Take the more promising diff
        std::string pure_content = rdiff.length() > diff_tools.left.length() ? rdiff : diff_tools.left;
        size_t      pos          = pure_content.find("Response text");
        if (pos == std::string::npos) {
            LOG_DBG("C1: Error: response text not found - improper template application?");
            return;
        }
        result.markers.content_start = trim_leading_whitespace(pure_content.substr(0, pos));
        result.markers.content_end =
            trim_leading_whitespace(pure_content.substr(pos + 13));  // 13 - len of "Response text"
        // TODO: WRAPPED_WITH_REASONING
    }

    // Determine content mode
    if (!result.markers.content_start.empty() || !result.markers.content_end.empty()) {
        result.content = content_mode::ALWAYS_WRAPPED;
        LOG_DBG("C1: Content is ALWAYS_WRAPPED\n");
        // TODO: END_DELIMITED content mode - delimited at end but not at start?
    }
}

void differential_analyzer::analyze_tool_call_format(const std::string &    haystack,
                                                     const std::string &    fun_name_needle,
                                                     const std::string &    arg_name_needle,
                                                     diff_analysis_result & result) {
    if (fun_name_needle.empty() || arg_name_needle.empty() || haystack.empty()) {
        return;
    }

    auto in_json_haystack = [&haystack](const std::string & needle) -> bool {
        // Find the needle in the haystack
        size_t needle_pos = haystack.find(needle);
        if (needle_pos == std::string::npos) {
            return false;
        }
        if (needle_pos < 2) {
            return false;  // not enough space for a JSON structure
        }
        if (haystack[needle_pos - 1] == '\'' || haystack[needle_pos - 1] == '"') {
            int cur = needle_pos - 2;
            for (; cur >= 0 && std::isspace(haystack[cur]); cur--) {
            }
            if (haystack[cur] == ':' || haystack[cur] == '{') {
                return true;
            }
        }
        return false;
    };

    if (in_json_haystack(fun_name_needle)) {
        // no need to check further, we're in JSON land
        result.tools = tool_format::JSON_NATIVE;
    } else if (in_json_haystack(arg_name_needle)) {
        result.tools = tool_format::TAG_WITH_JSON;
    } else {
        result.tools = tool_format::TAG_WITH_TAGGED;
    }

    // first, remove any reasoning markers
    std::string clean_haystack = haystack;
    if (!result.markers.reasoning_start.empty()) {
        auto pos = haystack.find(result.markers.reasoning_start);
        if (pos != std::string::npos) {
            clean_haystack = haystack.substr(0, pos) + haystack.substr(pos + result.markers.reasoning_start.length());
        }
    }
    if (!result.markers.reasoning_end.empty()) {
        auto pos = clean_haystack.find(result.markers.reasoning_end);
        if (pos != std::string::npos) {
            clean_haystack =
                clean_haystack.substr(0, pos) + clean_haystack.substr(pos + result.markers.reasoning_end.length());
        }
    }

    if (result.tools == tool_format::JSON_NATIVE) {
        analyze_tool_call_format_json_native(clean_haystack, fun_name_needle, arg_name_needle, result);
    } else {
        analyze_tool_call_format_non_json(clean_haystack, fun_name_needle, result);
    }
    // always relax whitespace requirements on ending markers since they don't influence content
    result.markers.tool_section_end = trim_whitespace(result.markers.tool_section_end);
    result.markers.per_call_end     = trim_whitespace(result.markers.per_call_end);
}

void differential_analyzer::analyze_tool_call_format_json_native(const std::string &    clean_haystack,
                                                                 const std::string &    fun_name_needle,
                                                                 const std::string &    arg_name_needle,
                                                                 diff_analysis_result & result) {
    // we might not have the typical OpenAI tool calling structure
    int  json_start     = clean_haystack.find_first_of('{');
    int  json_end       = clean_haystack.find_last_of('}');
    std::string cut     = clean_haystack.substr(json_start, json_end - json_start + 1);
    json call_struct    = json::parse(cut);
    auto register_field = [&](const std::string & prefix, const nlohmann::detail::iteration_proxy_value<json::iterator> & subel) {
        if (subel.value().is_string() && std::string(subel.value()).find("call0000") != std::string::npos) {
            result.id_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.value().is_string() && std::string(subel.value()) == fun_name_needle) {
            result.name_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.value().dump().find(arg_name_needle) !=
                   std::string::npos) {  // handle both string and JSON obj variants
            result.args_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.key().find("id") != std::string::npos) {
            // heuristics for generated id field
            result.gen_id_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        }
    };
    for (const auto & el : call_struct.items()) {
        if (el.key() == fun_name_needle) {
            result.fun_name_is_key = true;
            // When function name is the key, there's no name field and args are direct
            result.name_field.clear();
            result.args_field.clear();
            // Don't register this element - the function name IS the key, not a field
        } else {
            if (el.value().is_object() &&
                el.value().dump().find(arg_name_needle) == std::string::npos) {  // not the args object
                result.function_field = el.key();
                for (const auto & subel : el.value().items()) {
                    register_field(el.key(), subel);
                }
            }
            // Register this element as a potential field
            register_field("", el);
        }
    }
    // TODO: support for generated (not provided) tool call IDs
    auto space_or_bracket = [](bool opening, char c) -> bool {
        return std::isspace(c) || (opening ? c == '[' : c == ']');
    };
    // now let's check if we're in an array construction, mark it if so and get out of it
    if (json_start > 0 && space_or_bracket(true, clean_haystack[json_start - 1])) {
        for (--json_start; space_or_bracket(true, clean_haystack[json_start]) && json_start >= 0; json_start--) {
            if (clean_haystack[json_start] == '[') {
                result.tools_array_wrapped = true;
                break;
            }
        }
        if (!result.tools_array_wrapped) {
            json_start++;  // we ate into the last pre-json character
        }
    }
    if (json_end < (int) clean_haystack.length() - 1 && space_or_bracket(false, clean_haystack[json_end + 1])) {
        for (++json_end;
             space_or_bracket(false, clean_haystack[json_end]) && json_end < (int) clean_haystack.length() - 1;
             json_end++) {
        }
    }

    std::vector<std::pair<size_t, std::string>> located_params;
    if (!result.name_field.empty()) {
        located_params.push_back({ clean_haystack.find(result.name_field), result.name_field });
    }
    if (!result.args_field.empty()) {
        located_params.push_back({ clean_haystack.find(result.args_field), result.args_field });
    }
    if (!result.id_field.empty()) {
        located_params.push_back({ clean_haystack.find(result.id_field), result.id_field });
    }
    if (!result.gen_id_field.empty()) {
        located_params.push_back({ clean_haystack.find(result.gen_id_field), result.gen_id_field });
    }
    std::sort(located_params.begin(), located_params.end());
    for (auto & pair : located_params) {
        result.parameter_order.push_back(pair.second);
    }
    // we can immediately extract tool calling markers too
    result.markers.tool_section_start = trim_leading_whitespace(clean_haystack.substr(0, json_start));
    result.markers.tool_section_end   = trim_whitespace(clean_haystack.substr(json_end));
    // When tools_array_wrapped is true, the closing bracket is part of the array structure,
    // not a separate section end marker. Clear tool_section_end to avoid duplicate brackets.
    if (result.tools_array_wrapped && result.markers.tool_section_end == "]") {
        result.markers.tool_section_end.clear();
    }
}

void differential_analyzer::analyze_tool_call_format_non_json(const std::string &    clean_haystack,
                                                              const std::string &    fun_name_needle,
                                                              diff_analysis_result & result) {
    // we need to split by markers...
    auto haystack_split = segmentize_markers(trim_leading_whitespace(clean_haystack));
    int  where_is_nemo  = 0;
    int  i              = 0;
    for (auto & segment : haystack_split) {
        if (segment.value.find(fun_name_needle) != std::string::npos) {
            where_is_nemo = i;
            break;
        }
        i++;
    }

    // basically the rule here is:
    // - we append everything adjacent to a marker to the marker (treat it as part of the marker)
    // - we assume symmetry (as many opening as closing markers)
    // - we count the number of opening markers and then try to move backwards from the end until we've
    //   eaten as many closing markers as there were opening markers
    if (where_is_nemo > 1) {  // we might have more than one marker set here
        std::vector<segment> preceding_markers;
        for (int seg = where_is_nemo - 1; seg >= 0; seg--) {
            if (haystack_split[seg].type == MARKER) {
                preceding_markers.push_back(haystack_split[seg]);
            }
        }
        size_t how_many_markers = preceding_markers.size();
        if (how_many_markers > 1) {
            bool had_marker = false;
            for (int seg = where_is_nemo - 1; seg >= 0; seg--) {
                if (haystack_split[seg].type == MARKER) {
                    if (!had_marker) {
                        had_marker                    = true;
                        result.markers.per_call_start = haystack_split[seg].value + result.markers.per_call_start;
                    } else {
                        result.markers.tool_section_start =
                            haystack_split[seg].value + result.markers.tool_section_start;
                    }
                } else {
                    if (had_marker) {
                        result.markers.tool_section_start =
                            haystack_split[seg].value + result.markers.tool_section_start;
                    } else {
                        result.markers.per_call_start = haystack_split[seg].value + result.markers.per_call_start;
                    }
                }
            }
            had_marker                = false;
            size_t backtracked_so_far = 0;
            for (size_t seg = haystack_split.size() - 1; seg > (size_t) where_is_nemo; seg--) {
                if (haystack_split[seg].type == MARKER) {
                    backtracked_so_far++;
                    if (!had_marker) {
                        had_marker                      = true;
                        result.markers.tool_section_end = haystack_split[seg].value + result.markers.tool_section_end;
                    } else {
                        result.markers.per_call_end = haystack_split[seg].value + result.markers.per_call_end;
                    }
                } else {
                    if (had_marker) {
                        result.markers.per_call_end = haystack_split[seg].value + result.markers.per_call_end;
                    } else {
                        result.markers.tool_section_end = haystack_split[seg].value + result.markers.tool_section_end;
                    }
                }
                if (backtracked_so_far >= how_many_markers) {
                    break;
                }
            }
        } else {
            for (int seg = 0; seg < where_is_nemo; seg++) {
                result.markers.tool_section_start += haystack_split[seg].value;
            }
            for (size_t seg = haystack_split.size() - 1; seg > (size_t) where_is_nemo; seg--) {
                result.markers.tool_section_end = haystack_split[seg].value + result.markers.tool_section_end;
                if (haystack_split[seg].type == segment_type::MARKER) {
                    break;
                }
            }
        }
    } else {
        result.markers.tool_section_start += haystack_split[0].value;
        for (size_t seg = haystack_split.size() - 1; seg > (size_t) where_is_nemo; seg--) {
            result.markers.tool_section_end = haystack_split[seg].value + result.markers.tool_section_end;
            if (haystack_split[seg].type == segment_type::MARKER) {
                break;
            }
        }
    }
}

void differential_analyzer::analyze_tools(const common_chat_template & tmpl, diff_analysis_result & result) {
    LOG_DBG(ANSI_ORANGE "Phase 3: Tool call analysis\n" ANSI_RESET);
    analyze_tool_calls(tmpl, result);

    if (result.tools == tool_format::NONE) {
        LOG_DBG("T1: No tool support found\n");
        // Continue anyway - we may still have useful markers
    } else if (result.tools != tool_format::JSON_NATIVE) {
        if (result.supports_parallel_calls) {
            check_per_call_markers(tmpl, result);
        }
        extract_function_markers(tmpl, result);
        extract_argument_separator(tmpl, result);
        extract_args_markers(tmpl, result);
        extract_call_id_markers(tmpl, result);
        if (result.tools == tool_format::TAG_WITH_TAGGED) {
            analyze_arguments(tmpl, result);
        }
    }
}

void differential_analyzer::check_per_call_markers(const common_chat_template & tmpl, diff_analysis_result & result) {
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
        tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_two_tools }); });

    if (!one_vs_two) {
        LOG_DBG("T2: Generating double tool call comparison failed\n");
        return;
    }

    std::string second_tool_content = trim_leading_whitespace(one_vs_two->diff.right);
    if (!result.markers.tool_section_start.empty() &&
        second_tool_content.find(result.markers.tool_section_start) == 0) {
        result.markers.per_call_start = result.markers.tool_section_start;
        result.markers.per_call_end   = result.markers.tool_section_end;
        result.markers.tool_section_start.clear();
        result.markers.tool_section_end.clear();
    }
}

void differential_analyzer::analyze_tool_calls(const common_chat_template & tmpl, diff_analysis_result & result) {
    json assistant_no_tools = json{
        { "role",    "assistant" },
        { "content", "Response." }
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
        tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_tools }); });

    if (!comparison) {
        LOG_DBG("T1: Template application failed\n");
        return;
    }

    const auto & diff = comparison->diff;
    LOG_DBG("T1 diff - prefix: '%s', suffix: '%s'\n", diff.prefix.c_str(), diff.suffix.c_str());
    LOG_DBG("T1 diff - left: '%s', right: '%s'\n", diff.left.c_str(), diff.right.c_str());

    std::string tool_section = diff.right;

    if (tool_section.empty()) {
        return;
    }

    analyze_tool_call_format(tool_section, "foofoo", "first", result);

    LOG_DBG("T1: tool_section_start='%s', tool_section_end='%s'\n", result.markers.tool_section_start.c_str(),
            result.markers.tool_section_end.c_str());
}

void differential_analyzer::extract_call_separator(const common_chat_template & tmpl,
                                                   diff_analysis_result &       result,
                                                   std::string &                second_call_content) {
    json assistant_one_call = json{
        { "role",       "assistant"                      },
        { "content",    ""                               },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_two_calls = json{
        { "role",       "assistant"                                        },
        { "content",    ""                                                 },
        { "tool_calls", json::array({ first_tool_call, second_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_one_call });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_two_calls }); });

    if (!comparison) {
        LOG_DBG("T2: Template application failed\n");
        return;
    }

    const auto & diff = comparison->diff;
    LOG_DBG("T2 diff - prefix: '%s', suffix: '%s'\n", diff.prefix.c_str(), diff.suffix.c_str());
    LOG_DBG("T2 diff - left: '%s', right: '%s'\n", diff.left.c_str(), diff.right.c_str());

    if (!diff.right.empty()) {
        std::string first_func_name  = "foofoo";
        std::string second_func_name = "barbar";

        std::string separator         = until_common_prefix(diff.right, first_func_name, second_func_name);
        result.markers.call_separator = trim_whitespace(separator);

        LOG_DBG("T2: call_separator='%s'\n", result.markers.call_separator.c_str());

        result.supports_parallel_calls = true;
        second_call_content            = diff.right;

        LOG_DBG("T2: second_call_content='%s', supports_parallel_calls=true\n", second_call_content.c_str());
    }
}

void differential_analyzer::extract_function_markers(const common_chat_template & tmpl, diff_analysis_result & result) {
    json assistant_nocall = json{
        { "role",    "assistant" },
        { "content", "BBBB"      },
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
        tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_barbar }); });

    if (!comparison) {
        LOG_DBG("T3: Template application failed\n");
        return;
    }

    const auto & diff = comparison->diff;
    LOG_DBG("T3 diff - suffix: '%s'\n", diff.suffix.c_str());
    LOG_DBG("T3 diff - left: '%s', right: '%s'\n", diff.left.c_str(), diff.right.c_str());

    if (diff.left.find("foofoo") != std::string::npos && diff.right.find("barbar") != std::string::npos) {
        std::string prefix_marker;
        if (!result.markers.per_call_start.empty()) {
            prefix_marker = result.markers.per_call_start;
        } else {
            prefix_marker = result.markers.tool_section_start;
        }
        if (!prefix_marker.empty() && diff.prefix.rfind(prefix_marker) != std::string::npos) {
            result.markers.func_name_prefix =
                diff.prefix.substr(diff.prefix.rfind(prefix_marker) + prefix_marker.size());
        }

        auto seg = segmentize_markers(diff.left);
        for (const auto & s : seg) {
            if (s.value.find("foofoo") == std::string::npos) {
                result.markers.func_name_prefix += s.value;
            } else {
                size_t      pos  = s.value.find("foofoo");
                std::string pre  = s.value.substr(0, pos);
                std::string post = s.value.substr(pos + 6);  // 6 = len("foofoo")
                result.markers.func_name_prefix += pre;
                result.markers.func_name_suffix += post;
                break;
            }
        }

        auto   seg_suf           = segmentize_markers(diff.suffix);
        size_t stop              = 0;
        size_t stop_internal_pos = 0;
        for (const auto & ss : seg_suf) {
            bool has_needle = false;
            if (result.tools == tool_format::TAG_WITH_JSON) {
                has_needle = (ss.type == segment_type::TEXT && ss.value.find_first_of("{[") != std::string::npos);
                if (has_needle) {
                    stop_internal_pos = ss.value.find_first_of("{[");
                    break;
                }
            } else {
                has_needle = ss.value.find("first") != std::string::npos;
                if (has_needle) {
                    stop_internal_pos = ss.value.find("first");
                    break;
                }
            }
            stop++;
        }
        if (stop < seg_suf.size() - 1) {
            if (result.tools == tool_format::TAG_WITH_TAGGED) {
                size_t how_far = 0;
                if (stop > 0) {
                    if (seg_suf[stop].type == segment_type::MARKER) {
                        how_far = stop;
                    } else {
                        how_far = stop - 1;
                    }
                    for (size_t i = 0; i < how_far; i++) {
                        result.markers.func_name_suffix += seg_suf[i].value;
                    }
                }
            } else {
                for (size_t i = 0; i < stop; i++) {
                    result.markers.func_name_suffix += seg_suf[i].value;
                }
                const std::string & stopper = seg_suf[stop].value;
                result.markers.func_name_suffix += stopper.substr(0, stop_internal_pos);
            }
        }

        // now just to find the closer
        std::string suffix_marker;
        if (!result.markers.per_call_end.empty()) {
            suffix_marker = result.markers.per_call_end;
        } else {
            suffix_marker = result.markers.tool_section_end;
        }
        std::string closer_suffix;
        if (suffix_marker.empty()) {
            // we'll have to rely on an extra diff with no-calls version
            auto notool_comp = compare_variants(
                tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_nocall }); });
            auto nt_diff  = notool_comp->diff;
            closer_suffix = nt_diff.left.substr(nt_diff.left.find("YYYY") + 4);
        } else {
            closer_suffix = diff.suffix.substr(0, diff.suffix.find(suffix_marker));
        }
        if (!closer_suffix.empty()) {
            auto   closer_seg             = segmentize_markers(closer_suffix);
            bool   need_to_eat_arg_marker = (result.tools == tool_format::TAG_WITH_TAGGED);
            size_t last_arg_seg           = closer_seg.size() - 1;
            for (int i = (int) closer_seg.size() - 1; i >= 0; i--) {
                if (closer_seg[i].value.find("YYYY") != std::string::npos) {
                    last_arg_seg = i;
                }
            }
            if (result.tools == tool_format::TAG_WITH_JSON) {
                const auto & entire_seg = closer_seg[last_arg_seg].value;
                size_t       pos        = entire_seg.find_last_of("}]");
                if (pos != std::string::npos && pos < entire_seg.size() - 1) {
                    result.markers.func_close = trim_leading_whitespace(entire_seg.substr(pos + 1));
                }
            }
            for (size_t i = last_arg_seg + 1; i < closer_seg.size(); i++) {
                if (closer_seg[i].type == segment_type::MARKER) {
                    if (need_to_eat_arg_marker) {
                        need_to_eat_arg_marker = false;
                    } else {
                        result.markers.func_close += closer_seg[i].value;
                    }
                } else if (!need_to_eat_arg_marker) {
                    result.markers.func_close += closer_seg[i].value;
                }
            }
        }
        result.markers.func_close = trim_leading_whitespace(result.markers.func_close);

        LOG_DBG("T3: func_name_prefix='%s', func_name_suffix='%s', func_close='%s'\n",
                result.markers.func_name_prefix.c_str(), result.markers.func_name_suffix.c_str(),
                result.markers.func_close.c_str());
    }
}

void differential_analyzer::extract_argument_separator(const common_chat_template & tmpl,
                                                       diff_analysis_result &       result) {
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
        tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_two_args }); });

    if (!comparison) {
        LOG_DBG("T4: Template application failed\n");
        return;
    }

    const auto & diff = comparison->diff;
    LOG_DBG("T4 diff - suffix: '%s'\n", diff.suffix.c_str());
    LOG_DBG("T4 diff - left: '%s', right: '%s'\n", diff.left.c_str(), diff.right.c_str());

    if (!diff.right.empty()) {
        std::string separator        = until_common_prefix(diff.right, "first", "second");
        result.markers.arg_separator = separator;
        LOG_DBG("T4: arg_separator='%s'\n", result.markers.arg_separator.c_str());
    }
}

void differential_analyzer::extract_args_markers(const common_chat_template & tmpl, diff_analysis_result & result) {
    json assistant_no_args = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_zero_args }) }
    };

    json assistant_with_args = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_no_args });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_args }); });

    if (!comparison) {
        LOG_DBG("T5: Template application failed\n");
        return;
    }

    const auto & diff = comparison->diff;
    LOG_DBG("T5 diff - suffix: '%s'\n", diff.suffix.c_str());
    LOG_DBG("T5 diff - left: '%s', right: '%s'\n", diff.left.c_str(), diff.right.c_str());

    if (result.markers.args_start.empty() && result.tools != tool_format::JSON_NATIVE) {
        std::string prefix_marker = !result.markers.tool_section_start.empty() ? result.markers.tool_section_start :
                                                                                 result.markers.per_call_start;
        std::string suffix_marker =
            !result.markers.tool_section_end.empty() ? result.markers.tool_section_end : result.markers.per_call_end;
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
            result.markers.args_start = args_start;
            result.markers.args_end   = args_end;
            LOG_DBG("T5: Custom argument container detected: start='%s', end='%s'\n", args_start.c_str(),
                    args_end.c_str());
        }
    }
}

void differential_analyzer::extract_call_id_markers(const common_chat_template & tmpl, diff_analysis_result & result) {
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
        tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_id2 }); });

    if (!comparison) {
        LOG_DBG("T6: Template application failed for call_id detection\n");
        return;
    }

    const auto & diff = comparison->diff;
    LOG_DBG("T6 diff (call_id) - prefix: '%s', suffix: '%s'\n", diff.prefix.c_str(), diff.suffix.c_str());
    LOG_DBG("T6 diff (call_id) - left: '%s', right: '%s'\n", diff.left.c_str(), diff.right.c_str());

    if (diff.left.empty() && diff.right.empty()) {
        LOG_DBG("T6: No call_id difference detected\n");
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
    std::string func_name           = "foofoo";
    size_t      func_name_in_prefix = diff.prefix.rfind(func_name);
    size_t      func_name_in_suffix = diff.suffix.find(func_name);

    if (func_name_in_prefix != std::string::npos && func_name_in_suffix == std::string::npos) {
        // Function name is only in prefix - call_id is BETWEEN_FUNC_AND_ARGS or POST_ARGS
        // Check if args indicator "{" is in prefix or suffix
        size_t args_in_prefix = diff.prefix.find('{', func_name_in_prefix);
        size_t args_in_suffix = diff.suffix.find('{');

        if (args_in_suffix != std::string::npos &&
            (args_in_prefix == std::string::npos || args_in_prefix > diff.prefix.length())) {
            // Args are in suffix, so call_id is BETWEEN_FUNC_AND_ARGS
            result.call_id_pos = call_id_position::BETWEEN_FUNC_AND_ARGS;
            LOG_DBG("T6: Detected BETWEEN_FUNC_AND_ARGS position\n");

            // The prefix ends with: ...<func_name><func_name_suffix><call_id_prefix><common_id_part>
            // Segmentize to find the call_id_prefix marker
            std::string after_func = diff.prefix.substr(func_name_in_prefix + func_name.length());
            auto        segments   = segmentize_markers(after_func);

            std::string marker_before_id;
            for (size_t i = 0; i < segments.size(); i++) {
                if (segments[i].type == segment_type::MARKER) {
                    // Check if the next segment (if any) contains the common_id_part
                    if (i + 1 < segments.size() && segments[i + 1].value.find(common_id_part) != std::string::npos) {
                        marker_before_id = segments[i].value;
                        break;
                    }
                    // Or if this is the last marker and the text after contains common_id_part
                    if (i == segments.size() - 1 ||
                        (i + 1 < segments.size() && segments[i + 1].type == segment_type::TEXT &&
                         segments[i + 1].value.find(common_id_part) != std::string::npos)) {
                        marker_before_id = segments[i].value;
                    }
                }
            }

            if (!marker_before_id.empty()) {
                result.markers.call_id_prefix = marker_before_id;
                LOG_DBG("T6: call_id_prefix='%s'\n", result.markers.call_id_prefix.c_str());
            } else {
                // Fallback: look for the last marker in after_func
                for (int i = (int) segments.size() - 1; i >= 0; i--) {
                    if (segments[i].type == segment_type::MARKER) {
                        result.markers.call_id_prefix = segments[i].value;
                        LOG_DBG("T6: call_id_prefix (fallback)='%s'\n", result.markers.call_id_prefix.c_str());
                        break;
                    }
                }
            }

            // Extract call_id_suffix: the first marker in the suffix before args
            auto suffix_segments = segmentize_markers(diff.suffix);
            for (size_t i = 0; i < suffix_segments.size(); i++) {
                if (suffix_segments[i].type == segment_type::MARKER) {
                    result.markers.call_id_suffix = suffix_segments[i].value;
                    LOG_DBG("T6: call_id_suffix='%s'\n", result.markers.call_id_suffix.c_str());
                    break;
                }
                // Stop if we hit the args
                if (suffix_segments[i].value.find('{') != std::string::npos) {
                    break;
                }
            }
        } else if (args_in_prefix != std::string::npos) {
            // Args are in prefix, so call_id is POST_ARGS
            result.call_id_pos = call_id_position::POST_ARGS;
            LOG_DBG("T6: POST_ARGS call_id position detected\n");

            // Extract markers from between args and the ID
            std::string after_args    = diff.prefix.substr(args_in_prefix);
            size_t      closing_brace = after_args.rfind('}');
            if (closing_brace != std::string::npos) {
                std::string between_args_and_id = after_args.substr(closing_brace + 1);
                auto        segments            = segmentize_markers(between_args_and_id);
                for (int i = (int) segments.size() - 1; i >= 0; i--) {
                    if (segments[i].type == segment_type::MARKER) {
                        result.markers.call_id_prefix = segments[i].value;
                        LOG_DBG("T6: call_id_prefix='%s'\n", result.markers.call_id_prefix.c_str());
                        break;
                    }
                }
            }

            // call_id_suffix would be in the suffix (first marker)
            auto suffix_segments = segmentize_markers(diff.suffix);
            for (const auto & seg : suffix_segments) {
                if (seg.type == segment_type::MARKER) {
                    result.markers.call_id_suffix = seg.value;
                    LOG_DBG("T6: call_id_suffix='%s'\n", result.markers.call_id_suffix.c_str());
                    break;
                }
            }
        }
    } else if (func_name_in_suffix != std::string::npos && func_name_in_prefix == std::string::npos) {
        // Function name is only in suffix - call_id is PRE_FUNC_NAME
        result.call_id_pos = call_id_position::PRE_FUNC_NAME;
        LOG_DBG("T6: PRE_FUNC_NAME call_id position detected\n");

        // Extract call_id_prefix from prefix (last marker before the common_id_part)
        auto prefix_segments = segmentize_markers(diff.prefix);
        for (int i = (int) prefix_segments.size() - 1; i >= 0; i--) {
            if (prefix_segments[i].type == segment_type::MARKER) {
                result.markers.call_id_prefix = prefix_segments[i].value;
                LOG_DBG("T6: call_id_prefix='%s'\n", result.markers.call_id_prefix.c_str());
                break;
            }
        }

        // Extract call_id_suffix from suffix (first marker before func_name)
        std::string before_func     = diff.suffix.substr(0, func_name_in_suffix);
        auto        suffix_segments = segmentize_markers(before_func);
        for (const auto & seg : suffix_segments) {
            if (seg.type == segment_type::MARKER) {
                result.markers.call_id_suffix = seg.value;
                LOG_DBG("T6: call_id_suffix='%s'\n", result.markers.call_id_suffix.c_str());
                break;
            }
        }
    } else {
        LOG_DBG("T6: Unable to determine call_id position\n");
    }

    // When call_id is detected, per_call_end may have been incorrectly set to include
    // the call_id_suffix and sample args. Clear it if it starts with call_id_suffix.
    if (result.call_id_pos != call_id_position::NONE && !result.markers.call_id_suffix.empty() &&
        result.markers.per_call_end.find(result.markers.call_id_suffix) == 0) {
        result.markers.per_call_end.clear();
        LOG_DBG("T6: Cleared per_call_end (was incorrectly including call_id_suffix)\n");
    }
}

void differential_analyzer::analyze_arguments(const common_chat_template & tmpl, diff_analysis_result & result) {
    LOG_DBG(ANSI_ORANGE "Phase 4: Argument analysis\n" ANSI_RESET);

    extract_argument_name_markers(tmpl, result);
    extract_argument_value_markers(tmpl, result);
}

void differential_analyzer::extract_argument_name_markers(const common_chat_template & tmpl,
                                                          diff_analysis_result &       result) {
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
        tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_second_arg }); });

    if (!comparison) {
        LOG_DBG("A1: Template application failed\n");
        return;
    }

    const auto & diff = comparison->diff;
    LOG_DBG("A1 diff - suffix: '%s', left: '%s', right: '%s'\n", diff.suffix.c_str(), diff.left.c_str(),
            diff.right.c_str());

    if (!diff.left.empty() && !diff.right.empty()) {
        size_t common_len = 0;
        size_t min_len    = std::min(diff.left.length(), diff.right.length());
        while (common_len < min_len && diff.left[common_len] == diff.right[common_len]) {
            common_len++;
        }

        if (common_len > 0) {  // we have a marker structure with the name *inside* the marker
            std::string common_prefix   = diff.left.substr(0, common_len);
            std::string left_remainder  = diff.left.substr(common_len);
            std::string right_remainder = diff.right.substr(common_len);
            size_t      left_close =
                left_remainder.find_first_of("\"X");  // because arg-val is XXXX, can be quoted or unquoted
            size_t right_close = right_remainder.find_first_of("\"Y");  // here arg-val is YYYY

            if (left_close != std::string::npos && right_close != std::string::npos) {
                std::string left_name  = left_remainder.substr(0, 5);   // 5 = len("first")
                std::string right_name = right_remainder.substr(0, 6);  // 6 = len("second")

                if (left_name == "first" && right_name == "second") {
                    result.markers.arg_name_prefix = trim_whitespace(common_prefix);
                    std::string suffix_left        = left_remainder.substr(5, left_close - 5);
                    std::string suffix_right       = right_remainder.substr(6, right_close - 6);
                    if (suffix_left == suffix_right) {
                        result.markers.arg_name_suffix = trim_leading_whitespace(suffix_left);
                    }
                    LOG_DBG("A1: arg_name_prefix='%s', arg_name_suffix='%s'\n", result.markers.arg_name_prefix.c_str(),
                            result.markers.arg_name_suffix.c_str());
                }
            }
        } else if (diff.left.substr(0, 5) == "first" && diff.right.substr(0, 6) == "second") {
            // we most likely have actual markers for argument names
            auto pre_seg = segmentize_markers(diff.prefix);
            for (int i = pre_seg.size() - 1; i >= 0; i--) {
                result.markers.arg_name_prefix = result.markers.arg_name_prefix + pre_seg[i].value;
                if (pre_seg[i].type == segment_type::MARKER) {
                    break;
                }
            }
            auto left_seg = segmentize_markers(diff.left);
            if (left_seg.size() == 1) {  // only the name + maybe extra whitespace / normal chars in differing part
                result.markers.arg_name_suffix = diff.left.substr(5);
                auto suf_seg= segmentize_markers(diff.suffix);
                for (size_t i = 0; i < suf_seg.size(); i++) {
                    result.markers.arg_name_suffix += suf_seg[i].value;
                    if (suf_seg[i].type == segment_type::MARKER) {
                        if (i < suf_seg.size() - 2 && suf_seg[i + 1].type == segment_type::TEXT &&
                            trim_whitespace(suf_seg[i + 1].value).empty()) {
                            // we need to include post-marker whitespace/newlines as well
                            result.markers.arg_name_suffix += suf_seg[i + 1].value;
                        }
                        break;
                    }
                }
            } else {
                for (size_t i = 0; i < left_seg.size(); i++) {
                    std::string to_add;
                    if (i == 0) {
                        to_add = left_seg[i].value.substr(5);
                    } else {
                        to_add = left_seg[i].value;
                    }
                    result.markers.arg_name_suffix += to_add;
                    if (left_seg[i].type == segment_type::MARKER) {
                        if (i < left_seg.size() - 2 && left_seg[i + 1].type == segment_type::TEXT &&
                            trim_whitespace(left_seg[i + 1].value).empty()) {
                            // we need to include post-marker whitespace/newlines as well
                            result.markers.arg_name_suffix += left_seg[i + 1].value;
                        }
                        break;
                    }
                }
            }
        }
    }
}

void differential_analyzer::extract_argument_value_markers(const common_chat_template & tmpl,
                                                           diff_analysis_result &       result) {
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
        tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_val_Y }); });

    if (!comparison) {
        LOG_DBG("A2: Template application failed\n");
        return;
    }

    const auto & diff = comparison->diff;
    LOG_DBG("A2 diff - suffix: '%s'\n", diff.suffix.c_str());
    LOG_DBG("A2 diff - left: '%s', right: '%s'\n", diff.left.c_str(), diff.right.c_str());

    if (diff.left == "XXXX" && diff.right == "YYYY") {
        std::string arg_name_ending = "first" + result.markers.arg_name_suffix;
        std::string prefix          = diff.prefix;
        if (prefix.rfind(arg_name_ending) != std::string::npos) {
            prefix = prefix.substr(prefix.rfind(arg_name_ending) + arg_name_ending.size());
        }
        if (!prefix.empty()) {
            auto seg_pre = segmentize_markers(prefix);
            for (int i = seg_pre.size() - 1; i >= 0; i--) {
                result.markers.arg_value_prefix = seg_pre[i].value + result.markers.arg_value_prefix;
                if (seg_pre[i].type == segment_type::MARKER) {
                    break;
                }
            }
        }

        std::string value_suffix = diff.suffix;
        if (!result.markers.func_close.empty()) {
            size_t func_close_pos = value_suffix.find(result.markers.func_close);
            if (func_close_pos != std::string::npos) {
                value_suffix = value_suffix.substr(0, func_close_pos);
            }
        } else if (!result.markers.per_call_end.empty() || !result.markers.tool_section_end.empty()) {
            std::string end_marker =
                !result.markers.per_call_end.empty() ? result.markers.per_call_end : result.markers.tool_section_end;
            size_t end_marker_pos = value_suffix.find(end_marker);
            if (end_marker_pos != std::string::npos) {
                value_suffix = value_suffix.substr(0, end_marker_pos);
            }
        }
        value_suffix = trim_leading_whitespace(value_suffix);
        if (!value_suffix.empty()) {
            result.markers.arg_value_suffix = value_suffix;
        }

        LOG_DBG("A2: arg_value_prefix='%s', arg_value_suffix='%s'\n", result.markers.arg_value_prefix.c_str(),
                result.markers.arg_value_suffix.c_str());
    }
}

void differential_analyzer::collect_preserved_tokens(diff_analysis_result & result) {
    auto & tokens = result.preserved_tokens;

    auto add_token = [&tokens](const std::string & org_token) {
        std::string token = trim_whitespace(org_token);
        if (!token.empty()) {
            // Avoid duplicates
            if (std::find(tokens.begin(), tokens.end(), token) == tokens.end()) {
                tokens.push_back(token);
            }
        }
    };

    add_token(result.markers.reasoning_start);
    add_token(result.markers.reasoning_end);
    add_token(result.markers.content_start);
    add_token(result.markers.content_end);
    add_token(result.markers.tool_section_start);
    add_token(result.markers.tool_section_end);
    add_token(result.markers.per_call_start);
    add_token(result.markers.per_call_end);
    add_token(result.markers.func_name_prefix);
    add_token(result.markers.func_name_suffix);
    add_token(result.markers.func_close);
    add_token(result.markers.arg_name_prefix);
    add_token(result.markers.arg_name_suffix);
    add_token(result.markers.arg_separator);
    add_token(result.markers.arg_value_prefix);
    add_token(result.markers.arg_value_suffix);
    add_token(result.markers.call_id_prefix);
    add_token(result.markers.call_id_suffix);
    add_token(result.markers.code_block_marker);
}
