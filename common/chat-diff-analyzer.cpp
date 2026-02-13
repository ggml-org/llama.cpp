#include "chat-diff-analyzer.h"

#include "chat-auto-parser-helpers.h"
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

namespace autoparser {

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

analyze_template::analyze_template(const common_chat_template & tmpl)
    : jinja_caps(tmpl.original_caps())
    , reasoning(tmpl, jinja_caps.supports_tool_calls)
    , content(tmpl, reasoning)
    , tools(jinja_caps.supports_tool_calls ? analyze_tools(tmpl, jinja_caps, reasoning) : analyze_tools())
{
    LOG_DBG(ANSI_PURPLE "=== Starting differential analysis ===\n" ANSI_RESET);

    collect_preserved_tokens();

    for (auto & workaround : workarounds) {
        workaround(tmpl, *this);
    }

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
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_reasoning }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed, skipping reasoning detection\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    const std::string reasoning_content = "Let me think about this.";

    if (!diff.right.empty() && diff.right.find(reasoning_content) != std::string::npos) {
        auto seg = prune_whitespace_segments(segmentize_markers(diff.right));
        if (seg.size() >= 3 && trim_whitespace(seg[1].value) == reasoning_content) {
            // easy one: opening marker - reasoning - closing marker (possibly with trailing whitespace)
            mode  = reasoning_mode::TAG_BASED;
            start = trim_whitespace(seg[0].value);
            end   = trim_leading_whitespace(seg[2].value);
            for (size_t i = 3; i < seg.size(); i++) {
                end += seg[i].value;
            }
            // we always truncate because this doesn't really influence correctness but model might not always generate newline
            end = trim_whitespace(end);
        } else if (seg.size() >= 2 && trim_whitespace(seg[0].value) == reasoning_content) {
            // delimited
            mode = reasoning_mode::DELIMITER;
            end  = trim_leading_whitespace(seg[1].value);
            for (size_t i = 2; i < seg.size(); i++) {
                end += seg[i].value;
            }
            end = trim_whitespace(end);
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
                    mode  = reasoning_mode::FORCED_CLOSED;
                    start = trim_whitespace(marker_seg.value);
                    end   = trim_whitespace(suf_seg[0].value);
                }
            }
        }
    }
}

void analyze_reasoning::compare_thinking_enabled() {
    json user_msg = json{
        { "role",    "user"  },
        { "content", "Hello" }
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

    std::string left_trimmed = diff.left;
    trim_whitespace(left_trimmed);

    if (left_trimmed.empty() && !diff.right.empty()) {
        std::string right_trimmed = diff.right;
        trim_whitespace(right_trimmed);

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
        std::string output_A = comparison->output_A;  // enable_thinking=false
        std::string output_B = comparison->output_B;  // enable_thinking=true

        // Both should end with the assistant role marker
        // Check if output_A has both reasoning_start and reasoning_end markers
        // while output_B has only reasoning_start
        if (!start.empty()) {
            // Check if output_A contains both start and end markers
            bool A_has_start = output_A.find(start) != std::string::npos;
            bool A_has_end   = !end.empty() && output_A.find(end) != std::string::npos;

            // Check if output_B contains only the start marker (and not the end marker)
            bool B_has_start = output_B.find(start) != std::string::npos;
            bool B_has_end   = !end.empty() && output_B.find(end) != std::string::npos;

            // For FORCED_CLOSED: A should have both, B should have only start
            if (A_has_start && A_has_end && B_has_start && !B_has_end) {
                mode = reasoning_mode::FORCED_CLOSED;
            }
        } else if (!end.empty()) {
            // We might not have detected the reasoning open marker until now,
            // but this is another chance to do so
            auto diff    = comparison->diff;
            auto diff_rt = trim_whitespace(diff.right);
            auto diff_lt = trim_whitespace(diff.left);
            if (diff_rt.empty() && diff_lt == end) {
                auto seg = segmentize_markers(trim_whitespace(diff.prefix));
                if (!seg.empty() && seg[seg.size() - 1].type == MARKER) {  // this is FORCED_CLOSED
                    start = seg[seg.size() - 1].value;
                    mode  = reasoning_mode::FORCED_CLOSED;
                }
            }
        }
    }

    if (start.empty() && end.empty()) {
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
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_reasoning_tools }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    std::string reasoning_content = "Let me think.";

    // Check if reasoning only appears in variant B (with tools)
    bool reasoning_in_A = comparison->output_A.find(reasoning_content) != std::string::npos;
    bool reasoning_in_B = comparison->output_B.find(reasoning_content) != std::string::npos;

    if (!reasoning_in_A && reasoning_in_B) {
        mode = reasoning_mode::TOOLS_ONLY;
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
                    start = segment.value;
                    break;
                }
            }

            // Find end marker after reasoning_content
            size_t      reasoning_end   = reasoning_pos + reasoning_content.length();
            std::string after_reasoning = comparison->output_B.substr(reasoning_end);
            after_reasoning             = trim_leading_whitespace(after_reasoning);

            if (!after_reasoning.empty()) {
                // Try to find matching end marker
                if (!start.empty()) {
                    auto segments = segmentize_markers(after_reasoning);
                    for (auto & segment : segments) {
                        if (segment.type == segment_type::MARKER) {
                            end = segment.value;
                            break;
                        }
                    }
                }
            }
        }
    }
}

analyze_content::analyze_content(const common_chat_template & tmpl, const analyze_reasoning & reasoning)
    : analyze_base(tmpl) {
    LOG_DBG(ANSI_ORANGE "Phase 2: Content analysis\n" ANSI_RESET);

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
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
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
            mode      = content_mode::PLAIN;
            found_plain_content = true;
        } else if (reasoning.mode != reasoning_mode::NONE && !reasoning.end.empty() &&
                   diff_reasoning.left.find(reasoning.end) != std::string::npos) {
            std::string post_closed_reasoning = diff_reasoning.left.substr(
                diff_reasoning.left.find(reasoning.end) + reasoning.end.length());
            if (trim_whitespace(post_closed_reasoning) == "Response text") {
                LOG_DBG("C1: No content markers after stripping reasoning close marker\n");
                mode      = content_mode::PLAIN;
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
        size_t      pos          = pure_content.find("Response text");
        if (pos == std::string::npos) {
            LOG_DBG(ANSI_ORANGE "%s: Error: response text not found - improper template application?\n" ANSI_RESET, __func__);
            return;
        }
        start = trim_leading_whitespace(pure_content.substr(0, pos));
        end = trim_leading_whitespace(pure_content.substr(pos + 13));  // 13 - len of "Response text"
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

    analyze_tool_call_format(tool_section, "foofoo", "first", reasoning);
}

void analyze_tools::analyze_tool_call_format(const std::string &       haystack,
                                             const std::string &       fun_name_needle,
                                             const std::string &       arg_name_needle,
                                             const analyze_reasoning & reasoning) {
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
        format.mode = tool_format::JSON_NATIVE;
    } else if (in_json_haystack(arg_name_needle)) {
        format.mode = tool_format::TAG_WITH_JSON;
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
    // TODO: support for generated (not provided) tool call IDs
    auto space_or_bracket = [](bool opening, char c) -> bool {
        return std::isspace(c) || (opening ? c == '[' : c == ']');
    };
    // now let's check if we're in an array construction, mark it if so and get out of it
    if (json_start > 0 && space_or_bracket(true, clean_haystack[json_start - 1])) {
        for (--json_start; space_or_bracket(true, clean_haystack[json_start]) && json_start > 0; json_start--) {
            if (clean_haystack[json_start] == '[') {
                format.tools_array_wrapped = true;
                break;
            }
        }
        if (!format.tools_array_wrapped) {
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
                        format.per_call_start = haystack_split[seg].value + format.per_call_start;
                    } else {
                        format.section_start = haystack_split[seg].value + format.section_start;
                    }
                } else {
                    if (had_marker) {
                        format.section_start = haystack_split[seg].value + format.section_start;
                    } else {
                        format.per_call_start = haystack_split[seg].value + format.per_call_start;
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
                        format.section_end = haystack_split[seg].value + format.section_end;
                    } else {
                        format.per_call_end = haystack_split[seg].value + format.per_call_end;
                    }
                } else {
                    if (had_marker) {
                        format.per_call_end = haystack_split[seg].value + format.per_call_end;
                    } else {
                        format.section_end = haystack_split[seg].value + format.section_end;
                    }
                }
                if (backtracked_so_far >= how_many_markers) {
                    break;
                }
            }
        } else {
            for (int seg = 0; seg < where_is_nemo; seg++) {
                format.section_start += haystack_split[seg].value;
            }
            for (size_t seg = haystack_split.size() - 1; seg > (size_t) where_is_nemo; seg--) {
                format.section_end = haystack_split[seg].value + format.section_end;
                if (haystack_split[seg].type == segment_type::MARKER) {
                    break;
                }
            }
        }
    } else {
        format.section_start += haystack_split[0].value;
        for (size_t seg = haystack_split.size() - 1; seg > (size_t) where_is_nemo; seg--) {
            format.section_end = haystack_split[seg].value + format.section_end;
            if (haystack_split[seg].type == segment_type::MARKER) {
                break;
            }
        }
    }
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
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_barbar }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (diff.left.find("foofoo") != std::string::npos && diff.right.find("barbar") != std::string::npos) {
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

        auto seg = segmentize_markers(diff.left);
        for (const auto & s : seg) {
            if (s.value.find("foofoo") == std::string::npos) {
                function.name_prefix += s.value;
            } else {
                size_t      pos  = s.value.find("foofoo");
                std::string pre  = s.value.substr(0, pos);
                std::string post = s.value.substr(pos + 6);  // 6 = len("foofoo")
                function.name_prefix += pre;
                function.name_suffix += post;
                break;
            }
        }

        auto   seg_suf           = segmentize_markers(diff.suffix);
        size_t stop              = 0;
        size_t stop_internal_pos = 0;
        for (const auto & ss : seg_suf) {
            bool has_needle = false;
            if (format.mode == tool_format::TAG_WITH_JSON) {
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
            if (format.mode == tool_format::TAG_WITH_TAGGED) {
                size_t how_far = 0;
                if (stop > 0) {
                    if (seg_suf[stop].type == segment_type::MARKER) {
                        how_far = stop;
                    } else {
                        how_far = stop - 1;
                    }
                    for (size_t i = 0; i < how_far; i++) {
                        function.name_suffix += seg_suf[i].value;
                    }
                }
            } else {
                for (size_t i = 0; i < stop; i++) {
                    function.name_suffix += seg_suf[i].value;
                }
                const std::string & stopper = seg_suf[stop].value;
                function.name_suffix += stopper.substr(0, stop_internal_pos);
            }
        }

        // now just to find the closer
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
            auto   closer_seg             = segmentize_markers(closer_suffix);
            bool   need_to_eat_arg_marker = (format.mode == tool_format::TAG_WITH_TAGGED);
            size_t last_arg_seg           = closer_seg.size() - 1;
            for (int i = (int) closer_seg.size() - 1; i >= 0; i--) {
                if (closer_seg[i].value.find("YYYY") != std::string::npos) {
                    last_arg_seg = i;
                }
            }
            if (format.mode == tool_format::TAG_WITH_JSON) {
                const auto & entire_seg = closer_seg[last_arg_seg].value;
                size_t       pos        = entire_seg.find_last_of("}]");
                if (pos != std::string::npos && pos < entire_seg.size() - 1) {
                    function.close = trim_leading_whitespace(entire_seg.substr(pos + 1));
                }
            }
            for (size_t i = last_arg_seg + 1; i < closer_seg.size(); i++) {
                if (closer_seg[i].type == segment_type::MARKER) {
                    if (need_to_eat_arg_marker) {
                        need_to_eat_arg_marker = false;
                    } else {
                        function.close += closer_seg[i].value;
                    }
                } else if (!need_to_eat_arg_marker) {
                    function.close += closer_seg[i].value;
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
                    arguments.name_prefix = trim_whitespace(common_prefix);
                    std::string suffix_left        = left_remainder.substr(5, left_close - 5);
                    std::string suffix_right       = right_remainder.substr(6, right_close - 6);
                    if (suffix_left == suffix_right) {
                        arguments.name_suffix = trim_leading_whitespace(suffix_left);
                    }
                }
            }
        } else if (diff.left.substr(0, 5) == "first" && diff.right.substr(0, 6) == "second") {
            // we most likely have actual markers for argument names
            auto pre_seg = segmentize_markers(diff.prefix);
            for (int i = pre_seg.size() - 1; i >= 0; i--) {
                arguments.name_prefix = arguments.name_prefix + pre_seg[i].value;
                if (pre_seg[i].type == segment_type::MARKER) {
                    break;
                }
            }
            auto left_seg = segmentize_markers(diff.left);
            if (left_seg.size() == 1) {  // only the name + maybe extra whitespace / normal chars in differing part
                arguments.name_suffix = diff.left.substr(5);
                auto suf_seg= segmentize_markers(diff.suffix);
                for (size_t i = 0; i < suf_seg.size(); i++) {
                    arguments.name_suffix += suf_seg[i].value;
                    if (suf_seg[i].type == segment_type::MARKER) {
                        if (i < suf_seg.size() - 2 && suf_seg[i + 1].type == segment_type::TEXT &&
                            trim_whitespace(suf_seg[i + 1].value).empty()) {
                            // we need to include post-marker whitespace/newlines as well
                            arguments.name_suffix += suf_seg[i + 1].value;
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
                    arguments.name_suffix += to_add;
                    if (left_seg[i].type == segment_type::MARKER) {
                        if (i < left_seg.size() - 2 && left_seg[i + 1].type == segment_type::TEXT &&
                            trim_whitespace(left_seg[i + 1].value).empty()) {
                            // we need to include post-marker whitespace/newlines as well
                            arguments.name_suffix += left_seg[i + 1].value;
                        }
                        break;
                    }
                }
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
        std::string arg_name_ending = "first" + arguments.name_suffix;
        std::string prefix          = diff.prefix;
        if (prefix.rfind(arg_name_ending) != std::string::npos) {
            prefix = prefix.substr(prefix.rfind(arg_name_ending) + arg_name_ending.size());
        }
        if (!prefix.empty()) {
            auto seg_pre = segmentize_markers(prefix);
            for (int i = seg_pre.size() - 1; i >= 0; i--) {
                arguments.value_prefix = seg_pre[i].value + arguments.value_prefix;
                if (seg_pre[i].type == segment_type::MARKER) {
                    break;
                }
            }
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
        std::string separator        = until_common_prefix(diff.right, "first", "second");
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
            call_id.pos = call_id_position::BETWEEN_FUNC_AND_ARGS;

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
                call_id.prefix = marker_before_id;
            } else {
                // Fallback: look for the last marker in after_func
                for (int i = (int) segments.size() - 1; i >= 0; i--) {
                    if (segments[i].type == segment_type::MARKER) {
                        call_id.prefix = segments[i].value;
                        break;
                    }
                }
            }

            // Extract call_id_suffix: the first marker in the suffix before args
            auto suffix_segments = segmentize_markers(diff.suffix);
            for (size_t i = 0; i < suffix_segments.size(); i++) {
                if (suffix_segments[i].type == segment_type::MARKER) {
                    call_id.suffix = suffix_segments[i].value;
                    break;
                }
                // Stop if we hit the args
                if (suffix_segments[i].value.find('{') != std::string::npos) {
                    break;
                }
            }
        } else if (args_in_prefix != std::string::npos) {
            // Args are in prefix, so call_id is POST_ARGS
            call_id.pos = call_id_position::POST_ARGS;

            // Extract markers from between args and the ID
            std::string after_args    = diff.prefix.substr(args_in_prefix);
            size_t      closing_brace = after_args.rfind('}');
            if (closing_brace != std::string::npos) {
                std::string between_args_and_id = after_args.substr(closing_brace + 1);
                auto        segments            = segmentize_markers(between_args_and_id);
                for (int i = (int) segments.size() - 1; i >= 0; i--) {
                    if (segments[i].type == segment_type::MARKER) {
                        call_id.prefix = segments[i].value;
                        break;
                    }
                }
            }

            // call_id_suffix would be in the suffix (first marker)
            auto suffix_segments = segmentize_markers(diff.suffix);
            for (const auto & seg : suffix_segments) {
                if (seg.type == segment_type::MARKER) {
                    call_id.suffix = seg.value;
                    break;
                }
            }
        }
    } else if (func_name_in_suffix != std::string::npos && func_name_in_prefix == std::string::npos) {
        // Function name is only in suffix - call_id is PRE_FUNC_NAME
        call_id.pos = call_id_position::PRE_FUNC_NAME;

        // Extract call_id_prefix from prefix (last marker before the common_id_part)
        auto prefix_segments = segmentize_markers(diff.prefix);
        for (int i = (int) prefix_segments.size() - 1; i >= 0; i--) {
            if (prefix_segments[i].type == segment_type::MARKER) {
                call_id.prefix = prefix_segments[i].value;
                break;
            }
        }

        // Extract call_id_suffix from suffix (first marker before func_name)
        std::string before_func     = diff.suffix.substr(0, func_name_in_suffix);
        auto        suffix_segments = segmentize_markers(before_func);
        for (const auto & seg : suffix_segments) {
            if (seg.type == segment_type::MARKER) {
                call_id.suffix = seg.value;
                break;
            }
        }
    }

    // When call_id is detected, per_call_end may have been incorrectly set to include
    // the call_id_suffix and sample args. Clear it if it starts with call_id_suffix.
    if (call_id.pos != call_id_position::NONE && !call_id.suffix.empty() &&
        format.per_call_end.find(call_id.suffix) == 0) {
        format.per_call_end.clear();
    }
}

}  // namespace autoparser
