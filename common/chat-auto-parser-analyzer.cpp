#include "chat-auto-parser.h"
#include "chat-auto-parser-helpers.h"

#include "log.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

using json = nlohmann::ordered_json;

// ============================================================================
// Internal Implementation Details
// ============================================================================

// Internal structure for differential analysis (not exposed in header)
struct InternalDiscoveredPattern {
    std::string tool_call_opener;
    std::string tool_call_closer;
    std::string function_opener;
    std::string function_closer;
    std::string function_name_suffix;
    std::string parameter_opener;
    std::string parameter_closer;
    std::string argument_separator;
    std::string parameter_key_prefix;
    std::string parameter_key_suffix;
    std::string tool_call_start_marker;
    std::string tool_call_end_marker;
    std::string reasoning_start_marker;
    std::string reasoning_end_marker;
    std::string content_start_marker;
    std::string content_end_marker;
    std::string tool_name_field  = "name";
    std::string tool_args_field  = "arguments";
    std::string tool_id_field;
};

// Internal enum for format classification
enum InternalToolFormat {
    FORMAT_JSON_NATIVE,
    FORMAT_XML_CONSTRUCTED,
    FORMAT_CONTENT_ONLY,
    FORMAT_UNKNOWN
};

// Forward declarations for internal helpers
static std::string find_string_difference(const std::string & base, const std::string & extended);
static std::string extract_json_field_name(const std::string & opener, const std::string & default_name,
                                           const std::vector<std::string> & candidates);
static InternalDiscoveredPattern extract_patterns_from_differences(const std::string & tool1_diff,
                                                                    const std::string & tool2_diff,
                                                                    const std::string & tool3_diff);
static InternalToolFormat determine_format_from_patterns(const InternalDiscoveredPattern & patterns);
static InternalDiscoveredPattern analyze_by_differential(const minja::chat_template & tmpl);
static std::string find_closing_pattern(const std::string & diff, size_t func_pos);
static std::string find_tool_call_start(const std::string & diff);
static std::string find_tool_call_end(const std::string & diff, size_t func_pos);
static std::string infer_tool_call_opener(const std::string & diff1, const std::string & diff2,
                                          const std::string & diff3);
static std::string infer_tool_call_closer(const std::string & diff1, const std::string & diff2,
                                          const std::string & diff3);

// ============================================================================
// Public API: Unified Two-Phase Analysis
// ============================================================================

TemplateAnalysisResult TemplateAnalyzer::analyze_template(const minja::chat_template & tmpl) {
    LOG_DBG("=== STARTING UNIFIED TEMPLATE ANALYSIS ===\n");

    TemplateAnalysisResult result;

    // Phase 1: Analyze content and reasoning structure (no tools involved)
    result.content = analyze_content_structure(tmpl);

    // Phase 2: Analyze tool call structure (layered on Phase 1)
    result.tools = analyze_tool_structure(tmpl, result.content);

    // Post-processing: Extract reasoning markers from tool_section_start if Phase 1 didn't detect them
    // Some templates (like Command-R7B) include reasoning markers in tool outputs but not in prompts
    if (result.content.reasoning_start.empty() && !result.tools.tool_section_start.empty()) {
        // Known reasoning end marker patterns that might be embedded in tool_section_start
        std::vector<std::pair<std::string, std::string>> reasoning_patterns = {
            { "<|START_THINKING|>", "<|END_THINKING|>" },
            { "<|START_THOUGHT|>", "<|END_THOUGHT|>" },
            { "<|START_REASON|>", "<|END_REASON|>" },
            { "<think>", "</think>" },
            { "<thinking>", "</thinking>" },
        };

        for (const auto & [start_marker, end_marker] : reasoning_patterns) {
            size_t end_pos = result.tools.tool_section_start.find(end_marker);
            if (end_pos != std::string::npos) {
                // Found reasoning end marker in tool_section_start
                // Extract it and clean up tool_section_start
                result.content.reasoning_start = start_marker;
                result.content.reasoning_end = end_marker;
                result.content.reasoning_mode = ContentStructure::REASONING_OPTIONAL;

                // Clean up tool_section_start: remove everything before and including the end marker
                size_t after_end = end_pos + end_marker.length();
                if (after_end < result.tools.tool_section_start.length()) {
                    result.tools.tool_section_start = result.tools.tool_section_start.substr(after_end);
                    // Trim leading whitespace
                    size_t first_non_ws = result.tools.tool_section_start.find_first_not_of(" \t\n\r");
                    if (first_non_ws != std::string::npos && first_non_ws > 0) {
                        result.tools.tool_section_start = result.tools.tool_section_start.substr(first_non_ws);
                    }
                }

                LOG_DBG("Post-processing: Extracted reasoning markers from tool_section_start\n");
                LOG_DBG("  reasoning_start: '%s', reasoning_end: '%s'\n",
                        result.content.reasoning_start.c_str(), result.content.reasoning_end.c_str());
                LOG_DBG("  cleaned tool_section_start: '%s'\n", result.tools.tool_section_start.c_str());
                break;
            }
        }
    }

    // Collect preserved tokens from both phases
    collect_preserved_tokens(result);

    LOG_DBG("=== UNIFIED TEMPLATE ANALYSIS COMPLETE ===\n");
    LOG_DBG("Content structure:\n");
    LOG_DBG("  reasoning_mode: %d\n", static_cast<int>(result.content.reasoning_mode));
    LOG_DBG("  reasoning_start: '%s'\n", result.content.reasoning_start.c_str());
    LOG_DBG("  reasoning_end: '%s'\n", result.content.reasoning_end.c_str());
    LOG_DBG("  content_mode: %d\n", static_cast<int>(result.content.content_mode));
    LOG_DBG("  content_start: '%s'\n", result.content.content_start.c_str());
    LOG_DBG("  content_end: '%s'\n", result.content.content_end.c_str());
    LOG_DBG("Tool structure:\n");
    LOG_DBG("  supports_tools: %s\n", result.tools.supports_tools ? "true" : "false");
    LOG_DBG("  function_format: %d\n", static_cast<int>(result.tools.function_format));
    LOG_DBG("  argument_format: %d\n", static_cast<int>(result.tools.argument_format));
    LOG_DBG("  tool_section_start: '%s'\n", result.tools.tool_section_start.c_str());
    LOG_DBG("  tool_section_end: '%s'\n", result.tools.tool_section_end.c_str());

    return result;
}

// ============================================================================
// PHASE 1: Content and Reasoning Analysis
// ============================================================================

ContentStructure TemplateAnalyzer::analyze_content_structure(const minja::chat_template & tmpl) {
    LOG_DBG("=== PHASE 1: ANALYZING CONTENT STRUCTURE ===\n");

    ContentStructure cs;

    // Step 1: Detect reasoning markers by toggling enable_thinking
    detect_reasoning_markers(tmpl, cs);

    // Step 2: Detect content wrapping markers
    detect_content_markers(tmpl, cs);

    // Step 3: Determine reasoning mode (NONE, OPTIONAL, FORCED_OPEN)
    minja::chat_template_inputs inputs;
    inputs.messages = {
        {{ "role", "user" }, { "content", "Hello" }}
    };
    inputs.add_generation_prompt = true;
    inputs.extra_context["enable_thinking"] = true;

    std::string prompt;
    try {
        prompt = tmpl.apply(inputs);
    } catch (...) {
        LOG_DBG("Failed to render template for reasoning mode detection\n");
        return cs;
    }

    cs.reasoning_mode = detect_reasoning_mode(tmpl, cs, prompt);

    LOG_DBG("Phase 1 complete: reasoning_mode=%d, content_mode=%d\n",
            static_cast<int>(cs.reasoning_mode), static_cast<int>(cs.content_mode));

    return cs;
}

void TemplateAnalyzer::detect_reasoning_markers(const minja::chat_template & tmpl, ContentStructure & cs) {
    LOG_DBG("=== DETECTING REASONING MARKERS ===\n");

    // Method 1: Compare outputs with reasoning_content field present vs absent
    json reasoning_msg = {
        { "role",              "assistant" },
        { "content",           "CONTENT_MARKER" },
        { "reasoning_content", "THOUGHT_MARKER" }
    };

    json base_msg = {
        { "role",    "assistant" },
        { "content", "CONTENT_MARKER" }
    };

    minja::chat_template_inputs inputs;

    inputs.messages = { reasoning_msg };
    std::string reasoning_output;
    try {
        reasoning_output = tmpl.apply(inputs);
    } catch (...) {
        LOG_DBG("Failed to render template with reasoning_content\n");
        reasoning_output = "";
    }

    inputs.messages = { base_msg };
    std::string base_output;
    try {
        base_output = tmpl.apply(inputs);
    } catch (...) {
        LOG_DBG("Failed to render base template\n");
        base_output = "";
    }

    // If outputs differ and we can find THOUGHT_MARKER, extract the reasoning markers
    if (!reasoning_output.empty() && reasoning_output != base_output) {
        size_t thought_pos = reasoning_output.find("THOUGHT_MARKER");
        size_t content_pos = reasoning_output.find("CONTENT_MARKER");

        if (thought_pos != std::string::npos && content_pos != std::string::npos && content_pos > thought_pos) {
            // Extract what's between THOUGHT_MARKER and CONTENT_MARKER as the end marker
            size_t thought_end = thought_pos + strlen("THOUGHT_MARKER");
            cs.reasoning_end = reasoning_output.substr(thought_end, content_pos - thought_end);

            // Find what's before THOUGHT_MARKER by comparing with base_output
            size_t diff_start = 0;
            while (diff_start < base_output.length() && diff_start < reasoning_output.length() &&
                   base_output[diff_start] == reasoning_output[diff_start]) {
                diff_start++;
            }

            if (diff_start < thought_pos) {
                cs.reasoning_start = reasoning_output.substr(diff_start, thought_pos - diff_start);
            }

            trim_whitespace(cs.reasoning_start);
            trim_whitespace(cs.reasoning_end);

            LOG_DBG("Method 1: Found reasoning markers via reasoning_content field\n");
            LOG_DBG("  start: '%s', end: '%s'\n", cs.reasoning_start.c_str(), cs.reasoning_end.c_str());
        }
    }

    // Method 2: Compare prompts with enable_thinking true vs false
    if (cs.reasoning_start.empty()) {
        LOG_DBG("Method 1 failed, trying Method 2 (enable_thinking toggle)\n");

        json user_msg = {
            { "role", "user" },
            { "content", "Hello" }
        };

        minja::chat_template_inputs inputs_prompt;
        inputs_prompt.messages = { user_msg };
        inputs_prompt.add_generation_prompt = true;

        inputs_prompt.extra_context["enable_thinking"] = false;
        std::string prompt_no_think;
        try {
            prompt_no_think = tmpl.apply(inputs_prompt);
        } catch (...) {
            prompt_no_think = "";
        }

        inputs_prompt.extra_context["enable_thinking"] = true;
        std::string prompt_think;
        try {
            prompt_think = tmpl.apply(inputs_prompt);
        } catch (...) {
            prompt_think = "";
        }

        if (!prompt_think.empty() && prompt_think != prompt_no_think) {
            // Find the difference - this should be the reasoning start marker
            size_t diff_pos = 0;
            while (diff_pos < prompt_no_think.length() && diff_pos < prompt_think.length() &&
                   prompt_no_think[diff_pos] == prompt_think[diff_pos]) {
                diff_pos++;
            }

            std::string diff = prompt_think.substr(diff_pos);

            // Only use if it looks like a tag
            if (diff.find('<') != std::string::npos || diff.find('[') != std::string::npos) {
                cs.reasoning_start = diff;
                cs.reasoning_end = create_closing_tag(diff);
                trim_whitespace(cs.reasoning_start);
                trim_whitespace(cs.reasoning_end);

                LOG_DBG("Method 2: Found reasoning markers via enable_thinking toggle\n");
                LOG_DBG("  start: '%s', end: '%s'\n", cs.reasoning_start.c_str(), cs.reasoning_end.c_str());
            }
        }
    }

    // Method 3: Check if the prompt ends with an unclosed reasoning tag
    if (cs.reasoning_start.empty()) {
        LOG_DBG("Method 2 failed, trying Method 3 (prompt ending with open tag)\n");

        json user_msg = {
            { "role", "user" },
            { "content", "Hello" }
        };

        minja::chat_template_inputs inputs_prompt;
        inputs_prompt.messages = { user_msg };
        inputs_prompt.add_generation_prompt = true;
        inputs_prompt.extra_context["enable_thinking"] = true;

        std::string prompt;
        try {
            prompt = tmpl.apply(inputs_prompt);
        } catch (...) {
            prompt = "";
        }

        if (!prompt.empty()) {
            // Save trailing whitespace before trimming
            std::string trailing_ws;
            size_t end_pos = prompt.length();
            while (end_pos > 0 && (prompt[end_pos - 1] == '\n' || prompt[end_pos - 1] == '\r')) {
                trailing_ws = prompt[end_pos - 1] + trailing_ws;
                end_pos--;
            }

            trim_trailing_newlines(prompt);

            // Find the last tag in the prompt
            size_t last_open_angle = prompt.rfind('<');
            size_t last_close_angle = prompt.rfind('>');

            // Check for closed tags at the end
            if (last_open_angle != std::string::npos && last_close_angle != std::string::npos &&
                last_close_angle == prompt.length() - 1 && last_close_angle > last_open_angle) {
                std::string tag = prompt.substr(last_open_angle);

                // Check if this looks like a reasoning tag (not a role marker)
                std::vector<std::string> blacklisted_tags = {
                    "<|CHATBOT_TOKEN|>", "<|SYSTEM_TOKEN|>", "<|USER_TOKEN|>", "<|ASSISTANT_TOKEN|>",
                    "<|im_start|>", "<|im_end|>", "<|start_of_role|>", "<|end_of_role|>",
                    "<|end_of_text|>", "<|end|>", "<|assistant|>", "<|user|>", "<|system|>",
                    "<assistant>", "<user>", "<system>"
                };

                bool is_blacklisted = false;
                for (const auto & blacklisted : blacklisted_tags) {
                    if (tag == blacklisted) {
                        is_blacklisted = true;
                        break;
                    }
                }

                // Check if it looks like a thinking/reasoning tag
                std::string lower_tag = tag;
                std::transform(lower_tag.begin(), lower_tag.end(), lower_tag.begin(), ::tolower);
                bool looks_like_reasoning = lower_tag.find("think") != std::string::npos ||
                                           lower_tag.find("reason") != std::string::npos ||
                                           lower_tag.find("thought") != std::string::npos;

                if (!is_blacklisted && looks_like_reasoning) {
                    cs.reasoning_start = tag + trailing_ws;
                    cs.reasoning_end = create_closing_tag(tag) + trailing_ws;
                    trim_whitespace(cs.reasoning_start);
                    trim_whitespace(cs.reasoning_end);

                    LOG_DBG("Method 3: Found reasoning markers via prompt ending with tag\n");
                    LOG_DBG("  start: '%s', end: '%s'\n", cs.reasoning_start.c_str(), cs.reasoning_end.c_str());
                }
            }
        }
    }

    // Method 4: Look for adjacent opening/closing tag pairs with common content in prompt
    // This detects patterns like <think></think>, <|START_THINKING|><|END_THINKING|>, [think][/think]
    if (cs.reasoning_start.empty()) {
        LOG_DBG("Method 3 failed, trying Method 4 (adjacent tag pairs with common content)\n");

        json user_msg = {
            { "role", "user" },
            { "content", "Hello" }
        };

        minja::chat_template_inputs inputs_prompt;
        inputs_prompt.messages = { user_msg };
        inputs_prompt.add_generation_prompt = true;
        // Try with thinking disabled - templates may output empty thinking blocks
        inputs_prompt.extra_context["enable_thinking"] = false;

        std::string prompt;
        try {
            prompt = tmpl.apply(inputs_prompt);
        } catch (...) {
            prompt = "";
        }

        if (!prompt.empty()) {
            // Look for patterns like <tag1><tag2> or <tag1>...<tag2> where tag1 and tag2 share a common word
            // Common patterns:
            //   <think></think>
            //   <|START_THINKING|><|END_THINKING|>
            //   [think][/think]

            // Find potential tag pairs by looking for closing tags that immediately follow opening tags
            // Pattern: opening tag followed by closing tag with same keyword
            std::vector<std::tuple<std::string, std::string, std::string>> tag_patterns = {
                // (opening pattern, closing pattern, keyword to match)
                { "<|START_", "<|END_", "THINKING" },
                { "<|START_", "<|END_", "THOUGHT" },
                { "<|START_", "<|END_", "REASON" },
                { "<think>", "</think>", "" },
                { "<Think>", "</Think>", "" },
                { "<THINK>", "</THINK>", "" },
                { "[think]", "[/think]", "" },
                { "[THINK]", "[/THINK]", "" },
                { "<thinking>", "</thinking>", "" },
                { "<THINKING>", "</THINKING>", "" },
                { "<|think|>", "<|/think|>", "" },
            };

            for (const auto & [open_prefix, close_prefix, keyword] : tag_patterns) {
                size_t open_pos = prompt.find(open_prefix);
                if (open_pos == std::string::npos) continue;

                std::string start_tag, end_tag;

                if (!keyword.empty()) {
                    // Pattern like <|START_THINKING|><|END_THINKING|>
                    std::string full_open = open_prefix + keyword;
                    size_t full_open_pos = prompt.find(full_open);
                    if (full_open_pos == std::string::npos) continue;

                    // Find the end of this tag (look for |> or >)
                    size_t tag_end = prompt.find("|>", full_open_pos + full_open.length());
                    if (tag_end == std::string::npos) {
                        tag_end = prompt.find(">", full_open_pos + full_open.length());
                    }
                    if (tag_end == std::string::npos) continue;

                    start_tag = prompt.substr(full_open_pos, tag_end - full_open_pos + (prompt[tag_end] == '|' ? 2 : 1));

                    // Look for the corresponding end tag
                    std::string expected_close = close_prefix + keyword;
                    size_t close_pos = prompt.find(expected_close, tag_end);
                    if (close_pos == std::string::npos) continue;

                    // Find end of close tag
                    size_t close_end = prompt.find("|>", close_pos + expected_close.length());
                    if (close_end == std::string::npos) {
                        close_end = prompt.find(">", close_pos + expected_close.length());
                    }
                    if (close_end == std::string::npos) continue;

                    end_tag = prompt.substr(close_pos, close_end - close_pos + (prompt[close_end] == '|' ? 2 : 1));
                } else {
                    // Simple pattern like <think></think>
                    start_tag = open_prefix;
                    size_t close_pos = prompt.find(close_prefix, open_pos + start_tag.length());
                    if (close_pos == std::string::npos) continue;
                    end_tag = close_prefix;
                }

                // Verify the tags are adjacent or nearly adjacent (only whitespace between)
                size_t start_end_pos = prompt.find(start_tag) + start_tag.length();
                size_t end_start_pos = prompt.find(end_tag, start_end_pos);
                if (end_start_pos != std::string::npos) {
                    std::string between = prompt.substr(start_end_pos, end_start_pos - start_end_pos);
                    // Allow only whitespace between the tags (empty thinking block)
                    bool only_whitespace = true;
                    for (char c : between) {
                        if (!std::isspace(static_cast<unsigned char>(c))) {
                            only_whitespace = false;
                            break;
                        }
                    }

                    if (only_whitespace) {
                        cs.reasoning_start = start_tag;
                        cs.reasoning_end = end_tag;
                        LOG_DBG("Method 4: Found reasoning markers via adjacent tag pairs\n");
                        LOG_DBG("  start: '%s', end: '%s'\n", cs.reasoning_start.c_str(), cs.reasoning_end.c_str());
                        break;
                    }
                }
            }
        }
    }

    if (cs.reasoning_start.empty()) {
        LOG_DBG("No reasoning markers detected\n");
    }
}

void TemplateAnalyzer::detect_content_markers(const minja::chat_template & tmpl, ContentStructure & cs) {
    LOG_DBG("=== DETECTING CONTENT MARKERS ===\n");

    // Render template with a unique content marker
    json user_msg = {
        { "role", "user" },
        { "content", "Hello" }
    };
    json assistant_msg = {
        { "role", "assistant" },
        { "content", "UNIQUE_CONTENT_12345" }
    };

    minja::chat_template_inputs inputs;
    inputs.messages = { user_msg, assistant_msg };
    // Try with thinking enabled first (some templates only wrap content when reasoning is present)
    inputs.extra_context["thinking"] = true;
    inputs.extra_context["enable_thinking"] = true;

    std::string output_with_thinking;
    try {
        output_with_thinking = tmpl.apply(inputs);
    } catch (...) {
        output_with_thinking = "";
    }

    // Also render without thinking
    inputs.extra_context["thinking"] = false;
    inputs.extra_context["enable_thinking"] = false;

    std::string output_no_thinking;
    try {
        output_no_thinking = tmpl.apply(inputs);
    } catch (...) {
        output_no_thinking = "";
    }

    // Check both outputs for content markers
    auto find_content_markers = [&](const std::string & output) -> std::pair<std::string, std::string> {
        size_t marker_pos = output.find("UNIQUE_CONTENT_12345");
        if (marker_pos == std::string::npos) {
            return {"", ""};
        }

        // Known content marker patterns
        std::vector<std::pair<std::string, std::string>> patterns = {
            { "<|START_RESPONSE|>", "<|END_RESPONSE|>" },
            { "<|response|>", "<|/response|>" },
            { "<response>", "</response>" },
            { "<output>", "</output>" },
            { "<answer>", "</answer>" },
        };

        for (const auto & [start_pattern, end_pattern] : patterns) {
            size_t start_pos = output.rfind(start_pattern, marker_pos);
            if (start_pos != std::string::npos) {
                // Check that there's only whitespace between the start pattern and our marker
                std::string between = output.substr(start_pos + start_pattern.length(),
                                                    marker_pos - start_pos - start_pattern.length());
                size_t first_non_ws = between.find_first_not_of(" \t\n\r");
                if (first_non_ws == std::string::npos) {
                    // Found valid start marker, look for end marker
                    size_t marker_end = marker_pos + strlen("UNIQUE_CONTENT_12345");
                    size_t end_pos = output.find(end_pattern, marker_end);
                    if (end_pos != std::string::npos) {
                        std::string after = output.substr(marker_end, end_pos - marker_end);
                        size_t first_non_ws_after = after.find_first_not_of(" \t\n\r");
                        if (first_non_ws_after == std::string::npos) {
                            return { start_pattern, end_pattern };
                        }
                    }
                }
            }
        }

        return {"", ""};
    };

    auto [start_with_thinking, end_with_thinking] = find_content_markers(output_with_thinking);
    auto [start_no_thinking, end_no_thinking] = find_content_markers(output_no_thinking);

    if (!start_with_thinking.empty() && !start_no_thinking.empty()) {
        // Content is always wrapped
        cs.content_mode = ContentStructure::CONTENT_ALWAYS_WRAPPED;
        cs.content_start = start_with_thinking;
        cs.content_end = end_with_thinking;
        LOG_DBG("Content markers found in both thinking modes (ALWAYS_WRAPPED)\n");
    } else if (!start_with_thinking.empty() && start_no_thinking.empty()) {
        // Content is wrapped only when reasoning is present
        cs.content_mode = ContentStructure::CONTENT_WRAPPED_WITH_REASONING;
        cs.content_start = start_with_thinking;
        cs.content_end = end_with_thinking;
        LOG_DBG("Content markers found only with thinking enabled (WRAPPED_WITH_REASONING)\n");
    } else if (!start_no_thinking.empty()) {
        // Unusual: content wrapped without thinking but not with? Use what we found
        cs.content_mode = ContentStructure::CONTENT_ALWAYS_WRAPPED;
        cs.content_start = start_no_thinking;
        cs.content_end = end_no_thinking;
        LOG_DBG("Content markers found only without thinking (treating as ALWAYS_WRAPPED)\n");
    } else {
        cs.content_mode = ContentStructure::CONTENT_PLAIN;
        LOG_DBG("No content markers detected (PLAIN)\n");
    }

    LOG_DBG("Content markers: start='%s', end='%s'\n", cs.content_start.c_str(), cs.content_end.c_str());
}

ContentStructure::ReasoningMode TemplateAnalyzer::detect_reasoning_mode(
    const minja::chat_template & tmpl,
    const ContentStructure & cs,
    const std::string & prompt) {

    (void)tmpl;  // Unused for now

    LOG_DBG("=== DETECTING REASONING MODE ===\n");

    // If no reasoning markers detected, mode is NONE
    if (cs.reasoning_start.empty()) {
        LOG_DBG("No reasoning markers, mode=REASONING_NONE\n");
        return ContentStructure::REASONING_NONE;
    }

    // Check if the prompt ends with the reasoning start marker (forced open)
    std::string trimmed_prompt = prompt;
    trim_trailing_newlines(trimmed_prompt);

    std::string trimmed_marker = cs.reasoning_start;
    trim_whitespace(trimmed_marker);

    if (string_ends_with(trimmed_prompt, trimmed_marker)) {
        LOG_DBG("Prompt ends with reasoning start marker, mode=REASONING_FORCED_OPEN\n");
        return ContentStructure::REASONING_FORCED_OPEN;
    }

    // Otherwise, reasoning is optional
    LOG_DBG("Reasoning markers present but not forced, mode=REASONING_OPTIONAL\n");
    return ContentStructure::REASONING_OPTIONAL;
}

// ============================================================================
// PHASE 2: Tool Call Structure Analysis
// ============================================================================

ToolCallStructure TemplateAnalyzer::analyze_tool_structure(
    const minja::chat_template & tmpl,
    const ContentStructure & content) {

    (void)content;  // May be used in future for better tool detection

    LOG_DBG("=== PHASE 2: ANALYZING TOOL STRUCTURE ===\n");

    ToolCallStructure ts;

    // Check if minja reports tool call support
    auto caps = tmpl.original_caps();
    ts.supports_tools = caps.supports_tool_calls;

    if (!ts.supports_tools) {
        LOG_DBG("Template does not support tool calls (per minja caps)\n");
    }

    // Use differential analysis to detect tool patterns
    auto discovered = analyze_by_differential(tmpl);
    auto format = determine_format_from_patterns(discovered);

    if (format == FORMAT_JSON_NATIVE) {
        ts.supports_tools = true;
        ts.function_format = ToolCallStructure::FUNC_JSON_OBJECT;
        ts.argument_format = ToolCallStructure::ARGS_JSON;
        ts.tool_section_start = discovered.tool_call_start_marker;
        ts.tool_section_end = discovered.tool_call_end_marker;
        ts.name_field = discovered.tool_name_field;
        ts.args_field = discovered.tool_args_field;
        ts.id_field = discovered.tool_id_field;

        // For JSON_NATIVE format, clean up tool_section_end to only include the closing tag
        // The differential analysis may include JSON closing braces (e.g., "}}\n</tool_call>")
        // but the parser handles JSON separately, so we only need the tag marker
        if (!ts.tool_section_end.empty()) {
            size_t tag_start = ts.tool_section_end.find("</");
            if (tag_start != std::string::npos) {
                size_t tag_end = ts.tool_section_end.find('>', tag_start);
                if (tag_end != std::string::npos) {
                    ts.tool_section_end = ts.tool_section_end.substr(tag_start, tag_end - tag_start + 1);
                }
            } else {
                // Try other closing patterns like ]<|END_ACTION|>
                tag_start = ts.tool_section_end.find("<|");
                if (tag_start != std::string::npos) {
                    size_t tag_end = ts.tool_section_end.find("|>", tag_start);
                    if (tag_end != std::string::npos) {
                        // Include the opening bracket if present
                        size_t bracket_pos = ts.tool_section_end.rfind(']', tag_start);
                        if (bracket_pos != std::string::npos && bracket_pos + 1 == tag_start) {
                            ts.tool_section_end = ts.tool_section_end.substr(bracket_pos, tag_end - bracket_pos + 2);
                        } else {
                            ts.tool_section_end = ts.tool_section_end.substr(tag_start, tag_end - tag_start + 2);
                        }
                    }
                }
            }
        }
    } else if (format == FORMAT_XML_CONSTRUCTED) {
        ts.supports_tools = true;
        ts.function_format = ToolCallStructure::FUNC_TAG_WITH_NAME;
        ts.tool_section_start = discovered.tool_call_start_marker;
        ts.tool_section_end = discovered.tool_call_end_marker;

        // Extract function tag patterns
        if (!discovered.function_opener.empty()) {
            if (discovered.function_opener == ">>>") {
                // Functionary-style prefix format: >>>function_name\n{args}
                ts.function_prefix = ">>>";
                ts.function_suffix = "\n";  // Function name ends with newline
                ts.function_close = "";      // No closing tag
            } else {
                size_t eq_pos = discovered.function_opener.find('=');
                if (eq_pos != std::string::npos) {
                    ts.function_prefix = discovered.function_opener.substr(0, eq_pos + 1);
                    ts.function_suffix = discovered.function_name_suffix;
                    ts.function_close = discovered.function_closer;
                }
            }
        }

        // Determine argument format
        if (!discovered.parameter_key_prefix.empty() && discovered.parameter_key_prefix.find('<') != std::string::npos) {
            ts.argument_format = ToolCallStructure::ARGS_TAGGED;
            ts.arg_prefix = discovered.parameter_key_prefix;
            ts.arg_suffix = discovered.parameter_key_suffix;
            ts.arg_close = discovered.parameter_closer;
            ts.arg_separator = discovered.argument_separator;
        } else {
            ts.argument_format = ToolCallStructure::ARGS_JSON;
        }
    }

    // Check for instruction-based tool formats in template source
    // Some templates (like Granite) describe the tool format in instructions rather than
    // using structured tool call rendering. Detect these patterns and override if needed.
    std::string src = tmpl.source();

    // Check for <|tool_call|> format (e.g., Granite)
    if (src.find("<|tool_call|>") != std::string::npos) {
        // Template uses instruction-based <|tool_call|> format
        ts.supports_tools = true;
        ts.function_format = ToolCallStructure::FUNC_JSON_OBJECT;
        ts.argument_format = ToolCallStructure::ARGS_JSON;
        ts.tool_section_start = "<|tool_call|>";
        ts.tool_section_end = "";  // No closing tag, just JSON array
        ts.name_field = "name";
        ts.args_field = "arguments";
        LOG_DBG("Detected instruction-based <|tool_call|> format from template source\n");
    }

    // Check for [TOOL_CALLS] format (e.g., Mistral Nemo)
    if (src.find("[TOOL_CALLS]") != std::string::npos && ts.tool_section_start.empty()) {
        // Template uses [TOOL_CALLS] format
        ts.supports_tools = true;
        ts.function_format = ToolCallStructure::FUNC_JSON_OBJECT;
        ts.argument_format = ToolCallStructure::ARGS_JSON;
        ts.tool_section_start = "[TOOL_CALLS]";
        ts.tool_section_end = "";  // No closing tag, just JSON array
        ts.name_field = "name";
        ts.args_field = "arguments";
        ts.id_field = "id";  // Mistral Nemo format includes tool call ID
        LOG_DBG("Detected instruction-based [TOOL_CALLS] format from template source\n");
    }

    // Check for >>> prefix format (Functionary v3.2)
    // Format: >>>function_name\n{args}
    // NOTE: The >>> is part of the generation prompt, so parsing starts AFTER >>>
    // The parser sees: function_name\n{args}
    if (src.find("'>>>' + tool_call") != std::string::npos ||
        src.find("'>>>' ~ tool_call") != std::string::npos ||
        src.find("\">>>\" + tool_call") != std::string::npos ||
        src.find("\">>>\" ~ tool_call") != std::string::npos) {
        ts.supports_tools = true;
        ts.function_format = ToolCallStructure::FUNC_TAG_WITH_NAME;
        ts.argument_format = ToolCallStructure::ARGS_JSON;
        ts.tool_section_start = "";  // No section marker
        ts.tool_section_end = "";
        // >>> is already in the prompt, so function_prefix is empty and we just need newline after name
        ts.function_prefix = "";
        ts.function_suffix = "\n";
        ts.function_close = "";
        LOG_DBG("Detected Functionary style format - parsing after >>> prefix\n");
    }

    // Check for DeepSeek R1 style format with special Unicode markers
    // Format: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>func_name\n```json\n{args}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    // Note: Using string concatenation to avoid hex escape sequence issues
    // ｜ = U+FF5C = EF BD 9C, ▁ = U+2581 = E2 96 81
    const std::string ds_pipe = "\xef\xbd\x9c";      // ｜
    const std::string ds_under = "\xe2\x96\x81";    // ▁

    if (src.find("<" + ds_pipe + "tool" + ds_under + "calls" + ds_under + "begin" + ds_pipe + ">") != std::string::npos ||
        src.find("tool_calls_begin") != std::string::npos) {
        // Check for the characteristic format markers
        bool has_sep_marker = src.find("<" + ds_pipe + "tool" + ds_under + "sep" + ds_pipe + ">") != std::string::npos ||
                              src.find("tool_sep") != std::string::npos;
        bool has_json_fence = src.find("```json") != std::string::npos;

        if (has_sep_marker && has_json_fence) {
            ts.supports_tools = true;
            ts.function_format = ToolCallStructure::FUNC_TAG_WITH_NAME;
            ts.argument_format = ToolCallStructure::ARGS_JSON;
            // Use the actual Unicode markers
            ts.tool_section_start = "<" + ds_pipe + "tool" + ds_under + "calls" + ds_under + "begin" + ds_pipe + ">";
            ts.tool_section_end = "<" + ds_pipe + "tool" + ds_under + "calls" + ds_under + "end" + ds_pipe + ">";
            // Function prefix includes the type marker and separator
            ts.function_prefix = "<" + ds_pipe + "tool" + ds_under + "call" + ds_under + "begin" + ds_pipe + ">function<" +
                                 ds_pipe + "tool" + ds_under + "sep" + ds_pipe + ">";
            ts.function_suffix = "\n```json\n";
            // Note: No leading \n in function_close because the JSON grammar's trailing space
            // consumes the newline after the JSON object. The close fence immediately follows.
            ts.function_close = "```<" + ds_pipe + "tool" + ds_under + "call" + ds_under + "end" + ds_pipe + ">";
            LOG_DBG("Detected DeepSeek R1 style format with special markers\n");
        }
    }

    // Fallback for raw JSON format: check if the template uses "parameters" instead of "arguments"
    // This handles templates like Llama 3.1 where differential analysis may not detect patterns
    if (ts.supports_tools && ts.function_format == ToolCallStructure::FUNC_JSON_OBJECT) {
        // Check which arguments field is actually used in the template
        bool has_parameters = src.find("\"parameters\":") != std::string::npos ||
                              src.find("\"parameters\"") != std::string::npos;
        bool has_arguments = src.find("\"arguments\":") != std::string::npos ||
                             src.find("\"arguments\"") != std::string::npos;

        // Prefer "parameters" if present and "arguments" is not, or if only "parameters" is used
        if (has_parameters && !has_arguments) {
            ts.args_field = "parameters";
            LOG_DBG("Override args_field to 'parameters' based on template source\n");
        }
    }

    LOG_DBG("Phase 2 complete: supports_tools=%s, function_format=%d, argument_format=%d\n",
            ts.supports_tools ? "true" : "false",
            static_cast<int>(ts.function_format),
            static_cast<int>(ts.argument_format));

    return ts;
}

// ============================================================================
// Preserved Token Collection
// ============================================================================

void TemplateAnalyzer::collect_preserved_tokens(TemplateAnalysisResult & result) {
    LOG_DBG("=== COLLECTING PRESERVED TOKENS ===\n");

    std::vector<std::string> tokens;

    // Add reasoning markers
    if (!result.content.reasoning_start.empty()) {
        tokens.push_back(result.content.reasoning_start);
    }
    if (!result.content.reasoning_end.empty()) {
        tokens.push_back(result.content.reasoning_end);
    }

    // Add content markers
    if (!result.content.content_start.empty()) {
        tokens.push_back(result.content.content_start);
    }
    if (!result.content.content_end.empty()) {
        tokens.push_back(result.content.content_end);
    }

    // Add tool section markers
    if (!result.tools.tool_section_start.empty()) {
        tokens.push_back(result.tools.tool_section_start);
    }
    if (!result.tools.tool_section_end.empty()) {
        tokens.push_back(result.tools.tool_section_end);
    }

    // Add function markers for tag-based formats
    if (result.tools.function_format == ToolCallStructure::FUNC_TAG_WITH_NAME) {
        if (!result.tools.function_prefix.empty()) {
            tokens.push_back(result.tools.function_prefix);
        }
        if (!result.tools.function_close.empty()) {
            tokens.push_back(result.tools.function_close);
        }
    }

    // Add argument markers for tagged formats
    if (result.tools.argument_format == ToolCallStructure::ARGS_TAGGED) {
        if (!result.tools.arg_prefix.empty()) {
            tokens.push_back(result.tools.arg_prefix);
        }
        if (!result.tools.arg_close.empty()) {
            tokens.push_back(result.tools.arg_close);
        }
    }

    result.preserved_tokens = tokens;
    LOG_DBG("Collected %zu preserved tokens\n", tokens.size());
}

// ============================================================================
// Internal: Differential Analysis for Tool Detection
// ============================================================================

static std::string find_string_difference(const std::string & base, const std::string & extended) {
    size_t common_prefix = 0;
    while (common_prefix < base.length() && common_prefix < extended.length() &&
           base[common_prefix] == extended[common_prefix]) {
        common_prefix++;
    }
    return extended.substr(common_prefix);
}

static std::string extract_json_field_name(const std::string & opener, const std::string & default_name,
                                           const std::vector<std::string> & candidates) {
    for (const auto & candidate : candidates) {
        std::string pattern = "\"" + candidate + "\"";
        if (opener.find(pattern) != std::string::npos) {
            LOG_DBG("Found JSON field name '%s' in opener\n", candidate.c_str());
            return candidate;
        }
    }
    return default_name;
}

static std::string find_closing_pattern(const std::string & diff, size_t func_pos) {
    std::vector<std::string> closers = { "</", "}", "]", ">", " " };

    std::string best_pattern;
    size_t best_pos = std::string::npos;

    for (const auto & pattern : closers) {
        size_t pos = diff.find(pattern, func_pos);
        if (pos != std::string::npos) {
            if (pos < best_pos) {
                if (pattern == "</") {
                    size_t end_pos = diff.find('>', pos);
                    if (end_pos != std::string::npos) {
                        best_pattern = diff.substr(pos, end_pos - pos + 1);
                        best_pos = pos;
                    }
                } else {
                    best_pattern = pattern;
                    best_pos = pos;
                }
            }
        }
    }
    return best_pattern;
}

static std::string find_tool_call_start(const std::string & diff) {
    std::vector<std::string> start_patterns = { "<", "[", "{", "call", "func", "tool", "TOOL" };
    for (const auto & pattern : start_patterns) {
        size_t pos = diff.find(pattern);
        if (pos < 5) {
            if (pattern == "<") {
                size_t end_pos = diff.find('>', pos);
                if (end_pos != std::string::npos) {
                    return diff.substr(pos, end_pos - pos + 1);
                }
            }
            if (pattern == "[" || pattern == "{") {
                size_t chunk_len = std::min(diff.length() - pos, (size_t)60);
                return diff.substr(pos, chunk_len);
            }

            size_t end_pos = diff.find_first_of(">]} \n", pos);
            if (end_pos != std::string::npos) {
                if (diff[end_pos] == '>' || diff[end_pos] == ']' || diff[end_pos] == '}') {
                    return diff.substr(pos, end_pos - pos + 1);
                }
                return diff.substr(pos, end_pos - pos);
            }
            return diff.substr(pos, pattern.length());
        }
    }
    return "";
}

static std::string find_tool_call_end(const std::string & diff, size_t func_pos) {
    char opener_char = 0;
    std::string start_tag_name;

    std::string openers = "[{<";
    size_t last_opener_pos = std::string::npos;
    for (char c : openers) {
        size_t p = diff.rfind(c, func_pos);
        if (p != std::string::npos) {
            if (last_opener_pos == std::string::npos || p > last_opener_pos) {
                last_opener_pos = p;
                opener_char = c;
            }
        }
    }

    size_t unclosed_bracket = diff.rfind('[', func_pos);
    if (unclosed_bracket != std::string::npos) {
        size_t closer = diff.find(']', unclosed_bracket);
        if (closer == std::string::npos || closer > func_pos) {
            opener_char = '[';
        }
    }

    if (opener_char == '<') {
        size_t tag_start = diff.find('<', last_opener_pos);
        if (tag_start != std::string::npos) {
            // Include '=' in search to handle <function=name> style tags
            // where the closing tag is </function>, not </function=name>
            size_t tag_end = diff.find_first_of(" >=\n", tag_start);
            if (tag_end != std::string::npos) {
                start_tag_name = diff.substr(tag_start + 1, tag_end - (tag_start + 1));
            }
        }
    }

    if (!start_tag_name.empty()) {
        std::string expected_closer = "</" + start_tag_name + ">";
        size_t pos = diff.find(expected_closer, func_pos);
        if (pos != std::string::npos) {
            if (opener_char == '[') {
                size_t bracket_pos = diff.rfind(']', pos);
                if (bracket_pos != std::string::npos && bracket_pos > func_pos) {
                    return diff.substr(bracket_pos, (pos + expected_closer.length()) - bracket_pos);
                }
            }
            return expected_closer;
        }
    }

    std::vector<std::string> end_patterns = { "</", "]", "}", ">", "\n", " " };
    std::string best_pattern;
    size_t best_pos = std::string::npos;

    auto is_structural = [](const std::string & s) {
        if (s.empty()) {
            return false;
        }
        return s[0] == ']' || s[0] == '}' || s[0] == '>' || (s.size() >= 2 && s.substr(0, 2) == "</");
    };

    for (const auto & pattern : end_patterns) {
        size_t pos = diff.find(pattern, func_pos);
        if (pos == std::string::npos) {
            continue;
        }

        bool current_is_struct = is_structural(pattern);
        bool best_is_struct = is_structural(best_pattern);

        bool better = false;
        if (best_pattern.empty()) {
            better = true;
        } else if (pos < best_pos) {
            better = !(best_is_struct && !current_is_struct) &&
                     !(opener_char == '[' && best_pattern[0] == ']' && pattern[0] == '}');
        } else {
            if (!best_is_struct && current_is_struct && pos < best_pos + 400) {
                better = true;
            } else if (best_is_struct && current_is_struct && opener_char == '[' && pattern[0] == ']' &&
                       best_pattern[0] == '}') {
                if (pos < best_pos + 100) {
                    better = true;
                }
            }
        }

        if (better) {
            best_pattern = pattern;
            best_pos = pos;

            if (current_is_struct && (pattern == "]" || pattern == "}")) {
                size_t tag_start = diff.find('<', best_pos + pattern.length());
                if (tag_start != std::string::npos && tag_start < best_pos + pattern.length() + 5) {
                    size_t tag_end = diff.find('>', tag_start);
                    if (tag_end != std::string::npos) {
                        best_pattern = diff.substr(best_pos, tag_end - best_pos + 1);
                    }
                }
            }
        }
    }

    return best_pattern;
}

static std::string infer_tool_call_opener(const std::string & diff1, const std::string & diff2,
                                          const std::string & diff3) {
    std::vector<std::string> differences = { diff1, diff2, diff3 };
    return find_common_prefix(differences);
}

static std::string infer_tool_call_closer(const std::string & diff1, const std::string & diff2,
                                          const std::string & diff3) {
    std::vector<std::string> differences = { diff1, diff2, diff3 };
    return find_common_suffix_generic(differences);
}

static InternalDiscoveredPattern extract_patterns_from_differences(const std::string & tool1_diff,
                                                                    const std::string & tool2_diff,
                                                                    const std::string & tool3_diff) {
    LOG_DBG("=== EXTRACTING PATTERNS FROM DIFFERENCES ===\n");

    InternalDiscoveredPattern patterns;

    size_t func1_pos = tool1_diff.find("test_function_name");
    size_t func2_pos = tool2_diff.find("test_function_name");

    if (func1_pos != std::string::npos && func2_pos != std::string::npos) {
        LOG_DBG("Found function names, extracting patterns...\n");

        patterns.tool_call_opener = tool1_diff.substr(0, func1_pos);

        patterns.tool_name_field = extract_json_field_name(patterns.tool_call_opener, "name",
                                                            { "tool_name", "name", "function_name", "function" });

        patterns.tool_args_field = extract_json_field_name(patterns.tool_call_opener + tool1_diff.substr(func1_pos),
                                                            "arguments",
                                                            { "parameters", "arguments", "args", "params", "input" });

        patterns.tool_id_field = extract_json_field_name(patterns.tool_call_opener, "",
                                                          { "tool_call_id", "tool_id", "id", "call_id" });

        // Extract parameter patterns from tool2_diff
        size_t param1_pos = tool2_diff.find("\"param1\"");
        bool param_has_quotes = (param1_pos != std::string::npos);
        size_t param2_pos = tool2_diff.find("\"param2\"");
        size_t value1_pos = tool2_diff.find("\"value1\"");

        if (param1_pos == std::string::npos) {
            param1_pos = tool2_diff.find("param1");
        }
        if (param_has_quotes && param1_pos != std::string::npos) {
            param1_pos++;
        }
        if (param2_pos == std::string::npos) {
            param2_pos = tool2_diff.find("param2");
        }
        if (param_has_quotes && param2_pos != std::string::npos) {
            param2_pos++;
        }
        if (value1_pos == std::string::npos) {
            value1_pos = tool2_diff.find("value1");
        }
        if (param_has_quotes && value1_pos != std::string::npos) {
            value1_pos++;
        }

        if (param1_pos != std::string::npos && value1_pos != std::string::npos) {
            size_t search_start = (param1_pos > 20) ? param1_pos - 20 : 0;
            std::string pre_param = tool2_diff.substr(search_start, param1_pos - search_start);

            size_t delim_pos = pre_param.find_last_of('\n');
            if (delim_pos == std::string::npos) {
                delim_pos = pre_param.find_last_of('>');
            }

            if (delim_pos != std::string::npos) {
                patterns.parameter_key_prefix = pre_param.substr(delim_pos + 1);
            } else {
                size_t start_marker = pre_param.find_last_of("<{[ \"");
                if (start_marker != std::string::npos) {
                    patterns.parameter_key_prefix = pre_param.substr(start_marker);
                } else {
                    patterns.parameter_key_prefix = pre_param;
                }
            }

            trim_whitespace(patterns.parameter_key_prefix);

            size_t key_end = param1_pos + std::string("param1").length();
            if (value1_pos > key_end) {
                patterns.parameter_key_suffix = tool2_diff.substr(key_end, value1_pos - key_end);
            }

            // Extract parameter closer (e.g., </parameter>)
            // Look for closing tag after value1
            size_t value1_end = value1_pos + std::string("value1").length();
            if (value1_end < tool2_diff.length()) {
                // Try to find XML-style closing tag like </parameter>
                size_t close_start = tool2_diff.find("</", value1_end);
                if (close_start != std::string::npos) {
                    size_t close_end = tool2_diff.find('>', close_start);
                    if (close_end != std::string::npos) {
                        patterns.parameter_closer = tool2_diff.substr(close_start, close_end - close_start + 1);
                    }
                }
            }
        }

        // Extract function opener/closer
        const std::string & func_context = tool1_diff;
        size_t open_pos = func_context.rfind('<', func1_pos);
        if (open_pos != std::string::npos && open_pos < func1_pos) {
            size_t close_pos = func_context.find('>', open_pos);
            if (close_pos != std::string::npos && close_pos < func1_pos) {
                patterns.function_opener = func_context.substr(open_pos, close_pos - open_pos + 1);
            } else {
                patterns.function_opener = func_context.substr(open_pos, func1_pos - open_pos);
            }
        } else {
            // Check for special prefix patterns like >>> (Functionary)
            size_t prefix_start = func_context.rfind(">>>", func1_pos);
            if (prefix_start != std::string::npos && prefix_start + 3 == func1_pos) {
                patterns.function_opener = ">>>";
            } else {
                for (int i = (int)func1_pos - 1; i >= 0; i--) {
                    if (func_context[i] == '{' || func_context[i] == '[' || func_context[i] == '(' ||
                        func_context[i] == '<') {
                        patterns.function_opener = func_context.substr(i, func1_pos - i);
                        break;
                    }
                }
            }
        }

        // Function name suffix
        size_t func_name_end = func1_pos + std::string("test_function_name").length();
        if (func_name_end < func_context.length()) {
            char next_char = func_context[func_name_end];
            if (next_char == '>' || next_char == ']' || next_char == '}') {
                patterns.function_name_suffix = std::string(1, next_char);
            } else if (next_char == '"') {
                if (func_name_end + 1 < func_context.length() && func_context[func_name_end + 1] == '>') {
                    patterns.function_name_suffix = "\">";
                } else {
                    patterns.function_name_suffix = "\"";
                }
            }
        }

        // Function closer
        size_t search_start = func_name_end;
        if (!patterns.function_name_suffix.empty()) {
            search_start += patterns.function_name_suffix.length();
        }
        patterns.function_closer = find_closing_pattern(func_context, search_start);

        // Tool call start marker
        if (patterns.function_opener.length() > 0 &&
            patterns.tool_call_opener.length() > patterns.function_opener.length()) {
            size_t opener_start = patterns.tool_call_opener.length() - patterns.function_opener.length();
            if (opener_start > 0) {
                std::string before_func = patterns.tool_call_opener.substr(0, opener_start);
                size_t last_bracket = before_func.find_last_of('[');
                size_t tool_obj_brace = std::string::npos;
                if (last_bracket != std::string::npos && last_bracket + 1 < before_func.length()) {
                    tool_obj_brace = before_func.find('{', last_bracket + 1);
                }

                if (tool_obj_brace != std::string::npos) {
                    patterns.tool_call_start_marker = before_func.substr(0, tool_obj_brace);
                } else if (last_bracket != std::string::npos) {
                    patterns.tool_call_start_marker = before_func.substr(0, last_bracket + 1);
                } else {
                    patterns.tool_call_start_marker = before_func;
                }
            }
        } else {
            patterns.tool_call_start_marker = find_tool_call_start(tool1_diff);
        }

        if (patterns.tool_call_opener.empty()) {
            patterns.tool_call_opener = infer_tool_call_opener(tool1_diff, tool2_diff, tool3_diff);
            if (func1_pos != std::string::npos && patterns.tool_call_opener.length() > func1_pos) {
                patterns.tool_call_opener = patterns.tool_call_opener.substr(0, func1_pos);
            }
        }
        if (patterns.tool_call_closer.empty()) {
            patterns.tool_call_closer = infer_tool_call_closer(tool1_diff, tool2_diff, tool3_diff);
        }

        patterns.tool_call_end_marker = find_tool_call_end(func_context, func1_pos);

        // Trim whitespace
        if (!patterns.tool_call_end_marker.empty()) {
            size_t first = patterns.tool_call_end_marker.find_first_not_of(" \n\t");
            size_t last = patterns.tool_call_end_marker.find_last_not_of(" \n\t");
            if (first != std::string::npos && last != std::string::npos) {
                patterns.tool_call_end_marker = patterns.tool_call_end_marker.substr(first, (last - first + 1));
            }
        }

        // If tool_call_end_marker matches function_closer, it found the wrong tag.
        // Use tool_call_closer instead which is derived from common suffix of diffs.
        if (!patterns.function_closer.empty() && patterns.tool_call_end_marker == patterns.function_closer) {
            if (!patterns.tool_call_closer.empty()) {
                // Try to extract a proper closing tag from tool_call_closer
                // Use rfind to get the LAST closing tag (e.g., </tool_call> not </function>)
                size_t close_start = patterns.tool_call_closer.rfind("</");
                if (close_start != std::string::npos) {
                    size_t close_end = patterns.tool_call_closer.find('>', close_start);
                    if (close_end != std::string::npos) {
                        patterns.tool_call_end_marker = patterns.tool_call_closer.substr(close_start, close_end - close_start + 1);
                    }
                }
            }
        }

        if (patterns.tool_call_start_marker.empty()) {
            std::vector<std::string> diffs = { tool1_diff, tool2_diff, tool3_diff };
            patterns.tool_call_start_marker = find_common_substring_limited(diffs, 20, " \n\t<[{");
        }

        // Truncate if needed
        if (func1_pos != std::string::npos && patterns.tool_call_start_marker.length() > func1_pos) {
            std::string candidate = patterns.tool_call_start_marker.substr(0, func1_pos);
            size_t last_opener = candidate.find_last_of("{[");
            if (last_opener != std::string::npos) {
                patterns.tool_call_start_marker = candidate.substr(0, last_opener);
            } else {
                patterns.tool_call_start_marker = candidate;
            }
        }

        // Final trim
        if (!patterns.tool_call_start_marker.empty()) {
            size_t first = patterns.tool_call_start_marker.find_first_not_of(" \n\t\r");
            size_t last = patterns.tool_call_start_marker.find_last_not_of(" \n\t\r");
            if (first != std::string::npos && last != std::string::npos) {
                patterns.tool_call_start_marker = patterns.tool_call_start_marker.substr(first, (last - first + 1));
            }
        }
    }

    return patterns;
}

static InternalToolFormat determine_format_from_patterns(const InternalDiscoveredPattern & patterns) {
    LOG_DBG("=== DETERMINING FORMAT FROM PATTERNS ===\n");

    if (patterns.tool_call_opener.empty() && patterns.tool_call_closer.empty() && patterns.function_opener.empty() &&
        patterns.function_closer.empty() && patterns.parameter_opener.empty() && patterns.parameter_closer.empty() &&
        patterns.argument_separator.empty() && patterns.tool_call_start_marker.empty() &&
        patterns.tool_call_end_marker.empty()) {
        LOG_DBG("All patterns are empty - template doesn't support tool calls\n");
        return FORMAT_UNKNOWN;
    }

    if (!patterns.tool_call_opener.empty()) {
        if (patterns.tool_call_opener.find("{\"name\":") != std::string::npos ||
            patterns.tool_call_opener.find("{&quot;name&quot;:") != std::string::npos) {
            LOG_DBG("Detected JSON_NATIVE format from tool_call_opener JSON structure\n");
            return FORMAT_JSON_NATIVE;
        }
    }

    if (!patterns.function_opener.empty() && patterns.function_opener.find('<') == 0) {
        bool has_substantial_param_markers = false;
        if (!patterns.parameter_opener.empty()) {
            has_substantial_param_markers = (count_non_whitespace(patterns.parameter_opener) > 1);
        }
        if (!has_substantial_param_markers && !patterns.parameter_closer.empty()) {
            has_substantial_param_markers = (count_non_whitespace(patterns.parameter_closer) > 1);
        }

        if (!has_substantial_param_markers) {
            if ((!patterns.tool_call_opener.empty() && (patterns.tool_call_opener.find('[') != std::string::npos ||
                                                        patterns.tool_call_opener.find('{') != std::string::npos)) ||
                (!patterns.tool_call_start_marker.empty() &&
                 (patterns.tool_call_start_marker.find('[') != std::string::npos ||
                  patterns.tool_call_start_marker.find('{') != std::string::npos))) {
                LOG_DBG("Detected JSON_NATIVE format (XML markers but JSON structure)\n");
                return FORMAT_JSON_NATIVE;
            }
        }

        LOG_DBG("Detected XML_CONSTRUCTED format from function_opener\n");
        return FORMAT_XML_CONSTRUCTED;
    }

    if (!patterns.function_opener.empty() && patterns.function_opener.find('{') == 0) {
        LOG_DBG("Detected JSON_NATIVE format from function_opener\n");
        return FORMAT_JSON_NATIVE;
    }

    if (!patterns.tool_call_start_marker.empty() &&
        (patterns.tool_call_start_marker.find('<') == 0 || patterns.tool_call_start_marker.find('[') == 0)) {
        bool is_prefix_marker = patterns.tool_call_start_marker.find("<|") == 0 ||
                                patterns.tool_call_start_marker.find("<tool_call>") == 0 ||
                                patterns.tool_call_start_marker.find("[TOOL_CALL]") == 0;
        if (is_prefix_marker) {
            LOG_DBG("Detected JSON_NATIVE format from prefix marker (instruction-based)\n");
            return FORMAT_JSON_NATIVE;
        }
        LOG_DBG("Detected XML_CONSTRUCTED format from tool_call_start_marker\n");
        return FORMAT_XML_CONSTRUCTED;
    }

    if (!patterns.tool_call_opener.empty() && (patterns.tool_call_opener.find("<function=") != std::string::npos ||
                                               patterns.tool_call_opener.find('<') != std::string::npos)) {
        LOG_DBG("Detected XML_CONSTRUCTED format from tool_call_opener\n");
        return FORMAT_XML_CONSTRUCTED;
    }

    if (!patterns.tool_call_opener.empty() &&
        (patterns.tool_call_opener.find('{') != std::string::npos ||
         patterns.tool_call_opener.find('[') != std::string::npos ||
         patterns.tool_call_opener.find("TOOLCALL") != std::string::npos)) {
        LOG_DBG("Detected JSON_NATIVE format from tool_call_opener\n");
        return FORMAT_JSON_NATIVE;
    }

    if (!patterns.tool_call_start_marker.empty() &&
        (patterns.tool_call_start_marker.find('{') != std::string::npos ||
         patterns.tool_call_start_marker.find('[') != std::string::npos ||
         patterns.tool_call_start_marker.find("TOOLCALL") != std::string::npos)) {
        LOG_DBG("Detected JSON_NATIVE format from tool_call_start_marker structure\n");
        return FORMAT_JSON_NATIVE;
    }

    if (patterns.tool_call_opener.find("\"name\"") != std::string::npos &&
        patterns.tool_call_opener.find("\"arguments\"") != std::string::npos) {
        LOG_DBG("Detected JSON_NATIVE format from JSON structure in arguments\n");
        return FORMAT_JSON_NATIVE;
    }

    // Check for >>> prefix format (Functionary v3.2)
    if (!patterns.function_opener.empty() && patterns.function_opener == ">>>") {
        LOG_DBG("Detected XML_CONSTRUCTED format from >>> prefix (Functionary style)\n");
        return FORMAT_XML_CONSTRUCTED;
    }

    LOG_DBG("Could not determine format from patterns - returning UNKNOWN\n");
    return FORMAT_UNKNOWN;
}

static InternalDiscoveredPattern analyze_by_differential(const minja::chat_template & tmpl) {
    InternalDiscoveredPattern patterns;

    try {
        LOG_DBG("=== STARTING TEMPLATE DIFFERENTIAL ANALYSIS ===\n");

        auto caps = tmpl.original_caps();
        bool minja_supports_tool_calls = caps.supports_tool_calls;
        if (!minja_supports_tool_calls) {
            LOG_DBG("Template doesn't support standard tool calls (per minja caps detection)\n");
        }

        json base_msg = {
            { "role",    "assistant" },
            { "content", "MARKER"    }
        };

        // Use nullptr for content to trigger tool_calls branch in templates that check "content is none"
        json tool_msg1 = {
            { "role",       "assistant" },
            { "content",    nullptr     },
            { "tool_calls",
             json::array({ { { "type", "function" },
                            { "function", { { "name", "test_function_name" }, { "arguments", json::object() } } } } }) }
        };

        json tool_msg2 = {
            { "role",       "assistant" },
            { "content",    nullptr     },
            { "tool_calls",
             json::array({ { { "type", "function" },
                            { "function",
                              { { "name", "test_function_name" },
                                { "arguments", json::object({ { "param1", "value1" }, { "param2", "value2" } }) } } } } }) }
        };

        json tool_msg3 = {
            { "role",       "assistant" },
            { "content",    nullptr     },
            { "tool_calls",
             json::array({ { { "type", "function" },
                            { "function", { { "name", "test_function_name" }, { "arguments", json::object() } } } },
                          { { "type", "function" },
                            { "function", { { "name", "another_test_function" }, { "arguments", json::object() } } } } }) }
        };

        json tools = {
            { { "type", "function" },
             { "function",
                { { "name", "test_function_name" },
                  { "description", "A test function" },
                  { "parameters",
                    { { "type", "object" },
                      { "properties",
                        { { "param1", { { "type", "string" }, { "description", "First parameter" } } },
                          { "param2", { { "type", "string" }, { "description", "Second parameter" } } } } },
                      { "required", json::array({ "param1", "param2" }) } } } } } },
            { { "type", "function" },
             { "function",
                { { "name", "another_test_function" },
                  { "description", "Another test function" },
                  { "parameters",
                    { { "type", "object" },
                      { "properties",
                        { { "param1", { { "type", "string" }, { "description", "First parameter" } } } } },
                      { "required", json::array({ "param1" }) } } } } } }
        };

        minja::chat_template_inputs inputs;
        inputs.tools = tools;

        inputs.messages = { base_msg };
        auto base_output = tmpl.apply(inputs);

        inputs.messages = { tool_msg1 };
        auto tool1_output = tmpl.apply(inputs);

        inputs.messages = { tool_msg2 };
        auto tool2_output = tmpl.apply(inputs);

        inputs.messages = { tool_msg3 };
        auto tool3_output = tmpl.apply(inputs);

        std::string tool1_diff = find_string_difference(base_output, tool1_output);
        std::string tool2_diff = find_string_difference(base_output, tool2_output);
        std::string tool3_diff = find_string_difference(base_output, tool3_output);

        LOG_DBG("Tool1 diff length: %zu\n", tool1_diff.length());
        LOG_DBG("Tool2 diff length: %zu\n", tool2_diff.length());
        LOG_DBG("Tool3 diff length: %zu\n", tool3_diff.length());

        if (tool1_diff.empty() && tool2_diff.empty() && tool3_diff.empty()) {
            LOG_DBG("All diffs are empty - template may not produce tool calls in output\n");
            // Try with add_generation_prompt variations
            json alternative_base_msg = {
                { "role", "assistant" },
                { "content", "MARKER" }
            };

            minja::chat_template_inputs alt_inputs;
            alt_inputs.tools = tools;
            alt_inputs.messages = { alternative_base_msg };
            alt_inputs.add_generation_prompt = false;
            auto alt_base = tmpl.apply(alt_inputs);

            alt_inputs.messages = { tool_msg1 };
            auto alt_tool1 = tmpl.apply(alt_inputs);

            tool1_diff = find_string_difference(alt_base, alt_tool1);
            if (!tool1_diff.empty()) {
                alt_inputs.messages = { tool_msg2 };
                tool2_diff = find_string_difference(alt_base, tmpl.apply(alt_inputs));
                alt_inputs.messages = { tool_msg3 };
                tool3_diff = find_string_difference(alt_base, tmpl.apply(alt_inputs));
            }
        }

        patterns = extract_patterns_from_differences(tool1_diff, tool2_diff, tool3_diff);

        LOG_DBG("=== ENDING TEMPLATE DIFFERENTIAL ANALYSIS ===\n");

    } catch (const std::exception & e) {
        LOG_DBG("Template differential analysis failed: %s\n", e.what());
    }

    return patterns;
}
