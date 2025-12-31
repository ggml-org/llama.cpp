#include "chat-auto-parser.h"
#include "chat-auto-parser-helpers.h"

#include "log.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

using json = nlohmann::ordered_json;

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
                    // Check if the detected tag is a close tag (starts with </)
                    // This handles templates like DeepSeek-V3.1 that end with </think> when thinking is disabled
                    bool is_close_tag = (tag.size() > 2 && tag[0] == '<' && tag[1] == '/');

                    if (is_close_tag) {
                        // The tag is a close tag (e.g., </think>)
                        // Derive the open tag by removing the '/'
                        std::string tag_name = extract_tag_name(tag);  // Returns "/think" for </think>
                        if (!tag_name.empty() && tag_name[0] == '/') {
                            tag_name = tag_name.substr(1);  // Remove leading '/'
                        }
                        cs.reasoning_start = "<" + tag_name + ">";
                        cs.reasoning_end = tag;
                        trim_whitespace(cs.reasoning_start);
                        trim_whitespace(cs.reasoning_end);

                        LOG_DBG("Method 3: Found reasoning markers via prompt ending with CLOSE tag\n");
                        LOG_DBG("  start: '%s', end: '%s'\n", cs.reasoning_start.c_str(), cs.reasoning_end.c_str());

                        // Note: The prompt ends with the close tag, meaning thinking is disabled.
                        // The reasoning_mode will be set in detect_reasoning_mode() which will
                        // correctly identify this as NOT forced open since the prompt ends with
                        // the end marker, not the start marker.
                    } else {
                        // Standard case: open tag at the end (e.g., <think>)
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

    // Use differential analysis to detect tool patterns
    // This now includes a robust test that renders two payloads:
    // 1. Tool definitions + content only
    // 2. Tool definitions + content + tool calls
    // If outputs are identical, the template doesn't support tool calls
    auto discovered = analyze_by_differential(tmpl);
    auto format = determine_format_from_patterns(discovered);

    if (format == FORMAT_UNKNOWN) {
        LOG_DBG("Template does not support tool calls (differential analysis returned no patterns)\n");
        ts.supports_tools = false;
        return ts;
    }

    // Check if minja reports tool call support (for informational purposes)
    auto caps = tmpl.original_caps();
    if (!caps.supports_tool_calls) {
        LOG_DBG("Note: minja caps indicate no tool support, but differential analysis found patterns\n");
    }

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
            char first = discovered.function_opener[0];
            if (first != '<' && first != '{' && first != '[') {
                // Non-XML/JSON prefix format (e.g., ">>>", "##", etc.)
                // Function name follows prefix directly, ends with newline
                ts.function_prefix = discovered.function_opener;
                ts.function_suffix = "\n";  // Function name typically ends with newline
                ts.function_close = "";      // No closing tag for prefix formats
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

    // Add markers for prefixed-indexed formats (e.g., Kimi-K2)
    if (result.tools.function_format == ToolCallStructure::FUNC_PREFIXED_INDEXED) {
        if (!result.tools.per_call_start.empty()) {
            tokens.push_back(result.tools.per_call_start);
        }
        if (!result.tools.args_marker.empty()) {
            tokens.push_back(result.tools.args_marker);
        }
        if (!result.tools.per_call_end.empty()) {
            tokens.push_back(result.tools.per_call_end);
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
