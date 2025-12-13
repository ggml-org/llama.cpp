#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat.h"
#include "log.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

template_analysis_result template_analyzer::analyze_template(const common_chat_template & tmpl) {
    LOG_DBG("=== STARTING UNIFIED TEMPLATE ANALYSIS ===\n");

    template_analysis_result result;

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
            { "<|START_THOUGHT|>",  "<|END_THOUGHT|>"  },
            { "<|START_REASON|>",   "<|END_REASON|>"   },
            { "<think>",            "</think>"         },
            { "<thinking>",         "</thinking>"      },
        };

        for (const auto & [start_marker, end_marker] : reasoning_patterns) {
            size_t end_pos = result.tools.tool_section_start.find(end_marker);
            if (end_pos != std::string::npos) {
                // Found reasoning end marker in tool_section_start
                // Extract it and clean up tool_section_start
                result.content.reasoning_start = start_marker;
                result.content.reasoning_end   = end_marker;
                result.content.reasoning_mode  = content_structure::REASONING_OPTIONAL;

                // Clean up tool_section_start: remove everything before and including the end marker
                size_t after_end = end_pos + end_marker.length();
                if (after_end < result.tools.tool_section_start.length()) {
                    result.tools.tool_section_start = result.tools.tool_section_start.substr(after_end);
                    // Trim leading whitespace
                    size_t first_non_ws             = result.tools.tool_section_start.find_first_not_of(" \t\n\r");
                    if (first_non_ws != std::string::npos && first_non_ws > 0) {
                        result.tools.tool_section_start = result.tools.tool_section_start.substr(first_non_ws);
                    }
                }

                LOG_DBG("Post-processing: Extracted reasoning markers from tool_section_start\n");
                LOG_DBG("  reasoning_start: '%s', reasoning_end: '%s'\n", result.content.reasoning_start.c_str(),
                        result.content.reasoning_end.c_str());
                LOG_DBG("  cleaned tool_section_start: '%s'\n", result.tools.tool_section_start.c_str());
                break;
            }
        }
    }

    // Post-processing: Detect content markers for recipient-based format
    // For recipient-based format, content is prefixed with tool_call_start_marker + recipient_name + \n
    // (e.g., ">>>all\n"). We need to detect and extract this as the content_start marker.
    if (result.tools.function_format == tool_call_structure::FUNC_RECIPIENT_BASED &&
        result.content.content_start.empty() && !result.tools.tool_section_start.empty()) {
        // Render template with content only (no tools) to detect the content marker
        templates_params inputs;
        inputs.messages = {
            { { "role", "user" },      { "content", "Hello" }               },
            { { "role", "assistant" }, { "content", "ACTUAL_CONTENT_HERE" } }
        };
        inputs.add_generation_prompt = true;

        std::string output;
        try {
            output = common_chat_template_direct_apply(tmpl, inputs);
        } catch (...) {
            output = "";
        }

        if (!output.empty()) {
            // Find where the actual content starts
            size_t content_pos = output.find("ACTUAL_CONTENT_HERE");

            if (content_pos != std::string::npos) {
                // For recipient-based format, find the last occurrence of tool_call_start_marker
                // before the content. The marker is from that position to the content (including the newline).
                size_t marker_pos = output.rfind(result.tools.tool_section_start, content_pos);

                if (marker_pos != std::string::npos && marker_pos < content_pos) {
                    // Find the newline after the marker
                    size_t newline_pos = output.find('\n', marker_pos);

                    if (newline_pos != std::string::npos && newline_pos < content_pos) {
                        // Extract everything up to and including the newline after the marker
                        std::string detected_marker = output.substr(marker_pos, newline_pos - marker_pos + 1);

                        // Verify the marker starts with tool_call_start_marker
                        if (detected_marker.find(result.tools.tool_section_start) == 0) {
                            result.content.content_start = detected_marker;
                            result.content.content_mode  = content_structure::CONTENT_ALWAYS_WRAPPED;
                            LOG_DBG("Post-processing: Detected recipient-based content marker: '%s'\n",
                                    result.content.content_start.c_str());
                        }
                    }
                }
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

content_structure template_analyzer::analyze_content_structure(const common_chat_template & tmpl) {
    LOG_DBG("=== PHASE 1: ANALYZING CONTENT STRUCTURE ===\n");

    content_structure cs;

    // Step 1: Detect reasoning markers by toggling enable_thinking
    detect_reasoning_markers(tmpl, cs);

    // Step 2: Detect content wrapping markers
    detect_content_markers(tmpl, cs);

    // Step 3: Determine reasoning mode (NONE, OPTIONAL, FORCED_OPEN)
    templates_params inputs;
    inputs.messages = {
        { { "role", "user" }, { "content", "Hello" } }
    };
    inputs.add_generation_prompt            = true;
    inputs.enable_thinking                  = true;

    std::string prompt;
    try {
        prompt = common_chat_template_direct_apply(tmpl, inputs);
    } catch (...) {
        LOG_DBG("Failed to render template for reasoning mode detection\n");
        return cs;
    }

    cs.reasoning_mode = detect_reasoning_mode(cs, prompt);

    LOG_DBG("Phase 1 complete: reasoning_mode=%d, content_mode=%d\n", static_cast<int>(cs.reasoning_mode),
            static_cast<int>(cs.content_mode));

    return cs;
}

void template_analyzer::detect_reasoning_markers(const common_chat_template & tmpl, content_structure & cs) {
    LOG_DBG("=== DETECTING REASONING MARKERS ===\n");

    // Method 1: Compare outputs with reasoning_content field present vs absent
    json reasoning_msg = {
        { "role",              "assistant"      },
        { "content",           "CONTENT_MARKER" },
        { "reasoning_content", "THOUGHT_MARKER" }
    };

    json base_msg = {
        { "role",    "assistant"      },
        { "content", "CONTENT_MARKER" }
    };

    templates_params inputs;

    inputs.messages = { reasoning_msg };
    std::string reasoning_output;
    try {
        reasoning_output = common_chat_template_direct_apply(tmpl, inputs);
    } catch (...) {
        LOG_DBG("Failed to render template with reasoning_content\n");
        reasoning_output = "";
    }

    inputs.messages = { base_msg };
    std::string base_output;
    try {
        base_output = common_chat_template_direct_apply(tmpl, inputs);
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
            cs.reasoning_end   = reasoning_output.substr(thought_end, content_pos - thought_end);

            // Find what's before THOUGHT_MARKER by comparing with base_output
            size_t diff_start = 0;
            while (diff_start < base_output.length() && diff_start < reasoning_output.length() &&
                   base_output[diff_start] == reasoning_output[diff_start]) {
                diff_start++;
            }

            // If diff_start is in the middle of a tag (previous char is '<'), back up to include it
            // This handles cases like base="</think>" vs reasoning="<think>" where both share '<'
            if (diff_start > 0 && diff_start < reasoning_output.length() &&
                reasoning_output[diff_start - 1] == '<') {
                diff_start--;
            }

            if (diff_start < thought_pos) {
                cs.reasoning_start = reasoning_output.substr(diff_start, thought_pos - diff_start);
            }

            trim_whitespace(cs.reasoning_start);
            trim_whitespace(cs.reasoning_end);

            // If we found reasoning_end but not reasoning_start, try to derive it from reasoning_end
            // For example: </think> -> <think>, </|END_THINKING|> -> <|START_THINKING|>
            if (cs.reasoning_start.empty() && !cs.reasoning_end.empty()) {
                // First, try to derive directly from the closing tag format
                if (cs.reasoning_end.length() > 3 && cs.reasoning_end[0] == '<' && cs.reasoning_end[1] == '/') {
                    // Standard XML closing tag like </think> -> <think>
                    size_t tag_end_pos = cs.reasoning_end.find('>');
                    if (tag_end_pos != std::string::npos) {
                        std::string tag_name = cs.reasoning_end.substr(2, tag_end_pos - 2);
                        cs.reasoning_start   = "<" + tag_name + ">";
                        LOG_DBG("Method 1: Derived reasoning_start from closing tag format\n");
                        LOG_DBG("  start: '%s', end: '%s'\n", cs.reasoning_start.c_str(), cs.reasoning_end.c_str());
                    }
                } else if (cs.reasoning_end.find("<|END_") == 0 || cs.reasoning_end.find("<|/") == 0) {
                    // Special token format like <|END_THINKING|> -> <|START_THINKING|>
                    // or <|/think|> -> <|think|>
                    if (cs.reasoning_end.find("<|END_") == 0) {
                        std::string core   = cs.reasoning_end.substr(6);  // Remove "<|END_"
                        cs.reasoning_start = "<|START_" + core;
                    } else {
                        std::string core   = cs.reasoning_end.substr(3);  // Remove "<|/"
                        cs.reasoning_start = "<|" + core;
                    }
                    LOG_DBG("Method 1: Derived reasoning_start from special token format\n");
                    LOG_DBG("  start: '%s', end: '%s'\n", cs.reasoning_start.c_str(), cs.reasoning_end.c_str());
                }
            }

            if (!cs.reasoning_start.empty()) {
                LOG_DBG("Method 1: Found reasoning markers via reasoning_content field\n");
                LOG_DBG("  start: '%s', end: '%s'\n", cs.reasoning_start.c_str(), cs.reasoning_end.c_str());
            }
        }
    }

    // Method 2: Compare prompts with enable_thinking true vs false
    if (cs.reasoning_start.empty()) {
        LOG_DBG("Method 1 failed, trying Method 2 (enable_thinking toggle)\n");

        json user_msg = {
            { "role",    "user"  },
            { "content", "Hello" }
        };

        templates_params inputs_prompt;
        inputs_prompt.messages              = { user_msg };
        inputs_prompt.add_generation_prompt = true;
        inputs_prompt.enable_thinking = false;
        std::string prompt_no_think;
        try {
            prompt_no_think = common_chat_template_direct_apply(tmpl, inputs_prompt);
        } catch (...) {
            prompt_no_think = "";
        }

        inputs_prompt.enable_thinking = true;
        std::string prompt_think;
        try {
            prompt_think = common_chat_template_direct_apply(tmpl, inputs_prompt);
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

            // Check which direction has extra content
            if (prompt_think.length() > prompt_no_think.length()) {
                // Normal case: enable_thinking=true adds content (e.g., <think> at the end)
                std::string diff = prompt_think.substr(diff_pos);

                // Only use if it looks like a tag
                if (diff.find('<') != std::string::npos || diff.find('[') != std::string::npos) {
                    cs.reasoning_start = diff;
                    cs.reasoning_end   = create_closing_tag(diff);
                    trim_whitespace(cs.reasoning_start);
                    trim_whitespace(cs.reasoning_end);

                    LOG_DBG("Method 2: Found reasoning markers via enable_thinking toggle\n");
                    LOG_DBG("  start: '%s', end: '%s'\n", cs.reasoning_start.c_str(), cs.reasoning_end.c_str());
                }
            } else {
                // Reverse case: enable_thinking=false adds content (e.g., GLM-4.6 adds <think></think>)
                // This means the template adds an empty thinking block when thinking is disabled
                std::string diff = prompt_no_think.substr(diff_pos);

                // Look for adjacent opening and closing tags like <think></think>
                size_t open_start = diff.find('<');
                if (open_start != std::string::npos) {
                    size_t open_end = diff.find('>', open_start);
                    if (open_end != std::string::npos) {
                        std::string opening_tag = diff.substr(open_start, open_end - open_start + 1);
                        // Skip if it looks like a role marker
                        if (opening_tag.find("assistant") == std::string::npos &&
                            opening_tag.find("user") == std::string::npos &&
                            opening_tag.find("system") == std::string::npos) {
                            std::string expected_close = create_closing_tag(opening_tag);
                            // Check if the closing tag follows immediately (empty thinking block)
                            size_t      close_pos      = diff.find(expected_close, open_end + 1);
                            if (close_pos != std::string::npos) {
                                // Verify only whitespace between tags
                                std::string between = diff.substr(open_end + 1, close_pos - open_end - 1);
                                bool        only_ws = true;
                                for (char c : between) {
                                    if (!std::isspace(static_cast<unsigned char>(c))) {
                                        only_ws = false;
                                        break;
                                    }
                                }
                                if (only_ws) {
                                    cs.reasoning_start = opening_tag;
                                    cs.reasoning_end   = expected_close;
                                    trim_whitespace(cs.reasoning_start);
                                    trim_whitespace(cs.reasoning_end);

                                    LOG_DBG("Method 2: Found reasoning markers via enable_thinking toggle (reverse)\n");
                                    LOG_DBG("  start: '%s', end: '%s'\n", cs.reasoning_start.c_str(),
                                            cs.reasoning_end.c_str());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Method 3: Check if the prompt ends with an unclosed reasoning tag
    if (cs.reasoning_start.empty()) {
        LOG_DBG("Method 2 failed, trying Method 3 (prompt ending with open tag)\n");

        json user_msg = {
            { "role",    "user"  },
            { "content", "Hello" }
        };

        templates_params inputs_prompt;
        inputs_prompt.messages                         = { user_msg };
        inputs_prompt.add_generation_prompt            = true;
        inputs_prompt.enable_thinking                  = true;

        std::string prompt;
        try {
            prompt = common_chat_template_direct_apply(tmpl, inputs_prompt);
        } catch (...) {
            prompt = "";
        }

        if (!prompt.empty()) {
            // Save trailing whitespace before trimming
            std::string trailing_ws;
            size_t      end_pos = prompt.length();
            while (end_pos > 0 && (prompt[end_pos - 1] == '\n' || prompt[end_pos - 1] == '\r')) {
                trailing_ws = prompt[end_pos - 1] + trailing_ws;
                end_pos--;
            }

            trim_trailing_newlines(prompt);

            // Find the last tag in the prompt
            size_t last_open_angle  = prompt.rfind('<');
            size_t last_close_angle = prompt.rfind('>');

            // Check for closed tags at the end
            if (last_open_angle != std::string::npos && last_close_angle != std::string::npos &&
                last_close_angle == prompt.length() - 1 && last_close_angle > last_open_angle) {
                std::string tag = prompt.substr(last_open_angle);

                // Check if this looks like a reasoning tag (not a role marker)
                std::vector<std::string> blacklisted_tags = {
                    "<|CHATBOT_TOKEN|>", "<|SYSTEM_TOKEN|>",  "<|USER_TOKEN|>",  "<|ASSISTANT_TOKEN|>", "<|im_start|>",
                    "<|im_end|>",        "<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>",     "<|end|>",
                    "<|assistant|>",     "<|user|>",          "<|system|>",      "<assistant>",         "<user>",
                    "<system>"
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
                            tag_name = tag_name.substr(1);             // Remove leading '/'
                        }
                        cs.reasoning_start = "<" + tag_name + ">";
                        cs.reasoning_end   = tag;
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
                        cs.reasoning_end   = create_closing_tag(tag) + trailing_ws;
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
            { "role",    "user"  },
            { "content", "Hello" }
        };

        templates_params inputs_prompt;
        inputs_prompt.messages                         = { user_msg };
        inputs_prompt.add_generation_prompt            = true;
        // Try with thinking disabled - templates may output empty thinking blocks
        inputs_prompt.enable_thinking                  = false;

        std::string prompt;
        try {
            prompt = common_chat_template_direct_apply(tmpl, inputs_prompt);
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
                { "<|START_",   "<|END_",      "THINKING" },
                { "<|START_",   "<|END_",      "THOUGHT"  },
                { "<|START_",   "<|END_",      "REASON"   },
                { "<think>",    "</think>",    ""         },
                { "<Think>",    "</Think>",    ""         },
                { "<THINK>",    "</THINK>",    ""         },
                { "[think]",    "[/think]",    ""         },
                { "[THINK]",    "[/THINK]",    ""         },
                { "<thinking>", "</thinking>", ""         },
                { "<THINKING>", "</THINKING>", ""         },
                { "<|think|>",  "<|/think|>",  ""         },
            };

            for (const auto & [open_prefix, close_prefix, keyword] : tag_patterns) {
                size_t open_pos = prompt.find(open_prefix);
                if (open_pos == std::string::npos) {
                    continue;
                }

                std::string start_tag;
                std::string end_tag;

                if (!keyword.empty()) {
                    // Pattern like <|START_THINKING|><|END_THINKING|>
                    std::string full_open     = open_prefix + keyword;
                    size_t      full_open_pos = prompt.find(full_open);
                    if (full_open_pos == std::string::npos) {
                        continue;
                    }

                    // Find the end of this tag (look for |> or >)
                    size_t tag_end = prompt.find("|>", full_open_pos + full_open.length());
                    if (tag_end == std::string::npos) {
                        tag_end = prompt.find('>', full_open_pos + full_open.length());
                    }
                    if (tag_end == std::string::npos) {
                        continue;
                    }

                    start_tag =
                        prompt.substr(full_open_pos, tag_end - full_open_pos + (prompt[tag_end] == '|' ? 2 : 1));

                    // Look for the corresponding end tag
                    std::string expected_close = close_prefix + keyword;
                    size_t      close_pos      = prompt.find(expected_close, tag_end);
                    if (close_pos == std::string::npos) {
                        continue;
                    }

                    // Find end of close tag
                    size_t close_end = prompt.find("|>", close_pos + expected_close.length());
                    if (close_end == std::string::npos) {
                        close_end = prompt.find('>', close_pos + expected_close.length());
                    }
                    if (close_end == std::string::npos) {
                        continue;
                    }

                    end_tag = prompt.substr(close_pos, close_end - close_pos + (prompt[close_end] == '|' ? 2 : 1));
                } else {
                    // Simple pattern like <think></think>
                    start_tag        = open_prefix;
                    size_t close_pos = prompt.find(close_prefix, open_pos + start_tag.length());
                    if (close_pos == std::string::npos) {
                        continue;
                    }
                    end_tag = close_prefix;
                }

                // Verify the tags are adjacent or nearly adjacent (only whitespace between)
                size_t start_end_pos = prompt.find(start_tag) + start_tag.length();
                size_t end_start_pos = prompt.find(end_tag, start_end_pos);
                if (end_start_pos != std::string::npos) {
                    std::string between         = prompt.substr(start_end_pos, end_start_pos - start_end_pos);
                    // Allow only whitespace between the tags (empty thinking block)
                    bool        only_whitespace = true;
                    for (char c : between) {
                        if (!std::isspace(static_cast<unsigned char>(c))) {
                            only_whitespace = false;
                            break;
                        }
                    }

                    if (only_whitespace) {
                        cs.reasoning_start = start_tag;
                        cs.reasoning_end   = end_tag;
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

void template_analyzer::detect_content_markers(const common_chat_template & tmpl, content_structure & cs) {
    LOG_DBG("=== DETECTING CONTENT MARKERS ===\n");

    // Render template with a unique content marker
    json user_msg = {
        { "role",    "user"  },
        { "content", "Hello" }
    };
    json assistant_msg = {
        { "role",    "assistant"            },
        { "content", "UNIQUE_CONTENT_12345" }
    };

    templates_params inputs;
    inputs.messages                         = { user_msg, assistant_msg };
    // Try with thinking enabled first (some templates only wrap content when reasoning is present)
    inputs.extra_context["thinking"]        = true;
    inputs.enable_thinking                  = true;

    std::string output_with_thinking;
    try {
        output_with_thinking = common_chat_template_direct_apply(tmpl, inputs);
    } catch (...) {
        output_with_thinking = "";
    }

    // Also render without thinking
    inputs.extra_context["thinking"]        = false;
    inputs.enable_thinking                  = false;

    std::string output_no_thinking;
    try {
        output_no_thinking = common_chat_template_direct_apply(tmpl, inputs);
    } catch (...) {
        output_no_thinking = "";
    }

    // Check both outputs for content markers
    auto find_content_markers = [&](const std::string & output) -> std::pair<std::string, std::string> {
        size_t marker_pos = output.find("UNIQUE_CONTENT_12345");
        if (marker_pos == std::string::npos) {
            return { "", "" };
        }

        // Known content marker patterns
        std::vector<std::pair<std::string, std::string>> patterns = {
            { "<|START_RESPONSE|>", "<|END_RESPONSE|>"      },
            { "<|response|>",       "<|/response|>"         },
            { "<response>",         "</response>"           },
            { "<output>",           "</output>"             },
            { "<answer>",           "</answer>"             },
            { "<|CHATBOT_TOKEN|>",  "<|END_OF_TURN_TOKEN|>" },
        };

        for (const auto & [start_pattern, end_pattern] : patterns) {
            size_t start_pos = output.rfind(start_pattern, marker_pos);
            if (start_pos != std::string::npos) {
                // Check that there's only whitespace between the start pattern and our marker
                std::string between =
                    output.substr(start_pos + start_pattern.length(), marker_pos - start_pos - start_pattern.length());
                size_t first_non_ws = between.find_first_not_of(" \t\n\r");
                if (first_non_ws == std::string::npos) {
                    // Found valid start marker, look for end marker
                    size_t marker_end = marker_pos + strlen("UNIQUE_CONTENT_12345");
                    size_t end_pos    = output.find(end_pattern, marker_end);
                    if (end_pos != std::string::npos) {
                        std::string after              = output.substr(marker_end, end_pos - marker_end);
                        size_t      first_non_ws_after = after.find_first_not_of(" \t\n\r");
                        if (first_non_ws_after == std::string::npos) {
                            return { start_pattern, end_pattern };
                        }
                    }
                }
            }
        }

        return { "", "" };
    };

    auto [start_with_thinking, end_with_thinking] = find_content_markers(output_with_thinking);
    auto [start_no_thinking, end_no_thinking]     = find_content_markers(output_no_thinking);

    if (!start_with_thinking.empty() && !start_no_thinking.empty()) {
        // Content is always wrapped
        cs.content_mode  = content_structure::CONTENT_ALWAYS_WRAPPED;
        cs.content_start = start_with_thinking;
        cs.content_end   = end_with_thinking;
        LOG_DBG("Content markers found in both thinking modes (ALWAYS_WRAPPED)\n");
    } else if (!start_with_thinking.empty() && start_no_thinking.empty()) {
        // Content is wrapped only when reasoning is present
        cs.content_mode  = content_structure::CONTENT_WRAPPED_WITH_REASONING;
        cs.content_start = start_with_thinking;
        cs.content_end   = end_with_thinking;
        LOG_DBG("Content markers found only with thinking enabled (WRAPPED_WITH_REASONING)\n");
    } else if (!start_no_thinking.empty()) {
        // Unusual: content wrapped without thinking but not with? Use what we found
        cs.content_mode  = content_structure::CONTENT_ALWAYS_WRAPPED;
        cs.content_start = start_no_thinking;
        cs.content_end   = end_no_thinking;
        LOG_DBG("Content markers found only without thinking (treating as ALWAYS_WRAPPED)\n");
    } else {
        cs.content_mode = content_structure::CONTENT_PLAIN;
        LOG_DBG("No content markers detected (PLAIN)\n");
    }

    LOG_DBG("Content markers: start='%s', end='%s'\n", cs.content_start.c_str(), cs.content_end.c_str());
}

content_structure::reasoning_mode_type template_analyzer::detect_reasoning_mode(const content_structure & cs,
                                                                                const std::string &       prompt) {
    LOG_DBG("=== DETECTING REASONING MODE ===\n");

    // If both markers are empty, mode is NONE
    if (cs.reasoning_start.empty() && cs.reasoning_end.empty()) {
        LOG_DBG("No reasoning markers, mode=REASONING_NONE\n");
        return content_structure::REASONING_NONE;
    }

    // Handle case with end marker but no start marker (implicit start)
    if (cs.reasoning_start.empty() && !cs.reasoning_end.empty()) {
        LOG_DBG("Reasoning end marker present but no start marker, mode=REASONING_FORCED_OPEN\n");
        return content_structure::REASONING_FORCED_OPEN;
    }

    // Check if the prompt ends with the reasoning start marker (forced open)
    std::string trimmed_prompt = prompt;
    trim_trailing_newlines(trimmed_prompt);

    std::string trimmed_marker = cs.reasoning_start;
    trim_whitespace(trimmed_marker);

    if (string_ends_with(trimmed_prompt, trimmed_marker)) {
        LOG_DBG("Prompt ends with reasoning start marker, mode=REASONING_FORCED_OPEN\n");
        return content_structure::REASONING_FORCED_OPEN;
    }

    // Otherwise, reasoning is optional
    LOG_DBG("Reasoning markers present but not forced, mode=REASONING_OPTIONAL\n");
    return content_structure::REASONING_OPTIONAL;
}

tool_call_structure template_analyzer::analyze_tool_structure(const common_chat_template & tmpl,
                                                              const content_structure &    content) {
    (void) content;  // May be used in future for better tool detection

    LOG_DBG("=== PHASE 2: ANALYZING TOOL STRUCTURE ===\n");

    tool_call_structure ts;

    // Use differential analysis to detect tool patterns
    // This now includes a robust test that renders two payloads:
    // 1. Tool definitions + content only
    // 2. Tool definitions + content + tool calls
    // If outputs are identical, the template doesn't support tool calls
    auto discovered = analyze_by_differential(tmpl);
    auto format     = determine_format_from_patterns(discovered);

    // Strip EOS tokens from discovered patterns (handles both standard <|eos|> and fullwidth <｜end▁of▁sentence｜>)
    if (!discovered.tool_call_closer.empty()) {
        LOG_DBG("Before stripping: tool_call_closer='%s' (len=%zu)\n", discovered.tool_call_closer.c_str(),
                discovered.tool_call_closer.length());
        discovered.tool_call_closer = strip_eos_token(discovered.tool_call_closer);
        LOG_DBG("After stripping: tool_call_closer='%s'\n", discovered.tool_call_closer.c_str());
    }
    if (!discovered.tool_call_end_marker.empty()) {
        discovered.tool_call_end_marker = strip_eos_token(discovered.tool_call_end_marker);
    }

    if (format == FORMAT_UNKNOWN) {
        LOG_DBG("Template does not support tool calls (differential analysis returned no patterns)\n");
        ts.supports_tools = false;
        return ts;
    }

    // Propagate requires_nonnull_content flag from differential analysis
    ts.requires_nonnull_content = discovered.requires_nonnull_content;
    if (ts.requires_nonnull_content) {
        LOG_DBG("Template requires non-null content (renders null as 'None')\n");
    }

    // Check if minja reports tool call support (for informational purposes)
    auto caps = tmpl.original_caps();
    if (!caps.supports_tool_calls) {
        LOG_DBG("Note: minja caps indicate no tool support, but differential analysis found patterns\n");
    }

    if (format == FORMAT_JSON_NATIVE) {
        analyze_json_format(ts, discovered);
    } else if (format == FORMAT_XML_CONSTRUCTED) {
        analyze_xml_format(ts, discovered);
    } else if (format == FORMAT_BRACKET_TAG) {
        analyze_bracket_tag_format(ts, discovered);
    } else if (format == FORMAT_RECIPIENT_BASED) {
        analyze_recipient_based_format(ts, discovered);
    } else if (format == FORMAT_MARKDOWN_CODE_BLOCK) {
        analyze_markdown_code_block_format(ts, discovered);
    }

    return ts;
}

void template_analyzer::collect_preserved_tokens(template_analysis_result & result) {
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
    if (result.tools.function_format == tool_call_structure::FUNC_TAG_WITH_NAME) {
        if (!result.tools.function_prefix.empty()) {
            tokens.push_back(result.tools.function_prefix);
        }
        if (!result.tools.function_close.empty()) {
            tokens.push_back(result.tools.function_close);
        }
    }

    // Add markers for prefixed-indexed formats (e.g., Kimi-K2)
    if (result.tools.function_format == tool_call_structure::FUNC_PREFIXED_INDEXED) {
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
    if (result.tools.argument_format == tool_call_structure::ARGS_TAGGED) {
        if (!result.tools.arg_prefix.empty()) {
            tokens.push_back(result.tools.arg_prefix);
        }
        if (!result.tools.arg_close.empty()) {
            tokens.push_back(result.tools.arg_close);
        }
    }

    // Add markers for markdown code block format (Cohere Command-R Plus)
    if (result.tools.function_format == tool_call_structure::FUNC_MARKDOWN_CODE_BLOCK) {
        if (!result.tools.code_block_marker.empty()) {
            tokens.push_back(result.tools.code_block_marker);
        }
        if (!result.tools.tool_section_end.empty()) {
            tokens.push_back(result.tools.tool_section_end);  // Closing code fence ```
        }
    }

    result.preserved_tokens = tokens;
    LOG_DBG("Collected %zu preserved tokens\n", tokens.size());
}

void template_analyzer::analyze_json_format(tool_call_structure & ts, const internal_discovered_pattern & discovered) {
    ts.supports_tools     = true;
    ts.function_format    = tool_call_structure::FUNC_JSON_OBJECT;
    ts.argument_format    = tool_call_structure::ARGS_JSON;
    ts.tool_section_start = discovered.tool_call_start_marker;
    ts.tool_section_end   = discovered.tool_call_end_marker;
    ts.name_field         = discovered.tool_name_field;
    ts.args_field         = discovered.tool_args_field;
    ts.id_field           = discovered.tool_id_field;

    // Check for FUNC_NAME_AS_KEY format (e.g. Apertus: {"function_name": args})
    // This is characterized by the opener ending in {" and no explicit name field found yet
    if (!discovered.tool_call_opener.empty() && discovered.tool_call_opener.length() >= 2 &&
        discovered.tool_call_opener.substr(discovered.tool_call_opener.length() - 2) == "{\"") {
        LOG_DBG("Detected FUNC_NAME_AS_KEY format from tool_call_opener ending in '{\"' \n");
        ts.function_format = tool_call_structure::FUNC_NAME_AS_KEY;
    }

    // For JSON_NATIVE format, clean up tool_section_end to only include the closing tag
    // The differential analysis may include JSON closing braces (e.g., "}}\n</tool_call>")
    // but the parser handles JSON separately, so we only need the tag marker
    if (!ts.tool_section_end.empty()) {
        size_t tag_start = ts.tool_section_end.find("</");
        if (tag_start != std::string::npos) {
            size_t tag_end = ts.tool_section_end.find('>', tag_start);
            if (tag_end != std::string::npos) {
                // Check if there is a closing bracket ']' before the tag
                size_t bracket_pos = ts.tool_section_end.rfind(']', tag_start);
                if (bracket_pos != std::string::npos) {
                    // Include the bracket
                    ts.tool_section_end = ts.tool_section_end.substr(bracket_pos, tag_end - bracket_pos + 1);
                } else {
                    ts.tool_section_end = ts.tool_section_end.substr(tag_start, tag_end - tag_start + 1);
                }
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
}

void template_analyzer::analyze_xml_format(tool_call_structure & ts, const internal_discovered_pattern & discovered) {
    ts.supports_tools     = true;
    ts.function_format    = tool_call_structure::FUNC_TAG_WITH_NAME;
    ts.tool_section_start = discovered.tool_call_start_marker;
    ts.tool_section_end   = discovered.tool_call_end_marker;

    // Extract function tag patterns
    if (!discovered.function_opener.empty()) {
        char first = discovered.function_opener[0];
        if (first != '<' && first != '{' && first != '[') {
            // Non-XML/JSON prefix format (e.g., ">>>", "##", etc.)
            // Function name follows prefix directly, ends with newline
            ts.function_prefix = discovered.function_opener;
            ts.function_suffix = "\n";  // Function name typically ends with newline
            ts.function_close  = "";    // No closing tag for prefix formats
        } else {
            size_t eq_pos = discovered.function_opener.find('=');
            if (eq_pos != std::string::npos) {
                // Check if there's a quote after the equals sign
                if (eq_pos + 1 < discovered.function_opener.length() &&
                    (discovered.function_opener[eq_pos + 1] == '"' || discovered.function_opener[eq_pos + 1] == '\'')) {
                    ts.function_prefix = discovered.function_opener.substr(0, eq_pos + 2);
                } else {
                    ts.function_prefix = discovered.function_opener.substr(0, eq_pos + 1);
                }
                ts.function_suffix = discovered.function_name_suffix;

                // For formats like <function=name>{args}</function>, where function_prefix
                // IS the section start (no separate wrapper), tool_section_end is the function close.
                // But for nested formats like <tool_call><function=name>...</function></tool_call>,
                // the function_close is separate from tool_section_end.
                // We detect the non-nested case when tool_section_start matches function_prefix
                // (or tool_section_start was already cleared because it matched).
                bool section_start_matches_prefix = ts.tool_section_start.empty() ||
                                                    ts.tool_section_start.find(ts.function_prefix) == 0 ||
                                                    ts.function_prefix.find(ts.tool_section_start) == 0;
                if (section_start_matches_prefix && ts.function_prefix.find('<') == 0 && !ts.tool_section_end.empty() &&
                    ts.tool_section_end.find("</") == 0) {
                    ts.function_close   = ts.tool_section_end;
                    ts.tool_section_end = "";  // Clear to avoid double wrapping
                } else {
                    ts.function_close = discovered.function_closer;
                }
            } else if (!discovered.function_opener.empty() && discovered.function_opener[0] == '<') {
                // Check for FUNC_PREFIXED_INDEXED format
                // Detected by: function_opener ends with "." (namespace separator)
                //              AND function_name_suffix starts with ":" followed by digit (index)
                // Example: <|tool_call_begin|>functions.name:0<|tool_call_argument_begin|>
                size_t namespace_dot = discovered.function_opener.rfind('.');
                bool   has_namespace =
                    (namespace_dot != std::string::npos && namespace_dot == discovered.function_opener.length() - 1);

                bool has_index =
                    (!discovered.function_name_suffix.empty() && discovered.function_name_suffix[0] == ':' &&
                     discovered.function_name_suffix.length() > 1 &&
                     std::isdigit(static_cast<unsigned char>(discovered.function_name_suffix[1])));

                if (has_namespace && has_index) {
                    LOG_DBG("Detected FUNC_PREFIXED_INDEXED format: namespace ends with '.', suffix has ':N' index\n");
                    ts.function_format = tool_call_structure::FUNC_PREFIXED_INDEXED;

                    // Split function_opener into per_call_start and function_namespace
                    // e.g., "<|tool_call_begin|>functions." -> "<|tool_call_begin|>" + "functions."
                    // Find where the namespace starts (after the last '>' before the '.')
                    size_t namespace_start = discovered.function_opener.rfind('>');
                    if (namespace_start != std::string::npos && namespace_start < namespace_dot) {
                        ts.per_call_start     = discovered.function_opener.substr(0, namespace_start + 1);
                        ts.function_namespace = discovered.function_opener.substr(namespace_start + 1);
                    } else {
                        // Fallback: namespace is just the part ending with '.'
                        ts.per_call_start     = discovered.function_opener.substr(0, namespace_dot);
                        ts.function_namespace = ".";
                    }

                    // Extract args_marker from function_name_suffix
                    // Format: ":0<|some_marker|>" -> index is ":0", args_marker is "<|some_marker|>"
                    size_t args_marker_start = discovered.function_name_suffix.find('<');
                    if (args_marker_start != std::string::npos) {
                        size_t args_marker_end = discovered.function_name_suffix.find('>', args_marker_start);
                        if (args_marker_end != std::string::npos) {
                            ts.args_marker = discovered.function_name_suffix.substr(
                                args_marker_start, args_marker_end - args_marker_start + 1);
                        }
                    }

                    // Derive per_call_end from tool_call_closer by finding corresponding end marker
                    // tool_call_closer contains per_call_end + tool_section_end
                    // We find per_call_end by looking for a marker that structurally matches per_call_start
                    if (!discovered.tool_call_closer.empty() && !ts.per_call_start.empty()) {
                        // Extract structural pattern from per_call_start
                        // e.g., "<|tool_call_begin|>" -> look for "<|tool_call_...|>" in closer
                        size_t start_marker_begin = ts.per_call_start.find("<|");
                        size_t start_marker_end   = ts.per_call_start.rfind("|>");
                        if (start_marker_begin != std::string::npos && start_marker_end != std::string::npos) {
                            // Find the base pattern (e.g., "<|tool_call" from "<|tool_call_begin|>")
                            std::string start_content = ts.per_call_start.substr(
                                start_marker_begin + 2, start_marker_end - start_marker_begin - 2);
                            // Find a related marker in the closer
                            size_t closer_pos = discovered.tool_call_closer.find("<|");
                            while (closer_pos != std::string::npos) {
                                size_t closer_end = discovered.tool_call_closer.find("|>", closer_pos);
                                if (closer_end != std::string::npos) {
                                    std::string candidate =
                                        discovered.tool_call_closer.substr(closer_pos, closer_end - closer_pos + 2);
                                    // Check if this marker shares a common prefix with per_call_start
                                    // (ignoring _begin vs _end suffix differences)
                                    std::string candidate_content = candidate.substr(2, candidate.length() - 4);
                                    // Find common prefix between start_content and candidate_content
                                    size_t      common_len        = 0;
                                    while (common_len < start_content.length() &&
                                           common_len < candidate_content.length() &&
                                           start_content[common_len] == candidate_content[common_len]) {
                                        common_len++;
                                    }
                                    // If substantial overlap (>50%), this is likely the per_call_end
                                    if (common_len > start_content.length() / 2 &&
                                        candidate_content.find("end") != std::string::npos) {
                                        ts.per_call_end = candidate;
                                        break;
                                    }
                                }
                                closer_pos = discovered.tool_call_closer.find("<|", closer_pos + 1);
                            }
                        }
                    }

                    // Derive tool_section_end from tool_section_start by finding matching end marker
                    // For FUNC_PREFIXED_INDEXED, we always derive this to get the correct marker
                    // (the default discovered.tool_call_end_marker may contain extra content)
                    if (!ts.tool_section_start.empty()) {
                        size_t start_marker_begin = ts.tool_section_start.find("<|");
                        size_t start_marker_end   = ts.tool_section_start.rfind("|>");
                        if (start_marker_begin != std::string::npos && start_marker_end != std::string::npos) {
                            std::string start_content = ts.tool_section_start.substr(
                                start_marker_begin + 2, start_marker_end - start_marker_begin - 2);
                            size_t closer_pos = discovered.tool_call_closer.find("<|");
                            while (closer_pos != std::string::npos) {
                                size_t closer_end = discovered.tool_call_closer.find("|>", closer_pos);
                                if (closer_end != std::string::npos) {
                                    std::string candidate =
                                        discovered.tool_call_closer.substr(closer_pos, closer_end - closer_pos + 2);
                                    std::string candidate_content = candidate.substr(2, candidate.length() - 4);
                                    size_t      common_len        = 0;
                                    while (common_len < start_content.length() &&
                                           common_len < candidate_content.length() &&
                                           start_content[common_len] == candidate_content[common_len]) {
                                        common_len++;
                                    }
                                    if (common_len > start_content.length() / 2 &&
                                        candidate_content.find("end") != std::string::npos) {
                                        ts.tool_section_end = candidate;
                                        break;
                                    }
                                }
                                closer_pos = discovered.tool_call_closer.find("<|", closer_pos + 1);
                            }
                        }
                    }

                    LOG_DBG(
                        "FUNC_PREFIXED_INDEXED: per_call_start='%s', namespace='%s', args_marker='%s', "
                        "per_call_end='%s'\n",
                        ts.per_call_start.c_str(), ts.function_namespace.c_str(), ts.args_marker.c_str(),
                        ts.per_call_end.c_str());
                } else {
                    // Other formats like <|tool_call_begin|>name (non-indexed)
                    // Use function_opener as default, but try to use full tool_call_opener if it contains more
                    ts.function_prefix = discovered.function_opener;
                    LOG_DBG("Initial function_prefix: '%s', tool_call_opener: '%s', tool_section_start: '%s'\n",
                            ts.function_prefix.c_str(), discovered.tool_call_opener.c_str(),
                            ts.tool_section_start.c_str());
                    if (!ts.tool_section_start.empty() &&
                        discovered.tool_call_opener.find(ts.tool_section_start) == 0) {
                        std::string remainder = discovered.tool_call_opener.substr(ts.tool_section_start.length());
                        LOG_DBG("Derived remainder: '%s'\n", remainder.c_str());
                        if (remainder.length() > ts.function_prefix.length()) {
                            ts.function_prefix = remainder;
                        }
                    }
                    ts.function_suffix = discovered.function_name_suffix;
                    ts.function_close  = discovered.function_closer;
                }
            }
        }
    }

    // Fix for templates where tool_section_start matches function_prefix (double wrapping)
    // e.g. Functionary: tool_section_start="<function=", function_prefix="<function="
    if (!ts.tool_section_start.empty() && !ts.function_prefix.empty() && ts.tool_section_start == ts.function_prefix) {
        LOG_DBG("tool_section_start matches function_prefix, clearing section start to avoid double wrapping\n");
        ts.tool_section_start = "";
    }

    // Similar check for tool_section_end matching function_close
    if (!ts.tool_section_end.empty() && !ts.function_close.empty() && ts.tool_section_end == ts.function_close) {
        LOG_DBG("tool_section_end matches function_close, clearing section end to avoid double wrapping\n");
        ts.tool_section_end = "";
    }

    // Handle nested container markers (e.g., DeepSeek R1 style)
    // If function_suffix contains markdown code block (```), the template uses nested markers
    // tool_section_start might be: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function
    // We need to derive tool_section_end from the outer marker pattern
    if (ts.function_suffix.find("```") != std::string::npos && !ts.tool_section_start.empty()) {
        // Check if tool_section_start contains nested markers (both outer and per-call)
        // Pattern: <X_calls_begin><X_call_begin>...
        // We look for "calls" pattern which indicates an outer container
        size_t calls_pos = ts.tool_section_start.find("calls");
        if (calls_pos != std::string::npos && calls_pos < ts.tool_section_start.length()) {
            // Find where the outer marker ends (after the first >)
            size_t first_close = ts.tool_section_start.find('>', calls_pos);
            if (first_close != std::string::npos && first_close < ts.tool_section_start.length() - 1) {
                // Extract the outer marker (e.g., "<｜tool▁calls▁begin｜>")
                std::string outer_start = ts.tool_section_start.substr(0, first_close + 1);
                // Derive the outer end marker by replacing "begin" with "end"
                size_t      begin_pos   = outer_start.find("begin");
                if (begin_pos != std::string::npos) {
                    std::string outer_end =
                        outer_start.substr(0, begin_pos) + "end" + outer_start.substr(begin_pos + 5);
                    ts.tool_section_end = outer_end;

                    // Strip outer marker from function_prefix and function_opener if they were combined
                    if (ts.tool_section_start.find(outer_start) == 0) {
                        std::string remainder    = ts.tool_section_start.substr(outer_start.length());
                        // Trim leading whitespace from remainder
                        size_t      first_non_ws = remainder.find_first_not_of(" \t\n\r");
                        if (first_non_ws != std::string::npos && first_non_ws > 0) {
                            remainder = remainder.substr(first_non_ws);
                        }

                        // Concatenate with existing function_prefix (e.g. separator tag)
                        // but avoid double-concatenation if already present
                        if (!remainder.empty() && ts.function_prefix.find(remainder) == std::string::npos) {
                            ts.function_prefix = remainder + ts.function_prefix;
                        }
                    }

                    // Update tool_section_start to be just the outer marker
                    ts.tool_section_start = outer_start;

                    // Check if there's a fence in tool_call_closer that should be in function_close
                    // (DeepSeek R1 wraps JSON in markdown blocks within the custom tags)
                    if (discovered.tool_call_closer.find("```") != std::string::npos) {
                        size_t fence_pos = discovered.tool_call_closer.find("```");
                        // Include leading newlines if present before the fence
                        while (fence_pos > 0 && (discovered.tool_call_closer[fence_pos - 1] == '\n' ||
                                                 discovered.tool_call_closer[fence_pos - 1] == '\r')) {
                            fence_pos--;
                        }
                        ts.function_close = discovered.tool_call_closer.substr(fence_pos);

                        // Clip function_close to not include tool_section_end (if they were combined in differential analysis)
                        if (!ts.tool_section_end.empty()) {
                            size_t end_pos = ts.function_close.find(ts.tool_section_end);
                            if (end_pos != std::string::npos) {
                                ts.function_close = ts.function_close.substr(0, end_pos);
                            }
                        }

                        // Further trim any trailing EOS or prompt garbage
                        ts.function_close     = strip_eos_token(ts.function_close);
                        size_t prompt_garbage = ts.function_close.find("<｜");
                        if (prompt_garbage != std::string::npos && prompt_garbage > 0 &&
                            ts.function_close.substr(prompt_garbage).find("Assistant") != std::string::npos) {
                            ts.function_close = ts.function_close.substr(0, prompt_garbage);
                        }
                    }
                }
            }
        }
    }

    // General cleanup for tool_section_end when tool_section_start uses token markers (<|...|> or <｜...｜>)
    // If tool_section_start contains a token marker with "begin" and tool_section_end is messy (contains }
    // or multiple markers), derive tool_section_end by finding matching end marker in tool_call_closer
    if (!ts.tool_section_start.empty() && !discovered.tool_call_closer.empty()) {
        // Check if tool_section_start contains a token marker
        size_t start_opener_pos = find_token_opener(ts.tool_section_start, 0);
        size_t start_closer_pos = find_token_closer(ts.tool_section_start, start_opener_pos);
        if (start_opener_pos != std::string::npos && start_closer_pos != std::string::npos) {
            size_t      opener_len    = get_token_opener_length(ts.tool_section_start, start_opener_pos);
            // Extract the token content (between opener and closer)
            std::string start_content = ts.tool_section_start.substr(start_opener_pos + opener_len,
                                                                     start_closer_pos - start_opener_pos - opener_len);

            // Check if tool_section_end needs cleanup (starts with } or contains multiple markers)
            bool needs_cleanup = false;
            if (!ts.tool_section_end.empty() && ts.tool_section_end[0] == '}') {
                needs_cleanup = true;
            }
            // Count tokens in tool_section_end
            size_t token_count = 0;
            size_t pos         = 0;
            while ((pos = find_token_opener(ts.tool_section_end, pos)) != std::string::npos) {
                token_count++;
                pos += get_token_opener_length(ts.tool_section_end, pos);
            }
            if (token_count > 1) {
                needs_cleanup = true;
            }

            if (needs_cleanup) {
                // Find matching end marker in tool_call_closer
                // Look for a token that has similar content but with "end" instead of "begin"
                pos = 0;
                while ((pos = find_token_opener(discovered.tool_call_closer, pos)) != std::string::npos) {
                    size_t end_closer_pos = find_token_closer(discovered.tool_call_closer, pos);
                    if (end_closer_pos != std::string::npos) {
                        size_t      op_len    = get_token_opener_length(discovered.tool_call_closer, pos);
                        size_t      cl_len    = get_token_closer_length(discovered.tool_call_closer, end_closer_pos);
                        std::string candidate = discovered.tool_call_closer.substr(pos, end_closer_pos + cl_len - pos);
                        std::string candidate_content =
                            discovered.tool_call_closer.substr(pos + op_len, end_closer_pos - pos - op_len);

                        // Check if this candidate matches our start marker structure
                        // Start content might be "tool▁calls▁begin" and candidate might be "tool▁calls▁end"
                        size_t begin_in_start   = start_content.find("begin");
                        size_t end_in_candidate = candidate_content.find("end");
                        if (begin_in_start != std::string::npos && end_in_candidate != std::string::npos) {
                            // Check if they share a common prefix (e.g., "tool▁calls▁")
                            std::string start_base = start_content.substr(0, begin_in_start);
                            std::string cand_base  = candidate_content.substr(0, end_in_candidate);
                            if (start_base == cand_base) {
                                ts.tool_section_end = candidate;
                                LOG_DBG(
                                    "Derived tool_section_end='%s' from tool_section_start='%s' using token matching\n",
                                    ts.tool_section_end.c_str(), ts.tool_section_start.c_str());
                                break;
                            }
                        }
                    }
                    pos += get_token_opener_length(discovered.tool_call_closer, pos);
                }
            }
        }
    }

    // Determine argument format
    if (!discovered.parameter_key_prefix.empty() && discovered.parameter_key_prefix.find('<') != std::string::npos) {
        ts.argument_format = tool_call_structure::ARGS_TAGGED;
        ts.arg_prefix      = discovered.parameter_key_prefix;
        ts.arg_suffix      = discovered.parameter_key_suffix;
        ts.arg_close       = discovered.parameter_closer;
        ts.arg_separator   = discovered.argument_separator;

        // Check for specific GLM-4 style key-value tags
        // Format: <arg_key>key</arg_key>\n<arg_value>value</arg_value>
        // Analyzer detects suffix as: </arg_key>\n<arg_value>
        if (ts.arg_suffix.find("<arg_value>") != std::string::npos) {
            ts.argument_format = tool_call_structure::ARGS_KEY_VALUE_TAGS;

            // Clean up suffix to be just the key closer
            size_t val_opener = ts.arg_suffix.find("<arg_value>");
            if (val_opener != std::string::npos) {
                // Extract just the </arg_key> part (trimming whitespace/newlines before <arg_value>)
                std::string key_closer = ts.arg_suffix.substr(0, val_opener);
                // Trim trailing whitespace/newlines
                while (!key_closer.empty() &&
                       (key_closer.back() == '\n' || key_closer.back() == '\r' || key_closer.back() == ' ')) {
                    key_closer.pop_back();
                }
                ts.arg_suffix = key_closer;
            }
        }
    } else {
        ts.argument_format = tool_call_structure::ARGS_JSON;
    }

    LOG_DBG("%s: final markers: section_start='%s', section_end='%s', prefix='%s', close='%s'\n", __func__,
            ts.tool_section_start.c_str(), ts.tool_section_end.c_str(), ts.function_prefix.c_str(),
            ts.function_close.c_str());
}

void template_analyzer::analyze_bracket_tag_format(tool_call_structure &               ts,
                                                   const internal_discovered_pattern & discovered) {
    // Bracket-tag format: [TOOL_CALLS]name[CALL_ID]id[ARGS]{...} (Mistral Small 3.2)
    ts.supports_tools  = true;
    ts.function_format = tool_call_structure::FUNC_BRACKET_TAG;
    ts.argument_format = tool_call_structure::ARGS_JSON;

    // The function_opener contains the bracket tag before the function name (e.g., "[TOOL_CALLS]")
    // Each tool call starts with this tag, so it's the per_call_start, not a section wrapper
    // tool_section_start/end should be empty since there's no overall section wrapper
    ts.tool_section_start = "";
    ts.tool_section_end   = "";
    ts.per_call_start     = discovered.function_opener;

    // Extract markers from function_name_suffix (e.g., "[CALL_ID]call_0001[ARGS]" or just "[ARGS]")
    // Pattern: [ID_MARKER]...[ARGS_MARKER] or just [ARGS_MARKER]
    if (!discovered.function_name_suffix.empty()) {
        // Find all bracket tags in the suffix
        std::vector<std::string> tags;
        size_t                   pos = 0;
        while ((pos = discovered.function_name_suffix.find('[', pos)) != std::string::npos) {
            size_t end = discovered.function_name_suffix.find(']', pos);
            if (end != std::string::npos) {
                tags.push_back(discovered.function_name_suffix.substr(pos, end - pos + 1));
                pos = end + 1;
            } else {
                break;
            }
        }

        // Classify tags: args marker contains "ARG", id marker contains "ID" or "CALL"
        for (const auto & tag : tags) {
            std::string upper_tag = tag;
            for (auto & c : upper_tag) {
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            }
            if (upper_tag.find("ARG") != std::string::npos) {
                ts.args_marker = tag;
            } else if (upper_tag.find("ID") != std::string::npos || upper_tag.find("CALL") != std::string::npos) {
                ts.id_marker = tag;
            }
        }
    }

    LOG_DBG("FUNC_BRACKET_TAG: per_call_start='%s', id_marker='%s', args_marker='%s'\n", ts.per_call_start.c_str(),
            ts.id_marker.c_str(), ts.args_marker.c_str());
}

void template_analyzer::analyze_recipient_based_format(tool_call_structure &               ts,
                                                       const internal_discovered_pattern & discovered) {
    // Recipient-based format (Functionary v3.2): >>>recipient\n{content}
    // where recipient is either "all" (for content) or a function name (for tools)
    ts.supports_tools  = true;
    ts.function_format = tool_call_structure::FUNC_RECIPIENT_BASED;
    ts.argument_format = tool_call_structure::ARGS_JSON;  // Python dict format, parse as JSON

    // The tool_call_start_marker is used as the recipient delimiter
    ts.tool_section_start = discovered.tool_call_start_marker;
    ts.tool_section_end   = "";

    // For recipient-based format, content is wrapped in tool_call_start_marker + "all\n"
    // This needs to be detected and stripped. We detect this by checking if the
    // content_start marker (from phase 1 analysis) starts with tool_call_start_marker
    // If not already detected, infer it from the pattern.
    // Note: This is set on the ContentStructure result, not ToolCallStructure
    // The caller (analyze_template) will have the ContentStructure to modify

    LOG_DBG("FUNC_RECIPIENT_BASED: delimiter='%s'\n", ts.tool_section_start.c_str());
}

void template_analyzer::analyze_markdown_code_block_format(tool_call_structure &               ts,
                                                           const internal_discovered_pattern & discovered) {
    // Markdown code block format (Cohere Command-R Plus):
    // Action:
    // ```json
    // [
    //     {
    //         "tool_name": "...",
    //         "parameters": {...}
    //     }
    // ]
    // ```
    ts.supports_tools  = true;
    ts.function_format = tool_call_structure::FUNC_MARKDOWN_CODE_BLOCK;
    ts.argument_format = tool_call_structure::ARGS_JSON;

    // Extract the code block marker (e.g., "Action:")
    // The tool_call_start_marker should contain "Action:" followed by newline
    if (!discovered.tool_call_start_marker.empty()) {
        // Extract just the marker text (e.g., "Action:")
        // The marker may be followed by whitespace/newline in the template
        size_t marker_end = discovered.tool_call_start_marker.find_first_of(" \n\r\t");
        if (marker_end != std::string::npos) {
            ts.code_block_marker = discovered.tool_call_start_marker.substr(0, marker_end);
        } else {
            ts.code_block_marker = discovered.tool_call_start_marker;
        }
    }

    // Extract the code block language (e.g., "json")
    // For Command-R Plus format: Action:\n```json\n[...]
    // The code fence is in tool_call_opener (before the function name), not function_name_suffix
    if (!discovered.function_name_suffix.empty() && discovered.function_name_suffix.find("```") != std::string::npos) {
        // Format: ```json or ```json\n
        size_t code_fence_pos = discovered.function_name_suffix.find("```");
        size_t lang_start     = code_fence_pos + 3;
        // Find the end of the language identifier (newline, space, or end of string)
        size_t lang_end       = discovered.function_name_suffix.find_first_of(" \n\r\t", lang_start);
        if (lang_end != std::string::npos && lang_end > lang_start) {
            ts.code_block_language = discovered.function_name_suffix.substr(lang_start, lang_end - lang_start);
        } else {
            // No language identifier after ```, will use "json" as default
            ts.code_block_language = "json";
        }
    } else if (!discovered.tool_call_opener.empty() && discovered.tool_call_opener.find("```") != std::string::npos) {
        // Code fence is in tool_call_opener (before the function name)
        // Format: Action:\n```json\n[...
        size_t code_fence_pos = discovered.tool_call_opener.find("```");
        size_t lang_start     = code_fence_pos + 3;
        // Find the end of the language identifier (newline, space, or end of string)
        size_t lang_end       = discovered.tool_call_opener.find_first_of(" \n\r\t", lang_start);
        if (lang_end != std::string::npos && lang_end > lang_start) {
            ts.code_block_language = discovered.tool_call_opener.substr(lang_start, lang_end - lang_start);
        } else {
            // No language identifier after ```, will use "json" as default
            ts.code_block_language = "json";
        }
    } else {
        // Default to "json" if no code fence found
        ts.code_block_language = "json";
    }

    // The tool_section_end should be the closing code fence: ```
    if (!discovered.tool_call_closer.empty() && discovered.tool_call_closer.find("```") != std::string::npos) {
        // Extract just the closing code fence (may have trailing content)
        size_t fence_pos = discovered.tool_call_closer.find("```");
        size_t fence_end = fence_pos + 3;
        // Include any non-newline characters after ``` (like language identifier if present)
        while (fence_end < discovered.tool_call_closer.length() && discovered.tool_call_closer[fence_end] != '\n' &&
               discovered.tool_call_closer[fence_end] != '\r') {
            fence_end++;
        }
        ts.tool_section_end = discovered.tool_call_closer.substr(fence_pos, fence_end - fence_pos);
    } else {
        // Default closing code fence
        ts.tool_section_end = "```";
    }

    // JSON array format for function calls
    ts.name_field = discovered.tool_name_field;
    ts.args_field = discovered.tool_args_field;
    ts.id_field   = discovered.tool_id_field;

    LOG_DBG("FUNC_MARKDOWN_CODE_BLOCK: marker='%s', language='%s', section_end='%s'\n", ts.code_block_marker.c_str(),
            ts.code_block_language.c_str(), ts.tool_section_end.c_str());
}
