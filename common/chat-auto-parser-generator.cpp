#include "chat-auto-parser.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "chat-peg-parser.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>
#include <stdexcept>

using json = nlohmann::ordered_json;

common_chat_params UniversalPEGGenerator::generate_parser(const TemplatePattern &         pattern,
                                                          const minja::chat_template &    tmpl,
                                                          const struct templates_params & inputs) {
    common_chat_params data;
    TemplatePattern    local_pattern = pattern;

    try {
        LOG_DBG("=== GENERATING PEG PARSER ===\n");
        LOG_DBG("Pattern format: %d\n", local_pattern.format);
        LOG_DBG("Markers:\n");
        LOG_DBG("  tool_call_start: '%s'\n", local_pattern.special_markers.at("tool_call_start_marker").c_str());
        LOG_DBG("  tool_call_end:   '%s'\n", local_pattern.special_markers.at("tool_call_end_marker").c_str());
        LOG_DBG("  function_opener: '%s'\n", local_pattern.special_markers.at("function_opener").c_str());
        LOG_DBG("  reasoning_start: '%s'\n", local_pattern.special_markers.at("reasoning_start_marker").c_str());
        LOG_DBG("  reasoning_end:   '%s'\n", local_pattern.special_markers.at("reasoning_end_marker").c_str());

        // Calculate prompt first to detect forced thinking
        data.prompt = apply_template(tmpl, inputs);

        bool        thinking_forced_open = false;
        std::string start_marker         = local_pattern.special_markers.at("reasoning_start_marker");

        // Robust check for forced thinking (trim whitespace)
        std::string prompt_trimmed = data.prompt;
        while (!prompt_trimmed.empty() && std::isspace(static_cast<unsigned char>(prompt_trimmed.back()))) {
            prompt_trimmed.pop_back();
        }

        LOG_DBG("Prompt trimmed end: '%s'\n",
                prompt_trimmed.length() > 20 ? ("..." + prompt_trimmed.substr(prompt_trimmed.length() - 20)).c_str() :
                                               prompt_trimmed.c_str());

        if (!start_marker.empty()) {
            if (string_ends_with(prompt_trimmed, start_marker)) {
                if (!inputs.enable_thinking) {
                    data.prompt += local_pattern.special_markers.at("reasoning_end_marker");
                } else {
                    fprintf(stderr, "Thinking forced open via start marker match\n");
                    thinking_forced_open = true;
                }
            }
        } else if (prompt_trimmed.length() > 2 && prompt_trimmed.back() == '>' && inputs.enable_thinking) {
            fprintf(stderr, "Checking inference for prompt ending with >\n");
            // ... generic inference ...
            size_t open = prompt_trimmed.rfind('<');
            if (open != std::string::npos) {
                std::string tag = prompt_trimmed.substr(open);

                // Check if this looks like a reasoning tag (not just any tag)
                // Reasoning tags typically contain words like "think", "reason", "thought", etc.
                std::string tag_name  = tag.substr(1, tag.length() - 2);
                size_t      space_pos = tag_name.find(' ');
                if (space_pos != std::string::npos) {
                    tag_name = tag_name.substr(0, space_pos);
                }

                // Only infer reasoning if the tag name suggests reasoning
                std::string lower_tag_name = tag_name;
                std::transform(lower_tag_name.begin(), lower_tag_name.end(), lower_tag_name.begin(), ::tolower);

                if (lower_tag_name.find("think") != std::string::npos ||
                    lower_tag_name.find("reason") != std::string::npos ||
                    lower_tag_name.find("thought") != std::string::npos ||
                    lower_tag_name.find("reflect") != std::string::npos) {
                    LOG_DBG("Inferred reasoning tag from prompt: '%s'\n", tag.c_str());
                    local_pattern.special_markers["reasoning_start_marker"] = tag;
                    start_marker                                            = tag;

                    // Infer end marker
                    std::string end_marker;
                    std::string name  = tag.substr(1, tag.length() - 2);
                    size_t      space = name.find(' ');
                    if (space != std::string::npos) {
                        name = name.substr(0, space);
                    }
                    end_marker = "</" + name + ">";
                    if (tag[0] == '[') {  // Fix: check tag type for end marker inference
                        std::string name_sq = tag.substr(1, tag.length() - 2);
                        end_marker          = "[/" + name_sq + "]";
                    }

                    local_pattern.special_markers["reasoning_end_marker"] = end_marker;
                    fprintf(stderr, "Thinking forced open via inference\n");
                    thinking_forced_open = true;
                } else {
                    LOG_DBG("Tag '%s' does not appear to be a reasoning tag, skipping inference\n", tag.c_str());
                }
            }
        }

        data.thinking_forced_open = thinking_forced_open;
        fprintf(stderr, "Detailed thinking forced open: %d (marker=%s, prompt_end=%c, enable=%d)\n",
                thinking_forced_open, start_marker.c_str(), prompt_trimmed.empty() ? '?' : prompt_trimmed.back(),
                static_cast<int>(inputs.enable_thinking));

        // Calculate has_tools early for validation checks
        bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();

        // ... validation ...
        if (local_pattern.format == TemplatePattern::JSON_NATIVE) {
            if (has_tools && local_pattern.special_markers.at("tool_call_start_marker").empty() &&
                local_pattern.special_markers.at("tool_call_opener").empty() &&
                local_pattern.special_markers.at("function_opener").empty()) {
                LOG_DBG("JSON_NATIVE format detected but no meaningful markers - falling back to generic parser\n");
                throw std::runtime_error("Template analysis failed: JSON_NATIVE format without meaningful markers");
            }
        } else if (local_pattern.format == TemplatePattern::XML_CONSTRUCTED) {
            if (has_tools && local_pattern.special_markers.at("tool_call_start_marker").empty() &&
                local_pattern.special_markers.at("function_opener").empty()) {
                LOG_DBG("XML_CONSTRUCTED format detected but no meaningful markers - falling back to generic parser\n");
                throw std::runtime_error("Template analysis failed: XML_CONSTRUCTED format without meaningful markers");
            }
        }

        common_peg_arena arena;

        if (local_pattern.format == TemplatePattern::JSON_NATIVE) {
            arena       = build_native_parser(local_pattern, tmpl, inputs, thinking_forced_open);
            // Format is based on template structure, not whether tools are provided at runtime
            data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
            LOG_DBG("Generated JSON_NATIVE parser successfully (format: PEG_NATIVE)\n");
        } else if (local_pattern.format == TemplatePattern::XML_CONSTRUCTED) {
            arena       = build_constructed_parser(local_pattern, tmpl, inputs, thinking_forced_open);
            // Format is based on template structure, not whether tools are provided at runtime
            data.format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;
            LOG_DBG("Generated XML_CONSTRUCTED parser successfully (format: PEG_CONSTRUCTED)\n");
        } else {
            // Treat as content only
            arena       = build_chat_peg_native_parser([&](common_chat_peg_native_builder & p) {
                auto content = p.content(p.rest());
                // Only parse reasoning when reasoning_format != NONE
                // When reasoning_format == NONE, all output (including any thinking markers) goes into content
                if (inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE && thinking_forced_open &&
                    !local_pattern.special_markers.at("reasoning_end_marker").empty()) {
                    return p.reasoning_block(
                               p.reasoning(p.until(local_pattern.special_markers.at("reasoning_end_marker"))) +
                               p.literal(local_pattern.special_markers.at("reasoning_end_marker"))) +
                           content;
                }
                return content;
            });
            data.format = COMMON_CHAT_FORMAT_PEG_SIMPLE;
            LOG_DBG("Generated CONTENT_ONLY parser successfully\n");
        }

        data.parser = arena.save();

        // Determine trigger word for lazy grammar
        std::string trigger_word;
        if (!local_pattern.special_markers.at("tool_call_start_marker").empty()) {
            trigger_word = local_pattern.special_markers.at("tool_call_start_marker");
        } else if (!local_pattern.special_markers.at("function_opener").empty()) {
            trigger_word = local_pattern.special_markers.at("function_opener");
        }

        // Build grammar for tool calls based on discovered patterns
        data.grammar_lazy = inputs.tools.is_array() && !inputs.tools.empty();

        // If thinking forced open, we must constrain from the start (reasoning content)
        if (data.thinking_forced_open) {
            data.grammar_lazy = false;
        }

        if (data.grammar_lazy && !trigger_word.empty()) {
            data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_word });
        }

        // Always build grammar (full or lazy)
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            // Add tool definitions if any
            if (inputs.tools.is_array()) {
                for (const auto & tool : inputs.tools) {
                    if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
                        continue;
                    }
                    const auto & function = tool.at("function");
                    if (function.contains("parameters")) {
                        auto params = function.at("parameters");
                        builder.resolve_refs(params);
                    }
                }
            }
            arena.build_grammar(builder, data.grammar_lazy);
        });

        // Set preserved tokens - include all discovered markers plus trigger words
        std::vector<std::string> preserved;

        // Add trigger word if present
        if (!trigger_word.empty()) {
            preserved.push_back(trigger_word);
        }

        // Add all non-empty special markers
        for (const auto & [key, value] : local_pattern.special_markers) {
            if (!value.empty()) {
                // Avoid duplicates
                if (std::find(preserved.begin(), preserved.end(), value) == preserved.end()) {
                    preserved.push_back(value);
                }
            }
        }

        // Add any from the original pattern
        for (const auto & token : local_pattern.preserved_tokens) {
            if (!token.empty() && std::find(preserved.begin(), preserved.end(), token) == preserved.end()) {
                preserved.push_back(token);
            }
        }

        data.preserved_tokens = preserved;

        // data.prompt was already set

        LOG_DBG("=== PEG PARSER GENERATION COMPLETED ===\n");

    } catch (const std::exception & e) {
        LOG_DBG("Automatic parser generation failed: %s\n", e.what());
        throw;
    }

    return data;
}

common_peg_arena UniversalPEGGenerator::build_native_parser(const TemplatePattern &         pattern,
                                                            const minja::chat_template &    tmpl,
                                                            const struct templates_params & inputs,
                                                            bool                            thinking_forced_open) {
    GGML_UNUSED(tmpl);

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    auto parser = build_chat_peg_native_parser([&](common_chat_peg_native_builder & p) {
        auto reasoning_start = pattern.special_markers.at("reasoning_start_marker");
        auto reasoning_end   = pattern.special_markers.at("reasoning_end_marker");
        auto tool_call_start = pattern.special_markers.at("tool_call_start_marker");
        auto tool_call_end   = pattern.special_markers.at("tool_call_end_marker");
        auto content_start   = pattern.special_markers.at("content_start_marker");
        auto content_end     = pattern.special_markers.at("content_end_marker");

        // Reasoning/thinking block handling
        auto reasoning = p.eps();

        if (!reasoning_start.empty() && !reasoning_end.empty()) {
            if (inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE) {
                // Parse thinking into reasoning_content
                if (thinking_forced_open) {
                    LOG_DBG("Building mandatory reasoning block with end marker '%s'\n", reasoning_start.c_str());
                    reasoning = p.reasoning(p.until(reasoning_end)) + reasoning_end;
                } else {
                    LOG_DBG("Building optional reasoning block with start '%s' and end '%s'\n", reasoning_start.c_str(),
                            reasoning_end.c_str());
                    reasoning = p.optional(reasoning_start + p.reasoning(p.until(reasoning_end)) + reasoning_end);
                }
            }
            // When reasoning_format == NONE, reasoning stays as eps() - no special handling for thinking markers
        }

        // Build content parser based on available markers
        // Content markers should be stripped when present, but content without markers should also work
        // When reasoning_format=NONE, reasoning markers are in the input and should be preserved
        auto build_content = [&]() {
            if (!content_start.empty() && !content_end.empty()) {
                // Have content markers - handle both cases:
                // 1. Content starts with markers: strip them
                // 2. Content comes after other markers (e.g., reasoning markers): skip to start of content markers, then strip them
                LOG_DBG("Using optional content markers for content parsing\n");

                // When reasoning_format=NONE, reasoning markers are in the input.
                // We need to match: [any prefix] + content_start + content + content_end
                // The [any prefix] captures reasoning markers, which should be preserved in content
                auto match_content_with_markers = [&]() {
                    LOG_DBG(
                        "match_content_with_markers: reasoning_start='%s', reasoning_end='%s', content_start='%s', "
                        "reasoning_format=%d\n",
                        reasoning_start.c_str(), reasoning_end.c_str(), content_start.c_str(),
                        (int) inputs.reasoning_format);
                    if (!reasoning_start.empty() && !reasoning_end.empty() &&
                        inputs.reasoning_format == COMMON_REASONING_FORMAT_NONE) {
                        // When reasoning_format=NONE, preserve reasoning markers in content
                        // Match: (anything up to content_start as content) + content_start + (content) + content_end
                        // Both the prefix (including reasoning markers) and the inner content are tagged as content
                        LOG_DBG(
                            "preserve reasoning markers in content: using p.content(p.until('%s')) + literal + "
                            "p.content(until('%s')) + literal\n",
                            content_start.c_str(), content_end.c_str());
                        return p.content(p.until(content_start)) + content_start + p.content(p.until(content_end)) +
                               content_end;
                    }
                    // Normal case: just match content_start + content + content_end
                    return content_start + p.content(p.until(content_end)) + content_end;
                };

                // Fallback for content without markers (or when markers come after reasoning)
                auto build_fallback_content = [&]() {
                    if (!reasoning_start.empty() && !reasoning_end.empty() &&
                        inputs.reasoning_format == COMMON_REASONING_FORMAT_NONE) {
                        // When reasoning_format=NONE, look for reasoning markers then capture rest
                        // Skip optional reasoning block: (reasoning_start + reasoning + reasoning_end)? + content
                        return p.optional(reasoning_start + p.until(reasoning_end) + reasoning_end) +
                               p.content(p.rest());
                    }
                    // Just capture rest
                    return p.content(p.rest());
                };

                // Try: (content with markers) | (reasoning markers + content) | (just content)
                return p.choice({ match_content_with_markers(), build_fallback_content(), p.content(p.rest()) });
            }
            // No content markers - capture rest as content
            return p.content(p.rest());
        };

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (pattern.format == TemplatePattern::JSON_NATIVE) {
                bool force_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

                // Get custom field names or use defaults
                std::string name_field = "name";
                std::string args_field = "arguments";
                std::string id_field;

                auto name_it = pattern.special_markers.find("tool_name_field");
                auto args_it = pattern.special_markers.find("tool_args_field");
                auto id_it   = pattern.special_markers.find("tool_id_field");

                if (name_it != pattern.special_markers.end() && !name_it->second.empty()) {
                    name_field = name_it->second;
                }
                if (args_it != pattern.special_markers.end() && !args_it->second.empty()) {
                    args_field = args_it->second;
                }
                if (id_it != pattern.special_markers.end() && !id_it->second.empty()) {
                    id_field = id_it->second;
                }

                auto tool_call_start_it = pattern.special_markers.find("tool_call_start_marker");
                if (tool_call_start_it != pattern.special_markers.end() && !tool_call_start_it->second.empty()) {
                    tool_call_start = tool_call_start_it->second;

                    // If we switched to start_marker (which doesn't consume the opening brace),
                    // we need to find the corresponding list closer ']' in the end marker
                    // and strip any structural closing braces '}' that belonged to the deep opener.
                    if (!tool_call_end.empty()) {
                        size_t bracket_pos = tool_call_end.find(']');
                        if (bracket_pos != std::string::npos) {
                            tool_call_end = tool_call_end.substr(bracket_pos);

                            // Also truncate after the closing brace '}' if present
                            size_t brace_pos = tool_call_end.find('}');
                            if (brace_pos != std::string::npos) {
                                tool_call_end = tool_call_end.substr(0, brace_pos + 1);
                            }
                        }
                    }
                }

                LOG_DBG("Using JSON field names: name='%s', args='%s', id='%s'\n", name_field.c_str(),
                        args_field.c_str(), id_field.c_str());

                auto tool_calls =
                    p.standard_json_tools(tool_call_start, tool_call_end, inputs.tools, inputs.parallel_tool_calls,
                                          force_calls, name_field, args_field, id_field);

                auto content_before_tools = tool_call_start.empty() ? p.eps() : p.content(p.until(tool_call_start));

                return p.sequence({ reasoning, content_before_tools, p.space(), tool_calls, p.space(), p.end() });
            }
            throw std::runtime_error("Native parser requires JSON tool format");
        }

        auto content_parser = build_content();

        // Only add space if reasoning is not empty (has actual meaning)
        // When reasoning = eps() (reasoning_format=NONE), there's no reason to consume space
        if (reasoning != p.eps()) {
            content_parser = p.space() + content_parser;
        }

        return p.sequence({ reasoning, content_parser, p.end() });
    });

    return parser;
}

common_peg_arena UniversalPEGGenerator::build_constructed_parser(const TemplatePattern &         pattern,
                                                                 const minja::chat_template &    tmpl,
                                                                 const struct templates_params & inputs,
                                                                 bool                            thinking_forced_open) {
    (void) tmpl;  // Suppress unused parameter warning

    auto parser = build_chat_peg_constructed_parser([&](common_chat_peg_constructed_builder & p) {
        auto reasoning_start = pattern.special_markers.at("reasoning_start_marker");
        auto reasoning_end   = pattern.special_markers.at("reasoning_end_marker");
        auto tool_call_start = pattern.special_markers.at("tool_call_start_marker");
        auto content_start   = pattern.special_markers.at("content_start_marker");
        auto content_end     = pattern.special_markers.at("content_end_marker");

        LOG_DBG(
            "build_constructed_parser: reasoning_start='%s', reasoning_end='%s', content_start='%s', "
            "content_end='%s'\n",
            reasoning_start.c_str(), reasoning_end.c_str(), content_start.c_str(), content_end.c_str());

        // Reasoning/thinking block handling
        auto reasoning = p.eps();

        if (!reasoning_start.empty() && !reasoning_end.empty()) {
            if (inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE) {
                // Parse thinking into reasoning_content
                if (thinking_forced_open) {
                    LOG_DBG("Building mandatory reasoning block for constructed parser\n");
                    reasoning = p.rule(
                        "reasoning", p.reasoning_block(p.reasoning(p.until(reasoning_end)) + p.literal(reasoning_end)));
                } else {
                    LOG_DBG("Building optional reasoning block for constructed parser\n");
                    reasoning = p.optional(p.rule("reasoning", p.reasoning_block(p.literal(reasoning_start) +
                                                                                 p.reasoning(p.until(reasoning_end)) +
                                                                                 p.literal(reasoning_end))));
                }
            }
            // When reasoning_format == NONE, reasoning stays as eps() - no special handling for thinking markers
            // The content parser will capture everything including thinking markers
        }

        bool has_tools   = inputs.tools.is_array() && !inputs.tools.empty();
        bool force_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        auto tool_calls  = p.standard_constructed_tools(pattern.special_markers, inputs.tools,
                                                        inputs.parallel_tool_calls, force_calls);

        // Build content parser based on available markers
        // Content markers (like <|START_RESPONSE|>...<|END_RESPONSE|>) should be stripped when present
        // but content without markers should also be accepted
        // When reasoning_format=NONE, reasoning markers are in the input and should be preserved
        auto build_content = [&]() {
            if (!content_start.empty() && !content_end.empty()) {
                // Have content markers - handle both cases:
                // 1. Content starts with markers: strip them
                // 2. Content comes after other markers (e.g., reasoning markers): skip to start of content markers, then strip them
                LOG_DBG("Using optional content markers for content parsing\n");

                // When reasoning_format=NONE, reasoning markers are in the input.
                // We need to match: [any prefix] + content_start + content + content_end
                // The [any prefix] captures reasoning markers, which should be preserved in content
                auto match_content_with_markers = [&]() {
                    if (!reasoning_start.empty() && !reasoning_end.empty() &&
                        inputs.reasoning_format == COMMON_REASONING_FORMAT_NONE) {
                        // When reasoning_format=NONE, preserve reasoning markers in content
                        // Match: (anything up to content_start as content) + content_start + content + content_end
                        // Both the prefix (including reasoning markers) and the inner content are tagged as content
                        LOG_DBG("preserve reasoning markers in content with content markers\n");
                        return p.content(p.until(content_start)) + p.literal(content_start) +
                               p.rule("content", p.content(p.until(content_end))) + p.literal(content_end);
                    }
                    // Normal case: just match content_start + content + content_end
                    LOG_DBG("normal content markers without preserving reasoning markers\n");
                    return p.literal(content_start) + p.rule("content", p.content(p.until(content_end))) +
                           p.literal(content_end);
                };

                // Try: (content with markers) | (content without markers)
                return p.choice({ match_content_with_markers(), p.rule("content", p.content(p.rest())) });
            }
            // No content markers - capture rest as content
            return p.rule("content", p.content(p.rest()));
        };

        if (has_tools && !tool_call_start.empty()) {
            // With tools: reasoning -> content (optionally wrapped) -> tool calls -> trailing content
            return reasoning + p.space() + p.tag_with_safe_content("content", tool_call_start, tool_calls) +
                   p.optional(p.rule("content", p.content(p.rest())));
        }

        // No tools - just reasoning (if any) followed by content
        return reasoning + p.space() + build_content();
    });

    return parser;
}
