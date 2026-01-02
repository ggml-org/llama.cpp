#include "chat-auto-parser.h"
#include "chat-auto-parser-helpers.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "chat-peg-parser.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>
#include <optional>
#include <stdexcept>

using json = nlohmann::ordered_json;

// ============================================================================
// Unified PEG Parser Generator
// ============================================================================

common_chat_params UniversalPEGGenerator::generate_parser(
    const TemplateAnalysisResult &  analysis,
    const minja::chat_template &    tmpl,
    const struct templates_params & inputs) {

    common_chat_params data;

    try {
        LOG_DBG("=== GENERATING UNIFIED PEG PARSER ===\n");
        LOG_DBG("Content structure:\n");
        LOG_DBG("  reasoning_mode: %d\n", static_cast<int>(analysis.content.reasoning_mode));
        LOG_DBG("  reasoning_start: '%s'\n", analysis.content.reasoning_start.c_str());
        LOG_DBG("  reasoning_end: '%s'\n", analysis.content.reasoning_end.c_str());
        LOG_DBG("  content_mode: %d\n", static_cast<int>(analysis.content.content_mode));
        LOG_DBG("  content_start: '%s'\n", analysis.content.content_start.c_str());
        LOG_DBG("  content_end: '%s'\n", analysis.content.content_end.c_str());
        LOG_DBG("Tool structure:\n");
        LOG_DBG("  supports_tools: %s\n", analysis.tools.supports_tools ? "true" : "false");
        LOG_DBG("  function_format: %d\n", static_cast<int>(analysis.tools.function_format));
        LOG_DBG("  argument_format: %d\n", static_cast<int>(analysis.tools.argument_format));
        LOG_DBG("  tool_section_start: '%s'\n", analysis.tools.tool_section_start.c_str());
        LOG_DBG("  tool_section_end: '%s'\n", analysis.tools.tool_section_end.c_str());

        // Patch messages if template requires non-null content
        // Some templates (e.g., iquest) render null as "None" when concatenating strings
        std::optional<json> messages_override;
        if (analysis.tools.requires_nonnull_content && !inputs.messages.empty()) {
            LOG_DBG("Patching null content to empty string (template requires non-null content)\n");
            json patched_messages = inputs.messages;
            for (auto & msg : patched_messages) {
                if (msg.contains("content") && msg["content"].is_null()) {
                    msg["content"] = "";
                }
            }
            messages_override = patched_messages;
        }

        if (inputs.messages.empty()) {
            // Some templates don't handle empty messages well - always leave something in
            json message = { { { "role", "user" }, { "content", "Hello" } } };
            messages_override.emplace(message);
        }

        // Calculate prompt first to detect forced thinking
        data.prompt = apply_template(tmpl, inputs, messages_override);

        // Determine if thinking is forced open based on prompt ending
        bool thinking_forced_open = false;
        if (analysis.content.reasoning_mode == ContentStructure::REASONING_FORCED_OPEN) {
            if (inputs.enable_thinking) {
                thinking_forced_open = true;
                LOG_DBG("Thinking forced open based on template analysis\n");
            } else {
                // Template ends with reasoning start marker but thinking is disabled
                // Append the end marker to close it
                data.prompt += analysis.content.reasoning_end;
                LOG_DBG("Appended reasoning end marker since thinking is disabled\n");
            }
        }
        data.thinking_forced_open = thinking_forced_open;

        // Build the unified parser
        auto arena = build_parser(analysis, tmpl, inputs, thinking_forced_open);
        data.parser = arena.save();

        // Determine format
        bool has_tools = inputs.tools.is_array() && !inputs.tools.empty() &&
                        inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

        if (has_tools && analysis.tools.supports_tools) {
            // Unified format that handles both JSON and tagged tool calls
            data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
            LOG_DBG("Generated unified parser with tool support (format: PEG_NATIVE)\n");
        } else if (analysis.content.reasoning_mode != ContentStructure::REASONING_NONE) {
            // Reasoning markers detected - use PEG parser to handle thinking blocks
            data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
            LOG_DBG("Generated unified parser for reasoning handling (format: PEG_NATIVE)\n");
        } else if (analysis.content.content_mode != ContentStructure::CONTENT_PLAIN) {
            // Content markers detected - use PEG parser to strip them even without tools
            data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
            LOG_DBG("Generated unified parser for content marker stripping (format: PEG_NATIVE)\n");
        } else if (analysis.tools.function_format == ToolCallStructure::FUNC_TAG_WITH_NAME) {
            // Tag-with-name format (e.g., func_name\n{args} for Functionary)
            // Need PEG parser to handle function name parsing
            data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
            LOG_DBG("Generated unified parser for tag-with-name format (format: PEG_NATIVE)\n");
        } else if (analysis.tools.function_format == ToolCallStructure::FUNC_BRACKET_TAG) {
            // Bracket-tag format (e.g., [TOOL_CALLS]name[CALL_ID]id[ARGS]{...} for Mistral Small 3.2)
            // Need PEG parser to handle bracket tag parsing
            data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
            LOG_DBG("Generated unified parser for bracket-tag format (format: PEG_NATIVE)\n");
        } else if (analysis.tools.function_format == ToolCallStructure::FUNC_PREFIXED_INDEXED) {
            // Prefixed-indexed format (e.g., Kimi-K2)
            // Need PEG parser to handle namespace and indexed format
            data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
            LOG_DBG("Generated unified parser for prefixed-indexed format (format: PEG_NATIVE)\n");
        } else {
            data.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
            LOG_DBG("Generated unified parser without tools or content markers (format: CONTENT_ONLY)\n");
        }

        // Determine trigger word for lazy grammar
        std::string trigger_word;
        if (!analysis.tools.tool_section_start.empty()) {
            trigger_word = analysis.tools.tool_section_start;
        } else if (analysis.tools.function_format == ToolCallStructure::FUNC_TAG_WITH_NAME) {
            trigger_word = analysis.tools.function_prefix;
        } else if (analysis.tools.function_format == ToolCallStructure::FUNC_BRACKET_TAG ||
                   analysis.tools.function_format == ToolCallStructure::FUNC_PREFIXED_INDEXED) {
            // For formats with per-call markers, use per_call_start as trigger
            trigger_word = analysis.tools.per_call_start;
        }

        // Build grammar for tool calls
        data.grammar_lazy = analysis.tools.supports_tools && has_tools;

        // If thinking forced open, we must constrain from the start
        if (data.thinking_forced_open) {
            data.grammar_lazy = false;
        }

        // For FUNC_TAG_WITH_NAME with empty prefix (Functionary), disable lazy grammar
        // since there's no clear trigger word - constrain from the start
        if (analysis.tools.function_format == ToolCallStructure::FUNC_TAG_WITH_NAME &&
            analysis.tools.function_prefix.empty()) {
            data.grammar_lazy = false;
        }

        if (data.grammar_lazy) {
            if (!trigger_word.empty()) {
                data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_word });
            } else if (analysis.tools.function_format == ToolCallStructure::FUNC_JSON_OBJECT &&
                       analysis.tools.supports_tools) {
                // Raw JSON format without markers (e.g., Llama 3.1) - use regex trigger
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                    "(\\{\\s*(?:\"type\"\\s*:\\s*\"function\"\\s*,\\s*)?\"name\"\\s*:\\s*\")[\\s\\S]*",
                });
            }
        }

        // Build grammar
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
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

        // Set preserved tokens from analysis
        data.preserved_tokens = analysis.preserved_tokens;

        LOG_DBG("=== UNIFIED PEG PARSER GENERATION COMPLETED ===\n");

    } catch (const std::exception & e) {
        LOG_DBG("Unified parser generation failed: %s\n", e.what());
        throw;
    }

    return data;
}

common_peg_arena UniversalPEGGenerator::build_parser(
    const TemplateAnalysisResult &  analysis,
    const minja::chat_template &    tmpl,
    const struct templates_params & inputs,
    bool                            thinking_forced_open) {

    GGML_UNUSED(tmpl);

    auto parser = build_chat_peg_unified_parser([&](common_chat_peg_unified_builder & p) {
        // Build reasoning block using ContentStructure
        auto reasoning = p.build_reasoning_block(analysis.content, inputs.reasoning_format, thinking_forced_open);

        // Build content block using ContentStructure
        auto content = p.build_content_block(analysis.content, inputs.reasoning_format);

        // Build tool section using ToolCallStructure (if applicable)
        bool has_tools = inputs.tools.is_array() && !inputs.tools.empty() &&
                        inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

        if (has_tools && analysis.tools.supports_tools) {
            bool force_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            auto tool_section = p.build_tool_section(analysis.tools, inputs.tools,
                                                     inputs.parallel_tool_calls, force_calls);

            // Compose: reasoning -> content before tools -> tool_section -> trailing content
            // Check if this is a DeepSeek R1 style template with separate section and per-call markers
            // These templates go directly from reasoning to tool calls (no content before)
            bool has_per_call_markers_in_prefix =
                analysis.tools.function_format == ToolCallStructure::FUNC_TAG_WITH_NAME &&
                !analysis.tools.function_prefix.empty() &&
                analysis.tools.function_prefix.find("call") != std::string::npos &&
                (analysis.tools.function_prefix.find("begin") != std::string::npos ||
                 analysis.tools.function_prefix.find("start") != std::string::npos);

            // When thinking is forced open, the reasoning block expects </think>.
            // For tool-only messages (no thinking content), the model may output tools directly
            // without the </think> tag, so we need to make reasoning optional in that case.
            // But if reasoning_format is NONE, the reasoning block is already eps() - don't wrap it
            // in optional() as that would generate invalid grammar.
            auto reasoning_for_tools = (thinking_forced_open && inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE)
                                     ? p.optional(reasoning)
                                     : reasoning;

            if (has_per_call_markers_in_prefix && !analysis.tools.tool_section_start.empty()) {
                // DeepSeek R1 style: section markers wrap all calls, per-call markers in function_prefix
                // No content before tools - go directly from reasoning to tool section
                return p.sequence({ reasoning_for_tools, p.space(), tool_section, p.space(),
                                   p.optional(p.content(p.rest())), p.end() });
            } else if (!analysis.tools.tool_section_start.empty()) {
                // With section markers: look for start marker to delimit content
                auto content_before_tools = p.content(p.until(analysis.tools.tool_section_start));
                return p.sequence({ reasoning_for_tools, p.space(), content_before_tools, p.space(),
                                   tool_section, p.space(), p.optional(p.content(p.rest())), p.end() });
            } else if (analysis.tools.function_format == ToolCallStructure::FUNC_TAG_WITH_NAME &&
                       !analysis.tools.function_prefix.empty()) {
                // Tag-with-name format (e.g., >>>func_name): content stops at function prefix
                auto content_before_tools = p.content(p.until(analysis.tools.function_prefix));
                return p.sequence({ reasoning_for_tools, p.space(), content_before_tools, p.space(), tool_section, p.end() });
            } else if (analysis.tools.function_format == ToolCallStructure::FUNC_TAG_WITH_NAME) {
                // Functionary-style format: tool call starts immediately (e.g., func_name\n{args})
                // No content before tools in this format - the entire output is the tool call
                return p.sequence({ reasoning_for_tools, p.space(), tool_section, p.end() });
            } else {
                // No section markers (raw JSON format): content must stop at JSON object start
                // Tool calls start with "{", so use that as a delimiter
                auto content_before_tools = p.content(p.until("{"));
                return p.sequence({ reasoning_for_tools, p.space(), content_before_tools, p.space(), tool_section, p.end() });
            }
        }

        // No tools - just reasoning (if any) followed by content
        return p.sequence({ reasoning, p.space(), content, p.end() });
    });

    return parser;
}
