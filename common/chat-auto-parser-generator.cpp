#include "chat-auto-parser.h"
#include "chat-diff-analyzer.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "json-schema-to-grammar.h"
#include "nlohmann/json.hpp"
#include <string>


using json = nlohmann::ordered_json;

// Helper to iterate over tools/functions
static void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            continue;
        }
        fn(tool);
    }
}

common_chat_params universal_peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                            const struct templates_params & inputs) {
    // Run differential analysis to extract template structure
    auto analysis = differential_analyzer::analyze(tmpl);
    return generate_parser(tmpl, inputs, analysis);
}

common_chat_params universal_peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                            const struct templates_params & inputs,
                                                            const diff_analysis_result &    analysis) {
    // Check for thinking forced open
    bool thinking_forced_open = (analysis.reasoning == reasoning_mode::FORCED_OPEN);
    bool thinking_forced_closed = (analysis.reasoning == reasoning_mode::FORCED_CLOSED);

    // Build the parser using the analysis results
    auto parser = build_parser(analysis, inputs, thinking_forced_open, thinking_forced_closed);

    // Create the result structure
    common_chat_params data;
    data.prompt = common_chat_template_direct_apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens = analysis.preserved_tokens;
    data.parser = parser.save();

    // Build grammar if tools are present
    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    bool include_grammar = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Set grammar triggers based on tool section markers (fall back to per-call markers)
        std::string trigger_marker = !analysis.markers.tool_section_start.empty()
            ? analysis.markers.tool_section_start
            : analysis.markers.per_call_start;
        if (!trigger_marker.empty()) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_marker }
            };
        }
    }

    return data;
}

common_peg_arena universal_peg_generator::build_parser(const diff_analysis_result &    analysis,
                                                        const struct templates_params & inputs,
                                                        bool                            thinking_forced_open,
                                                        bool                            thinking_forced_closed) {
    return build_chat_peg_unified_parser([&](common_chat_peg_unified_builder & p) {
        p.set_allow_python_dict_format(true);
        const auto & m = analysis.markers;

        common_peg_parser reasoning = p.eps();
        bool extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
        bool enable_thinking = inputs.enable_thinking;

        if (extract_reasoning && enable_thinking && analysis.reasoning != reasoning_mode::NONE) {
            if (thinking_forced_open || thinking_forced_closed) {
                // Thinking is forced open OR forced closed with enable_thinking=true
                // In both cases, expect only the closing tag (opening was in template)
                reasoning = p.reasoning(p.until(m.reasoning_end)) + m.reasoning_end;
            } else if (analysis.reasoning == reasoning_mode::TAG_BASED ||
                       analysis.reasoning == reasoning_mode::TOOLS_ONLY) {
                // Standard tag-based reasoning OR tools-only mode (reasoning appears with tools)
                // Both use the same tag-based pattern if markers are available
                if (!m.reasoning_start.empty() && !m.reasoning_end.empty()) {
                    reasoning = p.optional(m.reasoning_start + p.reasoning(p.until(m.reasoning_end)) + m.reasoning_end);
                }
            } else if (analysis.reasoning == reasoning_mode::DELIMITER) {
                reasoning = p.optional(p.reasoning(p.until(m.reasoning_end)) + m.reasoning_end);
            }
        }

        bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
        bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();

        if (has_response_format) {
            return reasoning + p.space() + p.content(p.schema(p.json(), "response-format", inputs.json_schema)) + p.end();
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && analysis.supports_tools) {
            return build_tool_parser(p, analysis, inputs, reasoning);
        }

        if (analysis.content == content_mode::ALWAYS_WRAPPED &&
            !m.content_start.empty() && !m.content_end.empty()) {

            bool extracting_reasoning = extract_reasoning && enable_thinking && analysis.reasoning != reasoning_mode::NONE;

            if (extracting_reasoning) {
                return reasoning + m.content_start + p.content(p.until(m.content_end)) + m.content_end + p.end();
            }
            return p.content(p.until(m.content_start)) + m.content_start + p.content(p.until(m.content_end)) + m.content_end + p.end();
        }
        return reasoning + p.content(p.rest()) + p.end();
    });
}

common_peg_parser universal_peg_generator::build_tool_parser(
        common_chat_peg_unified_builder & p,
        const diff_analysis_result & analysis,
        const templates_params & inputs,
        const common_peg_parser & reasoning) {

    const auto & m = analysis.markers;

    // Build tool choice parser based on format
    common_peg_parser tool_choice = p.choice();

    if (analysis.tools == tool_format::JSON_NATIVE) {
        // Pure JSON format: use standard_json_tools helper
        // Build effective field names with dot notation if function_field is set
        std::string name_field = analysis.name_field;
        std::string args_field = analysis.args_field;

        if (!analysis.function_field.empty() &&
            analysis.function_field != "function" &&
            name_field.find('.') == std::string::npos) {
            name_field = analysis.function_field + "." + name_field;
            args_field = analysis.function_field + "." + args_field;
        }

        auto tools_parser = p.standard_json_tools(
            m.tool_section_start,
            m.tool_section_end,
            inputs.tools,
            inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED,
            name_field,
            args_field,
            analysis.tools_array_wrapped,
            analysis.fun_name_is_key,
            analysis.id_field,
            analysis.gen_id_field,
            analysis.parameter_order
        );

        // Handle content wrappers if present
        if (analysis.content == content_mode::ALWAYS_WRAPPED &&
            !m.content_start.empty() && !m.content_end.empty()) {
            auto wrapped_content = p.optional(m.content_start + p.content(p.until(m.content_end)) + m.content_end);
            return reasoning + wrapped_content + tools_parser + p.end();
        }

        auto content_before_tools = m.tool_section_start.empty() ? p.eps() : p.until(m.tool_section_start);
        return reasoning + p.optional(p.content(content_before_tools)) + tools_parser + p.end();
    }

    if (analysis.tools == tool_format::TAG_WITH_JSON) {
        // Tag-based with JSON args: <function=name>{args}</function>
        // With optional call_id: <function=name>[CALL_ID]id[ARGS]{args}</function>
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto & schema   = function.at("parameters");

            // Build call_id parser based on position (if supported)
            common_peg_parser call_id_section = p.eps();
            if (analysis.call_id_pos == call_id_position::BETWEEN_FUNC_AND_ARGS &&
                !m.call_id_prefix.empty() && !m.call_id_suffix.empty()) {
                // Optional call_id followed by required call_id_suffix (which is also args_start)
                // Format: optional([CALL_ID] + call_id_value) + [ARGS]
                call_id_section = p.optional(m.call_id_prefix + p.tool_id(p.until(m.call_id_suffix))) + m.call_id_suffix;
            }

            auto func_parser = p.tool_open(m.func_name_prefix + p.tool_name(p.literal(name)) + m.func_name_suffix) +
                               call_id_section +
                               p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema));

            if (!m.func_close.empty()) {
                func_parser = func_parser + m.func_close;
            }

            tool_choice |= p.rule("tool-" + name, func_parser);
        });

        auto require_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        common_peg_parser tool_calls = p.eps();

        if (!m.per_call_start.empty()) {
            // Per-call wrapping: each call individually wrapped
            auto wrapped_call = m.per_call_start + tool_choice + m.per_call_end;
            if (inputs.parallel_tool_calls) {
                tool_calls = p.trigger_rule("tool-call",
                    wrapped_call + p.zero_or_more(p.space() + wrapped_call));
            } else {
                tool_calls = p.trigger_rule("tool-call", wrapped_call);
            }
            if (!m.tool_section_start.empty()) {
                tool_calls = p.trigger_rule("tool-calls", p.literal(m.tool_section_start) + p.space() +
                    tool_calls + p.space() + (m.tool_section_end.empty() ? p.end() : p.literal(m.tool_section_end)));
            }
        } else {
            std::string separator = m.call_separator;
            if (separator.empty()) {
                separator = ", ";  // Default
            }

            if (inputs.parallel_tool_calls) {
                tool_calls = p.trigger_rule("tool-call",
                    m.tool_section_start + tool_choice + p.zero_or_more(separator + tool_choice) + m.tool_section_end);
            } else {
                tool_calls = p.trigger_rule("tool-call",
                    m.tool_section_start + tool_choice + m.tool_section_end);
            }
        }

        if (!require_calls) {
            tool_calls = p.optional(tool_calls);
        }

        std::string trigger_marker = !m.tool_section_start.empty() ? m.tool_section_start : m.per_call_start;
        auto content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
        return reasoning + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
    }

    if (analysis.tools == tool_format::TAG_WITH_TAGGED) {
        // Tag-based with tagged args: <function=name><param=key>value</param></function>
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto & params   = function.at("parameters");

            if (!params.contains("properties") || !params.at("properties").is_object()) {
                return;
            }

            const auto & properties = params.at("properties");
            std::set<std::string> required;
            if (params.contains("required") && params.at("required").is_array()) {
                params.at("required").get_to(required);
            }

            // Build parser for each argument
            std::vector<common_peg_parser> arg_parsers;
            for (const auto & [param_name, param_schema] : properties.items()) {
                bool is_required = required.find(param_name) != required.end();
                auto type = param_schema.value("type", "object");

                auto arg = p.tool_arg(
                    p.tool_arg_open(m.arg_name_prefix + p.tool_arg_name(p.literal(param_name)) + m.arg_name_suffix) + m.arg_value_prefix +
                    (type == "string" ?
                        p.tool_arg_string_value(p.schema(p.until(m.arg_value_suffix),
                            "tool-" + name + "-arg-" + param_name + "-schema", param_schema, true)) :
                        p.tool_arg_json_value(p.schema(p.json(),
                            "tool-" + name + "-arg-" + param_name + "-schema", param_schema)) + p.space()) +
                    p.tool_arg_close(p.literal(m.arg_value_suffix))
                );

                if (is_required) {
                    arg_parsers.push_back(p.rule("tool-" + name + "-arg-" + param_name, arg));
                } else {
                    arg_parsers.push_back(p.optional(p.rule("tool-" + name + "-arg-" + param_name, arg)));
                }
            }

            // Build arg sequence with space() between consecutive args
            common_peg_parser args_seq = p.eps();
            for (size_t i = 0; i < arg_parsers.size(); i++) {
                if (i > 0) {
                    args_seq = args_seq + p.space();
                }
                args_seq = args_seq + arg_parsers[i];
            }

            // Build call_id parser based on position (if supported)
            common_peg_parser call_id_section = p.eps();
            if (analysis.call_id_pos == call_id_position::BETWEEN_FUNC_AND_ARGS &&
                !m.call_id_prefix.empty() && !m.call_id_suffix.empty()) {
                // Optional call_id followed by required call_id_suffix
                call_id_section = p.optional(m.call_id_prefix + p.tool_id(p.until(m.call_id_suffix))) + m.call_id_suffix;
            }

            auto func_parser = p.tool_open(m.func_name_prefix + p.tool_name(p.literal(name)) + m.func_name_suffix) +
                               call_id_section +
                               p.space() + args_seq;

            if (!m.func_close.empty()) {
                func_parser = func_parser + p.space() + p.tool_close(p.literal(m.func_close));
            } else {
                func_parser = func_parser + p.tool_close(p.space()); // force this to process tool closing callbacks in mapper
            }

            tool_choice |= p.rule("tool-" + name, func_parser);
        });

        auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        common_peg_parser tool_calls = p.eps();

        if (!m.per_call_start.empty()) {
            // Per-call wrapping: each call individually wrapped (e.g., <tool_call>...</tool_call>)
            auto wrapped_call = m.per_call_start + p.space() + tool_choice + p.space() + m.per_call_end;
            if (inputs.parallel_tool_calls) {
                tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call));
            } else {
                tool_calls = p.trigger_rule("tool-call", wrapped_call);
            }
            if (!m.tool_section_start.empty()) {
                tool_calls = p.trigger_rule("tool-calls", p.literal(m.tool_section_start) + p.space() +
                    tool_calls + p.space() + (m.tool_section_end.empty() ? p.end() : p.literal(m.tool_section_end)));
            }
        } else {
            std::string separator = m.call_separator;
            if (separator.empty()) {
                separator = ", ";  // Default
            }

            if (inputs.parallel_tool_calls) {
                tool_calls = p.trigger_rule("tool-call",
                    m.tool_section_start + p.space() + tool_choice + p.zero_or_more(separator + tool_choice) + p.space() + m.tool_section_end);
            } else {
                tool_calls = p.trigger_rule("tool-call",
                    m.tool_section_start + p.space() + tool_choice + p.space() + m.tool_section_end);
            }
        }

        if (!require_tools) {
            tool_calls = p.optional(tool_calls);
        }

        std::string trigger_marker = !m.tool_section_start.empty() ? m.tool_section_start : m.per_call_start;
        auto content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
        return reasoning + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
    }

    GGML_ABORT("Unable to create tool parser");
}
