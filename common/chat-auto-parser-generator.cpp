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
    bool thinking_forced_open = (analysis.reasoning.mode == reasoning_mode::FORCED_OPEN);
    bool thinking_forced_closed = (analysis.reasoning.mode == reasoning_mode::FORCED_CLOSED);

    // Build the parser using the analysis results
    auto parser = build_parser(analysis, inputs, thinking_forced_open, thinking_forced_closed);

    // Create the result structure
    common_chat_params data;
    data.prompt = common_chat_template_direct_apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens = analysis.preserved_tokens;
    data.parser = parser.save();

    // Build grammar if tools are present
    bool has_tools = analysis.tools.format.mode != tool_format::NONE && inputs.tools.is_array() && !inputs.tools.empty();
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
        std::string trigger_marker = !analysis.tools.format.section_start.empty()
            ? analysis.tools.format.section_start
            : analysis.tools.format.per_call_start;
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
        common_peg_parser reasoning = p.eps();
        bool extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
        bool enable_thinking = inputs.enable_thinking;

        if (extract_reasoning && enable_thinking && analysis.reasoning.mode != reasoning_mode::NONE) {
            if (thinking_forced_open || thinking_forced_closed) {
                // Thinking is forced open OR forced closed with enable_thinking=true
                // In both cases, expect only the closing tag (opening was in template)
                reasoning = p.reasoning(p.until(analysis.reasoning.end)) + analysis.reasoning.end;
            } else if (analysis.reasoning.mode == reasoning_mode::TAG_BASED ||
                       analysis.reasoning.mode == reasoning_mode::TOOLS_ONLY) {
                // Standard tag-based reasoning OR tools-only mode (reasoning appears with tools)
                // Both use the same tag-based pattern if markers are available
                if (!analysis.reasoning.start.empty() && !analysis.reasoning.end.empty()) {
                    reasoning = p.optional(analysis.reasoning.start + p.reasoning(p.until(analysis.reasoning.end)) + analysis.reasoning.end);
                }
            } else if (analysis.reasoning.mode == reasoning_mode::DELIMITER) {
                reasoning = p.optional(p.reasoning(p.until(analysis.reasoning.end)) + analysis.reasoning.end);
            }
        }

        bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
        bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();

        if (has_response_format) {
            return reasoning + p.space() + p.content(p.schema(p.json(), "response-format", inputs.json_schema)) + p.end();
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && analysis.jinja_caps.supports_tool_calls) {
            return build_tool_parser(p, analysis, inputs, reasoning);
        }

        if (analysis.content.mode == content_mode::ALWAYS_WRAPPED &&
            !analysis.content.start.empty() && !analysis.content.end.empty()) {

            bool extracting_reasoning = extract_reasoning && enable_thinking && analysis.reasoning.mode != reasoning_mode::NONE;

            if (extracting_reasoning) {
                return reasoning + analysis.content.start + p.content(p.until(analysis.content.end)) + analysis.content.end + p.end();
            }
            return p.content(p.until(analysis.content.start)) + analysis.content.start + p.content(p.until(analysis.content.end)) + analysis.content.end + p.end();
        }
        return reasoning + p.content(p.rest()) + p.end();
    });
}

common_peg_parser universal_peg_generator::build_tool_parser(
        common_chat_peg_unified_builder & p,
        const diff_analysis_result & analysis,
        const templates_params & inputs,
        const common_peg_parser & reasoning) {

    switch (analysis.tools.format.mode) {
        case tool_format::JSON_NATIVE:
            return build_tool_parser_json_native(p, analysis, inputs, reasoning);
        case tool_format::TAG_WITH_JSON:
            return build_tool_parser_tag_json(p, analysis, inputs, reasoning);
        case tool_format::TAG_WITH_TAGGED:
            return build_tool_parser_tag_tagged(p, analysis, inputs, reasoning);
        default:
            GGML_ABORT("Unable to create tool parser");
    }
}

common_peg_parser universal_peg_generator::build_tool_parser_json_native(
        common_chat_peg_unified_builder & p,
        const diff_analysis_result & analysis,
        const templates_params & inputs,
        const common_peg_parser & reasoning) {

    // Build effective field names with dot notation if function_field is set
    std::string name_field = analysis.tools.format.name_field;
    std::string args_field = analysis.tools.format.args_field;

    if (!analysis.tools.format.function_field.empty() &&
        analysis.tools.format.function_field != "function" &&
        name_field.find('.') == std::string::npos) {
        name_field = analysis.tools.format.function_field + "." + name_field;
        args_field = analysis.tools.format.function_field + "." + args_field;
    }

    auto tools_parser = p.standard_json_tools(
        analysis.tools.format.section_start,
        analysis.tools.format.section_end,
        inputs.tools,
        inputs.parallel_tool_calls,
        inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED,
        name_field,
        args_field,
        analysis.tools.format.tools_array_wrapped,
        analysis.tools.format.fun_name_is_key,
        analysis.tools.format.id_field,
        analysis.tools.format.gen_id_field,
        analysis.tools.format.parameter_order
    );

    // Handle content wrappers if present
    if (analysis.content.mode == content_mode::ALWAYS_WRAPPED &&
        !analysis.content.start.empty() && !analysis.content.end.empty()) {
        auto wrapped_content = p.optional(analysis.content.start + p.content(p.until(analysis.content.end)) + analysis.content.end);
        return reasoning + wrapped_content + tools_parser + p.end();
    }

    auto content_before_tools = analysis.tools.format.section_start.empty() ? p.eps() : p.until(analysis.tools.format.section_start);
    return reasoning + p.optional(p.content(content_before_tools)) + tools_parser + p.end();
}

common_peg_parser universal_peg_generator::build_tool_parser_tag_json(
        common_chat_peg_unified_builder & p,
        const diff_analysis_result & analysis,
        const templates_params & inputs,
        const common_peg_parser & reasoning) {

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & function = tool.at("function");
        std::string  name     = function.at("name");
        const auto & schema   = function.at("parameters");

        // Build call_id parser based on position (if supported)
        common_peg_parser call_id_section = p.eps();
        if (analysis.tools.call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS &&
            !analysis.tools.call_id.prefix.empty() && !analysis.tools.call_id.suffix.empty()) {
            call_id_section = p.optional(analysis.tools.call_id.prefix + p.tool_id(p.until(analysis.tools.call_id.suffix))) + analysis.tools.call_id.suffix;
        }

        auto func_parser = p.tool_open(analysis.tools.function.name_prefix + p.tool_name(p.literal(name)) + analysis.tools.function.name_suffix) +
                           call_id_section +
                           p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema));

        if (!analysis.tools.function.close.empty()) {
            func_parser = func_parser + analysis.tools.function.close;
        }

        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!analysis.tools.format.per_call_start.empty()) {
        auto wrapped_call = analysis.tools.format.per_call_start + tool_choice + analysis.tools.format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                wrapped_call + p.zero_or_more(p.space() + wrapped_call));
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call);
        }
        if (!analysis.tools.format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls", p.literal(analysis.tools.format.section_start) + p.space() +
                tool_calls + p.space() + (analysis.tools.format.section_end.empty() ? p.end() : p.literal(analysis.tools.format.section_end)));
        }
    } else {
        std::string separator = ", ";  // Default
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                analysis.tools.format.section_start + tool_choice + p.zero_or_more(separator + tool_choice) + analysis.tools.format.section_end);
        } else {
            tool_calls = p.trigger_rule("tool-call",
                analysis.tools.format.section_start + tool_choice + analysis.tools.format.section_end);
        }
    }

    if (!require_calls) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker = !analysis.tools.format.section_start.empty() ? analysis.tools.format.section_start : analysis.tools.format.per_call_start;
    auto content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return reasoning + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
}

common_peg_parser universal_peg_generator::build_tool_parser_tag_tagged(
        common_chat_peg_unified_builder & p,
        const diff_analysis_result & analysis,
        const templates_params & inputs,
        const common_peg_parser & reasoning) {

    common_peg_parser tool_choice = p.choice();

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
                p.tool_arg_open(analysis.tools.arguments.name_prefix + p.tool_arg_name(p.literal(param_name)) + analysis.tools.arguments.name_suffix) + analysis.tools.arguments.value_prefix +
                (type == "string" ?
                    p.tool_arg_string_value(p.schema(p.until(analysis.tools.arguments.value_suffix),
                        "tool-" + name + "-arg-" + param_name + "-schema", param_schema, true)) :
                    p.tool_arg_json_value(p.schema(p.json(),
                        "tool-" + name + "-arg-" + param_name + "-schema", param_schema)) + p.space()) +
                p.tool_arg_close(p.literal(analysis.tools.arguments.value_suffix))
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
        if (analysis.tools.call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS &&
            !analysis.tools.call_id.prefix.empty() && !analysis.tools.call_id.suffix.empty()) {
            call_id_section = p.optional(analysis.tools.call_id.prefix + p.tool_id(p.until(analysis.tools.call_id.suffix))) + analysis.tools.call_id.suffix;
        }

        auto func_parser = p.tool_open(analysis.tools.function.name_prefix + p.tool_name(p.literal(name)) + analysis.tools.function.name_suffix) +
                           call_id_section +
                           p.space() + args_seq;

        if (!analysis.tools.function.close.empty()) {
            func_parser = func_parser + p.space() + p.tool_close(p.literal(analysis.tools.function.close));
        } else if (!analysis.tools.format.per_call_end.empty()) {
            // When there's no func_close but there is a per_call_end marker, use peek() to ensure
            // we only emit tool_close when we can actually see the closing marker. This prevents
            // premature closing during partial parsing when we've seen e.g. "</" which could be
            // either "</tool_call>" (end) or "<arg_key>" prefix that failed to match.
            func_parser = func_parser + p.tool_close(p.peek(p.literal(analysis.tools.format.per_call_end)));
        } else {
            func_parser = func_parser + p.tool_close(p.space()); // force this to process tool closing callbacks in mapper
        }

        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!analysis.tools.format.per_call_start.empty()) {
        auto wrapped_call = analysis.tools.format.per_call_start + p.space() + tool_choice + p.space() + analysis.tools.format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call));
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call);
        }
        if (!analysis.tools.format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls", p.literal(analysis.tools.format.section_start) + p.space() +
                tool_calls + p.space() + (analysis.tools.format.section_end.empty() ? p.end() : p.literal(analysis.tools.format.section_end)));
        }
    } else {
        std::string separator = ", ";  // Default

        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                analysis.tools.format.section_start + p.space() + tool_choice + p.zero_or_more(separator + tool_choice) + p.space() + analysis.tools.format.section_end);
        } else {
            tool_calls = p.trigger_rule("tool-call",
                analysis.tools.format.section_start + p.space() + tool_choice + p.space() + analysis.tools.format.section_end);
        }
    }

    if (!require_tools) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker = !analysis.tools.format.section_start.empty() ? analysis.tools.format.section_start : analysis.tools.format.per_call_start;
    auto content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return reasoning + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
}
