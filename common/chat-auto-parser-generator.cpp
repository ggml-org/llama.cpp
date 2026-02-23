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

namespace autoparser {

parser_build_context::parser_build_context(common_chat_peg_builder & p, const templates_params & inputs) :
    p(p),
    inputs(inputs),
    reasoning_parser(p.eps()) {}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct templates_params & inputs) {
    // Run differential analysis to extract template structure
    autoparser analysis(tmpl);
    return generate_parser(tmpl, inputs, analysis);
}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct templates_params & inputs,
                                                  const autoparser &              analysis) {
    // Build the parser using the analysis results
    auto parser = analysis.build_parser(inputs);

    // Create the result structure
    common_chat_params data;
    data.prompt           = common_chat_template_direct_apply(tmpl, inputs);
    data.format           = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens = analysis.preserved_tokens;
    data.parser           = parser.save();

    // Build grammar if tools are present
    bool has_tools =
        analysis.tools.format.mode != tool_format::NONE && inputs.tools.is_array() && !inputs.tools.empty();
    std::string trigger_marker = !analysis.tools.format.section_start.empty() ? analysis.tools.format.section_start :
                                                                                analysis.tools.format.per_call_start;
    bool        include_grammar =
        has_tools && ((inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO && !trigger_marker.empty()) ||
                      inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED);

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Set grammar triggers based on tool section markers (fall back to per-call markers)
        if (data.grammar_lazy) {  // only do triggers on lazy grammar
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_marker }
            };
        }
    }

    return data;
}

common_peg_arena autoparser::build_parser(const templates_params & inputs) const {
    return build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // If the template uses Python dict format (single-quoted strings in JSON structures),
        // pre-register a json-string rule that accepts both quote styles. This must happen
        // before any call to p.json() so that all JSON parsing inherits the flexible rule.
        if (tools.format.uses_python_dicts) {
            p.rule("json-string", [&]() { return p.choice({ p.double_quoted_string(), p.single_quoted_string() }); });
        }

        parser_build_context ctx(p, inputs);
        bool                 extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
        bool                 enable_thinking   = inputs.enable_thinking;

        ctx.extracting_reasoning = extract_reasoning && enable_thinking && reasoning.mode != reasoning_mode::NONE;
        ctx.content              = &content;

        // Build reasoning parser
        ctx.reasoning_parser = reasoning.build_parser(ctx);

        bool has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
        bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();

        if (has_response_format) {
            return ctx.reasoning_parser + p.space() +
                   p.content(p.schema(p.json(), "response-format", inputs.json_schema)) + p.end();
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && jinja_caps.supports_tool_calls) {
            return tools.build_parser(ctx);
        }

        return content.build_parser(ctx);
    });
}

common_peg_parser analyze_reasoning::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (!ctx.extracting_reasoning) {
        return p.eps();
    }

    bool thinking_forced_open   = (mode == reasoning_mode::FORCED_OPEN);
    bool thinking_forced_closed = (mode == reasoning_mode::FORCED_CLOSED);

    if (thinking_forced_open || thinking_forced_closed) {
        // Thinking is forced open OR forced closed with enable_thinking=true
        // In both cases, expect only the closing tag (opening was in template)
        return p.reasoning(p.until(end)) + end;
    }
    if (mode == reasoning_mode::TAG_BASED || mode == reasoning_mode::TOOLS_ONLY) {
        // Standard tag-based reasoning OR tools-only mode (reasoning appears with tools)
        // Both use the same tag-based pattern if markers are available
        if (!start.empty() && !end.empty()) {
            return p.optional(start + p.reasoning(p.until(end)) + end);
        }
    } else if (mode == reasoning_mode::DELIMITER) {
        return p.optional(p.reasoning(p.until(end)) + end);
    }

    return p.eps();
}

common_peg_parser analyze_content::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        if (ctx.extracting_reasoning) {
            return ctx.reasoning_parser + start + p.content(p.until(end)) + end + p.end();
        }
        return p.content(p.until(start)) + start + p.content(p.until(end)) + end + p.end();
    }
    return ctx.reasoning_parser + p.content(p.rest()) + p.end();
}

common_peg_parser analyze_content::build_optional_wrapped(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        return p.optional(start + p.content(p.until(end)) + end);
    }
    return p.eps();
}

common_peg_parser analyze_tools::build_parser(parser_build_context & ctx) const {
    switch (format.mode) {
        case tool_format::JSON_NATIVE:
            return build_tool_parser_json_native(ctx);
        case tool_format::TAG_WITH_JSON:
            return build_tool_parser_tag_json(ctx);
        case tool_format::TAG_WITH_TAGGED:
            return build_tool_parser_tag_tagged(ctx);
        default:
            GGML_ABORT("Unable to create tool parser");
    }
}

common_peg_parser analyze_tools::build_tool_parser_json_native(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;
    bool         force_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    // Build effective field names with dot notation if function_field is set
    std::string name_field = format.name_field;
    std::string args_field = format.args_field;

    if (!format.function_field.empty() && format.function_field != "function" &&
        name_field.find('.') == std::string::npos) {
        name_field = format.function_field + "." + name_field;
        args_field = format.function_field + "." + args_field;
    }

    auto tools_parser = p.standard_json_tools(
        format.section_start, format.section_end, inputs.tools, inputs.parallel_tool_calls,
        inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
        format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order);

    // Handle content wrappers if present
    if (ctx.content && ctx.content->is_always_wrapped()) {
        auto wrapped_content = ctx.content->build_optional_wrapped(ctx);
        return ctx.reasoning_parser + wrapped_content + tools_parser + p.end();
    }

    std::string tool_start = "{";
    if (!format.section_start.empty()) {
        tool_start = format.section_start;
    } else if (!format.per_call_start.empty()) {
        tool_start = format.per_call_start;
    }

    return ctx.reasoning_parser + (force_tools ? p.eps() : p.optional(p.content(p.until(tool_start)))) + tools_parser +
           p.end();
}

common_peg_parser analyze_tools::build_tool_parser_tag_json(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;
    bool         force_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func   = tool.at("function");
        std::string  name   = func.at("name");
        const auto & schema = func.at("parameters");

        // Build call_id parser based on position (if supported)
        common_peg_parser call_id_section = p.eps();
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            !call_id.suffix.empty()) {
            call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix))) + call_id.suffix;
        }

        auto func_parser = p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix) +
                           call_id_section + p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema));
        if (!function.close.empty()) {
            func_parser = func_parser + function.close;
        }

        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + tool_choice + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call));
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call);
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end)));
        }
    } else {
        std::string separator = ", ";  // Default
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + format.section_end);
        } else {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice + format.section_end);
        }
    }

    if (!require_calls) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return ctx.reasoning_parser + (force_tools ? p.eps() : p.optional(p.content(content_before_tools))) + tool_calls +
           p.end();
}

common_peg_parser analyze_tools::build_tool_parser_tag_tagged(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;
    bool         force_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func   = tool.at("function");
        std::string  name   = func.at("name");
        const auto & params = func.at("parameters");

        if (!params.contains("properties") || !params.at("properties").is_object()) {
            return;
        }

        const auto &          properties = params.at("properties");
        std::set<std::string> required;
        if (params.contains("required") && params.at("required").is_array()) {
            params.at("required").get_to(required);
        }

        // Build parser for each argument
        std::vector<common_peg_parser> arg_parsers;
        for (const auto & [param_name, param_schema] : properties.items()) {
            bool        is_required = required.find(param_name) != required.end();
            std::string type        = "object";
            auto        type_obj    = param_schema.contains("type") ? param_schema.at("type") : json::object();
            if (type_obj.is_string()) {
                type_obj.get_to(type);
            } else if (type_obj.is_object()) {
                if (type_obj.contains("type") && type_obj.at("type").is_string()) {
                    type_obj.at("type").get_to(type);
                }
            }

            auto arg = p.tool_arg(
                p.tool_arg_open(arguments.name_prefix + p.tool_arg_name(p.literal(param_name)) +
                                arguments.name_suffix) +
                arguments.value_prefix +
                (type == "string" ? p.tool_arg_string_value(p.schema(p.until(arguments.value_suffix),
                                                                     "tool-" + name + "-arg-" + param_name + "-schema",
                                                                     param_schema, true)) :
                                    p.tool_arg_json_value(p.schema(
                                        p.json(), "tool-" + name + "-arg-" + param_name + "-schema", param_schema)) +
                                        p.space()) +
                p.tool_arg_close(p.literal(arguments.value_suffix)));

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
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            !call_id.suffix.empty()) {
            call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix))) + call_id.suffix;
        }

        auto func_parser = p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix) +
                           call_id_section + p.space() + args_seq;

        if (!function.close.empty()) {
            func_parser = func_parser + p.space() + p.tool_close(p.literal(function.close));
        } else if (!format.per_call_end.empty()) {
            // When there's no func_close but there is a per_call_end marker, use peek() to ensure
            // we only emit tool_close when we can actually see the closing marker. This prevents
            // premature closing during partial parsing when we've seen e.g. "</" which could be
            // either "</tool_call>" (end) or "<arg_key>" prefix that failed to match.
            func_parser = func_parser + p.tool_close(p.peek(p.literal(format.per_call_end)));
        } else {
            func_parser =
                func_parser + p.tool_close(p.space());  // force this to process tool closing callbacks in mapper
        }

        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + p.space() + tool_choice + p.space() + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call));
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call);
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end)));
        }
    } else {
        std::string separator = ", ";  // Default

        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + p.space() + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + p.space() +
                                                         format.section_end);
        } else {
            tool_calls = p.trigger_rule(
                "tool-call", format.section_start + p.space() + tool_choice + p.space() + format.section_end);
        }
    }

    if (!require_tools) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return ctx.reasoning_parser + (force_tools ? p.eps() : p.optional(p.content(content_before_tools))) + tool_calls +
           p.end();
}

}  // namespace autoparser
