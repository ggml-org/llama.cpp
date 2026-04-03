#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

using json = nlohmann::ordered_json;

namespace {

// Gemma4-specific PEG builder extending the standard chat builder.
// Adds value type parsers that use <|\"|> as string delimiters
// instead of JSON's double quotes, and disables json-to-schema
// conversion for these types.
class common_peg_gemma4_builder {
    common_chat_peg_builder & p_;
    static constexpr const char * QUOTE = "<|\"|>";

public:
    explicit common_peg_gemma4_builder(common_chat_peg_builder & p) : p_(p) {}

    common_peg_parser gemma4_string() {
        return p_.rule("gemma4-string", [&]() {
            return p_.literal(QUOTE) + p_.until(QUOTE) + p_.literal(QUOTE);
        });
    }

    common_peg_parser gemma4_number() {
        return p_.rule("gemma4-number", [&]() {
            auto digit1_9 = p_.chars("[1-9]", 1, 1);
            auto digits   = p_.chars("[0-9]");
            auto int_part = p_.choice({p_.literal("0"), p_.sequence({digit1_9, p_.chars("[0-9]", 0, -1)})});
            auto frac     = p_.sequence({p_.literal("."), digits});
            auto exp      = p_.sequence({p_.choice({p_.literal("e"), p_.literal("E")}),
                                         p_.optional(p_.chars("[+-]", 1, 1)), digits});
            auto not_number_continuation = p_.negate(p_.chars("[0-9.eE+-]", 1, 1));
            return p_.sequence({p_.optional(p_.literal("-")), int_part, p_.optional(frac),
                                p_.optional(exp), not_number_continuation});
        });
    }

    common_peg_parser gemma4_bool() {
        return p_.rule("gemma4-bool", [&]() {
            return p_.choice({p_.literal("true"), p_.literal("false")});
        });
    }

    common_peg_parser gemma4_null() {
        return p_.rule("gemma4-null", [&]() {
            return p_.literal("null");
        });
    }

    common_peg_parser gemma4_dict() {
        return p_.rule("gemma4-dict", [&]() {
            auto ws = p_.space();
            auto key = p_.until(":");
            auto member = p_.sequence({key, p_.literal(":"), ws, gemma4_value()});
            auto members = p_.sequence({member, p_.zero_or_more(p_.sequence({p_.literal(","), ws, member}))});
            return p_.sequence({
                p_.literal("{"), ws,
                p_.choice({p_.literal("}"), p_.sequence({members, ws, p_.literal("}")})})
            });
        });
    }

    common_peg_parser gemma4_array() {
        return p_.rule("gemma4-array", [&]() {
            auto ws = p_.space();
            auto elements = p_.sequence({gemma4_value(), p_.zero_or_more(p_.sequence({p_.literal(","), ws, gemma4_value()}))});
            return p_.sequence({
                p_.literal("["), ws,
                p_.choice({p_.literal("]"), p_.sequence({elements, ws, p_.literal("]")})})
            });
        });
    }

    common_peg_parser gemma4_value() {
        return p_.rule("gemma4-value", [&]() {
            return p_.choice({gemma4_string(), gemma4_dict(), gemma4_array(),
                              gemma4_number(), gemma4_bool(), gemma4_null()});
        });
    }

    // Select the appropriate value parser based on JSON schema type.
    // Does NOT use schema() - the gemma4 types are pure PEG without
    // JSON schema metadata, so GBNF is generated directly from the
    // PEG structure.
    common_peg_parser gemma4_value_for_type(const json & schema) {
        if (!schema.contains("type") || !schema.at("type").is_string()) {
            return gemma4_value();
        }
        std::string type = schema.at("type").get<std::string>();
        if (type == "string")  { return gemma4_string(); }
        if (type == "number")  { return gemma4_number(); }
        if (type == "integer") { return gemma4_number(); }
        if (type == "boolean") { return gemma4_bool(); }
        if (type == "object")  { return gemma4_dict(); }
        if (type == "array")   { return gemma4_array(); }
        return gemma4_value();
    }
};

}  // anonymous namespace

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

parser_build_context::parser_build_context(common_chat_peg_builder & p, const generation_params & inputs) :
    p(p),
    inputs(inputs),
    reasoning_parser(p.eps()) {}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs) {
    // Run differential analysis to extract template structure
    struct autoparser autoparser;
    autoparser.analyze_template(tmpl);
    return generate_parser(tmpl, inputs, autoparser);
}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs,
                                                  const autoparser &              autoparser) {
    // Create the result structure
    common_chat_params data;
    data.prompt           = common_chat_template_direct_apply(tmpl, inputs);
    data.format           = (autoparser.tools.format.mode == tool_format::TAG_WITH_GEMMA4_DICT)
                            ? COMMON_CHAT_FORMAT_PEG_GEMMA4
                            : COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens = autoparser.preserved_tokens;

    auto parser = autoparser.build_parser(inputs);
    data.parser = parser.save();

    // Build grammar if tools are present
    bool has_tools =
        autoparser.tools.format.mode != tool_format::NONE && inputs.tools.is_array() && !inputs.tools.empty();
    std::string trigger_marker = !autoparser.tools.format.section_start.empty() ? autoparser.tools.format.section_start :
                                                                                  autoparser.tools.format.per_call_start;

    bool has_response_format = !inputs.json_schema.empty() && inputs.json_schema.is_object();
    bool include_grammar = has_response_format || (has_tools &&
            ((inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO && !trigger_marker.empty()) ||
              inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Set grammar triggers based on tool section markers (fall back to per-call markers)
        if (data.grammar_lazy) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_marker }
            };
        }
    }

    return data;
}

common_peg_arena autoparser::build_parser(const generation_params & inputs) const {
    if (!analysis_complete) {
        throw std::invalid_argument("Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)");
    }
    return build_chat_peg_parser([&](common_chat_peg_builder & p) {
        parser_build_context ctx(p, inputs);
        bool                 extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

        ctx.extracting_reasoning = extract_reasoning && reasoning.mode != reasoning_mode::NONE;
        ctx.content              = &content;
        ctx.reasoning            = &reasoning;

        // Build reasoning parser
        ctx.reasoning_parser = reasoning.build_parser(ctx);

        auto parser = p.eps();

        bool has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
        bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
        bool pure_content        = reasoning.mode == reasoning_mode::NONE;

        if (has_response_format) {
            auto response_format = p.rule("response-format", p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));
            parser = ctx.reasoning_parser + p.space() + p.choice({
                p.literal("```json") + p.space() + response_format + p.space() + p.literal("```"),
                response_format
            }) + p.end();
            pure_content = false;
        } else if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && jinja_caps.supports_tool_calls) {
            parser = tools.build_parser(ctx);
            pure_content = false;
        } else {
            parser = content.build_parser(ctx);
        }
        return pure_content ? p.prefix(inputs.generation_prompt, reasoning.start) + parser : p.prefix(inputs.generation_prompt, reasoning.start) << parser;
    });
}

common_peg_parser analyze_reasoning::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (!ctx.extracting_reasoning) {
        return p.eps();
    }

    if (mode == reasoning_mode::TAG_BASED || mode == reasoning_mode::TOOLS_ONLY) {
        if (!end.empty()) {
            if (!start.empty()) {
                // Standard tag-based: optional(<think>reasoning</think>)
                return p.optional(start + p.reasoning(p.until(end)) + end + p.space());
            }
            // Delimiter-style (empty start)
            return p.optional(p.reasoning(p.until(end)) + end + p.space());
        }
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
        case tool_format::TAG_WITH_GEMMA4_DICT:
            return build_tool_parser_tag_gemma4_dict(ctx);
        default:
            LOG_ERR("[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work properly. "
                "Check for a fixed template for your model in the models/templates directory of your llama.cpp installation or "
                "report an issue at https://github.com/ggml-org/llama.cpp/issues\n");
            return ctx.p.eps();
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
        const auto & schema = func.contains("parameters") ? func.at("parameters") : json::object();

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
        const auto &          func       = tool.at("function");
        std::string           name       = func.at("name");
        const auto &          params     = func.contains("parameters") ? func.at("parameters") : json::object();
        const auto &          properties = params.contains("properties") ? params.at("properties") : json::object();
        std::set<std::string> required;

        // Build parser for each argument, separating required and optional
        std::vector<common_peg_parser> required_parsers;
        std::vector<common_peg_parser> optional_parsers;
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

            auto arg =
                p.tool_arg(p.tool_arg_open(arguments.name_prefix + p.tool_arg_name(p.literal(param_name)) +
                                           arguments.name_suffix) +
                           arguments.value_prefix +
                           (type == "string" ?
                                p.tool_arg_string_value(p.schema(p.until(arguments.value_suffix),
                                                                 "tool-" + name + "-arg-" + param_name + "-schema",
                                                                 param_schema, true)) :
                                p.tool_arg_json_value(p.schema(
                                    p.json(), "tool-" + name + "-arg-" + param_name + "-schema", param_schema, false)) +
                                    p.space()) +
                           p.tool_arg_close(p.literal(arguments.value_suffix)));

            auto named_arg = p.rule("tool-" + name + "-arg-" + param_name, arg);
            if (is_required) {
                required_parsers.push_back(named_arg);
            } else {
                optional_parsers.push_back(named_arg);
            }
        }

        // Build required arg sequence in definition order
        common_peg_parser args_seq = p.eps();
        for (size_t i = 0; i < required_parsers.size(); i++) {
            if (i > 0) {
                args_seq = args_seq + p.space();
            }
            args_seq = args_seq + required_parsers[i];
        }

        // Build optional args with flexible ordering
        if (!optional_parsers.empty()) {
            common_peg_parser any_opt = p.choice();
            for (const auto & opt : optional_parsers) {
                any_opt |= opt;
            }
            args_seq = args_seq + p.repeat(p.space() + any_opt, 0, (int) optional_parsers.size());
        }

        // Build call_id parser based on position (if supported)
        common_peg_parser call_id_section = p.eps();
        bool have_call_id = false;
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            !call_id.suffix.empty()) {
            have_call_id = true;
            call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix)) + call_id.suffix);
        }

        bool matched_atomic = false;
        common_peg_parser func_parser = p.eps();
        if (!function.name_suffix.empty()) {
            func_parser = p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix) +
                call_id_section + p.space() + args_seq;
            matched_atomic = true;
        } else if (have_call_id) {
            func_parser = p.atomic(p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix) +
                call_id_section) + p.space() + args_seq;
            matched_atomic = true;
        } else if (!arguments.name_prefix.empty() && !required_parsers.empty()) {
            // Only peek for an arg tag when there are required args that must follow.
            // When all args are optional, the model may emit no arg tags at all (#20650).
            func_parser = p.atomic(p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix) +
                call_id_section + p.space() + p.peek(p.literal(arguments.name_prefix))) + args_seq;
            matched_atomic = true;
        } else {
            func_parser = p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix) +
                call_id_section + p.space() + args_seq;
        }

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
        if (!matched_atomic) {
            func_parser = p.atomic(func_parser);
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

common_peg_parser analyze_tools::build_tool_parser_tag_gemma4_dict(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;
    bool         force_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_gemma4_builder g4(p);
    static const std::string QUOTE = "<|\"|>";

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func   = tool.at("function");
        std::string  name   = func.at("name");
        const auto & params = func.at("parameters");

        if (!params.contains("properties") || !params.at("properties").is_object()) {
            auto func_parser = p.atomic(
                p.tool_open(p.literal(function.name_prefix) + p.tool_name(p.literal(name)) + p.literal("{")) +
                p.tool_args(p.eps()) +
                p.tool_close(p.literal("}")));
            tool_choice |= p.rule("tool-" + name, func_parser);
            return;
        }

        const auto &          properties = params.at("properties");
        std::set<std::string> required;
        if (params.contains("required") && params.at("required").is_array()) {
            params.at("required").get_to(required);
        }

        // Build per-argument parsers, sorted alphabetically (matching template's dictsort)
        struct arg_entry {
            std::string       param_name;
            common_peg_parser parser;
        };
        std::vector<arg_entry> arg_entries;

        for (const auto & [param_name, param_schema] : properties.items()) {
            std::string type    = "object";
            auto        type_v  = param_schema.contains("type") ? param_schema.at("type") : json::object();
            if (type_v.is_string()) type_v.get_to(type);

            common_peg_parser value_parser = p.eps();
            if (type == "string") {
                // String values are delimited by <|"|>...<|"|>
                value_parser =
                    p.literal(QUOTE) +
                    p.tool_arg_string_value(p.schema(p.until(QUOTE),
                        "tool-" + name + "-arg-" + param_name + "-schema", param_schema, true)) +
                    p.literal(QUOTE);
            } else if (type == "number" || type == "integer") {
                value_parser = p.tool_arg_value(g4.gemma4_number());
            } else if (type == "boolean") {
                value_parser = p.tool_arg_value(g4.gemma4_bool());
            } else if (type == "null") {
                value_parser = p.tool_arg_value(g4.gemma4_null());
            } else if (type == "object") {
                value_parser = p.tool_arg_value(g4.gemma4_dict());
            } else if (type == "array") {
                value_parser = p.tool_arg_value(g4.gemma4_array());
            } else {
                value_parser = p.tool_arg_value(g4.gemma4_value());
            }

            auto arg = p.tool_arg(
                p.tool_arg_open(p.tool_arg_name(p.literal(param_name)) + p.literal(":")) +
                value_parser +
                p.tool_arg_close(p.eps()));

            arg_entries.push_back({param_name, p.rule("tool-" + name + "-arg-" + param_name, arg)});
        }

        // Sort alphabetically to match Jinja's dictsort
        std::sort(arg_entries.begin(), arg_entries.end(), [](const auto & a, const auto & b) {
            return a.param_name < b.param_name;
        });

        // Build arg sequence: any arg, then zero-or-more comma-separated additional args
        common_peg_parser args_seq = p.eps();
        if (!arg_entries.empty()) {
            common_peg_parser any_arg = p.choice();
            for (auto & entry : arg_entries) {
                any_arg |= entry.parser;
            }
            args_seq = p.optional(
                any_arg + p.repeat(p.literal(",") + any_arg, 0, (int) arg_entries.size() - 1));
        }

        // Full parser: call:name{args}
        auto func_parser = p.atomic(
            p.tool_open(p.literal(function.name_prefix) + p.tool_name(p.literal(name)) + p.literal("{")) +
            p.tool_args(args_seq) +
            p.tool_close(p.literal("}")));

        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    // Wrap each call in <|tool_call>...</tool_call|>
    auto wrapped_call = p.literal(format.per_call_start) + tool_choice + p.literal(format.per_call_end);

    common_peg_parser tool_calls = p.eps();
    if (inputs.parallel_tool_calls) {
        tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call));
    } else {
        tool_calls = p.trigger_rule("tool-call", wrapped_call);
    }

    if (!force_tools) {
        tool_calls = p.optional(tool_calls);
    }

    auto content_before_tools = p.until_one_of({ format.per_call_start, ctx.reasoning->start });
    return ctx.reasoning_parser +
           (force_tools ? p.eps() : p.optional(p.content(content_before_tools) + p.optional(ctx.reasoning_parser))) +
           tool_calls + p.end();
}

}  // namespace autoparser
