#include "chat-peg-parser.h"

// #include <cstdint>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static std::string_view trim_trailing_space(std::string_view sv, int max = -1) {
    int count = 0;
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        if (max != -1 && count <= max) {
            break;
        }
        sv.remove_suffix(1);
        count++;
    }
    return sv;
}

void common_chat_peg_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    arena.visit(result, [this](const common_peg_ast_node & node) { map(node); });
}

void common_chat_peg_mapper::map(const common_peg_ast_node & node) {
    bool is_reasoning = node.tag == common_chat_peg_builder::REASONING;
    bool is_content   = node.tag == common_chat_peg_builder::CONTENT;

    if (is_reasoning) {
        result.reasoning_content = std::string(trim_trailing_space(node.text));
    }

    if (is_content) {
        result.content = std::string(trim_trailing_space(node.text));
    }
}

void common_chat_peg_native_mapper::map(const common_peg_ast_node & node) {
    common_chat_peg_mapper::map(node);

    bool is_tool_open = node.tag == common_chat_peg_native_builder::TOOL_OPEN;
    bool is_tool_name = node.tag == common_chat_peg_native_builder::TOOL_NAME;
    bool is_tool_id   = node.tag == common_chat_peg_native_builder::TOOL_ID;
    bool is_tool_args = node.tag == common_chat_peg_native_builder::TOOL_ARGS;

    if (is_tool_open) {
        result.tool_calls.emplace_back();
        current_tool = &result.tool_calls.back();
    }

    if (is_tool_id && current_tool) {
        current_tool->id = std::string(trim_trailing_space(node.text));
    }

    if (is_tool_name && current_tool) {
        current_tool->name = std::string(trim_trailing_space(node.text));
    }

    if (is_tool_args && current_tool) {
        current_tool->arguments = std::string(trim_trailing_space(node.text));
    }
}

void common_chat_peg_constructed_mapper::map(const common_peg_ast_node & node) {
    common_chat_peg_mapper::map(node);

    bool is_tool_open  = node.tag == common_chat_peg_constructed_builder::TOOL_OPEN;
    bool is_tool_name  = node.tag == common_chat_peg_constructed_builder::TOOL_NAME;
    bool is_tool_close = node.tag == common_chat_peg_constructed_builder::TOOL_CLOSE;
    bool is_arg_open   = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_OPEN;
    bool is_arg_close  = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_CLOSE;
    bool is_arg_name   = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_NAME;
    bool is_arg_string = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_STRING_VALUE;
    bool is_arg_json   = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_JSON_VALUE;

    if (is_tool_open) {
        result.tool_calls.emplace_back();
        current_tool = &result.tool_calls.back();
        arg_count    = 0;
    }

    if (is_tool_name) {
        current_tool->name      = std::string(node.text);
        current_tool->arguments = "{";
    }

    if (is_arg_open) {
        needs_closing_quote = false;
    }

    if (is_arg_name && current_tool) {
        if (arg_count > 0) {
            current_tool->arguments += ",";
        }
        current_tool->arguments += json(trim_trailing_space(node.text)).dump() + ":";
        ++arg_count;
    }

    if (is_arg_string && current_tool) {
        // Serialize to JSON, but exclude the end quote
        std::string dumped = json(trim_trailing_space(node.text)).dump();
        current_tool->arguments += dumped.substr(0, dumped.size() - 1);
        needs_closing_quote = true;
    }

    if (is_arg_close && current_tool) {
        if (needs_closing_quote) {
            current_tool->arguments += "\"";
            needs_closing_quote = false;
        }
    }

    if (is_arg_json && current_tool) {
        // When handling JSON args, only add to output when there's actual content
        // This prevents empty arguments from being output as "" which won't diff correctly
        // against primitive values like 1, true, null, etc.
        std::string content = std::string(trim_trailing_space(node.text));
        if (!content.empty()) {
            current_tool->arguments += content;
        }
    }

    if (is_tool_close && current_tool) {
        if (needs_closing_quote) {
            current_tool->arguments += "\"";
            needs_closing_quote = false;
        }
        current_tool->arguments += "}";
    }
}

common_peg_parser common_chat_peg_native_builder::standard_json_tools(const std::string &    open_tag,
                                                                      const std::string &    close_tag,
                                                                      const nlohmann::json & tool_defs,
                                                                      bool                   parallel_tool_calls,
                                                                      bool                   force_tool_calls) {
    auto tools = choice();
    for (const auto & tool : tool_defs) {
        const auto & function = tool.at("function");
        std::string  name     = function.at("name");
        const auto & schema_  = function.at("parameters");

        auto tool_name_ = json_member("name", "\"" + tool_name(literal(name)) + "\"");
        auto tool_args_ = json_member("arguments", tool_args(schema(json(), "tool-" + name + "-schema", schema_)));

        tools |= rule("tool-" + name, tool_open(literal("{")) << space() << tool_name_ << space() << "," << space()
                                                              << tool_args_ << space() << "}");
    };

    auto parallel_calls = eps();
    if (parallel_tool_calls) {
        parallel_calls = zero_or_more(space() << "," << space() << tools);
    }

    auto tool_call =
        trigger_rule("tool-call", sequence({ literal(open_tag), tools, parallel_calls, literal(close_tag) }));

    if (!force_tool_calls) {
        tool_call = optional(tool_call);
    }
    return tool_call;
}

common_peg_parser common_chat_peg_builder::tag_with_safe_content(const std::string &       tag_name,
                                                                 const std::string &       marker,
                                                                 const common_peg_parser & p) {
    if (marker.empty()) {
        return zero_or_more(choice({ p, rule(tag_name, content(any())) }));
    }
    auto content_chunk = rule(tag_name, content(negate(literal(marker)) + any() + until(marker)));
    return zero_or_more(choice({ p, content_chunk }));
}

common_peg_parser common_chat_peg_constructed_builder::standard_constructed_tools(
    const std::map<std::string, std::string> & markers,
    const nlohmann::json &                     tool_defs,
    bool                                       parallel_tool_calls,
    bool                                       force_tool_calls) {
    (void) force_tool_calls;
    auto marker_at = [&](const std::string & key) -> std::string {
        auto it = markers.find(key);
        return it != markers.end() ? it->second : "";
    };

    std::string tool_call_start  = marker_at("tool_call_start_marker");
    std::string tool_call_end    = marker_at("tool_call_end_marker");
    std::string func_opener      = marker_at("function_opener");
    std::string func_closer      = marker_at("function_closer");
    std::string func_name_suffix = marker_at("function_name_suffix");
    std::string param_key_prefix = marker_at("parameter_key_prefix");
    std::string param_key_suffix = marker_at("parameter_key_suffix");
    std::string param_closer     = marker_at("parameter_closer");
    std::string arg_separator    = marker_at("argument_separator");

    if (tool_call_start.empty() || tool_call_end.empty()) {
        // Fallback for models that might use function openers without global tool markers
        if (!func_opener.empty() && !func_closer.empty()) {
            return tool(tool_open(literal(func_opener)) + space() + tool_name(until(">")) +
                        content(until(func_closer)) + tool_close(literal(func_closer)));
        }
        return negate(eps());
    }

    auto build_args = [&](const nlohmann::json & parameters) -> common_peg_parser {
        if (param_key_prefix.empty()) {
            return content(until(tool_call_end));
        }
        if (parameters.contains("properties") && !parameters.at("properties").empty()) {
            auto arg_choice = choice();
            for (const auto & el : parameters.at("properties").items()) {
                const std::string & prop_name   = el.key();
                const auto &        prop_schema = el.value();

                auto arg_name_parser =
                    choice({ literal(prop_name), literal("\"" + prop_name + "\""), literal("'" + prop_name + "'") });

                // During differential analysis, determine if primitives are accepted as JSON args
                bool accepts_primitives = false;
                if (prop_schema.contains("type")) {
                    const std::string & type = prop_schema.at("type");
                    if (type == "number" || type == "integer" || type == "boolean" || type == "array" ||
                        type == "object") {
                        accepts_primitives = true;
                    }
                }

                // Build parser based on whether primitives are accepted
                common_peg_parser value_parser =
                    accepts_primitives ? tool_arg_json_value(until(param_closer.empty() ? "</" : param_closer)) :
                                         tool_arg_string_value(until(param_closer.empty() ? "</" : param_closer));

                auto arg_rule =
                    tool_arg(tool_arg_open(literal(param_key_prefix)) + tool_arg_name(arg_name_parser) +
                             (param_key_suffix.empty() ? literal(">") : literal(param_key_suffix)) + value_parser +
                             (param_closer.empty() ? eps() : tool_arg_close(literal(param_closer))) +
                             (arg_separator.empty() ? eps() : optional(literal(arg_separator))));
                arg_choice |= arg_rule;
            }
            return zero_or_more(arg_choice + space());
        }
        return eps();
    };

    // Build a lookahead parser that verifies the tool name is complete.
    // This prevents "special_function" from matching when input is "special_function_with_opt".
    auto build_name_terminator_peek = [&]() -> common_peg_parser {
        if (!func_name_suffix.empty()) {
            return peek(literal(func_name_suffix));
        }
        if (!func_closer.empty()) {
            return peek(literal(func_closer));
        }
        if (param_key_prefix.empty()) {
            // Tool call end marker or space should follow
            return peek(literal(tool_call_end) | chars(" \t\n\r"));
        }
        // Space should follow before args
        return peek(chars(" \t\n\r"));
    };

    auto build_tool_rule = [&](const std::string & name, const common_peg_parser & args) {
        return tool(tool_open(literal(tool_call_start)) + space() +
                    (!func_opener.empty() ? literal(func_opener) : eps()) +
                    tool_name(literal(name) + build_name_terminator_peek()) +
                    (!func_name_suffix.empty() ? literal(func_name_suffix) : eps()) +
                    (func_name_suffix.empty() && !func_closer.empty() ? literal(func_closer) : eps()) +
                    (param_key_prefix.empty() ? until(tool_call_end) : space()) + args + space() +
                    (!func_name_suffix.empty() && !func_closer.empty() && func_closer != func_name_suffix ?
                         literal(func_closer) :
                         eps()) +
                    until(tool_call_end) + tool_close(literal(tool_call_end)));
    };

    auto tool_choices      = choice();
    bool has_defined_tools = false;

    if (tool_defs.is_array() && !tool_defs.empty()) {
        for (const auto & tool : tool_defs) {
            if (!tool.contains("function")) {
                continue;
            }
            has_defined_tools      = true;
            std::string name       = tool.at("function").at("name");
            auto        parameters = tool.at("function").contains("parameters") ? tool.at("function").at("parameters") :
                                                                                  nlohmann::json::object();
            tool_choices |= rule("tool-" + name, build_tool_rule(name, build_args(parameters)));
        }
    }

    if (!has_defined_tools) {
        return rule("tool_call", negate(eps()));
    }

    auto tool_call = tool_choices;
    if (parallel_tool_calls) {
        tool_call = one_or_more(tool_call + space());
    }

    return trigger_rule("tool_call", tool_call);
}
