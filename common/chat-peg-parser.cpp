#include "chat-peg-parser.h"

#include "chat-auto-parser.h"
#include "ggml.h"
#include "peg-parser.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

static std::string_view trim_trailing_space(std::string_view sv, int max = -1) {
    int count = 0;
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        if (max != -1 && count >= max) {
            break;
        }
        sv.remove_suffix(1);
        count++;
    }
    return sv;
}

static std::string_view trim_leading_space(std::string_view sv, int max = -1) {
    int count = 0;
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
        if (max != -1 && count >= max) {
            break;
        }
        sv.remove_prefix(1);
        count++;
    }
    return sv;
}

static std::string_view trim(std::string_view sv) {
    return trim_trailing_space(trim_leading_space(sv, 1));
}

// Count the number of unclosed '{' braces in a JSON-like string,
// properly skipping braces inside quoted strings.
static int json_brace_depth(const std::string & s) {
    int  depth     = 0;
    bool in_string = false;
    bool escaped   = false;
    for (char c : s) {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\' && in_string) {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (!in_string) {
            if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
            }
        }
    }
    return depth;
}

// JSON-escape a string and return the inner content (without surrounding quotes).
static std::string escape_json_string_inner(const std::string & s) {
    std::string escaped = json(s).dump();
    if (escaped.size() >= 2 && escaped.front() == '"' && escaped.back() == '"') {
        return escaped.substr(1, escaped.size() - 2);
    }
    return escaped;
}

// Convert Python-style single-quoted strings to JSON double-quoted strings
// Only converts outer string delimiters, properly handling escape sequences:
// - {'key': 'value'} -> {"key": "value"}
// - {'code': 'print(\'hello\')'} -> {"code": "print('hello')"}
// - {'msg': 'He said "hi"'} -> {"msg": "He said \"hi\""}
static std::string normalize_quotes_to_json(const std::string & input) {
    std::string result;
    result.reserve(input.size() + 16);  // May need extra space for escaping

    bool in_single_quoted = false;
    bool in_double_quoted = false;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        // Handle escape sequences
        if (c == '\\' && i + 1 < input.size()) {
            char next = input[i + 1];

            if (in_single_quoted) {
                // Inside a single-quoted string being converted to double quotes
                if (next == '\'') {
                    // \' -> ' (escaped single quote becomes unescaped in double-quoted string)
                    result += '\'';
                    ++i;
                    continue;
                }
                if (next == '"') {
                    // \" stays as \" (already escaped, works in double-quoted string)
                    result += "\\\"";
                    ++i;
                    continue;
                }
                // Other escapes (\n, \\, etc.): pass through both characters
                result += c;
                result += next;
                ++i;
                continue;
            }

            if (in_double_quoted) {
                // Inside a double-quoted string - pass through escape sequences as-is
                result += c;
                result += next;
                ++i;
                continue;
            }

            // Outside any string - just pass through the backslash
            result += c;
            continue;
        }

        // Handle quote characters
        if (c == '"') {
            if (in_single_quoted) {
                // Unescaped double quote inside single-quoted string -> must escape for JSON
                result += "\\\"";
            } else {
                // Double quote as string delimiter or outside strings
                in_double_quoted = !in_double_quoted;
                result += c;
            }
        } else if (c == '\'') {
            if (in_double_quoted) {
                // Single quote inside double-quoted string -> pass through
                result += c;
            } else if (in_single_quoted) {
                // Closing single quote -> convert to double quote
                in_single_quoted = false;
                result += '"';
            } else {
                // Opening single quote -> convert to double quote
                in_single_quoted = true;
                result += '"';
            }
        } else {
            result += c;
        }
    }

    return result;
}

void tag_based_peg_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    arena.visit(result, [this](const common_peg_ast_node & node) {
        if (!node.tag.empty()) {
            tags[node.tag] = std::string(node.text);
        }
    });
}

tagged_parse_result tagged_peg_parser::parse_and_extract(const std::string & input, bool is_partial) const {
    common_peg_parse_context ctx(input, is_partial);
    auto parse_result = arena.parse(ctx);

    tag_based_peg_mapper mapper;
    mapper.from_ast(ctx.ast, parse_result);

    return { std::move(parse_result), std::move(mapper.tags) };
}

tagged_parse_result tagged_peg_parser::parse_anywhere_and_extract(const std::string & input) const {
    if (input.empty()) {
        return parse_and_extract(input, false);
    }
    for (size_t i = 0; i < input.size(); i++) {
        common_peg_parse_context ctx(input, false);
        ctx.debug = debug;
        auto parse_result = arena.parse(ctx, i);
        if (parse_result.success() || i == input.size() - 1) {
            tag_based_peg_mapper mapper;
            mapper.from_ast(ctx.ast, parse_result);
            return { std::move(parse_result), std::move(mapper.tags) };
        }
    }
    GGML_ABORT("Should not happen");
}

tagged_peg_parser build_tagged_peg_parser(
    const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn) {
    common_peg_parser_builder builder;
    builder.set_root(fn(builder));
    return { builder.build() };
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

std::string & common_chat_peg_mapper::args_target() {
    return (current_tool && !current_tool->name.empty()) ? current_tool->arguments : args_buffer;
}

void common_chat_peg_mapper::from_ast(const common_peg_ast_arena &    arena,
                                      const common_peg_parse_result & parse_result_arg) {
    arena.visit(parse_result_arg, [this](const common_peg_ast_node & node) { map(node); });
    // Flush any pending tool call that was started but never got a name
    // This happens during partial parsing when the tool call is incomplete
    if (pending_tool_call.has_value() && !pending_tool_call->name.empty()) {
        if (!args_buffer.empty()) {
            pending_tool_call->arguments = args_buffer;
        }
        if (closing_quote_pending && !pending_tool_call->arguments.empty()) {
            pending_tool_call->arguments += "\"";
        }
        result.tool_calls.push_back(pending_tool_call.value());
        pending_tool_call.reset();
    }
}

void common_chat_peg_mapper::map(const common_peg_ast_node & node) {
    // Handle reasoning/content tags
    bool is_reasoning = node.tag == common_chat_peg_builder::REASONING;
    bool is_content   = node.tag == common_chat_peg_builder::CONTENT;

    if (is_reasoning) { // GPT OSS can have more than 1 reasoning block, so concatenate here
        result.reasoning_content += std::string(node.text);
    }

    if (is_content) {
        // Concatenate content from multiple content nodes (e.g., when reasoning markers
        // are preserved before content markers in reasoning_format=NONE mode)
        result.content += std::string(node.text);
    }

    // Handle tool-related tags (unified version supporting both JSON and tagged formats)
    bool is_tool_open  = node.tag == common_chat_peg_builder::TOOL_OPEN;
    bool is_tool_close = node.tag == common_chat_peg_builder::TOOL_CLOSE;
    bool is_tool_name  = node.tag == common_chat_peg_builder::TOOL_NAME;
    bool is_tool_id    = node.tag == common_chat_peg_builder::TOOL_ID;
    bool is_tool_args  = node.tag == common_chat_peg_builder::TOOL_ARGS;
    bool is_arg_open   = node.tag == common_chat_peg_builder::TOOL_ARG_OPEN;
    bool is_arg_close  = node.tag == common_chat_peg_builder::TOOL_ARG_CLOSE;
    bool is_arg_name         = node.tag == common_chat_peg_builder::TOOL_ARG_NAME;
    bool is_arg_value        = node.tag == common_chat_peg_builder::TOOL_ARG_VALUE;
    bool is_arg_string_value = node.tag == common_chat_peg_builder::TOOL_ARG_STRING_VALUE;

    if (is_tool_open) {
        pending_tool_call     = common_chat_tool_call();
        current_tool          = &pending_tool_call.value();
        arg_count             = 0;
        args_buffer.clear();
        closing_quote_pending = false;
    }

    if (is_tool_id && current_tool) {
        auto text = trim_trailing_space(node.text);
        if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
            text = text.substr(1, text.size() - 2);
        }
        current_tool->id = std::string(text);
    }

    if (is_tool_name && current_tool) {
        current_tool->name = std::string(trim_trailing_space(node.text));
        // Now that we have the name, populate the arguments from the buffer
        if (!args_buffer.empty()) {
            current_tool->arguments = args_buffer;
            args_buffer.clear();
        } else if (current_tool->arguments.empty()) {
            current_tool->arguments = "{";
        }
        // Add the tool call to results so streaming can see it
        if (pending_tool_call.has_value()) {
            result.tool_calls.push_back(pending_tool_call.value());
            pending_tool_call.reset();
            current_tool = &result.tool_calls.back();
        }
    }

    if (is_tool_args && current_tool) {
        // For JSON format: arguments come as a complete JSON object
        // For tagged format: built up from individual arg_name/arg_value nodes
        auto text = trim_trailing_space(node.text);
        if (!text.empty() && text.front() == '{') {
            args_target() = std::string(text);
        }
    }

    if (is_arg_open) {
        closing_quote_pending = false;
    }

    if (is_arg_name && current_tool) {
        std::string arg_entry;
        if (arg_count > 0) {
            arg_entry = ",";
        }
        arg_entry += json(trim(node.text)).dump() + ":";
        ++arg_count;

        auto & target = args_target();
        if (target.empty()) {
            target = "{";
        }
        target += arg_entry;
    }

    if ((is_arg_value || is_arg_string_value) && current_tool) {
        std::string value_content = std::string(trim_trailing_space(trim_leading_space(node.text, 1), 1));

        std::string value_to_add;
        if (value_content.empty() && is_arg_string_value) {
            // Empty string value - arg_close will add the closing quote
            value_to_add          = "\"";
            closing_quote_pending = true;
        } else if (!value_content.empty() && is_arg_string_value) {
            // Schema declares this as string type - always treat as literal string value
            if (!closing_quote_pending) {
                value_to_add          = "\"";
                closing_quote_pending = true;
            }
            value_to_add += escape_json_string_inner(value_content);
        } else if (!value_content.empty()) {
            // For potential containers, normalize Python-style single quotes to JSON double quotes
            bool is_potential_container = value_content[0] == '[' || value_content[0] == '{';
            if (is_potential_container) {
                value_content = normalize_quotes_to_json(value_content);
            }

            // Try to parse as JSON value (number, bool, null, object, array)
            try {
                json parsed = json::parse(value_content);
                if (parsed.is_string()) {
                    // Don't add closing quote yet (added by arg_close) for monotonic streaming
                    std::string escaped = parsed.dump();
                    if (!escaped.empty() && escaped.back() == '"') {
                        escaped.pop_back();
                    }
                    value_to_add          = escaped;
                    closing_quote_pending = true;
                } else {
                    // Non-string values: use raw content to preserve whitespace for monotonicity
                    value_to_add = value_content;
                }
            } catch (...) {
                if (node.is_partial && is_potential_container) {
                    // Partial container: pass through the already-normalized content
                    value_to_add = value_content;
                } else {
                    // Not valid JSON - treat as string value
                    if (!closing_quote_pending) {
                        value_to_add          = "\"";
                        closing_quote_pending = true;
                    }
                    value_to_add += escape_json_string_inner(value_content);
                }
            }
        }

        args_target() += value_to_add;
    }

    if (is_arg_close && current_tool) {
        if (closing_quote_pending) {
            args_target() += "\"";
            closing_quote_pending = false;
        }
    }

    if (is_tool_close && current_tool) {
        // Flush buffer to arguments if tool name was never seen
        if (current_tool->name.empty() && !args_buffer.empty()) {
            current_tool->arguments = args_buffer;
            args_buffer.clear();
        }
        // Close any pending string quote
        if (closing_quote_pending) {
            current_tool->arguments += "\"";
            closing_quote_pending = false;
        }
        // Close any unclosed braces (accounts for nested objects)
        for (int d = json_brace_depth(current_tool->arguments); d > 0; d--) {
            current_tool->arguments += "}";
        }
        // Add tool call to results if named; otherwise discard
        if (pending_tool_call.has_value()) {
            if (!current_tool->name.empty()) {
                result.tool_calls.push_back(pending_tool_call.value());
            }
            pending_tool_call.reset();
        }
    }
}

common_peg_parser common_chat_peg_builder::standard_constructed_tools(
    const std::map<std::string, std::string> & markers,
    const nlohmann::json &                     tools,
    bool                                       parallel_tool_calls,
    bool                                       force_tool_calls) {
    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    // Extract markers with defaults
    auto get_marker = [&markers](const std::string & key, const std::string & default_val = "") -> std::string {
        auto it = markers.find(key);
        return it != markers.end() ? it->second : default_val;
    };

    std::string section_start    = get_marker("tool_call_start_marker", "<tool_call>");
    std::string section_end      = get_marker("tool_call_end_marker", "</tool_call>");
    std::string func_opener      = get_marker("function_opener", "<function=");
    std::string func_name_suffix = get_marker("function_name_suffix", ">");
    std::string func_closer      = get_marker("function_closer", "</function>");
    std::string param_key_prefix = get_marker("parameter_key_prefix", "<param=");
    std::string param_key_suffix = get_marker("parameter_key_suffix", ">");
    std::string param_closer     = get_marker("parameter_closer", "</param>");

    // Build tool choices for tagged format
    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        nlohmann::json params = function.contains("parameters") ? function.at("parameters") : nlohmann::json::object();

        // Build argument parsers
        auto args = eps();
        if (params.contains("properties") && !params["properties"].empty()) {
            auto arg_choice = choice();
            for (const auto & el : params["properties"].items()) {
                const std::string & prop_name = el.key();

                auto arg_name_parser =
                    choice({ literal(prop_name), literal("\"" + prop_name + "\""), literal("'" + prop_name + "'") });

                auto arg_rule = tool_arg(tool_arg_open(literal(param_key_prefix)) + tool_arg_name(arg_name_parser) +
                                         literal(param_key_suffix) + tool_arg_value(until(param_closer)) +
                                         tool_arg_close(literal(param_closer)));
                arg_choice |= arg_rule;
            }
            args = zero_or_more(arg_choice + space());
        }

        // Build function parser: <function=name>args</function>
        auto tool_parser = tool(tool_open(literal(func_opener) + tool_name(literal(name)) + literal(func_name_suffix)) +
                                space() + tool_args(args) + space() + tool_close(literal(func_closer)));

        tool_choices |= rule("tool-" + name, tool_parser);
    }

    // Build the section with markers
    auto section =
        parallel_tool_calls ?
            trigger_rule("tool-call", literal(section_start) + space() + one_or_more(tool_choices + space()) +
                                          literal(section_end)) :
            trigger_rule("tool-call", literal(section_start) + space() + tool_choices + space() + literal(section_end));

    return force_tool_calls ? section : optional(section);
}

// Helper: Parse dot notation key into prefix and field name
static std::pair<std::string, std::string> parse_key_spec(const std::string & key) {
    auto dot_pos = key.find('.');
    if (dot_pos == std::string::npos) {
        return {"", key};  // Top-level field
    }
    return {key.substr(0, dot_pos), key.substr(dot_pos + 1)};
}

// Mode 1: function_is_key — parse {"function_name": {...}}
common_peg_parser common_chat_peg_builder::build_json_tools_function_is_key(
    const nlohmann::json & tools,
    const std::string &    args_key,
    const std::string &    effective_args_key,
    const std::string &    call_id_key,
    const std::string &    gen_call_id_key) {

    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        nlohmann::json params = function.contains("parameters") ? function.at("parameters") : nlohmann::json::object();

        // Build inner object fields
        std::vector<common_peg_parser> inner_fields;

        if (!call_id_key.empty()) {
            auto id_parser = atomic(
                literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                literal("\"") + tool_id(json_string_content()) + literal("\"")
            );
            inner_fields.push_back(optional(id_parser + space() + optional(literal(",") + space())));
        }

        if (!gen_call_id_key.empty()) {
            auto gen_id_parser = atomic(
                literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(json_string_content()) + literal("\""),
                    tool_id(json_number())
                })
            );
            inner_fields.push_back(optional(gen_id_parser + space() + optional(literal(",") + space())));
        }

        // Arguments — either wrapped in args_key or parsed directly
        common_peg_parser args_parser = eps();
        if (args_key.empty()) {
            args_parser = tool_args(schema(json(), "tool-" + name + "-schema", params));
        } else {
            args_parser = literal("\"" + effective_args_key + "\"") + space() + literal(":") + space() +
                          tool_args(schema(json(), "tool-" + name + "-schema", params));
        }
        inner_fields.push_back(args_parser);

        // Build inner object parser
        common_peg_parser inner_object = eps();
        if (args_key.empty() && inner_fields.size() == 1) {
            inner_object = inner_fields[0];
        } else {
            inner_object = literal("{") + space();
            for (size_t i = 0; i < inner_fields.size(); i++) {
                inner_object = inner_object + inner_fields[i];
                if (i < inner_fields.size() - 1) {
                    inner_object = inner_object + space();
                }
            }
            inner_object = inner_object + space() + literal("}");
        }

        auto tool_parser = tool(
            tool_open(literal("{")) + space() +
            literal("\"") + tool_name(literal(name)) + literal("\"") +
            space() + literal(":") + space() +
            inner_object +
            space() + tool_close(literal("}"))
        );

        tool_choices |= rule("tool-" + name, tool_parser);
    }

    return tool_choices;
}

// Mode 2: Nested keys (dot notation like "function.name")
common_peg_parser common_chat_peg_builder::build_json_tools_nested_keys(
    const nlohmann::json & tools,
    const std::string &    effective_name_key,
    const std::string &    effective_args_key,
    const std::string &    call_id_key,
    const std::string &    gen_call_id_key) {

    auto tool_choices = choice();

    auto name_spec = parse_key_spec(effective_name_key);
    auto args_spec = parse_key_spec(effective_args_key);

    std::string nested_prefix     = !name_spec.first.empty() ? name_spec.first  : args_spec.first;
    std::string nested_name_field = !name_spec.first.empty() ? name_spec.second  : effective_name_key;
    std::string nested_args_field = !args_spec.first.empty() ? args_spec.second  : effective_args_key;

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        nlohmann::json params = function.contains("parameters") ? function.at("parameters") : nlohmann::json::object();

        auto nested_name = literal("\"" + nested_name_field + "\"") + space() + literal(":") + space() +
                          literal("\"") + tool_name(literal(name)) + literal("\"");
        auto nested_args = literal("\"" + nested_args_field + "\"") + space() + literal(":") + space() +
                          tool_args(schema(json(), "tool-" + name + "-schema", params));

        auto nested_object = literal("{") + space() +
                            nested_name + space() + literal(",") + space() +
                            nested_args +
                            space() + literal("}");

        // Format: { id?, "function": {...} }
        auto tool_parser_body = tool_open(literal("{")) + space();

        if (!call_id_key.empty()) {
            auto id_spec = parse_key_spec(call_id_key);
            if (id_spec.first.empty()) {
                auto id_parser = atomic(
                    literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                    literal("\"") + tool_id(json_string_content()) + literal("\"")
                );
                tool_parser_body = tool_parser_body + optional(id_parser + space() + literal(",") + space());
            }
        }

        if (!gen_call_id_key.empty()) {
            auto gen_id_spec = parse_key_spec(gen_call_id_key);
            if (gen_id_spec.first.empty()) {
                auto gen_id_parser = atomic(
                    literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                    choice({
                        literal("\"") + tool_id(json_string_content()) + literal("\""),
                        tool_id(json_number())
                    })
                );
                tool_parser_body = tool_parser_body + optional(gen_id_parser + space() + literal(",") + space());
            }
        }

        auto nested_field = literal("\"" + nested_prefix + "\"") + space() + literal(":") + space() + nested_object;
        tool_parser_body = tool_parser_body + nested_field + space() + tool_close(literal("}"));

        tool_choices |= rule("tool-" + name, tool(tool_parser_body));
    }

    return tool_choices;
}

// Mode 3: Flat keys with optional ID fields and parameter ordering
common_peg_parser common_chat_peg_builder::build_json_tools_flat_keys(
    const nlohmann::json &           tools,
    const std::string &              effective_name_key,
    const std::string &              effective_args_key,
    const std::string &              call_id_key,
    const std::string &              gen_call_id_key,
    const std::vector<std::string> & parameters_order) {

    auto tool_choices    = choice();
    auto name_key_parser = literal("\"" + effective_name_key + "\"");
    auto args_key_parser = literal("\"" + effective_args_key + "\"");

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        nlohmann::json params = function.contains("parameters") ? function.at("parameters") : nlohmann::json::object();

        auto tool_name_ = name_key_parser + space() + literal(":") + space() +
                         literal("\"") + tool_name(literal(name)) + literal("\"");
        auto tool_args_ = args_key_parser + space() + literal(":") + space() +
                         tool_args(schema(json(), "tool-" + name + "-schema", params));

        // Build ID parsers if keys are provided
        common_peg_parser id_parser = eps();
        if (!call_id_key.empty()) {
            id_parser = atomic(
                literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(json_string_content()) + literal("\""),
                    tool_id(json_number())
                })
            );
        }

        common_peg_parser gen_id_parser = eps();
        if (!gen_call_id_key.empty()) {
            gen_id_parser = atomic(
                literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(json_string_content()) + literal("\""),
                    tool_id(json_number())
                })
            );
        }

        // Create (parser, key) pairs for all fields, then sort by parameters_order
        std::vector<std::pair<common_peg_parser, std::string>> parser_pairs;
        parser_pairs.emplace_back(tool_name_, effective_name_key);
        parser_pairs.emplace_back(tool_args_, effective_args_key);
        if (!call_id_key.empty()) {
            parser_pairs.emplace_back(optional(id_parser), call_id_key);
        }
        if (!gen_call_id_key.empty()) {
            parser_pairs.emplace_back(optional(gen_id_parser), gen_call_id_key);
        }

        std::sort(parser_pairs.begin(), parser_pairs.end(),
            [&parameters_order](const auto & a, const auto & b) {
                auto pos_a = std::find(parameters_order.begin(), parameters_order.end(), a.second);
                auto pos_b = std::find(parameters_order.begin(), parameters_order.end(), b.second);
                size_t idx_a = (pos_a == parameters_order.end()) ? parameters_order.size() : std::distance(parameters_order.begin(), pos_a);
                size_t idx_b = (pos_b == parameters_order.end()) ? parameters_order.size() : std::distance(parameters_order.begin(), pos_b);
                return idx_a < idx_b;
            });

        auto ordered_body = tool_open(literal("{")) + space();
        for (size_t i = 0; i < parser_pairs.size(); i++) {
            ordered_body = ordered_body + parser_pairs[i].first;
            if (i < parser_pairs.size() - 1) {
                ordered_body = ordered_body + space() + literal(",") + space();
            }
        }
        ordered_body = ordered_body + space() + tool_close(literal("}"));

        tool_choices |= rule("tool-" + name, tool(ordered_body));
    }

    return tool_choices;
}

common_peg_parser common_chat_peg_builder::standard_json_tools(
                                                       const std::string &              section_start,
                                                       const std::string &              section_end,
                                                       const nlohmann::json &           tools,
                                                       bool                             parallel_tool_calls,
                                                       bool                             force_tool_calls,
                                                       const std::string &              name_key,
                                                       const std::string &              args_key,
                                                       bool                             array_wrapped,
                                                       bool                             function_is_key,
                                                       const std::string &              call_id_key,
                                                       const std::string &              gen_call_id_key,
                                                       const std::vector<std::string> & parameters_order) {
    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    std::string effective_name_key = name_key.empty() ? "name" : name_key;
    std::string effective_args_key = args_key.empty() ? "arguments" : args_key;

    // Dispatch to the appropriate builder based on the JSON layout mode
    common_peg_parser tool_choices = eps();
    if (function_is_key) {
        tool_choices = build_json_tools_function_is_key(tools, args_key, effective_args_key, call_id_key, gen_call_id_key);
    } else {
        auto name_spec = parse_key_spec(effective_name_key);
        auto args_spec = parse_key_spec(effective_args_key);
        if (!name_spec.first.empty() || !args_spec.first.empty()) {
            tool_choices = build_json_tools_nested_keys(tools, effective_name_key, effective_args_key, call_id_key, gen_call_id_key);
        } else {
            tool_choices = build_json_tools_flat_keys(tools, effective_name_key, effective_args_key, call_id_key, gen_call_id_key, parameters_order);
        }
    }

    // Build the section with markers
    auto tool_calls = tool_choices;
    if (parallel_tool_calls) {
        tool_calls = tool_calls + zero_or_more(space() + literal(",") + space() + tool_choices);
    }

    if (array_wrapped) {
        tool_calls = literal("[") + space() + tool_calls + space() + literal("]");
    }

    auto section =
        trigger_rule("tool-call", literal(section_start) + space() + tool_calls + space() + literal(section_end));

    return force_tool_calls ? section : optional(section);
}
