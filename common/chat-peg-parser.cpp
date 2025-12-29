#include "chat-peg-parser.h"
#include "chat-auto-parser.h"  // For ContentStructure, ToolCallStructure

#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

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

// ============================================================================
// Base Mapper Implementation
// ============================================================================

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
        // Concatenate content from multiple content nodes (e.g., when reasoning markers
        // are preserved before content markers in reasoning_format=NONE mode)
        result.content += std::string(trim_trailing_space(node.text));
    }
}

// ============================================================================
// Base Builder Helper
// ============================================================================

common_peg_parser common_chat_peg_builder::tag_with_safe_content(const std::string &       tag_name,
                                                                 const std::string &       marker,
                                                                 const common_peg_parser & p) {
    if (marker.empty()) {
        return zero_or_more(choice({ p, rule(tag_name, content(any())) }));
    }
    auto content_chunk = rule(tag_name, content(negate(literal(marker)) + any() + until(marker)));
    return zero_or_more(choice({ p, content_chunk }));
}

// ============================================================================
// Unified Builder Implementation
// ============================================================================

common_peg_parser common_chat_peg_unified_builder::build_reasoning_block(
    const ContentStructure & cs,
    common_reasoning_format  reasoning_format,
    bool                     thinking_forced_open) {

    // If reasoning is explicitly disabled, return empty
    if (reasoning_format == COMMON_REASONING_FORMAT_NONE) {
        return eps();
    }

    // Get reasoning markers - use from ContentStructure or fallback for DEEPSEEK format
    std::string reason_start = cs.reasoning_start;
    std::string reason_end = cs.reasoning_end;

    // If DEEPSEEK format is specified but markers weren't detected, use fallback markers
    if ((reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK ||
         reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY) &&
        (reason_start.empty() || reason_end.empty())) {
        // Try standard DeepSeek markers
        if (reason_start.empty()) {
            reason_start = "<think>";
        }
        if (reason_end.empty()) {
            reason_end = "</think>";
        }
    }

    // If still no markers, return empty
    if (reason_start.empty() || reason_end.empty()) {
        return eps();
    }

    if (thinking_forced_open) {
        // Mandatory reasoning: parse from current position to end marker
        return rule("reasoning",
                    reasoning_block(reasoning(until(reason_end)) + literal(reason_end)));
    } else {
        // Optional reasoning: may or may not appear
        // Also try <|START_THINKING|> style markers if standard markers don't match
        auto standard_reasoning = reasoning_block(literal(reason_start) + reasoning(until(reason_end)) +
                                                   literal(reason_end));

        // For templates that use <|START_THINKING|> style markers
        if (reason_start == "<think>" && reason_end == "</think>") {
            auto alt_reasoning = reasoning_block(literal("<|START_THINKING|>") +
                                                  reasoning(until("<|END_THINKING|>")) +
                                                  literal("<|END_THINKING|>"));
            return optional(rule("reasoning", choice({ standard_reasoning, alt_reasoning })));
        }

        return optional(rule("reasoning", standard_reasoning));
    }
}

common_peg_parser common_chat_peg_unified_builder::build_content_block(
    const ContentStructure & cs,
    common_reasoning_format  reasoning_format) {

    std::string content_start = cs.content_start;
    std::string content_end = cs.content_end;

    // Add fallback content markers for DEEPSEEK format if not detected
    // Some templates use <response> tags for content when reasoning is enabled
    if ((reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK ||
         reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY) &&
        (content_start.empty() || content_end.empty())) {
        content_start = "<response>";
        content_end = "</response>";
    }

    // Handle content markers
    if (cs.content_mode != ContentStructure::CONTENT_PLAIN && !cs.content_start.empty() && !cs.content_end.empty()) {
        // Content is wrapped in markers
        if (reasoning_format == COMMON_REASONING_FORMAT_NONE) {
            // When reasoning_format=NONE, preserve any content before the content start marker
            // (this may include reasoning/thinking markers that the model generates).
            // This applies even if reasoning markers weren't detected by the analyzer.
            auto with_markers = content(until(cs.content_start)) + literal(cs.content_start) +
                                content(until(cs.content_end)) + literal(cs.content_end);
            auto without_markers = content(rest());
            return choice({ with_markers, without_markers });
        } else {
            // When reasoning is parsed separately, content starts directly after reasoning block
            auto with_markers    = literal(cs.content_start) + content(until(cs.content_end)) + literal(cs.content_end);
            auto without_markers = content(rest());
            return choice({ with_markers, without_markers });
        }
    }

    // For DEEPSEEK format, try fallback content markers even if not detected
    if (!content_start.empty() && !content_end.empty()) {
        auto with_markers    = literal(content_start) + content(until(content_end)) + literal(content_end);
        auto without_markers = content(rest());
        return choice({ with_markers, without_markers });
    }

    // Plain content - capture rest
    return content(rest());
}

common_peg_parser common_chat_peg_unified_builder::build_tool_section(
    const ToolCallStructure & ts,
    const nlohmann::json &    tools,
    bool                      parallel_tool_calls,
    bool                      force_tool_calls) {

    if (!ts.supports_tools || !tools.is_array() || tools.empty()) {
        return eps();
    }

    // Build tool choices based on function format
    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto & function = tool_def.at("function");
        std::string  name     = function.at("name");
        nlohmann::json params = function.contains("parameters") ? function.at("parameters") : nlohmann::json::object();

        tool_choices |= rule("tool-" + name, build_function(ts, name, params));
    }

    // Build the section with or without markers
    auto build_section = [&]() -> common_peg_parser {
        if (!ts.tool_section_start.empty() && !ts.tool_section_end.empty()) {
            // Check if this format has SEPARATE section markers and per-call markers.
            // This happens when:
            // - Section markers wrap the ENTIRE section (e.g., <tool_calls_begin>...<tool_calls_end>)
            // - Function prefix contains its own per-call marker (e.g., <tool_call_begin>...)
            // Example: DeepSeek R1 with section and call markers, Kimi-K2 with prefixed-indexed format
            // We detect this by checking if function_prefix contains a per-call START marker
            // (indicated by words like "call_begin", "call_start", or similar patterns)
            bool has_separate_section_and_call_markers = false;

            // FUNC_PREFIXED_INDEXED always has separate section and per-call markers
            if (ts.function_format == ToolCallStructure::FUNC_PREFIXED_INDEXED) {
                has_separate_section_and_call_markers = true;
            } else if (ts.function_format == ToolCallStructure::FUNC_NAME_AS_KEY) {
                // FUNC_NAME_AS_KEY uses comma-separated JSON objects in an array
                // Format: [{"func1": args}, {"func2": args}]
                // The brackets are included in section markers
                auto tool_call = trigger_rule("tool-call", tool_choices);
                auto tool_calls = tool_call;
                if (parallel_tool_calls) {
                    tool_calls = tool_call + zero_or_more(space() + literal(",") + space() + tool_call);
                }
                return literal(ts.tool_section_start) + space() + tool_calls + space() + literal(ts.tool_section_end);
            } else if (ts.function_format == ToolCallStructure::FUNC_TAG_WITH_NAME && !ts.function_prefix.empty()) {
                // Check if function_prefix contains a per-call marker like "<tool_call_begin>"
                // This differentiates DeepSeek R1 (where function_prefix has its own call marker)
                // from Nemotron (where function_prefix is just "<function=")
                // DeepSeek pattern: function_prefix = "<｜tool▁call▁begin｜>function<｜tool▁sep｜>"
                // Nemotron pattern: function_prefix = "<function="
                bool prefix_has_call_marker = ts.function_prefix.find("call") != std::string::npos &&
                                             (ts.function_prefix.find("begin") != std::string::npos ||
                                              ts.function_prefix.find("start") != std::string::npos);
                if (prefix_has_call_marker) {
                    has_separate_section_and_call_markers = true;
                }
            }
            if (has_separate_section_and_call_markers) {
                // Section markers wrap all calls, per-call markers are in function_prefix/close
                // Format: <section_start> <call1> <call2> ... <section_end>
                auto tool_call = trigger_rule("tool-call", tool_choices);
                auto tool_calls = parallel_tool_calls ?
                    one_or_more(tool_call + space()) : tool_call;
                return literal(ts.tool_section_start) + space() + tool_calls + space() + literal(ts.tool_section_end);
            } else {
                // Each tool call has its own wrapper: <tool_call>tool</tool_call>
                auto single_tool_section = trigger_rule("tool-call",
                                                         literal(ts.tool_section_start) + space() + tool_choices + space() +
                                                             literal(ts.tool_section_end));
                if (parallel_tool_calls) {
                    // Multiple wrapped tool calls
                    return one_or_more(single_tool_section + space());
                } else {
                    return single_tool_section;
                }
            }
        } else if (!ts.tool_section_start.empty()) {
            // Start marker only (no end marker) - e.g., <|tool_call|>[...]
            // Wrap all tool calls in an array after the start marker
            auto tools_array = literal("[") + space();
            if (parallel_tool_calls) {
                tools_array = tools_array + tool_choices;
                tools_array = tools_array + zero_or_more(space() + literal(",") + space() + tool_choices);
            } else {
                tools_array = tools_array + optional(tool_choices);
            }
            tools_array = tools_array + space() + literal("]");

            return trigger_rule("tool-call", literal(ts.tool_section_start) + tools_array);
        } else {
            // No section markers (raw JSON format, e.g., Llama 3.1)
            // Use trigger rule since tool calls are identified by regex trigger on the grammar
            if (parallel_tool_calls) {
                return trigger_rule("tool-call", one_or_more(tool_choices + space()));
            } else {
                return trigger_rule("tool-call", tool_choices);
            }
        }
    };

    auto section = build_section();
    if (!force_tool_calls) {
        section = optional(section);
    }

    return section;
}

common_peg_parser common_chat_peg_unified_builder::build_function(
    const ToolCallStructure & ts,
    const std::string &       name,
    const nlohmann::json &    params) {

    auto args = build_arguments(ts, params);

    switch (ts.function_format) {
        case ToolCallStructure::FUNC_JSON_OBJECT: {
            // Build JSON object parser that accepts id field in either position:
            // - Before name: {"id": "...", "name": "X", "arguments": {...}} (R7B style)
            // - After args:  {"name": "X", "arguments": {...}, "id": "..."} (Mistral style)
            auto tool_name_ = json_member(ts.name_field, "\"" + tool_name(literal(name)) + "\"");
            auto tool_args_ = json_member(ts.args_field, tool_args(args));

            // id can appear before name or after args
            auto id_member = json_member(ts.id_field, tool_id(json_string()));
            auto id_before = ts.id_field.empty() ? eps() :
                             optional(id_member << space() << "," << space());
            auto id_after  = ts.id_field.empty() ? eps() :
                             optional(space() << "," << space() << id_member);

            return tool(tool_open(literal("{")) << space()
                                                << id_before  // optional id before name (R7B style)
                                                << tool_name_ << space() << "," << space()
                                                << tool_args_
                                                << id_after   // optional id after args (Mistral style)
                                                << zero_or_more(space() << "," << space() << json_string() << space()
                                                                        << ":" << space() << json())
                                                << space() << "}");
        }

        case ToolCallStructure::FUNC_TAG_WITH_NAME: {
            // Build tag parser: <function=X>{...}</function>
            // Combine prefix + name + suffix into tool_open to ensure the tool is only created
            // when the FULL opening tag is confirmed. This prevents partial name matches during
            // incremental parsing (e.g., matching "special_function" when input is "special_function_")
            auto opening = literal(ts.function_prefix) + tool_name(literal(name)) + literal(ts.function_suffix);
            // Note: No space() before tool_close because function_close may start with newline
            // (e.g., "\n```<close_tag>") and space() would consume it, preventing the literal match
            return tool(tool_open(opening) + space() + tool_args(args) +
                        tool_close(literal(ts.function_close)));
        }

        case ToolCallStructure::FUNC_TAG_NAME_ONLY: {
            // Build tag parser: <X>...</X>
            // Combine < + name + > into tool_open to prevent partial matches
            auto opening = literal("<") + tool_name(literal(name)) + literal(">");
            return tool(tool_open(opening) + space() + tool_args(args) +
                        space() + tool_close(literal("</" + name + ">")));
        }

        case ToolCallStructure::FUNC_PREFIXED_INDEXED: {
            // Build prefixed-indexed parser (e.g., Kimi-K2):
            // <|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{...}<|tool_call_end|>
            // The index number after : is ignored (we use zero_or_more(digit) to skip it)
            auto opening = literal(ts.per_call_start) +
                           literal(ts.function_namespace) +
                           tool_name(literal(name)) +
                           literal(":") +
                           zero_or_more(chars("0-9", 1, 1)) +  // Skip the index
                           literal(ts.args_marker);
            return tool(tool_open(opening) + space() + tool_args(args) +
                        space() + tool_close(literal(ts.per_call_end)));
        }

        case ToolCallStructure::FUNC_NAME_AS_KEY: {
            // Build name-as-key parser (e.g., Apertus):
            // {"function_name": {...arguments...}}
            // The function name IS the JSON key, and arguments are the value directly
            auto opening = literal("{\"") + tool_name(literal(name)) + literal("\":");
            return tool(tool_open(opening) + space() + tool_args(args) + space() + literal("}"));
        }
    }

    return eps();
}

common_peg_parser common_chat_peg_unified_builder::build_arguments(
    const ToolCallStructure & ts,
    const nlohmann::json &    params) {

    switch (ts.argument_format) {
        case ToolCallStructure::ARGS_JSON: {
            // Standard JSON object arguments
            if (params.is_object()) {
                return schema(json(), "args", params);
            }
            return json();
        }

        case ToolCallStructure::ARGS_TAGGED: {
            // Tagged arguments: <param=key>value</param>
            if (!params.contains("properties") || params.at("properties").empty()) {
                return eps();
            }

            auto arg_choice = choice();
            for (const auto & el : params.at("properties").items()) {
                const std::string & prop_name = el.key();

                auto arg_name_parser =
                    choice({ literal(prop_name), literal("\"" + prop_name + "\""), literal("'" + prop_name + "'") });

                auto arg_rule = tool_arg(tool_arg_open(literal(ts.arg_prefix)) + tool_arg_name(arg_name_parser) +
                                         literal(ts.arg_suffix) + tool_arg_value(until(ts.arg_close)) +
                                         tool_arg_close(literal(ts.arg_close)) +
                                         (ts.arg_separator.empty() ? eps() : optional(literal(ts.arg_separator))));
                arg_choice |= arg_rule;
            }
            return zero_or_more(arg_choice + space());
        }

        case ToolCallStructure::ARGS_KEY_VALUE_TAGS: {
            // Key-value tag arguments (GLM-4.6 style):
            // <arg_key>key</arg_key>
            // <arg_value>value</arg_value>
            if (!params.contains("properties") || params.at("properties").empty()) {
                return eps();
            }

            auto arg_choice = choice();
            for (const auto & el : params.at("properties").items()) {
                const std::string & prop_name = el.key();

                // Parse: <arg_key>key</arg_key>\n<arg_value>value</arg_value>
                // ts.arg_prefix = "<arg_key>", ts.arg_suffix = "</arg_key>", ts.arg_close = "</arg_value>"
                auto arg_rule = tool_arg(
                    tool_arg_open(literal(ts.arg_prefix)) +
                    tool_arg_name(literal(prop_name)) +
                    literal(ts.arg_suffix) +  // </arg_key>
                    space() +
                    literal("<arg_value>") +
                    tool_arg_value(until(ts.arg_close)) +
                    tool_arg_close(literal(ts.arg_close))
                );
                arg_choice |= arg_rule;
            }
            return zero_or_more(arg_choice + space());
        }
    }

    return eps();
}

common_peg_parser common_chat_peg_unified_builder::standard_json_tools(
    const std::string &    section_start,
    const std::string &    section_end,
    const nlohmann::json & tools,
    bool                   parallel_tool_calls,
    bool                   force_tool_calls) {

    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    // Build tool choices for JSON format
    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto & function = tool_def.at("function");
        std::string  name     = function.at("name");
        nlohmann::json params = function.contains("parameters") ? function.at("parameters") : nlohmann::json::object();

        // Build JSON object parser: {"name": "X", "arguments": {...}}
        auto tool_name_ = json_member("name", "\"" + tool_name(literal(name)) + "\"");
        auto tool_args_ = json_member("arguments", tool_args(schema(json(), "tool-" + name + "-schema", params)));

        auto tool_parser = tool(tool_open(literal("{")) << space() << tool_name_ << space() << "," << space()
                                                        << tool_args_
                                                        << zero_or_more(space() << "," << space() << json_string() << space()
                                                                                << ":" << space() << json())
                                                        << space() << "}");

        tool_choices |= rule("tool-" + name, tool_parser);
    }

    // Build the section with markers
    auto section = parallel_tool_calls
        ? trigger_rule("tool-call",
                      literal(section_start) + space() + one_or_more(tool_choices + space()) +
                          literal(section_end))
        : trigger_rule("tool-call",
                      literal(section_start) + space() + tool_choices + space() +
                          literal(section_end));

    return force_tool_calls ? section : optional(section);
}

common_peg_parser common_chat_peg_unified_builder::standard_constructed_tools(
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

    std::string section_start     = get_marker("tool_call_start_marker", "<tool_call>");
    std::string section_end       = get_marker("tool_call_end_marker", "</tool_call>");
    std::string func_opener       = get_marker("function_opener", "<function=");
    std::string func_name_suffix  = get_marker("function_name_suffix", ">");
    std::string func_closer       = get_marker("function_closer", "</function>");
    std::string param_key_prefix  = get_marker("parameter_key_prefix", "<param=");
    std::string param_key_suffix  = get_marker("parameter_key_suffix", ">");
    std::string param_closer      = get_marker("parameter_closer", "</param>");

    // Build tool choices for tagged format
    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto & function = tool_def.at("function");
        std::string  name     = function.at("name");
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
                               space() + tool_args(args) + space() +
                               tool_close(literal(func_closer)));

        tool_choices |= rule("tool-" + name, tool_parser);
    }

    // Build the section with markers
    auto section = parallel_tool_calls
        ? trigger_rule("tool-call",
                      literal(section_start) + space() + one_or_more(tool_choices + space()) +
                          literal(section_end))
        : trigger_rule("tool-call",
                      literal(section_start) + space() + tool_choices + space() +
                          literal(section_end));

    return force_tool_calls ? section : optional(section);
}

// ============================================================================
// Unified Mapper Implementation
// ============================================================================

void common_chat_peg_unified_mapper::map(const common_peg_ast_node & node) {
    // First call base class for reasoning/content handling
    common_chat_peg_mapper::map(node);

    // Handle tool-related tags (unified version supporting both JSON and tagged formats)
    bool is_tool_open   = node.tag == common_chat_peg_unified_builder::TOOL_OPEN;
    bool is_tool_close  = node.tag == common_chat_peg_unified_builder::TOOL_CLOSE;
    bool is_tool_name   = node.tag == common_chat_peg_unified_builder::TOOL_NAME;
    bool is_tool_id     = node.tag == common_chat_peg_unified_builder::TOOL_ID;
    bool is_tool_args   = node.tag == common_chat_peg_unified_builder::TOOL_ARGS;
    bool is_arg_open    = node.tag == common_chat_peg_unified_builder::TOOL_ARG_OPEN;
    bool is_arg_close   = node.tag == common_chat_peg_unified_builder::TOOL_ARG_CLOSE;
    bool is_arg_name    = node.tag == common_chat_peg_unified_builder::TOOL_ARG_NAME;
    bool is_arg_value   = node.tag == common_chat_peg_unified_builder::TOOL_ARG_VALUE;

    if (is_tool_open) {
        result.tool_calls.emplace_back();
        current_tool = &result.tool_calls.back();
        arg_count    = 0;
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
        // Initialize arguments if we're using tagged format
        if (current_tool->arguments.empty()) {
            current_tool->arguments = "{";
        }
    }

    if (is_tool_args && current_tool) {
        // For JSON format, the arguments come as a complete JSON object
        // For tagged format, we build up arguments from individual arg_name/arg_value nodes
        // Check if this looks like JSON (starts with {) vs tagged format (starts with <)
        auto text = trim_trailing_space(node.text);
        if (!text.empty() && text.front() == '{') {
            current_tool->arguments = std::string(text);
        }
        // If it's tagged format, we ignore this and let arg_name/arg_value build up the JSON
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

    if (is_arg_value && current_tool) {
        std::string value_content = std::string(trim_trailing_space(node.text));

        // During incremental parsing, the value might contain the start or full closing tag
        // Strip any content that looks like a closing tag (starts with </)
        // This handles both partial tags like "</para" and full tags like "</parameter>"
        // Use find() not rfind() to get the FIRST occurrence - handles cases like
        // "1\n</parameter>\n</" where we want to strip from the first </
        auto tag_start = value_content.find("</");
        if (tag_start != std::string::npos) {
            value_content = value_content.substr(0, tag_start);
        } else {
            // Also handle just < at the end (start of any tag)
            if (!value_content.empty() && value_content.back() == '<') {
                value_content.pop_back();
            }
        }

        // Trim trailing whitespace again after stripping
        while (!value_content.empty() && std::isspace(static_cast<unsigned char>(value_content.back()))) {
            value_content.pop_back();
        }

        if (!value_content.empty()) {
            // Try to parse as JSON value (number, bool, null, object, array)
            // For strings, we need special handling to support incremental parsing
            try {
                json parsed = json::parse(value_content);
                if (parsed.is_string()) {
                    // For string values, don't add closing quote yet (added by arg_close)
                    // This ensures incremental parsing produces monotonic arguments
                    std::string escaped = parsed.dump();
                    // Remove the trailing quote
                    if (!escaped.empty() && escaped.back() == '"') {
                        escaped.pop_back();
                    }
                    current_tool->arguments += escaped;
                    needs_closing_quote = true;
                } else {
                    // For non-string values (number, bool, null, object, array), add complete value
                    current_tool->arguments += parsed.dump();
                }
            } catch (...) {
                // Not valid JSON - treat as string value
                // Add opening quote if not already in a string
                if (!needs_closing_quote) {
                    current_tool->arguments += "\"";
                    needs_closing_quote = true;
                }
                // Escape special characters in the string content
                std::string escaped = json(value_content).dump();
                // Remove the surrounding quotes from the escaped string
                if (escaped.size() >= 2 && escaped.front() == '"' && escaped.back() == '"') {
                    escaped = escaped.substr(1, escaped.size() - 2);
                }
                current_tool->arguments += escaped;
            }
        }
    }

    if (is_arg_close && current_tool) {
        if (needs_closing_quote) {
            current_tool->arguments += "\"";
            needs_closing_quote = false;
        }
    }

    if (is_tool_close && current_tool) {
        if (needs_closing_quote) {
            current_tool->arguments += "\"";
            needs_closing_quote = false;
        }
        // Close the arguments object if using tagged format
        if (!current_tool->arguments.empty() && current_tool->arguments.back() != '}') {
            current_tool->arguments += "}";
        }
    }
}
