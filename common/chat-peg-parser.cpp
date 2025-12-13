#include "chat-peg-parser.h"

#include "chat-auto-parser.h"
#include "ggml.h"

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

void common_chat_peg_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    arena.visit(result, [this](const common_peg_ast_node & node) { map(node); });
}

void common_chat_peg_mapper::map(const common_peg_ast_node & node) {
    bool is_reasoning = node.tag == common_chat_peg_builder::REASONING;
    bool is_content   = node.tag == common_chat_peg_builder::CONTENT;

    if (is_reasoning) { // GPT OSS can have more than 1 reasoning block, so concatenate here
        result.reasoning_content += std::string(trim_trailing_space(node.text));
    }

    if (is_content) {
        // Concatenate content from multiple content nodes (e.g., when reasoning markers
        // are preserved before content markers in reasoning_format=NONE mode)
        result.content += std::string(trim_trailing_space(node.text));
    }
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

common_peg_parser common_chat_peg_unified_builder::build_reasoning_block(const content_structure & cs,
                                                                         common_reasoning_format   reasoning_format,
                                                                         bool thinking_forced_open) {
    // If reasoning is explicitly disabled, return empty
    if (reasoning_format == COMMON_REASONING_FORMAT_NONE) {
        return eps();
    }

    // Get reasoning markers - use from content_structure or fallback for DEEPSEEK format
    std::string reason_start = cs.reasoning_start;
    std::string reason_end   = cs.reasoning_end;

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
    // But allow empty start marker if thinking is forced open (implicit start)
    if ((reason_start.empty() && !thinking_forced_open) || reason_end.empty()) {
        return eps();
    }

    if (thinking_forced_open) {
        // Mandatory reasoning: parse from current position to end marker
        auto parser = reasoning(until(reason_end)) + literal(reason_end);
        return rule("reasoning", reasoning_block(parser));
    }
    // Optional reasoning: may or may not appear
    // Also try <|START_THINKING|> style markers if standard markers don't match
    auto standard_reasoning =
        reasoning_block(literal(reason_start) + reasoning(until(reason_end)) + literal(reason_end));

    // For templates that use <|START_THINKING|> style markers
    if (reason_start == "<think>" && reason_end == "</think>") {
        auto alt_reasoning = reasoning_block(literal("<|START_THINKING|>") + reasoning(until("<|END_THINKING|>")) +
                                             literal("<|END_THINKING|>"));
        return optional(rule("reasoning", choice({ standard_reasoning, alt_reasoning })));
    }

    return optional(rule("reasoning", standard_reasoning));
}

common_peg_parser common_chat_peg_unified_builder::build_content_block(const content_structure & cs,
                                                                       common_reasoning_format   reasoning_format,
                                                                       const std::string &       tool_section_start) {
    GGML_UNUSED(tool_section_start);  // leaving for now just in case
    std::string content_start = cs.content_start;
    std::string content_end   = cs.content_end;

    // Add fallback content markers for DEEPSEEK format if not detected
    // Some templates use <response> tags for content when reasoning is enabled
    if ((reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK ||
         reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY) &&
        (content_start.empty() || content_end.empty())) {
        content_start = "<response>";
        content_end   = "</response>";
    }

    // Handle content markers with both start and end
    if (cs.content_mode != content_structure::CONTENT_PLAIN && !cs.content_start.empty() && !cs.content_end.empty()) {
        // Content is wrapped in markers
        if (reasoning_format == COMMON_REASONING_FORMAT_NONE) {
            // When reasoning_format=NONE, preserve any content before the content start marker
            // (this may include reasoning/thinking markers that the model generates).
            // This applies even if reasoning markers weren't detected by the analyzer.
            auto with_markers = content(until(cs.content_start)) + literal(cs.content_start) +
                                content(until(cs.content_end)) + literal(cs.content_end);
            // Fallback: content wrapped in end marker only (start marker might be in prompt)
            auto implicit_markers = content(until(cs.content_end)) + literal(cs.content_end);
            auto without_markers  = content(rest());
            return choice({ with_markers, implicit_markers, without_markers });
        }  // When reasoning is parsed separately, content starts directly after reasoning block
        auto with_markers     = literal(cs.content_start) + content(until(cs.content_end)) + literal(cs.content_end);
        auto implicit_markers = content(until(cs.content_end)) + literal(cs.content_end);
        auto without_markers  = content(rest());
        return choice({ with_markers, implicit_markers, without_markers });
    }

    // Handle content with only start marker (no end marker)
    // This is for formats like recipient-based (Functionary v3.2) where content is prefixed with
    // a marker but has no explicit closing marker - content ends at end of message or before tool calls
    if (cs.content_mode != content_structure::CONTENT_PLAIN && !cs.content_start.empty() && cs.content_end.empty()) {
        if (reasoning_format == COMMON_REASONING_FORMAT_NONE) {
            // Preserve any content before the start marker, then consume the marker and capture rest
            auto with_start_marker = content(until(cs.content_start)) + literal(cs.content_start) + content(rest());
            auto without_markers   = content(rest());
            return choice({ with_start_marker, without_markers });
        }  // Content starts directly after reasoning block
        auto with_start_marker = literal(cs.content_start) + content(rest());
        auto without_markers   = content(rest());
        return choice({ with_start_marker, without_markers });
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

common_peg_parser common_chat_peg_unified_builder::build_tool_section(const tool_call_structure & ts,
                                                                      const nlohmann::json &      tools,
                                                                      bool                        parallel_tool_calls,
                                                                      bool                        force_tool_calls) {
    if (!ts.supports_tools || !tools.is_array() || tools.empty()) {
        return eps();
    }

    // Build tool choices based on function format
    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        nlohmann::json params = function.contains("parameters") ? function.at("parameters") : nlohmann::json::object();

        tool_choices |= rule("tool-" + name, build_function(ts, name, params));
    }

    // Build the section with or without markers
    auto build_section = [&]() -> common_peg_parser {
        // Markdown code block format (Cohere Command-R Plus):
        // Action:\n```json\n[{...}]\n```
        if (ts.function_format == tool_call_structure::FUNC_MARKDOWN_CODE_BLOCK) {
            // Build the opening: "Action:\n```json"
            std::string code_fence_open = "```";
            if (!ts.code_block_language.empty()) {
                code_fence_open += ts.code_block_language;
            }

            auto opening = literal(ts.code_block_marker) + literal("\n") + literal(code_fence_open) + literal("\n");
            auto closing = literal("\n") + literal(ts.tool_section_end);  // "\n```"

            // Build the JSON array of tool calls
            // Don't use trigger_rule here since we're nested inside a sequence
            auto tools_array = literal("[") + space();
            if (parallel_tool_calls) {
                tools_array = tools_array + tool_choices;
                tools_array = tools_array + zero_or_more(space() + literal(",") + space() + tool_choices);
            } else {
                tools_array = tools_array + optional(tool_choices);
            }
            tools_array = tools_array + space() + literal("]");

            // Full section: Action:\n```json\n[{...}]\n```
            return trigger_rule("tool-call", opening + tools_array + closing);
        }

        // Recipient-based format (Functionary v3.2): >>>function_name\n{arguments}
        // Uses tool_section_start as delimiter, but no array wrapper or section markers
        if (ts.function_format == tool_call_structure::FUNC_RECIPIENT_BASED) {
            auto tool_call = trigger_rule("tool-call", tool_choices);
            if (parallel_tool_calls) {
                // Multiple tool calls: each starts with >>>
                return one_or_more(tool_call + space());
            }
            return tool_call;
        }

        if (!ts.tool_section_start.empty() && !ts.tool_section_end.empty()) {
            // Check if this format has SEPARATE section markers and per-call markers.
            // This happens when:
            // - Section markers wrap the ENTIRE section (e.g., <tool_calls_begin>...<tool_calls_end>)
            // - Function prefix contains its own per-call marker (e.g., <tool_call_begin>...)
            // Example: DeepSeek R1 with section and call markers, Kimi-K2 with prefixed-indexed format
            // We detect this by checking if function_prefix contains a per-call START marker
            // (indicated by words like "call_begin", "call_start", or similar patterns)
            bool has_separate_section_and_call_markers = false;

            // FUNC_PREFIXED_INDEXED and FUNC_BRACKET_TAG always have separate section and per-call markers
            if (ts.function_format == tool_call_structure::FUNC_PREFIXED_INDEXED ||
                ts.function_format == tool_call_structure::FUNC_BRACKET_TAG) {
                has_separate_section_and_call_markers = true;
            } else if (ts.function_format == tool_call_structure::FUNC_NAME_AS_KEY) {
                // FUNC_NAME_AS_KEY uses comma-separated JSON objects in an array
                // Format: [{"func1": args}, {"func2": args}]
                // The brackets are included in section markers
                auto tool_call  = trigger_rule("tool-call", tool_choices);
                auto tool_calls = tool_call;
                if (parallel_tool_calls) {
                    tool_calls = tool_call + zero_or_more(space() + literal(",") + space() + tool_call);
                }
                return literal(ts.tool_section_start) + space() + tool_calls + space() + literal(ts.tool_section_end);
            } else if (ts.function_format == tool_call_structure::FUNC_TAG_WITH_NAME && !ts.function_prefix.empty()) {
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
                auto tool_call  = trigger_rule("tool-call", tool_choices);
                auto tool_calls = parallel_tool_calls ? one_or_more(tool_call + space()) : tool_call;
                return literal(ts.tool_section_start) + space() + tool_calls + space() + literal(ts.tool_section_end);
            }  // Each tool call has its own wrapper: <tool_call>tool</tool_call>
            auto single_tool_section =
                trigger_rule("tool-call", literal(ts.tool_section_start) + space() + tool_choices + space() +
                                              literal(ts.tool_section_end));
            if (parallel_tool_calls) {
                // Multiple wrapped tool calls
                return one_or_more(single_tool_section + space());
            }
            return single_tool_section;
        }
        if (!ts.tool_section_start.empty()) {
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
        }  // No section markers (raw JSON format, e.g., Llama 3.1)
        // Use trigger rule since tool calls are identified by regex trigger on the grammar
        if (parallel_tool_calls) {
            return trigger_rule("tool-call", one_or_more(tool_choices + space()));
        }
        return trigger_rule("tool-call", tool_choices);
    };

    auto section = build_section();
    if (!force_tool_calls) {
        section = optional(section);
    }

    return section;
}

common_peg_parser common_chat_peg_unified_builder::build_function(const tool_call_structure & ts,
                                                                  const std::string &         name,
                                                                  const nlohmann::json &      schema) {
    auto args = build_arguments(ts, schema);

    switch (ts.function_format) {
        case tool_call_structure::FUNC_JSON_OBJECT:
            {
                // Build JSON object parser that accepts id field in either position:
                // - Before name: {"id": "...", "name": "X", "arguments": {...}} (R7B style)
                // - After args:  {"name": "X", "arguments": {...}, "id": "..."} (Mistral style)
                auto tool_name_ = json_member(ts.name_field, "\"" + tool_name(literal(name)) + "\"");
                auto tool_args_ = json_member(ts.args_field, tool_args(args));

                // id can appear before name or after args
                auto id_member = json_member(ts.id_field, tool_id(json_string()));
                auto id_before = ts.id_field.empty() ? eps() : optional(id_member << space() << "," << space());
                auto id_after  = ts.id_field.empty() ? eps() : optional(space() << "," << space() << id_member);

                return tool(tool_open(literal("{")) << space() << id_before  // optional id before name (R7B style)
                                                    << tool_name_ << space() << "," << space() << tool_args_
                                                    << id_after              // optional id after args (Mistral style)
                                                    << zero_or_more(space() << "," << space() << json_string()
                                                                            << space() << ":" << space() << json())
                                                    << space() << "}");
            }

        case tool_call_structure::FUNC_TAG_WITH_NAME:
            {
                // Build tag parser: <function=X>{...}</function>
                // Combine prefix + name + suffix into tool_open to ensure the tool is only created
                // when the FULL opening tag is confirmed. This prevents partial name matches during
                // incremental parsing (e.g., matching "special_function" when input is "special_function_")
                auto opening = literal(ts.function_prefix) + tool_name(literal(name)) + literal(ts.function_suffix);
                // Note: No space() before tool_close because function_close may start with newline
                // (e.g., "\n```<close_tag>") and space() would consume it, preventing the literal match
                return tool(tool_open(opening) + space() + tool_args(args) + tool_close(literal(ts.function_close)));
            }

        case tool_call_structure::FUNC_TAG_NAME_ONLY:
            {
                // Build tag parser: <X>...</X>
                // Combine < + name + > into tool_open to prevent partial matches
                auto opening = literal("<") + tool_name(literal(name)) + literal(">");
                return tool(tool_open(opening) + space() + tool_args(args) + space() +
                            tool_close(literal("</" + name + ">")));
            }

        case tool_call_structure::FUNC_PREFIXED_INDEXED:
            {
                // Build prefixed-indexed parser (e.g., Kimi-K2):
                // <|tool_call_begin|>functions.special_function:0<|tool_call_argument_begin|>{...}<|tool_call_end|>
                // The index number after : is ignored (we use zero_or_more(digit) to skip it)
                auto opening = literal(ts.per_call_start) + literal(ts.function_namespace) + tool_name(literal(name)) +
                               literal(":") + zero_or_more(chars("0-9", 1, 1)) +  // Skip the index
                               literal(ts.args_marker);
                return tool(tool_open(opening) + space() + tool_args(args) + space() +
                            tool_close(literal(ts.per_call_end)));
            }

        case tool_call_structure::FUNC_NAME_AS_KEY:
            {
                // Build name-as-key parser (e.g., Apertus):
                // {"function_name": {...arguments...}}
                // The function name IS the JSON key, and arguments are the value directly
                auto opening = literal("{\"") + tool_name(literal(name)) + literal("\":");
                return tool(tool_open(opening) + space() + tool_args(args) + space() + literal("}"));
            }

        case tool_call_structure::FUNC_BRACKET_TAG:
            {
                // Build bracket-tag parser (e.g., Mistral Small 3.2):
                // [TOOL_CALLS]function_name[CALL_ID]call_id[ARGS]{...}
                // per_call_start = "[TOOL_CALLS]"
                // id_marker = "[CALL_ID]"
                // args_marker = "[ARGS]"
                auto opening = literal(ts.per_call_start) + tool_name(literal(name));
                if (!ts.id_marker.empty()) {
                    // Add id_marker + id value (captured as tool_id)
                    opening = opening + literal(ts.id_marker) + tool_id(until(ts.args_marker));
                }
                if (!ts.args_marker.empty()) {
                    opening = opening + literal(ts.args_marker);
                }
                // No explicit closer for this format (EOS terminates)
                return tool(tool_open(opening) + space() + tool_args(args));
            }

        case tool_call_structure::FUNC_RECIPIENT_BASED:
            {
                // Build recipient-based parser (e.g., Functionary v3.2):
                // >>>function_name
                // {'param1': 'value1', 'param2': 'value2'}
                // tool_section_start = ">>>"
                // Function name directly follows ">>>" with newline, arguments are Python dict (parse as JSON)
                auto opening = literal(ts.tool_section_start) + tool_name(literal(name));
                // No explicit closer (newline + arguments, then EOS or next >>>)
                return tool(tool_open(opening) + space() + tool_args(args));
            }

        case tool_call_structure::FUNC_MARKDOWN_CODE_BLOCK:
            {
                // Build markdown code block parser (e.g., Cohere Command-R Plus):
                // Action:
                // ```json
                // [
                //     {
                //         "tool_name": "function_name",
                //         "parameters": {...}
                //     }
                // ]
                // ```
                // The individual function is a JSON object within the array
                auto tool_name_ = json_member(ts.name_field, "\"" + tool_name(literal(name)) + "\"");
                auto tool_args_ = json_member(ts.args_field, tool_args(args));

                // Build the JSON object: {"tool_name": "...", "parameters": {...}}
                // Use same pattern as FUNC_JSON_OBJECT: tool_open with atomic wrapper
                return tool(tool_open(literal("{")) << space() << tool_name_ << space() << "," << space() << tool_args_
                                                    << zero_or_more(space() << "," << space() << json_string()
                                                                            << space() << ":" << space() << json())
                                                    << space() << "}");
            }
    }

    return eps();
}

common_peg_parser common_chat_peg_unified_builder::build_arguments(const tool_call_structure & ts,
                                                                   const nlohmann::json &      params) {
    switch (ts.argument_format) {
        case tool_call_structure::ARGS_JSON:
            {
                // Standard JSON object arguments
                if (params.is_object()) {
                    return schema(json(), "args", params);
                }
                return json();
            }

        case tool_call_structure::ARGS_TAGGED:
            {
                // Tagged arguments: <param=key>value</param>
                if (!params.contains("properties") || params.at("properties").empty()) {
                    return eps();
                }

                auto arg_choice = choice();
                for (const auto & el : params.at("properties").items()) {
                    const std::string & prop_name   = el.key();
                    const auto &        prop_schema = el.value();

                    // Check if the schema declares this as a string type
                    bool is_string_type = prop_schema.contains("type") && prop_schema.at("type") == "string";

                    auto arg_name_parser = choice(
                        { literal(prop_name), literal("\"" + prop_name + "\""), literal("'" + prop_name + "'") });

                    // Use tool_arg_string_value for string types to prevent treating "[..." as JSON array
                    auto value_parser = is_string_type ? tool_arg_string_value(until(ts.arg_close))
                                                       : tool_arg_value(until(ts.arg_close));

                    auto arg_rule = tool_arg(tool_arg_open(literal(ts.arg_prefix)) + tool_arg_name(arg_name_parser) +
                                             literal(ts.arg_suffix) + value_parser +
                                             tool_arg_close(literal(ts.arg_close)) +
                                             (ts.arg_separator.empty() ? eps() : optional(literal(ts.arg_separator))));
                    arg_choice |= arg_rule;
                }
                return zero_or_more(arg_choice + space());
            }

        case tool_call_structure::ARGS_KEY_VALUE_TAGS:
            {
                // Key-value tag arguments (GLM-4.6 style):
                // <arg_key>key</arg_key>
                // <arg_value>value</arg_value>
                if (!params.contains("properties") || params.at("properties").empty()) {
                    return eps();
                }

                auto arg_choice = choice();
                for (const auto & el : params.at("properties").items()) {
                    const std::string & prop_name   = el.key();
                    const auto &        prop_schema = el.value();

                    // Check if the schema declares this as a string type
                    bool is_string_type = prop_schema.contains("type") && prop_schema.at("type") == "string";

                    // Parse: <arg_key>key</arg_key>\n<arg_value>value</arg_value>
                    // ts.arg_prefix = "<arg_key>", ts.arg_suffix = "</arg_key>", ts.arg_close = "</arg_value>"
                    // Use tool_arg_string_value for string types to prevent treating "[..." as JSON array
                    auto value_parser = is_string_type ? tool_arg_string_value(until(ts.arg_close))
                                                       : tool_arg_value(until(ts.arg_close));

                    auto arg_rule = tool_arg(tool_arg_open(literal(ts.arg_prefix)) + tool_arg_name(literal(prop_name)) +
                                             literal(ts.arg_suffix) +  // </arg_key>
                                             space() + literal("<arg_value>") + value_parser +
                                             tool_arg_close(literal(ts.arg_close)));
                    arg_choice |= arg_rule;
                }
                return zero_or_more(arg_choice + space());
            }
    }

    return eps();
}

common_peg_parser common_chat_peg_unified_builder::standard_json_tools(const std::string &    section_start,
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
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        nlohmann::json params = function.contains("parameters") ? function.at("parameters") : nlohmann::json::object();

        // Build JSON object parser: {"name": "X", "arguments": {...}}
        auto tool_name_ = json_member("name", "\"" + tool_name(literal(name)) + "\"");
        auto tool_args_ = json_member("arguments", tool_args(schema(json(), "tool-" + name + "-schema", params)));

        auto tool_parser =
            tool(tool_open(literal("{")) << space() << tool_name_ << space() << "," << space() << tool_args_
                                         << zero_or_more(space() << "," << space() << json_string() << space() << ":"
                                                                 << space() << json())
                                         << space() << "}");

        tool_choices |= rule("tool-" + name, tool_parser);
    }

    // Build the section with markers
    auto tool_calls = tool_choices;
    if (parallel_tool_calls) {
        tool_calls = tool_calls + zero_or_more(space() + literal(",") + space() + tool_choices);
    }

    auto section =
        trigger_rule("tool-call", literal(section_start) + space() + tool_calls + space() + literal(section_end));

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

void common_chat_peg_unified_mapper::from_ast(const common_peg_ast_arena &    arena,
                                              const common_peg_parse_result & parse_result_arg) {
    // Call base class to visit all nodes
    common_chat_peg_mapper::from_ast(arena, parse_result_arg);

    // Flush any pending tool call that was started but never got a name
    // This happens during partial parsing when the tool call is incomplete
    if (pending_tool_call.has_value()) {
        // Transfer any buffered arguments
        if (!args_buffer.empty()) {
            pending_tool_call->arguments = args_buffer;
        }
        // Close any open quotes in buffered args
        if (buffer_needs_closing_quote && !pending_tool_call->arguments.empty()) {
            pending_tool_call->arguments += "\"";
        }
        // Add the incomplete tool call to results
        result.tool_calls.push_back(pending_tool_call.value());
        pending_tool_call.reset();
    }
}

void common_chat_peg_unified_mapper::map(const common_peg_ast_node & node) {
    // First call base class for reasoning/content handling
    common_chat_peg_mapper::map(node);

    // Handle tool-related tags (unified version supporting both JSON and tagged formats)
    bool is_tool_open  = node.tag == common_chat_peg_unified_builder::TOOL_OPEN;
    bool is_tool_close = node.tag == common_chat_peg_unified_builder::TOOL_CLOSE;
    bool is_tool_name  = node.tag == common_chat_peg_unified_builder::TOOL_NAME;
    bool is_tool_id    = node.tag == common_chat_peg_unified_builder::TOOL_ID;
    bool is_tool_args  = node.tag == common_chat_peg_unified_builder::TOOL_ARGS;
    bool is_arg_open   = node.tag == common_chat_peg_unified_builder::TOOL_ARG_OPEN;
    bool is_arg_close  = node.tag == common_chat_peg_unified_builder::TOOL_ARG_CLOSE;
    bool is_arg_name         = node.tag == common_chat_peg_unified_builder::TOOL_ARG_NAME;
    bool is_arg_value        = node.tag == common_chat_peg_unified_builder::TOOL_ARG_VALUE;
    bool is_arg_string_value = node.tag == common_chat_peg_unified_builder::TOOL_ARG_STRING_VALUE;

    if (is_tool_open) {
        // Don't create tool call yet - wait for name to be known
        // This prevents sending incomplete tool calls in streaming mode
        pending_tool_call = common_chat_tool_call();
        current_tool      = &pending_tool_call.value();
        arg_count         = 0;
        // Clear the arguments buffer for the new tool
        args_buffer.clear();
        needs_closing_quote        = false;
        buffer_needs_closing_quote = false;
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
        // Now that we have the name, we can populate the arguments from the buffer
        if (!args_buffer.empty()) {
            current_tool->arguments = args_buffer;
            args_buffer.clear();
        } else if (current_tool->arguments.empty()) {
            // Initialize arguments if we're using tagged format and no buffered args
            current_tool->arguments = "{";
        }
        // Now that we have the name, add the tool call to the result
        if (pending_tool_call.has_value()) {
            result.tool_calls.push_back(pending_tool_call.value());
            pending_tool_call.reset();
            current_tool = &result.tool_calls.back();
        }
    }

    if (is_tool_args && current_tool) {
        // For JSON format, the arguments come as a complete JSON object
        // For tagged format, we build up arguments from individual arg_name/arg_value nodes
        // Check if this looks like JSON (starts with {) vs tagged format (starts with <)
        auto text = trim_trailing_space(node.text);
        if (!text.empty() && text.front() == '{') {
            // If we have the tool name, populate directly; otherwise buffer
            if (!current_tool->name.empty()) {
                current_tool->arguments = std::string(text);
            } else {
                args_buffer = std::string(text);
            }
        }
        // If it's tagged format, we ignore this and let arg_name/arg_value build up the JSON
    }

    if (is_arg_open) {
        // Reset for new argument
        if (!current_tool->name.empty()) {
            needs_closing_quote = false;
        } else {
            buffer_needs_closing_quote = false;
        }
    }

    if (is_arg_name && current_tool) {
        std::string arg_entry;
        if (arg_count > 0) {
            arg_entry = ",";
        }
        arg_entry += json(trim(node.text)).dump() + ":";
        ++arg_count;

        // If we have the tool name, add directly; otherwise buffer
        if (!current_tool->name.empty()) {
            current_tool->arguments += arg_entry;
        } else {
            if (args_buffer.empty()) {
                args_buffer = "{";
            }
            args_buffer += arg_entry;
        }
    }

    if ((is_arg_value || is_arg_string_value) && current_tool) {
        std::string value_content = std::string(trim_trailing_space(trim_leading_space(node.text, 1), 1));

        std::string value_to_add;
        if (!value_content.empty()) {
            // For potential containers, normalize Python-style single quotes to JSON double quotes first
            // This ensures consistent output during both partial and final parsing
            // Note: is_arg_string_value means the schema explicitly declares this as a string type,
            // so we should NOT treat it as a potential container even if it starts with [ or {
            bool is_potential_container = !is_arg_string_value &&
                (value_content[0] == '[' || value_content[0] == '{');
            if (is_potential_container) {
                value_content = normalize_quotes_to_json(value_content);
            }

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
                    value_to_add = escaped;
                    if (!current_tool->name.empty()) {
                        needs_closing_quote = true;
                    } else {
                        buffer_needs_closing_quote = true;
                    }
                } else {
                    // For non-string values (number, bool, null, object, array), add raw value content
                    // Using raw content instead of dump() ensures monotonicity for streaming
                    // (prevents issues with spaces being removed by dump())
                    value_to_add = value_content;
                }
            } catch (...) {
                // JSON parsing failed - content is either incomplete (partial) or not valid JSON
                // Note: potential containers were already normalized above, so value_content
                // already has double quotes if it started with [ or {

                if (node.is_partial && is_potential_container) {
                    // During incremental parsing, if it looks like a JSON container, don't wrap in quotes yet
                    // and don't escape. Just pass through the (already normalized) content.
                    value_to_add = value_content;
                } else {
                    // Not valid JSON and NOT a potential partial container - treat as string value
                    // Add opening quote if not already in a string
                    if (!current_tool->name.empty()) {
                        if (!needs_closing_quote) {
                            value_to_add        = "\"";
                            needs_closing_quote = true;
                        }
                    } else {
                        if (!buffer_needs_closing_quote) {
                            value_to_add               = "\"";
                            buffer_needs_closing_quote = true;
                        }
                    }
                    // Escape special characters in the string content
                    std::string escaped = json(value_content).dump();
                    // Remove the surrounding quotes from the escaped string
                    if (escaped.size() >= 2 && escaped.front() == '"' && escaped.back() == '"') {
                        escaped = escaped.substr(1, escaped.size() - 2);
                    }
                    value_to_add += escaped;
                }
            }
        }

        // If we have the tool name, add directly; otherwise buffer
        if (!current_tool->name.empty()) {
            current_tool->arguments += value_to_add;
        } else {
            if (args_buffer.empty()) {
                args_buffer = "{";
            }
            args_buffer += value_to_add;
        }
    }

    if (is_arg_close && current_tool) {
        if (!current_tool->name.empty()) {
            if (needs_closing_quote) {
                current_tool->arguments += "\"";
                needs_closing_quote = false;
            }
        } else {
            if (buffer_needs_closing_quote) {
                if (args_buffer.empty()) {
                    args_buffer = "{";
                }
                args_buffer += "\"";
                buffer_needs_closing_quote = false;
            }
        }
    }

    if (is_tool_close && current_tool) {
        if (!current_tool->name.empty()) {
            if (needs_closing_quote) {
                current_tool->arguments += "\"";
                needs_closing_quote = false;
            }
            // Close the arguments object if using tagged format
            if (!current_tool->arguments.empty() && current_tool->arguments.back() != '}') {
                current_tool->arguments += "}";
            }
            // If we have a pending tool call that wasn't added yet, add it now
            if (pending_tool_call.has_value()) {
                result.tool_calls.push_back(pending_tool_call.value());
                pending_tool_call.reset();
            }
        } else {
            // We're closing a tool without a name - flush the buffer
            if (!args_buffer.empty()) {
                current_tool->arguments = args_buffer;
                args_buffer.clear();
            }
            if (buffer_needs_closing_quote) {
                current_tool->arguments += "\"";
                buffer_needs_closing_quote = false;
            }
            // Close the arguments object if using tagged format
            if (!current_tool->arguments.empty() && current_tool->arguments.back() != '}') {
                current_tool->arguments += "}";
            }
            // Don't add to result if no name - this prevents incomplete tool calls
            pending_tool_call.reset();
        }
    }
}
