#include "parsers.h"

#include "unicode.h"

#include <algorithm>
#include <map>
#include <string_view>

namespace {

// Recover complete XML calls while PEG handles reasoning and the generation grammar.
constexpr std::string_view function_open = "<function";
constexpr std::string_view function_close = "</function>";
constexpr std::string_view param_open = "<param";
constexpr std::string_view param_close = "</param>";
constexpr std::string_view cdata_open = "<![CDATA[";
constexpr std::string_view cdata_close = "]]>";

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.substr(0, prefix.size()) == prefix;
}

size_t space_size(std::string_view text) {
    if (text.empty()) {
        return 0;
    }

    const auto ch = common_parse_utf8_codepoint(text, 0);
    if (ch.status != utf8_parse_result::SUCCESS) {
        return 0;
    }

    const auto cp = ch.codepoint;
    const bool space =
        (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20) ||
        cp == 0x85 || cp == 0xa0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
    return space ? ch.bytes_consumed : 0;
}

size_t skip_space(std::string_view text) {
    size_t pos = 0;
    while (const auto len = space_size(text.substr(pos))) {
        pos += len;
    }
    return pos;
}

std::string_view trim_space(std::string_view text) {
    text.remove_prefix(skip_space(text));
    while (!text.empty()) {
        size_t pos = text.size() - 1;
        while (pos > 0 && (static_cast<unsigned char>(text[pos]) & 0xc0) == 0x80) {
            --pos;
        }
        if (space_size(text.substr(pos)) != text.size() - pos) {
            break;
        }
        text = text.substr(0, pos);
    }
    return text;
}

struct named_tag {
    std::string name;
    size_t end;
};

// Accept whitespace, either quote, and attributes after name.
std::optional<named_tag> read_named_tag(std::string_view text, std::string_view tag, size_t pos) {
    if (!starts_with(text.substr(pos), tag)) {
        return std::nullopt;
    }
    pos += tag.size();
    const size_t spaces = skip_space(text.substr(pos));
    if (!spaces) {
        return std::nullopt;
    }
    pos += spaces;
    constexpr std::string_view attribute = "name=";
    if (!starts_with(text.substr(pos), attribute)) {
        return std::nullopt;
    }
    pos += attribute.size();
    if (pos == text.size() || (text[pos] != '\'' && text[pos] != '"')) {
        return std::nullopt;
    }

    const size_t first = pos + 1;
    const size_t last  = text.find(text[pos], first);
    if (last == text.npos || first == last) {
        return std::nullopt;
    }
    const size_t end = text.find('>', last + 1);
    if (end == text.npos) {
        return std::nullopt;
    }
    return named_tag{std::string(trim_space(text.substr(first, last - first))), end + 1};
}

size_t find_function_close(std::string_view text, size_t begin, bool is_partial) {
    while (true) {
        const size_t close = text.find(function_close, begin);
        if (close == text.npos) {
            return close;
        }
        const size_t cdata = text.find(cdata_open, begin);
        if (cdata == text.npos || cdata > close) {
            return close;
        }
        const size_t end = text.find(cdata_close, cdata + cdata_open.size());
        // A fresh CDATA parameter after a closed function marks a new call.
        // Literal function markup without a fresh CDATA remains parameter text.
        for (size_t next = text.find(function_open, close + function_close.size());
             next != text.npos && (end == text.npos || next < end);
             next = text.find(function_open, next + function_open.size())) {
            const auto function = read_named_tag(text, function_open, next);
            if (!function) {
                continue;
            }
            const size_t param_pos = function->end + skip_space(text.substr(function->end));
            const auto param = read_named_tag(text, param_open, param_pos);
            if (param) {
                const size_t value = param->end + skip_space(text.substr(param->end));
                if ((end == text.npos || value < end) && starts_with(text.substr(value), cdata_open)) {
                    return close;
                }
            }
        }
        if (end == text.npos) {
            return is_partial ? text.npos : close;
        }
        begin = end + cdata_close.size();
    }
}

std::string_view read_param_value(std::string_view text, size_t & pos) {
    const size_t cdata = pos + skip_space(text.substr(pos));
    if (starts_with(text.substr(cdata), cdata_open)) {
        pos = cdata + cdata_open.size();
        const size_t end = text.find(cdata_close, pos);
        if (end != text.npos) {
            const auto value = text.substr(pos, end - pos);
            pos = end + cdata_close.size();
            if (starts_with(text.substr(pos), param_close)) {
                pos += param_close.size();
            }
            return value;
        }
    }

    for (size_t end = text.find('<', pos); end != text.npos; end = text.find('<', end + 1)) {
        const auto boundary = text.substr(end);
        const bool param_ends = starts_with(boundary, param_close);
        if (param_ends || starts_with(boundary, param_open) || starts_with(boundary, function_close)) {
            const auto value = param_ends ? text.substr(pos, end - pos) : trim_space(text.substr(pos, end - pos));
            pos = end + (param_ends ? param_close.size() : 0);
            return value;
        }
    }

    const auto value = text.substr(pos);
    pos = text.size();
    return value;
}

json get_schema_properties(const json & schema, size_t depth = 0) {
    if (depth > COMMON_RECURSIVE_LIMIT) {
        throw std::invalid_argument("Tool schema nesting is too deep");
    }
    if (!schema.is_object()) {
        return json::object();
    }
    if (schema.contains("properties") && schema.at("properties").is_object()) {
        return schema.at("properties");
    }

    json merged = json::object();
    for (const char * keyword : {"anyOf", "oneOf", "allOf"}) {
        if (!schema.contains(keyword) || !schema.at(keyword).is_array()) {
            continue;
        }
        for (const auto & branch : schema.at(keyword)) {
            const auto nested = get_schema_properties(branch, depth + 1);
            for (const auto & item : nested.items()) {
                if (!merged.contains(item.key())) {
                    merged[item.key()] = item.value();
                }
            }
        }
    }
    return merged;
}

struct tool_schema {
    json properties;
    std::vector<std::string> required;
};

std::optional<json> parse_arguments(std::string_view text, size_t pos, size_t body_end, const tool_schema & schema) {
    json arguments = json::object();
    while (pos < body_end) {
        pos += skip_space(text.substr(pos, body_end - pos));
        if (pos == body_end) {
            break;
        }

        auto param = read_named_tag(text, param_open, pos);
        if (param) {
            pos = param->end;
            const auto value = read_param_value(text, pos);
            const auto & properties = schema.properties;
            const auto * prop = properties.contains(param->name) ? &properties.at(param->name) : nullptr;
            if (arguments.contains(param->name) || (!properties.empty() && !prop)) {
                continue;
            }
            if (prop && !prop->is_object()) {
                return std::nullopt;
            }
            const auto type = prop ? prop->value("type", json()) : json();
            if (!type.is_null() && type != "string") {
                arguments[param->name] = common_chat_parse_minicpm5_value(value);
            } else {
                arguments[param->name] = std::string(value);
            }
        } else if (starts_with(text.substr(pos), param_open)) {
            const size_t end = text.find(param_close, pos);
            if (end == text.npos || end + param_close.size() > body_end) {
                break;
            }
            pos = end + param_close.size();
        } else {
            break;
        }
    }
    return arguments;
}

common_peg_parser minicpm5_prefix(common_chat_peg_builder & p, bool extract_reasoning) {
    auto prefix = p.literal("<|im_start|>assistant\n");
    if (extract_reasoning) {
        prefix = prefix + ("<think>" << p.reasoning(p.until("</think>")) << "</think>");
    }
    return prefix;
}

}  // namespace

struct common_chat_minicpm5_xml {
    std::map<std::string, tool_schema> tools;
    bool extract_reasoning;

    common_chat_minicpm5_xml(const json & tool_specs, bool extract_reasoning)
        : extract_reasoning(extract_reasoning) {
        foreach_function(tool_specs, [&](size_t, const json & tool) {
            const auto & function = tool.at("function");
            const auto name = function.value("name", std::string());
            if (name.empty()) {
                return;
            }

            const auto params = function.value("parameters", json());
            tool_schema schema{get_schema_properties(params), {}};
            if (params.is_object() && params.contains("required")) {
                const auto & required = params.at("required");
                if (required.is_array()) {
                    for (const auto & key : required) {
                        if (key.is_string()) {
                            schema.required.push_back(key.get<std::string>());
                        }
                    }
                }
            }
            tools.insert_or_assign(name, std::move(schema));
        });
    }
};

static void parse_xml(std::string_view text, bool is_partial, const common_chat_minicpm5_xml & config, common_chat_msg & msg) {
    size_t pos = 0;
    while (true) {
        const size_t start = text.find(function_open, pos);
        if (start == text.npos) {
            size_t end = text.size();
            if (is_partial) {
                for (size_t n = 1; n < function_open.size() && n <= text.size() - pos; ++n) {
                    if (text.substr(text.size() - n) == function_open.substr(0, n)) {
                        end = text.size() - n;
                    }
                }
            }
            msg.content += text.substr(pos, end - pos);
            return;
        }

        msg.content += text.substr(pos, start - pos);
        const auto opening = read_named_tag(text, function_open, start);
        if (!opening) {
            // A split opening tag may still become a tool call.
            if (is_partial && text.find('>', start) == text.npos) {
                return;
            }
            msg.content += function_open;
            pos = start + function_open.size();
            continue;
        }
        const size_t close = find_function_close(text, opening->end, is_partial);
        if (close == text.npos) {
            if (!is_partial) {
                msg.content += text.substr(start);
            }
            return;
        }

        pos = close + function_close.size();
        const auto tool = config.tools.find(opening->name);
        if (tool == config.tools.end()) {
            msg.content += text.substr(start, pos - start);
            continue;
        }

        const auto args = parse_arguments(text.substr(0, close), opening->end, close, tool->second);
        if (!args) {
            // Keep the original text if a supplied property is not a schema object.
            msg.content = text;
            msg.tool_calls.clear();
            return;
        }
        const auto & required = tool->second.required;
        if (std::all_of(required.begin(), required.end(), [&](const std::string & key) { return args->contains(key); })) {
            msg.tool_calls.push_back({opening->name, args->dump(), {}});
        }
    }
}

common_chat_msg common_chat_parse_minicpm5_xml(const common_chat_input & input,
                                               bool is_partial,
                                               const common_chat_parser_params & params) {
    static const auto plain = build_chat_peg_parser([](common_chat_peg_builder & p) {
        return minicpm5_prefix(p, false) + p.content(p.rest()) + p.end();
    });
    static const auto reasoning = build_chat_peg_parser([](common_chat_peg_builder & p) {
        return minicpm5_prefix(p, true) + p.space() + p.content(p.rest()) + p.end();
    });
    const auto & config = *params.minicpm5_xml;
    auto msg = common_chat_peg_parse(config.extract_reasoning ? reasoning : plain, input, is_partial, params);
    const auto raw = std::move(msg.content);
    msg.content.clear();

    parse_xml(raw, is_partial, config, msg);
    return msg;
}

// MiniCPM5 format:
// - Reasoning: <think>{reasoning}</think> (optional)
// - Tool calls: <function name="foo"><param name="bar">value</param></function>
common_chat_params common_chat_params_init_minicpm5(const common_chat_template &          tmpl,
                                                           const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.preserved_tokens  = {
        "<function",
        "<param",
        "</function>",
        "</param>",
        "<think>",
        "</think>",
    };

    data.thinking_start_tag = "<think>";
    data.thinking_end_tags  = {"</think>"};

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<|im_start|>assistant"             },
        { COMMON_CHAT_ROLE_TOOL,      "<|im_start|>user\n<tool_response>" },
        { COMMON_CHAT_ROLE_USER,      "<|im_start|>user"                  },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|im_start|>system"                },
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = "<|im_start|>assistant\n<think>\n" + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += "\n</think>\n\n" + msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto prefix = minicpm5_prefix(p, extract_reasoning);
        if (extract_reasoning) {
            prefix += p.space();
        }

        // Response format parser
        if (has_response_format) {
            return prefix + p.content(p.schema(p.json(), "response-format", inputs.json_schema));
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // CDATA lets a value carry characters that would otherwise close the tag (e.g.
            // </param>); capture the inner text only, excluding the CDATA markers.
            auto string_value = p.choice({
                p.literal("<![CDATA[") + p.ac(p.tool_arg_string_value(p.until("]]>")) + p.literal("]]>"), "]]>") + p.tool_arg_close(p.literal("</param>")),
                p.negate(p.literal("<![CDATA[")) + p.ac(p.tool_arg_string_value(p.until("</param>")) + p.tool_arg_close(p.literal("</param>")), "</param>")
            });

            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](size_t tool_index, const json & tool) {
                const auto &      function = tool.at("function");
                const std::string name     = function.at("name");

                std::vector<common_peg_parser> arg_rules;
                foreach_parameter(function, [&](size_t param_index, const common_chat_schema_property & prop, const common_chat_schema_document_ptr & doc) {
                    auto value_parser = p.eps();
                    if (prop.schema->may_be_string()) {
                        value_parser = string_value;
                    } else {
                        value_parser = p.tool_arg_json_value(
                                p.schema(p.json(), "tool-" + std::to_string(tool_index) + "-arg-" + std::to_string(param_index) + "-schema", doc, *prop.schema)
                            ) + p.tool_arg_close(p.literal("</param>"));
                    }

                    arg_rules.push_back(p.tool_arg(
                        p.tool_arg_open(p.literal("<param name=\"") + p.tool_arg_name(p.literal(prop.name)) + p.literal("\">")) +
                        value_parser
                    ));
                });

                auto args = p.eps();
                if (!arg_rules.empty()) {
                    args = p.zero_or_more(p.choice(arg_rules) + p.space());
                }

                auto tool_parser = p.tool(
                    p.tool_open(p.literal("<function name=\"") + p.tool_name(p.literal(name)) + p.literal("\">"))
                    << p.tool_args(args)
                    << p.tool_close(p.literal("</function>")));

                tool_choice |= p.rule("tool-" + std::to_string(tool_index), tool_parser);
            });

            auto max_calls  = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call", p.repeat(tool_choice + p.space(), 1, max_calls));

            auto content = p.content(p.until("<function"));

            return prefix + content + tool_calls + p.end();
        }

        return prefix + p.content(p.rest()) + p.end();
    });

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<function" },
        };
    }

    if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !has_response_format) {
        data.minicpm5_xml = std::make_shared<common_chat_minicpm5_xml>(inputs.tools, extract_reasoning);
    }

    return data;
}
