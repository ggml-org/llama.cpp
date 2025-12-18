#include "chat-peg-parser.h"

#include <cstdint>
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
    arena.visit(result, [this](const common_peg_ast_node & node) {
        map(node);
    });
}

void common_chat_peg_mapper::map(const common_peg_ast_node & node) {
    bool is_reasoning = node.tag == common_chat_peg_builder::REASONING;
    bool is_content = node.tag == common_chat_peg_builder::CONTENT;

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
    bool is_tool_id = node.tag == common_chat_peg_native_builder::TOOL_ID;
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

    bool is_tool_open = node.tag == common_chat_peg_constructed_builder::TOOL_OPEN;
    bool is_tool_name = node.tag == common_chat_peg_constructed_builder::TOOL_NAME;
    bool is_tool_close = node.tag == common_chat_peg_constructed_builder::TOOL_CLOSE;
    bool is_arg_open = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_OPEN;
    bool is_arg_close = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_CLOSE;
    bool is_arg_name = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_NAME;
    bool is_arg_string = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_STRING_VALUE;
    bool is_arg_json = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_JSON_VALUE;

    if (is_tool_open) {
        result.tool_calls.emplace_back();
        current_tool = &result.tool_calls.back();
        arg_count = 0;
    }

    if (is_tool_name) {
        current_tool->name = std::string(node.text);
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
        current_tool->arguments += std::string(trim_trailing_space(node.text));
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
                                                                      bool                   force_tool_calls
                                                                    ) {
    auto tools = choice();
    for (const auto & tool : tool_defs) {
        const auto & function = tool.at("function");
        std::string  name     = function.at("name");
        const auto & schema_  = function.at("parameters");

        auto tool_name_ = json_member("name", "\"" + tool_name(literal(name)) + "\"");
        auto tool_args_ = json_member("arguments", tool_args(schema(json(), "tool-" + name + "-schema", schema_)));

        tools |= rule("tool-" + name, tool_open(literal("{")) << tool_name_ << "," << tool_args_ << "}");
    };

    auto parallel_calls = eps();
    if (parallel_tool_calls) {
        parallel_calls = zero_or_more("," << tools);
    }
    
    auto tool_call =
        trigger_rule("tool-call", sequence({ literal(open_tag), tools, parallel_calls, literal(close_tag) }));

    if (!force_tool_calls) {
        tool_call = optional(tool_call);
    }
    return tool_call;
}
