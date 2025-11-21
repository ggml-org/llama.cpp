#include "chat-peg-parser.h"
#include "peg-parser.h"
#include <sstream>

/*
common_peg_parser common_chat_peg_parser_builder::reasoning(const std::string & tag) {
    std::string open_tag;
    open_tag.append("<").append(tag).append(">");
    std::string close_tag;
    close_tag.append("</").append(tag).append(">");
    return rule("raw-reasoning", literal(open_tag) << rule("reasoning-content", until(close_tag)) << literal(close_tag));
}

common_peg_parser common_chat_peg_parser_builder::content_before_tools(const std::string & tag) {
    return rule("content", until(tag));
}

common_peg_parser common_chat_peg_parser_builder::quasi_xml_no_attr(
    const std::string &              function_name,
    const std::vector<std::string> & parameters,
    const std::string &              function_tag,
    const std::string &              param_tag) {
    std::vector<common_peg_parser> args;

    for (auto it = parameters.begin(); it != parameters.end(); it++) {
        std::string arg_start_name;
        arg_start_name.append("arg-start-").append(*it);

        std::string param_open;
        param_open.append("<").append(param_tag).append("=");

        std::string param_open_after_name = ">";

        auto arg_name = rule(arg_start_name, literal(param_open) + capture("arg-name", literal(*it)) + literal(param_open_after_name));

        std::string param_close_end;
        param_close_end.append("</").append(param_tag).append(">");

        std::string param_close_peek;
        param_close_peek.append("</").append(function_tag).append(">");

        std::string param_peek_open;
        param_peek_open.append("<").append(param_tag).append("=");
        auto arg_end = rule("arg-end", literal(param_close_end) + peek(literal(param_peek_open) | literal(param_close_peek)));

        std::string string_content_1;
        string_content_1.append("</").append(param_tag).append("><").append(param_tag).append("=");

        std::string string_content_2;
        string_content_2.append("</").append(param_tag).append("></").append(function_tag).append(">");

        auto string_arg_content = rule("arg-string-content", until_one_of({ string_content_1, string_content_2 }));

        std::string arg_string_name;
        arg_string_name.append("arg-string-").append(*it);
        auto string_arg = rule(arg_string_name, "arg-string", arg_name + string_arg_content + arg_end);
        auto json_sec   = json();

        std::string arg_json_name;
        arg_json_name.append("arg-json-").append(*it);
        auto json_arg           = rule(arg_json_name, arg_name + rule("arg-json-content", json_sec) + arg_end);
        auto arg_json_or_string = one_or_more(json_arg | string_arg);
        args.push_back(arg_json_or_string);
    }
    auto args_sequence = sequence(args);

    std::string function_start_name;
    function_start_name.append("function-start-").append(function_name);

    std::string function_open;
    function_open.append("<").append(function_tag).append("=");

    std::string function_open_after_name;
    function_open_after_name = ">";

    std::string function_close;
    function_close.append("</").append(function_tag).append(">");

    std::string function_rule_name;
    function_rule_name.append("function-").append(function_name);
    auto function = rule(function_rule_name, rule(function_start_name, literal(function_open) + capture("tool-name", literal(function_name)) + literal(function_open_after_name)) + args_sequence + literal(function_close));

    return function;
}

common_peg_parser common_chat_peg_parser_builder::quasi_xml_attr(
    const std::string &              function_name,
    const std::vector<std::string> & parameters,
    const std::string &              function_tag,
    const std::string &              param_tag,
    const std::string &              name_attr) {
    std::vector<common_peg_parser> args;

    for (auto it = parameters.begin(); it != parameters.end(); it++) {
        std::string arg_start_name;
        arg_start_name.append("arg-start-").append(*it);

        std::string param_open;
        param_open.append("<").append(param_tag).append(" ").append(name_attr).append("=\"");

        std::string param_open_after_name ="\">";

        auto arg_name = rule(arg_start_name, literal(param_open) + capture("arg-name", literal(*it)) + literal(param_open_after_name));

        std::string param_close_end;
        param_close_end.append("</").append(param_tag).append(">");

        std::string param_close_peek;
        param_close_peek.append("</").append(function_tag).append(">");

        std::string param_peek_open;
        param_peek_open.append("<").append(param_tag).append(" ").append(name_attr).append("=\"");
        auto arg_end = rule("arg-end", literal(param_close_end) + peek(literal(param_peek_open) | literal(param_close_peek)));

        std::string string_content_1;
        string_content_1.append("</").append(param_tag).append("><").append(param_tag).append("=");

        std::string string_content_2;
        string_content_2.append("</").append(param_tag).append("></").append(function_tag).append(">");

        auto string_arg_content = rule("arg-string-content", until_one_of({ string_content_1, string_content_2 }));

        std::string arg_string_name;
        arg_string_name.append("arg-string-").append(*it);
        auto string_arg = rule(arg_string_name, "arg-string", arg_name + string_arg_content + arg_end);
        auto json_sec   = json();

        std::string arg_json_name;
        arg_json_name.append("arg-json-").append(*it);
        auto json_arg           = rule(arg_json_name, arg_name + rule("arg-json-content", json_sec) + arg_end);
        auto arg_json_or_string = one_or_more(json_arg | string_arg);
        args.push_back(arg_json_or_string);
    }
    auto args_sequence = sequence(args);

    std::string function_start_name;
    function_start_name.append("function-start-").append(function_name);

    std::string function_open;
    function_open.append("<").append(function_tag).append(" ").append(name_attr).append("=\"");

    std::string function_open_after_name = "\">";

    std::string function_close;
    function_close.append("</").append(function_tag).append(">");

    std::string function_rule_name;
    function_rule_name.append("function-").append(function_name);
    auto function = rule(function_rule_name,
        rule(function_start_name, literal(function_open) + capture("tool-name", literal(function_name)) +
            literal(function_open_after_name)) + args_sequence + literal(function_close));

    return function;
}
*/

common_peg_ast_visitor common_chat_peg_constructed_builder::extractor::visitor() {
    return [this](const common_peg_ast_node & node) {
        extract(node);
    };
}

void common_chat_peg_constructed_builder::extractor::extract(const common_peg_ast_node & node) {
    bool is_reasoning_block = node.tag == REASONING_BLOCK;
    bool is_reasoning = node.tag == REASONING;
    bool is_content = node.tag == CONTENT;
    bool is_tool_name = node.tag == TOOL_NAME;
    bool is_tool_close = node.tag == TOOL_CLOSE;
    bool is_arg_open = node.tag == TOOL_ARG_OPEN;
    bool is_arg_close = node.tag == TOOL_ARG_CLOSE;
    bool is_arg_name = node.tag == TOOL_ARG_NAME;
    bool is_arg_string = node.tag == TOOL_ARG_STRING_VALUE;
    bool is_arg_json = node.tag == TOOL_ARG_JSON_VALUE;

    if (is_reasoning_block) {
        result.reasoning_content = std::string(node.text);
    }

    if (is_reasoning) {
        result.reasoning_content = std::string(node.text);
    }

    if (is_content) {
        result.content = std::string(node.text);
    }

    if (is_tool_name) {
        result.tool_calls.emplace_back();
        current_tool = &result.tool_calls.back();
        arg_count = 0;

        current_tool->name = std::string(node.text);
        current_tool->arguments = "{";
    }

    if (is_arg_open) {
        needs_closing_quote = false;
    }

    if (is_arg_name) {
        if (arg_count > 0) {
            current_tool->arguments += ",";
        }
        current_tool->arguments += "\"" + std::string(node.text) + "\":";
        ++arg_count;
    }

    if (is_arg_string) {
        current_tool->arguments += "\"" + std::string(node.text);
        needs_closing_quote = true;
    }

    if (is_arg_close) {
        if (needs_closing_quote) {
            current_tool->arguments += "\"";
        }
    }

    if (is_arg_json) {
        current_tool->arguments += std::string(node.text);
    }

    if (is_tool_close) {
        current_tool->arguments += "}";
    }
}
