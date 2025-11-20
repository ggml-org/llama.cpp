#include "chat-peg-parser-helper.h"
#include "peg-parser.h"
#include <sstream>

common_peg_parser common_peg_parser_builder_helper::reasoning(const std::string & tag) {
    std::string open_tag;
    open_tag.append("<").append(tag).append(">");
    std::string close_tag;
    close_tag.append("</").append(tag).append(">");
    return rule("raw-reasoning", literal(open_tag) << rule("reasoning-content", until(close_tag)) << literal(close_tag));
}

common_peg_parser common_peg_parser_builder_helper::content_before_tools(const std::string & tag) {
    return rule("content", until(tag));
}

common_peg_parser common_peg_parser_builder_helper::quasi_xml_no_attr(
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

common_peg_parser common_peg_parser_builder_helper::quasi_xml_attr(
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

void common_peg_parse_simple_handler::operator()(const common_peg_parse_event & ev, common_peg_parse_semantics & semantics) const {
    if (log) {
        std::stringstream ss;
        ss << "Event: type=" << (ev.type == COMMON_PEG_PARSE_EVENT_NODE_START ? "start" : "end  ");
        ss << " rule=" << ev.rule;
        ss << " result=" << common_peg_parse_result_type_name(ev.status);
        ss << " text=" << ev.text;
        log(ss.str());
    }

    if (ev.rule == "reasoning-content" && ev.ending()) {
        semantics.reasoning_content = ev.text;
        if (log) {
            log("  reasoning_content=" + semantics.reasoning_content);
        }
    }

    if (ev.rule == "content" && ev.ending()) {
        semantics.content = ev.text;
        if (log) {
            log("  content=" + semantics.content);
        }
    }

    if (ev.rule.find("function-start") != std::string::npos && ev.ending() && ev.success()) {
        semantics.tool_calls.emplace_back();
        auto & tc = semantics.tool_calls.back();
        tc.name = semantics.captures["tool-name"];
        if (log) {
            log("  tool call added");
            log("    name=" + tc.name);
        }
    }

    if (ev.rule.find("arg-start") != std::string::npos && ev.ending() && ev.success()) {
        auto & tc = semantics.tool_calls.back();
        auto name = semantics.captures["arg-name"];
        if (tc.arguments.empty()) {
            tc.arguments += "{";
        } else {
            tc.arguments += ", ";
        }
        tc.arguments += "\"" + name + "\": ";
    }

    if (ev.rule == "arg-string-content" && ev.ending() && ev.success()) {
        auto & tc = semantics.tool_calls.back();
        tc.arguments += "\"" + std::string(ev.text);
    }

    if (ev.annotation == "arg-string" && ev.ending() && ev.success()) {
        auto & tc = semantics.tool_calls.back();
        tc.arguments += "\"";
        if (log) {
            log("    args=" + tc.arguments);
        }
    }

    if (ev.rule == "arg-json-content" && ev.ending() && (ev.success() || ev.need_more_input())) {
        auto & tc = semantics.tool_calls.back();
        tc.arguments += std::string(ev.text);
        if (log) {
            log("    args=" + tc.arguments);
        }
    }
}
