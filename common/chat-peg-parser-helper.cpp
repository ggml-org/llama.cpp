#include "chat-peg-parser-helper.h"
#include "chat-peg-parser.h"

common_chat_peg_parser common_chat_peg_parser_builder_helper::reasoning(const std::string &tag) {
    return add_rule("raw-reasoning", std::string("<" + tag + ">") << add_rule("reasoning-content", until("</" + tag + ">")) << "</" + tag + ">");
}

common_chat_peg_parser common_chat_peg_parser_builder_helper::content_before_tools(const std::string &tag) {
    return add_rule("content", until(tag));
}

common_chat_peg_parser common_chat_peg_parser_builder_helper::quasi_xml_no_attr(const std::string &function_name, const std::vector<std::string> &parameters,
    const std::string &function_tag, const std::string &param_tag) {
    std::vector<common_chat_peg_parser> args;

    for (auto it = parameters.begin(); it != parameters.end(); it++) {
        auto arg_name = add_rule(std::string("arg-start-" + *it), literal("<" + param_tag + "=" + *it + ">"));
        auto arg_end = add_rule("arg-end", "</" + param_tag + ">" + peek(literal("<" + param_tag + "=") | ("</" + function_tag + ">")));
        auto string_arg_content = add_rule("arg-string-content",
            until_one_of({"</" + param_tag + "><" + param_tag + "=", "</" + param_tag + "></" + function_tag + ">"}));
        auto string_arg = add_rule("arg-string-" + *it, arg_name + string_arg_content + arg_end);
        auto json_sec = json();
        auto json_arg = add_rule("arg-json-" + *it, arg_name + add_rule("arg-json-content", json_sec) + arg_end);
        auto arg_json_or_string = one_or_more(json_arg | string_arg);
        args.push_back(arg_json_or_string);
    }

    auto args_sequence = sequence(args);
    auto function = add_rule("function-" + function_name,
                add_rule("function-start-" + function_name, "<" + function_tag + "=" + function_name + ">")
                + args_sequence + "</" + function_tag + ">");

    return function;
}

common_chat_peg_parser build_peg_parser_helper(const std::function<common_chat_peg_parser(common_chat_peg_parser_builder_helper&)> & fn) {
    common_chat_peg_parser_builder_helper builder;
    auto root = fn(builder);
    builder.set_root(root);
    return builder.build();
}
