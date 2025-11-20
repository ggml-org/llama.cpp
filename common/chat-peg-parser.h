#pragma once

#include "peg-parser.h"
#include "log.h"

class common_chat_peg_parser_builder : public common_peg_parser_builder {
  public:
    // Adds raw-reasoning for the entire reasoning block plus reasoning-content for the contents, by default thinking tag is "think"
    common_peg_parser reasoning(const std::string & tag = "think");

    // Adds main content block before tool call block, due to the varied nature of tool call openers (not always XML-like) full tag is required
    common_peg_parser content_before_tools(const std::string &tag);

    // Adds a quasi-XML tool call spec without a separate name attribute (Qwen3 style);
    // TODO: accept parameter schemas (required, value types etc.)
    common_peg_parser quasi_xml_no_attr(const std::string &function_name, const std::vector<std::string> &parameters,
        const std::string &function_tag = "function", const std::string &param_tag = "parameter");

    // Adds a quasi-XML tool call spec with a separate name attribute (Minimax-M2 style)
    // TODO: accept parameter schemas (required, value types etc.)
    common_peg_parser quasi_xml_attr(const std::string &function_name, const std::vector<std::string> &parameters,
        const std::string &function_tag = "invoke", const std::string &param_tag = "parameter",
        const std::string &name_attr = "name");
};

template<typename F>
common_peg_arena build_chat_peg_parser(F && fn) {
    common_chat_peg_parser_builder builder;
    auto root = fn(builder);
    builder.set_root(root);
    return builder.build();
}

struct common_chat_peg_simple_handler {
    std::function<void(const std::string & msg)> log;
    void operator()(const common_peg_parse_event & ev, common_peg_parse_semantics & semantics) const;
};
