#pragma once

#include "chat.h"
#include "peg-parser.h"

/*
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
*/

class common_chat_peg_constructed_builder : public common_peg_parser_builder {
  public:
    static constexpr const char * REASONING_BLOCK = "reasoning-block";
    static constexpr const char * REASONING = "reasoning";
    static constexpr const char * CONTENT = "content";
    static constexpr const char * TOOL = "tool";
    static constexpr const char * TOOL_OPEN = "tool-open";
    static constexpr const char * TOOL_CLOSE = "tool-close";
    static constexpr const char * TOOL_NAME = "tool-name";
    static constexpr const char * TOOL_ARG = "tool-arg";
    static constexpr const char * TOOL_ARG_OPEN = "tool-arg-open";
    static constexpr const char * TOOL_ARG_CLOSE = "tool-arg-close";
    static constexpr const char * TOOL_ARG_NAME = "tool-arg-name";
    static constexpr const char * TOOL_ARG_STRING_VALUE = "tool-arg-string-value";
    static constexpr const char * TOOL_ARG_JSON_VALUE = "tool-arg-json-value";

    struct extractor {
        common_chat_msg & result;
        common_chat_tool_call * current_tool;
        int arg_count = 0;
        bool needs_closing_quote = false;

        extractor(common_chat_msg & msg) : result(msg) { }

        void extract(const common_peg_ast_node & node);
        common_peg_ast_visitor visitor();
    };

    common_peg_parser reasoning_block(const common_peg_parser & p) { return tag(REASONING_BLOCK, p); }
    common_peg_parser reasoning(const common_peg_parser & p) { return tag(REASONING, p); }
    common_peg_parser content(const common_peg_parser & p) { return tag(CONTENT, p); }
    common_peg_parser tool(const common_peg_parser & p) { return tag(TOOL, p); }
    common_peg_parser tool_open(const common_peg_parser & p) { return atomic(tag(TOOL_OPEN, p)); }
    common_peg_parser tool_close(const common_peg_parser & p) { return atomic(tag(TOOL_CLOSE, p)); }
    common_peg_parser tool_name(const common_peg_parser & p) { return atomic(tag(TOOL_NAME, p)); }
    common_peg_parser tool_arg(const common_peg_parser & p) { return tag(TOOL_ARG, p); }
    common_peg_parser tool_arg_open(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_OPEN, p)); }
    common_peg_parser tool_arg_close(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_CLOSE, p)); }
    common_peg_parser tool_arg_name(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_NAME, p)); }
    common_peg_parser tool_arg_string_value(const common_peg_parser & p) { return tag(TOOL_ARG_STRING_VALUE, p); }
    common_peg_parser tool_arg_json_value(const common_peg_parser & p) { return tag(TOOL_ARG_JSON_VALUE, p); }
};

inline common_peg_arena build_chat_peg_constructed_parser(const std::function<common_peg_parser(common_chat_peg_constructed_builder & builder)> & fn) {
    common_chat_peg_constructed_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}
