#include "chat-peg-parser.h"

class common_chat_peg_parser_builder_helper : public common_chat_peg_parser_builder {

public:
    // Helper methods for common patterns

    // Adds raw-reasoning for the entire reasoning block plus reasoning-content for the contents, by default thinking tag is "think"
    common_chat_peg_parser reasoning(const std::string & tag = "think");

    // Adds main content block before tool call block, due to the varied nature of tool call openers (not always XML-like) full tag is required
    common_chat_peg_parser content_before_tools(const std::string &tag);

    // Adds a quasi-XML tool call spec without a separate name attribute (Qwen3 style);
    // TODO: accept parameter schemas (required, value types etc.)
    common_chat_peg_parser quasi_xml_no_attr(const std::string &function_name, const std::vector<std::string> &parameters,
        const std::string &function_tag = "function", const std::string &param_tag = "parameter");

    // Adds a quasi-XML tool call spec with a separate name attribute (Minimax-M2 style)
    // TODO: accept parameter schemas (required, value types etc.)
    common_chat_peg_parser quasi_xml_attr(const std::string &function_name, const std::vector<std::string> &parameters,
        const std::string &function_tag = "invoke", const std::string &param_tag = "parameter",
        const std::string &name_attr = "name");
};

common_chat_peg_parser build_peg_parser_helper(const std::function<common_chat_peg_parser(common_chat_peg_parser_builder_helper&)> & fn);

inline void parser_semantic_handler(const common_chat_parse_event & ev, common_chat_parse_semantics & semantics) {
    if (ev.rule == "reasoning-content" && ev.ending()) {
        semantics.reasoning_content = ev.text;
    }

    if (ev.rule == "content" && ev.ending()) {
        semantics.content = ev.text;
    }

    if (ev.rule.find("function-start") != std::string::npos && ev.ending() && ev.success()) {
        semantics.tool_calls.emplace_back();
        auto & tc = semantics.tool_calls.back();
        tc.name = semantics.captures["tool-name"];
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

    if (ev.rule.find("arg-string") != std::string::npos && ev.ending() && ev.success()) {
        auto & tc = semantics.tool_calls.back();
        tc.arguments += "\"";
    }

    if (ev.rule == "arg-json-content" && ev.ending() && (ev.success() || ev.need_more_input())) {
        auto & tc = semantics.tool_calls.back();
        tc.arguments += std::string(ev.text);
    }
}
