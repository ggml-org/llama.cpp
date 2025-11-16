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
    // common_chat_peg_parser quasi_xml_attr(const std::string &function_name, const std::vector<std::string> &parameters,
    //     const std::string &function_tag = "invoke", const std::string &param_tag = "parameter",
    //     const std::string &name_attr = "name");
};

common_chat_peg_parser build_peg_parser_helper(const std::function<common_chat_peg_parser(common_chat_peg_parser_builder_helper&)> & fn);
