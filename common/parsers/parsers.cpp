#include "parsers.h"

#include "log.h"

void foreach_function(const json & tools, const std::function<void(size_t, const json &)> & fn) {
    for (size_t i = 0; i < tools.size(); i++) {
        const auto & tool = tools[i];
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(i, tool);
    }
}

void foreach_parameter(const json & function, const std::function<void(size_t, const common_chat_schema_property &, const common_chat_schema_document_ptr &)> & fn) {
    auto         params = common_chat_tool_parameters(function);
    auto         doc    = std::make_shared<const common_chat_schema_document>(common_chat_schema_from_json(params));
    const auto * object = dynamic_cast<const common_chat_schema_object *>(doc->root.get());
    if (!object) {
        return;
    }
    for (size_t i = 0; i < object->properties.size(); i++) {
        fn(i, object->properties[i], doc);
    }
}
