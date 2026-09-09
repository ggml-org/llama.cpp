#include "parsers.h"

#include "log.h"

void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(tool);
    }
}

common_schema_document_ptr parse_parameters(const json & function) {
    auto params = function.contains("parameters") ? function.at("parameters") : json::object();
    return std::make_shared<const common_schema_document>(common_schema_parse(params));
}

void foreach_parameter(const json & function, const std::function<void(const common_schema_property &, const common_schema_document_ptr &)> & fn) {
    auto         doc    = parse_parameters(function);
    const auto * object = dynamic_cast<const common_schema_object *>(doc->root.get());
    if (!object) {
        return;
    }
    for (const auto & prop : object->properties) {
        fn(prop, doc);
    }
}
