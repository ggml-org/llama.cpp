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

void foreach_parameter(const json & function, const std::function<void(const common_schema_property &, const json &)> & fn) {
    if (!function.contains("parameters") || !function.at("parameters").is_object()) {
        return;
    }
    const auto & params = function.at("parameters");
    auto         doc    = common_schema_parse(params);
    const auto * object = dynamic_cast<const common_schema_object *>(doc.root.get());
    if (!object) {
        return;
    }
    for (const auto & prop : object->properties) {
        fn(prop, params.at("properties").at(prop.name));
    }
}
