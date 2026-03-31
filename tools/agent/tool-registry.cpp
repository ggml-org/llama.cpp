#include "tool-registry.h"

tool_registry & tool_registry::instance() {
    static tool_registry inst;
    return inst;
}

void tool_registry::register_tool(const tool_def & tool) {
    tools_[tool.name] = tool;
}

const tool_def * tool_registry::get_tool(const std::string & name) const {
    auto it = tools_.find(name);
    if (it != tools_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<const tool_def *> tool_registry::get_all_tools() const {
    std::vector<const tool_def *> result;
    for (const auto & [name, tool] : tools_) {
        result.push_back(&tool);
    }
    return result;
}

std::vector<common_chat_tool> tool_registry::to_chat_tools() const {
    std::vector<common_chat_tool> result;
    for (const auto & [name, tool] : tools_) {
        result.push_back(tool.to_chat_tool());
    }
    return result;
}

tool_result tool_registry::execute(const std::string & name, const json & args, const tool_context & ctx) const {
    const tool_def * tool = get_tool(name);
    if (!tool) {
        return {false, "", "Unknown tool: " + name};
    }

    try {
        return tool->execute(args, ctx);
    } catch (const std::exception & e) {
        return {false, "", std::string("Tool execution error: ") + e.what()};
    }
}
