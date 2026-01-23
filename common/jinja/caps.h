#pragma once

#include "runtime.h"

#include <string>
#include <map>

namespace jinja {

struct caps {
    bool supports_tools = true;
    bool supports_tool_calls = true;
    bool supports_system_role = true;
    bool supports_parallel_tool_calls = true;
    bool supports_preserve_reasoning = false; // support assistant message with reasoning_content

    bool requires_typed_content = false; // default: use string content
    bool requires_non_null_content = false; // requires "" instead of null for content in tool calls

    // for reporting on server
    std::map<std::string, bool> to_map() const;

    // for debugging
    std::string to_string() const;
};

caps caps_get(jinja::program & prog);

} // namespace jinja
