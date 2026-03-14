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

    // one of the 2 content capabilities must be true
    bool supports_string_content = true;
    bool supports_typed_content = false;

    // for reporting on server
    std::map<std::string, bool> to_map() const;

    // for debugging
    std::string to_string() const;
};

caps caps_get(jinja::program & prog);

} // namespace jinja
