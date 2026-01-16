#pragma once

#include "runtime.h"

#include <string>

namespace jinja {

struct caps {
    bool supports_tools = true;
    bool supports_tool_calls = true;
    bool supports_system_role = true;
    bool supports_parallel_tool_calls = true;

    bool requires_typed_content = false; // default: use string content

    // for debugging
    std::string to_string() const;
};

caps caps_get(jinja::program & prog);
void debug_print_caps(const caps & c);

} // namespace jinja
