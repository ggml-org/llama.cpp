#pragma once

#include <vector>
#include <string>
#include <sstream>

#include "value.h"
#include "runtime.h"

namespace jinja {

struct caps {
    bool supports_tools = true;
    bool supports_tool_calls = true;
    bool supports_system_role = true;
    bool supports_parallel_tool_calls = true;

    bool requires_typed_content = false; // default: use string content

    // for debugging
    std::string to_string() const {
        std::ostringstream ss;
        ss << "Caps(\n";
        ss << "  requires_typed_content=" << requires_typed_content << "\n";
        ss << "  supports_tools=" << supports_tools << "\n";
        ss << "  supports_tool_calls=" << supports_tool_calls << "\n";
        ss << "  supports_parallel_tool_calls=" << supports_parallel_tool_calls << "\n";
        ss << "  supports_system_role=" << supports_system_role << "\n";
        ss << ")";
        return ss.str();
    }
};

caps caps_get(jinja::program & prog);
void debug_print_caps(const caps & c);

} // namespace jinja
