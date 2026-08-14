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

    // supports preserve reasoning trace in the full history, not just the last assistant message
    bool supports_preserve_reasoning = false;

    // one of the 2 content capabilities must be true
    bool supports_string_content = true;
    bool supports_typed_content = false;

    bool supports_object_arguments = false;

    // for reporting on server
    std::map<std::string, bool> to_map() const;

    // for debugging
    std::string to_string() const;
};

caps caps_get(jinja::program & prog);

void caps_apply_preserve_reasoning(jinja::context & ctx, bool enabled);

} // namespace jinja
