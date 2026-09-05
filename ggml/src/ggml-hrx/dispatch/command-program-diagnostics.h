#pragma once

#include "command-program.h"

#include <string>

namespace ggml::hrx {

std::string command_kind_name(CommandKind kind);
std::string command_binding_origin_name(CommandBindingOrigin origin);
std::string resource_access_name(ResourceAccess access);

std::string format_command_binding(const CommandBinding & binding);
std::string format_command(const Command & command);
std::string format_command_program(const CommandProgram & program);

}  // namespace ggml::hrx
