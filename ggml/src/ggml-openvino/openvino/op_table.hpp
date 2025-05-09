#pragma once

#include "node_context.hpp"

namespace ov {
namespace frontend {
namespace ggml {

const std::unordered_map<std::string, CreatorFunction> get_supported_ops();

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
