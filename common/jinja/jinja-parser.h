#pragma once

#include "jinja-lexer.h"
#include "jinja-interpreter.h"

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>

namespace jinja {

program parse_from_tokens(const std::vector<token> & tokens);

program parse_from_tokens(const lexer_result & lexer_res);

} // namespace jinja
