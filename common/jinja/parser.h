#pragma once

#include "lexer.h"
#include "runtime.h"
#include "utils.h"

#include <string>
#include <stdexcept>

namespace jinja {

// parse from a list of tokens into an AST (program)
// may throw parser_exception on error
program parse_from_tokens(const lexer_result & lexer_res);

struct parser_exception : public std::runtime_error {
    parser_exception(const std::string & msg, const std::string & source, size_t pos)
        : std::runtime_error(fmt_error_with_source("parser", msg, source, pos)) {}
};

} // namespace jinja
