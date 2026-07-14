#pragma once

#include "utils.h"

#include <cctype>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace jinja {

struct token {
    enum type {
        eof, // end of source
        text, // The text between Jinja statements or expressions

        numeric_literal, // e.g., 123, 1.0
        string_literal, // 'string'
        identifier, // Variables, functions, statements, booleans, etc.
        equals, // =
        open_paren, // (
        close_paren, // )
        open_statement, // {%
        close_statement, // %}
        open_expression, // {{
        close_expression, // }}
        open_square_bracket, // [
        close_square_bracket, // ]
        open_curly_bracket, // {
        close_curly_bracket, // }
        comma, // ,
        dot, // .
        colon, // :
        pipe, // |

        call_operator, // ()
        additive_binary_operator, // + - ~
        multiplicative_binary_operator, // * / %
        comparison_binary_operator, // < > <= >= == !=
        unary_operator, // ! - +
        comment, // {# ... #}
    };
    type t;
    std::string value;
    size_t pos;
};

static std::string type_to_string(token::type t) {
    switch (t) {
        case token::eof: return "eof";
        case token::text: return "text";
        case token::numeric_literal: return "numeric_literal";
        case token::string_literal: return "string_literal";
        case token::identifier: return "identifier";
        case token::equals: return "equals";
        case token::open_paren: return "open_paren";
        case token::close_paren: return "close_paren";
        case token::open_statement: return "open_statement";
        case token::close_statement: return "close_statement";
        case token::open_expression: return "open_expression";
        case token::close_expression: return "close_expression";
        case token::open_square_bracket: return "open_square_bracket";
        case token::close_square_bracket: return "close_square_bracket";
        case token::open_curly_bracket: return "open_curly_bracket";
        case token::close_curly_bracket: return "close_curly_bracket";
        case token::comma: return "comma";
        case token::dot: return "dot";
        case token::colon: return "colon";
        case token::pipe: return "pipe";
        case token::call_operator: return "call_operator";
        case token::additive_binary_operator: return "additive_binary_operator";
        case token::multiplicative_binary_operator: return "multiplicative_binary_operator";
        case token::comparison_binary_operator: return "comparison_binary_operator";
        case token::unary_operator: return "unary_operator";
        case token::comment: return "comment";
        default: return "unknown";
    }
}

struct lexer_result {
    std::vector<token> tokens;
    std::string source;
};

struct lexer {
    const std::map<char, char> escape_chars = {
        {'n', '\n'},
        {'t', '\t'},
        {'r', '\r'},
        {'b', '\b'},
        {'f', '\f'},
        {'v', '\v'},
        {'\\', '\\'},
        {'\'', '\''},
        {'\"', '\"'},
    };

    static bool is_word(char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    }

    static bool is_integer(char c) {
        return std::isdigit(static_cast<unsigned char>(c));
    }

    const std::vector<std::pair<std::string, token::type>> ordered_mapping_table = {
        // Trimmed control sequences
        {"{%-", token::open_statement},
        {"-%}", token::close_statement},
        {"{{-", token::open_expression},
        {"-}}", token::close_expression},
        // Control sequences
        {"{%", token::open_statement},
        {"%}", token::close_statement},
        {"{{", token::open_expression},
        {"}}", token::close_expression},
        // Single character tokens
        {"(", token::open_paren},
        {")", token::close_paren},
        {"{", token::open_curly_bracket},
        {"}", token::close_curly_bracket},
        {"[", token::open_square_bracket},
        {"]", token::close_square_bracket},
        {",", token::comma},
        {".", token::dot},
        {":", token::colon},
        {"|", token::pipe},
        // Comparison operators
        {"<=", token::comparison_binary_operator},
        {">=", token::comparison_binary_operator},
        {"==", token::comparison_binary_operator},
        {"!=", token::comparison_binary_operator},
        {"<", token::comparison_binary_operator},
        {">", token::comparison_binary_operator},
        // Arithmetic operators
        {"+", token::additive_binary_operator},
        {"-", token::additive_binary_operator},
        {"~", token::additive_binary_operator},
        {"*", token::multiplicative_binary_operator},
        {"/", token::multiplicative_binary_operator},
        {"%", token::multiplicative_binary_operator},
        // Assignment operator
        {"=", token::equals},
    };

    // tokenize the source string into a list of tokens
    // may throw lexer_exception on error
    lexer_result tokenize(const std::string & source);
};

struct lexer_exception : public std::runtime_error {
    lexer_exception(const std::string & msg, const std::string & source, size_t pos)
        : std::runtime_error(fmt_error_with_source("lexer", msg, source, pos)) {}
};

} // namespace jinja
