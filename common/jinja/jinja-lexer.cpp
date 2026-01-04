#include "jinja-lexer.h"
#include "jinja-interpreter.h"

#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include <cctype>
#include <functional>
#include <string_view>

#define FILENAME "jinja-lexer"

namespace jinja {

static void string_lstrip(std::string & s, const char * chars) {
    size_t start = s.find_first_not_of(chars);
    if (start == std::string::npos) {
        s.clear();
    } else {
        s.erase(0, start);
    }
}

static void string_rstrip(std::string & s, const char * chars) {
    size_t end = s.find_last_not_of(chars);
    if (end == std::string::npos) {
        s.clear();
    } else {
        s.erase(end + 1);
    }
}

lexer_result lexer::tokenize(const std::string & source) {
    std::vector<token> tokens;

    // NOTE: do NOT transform the source string (i.e. preprocessing), as we need to keep
    //       the original character positions for error reporting etc.
    std::string src = source;

    if (source.empty()) {
        return {tokens, src};
    }

    // In the default configuration:
    //  - a single trailing newline is stripped if present
    //  - other whitespace (spaces, tabs, newlines etc.) is returned unchanged
    if (source.back() == '\n') {
        src.pop_back();
    }

    size_t pos = 0;
    size_t start_pos = 0;
    size_t curly_bracket_depth = 0;

    using pred = std::function<bool(char)>;
    auto consume_while = [&](pred predicate) -> std::string {
        std::string str;
        while (predicate(src[pos])) {
            // check for escape char
            if (src[pos] == '\\') {
                // consume backslash
                ++pos;
                // check for end of input
                if (pos >= src.size()) {
                    throw std::runtime_error("lexer: unexpected end of input after escape character");
                }
                // add escaped char
                char escaped_char = src[pos++];
                if (escape_chars.find(escaped_char) == escape_chars.end()) {
                    throw std::runtime_error(std::string("lexer: unknown escape character \\") + escaped_char);
                }
                char unescaped_char = escape_chars.at(escaped_char);
                str += unescaped_char;
                continue;
            }

            str += src[pos++];
            if (pos > src.size()) {
                throw std::runtime_error("lexer: unexpected end of input during consume_while");
            }
        }
        return str;
    };

    auto next_pos_is = [&](std::initializer_list<char> chars, size_t n = 1) -> bool {
        if (pos + n >= src.size()) return false;
        for (char c : chars) {
            if (src[pos + n] == c) return true;
        }
        return false;
    };

    bool is_lstrip_block = false; // example: {%-
    bool is_rstrip_block = false; // example: -%}

    while (pos < src.size()) {
        start_pos = pos;
        // JJ_DEBUG("lexer main loop at pos %zu: '%s...'", pos, src.substr(pos, 10).c_str());

        // First, consume all text that is outside of a Jinja statement or expression
        token::type last_token_type = tokens.empty()
                                            ? token::undefined
                                            : tokens.back().t;
        if (last_token_type == token::undefined ||
            last_token_type == token::close_statement ||
            last_token_type == token::close_expression ||
            last_token_type == token::comment) {
            std::string text;
            while (pos < src.size() &&
                    // Keep going until we hit the next Jinja statement or expression
                    !(
                        src[pos] == '{' &&
                        next_pos_is( {'%', '{', '#'} )
                    )) {
                text += src[pos++];
            }

            // always strip single leading newline
            // example: {{block}}\ntext
            if (!text.empty() && text.front() == '\n') {
                text.erase(0, 1);
            }

            if (is_rstrip_block) {
                // example: {last_block}[space]text
                // doing lstrip on text, effectively rstrip the LAST block
                // JJ_DEBUG("RSTRIP block detected, current text: '%s'", text.c_str());
                string_lstrip(text, " \t"); // not stripping newlines
            }

            // note: we always lstrip if the block is control or comment
            is_lstrip_block = next_pos_is({'%', '#'}) || next_pos_is({'-'}, 2);
            if (is_lstrip_block) {
                // example: text[space]{current_block}
                // doing rstrip on text, effectively lstrip the CURRENT block
                // JJ_DEBUG("LSTRIP block detected, current text: '%s'", text.c_str());
                string_rstrip(text, " \t"); // not stripping newlines
            }

            if (!text.empty()) {
                // JJ_DEBUG("consumed text: '%s'", text.c_str());
                tokens.push_back({token::text, text, start_pos});
                continue;
            }
        }

        // Possibly consume a comment
        // TODO: handle lstrip/rstrip for comments? (not important for now)
        if (src[pos] == '{' && next_pos_is( {'#'} )) {
            start_pos = pos;
            pos += 2; // Skip the opening {#
            std::string comment;
            while (!(src[pos] == '#' && next_pos_is( {'}'} ))) {
                if (pos + 2 >= src.size()) {
                    throw std::runtime_error("lexer: missing end of comment tag");
                }
                comment += src[pos++];
            }
            JJ_DEBUG("consumed comment: '%s'", comment.c_str());
            tokens.push_back({token::comment, comment, start_pos});
            pos += 2; // Skip the closing #}

            // always do rstrip for comments
            is_rstrip_block = true;
            continue;
        }

        if (src[pos] == '-' && (
                last_token_type == token::open_expression ||
                last_token_type == token::open_statement)
        ) {
            JJ_DEBUG("lexer main loop at pos %zu: '%s...'", pos, src.substr(pos, 10).c_str());
            pos++; // consume '-' in {%- or {{-
            if (pos >= src.size()) break;
        }

        // Consume (and ignore) all whitespace inside Jinja statements or expressions
        consume_while([](char c) { return std::isspace(static_cast<unsigned char>(c)); });

        if (pos >= src.size()) break;

        char ch = src[pos];

        bool is_closing_block = ch == '-' && next_pos_is( {'%', '}'} );

        // Check for unary operators
        if (!is_closing_block && (ch == '-' || ch == '+')) {
            start_pos = pos;
            token::type last_token_type = tokens.empty() ? token::undefined : tokens.back().t;
            if (last_token_type == token::text || last_token_type == token::undefined) {
                throw std::runtime_error(std::string("lexer: unexpected character: ") + ch);
            }
            switch (last_token_type) {
                case token::identifier:
                case token::numeric_literal:
                case token::string_literal:
                case token::close_paren:
                case token::close_square_bracket:
                    // Part of a binary operator
                    // a - 1, 1 - 1, true - 1, "apple" - 1, (1) - 1, a[1] - 1
                    // Continue parsing normally
                    break;
                default: {
                    // Is part of a unary operator
                    // (-1), [-1], (1 + -1), not -1, -apple
                    ++pos; // Consume the operator

                    // Check for numbers following the unary operator
                    std::string num = consume_while(is_integer);
                    std::string value = std::string(1, ch) + num;
                    token::type t = num.empty() ? token::unary_operator : token::numeric_literal;
                    // JJ_DEBUG("consumed unary operator or numeric literal: '%s'", value.c_str());
                    tokens.push_back({t, value, start_pos});
                    continue;
                }
            }
        }

        // Try to match one of the tokens in the mapping table
        bool matched = false;
        for (const auto & [seq, typ] : ordered_mapping_table) {
            start_pos = pos;
            // Inside an object literal, don't treat "}}" as expression-end
            if (seq == "}}" && curly_bracket_depth > 0) {
                continue;
            }
            if (pos + seq.size() <= src.size() && src.substr(pos, seq.size()) == seq) {
                tokens.push_back({typ, seq, start_pos});
                if (typ == token::open_expression) {
                    curly_bracket_depth = 0;
                } else if (typ == token::open_curly_bracket) {
                    ++curly_bracket_depth;
                } else if (typ == token::close_curly_bracket) {
                    --curly_bracket_depth;
                }

                // optionally handle rstrip for this block
                // this will affect the next text chunk
                if (typ == token::close_statement || typ == token::close_expression) {
                    is_rstrip_block = src[pos] == '-';
                }

                pos += seq.size();
                matched = true;
                break; // continue main loop
            }
        }
        if (matched) continue; // continue main loop

        // Strings
        if (ch == '\'' || ch == '"') {
            start_pos = pos;
            ++pos; // Skip opening quote
            std::string str = consume_while([ch](char c) { return c != ch; });
            // JJ_DEBUG("consumed string literal: '%s'", str.c_str());
            tokens.push_back({token::string_literal, str, start_pos});
            ++pos; // Skip closing quote
            continue;
        }

        // Numbers
        if (is_integer(ch)) {
            start_pos = pos;
            std::string num = consume_while(is_integer);
            if (pos < src.size() && src[pos] == '.' && pos + 1 < src.size() && is_integer(src[pos + 1])) {
                ++pos; // Consume '.'
                std::string frac = consume_while(is_integer);
                num += "." + frac;
            }
            // JJ_DEBUG("consumed numeric literal: '%s'", num.c_str());
            tokens.push_back({token::numeric_literal, num, start_pos});
            continue;
        }

        // Identifiers
        if (is_word(ch)) {
            start_pos = pos;
            std::string word = consume_while(is_word);
            // JJ_DEBUG("consumed identifier: '%s'", word.c_str());
            tokens.push_back({token::identifier, word, start_pos});
            continue;
        }

        throw std::runtime_error(std::string("lexer: unexpected character: ") + ch);
    }

    return {std::move(tokens), src};
}

} // namespace jinja
