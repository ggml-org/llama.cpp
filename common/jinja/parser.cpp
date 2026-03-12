#include "lexer.h"
#include "runtime.h"
#include "parser.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define FILENAME "jinja-parser"

namespace jinja {

// Helper to check type without asserting (useful for logic)
template<typename T>
static bool is_type(const statement_ptr & ptr) {
    return dynamic_cast<const T*>(ptr.get()) != nullptr;
}

class parser {
    const std::vector<token> & tokens;
    size_t current = 0;

    std::string source; // for error reporting

public:
    parser(const std::vector<token> & t, const std::string & src) : tokens(t), source(src) {}

    program parse() {
        statements body;
        while (current < tokens.size()) {
            body.push_back(parse_any());
        }
        return program(std::move(body));
    }

    // NOTE: start_pos is the token index, used for error reporting
    template<typename T, typename... Args>
    std::unique_ptr<T> mk_stmt(size_t start_pos, Args&&... args) {
        auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
        assert(start_pos < tokens.size());
        ptr->pos = tokens[start_pos].pos;
        return ptr;
    }

private:
    const token & peek(size_t offset = 0) const {
        if (current + offset >= tokens.size()) {
            static const token end_token{token::eof, "", 0};
            return end_token;
        }
        return tokens[current + offset];
    }

    token expect(token::type type, const std::string&  error) {
        const auto & t = peek();
        if (t.t != type) {
            throw parser_exception("Parser Error: " + error + " (Got " + t.value + ")", source, t.pos);
        }
        current++;
        return t;
    }

    void expect_identifier(const std::string & name) {
        const auto & t = peek();
        if (t.t != token::identifier || t.value != name) {
            throw parser_exception("Expected identifier: " + name, source, t.pos);
        }
        current++;
    }

    bool is(token::type type) const {
        return peek().t == type;
    }

    bool is_identifier(const std::string & name) const {
        return peek().t == token::identifier && peek().value == name;
    }

    bool is_statement(const std::vector<std::string> & names) const {
        if (peek(0).t != token::open_statement || peek(1).t != token::identifier) {
            return false;
        }
        std::string val = peek(1).value;
        return std::find(names.begin(), names.end(), val) != names.end();
    }

    statement_ptr parse_any() {
        size_t start_pos = current;
        switch (peek().t) {
            case token::comment:
                return mk_stmt<comment_statement>(start_pos, tokens[current++].value);
            case token::text:
                return mk_stmt<string_literal>(start_pos, tokens[current++].value);
            case token::open_statement:
                return parse_jinja_statement();
            case token::open_expression:
                return parse_jinja_expression();
            default:
                throw std::runtime_error("Unexpected token type");
        }
    }

    statement_ptr parse_jinja_expression() {
        // Consume {{ }} tokens
        expect(token::open_expression, "Expected {{");
        auto result = parse_expression();
        expect(token::close_expression, "Expected }}");
        return result;
    }

    statement_ptr parse_jinja_statement() {
        // Consume {% token
        expect(token::open_statement, "Expected {%");

        if (peek().t != token::identifier) {
            throw std::runtime_error("Unknown statement");
        }

        size_t start_pos = current;
        std::string name = peek().value;
        current++; // consume identifier

        statement_ptr result;
        if (name == "set") {
            result = parse_set_statement(start_pos);

        } else if (name == "if") {
            result = parse_if_statement(start_pos);
            // expect {% endif %}
            expect(token::open_statement, "Expected {%");
            expect_identifier("endif");
            expect(token::close_statement, "Expected %}");

        } else if (name == "macro") {
            result = parse_macro_statement(start_pos);
            // expect {% endmacro %}
            expect(token::open_statement, "Expected {%");
            expect_identifier("endmacro");
            expect(token::close_statement, "Expected %}");

        } else if (name == "for") {
            result = parse_for_statement(start_pos);
            // expect {% endfor %}
            expect(token::open_statement, "Expected {%");
            expect_identifier("endfor");
            expect(token::close_statement, "Expected %}");

        } else if (name == "break") {
            expect(token::close_statement, "Expected %}");
            result = mk_stmt<break_statement>(start_pos);

        } else if (name == "continue") {
            expect(token::close_statement, "Expected %}");
            result = mk_stmt<continue_statement>(start_pos);

        } else if (name == "call") {
            statements caller_args;
            // bool has_caller_args = false;
            if (is(token::open_paren)) {
                // Optional caller arguments, e.g. {% call(user) dump_users(...) %}
                caller_args = parse_args();
                // has_caller_args = true;
            }
            auto callee = parse_primary_expression();
            if (!is_type<identifier>(callee)) throw std::runtime_error("Expected identifier");

            auto call_args = parse_args();
            expect(token::close_statement, "Expected %}");

            statements body;
            while (!is_statement({"endcall"})) {
                body.push_back(parse_any());
            }

            expect(token::open_statement, "Expected {%");
            expect_identifier("endcall");
            expect(token::close_statement, "Expected %}");

            auto call_expr = mk_stmt<call_expression>(start_pos, std::move(callee), std::move(call_args));
            result = mk_stmt<call_statement>(start_pos, std::move(call_expr), std::move(caller_args), std::move(body));

        } else if (name == "filter") {
            auto filter_node = parse_primary_expression();
            if (is_type<identifier>(filter_node) && is(token::open_paren)) {
                filter_node = parse_call_expression(std::move(filter_node));
            }
            expect(token::close_statement, "Expected %}");

            statements body;
            while (!is_statement({"endfilter"})) {
                body.push_back(parse_any());
            }

            expect(token::open_statement, "Expected {%");
            expect_identifier("endfilter");
            expect(token::close_statement, "Expected %}");
            result = mk_stmt<filter_statement>(start_pos, std::move(filter_node), std::move(body));

        } else if (name == "generation" || name == "endgeneration") {
            // Ignore generation blocks (transformers-specific)
            // See https://github.com/huggingface/transformers/pull/30650 for more information.
            result = mk_stmt<noop_statement>(start_pos);
            current++;

        } else {
            throw std::runtime_error("Unknown statement: " + name);
        }
        return result;
    }

    statement_ptr parse_set_statement(size_t start_pos) {
        // NOTE: `set` acts as both declaration statement and assignment expression
        auto left = parse_expression_sequence();
        statement_ptr value = nullptr;
        statements body;

        if (is(token::equals)) {
            current++;
            value = parse_expression_sequence();
        } else {
            // parsing multiline set here
            expect(token::close_statement, "Expected %}");
            while (!is_statement({"endset"})) {
                body.push_back(parse_any());
            }
            expect(token::open_statement, "Expected {%");
            expect_identifier("endset");
        }
        expect(token::close_statement, "Expected %}");
        return mk_stmt<set_statement>(start_pos, std::move(left), std::move(value), std::move(body));
    }

    statement_ptr parse_if_statement(size_t start_pos) {
        auto test = parse_expression();
        expect(token::close_statement, "Expected %}");

        statements body;
        statements alternate;

        // Keep parsing 'if' body until we reach the first {% elif %} or {% else %} or {% endif %}
        while (!is_statement({"elif", "else", "endif"})) {
            body.push_back(parse_any());
        }

        if (is_statement({"elif"})) {
            size_t pos0 = current;
            ++current; // consume {%
            ++current; // consume 'elif'
            alternate.push_back(parse_if_statement(pos0)); // nested If
        } else if (is_statement({"else"})) {
            ++current; // consume {%
            ++current; // consume 'else'
            expect(token::close_statement, "Expected %}");

            // keep going until we hit {% endif %}
            while (!is_statement({"endif"})) {
                alternate.push_back(parse_any());
            }
        }
        return mk_stmt<if_statement>(start_pos, std::move(test), std::move(body), std::move(alternate));
    }

    statement_ptr parse_macro_statement(size_t start_pos) {
        auto name = parse_primary_expression();
        auto args = parse_args();
        expect(token::close_statement, "Expected %}");
        statements body;
        // Keep going until we hit {% endmacro
        while (!is_statement({"endmacro"})) {
            body.push_back(parse_any());
        }
        return mk_stmt<macro_statement>(start_pos, std::move(name), std::move(args), std::move(body));
    }

    statement_ptr parse_expression_sequence(bool primary = false) {
        size_t start_pos = current;
        statements exprs;
        exprs.push_back(primary ? parse_primary_expression() : parse_expression());
        bool is_tuple = is(token::comma);
        while (is(token::comma)) {
            current++; // consume comma
            exprs.push_back(primary ? parse_primary_expression() : parse_expression());
        }
        return is_tuple ? mk_stmt<tuple_literal>(start_pos, std::move(exprs)) : std::move(exprs[0]);
    }

    statement_ptr parse_for_statement(size_t start_pos) {
        // e.g., `message` in `for message in messages`
        auto loop_var = parse_expression_sequence(true); // should be an identifier/tuple
        if (!is_identifier("in")) throw std::runtime_error("Expected 'in'");
        current++;

        // `messages` in `for message in messages`
        auto iterable = parse_expression();
        expect(token::close_statement, "Expected %}");

        statements body;
        statements alternate;

        // Keep going until we hit {% endfor or {% else
        while (!is_statement({"endfor", "else"})) {
            body.push_back(parse_any());
        }

        if (is_statement({"else"})) {
            current += 2;
            expect(token::close_statement, "Expected %}");
            while (!is_statement({"endfor"})) {
                alternate.push_back(parse_any());
            }
        }
        return mk_stmt<for_statement>(
            start_pos,
            std::move(loop_var), std::move(iterable),
            std::move(body), std::move(alternate));
    }

    statement_ptr parse_expression() {
        // Choose parse function with lowest precedence
        return parse_if_expression();
    }

    statement_ptr parse_if_expression() {
        auto a = parse_logical_or_expression();
        if (is_identifier("if")) {
            // Ternary expression
            size_t start_pos = current;
            ++current; // consume 'if'
            auto test = parse_logical_or_expression();
            if (is_identifier("else")) {
                // Ternary expression with else
                size_t pos0 = current;
                ++current; // consume 'else'
                auto false_expr = parse_if_expression(); // recurse to support chained ternaries
                return mk_stmt<ternary_expression>(pos0, std::move(test), std::move(a), std::move(false_expr));
            } else {
                // Select expression on iterable
                return mk_stmt<select_expression>(start_pos, std::move(a), std::move(test));
            }
        }
        return a;
    }

    statement_ptr parse_logical_or_expression() {
        auto left = parse_logical_and_expression();
        while (is_identifier("or")) {
            size_t start_pos = current;
            token op = tokens[current++];
            left = mk_stmt<binary_expression>(start_pos, op, std::move(left), parse_logical_and_expression());
        }
        return left;
    }

    statement_ptr parse_logical_and_expression() {
        auto left = parse_logical_negation_expression();
        while (is_identifier("and")) {
            size_t start_pos = current;
            auto op = tokens[current++];
            left = mk_stmt<binary_expression>(start_pos, op, std::move(left), parse_logical_negation_expression());
        }
        return left;
    }

    statement_ptr parse_logical_negation_expression() {
        // Try parse unary operators
        if (is_identifier("not")) {
            size_t start_pos = current;
            auto op = tokens[current++];
            return mk_stmt<unary_expression>(start_pos, op, parse_logical_negation_expression());
        }
        return parse_comparison_expression();
    }

    statement_ptr parse_comparison_expression() {
        // NOTE: membership has same precedence as comparison
        // e.g., ('a' in 'apple' == 'b' in 'banana') evaluates as ('a' in ('apple' == ('b' in 'banana')))
        auto left = parse_additive_expression();
        while (true) {
            token op;
            size_t start_pos = current;
            if (is_identifier("not") && peek(1).t == token::identifier && peek(1).value == "in") {
                op = {token::identifier, "not in", tokens[current].pos};
                current += 2;
            } else if (is_identifier("in")) {
                op = tokens[current++];
            } else if (is(token::comparison_binary_operator)) {
                op = tokens[current++];
            } else break;
            left = mk_stmt<binary_expression>(start_pos, op, std::move(left), parse_additive_expression());
        }
        return left;
    }

    statement_ptr parse_additive_expression() {
        auto left = parse_multiplicative_expression();
        while (is(token::additive_binary_operator)) {
            size_t start_pos = current;
            auto op = tokens[current++];
            left = mk_stmt<binary_expression>(start_pos, op, std::move(left), parse_multiplicative_expression());
        }
        return left;
    }

    statement_ptr parse_multiplicative_expression() {
        auto left = parse_test_expression();
        while (is(token::multiplicative_binary_operator)) {
            size_t start_pos = current;
            auto op = tokens[current++];
            left = mk_stmt<binary_expression>(start_pos, op, std::move(left), parse_test_expression());
        }
        return left;
    }

    statement_ptr parse_test_expression() {
        auto operand = parse_filter_expression();
        while (is_identifier("is")) {
            size_t start_pos = current;
            current++;
            bool negate = false;
            if (is_identifier("not")) { current++; negate = true; }
            auto test_id = parse_primary_expression();
            // FIXME: tests can also be expressed like this: if x is eq 3
            if (is(token::open_paren)) test_id = parse_call_expression(std::move(test_id));
            operand = mk_stmt<test_expression>(start_pos, std::move(operand), negate, std::move(test_id));
        }
        return operand;
    }

    statement_ptr parse_filter_expression() {
        auto operand = parse_call_member_expression();
        while (is(token::pipe)) {
            size_t start_pos = current;
            current++;
            auto filter = parse_primary_expression();
            if (is(token::open_paren)) filter = parse_call_expression(std::move(filter));
            operand = mk_stmt<filter_expression>(start_pos, std::move(operand), std::move(filter));
        }
        return operand;
    }

    statement_ptr parse_call_member_expression() {
        // Handle member expressions recursively
        auto member = parse_member_expression(parse_primary_expression());
        return is(token::open_paren)
            ? parse_call_expression(std::move(member)) // foo.x()
            : std::move(member);
    }

    statement_ptr parse_call_expression(statement_ptr callee) {
        size_t start_pos = current;
        auto expr = mk_stmt<call_expression>(start_pos, std::move(callee), parse_args());
        auto member = parse_member_expression(std::move(expr)); // foo.x().y
        return is(token::open_paren)
            ? parse_call_expression(std::move(member)) // foo.x()()
            : std::move(member);
    }

    statements parse_args() {
        // comma-separated arguments list
        expect(token::open_paren, "Expected (");
        statements args;
        while (!is(token::close_paren)) {
            statement_ptr arg;
            // unpacking: *expr
            if (peek().t == token::multiplicative_binary_operator && peek().value == "*") {
                size_t start_pos = current;
                ++current; // consume *
                arg = mk_stmt<spread_expression>(start_pos, parse_expression());
            } else {
                arg = parse_expression();
                if (is(token::equals)) {
                    // keyword argument
                    // e.g., func(x = 5, y = a or b)
                    size_t start_pos = current;
                    ++current; // consume equals
                    arg = mk_stmt<keyword_argument_expression>(start_pos, std::move(arg), parse_expression());
                }
            }
            args.push_back(std::move(arg));
            if (is(token::comma)) {
                ++current; // consume comma
            }
        }
        expect(token::close_paren, "Expected )");
        return args;
    }

    statement_ptr parse_member_expression(statement_ptr object) {
        size_t start_pos = current;
        while (is(token::dot) || is(token::open_square_bracket)) {
            auto op = tokens[current++];
            bool computed = op.t == token::open_square_bracket;
            statement_ptr prop;
            if (computed) {
                prop = parse_member_expression_arguments();
                expect(token::close_square_bracket, "Expected ]");
            } else {
                prop = parse_primary_expression();
            }
            object = mk_stmt<member_expression>(start_pos, std::move(object), std::move(prop), computed);
        }
        return object;
    }

    statement_ptr parse_member_expression_arguments() {
        // NOTE: This also handles slice expressions colon-separated arguments list
        // e.g., ['test'], [0], [:2], [1:], [1:2], [1:2:3]
        statements slices;
        bool is_slice = false;
        size_t start_pos = current;
        while (!is(token::close_square_bracket)) {
            if (is(token::colon)) {
                // A case where a default is used
                // e.g., [:2] will be parsed as [undefined, 2]
                slices.push_back(nullptr);
                ++current; // consume colon
                is_slice = true;
            } else {
                slices.push_back(parse_expression());
                if (is(token::colon)) {
                    ++current; // consume colon after expression, if it exists
                    is_slice = true;
                }
            }
        }
        if (is_slice) {
            statement_ptr start = slices.size() > 0 ? std::move(slices[0]) : nullptr;
            statement_ptr stop = slices.size() > 1 ? std::move(slices[1]) : nullptr;
            statement_ptr step = slices.size() > 2 ? std::move(slices[2]) : nullptr;
            return mk_stmt<slice_expression>(start_pos, std::move(start), std::move(stop), std::move(step));
        }
        return std::move(slices[0]);
    }

    statement_ptr parse_primary_expression() {
        size_t start_pos = current;
        auto t = tokens[current++];
        switch (t.t) {
            case token::numeric_literal:
                if (t.value.find('.') != std::string::npos) {
                    return mk_stmt<float_literal>(start_pos, std::stod(t.value));
                } else {
                    return mk_stmt<integer_literal>(start_pos, std::stoll(t.value));
                }
            case token::string_literal: {
                std::string val = t.value;
                while (is(token::string_literal)) {
                    val += tokens[current++].value;
                }
                return mk_stmt<string_literal>(start_pos, val);
            }
            case token::identifier:
                return mk_stmt<identifier>(start_pos, t.value);
            case token::open_paren: {
                auto expr = parse_expression_sequence();
                expect(token::close_paren, "Expected )");
                return expr;
            }
            case token::open_square_bracket: {
                statements vals;
                while (!is(token::close_square_bracket)) {
                    vals.push_back(parse_expression());
                    if (is(token::comma)) current++;
                }
                current++;
                return mk_stmt<array_literal>(start_pos, std::move(vals));
            }
            case token::open_curly_bracket: {
                std::vector<std::pair<statement_ptr, statement_ptr>> pairs;
                while (!is(token::close_curly_bracket)) {
                    auto key = parse_expression();
                    expect(token::colon, "Expected :");
                    pairs.push_back({std::move(key), parse_expression()});
                    if (is(token::comma)) current++;
                }
                current++;
                return mk_stmt<object_literal>(start_pos, std::move(pairs));
            }
            default:
                throw std::runtime_error("Unexpected token: " + t.value + " of type " + std::to_string(t.t));
        }
    }
};

program parse_from_tokens(const lexer_result & lexer_res) {
    return parser(lexer_res.tokens, lexer_res.source).parse();
}

} // namespace jinja
