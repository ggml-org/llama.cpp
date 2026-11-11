#include "runtime.h"

#include "lexer.h"
#include "utils.h"
#include "value.h"

#include <cmath>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define FILENAME "jinja-runtime"

bool g_jinja_debug = false;

namespace jinja {

void enable_debug(bool enable) {
    g_jinja_debug = enable;
}

static void render_statements(const statements & stmts, context & ctx, string & output) {
    for (const auto & stmt : stmts) {
        stmt->render(ctx, output);
        if (ctx.flow() != loop_flow::NONE) {
            break;
        }
    }
}

static value_string exec_statements(const statements & stmts, context & ctx) {
    auto output = mk_val<value_string>();
    render_statements(stmts, ctx, output->val_str);
    return output;
}

value_string runtime::execute(const program & prog) {
    auto result = exec_statements(prog.body, ctx);
    if (ctx.flow() != loop_flow::NONE) {
        throw std::runtime_error("Loop control used outside a loop");
    }
    return result;
}

static std::string get_line_col(const std::string & source, size_t pos) {
    size_t line = 1;
    size_t col = 1;
    for (size_t i = 0; i < pos && i < source.size(); i++) {
        if (source[i] == '\n') {
            line++;
            col = 1;
        } else {
            col++;
        }
    }
    return "line " + std::to_string(line) + ", column " + std::to_string(col);
}

static void ensure_key_type_allowed(const value & val) {
    if (!val->is_hashable()) {
        throw std::runtime_error("Type: " + val->type() + " is not allowed as object key");
    }
}

template <class Operation>
static decltype(auto) evaluate(const statement & node, context & ctx, Operation && operation) {
    std::exception_ptr pending;
    std::string message;
    try {
        return operation();
    } catch (const rethrown_exception & /* ex */) {
        pending = std::current_exception();
    } catch (const not_implemented_exception & /* ex */) {
        pending = std::current_exception();
    } catch (const std::exception & e) {
        const std::string & source = *ctx.src;
        if (source.empty()) {
            std::ostringstream oss;
            oss << "\nError executing " << node.type() << " at position " << node.pos << ": " << e.what();
            message = oss.str();
        } else {
            std::ostringstream oss;
            oss << "\n------------\n";
            oss << "While executing " << node.type() << " at " << get_line_col(source, node.pos) << " in source:\n";
            oss << peak_source(source, node.pos) << "\n";
            oss << "Error: " << e.what();
            message = oss.str();
        }
    }
    // throw outside the catch: MSVC keeps the unwound frames on the stack while a catch block runs
    if (pending) {
        std::rethrow_exception(pending);
    }
    // throw as another exception to avoid repeated formatting
    throw rethrown_exception(message);
}

value statement::execute(context & ctx) const {
    return evaluate(*this, ctx, [&] { return execute_impl(ctx); });
}

void statement::render(context & ctx, string & output) const {
    evaluate(*this, ctx, [&] { render_impl(ctx, output); });
}

void statement::render_impl(context & ctx, string & output) const {
    const auto result = execute_impl(ctx);
    if (!result) {
        return;
    }
    if (is_val<value_string>(result)) {
        output.append(cast_val<value_string>(result)->val_str);
    } else {
        output.append(runtime::gather_string_parts(result)->val_str);
    }
}

value identifier::execute_impl(context & ctx) const {
    auto it = ctx.find_val(val);
    if (it && !it->is_undefined()) {
        if (ctx.is_get_stats) {
            value_t::stats_t::mark_used(it);
        }
        JJ_DEBUG("Identifier '%s' found, type = %s", val.c_str(), it->type().c_str());
        return it;
    }
    const auto & builtins = global_builtins();
    auto builtin = builtins.find(val);
    if (builtin != builtins.end()) {
        JJ_DEBUG("Identifier '%s' found in builtins", val.c_str());
        return mk_val<value_func>(val, builtin->second);
    }
    JJ_DEBUG("Identifier '%s' not found, returning undefined", val.c_str());
    return mk_val<value_undefined>(val);
}

value object_literal::execute_impl(context & ctx) const {
    auto obj = mk_val<value_object>();
    for (const auto & pair : val) {
        value key = pair.first->execute(ctx);
        value val = pair.second->execute(ctx);
        JJ_DEBUG("Object literal: setting key '%s' with value type %s", key->as_string().str().c_str(), val->type().c_str());
        obj->insert(key, val);
    }
    return obj;
}

static value exec_binary_op(context & ctx, const token & op, value left_val, value right_val) {
    // Equality operators
    JJ_DEBUG("Executing binary expression %s '%s' %s", left_val->type().c_str(), op.value.c_str(), right_val->type().c_str());
    if (op.value == "==") {
        return mk_val<value_bool>(*left_val == *right_val);
    } else if (op.value == "!=") {
        return mk_val<value_bool>(!(*left_val == *right_val));
    }

    auto workaround_concat_null_with_str = [&](value & res) -> bool {
        bool is_left_null  = left_val->is_none()  || left_val->is_undefined();
        bool is_right_null = right_val->is_none() || right_val->is_undefined();
        bool is_left_str   = is_val<value_string>(left_val);
        bool is_right_str  = is_val<value_string>(right_val);
        if ((is_left_null && is_right_str) || (is_right_null && is_left_str)) {
            JJ_DEBUG("%s", "Workaround: treating null/undefined as empty string for string concatenation");
            string left_str  = is_left_null  ? string() : left_val->as_string();
            string right_str = is_right_null ? string() : right_val->as_string();
            auto output = left_str.append(right_str);
            res = mk_val<value_string>(std::move(output));
            return true;
        }
        return false;
    };

    auto test_is_in = [&]() -> bool {
        func_args args(ctx);
        args.push_back(left_val);
        args.push_back(right_val);
        return global_builtins().at("test_is_in")(args)->as_bool();
    };

    // Handle undefined and null values
    if (is_val<value_undefined>(left_val) || is_val<value_undefined>(right_val)) {
        if (is_val<value_undefined>(right_val) && (op.value == "in" || op.value == "not in")) {
            // Special case: `anything in undefined` is `false` and `anything not in undefined` is `true`
            return mk_val<value_bool>(op.value == "not in");
        }
        if (op.value == "+" || op.value == "~") {
            value res = mk_val<value_undefined>();
            if (workaround_concat_null_with_str(res)) {
                return res;
            }
        }
        throw std::runtime_error("Cannot perform operation " + op.value + " on undefined values");
    } else if (is_val<value_none>(left_val) || is_val<value_none>(right_val)) {
        if (!is_val<value_none>(right_val) && (op.value == "in" || op.value == "not in")) {
            // case: none in {'low': 1}
            // A null left operand is looked up like any other value.
            bool member = test_is_in();
            return mk_val<value_bool>(op.value == "in" ? member : !member);
        }
        if (op.value == "+" || op.value == "~") {
            value res = mk_val<value_undefined>();
            if (workaround_concat_null_with_str(res)) {
                return res;
            }
        }
        throw std::runtime_error("Cannot perform operation on null values");
    }

    // Float operations
    if ((is_val<value_int>(left_val) || is_val<value_float>(left_val)) &&
        (is_val<value_int>(right_val) || is_val<value_float>(right_val))) {
        double a = left_val->as_float();
        double b = right_val->as_float();
        if (op.value == "+" || op.value == "-" || op.value == "*") {
            double res = (op.value == "+") ? a + b : (op.value == "-") ? a - b : a * b;
            JJ_DEBUG("Arithmetic operation: %f %s %f = %f", a, op.value.c_str(), b, res);
            bool is_float = is_val<value_float>(left_val) || is_val<value_float>(right_val);
            if (is_float) {
                return mk_val<value_float>(res);
            } else {
                return mk_val<value_int>(static_cast<int64_t>(res));
            }
        } else if (op.value == "/") {
            JJ_DEBUG("Division operation: %f / %f", a, b);
            return mk_val<value_float>(a / b);
        } else if (op.value == "%") {
            double rem = std::fmod(a, b);
            JJ_DEBUG("Modulo operation: %f %% %f = %f", a, b, rem);
            bool is_float = is_val<value_float>(left_val) || is_val<value_float>(right_val);
            if (is_float) {
                return mk_val<value_float>(rem);
            } else {
                return mk_val<value_int>(static_cast<int64_t>(rem));
            }
        } else if (op.value == "<") {
            JJ_DEBUG("Comparison operation: %f < %f is %d", a, b, a < b);
            return mk_val<value_bool>(a < b);
        } else if (op.value == ">") {
            JJ_DEBUG("Comparison operation: %f > %f is %d", a, b, a > b);
            return mk_val<value_bool>(a > b);
        } else if (op.value == ">=") {
            JJ_DEBUG("Comparison operation: %f >= %f is %d", a, b, a >= b);
            return mk_val<value_bool>(a >= b);
        } else if (op.value == "<=") {
            JJ_DEBUG("Comparison operation: %f <= %f is %d", a, b, a <= b);
            return mk_val<value_bool>(a <= b);
        }
    }

    // Array operations
    if (is_val<value_array>(left_val) && is_val<value_array>(right_val)) {
        if (op.value == "+") {
            auto & left_arr = left_val->as_array();
            auto & right_arr = right_val->as_array();
            auto result = mk_val<value_array>();
            for (const auto & item : left_arr) {
                result->push_back(item);
            }
            for (const auto & item : right_arr) {
                result->push_back(item);
            }
            return result;
        }
    } else if (is_val<value_array>(right_val)) {
        // case: 1 in [0, 1, 2]
        bool member = test_is_in();
        if (op.value == "in") {
            return mk_val<value_bool>(member);
        } else if (op.value == "not in") {
            return mk_val<value_bool>(!member);
        }
    }

    // String concatenation with ~ and +
    if ((is_val<value_string>(left_val) || is_val<value_string>(right_val)) &&
            (op.value == "~" || op.value == "+")) {
        JJ_DEBUG("String concatenation with %s operator", op.value.c_str());
        auto output = left_val->as_string().append(right_val->as_string());
        auto res = mk_val<value_string>();
        res->val_str = std::move(output);
        return res;
    }

    // Python-style string repetition
    // TODO: support array/tuple repetition (e.g., [1, 2] * 3 → [1, 2, 1, 2, 1, 2])
    if (op.value == "*" &&
            ((is_val<value_string>(left_val) && is_val<value_int>(right_val)) ||
             (is_val<value_int>(left_val) && is_val<value_string>(right_val)))) {
        const auto & str = is_val<value_string>(left_val) ? left_val->as_string() : right_val->as_string();
        const int64_t repeat = is_val<value_int>(right_val) ? right_val->as_int() : left_val->as_int();
        auto res = mk_val<value_string>();
        if (repeat <= 0) {
            return res;
        }
        for (int64_t i = 0; i < repeat; ++i) {
            res->val_str.append(str);
        }
        return res;
    }

    // String membership
    if (is_val<value_string>(left_val) && is_val<value_string>(right_val)) {
        // case: "a" in "abc"
        bool member = test_is_in();
        if (op.value == "in") {
            return mk_val<value_bool>(member);
        } else if (op.value == "not in") {
            return mk_val<value_bool>(!member);
        }
    }

    // Value key in object
    if (is_val<value_object>(right_val)) {
        // case: key in {key: value}
        bool member = test_is_in();
        if (op.value == "in") {
            return mk_val<value_bool>(member);
        } else if (op.value == "not in") {
            return mk_val<value_bool>(!member);
        }
    }

    throw std::runtime_error("Unknown operator \"" + op.value + "\" between " + left_val->type() + " and " + right_val->type());
}

value binary_expression::execute_impl(context & ctx) const {
    value left_val = left->execute(ctx);

    // Logical operators
    if (op.value == "and") {
        JJ_DEBUG("Executing logical test: %s AND %s", left->type().c_str(), right->type().c_str());
        return left_val->as_bool() ? right->execute(ctx) : std::move(left_val);
    } else if (op.value == "or") {
        JJ_DEBUG("Executing logical test: %s OR %s", left->type().c_str(), right->type().c_str());
        return left_val->as_bool() ? std::move(left_val) : right->execute(ctx);
    }

    return exec_binary_op(ctx, op, std::move(left_val), right->execute(ctx));
}

value comparison_chain_expression::execute_impl(context & ctx) const {
    value left_val = operands[0]->execute(ctx);
    value res;
    for (size_t i = 0; i < ops.size(); ++i) {
        value right_val = operands[i + 1]->execute(ctx);
        res = exec_binary_op(ctx, ops[i], left_val, right_val);
        if (!res->as_bool()) {
            break; // same as "and", the next operands are not executed
        }
        left_val = std::move(right_val);
    }
    return res;
}

static void exec_call_args(context & ctx, const statements & arg_stmts, func_args & args) {
    for (const auto & arg_stmt : arg_stmts) {
        if (!is_stmt<spread_expression>(arg_stmt)) {
            auto arg_val = arg_stmt->execute(ctx);
            JJ_DEBUG("  Argument type: %s", arg_val->type().c_str());
            args.push_back(arg_val);
            continue;
        }
        auto spread = cast_stmt<spread_expression>(arg_stmt);
        value val = spread->argument->execute(ctx);
        JJ_DEBUG("  Unpacked argument type: %s", val->type().c_str());
        if (spread->is_kwargs) {
            if (!is_val<value_object>(val)) {
                throw std::runtime_error("Argument after ** must be an object, got " + val->type());
            }
            for (const auto & pair : val->as_ordered_object()) {
                if (!is_val<value_string>(pair.first)) {
                    throw std::runtime_error("Keyword argument key must be a string, got " + pair.first->type());
                }
                args.push_back(mk_val<value_kwarg>(pair.first->as_string().str(), pair.second));
            }
        } else {
            if (!is_val<value_array>(val)) {
                throw std::runtime_error("Argument after * must be an array, got " + val->type());
            }
            for (const auto & item : val->as_array()) {
                args.push_back(item);
            }
        }
    }
}

static value apply_filter(context & ctx, const statement_ptr & filter, value input) {
    JJ_DEBUG("Applying filter to %s", input->type().c_str());

    if (is_stmt<identifier>(filter)) {
        auto filter_id = cast_stmt<identifier>(filter)->val;
        JJ_DEBUG("Applying filter '%s' to %s", filter_id.c_str(), input->type().c_str());
        return apply_filter_by_name(ctx, filter_id, input, func_args(ctx));

    } else if (is_stmt<call_expression>(filter)) {
        auto call = cast_stmt<call_expression>(filter);
        if (!is_stmt<identifier>(call->callee)) {
            throw std::runtime_error("Filter callee must be an identifier");
        }
        auto filter_id = cast_stmt<identifier>(call->callee)->val;

        JJ_DEBUG("Applying filter '%s' with arguments to %s", filter_id.c_str(), input->type().c_str());
        func_args args(ctx);
        exec_call_args(ctx, call->args, args);

        return apply_filter_by_name(ctx, filter_id, input, args);

    } else {
        throw std::runtime_error("Invalid filter expression");
    }
}

value filter_expression::execute_impl(context & ctx) const {
    return apply_filter(ctx, filter, operand->execute(ctx));
}

value filter_statement::execute_impl(context & ctx) const {
    auto input = exec_statements(body, ctx);
    if (ctx.flow() != loop_flow::NONE) {
        return {};
    }
    return apply_filter(ctx, filter, input);
}

value test_expression::execute_impl(context & ctx) const {
    // NOTE: "value is something" translates to function call "test_is_something(value)"
    const auto & builtins = global_builtins();

    std::string test_id;
    value input = operand->execute(ctx);

    func_args args(ctx);
    args.push_back(input);

    if (is_stmt<identifier>(test)) {
        test_id = cast_stmt<identifier>(test)->val;
    } else if (is_stmt<call_expression>(test)) {
        auto call = cast_stmt<call_expression>(test);
        if (!is_stmt<identifier>(call->callee)) {
            throw std::runtime_error("Test callee must be an identifier");
        }
        test_id = cast_stmt<identifier>(call->callee)->val;

        JJ_DEBUG("Applying test '%s' with arguments to %s", test_id.c_str(), input->type().c_str());
        exec_call_args(ctx, call->args, args);

    } else {
        throw std::runtime_error("Invalid test expression");
    }

    const std::string test_name = "test_is_" + test_id;
    auto it = builtins.find(test_name);
    JJ_DEBUG("Test expression %s '%s' %s (using function '%s')", operand->type().c_str(), test_id.c_str(), negate ? "(negate)" : "", test_name.c_str());
    if (it == builtins.end()) {
        throw std::runtime_error("Unknown test '" + test_id + "'");
    }

    if (ctx.is_get_stats) {
        value_t::stats_t::mark_used(input);
        input->stats.ops.insert(test_name);
    }

    auto res = it->second(args);

    if (negate) {
        return mk_val<value_bool>(!res->as_bool());
    } else {
        return res;
    }
}

value unary_expression::execute_impl(context & ctx) const {
    value operand_val = argument->execute(ctx);
    JJ_DEBUG("Executing unary expression with operator '%s'", op.value.c_str());

    if (op.value == "not") {
        return mk_val<value_bool>(!operand_val->as_bool());
    } else if (op.value == "-") {
        if (is_val<value_int>(operand_val)) {
            return mk_val<value_int>(-operand_val->as_int());
        } else if (is_val<value_float>(operand_val)) {
            return mk_val<value_float>(-operand_val->as_float());
        } else {
            throw std::runtime_error("Unary - operator requires numeric operand");
        }
    } else if (op.value == "+") {
        if (is_val<value_int>(operand_val) || is_val<value_float>(operand_val)) {
            return operand_val;
        }
        throw std::runtime_error("Unary + operator requires numeric operand");
    }

    throw std::runtime_error("Unknown unary operator '" + op.value + "'");
}

value if_statement::execute_impl(context & ctx) const {
    auto output = mk_val<value_string>();
    render_impl(ctx, output->val_str);
    return output;
}

void if_statement::render_impl(context & ctx, string & output) const {
    auto condition = test->execute(ctx);
    render_statements(condition->as_bool() ? body : alternate, ctx, output);
}

namespace {
class iterable_cursor {
  public:
    explicit iterable_cursor(value input, bool object_items = false) :
        input(std::move(input)),
        object_items(object_items) {
        if (!is_val<value_array>(this->input) && !is_val<value_object>(this->input) && !this->input->is_undefined()) {
            throw std::runtime_error("Expected iterable or object type in for loop: got " + this->input->type());
        }
    }

    uint64_t size() const {
        if (is_val<value_array>(input)) {
            return cast_val<value_array>(input)->size();
        }
        if (is_val<value_object>(input)) {
            return input->as_ordered_object().size();
        }
        return 0;
    }

    value next() {
        if (exhausted || offset >= size()) {
            exhausted = true;
            return {};
        }
        if (is_val<value_array>(input)) {
            return cast_val<value_array>(input)->element(offset++);
        }
        const auto & entry = input->as_ordered_object()[offset++];
        return object_items ? mk_val<value_tuple>(entry) : entry.first;
    }
  private:
    value    input;
    bool     object_items;
    bool     exhausted = false;
    uint64_t offset    = 0;
};

class variable_binding {
  public:
    explicit variable_binding(const statement_ptr & target) {
        if (const auto * id = cast_stmt<identifier>(target)) {
            names_.push_back(id->val);
        } else if (const auto * tuple = cast_stmt<tuple_literal>(target)) {
            unpack_ = true;
            for (const auto & item : tuple->val) {
                const auto * id = cast_stmt<identifier>(item);
                if (!id) {
                    throw std::runtime_error("Cannot unpack non-identifier type");
                }
                names_.push_back(id->val);
            }
        } else {
            throw std::runtime_error("Invalid loop variable: " + target->type());
        }
    }

    void bind(context & scope, const value & item) const {
        if (!unpack_) {
            scope.set_val(names_.front(), item);
            return;
        }
        iterable_cursor values(item);
        for (const auto & name : names_) {
            auto v = values.next();
            if (!v) {
                throw std::runtime_error("Too few items to unpack");
            }
            scope.set_val(name, v);
        }
        if (values.next()) {
            throw std::runtime_error("Too many items to unpack");
        }
    }

    bool unpack() const { return unpack_; }

  private:
    std::vector<std::string> names_;
    bool                     unpack_ = false;
};

class loop_value final : public value_object_t {
  public:
    loop_value(value input, context & parent, variable_binding binding, const statement * filter) :
        source_(std::move(input), binding.unpack()),
        filter_scope_(parent),
        binding_(std::move(binding)),
        filter_(filter) {
        has_builtins = false;
    }

    bool advance(context & body) {
        auto item = take();
        if (!item) {
            return false;
        }
        previous_ = std::move(current_);
        current_  = std::move(item);
        ++index_;
        key_index.clear();
        val_obj.clear();
        materialized_ = false;
        binding_.bind(body, current_);
        return true;
    }

    bool as_bool() const override { return true; }

    value & at(const value & key, value & fallback) override {
        filter_scope_.ensure_active();
        if (!key_index.empty()) {
            auto found = key_index.find(key);
            if (found != key_index.end()) {
                return val_obj[found->second].second;
            }
        }
        const auto * text = cast_val<value_string>(key);
        if (!text) {
            return fallback;
        }
        const auto & name = text->val_str;
        if (name.equals("index")) {
            cached_ = number(index_);
        } else if (name.equals("index0")) {
            cached_ = number(index_ - 1);
        } else if (name.equals("first")) {
            cached_ = mk_val<value_bool>(index_ == 1);
        } else if (name.equals("last")) {
            cached_ = mk_val<value_bool>(!peek());
        } else if (name.equals("length")) {
            cached_ = number(length());
        } else if (name.equals("revindex")) {
            cached_ = number(length() - index_ + 1);
        } else if (name.equals("revindex0")) {
            cached_ = number(length() - index_);
        } else if (name.equals("previtem")) {
            cached_ = previous_ ? previous_ : mk_val<value_undefined>("previtem");
        } else if (name.equals("nextitem")) {
            const auto item = peek();
            cached_         = item ? item : mk_val<value_undefined>("nextitem");
        } else {
            return fallback;
        }
        return cached_;
    }

    value & at(const value & key) override {
        value  missing;
        auto & result = at(key, missing);
        if (!result) {
            throw std::runtime_error("Unknown loop attribute: " + key->as_string().str());
        }
        return result;
    }

    bool has_key(const value & key) override {
        const auto name = key->as_string().str();
        for (const auto * field : fields) {
            if (name == field) {
                return true;
            }
        }
        return value_object_t::has_key(key);
    }

    const std::vector<std::pair<value, value>> & as_ordered_object() const override {
        const_cast<loop_value *>(this)->materialize_fields();
        return val_obj;
    }

    string as_string() const override {
        const_cast<loop_value *>(this)->materialize_fields();
        return value_object_t::as_string();
    }

    void insert(const value & key, const value & val) override {
        materialize_fields();
        value_object_t::insert(key, val);
    }

  private:
    static constexpr const char * fields[]      = { "index", "index0", "revindex", "revindex0", "first",
                                                    "last",  "length", "previtem", "nextitem" };
    bool                          materialized_ = false;

    void materialize_fields() {
        if (materialized_) {
            return;
        }
        for (const auto * field : fields) {
            value key = mk_val<value_string>(field);
            value missing;
            value item = at(key, missing);
            value_object_t::insert(key, item);
        }
        materialized_ = true;
    }

    static value number(uint64_t n) {
        if (n > INT64_MAX) {
            throw std::runtime_error("loop length exceeds int64");
        }
        return mk_val<value_int>(static_cast<int64_t>(n));
    }

    value read() {
        filter_scope_.ensure_active();
        while (auto item = source_.next()) {
            if (filter_) {
                filter_scope_.clear_locals();
                binding_.bind(filter_scope_, item);
                if (!filter_->execute(filter_scope_)->as_bool()) {
                    continue;
                }
            }
            return item;
        }
        return {};
    }

    value take() {
        if (pending_.empty()) {
            return read();
        }
        auto item = std::move(pending_.front());
        pending_.pop_front();
        return item;
    }

    value peek() {
        if (pending_.empty()) {
            auto item = read();
            if (item) {
                pending_.push_back(std::move(item));
            }
        }
        return pending_.empty() ? value{} : pending_.front();
    }

    uint64_t length() {
        if (!length_) {
            if (filter_) {
                while (auto item = read()) {
                    pending_.push_back(std::move(item));
                }
                length_ = index_ + pending_.size();
            } else {
                length_ = source_.size();
            }
        }
        return *length_;
    }

    iterable_cursor         source_;
    context                 filter_scope_;
    variable_binding        binding_;
    const statement *       filter_;
    std::deque<value>       pending_;
    std::optional<uint64_t> length_;
    uint64_t                index_ = 0;
    value                   current_, previous_, cached_;
};
}  // namespace

value for_statement::execute_impl(context & ctx) const {
    auto output = mk_val<value_string>();
    render_impl(ctx, output->val_str);
    return output;
}

void for_statement::render_impl(context & ctx, string & output) const {
    const auto * selected = cast_stmt<select_expression>(iterable);
    auto         input    = (selected ? selected->lhs : iterable)->execute(ctx);
    if (ctx.is_get_stats) {
        value_t::stats_t::mark_used(input);
        input->stats.ops.insert(is_val<value_object>(input) ? "object_access" : "array_access");
    }
    auto loop         = std::make_shared<loop_value>(std::move(input), ctx, variable_binding(loopvar),
                                             selected ? selected->test.get() : nullptr);
    bool no_iteration = true;
    for (;;) {
        context body_scope(ctx);
        if (!loop->advance(body_scope)) {
            break;
        }
        body_scope.set_val("loop", loop);
        render_statements(body, body_scope, output);
        const auto flow = std::exchange(ctx.flow(), loop_flow::NONE);
        if (flow == loop_flow::CONTINUE) {
            continue;
        }
        if (flow == loop_flow::BREAK) {
            break;
        }
        no_iteration = false;
    }
    if (no_iteration) {
        render_statements(default_block, ctx, output);
    }
}

value set_statement::execute_impl(context & ctx) const {
    auto rhs = val ? val->execute(ctx) : exec_statements(body, ctx);
    if (ctx.flow() != loop_flow::NONE) {
        return {};
    }

    if (is_stmt<identifier>(assignee)) {
        // case: {% set my_var = value %}
        auto var_name = cast_stmt<identifier>(assignee)->val;
        JJ_DEBUG("Setting global variable '%s' with value type %s", var_name.c_str(), rhs->type().c_str());
        ctx.set_val(var_name, rhs);

    } else if (is_stmt<tuple_literal>(assignee)) {
        // case: {% set a, b = value %}
        auto tuple = cast_stmt<tuple_literal>(assignee);
        if (!is_val<value_array>(rhs)) {
            throw std::runtime_error("Cannot unpack non-iterable type in set: " + rhs->type());
        }
        auto & arr = rhs->as_array();
        if (arr.size() != tuple->val.size()) {
            throw std::runtime_error(std::string("Too ") + (tuple->val.size() > arr.size() ? "few" : "many") + " items to unpack in set");
        }
        for (size_t i = 0; i < tuple->val.size(); ++i) {
            auto & elem = tuple->val[i];
            if (!is_stmt<identifier>(elem)) {
                throw std::runtime_error("Cannot unpack to non-identifier in set: " + elem->type());
            }
            auto var_name = cast_stmt<identifier>(elem)->val;
            ctx.set_val(var_name, arr[i]);
        }

    } else if (is_stmt<member_expression>(assignee)) {
        // case: {% set ns.my_var = value %}
        auto member = cast_stmt<member_expression>(assignee);
        if (member->computed) {
            throw std::runtime_error("Cannot assign to computed member");
        }
        if (!is_stmt<identifier>(member->property)) {
            throw std::runtime_error("Cannot assign to member with non-identifier property");
        }
        auto prop_name = cast_stmt<identifier>(member->property)->val;

        value object = member->object->execute(ctx);
        if (!is_val<value_object>(object)) {
            throw std::runtime_error("Cannot assign to member of non-object");
        }
        auto obj_ptr = cast_val<value_object>(object);
        JJ_DEBUG("Setting object property '%s' with value type %s", prop_name.c_str(), rhs->type().c_str());
        obj_ptr->insert(cached_val<value_string>(ctx, member->static_property), rhs);

    } else {
        throw std::runtime_error("Invalid LHS inside assignment expression: " + assignee->type());
    }
    return {};
}

static inline void bind_parameters(const std::string & name, const statements & this_args, const func_args & args, context & ctx) {
    const size_t expected_count = this_args.size();
    const size_t input_count = args.count();

    JJ_DEBUG("Invoking '%s' with %zu input arguments (expected %zu)", name.c_str(), input_count, expected_count);
    for (size_t i = 0; i < expected_count; ++i) {
        if (i < input_count) {
            if (is_stmt<identifier>(this_args[i])) {
                // normal parameter
                std::string param_name = cast_stmt<identifier>(this_args[i])->val;
                value param_value = args.get_kwarg_or_pos(param_name, i);
                JJ_DEBUG("  Binding parameter '%s' to argument of type %s", param_name.c_str(), param_value->type().c_str());
                ctx.set_val(param_name, param_value);
            } else if (is_stmt<keyword_argument_expression>(this_args[i])) {
                // default argument used as normal parameter
                auto kwarg = cast_stmt<keyword_argument_expression>(this_args[i]);
                if (!is_stmt<identifier>(kwarg->key)) {
                    throw std::runtime_error("Keyword argument key must be an identifier in '" + name + "'");
                }
                std::string param_name = cast_stmt<identifier>(kwarg->key)->val;
                value param_value = args.get_kwarg_or_pos(param_name, i);
                JJ_DEBUG("  Binding parameter '%s' to argument of type %s", param_name.c_str(), param_value->type().c_str());
                ctx.set_val(param_name, param_value);
            } else {
                throw std::runtime_error("Invalid parameter type in '" + name + "'");
            }
        } else {
            auto & default_arg = this_args[i];
            if (is_stmt<keyword_argument_expression>(default_arg)) {
                auto kwarg = cast_stmt<keyword_argument_expression>(default_arg);
                if (!is_stmt<identifier>(kwarg->key)) {
                    throw std::runtime_error("Keyword argument key must be an identifier in '" + name + "'");
                }
                std::string param_name = cast_stmt<identifier>(kwarg->key)->val;
                JJ_DEBUG("  Binding parameter '%s' to default argument of type %s", param_name.c_str(), kwarg->val->type().c_str());
                ctx.set_val(param_name, kwarg->val->execute(args.ctx));
            } else {
                throw std::runtime_error("Not enough arguments provided to '" + name + "'");
            }
            //std::string param_name = cast_stmt<identifier>(default_args[i])->val;
            //JJ_DEBUG("  Binding parameter '%s' to default", param_name.c_str());
            //ctx.var[param_name] = default_args[i]->execute(ctx);
        }
    }
}

value macro_statement::execute_impl(context & ctx) const {
    if (!is_stmt<identifier>(this->name)) {
        throw std::runtime_error("Macro name must be an identifier");
    }
    std::string name = cast_stmt<identifier>(this->name)->val;

    const func_handler func = [this, name](const func_args & args) -> value {
        if (args.ctx.macro_depth >= JINJA_RECURSIVE_LIMIT) {
            throw std::runtime_error("macro calls are nested more than " + std::to_string(JINJA_RECURSIVE_LIMIT) + " levels deep");
        }
        context macro_ctx(args.ctx);
        macro_ctx.macro_depth = args.ctx.macro_depth + 1;

        bind_parameters(name, this->args, args, macro_ctx);

        // execute macro body
        JJ_DEBUG("Executing macro '%s' body with %zu statements", name.c_str(), this->body.size());
        auto res = exec_statements(this->body, macro_ctx);
        JJ_DEBUG("Macro '%s' execution complete, result: %s", name.c_str(), res->val_str.str().c_str());
        return res;
    };

    JJ_DEBUG("Defining macro '%s' with %zu parameters", name.c_str(), args.size());
    ctx.set_val(name, mk_val<value_func>(name, func));
    return {};
}

value call_statement::execute_impl(context & ctx) const {
    auto call_expr = cast_stmt<call_expression>(this->call);
    if (!call_expr) {
        throw std::runtime_error("Call statement requires a valid call expression");
    }

    value callee_val = call_expr->callee->execute(ctx);
    if (!is_val<value_func>(callee_val)) {
        throw std::runtime_error("Callee is not a function: got " + callee_val->type());
    }
    auto * callee_func = cast_val<value_func>(callee_val);

    const func_handler func = [this, caller_ctx = ctx.snapshot()](const func_args & args) -> value {
        if (args.ctx.macro_depth >= JINJA_RECURSIVE_LIMIT) {
            throw std::runtime_error("macro calls are nested more than " + std::to_string(JINJA_RECURSIVE_LIMIT) + " levels deep");
        }
        context block_ctx(caller_ctx);
        block_ctx.macro_depth = args.ctx.macro_depth + 1;

        bind_parameters("caller", this->caller_args, args, block_ctx);

        JJ_DEBUG("Executing call body with %zu statements", this->body.size());
        auto res = exec_statements(this->body, block_ctx);
        JJ_DEBUG("Call body execution complete, result: %s", res->val_str.str().c_str());
        return res;
    };

    context call_ctx(ctx);
    call_ctx.set_val("caller", mk_val<value_func>("caller", func));

    func_args args(call_ctx);
    exec_call_args(ctx, call_expr->args, args);

    JJ_DEBUG("Calling macro '%s' with %zu arguments", callee_func->name.c_str(), args.count());
    return callee_func->invoke(args);
}

member_expression::member_expression(statement_ptr && object, statement_ptr && property, bool computed) :
    object(std::move(object)),
    property(std::move(property)),
    computed(computed) {
    chk_type<expression>(this->object);
    chk_type<expression>(this->property);
    if (!computed) {
        if (const auto * id = cast_stmt<identifier>(this->property)) {
            static_property = id->key;
        }
    }
}

value member_expression::execute_impl(context & ctx) const {
    value object = this->object->execute(ctx);

    value property;
    if (this->computed) {
        // syntax: obj[expr]
        JJ_DEBUG("Member expression, computing property type %s", this->property->type().c_str());

        if (is_stmt<slice_expression>(this->property)) {
            int64_t arr_size = 0;
            if (is_val<value_array>(object)) {
                auto count = cast_val<value_array>(object)->size();
                if (count > INT64_MAX) {
                    throw std::length_error("array length exceeds int64");
                }
                arr_size = static_cast<int64_t>(count);
            } else if (is_val<value_string>(object)) {
                arr_size = static_cast<int64_t>(cast_val<value_string>(object)->val_str.length());
            }
            auto s = cast_stmt<slice_expression>(this->property);
            value step_val  = s->step_expr  ? s->step_expr->execute(ctx)  : mk_val<value_int>(1);
            value start_val = s->start_expr ? s->start_expr->execute(ctx) : (step_val->as_int() < 0 ? mk_val<value_int>(arr_size - 1) : mk_val<value_int>(0));
            value stop_val  = s->stop_expr  ? s->stop_expr->execute(ctx)  : (step_val->as_int() < 0 ? mk_val<value_int>(-1) : mk_val<value_int>(arr_size));

            // translate to function call: obj.slice(start, stop, step)
            JJ_DEBUG("Member expression is a slice: start %s, stop %s, step %s",
                     start_val->as_repr().c_str(),
                     stop_val->as_repr().c_str(),
                     step_val->as_repr().c_str());
            auto slice_func = try_builtin_func(ctx, "slice", object);
            func_args args(ctx);
            args.push_back(start_val);
            args.push_back(stop_val);
            args.push_back(step_val);
            return slice_func->invoke(args);
        } else {
            property = this->property->execute(ctx);
        }
    } else if (is_stmt<integer_literal>(this->property)) {
        // syntax: obj.index
        property = this->property->execute(ctx);
        if (property->as_int() < 0) {
            throw std::runtime_error("Static member property cannot be negative");
        }
    } else {
        // syntax: obj.prop
        if (!is_stmt<identifier>(this->property)) {
            throw std::runtime_error("Static member property must be an identifier");
        }
        property = cached_val<value_string>(ctx, static_property);
        const auto & prop = cast_stmt<identifier>(this->property)->val;
        JJ_DEBUG("Member expression, object type %s, static property '%s'", object->type().c_str(), prop.c_str());

        // behavior of jinja2: obj having prop as a built-in function AND 'prop', as an object key,
        // then obj.prop returns the built-in function, not the property value.
        // while obj['prop'] returns the property value.
        // example: {"obj": {"items": 123}} -> obj.items is the built-in function, obj['items'] is 123

        value val = find_builtin_func(ctx, prop, object);
        if (val) {
            return val;
        }
        // else, fallthrough to normal property access below
    }

    JJ_DEBUG("Member expression on object type %s, property type %s", object->type().c_str(), property->type().c_str());
    value val;

    if (property->is_undefined()) {
        JJ_DEBUG("%s", "Member expression property is undefined, returning undefined");
        return mk_val<value_undefined>("object_property");
    }

    ensure_key_type_allowed(property);

    if (is_val<value_undefined>(object)) {
        JJ_DEBUG("%s", "Accessing property on undefined object, returning undefined");
        return mk_val<value_undefined>("object_property");

    } else if (is_val<value_object>(object)) {
        val = object->find(property);
        if (!val || is_val<value_undefined>(val)) {
            val = try_builtin_func(ctx, property->as_string().str(), object, true);
        }
        JJ_DEBUG("Accessed property '%s' value, got type: %s", property->as_string().str().c_str(), val->type().c_str());

    } else if (is_val<value_array>(object) || is_val<value_string>(object)) {
        if (is_val<value_int>(property) || is_val<value_bool>(property)) {
            int64_t index = property->as_int();
            JJ_DEBUG("Accessing %s index %d", object->type().c_str(), (int)index);
            if (is_val<value_array>(object)) {
                const auto * arr = cast_val<value_array>(object);
                const auto count = arr->size();
                if (index < 0) {
                    // Avoid negating INT64_MIN.
                    const uint64_t distance = static_cast<uint64_t>(-(index + 1)) + 1;
                    if (distance <= count) {
                        val = arr->element(count - distance);
                    }
                } else if (static_cast<uint64_t>(index) < count) {
                    val = arr->element(static_cast<uint64_t>(index));
                }
            } else { // value_string
                if (index >= 0) {
                    const auto byte = cast_val<value_string>(object)->val_str.byte_at(static_cast<size_t>(index));
                    if (byte) {
                        val = mk_val<value_string>(std::string(1, *byte));
                    }
                }
            }

        } else if (is_val<value_string>(property)) {
            auto key = property->as_string().str();
            JJ_DEBUG("Accessing %s built-in '%s'", is_val<value_array>(object) ? "array" : "string", key.c_str());
            val = try_builtin_func(ctx, key, object, true);

        }
    } else {
        if (!is_val<value_string>(property)) {
            throw std::runtime_error("Cannot access property with non-string: got " + property->type());
        }
        auto key = property->as_string().str();
        val = try_builtin_func(ctx, key, object, true);
    }

    if (!val) {
        val = mk_val<value_undefined>("object_property");
    }
    if (ctx.is_get_stats && object && property) {
        value_t::stats_t::mark_used(val);
        value_t::stats_t::mark_used(object);
        value_t::stats_t::mark_used(property);
        if (is_val<value_object>(object) || is_val<value_string>(property) || is_val<value_float>(property) || is_val<value_array>(property) || is_val<value_none>(property)) {
            object->stats.ops.insert("object_access");
        } else if (is_val<value_int>(property) || is_val<value_bool>(property)) {
            object->stats.ops.insert("array_access");
        }
    }

    return val;
}

value call_expression::execute_impl(context & ctx) const {
    // gather arguments
    func_args args(ctx);
    exec_call_args(ctx, this->args, args);
    // execute callee
    value callee_val = callee->execute(ctx);
    if (!is_val<value_func>(callee_val)) {
        throw std::runtime_error("Callee is not a function: got " + callee_val->type());
    }
    auto * callee_func = cast_val<value_func>(callee_val);
    JJ_DEBUG("Calling function '%s' with %zu arguments", callee_func->name.c_str(), args.count());
    return callee_func->invoke(args);
}

value keyword_argument_expression::execute_impl(context & ctx) const {
    if (!is_stmt<identifier>(key)) {
        throw std::runtime_error("Keyword argument key must be identifiers");
    }

    std::string k = cast_stmt<identifier>(key)->val;
    JJ_DEBUG("Keyword argument expression key: %s, value: %s", k.c_str(), val->type().c_str());

    value v = val->execute(ctx);
    JJ_DEBUG("Keyword argument value executed, type: %s", v->type().c_str());

    return mk_val<value_kwarg>(k, v);
}

std::string runtime::debug_dump_program(const program & prog, const std::string & src) {
    std::ostringstream oss;
    size_t lvl = 0;
    context ctx;
    ctx.src.reset(new std::string(src));

    auto indent = [](size_t lvl) -> std::string {
        return std::string(lvl * 2, ' ');
    };

    ctx.visitor = [&](bool is_leaf, const statement * node, std::vector<visitor_pair> children) {
        oss << indent(lvl) << node->type() << ":\n";
        lvl++;
        if (is_leaf) {
            const auto & pos = node->pos;
            oss << indent(lvl) << "(leaf) at " << get_line_col(src, pos) << " in source:\n";
            std::string snippet = peak_source(src, pos);
            string_replace_all(snippet, "\n", "\n" + indent(lvl));
            oss << indent(lvl) << snippet << "\n";
        } else {
            for (auto & [label, children_vec] : children) {
                oss << indent(lvl) << label << ":\n";
                lvl++;
                if (children_vec.empty()) {
                    oss << indent(lvl) << "<empty>\n\n";
                } else {
                    for (auto * child : children_vec) {
                        if (!child) {
                            continue;
                        }
                        child->visit(ctx);
                    }
                }
                lvl--;
            }
        }
        lvl--;
    };

    for (const auto & stmt : prog.body) {
        stmt->visit(ctx);
    }

    return oss.str();
}

} // namespace jinja
