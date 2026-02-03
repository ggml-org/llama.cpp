#include "lexer.h"
#include "runtime.h"
#include "value.h"
#include "utils.h"

#include <string>
#include <vector>
#include <memory>
#include <cmath>

#define FILENAME "jinja-runtime"

bool g_jinja_debug = false;

namespace jinja {

void enable_debug(bool enable) {
    g_jinja_debug = enable;
}

static value_string exec_statements(const statements & stmts, context & ctx) {
    auto result = mk_val<value_array>();
    for (const auto & stmt : stmts) {
        JJ_DEBUG("Executing statement of type %s", stmt->type().c_str());
        result->push_back(stmt->execute(ctx));
    }
    // convert to string parts
    value_string str = mk_val<value_string>();
    gather_string_parts_recursive(result, str);
    return str;
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

// execute with error handling
value statement::execute(context & ctx) {
    try {
        return execute_impl(ctx);
    } catch (const continue_statement::signal & /* ex */) {
        throw;
    } catch (const break_statement::signal & /* ex */) {
        throw;
    } catch (const rethrown_exception & /* ex */) {
        throw;
    } catch (const not_implemented_exception & /* ex */) {
        throw;
    } catch (const std::exception & e) {
        const std::string & source = *ctx.src;
        if (source.empty()) {
            std::ostringstream oss;
            oss << "\nError executing " << type() << " at position " << pos << ": " << e.what();
            throw rethrown_exception(oss.str());
        } else {
            std::ostringstream oss;
            oss << "\n------------\n";
            oss << "While executing " << type() << " at " << get_line_col(source, pos) << " in source:\n";
            oss << peak_source(source, pos) << "\n";
            oss << "Error: " << e.what();
            // throw as another exception to avoid repeated formatting
            throw rethrown_exception(oss.str());
        }
    }
}

value identifier::execute_impl(context & ctx) {
    auto it = ctx.get_val(val);
    auto builtins = global_builtins();
    if (!it->is_undefined()) {
        if (ctx.is_get_stats) {
            it->stats.used = true;
        }
        JJ_DEBUG("Identifier '%s' found, type = %s", val.c_str(), it->type().c_str());
        return it;
    } else if (builtins.find(val) != builtins.end()) {
        JJ_DEBUG("Identifier '%s' found in builtins", val.c_str());
        return mk_val<value_func>(val, builtins.at(val));
    } else {
        JJ_DEBUG("Identifier '%s' not found, returning undefined", val.c_str());
        return mk_val<value_undefined>(val);
    }
}

value object_literal::execute_impl(context & ctx) {
    auto obj = mk_val<value_object>();
    for (const auto & pair : val) {
        value key = pair.first->execute(ctx);
        value val = pair.second->execute(ctx);
        JJ_DEBUG("Object literal: setting key '%s' with value type %s", key->as_string().str().c_str(), val->type().c_str());
        obj->insert(key, val);
    }
    return obj;
}

value binary_expression::execute_impl(context & ctx) {
    value left_val = left->execute(ctx);

    // Logical operators
    if (op.value == "and") {
        return left_val->as_bool() ? right->execute(ctx) : std::move(left_val);
    } else if (op.value == "or") {
        return left_val->as_bool() ? std::move(left_val) : right->execute(ctx);
    }

    // Equality operators
    value right_val = right->execute(ctx);
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

static value try_builtin_func(context & ctx, const std::string & name, value & input, bool undef_on_missing = false) {
    JJ_DEBUG("Trying built-in function '%s' for type %s", name.c_str(), input->type().c_str());
    if (ctx.is_get_stats) {
        input->stats.used = true;
        input->stats.ops.insert(name);
    }
    auto builtins = input->get_builtins();
    auto it = builtins.find(name);
    if (it != builtins.end()) {
        JJ_DEBUG("Binding built-in '%s'", name.c_str());
        return mk_val<value_func>(name, it->second, input);
    }
    if (undef_on_missing) {
        return mk_val<value_undefined>(name);
    }
    throw std::runtime_error("Unknown (built-in) filter '" + name + "' for type " + input->type());
}

value filter_expression::execute_impl(context & ctx) {
    value input = operand ? operand->execute(ctx) : val;

    JJ_DEBUG("Applying filter to %s", input->type().c_str());

    if (is_stmt<identifier>(filter)) {
        auto filter_id = cast_stmt<identifier>(filter)->val;

        if (filter_id == "trim") {
            filter_id = "strip"; // alias
        }
        JJ_DEBUG("Applying filter '%s' to %s", filter_id.c_str(), input->type().c_str());
        return try_builtin_func(ctx, filter_id, input)->invoke(func_args(ctx));

    } else if (is_stmt<call_expression>(filter)) {
        auto call = cast_stmt<call_expression>(filter);
        if (!is_stmt<identifier>(call->callee)) {
            throw std::runtime_error("Filter callee must be an identifier");
        }
        auto filter_id = cast_stmt<identifier>(call->callee)->val;

        if (filter_id == "trim") {
            filter_id = "strip"; // alias
        }
        JJ_DEBUG("Applying filter '%s' with arguments to %s", filter_id.c_str(), input->type().c_str());
        func_args args(ctx);
        for (const auto & arg_expr : call->args) {
            args.push_back(arg_expr->execute(ctx));
        }

        return try_builtin_func(ctx, filter_id, input)->invoke(args);

    } else {
        throw std::runtime_error("Invalid filter expression");
    }
}

value filter_statement::execute_impl(context & ctx) {
    // eval body as string, then apply filter
    auto body_val = exec_statements(body, ctx);
    value_string parts = mk_val<value_string>();
    gather_string_parts_recursive(body_val, parts);

    JJ_DEBUG("FilterStatement: applying filter to body string of length %zu", parts->val_str.length());
    filter_expression filter_expr(std::move(parts), std::move(filter));
    value out = filter_expr.execute(ctx);

    // this node can be reused later, make sure filter is preserved
    this->filter = std::move(filter_expr.filter);
    return out;
}

value test_expression::execute_impl(context & ctx) {
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
        for (const auto & arg_expr : call->args) {
            args.push_back(arg_expr->execute(ctx));
        }

    } else {
        throw std::runtime_error("Invalid test expression");
    }

    auto it = builtins.find("test_is_" + test_id);
    JJ_DEBUG("Test expression %s '%s' %s (using function 'test_is_%s')", operand->type().c_str(), test_id.c_str(), negate ? "(negate)" : "", test_id.c_str());
    if (it == builtins.end()) {
        throw std::runtime_error("Unknown test '" + test_id + "'");
    }

    auto res = it->second(args);

    if (negate) {
        return mk_val<value_bool>(!res->as_bool());
    } else {
        return res;
    }
}

value unary_expression::execute_impl(context & ctx) {
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
    }

    throw std::runtime_error("Unknown unary operator '" + op.value + "'");
}

value if_statement::execute_impl(context & ctx) {
    value test_val = test->execute(ctx);

    auto out = mk_val<value_array>();
    if (test_val->as_bool()) {
        for (auto & stmt : body) {
            JJ_DEBUG("IF --> Executing THEN body, current block: %s", stmt->type().c_str());
            out->push_back(stmt->execute(ctx));
        }
    } else {
        for (auto & stmt : alternate) {
            JJ_DEBUG("IF --> Executing ELSE body, current block: %s", stmt->type().c_str());
            out->push_back(stmt->execute(ctx));
        }
    }
    // convert to string parts
    value_string str = mk_val<value_string>();
    gather_string_parts_recursive(out, str);
    return str;
}

value for_statement::execute_impl(context & ctx) {
    context scope(ctx); // new scope for loop variables

    jinja::select_expression * select_expr = cast_stmt<select_expression>(iterable);
    statement_ptr test_expr_nullptr;

    statement_ptr & iter_expr = [&]() -> statement_ptr & {
        auto tmp = cast_stmt<select_expression>(iterable);
        return tmp ? tmp->lhs : iterable;
    }();
    statement_ptr & test_expr = [&]() -> statement_ptr & {
        auto tmp = cast_stmt<select_expression>(iterable);
        return tmp ? tmp->test : test_expr_nullptr;
    }();

    JJ_DEBUG("Executing for statement, iterable type: %s", iter_expr->type().c_str());

    value iterable_val = iter_expr->execute(scope);

    if (iterable_val->is_undefined()) {
        JJ_DEBUG("%s", "For loop iterable is undefined, skipping loop");
        iterable_val = mk_val<value_array>();
    }

    if (!is_val<value_array>(iterable_val) && !is_val<value_object>(iterable_val)) {
        throw std::runtime_error("Expected iterable or object type in for loop: got " + iterable_val->type());
    }

    std::vector<value> items;
    if (is_val<value_object>(iterable_val)) {
        JJ_DEBUG("%s", "For loop over object keys");
        auto & obj = iterable_val->as_ordered_object();
        for (auto & p : obj) {
            auto tuple = mk_val<value_tuple>(p);
            items.push_back(std::move(tuple));
        }
        if (ctx.is_get_stats) {
            iterable_val->stats.used = true;
            iterable_val->stats.ops.insert("object_access");
        }
    } else {
        JJ_DEBUG("%s", "For loop over array items");
        auto & arr = iterable_val->as_array();
        for (const auto & item : arr) {
            items.push_back(item);
        }
        if (ctx.is_get_stats) {
            iterable_val->stats.used = true;
            iterable_val->stats.ops.insert("array_access");
        }
    }

    std::vector<std::function<void(context &)>> scope_update_fns;

    std::vector<value> filtered_items;
    for (size_t i = 0; i < items.size(); ++i) {
        context loop_scope(scope);

        value current = items[i];

        std::function<void(context&)> scope_update_fn = [](context &) { /* no-op */};
        if (is_stmt<identifier>(loopvar)) {
            auto id = cast_stmt<identifier>(loopvar)->val;

            if (is_val<value_object>(iterable_val)) {
                // case example: {% for key in dict %}
                current = items[i]->as_array()[0];
                scope_update_fn = [id, &items, i](context & ctx) {
                    ctx.set_val(id, items[i]->as_array()[0]);
                };
            } else {
                // case example: {% for item in list %}
                scope_update_fn = [id, &items, i](context & ctx) {
                    ctx.set_val(id, items[i]);
                };
            }

        } else if (is_stmt<tuple_literal>(loopvar)) {
            // case example: {% for key, value in dict %}
            auto tuple = cast_stmt<tuple_literal>(loopvar);
            if (!is_val<value_array>(current)) {
                throw std::runtime_error("Cannot unpack non-iterable type: " + current->type());
            }
            auto & c_arr = current->as_array();
            if (tuple->val.size() != c_arr.size()) {
                throw std::runtime_error(std::string("Too ") + (tuple->val.size() > c_arr.size() ? "few" : "many") + " items to unpack");
            }
            scope_update_fn = [tuple, &items, i](context & ctx) {
                auto & c_arr = items[i]->as_array();
                for (size_t j = 0; j < tuple->val.size(); ++j) {
                    if (!is_stmt<identifier>(tuple->val[j])) {
                        throw std::runtime_error("Cannot unpack non-identifier type: " + tuple->val[j]->type());
                    }
                    auto id = cast_stmt<identifier>(tuple->val[j])->val;
                    ctx.set_val(id, c_arr[j]);
                }
            };

        } else {
            throw std::runtime_error("Invalid loop variable(s): " + loopvar->type());
        }

        if (select_expr && test_expr) {
            scope_update_fn(loop_scope);
            value test_val = test_expr->execute(loop_scope);
            if (!test_val->as_bool()) {
                continue;
            }
        }
        JJ_DEBUG("For loop: adding item type %s at index %zu", current->type().c_str(), i);
        filtered_items.push_back(current);
        scope_update_fns.push_back(scope_update_fn);
    }
    JJ_DEBUG("For loop: %zu items after filtering", filtered_items.size());

    auto result = mk_val<value_array>();

    bool noIteration = true;
    for (size_t i = 0; i < filtered_items.size(); i++) {
        JJ_DEBUG("For loop iteration %zu/%zu", i + 1, filtered_items.size());
        value_object loop_obj = mk_val<value_object>();
        loop_obj->has_builtins = false; // loop object has no builtins
        loop_obj->insert("index", mk_val<value_int>(i + 1));
        loop_obj->insert("index0", mk_val<value_int>(i));
        loop_obj->insert("revindex", mk_val<value_int>(filtered_items.size() - i));
        loop_obj->insert("revindex0", mk_val<value_int>(filtered_items.size() - i - 1));
        loop_obj->insert("first", mk_val<value_bool>(i == 0));
        loop_obj->insert("last", mk_val<value_bool>(i == filtered_items.size() - 1));
        loop_obj->insert("length", mk_val<value_int>(filtered_items.size()));
        loop_obj->insert("previtem", i > 0 ? filtered_items[i - 1] : mk_val<value_undefined>("previtem"));
        loop_obj->insert("nextitem", i < filtered_items.size() - 1 ? filtered_items[i + 1] : mk_val<value_undefined>("nextitem"));
        scope.set_val("loop", loop_obj);
        scope_update_fns[i](scope);
        try {
            for (auto & stmt : body) {
                value val = stmt->execute(scope);
                result->push_back(val);
            }
        } catch (const continue_statement::signal &) {
            continue;
        } catch (const break_statement::signal &) {
            break;
        }
        noIteration = false;
    }

    JJ_DEBUG("For loop complete, total iterations: %zu", filtered_items.size());
    if (noIteration) {
        for (auto & stmt : default_block) {
            value val = stmt->execute(ctx);
            result->push_back(val);
        }
    }

    // convert to string parts
    value_string str = mk_val<value_string>();
    gather_string_parts_recursive(result, str);
    return str;
}

value set_statement::execute_impl(context & ctx) {
    auto rhs = val ? val->execute(ctx) : exec_statements(body, ctx);

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
        obj_ptr->insert(prop_name, rhs);

    } else {
        throw std::runtime_error("Invalid LHS inside assignment expression: " + assignee->type());
    }
    return mk_val<value_undefined>();
}

value macro_statement::execute_impl(context & ctx) {
    if (!is_stmt<identifier>(this->name)) {
        throw std::runtime_error("Macro name must be an identifier");
    }
    std::string name = cast_stmt<identifier>(this->name)->val;

    const func_handler func = [this, name, &ctx](const func_args & args) -> value {
        size_t expected_count = this->args.size();
        size_t input_count = args.count();

        JJ_DEBUG("Invoking macro '%s' with %zu input arguments (expected %zu)", name.c_str(), input_count, expected_count);
        context macro_ctx(ctx); // new scope for macro execution

        // bind parameters
        for (size_t i = 0; i < expected_count; ++i) {
            if (i < input_count) {
                if (is_stmt<identifier>(this->args[i])) {
                    // normal parameter
                    std::string param_name = cast_stmt<identifier>(this->args[i])->val;
                    JJ_DEBUG("  Binding parameter '%s' to argument of type %s", param_name.c_str(), args.get_pos(i)->type().c_str());
                    macro_ctx.set_val(param_name, args.get_pos(i));
                } else if (is_stmt<keyword_argument_expression>(this->args[i])) {
                    // default argument used as normal parameter
                    auto kwarg = cast_stmt<keyword_argument_expression>(this->args[i]);
                    if (!is_stmt<identifier>(kwarg->key)) {
                        throw std::runtime_error("Keyword argument key must be an identifier in macro '" + name + "'");
                    }
                    std::string param_name = cast_stmt<identifier>(kwarg->key)->val;
                    JJ_DEBUG("  Binding parameter '%s' to argument of type %s", param_name.c_str(), args.get_pos(i)->type().c_str());
                    macro_ctx.set_val(param_name, args.get_pos(i));
                } else {
                    throw std::runtime_error("Invalid parameter type in macro '" + name + "'");
                }
            } else {
                auto & default_arg = this->args[i];
                if (is_stmt<keyword_argument_expression>(default_arg)) {
                    auto kwarg = cast_stmt<keyword_argument_expression>(default_arg);
                    if (!is_stmt<identifier>(kwarg->key)) {
                        throw std::runtime_error("Keyword argument key must be an identifier in macro '" + name + "'");
                    }
                    std::string param_name = cast_stmt<identifier>(kwarg->key)->val;
                    JJ_DEBUG("  Binding parameter '%s' to default argument of type %s", param_name.c_str(), kwarg->val->type().c_str());
                    macro_ctx.set_val(param_name, kwarg->val->execute(ctx));
                } else {
                    throw std::runtime_error("Not enough arguments provided to macro '" + name + "'");
                }
                //std::string param_name = cast_stmt<identifier>(default_args[i])->val;
                //JJ_DEBUG("  Binding parameter '%s' to default", param_name.c_str());
                //macro_ctx.var[param_name] = default_args[i]->execute(ctx);
            }
        }

        // execute macro body
        JJ_DEBUG("Executing macro '%s' body with %zu statements", name.c_str(), this->body.size());
        auto res = exec_statements(this->body, macro_ctx);
        JJ_DEBUG("Macro '%s' execution complete, result: %s", name.c_str(), res->val_str.str().c_str());
        return res;
    };

    JJ_DEBUG("Defining macro '%s' with %zu parameters", name.c_str(), args.size());
    ctx.set_val(name, mk_val<value_func>(name, func));
    return mk_val<value_undefined>();
}

value member_expression::execute_impl(context & ctx) {
    value object = this->object->execute(ctx);

    value property;
    if (this->computed) {
        // syntax: obj[expr]
        JJ_DEBUG("Member expression, computing property type %s", this->property->type().c_str());

        int64_t arr_size = 0;
        if (is_val<value_array>(object)) {
            arr_size = object->as_array().size();
        }

        if (is_stmt<slice_expression>(this->property)) {
            auto s = cast_stmt<slice_expression>(this->property);
            value start_val = s->start_expr ? s->start_expr->execute(ctx) : mk_val<value_int>(0);
            value stop_val  = s->stop_expr  ? s->stop_expr->execute(ctx)  : mk_val<value_int>(arr_size);
            value step_val  = s->step_expr  ? s->step_expr->execute(ctx)  : mk_val<value_int>(1);

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
    } else {
        // syntax: obj.prop
        if (!is_stmt<identifier>(this->property)) {
            throw std::runtime_error("Static member property must be an identifier");
        }
        property = mk_val<value_string>(cast_stmt<identifier>(this->property)->val);
        std::string prop = property->as_string().str();
        JJ_DEBUG("Member expression, object type %s, static property '%s'", object->type().c_str(), prop.c_str());

        // behavior of jinja2: obj having prop as a built-in function AND 'prop', as an object key,
        // then obj.prop returns the built-in function, not the property value.
        // while obj['prop'] returns the property value.
        // example: {"obj": {"items": 123}} -> obj.items is the built-in function, obj['items'] is 123

        value val = try_builtin_func(ctx, prop, object, true);
        if (!is_val<value_undefined>(val)) {
            return val;
        }
        // else, fallthrough to normal property access below
    }

    JJ_DEBUG("Member expression on object type %s, property type %s", object->type().c_str(), property->type().c_str());
    ensure_key_type_allowed(property);

    value val = mk_val<value_undefined>("object_property");

    if (is_val<value_undefined>(object)) {
        JJ_DEBUG("%s", "Accessing property on undefined object, returning undefined");
        return val;

    } else if (is_val<value_object>(object)) {
        auto key = property->as_string().str();
        val = object->at(property, val);
        if (is_val<value_undefined>(val)) {
            val = try_builtin_func(ctx, key, object, true);
        }
        JJ_DEBUG("Accessed property '%s' value, got type: %s", key.c_str(), val->type().c_str());

    } else if (is_val<value_array>(object) || is_val<value_string>(object)) {
        if (is_val<value_int>(property)) {
            int64_t index = property->as_int();
            JJ_DEBUG("Accessing %s index %d", object->type().c_str(), (int)index);
            if (is_val<value_array>(object)) {
                auto & arr = object->as_array();
                if (index < 0) {
                    index += static_cast<int64_t>(arr.size());
                }
                if (index >= 0 && index < static_cast<int64_t>(arr.size())) {
                    val = arr[index];
                }
            } else { // value_string
                auto str = object->as_string().str();
                if (index >= 0 && index < static_cast<int64_t>(str.size())) {
                    val = mk_val<value_string>(std::string(1, str[index]));
                }
            }

        } else if (is_val<value_string>(property)) {
            auto key = property->as_string().str();
            JJ_DEBUG("Accessing %s built-in '%s'", is_val<value_array>(object) ? "array" : "string", key.c_str());
            val = try_builtin_func(ctx, key, object, true);

        } else {
            throw std::runtime_error("Cannot access property with non-string/non-number: got " + property->type());
        }
    } else {
        if (!is_val<value_string>(property)) {
            throw std::runtime_error("Cannot access property with non-string: got " + property->type());
        }
        auto key = property->as_string().str();
        val = try_builtin_func(ctx, key, object, true);
    }

    if (ctx.is_get_stats && val && object && property) {
        val->stats.used = true;
        object->stats.used = true;
        if (is_val<value_int>(property)) {
            object->stats.ops.insert("array_access");
        } else if (is_val<value_string>(property)) {
            object->stats.ops.insert("object_access");
        }
    }

    return val;
}

value call_expression::execute_impl(context & ctx) {
    // gather arguments
    func_args args(ctx);
    for (auto & arg_stmt : this->args) {
        auto arg_val = arg_stmt->execute(ctx);
        JJ_DEBUG("  Argument type: %s", arg_val->type().c_str());
        args.push_back(std::move(arg_val));
    }
    // execute callee
    value callee_val = callee->execute(ctx);
    if (!is_val<value_func>(callee_val)) {
        throw std::runtime_error("Callee is not a function: got " + callee_val->type());
    }
    auto * callee_func = cast_val<value_func>(callee_val);
    JJ_DEBUG("Calling function '%s' with %zu arguments", callee_func->name.c_str(), args.count());
    return callee_func->invoke(args);
}

value keyword_argument_expression::execute_impl(context & ctx) {
    if (!is_stmt<identifier>(key)) {
        throw std::runtime_error("Keyword argument key must be identifiers");
    }

    std::string k = cast_stmt<identifier>(key)->val;
    JJ_DEBUG("Keyword argument expression key: %s, value: %s", k.c_str(), val->type().c_str());

    value v = val->execute(ctx);
    JJ_DEBUG("Keyword argument value executed, type: %s", v->type().c_str());

    return mk_val<value_kwarg>(k, v);
}

} // namespace jinja
