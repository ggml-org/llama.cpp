#include "runtime.h"
#include "value.h"

// for converting from JSON to jinja values
#include <nlohmann/json.hpp>

#include <string>
#include <cctype>
#include <vector>
#include <optional>
#include <algorithm>

#define FILENAME "jinja-value"

namespace jinja {

// func_args method implementations

value func_args::get_kwarg(const std::string & key, value default_val) const {
    for (const auto & arg : args) {
        if (is_val<value_kwarg>(arg)) {
            auto * kwarg = cast_val<value_kwarg>(arg);
            if (kwarg->key == key) {
                return kwarg->val;
            }
        }
    }
    return default_val;
}

value func_args::get_kwarg_or_pos(const std::string & key, size_t pos) const {
    value val = get_kwarg(key, mk_val<value_undefined>());

    if (val->is_undefined() && pos < count() && !is_val<value_kwarg>(args[pos])) {
        return args[pos];
    }

    return val;
}

value func_args::get_pos(size_t pos) const {
    if (count() > pos) {
        return args[pos];
    }
    throw raised_exception("Function '" + func_name + "' expected at least " + std::to_string(pos + 1) + " arguments, got " + std::to_string(count()));
}

value func_args::get_pos(size_t pos, value default_val) const {
    if (count() > pos) {
        return args[pos];
    }
    return default_val;
}

void func_args::push_back(const value & val) {
    args.push_back(val);
}

void func_args::push_front(const value & val) {
    args.insert(args.begin(), val);
}

const std::vector<value> & func_args::get_args() const {
    return args;
}

/**
 * Function that mimics Python's array slicing.
 */
template<typename T>
static T slice(const T & array, int64_t start, int64_t stop, int64_t step = 1) {
    int64_t len = static_cast<int64_t>(array.size());
    int64_t direction = (step > 0) ? 1 : ((step < 0) ? -1 : 0);
    int64_t start_val = 0;
    int64_t stop_val = 0;
    if (direction >= 0) {
        start_val = start;
        if (start_val < 0) {
            start_val = std::max(len + start_val, (int64_t)0);
        } else {
            start_val = std::min(start_val, len);
        }

        stop_val = stop;
        if (stop_val < 0) {
            stop_val = std::max(len + stop_val, (int64_t)0);
        } else {
            stop_val = std::min(stop_val, len);
        }
    } else {
        start_val = len - 1;
        if (start_val < 0) {
            start_val = std::max(len + start_val, (int64_t)-1);
        } else {
            start_val = std::min(start_val, len - 1);
        }

        stop_val = -1;
        if (stop_val < -1) {
            stop_val = std::max(len + stop_val, (int64_t)-1);
        } else {
            stop_val = std::min(stop_val, len - 1);
        }
    }
    T result;
    if (direction == 0) {
        return result;
    }
    for (int64_t i = start_val; direction * i < direction * stop_val; i += step) {
        if (i >= 0 && i < len) {
            result.push_back(array[static_cast<size_t>(i)]);
        }
    }
    return result;
}

template<typename T>
static value test_type_fn(const func_args & args) {
    args.ensure_count(1);
    bool is_type = is_val<T>(args.get_pos(0));
    JJ_DEBUG("test_type_fn: type=%s result=%d", typeid(T).name(), is_type ? 1 : 0);
    return mk_val<value_bool>(is_type);
}
template<typename T, typename U>
static value test_type_fn(const func_args & args) {
    args.ensure_count(1);
    bool is_type = is_val<T>(args.get_pos(0)) || is_val<U>(args.get_pos(0));
    JJ_DEBUG("test_type_fn: type=%s or %s result=%d", typeid(T).name(), typeid(U).name(), is_type ? 1 : 0);
    return mk_val<value_bool>(is_type);
}
template<value_compare_op op>
static value test_compare_fn(const func_args & args) {
    args.ensure_count(2, 2);
    return mk_val<value_bool>(value_compare(args.get_pos(0), args.get_pos(1), op));
}

static value tojson(const func_args & args) {
    args.ensure_count(1, 5);
    value val_ascii      = args.get_kwarg_or_pos("ensure_ascii", 1);
    value val_indent     = args.get_kwarg_or_pos("indent",       2);
    value val_separators = args.get_kwarg_or_pos("separators",   3);
    value val_sort       = args.get_kwarg_or_pos("sort_keys",    4);
    int indent = -1;
    if (is_val<value_int>(val_indent)) {
        indent = static_cast<int>(val_indent->as_int());
    }
    if (val_ascii->as_bool()) { // undefined == false
        throw not_implemented_exception("tojson ensure_ascii=true not implemented");
    }
    if (val_sort->as_bool()) { // undefined == false
        throw not_implemented_exception("tojson sort_keys=true not implemented");
    }
    auto separators = (is_val<value_array>(val_separators) ? val_separators : mk_val<value_array>())->as_array();
    std::string item_sep = separators.size() > 0 ? separators[0]->as_string().str() : (indent < 0 ? ", " : ",");
    std::string key_sep = separators.size() > 1 ? separators[1]->as_string().str() : ": ";
    std::string json_str = value_to_json(args.get_pos(0), indent, item_sep, key_sep);
    return mk_val<value_string>(json_str);
}

template<bool is_reject>
static value selectattr(const func_args & args) {
    args.ensure_count(2, 4);
    args.ensure_vals<value_array, value_string, value_string, value_string>(true, true, false, false);

    auto arr = args.get_pos(0)->as_array();
    auto attr_name = args.get_pos(1)->as_string().str();
    auto out = mk_val<value_array>();
    value val_default = mk_val<value_undefined>();

    if (args.count() == 2) {
        // example: array | selectattr("active")
        for (const auto & item : arr) {
            if (!is_val<value_object>(item)) {
                throw raised_exception("selectattr: item is not an object");
            }
            value attr_val = item->at(attr_name, val_default);
            bool is_selected = attr_val->as_bool();
            if constexpr (is_reject) is_selected = !is_selected;
            if (is_selected) out->push_back(item);
        }
        return out;

    } else if (args.count() == 3) {
        // example: array | selectattr("equalto", "text")
        // translated to: test_is_equalto(item, "text")
        std::string test_name = args.get_pos(1)->as_string().str();
        value test_val = args.get_pos(2);
        auto & builtins = global_builtins();
        auto it = builtins.find("test_is_" + test_name);
        if (it == builtins.end()) {
            throw raised_exception("selectattr: unknown test '" + test_name + "'");
        }
        auto test_fn = it->second;
        for (const auto & item : arr) {
            func_args test_args(args.ctx);
            test_args.push_back(item); // current object
            test_args.push_back(test_val); // extra argument
            value test_result = test_fn(test_args);
            bool is_selected = test_result->as_bool();
            if constexpr (is_reject) is_selected = !is_selected;
            if (is_selected) out->push_back(item);
        }
        return out;

    } else if (args.count() == 4) {
        // example: array | selectattr("status", "equalto", "active")
        // translated to: test_is_equalto(item.status, "active")
        std::string test_name = args.get_pos(2)->as_string().str();
        auto extra_arg = args.get_pos(3);
        auto & builtins = global_builtins();
        auto it = builtins.find("test_is_" + test_name);
        if (it == builtins.end()) {
            throw raised_exception("selectattr: unknown test '" + test_name + "'");
        }
        auto test_fn = it->second;
        for (const auto & item : arr) {
            if (!is_val<value_object>(item)) {
                throw raised_exception("selectattr: item is not an object");
            }
            value attr_val = item->at(attr_name, val_default);
            func_args test_args(args.ctx);
            test_args.push_back(attr_val); // attribute value
            test_args.push_back(extra_arg); // extra argument
            value test_result = test_fn(test_args);
            bool is_selected = test_result->as_bool();
            if constexpr (is_reject) is_selected = !is_selected;
            if (is_selected) out->push_back(item);
        }
        return out;
    } else {
        throw raised_exception("selectattr: invalid number of arguments");
    }

    return out;
}

static value default_value(const func_args & args) {
    args.ensure_count(2, 3);
    value val_check = args.get_kwarg_or_pos("boolean", 2);
    bool check_bool = val_check->as_bool(); // undefined == false
    bool no_value = check_bool
        ? (!args.get_pos(0)->as_bool())
        : (args.get_pos(0)->is_undefined() || args.get_pos(0)->is_none());
    return no_value ? args.get_pos(1) : args.get_pos(0);
}

const func_builtins & global_builtins() {
    static const func_builtins builtins = {
        {"raise_exception", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string msg = args.get_pos(0)->as_string().str();
            throw raised_exception("Jinja Exception: " + msg);
        }},
        {"namespace", [](const func_args & args) -> value {
            auto out = mk_val<value_object>();
            for (const auto & arg : args.get_args()) {
                if (!is_val<value_kwarg>(arg)) {
                    throw raised_exception("namespace() arguments must be kwargs");
                }
                auto kwarg = cast_val<value_kwarg>(arg);
                JJ_DEBUG("namespace: adding key '%s'", kwarg->key.c_str());
                out->insert(kwarg->key, kwarg->val);
            }
            return out;
        }},
        {"strftime_now", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string format = args.get_pos(0)->as_string().str();
            // get current time
            // TODO: make sure this is the same behavior as Python's strftime
            char buf[100];
            if (std::strftime(buf, sizeof(buf), format.c_str(), std::localtime(&args.ctx.current_time))) {
                return mk_val<value_string>(std::string(buf));
            } else {
                throw raised_exception("strftime_now: failed to format time");
            }
        }},
        {"range", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            args.ensure_vals<value_int, value_int, value_int>(true, false, false);

            auto arg0 = args.get_pos(0);
            auto arg1 = args.get_pos(1, mk_val<value_undefined>());
            auto arg2 = args.get_pos(2, mk_val<value_undefined>());

            int64_t start, stop, step;
            if (args.count() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.count() == 2) {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = 1;
            } else {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = arg2->as_int();
            }

            auto out = mk_val<value_array>();
            if (step == 0) {
                throw raised_exception("range() step argument must not be zero");
            }
            if (step > 0) {
                for (int64_t i = start; i < stop; i += step) {
                    out->push_back(mk_val<value_int>(i));
                }
            } else {
                for (int64_t i = start; i > stop; i += step) {
                    out->push_back(mk_val<value_int>(i));
                }
            }
            return out;
        }},
        {"tojson", tojson},

        // tests
        {"test_is_boolean", test_type_fn<value_bool>},
        {"test_is_callable", test_type_fn<value_func>},
        {"test_is_odd", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.get_pos(0)->as_int();
            return mk_val<value_bool>(val % 2 != 0);
        }},
        {"test_is_even", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.get_pos(0)->as_int();
            return mk_val<value_bool>(val % 2 == 0);
        }},
        {"test_is_false", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool val = is_val<value_bool>(args.get_pos(0)) && !args.get_pos(0)->as_bool();
            return mk_val<value_bool>(val);
        }},
        {"test_is_true", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool val = is_val<value_bool>(args.get_pos(0)) && args.get_pos(0)->as_bool();
            return mk_val<value_bool>(val);
        }},
        {"test_is_divisibleby", [](const func_args & args) -> value {
            args.ensure_vals<value_int, value_int>();
            bool res = args.get_pos(0)->val_int % args.get_pos(1)->val_int == 0;
            return mk_val<value_bool>(res);
        }},
        {"test_is_string", test_type_fn<value_string>},
        {"test_is_integer", test_type_fn<value_int>},
        {"test_is_float", test_type_fn<value_float>},
        {"test_is_number", test_type_fn<value_int, value_float>},
        {"test_is_iterable", test_type_fn<value_array, value_string>},
        {"test_is_sequence", test_type_fn<value_array, value_string>},
        {"test_is_mapping", test_type_fn<value_object>},
        {"test_is_lower", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            return mk_val<value_bool>(args.get_pos(0)->val_str.is_lowercase());
        }},
        {"test_is_upper", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            return mk_val<value_bool>(args.get_pos(0)->val_str.is_uppercase());
        }},
        {"test_is_none", test_type_fn<value_none>},
        {"test_is_defined", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool res = !args.get_pos(0)->is_undefined();
            JJ_DEBUG("test_is_defined: result=%d", res ? 1 : 0);
            return mk_val<value_bool>(res);
        }},
        {"test_is_undefined", test_type_fn<value_undefined>},
        {"test_is_eq", test_compare_fn<value_compare_op::eq>},
        {"test_is_equalto", test_compare_fn<value_compare_op::eq>},
        {"test_is_ge", test_compare_fn<value_compare_op::ge>},
        {"test_is_gt", test_compare_fn<value_compare_op::gt>},
        {"test_is_greaterthan", test_compare_fn<value_compare_op::gt>},
        {"test_is_lt", test_compare_fn<value_compare_op::lt>},
        {"test_is_lessthan", test_compare_fn<value_compare_op::lt>},
        {"test_is_ne", test_compare_fn<value_compare_op::ne>},
        {"test_is_test", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            auto & builtins = global_builtins();
            std::string test_name = args.get_pos(0)->val_str.str();
            auto it = builtins.find("test_is_" + test_name);
            bool res = it != builtins.end();
            return mk_val<value_bool>(res);
        }},
        {"test_is_sameas", [](const func_args & args) -> value {
            // Check if an object points to the same memory address as another object
            (void)args;
            throw not_implemented_exception("sameas test not implemented");
        }},
        {"test_is_escaped", [](const func_args & args) -> value {
            (void)args;
            throw not_implemented_exception("escaped test not implemented");
        }},
        {"test_is_filter", [](const func_args & args) -> value {
            (void)args;
            throw not_implemented_exception("filter test not implemented");
        }},
    };
    return builtins;
}


const func_builtins & value_int_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.get_pos(0)->as_int();
            return mk_val<value_int>(val < 0 ? -val : val);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            double val = static_cast<double>(args.get_pos(0)->as_int());
            return mk_val<value_float>(val);
        }},
        {"tojson", tojson},
        {"string", tojson},
    };
    return builtins;
}


const func_builtins & value_float_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            double val = args.get_pos(0)->as_float();
            return mk_val<value_float>(val < 0.0 ? -val : val);
        }},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            int64_t val = static_cast<int64_t>(args.get_pos(0)->as_float());
            return mk_val<value_int>(val);
        }},
        {"tojson", tojson},
        {"string", tojson},
    };
    return builtins;
}

static bool string_startswith(const std::string & str, const std::string & prefix) {
    if (str.length() < prefix.length()) return false;
    return str.compare(0, prefix.length(), prefix) == 0;
}

static bool string_endswith(const std::string & str, const std::string & suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

const func_builtins & value_string_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"upper", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().uppercase();
            return mk_val<value_string>(str);
        }},
        {"lower", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().lowercase();
            return mk_val<value_string>(str);
        }},
        {"strip", [](const func_args & args) -> value {
            value val_input = args.get_pos(0);
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("strip() first argument must be a string");
            }
            value val_chars = args.get_kwarg_or_pos("chars", 1);
            if (val_chars->is_undefined()) {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, true));
            } else {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, true, val_chars->as_string().str()));
            }
        }},
        {"rstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            value val_chars = args.get_kwarg_or_pos("chars", 1);
            if (val_chars->is_undefined()) {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(false, true));
            } else {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(false, true, val_chars->as_string().str()));
            }
        }},
        {"lstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            value val_chars = args.get_kwarg_or_pos("chars", 1);
            if (val_chars->is_undefined()) {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, false));
            } else {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, false, val_chars->as_string().str()));
            }
        }},
        {"title", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().titlecase();
            return mk_val<value_string>(str);
        }},
        {"capitalize", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().capitalize();
            return mk_val<value_string>(str);
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string();
            return mk_val<value_int>(str.length());
        }},
        {"startswith", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            std::string str = args.get_pos(0)->as_string().str();
            std::string prefix = args.get_pos(1)->as_string().str();
            return mk_val<value_bool>(string_startswith(str, prefix));
        }},
        {"endswith", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            std::string str = args.get_pos(0)->as_string().str();
            std::string suffix = args.get_pos(1)->as_string().str();
            return mk_val<value_bool>(string_endswith(str, suffix));
        }},
        {"split", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            value val_input = args.get_pos(0);
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("split() first argument must be a string");
            }
            std::string str = val_input->as_string().str();
            // FIXME: Support non-specified delimiter (split on consecutive (no leading or trailing) whitespace)
            std::string delim = (args.count() > 1) ? args.get_pos(1)->as_string().str() : " ";
            int64_t maxsplit = (args.count() > 2) ? args.get_pos(2)->as_int() : -1;
            auto result = mk_val<value_array>();
            size_t pos = 0;
            std::string token;
            while ((pos = str.find(delim)) != std::string::npos && maxsplit != 0) {
                token = str.substr(0, pos);
                result->push_back(mk_val<value_string>(token));
                str.erase(0, pos + delim.length());
                --maxsplit;
            }
            auto res = mk_val<value_string>(str);
            res->val_str.mark_input_based_on(args.get_pos(0)->val_str);
            result->push_back(std::move(res));
            return result;
        }},
        {"rsplit", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            value val_input = args.get_pos(0);
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("rsplit() first argument must be a string");
            }
            std::string str = val_input->as_string().str();
            // FIXME: Support non-specified delimiter (split on consecutive (no leading or trailing) whitespace)
            std::string delim = (args.count() > 1) ? args.get_pos(1)->as_string().str() : " ";
            int64_t maxsplit = (args.count() > 2) ? args.get_pos(2)->as_int() : -1;
            auto result = mk_val<value_array>();
            size_t pos = 0;
            std::string token;
            while ((pos = str.rfind(delim)) != std::string::npos && maxsplit != 0) {
                token = str.substr(pos + delim.length());
                result->push_back(mk_val<value_string>(token));
                str.erase(pos);
                --maxsplit;
            }
            auto res = mk_val<value_string>(str);
            res->val_str.mark_input_based_on(args.get_pos(0)->val_str);
            result->push_back(std::move(res));
            result->reverse();
            return result;
        }},
        {"replace", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string, value_string, value_int>(true, true, true, false);
            std::string str = args.get_pos(0)->as_string().str();
            std::string old_str = args.get_pos(1)->as_string().str();
            std::string new_str = args.get_pos(2)->as_string().str();
            int64_t count = args.count() > 3 ? args.get_pos(3)->as_int() : -1;
            if (count > 0) {
                throw not_implemented_exception("String replace with count argument not implemented");
            }
            size_t pos = 0;
            while ((pos = str.find(old_str, pos)) != std::string::npos) {
                str.replace(pos, old_str.length(), new_str);
                pos += new_str.length();
            }
            auto res = mk_val<value_string>(str);
            res->val_str.mark_input_based_on(args.get_pos(0)->val_str);
            return res;
        }},
        {"int", [](const func_args & args) -> value {
            value val_input   = args.get_pos(0);
            value val_default = args.get_kwarg_or_pos("default", 1);
            value val_base    = args.get_kwarg_or_pos("base",    2);
            const int base = val_base->is_undefined() ? 10 : val_base->as_int();
            if (is_val<value_string>(val_input) == false) {
                throw raised_exception("int() first argument must be a string");
            }
            std::string str = val_input->as_string().str();
            try {
                return mk_val<value_int>(std::stoi(str, nullptr, base));
            } catch (...) {
                return mk_val<value_int>(val_default->is_undefined() ? 0 : val_default->as_int());
            }
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            value val_default = args.get_kwarg_or_pos("default", 1);
            std::string str = args.get_pos(0)->as_string().str();
            try {
                return mk_val<value_float>(std::stod(str));
            } catch (...) {
                return mk_val<value_float>(val_default->is_undefined() ? 0.0 : val_default->as_float());
            }
        }},
        {"string", [](const func_args & args) -> value {
            // no-op
            args.ensure_vals<value_string>();
            return mk_val<value_string>(args.get_pos(0)->as_string());
        }},
        {"default", [](const func_args & args) -> value {
            value input = args.get_pos(0);
            if (!is_val<value_string>(input)) {
                throw raised_exception("default() first argument must be a string");
            }
            value default_val = mk_val<value_string>("");
            if (args.count() > 1 && !args.get_pos(1)->is_undefined()) {
                default_val = args.get_pos(1);
            }
            value boolean_val = args.get_kwarg_or_pos("boolean", 2); // undefined == false
            if (input->is_undefined() || (boolean_val->as_bool() && !input->as_bool())) {
                return default_val;
            } else {
                return input;
            }
        }},
        {"slice", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            args.ensure_vals<value_string, value_int, value_int, value_int>(true, true, false, false);

            auto arg0 = args.get_pos(1);
            auto arg1 = args.get_pos(2, mk_val<value_undefined>());
            auto arg2 = args.get_pos(3, mk_val<value_undefined>());

            int64_t start, stop, step;
            if (args.count() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.count() == 2) {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = 1;
            } else {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = arg2->as_int();
            }
            if (step == 0) {
                throw raised_exception("slice step cannot be zero");
            }
            auto input = args.get_pos(0);
            auto sliced = slice(input->as_string().str(), start, stop, step);
            auto res = mk_val<value_string>(sliced);
            res->val_str.mark_input_based_on(input->as_string());
            return res;
        }},
        {"safe", [](const func_args & args) -> value {
            // no-op for now
            args.ensure_vals<value_string>();
            return args.get_pos(0);
        }},
        {"tojson", tojson},
        {"indent", [](const func_args &) -> value {
            throw not_implemented_exception("String indent builtin not implemented");
        }},
        {"join", [](const func_args &) -> value {
            throw not_implemented_exception("String join builtin not implemented");
        }},
    };
    return builtins;
}


const func_builtins & value_bool_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.get_pos(0)->as_bool();
            return mk_val<value_int>(val ? 1 : 0);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.get_pos(0)->as_bool();
            return mk_val<value_float>(val ? 1.0 : 0.0);
        }},
        {"string", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.get_pos(0)->as_bool();
            return mk_val<value_string>(val ? "True" : "False");
        }},
        {"tojson", tojson},
    };
    return builtins;
}


const func_builtins & value_array_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"list", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            auto result = mk_val<value_array>();
            for (const auto& v : arr) {
                result->push_back(v);
            }
            return result;
        }},
        {"first", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            return arr[0];
        }},
        {"last", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            return arr[arr.size() - 1];
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            return mk_val<value_int>(static_cast<int64_t>(arr.size()));
        }},
        {"slice", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            args.ensure_vals<value_array, value_int, value_int, value_int>(true, true, false, false);

            auto arg0 = args.get_pos(1);
            auto arg1 = args.get_pos(2, mk_val<value_undefined>());
            auto arg2 = args.get_pos(3, mk_val<value_undefined>());

            int64_t start, stop, step;
            if (args.count() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.count() == 2) {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = 1;
            } else {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = arg2->as_int();
            }
            if (step == 0) {
                throw raised_exception("slice step cannot be zero");
            }
            auto arr = slice(args.get_pos(0)->as_array(), start, stop, step);
            auto res = mk_val<value_array>();
            res->val_arr = std::move(arr);
            return res;
        }},
        {"selectattr", selectattr<false>},
        {"select", selectattr<false>},
        {"rejectattr", selectattr<true>},
        {"reject", selectattr<true>},
        {"join", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("join() first argument must be an array");
            }
            value val_delim = args.get_kwarg_or_pos("d",         1);
            value attribute = args.get_kwarg_or_pos("attribute", 2);
            const auto & arr = args.get_pos(0)->as_array();
            const bool attr_is_int = is_val<value_int>(attribute);
            if (!attribute->is_undefined() && !is_val<value_string>(attribute) && !attr_is_int) {
                throw raised_exception("join() attribute must be string or integer");
            }
            const int64_t attr_int = attr_is_int ? attribute->as_int() : 0;
            const std::string delim = val_delim->is_undefined() ? "" : val_delim->as_string().str();
            const std::string attr_name = attribute->is_undefined() ? "" : attribute->as_string().str();
            std::string result;
            for (size_t i = 0; i < arr.size(); ++i) {
                value val_arr = arr[i];
                if (!attribute->is_undefined()) {
                    if (attr_is_int && is_val<value_array>(val_arr)) {
                        val_arr = val_arr->at(attr_int);
                    } else if (!attr_is_int && !attr_name.empty() && is_val<value_object>(val_arr)) {
                        val_arr = val_arr->at(attr_name);
                    }
                }
                if (!is_val<value_string>(val_arr) && !is_val<value_int>(val_arr) && !is_val<value_float>(val_arr)) {
                    throw raised_exception("join() can only join arrays of strings or numerics");
                }
                result += val_arr->as_string().str();
                if (i < arr.size() - 1) {
                    result += delim;
                }
            }
            return mk_val<value_string>(result);
        }},
        {"string", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            auto str = mk_val<value_string>();
            gather_string_parts_recursive(args.get_pos(0), str);
            return str;
        }},
        {"tojson", tojson},
        {"map", [](const func_args & args) -> value {
            args.ensure_count(2);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("map: first argument must be an array");
            }
            if (!is_val<value_kwarg>(args.get_args().at(1))) {
                throw not_implemented_exception("map: filter-mapping not implemented");
            }
            value attribute = args.get_kwarg_or_pos("attribute", 1);
            const bool attr_is_int = is_val<value_int>(attribute);
            if (!is_val<value_string>(attribute) && !attr_is_int) {
                throw raised_exception("map: attribute must be string or integer");
            }
            const int64_t attr_int = attr_is_int ? attribute->as_int() : 0;
            const std::string attr_name = attribute->as_string().str();
            value default_val = args.get_kwarg("default", mk_val<value_undefined>());
            auto out = mk_val<value_array>();
            auto arr = args.get_pos(0)->as_array();
            for (const auto & item : arr) {
                value attr_val;
                if (attr_is_int) {
                    attr_val = is_val<value_array>(item) ? item->at(attr_int, default_val) : default_val;
                } else {
                    attr_val = is_val<value_object>(item) ? item->at(attr_name, default_val) : default_val;
                }
                out->push_back(attr_val);
            }
            return out;
        }},
        {"append", [](const func_args & args) -> value {
            args.ensure_count(2);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("append: first argument must be an array");
            }
            const value_array_t * arr = cast_val<value_array>(args.get_pos(0));
            // need to use const_cast here to modify the array
            value_array_t * arr_editable = const_cast<value_array_t *>(arr);
            arr_editable->push_back(args.get_pos(1));
            return args.get_pos(0);
        }},
        {"pop", [](const func_args & args) -> value {
            args.ensure_count(1, 2);
            args.ensure_vals<value_array, value_int>(true, false);
            int64_t index = args.count() == 2 ? args.get_pos(1)->as_int() : -1;
            const value_array_t * arr = cast_val<value_array>(args.get_pos(0));
            // need to use const_cast here to modify the array
            value_array_t * arr_editable = const_cast<value_array_t *>(arr);
            return arr_editable->pop_at(index);
        }},
        {"sort", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("sort: first argument must be an array");
            }
            value val_reverse = args.get_kwarg_or_pos("reverse",        1);
            value val_case    = args.get_kwarg_or_pos("case_sensitive", 2);
            value attribute   = args.get_kwarg_or_pos("attribute",      3);
            // FIXME: sorting is currently always case sensitive
            //const bool case_sensitive = val_case->as_bool(); // undefined == false
            const bool reverse = val_reverse->as_bool(); // undefined == false
            const bool attr_is_int = is_val<value_int>(attribute);
            const int64_t attr_int = attr_is_int ? attribute->as_int() : 0;
            const std::string attr_name = attribute->is_undefined() ? "" : attribute->as_string().str();
            std::vector<value> arr = cast_val<value_array>(args.get_pos(0))->as_array(); // copy
            std::sort(arr.begin(), arr.end(),[&](const value & a, const value & b) {
                value val_a = a;
                value val_b = b;
                if (!attribute->is_undefined()) {
                    if (attr_is_int && is_val<value_array>(a) && is_val<value_array>(b)) {
                        val_a = a->at(attr_int);
                        val_b = b->at(attr_int);
                    } else if (!attr_is_int && !attr_name.empty() && is_val<value_object>(a) && is_val<value_object>(b)) {
                        val_a = a->at(attr_name);
                        val_b = b->at(attr_name);
                    } else {
                        throw raised_exception("sort: unsupported object attribute comparison");
                    }
                }
                return value_compare(val_a, val_b, reverse ? value_compare_op::gt : value_compare_op::lt);
            });
            return mk_val<value_array>(arr);
        }},
        {"reverse", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            std::vector<value> arr = cast_val<value_array>(args.get_pos(0))->as_array(); // copy
            std::reverse(arr.begin(), arr.end());
            return mk_val<value_array>(arr);
        }},
        {"unique", [](const func_args &) -> value {
            throw not_implemented_exception("Array unique builtin not implemented");
        }},
    };
    return builtins;
}


const func_builtins & value_object_t::get_builtins() const {
    if (!has_builtins) {
        static const func_builtins no_builtins = {};
        return no_builtins;
    }

    static const func_builtins builtins = {
        // {"default", default_value}, // cause issue with gpt-oss
        {"get", [](const func_args & args) -> value {
            args.ensure_count(2, 3);
            if (!is_val<value_object>(args.get_pos(0))) {
                throw raised_exception("get: first argument must be an object");
            }
            if (!is_val<value_string>(args.get_pos(1))) {
                throw raised_exception("get: second argument must be a string (key)");
            }
            value default_val = mk_val<value_none>();
            if (args.count() == 3) {
                default_val = args.get_pos(2);
            }
            const value obj = args.get_pos(0);
            std::string key = args.get_pos(1)->as_string().str();
            return obj->at(key, default_val);
        }},
        {"keys", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.get_pos(0)->as_ordered_object();
            auto result = mk_val<value_array>();
            for (const auto & pair : obj) {
                result->push_back(mk_val<value_string>(pair.first));
            }
            return result;
        }},
        {"values", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.get_pos(0)->as_ordered_object();
            auto result = mk_val<value_array>();
            for (const auto & pair : obj) {
                result->push_back(pair.second);
            }
            return result;
        }},
        {"items", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.get_pos(0)->as_ordered_object();
            auto result = mk_val<value_array>();
            for (const auto & pair : obj) {
                auto item = mk_val<value_array>();
                item->push_back(mk_val<value_string>(pair.first));
                item->push_back(pair.second);
                result->push_back(std::move(item));
            }
            return result;
        }},
        {"tojson", tojson},
        {"string", tojson},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.get_pos(0)->as_ordered_object();
            return mk_val<value_int>(static_cast<int64_t>(obj.size()));
        }},
        {"tojson", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            // use global to_json
            return global_builtins().at("tojson")(args);
        }},
        {"dictsort", [](const func_args & args) -> value {
            value val_input   = args.get_pos(0);
            value val_case    = args.get_kwarg_or_pos("case_sensitive", 1);
            value val_by      = args.get_kwarg_or_pos("by",             2);
            value val_reverse = args.get_kwarg_or_pos("reverse",        3);
            // FIXME: sorting is currently always case sensitive
            //const bool case_sensitive = val_case->as_bool(); // undefined == false
            const bool reverse = val_reverse->as_bool(); // undefined == false
            const bool by_value = is_val<value_string>(val_by) && val_by->as_string().str() == "value" ? true : false;
            auto result = mk_val<value_object>(val_input); // copy
            std::sort(result->val_obj.ordered.begin(), result->val_obj.ordered.end(), [&](const auto & a, const auto & b) {
                if (by_value) {
                    return value_compare(a.second, b.second, reverse ? value_compare_op::gt : value_compare_op::lt);
                } else {
                    return reverse ? a.first > b.first : a.first < b.first;
                }
            });
            return result;
        }},
        {"join", [](const func_args &) -> value {
            throw not_implemented_exception("object join not implemented");
        }},
    };
    return builtins;
}

const func_builtins & value_none_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"tojson", tojson},
        {"string", [](const func_args &) -> value { return mk_val<value_string>("None"); }}
    };
    return builtins;
}


const func_builtins & value_undefined_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"tojson", [](const func_args & args) -> value {
            args.ensure_vals<value_undefined>();
            return mk_val<value_string>("null");
        }},
    };
    return builtins;
}


//////////////////////////////////


static value from_json(const nlohmann::ordered_json & j, bool mark_input) {
    if (j.is_null()) {
        return mk_val<value_none>();
    } else if (j.is_boolean()) {
        return mk_val<value_bool>(j.get<bool>());
    } else if (j.is_number_integer()) {
        return mk_val<value_int>(j.get<int64_t>());
    } else if (j.is_number_float()) {
        return mk_val<value_float>(j.get<double>());
    } else if (j.is_string()) {
        auto str = mk_val<value_string>(j.get<std::string>());
        if (mark_input) {
            str->mark_input();
        }
        return str;
    } else if (j.is_array()) {
        auto arr = mk_val<value_array>();
        for (const auto & item : j) {
            arr->push_back(from_json(item, mark_input));
        }
        return arr;
    } else if (j.is_object()) {
        auto obj = mk_val<value_object>();
        for (auto it = j.begin(); it != j.end(); ++it) {
            obj->insert(it.key(), from_json(it.value(), mark_input));
        }
        return obj;
    } else {
        throw std::runtime_error("Unsupported JSON value type");
    }
}

// compare operator for value_t
bool value_compare(const value & a, const value & b, value_compare_op op) {
    auto cmp = [&]() {
        // compare numeric types
        if ((is_val<value_int>(a) || is_val<value_float>(a)) &&
            (is_val<value_int>(b) || is_val<value_float>(b))){
            try {
                if (op == value_compare_op::eq) {
                    return a->as_float() == b->as_float();
                } else if (op == value_compare_op::ge) {
                    return a->as_float() >= b->as_float();
                } else if (op == value_compare_op::gt) {
                    return a->as_float() > b->as_float();
                } else if (op == value_compare_op::lt) {
                    return a->as_float() < b->as_float();
                } else if (op == value_compare_op::ne) {
                    return a->as_float() != b->as_float();
                } else {
                    throw std::runtime_error("Unsupported comparison operator for numeric types");
                }
            } catch (...) {}
        }
        // compare string and number
        // TODO: not sure if this is the right behavior
        if ((is_val<value_string>(b) && (is_val<value_int>(a) || is_val<value_float>(a))) ||
            (is_val<value_string>(a) && (is_val<value_int>(b) || is_val<value_float>(b))) ||
            (is_val<value_string>(a) && is_val<value_string>(b))) {
            try {
                if (op == value_compare_op::eq) {
                    return a->as_string().str() == b->as_string().str();
                } else if (op == value_compare_op::ge) {
                    return a->as_string().str() >= b->as_string().str();
                } else if (op == value_compare_op::gt) {
                    return a->as_string().str() > b->as_string().str();
                } else if (op == value_compare_op::lt) {
                    return a->as_string().str() < b->as_string().str();
                } else if (op == value_compare_op::ne) {
                    return a->as_string().str() != b->as_string().str();
                } else {
                    throw std::runtime_error("Unsupported comparison operator for string/number types");
                }
            } catch (...) {}
        }
        // compare boolean simple
        if (is_val<value_bool>(a) && is_val<value_bool>(b)) {
            if (op == value_compare_op::eq) {
                return a->as_bool() == b->as_bool();
            } else if (op == value_compare_op::ne) {
                return a->as_bool() != b->as_bool();
            } else {
                throw std::runtime_error("Unsupported comparison operator for bool type");
            }
        }
        // compare by type
        if (a->type() != b->type()) {
            return false;
        }
        return false;
    };
    auto result = cmp();
    JJ_DEBUG("Comparing types: %s and %s result=%d", a->type().c_str(), b->type().c_str(), result);
    return result;
}

template<>
void global_from_json(context & ctx, const nlohmann::ordered_json & json_obj, bool mark_input) {
    // printf("global_from_json: %s\n" , json_obj.dump(2).c_str());
    if (json_obj.is_null() || !json_obj.is_object()) {
        throw std::runtime_error("global_from_json: input JSON value must be an object");
    }
    for (auto it = json_obj.begin(); it != json_obj.end(); ++it) {
        JJ_DEBUG("global_from_json: setting key '%s'", it.key().c_str());
        ctx.set_val(it.key(), from_json(it.value(), mark_input));
    }
}

static void value_to_json_internal(std::ostringstream & oss, const value & val, int curr_lvl, int indent, const std::string_view item_sep, const std::string_view key_sep) {
    auto indent_str = [indent, curr_lvl]() -> std::string {
        return (indent > 0) ? std::string(curr_lvl * indent, ' ') : "";
    };
    auto newline = [indent]() -> std::string {
        return (indent >= 0) ? "\n" : "";
    };

    if (is_val<value_none>(val) || val->is_undefined()) {
        oss << "null";
    } else if (is_val<value_bool>(val)) {
        oss << (val->as_bool() ? "true" : "false");
    } else if (is_val<value_int>(val)) {
        oss << val->as_int();
    } else if (is_val<value_float>(val)) {
        oss << val->as_float();
    } else if (is_val<value_string>(val)) {
        oss << "\"";
        for (char c : val->as_string().str()) {
            switch (c) {
                case '"': oss << "\\\""; break;
                case '\\': oss << "\\\\"; break;
                case '\b': oss << "\\b"; break;
                case '\f': oss << "\\f"; break;
                case '\n': oss << "\\n"; break;
                case '\r': oss << "\\r"; break;
                case '\t': oss << "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[7];
                        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                        oss << buf;
                    } else {
                        oss << c;
                    }
            }
        }
        oss << "\"";
    } else if (is_val<value_array>(val)) {
        const auto & arr = val->as_array();
        oss << "[";
        if (!arr.empty()) {
            oss << newline();
            for (size_t i = 0; i < arr.size(); ++i) {
                oss << indent_str() << (indent > 0 ? std::string(indent, ' ') : "");
                value_to_json_internal(oss, arr[i], curr_lvl + 1, indent, item_sep, key_sep);
                if (i < arr.size() - 1) {
                    oss << item_sep;
                }
                oss << newline();
            }
            oss << indent_str();
        }
        oss << "]";
    } else if (is_val<value_object>(val)) {
        const auto & obj = val->as_ordered_object(); // IMPORTANT: need to keep exact order
        oss << "{";
        if (!obj.empty()) {
            oss << newline();
            size_t i = 0;
            for (const auto & pair : obj) {
                oss << indent_str() << (indent > 0 ? std::string(indent, ' ') : "");
                oss << "\"" << pair.first << "\"" << key_sep;
                value_to_json_internal(oss, pair.second, curr_lvl + 1, indent, item_sep, key_sep);
                if (i < obj.size() - 1) {
                    oss << item_sep;
                }
                oss << newline();
                ++i;
            }
            oss << indent_str();
        }
        oss << "}";
    } else {
        oss << "null";
    }
}

std::string value_to_json(const value & val, int indent, const std::string_view item_sep, const std::string_view key_sep) {
    std::ostringstream oss;
    value_to_json_internal(oss, val, 0, indent, item_sep, key_sep);
    JJ_DEBUG("value_to_json: result=%s", oss.str().c_str());
    return oss.str();
}

} // namespace jinja
