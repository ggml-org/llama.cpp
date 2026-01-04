#include "jinja-lexer.h"
#include "jinja-interpreter.h"
#include "jinja-parser.h"
#include "jinja-value.h"

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

value func_args::get_kwarg(const std::string & key) const  {
    for (const auto & arg : args) {
        if (is_val<value_kwarg>(arg)) {
            auto * kwarg = cast_val<value_kwarg>(arg);
            if (kwarg->key == key) {
                return kwarg->val;
            }
        }
    }
    return mk_val<value_undefined>();
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
    bool is_type = is_val<T>(args.args[0]);
    JJ_DEBUG("test_type_fn: type=%s result=%d", typeid(T).name(), is_type ? 1 : 0);
    return mk_val<value_bool>(is_type);
}
template<typename T, typename U>
static value test_type_fn(const func_args & args) {
    args.ensure_count(1);
    bool is_type = is_val<T>(args.args[0]) || is_val<U>(args.args[0]);
    JJ_DEBUG("test_type_fn: type=%s or %s result=%d", typeid(T).name(), typeid(U).name(), is_type ? 1 : 0);
    return mk_val<value_bool>(is_type);
}

static value tojson(const func_args & args) {
    args.ensure_count(1, 2);
    int indent = 0;
    if (args.args.size() == 2 && is_val<value_int>(args.args[1])) {
        indent = static_cast<int>(args.args[1]->as_int());
    }
    std::string json_str = value_to_json(args.args[0], indent);
    return mk_val<value_string>(json_str);
}

template<bool is_reject>
static value selectattr(const func_args & args) {
    args.ensure_count(2, 4);
    args.ensure_vals<value_array, value_string, value_string, value_string>(true, true, false, false);

    auto arr = args.args[0]->as_array();
    auto attr_name = args.args[1]->as_string().str();
    auto out = mk_val<value_array>();
    value val_default = mk_val<value_undefined>();

    if (args.args.size() == 2) {
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

    } else if (args.args.size() == 3) {
        // example: array | selectattr("equalto", "text")
        // translated to: test_is_equalto(item, "text")
        std::string test_name = args.args[1]->as_string().str();
        value test_val = args.args[2];
        auto & builtins = global_builtins();
        auto it = builtins.find("test_is_" + test_name);
        if (it == builtins.end()) {
            throw raised_exception("selectattr: unknown test '" + test_name + "'");
        }
        auto test_fn = it->second;
        for (const auto & item : arr) {
            func_args test_args(args.ctx);
            test_args.args.push_back(item); // current object
            test_args.args.push_back(test_val); // extra argument
            value test_result = test_fn(test_args);
            bool is_selected = test_result->as_bool();
            if constexpr (is_reject) is_selected = !is_selected;
            if (is_selected) out->push_back(item);
        }
        return out;

    } else if (args.args.size() == 4) {
        // example: array | selectattr("status", "equalto", "active")
        // translated to: test_is_equalto(item.status, "active")
        std::string test_name = args.args[2]->as_string().str();
        func_args test_args(args.ctx);
        test_args.args.push_back(val_default); // placeholder for current object
        test_args.args.push_back(args.args[3]); // extra argument
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
            test_args.args[0] = attr_val;
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
    bool check_bool = false;
    if (args.args.size() == 3) {
        check_bool = args.args[2]->as_bool();
    }
    bool no_value = check_bool
        ? (!args.args[0]->as_bool())
        : (args.args[0]->is_undefined() || args.args[0]->is_null());
    return no_value ? args.args[1] : args.args[0];
}

const func_builtins & global_builtins() {
    static const func_builtins builtins = {
        {"raise_exception", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string msg = args.args[0]->as_string().str();
            throw raised_exception("Jinja Exception: " + msg);
        }},
        {"namespace", [](const func_args & args) -> value {
            auto out = mk_val<value_object>();
            for (const auto & arg : args.args) {
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
            std::string format = args.args[0]->as_string().str();
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

            auto & arg0 = args.args[0];
            auto & arg1 = args.args[1];
            auto & arg2 = args.args[2];

            int64_t start, stop, step;
            if (args.args.size() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.args.size() == 2) {
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
            int64_t val = args.args[0]->as_int();
            return mk_val<value_bool>(val % 2 != 0);
        }},
        {"test_is_even", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.args[0]->as_int();
            return mk_val<value_bool>(val % 2 == 0);
        }},
        {"test_is_false", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool val = is_val<value_bool>(args.args[0]) && !args.args[0]->as_bool();
            return mk_val<value_bool>(val);
        }},
        {"test_is_true", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool val = is_val<value_bool>(args.args[0]) && args.args[0]->as_bool();
            return mk_val<value_bool>(val);
        }},
        {"test_is_string", test_type_fn<value_string>},
        {"test_is_integer", test_type_fn<value_int>},
        {"test_is_number", test_type_fn<value_int, value_float>},
        {"test_is_iterable", test_type_fn<value_array, value_string>},
        {"test_is_sequence", test_type_fn<value_array, value_string>},
        {"test_is_mapping", test_type_fn<value_object>},
        {"test_is_lower", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            return mk_val<value_bool>(args.args[0]->val_str.is_lowercase());
        }},
        {"test_is_upper", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            return mk_val<value_bool>(args.args[0]->val_str.is_uppercase());
        }},
        {"test_is_none", test_type_fn<value_null>},
        {"test_is_defined", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool res = !args.args[0]->is_undefined();
            JJ_DEBUG("test_is_defined: result=%d", res ? 1 : 0);
            return mk_val<value_bool>(res);
        }},
        {"test_is_undefined", test_type_fn<value_undefined>},
        {"test_is_equalto", [](const func_args & args) -> value {
            // alias for is_eq
            args.ensure_count(2);
            return mk_val<value_bool>(value_compare(args.args[0], args.args[1]));
        }},
    };
    return builtins;
}


const func_builtins & value_int_t::get_builtins() const {
    static const func_builtins builtins = {
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.args[0]->as_int();
            return mk_val<value_int>(val < 0 ? -val : val);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            double val = static_cast<double>(args.args[0]->as_int());
            return mk_val<value_float>(val);
        }},
    };
    return builtins;
}


const func_builtins & value_float_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            double val = args.args[0]->as_float();
            return mk_val<value_float>(val < 0.0 ? -val : val);
        }},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            int64_t val = static_cast<int64_t>(args.args[0]->as_float());
            return mk_val<value_int>(val);
        }},
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
            jinja::string str = args.args[0]->as_string().uppercase();
            return mk_val<value_string>(str);
        }},
        {"lower", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().lowercase();
            return mk_val<value_string>(str);
        }},
        {"strip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().strip(true, true);
            return mk_val<value_string>(str);
        }},
        {"rstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().strip(false, true);
            return mk_val<value_string>(str);
        }},
        {"lstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().strip(true, false);
            return mk_val<value_string>(str);
        }},
        {"title", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().titlecase();
            return mk_val<value_string>(str);
        }},
        {"capitalize", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string().capitalize();
            return mk_val<value_string>(str);
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.args[0]->as_string();
            return mk_val<value_int>(str.length());
        }},
        {"startswith", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            std::string str = args.args[0]->as_string().str();
            std::string prefix = args.args[1]->as_string().str();
            return mk_val<value_bool>(string_startswith(str, prefix));
        }},
        {"endswith", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            std::string str = args.args[0]->as_string().str();
            std::string suffix = args.args[1]->as_string().str();
            return mk_val<value_bool>(string_endswith(str, suffix));
        }},
        {"split", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string().str();
            std::string delim = (args.args.size() > 1) ? args.args[1]->as_string().str() : " ";
            auto result = mk_val<value_array>();
            size_t pos = 0;
            std::string token;
            while ((pos = str.find(delim)) != std::string::npos) {
                token = str.substr(0, pos);
                result->push_back(mk_val<value_string>(token));
                str.erase(0, pos + delim.length());
            }
            auto res = mk_val<value_string>(str);
            res->val_str.mark_input_based_on(args.args[0]->val_str);
            result->push_back(std::move(res));
            return result;
        }},
        {"replace", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string, value_string>();
            std::string str = args.args[0]->as_string().str();
            std::string old_str = args.args[1]->as_string().str();
            std::string new_str = args.args[2]->as_string().str();
            size_t pos = 0;
            while ((pos = str.find(old_str, pos)) != std::string::npos) {
                str.replace(pos, old_str.length(), new_str);
                pos += new_str.length();
            }
            auto res = mk_val<value_string>(str);
            res->val_str.mark_input_based_on(args.args[0]->val_str);
            return res;
        }},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string().str();
            try {
                return mk_val<value_int>(std::stoi(str));
            } catch (...) {
                throw std::runtime_error("Cannot convert string '" + str + "' to int");
            }
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string str = args.args[0]->as_string().str();
            try {
                return mk_val<value_float>(std::stod(str));
            } catch (...) {
                throw std::runtime_error("Cannot convert string '" + str + "' to float");
            }
        }},
        {"string", [](const func_args & args) -> value {
            // no-op
            args.ensure_vals<value_string>();
            return mk_val<value_string>(args.args[0]->as_string());
        }},
        {"default", [](const func_args & args) -> value {
            value input = args.args[0];
            if (!is_val<value_string>(input)) {
                throw raised_exception("default() first argument must be a string");
            }
            value default_val = mk_val<value_string>("");
            if (args.args.size() > 1 && !args.args[1]->is_undefined()) {
                default_val = args.args[1];
            }
            value boolean_val = mk_val<value_bool>(false);
            if (args.args.size() > 1) {
                boolean_val = args.args[1];
            }
            if (input->is_undefined() || (boolean_val->as_bool() && !input->as_bool())) {
                return default_val;
            } else {
                return input;
            }
        }},
        {"slice", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            args.ensure_vals<value_string, value_int, value_int, value_int>(true, true, false, false);

            auto & arg0 = args.args[1];
            auto & arg1 = args.args[2];
            auto & arg2 = args.args[3];

            int64_t start, stop, step;
            if (args.args.size() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.args.size() == 2) {
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
            auto & input = args.args[0];
            auto sliced = slice(input->as_string().str(), start, stop, step);
            auto res = mk_val<value_string>(sliced);
            res->val_str.mark_input_based_on(input->as_string());
            return res;
        }},
        {"safe", [](const func_args & args) -> value {
            // no-op for now
            args.ensure_vals<value_string>();
            return args.args[0];
        }},
        {"tojson", tojson},
        {"indent", [](const func_args &) -> value {
            throw std::runtime_error("String indent builtin not implemented");
        }},
        {"join", [](const func_args &) -> value {
            throw std::runtime_error("String join builtin not implemented");
        }},
    };
    return builtins;
}


const func_builtins & value_bool_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.args[0]->as_bool();
            return mk_val<value_int>(val ? 1 : 0);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.args[0]->as_bool();
            return mk_val<value_float>(val ? 1.0 : 0.0);
        }},
        {"string", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.args[0]->as_bool();
            return mk_val<value_string>(val ? "True" : "False");
        }},
    };
    return builtins;
}


const func_builtins & value_array_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"list", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            auto result = mk_val<value_array>();
            for (const auto& v : arr) {
                result->push_back(v);
            }
            return result;
        }},
        {"first", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            return arr[0];
        }},
        {"last", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            return arr[arr.size() - 1];
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.args[0]->as_array();
            return mk_val<value_int>(static_cast<int64_t>(arr.size()));
        }},
        {"slice", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            args.ensure_vals<value_array, value_int, value_int, value_int>(true, true, false, false);

            auto & arg0 = args.args[1];
            auto & arg1 = args.args[2];
            auto & arg2 = args.args[3];

            int64_t start, stop, step;
            if (args.args.size() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.args.size() == 2) {
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
            auto arr = slice(args.args[0]->as_array(), start, stop, step);
            auto res = mk_val<value_array>();
            res->val_arr = std::move(arr);
            return res;
        }},
        {"selectattr", selectattr<false>},
        {"select", selectattr<false>},
        {"rejectattr", selectattr<true>},
        {"reject", selectattr<true>},
        {"join", [](const func_args & args) -> value {
            if (args.args.size() < 1 || args.args.size() > 2) {
                throw raised_exception("join() takes one or two arguments");
            }
            if (!is_val<value_array>(args.args[0])) {
                throw raised_exception("join() first argument must be an array");
            }
            const auto & arr = args.args[0]->as_array();
            std::string delim = (args.args.size() > 1 && is_val<value_string>(args.args[1])) ? args.args[1]->as_string().str() : "";
            std::string result;
            for (size_t i = 0; i < arr.size(); ++i) {
                if (!is_val<value_string>(arr[i])) {
                    throw raised_exception("join() can only join arrays of strings");
                }
                result += arr[i]->as_string().str();
                if (i < arr.size() - 1) {
                    result += delim;
                }
            }
            return mk_val<value_string>(result);
        }},
        {"string", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            auto str = mk_val<value_string>();
            gather_string_parts_recursive(args.args[0], str);
            return str;
        }},
        {"tojson", tojson},
        {"map", [](const func_args & args) -> value {
            args.ensure_count(2, 3);
            if (!is_val<value_array>(args.args[0])) {
                throw raised_exception("map: first argument must be an array");
            }
            std::string attribute = args.get_kwarg("attribute")->as_string().str();
            value default_val = args.get_kwarg("default");
            auto out = mk_val<value_array>();
            auto arr = args.args[0]->as_array();
            for (const auto & item : arr) {
                if (!is_val<value_object>(item)) {
                    throw raised_exception("map: item is not an object");
                }
                value attr_val = item->at(attribute, default_val);
                out->push_back(attr_val);
            }
            return out;
        }},
        {"sort", [](const func_args &) -> value {
            throw std::runtime_error("Array sort builtin not implemented");
        }},
        {"reverse", [](const func_args &) -> value {
            throw std::runtime_error("Array reverse builtin not implemented");
        }},
        {"unique", [](const func_args &) -> value {
            throw std::runtime_error("Array unique builtin not implemented");
        }},
    };
    return builtins;
}


const func_builtins & value_object_t::get_builtins() const {
    static const func_builtins builtins = {
        // {"default", default_value}, // cause issue with gpt-oss
        {"get", [](const func_args & args) -> value {
            args.ensure_vals<value_object, value_string>(); // TODO: add default value
            const auto & obj = args.args[0]->as_object();
            std::string key = args.args[1]->as_string().str();
            auto it = obj.find(key);
            if (it != obj.end()) {
                return it->second;
            } else {
                return mk_val<value_undefined>();
            }
        }},
        {"keys", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.args[0]->as_object();
            auto result = mk_val<value_array>();
            for (const auto & pair : obj) {
                result->push_back(mk_val<value_string>(pair.first));
            }
            return result;
        }},
        {"values", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.args[0]->as_object();
            auto result = mk_val<value_array>();
            for (const auto & pair : obj) {
                result->push_back(pair.second);
            }
            return result;
        }},
        {"items", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.args[0]->as_object();
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
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.args[0]->as_object();
            return mk_val<value_int>(static_cast<int64_t>(obj.size()));
        }},
        {"string", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            return mk_val<value_string>("TO BE IMPLEMENTED");
        }},
        {"tojson", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            // use global to_json
            return global_builtins().at("tojson")(args);
        }},
        {"dictsort", [](const func_args & args) -> value {
            // no-op
            args.ensure_vals<value_object>();
            return args.args[0];
        }},
    };
    return builtins;
}

const func_builtins & value_null_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"tojson", tojson},
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


static value from_json(const nlohmann::ordered_json & j) {
    if (j.is_null()) {
        return mk_val<value_null>();
    } else if (j.is_boolean()) {
        return mk_val<value_bool>(j.get<bool>());
    } else if (j.is_number_integer()) {
        return mk_val<value_int>(j.get<int64_t>());
    } else if (j.is_number_float()) {
        return mk_val<value_float>(j.get<double>());
    } else if (j.is_string()) {
        return mk_val<value_string>(j.get<std::string>());
    } else if (j.is_array()) {
        auto arr = mk_val<value_array>();
        for (const auto & item : j) {
            arr->push_back(from_json(item));
        }
        return arr;
    } else if (j.is_object()) {
        if (j.contains("__input__")) {
            // handle input marking
            auto str = mk_val<value_string>(j.at("__input__").get<std::string>());
            str->mark_input();
            return str;
        } else {
            // normal object
            auto obj = mk_val<value_object>();
            for (auto it = j.begin(); it != j.end(); ++it) {
                obj->insert(it.key(), from_json(it.value()));
            }
            return obj;
        }
    } else {
        throw std::runtime_error("Unsupported JSON value type");
    }
}

template<>
void global_from_json(context & ctx, const nlohmann::ordered_json & json_obj) {
    // printf("global_from_json: %s\n" , json_obj.dump(2).c_str());
    if (json_obj.is_null() || !json_obj.is_object()) {
        throw std::runtime_error("global_from_json: input JSON value must be an object");
    }
    for (auto it = json_obj.begin(); it != json_obj.end(); ++it) {
        JJ_DEBUG("global_from_json: setting key '%s'", it.key().c_str());
        ctx.set_val(it.key(), from_json(it.value()));
    }
}

static void value_to_json_internal(std::ostringstream & oss, const value & val, int curr_lvl, int indent) {
    auto indent_str = [indent, curr_lvl]() -> std::string {
        return (indent > 0) ? std::string(curr_lvl * indent, ' ') : "";
    };
    auto newline = [indent]() -> std::string {
        return (indent > 0) ? "\n" : "";
    };

    if (is_val<value_null>(val) || val->is_undefined()) {
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
                oss << indent_str() << std::string(indent, ' ');
                value_to_json_internal(oss, arr[i], curr_lvl + 1, indent);
                if (i < arr.size() - 1) {
                    oss << ",";
                    if (indent == 0) oss << " ";
                }
                oss << newline();
            }
            oss << indent_str();
        }
        oss << "]";
    } else if (is_val<value_object>(val)) {
        const auto & obj = val->val_obj.ordered; // IMPORTANT: need to keep exact order
        oss << "{";
        if (!obj.empty()) {
            oss << newline();
            size_t i = 0;
            for (const auto & pair : obj) {
                oss << indent_str() << std::string(indent, ' ');
                oss << "\"" << pair.first << "\":";
                if (indent > 0) oss << " ";
                else oss << " ";
                value_to_json_internal(oss, pair.second, curr_lvl + 1, indent);
                if (i < obj.size() - 1) {
                    oss << ",";
                    if (indent == 0) oss << " ";
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

std::string value_to_json(const value & val, int indent) {
    std::ostringstream oss;
    value_to_json_internal(oss, val, 0, indent);
    JJ_DEBUG("value_to_json: result=%s", oss.str().c_str());
    return oss.str();
}

} // namespace jinja
