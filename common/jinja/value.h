#pragma once

#include "string.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace jinja {

struct value_t;
using value = std::shared_ptr<value_t>;


// Helper to check the type of a value
template<typename T>
struct extract_pointee {
    using type = T;
};
template<typename U>
struct extract_pointee<std::shared_ptr<U>> {
    using type = U;
};
template<typename T>
bool is_val(const value & ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<const PointeeType*>(ptr.get()) != nullptr;
}
template<typename T>
bool is_val(const value_t * ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<const PointeeType*>(ptr) != nullptr;
}
template<typename T, typename... Args>
std::shared_ptr<typename extract_pointee<T>::type> mk_val(Args&&... args) {
    using PointeeType = typename extract_pointee<T>::type;
    return std::make_shared<PointeeType>(std::forward<Args>(args)...);
}
template<typename T>
const typename extract_pointee<T>::type * cast_val(const value & ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<const PointeeType*>(ptr.get());
}
template<typename T>
typename extract_pointee<T>::type * cast_val(value & ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<PointeeType*>(ptr.get());
}
// End Helper


struct context; // forward declaration


// for converting from JSON to jinja values
// example input JSON:
// {
//   "messages": [
//     {"role": "user", "content": "Hello!"},
//     {"role": "assistant", "content": "Hi there!"}
//   ],
//   "bos_token": "<s>",
//   "eos_token": "</s>",
// }
//
// to mark strings as user input, wrap them in a special object:
// {
//   "messages": [
//     {
//       "role": "user",
//       "content": {"__input__": "Hello!"}  // this string is user input
//     },
//     ...
//   ],
// }
//
// marking input can be useful for tracking data provenance
// and preventing template injection attacks
//
// Note: T_JSON can be nlohmann::ordered_json
template<typename T_JSON>
void global_from_json(context & ctx, const T_JSON & json_obj, bool mark_input);

//
// base value type
//

struct func_args; // function argument values

using func_handler = std::function<value(const func_args &)>;
using func_builtins = std::map<std::string, func_handler>;

enum value_compare_op { eq, ge, gt, lt, ne };
bool value_compare(const value & a, const value & b, value_compare_op op);

struct value_t {
    int64_t val_int;
    double val_flt;
    string val_str;
    bool val_bool;

    std::vector<value> val_arr;

    struct map {
        // once set to true, all keys must be numeric
        // caveat: we only allow either all numeric keys or all non-numeric keys
        // for now, this only applied to for_statement in case of iterating over object keys/items
        bool is_key_numeric = false;
        std::map<std::string, value> unordered;
        std::vector<std::pair<std::string, value>> ordered;
        void insert(const std::string & key, const value & val) {
            if (unordered.find(key) != unordered.end()) {
                // if key exists, remove from ordered list
                ordered.erase(std::remove_if(ordered.begin(), ordered.end(),
                    [&](const std::pair<std::string, value> & p) { return p.first == key; }),
                    ordered.end());
            }
            unordered[key] = val;
            ordered.push_back({key, val});
        }
    } val_obj;

    func_handler val_func;

    // only used if ctx.is_get_stats = true
    struct stats_t {
        bool used = false;
        // ops can be builtin calls or operators: "array_access", "object_access"
        std::set<std::string> ops;
    } stats;

    value_t() = default;
    value_t(const value_t &) = default;
    virtual ~value_t() = default;

    virtual std::string type() const { return ""; }

    virtual int64_t as_int() const { throw std::runtime_error(type() + " is not an int value"); }
    virtual double as_float() const { throw std::runtime_error(type() + " is not a float value"); }
    virtual string as_string() const { throw std::runtime_error(type() + " is not a string value"); }
    virtual bool as_bool() const { throw std::runtime_error(type() + " is not a bool value"); }
    virtual const std::vector<value> & as_array() const { throw std::runtime_error(type() + " is not an array value"); }
    virtual const std::vector<std::pair<std::string, value>> & as_ordered_object() const { throw std::runtime_error(type() + " is not an object value"); }
    virtual value invoke(const func_args &) const { throw std::runtime_error(type() + " is not a function value"); }
    virtual bool is_none() const { return false; }
    virtual bool is_undefined() const { return false; }
    virtual const func_builtins & get_builtins() const {
        throw std::runtime_error("No builtins available for type " + type());
    }

    virtual bool has_key(const std::string & key) {
        return val_obj.unordered.find(key) != val_obj.unordered.end();
    }
    virtual value & at(const std::string & key, value & default_val) {
        auto it = val_obj.unordered.find(key);
        if (it == val_obj.unordered.end()) {
            return default_val;
        }
        return val_obj.unordered.at(key);
    }
    virtual value & at(const std::string & key) {
        auto it = val_obj.unordered.find(key);
        if (it == val_obj.unordered.end()) {
            throw std::runtime_error("Key '" + key + "' not found in value of type " + type());
        }
        return val_obj.unordered.at(key);
    }
    virtual value & at(int64_t index, value & default_val) {
        if (index < 0) {
            index += val_arr.size();
        }
        if (index < 0 || static_cast<size_t>(index) >= val_arr.size()) {
            return default_val;
        }
        return val_arr[index];
    }
    virtual value & at(int64_t index) {
        if (index < 0) {
            index += val_arr.size();
        }
        if (index < 0 || static_cast<size_t>(index) >= val_arr.size()) {
            throw std::runtime_error("Index " + std::to_string(index) + " out of bounds for array of size " + std::to_string(val_arr.size()));
        }
        return val_arr[index];
    }

    virtual std::string as_repr() const { return as_string().str(); }
};

//
// primitive value types
//

struct value_int_t : public value_t {
    value_int_t(int64_t v) { val_int = v; }
    virtual std::string type() const override { return "Integer"; }
    virtual int64_t as_int() const override { return val_int; }
    virtual double as_float() const override { return static_cast<double>(val_int); }
    virtual string as_string() const override { return std::to_string(val_int); }
    virtual bool as_bool() const override {
        return val_int != 0;
    }
    virtual const func_builtins & get_builtins() const override;
};
using value_int = std::shared_ptr<value_int_t>;


struct value_float_t : public value_t {
    value_float_t(double v) { val_flt = v; }
    virtual std::string type() const override { return "Float"; }
    virtual double as_float() const override { return val_flt; }
    virtual int64_t as_int() const override { return static_cast<int64_t>(val_flt); }
    virtual string as_string() const override {
        std::string out = std::to_string(val_flt);
        out.erase(out.find_last_not_of('0') + 1, std::string::npos); // remove trailing zeros
        if (out.back() == '.') out.push_back('0'); // leave one zero if no decimals
        return out;
    }
    virtual bool as_bool() const override {
        return val_flt != 0.0;
    }
    virtual const func_builtins & get_builtins() const override;
};
using value_float = std::shared_ptr<value_float_t>;


struct value_string_t : public value_t {
    value_string_t() { val_str = string(); }
    value_string_t(const std::string & v) { val_str = string(v); }
    value_string_t(const string & v) { val_str = v; }
    virtual std::string type() const override { return "String"; }
    virtual string as_string() const override { return val_str; }
    virtual std::string as_repr() const override {
        std::ostringstream ss;
        for (const auto & part : val_str.parts) {
            ss << (part.is_input ? "INPUT: " : "TMPL:  ") << part.val << "\n";
        }
        return ss.str();
    }
    virtual bool as_bool() const override {
        return val_str.length() > 0;
    }
    virtual const func_builtins & get_builtins() const override;
    void mark_input() {
        val_str.mark_input();
    }
};
using value_string = std::shared_ptr<value_string_t>;


struct value_bool_t : public value_t {
    value_bool_t(bool v) { val_bool = v; }
    virtual std::string type() const override { return "Boolean"; }
    virtual bool as_bool() const override { return val_bool; }
    virtual string as_string() const override { return std::string(val_bool ? "True" : "False"); }
    virtual const func_builtins & get_builtins() const override;
};
using value_bool = std::shared_ptr<value_bool_t>;


struct value_array_t : public value_t {
    value_array_t() = default;
    value_array_t(value & v) {
        val_arr = v->val_arr;
    }
    value_array_t(const std::vector<value> & arr) {
        val_arr = arr;
    }
    void reverse() { std::reverse(val_arr.begin(), val_arr.end()); }
    void push_back(const value & val) { val_arr.push_back(val); }
    void push_back(value && val) { val_arr.push_back(std::move(val)); }
    value pop_at(int64_t index) {
        if (index < 0) {
            index = static_cast<int64_t>(val_arr.size()) + index;
        }
        if (index < 0 || index >= static_cast<int64_t>(val_arr.size())) {
            throw std::runtime_error("Index " + std::to_string(index) + " out of bounds for array of size " + std::to_string(val_arr.size()));
        }
        value val = val_arr.at(static_cast<size_t>(index));
        val_arr.erase(val_arr.begin() + index);
        return val;
    }
    virtual std::string type() const override { return "Array"; }
    virtual const std::vector<value> & as_array() const override { return val_arr; }
    virtual string as_string() const override {
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < val_arr.size(); i++) {
            if (i > 0) ss << ", ";
            ss << val_arr.at(i)->as_repr();
        }
        ss << "]";
        return ss.str();
    }
    virtual bool as_bool() const override {
        return !val_arr.empty();
    }
    virtual const func_builtins & get_builtins() const override;
};
using value_array = std::shared_ptr<value_array_t>;


struct value_object_t : public value_t {
    bool has_builtins = true; // context and loop objects do not have builtins
    value_object_t() = default;
    value_object_t(value & v) {
        val_obj = v->val_obj;
    }
    value_object_t(const std::map<std::string, value> & obj) {
        for (const auto & pair : obj) {
            val_obj.insert(pair.first, pair.second);
        }
    }
    value_object_t(const std::vector<std::pair<std::string, value>> & obj) {
        for (const auto & pair : obj) {
            val_obj.insert(pair.first, pair.second);
        }
    }
    void insert(const std::string & key, const value & val) {
        val_obj.insert(key, val);
    }
    virtual std::string type() const override { return "Object"; }
    virtual const std::vector<std::pair<std::string, value>> & as_ordered_object() const override { return val_obj.ordered; }
    virtual bool as_bool() const override {
        return !val_obj.unordered.empty();
    }
    virtual const func_builtins & get_builtins() const override;
};
using value_object = std::shared_ptr<value_object_t>;

//
// null and undefined types
//

struct value_none_t : public value_t {
    virtual std::string type() const override { return "None"; }
    virtual bool is_none() const override { return true; }
    virtual bool as_bool() const override { return false; }
    virtual string as_string() const override { return string("None"); }
    virtual std::string as_repr() const override { return type(); }
    virtual const func_builtins & get_builtins() const override;
};
using value_none = std::shared_ptr<value_none_t>;

struct value_undefined_t : public value_t {
    std::string hint; // for debugging, to indicate where undefined came from
    value_undefined_t(const std::string & h = "") : hint(h) {}
    virtual std::string type() const override { return hint.empty() ? "Undefined" : "Undefined (hint: '" + hint + "')"; }
    virtual bool is_undefined() const override { return true; }
    virtual bool as_bool() const override { return false; }
    virtual std::string as_repr() const override { return type(); }
    virtual const func_builtins & get_builtins() const override;
};
using value_undefined = std::shared_ptr<value_undefined_t>;

//
// function type
//

struct func_args {
public:
    std::string func_name; // for error messages
    context & ctx;
    func_args(context & ctx) : ctx(ctx) {}
    value get_kwarg(const std::string & key, value default_val) const;
    value get_kwarg_or_pos(const std::string & key, size_t pos) const;
    value get_pos(size_t pos) const;
    value get_pos(size_t pos, value default_val) const;
    const std::vector<value> & get_args() const;
    size_t count() const { return args.size(); }
    void push_back(const value & val);
    void push_front(const value & val);
    void ensure_count(size_t min, size_t max = 999) const {
        size_t n = args.size();
        if (n < min || n > max) {
            throw std::runtime_error("Function '" + func_name + "' expected between " + std::to_string(min) + " and " + std::to_string(max) + " arguments, got " + std::to_string(n));
        }
    }
    template<typename T> void ensure_val(const value & ptr) const {
        if (!is_val<T>(ptr)) {
            throw std::runtime_error("Function '" + func_name + "' expected value of type " + std::string(typeid(T).name()) + ", got " + ptr->type());
        }
    }
    void ensure_count(bool require0, bool require1, bool require2, bool require3) const {
        static auto bool_to_int = [](bool b) { return b ? 1 : 0; };
        size_t required = bool_to_int(require0) + bool_to_int(require1) + bool_to_int(require2) + bool_to_int(require3);
        ensure_count(required);
    }
    template<typename T0> void ensure_vals(bool required0 = true) const {
        ensure_count(required0, false, false, false);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
    }
    template<typename T0, typename T1> void ensure_vals(bool required0 = true, bool required1 = true) const {
        ensure_count(required0, required1, false, false);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
        if (required1 && args.size() > 1) ensure_val<T1>(args[1]);
    }
    template<typename T0, typename T1, typename T2> void ensure_vals(bool required0 = true, bool required1 = true, bool required2 = true) const {
        ensure_count(required0, required1, required2, false);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
        if (required1 && args.size() > 1) ensure_val<T1>(args[1]);
        if (required2 && args.size() > 2) ensure_val<T2>(args[2]);
    }
    template<typename T0, typename T1, typename T2, typename T3> void ensure_vals(bool required0 = true, bool required1 = true, bool required2 = true, bool required3 = true) const {
        ensure_count(required0, required1, required2, required3);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
        if (required1 && args.size() > 1) ensure_val<T1>(args[1]);
        if (required2 && args.size() > 2) ensure_val<T2>(args[2]);
        if (required3 && args.size() > 3) ensure_val<T3>(args[3]);
    }
private:
    std::vector<value> args;
};

struct value_func_t : public value_t {
    std::string name;
    value arg0; // bound "this" argument, if any
    value_func_t(const std::string & name, const func_handler & func) : name(name) {
        val_func = func;
    }
    value_func_t(const std::string & name, const func_handler & func, const value & arg_this) : name(name), arg0(arg_this) {
        val_func = func;
    }
    virtual value invoke(const func_args & args) const override {
        func_args new_args(args); // copy
        new_args.func_name = name;
        if (arg0) {
            new_args.push_front(arg0);
        }
        return val_func(new_args);
    }
    virtual std::string type() const override { return "Function"; }
    virtual std::string as_repr() const override { return type(); }
};
using value_func = std::shared_ptr<value_func_t>;

// special value for kwarg
struct value_kwarg_t : public value_t {
    std::string key;
    value val;
    value_kwarg_t(const std::string & k, const value & v) : key(k), val(v) {}
    virtual std::string type() const override { return "KwArg"; }
    virtual std::string as_repr() const override { return type(); }
};
using value_kwarg = std::shared_ptr<value_kwarg_t>;


// utils

const func_builtins & global_builtins();
std::string value_to_json(const value & val, int indent = -1, const std::string_view item_sep = ", ", const std::string_view key_sep = ": ");

struct not_implemented_exception : public std::runtime_error {
    not_implemented_exception(const std::string & msg) : std::runtime_error("NotImplemented: " + msg) {}
};


} // namespace jinja
