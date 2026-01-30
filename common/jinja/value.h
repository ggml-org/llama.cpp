#pragma once

#include "string.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
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

using func_hptr = value(const func_args &);
using func_handler = std::function<func_hptr>;
using func_builtins = std::map<std::string, func_handler>;

enum value_compare_op { eq, ge, gt, lt, ne };
bool value_compare(const value & a, const value & b, value_compare_op op);

struct value_t {
    int64_t val_int;
    double val_flt;
    string val_str;

    std::vector<value> val_arr;
    std::vector<std::pair<value, value>> val_obj;

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

    // Note: only for debugging and error reporting purposes
    virtual std::string type() const { return ""; }

    virtual int64_t as_int() const { throw std::runtime_error(type() + " is not an int value"); }
    virtual double as_float() const { throw std::runtime_error(type() + " is not a float value"); }
    virtual string as_string() const { throw std::runtime_error(type() + " is not a string value"); }
    virtual bool as_bool() const { throw std::runtime_error(type() + " is not a bool value"); }
    virtual const std::vector<value> & as_array() const { throw std::runtime_error(type() + " is not an array value"); }
    virtual const std::vector<std::pair<value, value>> & as_ordered_object() const { throw std::runtime_error(type() + " is not an object value"); }
    virtual value invoke(const func_args &) const { throw std::runtime_error(type() + " is not a function value"); }
    virtual bool is_none() const { return false; }
    virtual bool is_undefined() const { return false; }
    virtual const func_builtins & get_builtins() const {
        throw std::runtime_error("No builtins available for type " + type());
    }

    virtual bool has_key(const value &) { throw std::runtime_error(type() + " is not an object value"); }
    virtual void insert(const value & /* key */, const value & /* val */) { throw std::runtime_error(type() + " is not an object value"); }
    virtual value & at(const value & /* key */, value & /* default_val */) { throw std::runtime_error(type() + " is not an object value"); }
    virtual value & at(const value & /* key */) { throw std::runtime_error(type() + " is not an object value"); }
    virtual value & at(const std::string & /* key */, value & /* default_val */) { throw std::runtime_error(type() + " is not an object value"); }
    virtual value & at(const std::string & /* key */) { throw std::runtime_error(type() + " is not an object value"); }
    virtual value & at(int64_t /* idx */, value & /* default_val */) { throw std::runtime_error(type() + " is not an array value"); }
    virtual value & at(int64_t /* idx */) { throw std::runtime_error(type() + " is not an array value"); }

    virtual bool is_numeric() const { return false; }
    virtual bool is_hashable() const { return false; }
    virtual bool is_immutable() const { return true; }
    virtual hasher unique_hash() const noexcept = 0;
    // TODO: C++20 <=> operator
    // NOTE: We are treating == as equivalent (for normal comparisons) and != as strict nonequal (for strict (is) comparisons)
    virtual bool operator==(const value_t & other) const { return equivalent(other); }
    virtual bool operator!=(const value_t & other) const { return nonequal(other); }

    // Note: only for debugging purposes
    virtual std::string as_repr() const { return as_string().str(); }

protected:
    virtual bool equivalent(const value_t &) const = 0;
    virtual bool nonequal(const value_t & other) const { return !equivalent(other); }
};

//
// utils
//

const func_builtins & global_builtins();

std::string value_to_json(const value & val, int indent = -1, const std::string_view item_sep = ", ", const std::string_view key_sep = ": ");

// Note: only used for debugging purposes
std::string value_to_string_repr(const value & val);

struct not_implemented_exception : public std::runtime_error {
    not_implemented_exception(const std::string & msg) : std::runtime_error("NotImplemented: " + msg) {}
};

struct value_hasher {
    size_t operator()(const value & val) const noexcept {
        return val->unique_hash().digest();
    }
};

struct value_equivalence {
    bool operator()(const value & lhs, const value & rhs) const {
        return *lhs == *rhs;
    }
    bool operator()(const std::pair<value, value> & lhs, const std::pair<value, value> & rhs) const {
        return *(lhs.first) == *(rhs.first) && *(lhs.second) == *(rhs.second);
    }
};

struct value_equality {
    bool operator()(const value & lhs, const value & rhs) const {
        return !(*lhs != *rhs);
    }
};

//
// primitive value types
//

struct value_int_t : public value_t {
    value_int_t(int64_t v) {
        val_int = v;
        val_flt = static_cast<double>(v);
        if (static_cast<int64_t>(val_flt) != v) {
            val_flt = v < 0 ? -INFINITY : INFINITY;
        }
    }
    virtual std::string type() const override { return "Integer"; }
    virtual int64_t as_int() const override { return val_int; }
    virtual double as_float() const override { return val_flt; }
    virtual string as_string() const override { return std::to_string(val_int); }
    virtual bool as_bool() const override {
        return val_int != 0;
    }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_numeric() const override { return true; }
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        return hasher(typeid(*this))
            .update(&val_int, sizeof(val_int))
            .update(&val_flt, sizeof(val_flt));
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return other.is_numeric() && val_int == other.val_int && val_flt == other.val_flt;
    }
    virtual bool nonequal(const value_t & other) const override {
        return !(typeid(*this) == typeid(other) && val_int == other.val_int);
    }
};
using value_int = std::shared_ptr<value_int_t>;


struct value_float_t : public value_t {
    value val;
    value_float_t(double v) {
        val_flt = v;
        val_int = std::isfinite(v) ? static_cast<int64_t>(v) : 0;
        val = mk_val<value_int>(val_int);
    }
    virtual std::string type() const override { return "Float"; }
    virtual double as_float() const override { return val_flt; }
    virtual int64_t as_int() const override { return val_int; }
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
    virtual bool is_numeric() const override { return true; }
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        if (static_cast<double>(val_int) == val_flt) {
            return val->unique_hash();
        } else {
            return hasher(typeid(*this))
                .update(&val_int, sizeof(val_int))
                .update(&val_flt, sizeof(val_flt));
        }
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return other.is_numeric() && val_int == other.val_int && val_flt == other.val_flt;
    }
    virtual bool nonequal(const value_t & other) const override {
        return !(typeid(*this) == typeid(other) && val_flt == other.val_flt);
    }
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
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        const auto type_hash = typeid(*this).hash_code();
        auto hash = hasher();
        hash.update(&type_hash, sizeof(type_hash));
        val_str.hash_update(hash);
        return hash;
    }
    void mark_input() {
        val_str.mark_input();
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return typeid(*this) == typeid(other) && val_str.str() == other.val_str.str();
    }
};
using value_string = std::shared_ptr<value_string_t>;


struct value_bool_t : public value_t {
    value val;
    value_bool_t(bool v) {
        val_int = static_cast<int64_t>(v);
        val_flt = static_cast<double>(v);
        val = mk_val<value_int>(val_int);
    }
    virtual std::string type() const override { return "Boolean"; }
    virtual int64_t as_int() const override { return val_int; }
    virtual bool as_bool() const override { return val_int; }
    virtual string as_string() const override { return std::string(val_int ? "True" : "False"); }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_numeric() const override { return true; }
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        return val->unique_hash();
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return other.is_numeric() && val_int == other.val_int && val_flt == other.val_flt;
    }
    virtual bool nonequal(const value_t & other) const override {
        return !(typeid(*this) == typeid(other) && val_int == other.val_int);
    }
};
using value_bool = std::shared_ptr<value_bool_t>;


struct value_array_t : public value_t {
    value_array_t() = default;
    value_array_t(value & v) {
        val_arr = v->val_arr;
    }
    value_array_t(std::vector<value> && arr) {
        val_arr = arr;
    }
    value_array_t(const std::vector<value> & arr) {
        val_arr = arr;
    }
    void reverse() {
        if (is_immutable()) {
            throw std::runtime_error("Attempting to modify immutable type");
        }
        std::reverse(val_arr.begin(), val_arr.end());
    }
    void push_back(const value & val) {
        if (is_immutable()) {
            throw std::runtime_error("Attempting to modify immutable type");
        }
        val_arr.push_back(val);
    }
    void push_back(value && val) {
        if (is_immutable()) {
            throw std::runtime_error("Attempting to modify immutable type");
        }
        val_arr.push_back(std::move(val));
    }
    value pop_at(int64_t index) {
        if (is_immutable()) {
            throw std::runtime_error("Attempting to modify immutable type");
        }
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
    virtual bool is_immutable() const override { return false; }
    virtual const std::vector<value> & as_array() const override { return val_arr; }
    virtual string as_string() const override {
        const bool immutable = is_immutable();
        std::ostringstream ss;
        ss << (immutable ? "(" : "[");
        for (size_t i = 0; i < val_arr.size(); i++) {
            if (i > 0) ss << ", ";
            value val = val_arr.at(i);
            ss << value_to_string_repr(val);
        }
        if (immutable && val_arr.size() == 1) {
            ss << ",";
        }
        ss << (immutable ? ")" : "]");
        return ss.str();
    }
    virtual bool as_bool() const override {
        return !val_arr.empty();
    }
    virtual value & at(int64_t index, value & default_val) override {
        if (index < 0) {
            index += val_arr.size();
        }
        if (index < 0 || static_cast<size_t>(index) >= val_arr.size()) {
            return default_val;
        }
        return val_arr[index];
    }
    virtual value & at(int64_t index) override {
        if (index < 0) {
            index += val_arr.size();
        }
        if (index < 0 || static_cast<size_t>(index) >= val_arr.size()) {
            throw std::runtime_error("Index " + std::to_string(index) + " out of bounds for array of size " + std::to_string(val_arr.size()));
        }
        return val_arr[index];
    }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_hashable() const override {
        if (std::all_of(val_arr.begin(), val_arr.end(), [&](auto & val) -> bool {
            return val->is_immutable() && val->is_hashable();
        })) {
            return true;
        }
        return false;
    }
    virtual hasher unique_hash() const noexcept override {
        auto hash = hasher(typeid(*this));
        for (const auto & val : val_arr) {
            // must use digest to prevent problems from "concatenation" property of hasher
            // for ex. hash of [ "ab", "c" ] should be different from [ "a", "bc" ]
            const size_t val_hash = val->unique_hash().digest();
            hash.update(&val_hash, sizeof(size_t));
        }
        return hash;
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return typeid(*this) == typeid(other) && is_hashable() && other.is_hashable() && std::equal(val_arr.begin(), val_arr.end(), other.val_arr.begin(), value_equivalence());
    }
};
using value_array = std::shared_ptr<value_array_t>;


struct value_tuple_t : public value_array_t {
    value_tuple_t(value & v) {
        val_arr = v->val_arr;
    }
    value_tuple_t(std::vector<value> && arr) {
        val_arr = arr;
    }
    value_tuple_t(const std::vector<value> & arr) {
        val_arr = arr;
    }
    value_tuple_t(const std::pair<value, value> & pair) {
        val_arr.push_back(pair.first);
        val_arr.push_back(pair.second);
    }
    virtual std::string type() const override { return "Tuple"; }
    virtual bool is_immutable() const override { return true; }
};
using value_tuple = std::shared_ptr<value_tuple_t>;


struct value_object_t : public value_t {
    std::unordered_map<value, value, value_hasher, value_equivalence> unordered;
    bool has_builtins = true; // context and loop objects do not have builtins
    value_object_t() = default;
    value_object_t(value & v) {
        val_obj = v->val_obj;
        for (const auto & pair : val_obj) {
            unordered[pair.first] = pair.second;
        }
    }
    value_object_t(const std::map<value, value> & obj) {
        for (const auto & pair : obj) {
            insert(pair.first, pair.second);
        }
    }
    value_object_t(const std::vector<std::pair<value, value>> & obj) {
        for (const auto & pair : obj) {
            insert(pair.first, pair.second);
        }
    }
    void insert(const std::string & key, const value & val) {
        insert(mk_val<value_string>(key), val);
    }
    virtual std::string type() const override { return "Object"; }
    virtual bool is_immutable() const override { return false; }
    virtual const std::vector<std::pair<value, value>> & as_ordered_object() const override { return val_obj; }
    virtual string as_string() const override {
        std::ostringstream ss;
        ss << "{";
        for (size_t i = 0; i < val_obj.size(); i++) {
            if (i > 0) ss << ", ";
            auto & [key, val] = val_obj.at(i);
            ss << value_to_string_repr(key) << ": " << value_to_string_repr(val);
        }
        ss << "}";
        return ss.str();
    }
    virtual bool as_bool() const override {
        return !unordered.empty();
    }
    virtual bool has_key(const value & key) override {
        if (!key->is_immutable() || !key->is_hashable()) {
            throw std::runtime_error("Object key of unhashable type: " + key->type());
        }
        return unordered.find(key) != unordered.end();
    }
    virtual void insert(const value & key, const value & val) override {
        bool replaced = false;
        if (is_immutable()) {
            throw std::runtime_error("Attempting to modify immutable type");
        }
        if (has_key(key)) {
            // if key exists, replace value in ordered list instead of appending
            for (auto & pair : val_obj) {
                if (*(pair.first) == *key) {
                    pair.second = val;
                    replaced = true;
                    break;
                }
            }
        }
        unordered[key] = val;
        if (!replaced) {
            val_obj.push_back({key, val});
        }
    }
    virtual value & at(const value & key, value & default_val) override {
        if (!has_key(key)) {
            return default_val;
        }
        return unordered.at(key);
    }
    virtual value & at(const value & key) override {
        if (!has_key(key)) {
            throw std::runtime_error("Key '" + key->as_string().str() + "' not found in value of type " + type());
        }
        return unordered.at(key);
    }
    virtual value & at(const std::string & key, value & default_val) override {
        value key_val = mk_val<value_string>(key);
        return at(key_val, default_val);
    }
    virtual value & at(const std::string & key) override {
        value key_val = mk_val<value_string>(key);
        return at(key_val);
    }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_hashable() const override {
        if (std::all_of(val_obj.begin(), val_obj.end(), [&](auto & pair) -> bool {
            const auto & val = pair.second;
            return val->is_immutable() && val->is_hashable();
        })) {
            return true;
        }
        return false;
    }
    virtual hasher unique_hash() const noexcept override {
        auto hash = hasher(typeid(*this));
        for (const auto & [key, val] : val_obj) {
            // must use digest to prevent problems from "concatenation" property of hasher
            // for ex. hash of key="ab", value="c" should be different from key="a", value="bc"
            const size_t key_hash = key->unique_hash().digest();
            const size_t val_hash = val->unique_hash().digest();
            hash.update(&key_hash, sizeof(key_hash));
            hash.update(&val_hash, sizeof(val_hash));
        }
        return hash;
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return typeid(*this) == typeid(other) && is_hashable() && other.is_hashable() && std::equal(val_obj.begin(), val_obj.end(), other.val_obj.begin(), value_equivalence());
    }
};
using value_object = std::shared_ptr<value_object_t>;

//
// none and undefined types
//

struct value_none_t : public value_t {
    virtual std::string type() const override { return "None"; }
    virtual bool is_none() const override { return true; }
    virtual bool as_bool() const override { return false; }
    virtual string as_string() const override { return string(type()); }
    virtual std::string as_repr() const override { return type(); }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        return hasher(typeid(*this));
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return typeid(*this) == typeid(other);
    }
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
    virtual hasher unique_hash() const noexcept override {
        return hasher(typeid(*this));
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return is_undefined() == other.is_undefined();
    }
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
    virtual std::string as_repr() const override { return type() + "<" + name + ">(" + (arg0 ? arg0->as_repr() : "") + ")"; }
    virtual bool is_hashable() const override { return false; }
    virtual hasher unique_hash() const noexcept override {
        // Note: this is unused for now, we don't support function as object keys
        // use function pointer as unique identifier
        const auto target = val_func.target<func_hptr>();
        return hasher(typeid(*this)).update(&target, sizeof(target));
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        // Note: this is unused for now, we don't support function as object keys
        // compare function pointers
        // (val_func == other.val_func does not work as std::function::operator== is only used for nullptr check)
        const auto target_this  = this->val_func.target<func_hptr>();
        const auto target_other = other.val_func.target<func_hptr>();
        return typeid(*this) == typeid(other) && target_this == target_other;
    }
};
using value_func = std::shared_ptr<value_func_t>;

// special value for kwarg
struct value_kwarg_t : public value_t {
    std::string key;
    value val;
    value_kwarg_t(const std::string & k, const value & v) : key(k), val(v) {}
    virtual std::string type() const override { return "KwArg"; }
    virtual std::string as_repr() const override { return type(); }
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        const auto type_hash = typeid(*this).hash_code();
        auto hash = val->unique_hash();
        hash.update(&type_hash, sizeof(type_hash))
            .update(key.data(), key.size());
        return hash;
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        const value_kwarg_t & other_val = static_cast<const value_kwarg_t &>(other);
        return typeid(*this) == typeid(other) && key == other_val.key && val == other_val.val;
    }
};
using value_kwarg = std::shared_ptr<value_kwarg_t>;


} // namespace jinja
