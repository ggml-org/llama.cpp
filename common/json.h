#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// common_json, a thin wrapper around vendor json library
// the underlay library is pimpl, we are using nlohmann::json for now
//
// many features of the library are deliberately left out, to keep this interface small and generic and to keep compile time down
//
// some main differences compared to nlohmann::json :
// - object keys keep the order in which they are added
// - errors are always throw as common_json_error
// - obj.push_back({key, val}) is intentionally unsupported to avoid confusion with push_back on a vector; write it as obj[key] = val for clarity
// - a braced pair in value position does not build, e.g. {"key", {"a", "b"}}; write array({"a", "b"}) where nlohmann made an array
//
// in doubt, search the code base for an existing usage example; do not add anything to this header unless absolutely necessary

class common_json;

// common_json_value holds a list of these, and each of them holds a value, so one must come first
struct common_json_item;

struct common_json_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// one value, tagged so that this header stays free of the backing library
// note: a value that holds a tree is single use, the second use gives null
struct common_json_value {
    enum value_type {
        VAL_NULL,
        VAL_BOOL,
        VAL_INT,
        VAL_UINT,
        VAL_DOUBLE,
        VAL_STRING,
        VAL_JSON,
    };

    value_type type = VAL_NULL;

    union {
        bool     val_bool;
        int64_t  val_int;
        uint64_t val_uint = 0;
        double   val_double;
    };

    std::string                  val_string;
    std::shared_ptr<common_json> val_json;

    common_json_value(std::nullptr_t = nullptr) : type(VAL_NULL) {}
    common_json_value(bool val) : type(VAL_BOOL), val_bool(val) {}
    common_json_value(std::string val) : type(VAL_STRING), val_string(std::move(val)) {}
    // without this a string_view lands on the common_json ctor below and recurses
    common_json_value(std::string_view val) : type(VAL_STRING), val_string(val) {}
    common_json_value(const char * val);
    common_json_value(const common_json & val);
    common_json_value(common_json && val);
    // only for the types instantiated in json.cpp, the rest fails at link time
    template <typename T> common_json_value(const std::vector<T> & vals);
    // a set becomes an array, in the set's own order
    template <typename T> common_json_value(const std::set<T> & vals);
    // a map becomes an object, keyed in the map's own order
    template <typename T> common_json_value(const std::map<std::string, T> & vals);
    template <typename T> common_json_value(const std::unordered_map<std::string, T> & vals);

    // nested object, e.g. {"fn", {{"name", "x"}}}
    // note: a nested pair {"a", "b"} does not build, use common_json::array({"a", "b"}) for an array
    common_json_value(std::initializer_list<common_json_item> items);

    template <typename T, typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, int>::type = 0>
    common_json_value(T val) : type(std::is_signed<T>::value ? VAL_INT : VAL_UINT) {
        if (std::is_signed<T>::value) {
            val_int = (int64_t) val;
        } else {
            val_uint = (uint64_t) val;
        }
    }

    template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
    common_json_value(T val) : type(VAL_DOUBLE), val_double((double) val) {}
};

struct common_json_item {
    std::string       key;
    common_json_value val;

    template <typename T>
    common_json_item(std::string key, T && val) :
        key(std::move(key)), val(std::forward<T>(val)) {}

    // a braced list cannot deduce T, so it needs its own overload
    common_json_item(std::string key, std::initializer_list<common_json_item> items) :
        key(std::move(key)), val(items) {}
};

// the types common_json_value holds on its own
// anything else reaches its common_json ctor and recurses forever
template <typename T> struct common_json_is_value : std::integral_constant<bool,
    std::is_arithmetic<T>::value ||
    std::is_same<T, std::nullptr_t>::value ||
    std::is_same<T, std::string>::value ||
    std::is_same<T, std::string_view>::value ||
    std::is_same<T, char *>::value ||
    std::is_same<T, const char *>::value ||
    std::is_same<T, common_json>::value> {};

template <typename T, typename A>
struct common_json_is_value<std::vector<T, A>> : std::true_type {};

template <typename T, typename C, typename A>
struct common_json_is_value<std::set<T, C, A>> : std::true_type {};

template <typename V, typename C, typename A>
struct common_json_is_value<std::map<std::string, V, C, A>> : std::true_type {};

template <typename V, typename H, typename E, typename A>
struct common_json_is_value<std::unordered_map<std::string, V, H, E, A>> : std::true_type {};

class common_json {
  public:
    common_json();
    common_json(const common_json & other);
    common_json(common_json && other) noexcept;
    common_json(std::initializer_list<common_json_item> items);
    common_json(const common_json_value & val);

    // direct, a value would need two conversions in a row
    common_json(std::nullptr_t);

    // one step, so that "abc" or a vector can go straight into a common_json
    template <typename T, typename std::enable_if<!std::is_same<typename std::decay<T>::type, common_json>::value &&
                                                  !std::is_same<typename std::decay<T>::type, common_json_value>::value, int>::type = 0>
    common_json(T && val) : common_json(common_json_value(std::forward<T>(val))) {
        static_assert(common_json_is_value<typename std::decay<T>::type>::value,
                      "no common_json_value ctor holds this type, add one instead of letting it recurse");
    }

    // by value, same as the backing library
    // the right side is copied before the left side can invalidate it, e.g. msg["a"] = msg.at("b")
    common_json & operator=(common_json other) noexcept;

    ~common_json();

    // throws common_json_error if the text is not valid JSON
    static common_json parse(const std::string & text);

    // gives a discarded value instead of throwing, check it with is_discarded()
    static common_json parse_no_throw(const std::string & text);

    bool is_discarded() const;

    static common_json array();
    static common_json array(std::initializer_list<common_json_value> vals);
    static common_json object();
    static common_json object(std::initializer_list<common_json_item> items);

    // holds a single value, e.g. make("abc").dump() gives "\"abc\""
    static common_json make(const common_json_value & val);

    bool is_null()    const;
    bool is_object()  const;
    bool is_array()   const;
    bool is_string()  const;
    bool is_boolean() const;
    bool is_number()  const;
    bool is_number_integer() const;
    bool is_number_float()   const;

    bool   empty() const;
    size_t size()  const;

    bool contains(const std::string & key) const;

    bool operator==(const common_json_value & val) const;
    bool operator!=(const common_json_value & val) const;

    // at() throws common_json_error if the key is missing, operator[] adds a null value instead
    // note: a const operator[] cannot add, it throws like at()
    common_json       & at(const std::string & key);
    const common_json & at(const std::string & key) const;
    common_json       & at(size_t idx);
    const common_json & at(size_t idx) const;

    common_json       & operator[](const std::string & key);
    const common_json & operator[](const std::string & key) const;
    common_json       & operator[](const char * key)       { return (*this)[std::string(key)]; }
    const common_json & operator[](const char * key) const { return (*this)[std::string(key)]; }
    common_json       & operator[](int idx)       { return (*this)[to_idx(idx)]; }
    const common_json & operator[](int idx) const { return (*this)[to_idx(idx)]; }
    common_json       & operator[](size_t idx);
    const common_json & operator[](size_t idx) const;

    common_json       & front();
    const common_json & front() const;
    common_json       & back();
    const common_json & back()  const;

    void clear();

    void erase(const std::string & key);
    void erase(size_t idx);

    // only for the types instantiated in json.cpp, the rest fails at link time
    template <typename T> T get() const;

    // implicit get<T>() for plain values, so they can be assigned to their C++ type directly
    // note: kept to this short list on purpose, a wider one makes j["key"] ambiguous
    // note: a numeric one would make "str = json;" ambiguous, a number converts to char too
    operator std::string() const;

    template <typename T>
    T value(const std::string & key, T def) const {
        return contains(key) ? at(key).get<T>() : def;
    }

    std::string value(const std::string & key, const char * def) const;

    // a JSON default needs no get<T>(), it is already the right type
    common_json value(const std::string & key, const common_json & def) const {
        return contains(key) ? at(key) : def;
    }

    void assign(const common_json_value & val);
    void set(const common_json_item & item);
    void push_back(const common_json_value & val);

    // appends one object, e.g. push_back({{"a", 1}})
    void push_back(std::initializer_list<common_json_item> items);

    // 1 if the key is there, 0 if not
    size_t count(const std::string & key) const;

    // appends every value of another array; inserting an array into itself throws
    void insert(const common_json & vals);

    // a common_json goes through the copy assignment above, everything else becomes a value
    template <typename T, typename std::enable_if<!std::is_same<typename std::decay<T>::type, common_json>::value, int>::type = 0>
    common_json & operator=(T && val) {
        assign(common_json_value(std::forward<T>(val)));
        return *this;
    }

    std::string dump(int indent = -1) const;

    // same as dump(), but bad UTF-8 gets replaced instead of throwing
    std::string dump_safe(int indent = -1) const;

    // walks an array by index, or an object in insertion order
    // a plain value gives itself once, same as the backing library
    class iterator {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = common_json;
        using difference_type   = std::ptrdiff_t;
        using pointer           = common_json *;
        using reference         = common_json &;

        iterator(common_json * node, size_t idx) : node(node), idx(idx) {}

        common_json & operator*() const;
        common_json & value()     const { return **this; }
        std::string   key()       const;

        iterator & operator++() {
            idx++;
            return *this;
        }

        bool operator!=(const iterator & other) const { return idx != other.idx; }
        bool operator==(const iterator & other) const { return idx == other.idx; }

      private:
        common_json * node;
        size_t        idx;
    };

    iterator begin() const;
    iterator end()   const;

    // allows: for (const auto & [key, val] : obj.items())
    class items_view {
      public:
        // the members are public, so an entry also works with structured bindings
        struct entry {
            std::string   k;
            common_json & v;

            const std::string & key()   const { return k; }
            common_json &       value() const { return v; }
        };

        items_view(common_json * node, size_t n) : node(node), n(n) {}

        class iterator {
          public:
            iterator(common_json * node, size_t idx) : node(node), idx(idx) {}

            entry operator*() const;

            iterator & operator++() {
                idx++;
                return *this;
            }

            bool operator!=(const iterator & other) const { return idx != other.idx; }

          private:
            common_json * node;
            size_t        idx;
        };

        iterator begin() const { return iterator(node, 0); }
        iterator end()   const { return iterator(node, n); }

      private:
        common_json * node;
        size_t        n;
    };

    items_view items() const;

  private:
    // a negative index must not turn into a huge size_t
    static size_t to_idx(int idx) {
        if (idx < 0) {
            throw common_json_error("negative array index");
        }
        return (size_t) idx;
    }

    // the backing value is built here, json.cpp checks that it fits
    // it cannot be a pointer: a value inside a tree would then not be a common_json
    // at() could then only give back a copy instead of a real reference
    alignas(8) unsigned char storage[32];
};

using common_json_entry = common_json::items_view::entry;
