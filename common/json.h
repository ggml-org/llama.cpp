#pragma once

// JSON object, it works without the need to include a JSON library header
// the underlay library is pimpl, it should never be exposed here
// note: object keys keep the order in which they are added

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

class common_json;
class common_json_ref;

// common_json_value holds a list of these, and each of them holds a value, so one must come first
struct common_json_item;

// one value of the backing library, only json.cpp knows what it is
struct common_json_node;

struct common_json_node_deleter {
    void operator()(common_json_node * node) const;
};

struct common_json_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// one value, tagged so that this header stays free of the backing library
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
    common_json_value(const char * val);
    common_json_value(const common_json & val);
    common_json_value(const common_json_ref & val);

    // nested object, e.g. {"fn", {{"name", "x"}}}
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

// view to a value owned by a common_json, it goes stale if the owner gets a new key
class common_json_ref {
  public:
    explicit common_json_ref(common_json_node * node) : node(node) {}

    common_json_ref(const common_json_ref &) = default;

    // rebinding a view is almost always a write-through by mistake, use assign() to write
    common_json_ref & operator=(const common_json_ref &) = delete;

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

    // at() throws if the key is missing, operator[] adds a null value instead
    common_json_ref at(const std::string & key) const;
    common_json_ref operator[](const std::string & key) const;
    common_json_ref operator[](size_t idx) const;

    // only for the types instantiated in json.cpp, the rest fails at link time
    template <typename T> T get() const;

    template <typename T>
    T value(const std::string & key, T def) const {
        return contains(key) ? at(key).get<T>() : def;
    }

    std::string value(const std::string & key, const char * def) const {
        return contains(key) ? at(key).get<std::string>() : std::string(def);
    }

    void assign(const common_json_value & val);
    void set(const common_json_item & item);
    void push_back(const common_json_value & val);

    template <typename T>
    common_json_ref & operator=(T && val) {
        assign(common_json_value(std::forward<T>(val)));
        return *this;
    }

    std::string dump(int indent = -1) const;

    // walks an array by index, or an object in insertion order
    class iterator {
      public:
        iterator(common_json_node * node, size_t idx) : node(node), idx(idx) {}

        common_json_ref operator*() const;
        std::string     key()       const;

        iterator & operator++() {
            idx++;
            return *this;
        }

        bool operator!=(const iterator & other) const { return idx != other.idx; }
        bool operator==(const iterator & other) const { return idx == other.idx; }

      private:
        common_json_node * node;
        size_t             idx;
    };

    iterator begin() const { return iterator(node, 0); }
    iterator end()   const { return iterator(node, size()); }

    // allows: for (const auto & [key, val] : obj.items())
    class items_view {
      public:
        items_view(common_json_node * node, size_t n) : node(node), n(n) {}

        class iterator {
          public:
            iterator(common_json_node * node, size_t idx) : node(node), idx(idx) {}

            std::pair<std::string, common_json_ref> operator*() const;

            iterator & operator++() {
                idx++;
                return *this;
            }

            bool operator!=(const iterator & other) const { return idx != other.idx; }

          private:
            common_json_node * node;
            size_t             idx;
        };

        iterator begin() const { return iterator(node, 0); }
        iterator end()   const { return iterator(node, n); }

      private:
        common_json_node * node;
        size_t             n;
    };

    items_view items() const { return items_view(node, size()); }

    common_json_node * get_node() const { return node; }

  protected:
    common_json_node * node;
};

// owns the value it points to
class common_json : public common_json_ref {
  public:
    common_json();
    common_json(std::initializer_list<common_json_item> items);
    common_json(const common_json & other);
    common_json(common_json && other) noexcept;

    common_json & operator=(const common_json & other);
    common_json & operator=(common_json && other) noexcept;

    // out-of-line, the deleter needs to know the real type
    ~common_json();

    // throws common_json_error if the text is not valid JSON
    static common_json parse(const std::string & text);

    static common_json array();
    static common_json array(std::initializer_list<common_json_value> vals);
    static common_json object();

    // holds a single value, e.g. make("abc").dump() gives "\"abc\""
    static common_json make(const common_json_value & val);

  private:
    std::unique_ptr<common_json_node, common_json_node_deleter> pimpl;
};

// bridge for code that still uses internal component from nlohmann::json
// usage: common_json_raw<nlohmann::ordered_json>(j)
// TODO: maybe completely remove this in the future

template <typename T> T       & common_json_raw(common_json_ref & json);
template <typename T> const T & common_json_raw(const common_json_ref & json);

template <typename T> common_json common_json_from_raw(const T & json);

// view over a value of the backing library, it does not copy
template <typename T> common_json_ref common_json_ref_from_raw(T & json);
