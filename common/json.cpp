#include "json.h"

#include "ggml.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <iterator>
#include <new>
#include <set>
#include <unordered_map>
#include <vector>

using nlohmann::ordered_json;

// a common_json is the backing value, so any value of a tree can be used as a common_json
static_assert(sizeof(ordered_json)  <= sizeof(common_json),  "common_json storage is too small");
static_assert(alignof(ordered_json) <= alignof(common_json), "common_json alignment is too weak");

// runs fn and gives every error of the backing library as a common_json_error
template <typename F>
static decltype(auto) guard(F && fn) {
    try {
        return fn();
    } catch (const ordered_json::exception & e) {
        throw common_json_error(e.what());
    }
}

static ordered_json & as_json(common_json * self) {
    return *reinterpret_cast<ordered_json *>(self);
}

static const ordered_json & as_json(const common_json * self) {
    return *reinterpret_cast<const ordered_json *>(self);
}

static common_json & as_common(ordered_json & json) {
    return *reinterpret_cast<common_json *>(&json);
}

static const common_json & as_common(const ordered_json & json) {
    return *reinterpret_cast<const common_json *>(&json);
}

static ordered_json to_json(const common_json_value & val) {
    switch (val.type) {
        case common_json_value::VAL_NULL:   return nullptr;
        case common_json_value::VAL_BOOL:   return val.val_bool;
        case common_json_value::VAL_INT:    return val.val_int;
        case common_json_value::VAL_UINT:   return val.val_uint;
        case common_json_value::VAL_DOUBLE: return val.val_double;
        case common_json_value::VAL_STRING: return val.val_string;
        case common_json_value::VAL_JSON:
            // one owner means no one else can see this tree, so it is safe to move it out
            // note: this makes a value single use, same as the json_ref of the backing library
            if (val.val_json.use_count() == 1) {
                return std::move(as_json(val.val_json.get()));
            }
            return as_json(val.val_json.get());
    }

    return nullptr;
}

common_json_value::common_json_value(const char * val) {
    if (val) {
        type       = VAL_STRING;
        val_string = val;
    } else {
        type = VAL_NULL;
    }
}

common_json_value::common_json_value(const common_json & val) :
    type(VAL_JSON), val_json(std::make_shared<common_json>(val)) {}

common_json_value::common_json_value(common_json && val) :
    type(VAL_JSON), val_json(std::make_shared<common_json>(std::move(val))) {}

// the ctors and get<T>() below are explicit specializations, giving strong symbols
// an explicit instantiation is a weak symbol, dropped by some LTO builds (clang-cl)
template <typename T>
static std::shared_ptr<common_json> set_json(const std::set<T> & vals) {
    common_json out = common_json::array();

    for (const auto & val : vals) {
        out.push_back(val);
    }

    return std::make_shared<common_json>(std::move(out));
}

// a set value is usable only for the types below
#define COMMON_JSON_SET(...) template <> common_json_value::common_json_value(const std::set<__VA_ARGS__> & vals) : type(VAL_JSON), val_json(set_json(vals)) {}

COMMON_JSON_SET(int)
COMMON_JSON_SET(std::string)

#undef COMMON_JSON_SET

template <typename T>
static std::shared_ptr<common_json> map_json(const T & vals) {
    common_json out = common_json::object();

    for (const auto & val : vals) {
        out.set({ val.first, val.second });
    }

    return std::make_shared<common_json>(std::move(out));
}

// a map value is usable only for the types below
#define COMMON_JSON_MAP(...) template <> common_json_value::common_json_value(const std::map<std::string, __VA_ARGS__> & vals) : type(VAL_JSON), val_json(map_json(vals)) {}

COMMON_JSON_MAP(bool)
COMMON_JSON_MAP(std::string)

#undef COMMON_JSON_MAP

// an unordered map value is usable only for the types below
#define COMMON_JSON_UMAP(...) template <> common_json_value::common_json_value(const std::unordered_map<std::string, __VA_ARGS__> & vals) : type(VAL_JSON), val_json(map_json(vals)) {}

COMMON_JSON_UMAP(size_t)

#undef COMMON_JSON_UMAP

template <typename T>
static std::shared_ptr<common_json> vec_json(const std::vector<T> & vals) {
    common_json out = common_json::array();

    for (const auto & val : vals) {
        out.push_back(val);
    }

    return std::make_shared<common_json>(std::move(out));
}

// a vector value is usable only for the types below
// note: std::vector<bool> is not here, its proxy reference does not convert
#define COMMON_JSON_VEC(...) template <> common_json_value::common_json_value(const std::vector<__VA_ARGS__> & vals) : type(VAL_JSON), val_json(vec_json(vals)) {}

COMMON_JSON_VEC(int)
COMMON_JSON_VEC(unsigned char)
COMMON_JSON_VEC(unsigned int)
COMMON_JSON_VEC(long)
COMMON_JSON_VEC(unsigned long)
COMMON_JSON_VEC(long long)
COMMON_JSON_VEC(unsigned long long)
COMMON_JSON_VEC(float)
COMMON_JSON_VEC(double)
COMMON_JSON_VEC(std::string)
COMMON_JSON_VEC(std::vector<float>)
COMMON_JSON_VEC(common_json)

#undef COMMON_JSON_VEC

common_json_value::common_json_value(std::initializer_list<common_json_item> items) :
    type(VAL_JSON), val_json(std::make_shared<common_json>(items)) {}

// null, same as the backing library
// operator[] turns it into an object, push_back() into an array
common_json::common_json() {
    new (storage) ordered_json();
}

common_json::common_json(const common_json & other) {
    new (storage) ordered_json(as_json(&other));
}

common_json::common_json(common_json && other) noexcept {
    new (storage) ordered_json(std::move(as_json(&other)));
}

common_json::common_json(std::initializer_list<common_json_item> items) {
    new (storage) ordered_json(ordered_json::object());

    for (const auto & item : items) {
        set(item);
    }
}

common_json::common_json(const common_json_value & val) {
    new (storage) ordered_json(to_json(val));
}

common_json::common_json(std::nullptr_t) {
    new (storage) ordered_json(nullptr);
}

common_json & common_json::operator=(common_json other) noexcept {
    as_json(this).swap(as_json(&other));

    return *this;
}

common_json::~common_json() {
    as_json(this).~basic_json();
}

common_json common_json::parse(const std::string & text) {
    try {
        // the assignment moves the parsed tree in, it does not copy
        common_json out;
        as_json(&out) = ordered_json::parse(text);
        return out;
    } catch (const std::exception & e) {
        throw common_json_error(e.what());
    }
}

common_json common_json::parse_no_throw(const std::string & text) {
    common_json out;
    as_json(&out) = ordered_json::parse(text, nullptr, false);
    return out;
}

bool common_json::is_discarded() const {
    return as_json(this).is_discarded();
}

common_json common_json::array() {
    common_json out;
    as_json(&out) = ordered_json::array();
    return out;
}

common_json common_json::array(std::initializer_list<common_json_value> vals) {
    common_json out;
    ordered_json & arr = as_json(&out);
    arr = ordered_json::array();

    for (const auto & val : vals) {
        arr.push_back(to_json(val));
    }

    return out;
}

common_json common_json::object() {
    common_json out;
    as_json(&out) = ordered_json::object();
    return out;
}

common_json common_json::object(std::initializer_list<common_json_item> items) {
    return common_json(items);
}

common_json common_json::make(const common_json_value & val) {
    return common_json(val);
}

bool common_json::is_null()           const { return as_json(this).is_null(); }
bool common_json::is_object()         const { return as_json(this).is_object(); }
bool common_json::is_array()          const { return as_json(this).is_array(); }
bool common_json::is_string()         const { return as_json(this).is_string(); }
bool common_json::is_boolean()        const { return as_json(this).is_boolean(); }
bool common_json::is_number()         const { return as_json(this).is_number(); }
bool common_json::is_number_integer() const { return as_json(this).is_number_integer(); }
bool common_json::is_number_float()   const { return as_json(this).is_number_float(); }

bool   common_json::empty() const { return as_json(this).empty(); }
size_t common_json::size()  const { return as_json(this).size(); }

bool common_json::contains(const std::string & key) const {
    return as_json(this).contains(key);
}

bool common_json::operator==(const common_json_value & val) const {
    // compare a tree in place, to_json() would copy it
    if (val.type == common_json_value::VAL_JSON) {
        return as_json(this) == as_json(val.val_json.get());
    }
    return as_json(this) == to_json(val);
}

bool common_json::operator!=(const common_json_value & val) const {
    return !(*this == val);
}

common_json       & common_json::at(const std::string & key)       { return guard([&]() -> common_json       & { return as_common(as_json(this).at(key)); }); }
const common_json & common_json::at(const std::string & key) const { return guard([&]() -> const common_json & { return as_common(as_json(this).at(key)); }); }
common_json       & common_json::at(size_t idx)                    { return guard([&]() -> common_json       & { return as_common(as_json(this).at(idx)); }); }
const common_json & common_json::at(size_t idx)              const { return guard([&]() -> const common_json & { return as_common(as_json(this).at(idx)); }); }

common_json       & common_json::operator[](const std::string & key)       { return guard([&]() -> common_json       & { return as_common(as_json(this)[key]); }); }
const common_json & common_json::operator[](const std::string & key) const { return guard([&]() -> const common_json & { return as_common(as_json(this).at(key)); }); }
common_json       & common_json::operator[](size_t idx)                    { return guard([&]() -> common_json       & { return as_common(as_json(this)[idx]); }); }
const common_json & common_json::operator[](size_t idx)              const { return guard([&]() -> const common_json & { return as_common(as_json(this).at(idx)); }); }

common_json       & common_json::front()       { return as_common(as_json(this).front()); }
const common_json & common_json::front() const { return as_common(as_json(this).front()); }
common_json       & common_json::back()        { return as_common(as_json(this).back()); }
const common_json & common_json::back()  const { return as_common(as_json(this).back()); }

void common_json::clear() {
    as_json(this).clear();
}

void common_json::erase(const std::string & key) {
    guard([&] { as_json(this).erase(key); });
}

void common_json::erase(size_t idx) {
    guard([&] { as_json(this).erase(idx); });
}

void common_json::assign(const common_json_value & val) {
    as_json(this) = to_json(val);
}

void common_json::set(const common_json_item & item) {
    guard([&] { as_json(this)[item.key] = to_json(item.val); });
}

void common_json::push_back(const common_json_value & val) {
    guard([&] { as_json(this).push_back(to_json(val)); });
}

void common_json::push_back(std::initializer_list<common_json_item> items) {
    common_json val(items);

    guard([&] { as_json(this).push_back(std::move(as_json(&val))); });
}

size_t common_json::count(const std::string & key) const {
    return as_json(this).count(key);
}

void common_json::insert(const common_json & vals) {
    guard([&] {
        ordered_json & self = as_json(this);

        self.insert(self.end(), as_json(&vals).begin(), as_json(&vals).end());
    });
}

std::string common_json::dump(int indent) const {
    return guard([&] { return as_json(this).dump(indent); });
}

std::string common_json::dump_safe(int indent) const {
    return as_json(this).dump(indent, ' ', false, ordered_json::error_handler_t::replace);
}

// an array is indexed directly, an object needs a walk from the start
common_json & common_json::iterator::operator*() const {
    return guard([&]() -> common_json & {
        ordered_json & j = as_json(node);

        if (j.is_object()) {
            return as_common(std::next(j.begin(), idx).value());
        }
        if (j.is_array()) {
            return as_common(j[idx]);
        }

        // a plain value gives itself once, same as the backing library
        return *node;
    });
}

std::string common_json::iterator::key() const {
    return guard([&] { return std::next(as_json(node).begin(), idx).key(); });
}

common_json::iterator common_json::begin() const {
    return iterator(const_cast<common_json *>(this), 0);
}

common_json::iterator common_json::end() const {
    return iterator(const_cast<common_json *>(this), size());
}

// the keys follow the backing library: the index for an array, "" for a plain value
common_json::items_view::entry common_json::items_view::iterator::operator*() const {
    return guard([&]() -> entry {
        ordered_json & j = as_json(node);

        if (j.is_object()) {
            auto it = std::next(j.begin(), idx);

            return { it.key(), as_common(it.value()) };
        }
        if (j.is_array()) {
            return { std::to_string(idx), as_common(j[idx]) };
        }

        return { std::string(), *node };
    });
}

common_json::items_view common_json::items() const {
    return items_view(const_cast<common_json *>(this), size());
}

// the backing library cannot build a common_json, so this one is just a copy
template <> common_json common_json::get<common_json>() const {
    return *this;
}

// get<T>() is usable only for the types below

#define COMMON_JSON_GET(...) template <> __VA_ARGS__ common_json::get<__VA_ARGS__>() const { return guard([&] { return as_json(this).get<__VA_ARGS__>(); }); }

COMMON_JSON_GET(bool)
COMMON_JSON_GET(int)
COMMON_JSON_GET(unsigned int)
COMMON_JSON_GET(long)
COMMON_JSON_GET(unsigned long)
COMMON_JSON_GET(long long)
COMMON_JSON_GET(unsigned long long)
COMMON_JSON_GET(float)
COMMON_JSON_GET(double)
COMMON_JSON_GET(std::string)
COMMON_JSON_GET(std::vector<float>)
COMMON_JSON_GET(std::vector<std::string>)
COMMON_JSON_GET(std::set<std::string>)
COMMON_JSON_GET(std::vector<int>)
COMMON_JSON_GET(std::vector<size_t>)
COMMON_JSON_GET(std::unordered_map<std::string, size_t>)

#undef COMMON_JSON_GET

// must stay below the get<std::string> specialization
common_json::operator std::string() const {
    return get<std::string>();
}

std::string common_json::value(const std::string & key, const char * def) const {
    return contains(key) ? at(key).get<std::string>() : std::string(def);
}
