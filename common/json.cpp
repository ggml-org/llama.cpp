#include "json.h"
// defines the shim
#include "json-shim.h"

#include "ggml.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <iterator>
#include <new>
#include <set>
#include <vector>

using nlohmann::ordered_json;

// a common_json is the backing value, so any value of a tree can be used as a common_json
static_assert(sizeof(ordered_json)  <= sizeof(common_json),  "common_json storage is too small");
static_assert(alignof(ordered_json) <= alignof(common_json), "common_json alignment is too weak");

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
        case common_json_value::VAL_JSON:   return as_json(val.val_json.get());
    }

    return nullptr;
}

template <typename T> T & common_json_raw(common_json & json) {
    return as_json(&json);
}

template <typename T> const T & common_json_raw(const common_json & json) {
    return as_json(&json);
}

template <typename T> common_json common_json_from_raw(const T & json) {
    return common_json(as_common(json));
}

template <typename T> common_json & common_json_ref_from_raw(T & json) {
    return as_common(json);
}

// the bridge is usable only for the type below
template ordered_json       & common_json_raw<ordered_json>(common_json &);
template const ordered_json & common_json_raw<ordered_json>(const common_json &);
template common_json          common_json_from_raw<ordered_json>(const ordered_json &);
template common_json        & common_json_ref_from_raw<ordered_json>(ordered_json &);

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

template <typename T>
common_json_value::common_json_value(const std::vector<T> & vals) : type(VAL_JSON) {
    common_json out = common_json::array();

    for (const auto & val : vals) {
        out.push_back(val);
    }

    val_json = std::make_shared<common_json>(std::move(out));
}

// a vector value is usable only for the types below
// note: std::vector<bool> is not here, its proxy reference does not convert
#define COMMON_JSON_VEC(...) template common_json_value::common_json_value(const std::vector<__VA_ARGS__> &);

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

// null, same as the backing library. operator[] turns it into an object,
// push_back() into an array
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
        return common_json_from_raw(ordered_json::parse(text));
    } catch (const std::exception & e) {
        throw common_json_error(e.what());
    }
}

common_json common_json::parse_no_throw(const std::string & text) {
    return common_json_from_raw(ordered_json::parse(text, nullptr, false));
}

bool common_json::is_discarded() const {
    return as_json(this).is_discarded();
}

common_json common_json::array() {
    return common_json_from_raw(ordered_json::array());
}

common_json common_json::array(std::initializer_list<common_json_value> vals) {
    ordered_json out = ordered_json::array();

    for (const auto & val : vals) {
        out.push_back(to_json(val));
    }

    return common_json_from_raw(out);
}

common_json common_json::object() {
    return common_json_from_raw(ordered_json::object());
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
    return as_json(this) == to_json(val);
}

bool common_json::operator!=(const common_json_value & val) const {
    return !(*this == val);
}

common_json       & common_json::at(const std::string & key)       { return as_common(as_json(this).at(key)); }
const common_json & common_json::at(const std::string & key) const { return as_common(as_json(this).at(key)); }
common_json       & common_json::at(size_t idx)                    { return as_common(as_json(this).at(idx)); }
const common_json & common_json::at(size_t idx)              const { return as_common(as_json(this).at(idx)); }

common_json       & common_json::operator[](const std::string & key)       { return as_common(as_json(this)[key]); }
const common_json & common_json::operator[](const std::string & key) const { return as_common(as_json(this).at(key)); }
common_json       & common_json::operator[](size_t idx)                    { return as_common(as_json(this)[idx]); }
const common_json & common_json::operator[](size_t idx)              const { return as_common(as_json(this).at(idx)); }

common_json       & common_json::front()       { return as_common(as_json(this).front()); }
const common_json & common_json::front() const { return as_common(as_json(this).front()); }
common_json       & common_json::back()        { return as_common(as_json(this).back()); }
const common_json & common_json::back()  const { return as_common(as_json(this).back()); }

void common_json::clear() {
    as_json(this).clear();
}

void common_json::erase(const std::string & key) {
    as_json(this).erase(key);
}

void common_json::erase(size_t idx) {
    as_json(this).erase(idx);
}

void common_json::assign(const common_json_value & val) {
    as_json(this) = to_json(val);
}

void common_json::set(const common_json_item & item) {
    as_json(this)[item.key] = to_json(item.val);
}

void common_json::push_back(const common_json_value & val) {
    as_json(this).push_back(to_json(val));
}

void common_json::push_back(std::initializer_list<common_json_item> items) {
    common_json val(items);

    as_json(this).push_back(as_json(&val));
}

size_t common_json::count(const std::string & key) const {
    return as_json(this).count(key);
}

void common_json::insert(const common_json & vals) {
    ordered_json & self = as_json(this);

    self.insert(self.end(), as_json(&vals).begin(), as_json(&vals).end());
}

std::string common_json::dump(int indent) const {
    return as_json(this).dump(indent);
}

std::string common_json::dump_safe(int indent) const {
    return as_json(this).dump(indent, ' ', false, ordered_json::error_handler_t::replace);
}

// an array is indexed directly, an object needs a walk from the start
common_json & common_json::iterator::operator*() const {
    if (as_json(node).is_object()) {
        return as_common(std::next(as_json(node).begin(), idx).value());
    }

    return as_common(as_json(node)[idx]);
}

std::string common_json::iterator::key() const {
    return std::next(as_json(node).begin(), idx).key();
}

common_json::iterator common_json::begin() const {
    return iterator(const_cast<common_json *>(this), 0);
}

common_json::iterator common_json::end() const {
    return iterator(const_cast<common_json *>(this), size());
}

common_json::items_view::entry common_json::items_view::iterator::operator*() const {
    auto it = std::next(as_json(node).begin(), idx);

    return { it.key(), as_common(it.value()) };
}

common_json::items_view common_json::items() const {
    return items_view(const_cast<common_json *>(this), size());
}

template <typename T> T common_json::get() const {
    return as_json(this).get<T>();
}

// the backing library cannot build a common_json, so this one is just a copy
template <> common_json common_json::get<common_json>() const {
    return *this;
}

// get<T>() is usable only for the types below

#define COMMON_JSON_GET(...) template __VA_ARGS__ common_json::get<__VA_ARGS__>() const;

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

#undef COMMON_JSON_GET
