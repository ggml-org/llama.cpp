#include "json.h"

#include "ggml.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <iterator>
#include <vector>

using nlohmann::ordered_json;

// common_json_node is never defined, it only stands for an ordered_json in this file
static ordered_json & as_json(common_json_node * node) {
    return *reinterpret_cast<ordered_json *>(node);
}

static common_json_node * as_node(ordered_json * json) {
    return reinterpret_cast<common_json_node *>(json);
}

void common_json_node_deleter::operator()(common_json_node * node) const {
    delete reinterpret_cast<ordered_json *>(node);
}

static common_json make_json(const ordered_json & val) {
    common_json out;
    as_json(out.get_node()) = val;
    return out;
}

static ordered_json to_json(const common_json_value & val) {
    switch (val.type) {
        case common_json_value::VAL_NULL:   return nullptr;
        case common_json_value::VAL_BOOL:   return val.val_bool;
        case common_json_value::VAL_INT:    return val.val_int;
        case common_json_value::VAL_UINT:   return val.val_uint;
        case common_json_value::VAL_DOUBLE: return val.val_double;
        case common_json_value::VAL_STRING: return val.val_string;
        case common_json_value::VAL_JSON:   return as_json(val.val_json->get_node());
    }

    return nullptr;
}

template <typename T> T & common_json_raw(common_json_ref & json) {
    return as_json(json.get_node());
}

template <typename T> const T & common_json_raw(const common_json_ref & json) {
    return as_json(json.get_node());
}

template <typename T> common_json common_json_from_raw(const T & json) {
    return make_json(json);
}

template <typename T> common_json_ref common_json_ref_from_raw(T & json) {
    return common_json_ref(as_node(&json));
}

// the bridge is usable only for the type below
template ordered_json       & common_json_raw<ordered_json>(common_json_ref &);
template const ordered_json & common_json_raw<ordered_json>(const common_json_ref &);
template common_json          common_json_from_raw<ordered_json>(const ordered_json &);
template common_json_ref      common_json_ref_from_raw<ordered_json>(ordered_json &);

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

common_json_value::common_json_value(std::initializer_list<common_json_item> items) :
    type(VAL_JSON), val_json(std::make_shared<common_json>(items)) {}

common_json_value::common_json_value(const common_json_ref & val) :
    type(VAL_JSON), val_json(std::make_shared<common_json>(make_json(as_json(val.get_node())))) {}

bool common_json_ref::is_null()           const { return as_json(node).is_null(); }
bool common_json_ref::is_object()         const { return as_json(node).is_object(); }
bool common_json_ref::is_array()          const { return as_json(node).is_array(); }
bool common_json_ref::is_string()         const { return as_json(node).is_string(); }
bool common_json_ref::is_boolean()        const { return as_json(node).is_boolean(); }
bool common_json_ref::is_number()         const { return as_json(node).is_number(); }
bool common_json_ref::is_number_integer() const { return as_json(node).is_number_integer(); }
bool common_json_ref::is_number_float()   const { return as_json(node).is_number_float(); }

bool   common_json_ref::empty() const { return as_json(node).empty(); }
size_t common_json_ref::size()  const { return as_json(node).size(); }

bool common_json_ref::contains(const std::string & key) const {
    return as_json(node).contains(key);
}

bool common_json_ref::operator==(const common_json_value & val) const {
    return as_json(node) == to_json(val);
}

bool common_json_ref::operator!=(const common_json_value & val) const {
    return !(*this == val);
}

common_json_ref common_json_ref::at(const std::string & key) const {
    return common_json_ref(as_node(&as_json(node).at(key)));
}

common_json_ref common_json_ref::operator[](const std::string & key) const {
    return common_json_ref(as_node(&as_json(node)[key]));
}

common_json_ref common_json_ref::operator[](size_t idx) const {
    return common_json_ref(as_node(&as_json(node)[idx]));
}

void common_json_ref::assign(const common_json_value & val) {
    as_json(node) = to_json(val);
}

void common_json_ref::set(const common_json_item & item) {
    as_json(node)[item.key] = to_json(item.val);
}

void common_json_ref::push_back(const common_json_value & val) {
    as_json(node).push_back(to_json(val));
}

std::string common_json_ref::dump(int indent) const {
    return as_json(node).dump(indent);
}

// an array is indexed directly, an object needs a walk from the start
common_json_ref common_json_ref::iterator::operator*() const {
    if (as_json(node).is_object()) {
        return common_json_ref(as_node(&std::next(as_json(node).begin(), idx).value()));
    }

    return common_json_ref(as_node(&as_json(node)[idx]));
}

std::string common_json_ref::iterator::key() const {
    return std::next(as_json(node).begin(), idx).key();
}

std::pair<std::string, common_json_ref> common_json_ref::items_view::iterator::operator*() const {
    auto it = std::next(as_json(node).begin(), idx);

    return { it.key(), common_json_ref(as_node(&it.value())) };
}

common_json::common_json() :
    common_json_ref(nullptr), pimpl(as_node(new ordered_json(ordered_json::object()))) {
    node = pimpl.get();
}

common_json::common_json(std::initializer_list<common_json_item> items) : common_json() {
    for (const auto & item : items) {
        set(item);
    }
}

common_json::common_json(const common_json & other) :
    common_json_ref(nullptr), pimpl(as_node(new ordered_json(as_json(other.node)))) {
    node = pimpl.get();
}

common_json::common_json(common_json && other) noexcept :
    common_json_ref(other.node), pimpl(std::move(other.pimpl)) {
    other.node = nullptr;
}

common_json & common_json::operator=(const common_json & other) {
    as_json(node) = as_json(other.node);

    return *this;
}

common_json & common_json::operator=(common_json && other) noexcept {
    pimpl = std::move(other.pimpl);
    node  = pimpl.get();

    other.node = nullptr;

    return *this;
}

common_json::~common_json() = default;

common_json common_json::parse(const std::string & text) {
    try {
        return make_json(ordered_json::parse(text));
    } catch (const std::exception & e) {
        throw common_json_error(e.what());
    }
}

common_json common_json::array() {
    return make_json(ordered_json::array());
}

common_json common_json::array(std::initializer_list<common_json_value> vals) {
    ordered_json out = ordered_json::array();

    for (const auto & val : vals) {
        out.push_back(to_json(val));
    }

    return make_json(out);
}

common_json common_json::object() {
    return common_json();
}

common_json common_json::make(const common_json_value & val) {
    return make_json(to_json(val));
}

template <typename T> T common_json_ref::get() const {
    return as_json(node).get<T>();
}

// get<T>() is usable only for the types below

#define COMMON_JSON_GET(...) template __VA_ARGS__ common_json_ref::get<__VA_ARGS__>() const;

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
COMMON_JSON_GET(std::vector<std::string>)

#undef COMMON_JSON_GET
