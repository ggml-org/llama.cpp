#include "json-schema.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class common_schema_parser {
    const common_json &    root_;
    common_schema_document doc_;

    // ref nodes get their target once every $ref is parsed, a cycle would otherwise need it too early
    std::vector<common_schema_ref *> pending_;

    [[noreturn]] static void fail(const std::string & path, const std::string & msg) {
        throw std::runtime_error("JSON schema error at " + path + ": " + msg);
    }

    static int get_count(const common_json & schema, const std::string & key, const std::string & path, int def) {
        if (!schema.contains(key)) {
            return def;
        }
        const common_json & value = schema.at(key);
        if (!value.is_number_integer() || value.get<int>() < 0) {
            fail(path, key + " must be a non-negative integer");
        }
        return value.get<int>();
    }

    // a fractional bound is rounded inwards, towards the integers it still admits
    static int64_t get_bound(const common_json & schema, const std::string & key, const std::string & path, bool round_up) {
        const common_json & value = schema.at(key);
        if (value.is_number_integer()) {
            return value.get<int64_t>();
        }
        if (!value.is_number()) {
            fail(path, key + " must be a number");
        }
        double d = value.get<double>();
        return (int64_t) (round_up ? std::ceil(d) : std::floor(d));
    }

    static common_schema_format get_format(const common_json & schema, const std::string & path) {
        if (!schema.contains("format")) {
            return COMMON_SCHEMA_FORMAT_NONE;
        }
        const common_json & value = schema.at("format");
        if (!value.is_string()) {
            fail(path, "format must be a string");
        }
        std::string format = value.get<std::string>();
        if (format == "date") {
            return COMMON_SCHEMA_FORMAT_DATE;
        }
        if (format == "time") {
            return COMMON_SCHEMA_FORMAT_TIME;
        }
        if (format == "date-time") {
            return COMMON_SCHEMA_FORMAT_DATE_TIME;
        }
        if (format == "uuid" || (format.size() == 5 && format.compare(0, 4, "uuid") == 0 && format[4] >= '1' && format[4] <= '5')) {
            return COMMON_SCHEMA_FORMAT_UUID;
        }
        // any other format is a plain string
        return COMMON_SCHEMA_FORMAT_NONE;
    }

    const common_json & resolve_ref(const std::string & ref, const std::string & path) {
        const common_json * target = &root_;
        auto tokens = string_split(ref.substr(1), "/");
        for (size_t i = 1; i < tokens.size(); i++) {
            const std::string & sel = tokens[i];
            if (target->is_object() && target->contains(sel)) {
                target = &target->at(sel);
            } else if (target->is_array()) {
                size_t idx;
                try {
                    idx = std::stoull(sel);
                } catch (const std::logic_error &) {
                    idx = target->size();
                }
                if (idx >= target->size()) {
                    fail(path, "cannot resolve $ref " + ref + ", " + sel + " is out of range");
                }
                target = &target->at(idx);
            } else {
                fail(path, "cannot resolve $ref " + ref + ", " + sel + " not found");
            }
        }
        return *target;
    }

    common_schema_ptr parse_ref(const common_json & value, const std::string & path) {
        if (!value.is_string()) {
            fail(path, "$ref must be a string");
        }
        std::string ref = value.get<std::string>();
        if (ref.compare(0, 2, "#/") != 0) {
            fail(path, "unsupported $ref " + ref + ", only references into the same document are supported");
        }
        if (doc_.refs.find(ref) == doc_.refs.end()) {
            // reserve the key first, so that a cycle back to this $ref stops here
            doc_.refs[ref] = nullptr;
            doc_.refs[ref] = parse_schema(resolve_ref(ref, path), ref);
        }
        auto node = std::make_unique<common_schema_ref>(ref);
        pending_.push_back(node.get());
        return node;
    }

    template <typename T>
    common_schema_ptr parse_alternatives(const common_json & alts, const std::string & path) {
        if (!alts.is_array()) {
            fail(path, "must be an array of schemas");
        }
        if (alts.empty()) {
            fail(path, "must not be empty");
        }
        auto node = std::make_unique<T>();
        size_t i = 0;
        for (const auto & alt : alts) {
            node->children.push_back(parse_schema(alt, path + "/" + std::to_string(i++)));
        }
        return node;
    }

    common_schema_ptr parse_object(const common_json & schema, const std::string & path) {
        auto node = std::make_unique<common_schema_object>();

        std::unordered_set<std::string> required;
        if (schema.contains("required") && schema.at("required").is_array()) {
            for (const auto & name : schema.at("required")) {
                if (name.is_string()) {
                    required.insert(name.get<std::string>());
                }
            }
        }

        if (schema.contains("properties")) {
            const common_json & properties = schema.at("properties");
            if (!properties.is_object()) {
                fail(path, "properties must be an object");
            }
            for (const auto & [name, prop] : properties.items()) {
                node->properties.push_back({name, parse_schema(prop, path + "/properties/" + name), required.count(name) > 0});
            }
        }

        if (schema.contains("additionalProperties")) {
            const common_json & additional = schema.at("additionalProperties");
            if (additional.is_boolean()) {
                if (additional.get<bool>()) {
                    node->additional_properties = std::make_unique<common_schema_any>();
                }
            } else if (additional.is_object()) {
                node->additional_properties = parse_schema(additional, path + "/additionalProperties");
            } else {
                fail(path, "additionalProperties must be a boolean or a schema");
            }
        } else if (!schema.contains("properties")) {
            // {"type": "object"} on its own accepts any object
            node->additional_properties = std::make_unique<common_schema_any>();
        }

        return node;
    }

    common_schema_ptr parse_array(const common_json & schema, const std::string & path) {
        auto node = std::make_unique<common_schema_array>();
        if (schema.contains("items") || schema.contains("prefixItems")) {
            // "items" given as an array is the older spelling of "prefixItems", and wins when both are present
            const std::string key = schema.contains("items") ? "items" : "prefixItems";
            const common_json & items = schema.at(key);
            if (items.is_array()) {
                auto tuple = std::make_unique<common_schema_tuple>();
                size_t i = 0;
                for (const auto & item : items) {
                    tuple->items.push_back(parse_schema(item, path + "/" + key + "/" + std::to_string(i++)));
                }
                return tuple;
            }
            if (key == "prefixItems") {
                fail(path, "prefixItems must be an array");
            }
            node->items = parse_schema(items, path + "/" + key);
        } else {
            node->items = std::make_unique<common_schema_any>();
        }
        node->min_items = get_count(schema, "minItems", path, 0);
        node->max_items = get_count(schema, "maxItems", path, -1);
        return node;
    }

    common_schema_ptr parse_string(const common_json & schema, const std::string & path) {
        auto node = std::make_unique<common_schema_string>();
        if (schema.contains("pattern")) {
            const common_json & pattern = schema.at("pattern");
            if (!pattern.is_string()) {
                fail(path, "pattern must be a string");
            }
            node->pattern = pattern.get<std::string>();
        }
        node->format     = get_format(schema, path);
        node->min_length = get_count(schema, "minLength", path, 0);
        node->max_length = get_count(schema, "maxLength", path, -1);
        return node;
    }

    common_schema_ptr parse_integer(const common_json & schema, const std::string & path) {
        auto node = std::make_unique<common_schema_integer>();
        if (schema.contains("minimum")) {
            node->minimum = get_bound(schema, "minimum", path, /* round_up */ true);
        } else if (schema.contains("exclusiveMinimum")) {
            node->minimum = get_bound(schema, "exclusiveMinimum", path, /* round_up */ false) + 1;
        }
        if (schema.contains("maximum")) {
            node->maximum = get_bound(schema, "maximum", path, /* round_up */ false);
        } else if (schema.contains("exclusiveMaximum")) {
            node->maximum = get_bound(schema, "exclusiveMaximum", path, /* round_up */ true) - 1;
        }
        return node;
    }

    common_schema_ptr parse_schema(const common_json & schema, const std::string & path) {
        if (!schema.is_object()) {
            fail(path, "schema must be an object");
        }
        if (schema.contains("$ref")) {
            return parse_ref(schema.at("$ref"), path);
        }
        if (schema.contains("oneOf") || schema.contains("anyOf")) {
            const std::string key = schema.contains("oneOf") ? "oneOf" : "anyOf";
            return parse_alternatives<common_schema_any_of>(schema.at(key), path + "/" + key);
        }

        common_json type;
        if (schema.contains("type")) {
            type = schema.at("type");
        }
        if (type.is_array()) {
            // {"type": ["a", "b"], ...} is {"anyOf": [{"type": "a", ...}, {"type": "b", ...}]}
            if (type.empty()) {
                fail(path, "type must not be empty");
            }
            auto node = std::make_unique<common_schema_any_of>();
            size_t i = 0;
            for (const auto & t : type) {
                common_json alt = schema;
                alt["type"] = t;
                node->children.push_back(parse_schema(alt, path + "/type/" + std::to_string(i++)));
            }
            return node;
        }
        if (schema.contains("const")) {
            return std::make_unique<common_schema_const>(schema.at("const"));
        }
        if (schema.contains("enum")) {
            const common_json & values = schema.at("enum");
            if (!values.is_array() || values.empty()) {
                fail(path, "enum must be a non-empty array");
            }
            auto node = std::make_unique<common_schema_enum>();
            for (const auto & value : values) {
                node->values.push_back(value);
            }
            return node;
        }
        if (!type.is_null() && !type.is_string()) {
            fail(path, "type must be a string or an array of strings");
        }

        const std::string type_name = type.is_string() ? type.get<std::string>() : "";
        const bool has_properties = schema.contains("properties") ||
            (schema.contains("additionalProperties") && schema.at("additionalProperties") != true);

        if (type_name.empty()) {
            // without a type the structural keywords decide, in the same order as the converter
            if (has_properties) {
                return parse_object(schema, path);
            }
            if (schema.contains("allOf")) {
                return parse_alternatives<common_schema_all_of>(schema.at("allOf"), path + "/allOf");
            }
            if (schema.contains("items") || schema.contains("prefixItems")) {
                return parse_array(schema, path);
            }
            if (schema.contains("pattern") || get_format(schema, path) != COMMON_SCHEMA_FORMAT_NONE) {
                return parse_string(schema, path);
            }
            return std::make_unique<common_schema_any>();
        }
        if (type_name == "object") {
            if (!has_properties && schema.contains("allOf")) {
                return parse_alternatives<common_schema_all_of>(schema.at("allOf"), path + "/allOf");
            }
            return parse_object(schema, path);
        }
        if (type_name == "string") {
            if (schema.contains("allOf")) {
                return parse_alternatives<common_schema_all_of>(schema.at("allOf"), path + "/allOf");
            }
            return parse_string(schema, path);
        }
        if (type_name == "array") {
            return parse_array(schema, path);
        }
        if (type_name == "integer") {
            return parse_integer(schema, path);
        }
        if (type_name == "number") {
            return std::make_unique<common_schema_number>();
        }
        if (type_name == "boolean") {
            return std::make_unique<common_schema_boolean>();
        }
        if (type_name == "null") {
            return std::make_unique<common_schema_null>();
        }
        fail(path, "unrecognized type " + type_name);
    }

  public:
    explicit common_schema_parser(const common_json & root) : root_(root) {}

    common_schema_document parse() {
        doc_.root = parse_schema(root_, "#");
        for (auto * node : pending_) {
            node->target = doc_.refs.at(node->ref).get();
        }
        return std::move(doc_);
    }
};

common_schema_document common_schema_parse(const common_json & schema) {
    return common_schema_parser(schema).parse();
}

class common_schema_optimizer {
    common_schema_document & doc_;

    // set when a $ref got replaced by the none its target became, the pass is then repeated
    bool changed_ = false;

    // the target pairs being intersected further up, a recursive schema would otherwise never bottom out
    std::set<std::pair<const common_schema *, const common_schema *>> active_;

    template <typename T>
    static const T & as(const common_schema & node) {
        return static_cast<const T &>(node);
    }

    template <typename T>
    static T & as(common_schema & node) {
        return static_cast<T &>(node);
    }

    static bool is(const common_schema & node, common_schema_kind kind) {
        return node.kind() == kind;
    }

    static bool is(const common_schema_ptr & node, common_schema_kind kind) {
        return node && node->kind() == kind;
    }

    static common_schema_ptr none() {
        return std::make_unique<common_schema_none>();
    }

    // Follows a chain of $refs to a node of another kind.
    // Gives nullptr for a target that is being rewritten right now, or a chain that loops back on itself.
    const common_schema * resolve(const common_schema & node) const {
        const common_schema * cur = &node;
        for (size_t hops = 0; is(*cur, COMMON_SCHEMA_KIND_REF); hops++) {
            if (hops > doc_.refs.size()) {
                return nullptr;
            }
            auto it = doc_.refs.find(as<common_schema_ref>(*cur).ref);
            if (it == doc_.refs.end() || !it->second) {
                return nullptr;
            }
            cur = it->second.get();
        }
        return cur;
    }

    static std::vector<common_schema_ptr> clone_all(const std::vector<common_schema_ptr> & nodes) {
        std::vector<common_schema_ptr> out;
        for (const auto & node : nodes) {
            out.push_back(clone(*node));
        }
        return out;
    }

    static common_schema_ptr clone(const common_schema & node) {
        switch (node.kind()) {
            case COMMON_SCHEMA_KIND_ANY:     return std::make_unique<common_schema_any>();
            case COMMON_SCHEMA_KIND_NONE:    return none();
            case COMMON_SCHEMA_KIND_NULL:    return std::make_unique<common_schema_null>();
            case COMMON_SCHEMA_KIND_BOOLEAN: return std::make_unique<common_schema_boolean>();
            case COMMON_SCHEMA_KIND_NUMBER:  return std::make_unique<common_schema_number>();
            case COMMON_SCHEMA_KIND_INTEGER: return std::make_unique<common_schema_integer>(as<common_schema_integer>(node));
            case COMMON_SCHEMA_KIND_STRING:  return std::make_unique<common_schema_string>(as<common_schema_string>(node));
            case COMMON_SCHEMA_KIND_CONST:   return std::make_unique<common_schema_const>(as<common_schema_const>(node).value);
            case COMMON_SCHEMA_KIND_REF: {
                const auto & ref = as<common_schema_ref>(node);
                auto out = std::make_unique<common_schema_ref>(ref.ref);
                out->target = ref.target;
                return out;
            }
            case COMMON_SCHEMA_KIND_ENUM: {
                auto out = std::make_unique<common_schema_enum>();
                out->values = as<common_schema_enum>(node).values;
                return out;
            }
            case COMMON_SCHEMA_KIND_ANY_OF: {
                auto out = std::make_unique<common_schema_any_of>();
                out->children = clone_all(as<common_schema_any_of>(node).children);
                return out;
            }
            case COMMON_SCHEMA_KIND_ALL_OF: {
                auto out = std::make_unique<common_schema_all_of>();
                out->children = clone_all(as<common_schema_all_of>(node).children);
                return out;
            }
            case COMMON_SCHEMA_KIND_ARRAY: {
                const auto & arr = as<common_schema_array>(node);
                auto out = std::make_unique<common_schema_array>();
                out->items     = clone(*arr.items);
                out->min_items = arr.min_items;
                out->max_items = arr.max_items;
                return out;
            }
            case COMMON_SCHEMA_KIND_TUPLE: {
                auto out = std::make_unique<common_schema_tuple>();
                out->items = clone_all(as<common_schema_tuple>(node).items);
                return out;
            }
            case COMMON_SCHEMA_KIND_OBJECT: {
                const auto & obj = as<common_schema_object>(node);
                auto out = std::make_unique<common_schema_object>();
                for (const auto & prop : obj.properties) {
                    out->properties.push_back({prop.name, clone(*prop.schema), prop.required});
                }
                if (obj.additional_properties) {
                    out->additional_properties = clone(*obj.additional_properties);
                }
                return out;
            }
        }
        return none();
    }

    static bool contains(const std::vector<common_json> & values, const common_json & value) {
        return std::any_of(values.begin(), values.end(), [&](const common_json & v) { return v == value; });
    }

    // the values a const or enum accepts
    static bool get_values(const common_schema & node, std::vector<common_json> & values) {
        if (is(node, COMMON_SCHEMA_KIND_CONST)) {
            values.push_back(as<common_schema_const>(node).value);
            return true;
        }
        if (is(node, COMMON_SCHEMA_KIND_ENUM)) {
            values = as<common_schema_enum>(node).values;
            return true;
        }
        return false;
    }

    // a const for one value, none for no value at all
    static common_schema_ptr make_values(std::vector<common_json> values) {
        if (values.empty()) {
            return none();
        }
        if (values.size() == 1) {
            return std::make_unique<common_schema_const>(std::move(values[0]));
        }
        auto node = std::make_unique<common_schema_enum>();
        node->values = std::move(values);
        return node;
    }

    // The union of rewritten alternatives.
    static common_schema_ptr make_any_of(std::vector<common_schema_ptr> children) {
        std::vector<common_schema_ptr> flat;
        for (auto & child : children) {
            if (is(child, COMMON_SCHEMA_KIND_ANY_OF)) {
                // a nested one is flat already
                for (auto & c : as<common_schema_any_of>(*child).children) {
                    flat.push_back(std::move(c));
                }
            } else {
                flat.push_back(std::move(child));
            }
        }
        std::vector<common_schema_ptr> out;
        for (auto & child : flat) {
            if (is(child, COMMON_SCHEMA_KIND_ANY)) {
                return std::make_unique<common_schema_any>();
            }
            if (!is(child, COMMON_SCHEMA_KIND_NONE)) {
                out.push_back(std::move(child));
            }
        }
        if (out.empty()) {
            return none();
        }
        if (out.size() == 1) {
            return std::move(out[0]);
        }
        auto node = std::make_unique<common_schema_any_of>();
        node->children = std::move(out);
        return node;
    }

    // An array with rewritten items.
    static common_schema_ptr make_array(common_schema_ptr items, int min_items, int max_items) {
        if ((max_items >= 0 && min_items > max_items) || (is(items, COMMON_SCHEMA_KIND_NONE) && min_items > 0)) {
            return none();
        }
        if (max_items == 0 || is(items, COMMON_SCHEMA_KIND_NONE)) {
            // only [] is left
            return std::make_unique<common_schema_tuple>();
        }
        auto node = std::make_unique<common_schema_array>();
        node->items     = std::move(items);
        node->min_items = min_items;
        node->max_items = max_items;
        return node;
    }

    // An object with rewritten property schemas.
    static common_schema_ptr make_object(std::vector<common_schema_property> properties, common_schema_ptr additional) {
        auto node = std::make_unique<common_schema_object>();
        for (auto & prop : properties) {
            if (is(prop.schema, COMMON_SCHEMA_KIND_NONE)) {
                // it can never be present
                if (prop.required) {
                    return none();
                }
                continue;
            }
            node->properties.push_back(std::move(prop));
        }
        if (additional && !is(additional, COMMON_SCHEMA_KIND_NONE)) {
            node->additional_properties = std::move(additional);
        }
        return node;
    }

    std::vector<common_schema_ptr> rewrite_all(std::vector<common_schema_ptr> nodes) {
        for (auto & node : nodes) {
            node = rewrite(std::move(node));
        }
        return nodes;
    }

    // Rewrites a node bottom-up.
    common_schema_ptr rewrite(common_schema_ptr node) {
        switch (node->kind()) {
            case COMMON_SCHEMA_KIND_ANY:
            case COMMON_SCHEMA_KIND_NONE:
            case COMMON_SCHEMA_KIND_CONST:
            case COMMON_SCHEMA_KIND_NULL:
            case COMMON_SCHEMA_KIND_BOOLEAN:
            case COMMON_SCHEMA_KIND_NUMBER:
                return node;
            case COMMON_SCHEMA_KIND_REF: {
                const common_schema * target = resolve(*node);
                if (target && is(*target, COMMON_SCHEMA_KIND_NONE)) {
                    changed_ = true;
                    return none();
                }
                return node;
            }
            case COMMON_SCHEMA_KIND_ENUM: {
                std::vector<common_json> values;
                for (const auto & v : as<common_schema_enum>(*node).values) {
                    if (!contains(values, v)) {
                        values.push_back(v);
                    }
                }
                return make_values(std::move(values));
            }
            case COMMON_SCHEMA_KIND_INTEGER: {
                const auto & i = as<common_schema_integer>(*node);
                return i.minimum > i.maximum ? none() : std::move(node);
            }
            case COMMON_SCHEMA_KIND_STRING: {
                const auto & s = as<common_schema_string>(*node);
                return s.max_length >= 0 && s.min_length > s.max_length ? none() : std::move(node);
            }
            case COMMON_SCHEMA_KIND_ANY_OF:
                return make_any_of(rewrite_all(std::move(as<common_schema_any_of>(*node).children)));
            case COMMON_SCHEMA_KIND_ALL_OF:
                return intersect_all(rewrite_all(std::move(as<common_schema_all_of>(*node).children)));
            case COMMON_SCHEMA_KIND_ARRAY: {
                auto & arr = as<common_schema_array>(*node);
                return make_array(rewrite(std::move(arr.items)), arr.min_items, arr.max_items);
            }
            case COMMON_SCHEMA_KIND_TUPLE: {
                auto & tup = as<common_schema_tuple>(*node);
                tup.items  = rewrite_all(std::move(tup.items));
                for (const auto & item : tup.items) {
                    if (is(item, COMMON_SCHEMA_KIND_NONE)) {
                        return none();
                    }
                }
                return node;
            }
            case COMMON_SCHEMA_KIND_OBJECT: {
                auto & obj = as<common_schema_object>(*node);
                for (auto & prop : obj.properties) {
                    prop.schema = rewrite(std::move(prop.schema));
                }
                if (obj.additional_properties) {
                    obj.additional_properties = rewrite(std::move(obj.additional_properties));
                }
                return make_object(std::move(obj.properties), std::move(obj.additional_properties));
            }
        }
        return node;
    }

    static common_schema_ptr make_all_of(std::vector<common_schema_ptr> children) {
        if (children.size() == 1) {
            return std::move(children[0]);
        }
        auto node = std::make_unique<common_schema_all_of>();
        node->children = std::move(children);
        return node;
    }

    // what is left when two nodes cannot be combined
    static common_schema_ptr irreducible(const common_schema & a, const common_schema & b) {
        std::vector<common_schema_ptr> children;
        children.push_back(clone(a));
        children.push_back(clone(b));
        return make_all_of(std::move(children));
    }

    // The intersection of rewritten nodes, an allOf of those that could not be combined.
    common_schema_ptr intersect_all(std::vector<common_schema_ptr> children) {
        std::vector<common_schema_ptr> flat;
        for (auto & child : children) {
            if (is(child, COMMON_SCHEMA_KIND_ALL_OF)) {
                for (auto & c : as<common_schema_all_of>(*child).children) {
                    flat.push_back(std::move(c));
                }
            } else {
                flat.push_back(std::move(child));
            }
        }

        std::vector<common_schema_ptr> residual;
        for (auto & child : flat) {
            bool merged = false;
            for (auto & r : residual) {
                auto cand = intersect(*r, *child);
                if (is(cand, COMMON_SCHEMA_KIND_NONE)) {
                    return none();
                }
                if (!is(cand, COMMON_SCHEMA_KIND_ALL_OF)) {
                    r      = std::move(cand);
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                residual.push_back(std::move(child));
            }
        }
        if (residual.empty()) {
            return std::make_unique<common_schema_any>();
        }
        return make_all_of(std::move(residual));
    }

    static const common_schema_property * find_property(const common_schema_object & obj, const std::string & name) {
        for (const auto & prop : obj.properties) {
            if (prop.name == name) {
                return &prop;
            }
        }
        return nullptr;
    }

    // Properties are merged the way json_schema_to_grammar() merges an allOf: a property the other
    // side does not list survives, a side that is closed does not shut it out.
    common_schema_ptr intersect_object(const common_schema_object & oa, const common_schema_object & ob) {
        std::vector<common_schema_property> properties;
        auto add = [&](const common_schema_object & x, const common_schema_object & y, bool skip_shared) {
            for (const auto & px : x.properties) {
                const auto * py = find_property(y, px.name);
                if (py && skip_shared) {
                    continue;
                }
                common_schema_property prop;
                prop.name     = px.name;
                prop.required = px.required || (py && py->required);
                if (py) {
                    prop.schema = intersect(*px.schema, *py->schema);
                } else if (y.additional_properties) {
                    prop.schema = intersect(*px.schema, *y.additional_properties);
                } else {
                    prop.schema = clone(*px.schema);
                }
                properties.push_back(std::move(prop));
            }
        };
        add(oa, ob, false);
        add(ob, oa, true);

        common_schema_ptr additional;
        if (oa.additional_properties && ob.additional_properties) {
            additional = intersect(*oa.additional_properties, *ob.additional_properties);
        }
        return make_object(std::move(properties), std::move(additional));
    }

    // The intersection of two rewritten nodes, an allOf of both where they cannot be combined.
    common_schema_ptr intersect(const common_schema & a, const common_schema & b) {
        if (is(a, COMMON_SCHEMA_KIND_ANY)) {
            return clone(b);
        }
        if (is(b, COMMON_SCHEMA_KIND_ANY)) {
            return clone(a);
        }
        if (is(a, COMMON_SCHEMA_KIND_NONE) || is(b, COMMON_SCHEMA_KIND_NONE)) {
            return none();
        }
        if (is(a, COMMON_SCHEMA_KIND_REF) || is(b, COMMON_SCHEMA_KIND_REF)) {
            if (is(a, COMMON_SCHEMA_KIND_REF) && is(b, COMMON_SCHEMA_KIND_REF) && as<common_schema_ref>(a).ref == as<common_schema_ref>(b).ref) {
                return clone(a);
            }
            const common_schema * ta = resolve(a);
            const common_schema * tb = resolve(b);
            if (!ta || !tb) {
                return irreducible(a, b);
            }
            auto key = std::make_pair(ta, tb);
            if (!active_.insert(key).second) {
                // the same pair is already being intersected further up, a recursive schema
                return irreducible(a, b);
            }
            auto out = intersect(*ta, *tb);
            active_.erase(key);
            return out;
        }
        if (is(a, COMMON_SCHEMA_KIND_ANY_OF) || is(b, COMMON_SCHEMA_KIND_ANY_OF)) {
            // (a1 | a2) & b = (a1 & b) | (a2 & b)
            std::vector<common_schema_ptr> alts;
            if (is(a, COMMON_SCHEMA_KIND_ANY_OF)) {
                for (const auto & child : as<common_schema_any_of>(a).children) {
                    alts.push_back(intersect(*child, b));
                }
            } else {
                for (const auto & child : as<common_schema_any_of>(b).children) {
                    alts.push_back(intersect(a, *child));
                }
            }
            return make_any_of(std::move(alts));
        }
        if (is(a, COMMON_SCHEMA_KIND_ALL_OF) || is(b, COMMON_SCHEMA_KIND_ALL_OF)) {
            std::vector<common_schema_ptr> parts;
            for (const auto * node : {&a, &b}) {
                if (is(*node, COMMON_SCHEMA_KIND_ALL_OF)) {
                    for (const auto & child : as<common_schema_all_of>(*node).children) {
                        parts.push_back(clone(*child));
                    }
                } else {
                    parts.push_back(clone(*node));
                }
            }
            return intersect_all(std::move(parts));
        }

        std::vector<common_json> va;
        std::vector<common_json> vb;
        bool values_a = get_values(a, va);
        bool values_b = get_values(b, vb);
        if (values_a && values_b) {
            std::vector<common_json> common;
            for (const auto & v : va) {
                if (contains(vb, v)) {
                    common.push_back(v);
                }
            }
            return make_values(std::move(common));
        }
        if (values_a || values_b) {
            // whether a value fits a schema is not checked
            return irreducible(a, b);
        }

        if (a.kind() != b.kind()) {
            if (is(a, COMMON_SCHEMA_KIND_NUMBER) && is(b, COMMON_SCHEMA_KIND_INTEGER)) {
                return clone(b);
            }
            if (is(a, COMMON_SCHEMA_KIND_INTEGER) && is(b, COMMON_SCHEMA_KIND_NUMBER)) {
                return clone(a);
            }
            if ((is(a, COMMON_SCHEMA_KIND_ARRAY) && is(b, COMMON_SCHEMA_KIND_TUPLE)) || (is(a, COMMON_SCHEMA_KIND_TUPLE) && is(b, COMMON_SCHEMA_KIND_ARRAY))) {
                return irreducible(a, b);
            }
            return none();
        }
        switch (a.kind()) {
            case COMMON_SCHEMA_KIND_INTEGER: {
                const auto & ia = as<common_schema_integer>(a);
                const auto & ib = as<common_schema_integer>(b);
                auto node = std::make_unique<common_schema_integer>();
                node->minimum = std::max(ia.minimum, ib.minimum);
                node->maximum = std::min(ia.maximum, ib.maximum);
                return node->minimum > node->maximum ? none() : std::move(node);
            }
            case COMMON_SCHEMA_KIND_STRING: {
                const auto & sa = as<common_schema_string>(a);
                const auto & sb = as<common_schema_string>(b);
                if (!sa.pattern.empty() && !sb.pattern.empty() && sa.pattern != sb.pattern) {
                    return irreducible(a, b);
                }
                if (sa.format != COMMON_SCHEMA_FORMAT_NONE && sb.format != COMMON_SCHEMA_FORMAT_NONE && sa.format != sb.format) {
                    return none();
                }
                auto node = std::make_unique<common_schema_string>();
                node->pattern    = sa.pattern.empty() ? sb.pattern : sa.pattern;
                node->format     = sa.format == COMMON_SCHEMA_FORMAT_NONE ? sb.format : sa.format;
                node->min_length = std::max(sa.min_length, sb.min_length);
                node->max_length = sa.max_length < 0 ? sb.max_length : sb.max_length < 0 ? sa.max_length : std::min(sa.max_length, sb.max_length);
                return node->max_length >= 0 && node->min_length > node->max_length ? none() : std::move(node);
            }
            case COMMON_SCHEMA_KIND_ARRAY: {
                const auto & aa = as<common_schema_array>(a);
                const auto & ab = as<common_schema_array>(b);
                int min_items = std::max(aa.min_items, ab.min_items);
                int max_items = aa.max_items < 0 ? ab.max_items : ab.max_items < 0 ? aa.max_items : std::min(aa.max_items, ab.max_items);
                return make_array(intersect(*aa.items, *ab.items), min_items, max_items);
            }
            case COMMON_SCHEMA_KIND_TUPLE: {
                const auto & ta = as<common_schema_tuple>(a);
                const auto & tb = as<common_schema_tuple>(b);
                if (ta.items.size() != tb.items.size()) {
                    return none();
                }
                auto node = std::make_unique<common_schema_tuple>();
                for (size_t i = 0; i < ta.items.size(); i++) {
                    auto item = intersect(*ta.items[i], *tb.items[i]);
                    if (is(item, COMMON_SCHEMA_KIND_NONE)) {
                        return none();
                    }
                    node->items.push_back(std::move(item));
                }
                return node;
            }
            case COMMON_SCHEMA_KIND_OBJECT:
                return intersect_object(as<common_schema_object>(a), as<common_schema_object>(b));
            default:
                // null, boolean and number have no fields to differ in
                return clone(a);
        }
    }

    // Moves the refs the node reaches into live, and points each ref node at its target there.
    void link(common_schema & node, std::map<std::string, common_schema_ptr> & live) {
        switch (node.kind()) {
            case COMMON_SCHEMA_KIND_REF: {
                auto & ref = as<common_schema_ref>(node);
                auto it = live.find(ref.ref);
                if (it == live.end()) {
                    it = live.emplace(ref.ref, std::move(doc_.refs.at(ref.ref))).first;
                    link(*it->second, live);
                }
                ref.target = it->second.get();
                return;
            }
            case COMMON_SCHEMA_KIND_ANY_OF:
                for (auto & child : as<common_schema_any_of>(node).children) {
                    link(*child, live);
                }
                return;
            case COMMON_SCHEMA_KIND_ALL_OF:
                for (auto & child : as<common_schema_all_of>(node).children) {
                    link(*child, live);
                }
                return;
            case COMMON_SCHEMA_KIND_ARRAY:
                link(*as<common_schema_array>(node).items, live);
                return;
            case COMMON_SCHEMA_KIND_TUPLE:
                for (auto & item : as<common_schema_tuple>(node).items) {
                    link(*item, live);
                }
                return;
            case COMMON_SCHEMA_KIND_OBJECT: {
                auto & obj = as<common_schema_object>(node);
                for (auto & prop : obj.properties) {
                    link(*prop.schema, live);
                }
                if (obj.additional_properties) {
                    link(*obj.additional_properties, live);
                }
                return;
            }
            default:
                return;
        }
    }

  public:
    explicit common_schema_optimizer(common_schema_document & doc) : doc_(doc) {}

    void run() {
        // a $ref whose target became none is none too, which the next pass can then prune in its parent
        do {
            changed_ = false;
            for (auto & entry : doc_.refs) {
                entry.second = rewrite(std::move(entry.second));
            }
            doc_.root = rewrite(std::move(doc_.root));
        } while (changed_);

        // only the refs the root still reaches are kept, and every ref node gets its new target
        std::map<std::string, common_schema_ptr> live;
        link(*doc_.root, live);
        doc_.refs = std::move(live);
    }
};

void common_schema_optimize(common_schema_document & doc) {
    common_schema_optimizer(doc).run();
}
