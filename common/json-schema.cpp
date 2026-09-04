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

    // the target pairs an intersection is working through, to stop a recursive schema from recursing forever
    std::set<std::pair<const common_schema *, const common_schema *>> active_;

    // whether a value matches a schema, unknown where the check would need a regex
    enum match_t {
        MATCH_NO,
        MATCH_YES,
        MATCH_UNKNOWN,
    };

    static match_t both(match_t a, match_t b) {
        if (a == MATCH_NO || b == MATCH_NO) {
            return MATCH_NO;
        }
        if (a == MATCH_UNKNOWN || b == MATCH_UNKNOWN) {
            return MATCH_UNKNOWN;
        }
        return MATCH_YES;
    }

    template <typename T>
    static const T & as(const common_schema & node) {
        return static_cast<const T &>(node);
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
        out.reserve(nodes.size());
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

    static bool equal_all(const std::vector<common_schema_ptr> & a, const std::vector<common_schema_ptr> & b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); i++) {
            if (!equal(*a[i], *b[i])) {
                return false;
            }
        }
        return true;
    }

    // structural equality, a $ref only equals the same $ref
    static bool equal(const common_schema & a, const common_schema & b) {
        if (a.kind() != b.kind()) {
            return false;
        }
        switch (a.kind()) {
            case COMMON_SCHEMA_KIND_ANY:
            case COMMON_SCHEMA_KIND_NONE:
            case COMMON_SCHEMA_KIND_NULL:
            case COMMON_SCHEMA_KIND_BOOLEAN:
            case COMMON_SCHEMA_KIND_NUMBER:
                return true;
            case COMMON_SCHEMA_KIND_REF:
                return as<common_schema_ref>(a).ref == as<common_schema_ref>(b).ref;
            case COMMON_SCHEMA_KIND_CONST:
                return as<common_schema_const>(a).value == as<common_schema_const>(b).value;
            case COMMON_SCHEMA_KIND_ENUM: {
                const auto & va = as<common_schema_enum>(a).values;
                const auto & vb = as<common_schema_enum>(b).values;
                if (va.size() != vb.size()) {
                    return false;
                }
                for (size_t i = 0; i < va.size(); i++) {
                    if (va[i] != vb[i]) {
                        return false;
                    }
                }
                return true;
            }
            case COMMON_SCHEMA_KIND_INTEGER: {
                const auto & ia = as<common_schema_integer>(a);
                const auto & ib = as<common_schema_integer>(b);
                return ia.minimum == ib.minimum && ia.maximum == ib.maximum;
            }
            case COMMON_SCHEMA_KIND_STRING: {
                const auto & sa = as<common_schema_string>(a);
                const auto & sb = as<common_schema_string>(b);
                return sa.pattern == sb.pattern && sa.format == sb.format && sa.min_length == sb.min_length && sa.max_length == sb.max_length;
            }
            case COMMON_SCHEMA_KIND_ANY_OF:
                return equal_all(as<common_schema_any_of>(a).children, as<common_schema_any_of>(b).children);
            case COMMON_SCHEMA_KIND_ALL_OF:
                return equal_all(as<common_schema_all_of>(a).children, as<common_schema_all_of>(b).children);
            case COMMON_SCHEMA_KIND_ARRAY: {
                const auto & aa = as<common_schema_array>(a);
                const auto & ab = as<common_schema_array>(b);
                return aa.min_items == ab.min_items && aa.max_items == ab.max_items && equal(*aa.items, *ab.items);
            }
            case COMMON_SCHEMA_KIND_TUPLE:
                return equal_all(as<common_schema_tuple>(a).items, as<common_schema_tuple>(b).items);
            case COMMON_SCHEMA_KIND_OBJECT: {
                const auto & oa = as<common_schema_object>(a);
                const auto & ob = as<common_schema_object>(b);
                if (oa.properties.size() != ob.properties.size()) {
                    return false;
                }
                for (size_t i = 0; i < oa.properties.size(); i++) {
                    const auto & pa = oa.properties[i];
                    const auto & pb = ob.properties[i];
                    if (pa.name != pb.name || pa.required != pb.required || !equal(*pa.schema, *pb.schema)) {
                        return false;
                    }
                }
                if (!oa.additional_properties || !ob.additional_properties) {
                    return !oa.additional_properties && !ob.additional_properties;
                }
                return equal(*oa.additional_properties, *ob.additional_properties);
            }
        }
        return false;
    }
    static const common_schema_property * find_property(const common_schema_object & obj, const std::string & name) {
        for (const auto & prop : obj.properties) {
            if (prop.name == name) {
                return &prop;
            }
        }
        return nullptr;
    }

    // code points, the length the grammar's char rule counts
    static int utf8_length(const std::string & s) {
        int n = 0;
        for (unsigned char c : s) {
            if ((c & 0xC0) != 0x80) {
                n++;
            }
        }
        return n;
    }

    // Whether a value matches a schema, unknown where a pattern or format would have to be checked.
    match_t satisfies(const common_json & value, const common_schema & schema) const {
        const common_schema * node = resolve(schema);
        if (!node) {
            return MATCH_UNKNOWN;
        }
        switch (node->kind()) {
            case COMMON_SCHEMA_KIND_ANY:
                return MATCH_YES;
            case COMMON_SCHEMA_KIND_NONE:
            case COMMON_SCHEMA_KIND_REF:
                return MATCH_NO;
            case COMMON_SCHEMA_KIND_ANY_OF: {
                match_t res = MATCH_NO;
                for (const auto & child : as<common_schema_any_of>(*node).children) {
                    match_t m = satisfies(value, *child);
                    if (m == MATCH_YES) {
                        return MATCH_YES;
                    }
                    if (m == MATCH_UNKNOWN) {
                        res = MATCH_UNKNOWN;
                    }
                }
                return res;
            }
            case COMMON_SCHEMA_KIND_ALL_OF: {
                match_t res = MATCH_YES;
                for (const auto & child : as<common_schema_all_of>(*node).children) {
                    res = both(res, satisfies(value, *child));
                    if (res == MATCH_NO) {
                        return MATCH_NO;
                    }
                }
                return res;
            }
            case COMMON_SCHEMA_KIND_CONST:
                return value == as<common_schema_const>(*node).value ? MATCH_YES : MATCH_NO;
            case COMMON_SCHEMA_KIND_ENUM:
                for (const auto & v : as<common_schema_enum>(*node).values) {
                    if (value == v) {
                        return MATCH_YES;
                    }
                }
                return MATCH_NO;
            case COMMON_SCHEMA_KIND_NULL:
                return value.is_null() ? MATCH_YES : MATCH_NO;
            case COMMON_SCHEMA_KIND_BOOLEAN:
                return value.is_boolean() ? MATCH_YES : MATCH_NO;
            case COMMON_SCHEMA_KIND_NUMBER:
                return value.is_number() ? MATCH_YES : MATCH_NO;
            case COMMON_SCHEMA_KIND_INTEGER: {
                if (!value.is_number_integer()) {
                    return MATCH_NO;
                }
                const auto & i = as<common_schema_integer>(*node);
                int64_t v = value.get<int64_t>();
                return v >= i.minimum && v <= i.maximum ? MATCH_YES : MATCH_NO;
            }
            case COMMON_SCHEMA_KIND_STRING: {
                if (!value.is_string()) {
                    return MATCH_NO;
                }
                const auto & s = as<common_schema_string>(*node);
                int len = utf8_length(value.get<std::string>());
                if (len < s.min_length || (s.max_length >= 0 && len > s.max_length)) {
                    return MATCH_NO;
                }
                return s.pattern.empty() && s.format == COMMON_SCHEMA_FORMAT_NONE ? MATCH_YES : MATCH_UNKNOWN;
            }
            case COMMON_SCHEMA_KIND_ARRAY: {
                if (!value.is_array()) {
                    return MATCH_NO;
                }
                const auto & arr = as<common_schema_array>(*node);
                int n = (int) value.size();
                if (n < arr.min_items || (arr.max_items >= 0 && n > arr.max_items)) {
                    return MATCH_NO;
                }
                match_t res = MATCH_YES;
                for (const auto & item : value) {
                    res = both(res, satisfies(item, *arr.items));
                    if (res == MATCH_NO) {
                        return MATCH_NO;
                    }
                }
                return res;
            }
            case COMMON_SCHEMA_KIND_TUPLE: {
                const auto & tup = as<common_schema_tuple>(*node);
                if (!value.is_array() || value.size() != tup.items.size()) {
                    return MATCH_NO;
                }
                match_t res = MATCH_YES;
                for (size_t i = 0; i < tup.items.size(); i++) {
                    res = both(res, satisfies(value.at(i), *tup.items[i]));
                    if (res == MATCH_NO) {
                        return MATCH_NO;
                    }
                }
                return res;
            }
            case COMMON_SCHEMA_KIND_OBJECT: {
                if (!value.is_object()) {
                    return MATCH_NO;
                }
                const auto & obj = as<common_schema_object>(*node);
                match_t res = MATCH_YES;
                for (const auto & prop : obj.properties) {
                    if (value.contains(prop.name)) {
                        res = both(res, satisfies(value.at(prop.name), *prop.schema));
                    } else if (prop.required) {
                        return MATCH_NO;
                    }
                    if (res == MATCH_NO) {
                        return MATCH_NO;
                    }
                }
                for (const auto & [key, val] : value.items()) {
                    if (find_property(obj, key)) {
                        continue;
                    }
                    if (!obj.additional_properties) {
                        return MATCH_NO;
                    }
                    res = both(res, satisfies(val, *obj.additional_properties));
                    if (res == MATCH_NO) {
                        return MATCH_NO;
                    }
                }
                return res;
            }
        }
        return MATCH_UNKNOWN;
    }

    bool subsumes_all(const common_schema & a, const std::vector<common_schema_ptr> & items) const {
        for (const auto & item : items) {
            if (!subsumes(a, *item)) {
                return false;
            }
        }
        return true;
    }

    // Whether a accepts every value b accepts. False when unsure, so a $ref only subsumes the same $ref.
    bool subsumes(const common_schema & a, const common_schema & b) const {
        if (is(a, COMMON_SCHEMA_KIND_ANY) || is(b, COMMON_SCHEMA_KIND_NONE) || equal(a, b)) {
            return true;
        }
        if (is(a, COMMON_SCHEMA_KIND_REF) || is(b, COMMON_SCHEMA_KIND_REF)) {
            return false;
        }
        if (is(b, COMMON_SCHEMA_KIND_ANY_OF)) {
            return subsumes_all(a, as<common_schema_any_of>(b).children);
        }
        if (is(a, COMMON_SCHEMA_KIND_ANY_OF)) {
            for (const auto & child : as<common_schema_any_of>(a).children) {
                if (subsumes(*child, b)) {
                    return true;
                }
            }
            return false;
        }
        if (is(b, COMMON_SCHEMA_KIND_ALL_OF)) {
            // b is within each of its children, so within a once one of them is
            for (const auto & child : as<common_schema_all_of>(b).children) {
                if (subsumes(a, *child)) {
                    return true;
                }
            }
            return false;
        }
        if (is(a, COMMON_SCHEMA_KIND_ALL_OF)) {
            for (const auto & child : as<common_schema_all_of>(a).children) {
                if (!subsumes(*child, b)) {
                    return false;
                }
            }
            return true;
        }
        if (is(b, COMMON_SCHEMA_KIND_CONST)) {
            return satisfies(as<common_schema_const>(b).value, a) == MATCH_YES;
        }
        if (is(b, COMMON_SCHEMA_KIND_ENUM)) {
            for (const auto & v : as<common_schema_enum>(b).values) {
                if (satisfies(v, a) != MATCH_YES) {
                    return false;
                }
            }
            return true;
        }
        if (is(a, COMMON_SCHEMA_KIND_ARRAY) && is(b, COMMON_SCHEMA_KIND_TUPLE)) {
            const auto & arr = as<common_schema_array>(a);
            const auto & tup = as<common_schema_tuple>(b);
            int n = (int) tup.items.size();
            return n >= arr.min_items && (arr.max_items < 0 || n <= arr.max_items) && subsumes_all(*arr.items, tup.items);
        }
        if (a.kind() != b.kind()) {
            return is(a, COMMON_SCHEMA_KIND_NUMBER) && is(b, COMMON_SCHEMA_KIND_INTEGER);
        }
        switch (a.kind()) {
            case COMMON_SCHEMA_KIND_INTEGER: {
                const auto & ia = as<common_schema_integer>(a);
                const auto & ib = as<common_schema_integer>(b);
                return ia.minimum <= ib.minimum && ia.maximum >= ib.maximum;
            }
            case COMMON_SCHEMA_KIND_STRING: {
                const auto & sa = as<common_schema_string>(a);
                const auto & sb = as<common_schema_string>(b);
                return (sa.pattern.empty() || sa.pattern == sb.pattern) &&
                    (sa.format == COMMON_SCHEMA_FORMAT_NONE || sa.format == sb.format) &&
                    sa.min_length <= sb.min_length &&
                    (sa.max_length < 0 || (sb.max_length >= 0 && sa.max_length >= sb.max_length));
            }
            case COMMON_SCHEMA_KIND_ARRAY: {
                const auto & aa = as<common_schema_array>(a);
                const auto & ab = as<common_schema_array>(b);
                return aa.min_items <= ab.min_items &&
                    (aa.max_items < 0 || (ab.max_items >= 0 && aa.max_items >= ab.max_items)) &&
                    subsumes(*aa.items, *ab.items);
            }
            case COMMON_SCHEMA_KIND_TUPLE: {
                const auto & ta = as<common_schema_tuple>(a);
                const auto & tb = as<common_schema_tuple>(b);
                if (ta.items.size() != tb.items.size()) {
                    return false;
                }
                for (size_t i = 0; i < ta.items.size(); i++) {
                    if (!subsumes(*ta.items[i], *tb.items[i])) {
                        return false;
                    }
                }
                return true;
            }
            case COMMON_SCHEMA_KIND_OBJECT: {
                const auto & oa = as<common_schema_object>(a);
                const auto & ob = as<common_schema_object>(b);
                for (const auto & pb : ob.properties) {
                    const auto * pa = find_property(oa, pb.name);
                    if (pa) {
                        if ((pa->required && !pb.required) || !subsumes(*pa->schema, *pb.schema)) {
                            return false;
                        }
                    } else if (!oa.additional_properties || !subsumes(*oa.additional_properties, *pb.schema)) {
                        return false;
                    }
                }
                for (const auto & pa : oa.properties) {
                    if (find_property(ob, pa.name)) {
                        continue;
                    }
                    // b may leave it out, or fill it through its additionalProperties
                    if (pa.required || (ob.additional_properties && !subsumes(*pa.schema, *ob.additional_properties))) {
                        return false;
                    }
                }
                if (ob.additional_properties) {
                    return oa.additional_properties && subsumes(*oa.additional_properties, *ob.additional_properties);
                }
                return true;
            }
            default:
                return false;
        }
    }
    template <typename T>
    static T & as(common_schema & node) {
        return static_cast<T &>(node);
    }

    static void add_value(std::vector<common_json> & values, const common_json & value) {
        for (const auto & v : values) {
            if (v == value) {
                return;
            }
        }
        values.push_back(value);
    }

    static common_schema_ptr make_values(std::vector<common_json> values) {
        if (values.size() == 1) {
            return std::make_unique<common_schema_const>(std::move(values[0]));
        }
        auto node = std::make_unique<common_schema_enum>();
        node->values = std::move(values);
        return node;
    }

    // whether two ranges overlap or sit next to each other, so that their union is one range
    static bool touching(const common_schema_integer & a, const common_schema_integer & b) {
        if (a.minimum <= b.maximum && b.minimum <= a.maximum) {
            return true;
        }
        return (b.maximum < INT64_MAX && a.minimum == b.maximum + 1) || (a.maximum < INT64_MAX && b.minimum == a.maximum + 1);
    }

    // The union of rewritten alternatives.
    common_schema_ptr make_any_of(std::vector<common_schema_ptr> children) {
        std::vector<common_schema_ptr> flat;
        for (auto & child : children) {
            if (is(child, COMMON_SCHEMA_KIND_ANY_OF)) {
                for (auto & c : as<common_schema_any_of>(*child).children) {
                    flat.push_back(std::move(c));
                }
            } else {
                flat.push_back(std::move(child));
            }
        }

        // consts and enums become one enum, in order of first appearance
        std::vector<common_json>       values;
        std::vector<common_schema_ptr> out;
        for (auto & child : flat) {
            if (is(child, COMMON_SCHEMA_KIND_NONE)) {
                continue;
            }
            if (is(child, COMMON_SCHEMA_KIND_ANY)) {
                return std::make_unique<common_schema_any>();
            }
            if (is(child, COMMON_SCHEMA_KIND_CONST)) {
                add_value(values, as<common_schema_const>(*child).value);
                continue;
            }
            if (is(child, COMMON_SCHEMA_KIND_ENUM)) {
                for (const auto & v : as<common_schema_enum>(*child).values) {
                    add_value(values, v);
                }
                continue;
            }
            out.push_back(std::move(child));
        }

        // an integer range that overlaps or touches an earlier one widens it instead
        for (size_t i = 0; i < out.size();) {
            bool merged = false;
            if (is(out[i], COMMON_SCHEMA_KIND_INTEGER)) {
                auto & b = as<common_schema_integer>(*out[i]);
                for (size_t j = 0; j < i; j++) {
                    if (!is(out[j], COMMON_SCHEMA_KIND_INTEGER)) {
                        continue;
                    }
                    auto & a = as<common_schema_integer>(*out[j]);
                    if (touching(a, b)) {
                        a.minimum = std::min(a.minimum, b.minimum);
                        a.maximum = std::max(a.maximum, b.maximum);
                        merged    = true;
                        break;
                    }
                }
            }
            if (merged) {
                out.erase(out.begin() + i);
                i = 0;  // the widened range may now touch one it did not before
            } else {
                i++;
            }
        }

        // a value another alternative already accepts is dropped
        for (size_t i = 0; i < values.size();) {
            bool covered = false;
            for (const auto & child : out) {
                if (satisfies(values[i], *child) == MATCH_YES) {
                    covered = true;
                    break;
                }
            }
            if (covered) {
                values.erase(values.begin() + i);
            } else {
                i++;
            }
        }
        if (!values.empty()) {
            out.push_back(make_values(std::move(values)));
        }

        // an alternative within another is dropped, of two within each other the first stays
        std::vector<bool> dropped(out.size(), false);
        for (size_t i = 0; i < out.size(); i++) {
            for (size_t j = 0; j < out.size() && !dropped[i]; j++) {
                if (j == i || dropped[j] || !subsumes(*out[j], *out[i])) {
                    continue;
                }
                dropped[i] = j < i || !subsumes(*out[i], *out[j]);
            }
        }
        std::vector<common_schema_ptr> kept;
        for (size_t i = 0; i < out.size(); i++) {
            if (!dropped[i]) {
                kept.push_back(std::move(out[i]));
            }
        }

        if (kept.empty()) {
            return none();
        }
        if (kept.size() == 1) {
            return std::move(kept[0]);
        }
        auto node = std::make_unique<common_schema_any_of>();
        node->children = std::move(kept);
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
                    add_value(values, v);
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

    common_schema_ptr intersect_tuple(const common_schema_array & arr, const common_schema_tuple & tup) {
        int n = (int) tup.items.size();
        if (n < arr.min_items || (arr.max_items >= 0 && n > arr.max_items)) {
            return none();
        }
        auto node = std::make_unique<common_schema_tuple>();
        for (const auto & item : tup.items) {
            auto out = intersect(*arr.items, *item);
            if (is(out, COMMON_SCHEMA_KIND_NONE)) {
                return none();
            }
            node->items.push_back(std::move(out));
        }
        return node;
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
        if (equal(a, b)) {
            return clone(a);
        }
        if (is(a, COMMON_SCHEMA_KIND_REF) || is(b, COMMON_SCHEMA_KIND_REF)) {
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
        if (is(a, COMMON_SCHEMA_KIND_CONST) || is(b, COMMON_SCHEMA_KIND_CONST)) {
            const auto & c     = as<common_schema_const>(is(a, COMMON_SCHEMA_KIND_CONST) ? a : b);
            const auto & other = is(a, COMMON_SCHEMA_KIND_CONST) ? b : a;
            switch (satisfies(c.value, other)) {
                case MATCH_YES: return clone(c);
                case MATCH_NO:  return none();
                default:        return irreducible(a, b);
            }
        }
        if (is(a, COMMON_SCHEMA_KIND_ENUM) || is(b, COMMON_SCHEMA_KIND_ENUM)) {
            const auto & e     = as<common_schema_enum>(is(a, COMMON_SCHEMA_KIND_ENUM) ? a : b);
            const auto & other = is(a, COMMON_SCHEMA_KIND_ENUM) ? b : a;
            std::vector<common_json> values;
            for (const auto & v : e.values) {
                switch (satisfies(v, other)) {
                    case MATCH_YES: values.push_back(v); break;
                    case MATCH_NO:  break;
                    default:        return irreducible(a, b);
                }
            }
            return values.empty() ? none() : make_values(std::move(values));
        }
        if (a.kind() != b.kind()) {
            if (is(a, COMMON_SCHEMA_KIND_NUMBER) && is(b, COMMON_SCHEMA_KIND_INTEGER)) {
                return clone(b);
            }
            if (is(a, COMMON_SCHEMA_KIND_INTEGER) && is(b, COMMON_SCHEMA_KIND_NUMBER)) {
                return clone(a);
            }
            if (is(a, COMMON_SCHEMA_KIND_ARRAY) && is(b, COMMON_SCHEMA_KIND_TUPLE)) {
                return intersect_tuple(as<common_schema_array>(a), as<common_schema_tuple>(b));
            }
            if (is(a, COMMON_SCHEMA_KIND_TUPLE) && is(b, COMMON_SCHEMA_KIND_ARRAY)) {
                return intersect_tuple(as<common_schema_array>(b), as<common_schema_tuple>(a));
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
                // null, boolean and number have no fields, so unequal ones cannot be of the same kind
                return none();
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
