#include "json-schema.h"
#include "common.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class common_schema_parser {
    const common_json &      root_;
    common_schema_document & doc_;

    // the targets parsed here, moved into doc_ once the whole schema parsed
    std::map<std::string, common_schema_ptr> refs_;

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
        if (doc_.refs.find(ref) == doc_.refs.end() && refs_.find(ref) == refs_.end()) {
            // reserve the key first, so that a cycle back to this $ref stops here
            refs_[ref] = nullptr;
            refs_[ref] = parse_schema(resolve_ref(ref, path), ref);
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
            // "items" wins when both are present; as in the converter, a schema instead of an array is the item schema
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
    common_schema_parser(const common_json & root, common_schema_document & doc) : root_(root), doc_(doc) {}

    common_schema_ptr parse() {
        auto node = parse_schema(root_, "#");
        for (auto & entry : refs_) {
            doc_.refs[entry.first] = std::move(entry.second);
        }
        for (auto * ref : pending_) {
            ref->target = doc_.refs.at(ref->ref).get();
        }
        return node;
    }
};

common_schema_document common_schema_parse(const common_json & schema) {
    common_schema_document doc;
    doc.root = common_schema_parser(schema, doc).parse();
    return doc;
}

common_schema_ptr common_schema_parse(const common_json & schema, common_schema_document & doc) {
    return common_schema_parser(schema, doc).parse();
}

static common_schema_kind kind_of(const common_json & value) {
    if (value.is_null()) {
        return COMMON_SCHEMA_KIND_NULL;
    }
    if (value.is_boolean()) {
        return COMMON_SCHEMA_KIND_BOOLEAN;
    }
    if (value.is_number_integer()) {
        return COMMON_SCHEMA_KIND_INTEGER;
    }
    if (value.is_number()) {
        return COMMON_SCHEMA_KIND_NUMBER;
    }
    if (value.is_string()) {
        return COMMON_SCHEMA_KIND_STRING;
    }
    if (value.is_array()) {
        return COMMON_SCHEMA_KIND_ARRAY;
    }
    return COMMON_SCHEMA_KIND_OBJECT;
}

static common_schema_kinds resolve_kinds(const common_schema & s, std::unordered_set<const common_schema *> & visited) {
    switch (s.kind()) {
        case COMMON_SCHEMA_KIND_ANY:
            return common_schema_kinds::all();
        case COMMON_SCHEMA_KIND_NUMBER:
            return { COMMON_SCHEMA_KIND_NUMBER, COMMON_SCHEMA_KIND_INTEGER };
        case COMMON_SCHEMA_KIND_TUPLE:
            return { COMMON_SCHEMA_KIND_ARRAY };
        case COMMON_SCHEMA_KIND_NULL:
        case COMMON_SCHEMA_KIND_BOOLEAN:
        case COMMON_SCHEMA_KIND_INTEGER:
        case COMMON_SCHEMA_KIND_STRING:
        case COMMON_SCHEMA_KIND_ARRAY:
        case COMMON_SCHEMA_KIND_OBJECT:
            return { s.kind() };
        case COMMON_SCHEMA_KIND_CONST:
            return { kind_of(static_cast<const common_schema_const &>(s).value) };
        case COMMON_SCHEMA_KIND_ENUM: {
            common_schema_kinds kinds;
            for (const auto & value : static_cast<const common_schema_enum &>(s).values) {
                kinds.add(kind_of(value));
            }
            return kinds;
        }
        case COMMON_SCHEMA_KIND_REF: {
            const auto * target = static_cast<const common_schema_ref &>(s).target;
            if (!target || !visited.insert(target).second) {
                // a cycle contributes no kind, to be safe
                return {};
            }
            auto kinds = resolve_kinds(*target, visited);
            visited.erase(target);
            return kinds;
        }
        case COMMON_SCHEMA_KIND_ANY_OF: {
            common_schema_kinds kinds;
            for (const auto & child : static_cast<const common_schema_any_of &>(s).children) {
                kinds |= resolve_kinds(*child, visited);
            }
            return kinds;
        }
        case COMMON_SCHEMA_KIND_ALL_OF: {
            auto kinds = common_schema_kinds::all();
            for (const auto & child : static_cast<const common_schema_all_of &>(s).children) {
                kinds &= resolve_kinds(*child, visited);
            }
            return kinds;
        }
    }
    return {};
}

common_schema_kinds common_schema::resolve_kinds() const {
    std::unordered_set<const common_schema *> visited;
    return ::resolve_kinds(*this, visited);
}

const char * common_schema_kind_name(common_schema_kind kind) {
    switch (kind) {
        case COMMON_SCHEMA_KIND_ANY:     return "any";
        case COMMON_SCHEMA_KIND_REF:     return "ref";
        case COMMON_SCHEMA_KIND_ANY_OF:  return "anyOf";
        case COMMON_SCHEMA_KIND_ALL_OF:  return "allOf";
        case COMMON_SCHEMA_KIND_CONST:   return "const";
        case COMMON_SCHEMA_KIND_ENUM:    return "enum";
        case COMMON_SCHEMA_KIND_NULL:    return "null";
        case COMMON_SCHEMA_KIND_BOOLEAN: return "boolean";
        case COMMON_SCHEMA_KIND_NUMBER:  return "number";
        case COMMON_SCHEMA_KIND_INTEGER: return "integer";
        case COMMON_SCHEMA_KIND_STRING:  return "string";
        case COMMON_SCHEMA_KIND_ARRAY:   return "array";
        case COMMON_SCHEMA_KIND_TUPLE:   return "tuple";
        case COMMON_SCHEMA_KIND_OBJECT:  return "object";
    }
    return "?";
}

static bool resolves_to_string(const common_schema & s, std::unordered_set<const common_schema *> & visited) {
    switch (s.kind()) {
        case COMMON_SCHEMA_KIND_STRING:
            return true;
        case COMMON_SCHEMA_KIND_CONST:
            return static_cast<const common_schema_const &>(s).value.is_string();
        case COMMON_SCHEMA_KIND_ENUM:
            for (const auto & v : static_cast<const common_schema_enum &>(s).values) {
                if (v.is_string()) {
                    return true;
                }
            }
            return false;
        case COMMON_SCHEMA_KIND_REF: {
            // a cycle is taken as not a string, to be safe
            const auto * target = static_cast<const common_schema_ref &>(s).target;
            return target && visited.insert(target).second && resolves_to_string(*target, visited);
        }
        case COMMON_SCHEMA_KIND_ANY_OF:
            for (const auto & child : static_cast<const common_schema_any_of &>(s).children) {
                if (resolves_to_string(*child, visited)) {
                    return true;
                }
            }
            return false;
        case COMMON_SCHEMA_KIND_ALL_OF: {
            // every child must allow a string, an any child constrains nothing
            bool any_string = false;
            for (const auto & child : static_cast<const common_schema_all_of &>(s).children) {
                if (child->kind() == COMMON_SCHEMA_KIND_ANY) {
                    continue;
                }
                if (!resolves_to_string(*child, visited)) {
                    return false;
                }
                any_string = true;
            }
            return any_string;
        }
        default:
            return false;
    }
}

bool common_schema::resolves_to_string() const {
    std::unordered_set<const common_schema *> visited;
    return ::resolves_to_string(*this, visited);
}
