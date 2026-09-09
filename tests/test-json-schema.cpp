#include "json-schema.h"
#include "json.h"
#include "testing.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

static common_schema_document parse(const std::string & schema) {
    return common_schema_parse(common_json::parse(schema));
}

// the node as T, aborting the current test when it is some other kind
template <typename T>
static const T & as(testing & t, const common_schema * node, const std::string & what) {
    const T * typed = dynamic_cast<const T *>(node);
    if (!t.assert_true(what + " has the expected kind", typed != nullptr)) {
        throw std::runtime_error(what + " has the wrong kind");
    }
    return *typed;
}

template <typename T>
static const T & root(testing & t, const common_schema_document & doc) {
    return as<T>(t, doc.root.get(), "root");
}

static std::string dump(const common_schema & node);

static std::string dump_all(const std::vector<common_schema_ptr> & nodes) {
    std::string out;
    for (const auto & node : nodes) {
        out += (out.empty() ? "" : ", ") + dump(*node);
    }
    return out;
}

// a bound that is left out when it is the default
static std::string dump_range(int64_t min, int64_t min_def, int64_t max, int64_t max_def) {
    if (min == min_def && max == max_def) {
        return "";
    }
    return "[" + (min == min_def ? "" : std::to_string(min)) + ".." + (max == max_def ? "" : std::to_string(max)) + "]";
}

// one line per node, e.g. object{a: string, b?: integer[1..], *: any}
static std::string dump(const common_schema & node) {
    switch (node.kind()) {
        case COMMON_SCHEMA_KIND_ANY:     return "any";
        case COMMON_SCHEMA_KIND_NULL:    return "null";
        case COMMON_SCHEMA_KIND_BOOLEAN: return "boolean";
        case COMMON_SCHEMA_KIND_NUMBER:  return "number";
        case COMMON_SCHEMA_KIND_REF:     return "ref(" + static_cast<const common_schema_ref &>(node).ref + ")";
        case COMMON_SCHEMA_KIND_ANY_OF:  return "anyOf(" + dump_all(static_cast<const common_schema_any_of &>(node).children) + ")";
        case COMMON_SCHEMA_KIND_ALL_OF:  return "allOf(" + dump_all(static_cast<const common_schema_all_of &>(node).children) + ")";
        case COMMON_SCHEMA_KIND_CONST:   return "const(" + static_cast<const common_schema_const &>(node).value.dump() + ")";
        case COMMON_SCHEMA_KIND_TUPLE:   return "tuple(" + dump_all(static_cast<const common_schema_tuple &>(node).items) + ")";
        case COMMON_SCHEMA_KIND_ENUM: {
            std::string out;
            for (const auto & v : static_cast<const common_schema_enum &>(node).values) {
                out += (out.empty() ? "" : ", ") + v.dump();
            }
            return "enum(" + out + ")";
        }
        case COMMON_SCHEMA_KIND_INTEGER: {
            const auto & i = static_cast<const common_schema_integer &>(node);
            return "integer" + dump_range(i.minimum, INT64_MIN, i.maximum, INT64_MAX);
        }
        case COMMON_SCHEMA_KIND_STRING: {
            const auto & s = static_cast<const common_schema_string &>(node);
            static const char * formats[] = {"", "uuid", "date", "time", "date-time"};
            std::string out = "string";
            if (!s.pattern.empty()) {
                out += "/" + s.pattern + "/";
            }
            if (s.format != COMMON_SCHEMA_FORMAT_NONE) {
                out += std::string(":") + formats[s.format];
            }
            return out + dump_range(s.min_length, 0, s.max_length, -1);
        }
        case COMMON_SCHEMA_KIND_ARRAY: {
            const auto & a = static_cast<const common_schema_array &>(node);
            return "array(" + dump(*a.items) + ")" + dump_range(a.min_items, 0, a.max_items, -1);
        }
        case COMMON_SCHEMA_KIND_OBJECT: {
            const auto & o = static_cast<const common_schema_object &>(node);
            std::string out;
            for (const auto & p : o.properties) {
                out += (out.empty() ? "" : ", ") + p.name + (p.required ? ": " : "?: ") + dump(*p.schema);
            }
            if (o.additional_properties) {
                out += (out.empty() ? "" : ", ") + std::string("*: ") + dump(*o.additional_properties);
            }
            return "object{" + out + "}";
        }
    }
    return "?";
}

static void assert_error(testing & t, const std::string & schema, const std::string & needle) {
    try {
        parse(schema);
        t.assert_true(schema + " is rejected", false);
    } catch (const std::runtime_error & e) {
        std::string what = e.what();
        t.assert_true(schema + " -> " + what, what.find(needle) != std::string::npos);
    }
}

static void test_any(testing & t) {
    t.test("empty schema", [](testing & t) {
        auto doc = parse("{}");
        root<common_schema_any>(t, doc);
        t.assert_true("no refs", doc.refs.empty());
    });

    t.test("unrecognized keywords only", [](testing & t) {
        auto doc = parse(R"({"description": "x", "title": "y", "default": 1})");
        root<common_schema_any>(t, doc);
    });

    t.test("unknown format without a type", [](testing & t) {
        auto doc = parse(R"({"format": "email"})");
        root<common_schema_any>(t, doc);
    });

    t.test("length keywords do not imply a string", [](testing & t) {
        auto doc = parse(R"({"minLength": 3, "maxLength": 5})");
        root<common_schema_any>(t, doc);
    });

    t.test("additionalProperties true alone", [](testing & t) {
        auto doc = parse(R"({"additionalProperties": true})");
        root<common_schema_any>(t, doc);
    });
}

static void test_primitives(testing & t) {
    t.test("null", [](testing & t) {
        auto doc = parse(R"({"type": "null"})");
        root<common_schema_null>(t, doc);
    });

    t.test("boolean", [](testing & t) {
        auto doc = parse(R"({"type": "boolean"})");
        root<common_schema_boolean>(t, doc);
    });

    t.test("number", [](testing & t) {
        auto doc = parse(R"({"type": "number"})");
        root<common_schema_number>(t, doc);
    });

    t.test("number ignores bounds", [](testing & t) {
        auto doc = parse(R"({"type": "number", "minimum": 1, "maximum": 2})");
        root<common_schema_number>(t, doc);
    });
}

static void test_integer(testing & t) {
    t.test("unbounded", [](testing & t) {
        auto doc = parse(R"({"type": "integer"})");
        const auto & i = root<common_schema_integer>(t, doc);
        t.assert_equal("minimum", INT64_MIN, i.minimum);
        t.assert_equal("maximum", INT64_MAX, i.maximum);
    });

    t.test("inclusive bounds", [](testing & t) {
        auto doc = parse(R"({"type": "integer", "minimum": -5, "maximum": 10})");
        const auto & i = root<common_schema_integer>(t, doc);
        t.assert_equal("minimum", -5, i.minimum);
        t.assert_equal("maximum", 10, i.maximum);
    });

    t.test("exclusive bounds are folded", [](testing & t) {
        auto doc = parse(R"({"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 10})");
        const auto & i = root<common_schema_integer>(t, doc);
        t.assert_equal("minimum", 1, i.minimum);
        t.assert_equal("maximum", 9, i.maximum);
    });

    t.test("inclusive bound wins over exclusive", [](testing & t) {
        auto doc = parse(R"({"type": "integer", "minimum": 3, "exclusiveMinimum": 0, "maximum": 7, "exclusiveMaximum": 100})");
        const auto & i = root<common_schema_integer>(t, doc);
        t.assert_equal("minimum", 3, i.minimum);
        t.assert_equal("maximum", 7, i.maximum);
    });

    t.test("fractional bounds round inwards", [](testing & t) {
        auto doc = parse(R"({"type": "integer", "minimum": 1.5, "maximum": 9.5})");
        const auto & i = root<common_schema_integer>(t, doc);
        t.assert_equal("minimum", 2, i.minimum);
        t.assert_equal("maximum", 9, i.maximum);
    });

    t.test("fractional exclusive bounds round inwards", [](testing & t) {
        auto doc = parse(R"({"type": "integer", "exclusiveMinimum": 1.5, "exclusiveMaximum": 9.5})");
        const auto & i = root<common_schema_integer>(t, doc);
        t.assert_equal("minimum", 2, i.minimum);
        t.assert_equal("maximum", 9, i.maximum);
    });

    t.test("integral floats as exclusive bounds", [](testing & t) {
        auto doc = parse(R"({"type": "integer", "exclusiveMinimum": 2.0, "exclusiveMaximum": 9.0})");
        const auto & i = root<common_schema_integer>(t, doc);
        t.assert_equal("minimum", 3, i.minimum);
        t.assert_equal("maximum", 8, i.maximum);
    });
}

static void test_string(testing & t) {
    t.test("defaults", [](testing & t) {
        auto doc = parse(R"({"type": "string"})");
        const auto & s = root<common_schema_string>(t, doc);
        t.assert_equal("pattern", "", s.pattern);
        t.assert_equal("format", COMMON_SCHEMA_FORMAT_NONE, s.format);
        t.assert_equal("min_length", 0, s.min_length);
        t.assert_equal("max_length", -1, s.max_length);
    });

    t.test("all keywords are kept", [](testing & t) {
        auto doc = parse(R"({"type": "string", "pattern": "^[a-z]+$", "format": "date", "minLength": 2, "maxLength": 8})");
        const auto & s = root<common_schema_string>(t, doc);
        t.assert_equal("pattern", "^[a-z]+$", s.pattern);
        t.assert_equal("format", COMMON_SCHEMA_FORMAT_DATE, s.format);
        t.assert_equal("min_length", 2, s.min_length);
        t.assert_equal("max_length", 8, s.max_length);
    });

    t.test("formats", [](testing & t) {
        auto expect = [&](const char * format, common_schema_format expected) {
            auto doc = parse(std::string(R"({"type": "string", "format": ")") + format + "\"}");
            t.assert_equal(format, expected, root<common_schema_string>(t, doc).format);
        };
        expect("date",      COMMON_SCHEMA_FORMAT_DATE);
        expect("time",      COMMON_SCHEMA_FORMAT_TIME);
        expect("date-time", COMMON_SCHEMA_FORMAT_DATE_TIME);
        expect("uuid",      COMMON_SCHEMA_FORMAT_UUID);
        expect("uuid1",     COMMON_SCHEMA_FORMAT_UUID);
        expect("uuid5",     COMMON_SCHEMA_FORMAT_UUID);
        expect("uuid6",     COMMON_SCHEMA_FORMAT_NONE);
        expect("email",     COMMON_SCHEMA_FORMAT_NONE);
        expect("uri",       COMMON_SCHEMA_FORMAT_NONE);
    });

    t.test("pattern implies a string", [](testing & t) {
        auto doc = parse(R"({"pattern": "^a$"})");
        t.assert_equal("pattern", "^a$", root<common_schema_string>(t, doc).pattern);
    });

    t.test("known format implies a string", [](testing & t) {
        auto doc = parse(R"({"format": "uuid"})");
        t.assert_equal("format", COMMON_SCHEMA_FORMAT_UUID, root<common_schema_string>(t, doc).format);
    });
}

static void test_array(testing & t) {
    t.test("items with bounds", [](testing & t) {
        auto doc = parse(R"({"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3})");
        const auto & a = root<common_schema_array>(t, doc);
        as<common_schema_integer>(t, a.items.get(), "items");
        t.assert_equal("min_items", 1, a.min_items);
        t.assert_equal("max_items", 3, a.max_items);
    });

    t.test("no items", [](testing & t) {
        auto doc = parse(R"({"type": "array"})");
        const auto & a = root<common_schema_array>(t, doc);
        as<common_schema_any>(t, a.items.get(), "items");
        t.assert_equal("min_items", 0, a.min_items);
        t.assert_equal("max_items", -1, a.max_items);
    });

    t.test("bounds without items", [](testing & t) {
        auto doc = parse(R"({"type": "array", "minItems": 2})");
        const auto & a = root<common_schema_array>(t, doc);
        as<common_schema_any>(t, a.items.get(), "items");
        t.assert_equal("min_items", 2, a.min_items);
    });

    t.test("items imply an array", [](testing & t) {
        auto doc = parse(R"({"items": {"type": "string"}})");
        const auto & a = root<common_schema_array>(t, doc);
        as<common_schema_string>(t, a.items.get(), "items");
    });

    t.test("prefixItems given as a schema is items", [](testing & t) {
        auto doc = parse(R"({"prefixItems": {"type": "string"}})");
        const auto & a = root<common_schema_array>(t, doc);
        as<common_schema_string>(t, a.items.get(), "items");
    });
}

static void test_tuple(testing & t) {
    t.test("prefixItems", [](testing & t) {
        auto doc = parse(R"({"prefixItems": [{"type": "string"}, {"type": "number"}]})");
        const auto & tup = root<common_schema_tuple>(t, doc);
        t.assert_equal("size", (size_t) 2, tup.items.size());
        as<common_schema_string>(t, tup.items[0].get(), "items[0]");
        as<common_schema_number>(t, tup.items[1].get(), "items[1]");
    });

    t.test("items as an array", [](testing & t) {
        auto doc = parse(R"({"type": "array", "items": [{"type": "boolean"}]})");
        const auto & tup = root<common_schema_tuple>(t, doc);
        t.assert_equal("size", (size_t) 1, tup.items.size());
        as<common_schema_boolean>(t, tup.items[0].get(), "items[0]");
    });

    t.test("items array wins over prefixItems", [](testing & t) {
        auto doc = parse(R"({"items": [{"type": "null"}], "prefixItems": [{"type": "string"}, {"type": "string"}]})");
        const auto & tup = root<common_schema_tuple>(t, doc);
        t.assert_equal("size", (size_t) 1, tup.items.size());
        as<common_schema_null>(t, tup.items[0].get(), "items[0]");
    });

    t.test("empty", [](testing & t) {
        auto doc = parse(R"({"prefixItems": []})");
        t.assert_true("no items", root<common_schema_tuple>(t, doc).items.empty());
    });
}

static void test_object(testing & t) {
    t.test("type alone accepts any object", [](testing & t) {
        auto doc = parse(R"({"type": "object"})");
        const auto & o = root<common_schema_object>(t, doc);
        t.assert_true("no properties", o.properties.empty());
        as<common_schema_any>(t, o.additional_properties.get(), "additional_properties");
    });

    t.test("properties", [](testing & t) {
        auto doc = parse(R"({
            "type": "object",
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "integer"},
                "c": {"type": "boolean"}
            },
            "required": ["a", "c"]
        })");
        const auto & o = root<common_schema_object>(t, doc);
        t.assert_equal("size", (size_t) 3, o.properties.size());
        t.assert_equal("order", "b", o.properties[0].name);
        t.assert_equal("order", "a", o.properties[1].name);
        t.assert_equal("order", "c", o.properties[2].name);
        t.assert_true("b optional", !o.properties[0].required);
        t.assert_true("a required", o.properties[1].required);
        t.assert_true("c required", o.properties[2].required);
        as<common_schema_string>(t, o.properties[0].schema.get(), "b");
        as<common_schema_integer>(t, o.properties[1].schema.get(), "a");
        as<common_schema_boolean>(t, o.properties[2].schema.get(), "c");
        t.assert_true("closed", o.additional_properties == nullptr);
    });

    t.test("properties imply an object", [](testing & t) {
        auto doc = parse(R"({"properties": {"a": {}}})");
        const auto & o = root<common_schema_object>(t, doc);
        t.assert_equal("size", (size_t) 1, o.properties.size());
        t.assert_true("closed", o.additional_properties == nullptr);
    });

    t.test("required entries that are not properties are ignored", [](testing & t) {
        auto doc = parse(R"({"properties": {"a": {}}, "required": ["a", "zzz", 1]})");
        const auto & o = root<common_schema_object>(t, doc);
        t.assert_equal("size", (size_t) 1, o.properties.size());
        t.assert_true("a required", o.properties[0].required);
    });

    t.test("required that is not an array is ignored", [](testing & t) {
        auto doc = parse(R"({"properties": {"a": {}}, "required": "a"})");
        t.assert_true("a optional", !root<common_schema_object>(t, doc).properties[0].required);
    });

    t.test("additionalProperties false", [](testing & t) {
        auto doc = parse(R"({"type": "object", "additionalProperties": false})");
        t.assert_true("closed", root<common_schema_object>(t, doc).additional_properties == nullptr);
    });

    t.test("additionalProperties false implies an object", [](testing & t) {
        auto doc = parse(R"({"additionalProperties": false})");
        const auto & o = root<common_schema_object>(t, doc);
        t.assert_true("no properties", o.properties.empty());
        t.assert_true("closed", o.additional_properties == nullptr);
    });

    t.test("additionalProperties true with properties", [](testing & t) {
        auto doc = parse(R"({"properties": {"a": {}}, "additionalProperties": true})");
        const auto & o = root<common_schema_object>(t, doc);
        t.assert_equal("size", (size_t) 1, o.properties.size());
        as<common_schema_any>(t, o.additional_properties.get(), "additional_properties");
    });

    t.test("additionalProperties schema", [](testing & t) {
        auto doc = parse(R"({"additionalProperties": {"type": "integer", "minimum": 0}})");
        const auto & o = root<common_schema_object>(t, doc);
        t.assert_true("no properties", o.properties.empty());
        const auto & v = as<common_schema_integer>(t, o.additional_properties.get(), "additional_properties");
        t.assert_equal("minimum", 0, v.minimum);
    });

    t.test("nested", [](testing & t) {
        auto doc = parse(R"({"properties": {"inner": {"properties": {"leaf": {"type": "null"}}, "required": ["leaf"]}}})");
        const auto & o = root<common_schema_object>(t, doc);
        const auto & inner = as<common_schema_object>(t, o.properties[0].schema.get(), "inner");
        t.assert_equal("leaf name", "leaf", inner.properties[0].name);
        t.assert_true("leaf required", inner.properties[0].required);
        as<common_schema_null>(t, inner.properties[0].schema.get(), "leaf");
    });
}

static void test_const_enum(testing & t) {
    t.test("const", [](testing & t) {
        auto doc = parse(R"({"const": "x"})");
        t.assert_equal("value", "\"x\"", root<common_schema_const>(t, doc).value.dump());
    });

    t.test("const object", [](testing & t) {
        auto doc = parse(R"({"const": {"a": [1, null]}})");
        t.assert_equal("value", R"({"a":[1,null]})", root<common_schema_const>(t, doc).value.dump());
    });

    t.test("enum", [](testing & t) {
        auto doc = parse(R"({"enum": ["a", 1, null, true]})");
        const auto & e = root<common_schema_enum>(t, doc);
        t.assert_equal("size", (size_t) 4, e.values.size());
        t.assert_equal("values[0]", "\"a\"", e.values[0].dump());
        t.assert_equal("values[1]", "1", e.values[1].dump());
        t.assert_equal("values[2]", "null", e.values[2].dump());
        t.assert_equal("values[3]", "true", e.values[3].dump());
    });

    t.test("const and enum win over type", [](testing & t) {
        auto doc_const = parse(R"({"type": "string", "const": "x"})");
        root<common_schema_const>(t, doc_const);
        auto doc_enum = parse(R"({"type": "integer", "enum": [1, 2]})");
        root<common_schema_enum>(t, doc_enum);
    });

    t.test("const wins over enum", [](testing & t) {
        auto doc = parse(R"({"const": "x", "enum": ["y"]})");
        t.assert_equal("value", "\"x\"", root<common_schema_const>(t, doc).value.dump());
    });
}

static void test_any_of(testing & t) {
    t.test("anyOf", [](testing & t) {
        auto doc = parse(R"({"anyOf": [{"type": "string"}, {"type": "number"}]})");
        const auto & u = root<common_schema_any_of>(t, doc);
        t.assert_equal("size", (size_t) 2, u.children.size());
        as<common_schema_string>(t, u.children[0].get(), "children[0]");
        as<common_schema_number>(t, u.children[1].get(), "children[1]");
    });

    t.test("oneOf", [](testing & t) {
        auto doc = parse(R"({"oneOf": [{"type": "null"}]})");
        const auto & u = root<common_schema_any_of>(t, doc);
        t.assert_equal("size", (size_t) 1, u.children.size());
        as<common_schema_null>(t, u.children[0].get(), "children[0]");
    });

    t.test("oneOf wins over anyOf and type", [](testing & t) {
        auto doc = parse(R"({"type": "string", "oneOf": [{"type": "null"}], "anyOf": [{"type": "number"}, {"type": "boolean"}]})");
        const auto & u = root<common_schema_any_of>(t, doc);
        t.assert_equal("size", (size_t) 1, u.children.size());
        as<common_schema_null>(t, u.children[0].get(), "children[0]");
    });

    t.test("type array expands with sibling keywords", [](testing & t) {
        auto doc = parse(R"({"type": ["string", "null", "integer"], "minLength": 2, "minimum": 5})");
        const auto & u = root<common_schema_any_of>(t, doc);
        t.assert_equal("size", (size_t) 3, u.children.size());
        t.assert_equal("min_length", 2, as<common_schema_string>(t, u.children[0].get(), "children[0]").min_length);
        as<common_schema_null>(t, u.children[1].get(), "children[1]");
        t.assert_equal("minimum", 5, as<common_schema_integer>(t, u.children[2].get(), "children[2]").minimum);
    });

    t.test("type array with structural keywords", [](testing & t) {
        auto doc = parse(R"({"type": ["object", "array"], "properties": {"a": {}}, "items": {"type": "null"}})");
        const auto & u = root<common_schema_any_of>(t, doc);
        t.assert_equal("size", (size_t) 2, u.children.size());
        t.assert_equal("properties", (size_t) 1, as<common_schema_object>(t, u.children[0].get(), "children[0]").properties.size());
        as<common_schema_null>(t, as<common_schema_array>(t, u.children[1].get(), "children[1]").items.get(), "items");
    });
}

static void test_all_of(testing & t) {
    t.test("without a type", [](testing & t) {
        auto doc = parse(R"({"allOf": [{"properties": {"a": {}}}, {"properties": {"b": {}}}]})");
        const auto & all = root<common_schema_all_of>(t, doc);
        t.assert_equal("size", (size_t) 2, all.children.size());
        as<common_schema_object>(t, all.children[0].get(), "children[0]");
        as<common_schema_object>(t, all.children[1].get(), "children[1]");
    });

    t.test("type string", [](testing & t) {
        auto doc = parse(R"({"type": "string", "allOf": [{"enum": ["a", "b"]}, {"enum": ["b", "c"]}]})");
        const auto & all = root<common_schema_all_of>(t, doc);
        t.assert_equal("size", (size_t) 2, all.children.size());
        as<common_schema_enum>(t, all.children[0].get(), "children[0]");
    });

    t.test("type object without properties", [](testing & t) {
        auto doc = parse(R"({"type": "object", "allOf": [{"properties": {"a": {}}}]})");
        root<common_schema_all_of>(t, doc);
    });

    t.test("properties win over allOf", [](testing & t) {
        auto doc = parse(R"({"type": "object", "properties": {"a": {}}, "allOf": [{"properties": {"b": {}}}]})");
        t.assert_equal("size", (size_t) 1, root<common_schema_object>(t, doc).properties.size());
    });

    t.test("nested anyOf component", [](testing & t) {
        auto doc = parse(R"({"allOf": [{"anyOf": [{"properties": {"a": {}}}, {"properties": {"b": {}}}]}]})");
        const auto & all = root<common_schema_all_of>(t, doc);
        const auto & any = as<common_schema_any_of>(t, all.children[0].get(), "children[0]");
        t.assert_equal("size", (size_t) 2, any.children.size());
    });

    t.test("other types ignore allOf", [](testing & t) {
        auto doc = parse(R"({"type": "integer", "allOf": [{"minimum": 1}]})");
        root<common_schema_integer>(t, doc);
    });
}

static void test_ref(testing & t) {
    t.test("target is owned by the document", [](testing & t) {
        auto doc = parse(R"({"$ref": "#/$defs/t", "$defs": {"t": {"type": "boolean"}}})");
        const auto & r = root<common_schema_ref>(t, doc);
        t.assert_equal("ref", "#/$defs/t", r.ref);
        t.assert_equal("refs", (size_t) 1, doc.refs.size());
        t.assert_true("target", r.target != nullptr && r.target == doc.refs.at("#/$defs/t").get());
        as<common_schema_boolean>(t, r.target, "target");
    });

    t.test("sibling keywords are ignored", [](testing & t) {
        auto doc = parse(R"({"$ref": "#/$defs/t", "type": "string", "$defs": {"t": {"type": "null"}}})");
        as<common_schema_null>(t, root<common_schema_ref>(t, doc).target, "target");
    });

    t.test("definitions", [](testing & t) {
        auto doc = parse(R"({"properties": {"a": {"$ref": "#/definitions/t"}}, "definitions": {"t": {"type": "number"}}})");
        const auto & o = root<common_schema_object>(t, doc);
        const auto & r = as<common_schema_ref>(t, o.properties[0].schema.get(), "a");
        as<common_schema_number>(t, r.target, "target");
    });

    t.test("same ref shares one target", [](testing & t) {
        auto doc = parse(R"({"properties": {"a": {"$ref": "#/$defs/t"}, "b": {"$ref": "#/$defs/t"}}, "$defs": {"t": {"type": "boolean"}}})");
        const auto & o = root<common_schema_object>(t, doc);
        const auto & ra = as<common_schema_ref>(t, o.properties[0].schema.get(), "a");
        const auto & rb = as<common_schema_ref>(t, o.properties[1].schema.get(), "b");
        t.assert_true("shared", ra.target != nullptr && ra.target == rb.target);
        t.assert_equal("refs", (size_t) 1, doc.refs.size());
    });

    t.test("recursive", [](testing & t) {
        auto doc = parse(R"({
            "$ref": "#/$defs/node",
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "next": {"$ref": "#/$defs/node"}
                    },
                    "required": ["value"]
                }
            }
        })");
        const auto & r = root<common_schema_ref>(t, doc);
        const auto & node = as<common_schema_object>(t, r.target, "node");
        t.assert_equal("properties", (size_t) 2, node.properties.size());
        const auto & next = as<common_schema_ref>(t, node.properties[1].schema.get(), "next");
        t.assert_true("cycle", next.target == r.target);
        t.assert_equal("refs", (size_t) 1, doc.refs.size());
    });

    t.test("mutually recursive", [](testing & t) {
        auto doc = parse(R"({
            "$ref": "#/$defs/a",
            "$defs": {
                "a": {"properties": {"b": {"$ref": "#/$defs/b"}}},
                "b": {"properties": {"a": {"$ref": "#/$defs/a"}}}
            }
        })");
        const auto & ra = root<common_schema_ref>(t, doc);
        const auto & a  = as<common_schema_object>(t, ra.target, "a");
        const auto & rb = as<common_schema_ref>(t, a.properties[0].schema.get(), "a.b");
        const auto & b  = as<common_schema_object>(t, rb.target, "b");
        const auto & back = as<common_schema_ref>(t, b.properties[0].schema.get(), "b.a");
        t.assert_true("cycle", back.target == ra.target);
        t.assert_equal("refs", (size_t) 2, doc.refs.size());
    });

    t.test("pointer through an array", [](testing & t) {
        auto doc = parse(R"({"oneOf": [{"type": "null"}, {"$ref": "#/oneOf/0"}]})");
        const auto & u = root<common_schema_any_of>(t, doc);
        const auto & r = as<common_schema_ref>(t, u.children[1].get(), "children[1]");
        as<common_schema_null>(t, r.target, "target");
    });

    t.test("targets survive moving the document", [](testing & t) {
        auto parsed = parse(R"({"items": {"$ref": "#/$defs/t"}, "$defs": {"t": {"type": "null"}}})");
        common_schema_document doc = std::move(parsed);
        const auto & a = root<common_schema_array>(t, doc);
        const auto & r = as<common_schema_ref>(t, a.items.get(), "items");
        t.assert_true("target", r.target == doc.refs.at("#/$defs/t").get());
        as<common_schema_null>(t, r.target, "target");
    });

    t.test("a schema parsed into a document reuses its refs", [](testing & t) {
        auto doc  = parse(R"({"properties": {"a": {"$ref": "#/$defs/t"}}, "$defs": {"t": {"type": "boolean"}}})");
        auto node = common_schema_parse(common_json::parse(R"({"items": {"$ref": "#/$defs/t"}})"), doc);
        const auto & a = as<common_schema_array>(t, node.get(), "node");
        const auto & r = as<common_schema_ref>(t, a.items.get(), "items");
        t.assert_true("shared target", r.target == doc.refs.at("#/$defs/t").get());
        t.assert_equal("refs", (size_t) 1, doc.refs.size());
    });

    t.test("a schema parsed into a document adds its refs", [](testing & t) {
        common_schema_document doc;
        auto node = common_schema_parse(common_json::parse(R"({"$ref": "#/$defs/t", "$defs": {"t": {"type": "null"}}})"), doc);
        as<common_schema_null>(t, as<common_schema_ref>(t, node.get(), "node").target, "target");
        t.assert_equal("refs", (size_t) 1, doc.refs.size());
    });

    t.test("a rejected schema leaves the document unchanged", [](testing & t) {
        common_schema_document doc;
        try {
            common_schema_parse(common_json::parse(R"({"allOf": [{"$ref": "#/$defs/t"}, {"type": "x"}], "$defs": {"t": {"type": "null"}}})"), doc);
            t.assert_true("rejected", false);
        } catch (const std::runtime_error &) {
            t.assert_true("no refs", doc.refs.empty());
        }
    });
}

static void test_errors(testing & t) {
    t.test("not a schema", [](testing & t) {
        assert_error(t, R"([])", "#: schema must be an object");
        assert_error(t, R"("string")", "#: schema must be an object");
    });

    t.test("type", [](testing & t) {
        assert_error(t, R"({"type": "char"})", "#: unrecognized type char");
        assert_error(t, R"({"type": 5})", "#: type must be a string or an array of strings");
        assert_error(t, R"({"type": []})", "#: type must not be empty");
        assert_error(t, R"({"type": ["string", "bad"]})", "#/type/1: unrecognized type bad");
    });

    t.test("ref", [](testing & t) {
        assert_error(t, R"({"$ref": 5})", "#: $ref must be a string");
        assert_error(t, R"({"$ref": "https://example.com/x.json"})", "#: unsupported $ref https://example.com/x.json");
        assert_error(t, R"({"$ref": "#"})", "#: unsupported $ref #");
        assert_error(t, R"({"$ref": "#/$defs/missing"})", "#: cannot resolve $ref #/$defs/missing, $defs not found");
        assert_error(t, R"({"$defs": {}, "$ref": "#/$defs/missing"})", "#: cannot resolve $ref #/$defs/missing, missing not found");
        assert_error(t, R"({"oneOf": [{}], "$ref": "#/oneOf/1"})", "#: cannot resolve $ref #/oneOf/1, 1 is out of range");
        assert_error(t, R"({"oneOf": [{}], "$ref": "#/oneOf/x"})", "#: cannot resolve $ref #/oneOf/x, x is out of range");
        assert_error(t, R"({"$defs": {"a": {"$ref": "#/$defs/a/nope"}}, "$ref": "#/$defs/a"})", "#/$defs/a: cannot resolve $ref #/$defs/a/nope, nope not found");
    });

    t.test("alternatives", [](testing & t) {
        assert_error(t, R"({"oneOf": []})", "#/oneOf: must not be empty");
        assert_error(t, R"({"anyOf": {}})", "#/anyOf: must be an array of schemas");
        assert_error(t, R"({"allOf": []})", "#/allOf: must not be empty");
        assert_error(t, R"({"anyOf": [{"type": "string"}, {"items": {"type": "x"}}]})", "#/anyOf/1/items: unrecognized type x");
    });

    t.test("enum", [](testing & t) {
        assert_error(t, R"({"enum": []})", "#: enum must be a non-empty array");
        assert_error(t, R"({"enum": "a"})", "#: enum must be a non-empty array");
    });

    t.test("string", [](testing & t) {
        assert_error(t, R"({"type": "string", "pattern": 5})", "#: pattern must be a string");
        assert_error(t, R"({"type": "string", "format": 5})", "#: format must be a string");
        assert_error(t, R"({"type": "string", "minLength": -1})", "#: minLength must be a non-negative integer");
        assert_error(t, R"({"type": "string", "maxLength": "5"})", "#: maxLength must be a non-negative integer");
    });

    t.test("integer", [](testing & t) {
        assert_error(t, R"({"type": "integer", "minimum": "1"})", "#: minimum must be a number");
        assert_error(t, R"({"type": "integer", "exclusiveMaximum": null})", "#: exclusiveMaximum must be a number");
    });

    t.test("array", [](testing & t) {
        assert_error(t, R"({"type": "array", "maxItems": 1.5})", "#: maxItems must be a non-negative integer");
        assert_error(t, R"({"type": "array", "items": {"type": "x"}})", "#/items: unrecognized type x");
        assert_error(t, R"({"prefixItems": [{"type": "x"}]})", "#/prefixItems/0: unrecognized type x");
    });

    t.test("object", [](testing & t) {
        assert_error(t, R"({"properties": []})", "#: properties must be an object");
        assert_error(t, R"({"properties": {"a": {"type": "nope"}}})", "#/properties/a: unrecognized type nope");
        assert_error(t, R"({"additionalProperties": null})", "#: additionalProperties must be a boolean or a schema");
        assert_error(t, R"({"additionalProperties": {"type": "x"}})", "#/additionalProperties: unrecognized type x");
    });
}

int main(int argc, char * argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    t.test("any", test_any);
    t.test("primitives", test_primitives);
    t.test("integer", test_integer);
    t.test("string", test_string);
    t.test("array", test_array);
    t.test("tuple", test_tuple);
    t.test("object", test_object);
    t.test("const and enum", test_const_enum);
    t.test("any_of", test_any_of);
    t.test("all_of", test_all_of);
    t.test("ref", test_ref);
    t.test("errors", test_errors);

    return t.summary();
}
