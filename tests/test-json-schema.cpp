#include "json-schema.h"
#include "json.h"
#include "testing.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

static common_schema_document parse(const std::string & schema) {
    return common_schema_from_json(common_json::parse(schema));
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

    t.test("keywords that do not imply a type", [](testing & t) {
        auto doc = parse(R"({"description": "x", "format": "email", "minLength": 3, "additionalProperties": true})");
        root<common_schema_any>(t, doc);
    });
}

static void test_primitives(testing & t) {
    t.test("null, boolean, number", [](testing & t) {
        auto doc_null = parse(R"({"type": "null"})");
        root<common_schema_null>(t, doc_null);
        auto doc_bool = parse(R"({"type": "boolean"})");
        root<common_schema_boolean>(t, doc_bool);
        auto doc_num = parse(R"({"type": "number", "minimum": 1, "maximum": 2})");
        root<common_schema_number>(t, doc_num);
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

    t.test("fractional bounds round inwards", [](testing & t) {
        auto doc = parse(R"({"type": "integer", "minimum": 1.5, "exclusiveMaximum": 9.5})");
        const auto & i = root<common_schema_integer>(t, doc);
        t.assert_equal("minimum", 2, i.minimum);
        t.assert_equal("maximum", 9, i.maximum);
    });
}

static void test_string(testing & t) {
    t.test("defaults", [](testing & t) {
        auto doc = parse(R"({"type": "string"})");
        const auto & s = root<common_schema_string>(t, doc);
        t.assert_equal("pattern", "", s.pattern);
        t.assert_equal("format", common_schema::FORMAT_NONE, s.format);
        t.assert_equal("min_length", 0, s.min_length);
        t.assert_equal("max_length", -1, s.max_length);
    });

    t.test("all keywords are kept", [](testing & t) {
        auto doc = parse(R"({"type": "string", "pattern": "^[a-z]+$", "format": "date", "minLength": 2, "maxLength": 8})");
        const auto & s = root<common_schema_string>(t, doc);
        t.assert_equal("pattern", "^[a-z]+$", s.pattern);
        t.assert_equal("format", common_schema::FORMAT_DATE, s.format);
        t.assert_equal("min_length", 2, s.min_length);
        t.assert_equal("max_length", 8, s.max_length);
    });

    t.test("formats", [](testing & t) {
        auto expect = [&](const char * format, common_schema::string_format expected) {
            auto doc = parse(std::string(R"({"type": "string", "format": ")") + format + "\"}");
            t.assert_equal(format, expected, root<common_schema_string>(t, doc).format);
        };
        expect("time",      common_schema::FORMAT_TIME);
        expect("date-time", common_schema::FORMAT_DATE_TIME);
        expect("uuid",      common_schema::FORMAT_UUID);
        expect("uuid5",     common_schema::FORMAT_UUID);
        expect("email",     common_schema::FORMAT_NONE);
    });

    t.test("pattern and known format imply a string", [](testing & t) {
        auto doc_pattern = parse(R"({"pattern": "^a$"})");
        t.assert_equal("pattern", "^a$", root<common_schema_string>(t, doc_pattern).pattern);
        auto doc_format = parse(R"({"format": "uuid"})");
        t.assert_equal("format", common_schema::FORMAT_UUID, root<common_schema_string>(t, doc_format).format);
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

    t.test("items imply an array", [](testing & t) {
        auto doc = parse(R"({"items": {"type": "string"}})");
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

    t.test("unknown required entries are ignored", [](testing & t) {
        auto doc = parse(R"({"properties": {"a": {}}, "required": ["a", "zzz", 1]})");
        const auto & o = root<common_schema_object>(t, doc);
        t.assert_equal("size", (size_t) 1, o.properties.size());
        t.assert_true("a required", o.properties[0].required);
    });

    t.test("additionalProperties false implies an object", [](testing & t) {
        auto doc = parse(R"({"additionalProperties": false})");
        const auto & o = root<common_schema_object>(t, doc);
        t.assert_true("no properties", o.properties.empty());
        t.assert_true("closed", o.additional_properties == nullptr);
    });

    t.test("additionalProperties schema", [](testing & t) {
        auto doc = parse(R"({"properties": {"a": {}}, "additionalProperties": {"type": "integer", "minimum": 0}})");
        const auto & o = root<common_schema_object>(t, doc);
        t.assert_equal("size", (size_t) 1, o.properties.size());
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

    t.test("const wins over enum, enum wins over type", [](testing & t) {
        auto doc_enum = parse(R"({"type": "integer", "enum": [1, 2]})");
        root<common_schema_enum>(t, doc_enum);
        auto doc_const = parse(R"({"type": "string", "const": "x", "enum": ["y"]})");
        t.assert_equal("value", "\"x\"", root<common_schema_const>(t, doc_const).value.dump());
    });
}

static void test_any_of(testing & t) {
    t.test("anyOf and oneOf", [](testing & t) {
        auto doc_any = parse(R"({"anyOf": [{"type": "string"}, {"type": "number"}]})");
        const auto & u = root<common_schema_any_of>(t, doc_any);
        t.assert_equal("size", (size_t) 2, u.children.size());
        as<common_schema_string>(t, u.children[0].get(), "children[0]");
        as<common_schema_number>(t, u.children[1].get(), "children[1]");

        auto doc_one = parse(R"({"oneOf": [{"type": "null"}]})");
        const auto & o = root<common_schema_any_of>(t, doc_one);
        t.assert_equal("size", (size_t) 1, o.children.size());
        as<common_schema_null>(t, o.children[0].get(), "children[0]");
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
}

static void test_all_of(testing & t) {
    t.test("components", [](testing & t) {
        auto doc = parse(R"({"allOf": [{"properties": {"a": {}}}, {"anyOf": [{"properties": {"b": {}}}, {"type": "null"}]}]})");
        const auto & all = root<common_schema_all_of>(t, doc);
        t.assert_equal("size", (size_t) 2, all.children.size());
        as<common_schema_object>(t, all.children[0].get(), "children[0]");
        as<common_schema_any_of>(t, all.children[1].get(), "children[1]");

        auto doc_typed = parse(R"({"type": "object", "allOf": [{"properties": {"a": {}}}]})");
        root<common_schema_all_of>(t, doc_typed);
    });

    t.test("properties win over allOf", [](testing & t) {
        auto doc = parse(R"({"type": "object", "properties": {"a": {}}, "allOf": [{"properties": {"b": {}}}]})");
        t.assert_equal("size", (size_t) 1, root<common_schema_object>(t, doc).properties.size());
    });

    t.test("other types ignore allOf", [](testing & t) {
        auto doc = parse(R"({"type": "integer", "allOf": [{"minimum": 1}]})");
        root<common_schema_integer>(t, doc);
    });
}

static void test_ref(testing & t) {
    t.test("target is owned by the document", [](testing & t) {
        auto doc = parse(R"({"$ref": "#/$defs/t", "type": "string", "$defs": {"t": {"type": "boolean"}}})");
        const auto & r = root<common_schema_ref>(t, doc);
        t.assert_equal("ref", "#/$defs/t", r.ref);
        t.assert_equal("refs", (size_t) 1, doc.refs.size());
        t.assert_true("target", r.target != nullptr && r.target == doc.refs.at("#/$defs/t").get());
        as<common_schema_boolean>(t, r.target, "target");
    });

    t.test("definitions", [](testing & t) {
        auto doc = parse(R"({"properties": {"a": {"$ref": "#/definitions/t"}}, "definitions": {"t": {"type": "number"}}})");
        const auto & o = root<common_schema_object>(t, doc);
        const auto & r = as<common_schema_ref>(t, o.properties[0].schema.get(), "a");
        as<common_schema_number>(t, r.target, "target");
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

    t.test("a schema parsed into a document shares its refs", [](testing & t) {
        auto doc  = parse(R"({"properties": {"a": {"$ref": "#/$defs/t"}}, "$defs": {"t": {"type": "boolean"}}})");
        auto node = common_schema_from_json(common_json::parse(R"({"items": {"$ref": "#/$defs/t"}})"), doc);
        const auto & a = as<common_schema_array>(t, node.get(), "node");
        const auto & r = as<common_schema_ref>(t, a.items.get(), "items");
        t.assert_true("shared target", r.target == doc.refs.at("#/$defs/t").get());
        t.assert_equal("refs", (size_t) 1, doc.refs.size());

        auto added = common_schema_from_json(common_json::parse(R"({"$ref": "#/$defs/u", "$defs": {"u": {"type": "null"}}})"), doc);
        as<common_schema_null>(t, as<common_schema_ref>(t, added.get(), "added").target, "target");
        t.assert_equal("refs", (size_t) 2, doc.refs.size());
    });

    t.test("a rejected schema leaves the document unchanged", [](testing & t) {
        common_schema_document doc;
        try {
            common_schema_from_json(common_json::parse(R"({"allOf": [{"$ref": "#/$defs/t"}, {"type": "x"}], "$defs": {"t": {"type": "null"}}})"), doc);
            t.assert_true("rejected", false);
        } catch (const std::runtime_error &) {
            t.assert_true("no refs", doc.refs.empty());
        }
    });
}

static void test_may_be_string(testing & t) {
    auto check = [](testing & t, const std::string & schema, bool expected) {
        t.assert_equal(schema, expected, parse(schema).root->may_be_string());
    };

    t.test("leaves", [&](testing & t) {
        check(t, R"({"type": "string"})", true);
        check(t, R"({"type": "integer"})", false);
        check(t, R"({"minLength": 1})", false);
        check(t, R"({"pattern": "^[a-z]+$"})", true);
        check(t, R"({"const": "hello"})", true);
        check(t, R"({"const": 123})", false);
        check(t, R"({"enum": [1, "a", null]})", true);
        check(t, R"({"enum": [1, 2, 3]})", false);
    });

    t.test("composites", [&](testing & t) {
        check(t, R"({"type": ["integer", "string"]})", true);
        check(t, R"({"anyOf": [{"type": "integer"}, {"type": "boolean"}]})", false);
        check(t, R"({"allOf": [{"type": "string"}, {"minLength": 1}]})", true);
        check(t, R"({"allOf": [{"type": "string"}, {"type": "integer"}]})", false);
        check(t, R"({"allOf": [{"minLength": 1}, {"maxLength": 2}]})", false);
    });

    t.test("ref", [&](testing & t) {
        check(t, R"({"$ref": "#/$defs/n", "$defs": {"n": {"anyOf": [{"$ref": "#/$defs/n"}, {"type": "string"}]}}})", true);
        check(t, R"({"$ref": "#/$defs/n", "$defs": {"n": {"$ref": "#/$defs/n"}}})", false);
    });
}

// e.g. {number, integer}, in type order
static std::string dump(const common_schema::type_set & types) {
    static const common_schema::value_type order[] = { common_schema::TYPE_NULL,   common_schema::TYPE_BOOLEAN, common_schema::TYPE_NUMBER,
                                                       common_schema::TYPE_INTEGER, common_schema::TYPE_STRING,  common_schema::TYPE_ARRAY,
                                                       common_schema::TYPE_OBJECT };
    std::string out;
    for (auto type : order) {
        if (types.has(type)) {
            out += (out.empty() ? "" : ", ") + std::string(common_schema::type_name(type));
        }
    }
    return "{" + out + "}";
}

static void test_value_types(testing & t) {
    auto check = [](testing & t, const std::string & schema, const common_schema::type_set & expected) {
        t.assert_equal(schema, dump(expected), dump(parse(schema).root->value_types()));
    };

    t.test("leaves", [&](testing & t) {
        check(t, R"({"type": "string"})", { common_schema::TYPE_STRING });
        check(t, R"({"type": "number"})", { common_schema::TYPE_NUMBER, common_schema::TYPE_INTEGER });
        check(t, R"({"minLength": 1})", common_schema::type_set::all());
        check(t, R"({"properties": {"a": {"type": "string"}}})", { common_schema::TYPE_OBJECT });
        check(t, R"({"items": {"type": "string"}})", { common_schema::TYPE_ARRAY });
        check(t, R"({"const": 1.5})", { common_schema::TYPE_NUMBER });
        check(t, R"({"enum": [1, "a", null]})", { common_schema::TYPE_INTEGER, common_schema::TYPE_STRING, common_schema::TYPE_NULL });
    });

    t.test("any_of is the union, all_of is the intersection", [&](testing & t) {
        check(t, R"({"type": ["string", "null"]})", { common_schema::TYPE_STRING, common_schema::TYPE_NULL });
        check(t, R"({"allOf": [{"type": ["string", "number"]}, {"type": ["number", "object"]}]})", { common_schema::TYPE_NUMBER, common_schema::TYPE_INTEGER });
        check(t, R"({"allOf": [{"type": "string"}, {"type": "integer"}]})", {});
    });

    t.test("ref", [&](testing & t) {
        check(t, R"({"$ref": "#/$defs/n", "$defs": {"n": {"anyOf": [{"$ref": "#/$defs/n"}, {"type": "string"}]}}})",
              { common_schema::TYPE_STRING });
    });
}

static void test_errors(testing & t) {
    t.test("not a schema", [](testing & t) {
        assert_error(t, R"([])", "#: schema must be an object");
    });

    t.test("type", [](testing & t) {
        assert_error(t, R"({"type": 5})", "#: type must be a string or an array of strings");
        assert_error(t, R"({"type": []})", "#: type must not be empty");
        assert_error(t, R"({"type": ["string", "bad"]})", "#/type/1: unrecognized type bad");
    });

    t.test("ref", [](testing & t) {
        assert_error(t, R"({"$ref": 5})", "#: $ref must be a string");
        assert_error(t, R"({"$ref": "https://example.com/x.json"})", "#: unsupported $ref https://example.com/x.json");
        assert_error(t, R"({"$defs": {}, "$ref": "#/$defs/missing"})", "#: cannot resolve $ref #/$defs/missing, missing not found");
        assert_error(t, R"({"oneOf": [{}], "$ref": "#/oneOf/1"})", "#: cannot resolve $ref #/oneOf/1, 1 is out of range");
        assert_error(t, R"({"$defs": {"a": {"$ref": "#/$defs/a/nope"}}, "$ref": "#/$defs/a"})", "#/$defs/a: cannot resolve $ref #/$defs/a/nope, nope not found");
    });

    t.test("alternatives", [](testing & t) {
        assert_error(t, R"({"oneOf": []})", "#/oneOf: must not be empty");
        assert_error(t, R"({"anyOf": {}})", "#/anyOf: must be an array of schemas");
        assert_error(t, R"({"anyOf": [{"type": "string"}, {"items": {"type": "x"}}]})", "#/anyOf/1/items: unrecognized type x");
    });

    t.test("keywords", [](testing & t) {
        assert_error(t, R"({"enum": []})", "#: enum must be a non-empty array");
        assert_error(t, R"({"type": "string", "pattern": 5})", "#: pattern must be a string");
        assert_error(t, R"({"type": "string", "minLength": -1})", "#: minLength must be a non-negative integer");
        assert_error(t, R"({"type": "integer", "minimum": "1"})", "#: minimum must be a number");
        assert_error(t, R"({"type": "array", "maxItems": 1.5})", "#: maxItems must be a non-negative integer");
        assert_error(t, R"({"properties": []})", "#: properties must be an object");
        assert_error(t, R"({"properties": {"a": {"type": "nope"}}})", "#/properties/a: unrecognized type nope");
        assert_error(t, R"({"additionalProperties": null})", "#: additionalProperties must be a boolean or a schema");
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
    t.test("may_be_string", test_may_be_string);
    t.test("value_types", test_value_types);
    t.test("errors", test_errors);

    return t.summary();
}
