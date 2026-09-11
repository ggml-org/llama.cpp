#pragma once

#include "json.h"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

// JSON schema, covering the subset that json_schema_to_grammar() can convert.

struct common_schema {
    enum node_kind {
        KIND_ANY,
        KIND_REF,
        KIND_ANY_OF,
        KIND_ALL_OF,
        KIND_CONST,
        KIND_ENUM,
        KIND_NULL,
        KIND_BOOLEAN,
        KIND_NUMBER,
        KIND_INTEGER,
        KIND_STRING,
        KIND_ARRAY,
        KIND_TUPLE,
        KIND_OBJECT,
    };

    enum value_type {
        TYPE_NULL,
        TYPE_BOOLEAN,
        TYPE_NUMBER,
        TYPE_INTEGER,
        TYPE_STRING,
        TYPE_ARRAY,
        TYPE_OBJECT,
    };

    enum string_format {
        FORMAT_NONE,
        FORMAT_UUID,  // uuid, uuid1 .. uuid5
        FORMAT_DATE,
        FORMAT_TIME,
        FORMAT_DATE_TIME,
    };

    class type_set {
        uint32_t mask_ = 0;

      public:
        type_set() = default;
        type_set(std::initializer_list<value_type> types) {
            for (auto type : types) {
                add(type);
            }
        }

        static type_set all() {
            return { TYPE_NULL, TYPE_BOOLEAN, TYPE_NUMBER, TYPE_INTEGER, TYPE_STRING, TYPE_ARRAY, TYPE_OBJECT };
        }

        void add(value_type type) { mask_ |= 1u << type; }

        bool has(value_type type) const { return (mask_ & (1u << type)) != 0; }
        bool is_only(value_type type) const { return mask_ == (1u << type); }
        bool empty() const { return mask_ == 0; }

        type_set & operator|=(const type_set & other) { mask_ |= other.mask_; return *this; }
        type_set & operator&=(const type_set & other) { mask_ &= other.mask_; return *this; }

        bool operator==(const type_set & other) const { return mask_ == other.mask_; }
        bool operator!=(const type_set & other) const { return mask_ != other.mask_; }
    };

    virtual ~common_schema() = default;
    virtual node_kind kind() const = 0;

    type_set value_types() const;

    // Whether a value matching the schema may be a string, through any branch of it.
    bool may_be_string() const;

    static const char * kind_name(node_kind kind);
    static const char * type_name(value_type type);
};

using common_schema_ptr = std::unique_ptr<common_schema>;

struct common_schema_any : common_schema {
    node_kind kind() const override { return KIND_ANY; }
};

// {"$ref": "#/..."}, only references into the same document are supported
struct common_schema_ref : common_schema {
    std::string           ref;
    const common_schema * target = nullptr;  // owned by common_schema_document::refs

    explicit common_schema_ref(std::string ref) : ref(std::move(ref)) {}

    node_kind kind() const override { return KIND_REF; }
};

// oneOf / anyOf, or a "type" array expanded to one alternative per type
struct common_schema_any_of : common_schema {
    std::vector<common_schema_ptr> children;

    node_kind kind() const override { return KIND_ANY_OF; }
};

struct common_schema_all_of : common_schema {
    std::vector<common_schema_ptr> children;

    node_kind kind() const override { return KIND_ALL_OF; }
};

struct common_schema_const : common_schema {
    common_json value;

    explicit common_schema_const(common_json value) : value(std::move(value)) {}

    node_kind kind() const override { return KIND_CONST; }
};

struct common_schema_enum : common_schema {
    std::vector<common_json> values;

    node_kind kind() const override { return KIND_ENUM; }
};

struct common_schema_null : common_schema {
    node_kind kind() const override { return KIND_NULL; }
};

struct common_schema_boolean : common_schema {
    node_kind kind() const override { return KIND_BOOLEAN; }
};

struct common_schema_number : common_schema {
    node_kind kind() const override { return KIND_NUMBER; }
};

// bounds are inclusive, exclusiveMinimum / exclusiveMaximum are folded in
struct common_schema_integer : common_schema {
    int64_t minimum = INT64_MIN;  // INT64_MIN for unbounded
    int64_t maximum = INT64_MAX;  // INT64_MAX for unbounded

    node_kind kind() const override { return KIND_INTEGER; }
};

struct common_schema_string : common_schema {
    std::string   pattern;  // empty when absent
    string_format format     = FORMAT_NONE;
    int           min_length = 0;
    int           max_length = -1;  // -1 for unbounded

    node_kind kind() const override { return KIND_STRING; }
};

struct common_schema_array : common_schema {
    common_schema_ptr items;  // a common_schema_any when "items" is absent
    int               min_items = 0;
    int               max_items = -1;  // -1 for unbounded

    node_kind kind() const override { return KIND_ARRAY; }
};

struct common_schema_tuple : common_schema {
    std::vector<common_schema_ptr> items;

    node_kind kind() const override { return KIND_TUPLE; }
};

struct common_schema_property {
    std::string       name;
    common_schema_ptr schema;
    bool              required = false;
};

struct common_schema_object : common_schema {
    std::vector<common_schema_property> properties;             // in schema order
    common_schema_ptr                   additional_properties;  // null when not allowed

    node_kind kind() const override { return KIND_OBJECT; }
};

struct common_schema_document {
    common_schema_ptr                        root;
    std::map<std::string, common_schema_ptr> refs;
};

// A document shared by the PEG parsers built from its nodes, which it keeps alive
using common_schema_document_ptr = std::shared_ptr<const common_schema_document>;

// Throws std::runtime_error when the schema falls outside the supported subset.
common_schema_document common_schema_from_json(const common_json & schema);

// Builds a schema that belongs to a document built earlier, e.g. one property of it.
// A $ref it cannot resolve on its own is looked up in doc.refs, the targets it resolves itself are added there.
// doc is unchanged when the schema is rejected.
common_schema_ptr common_schema_from_json(const common_json & schema, common_schema_document & doc);
