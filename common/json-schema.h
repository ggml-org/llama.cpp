#pragma once

#include "json.h"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

// JSON schema IR, covering the subset that json_schema_to_grammar() can convert.
// A $ref becomes a common_schema_ref whose target is owned by the common_schema_document, so a recursive schema stays finite.

enum common_schema_kind {
    COMMON_SCHEMA_KIND_ANY,
    COMMON_SCHEMA_KIND_NONE,
    COMMON_SCHEMA_KIND_REF,
    COMMON_SCHEMA_KIND_ANY_OF,
    COMMON_SCHEMA_KIND_ALL_OF,
    COMMON_SCHEMA_KIND_CONST,
    COMMON_SCHEMA_KIND_ENUM,
    COMMON_SCHEMA_KIND_NULL,
    COMMON_SCHEMA_KIND_BOOLEAN,
    COMMON_SCHEMA_KIND_NUMBER,
    COMMON_SCHEMA_KIND_INTEGER,
    COMMON_SCHEMA_KIND_STRING,
    COMMON_SCHEMA_KIND_ARRAY,
    COMMON_SCHEMA_KIND_TUPLE,
    COMMON_SCHEMA_KIND_OBJECT,
};

enum common_schema_format {
    COMMON_SCHEMA_FORMAT_NONE,
    COMMON_SCHEMA_FORMAT_UUID,  // uuid, uuid1 .. uuid5
    COMMON_SCHEMA_FORMAT_DATE,
    COMMON_SCHEMA_FORMAT_TIME,
    COMMON_SCHEMA_FORMAT_DATE_TIME,
};

// Base class for all nodes, the concrete ones are the common_schema_* structs below
struct common_schema {
    virtual ~common_schema() = default;
    virtual common_schema_kind kind() const = 0;
};

using common_schema_ptr = std::unique_ptr<common_schema>;

// {} or a schema with no recognized keywords: any JSON value
struct common_schema_any : common_schema {
    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_ANY; }
};

// Matches no value: what common_schema_optimize() leaves where an intersection turned out empty
struct common_schema_none : common_schema {
    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_NONE; }
};

// {"$ref": "#/..."}, only references into the same document are supported
struct common_schema_ref : common_schema {
    std::string           ref;
    const common_schema * target = nullptr;  // owned by common_schema_document::refs

    explicit common_schema_ref(std::string ref) : ref(std::move(ref)) {}

    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_REF; }
};

// oneOf / anyOf, or a "type" array expanded to one alternative per type
struct common_schema_any_of : common_schema {
    std::vector<common_schema_ptr> children;

    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_ANY_OF; }
};

struct common_schema_all_of : common_schema {
    std::vector<common_schema_ptr> children;

    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_ALL_OF; }
};

struct common_schema_const : common_schema {
    common_json value;

    explicit common_schema_const(common_json value) : value(std::move(value)) {}

    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_CONST; }
};

struct common_schema_enum : common_schema {
    std::vector<common_json> values;

    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_ENUM; }
};

struct common_schema_null : common_schema {
    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_NULL; }
};

struct common_schema_boolean : common_schema {
    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_BOOLEAN; }
};

struct common_schema_number : common_schema {
    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_NUMBER; }
};

// bounds are inclusive, exclusiveMinimum / exclusiveMaximum are folded in
struct common_schema_integer : common_schema {
    int64_t minimum = INT64_MIN;  // INT64_MIN for unbounded
    int64_t maximum = INT64_MAX;  // INT64_MAX for unbounded

    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_INTEGER; }
};

struct common_schema_string : common_schema {
    std::string          pattern;  // empty when absent
    common_schema_format format     = COMMON_SCHEMA_FORMAT_NONE;
    int                  min_length = 0;
    int                  max_length = -1;  // -1 for unbounded

    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_STRING; }
};

struct common_schema_array : common_schema {
    common_schema_ptr items;  // a common_schema_any when "items" is absent
    int               min_items = 0;
    int               max_items = -1;  // -1 for unbounded

    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_ARRAY; }
};

// "prefixItems", or "items" given as an array: one schema per position
struct common_schema_tuple : common_schema {
    std::vector<common_schema_ptr> items;

    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_TUPLE; }
};

struct common_schema_property {
    std::string       name;
    common_schema_ptr schema;
    bool              required = false;
};

struct common_schema_object : common_schema {
    std::vector<common_schema_property> properties;             // in schema order
    common_schema_ptr                   additional_properties;  // null when not allowed

    common_schema_kind kind() const override { return COMMON_SCHEMA_KIND_OBJECT; }
};

// A parsed schema: its root, and the target of every $ref it reaches keyed by the $ref string
struct common_schema_document {
    common_schema_ptr                        root;
    std::map<std::string, common_schema_ptr> refs;
};

// Parses a JSON schema into a document.
// Throws std::runtime_error when the schema falls outside the supported subset.
common_schema_document common_schema_parse(const common_json & schema);

// Parses a schema that belongs to a document parsed earlier, e.g. one property of it.
// A $ref it cannot resolve on its own is looked up in doc.refs, the targets it resolves itself are added there.
// doc is unchanged when the schema is rejected.
common_schema_ptr common_schema_parse(const common_json & schema, common_schema_document & doc);

// Rewrites a document in place into an equivalent one with less redundancy: allOf becomes the intersection
// of its children, nested anyOf are flattened, branches that can match nothing are pruned, up to a
// common_schema_none root, and $refs nothing reaches anymore are dropped.
// An allOf stays where the children cannot be combined, e.g. two different patterns or a const against a type.
void common_schema_optimize(common_schema_document & doc);

// A set of the kinds of value a schema may match: only the value kinds NULL to OBJECT occur, a tuple counts as an array
class common_schema_kinds {
    uint32_t mask_ = 0;

  public:
    common_schema_kinds() = default;
    common_schema_kinds(std::initializer_list<common_schema_kind> kinds) {
        for (auto kind : kinds) {
            add(kind);
        }
    }

    static common_schema_kinds all() {
        return { COMMON_SCHEMA_KIND_NULL,   COMMON_SCHEMA_KIND_BOOLEAN, COMMON_SCHEMA_KIND_NUMBER, COMMON_SCHEMA_KIND_INTEGER,
                 COMMON_SCHEMA_KIND_STRING, COMMON_SCHEMA_KIND_ARRAY,   COMMON_SCHEMA_KIND_OBJECT };
    }

    void add(common_schema_kind kind) { mask_ |= 1u << kind; }

    bool has(common_schema_kind kind) const { return (mask_ & (1u << kind)) != 0; }
    bool is_only(common_schema_kind kind) const { return mask_ == (1u << kind); }
    bool empty() const { return mask_ == 0; }

    common_schema_kinds & operator|=(const common_schema_kinds & other) { mask_ |= other.mask_; return *this; }
    common_schema_kinds & operator&=(const common_schema_kinds & other) { mask_ &= other.mask_; return *this; }

    bool operator==(const common_schema_kinds & other) const { return mask_ == other.mask_; }
    bool operator!=(const common_schema_kinds & other) const { return mask_ != other.mask_; }
};

// The kinds of value matching the schema: the union over anyOf, the intersection over allOf, every kind for an any.
// A number schema accepts integers too, so it resolves to both.
common_schema_kinds common_schema_resolve_kinds(const common_schema & schema);
