#include "json.h"
#include "testing.h"

using json       = common_json;
using json_value = common_json_value;

static void test_bool(testing & t);
static void test_enum(testing & t);
static void test_integer(testing & t);

int main(int /*argc*/, char * /*argv*/[]) {
    testing t;
    t.verbose = true;

    t.test("bool", test_bool);
    t.test("enum", test_enum);
    t.test("integer", test_integer);

    return t.summary();
}

template <typename T>
static void check_value(testing &           t,
                        const std::string & type_name,
                        T                   value,
                        const std::string & json_type_name,
                        bool (*json_check_func)(const json &)) {
    json cast_value = { json_value(value) };
    t.assert_true(type_name + " cast to value is " + json_type_name, json_check_func(cast_value));

    json bare_value = { value };
    t.assert_true(type_name + " is " + json_type_name, json_check_func(bare_value));

    json object = {
        { "v", value },
    };
    t.assert_true(type_name + " member is " + json_type_name, json_check_func(object["v"]));
}

void test_bool(testing & t) {
    check_value<bool>(t, "bool", false, "bool", [](const json & j) { return j.is_boolean(); });
}

template <typename T> static void check_integer(testing & t, const std::string & type_name, T value) {
    check_value<T>(t, type_name, value, "integer", [](const json & j) { return j.is_number_integer(); });
}

void test_enum(testing & t) {
    enum enum_type {
        ENUM_0,
    };

    check_integer<enum_type>(t, "enum", ENUM_0);
}

void test_integer(testing & t) {
    check_integer<int>(t, "int", 0);
    check_integer<int8_t>(t, "int8_t", 0);
    check_integer<int16_t>(t, "int16_t", 0);
    check_integer<int32_t>(t, "int32_t", 0);
    check_integer<int64_t>(t, "int64_t", 0);

    check_integer<unsigned int>(t, "unsigned int", 0);
    check_integer<uint8_t>(t, "uint8_t", 0);
    check_integer<uint16_t>(t, "uint16_t", 0);
    check_integer<uint32_t>(t, "uint32_t", 0);
    check_integer<uint64_t>(t, "uint64_t", 0);
}
