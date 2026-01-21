#include <string>
#include <iostream>
#include <random>
#include <cstdlib>

#include <nlohmann/json.hpp>
#include <sheredom/subprocess.h>

#include "jinja/runtime.h"
#include "jinja/parser.h"
#include "jinja/lexer.h"

#include "testing.h"

using json = nlohmann::ordered_json;

static void test_template(testing & t, const std::string & name, const std::string & tmpl, const json & vars, const std::string & expect);

static void test_whitespace_control(testing & t);
static void test_conditionals(testing & t);
static void test_loops(testing & t);
static void test_expressions(testing & t);
static void test_set_statement(testing & t);
static void test_filters(testing & t);
static void test_literals(testing & t);
static void test_comments(testing & t);
static void test_macros(testing & t);
static void test_namespace(testing & t);
static void test_tests(testing & t);
static void test_string_methods(testing & t);
static void test_array_methods(testing & t);
static void test_object_methods(testing & t);
static void test_fuzzing(testing & t);

static bool g_python_mode = false;

int main(int argc, char *argv[]) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-jinja [-py] [filter_regex]
    //  -py : enable python mode (use python jinja2 for rendering expected output)
    //        only use this for cross-checking, not for correctness
    //        note: the implementation of this flag is basic, only intented to be used by maintainers

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-py") {
            g_python_mode = true;
        } else {
            t.set_filter(arg);
        }
    }

    t.test("whitespace control", test_whitespace_control);
    t.test("conditionals", test_conditionals);
    t.test("loops", test_loops);
    t.test("expressions", test_expressions);
    t.test("set statement", test_set_statement);
    t.test("filters", test_filters);
    t.test("literals", test_literals);
    t.test("comments", test_comments);
    t.test("macros", test_macros);
    t.test("namespace", test_namespace);
    t.test("tests", test_tests);
    t.test("string methods", test_string_methods);
    t.test("array methods", test_array_methods);
    t.test("object methods", test_object_methods);
    if (!g_python_mode) {
        t.test("fuzzing", test_fuzzing);
    }

    return t.summary();
}

static void test_whitespace_control(testing & t) {
    test_template(t, "trim_blocks removes newline after tag",
        "{% if true %}\n"
        "hello\n"
        "{% endif %}\n",
        json::object(),
        "hello\n"
    );

    test_template(t, "lstrip_blocks removes leading whitespace",
        "    {% if true %}\n"
        "    hello\n"
        "    {% endif %}\n",
        json::object(),
        "    hello\n"
    );

    test_template(t, "for loop with trim_blocks",
        "{% for i in items %}\n"
        "{{ i }}\n"
        "{% endfor %}\n",
        {{"items", json::array({1, 2, 3})}},
        "1\n2\n3\n"
    );

    test_template(t, "explicit strip both",
        "  {%- if true -%}  \n"
        "hello\n"
        "  {%- endif -%}  \n",
        json::object(),
        "hello"
    );

    test_template(t, "expression whitespace control",
        "  {{- 'hello' -}}  \n",
        json::object(),
        "hello"
    );

    test_template(t, "inline block no newline",
        "{% if true %}yes{% endif %}",
        json::object(),
        "yes"
    );
}

static void test_conditionals(testing & t) {
    test_template(t, "if true",
        "{% if cond %}yes{% endif %}",
        {{"cond", true}},
        "yes"
    );

    test_template(t, "if false",
        "{% if cond %}yes{% endif %}",
        {{"cond", false}},
        ""
    );

    test_template(t, "if else",
        "{% if cond %}yes{% else %}no{% endif %}",
        {{"cond", false}},
        "no"
    );

    test_template(t, "if elif else",
        "{% if a %}A{% elif b %}B{% else %}C{% endif %}",
        {{"a", false}, {"b", true}},
        "B"
    );

    test_template(t, "nested if",
        "{% if outer %}{% if inner %}both{% endif %}{% endif %}",
        {{"outer", true}, {"inner", true}},
        "both"
    );

    test_template(t, "comparison operators",
        "{% if x > 5 %}big{% endif %}",
        {{"x", 10}},
        "big"
    );

    test_template(t, "logical and",
        "{% if a and b %}both{% endif %}",
        {{"a", true}, {"b", true}},
        "both"
    );

    test_template(t, "logical or",
        "{% if a or b %}either{% endif %}",
        {{"a", false}, {"b", true}},
        "either"
    );

    test_template(t, "logical not",
        "{% if not a %}negated{% endif %}",
        {{"a", false}},
        "negated"
    );

    test_template(t, "in operator",
        "{% if 'x' in items %}found{% endif %}",
        {{"items", json::array({"x", "y"})}},
        "found"
    );

    test_template(t, "is defined",
        "{% if x is defined %}yes{% else %}no{% endif %}",
        {{"x", 1}},
        "yes"
    );

    test_template(t, "is not defined",
        "{% if y is not defined %}yes{% else %}no{% endif %}",
        json::object(),
        "yes"
    );

    test_template(t, "is undefined falsy",
        "{{ 'yes' if not y else 'no' }}",
        json::object(),
        "yes"
    );

    test_template(t, "is undefined attribute falsy",
        "{{ 'yes' if not y.x else 'no' }}",
        {{"y", true}},
        "yes"
    );

    test_template(t, "is undefined key falsy",
        "{{ 'yes' if not y['x'] else 'no' }}",
        {{"y", {{}}}},
        "yes"
    );

    test_template(t, "is empty array falsy",
        "{{ 'yes' if not y else 'no' }}",
        {{"y", json::array()}},
        "yes"
    );

    test_template(t, "is empty object falsy",
        "{{ 'yes' if not y else 'no' }}",
        {{"y", json::object()}},
        "yes"
    );

    test_template(t, "is empty string falsy",
        "{{ 'yes' if not y else 'no' }}",
        {{"y", ""}},
        "yes"
    );

    test_template(t, "is 0 falsy",
        "{{ 'yes' if not y else 'no' }}",
        {{"y", 0}},
        "yes"
    );

    test_template(t, "is 0.0 falsy",
        "{{ 'yes' if not y else 'no' }}",
        {{"y", 0.0}},
        "yes"
    );

    test_template(t, "is non-empty array truthy",
        "{{ 'yes' if y else 'no' }}",
        {{"y", json::array({""})}},
        "yes"
    );

    test_template(t, "is non-empty object truthy",
        "{{ 'yes' if y else 'no' }}",
        {{"y", {"x", false}}},
        "yes"
    );

    test_template(t, "is non-empty string truthy",
        "{{ 'yes' if y else 'no' }}",
        {{"y", "0"}},
        "yes"
    );

    test_template(t, "is 1 truthy",
        "{{ 'yes' if y else 'no' }}",
        {{"y", 1}},
        "yes"
    );

    test_template(t, "is 1.0 truthy",
        "{{ 'yes' if y else 'no' }}",
        {{"y", 1.0}},
        "yes"
    );
}

static void test_loops(testing & t) {
    test_template(t, "simple for",
        "{% for i in items %}{{ i }}{% endfor %}",
        {{"items", json::array({1, 2, 3})}},
        "123"
    );

    test_template(t, "loop.index",
        "{% for i in items %}{{ loop.index }}{% endfor %}",
        {{"items", json::array({"a", "b", "c"})}},
        "123"
    );

    test_template(t, "loop.index0",
        "{% for i in items %}{{ loop.index0 }}{% endfor %}",
        {{"items", json::array({"a", "b", "c"})}},
        "012"
    );

    test_template(t, "loop.first and loop.last",
        "{% for i in items %}{% if loop.first %}[{% endif %}{{ i }}{% if loop.last %}]{% endif %}{% endfor %}",
        {{"items", json::array({1, 2, 3})}},
        "[123]"
    );

    test_template(t, "loop.length",
        "{% for i in items %}{{ loop.length }}{% endfor %}",
        {{"items", json::array({"a", "b"})}},
        "22"
    );

    test_template(t, "for over dict items",
        "{% for k, v in data.items() %}{{ k }}={{ v }} {% endfor %}",
        {{"data", {{"x", 1}, {"y", 2}}}},
        "x=1 y=2 "
    );

    test_template(t, "for else empty",
        "{% for i in items %}{{ i }}{% else %}empty{% endfor %}",
        {{"items", json::array()}},
        "empty"
    );

    test_template(t, "nested for",
        "{% for i in a %}{% for j in b %}{{ i }}{{ j }}{% endfor %}{% endfor %}",
        {{"a", json::array({1, 2})}, {"b", json::array({"x", "y"})}},
        "1x1y2x2y"
    );

    test_template(t, "for with range",
        "{% for i in range(3) %}{{ i }}{% endfor %}",
        json::object(),
        "012"
    );
}

static void test_expressions(testing & t) {
    test_template(t, "simple variable",
        "{{ x }}",
        {{"x", 42}},
        "42"
    );

    test_template(t, "dot notation",
        "{{ user.name }}",
        {{"user", {{"name", "Bob"}}}},
        "Bob"
    );

    test_template(t, "negative float (not dot notation)",
        "{{ -1.0 }}",
        json::object(),
        "-1.0"
    );

    test_template(t, "bracket notation",
        "{{ user['name'] }}",
        {{"user", {{"name", "Bob"}}}},
        "Bob"
    );

    test_template(t, "array access",
        "{{ items[1] }}",
        {{"items", json::array({"a", "b", "c"})}},
        "b"
    );

    test_template(t, "arithmetic",
        "{{ (a + b) * c }}",
        {{"a", 2}, {"b", 3}, {"c", 4}},
        "20"
    );

    test_template(t, "string concat ~",
        "{{ 'hello' ~ ' ' ~ 'world' }}",
        json::object(),
        "hello world"
    );

    test_template(t, "ternary",
        "{{ 'yes' if cond else 'no' }}",
        {{"cond", true}},
        "yes"
    );
}

static void test_set_statement(testing & t) {
    test_template(t, "simple set",
        "{% set x = 5 %}{{ x }}",
        json::object(),
        "5"
    );

    test_template(t, "set with expression",
        "{% set x = a + b %}{{ x }}",
        {{"a", 10}, {"b", 20}},
        "30"
    );

    test_template(t, "set list",
        "{% set items = [1, 2, 3] %}{{ items|length }}",
        json::object(),
        "3"
    );

    test_template(t, "set dict",
        "{% set d = {'a': 1} %}{{ d.a }}",
        json::object(),
        "1"
    );
}

static void test_filters(testing & t) {
    test_template(t, "upper",
        "{{ 'hello'|upper }}",
        json::object(),
        "HELLO"
    );

    test_template(t, "lower",
        "{{ 'HELLO'|lower }}",
        json::object(),
        "hello"
    );

    test_template(t, "capitalize",
        "{{ 'heLlo World'|capitalize }}",
        json::object(),
        "Hello world"
    );

    test_template(t, "title",
        "{{ 'hello world'|title }}",
        json::object(),
        "Hello World"
    );

    test_template(t, "trim",
        "{{ '  \r\n\thello\t\n\r  '|trim }}",
        json::object(),
        "hello"
    );

    test_template(t, "trim chars",
        "{{ 'xyxhelloxyx'|trim('xy') }}",
        json::object(),
        "hello"
    );

    test_template(t, "length string",
        "{{ 'hello'|length }}",
        json::object(),
        "5"
    );

    test_template(t, "replace",
        "{{ 'hello world'|replace('world', 'jinja') }}",
        json::object(),
        "hello jinja"
    );

    test_template(t, "length list",
        "{{ items|length }}",
        {{"items", json::array({1, 2, 3})}},
        "3"
    );

    test_template(t, "first",
        "{{ items|first }}",
        {{"items", json::array({10, 20, 30})}},
        "10"
    );

    test_template(t, "last",
        "{{ items|last }}",
        {{"items", json::array({10, 20, 30})}},
        "30"
    );

    test_template(t, "reverse",
        "{% for i in items|reverse %}{{ i }}{% endfor %}",
        {{"items", json::array({1, 2, 3})}},
        "321"
    );

    test_template(t, "sort",
        "{% for i in items|sort %}{{ i }}{% endfor %}",
        {{"items", json::array({3, 1, 2})}},
        "123"
    );

    test_template(t, "sort reverse",
        "{% for i in items|sort(true) %}{{ i }}{% endfor %}",
        {{"items", json::array({3, 1, 2})}},
        "321"
    );

    test_template(t, "sort with attribute",
        "{{ items|sort(attribute='name')|join(attribute='age') }}",
        {{"items", json::array({
            json({{"name", "c"}, {"age", 3}}),
            json({{"name", "a"}, {"age", 1}}),
            json({{"name", "b"}, {"age", 2}}),
        })}},
        "123"
    );

    test_template(t, "sort with numeric attribute",
        "{{ items|sort(attribute=0)|join(attribute=1) }}",
        {{"items", json::array({
            json::array({3, "z"}),
            json::array({1, "x"}),
            json::array({2, "y"}),
        })}},
        "xyz"
    );

    test_template(t, "join",
        "{{ items|join(', ') }}",
        {{"items", json::array({"a", "b", "c"})}},
        "a, b, c"
    );

    test_template(t, "join default separator",
        "{{ items|join }}",
        {{"items", json::array({"x", "y", "z"})}},
        "xyz"
    );

    test_template(t, "abs",
        "{{ -5|abs }}",
        json::object(),
        "5"
    );

    test_template(t, "int from string",
        "{{ '42'|int }}",
        json::object(),
        "42"
    );

    test_template(t, "int from string with default",
        "{{ ''|int(1) }}",
        json::object(),
        "1"
    );

    test_template(t, "int from string with base",
        "{{ '11'|int(base=2) }}",
        json::object(),
        "3"
    );

    test_template(t, "float from string",
        "{{ '3.14'|float }}",
        json::object(),
        "3.14"
    );

    test_template(t, "default with value",
        "{{ x|default('fallback') }}",
        {{"x", "actual"}},
        "actual"
    );

    test_template(t, "default without value",
        "{{ y|default('fallback') }}",
        json::object(),
        "fallback"
    );

    test_template(t, "default with falsy value",
        "{{ ''|default('fallback', true) }}",
        json::object(),
        "fallback"
    );

    test_template(t, "tojson ensure_ascii=true",
        "{{ data|tojson(ensure_ascii=true) }}",
        {{"data", "\u2713"}},
        "\"\\u2713\""
    );

    test_template(t, "tojson sort_keys=true",
        "{{ data|tojson(sort_keys=true) }}",
        {{"data", {{"b", 2}, {"a", 1}}}},
        "{\"a\": 1, \"b\": 2}"
    );

    test_template(t, "tojson",
        "{{ data|tojson }}",
        {{"data", {{"a", 1}, {"b", json::array({1, 2})}}}},
        "{\"a\": 1, \"b\": [1, 2]}"
    );

    test_template(t, "tojson indent=4",
        "{{ data|tojson(indent=4) }}",
        {{"data", {{"a", 1}, {"b", json::array({1, 2})}}}},
        "{\n    \"a\": 1,\n    \"b\": [\n        1,\n        2\n    ]\n}"
    );

    test_template(t, "tojson separators=(',',':')",
        "{{ data|tojson(separators=(',',':')) }}",
        {{"data", {{"a", 1}, {"b", json::array({1, 2})}}}},
        "{\"a\":1,\"b\":[1,2]}"
    );

    test_template(t, "tojson separators=(',',': ') indent=2",
        "{{ data|tojson(separators=(',',': '), indent=2) }}",
        {{"data", {{"a", 1}, {"b", json::array({1, 2})}}}},
        "{\n  \"a\": 1,\n  \"b\": [\n    1,\n    2\n  ]\n}"
    );

    test_template(t, "chained filters",
        "{{ '  HELLO  '|trim|lower }}",
        json::object(),
        "hello"
    );
}

static void test_literals(testing & t) {
    test_template(t, "integer",
        "{{ 42 }}",
        json::object(),
        "42"
    );

    test_template(t, "float",
        "{{ 3.14 }}",
        json::object(),
        "3.14"
    );

    test_template(t, "string",
        "{{ 'hello' }}",
        json::object(),
        "hello"
    );

    test_template(t, "boolean true",
        "{{ true }}",
        json::object(),
        "True"
    );

    test_template(t, "boolean false",
        "{{ false }}",
        json::object(),
        "False"
    );

    test_template(t, "none",
        "{% if x is none %}null{% endif %}",
        {{"x", nullptr}},
        "null"
    );

    test_template(t, "list literal",
        "{% for i in [1, 2, 3] %}{{ i }}{% endfor %}",
        json::object(),
        "123"
    );

    test_template(t, "dict literal",
        "{% set d = {'a': 1} %}{{ d.a }}",
        json::object(),
        "1"
    );

    test_template(t, "integer|abs",
        "{{ -42 | abs }}",
        json::object(),
        "42"
    );

    test_template(t, "integer|float",
        "{{ 42 | float }}",
        json::object(),
        "42.0"
    );

    test_template(t, "integer|tojson",
        "{{ 42 | tojson }}",
        json::object(),
        "42"
    );

    test_template(t, "float|abs",
        "{{ -3.14 | abs }}",
        json::object(),
        "3.14"
    );

    test_template(t, "float|int",
        "{{ 3.14 | int }}",
        json::object(),
        "3"
    );

    test_template(t, "float|tojson",
        "{{ 3.14 | tojson }}",
        json::object(),
        "3.14"
    );

    test_template(t, "string|tojson",
        "{{ 'hello' | tojson }}",
        json::object(),
        "\"hello\""
    );

    test_template(t, "boolean|int",
        "{{ true | int }}",
        json::object(),
        "1"
    );

    test_template(t, "boolean|float",
        "{{ true | float }}",
        json::object(),
        "1.0"
    );

    test_template(t, "boolean|tojson",
        "{{ true | tojson }}",
        json::object(),
        "true"
    );
}

static void test_comments(testing & t) {
    test_template(t, "inline comment",
        "before{# comment #}after",
        json::object(),
        "beforeafter"
    );

    test_template(t, "comment ignores code",
        "{% set x = 1 %}{# {% set x = 999 %} #}{{ x }}",
        json::object(),
        "1"
    );
}

static void test_macros(testing & t) {
    test_template(t, "simple macro",
        "{% macro greet(name) %}Hello {{ name }}{% endmacro %}{{ greet('World') }}",
        json::object(),
        "Hello World"
    );

    test_template(t, "macro default arg",
        "{% macro greet(name='Guest') %}Hi {{ name }}{% endmacro %}{{ greet() }}",
        json::object(),
        "Hi Guest"
    );
}

static void test_namespace(testing & t) {
    test_template(t, "namespace counter",
        "{% set ns = namespace(count=0) %}{% for i in range(3) %}{% set ns.count = ns.count + 1 %}{% endfor %}{{ ns.count }}",
        json::object(),
        "3"
    );
}

static void test_tests(testing & t) {
    test_template(t, "is odd",
        "{% if 3 is odd %}yes{% endif %}",
        json::object(),
        "yes"
    );

    test_template(t, "is even",
        "{% if 4 is even %}yes{% endif %}",
        json::object(),
        "yes"
    );

    test_template(t, "is false",
        "{{ 'yes' if x is false }}",
        {{"x", false}},
        "yes"
    );

    test_template(t, "is true",
        "{{ 'yes' if x is true }}",
        {{"x", true}},
        "yes"
    );

    test_template(t, "string is false",
        "{{ 'yes' if x is false else 'no' }}",
        {{"x", ""}},
        "no"
    );

    test_template(t, "is divisibleby",
        "{{ 'yes' if x is divisibleby(2) }}",
        {{"x", 2}},
        "yes"
    );

    test_template(t, "is eq",
        "{{ 'yes' if 3 is eq(3) }}",
        json::object(),
        "yes"
    );

    test_template(t, "is not equalto",
        "{{ 'yes' if 3 is not equalto(4) }}",
        json::object(),
        "yes"
    );

    test_template(t, "is ge",
        "{{ 'yes' if 3 is ge(3) }}",
        json::object(),
        "yes"
    );

    test_template(t, "is gt",
        "{{ 'yes' if 3 is gt(2) }}",
        json::object(),
        "yes"
    );

    test_template(t, "is greaterthan",
        "{{ 'yes' if 3 is greaterthan(2) }}",
        json::object(),
        "yes"
    );

    test_template(t, "is lt",
        "{{ 'yes' if 2 is lt(3) }}",
        json::object(),
        "yes"
    );

    test_template(t, "is lessthan",
        "{{ 'yes' if 2 is lessthan(3) }}",
        json::object(),
        "yes"
    );

    test_template(t, "is ne",
        "{{ 'yes' if 2 is ne(3) }}",
        json::object(),
        "yes"
    );

    test_template(t, "is lower",
        "{{ 'yes' if 'lowercase' is lower }}",
        json::object(),
        "yes"
    );

    test_template(t, "is upper",
        "{{ 'yes' if 'UPPERCASE' is upper }}",
        json::object(),
        "yes"
    );

    test_template(t, "is sameas",
        "{{ 'yes' if x is sameas(false) }}",
        {{"x", false}},
        "yes"
    );

    test_template(t, "is boolean",
        "{{ 'yes' if x is boolean }}",
        {{"x", true}},
        "yes"
    );

    test_template(t, "is callable",
        "{{ 'yes' if ''.strip is callable }}",
        json::object(),
        "yes"
    );

    test_template(t, "is escaped",
        "{{ 'yes' if 'foo'|safe is escaped }}",
        json::object(),
        "yes"
    );

    test_template(t, "is filter",
        "{{ 'yes' if 'trim' is filter }}",
        json::object(),
        "yes"
    );

    test_template(t, "is float",
        "{{ 'yes' if x is float }}",
        {{"x", 1.1}},
        "yes"
    );

    test_template(t, "is integer",
        "{{ 'yes' if x is integer }}",
        {{"x", 1}},
        "yes"
    );

    test_template(t, "is sequence",
        "{{ 'yes' if x is sequence }}",
        {{"x", json::array({1, 2, 3})}},
        "yes"
    );

    test_template(t, "is test",
        "{{ 'yes' if 'sequence' is test }}",
        json::object(),
        "yes"
    );

    test_template(t, "is undefined",
        "{{ 'yes' if x is undefined }}",
        json::object(),
        "yes"
    );

    test_template(t, "is none",
        "{% if x is none %}yes{% endif %}",
        {{"x", nullptr}},
        "yes"
    );

    test_template(t, "is string",
        "{% if x is string %}yes{% endif %}",
        {{"x", "hello"}},
        "yes"
    );

    test_template(t, "is number",
        "{% if x is number %}yes{% endif %}",
        {{"x", 42}},
        "yes"
    );

    test_template(t, "is iterable",
        "{% if x is iterable %}yes{% endif %}",
        {{"x", json::array({1, 2, 3})}},
        "yes"
    );

    test_template(t, "is mapping",
        "{% if x is mapping %}yes{% endif %}",
        {{"x", {{"a", 1}}}},
        "yes"
    );
}

static void test_string_methods(testing & t) {
    test_template(t, "string.upper()",
        "{{ s.upper() }}",
        {{"s", "hello"}},
        "HELLO"
    );

    test_template(t, "string.lower()",
        "{{ s.lower() }}",
        {{"s", "HELLO"}},
        "hello"
    );

    test_template(t, "string.strip()",
        "[{{ s.strip() }}]",
        {{"s", "  hello  "}},
        "[hello]"
    );

    test_template(t, "string.lstrip()",
        "[{{ s.lstrip() }}]",
        {{"s", "   hello"}},
        "[hello]"
    );

    test_template(t, "string.rstrip()",
        "[{{ s.rstrip() }}]",
        {{"s", "hello   "}},
        "[hello]"
    );

    test_template(t, "string.title()",
        "{{ s.title() }}",
        {{"s", "hello world"}},
        "Hello World"
    );

    test_template(t, "string.capitalize()",
        "{{ s.capitalize() }}",
        {{"s", "heLlo World"}},
        "Hello world"
    );

    test_template(t, "string.startswith() true",
        "{% if s.startswith('hel') %}yes{% endif %}",
        {{"s", "hello"}},
        "yes"
    );

    test_template(t, "string.startswith() false",
        "{% if s.startswith('xyz') %}yes{% else %}no{% endif %}",
        {{"s", "hello"}},
        "no"
    );

    test_template(t, "string.endswith() true",
        "{% if s.endswith('lo') %}yes{% endif %}",
        {{"s", "hello"}},
        "yes"
    );

    test_template(t, "string.endswith() false",
        "{% if s.endswith('xyz') %}yes{% else %}no{% endif %}",
        {{"s", "hello"}},
        "no"
    );

    test_template(t, "string.split() with sep",
        "{{ s.split(',')|join('-') }}",
        {{"s", "a,b,c"}},
        "a-b-c"
    );

    test_template(t, "string.split() with maxsplit",
        "{{ s.split(',', 1)|join('-') }}",
        {{"s", "a,b,c"}},
        "a-b,c"
    );

    test_template(t, "string.rsplit() with sep",
        "{{ s.rsplit(',')|join('-') }}",
        {{"s", "a,b,c"}},
        "a-b-c"
    );

    test_template(t, "string.rsplit() with maxsplit",
        "{{ s.rsplit(',', 1)|join('-') }}",
        {{"s", "a,b,c"}},
        "a,b-c"
    );

    test_template(t, "string.replace() basic",
        "{{ s.replace('world', 'jinja') }}",
        {{"s", "hello world"}},
        "hello jinja"
    );

    test_template(t, "string.replace() with count",
        "{{ s.replace('a', 'X', 2) }}",
        {{"s", "banana"}},
        "bXnXna"
    );
}

static void test_array_methods(testing & t) {
    test_template(t, "array|selectattr by attribute",
        "{% for item in items|selectattr('active') %}{{ item.name }} {% endfor %}",
        {{"items", json::array({
            {{"name", "a"}, {"active", true}},
            {{"name", "b"}, {"active", false}},
            {{"name", "c"}, {"active", true}}
        })}},
        "a c "
    );

    test_template(t, "array|selectattr with operator",
        "{% for item in items|selectattr('value', 'equalto', 5) %}{{ item.name }} {% endfor %}",
        {{"items", json::array({
            {{"name", "a"}, {"value", 3}},
            {{"name", "b"}, {"value", 5}},
            {{"name", "c"}, {"value", 5}}
        })}},
        "b c "
    );

    test_template(t, "array|tojson",
        "{{ arr|tojson }}",
        {{"arr", json::array({1, 2, 3})}},
        "[1, 2, 3]"
    );

    test_template(t, "array|tojson with strings",
        "{{ arr|tojson }}",
        {{"arr", json::array({"a", "b", "c"})}},
        "[\"a\", \"b\", \"c\"]"
    );

    test_template(t, "array|tojson nested",
        "{{ arr|tojson }}",
        {{"arr", json::array({json::array({1, 2}), json::array({3, 4})})}},
        "[[1, 2], [3, 4]]"
    );

    test_template(t, "array|last",
        "{{ arr|last }}",
        {{"arr", json::array({10, 20, 30})}},
        "30"
    );

    test_template(t, "array|last single element",
        "{{ arr|last }}",
        {{"arr", json::array({42})}},
        "42"
    );

    test_template(t, "array|join with separator",
        "{{ arr|join(', ') }}",
        {{"arr", json::array({"a", "b", "c"})}},
        "a, b, c"
    );

    test_template(t, "array|join with custom separator",
        "{{ arr|join(' | ') }}",
        {{"arr", json::array({1, 2, 3})}},
        "1 | 2 | 3"
    );

    test_template(t, "array|join default separator",
        "{{ arr|join }}",
        {{"arr", json::array({"x", "y", "z"})}},
        "xyz"
    );

    test_template(t, "array|join attribute",
        "{{ arr|join(attribute='age') }}",
        {{"arr", json::array({
            json({{"name", "a"}, {"age", 1}}),
            json({{"name", "b"}, {"age", 2}}),
            json({{"name", "c"}, {"age", 3}}),
        })}},
        "123"
    );

    test_template(t, "array|join numeric attribute",
        "{{ arr|join(attribute=-1) }}",
        {{"arr", json::array({json::array({1}), json::array({2}), json::array({3})})}},
        "123"
    );

    test_template(t, "array.pop() last",
        "{{ arr.pop() }}-{{ arr|join(',') }}",
        {{"arr", json::array({"a", "b", "c"})}},
        "c-a,b"
    );

    test_template(t, "array.pop() with index",
        "{{ arr.pop(0) }}-{{ arr|join(',') }}",
        {{"arr", json::array({"a", "b", "c"})}},
        "a-b,c"
    );

    test_template(t, "array.append()",
        "{% set _ = arr.append('d') %}{{ arr|join(',') }}",
        {{"arr", json::array({"a", "b", "c"})}},
        "a,b,c,d"
    );

    test_template(t, "array|map with attribute",
        "{% for v in arr|map(attribute='age') %}{{ v }} {% endfor %}",
        {{"arr", json::array({
            json({{"name", "a"}, {"age", 1}}),
            json({{"name", "b"}, {"age", 2}}),
            json({{"name", "c"}, {"age", 3}}),
        })}},
        "1 2 3 "
    );

    test_template(t, "array|map with attribute default",
        "{% for v in arr|map(attribute='age', default=3) %}{{ v }} {% endfor %}",
        {{"arr", json::array({
            json({{"name", "a"}, {"age", 1}}),
            json({{"name", "b"}, {"age", 2}}),
            json({{"name", "c"}}),
        })}},
        "1 2 3 "
    );

    test_template(t, "array|map without attribute default",
        "{% for v in arr|map(attribute='age') %}{{ v }} {% endfor %}",
        {{"arr", json::array({
            json({{"name", "a"}, {"age", 1}}),
            json({{"name", "b"}, {"age", 2}}),
            json({{"name", "c"}}),
        })}},
        "1 2  "
    );

    test_template(t, "array|map with numeric attribute",
        "{% for v in arr|map(attribute=0) %}{{ v }} {% endfor %}",
        {{"arr", json::array({
            json::array({10, "x"}),
            json::array({20, "y"}),
            json::array({30, "z"}),
        })}},
        "10 20 30 "
    );

    test_template(t, "array|map with negative attribute",
        "{% for v in arr|map(attribute=-1) %}{{ v }} {% endfor %}",
        {{"arr", json::array({
            json::array({10, "x"}),
            json::array({20, "y"}),
            json::array({30, "z"}),
        })}},
        "x y z "
    );

    test_template(t, "array|map with filter",
        "{{ arr|map('int')|sum }}",
        {{"arr", json::array({"1", "2", "3"})}},
        "6"
    );

    // not used by any chat templates
    // test_template(t, "array.insert()",
    //     "{% set _ = arr.insert(1, 'x') %}{{ arr|join(',') }}",
    //     {{"arr", json::array({"a", "b", "c"})}},
    //     "a,x,b,c"
    // );
}

static void test_object_methods(testing & t) {
    test_template(t, "object.get() existing key",
        "{{ obj.get('a') }}",
        {{"obj", {{"a", 1}, {"b", 2}}}},
        "1"
    );

    test_template(t, "object.get() missing key",
        "[{{ obj.get('c') is none }}]",
        {{"obj", {{"a", 1}}}},
        "[True]"
    );

    test_template(t, "object.get() missing key with default",
        "{{ obj.get('c', 'default') }}",
        {{"obj", {{"a", 1}}}},
        "default"
    );

    test_template(t, "object.items()",
        "{% for k, v in obj.items() %}{{ k }}={{ v }} {% endfor %}",
        {{"obj", {{"x", 1}, {"y", 2}}}},
        "x=1 y=2 "
    );

    test_template(t, "object.keys()",
        "{% for k in obj.keys() %}{{ k }} {% endfor %}",
        {{"obj", {{"a", 1}, {"b", 2}}}},
        "a b "
    );

    test_template(t, "object.values()",
        "{% for v in obj.values() %}{{ v }} {% endfor %}",
        {{"obj", {{"a", 1}, {"b", 2}}}},
        "1 2 "
    );

    test_template(t, "dictsort ascending by key",
        "{% for k, v in obj|dictsort %}{{ k }}={{ v }} {% endfor %}",
        {{"obj", {{"z", 2}, {"a", 3}, {"m", 1}}}},
        "a=3 m=1 z=2 "
    );

    test_template(t, "dictsort descending by key",
        "{% for k, v in obj|dictsort(reverse=true) %}{{ k }}={{ v }} {% endfor %}",
        {{"obj", {{"a", 1}, {"b", 2}, {"c", 3}}}},
        "c=3 b=2 a=1 "
    );

    test_template(t, "dictsort by value",
        "{% for k, v in obj|dictsort(by='value') %}{{ k }}={{ v }} {% endfor %}",
        {{"obj", {{"a", 3}, {"b", 1}, {"c", 2}}}},
        "b=1 c=2 a=3 "
    );

    test_template(t, "dictsort case sensitive",
        "{% for k, v in obj|dictsort(case_sensitive=true) %}{{ k }}={{ v }} {% endfor %}",
        {{"obj", {{"a", 1}, {"A", 1}, {"b", 2}, {"B", 2}, {"c", 3}}}},
        "A=1 B=2 a=1 b=2 c=3 "
    );

    test_template(t, "object|tojson",
        "{{ obj|tojson }}",
        {{"obj", {{"name", "test"}, {"value", 42}}}},
        "{\"name\": \"test\", \"value\": 42}"
    );

    test_template(t, "nested object|tojson",
        "{{ obj|tojson }}",
        {{"obj", {{"outer", {{"inner", "value"}}}}}},
        "{\"outer\": {\"inner\": \"value\"}}"
    );

    test_template(t, "array in object|tojson",
        "{{ obj|tojson }}",
        {{"obj", {{"items", json::array({1, 2, 3})}}}},
        "{\"items\": [1, 2, 3]}"
    );

    test_template(t, "object attribute and key access",
        "{{ obj.keys()|join(',') }} vs {{ obj['keys'] }} vs {{ obj.test }}",
        {{"obj", {{"keys", "value"}, {"test", "attr_value"}}}},
        "keys,test vs value vs attr_value"
    );

    test_template(t, "env should not have object methods",
        "{{ keys is undefined }} {{ obj.keys is defined }}",
        {{"obj", {{"a", "b"}}}},
        "True True"
    );
}

static void test_template_cpp(testing & t, const std::string & name, const std::string & tmpl, const json & vars, const std::string & expect) {
    t.test(name, [&tmpl, &vars, &expect](testing & t) {
        jinja::lexer lexer;
        auto lexer_res = lexer.tokenize(tmpl);

        jinja::program ast = jinja::parse_from_tokens(lexer_res);

        jinja::context ctx(tmpl);
        jinja::global_from_json(ctx, vars, true);

        jinja::runtime runtime(ctx);

        try {
            const jinja::value results = runtime.execute(ast);
            auto parts = runtime.gather_string_parts(results);

            std::string rendered;
            for (const auto & part : parts->as_string().parts) {
                rendered += part.val;
            }

            if (!t.assert_true("Template render mismatch", expect == rendered)) {
                t.log("Template: " + json(tmpl).dump());
                t.log("Expected: " + json(expect).dump());
                t.log("Actual  : " + json(rendered).dump());
            }
        } catch (const jinja::not_implemented_exception & e) {
            // TODO @ngxson : remove this when the test framework supports skipping tests
            t.log("Skipped: " + std::string(e.what()));
        }
    });
}

// keep this in-sync with https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py
// note: we use SandboxedEnvironment instead of ImmutableSandboxedEnvironment to allow usage of in-place array methods like append() and pop()
static std::string py_script = R"(
import jinja2
import jinja2.ext as jinja2_ext
import json
import sys
from datetime import datetime
from jinja2.sandbox import SandboxedEnvironment

tmpl = json.loads(sys.argv[1])
vars_json = json.loads(sys.argv[2])

env = SandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=[jinja2_ext.loopcontrols],
)

def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)

env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
env.globals["strftime_now"] = lambda format: datetime.now().strftime(format)
env.globals["raise_exception"] = raise_exception

template = env.from_string(tmpl)
result = template.render(**vars_json)
print(result, end='')
)";

static void test_template_py(testing & t, const std::string & name, const std::string & tmpl, const json & vars, const std::string & expect) {
    t.test(name, [&tmpl, &vars, &expect](testing & t) {
        // Prepare arguments
        std::string tmpl_json = json(tmpl).dump();
        std::string vars_json = vars.dump();

#ifdef _WIN32
        const char * python_executable = "python.exe";
#else
        const char * python_executable = "python3";
#endif

        const char * command_line[] = {python_executable, "-c", py_script.c_str(), tmpl_json.c_str(), vars_json.c_str(), NULL};

        struct subprocess_s subprocess;
        int options = subprocess_option_combined_stdout_stderr
                    | subprocess_option_no_window
                    | subprocess_option_inherit_environment
                    | subprocess_option_search_user_path;
        int result = subprocess_create(command_line, options, &subprocess);

        if (result != 0) {
            t.log("Failed to create subprocess, error code: " + std::to_string(result));
            t.assert_true("subprocess creation", false);
            return;
        }

        // Read output
        std::string output;
        char buffer[1024];
        FILE * p_stdout = subprocess_stdout(&subprocess);
        while (fgets(buffer, sizeof(buffer), p_stdout)) {
            output += buffer;
        }

        int process_return;
        subprocess_join(&subprocess, &process_return);
        subprocess_destroy(&subprocess);

        if (process_return != 0) {
            t.log("Python script failed with exit code: " + std::to_string(process_return));
            t.log("Output: " + output);
            t.assert_true("python execution", false);
            return;
        }

        if (!t.assert_true("Template render mismatch", expect == output)) {
            t.log("Template: " + json(tmpl).dump());
            t.log("Expected: " + json(expect).dump());
            t.log("Python  : " + json(output).dump());
        }
    });
}

static void test_template(testing & t, const std::string & name, const std::string & tmpl, const json & vars, const std::string & expect) {
    if (g_python_mode) {
        test_template_py(t, name, tmpl, vars, expect);
    } else {
        test_template_cpp(t, name, tmpl, vars, expect);
    }
}

//
// fuzz tests to ensure no crashes occur on malformed inputs
//

constexpr int JINJA_FUZZ_ITERATIONS = 100;

// Helper to generate random string
static std::string random_string(std::mt19937 & rng, size_t max_len) {
    static const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
    std::uniform_int_distribution<size_t> len_dist(0, max_len);
    std::uniform_int_distribution<size_t> char_dist(0, sizeof(charset) - 2);
    size_t len = len_dist(rng);
    std::string result;
    result.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        result += charset[char_dist(rng)];
    }
    return result;
}

// Helper to execute a fuzz test case - returns true if no crash occurred
static bool fuzz_test_template(const std::string & tmpl, const json & vars) {
    try {
        // printf("Fuzz testing template: %s\n", tmpl.c_str());
        jinja::lexer lexer;
        auto lexer_res = lexer.tokenize(tmpl);
        jinja::program ast = jinja::parse_from_tokens(lexer_res);
        jinja::context ctx(tmpl);
        jinja::global_from_json(ctx, vars, true);
        jinja::runtime runtime(ctx);
        const jinja::value results = runtime.execute(ast);
        runtime.gather_string_parts(results);
        return true; // success
    } catch (const std::exception &) {
        return true; // exception is acceptable, not a crash
    } catch (...) {
        return true; // any exception is acceptable, not a crash
    }
}

static void test_fuzzing(testing & t) {
    const int num_iterations = JINJA_FUZZ_ITERATIONS;
    const unsigned int seed = 42; // fixed seed for reproducibility
    std::mt19937 rng(seed);

    // Distribution helpers
    std::uniform_int_distribution<int> choice_dist(0, 100);
    std::uniform_int_distribution<int> int_dist(-1000, 1000);
    std::uniform_int_distribution<size_t> idx_dist(0, 1000);

    // Template fragments for fuzzing
    const std::vector<std::string> var_names = {
        "x", "y", "z", "arr", "obj", "items", "foo", "bar", "undefined_var",
        "none", "true", "false", "None", "True", "False"
    };
    const std::vector<std::string> filters = {
        "length", "first", "last", "reverse", "sort", "unique", "join", "upper", "lower",
        "trim", "default", "tojson", "string", "int", "float", "abs", "list", "dictsort"
    };
    const std::vector<std::string> builtins = {
        "range", "len", "dict", "list", "join", "str", "int", "float", "namespace"
    };

    t.test("out of bound array access", [&](testing & t) {
        for (int i = 0; i < num_iterations; ++i) {
            int idx = int_dist(rng);
            std::string tmpl = "{{ arr[" + std::to_string(idx) + "] }}";
            json vars = {{"arr", json::array({1, 2, 3})}};
            t.assert_true("should not crash", fuzz_test_template(tmpl, vars));
        }
    });

    t.test("non-existing variables", [&](testing & t) {
        for (int i = 0; i < num_iterations; ++i) {
            std::string var = random_string(rng, 20);
            std::string tmpl = "{{ " + var + " }}";
            json vars = json::object(); // empty context
            t.assert_true("should not crash", fuzz_test_template(tmpl, vars));
        }
    });

    t.test("non-existing nested attributes", [&](testing & t) {
        for (int i = 0; i < num_iterations; ++i) {
            std::string var1 = var_names[choice_dist(rng) % var_names.size()];
            std::string var2 = random_string(rng, 10);
            std::string var3 = random_string(rng, 10);
            std::string tmpl = "{{ " + var1 + "." + var2 + "." + var3 + " }}";
            json vars = {{var1, {{"other", 123}}}};
            t.assert_true("should not crash", fuzz_test_template(tmpl, vars));
        }
    });

    t.test("invalid filter arguments", [&](testing & t) {
        for (int i = 0; i < num_iterations; ++i) {
            std::string filter = filters[choice_dist(rng) % filters.size()];
            int val = int_dist(rng);
            std::string tmpl = "{{ " + std::to_string(val) + " | " + filter + " }}";
            json vars = json::object();
            t.assert_true("should not crash", fuzz_test_template(tmpl, vars));
        }
    });

    t.test("chained filters on various types", [&](testing & t) {
        for (int i = 0; i < num_iterations; ++i) {
            std::string f1 = filters[choice_dist(rng) % filters.size()];
            std::string f2 = filters[choice_dist(rng) % filters.size()];
            std::string var = var_names[choice_dist(rng) % var_names.size()];
            std::string tmpl = "{{ " + var + " | " + f1 + " | " + f2 + " }}";
            json vars = {
                {"x", 42},
                {"y", "hello"},
                {"arr", json::array({1, 2, 3})},
                {"obj", {{"a", 1}, {"b", 2}}},
                {"items", json::array({"a", "b", "c"})}
            };
            t.assert_true("should not crash", fuzz_test_template(tmpl, vars));
        }
    });

    t.test("invalid builtin calls", [&](testing & t) {
        for (int i = 0; i < num_iterations; ++i) {
            std::string builtin = builtins[choice_dist(rng) % builtins.size()];
            std::string arg;
            int arg_type = choice_dist(rng) % 4;
            switch (arg_type) {
                case 0: arg = "\"not a number\""; break;
                case 1: arg = "none"; break;
                case 2: arg = std::to_string(int_dist(rng)); break;
                case 3: arg = "[]"; break;
            }
            std::string tmpl = "{{ " + builtin + "(" + arg + ") }}";
            json vars = json::object();
            t.assert_true("should not crash", fuzz_test_template(tmpl, vars));
        }
    });

    t.test("macro edge cases", [&](testing & t) {
        // Macro with no args called with args
        t.assert_true("macro no args with args", fuzz_test_template(
            "{% macro foo() %}hello{% endmacro %}{{ foo(1, 2, 3) }}",
            json::object()
        ));

        // Macro with args called with no args
        t.assert_true("macro with args no args", fuzz_test_template(
            "{% macro foo(a, b, c) %}{{ a }}{{ b }}{{ c }}{% endmacro %}{{ foo() }}",
            json::object()
        ));

        // Recursive macro reference
        t.assert_true("recursive macro", fuzz_test_template(
            "{% macro foo(n) %}{% if n > 0 %}{{ foo(n - 1) }}{% endif %}{% endmacro %}{{ foo(5) }}",
            json::object()
        ));

        // Nested macro definitions
        for (int i = 0; i < num_iterations / 10; ++i) {
            std::string tmpl = "{% macro outer() %}{% macro inner() %}x{% endmacro %}{{ inner() }}{% endmacro %}{{ outer() }}";
            t.assert_true("nested macro", fuzz_test_template(tmpl, json::object()));
        }
    });

    t.test("empty and none operations", [&](testing & t) {
        const std::vector<std::string> empty_tests = {
            "{{ \"\" | first }}",
            "{{ \"\" | last }}",
            "{{ [] | first }}",
            "{{ [] | last }}",
            "{{ none.attr }}",
            "{{ none | length }}",
            "{{ none | default('fallback') }}",
            "{{ {} | first }}",
            "{{ {} | dictsort }}",
        };
        for (const auto & tmpl : empty_tests) {
            t.assert_true("empty/none: " + tmpl, fuzz_test_template(tmpl, json::object()));
        }
    });

    t.test("arithmetic edge cases", [&](testing & t) {
        const std::vector<std::string> arith_tests = {
            "{{ 1 / 0 }}",
            "{{ 1 // 0 }}",
            "{{ 1 % 0 }}",
            "{{ 999999999999999999 * 999999999999999999 }}",
            "{{ -999999999999999999 - 999999999999999999 }}",
            "{{ 1.0 / 0.0 }}",
            "{{ 0.0 / 0.0 }}",
        };
        for (const auto & tmpl : arith_tests) {
            t.assert_true("arith: " + tmpl, fuzz_test_template(tmpl, json::object()));
        }
    });

    t.test("deeply nested structures", [&](testing & t) {
        // Deeply nested loops
        for (int depth = 1; depth <= 10; ++depth) {
            std::string tmpl;
            for (int d = 0; d < depth; ++d) {
                tmpl += "{% for i" + std::to_string(d) + " in arr %}";
            }
            tmpl += "x";
            for (int d = 0; d < depth; ++d) {
                tmpl += "{% endfor %}";
            }
            json vars = {{"arr", json::array({1, 2})}};
            t.assert_true("nested loops depth " + std::to_string(depth), fuzz_test_template(tmpl, vars));
        }

        // Deeply nested conditionals
        for (int depth = 1; depth <= 10; ++depth) {
            std::string tmpl;
            for (int d = 0; d < depth; ++d) {
                tmpl += "{% if true %}";
            }
            tmpl += "x";
            for (int d = 0; d < depth; ++d) {
                tmpl += "{% endif %}";
            }
            t.assert_true("nested ifs depth " + std::to_string(depth), fuzz_test_template(tmpl, json::object()));
        }
    });

    t.test("special characters in strings", [&](testing & t) {
        const std::vector<std::string> special_tests = {
            "{{ \"}{%\" }}",
            "{{ \"}}{{\" }}",
            "{{ \"{%%}\" }}",
            "{{ \"\\n\\t\\r\" }}",
            "{{ \"'\\\"'\" }}",
            "{{ \"hello\\x00world\" }}",
        };
        for (const auto & tmpl : special_tests) {
            t.assert_true("special: " + tmpl, fuzz_test_template(tmpl, json::object()));
        }
    });

    t.test("random template generation", [&](testing & t) {
        const std::vector<std::string> fragments = {
            "{{ x }}", "{{ y }}", "{{ arr }}", "{{ obj }}",
            "{% if true %}a{% endif %}",
            "{% if false %}b{% else %}c{% endif %}",
            "{% for i in arr %}{{ i }}{% endfor %}",
            "{{ x | length }}", "{{ x | first }}", "{{ x | default(0) }}",
            "{{ x + y }}", "{{ x - y }}", "{{ x * y }}",
            "{{ x == y }}", "{{ x != y }}", "{{ x > y }}",
            "{{ range(3) }}", "{{ \"hello\" | upper }}",
            "text", " ", "\n",
        };

        for (int i = 0; i < num_iterations; ++i) {
            std::string tmpl;
            int num_frags = choice_dist(rng) % 10 + 1;
            for (int f = 0; f < num_frags; ++f) {
                tmpl += fragments[choice_dist(rng) % fragments.size()];
            }
            json vars = {
                {"x", int_dist(rng)},
                {"y", int_dist(rng)},
                {"arr", json::array({1, 2, 3})},
                {"obj", {{"a", 1}, {"b", 2}}}
            };
            t.assert_true("random template #" + std::to_string(i), fuzz_test_template(tmpl, vars));
        }
    });

    t.test("malformed templates (should error, not crash)", [&](testing & t) {
        const std::vector<std::string> malformed = {
            "{{ x",
            "{% if %}",
            "{% for %}",
            "{% for x in %}",
            "{% endfor %}",
            "{% endif %}",
            "{{ | filter }}",
            "{% if x %}", // unclosed
            "{% for i in x %}", // unclosed
            "{{ x | }}",
            "{% macro %}{% endmacro %}",
            "{{{{",
            "}}}}",
            "{%%}",
            "{% set %}",
            "{% set x %}",
        };
        for (const auto & tmpl : malformed) {
            t.assert_true("malformed: " + tmpl, fuzz_test_template(tmpl, json::object()));
        }
    });

    t.test("type coercion edge cases", [&](testing & t) {
        for (int i = 0; i < num_iterations; ++i) {
            int op_choice = choice_dist(rng) % 6;
            std::string op;
            switch (op_choice) {
                case 0: op = "+"; break;
                case 1: op = "-"; break;
                case 2: op = "*"; break;
                case 3: op = "/"; break;
                case 4: op = "=="; break;
                case 5: op = "~"; break; // string concat
            }

            std::string left_var = var_names[choice_dist(rng) % var_names.size()];
            std::string right_var = var_names[choice_dist(rng) % var_names.size()];
            std::string tmpl = "{{ " + left_var + " " + op + " " + right_var + " }}";

            json vars = {
                {"x", 42},
                {"y", "hello"},
                {"z", 3.14},
                {"arr", json::array({1, 2, 3})},
                {"obj", {{"a", 1}}},
                {"items", json::array()},
                {"foo", nullptr},
                {"bar", true}
            };
            t.assert_true("type coercion: " + tmpl, fuzz_test_template(tmpl, vars));
        }
    });

    t.test("fuzz builtin functions", [&](testing & t) {
        // pair of (type_name, builtin_name)
        std::vector<std::pair<std::string, std::string>> builtins;
        auto add_fns = [&](std::string type_name, const jinja::func_builtins & added) {
            for (const auto & it : added) {
                builtins.push_back({type_name, it.first});
            }
        };
        add_fns("global", jinja::global_builtins());
        add_fns("int",    jinja::value_int_t(0).get_builtins());
        add_fns("float",  jinja::value_float_t(0.0f).get_builtins());
        add_fns("string", jinja::value_string_t().get_builtins());
        add_fns("array",  jinja::value_array_t().get_builtins());
        add_fns("object", jinja::value_object_t().get_builtins());

        const int max_args = 5;
        const std::vector<std::string> kwarg_names = {
            "base", "attribute", "default", "reverse", "case_sensitive", "by", "safe", "chars", "separators", "sort_keys", "indent", "ensure_ascii",
        };

        // Generate random argument values
        auto gen_random_arg = [&]() -> std::string {
            int type = choice_dist(rng) % 8;
            switch (type) {
                case 0: return std::to_string(int_dist(rng));           // int
                case 1: return std::to_string(int_dist(rng)) + ".5";    // float
                case 2: return "\"" + random_string(rng, 10) + "\"";    // string
                case 3: return "true";                                   // bool true
                case 4: return "false";                                  // bool false
                case 5: return "none";                                   // none
                case 6: return "[1, 2, 3]";                              // array
                case 7: return "{\"a\": 1}";                             // object
                default: return "0";
            }
        };

        for (int i = 0; i < num_iterations; ++i) {
            // Pick a random builtin
            auto & [type_name, fn_name] = builtins[choice_dist(rng) % builtins.size()];

            // Generate random number of args
            int num_args = choice_dist(rng) % (max_args + 1);
            std::string args_str;
            for (int a = 0; a < num_args; ++a) {
                if (a > 0) args_str += ", ";
                // Sometimes use keyword args
                if (choice_dist(rng) % 3 == 0 && !kwarg_names.empty()) {
                    std::string kwarg = kwarg_names[choice_dist(rng) % kwarg_names.size()];
                    args_str += kwarg + "=" + gen_random_arg();
                } else {
                    args_str += gen_random_arg();
                }
            }

            std::string tmpl;
            if (type_name == "global") {
                // Global function call
                tmpl = "{{ " + fn_name + "(" + args_str + ") }}";
            } else {
                // Method call on a value
                std::string base_val;
                if (type_name == "int") {
                    base_val = std::to_string(int_dist(rng));
                } else if (type_name == "float") {
                    base_val = std::to_string(int_dist(rng)) + ".5";
                } else if (type_name == "string") {
                    base_val = "\"test_string\"";
                } else if (type_name == "array") {
                    base_val = "[1, 2, 3, \"a\", \"b\"]";
                } else if (type_name == "object") {
                    base_val = "{\"x\": 1, \"y\": 2}";
                } else {
                    base_val = "x";
                }
                tmpl = "{{ " + base_val + "." + fn_name + "(" + args_str + ") }}";
            }

            json vars = {
                {"x", 42},
                {"y", "hello"},
                {"arr", json::array({1, 2, 3})},
                {"obj", {{"a", 1}, {"b", 2}}}
            };

            t.assert_true("builtin " + type_name + "." + fn_name + " #" + std::to_string(i), fuzz_test_template(tmpl, vars));
        }
    });
}
