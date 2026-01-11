#include <string>
#include <iostream>

#include <nlohmann/json.hpp>

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

int main(int argc, char *argv[]) {
    testing t(std::cout);
    t.verbose = true;

    if (argc >= 2) {
        t.set_filter(argv[1]);
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

    // TODO: Check if an object points to the same memory address as another object
    // test_template(t, "is sameas",
    //     "{{ 'yes' if x is sameas false }}",
    //     {{"x", false}},
    //     "yes"
    // );

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

    // TODO:
    // test_template(t, "is escaped",
    //     "{{ 'yes' if 'foo'|safe is escaped }}",
    //     json::object(),
    //     "yes"
    // );

    // TODO:
    // test_template(t, "is filter",
    //     "{{ 'yes' if 'trim' is filter }}",
    //     json::object(),
    //     "yes"
    // );

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
        {{"obj", {{"z", 3}, {"a", 1}, {"m", 2}}}},
        "a=1 m=2 z=3 "
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
}

static void test_template(testing & t, const std::string & name, const std::string & tmpl, const json & vars, const std::string & expect) {
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
