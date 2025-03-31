#ifdef NDEBUG
#undef NDEBUG
#endif

#include "json-schema-to-grammar.h"

#include "llama-grammar.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <regex>

static std::string trim(const std::string & source) {
    std::string s(source);
    s.erase(0,s.find_first_not_of(" \n\r\t"));
    s.erase(s.find_last_not_of(" \n\r\t")+1);
    return std::regex_replace(s, std::regex("(^|\n)[ \t]+"), "$1");
}

enum TestCaseStatus {
    SUCCESS,
    FAILURE
};

struct TestCase {
    TestCaseStatus expected_status;
    std::string name;
    std::string schema;
    std::string expected_grammar;

    void _print_failure_header() const {
        fprintf(stderr, "#\n# Test '%s' failed.\n#\n%s\n", name.c_str(), schema.c_str());
    }
    void verify(const std::string & actual_grammar) const {
        if (trim(actual_grammar) != trim(expected_grammar)) {
        _print_failure_header();
        fprintf(stderr, "# EXPECTED:\n%s\n# ACTUAL:\n%s\n", expected_grammar.c_str(), actual_grammar.c_str());
        assert(false);
        }
    }
    void verify_expectation_parseable() const {
        try {
            llama_grammar_parser state;
            state.parse(expected_grammar.c_str());
            if (state.symbol_ids.find("root") == state.symbol_ids.end()) {
                throw std::runtime_error("Grammar failed to parse:\n" + expected_grammar);
            }
        } catch (const std::runtime_error & ex) {
            _print_failure_header();
            fprintf(stderr, "# GRAMMAR ERROR: %s\n", ex.what());
            assert(false);
        }
    }
    void verify_status(TestCaseStatus status) const {
        if (status != expected_status) {
            _print_failure_header();
            fprintf(stderr, "# EXPECTED STATUS: %s\n", expected_status == SUCCESS ? "SUCCESS" : "FAILURE");
            fprintf(stderr, "# ACTUAL STATUS: %s\n", status == SUCCESS ? "SUCCESS" : "FAILURE");
            assert(false);
        }
    }
};

static void write(const std::string & file, const std::string & content) {
    std::ofstream f;
    f.open(file.c_str());
    f << content.c_str();
    f.close();
}

static std::string read(const std::string & file) {
    std::ostringstream actuals;
    actuals << std::ifstream(file.c_str()).rdbuf();
    return actuals.str();
}

static void test_all(const std::string & lang, std::function<void(const TestCase &)> runner) {
    fprintf(stderr, "#\n# Testing JSON schema conversion (%s)\n#\n", lang.c_str());
    auto test = [&](const TestCase & tc) {
        fprintf(stderr, "- %s%s\n", tc.name.c_str(), tc.expected_status == FAILURE ? " (failure expected)" : "");
        runner(tc);
    };

    test({
        SUCCESS,
        "min 0",
        R"""({
            "type": "integer",
            "minimum": 0
        })""",
        R"""(
            root ::= ([0] | [1-9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 1",
        R"""({
            "type": "integer",
            "minimum": 1
        })""",
        R"""(
            root ::= ([1-9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 3",
        R"""({
            "type": "integer",
            "minimum": 3
        })""",
        R"""(
            root ::= ([1-2] [0-9]{1,15} | [3-9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 9",
        R"""({
            "type": "integer",
            "minimum": 9
        })""",
        R"""(
            root ::= ([1-8] [0-9]{1,15} | [9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 10",
        R"""({
            "type": "integer",
            "minimum": 10
        })""",
        R"""(
            root ::= ([1] ([0-9]{1,15}) | [2-9] [0-9]{1,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 25",
        R"""({
            "type": "integer",
            "minimum": 25
        })""",
        R"""(
            root ::= ([1] [0-9]{2,15} | [2] ([0-4] [0-9]{1,14} | [5-9] [0-9]{0,14}) | [3-9] [0-9]{1,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max 30",
        R"""({
            "type": "integer",
            "maximum": 30
        })""",
        R"""(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-2] [0-9] | [3] "0")) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -5",
        R"""({
            "type": "integer",
            "minimum": -5
        })""",
        R"""(
            root ::= ("-" ([0-5]) | [0] | [1-9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -123",
        R"""({
            "type": "integer",
            "minimum": -123
        })""",
        R"""(
            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0] | [1-9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max -5",
        R"""({
            "type": "integer",
            "maximum": -5
        })""",
        R"""(
            root ::= ("-" ([0-4] [0-9]{1,15} | [5-9] [0-9]{0,15})) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max 1",
        R"""({
            "type": "integer",
            "maximum": 1
        })""",
        R"""(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-1]) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max 100",
        R"""({
            "type": "integer",
            "maximum": 100
        })""",
        R"""(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-8] [0-9] | [9] [0-9]) | "100") space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 0 max 23",
        R"""({
            "type": "integer",
            "minimum": 0,
            "maximum": 23
        })""",
        R"""(
            root ::= ([0-9] | ([1] [0-9] | [2] [0-3])) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 15 max 300",
        R"""({
            "type": "integer",
            "minimum": 15,
            "maximum": 300
        })""",
        R"""(
            root ::= (([1] ([5-9]) | [2-9] [0-9]) | ([1-2] [0-9]{2} | [3] "00")) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 5 max 30",
        R"""({
            "type": "integer",
            "minimum": 5,
            "maximum": 30
        })""",
        R"""(
            root ::= ([5-9] | ([1-2] [0-9] | [3] "0")) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -123 max 42",
        R"""({
            "type": "integer",
            "minimum": -123,
            "maximum": 42
        })""",
        R"""(
            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0-9] | ([1-3] [0-9] | [4] [0-2])) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -10 max 10",
        R"""({
            "type": "integer",
            "minimum": -10,
            "maximum": 10
        })""",
        R"""(
            root ::= ("-" ([0-9] | "10") | [0-9] | "10") space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        FAILURE,
        "unknown type",
        R"""({
            "type": "kaboom"
        })""",
        ""
    });

    test({
        FAILURE,
        "invalid type",
        R"""({
            "type": 123
        })""",
        ""
    });

    test({
        SUCCESS,
        "empty schema (object)",
        "{}",
        R"""(
            array ::= "[" space ( value ("," space value)* )? "]" space
            boolean ::= ("true" | "false") space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null" space
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "exotic formats",
        R"""({
            "items": [
                { "format": "date" },
                { "format": "uuid" },
                { "format": "time" },
                { "format": "date-time" }
            ]
        })""",
        R"""(
            date ::= [0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )
            date-string ::= "\"" date "\"" space
            date-time ::= date "T" time
            date-time-string ::= "\"" date-time "\"" space
            root ::= "[" space tuple-0 "," space uuid "," space tuple-2 "," space tuple-3 "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            time ::= ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )
            time-string ::= "\"" time "\"" space
            tuple-0 ::= date-string
            tuple-2 ::= time-string
            tuple-3 ::= date-time-string
            uuid ::= "\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"" space
        )"""
    });

    test({
        SUCCESS,
        "string",
        R"""({
            "type": "string"
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char* "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ min length 1",
        R"""({
            "type": "string",
            "minLength": 1
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char+ "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ min length 3",
        R"""({
            "type": "string",
            "minLength": 3
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{3,} "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ max length",
        R"""({
            "type": "string",
            "maxLength": 3
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{0,3} "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ min & max length",
        R"""({
            "type": "string",
            "minLength": 1,
            "maxLength": 4
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{1,4} "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "boolean",
        R"""({
            "type": "boolean"
        })""",
        R"""(
            root ::= ("true" | "false") space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "integer",
        R"""({
            "type": "integer"
        })""",
        R"""(
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string const",
        R"""({
            "const": "foo"
        })""",
        R"""(
            root ::= "\"foo\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "non-string const",
        R"""({
            "const": 123
        })""",
        R"""(
            root ::= "123" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "non-string enum",
        R"""({
            "enum": ["red", "amber", "green", null, 42, ["foo"]]
        })""",
        R"""(
            root ::= ("\"red\"" | "\"amber\"" | "\"green\"" | "null" | "42" | "[\"foo\"]") space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string array",
        R"""({
            "type": "array",
            "prefixItems": { "type": "string" }
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space (string ("," space string)*)? "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "nullable string array",
        R"""({
            "type": ["array", "null"],
            "prefixItems": { "type": "string" }
        })""",
        R"""(
            alternative-0 ::= "[" space (string ("," space string)*)? "]" space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            null ::= "null" space
            root ::= alternative-0 | null
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "tuple1",
        R"""({
            "prefixItems": [{ "type": "string" }]
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space string "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "tuple2",
        R"""({
            "prefixItems": [{ "type": "string" }, { "type": "number" }]
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "[" space string "," space number "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "number",
        R"""({
            "type": "number"
        })""",
        R"""(
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "minItems",
        R"""({
            "items": {
                "type": "boolean"
            },
            "minItems": 2
        })""",
        R"""(
            boolean ::= ("true" | "false") space
            root ::= "[" space boolean ("," space boolean)+ "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "maxItems 1",
        R"""({
            "items": {
                "type": "boolean"
            },
            "maxItems": 1
        })""",
        R"""(
            boolean ::= ("true" | "false") space
            root ::= "[" space boolean? "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "maxItems 2",
        R"""({
            "items": {
                "type": "boolean"
            },
            "maxItems": 2
        })""",
        R"""(
            boolean ::= ("true" | "false") space
            root ::= "[" space (boolean ("," space boolean)?)? "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min + maxItems",
        R"""({
            "items": {
                "type": ["number", "integer"]
            },
            "minItems": 3,
            "maxItems": 5
        })""",
        R"""(
            decimal-part ::= [0-9]{1,16}
            integer ::= ("-"? integral-part) space
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= number | integer
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "[" space item ("," space item){2,4} "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min + max items with min + max values across zero",
        R"""({
            "items": {
                "type": "integer",
                "minimum": -12,
                "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
        })""",
        R"""(
            item ::= ("-" ([0-9] | "1" [0-2]) | [0-9] | ([1-8] [0-9] | [9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7])) space
            root ::= "[" space item ("," space item){2,4} "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min + max items with min + max values",
        R"""({
            "items": {
                "type": "integer",
                "minimum": 12,
                "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
        })""",
        R"""(
            item ::= (([1] ([2-9]) | [2-9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7])) space
            root ::= "[" space item ("," space item){2,4} "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "simple regexp",
        R"""({
            "type": "string",
            "pattern": "^abc?d*efg+(hij)?kl$"
        })""",
        R"""(
            root ::= "\"" ("ab" "c"? "d"* "ef" "g"+ ("hij")? "kl") "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp escapes",
        R"""({
            "type": "string",
            "pattern": "^\\[\\]\\{\\}\\(\\)\\|\\+\\*\\?$"
        })""",
        R"""(
            root ::= "\"" ("[]{}()|+*?") "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp quote",
        R"""({
            "type": "string",
            "pattern": "^\"$"
        })""",
        R"""(
            root ::= "\"" ("\"") "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp with top-level alternation",
        R"""({
            "type": "string",
            "pattern": "^A|B|C|D$"
        })""",
        R"""(
            root ::= "\"" ("A" | "B" | "C" | "D") "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp",
        R"""({
            "type": "string",
            "pattern": "^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} a{3,5}nd...$"
        })""",
        R"""(
            dot ::= [^\x0A\x0D]
            root ::= "\"" (("(" root-1{1,3} ")")? root-1{3,3} "-" root-1{4,4} " " "a"{3,5} "nd" dot dot dot) "\"" space
            root-1 ::= [0-9]
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "required props in original order",
        R"""({
            "type": "object",
            "properties": {
                "b": {"type": "string"},
                "c": {"type": "string"},
                "a": {"type": "string"}
            },
            "required": [
                "a",
                "b",
                "c"
            ],
            "additionalProperties": false,
            "definitions": {}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space b-kv "," space c-kv "," space a-kv "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "1 optional prop",
        R"""({
            "properties": {
                "a": {
                "type": "string"
                }
            },
            "additionalProperties": false
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "N optional props",
        R"""({
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"}
            },
            "additionalProperties": false
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            a-rest ::= ( "," space b-kv )? b-rest
            b-kv ::= "\"b\"" space ":" space string
            b-rest ::= ( "," space c-kv )?
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv a-rest | b-kv b-rest | c-kv )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "required + optional props each in original order",
        R"""({
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "string"},
                "d": {"type": "string"},
                "c": {"type": "string"}
            },
            "required": ["a", "b"],
            "additionalProperties": false
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            d-kv ::= "\"d\"" space ":" space string
            d-rest ::= ( "," space c-kv )?
            root ::= "{" space b-kv "," space a-kv ( "," space ( d-kv d-rest | c-kv ) )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "additional props",
        R"""({
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "number"}}
        })""",
        R"""(
            additional-kv ::= string ":" space additional-value
            additional-value ::= "[" space (number ("," space number)*)? "]" space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "{" space  (additional-kv ( "," space additional-kv )* )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "additional props (true)",
        R"""({
            "type": "object",
            "additionalProperties": true
        })""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? "]" space
            boolean ::= ("true" | "false") space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null" space
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "additional props (implicit)",
        R"""({
            "type": "object"
        })""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? "]" space
            boolean ::= ("true" | "false") space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null" space
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "empty w/o additional props",
        R"""({
            "type": "object",
            "additionalProperties": false
        })""",
        R"""(
            root ::= "{" space  "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "required + additional props",
        R"""({
            "type": "object",
            "properties": {
                "a": {"type": "number"}
            },
            "required": ["a"],
            "additionalProperties": {"type": "string"}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space number
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["] space
            additional-kv ::= additional-k ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "{" space a-kv ( "," space ( additional-kv ( "," space additional-kv )* ) )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "optional + additional props",
        R"""({
            "type": "object",
            "properties": {
                "a": {"type": "number"}
            },
            "additionalProperties": {"type": "number"}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space number
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["] space
            additional-kv ::= additional-k ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "{" space  (a-kv a-rest | additional-kv ( "," space additional-kv )* )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "required + optional + additional props",
        R"""({
            "type": "object",
            "properties": {
                "and": {"type": "number"},
                "also": {"type": "number"}
            },
            "required": ["and"],
            "additionalProperties": {"type": "number"}
        })""",
        R"""(
            additional-k ::= ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["] space
            additional-kv ::= additional-k ":" space number
            also-kv ::= "\"also\"" space ":" space number
            also-rest ::= ( "," space additional-kv )*
            and-kv ::= "\"and\"" space ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "{" space and-kv ( "," space ( also-kv also-rest | additional-kv ( "," space additional-kv )* ) )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "optional props with empty name",
        R"""({
            "properties": {
                "": {"type": "integer"},
                "a": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })""",
        R"""(
            -kv ::= "\"\"" space ":" space root
            -rest ::= ( "," space a-kv )? a-rest
            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* ) ["] space
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part) space
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part) space
            root0 ::= "{" space  (-kv -rest | a-kv a-rest | additional-kv ( "," space additional-kv )* )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "optional props with nested names",
        R"""({
            "properties": {
                "a": {"type": "integer"},
                "aa": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space aa-kv )? aa-rest
            aa-kv ::= "\"aa\"" space ":" space integer
            aa-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([a] char+ | [^"a] char*) | [^"a] char* )? ["] space
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part) space
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (a-kv a-rest | aa-kv aa-rest | additional-kv ( "," space additional-kv )* )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "optional props with common prefix",
        R"""({
            "properties": {
                "ab": {"type": "integer"},
                "ac": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })""",
        R"""(
            ab-kv ::= "\"ab\"" space ":" space integer
            ab-rest ::= ( "," space ac-kv )? ac-rest
            ac-kv ::= "\"ac\"" space ":" space integer
            ac-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([b] char+ | [c] char+ | [^"bc] char*) | [^"a] char* )? ["] space
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part) space
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (ab-kv ab-rest | ac-kv ac-rest | additional-kv ( "," space additional-kv )* )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "top-level $ref",
        R"""({
            "$ref": "#/definitions/foo",
            "definitions": {
                "foo": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "a"
                    ],
                    "additionalProperties": false
                }
            }
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            foo ::= "{" space foo-a-kv "}" space
            foo-a-kv ::= "\"a\"" space ":" space string
            root ::= foo
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "anyOf",
        R"""({
            "anyOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
                "foo": {
                    "properties": {"a": {"type": "number"}}
                },
                "bar": {
                    "properties": {"b": {"type": "number"}}
                }
            },
            "type": "object"
        })""",
        R"""(
            alternative-0 ::= foo
            alternative-1 ::= bar
            bar ::= "{" space  (bar-b-kv )? "}" space
            bar-b-kv ::= "\"b\"" space ":" space number
            decimal-part ::= [0-9]{1,16}
            foo ::= "{" space  (foo-a-kv )? "}" space
            foo-a-kv ::= "\"a\"" space ":" space number
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= alternative-0 | alternative-1
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "mix of allOf, anyOf and $ref (similar to https://json.schemastore.org/tsconfig.json)",
        R"""({
            "allOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"},
                {
                "anyOf": [
                    {"$ref": "#/definitions/baz"},
                    {"$ref": "#/definitions/bam"}
                ]
                }
            ],
            "definitions": {
                "foo": {
                    "properties": {"a": {"type": "number"}}
                },
                "bar": {
                    "properties": {"b": {"type": "number"}}
                },
                "bam": {
                    "properties": {"c": {"type": "number"}}
                },
                "baz": {
                    "properties": {"d": {"type": "number"}}
                }
            },
            "type": "object"
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space number
            b-kv ::= "\"b\"" space ":" space number
            c-kv ::= "\"c\"" space ":" space number
            d-kv ::= "\"d\"" space ":" space number
            d-rest ::= ( "," space c-kv )?
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "{" space a-kv "," space b-kv ( "," space ( d-kv d-rest | c-kv ) )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "conflicting names",
        R"""({
            "type": "object",
            "properties": {
                "number": {
                "type": "object",
                "properties": {
                    "number": {
                    "type": "object",
                        "properties": {
                            "root": {
                                "type": "number"
                            }
                        },
                        "required": [
                            "root"
                        ],
                        "additionalProperties": false
                    }
                },
                "required": [
                    "number"
                ],
                "additionalProperties": false
                }
            },
            "required": [
                "number"
            ],
            "additionalProperties": false,
            "definitions": {}
        })""",
        R"""(
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            number- ::= "{" space number-number-kv "}" space
            number-kv ::= "\"number\"" space ":" space number-
            number-number ::= "{" space number-number-root-kv "}" space
            number-number-kv ::= "\"number\"" space ":" space number-number
            number-number-root-kv ::= "\"root\"" space ":" space number
            root ::= "{" space number-kv "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });
}

int main() {
    fprintf(stderr, "LLAMA_NODE_AVAILABLE = %s\n", getenv("LLAMA_NODE_AVAILABLE") ? "true" : "false");
    fprintf(stderr, "LLAMA_PYTHON_AVAILABLE = %s\n", getenv("LLAMA_PYTHON_AVAILABLE") ? "true" : "false");

    test_all("C++", [](const TestCase & tc) {
        try {
            tc.verify(json_schema_to_grammar(nlohmann::ordered_json::parse(tc.schema), true));
            tc.verify_status(SUCCESS);
        } catch (const std::runtime_error & ex) {
            fprintf(stderr, "Error: %s\n", ex.what());
            tc.verify_status(FAILURE);
        }
    });

    if (getenv("LLAMA_SKIP_TESTS_SLOW_ON_EMULATOR")) {
        fprintf(stderr, "\033[33mWARNING: Skipping slow tests on emulator.\n\033[0m");
    } else {
        if (getenv("LLAMA_PYTHON_AVAILABLE") || (std::system("python -c \"import sys; exit(1) if sys.version_info < (3, 8) else print('Python version is sufficient')\"") == 0)) {
            test_all("Python", [](const TestCase & tc) {
                write("test-json-schema-input.tmp", tc.schema);
                tc.verify_status(std::system(
                    "python ./examples/json_schema_to_grammar.py test-json-schema-input.tmp > test-grammar-output.tmp") == 0 ? SUCCESS : FAILURE);
                tc.verify(read("test-grammar-output.tmp"));
            });
        } else {
            fprintf(stderr, "\033[33mWARNING: Python not found (min version required is 3.8), skipping Python JSON schema -> grammar tests.\n\033[0m");
        }

        if (getenv("LLAMA_NODE_AVAILABLE") || (std::system("node --version") == 0)) {
            test_all("JavaScript", [](const TestCase & tc) {
                write("test-json-schema-input.tmp", tc.schema);
                tc.verify_status(std::system(
                    "node ./tests/run-json-schema-to-grammar.mjs test-json-schema-input.tmp > test-grammar-output.tmp") == 0 ? SUCCESS : FAILURE);
                tc.verify(read("test-grammar-output.tmp"));
            });
        } else {
            fprintf(stderr, "\033[33mWARNING: Node not found, skipping JavaScript JSON schema -> grammar tests.\n\033[0m");
        }
    }

    test_all("Check Expectations Validity", [](const TestCase & tc) {
        if (tc.expected_status == SUCCESS) {
            tc.verify_expectation_parseable();
        }
    });
}
