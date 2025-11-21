#include <string>
#include <iostream>

#include "peg-parser/tests.h"

int main(int argc, char *argv[]) {
    testing t(std::cout);
    if (argc >= 2) {
        t.set_filter(argv[1]);
    }

    t.test("partial", test_partial_parsing);
    t.test("one", test_one);
    t.test("optional", test_optional);
    t.test("unicode", test_unicode);
    t.test("recursive", test_recursive_references);
    t.test("json", test_json_parser);
    t.test("gbnf", test_gbnf_generation);
    t.test("serialization", test_json_serialization);
    //t.test("qwen3_coder", test_example_qwen3_coder);
    //t.test("seed_oss", test_example_seed_oss);
    //t.test("minimax_m2", test_example_minimax_m2);
    //t.test("command7_parser_compare", test_command7_parser_compare);

    return t.summary();
}
