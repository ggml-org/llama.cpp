#include "combinator/tests.h"

int main() {
    test_partial_parsing partial_parsing_test;
    partial_parsing_test.run_all_tests();

    test_one one_test;
    one_test.run_all_tests();

    test_optional optional_test;
    optional_test.run_all_tests();

    test_recursive_references recursive_references_test;
    recursive_references_test.run_all_tests();

    test_json_parser json_parser_test;
    json_parser_test.run_all_tests();

    test_actions actions_test;
    actions_test.run_all_tests();

    test_gbnf_generation gbnf_generation_test;
    gbnf_generation_test.run_all_tests();

    test_example_qwen3_coder qwen3_coder_test;
    qwen3_coder_test.run_all_tests();

    test_command7_parser_compare command7_compare_test;
    command7_compare_test.run_comparison(100);

    return 0;
}
