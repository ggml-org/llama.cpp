#include "test-partial-parsing.cpp"
#include "test-one.cpp"
#include "test-optional.cpp"
#include "test-recursive-references.cpp"
#include "test-json-parser.cpp"
#include "test-complete-example.cpp"
#include "test-actions.cpp"
#include "test-gbnf-generation.cpp"

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

    test_complete_example complete_example_test;
    complete_example_test.run_all_tests();

    test_actions actions_test;
    actions_test.run_all_tests();

    test_gbnf_generation gbnf_generation_test;
    gbnf_generation_test.run_all_tests();
    
    return 0;
}