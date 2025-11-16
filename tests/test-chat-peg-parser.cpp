#include "chat-peg-parser/tests.h"

int main() {
    testing t(std::cout);

    test_partial_parsing(t);
    test_one(t);
    test_optional(t);
    test_unicode(t);
    test_recursive_references(t);
    test_json_parser(t);
    test_gbnf_generation(t);
    test_example_qwen3_coder(t);
    test_command7_parser_compare(t);

    return t.summary();
}
