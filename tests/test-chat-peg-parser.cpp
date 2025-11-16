#include "log.h"

#include "chat-peg-parser/tests.h"

int main() {
    common_log_set_verbosity_thold(LOG_DEFAULT_DEBUG);

    testing t;

    test_partial_parsing(t);
    test_one(t);
    test_optional(t);
    test_unicode(t);
    test_recursive_references(t);
    test_json_parser(t);
    test_gbnf_generation(t);
    test_example_qwen3_coder(t);
    test_example_seed_oss(t);
    test_example_minimax_m2(t);
    test_command7_parser_compare(t);

    return t.summary();
}
