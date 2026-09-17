#include "arg.h"
#include "common.h"
#include "log.h"

#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>

#include "peg-parser/tests.h"

int main(int argc, char *argv[]) {
    common_params params;
    params.model.path = "."; // this test takes no model
    common_init();

    // this test takes an optional filter as its only positional argument
    std::string filter;
    std::vector<char *> common_argv;
    common_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            common_argv.push_back(argv[i]); // an option: let common_params_parse handle it
        } else if (filter.empty()) {
            filter = argv[i];
        }
    }
    common_argv.push_back(nullptr);
    if (!common_params_parse((int) common_argv.size() - 1, common_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    LOG("%s: running\n", "test-peg-parser");

    testing t(std::cout);
    if (!filter.empty()) {
        t.set_filter(filter);
    }

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    t.test("basic", test_basic);
    t.test("unicode", test_unicode);
    t.test("json", test_json_parser);
    t.test("gbnf", test_gbnf_generation);
    t.test("serialization", test_json_serialization);
    t.test("python-dict", test_python_dict_parser);

    const int rc = t.summary();
    LOG("%s: %s\n", "test-peg-parser", rc == 0 ? "PASSED" : "FAILED");
    common_log_flush(common_log_main());
    return rc;
}
