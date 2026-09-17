#include "../src/unicode.h"

#include "arg.h"
#include "common.h"
#include "log.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;
    params.model.path = "."; // this test takes no model
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    LOG("%s: running\n", "test-unicode");

    const std::vector<std::string> regex_exprs = {
        "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
    };
    const std::vector<std::string> expected = { " ~", "foo" };
    const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

    if (actual != expected) {
        LOG_ERR("unexpected split:");
        for (const auto & piece : actual) {
            LOG_ERR(" [%s]", piece.c_str());
        }
        LOG_ERR("\n");
        LOG("%s: %s\n", "test-unicode", "FAILED");
        common_log_flush(common_log_main());
        return 1;
    }

    LOG("%s: %s\n", "test-unicode", "PASSED");
    common_log_flush(common_log_main());
    return 0;
}
