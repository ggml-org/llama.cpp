#include "test-case.h"

#include <iterator>

const char * test_status_str(test_status status) {
    switch (status) {
        case test_status::PASS: return "\033[1;32mOK\033[0m";
        case test_status::FAIL: return "\033[1;31mFAIL\033[0m";
        case test_status::SKIP: return "\033[1;33mSKIP\033[0m";
    }
    return "";
}

// the printable width of a verdict, the color escapes of test_status_str are not part of it
size_t test_status_width(test_status status) {
    switch (status) {
        case test_status::PASS: return 2; // "OK"
        case test_status::FAIL: return 4; // "FAIL"
        case test_status::SKIP: return 4; // "SKIP"
    }

    return 0;
}

test_status merge_status(test_status a, test_status b) {
    if (a == test_status::FAIL || b == test_status::FAIL) {
        return test_status::FAIL;
    }
    if (a == test_status::PASS || b == test_status::PASS) {
        return test_status::PASS;
    }
    return test_status::SKIP;
}

case_result case_result::pass() {
    return { test_status::PASS, {} };
}

case_result case_result::fail(const std::string & why) {
    return { test_status::FAIL, why };
}

case_result case_result::skip(const std::string & why) {
    return { test_status::SKIP, why };
}

std::string ctx_test::skip_reason(model_run & mr) const {
    // the recurrent cases need a recurrent cache, the KV cases work with any model
    if (group() == TEST_GROUP_RS && !mr.is_recurrent()) {
        return "not a recurrent model";
    }

    return {};
}

const std::vector<ctx_test_ptr> & test_cases() {
    static const std::vector<ctx_test_ptr> cases = [] {
        std::vector<ctx_test_ptr> res;

        for (const auto & maker : { make_kv_cases, make_gigachat_cases, make_rs_cases }) {
            auto part = maker();
            res.insert(res.end(), std::make_move_iterator(part.begin()), std::make_move_iterator(part.end()));
        }

        return res;
    }();

    return cases;
}
