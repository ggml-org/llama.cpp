#pragma once

#include "test-case.h"

#include <string>
#include <vector>

//
// the runner: the single place that decides how a case is executed
//

struct case_run {
    std::string name;
    test_status status = test_status::SKIP;
    std::string reason;
};

// the cases selected by the filter, all of them when it is empty, in table order
std::vector<const ctx_test *> select_cases(const std::string & filter);

// run the selected cases for one model and return one verdict per case - with report, print the
// verdict of every case as it finishes, for single model mode
std::vector<case_run> run_cases(model_run & mr, const std::vector<const ctx_test *> & cases, bool report);
