#include "test-runner.h"

#include "test-log.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <regex>
#include <utility>

std::vector<const ctx_test *> select_cases(const std::string & filter) {
    std::vector<const ctx_test *> selected;

    if (filter.empty()) {
        selected.reserve(test_cases().size());
        for (const auto & tc : test_cases()) {
            selected.push_back(tc.get());
        }

        return selected;
    }

    const std::regex re(filter);

    for (const auto & tc : test_cases()) {
        if (std::regex_match(tc->name(), re)) {
            selected.push_back(tc.get());
        }
    }

    return selected;
}

std::vector<case_run> run_cases(model_run & mr, const std::vector<const ctx_test *> & cases, bool report) {
    // the state cases keep the historical default of a unified KV cache when running a single
    // sequence, the recurrent cases use whatever was passed on the command line
    const bool unified_base  = mr.params.kv_unified;
    const bool unified_state = mr.params.n_parallel == 1 ? true : unified_base;

    size_t name_width = 4; // "case"
    for (const ctx_test * tc : cases) {
        name_width = std::max(name_width, strlen(tc->name()));
    }

    std::vector<case_run> runs;
    runs.reserve(cases.size());

    for (const ctx_test * tc : cases) {
        mr.params.kv_unified = tc->group() == TEST_GROUP_STATE ? unified_state : unified_base;

        case_run run;
        run.name = tc->name();

        // the log of a model is read on its own, mark where each case begins
        TLOG_INF("%s\n", run.name.c_str());

        // a case that throws is a case that failed, and the run goes on with the next one
        try {
            if (const std::string why = tc->skip_reason(mr); !why.empty()) {
                run.status = test_status::SKIP;
                run.reason = why;
            } else if (tc->needs_baseline() && !mr.have_baseline) {
                run.status = test_status::SKIP;
                run.reason = "no baseline";
            } else if (!tc->uses_fill()) {
                mr.fill = 0;

                const case_result res = tc->run(mr);

                run.status = res.status;
                run.reason = res.reason;
            } else {
                // the recurrent cases are also run against a cache pre-filled with a known
                // pattern, so that a restore reading cells it never wrote cannot pass

                for (const uint8_t fill : { 0, 0x3e }) {
                    TLOG_INF("(fill 0x%02x)\n", fill);

                    mr.fill = fill;

                    const case_result res = tc->run(mr);

                    run.status = merge_status(run.status, res.status);
                    if (run.reason.empty()) {
                        run.reason = res.reason;
                    }
                    if (res.status == test_status::FAIL) {
                        break;
                    }
                }
            }
        } catch (...) {
            run.status = test_status::FAIL;
            run.reason = "unhandled exception";
        }

        if (unified_state) {
            mr.params.kv_unified = unified_base;
        }

        // the verdict is one word, what it means goes to the log of the model
        if (!run.reason.empty()) {
            TLOG_INF("%s %s: %s\n", run.name.c_str(),
                    run.status == test_status::FAIL ? "failed" : "skipped", run.reason.c_str());
        }

        if (report) {
            fprintf(stderr, "%-*s  %s%s%s\n", (int) name_width, run.name.c_str(),
                    test_status_str(run.status),
                    run.reason.empty() ? "" : "  ", run.reason.c_str());
        }

        runs.push_back(std::move(run));
    }

    return runs;
}
