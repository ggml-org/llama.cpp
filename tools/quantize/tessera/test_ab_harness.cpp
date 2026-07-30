//
// test_ab_harness.cpp
//
// Smoke test for tessera-ab-harness. Verifies Kendall tau on rankings
// with known correlation, the composite-beats-single logic, and the
// receipt JSON.
//

#include "tessera-ab-harness.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(const char * name, bool ok) {
    if (!ok) {
        std::printf("FAIL %s\n", name);
        g_fail++;
    } else {
        std::printf("ok   %s\n", name);
    }
}

static void check_close(const char * name, float got, float want, float tol) {
    if (std::fabs(got - want) > tol) {
        std::printf("FAIL %-28s got %.7g want %.7g\n", name, (double)got, (double)want);
        g_fail++;
    } else {
        std::printf("ok   %-28s %.7g\n", name, (double)got);
    }
}

int main() {
    // Case 1: perfect correlation -> tau = 1.0
    {
        float a[10], b[10];
        for (int i = 0; i < 10; i++) {
            a[i] = (float)(i + 1);
            b[i] = 2.0f * (float)(i + 1) + 3.0f;  // monotonic increasing
        }
        check_close("tau perfect correlation", ts_ab_kendall_tau(a, b, 10), 1.0f, 1e-6f);
    }

    // Case 2: anti-correlation -> tau = -1.0
    {
        float a[10], b[10];
        for (int i = 0; i < 10; i++) {
            a[i] = (float)(i + 1);
            b[i] = (float)(10 - i);  // monotonic decreasing
        }
        check_close("tau anti-correlation", ts_ab_kendall_tau(a, b, 10), -1.0f, 1e-6f);
    }

    // Case 3: partial correlation -> tau in (-1, 1), exact value 2/3
    // (one swapped pair out of six: C=5, D=1, tau=(5-1)/6)
    {
        float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float b[4] = { 1.0f, 3.0f, 2.0f, 4.0f };
        const float tau = ts_ab_kendall_tau(a, b, 4);
        check_close("tau partial (2/3)", tau, 2.0f / 3.0f, 1e-6f);
        check("tau partial in (-1,1)", tau > -1.0f && tau < 1.0f);
    }

    // degenerate: n < 2 -> 0
    {
        float a[1] = { 1.0f };
        float b[1] = { 1.0f };
        check_close("tau n=1", ts_ab_kendall_tau(a, b, 1), 0.0f, 1e-9f);
    }

    // 10 synthetic tensor scores, partially correlated
    std::vector<ts_ab_tensor_scores> scores(10);
    const float kernel_t2[10] = { 0.5f, 0.3f, 0.9f, 0.1f, 0.7f, 0.2f, 0.8f, 0.4f, 0.6f, 1.0f };
    for (int i = 0; i < 10; i++) {
        scores[i].name = "tensor_" + std::to_string(i);
        scores[i].offline_proxy_mse = 0.1f * (float)(i + 1);  // 0.1 .. 1.0 increasing
        scores[i].kernel_direct_t2  = kernel_t2[i];
        scores[i].alpha_l = 1.0f;
    }

    ts_ab_harness_params params;
    params.n_heldout = 10;
    params.measure_ranking = true;
    params.verbose = false;

    ts_ab_harness_result result;
    int rc = ts_ab_run(&scores, &params, &result);
    check("ts_ab_run rc == 0", rc == 0);
    check("result has 10 scores", result.scores.size() == 10);

    // composites: sum of 0.1..1.0 = 5.5, sum of kernel_t2 = 5.5
    check_close("composite_offline", result.composite_offline, 5.5f, 1e-5f);
    check_close("composite_kernel", result.composite_kernel, 5.5f, 1e-5f);

    // with uniform alpha=1.0 the composite (a sum) cannot beat the best
    // single tensor (min kernel_direct_t2 = 0.1)
    check("composite_beats_single == false", result.composite_beats_single == false);

    // ranking is partial: tau and disagreement strictly interior
    check("tau in (-1,1)", result.kendall_tau > -1.0f && result.kendall_tau < 1.0f);
    check("disagreement in (0,1)",
          result.ranking_disagreement > 0.0f && result.ranking_disagreement < 1.0f);

    // no ties -> disagreement == (1 - tau) / 2
    check_close("disagreement == (1-tau)/2",
                result.ranking_disagreement, (1.0f - result.kendall_tau) * 0.5f, 1e-5f);

    // receipt JSON: non-empty and carries the key fields
    const std::string receipt = ts_ab_receipt_json(&result);
    check("receipt non-empty", !receipt.empty());
    check("receipt contains kendall_tau", receipt.find("kendall_tau") != std::string::npos);
    check("receipt contains composite_kernel", receipt.find("composite_kernel") != std::string::npos);

    // null-safety
    check("null scores -> -1", ts_ab_run(nullptr, &params, &result) == -1);

    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
