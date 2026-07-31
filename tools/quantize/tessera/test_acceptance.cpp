//
// test_acceptance.cpp
//
// Smoke test for tessera-acceptance. Verifies composite-beats-single
// with well-separated regimes, ranking disagreement computation, JSON
// report round-trip, and verdict population.
//

#include "tessera-acceptance.h"
#include "tessera-ab-harness.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

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

// Build a tensor entry with explicit per-proxy scores.
static ts_acceptance_tensor make_tensor(const char * name,
                                        float comp, float awq, float rot,
                                        float lr, float hess,
                                        float offline, float kernel,
                                        bool held_out) {
    ts_acceptance_tensor t;
    memset(&t, 0, sizeof(t));
    snprintf(t.name, sizeof(t.name), "%s", name);
    t.composite_t2      = comp;
    t.awq_t2            = awq;
    t.rotation_t2       = rot;
    t.lowrank_t2        = lr;
    t.hessian_t2        = hess;
    t.offline_proxy_mse = offline;
    t.kernel_direct_t2  = kernel;
    t.held_out          = held_out;
    return t;
}

int main() {
    // ------------------------------------------------------------------
    // Case 1: well-separated regimes -> composite wins
    // ------------------------------------------------------------------
    // Four tensors, each with a different best proxy. The composite
    // routes each to its best, so composite mean < any single-proxy mean.
    {
        ts_acceptance_tensor tensors[4] = {
            // name, comp,  awq,   rot,   lr,    hess,  offline, kernel, held
            make_tensor("blk.0.attn_k.weight", 0.01f, 0.01f, 0.05f, 0.05f, 0.05f, 0.10f, 0.40f, true),
            make_tensor("blk.0.ffn_down.weight", 0.01f, 0.05f, 0.01f, 0.05f, 0.05f, 0.20f, 0.10f, true),
            make_tensor("blk.1.attn_q.weight", 0.01f, 0.05f, 0.05f, 0.01f, 0.05f, 0.30f, 0.30f, true),
            make_tensor("blk.1.ffn_gate.weight", 0.01f, 0.05f, 0.05f, 0.05f, 0.01f, 0.40f, 0.20f, true),
        };

        ts_acceptance_config cfg;
        ts_acceptance_default_config(&cfg);
        cfg.verbose = false;

        ts_acceptance_result res;
        int rc = ts_acceptance_run(&cfg, tensors, 4, &res);
        check("case1: rc == 0", rc == 0);

        // composite mean = 0.01, each single-proxy mean = (0.01+0.05*3)/4 = 0.04
        check_close("case1: composite_t2", res.composite_t2, 0.01f, 1e-6f);
        check_close("case1: best_single_t2", res.best_single_t2, 0.04f, 1e-6f);
        check("case1: composite_wins", res.composite_wins);
        check("case1: improvement > 0", res.improvement_pct > 0.0f);
        check_close("case1: improvement_pct", res.improvement_pct, 75.0f, 0.1f);

        // per-proxy breakdown
        check_close("case1: awq_t2", res.awq_t2, 0.04f, 1e-6f);
        check_close("case1: rotation_t2", res.rotation_t2, 0.04f, 1e-6f);
        check_close("case1: lowrank_t2", res.lowrank_t2, 0.04f, 1e-6f);
        check_close("case1: hessian_t2", res.hessian_t2, 0.04f, 1e-6f);

        // ranking: offline=[0.1,0.2,0.3,0.4], kernel=[0.4,0.1,0.3,0.2]
        // pairs: (0,1)D (0,2)D (0,3)D (1,2)C (1,3)C (2,3)D -> C=2 D=4 tau=-2/6
        check_close("case1: kendall_tau", res.kendall_tau, -1.0f / 3.0f, 1e-6f);
        check_close("case1: ranking_disagreement", res.ranking_disagreement, 2.0f / 3.0f, 1e-6f);
        check("case1: novelty_survives", res.novelty_survives);

        check("case1: acceptance_passed", res.acceptance_passed);
        check("case1: verdict non-empty", strlen(res.verdict) > 0);
        check("case1: verdict has PASS", strstr(res.verdict, "PASS") != nullptr);
        check("case1: n_tensors_total", res.n_tensors_total == 4);
        check("case1: n_tensors_heldout", res.n_tensors_heldout == 4);
    }

    // ------------------------------------------------------------------
    // Case 2: identical rankings -> novelty fails
    // ------------------------------------------------------------------
    {
        ts_acceptance_tensor tensors[4] = {
            make_tensor("t0", 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.1f, 0.1f, true),
            make_tensor("t1", 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.2f, 0.2f, true),
            make_tensor("t2", 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.3f, 0.3f, true),
            make_tensor("t3", 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.4f, 0.4f, true),
        };

        ts_acceptance_config cfg;
        ts_acceptance_default_config(&cfg);

        ts_acceptance_result res;
        int rc = ts_acceptance_run(&cfg, tensors, 4, &res);
        check("case2: rc == 0", rc == 0);
        check("case2: composite_wins", res.composite_wins);
        check_close("case2: tau == 1", res.kendall_tau, 1.0f, 1e-6f);
        check_close("case2: disagreement == 0", res.ranking_disagreement, 0.0f, 1e-6f);
        check("case2: novelty fails", !res.novelty_survives);
        check("case2: acceptance fails", !res.acceptance_passed);
        check("case2: verdict has FAIL", strstr(res.verdict, "FAIL") != nullptr);
    }

    // ------------------------------------------------------------------
    // Case 3: composite loses -> acceptance fails even with disagreement
    // ------------------------------------------------------------------
    {
        ts_acceptance_tensor tensors[4] = {
            make_tensor("t0", 0.10f, 0.01f, 0.02f, 0.03f, 0.04f, 0.1f, 0.4f, true),
            make_tensor("t1", 0.10f, 0.01f, 0.02f, 0.03f, 0.04f, 0.2f, 0.1f, true),
            make_tensor("t2", 0.10f, 0.01f, 0.02f, 0.03f, 0.04f, 0.3f, 0.3f, true),
            make_tensor("t3", 0.10f, 0.01f, 0.02f, 0.03f, 0.04f, 0.4f, 0.2f, true),
        };

        ts_acceptance_config cfg;
        ts_acceptance_default_config(&cfg);

        ts_acceptance_result res;
        int rc = ts_acceptance_run(&cfg, tensors, 4, &res);
        check("case3: rc == 0", rc == 0);
        check("case3: composite loses", !res.composite_wins);
        check("case3: novelty survives", res.novelty_survives);
        check("case3: acceptance fails", !res.acceptance_passed);
    }

    // ------------------------------------------------------------------
    // Case 4: held-out fraction (no explicit held_out flags)
    // ------------------------------------------------------------------
    {
        // partially shuffled kernel rankings (not perfectly anti-correlated)
        const float kernel_vals[10] = { 0.5f, 0.9f, 0.1f, 0.7f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 1.0f };
        ts_acceptance_tensor tensors[10];
        for (int i = 0; i < 10; i++) {
            tensors[i] = make_tensor(
                ("t" + std::to_string(i)).c_str(),
                0.01f, 0.05f, 0.05f, 0.05f, 0.05f,
                0.1f * (float)(i + 1), kernel_vals[i],
                false);  // not explicitly held out
        }

        ts_acceptance_config cfg;
        ts_acceptance_default_config(&cfg);
        cfg.heldout_fraction = 0.2f;  // hold out last 2

        ts_acceptance_result res;
        int rc = ts_acceptance_run(&cfg, tensors, 10, &res);
        check("case4: rc == 0", rc == 0);
        check("case4: 2 held out", res.n_tensors_heldout == 2);
        check("case4: 10 total", res.n_tensors_total == 10);
        // ranking uses all 10 tensors; partial shuffle -> tau strictly interior
        check("case4: tau in (-1,1)", res.kendall_tau > -1.0f && res.kendall_tau < 1.0f);
    }

    // ------------------------------------------------------------------
    // Case 5: JSON report round-trip
    // ------------------------------------------------------------------
    {
        ts_acceptance_tensor tensors[2] = {
            make_tensor("j0", 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.1f, 0.5f, true),
            make_tensor("j1", 0.02f, 0.03f, 0.01f, 0.04f, 0.05f, 0.2f, 0.1f, true),
        };

        ts_acceptance_config cfg;
        ts_acceptance_default_config(&cfg);
        snprintf(cfg.output_path, sizeof(cfg.output_path),
                 "/tmp/tessera_acceptance_test.json");

        ts_acceptance_result res;
        int rc = ts_acceptance_run(&cfg, tensors, 2, &res);
        check("case5: rc == 0", rc == 0);

        // verify the file was written
        FILE * f = fopen(cfg.output_path, "r");
        check("case5: json file exists", f != nullptr);
        if (f) {
            char buf[8192];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = '\0';
            fclose(f);

            std::string json(buf);
            check("case5: has schema", json.find("llama.tessera.acceptance.v1") != std::string::npos);
            check("case5: has acceptance_passed", json.find("acceptance_passed") != std::string::npos);
            check("case5: has composite_t2", json.find("composite_t2") != std::string::npos);
            check("case5: has kendall_tau", json.find("kendall_tau") != std::string::npos);
            check("case5: has per_proxy", json.find("per_proxy") != std::string::npos);
            check("case5: has tensors array", json.find("\"tensors\":[") != std::string::npos);
            check("case5: has tensor name j0", json.find("\"j0\"") != std::string::npos);
            check("case5: has verdict", json.find("verdict") != std::string::npos);
        }
    }

    // ------------------------------------------------------------------
    // Case 6: Kendall tau sanity (reuse ts_ab_kendall_tau directly)
    // ------------------------------------------------------------------
    {
        // anti-correlated rankings -> tau = -1, disagreement = 0 (|tau| = 1)
        float a[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        float b[5] = { 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
        float tau = ts_ab_kendall_tau(a, b, 5);
        check_close("case6: anti-corr tau", tau, -1.0f, 1e-6f);
        // disagreement = 1 - |tau| = 0 -> novelty fails for anti-correlation too
        check("case6: |tau| == 1 -> no disagreement", 1.0f - fabsf(tau) < 0.05f);
    }

    // ------------------------------------------------------------------
    // Case 7: null safety
    // ------------------------------------------------------------------
    {
        ts_acceptance_config cfg;
        ts_acceptance_default_config(&cfg);
        ts_acceptance_result res;
        check("null tensors -> -1", ts_acceptance_run(&cfg, nullptr, 0, &res) == -1);
        check("null config -> -1", ts_acceptance_run(nullptr, nullptr, 0, &res) == -1);
    }

    // ------------------------------------------------------------------
    // Case 8: empty tensor set
    // ------------------------------------------------------------------
    {
        ts_acceptance_config cfg;
        ts_acceptance_default_config(&cfg);
        ts_acceptance_result res;
        ts_acceptance_tensor dummy;
        int rc = ts_acceptance_run(&cfg, &dummy, 0, &res);
        check("case8: rc == 0", rc == 0);
        check("case8: verdict has SKIP", strstr(res.verdict, "SKIP") != nullptr);
    }

    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
