//
// test_capability_loop.cpp
//
// Smoke test for the self-improving capability loop: the multi-axis
// capability eval (ts_capability_score_load + the weighted-sum, Pareto,
// and guard lenses) and the guarded adaptation engine (ts_adapt_run).
// Writes throwaway JSON to /tmp and checks return codes + receipt shape.
//

#include "tessera-capability-eval.h"
#include "tessera-adapt.h"

#include <cmath>
#include <cstdio>
#include <fstream>
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

static void check_close(const char * name, double got, double want, double tol) {
    if (std::fabs(got - want) > tol) {
        std::printf("FAIL %-28s got %.7g want %.7g\n", name, got, want);
        g_fail++;
    } else {
        std::printf("ok   %-28s %.7g\n", name, got);
    }
}

static void write_file(const char * path, const std::string & body) {
    std::ofstream f(path, std::ios::binary);
    f << body;
}

static std::string read_file(const char * path) {
    std::ifstream f(path, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s;
}

// schema_version 1 instances: mechanical .8, api_currency .6, hard_tail .5,
// personal_style .7, general_competence .9; baseline gc .85.
static const char * kGoodInstances =
    "{\n"
    "  \"schema_version\": 1,\n"
    "  \"axes\": {\n"
    "    \"mechanical\":         {\"pass\": 8, \"fail\": 2},\n"
    "    \"api_currency\":       {\"pass\": 6, \"fail\": 4},\n"
    "    \"hard_tail\":          {\"pass\": 5, \"fail\": 5},\n"
    "    \"personal_style\":     {\"pass\": 7, \"fail\": 3},\n"
    "    \"general_competence\": {\"pass\": 9, \"fail\": 1}\n"
    "  },\n"
    "  \"baseline\": {\n"
    "    \"mechanical\": 0.7, \"api_currency\": 0.5, \"hard_tail\": 0.5,\n"
    "    \"personal_style\": 0.6, \"general_competence\": 0.85\n"
    "  }\n"
    "}\n";

int main() {
    const char * good_path = "/tmp/tessera_cap_good.json";

    // ------------------------------------------------------------------
    // Case 1: ts_capability_score_load - good input
    // ------------------------------------------------------------------
    {
        write_file(good_path, kGoodInstances);
        ts_capability_score score;
        ts_capability_score baseline;
        bool has_baseline = false;
        std::string err;
        const int rc = ts_capability_score_load(good_path, &score, &baseline, &has_baseline, &err);
        check("case1: load rc == 0", rc == 0);
        check_close("case1: mechanical",     score.mechanical,         0.8, 1e-9);
        check_close("case1: api_currency",   score.api_currency,       0.6, 1e-9);
        check_close("case1: hard_tail",      score.hard_tail,          0.5, 1e-9);
        check_close("case1: personal_style", score.personal_style,     0.7, 1e-9);
        check_close("case1: general_comp",   score.general_competence, 0.9, 1e-9);
        check("case1: has_baseline", has_baseline);
        check_close("case1: baseline gc", baseline.general_competence, 0.85, 1e-9);
    }

    // ------------------------------------------------------------------
    // Case 2: ts_capability_score_load - schema mismatch fails loudly
    // ------------------------------------------------------------------
    {
        const char * bad_path = "/tmp/tessera_cap_badschema.json";
        write_file(bad_path, "{ \"schema_version\": 2, \"axes\": {} }\n");
        ts_capability_score score;
        std::string err;
        const int rc = ts_capability_score_load(bad_path, &score, nullptr, nullptr, &err);
        check("case2: mismatch rc != 0", rc != 0);
        check("case2: err mentions schema", err.find("schema_version") != std::string::npos);
    }

    // ------------------------------------------------------------------
    // Case 3: ts_capability_score_load - missing axis fails loudly
    // ------------------------------------------------------------------
    {
        const char * bad_path = "/tmp/tessera_cap_missingaxis.json";
        write_file(bad_path,
            "{ \"schema_version\": 1, \"axes\": { \"mechanical\": {\"pass\": 1, \"fail\": 0} } }\n");
        ts_capability_score score;
        std::string err;
        const int rc = ts_capability_score_load(bad_path, &score, nullptr, nullptr, &err);
        check("case3: missing axis rc != 0", rc != 0);
        check("case3: err mentions axis", err.find("axis") != std::string::npos);
    }

    // ------------------------------------------------------------------
    // Case 4: weighted-sum lens (uniform weights over 4 optimization axes)
    // ------------------------------------------------------------------
    {
        ts_capability_score s = { 0.8, 0.6, 0.5, 0.7, 0.9 };
        const double uniform[5] = { 0.25, 0.25, 0.25, 0.25, 0.0 };
        // 0.25 * (0.8 + 0.6 + 0.5 + 0.7) = 0.65; guard axis not summed
        check_close("case4: weighted_sum", ts_capability_score_weighted_sum(&s, uniform), 0.65, 1e-9);
    }

    // ------------------------------------------------------------------
    // Case 5: Pareto domination lens
    // ------------------------------------------------------------------
    {
        ts_capability_score a = { 0.8, 0.6, 0.5, 0.7, 0.9 };
        ts_capability_score b = { 0.7, 0.5, 0.5, 0.6, 0.85 };
        ts_capability_score c = { 0.8, 0.6, 0.5, 0.7, 0.9 };  // == a
        check("case5: a dominates b",  ts_capability_score_dominates(&a, &b));
        check("case5: b !dominate a", !ts_capability_score_dominates(&b, &a));
        check("case5: equal !dominate", !ts_capability_score_dominates(&a, &c));
    }

    // ------------------------------------------------------------------
    // Case 6: guard lens
    // ------------------------------------------------------------------
    {
        ts_capability_score s        = { 0.8, 0.6, 0.5, 0.7, 0.9 };
        ts_capability_score base_ok  = { 0.7, 0.5, 0.5, 0.6, 0.85 };
        ts_capability_score base_bad = { 0.7, 0.5, 0.5, 0.6, 0.99 };
        check("case6: passes vs lower baseline",  ts_capability_score_passes_guard(&s, &base_ok, 0.02));
        check("case6: fails vs higher baseline", !ts_capability_score_passes_guard(&s, &base_bad, 0.02));
        check("case6: null baseline passes",      ts_capability_score_passes_guard(&s, nullptr, 0.02));
    }

    // ------------------------------------------------------------------
    // Case 7: ts_adapt_run - guard passes (rc 0), receipt written
    // ------------------------------------------------------------------
    {
        const char * receipt = "/tmp/tessera_adapt_pass.json";
        ts_adapt_params params;
        ts_adapt_default_params(&params);
        snprintf(params.input_eval_path,  sizeof(params.input_eval_path),  "%s", good_path);
        snprintf(params.output_receipt_path, sizeof(params.output_receipt_path), "%s", receipt);
        // score gc 0.9 vs baseline gc 0.85, epsilon 0.02 -> guard passes
        const int rc = ts_adapt_run(&params);
        check("case7: rc == 0", rc == 0);
        const std::string r = read_file(receipt);
        check("case7: receipt schema", r.find("llama.tessera.adapt.v1") != std::string::npos);
        check("case7: guard_passed true", r.find("\"guard_passed\": true") != std::string::npos);
        check("case7: has score", r.find("\"score\"") != std::string::npos);
        check("case7: has_baseline true", r.find("\"has_baseline\": true") != std::string::npos);
    }

    // ------------------------------------------------------------------
    // Case 8: ts_adapt_run - guard fails (rc 1), receipt still written
    // ------------------------------------------------------------------
    {
        const char * eval_path = "/tmp/tessera_adapt_regress.json";
        const char * receipt   = "/tmp/tessera_adapt_fail.json";
        // score gc 0.5 vs baseline gc 0.9, epsilon 0.02 -> guard fails
        write_file(eval_path,
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"axes\": {\n"
            "    \"mechanical\":         {\"pass\": 8, \"fail\": 2},\n"
            "    \"api_currency\":       {\"pass\": 6, \"fail\": 4},\n"
            "    \"hard_tail\":          {\"pass\": 5, \"fail\": 5},\n"
            "    \"personal_style\":     {\"pass\": 7, \"fail\": 3},\n"
            "    \"general_competence\": {\"pass\": 5, \"fail\": 5}\n"
            "  },\n"
            "  \"baseline\": {\n"
            "    \"mechanical\": 0.7, \"api_currency\": 0.5, \"hard_tail\": 0.5,\n"
            "    \"personal_style\": 0.6, \"general_competence\": 0.9\n"
            "  }\n"
            "}\n");
        ts_adapt_params params;
        ts_adapt_default_params(&params);
        params.dry_run = true;
        snprintf(params.input_eval_path,  sizeof(params.input_eval_path),  "%s", eval_path);
        snprintf(params.output_receipt_path, sizeof(params.output_receipt_path), "%s", receipt);
        const int rc = ts_adapt_run(&params);
        check("case8: rc == 1", rc == 1);
        const std::string r = read_file(receipt);
        check("case8: guard_passed false", r.find("\"guard_passed\": false") != std::string::npos);
        check("case8: dry_run recorded", r.find("\"dry_run\": true") != std::string::npos);
        check("case8: not adapted", r.find("\"adapted\": false") != std::string::npos);
    }

    // ------------------------------------------------------------------
    // Case 9: ts_adapt_run - bad args / unreadable eval (rc -1)
    // ------------------------------------------------------------------
    {
        ts_adapt_params params;
        ts_adapt_default_params(&params);  // empty input/output paths
        check("case9: empty paths rc == -1", ts_adapt_run(&params) == -1);

        ts_adapt_default_params(&params);
        snprintf(params.input_eval_path,  sizeof(params.input_eval_path),  "%s", "/tmp/tessera_does_not_exist.json");
        snprintf(params.output_receipt_path, sizeof(params.output_receipt_path), "%s", "/tmp/tessera_adapt_err.json");
        check("case9: missing eval rc == -1", ts_adapt_run(&params) == -1);
    }

    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
