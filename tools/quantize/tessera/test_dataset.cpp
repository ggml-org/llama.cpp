// test_dataset.cpp - offline tests for tessera-dataset (no model needed)
#include "tessera-dataset.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

static std::string write_tmp(const char * name, const std::string & content) {
    std::string path = std::string("/tmp/ts_dataset_test_") + name;
    std::ofstream f(path, std::ios::binary);
    f << content;
    return path;
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// A minimal valid llama.tessera.spec.v1 record.
static std::string make_record(int n_acc, int n_dft,
                               std::vector<int> drafted,
                               std::vector<int> accepted) {
    json j;
    j["schema"]          = "llama.tessera.spec.v1";
    j["seq_id"]          = 0;
    j["step_idx"]        = 0;
    j["prime_token"]     = 42;
    j["drafted"]         = n_dft;
    j["accepted"]        = n_acc;
    j["topk"]            = 2;
    j["drafted_tokens"]  = drafted;
    j["accepted_tokens"] = accepted;
    j["verifier_argmax"] = accepted;
    j["drafter_argmax"]  = drafted;
    // Minimal top-k (2 positions, 2 entries each).
    j["verifier_topk_tokens"] = json::array({json::array({10, 11}), json::array({12, 13})});
    j["verifier_topk_probs"]  = json::array({json::array({0.8, 0.2}), json::array({0.7, 0.3})});
    j["drafter_topk_tokens"]  = json::array({json::array({10, 11}), json::array({12, 14})});
    j["drafter_topk_probs"]   = json::array({json::array({0.6, 0.4}), json::array({0.5, 0.5})});
    return j.dump();
}

// A llama.tessera.spec.v1 record that also carries per-position confidence[],
// as the real imatrix spec-calibration path emits. verifier_argmax is the
// per-position ground truth (length n_dft+1); confidence is the acceptance
// proxy (length n_dft).
static std::string make_dflash_record(int n_acc, int n_dft,
                                      std::vector<int> drafted,
                                      std::vector<int> verifier_argmax,
                                      std::vector<double> confidence) {
    json j;
    j["schema"]          = "llama.tessera.spec.v1";
    j["seq_id"]          = 0;
    j["step_idx"]        = 0;
    j["prime_token"]     = 42;
    j["drafted"]         = n_dft;
    j["accepted"]        = n_acc;
    j["topk"]            = 2;
    j["drafted_tokens"]  = drafted;
    j["verifier_argmax"] = verifier_argmax;
    j["confidence"]      = confidence;
    return j.dump();
}

// -------------------------------------------------------------------------

static void test_mode_from_string() {
    ts_dataset_mode m;
    CHECK(ts_dataset_mode_from_string("text",  &m) == 0 && m == TS_DATASET_MODE_TEXT,  "mode text");
    CHECK(ts_dataset_mode_from_string("pairs", &m) == 0 && m == TS_DATASET_MODE_PAIRS, "mode pairs");
    CHECK(ts_dataset_mode_from_string("lk",    &m) == 0 && m == TS_DATASET_MODE_LK,    "mode lk");
    CHECK(ts_dataset_mode_from_string("dflash", &m) == 0 && m == TS_DATASET_MODE_DFLASH, "mode dflash");
    CHECK(ts_dataset_mode_from_string("bogus", &m) != 0, "mode bogus rejected");
}

static void test_text_mode() {
    const std::string jsonl =
        make_record(2, 3, {101, 102, 103}, {101, 102, 999}) + "\n" +
        make_record(1, 2, {201, 202}, {201, 888}) + "\n";
    const std::string in_path  = write_tmp("text_in.jsonl", jsonl);
    const std::string out_path = "/tmp/ts_dataset_test_text_out.txt";

    ts_dataset_params p;
    ts_dataset_default_params(&p);
    snprintf(p.input_path,  sizeof(p.input_path),  "%s", in_path.c_str());
    snprintf(p.output_path, sizeof(p.output_path), "%s", out_path.c_str());
    p.mode = TS_DATASET_MODE_TEXT;

    int n = 0;
    std::string err;
    const int rc = ts_dataset_run(&p, &n, nullptr, &err);
    CHECK(rc == 0, "text mode rc");
    CHECK(n == 2, "text mode 2 records");

    const std::string out = read_file(out_path);
    CHECK(out.find("101 102 999") != std::string::npos, "text line 1");
    CHECK(out.find("201 888")     != std::string::npos, "text line 2");
}

static void test_pairs_mode() {
    const std::string jsonl = make_record(2, 3, {101, 102, 103}, {101, 102, 999}) + "\n";
    const std::string in_path  = write_tmp("pairs_in.jsonl", jsonl);
    const std::string out_path = "/tmp/ts_dataset_test_pairs_out.jsonl";

    ts_dataset_params p;
    ts_dataset_default_params(&p);
    snprintf(p.input_path,  sizeof(p.input_path),  "%s", in_path.c_str());
    snprintf(p.output_path, sizeof(p.output_path), "%s", out_path.c_str());
    p.mode = TS_DATASET_MODE_PAIRS;

    int n = 0;
    std::string err;
    CHECK(ts_dataset_run(&p, &n, nullptr, &err) == 0, "pairs mode rc");
    CHECK(n == 1, "pairs mode 1 record");

    const std::string out = read_file(out_path);
    json rec = json::parse(out);
    CHECK(rec["context"].get<int>() == 42, "pairs context=prime_token");
    CHECK(rec["n_acc"].get<int>() == 2, "pairs n_acc");
    CHECK(rec["n_dft"].get<int>() == 3, "pairs n_dft");
    CHECK(rec["drafted"].size() == 3, "pairs drafted len");
    CHECK(rec["accepted"].size() == 3, "pairs accepted len");
}

static void test_lk_mode() {
    const std::string jsonl = make_record(2, 3, {101, 102, 103}, {101, 102, 999}) + "\n";
    const std::string in_path  = write_tmp("lk_in.jsonl", jsonl);
    const std::string out_path = "/tmp/ts_dataset_test_lk_out.jsonl";

    ts_dataset_params p;
    ts_dataset_default_params(&p);
    snprintf(p.input_path,  sizeof(p.input_path),  "%s", in_path.c_str());
    snprintf(p.output_path, sizeof(p.output_path), "%s", out_path.c_str());
    p.mode = TS_DATASET_MODE_LK;

    int n = 0;
    std::string err;
    CHECK(ts_dataset_run(&p, &n, nullptr, &err) == 0, "lk mode rc");
    // 2 positions in the top-k arrays -> 2 output records
    CHECK(n == 1, "lk mode 1 step record");

    const std::string out = read_file(out_path);
    // Should have 2 lines (one per position)
    int lines = 0;
    for (char c : out) if (c == '\n') lines++;
    CHECK(lines == 2, "lk mode 2 position lines");

    // First position: drafted[0]=101 == accepted[0]=101 -> accepted=true
    json first = json::parse(out.substr(0, out.find('\n')));
    CHECK(first["position"].get<int>() == 0, "lk pos 0");
    CHECK(first["accepted"].get<bool>() == true, "lk pos 0 accepted");
    CHECK(first["p_tokens"].size() == 2, "lk pos 0 p_tokens len");
    CHECK(first["q_tokens"].size() == 2, "lk pos 0 q_tokens len");
}

static void test_min_accepted_filter() {
    // Record with 0 accepted should be skipped with min_accepted=1
    const std::string jsonl =
        make_record(0, 3, {101, 102, 103}, {999}) + "\n" +
        make_record(2, 3, {201, 202, 203}, {201, 202, 888}) + "\n";
    const std::string in_path  = write_tmp("filter_in.jsonl", jsonl);
    const std::string out_path = "/tmp/ts_dataset_test_filter_out.txt";

    ts_dataset_params p;
    ts_dataset_default_params(&p);
    snprintf(p.input_path,  sizeof(p.input_path),  "%s", in_path.c_str());
    snprintf(p.output_path, sizeof(p.output_path), "%s", out_path.c_str());
    p.mode = TS_DATASET_MODE_TEXT;
    p.min_accepted = 1;

    int n = 0;
    std::string err;
    CHECK(ts_dataset_run(&p, &n, nullptr, &err) == 0, "filter rc");
    CHECK(n == 1, "filter skips 0-accepted record");
}

static void test_wrong_schema_skipped() {
    // Record carrying an unrelated schema name should be skipped while a
    // llama.tessera.spec.v1 record in the same file is processed.
    const std::string jsonl =
        "{\"schema\":\"some.other.schema\",\"seq_id\":0,\"drafted\":4,\"accepted\":3}\n" +
        make_record(2, 3, {101, 102, 103}, {101, 102, 999}) + "\n";
    const std::string in_path  = write_tmp("schema_in.jsonl", jsonl);
    const std::string out_path = "/tmp/ts_dataset_test_schema_out.txt";

    ts_dataset_params p;
    ts_dataset_default_params(&p);
    snprintf(p.input_path,  sizeof(p.input_path),  "%s", in_path.c_str());
    snprintf(p.output_path, sizeof(p.output_path), "%s", out_path.c_str());
    p.mode = TS_DATASET_MODE_TEXT;

    int n = 0;
    std::string err;
    CHECK(ts_dataset_run(&p, &n, nullptr, &err) == 0, "schema rc");
    CHECK(n == 1, "unrelated-schema record skipped, spec.v1 processed");
}

static void test_missing_input() {
    ts_dataset_params p;
    ts_dataset_default_params(&p);
    snprintf(p.input_path, sizeof(p.input_path), "/tmp/ts_nonexistent_dataset.jsonl");
    snprintf(p.output_path, sizeof(p.output_path), "/tmp/ts_dataset_test_missing_out.txt");
    int n = 0;
    std::string err;
    CHECK(ts_dataset_run(&p, &n, nullptr, &err) != 0, "missing input rejected");
}

static void test_dflash_mode() {
    // 3-position drafted block; acceptance falls off along the block so the
    // adaptive D-PACE weight should concentrate on the anchor.
    const std::string jsonl =
        make_dflash_record(2, 3, {101, 102, 104},
                           {101, 102, 103, 999}, {0.9, 0.5, 0.2}) + "\n";
    const std::string in_path  = write_tmp("dflash_in.jsonl", jsonl);
    const std::string out_path = "/tmp/ts_dataset_test_dflash_out.jsonl";

    ts_dataset_params p;
    ts_dataset_default_params(&p);
    snprintf(p.input_path,  sizeof(p.input_path),  "%s", in_path.c_str());
    snprintf(p.output_path, sizeof(p.output_path), "%s", out_path.c_str());
    p.mode = TS_DATASET_MODE_DFLASH;

    int n = 0;
    std::string err;
    CHECK(ts_dataset_run(&p, &n, nullptr, &err) == 0, "dflash mode rc");
    CHECK(n == 1, "dflash mode 1 block");

    const std::string out = read_file(out_path);
    json rec = json::parse(out.substr(0, out.find('\n')));
    CHECK(rec["schema"].get<std::string>() == "llama.tessera.dflash-block.v1", "dflash schema");
    CHECK(rec["block_size"].get<int>() == 3, "dflash block_size");
    CHECK(rec["n_acc"].get<int>() == 2, "dflash n_acc");
    CHECK(rec["n_dft"].get<int>() == 3, "dflash n_dft");

    // target_tokens = verifier_argmax[0..n_dft-1]
    CHECK(rec["target_tokens"].size() == 3, "dflash target_tokens len");
    CHECK(rec["target_tokens"][0].get<int>() == 101, "dflash target 0");
    CHECK(rec["target_tokens"][2].get<int>() == 103, "dflash target 2");

    // acceptance_probs echoed from confidence[]
    CHECK(rec["acceptance_probs"].size() == 3, "dflash acceptance_probs len");
    CHECK(std::abs(rec["acceptance_probs"][0].get<double>() - 0.9) < 1e-6, "dflash acc prob 0");

    // weights are normalized to sum == block_size (mean 1.0)
    double dpace_sum = 0.0, decay_sum = 0.0;
    for (const auto & w : rec["dpace_weights"]) dpace_sum += w.get<double>();
    for (const auto & w : rec["decay_weights"]) decay_sum += w.get<double>();
    CHECK(std::abs(dpace_sum - 3.0) < 1e-6, "dflash dpace weights normalized");
    CHECK(std::abs(decay_sum - 3.0) < 1e-6, "dflash decay weights normalized");

    // anchor concentrates adaptive weight; decay is monotonic
    CHECK(rec["dpace_weights"][0].get<double>() > rec["dpace_weights"][2].get<double>(),
          "dflash dpace anchor > tail");
    CHECK(rec["decay_weights"][0].get<double>() > rec["decay_weights"][1].get<double>() &&
          rec["decay_weights"][1].get<double>() > rec["decay_weights"][2].get<double>(),
          "dflash decay monotonic");

    CHECK(rec["surrogate"].get<double>() > 0.0, "dflash surrogate positive");
}

static void test_dflash_requires_confidence() {
    // A spec.v1 record without confidence[] cannot form a block -> skipped.
    const std::string jsonl =
        make_record(2, 3, {101, 102, 103}, {101, 102, 999}) + "\n";
    const std::string in_path  = write_tmp("dflash_noconf_in.jsonl", jsonl);
    const std::string out_path = "/tmp/ts_dataset_test_dflash_noconf_out.jsonl";

    ts_dataset_params p;
    ts_dataset_default_params(&p);
    snprintf(p.input_path,  sizeof(p.input_path),  "%s", in_path.c_str());
    snprintf(p.output_path, sizeof(p.output_path), "%s", out_path.c_str());
    p.mode = TS_DATASET_MODE_DFLASH;

    int n = 0;
    std::string err;
    CHECK(ts_dataset_run(&p, &n, nullptr, &err) == 0, "dflash no-confidence rc");
    CHECK(n == 0, "dflash skips record lacking confidence");
}

// -------------------------------------------------------------------------

int main() {
    test_mode_from_string();
    test_text_mode();
    test_pairs_mode();
    test_lk_mode();
    test_min_accepted_filter();
    test_wrong_schema_skipped();
    test_missing_input();
    test_dflash_mode();
    test_dflash_requires_confidence();

    printf("dataset: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
