//
// test_dflash_train_data.cpp
//
// Unit tests for tessera-dflash-train-data.{h,cpp}: the dflash-block.v1
// JSONL -> (tokens, sparse labels, D-PACE weights) densification pass used
// by tessera-train-dflash. Mirrors test_lk_train_data.cpp's structure.
//

#include "tessera-dflash-train-data.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; std::fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

#define EPS 1e-6f

// Write a list of JSONL lines to a file. Returns the path.
static std::string write_jsonl(const std::vector<std::string> & lines) {
    const std::string p = "/tmp/ts_dflash_train_test.jsonl";
    std::ofstream f(p, std::ios::binary);
    for (const auto & l : lines) {
        f << l << '\n';
    }
    return p;
}

static bool close_enough(float a, float b) {
    return std::fabs((double) a - (double) b) < (double) EPS;
}

static void test_usable_positive() {
    // A well-formed dflash-block.v1 record with n_dft=3 is usable.
    const std::string line =
        R"({"schema":"llama.tessera.dflash-block.v1","block_size":3,"n_dft":3,)"
        R"("target_tokens":[10,20,30],"acceptance_probs":[0.9,0.8,0.7],)"
        R"("dpace_weights":[0.5,0.7,0.9],"decay_weights":[1.0,0.5,0.25],)"
        R"("n_acc":3,"surrogate":2.4})";
    CHECK(ts_dflash_train_line_usable(line.c_str(), 3) == 1,
          "well-formed record usable");
}

static void test_usable_negative_wrong_schema() {
    const std::string line =
        R"({"schema":"llama.tessera.spec.v1","drafted":3,"prime_token":1,)"
        R"("verifier_argmax":[10,20,30]})";
    CHECK(ts_dflash_train_line_usable(line.c_str(), 3) == 0,
          "wrong schema rejected");
}

static void test_usable_negative_wrong_block_size() {
    // n_dft=3 but caller asks for block_size=4: not usable.
    const std::string line =
        R"({"schema":"llama.tessera.dflash-block.v1","block_size":3,"n_dft":3,)"
        R"("target_tokens":[10,20,30],"acceptance_probs":[0.9,0.8,0.7],)"
        R"("dpace_weights":[0.5,0.7,0.9],"decay_weights":[1.0,0.5,0.25],)"
        R"("n_acc":3,"surrogate":2.4})";
    CHECK(ts_dflash_train_line_usable(line.c_str(), 4) == 0,
          "block-size mismatch rejected");
}

static void test_usable_negative_missing_field() {
    // missing dpace_weights -> not usable.
    const std::string line =
        R"({"schema":"llama.tessera.dflash-block.v1","block_size":3,"n_dft":3,)"
        R"("target_tokens":[10,20,30],"acceptance_probs":[0.9,0.8,0.7],)"
        R"("decay_weights":[1.0,0.5,0.25],"n_acc":3,"surrogate":2.4})";
    CHECK(ts_dflash_train_line_usable(line.c_str(), 3) == 0,
          "missing dpace_weights rejected");
}

static void test_example_from_line_dpace() {
    // B=3, three positions. dpace_weights baked in: [0.1, 0.2, 0.3].
    const std::string line =
        R"({"schema":"llama.tessera.dflash-block.v1","block_size":3,"n_dft":3,)"
        R"("target_tokens":[10,20,30],"acceptance_probs":[0.9,0.8,0.7],)"
        R"("dpace_weights":[0.1,0.2,0.3],"decay_weights":[1.0,0.5,0.25],)"
        R"("n_acc":3,"surrogate":2.4})";

    int32_t tokens[4]   = { -1, -1, -1, -1 };
    int32_t labels[4]   = { -1, -1, -1, -1 };
    float   weights[4]  = { -1, -1, -1, -1 };
    const int rc = ts_dflash_train_example_from_line(line.c_str(), 3, /*scheme=*/0,
                                                     tokens, labels, weights);
    CHECK(rc == 1, "example written");

    // Pos 0: anchor. Token/labels are sentinels; weight must be 0.
    CHECK(weights[0] == 0.0f, "anchor weight = 0");
    CHECK(tokens[0] == 0, "anchor token sentinel = 0");
    CHECK(labels[0] == 0, "anchor label sentinel = 0");

    // Pos 1..3 (j+1 in our terms): drafted positions.
    // tokens come from accepted tokens in the dflash path. Here we use the
    // target_tokens as the on-policy prefix (the dataset does not store
    // drafted_tokens in dflash-block.v1, by design - the drafter
    // conditions on its own outputs in the training graph).
    // For our purposes, the contract is: tokens[j+1] = target_tokens[j].
    CHECK(tokens[1] == 10, "pos 1 token");
    CHECK(tokens[2] == 20, "pos 2 token");
    CHECK(tokens[3] == 30, "pos 3 token");

    CHECK(labels[1] == 10, "pos 1 label = target[0]");
    CHECK(labels[2] == 20, "pos 2 label = target[1]");
    CHECK(labels[3] == 30, "pos 3 label = target[2]");

    CHECK(close_enough(weights[1], 0.1f), "pos 1 dpace weight");
    CHECK(close_enough(weights[2], 0.2f), "pos 2 dpace weight");
    CHECK(close_enough(weights[3], 0.3f), "pos 3 dpace weight");
}

static void test_example_from_line_decay_scheme() {
    // Same record, but weight_scheme=1 (decay) -> use decay_weights.
    const std::string line =
        R"({"schema":"llama.tessera.dflash-block.v1","block_size":3,"n_dft":3,)"
        R"("target_tokens":[10,20,30],"acceptance_probs":[0.9,0.8,0.7],)"
        R"("dpace_weights":[0.1,0.2,0.3],"decay_weights":[1.0,0.5,0.25],)"
        R"("n_acc":3,"surrogate":2.4})";

    int32_t tokens[4]  = {};
    int32_t labels[4]  = {};
    float   weights[4] = {};
    const int rc = ts_dflash_train_example_from_line(line.c_str(), 3, /*scheme=*/1,
                                                     tokens, labels, weights);
    CHECK(rc == 1, "example written (decay scheme)");

    CHECK(weights[0] == 0.0f, "anchor weight = 0 (decay scheme)");
    CHECK(close_enough(weights[1], 1.0f),  "pos 1 decay weight");
    CHECK(close_enough(weights[2], 0.5f),  "pos 2 decay weight");
    CHECK(close_enough(weights[3], 0.25f), "pos 3 decay weight");
}

static void test_example_from_line_unusable() {
    // Wrong schema: returns 0, buffers untouched.
    const std::string line =
        R"({"schema":"llama.tessera.spec.v1","drafted":3,"prime_token":1})";
    int32_t tokens[4]  = { 7, 7, 7, 7 };
    int32_t labels[4]  = { 7, 7, 7, 7 };
    float   weights[4] = { 7, 7, 7, 7 };
    const int rc = ts_dflash_train_example_from_line(line.c_str(), 3, 0,
                                                     tokens, labels, weights);
    CHECK(rc == 0, "unusable line returns 0");
    CHECK(tokens[0] == 7 && tokens[1] == 7 && tokens[2] == 7 && tokens[3] == 7,
          "buffers untouched on skip");
}

static void test_detect_block_size_modal() {
    // Three records: n_dft=3, n_dft=3, n_dft=5. Modal = 3.
    const std::vector<std::string> lines = {
        R"({"schema":"llama.tessera.dflash-block.v1","block_size":3,"n_dft":3,"target_tokens":[1,2,3],"acceptance_probs":[0.9,0.8,0.7],"dpace_weights":[0.5,0.5,0.5],"decay_weights":[1.0,0.5,0.25],"n_acc":3,"surrogate":2.0})",
        R"({"schema":"llama.tessera.dflash-block.v1","block_size":3,"n_dft":3,"target_tokens":[4,5,6],"acceptance_probs":[0.9,0.8,0.7],"dpace_weights":[0.5,0.5,0.5],"decay_weights":[1.0,0.5,0.25],"n_acc":3,"surrogate":2.0})",
        R"({"schema":"llama.tessera.dflash-block.v1","block_size":5,"n_dft":5,"target_tokens":[1,2,3,4,5],"acceptance_probs":[0.9,0.8,0.7,0.6,0.5],"dpace_weights":[0.4,0.4,0.4,0.4,0.4],"decay_weights":[1.0,0.7,0.5,0.3,0.2],"n_acc":5,"surrogate":3.0})",
    };
    const std::string p = write_jsonl(lines);
    const int n = ts_dflash_train_detect_block_size(p.c_str());
    CHECK(n == 3, "modal n_dft = 3");
}

static void test_detect_block_size_tiebreak() {
    // Two records, one each: tie-break on smaller.
    const std::vector<std::string> lines = {
        R"({"schema":"llama.tessera.dflash-block.v1","block_size":3,"n_dft":3,"target_tokens":[1,2,3],"acceptance_probs":[0.9,0.8,0.7],"dpace_weights":[0.5,0.5,0.5],"decay_weights":[1.0,0.5,0.25],"n_acc":3,"surrogate":2.0})",
        R"({"schema":"llama.tessera.dflash-block.v1","block_size":5,"n_dft":5,"target_tokens":[1,2,3,4,5],"acceptance_probs":[0.9,0.8,0.7,0.6,0.5],"dpace_weights":[0.4,0.4,0.4,0.4,0.4],"decay_weights":[1.0,0.7,0.5,0.3,0.2],"n_acc":5,"surrogate":3.0})",
    };
    const std::string p = write_jsonl(lines);
    const int n = ts_dflash_train_detect_block_size(p.c_str());
    CHECK(n == 3, "tie-break picks smaller n_dft");
}

static void test_detect_block_size_empty() {
    // Empty file: -1.
    const std::string p = "/tmp/ts_dflash_train_empty.jsonl";
    std::ofstream f(p, std::ios::binary);
    // no lines
    const int n = ts_dflash_train_detect_block_size(p.c_str());
    CHECK(n == -1, "empty file -> -1");
}

static void test_detect_block_size_wrong_schema() {
    // All wrong-schema lines: -1.
    const std::vector<std::string> lines = {
        R"({"schema":"llama.tessera.spec.v1","drafted":3})",
        R"({"schema":"some.other.schema","step":1})",
    };
    const std::string p = write_jsonl(lines);
    const int n = ts_dflash_train_detect_block_size(p.c_str());
    CHECK(n == -1, "no dflash-block records -> -1");
}

static void test_off_by_one_anchor_mapping() {
    // The design contract: dataset target_tokens[k] -> model pos k+1, anchor
    // pos 0 carries weight 0. Verify with a single example.
    const std::string line =
        R"({"schema":"llama.tessera.dflash-block.v1","block_size":4,"n_dft":4,)"
        R"("target_tokens":[100,200,300,400],)"
        R"("acceptance_probs":[0.95,0.9,0.85,0.8],)"
        R"("dpace_weights":[0.25,0.3,0.35,0.4],)"
        R"("decay_weights":[1.0,0.5,0.25,0.125],"n_acc":4,"surrogate":3.0})";

    const int B = 4;
    int32_t tokens[5]  = {};
    int32_t labels[5]  = {};
    float   weights[5] = {};
    const int rc = ts_dflash_train_example_from_line(line.c_str(), B, 0,
                                                     tokens, labels, weights);
    CHECK(rc == 1, "B=4 example written");
    CHECK(weights[0] == 0.0f, "anchor weight = 0 (off-by-one invariant)");

    // pos 1 = dataset pos 0
    CHECK(labels[1] == 100, "pos 1 = target[0]");
    CHECK(close_enough(weights[1], 0.25f), "pos 1 weight = dpace[0]");
    // pos 2 = dataset pos 1
    CHECK(labels[2] == 200, "pos 2 = target[1]");
    CHECK(close_enough(weights[2], 0.3f), "pos 2 weight = dpace[1]");
    // pos 3 = dataset pos 2
    CHECK(labels[3] == 300, "pos 3 = target[2]");
    CHECK(close_enough(weights[3], 0.35f), "pos 3 weight = dpace[2]");
    // pos 4 = dataset pos 3
    CHECK(labels[4] == 400, "pos 4 = target[3]");
    CHECK(close_enough(weights[4], 0.4f), "pos 4 weight = dpace[3]");
}

int main() {
    test_usable_positive();
    test_usable_negative_wrong_schema();
    test_usable_negative_wrong_block_size();
    test_usable_negative_missing_field();
    test_example_from_line_dpace();
    test_example_from_line_decay_scheme();
    test_example_from_line_unusable();
    test_detect_block_size_modal();
    test_detect_block_size_tiebreak();
    test_detect_block_size_empty();
    test_detect_block_size_wrong_schema();
    test_off_by_one_anchor_mapping();

    std::printf("dflash_train_data: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
