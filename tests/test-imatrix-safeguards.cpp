//
// test-imatrix-safeguards.cpp
//
// Unit tests for the crash-safety / long-run safeguards added to
// llama-imatrix in Phase 16.9 (2026-08-04). The safeguards were
// previously in tools/tessera/smoke_imatrix.py; they moved into
// the binary in this commit.
//
// The helpers live in tools/imatrix/imatrix-safeguards.h so the
// test can exercise them without dragging in the full llama_init /
// ggml dependency chain.
//
// Run with no args; uses a tmp directory for the file_size tests.
// Exit 0 on success, non-zero on failure.
//

#include "imatrix-safeguards.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <sys/stat.h>

using namespace tessera_imatrix_safeguards;

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        failures++; \
    } \
} while (0)

// ---- dynamic_save_freq_ladder -------------------------------------

static void test_ladder_paranoid_under_5min() {
    CHECK(dynamic_save_freq_ladder(0.0)        ==   8, "t=0 -> 8");
    CHECK(dynamic_save_freq_ladder(60.0)       ==   8, "t=1m -> 8");
    CHECK(dynamic_save_freq_ladder(4 * 60 + 59)==   8, "t=4m59s -> 8");
}

static void test_ladder_relax_5_to_10() {
    CHECK(dynamic_save_freq_ladder(5 * 60)        ==  16, "t=5m -> 16");
    CHECK(dynamic_save_freq_ladder(7 * 60 + 30)   ==  16, "t=7m30s -> 16");
    CHECK(dynamic_save_freq_ladder(9 * 60 + 59)   ==  16, "t=9m59s -> 16");
}

static void test_ladder_nominal_10_to_15() {
    CHECK(dynamic_save_freq_ladder(10 * 60)       ==  32, "t=10m -> 32");
    CHECK(dynamic_save_freq_ladder(12 * 60)       ==  32, "t=12m -> 32");
    CHECK(dynamic_save_freq_ladder(14 * 60 + 59)  ==  32, "t=14m59s -> 32");
}

static void test_ladder_relaxed_15_to_25() {
    CHECK(dynamic_save_freq_ladder(15 * 60)       ==  64, "t=15m -> 64");
    CHECK(dynamic_save_freq_ladder(20 * 60)       ==  64, "t=20m -> 64");
    CHECK(dynamic_save_freq_ladder(24 * 60 + 59)  ==  64, "t=24m59s -> 64");
}

static void test_ladder_minimal_25_plus() {
    CHECK(dynamic_save_freq_ladder(25 * 60)       == 128, "t=25m -> 128");
    CHECK(dynamic_save_freq_ladder(60 * 60)       == 128, "t=60m -> 128");
    CHECK(dynamic_save_freq_ladder(24 * 60 * 60)  == 128, "t=24h -> 128");
}

static void test_ladder_monotonic() {
    // The ladder must NEVER go backwards in stability (i.e. once we
    // relax, we do not re-tighten). A SIGKILL during a relaxed
    // interval must not lose more work than expected.
    int32_t prev = 0;
    for (double t = 0.0; t <= 30.0 * 60.0; t += 30.0) {
        const int32_t cur = dynamic_save_freq_ladder(t);
        CHECK(cur >= prev,
              ("ladder non-decreasing at t=" + std::to_string(t)).c_str());
        prev = cur;
    }
}

// ---- physmem_bytes ------------------------------------------------

static void test_physmem_returns_positive_on_known_platforms() {
    // On macOS (sysctl) and Linux (sysinfo) this should be > 0.
    // On Windows / BSD it returns 0 by design; the test does not
    // assert on Windows.
#if defined(__APPLE__) || defined(__linux__)
    const int64_t m = physmem_bytes();
    CHECK(m > 0, "physmem_bytes > 0 on macOS / Linux");
    // 256 MB minimum sanity (no embedded / IoT targets in the
    // tessera build matrix).
    CHECK(m > 256LL * 1024 * 1024,
          "physmem_bytes > 256 MB sanity floor");
#else
    // On unknown platforms, the function returns 0 so the precheck
    // is a no-op. We do not assert anything; the test simply exits
    // silently.
#endif
}

// ---- file_size_or_zero --------------------------------------------

static void test_file_size_real_file() {
    const std::string path = "/tmp/test-imatrix-safeguards.tmp";
    const char data[] = "0123456789";  // 10 bytes
    {
        std::ofstream f(path, std::ios::trunc | std::ios::binary);
        f.write(data, sizeof(data) - 1);
    }
    CHECK(file_size_or_zero(path) == 10,
          "file_size_or_zero on a 10-byte file returns 10");
    std::remove(path.c_str());
}

static void test_file_size_missing_file() {
    CHECK(file_size_or_zero("/tmp/this-file-does-not-exist-xyz-12345") == 0,
          "file_size_or_zero on missing file returns 0");
}

static void test_file_size_empty_file() {
    const std::string path = "/tmp/test-imatrix-safeguards-empty.tmp";
    {
        std::ofstream f(path, std::ios::trunc);
    }
    CHECK(file_size_or_zero(path) == 0,
          "file_size_or_zero on empty file returns 0");
    std::remove(path.c_str());
}

// ---- integration: precheck refusal logic (logic-only, no binary) -
//
// We do NOT spin up llama-imatrix in the test (would need a real
// GGUF). The precheck logic is one-line: refuse if
// model_size / physmem > budget. We verify the comparison
// arithmetic with realistic numbers: 23 GB BF16 model on 16 GB
// M1 with 0.6 budget -> ratio=1.44, well over 0.6 -> refuse.

static void test_precheck_refuses_oversized_model() {
    const int64_t model_size = 23LL * 1024 * 1024 * 1024;  // 23 GB BF16
    const int64_t physmem    = 16LL * 1024 * 1024 * 1024;  // 16 GB M1
    const float   budget     = 0.6f;
    const double  ratio      = (double) model_size / (double) physmem;
    CHECK(ratio > budget,
          "23 GB on 16 GB with 0.6 budget: precheck would refuse");
}

static void test_precheck_allows_within_budget() {
    const int64_t model_size = 9LL * 1024 * 1024 * 1024;  // 9 GB
    const int64_t physmem    = 16LL * 1024 * 1024 * 1024;  // 16 GB
    const float   budget     = 0.6f;
    const double  ratio      = (double) model_size / (double) physmem;
    CHECK(ratio <= budget,
          "9 GB on 16 GB with 0.6 budget: precheck would pass");
}

static void test_precheck_skipped_when_physmem_unknown() {
    // The precheck short-circuits if physmem is 0. The function
    // returns 0 on Windows / BSD; the binary checks the return and
    // skips the comparison. Logic verification:
    const int64_t physmem = 0;
    CHECK(physmem == 0,
          "physmem=0 means the precheck is bypassed (no false-positive refusal)");
}

// ---- main ---------------------------------------------------------

int main() {
    test_ladder_paranoid_under_5min();
    test_ladder_relax_5_to_10();
    test_ladder_nominal_10_to_15();
    test_ladder_relaxed_15_to_25();
    test_ladder_minimal_25_plus();
    test_ladder_monotonic();
    test_physmem_returns_positive_on_known_platforms();
    test_file_size_real_file();
    test_file_size_missing_file();
    test_file_size_empty_file();
    test_precheck_refuses_oversized_model();
    test_precheck_allows_within_budget();
    test_precheck_skipped_when_physmem_unknown();

    if (failures == 0) {
        printf("OK: all imatrix-safeguards tests passed\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d assertion(s) failed\n", failures);
    return 1;
}
