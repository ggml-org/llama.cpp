//
// Test: Dispatch Tuning Mapping
//
// Verifies that dispatch tuning maps "unified_matmul" winners to the
// UNIFIED_MATMUL kernel enum (orchestrator entrypoint).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "dispatch-tuning.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_BEGIN(name)                         \
    do {                                         \
        g_tests_run++;                           \
        fprintf(stderr, "[TEST] %s ... ", name); \
    } while (0)

#define TEST_PASS()                  \
    do {                             \
        g_tests_passed++;            \
        fprintf(stderr, "PASSED\n"); \
    } while (0)

#define TEST_FAIL(msg)                        \
    do {                                      \
        fprintf(stderr, "FAILED: %s\n", msg); \
        return false;                         \
    } while (0)

#define TEST_ASSERT(cond, msg) \
    do {                       \
        if (!(cond)) {         \
            TEST_FAIL(msg);    \
        }                      \
    } while (0)

static bool write_temp_file(std::string & path, const std::string & content) {
    char tmpl[] = "/tmp/ggml_dispatch_tuningXXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        return false;
    }
    FILE * file = fdopen(fd, "w");
    if (!file) {
        close(fd);
        return false;
    }
    const size_t written = fwrite(content.data(), 1, content.size(), file);
    fclose(file);
    path = tmpl;
    return written == content.size();
}

static bool test_unified_matmul_mapping() {
    TEST_BEGIN("unified_matmul winner maps to UNIFIED_MATMUL");
    const std::string json =
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"quant\": \"Q4_0\",\n"
        "      \"dim_m\": 1,\n"
        "      \"dim_n\": 256,\n"
        "      \"dim_k\": 256,\n"
        "      \"tensor_instances\": 1,\n"
        "      \"winner\": \"unified_matmul\"\n"
        "    }\n"
        "  ]\n"
        "}\n";

    std::string path;
    TEST_ASSERT(write_temp_file(path, json), "failed to create temp tuning file");

    ggml_sycl::dispatch_tuning::DispatchTuningCache cache;
    std::string error;
    const bool loaded = ggml_sycl::dispatch_tuning::load_dispatch_tuning_from_file(path, cache, &error);
    std::remove(path.c_str());

    TEST_ASSERT(loaded, "failed to load tuning entries");
    auto key = ggml_sycl::dispatch_tuning::make_dispatch_tuning_key(GGML_TYPE_Q4_0, 1, 256, 256);
    auto entry = cache.lookup(key);
    TEST_ASSERT(entry.has_value(), "missing tuning entry");
    TEST_ASSERT(entry->kernel == ggml_sycl_mul_mat_kernel::UNIFIED_MATMUL,
                "winner did not map to UNIFIED_MATMUL");

    TEST_PASS();
    return true;
}

int main() {
    bool ok = true;
    ok = test_unified_matmul_mapping() && ok;

    if (!ok || g_tests_passed != g_tests_run) {
        fprintf(stderr, "\n[TEST SUMMARY] %d/%d tests passed\n", g_tests_passed, g_tests_run);
        return 1;
    }
    fprintf(stderr, "\n[TEST SUMMARY] %d/%d tests passed\n", g_tests_passed, g_tests_run);
    return 0;
}
