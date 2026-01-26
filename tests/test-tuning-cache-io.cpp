//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Unit tests for tuning-cache-io.hpp
// Tests JSON serialization/deserialization, file I/O, and atomic writes

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "../ggml/src/ggml-sycl/tuning-cache-io.hpp"

using namespace ggml_sycl_tuning;

// Test counter
static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) \
    static bool test_##name(); \
    static bool test_##name##_registered = [] { \
        return true; \
    }(); \
    static bool test_##name()

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "  ASSERT FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

#define RUN_TEST(name) \
    do { \
        std::cout << "Running " << #name << "... "; \
        if (test_##name()) { \
            std::cout << "[PASS]\n"; \
            g_passed++; \
        } else { \
            std::cout << "[FAIL]\n"; \
            g_failed++; \
        } \
    } while(0)

// =============================================================================
// Test: get_cache_dir returns valid directory
// =============================================================================
TEST(get_cache_dir) {
    std::string dir = get_cache_dir();
    ASSERT(!dir.empty());
    ASSERT(dir.find("llama.cpp") != std::string::npos);
    ASSERT(dir.find("sycl-tuning") != std::string::npos);
    return true;
}

// =============================================================================
// Test: get_cache_dir respects XDG_CACHE_HOME
// =============================================================================
TEST(get_cache_dir_xdg) {
    // Save original value
    const char* original = std::getenv("XDG_CACHE_HOME");
    std::string original_str = original ? original : "";

    // Set custom XDG_CACHE_HOME
    setenv("XDG_CACHE_HOME", "/custom/cache/path", 1);
    std::string dir = get_cache_dir();
    ASSERT(dir == "/custom/cache/path/llama.cpp/sycl-tuning");

    // Restore original
    if (original_str.empty()) {
        unsetenv("XDG_CACHE_HOME");
    } else {
        setenv("XDG_CACHE_HOME", original_str.c_str(), 1);
    }
    return true;
}

// =============================================================================
// Test: sanitize_device_name handles special characters
// =============================================================================
TEST(sanitize_device_name) {
    // Normal device name with spaces
    ASSERT(sanitize_device_name("Intel Arc A770") == "Intel_Arc_A770");

    // Device name with special characters
    ASSERT(sanitize_device_name("Intel(R) Arc(TM) A770") == "IntelR_ArcTM_A770");

    // Device name with colons and slashes
    ASSERT(sanitize_device_name("Device: GPU/0") == "Device_GPU0");

    // Empty string
    ASSERT(sanitize_device_name("") == "");

    // Already clean name
    ASSERT(sanitize_device_name("Arc-A770_v2") == "Arc-A770_v2");

    return true;
}

// =============================================================================
// Test: get_cache_file generates valid path
// =============================================================================
TEST(get_cache_file) {
    std::string file = get_cache_file("Intel Arc A770");
    ASSERT(file.find("Intel_Arc_A770") != std::string::npos);
    ASSERT(file.find(".json") != std::string::npos);
    ASSERT(file.find("llama.cpp") != std::string::npos);
    return true;
}

// =============================================================================
// Test: params_to_json serialization
// =============================================================================
TEST(params_to_json) {
    TunedParams p;
    p.tile_m = 8;
    p.tile_n = 32;
    p.tile_k = 64;
    p.workgroup_size = 256;
    p.slm_kb = 32;
    p.prefetch_depth = 2;
    p.use_dpas = true;
    p.layout_mode = 2;

    std::string json = params_to_json(p);

    ASSERT(json.find("\"tile_m\":8") != std::string::npos);
    ASSERT(json.find("\"tile_n\":32") != std::string::npos);
    ASSERT(json.find("\"tile_k\":64") != std::string::npos);
    ASSERT(json.find("\"workgroup_size\":256") != std::string::npos);
    ASSERT(json.find("\"slm_kb\":32") != std::string::npos);
    ASSERT(json.find("\"prefetch_depth\":2") != std::string::npos);
    ASSERT(json.find("\"use_dpas\":true") != std::string::npos);
    ASSERT(json.find("\"layout_mode\":2") != std::string::npos);

    return true;
}

// =============================================================================
// Test: params_from_json deserialization
// =============================================================================
TEST(params_from_json) {
    std::string json = R"({"tile_m":16,"tile_n":64,"tile_k":32,"workgroup_size":128,"slm_kb":16,"prefetch_depth":3,"use_dpas":false,"layout_mode":1})";

    TunedParams p = params_from_json(json);

    ASSERT(p.tile_m == 16);
    ASSERT(p.tile_n == 64);
    ASSERT(p.tile_k == 32);
    ASSERT(p.workgroup_size == 128);
    ASSERT(p.slm_kb == 16);
    ASSERT(p.prefetch_depth == 3);
    ASSERT(p.use_dpas == false);
    ASSERT(p.layout_mode == 1);

    return true;
}

// =============================================================================
// Test: TunedParams roundtrip serialization
// =============================================================================
TEST(params_roundtrip) {
    TunedParams original;
    original.tile_m = 32;
    original.tile_n = 16;
    original.tile_k = 128;
    original.workgroup_size = 512;
    original.slm_kb = 64;
    original.prefetch_depth = 4;
    original.use_dpas = true;
    original.layout_mode = 3;

    std::string json = params_to_json(original);
    TunedParams restored = params_from_json(json);

    ASSERT(original.tile_m == restored.tile_m);
    ASSERT(original.tile_n == restored.tile_n);
    ASSERT(original.tile_k == restored.tile_k);
    ASSERT(original.workgroup_size == restored.workgroup_size);
    ASSERT(original.slm_kb == restored.slm_kb);
    ASSERT(original.prefetch_depth == restored.prefetch_depth);
    ASSERT(original.use_dpas == restored.use_dpas);
    ASSERT(original.layout_mode == restored.layout_mode);

    return true;
}

// =============================================================================
// Test: key_to_string and key_from_string
// =============================================================================
TEST(key_serialization) {
    TuningKey key;
    key.quant_type = 2;  // GGML_TYPE_Q4_0
    key.batch_bucket = BatchBucket::MEDIUM;
    key.K = 4096;
    key.N = 32000;

    std::string str = key_to_string(key);
    ASSERT(str == "2_2_4096_32000");

    TuningKey restored = key_from_string(str);
    ASSERT(restored.quant_type == 2);
    ASSERT(restored.batch_bucket == BatchBucket::MEDIUM);
    ASSERT(restored.K == 4096);
    ASSERT(restored.N == 32000);

    return true;
}

// =============================================================================
// Test: entry_to_json and entry_from_json
// =============================================================================
TEST(entry_serialization) {
    TuningEntry entry;
    entry.key.quant_type = 8;  // GGML_TYPE_Q8_0
    entry.key.batch_bucket = BatchBucket::SINGLE;
    entry.key.K = 4096;
    entry.key.N = 4096;
    entry.params.tile_m = 8;
    entry.params.tile_n = 32;
    entry.params.tile_k = 64;
    entry.params.workgroup_size = 256;
    entry.params.slm_kb = 32;
    entry.params.prefetch_depth = 2;
    entry.params.use_dpas = true;
    entry.params.layout_mode = 1;
    entry.measured_tflops = 12.5f;
    entry.timestamp = 1700000000;

    std::string json = entry_to_json(entry);

    // Verify key is present
    ASSERT(json.find("\"key\":\"8_0_4096_4096\"") != std::string::npos);
    ASSERT(json.find("\"params\":") != std::string::npos);
    ASSERT(json.find("\"measured_tflops\":12.5") != std::string::npos);
    ASSERT(json.find("\"timestamp\":1700000000") != std::string::npos);

    TuningEntry restored = entry_from_json(json);
    ASSERT(restored.key.quant_type == 8);
    ASSERT(restored.key.batch_bucket == BatchBucket::SINGLE);
    ASSERT(restored.key.K == 4096);
    ASSERT(restored.key.N == 4096);
    ASSERT(restored.params.tile_m == 8);
    ASSERT(restored.params.use_dpas == true);
    ASSERT(std::abs(restored.measured_tflops - 12.5f) < 0.01f);

    return true;
}

// =============================================================================
// Test: create_dir_recursive
// =============================================================================
TEST(create_dir_recursive) {
    std::string test_dir = "/tmp/llama_test_cache_dir_" + std::to_string(getpid());
    std::string nested_dir = test_dir + "/level1/level2/level3";

    // Create nested directory
    create_dir_recursive(nested_dir);

    // Check directory exists
    struct stat st;
    ASSERT(stat(nested_dir.c_str(), &st) == 0);
    ASSERT(S_ISDIR(st.st_mode));

    // Cleanup
    rmdir((test_dir + "/level1/level2/level3").c_str());
    rmdir((test_dir + "/level1/level2").c_str());
    rmdir((test_dir + "/level1").c_str());
    rmdir(test_dir.c_str());

    return true;
}

// =============================================================================
// Test: TuningCache for_each iteration
// =============================================================================
TEST(cache_for_each) {
    TuningCache cache;

    // Insert some entries
    TuningEntry e1;
    e1.key.quant_type = 2;
    e1.key.batch_bucket = BatchBucket::SINGLE;
    e1.key.K = 4096;
    e1.key.N = 4096;
    e1.params.tile_m = 8;
    cache.insert(e1);

    TuningEntry e2;
    e2.key.quant_type = 8;
    e2.key.batch_bucket = BatchBucket::MEDIUM;
    e2.key.K = 4096;
    e2.key.N = 32000;
    e2.params.tile_m = 16;
    cache.insert(e2);

    // Count entries via for_each
    int count = 0;
    cache.for_each([&count](const TuningKey& key, const TuningEntry& entry) {
        (void)key;
        (void)entry;
        count++;
    });

    ASSERT(count == 2);
    ASSERT(cache.size() == 2);

    return true;
}

// =============================================================================
// Test: save_cache and load_cache roundtrip
// =============================================================================
TEST(cache_file_roundtrip) {
    // Use unique device name to avoid conflicts
    std::string device_name = "TestDevice_" + std::to_string(getpid());

    // Create cache with entries
    TuningCache cache;

    TuningEntry e1;
    e1.key.quant_type = 2;
    e1.key.batch_bucket = BatchBucket::SINGLE;
    e1.key.K = 4096;
    e1.key.N = 4096;
    e1.params.tile_m = 8;
    e1.params.tile_n = 32;
    e1.params.tile_k = 64;
    e1.params.workgroup_size = 256;
    e1.params.slm_kb = 32;
    e1.params.prefetch_depth = 2;
    e1.params.use_dpas = true;
    e1.params.layout_mode = 1;
    e1.measured_tflops = 15.5f;
    e1.timestamp = 1700000000;
    cache.insert(e1);

    TuningEntry e2;
    e2.key.quant_type = 8;
    e2.key.batch_bucket = BatchBucket::MEDIUM;
    e2.key.K = 4096;
    e2.key.N = 32000;
    e2.params.tile_m = 16;
    e2.params.tile_n = 64;
    e2.params.tile_k = 32;
    e2.params.workgroup_size = 512;
    e2.params.slm_kb = 64;
    e2.params.prefetch_depth = 4;
    e2.params.use_dpas = false;
    e2.params.layout_mode = 2;
    e2.measured_tflops = 8.2f;
    e2.timestamp = 1700001000;
    cache.insert(e2);

    // Save cache
    bool save_result = save_cache(cache, device_name);
    ASSERT(save_result);

    // Verify file exists
    ASSERT(cache_exists(device_name));

    // Load into new cache
    TuningCache loaded_cache;
    bool load_result = load_cache(loaded_cache, device_name);
    ASSERT(load_result);

    // Verify loaded cache has same entries
    ASSERT(loaded_cache.size() == 2);

    // Verify first entry
    auto lookup1 = loaded_cache.lookup(e1.key);
    ASSERT(lookup1.has_value());
    ASSERT(lookup1->params.tile_m == 8);
    ASSERT(lookup1->params.use_dpas == true);

    // Verify second entry
    auto lookup2 = loaded_cache.lookup(e2.key);
    ASSERT(lookup2.has_value());
    ASSERT(lookup2->params.tile_m == 16);
    ASSERT(lookup2->params.use_dpas == false);

    // Cleanup
    delete_cache(device_name);
    ASSERT(!cache_exists(device_name));

    return true;
}

// =============================================================================
// Test: load_cache handles missing file gracefully
// =============================================================================
TEST(load_missing_file) {
    TuningCache cache;
    bool result = load_cache(cache, "NonExistentDevice_12345678");
    ASSERT(result == false);
    ASSERT(cache.size() == 0);
    return true;
}

// =============================================================================
// Test: version check rejects incompatible versions
// =============================================================================
TEST(version_check) {
    std::string device_name = "TestVersionDevice_" + std::to_string(getpid());
    std::string path = get_cache_file(device_name);

    // Create directory and write file with wrong version
    create_dir_recursive(get_cache_dir());
    std::ofstream f(path);
    f << "{\n";
    f << "  \"version\": 999,\n";  // Invalid version
    f << "  \"device\": \"" << device_name << "\",\n";
    f << "  \"entries\": []\n";
    f << "}\n";
    f.close();

    // Try to load - should fail due to version mismatch
    TuningCache cache;
    bool result = load_cache(cache, device_name);
    ASSERT(result == false);

    // Cleanup
    std::remove(path.c_str());

    return true;
}

// =============================================================================
// Test: CACHE_VERSION constant
// =============================================================================
TEST(cache_version) {
    ASSERT(CACHE_VERSION >= 1);
    return true;
}

// =============================================================================
// Test: parse_int handles edge cases
// =============================================================================
TEST(parse_int_edge_cases) {
    // Normal case
    ASSERT(parse_int("{\"value\":42}", "value") == 42);

    // Negative number
    ASSERT(parse_int("{\"value\":-123}", "value") == -123);

    // Zero
    ASSERT(parse_int("{\"value\":0}", "value") == 0);

    // Missing key
    ASSERT(parse_int("{\"other\":42}", "value") == 0);

    // With whitespace
    ASSERT(parse_int("{\"value\":  42}", "value") == 42);

    return true;
}

// =============================================================================
// Test: parse_bool handles edge cases
// =============================================================================
TEST(parse_bool_edge_cases) {
    // True
    ASSERT(parse_bool("{\"flag\":true}", "flag") == true);

    // False
    ASSERT(parse_bool("{\"flag\":false}", "flag") == false);

    // Missing key
    ASSERT(parse_bool("{\"other\":true}", "flag") == false);

    // With whitespace
    ASSERT(parse_bool("{\"flag\":  true}", "flag") == true);

    return true;
}

// =============================================================================
// Test: parse_string handles edge cases
// =============================================================================
TEST(parse_string_edge_cases) {
    // Normal string
    ASSERT(parse_string("{\"name\":\"hello\"}", "name") == "hello");

    // Empty string
    ASSERT(parse_string("{\"name\":\"\"}", "name") == "");

    // String with spaces
    ASSERT(parse_string("{\"name\":\"hello world\"}", "name") == "hello world");

    // Missing key
    ASSERT(parse_string("{\"other\":\"hello\"}", "name") == "");

    return true;
}

// =============================================================================
// Test: Atomic write behavior (temp file cleanup on success)
// =============================================================================
TEST(atomic_write) {
    std::string device_name = "TestAtomicDevice_" + std::to_string(getpid());
    std::string path = get_cache_file(device_name);
    std::string temp_path = path + ".tmp";

    TuningCache cache;
    TuningEntry e;
    e.key.quant_type = 2;
    e.key.batch_bucket = BatchBucket::SINGLE;
    e.key.K = 4096;
    e.key.N = 4096;
    cache.insert(e);

    // Save cache
    bool result = save_cache(cache, device_name);
    ASSERT(result);

    // Verify final file exists
    struct stat st;
    ASSERT(stat(path.c_str(), &st) == 0);

    // Verify temp file does NOT exist (was renamed)
    ASSERT(stat(temp_path.c_str(), &st) != 0);

    // Cleanup
    delete_cache(device_name);

    return true;
}

// =============================================================================
// Main test runner
// =============================================================================
int main() {
    std::cout << "=== Tuning Cache I/O Tests ===\n\n";

    RUN_TEST(get_cache_dir);
    RUN_TEST(get_cache_dir_xdg);
    RUN_TEST(sanitize_device_name);
    RUN_TEST(get_cache_file);
    RUN_TEST(params_to_json);
    RUN_TEST(params_from_json);
    RUN_TEST(params_roundtrip);
    RUN_TEST(key_serialization);
    RUN_TEST(entry_serialization);
    RUN_TEST(create_dir_recursive);
    RUN_TEST(cache_for_each);
    RUN_TEST(cache_file_roundtrip);
    RUN_TEST(load_missing_file);
    RUN_TEST(version_check);
    RUN_TEST(cache_version);
    RUN_TEST(parse_int_edge_cases);
    RUN_TEST(parse_bool_edge_cases);
    RUN_TEST(parse_string_edge_cases);
    RUN_TEST(atomic_write);

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << g_passed << ", Failed: " << g_failed << "\n";

    return g_failed > 0 ? 1 : 0;
}
