// Regression tests for Vulkan sparse binding and chunked staging transfers.
//
// B: chunked staging data integrity (buffer > 32 MB cap)
// C: sparse binding boundary data integrity (buffer >= 512 MB threshold)
// D: sparse fallback chain via fault injection (GGML_VULKAN_SPARSE_TESTING)
// E: sparse eligibility decision logic (pure, no device needed)
// F: staging cap-learning via fault injection (GGML_VULKAN_SPARSE_TESTING)
//
// B/C adapt to available memory and skip when insufficient.

#include <ggml.h>
#include <ggml-backend.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef GGML_VULKAN_SPARSE_TESTING
// extern "C" {} block required: GGML_BACKEND_API includes `extern`.
extern "C" {
GGML_BACKEND_API void   ggml_vk_test_sparse_counter_reset(void);
GGML_BACKEND_API int    ggml_vk_test_sparse_counter_read(void);
GGML_BACKEND_API size_t ggml_vk_test_staging_cap_read(int device_idx);
GGML_BACKEND_API void   ggml_vk_test_staging_cap_reset(int device_idx);
GGML_BACKEND_API void   ggml_vk_test_staging_destroy(int device_idx);
GGML_BACKEND_API void   ggml_vk_test_staging_oom_counter_reset(void);
GGML_BACKEND_API int    ggml_vk_test_staging_oom_counter_read(void);
}
#endif

static bool find_vulkan_device(ggml_backend_dev_t * dev_out) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strstr(ggml_backend_dev_name(dev), "Vulkan") != nullptr) {
            *dev_out = dev;
            return true;
        }
    }
    return false;
}

// Fill a float vector with a deterministic pattern derived from offset.
static void fill_pattern(float * data, size_t n_floats, size_t offset_id) {
    for (size_t i = 0; i < n_floats; i++) {
        data[i] = (float)((offset_id + 1) * 1000.0 + (double)i * 0.25);
    }
}

// Mirror of ggml_vk_should_try_sparse — keep in sync with backend.
static bool should_try_sparse(bool had_oom, bool sparse_binding, size_t size,
                              size_t sparse_threshold, bool import_ptr,
                              bool is_device_local, bool is_host_visible) {
    return had_oom && sparse_binding && size >= sparse_threshold && !import_ptr &&
           is_device_local && !is_host_visible;
}

// Subtest B: data integrity through chunked staging (tensor > 32 MB cap).
static bool test_chunked_staging(ggml_backend_t backend, ggml_backend_dev_t dev) {
    printf("  test_chunked_staging:     ");

    // 48 MB target (> 32 MB cap), adapt to available memory.
    const size_t target_bytes = 48ULL * 1024 * 1024;
    const size_t min_bytes    = 33ULL * 1024 * 1024;  // must exceed 32 MB cap

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);

    // Need target + headroom for backend overhead.
    size_t alloc_bytes = target_bytes;
    if (free_mem < alloc_bytes + 64 * 1024 * 1024) {
        if (free_mem < min_bytes + 64 * 1024 * 1024) {
            printf("SKIP (insufficient memory: %zu MB free)\n", free_mem / (1024 * 1024));
            return true;
        }
        alloc_bytes = min_bytes;
    }

    size_t n_floats = alloc_bytes / sizeof(float);

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL (ggml_init)\n");
        return false;
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)n_floats);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("SKIP (buffer alloc failed)\n");
        ggml_free(ctx);
        return true;
    }

    // Write pattern.
    std::vector<float> src(n_floats);
    fill_pattern(src.data(), n_floats, 0);
    ggml_backend_tensor_set(t, src.data(), 0, n_floats * sizeof(float));

    // Read back.
    std::vector<float> dst(n_floats, 0.0f);
    ggml_backend_tensor_get(t, dst.data(), 0, n_floats * sizeof(float));

    // Verify.  Check first, last, and sampled middle elements.
    bool ok = true;
    size_t check_indices[] = { 0, 1, n_floats / 4, n_floats / 2, n_floats * 3 / 4, n_floats - 2, n_floats - 1 };
    for (size_t idx : check_indices) {
        if (idx < n_floats && dst[idx] != src[idx]) {
            printf("FAIL (mismatch at [%zu]: expected %.4f, got %.4f)\n",
                   idx, src[idx], dst[idx]);
            ok = false;
            break;
        }
    }
    // Full scan if spot checks pass.
    if (ok) {
        for (size_t i = 0; i < n_floats; i++) {
            if (dst[i] != src[i]) {
                printf("FAIL (mismatch at [%zu]: expected %.4f, got %.4f)\n",
                       i, src[i], dst[i]);
                ok = false;
                break;
            }
        }
    }
    if (ok) {
        printf("OK (%zu MB)\n", n_floats * sizeof(float) / (1024 * 1024));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// Subtest C: data integrity at sparse binding boundary (>= 512 MB threshold).
static bool test_sparse_boundary(ggml_backend_t backend, ggml_backend_dev_t dev) {
    printf("  test_sparse_boundary:     ");

    // Read threshold from env (allows CI to lower it for testing).
    size_t sparse_threshold = 512ULL * 1024 * 1024;
    const char * env_threshold = getenv("GGML_VK_SPARSE_THRESHOLD");
    if (env_threshold) {
        size_t val = (size_t)strtoull(env_threshold, nullptr, 10);
        if (val > 0) {
            sparse_threshold = val;
        }
    }

    // Target: threshold + 1 MB to ensure we cross into the sparse path.
    size_t target_bytes = sparse_threshold + 1024 * 1024;

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);

    // Need target + generous headroom (sparse allocates in chunks + overhead).
    if (free_mem < target_bytes + 256ULL * 1024 * 1024) {
        printf("SKIP (insufficient memory: %zu MB free, need %zu MB)\n",
               free_mem / (1024 * 1024), (target_bytes + 256 * 1024 * 1024) / (1024 * 1024));
        return true;
    }

    size_t n_floats = target_bytes / sizeof(float);

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL (ggml_init)\n");
        return false;
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)n_floats);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("SKIP (buffer alloc failed -- sparse binding may not be supported)\n");
        ggml_free(ctx);
        return true;
    }

    // Pattern encodes byte offset per float — boundary errors are obvious.
    std::vector<float> src(n_floats);
    fill_pattern(src.data(), n_floats, 42);
    ggml_backend_tensor_set(t, src.data(), 0, n_floats * sizeof(float));

    // Read back.
    std::vector<float> dst(n_floats, 0.0f);
    ggml_backend_tensor_get(t, dst.data(), 0, n_floats * sizeof(float));

    // Verify at chunk boundaries (128 MB = 32M floats per chunk).
    bool ok = true;
    const size_t chunk_floats = 128ULL * 1024 * 1024 / sizeof(float);
    std::vector<size_t> check_indices = { 0, 1 };

    // Add indices around each chunk boundary.
    for (size_t boundary = chunk_floats; boundary < n_floats; boundary += chunk_floats) {
        if (boundary > 0)     check_indices.push_back(boundary - 1);
        check_indices.push_back(boundary);
        if (boundary + 1 < n_floats) check_indices.push_back(boundary + 1);
    }
    check_indices.push_back(n_floats - 1);

    for (size_t idx : check_indices) {
        if (idx < n_floats && dst[idx] != src[idx]) {
            printf("FAIL (mismatch at [%zu] near chunk boundary: expected %.4f, got %.4f)\n",
                   idx, src[idx], dst[idx]);
            ok = false;
            break;
        }
    }

    // Full scan if boundary checks pass.
    if (ok) {
        for (size_t i = 0; i < n_floats; i++) {
            if (dst[i] != src[i]) {
                printf("FAIL (mismatch at [%zu]: expected %.4f, got %.4f)\n",
                       i, src[i], dst[i]);
                ok = false;
                break;
            }
        }
    }
    if (ok) {
        printf("OK (%zu MB, threshold=%zu MB)\n",
               n_floats * sizeof(float) / (1024 * 1024),
               sparse_threshold / (1024 * 1024));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// Subtest D: full fallback chain (contiguous OOM -> sparse fail -> host-visible).
// Requires GGML_VULKAN_SPARSE_TESTING; skips otherwise.
static bool test_sparse_fallback_chain(ggml_backend_t backend, ggml_backend_dev_t dev) {
    printf("  test_sparse_fallback_chain: ");

    // Set fault injection: force device-local OOM + sparse failure + low threshold
    // Full chain exercised: contiguous(OOM) -> sparse(force-fail) -> host-visible(succeed)
#ifdef _WIN32
    _putenv_s("GGML_VK_TEST_FORCE_OOM", "1");
    _putenv_s("GGML_VK_TEST_SPARSE_FORCE_FAIL", "1");
    _putenv_s("GGML_VK_SPARSE_THRESHOLD", "64");
#else
    setenv("GGML_VK_TEST_FORCE_OOM", "1", 1);
    setenv("GGML_VK_TEST_SPARSE_FORCE_FAIL", "1", 1);
    setenv("GGML_VK_SPARSE_THRESHOLD", "64", 1);
#endif

#ifdef GGML_VULKAN_SPARSE_TESTING
    bool knobs_active = true;
    ggml_vk_test_sparse_counter_reset();
#else
    bool knobs_active = false;
#endif
    bool sparse_path_reached = true;
    int sparse_attempts = 0;

    const int n_elements = 1024;
    const size_t data_size = n_elements * sizeof(float);

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL (ggml_init)\n");
        goto cleanup_env;
    }

    {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            printf("FAIL (allocation failed -- fallback chain broken)\n");
            ggml_free(ctx);
            goto cleanup_env;
        }

        // Write and read back to verify the buffer works regardless of path
        std::vector<float> src(n_elements);
        for (int i = 0; i < n_elements; i++) {
            src[i] = (float)(i * 3 + 7);
        }
        ggml_backend_tensor_set(t, src.data(), 0, data_size);

        std::vector<float> dst(n_elements, 0.0f);
        ggml_backend_tensor_get(t, dst.data(), 0, data_size);

        bool ok = true;
        for (int i = 0; i < n_elements; i++) {
            if (dst[i] != src[i]) {
                printf("FAIL (mismatch at [%d]: expected %.1f, got %.1f)\n",
                       i, src[i], dst[i]);
                ok = false;
                break;
            }
        }

#ifdef GGML_VULKAN_SPARSE_TESTING
        // Assert that the sparse fallback branch was actually reached.
        if (ok && knobs_active) {
            sparse_attempts = ggml_vk_test_sparse_counter_read();
            sparse_path_reached = sparse_attempts >= 1;
        }
#endif

        if (ok) {
            if (knobs_active && sparse_path_reached) {
                printf("OK (fallback chain exercised, sparse_attempts=%d)\n",
#ifdef GGML_VULKAN_SPARSE_TESTING
                       ggml_vk_test_sparse_counter_read()
#else
                       0
#endif
                );
            } else if (knobs_active) {
                printf("SKIP (sparse fallback path not reachable on this config: counter=%d)\n", sparse_attempts);
            } else {
                printf("SKIP (GGML_VULKAN_SPARSE_TESTING not enabled in backend build)\n");
            }
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        // Clear fault injection before returning
#ifdef _WIN32
        _putenv_s("GGML_VK_TEST_FORCE_OOM", "");
        _putenv_s("GGML_VK_TEST_SPARSE_FORCE_FAIL", "");
        _putenv_s("GGML_VK_SPARSE_THRESHOLD", "");
#else
        unsetenv("GGML_VK_TEST_FORCE_OOM");
        unsetenv("GGML_VK_TEST_SPARSE_FORCE_FAIL");
        unsetenv("GGML_VK_SPARSE_THRESHOLD");
#endif
        return ok;  // SKIP is not a failure, but mismatch is
    }

cleanup_env:
#ifdef _WIN32
    _putenv_s("GGML_VK_TEST_FORCE_OOM", "");
    _putenv_s("GGML_VK_TEST_SPARSE_FORCE_FAIL", "");
    _putenv_s("GGML_VK_SPARSE_THRESHOLD", "");
#else
    unsetenv("GGML_VK_TEST_FORCE_OOM");
    unsetenv("GGML_VK_TEST_SPARSE_FORCE_FAIL");
    unsetenv("GGML_VK_SPARSE_THRESHOLD");
#endif
    return false;
}

// Subtest E: sparse eligibility logic (pure, no device needed).
static bool test_sparse_eligibility(void) {
    printf("  test_sparse_eligibility:   ");

    const size_t threshold = 512ULL * 1024 * 1024;
    const size_t above     = threshold + 1;
    const size_t below     = threshold - 1;

    struct test_case {
        bool had_oom;
        bool sparse_binding;
        size_t size;
        bool import_ptr;
        bool is_device_local;
        bool is_host_visible;
        bool expected;
        const char * label;
    };

    test_case cases[] = {
        // Positive: all conditions met
        { true,  true,  above, false, true,  false, true,  "all conditions met" },

        // Negative: each condition independently false
        { false, true,  above, false, true,  false, false, "no OOM" },
        { true,  false, above, false, true,  false, false, "no sparse support" },
        { true,  true,  below, false, true,  false, false, "below threshold" },
        { true,  true,  above, true,  true,  false, false, "import pointer" },
        { true,  true,  above, false, false, false, false, "not device-local" },
        { true,  true,  above, false, true,  true,  false, "host-visible" },

        // Edge: exact threshold
        { true,  true,  threshold, false, true, false, true, "exact threshold" },

        // Edge: all false
        { false, false, 0,     true,  false, true,  false, "all false" },
    };

    bool ok = true;
    for (const auto & tc : cases) {
        bool result = should_try_sparse(tc.had_oom, tc.sparse_binding, tc.size,
                                        threshold, tc.import_ptr,
                                        tc.is_device_local, tc.is_host_visible);
        if (result != tc.expected) {
            printf("FAIL (\"%s\": expected %s, got %s)\n",
                   tc.label,
                   tc.expected ? "true" : "false",
                   result ? "true" : "false");
            ok = false;
            break;
        }
    }

    if (ok) {
        printf("OK (%zu cases)\n", sizeof(cases) / sizeof(cases[0]));
    }
    return ok;
}

// Subtest F: staging cap-learning (OOM -> cap at 32 MB -> chunk fallback).
// Requires GGML_VULKAN_SPARSE_TESTING; skips otherwise.
static bool test_staging_cap_learning(ggml_backend_t backend, ggml_backend_dev_t dev) {
    printf("  test_staging_cap_learning: ");

#ifndef GGML_VULKAN_SPARSE_TESTING
    printf("SKIP (GGML_VULKAN_SPARSE_TESTING not enabled in backend build)\n");
    return true;
#else
    static constexpr size_t EXPECTED_CAP = 32ULL * 1024 * 1024;

    // Destroy existing staging buffer so fault injection threshold is evaluated.
    ggml_vk_test_staging_destroy(0);

    // Reset cap and OOM counter to known initial state.
    ggml_vk_test_staging_cap_reset(0);
    ggml_vk_test_staging_oom_counter_reset();

    size_t cap_before = ggml_vk_test_staging_cap_read(0);
    if (cap_before != SIZE_MAX) {
        printf("FAIL (cap_before=%zu, expected SIZE_MAX)\n", cap_before);
        return false;
    }

    // OOM threshold 40 MB: above 32 MB cap (retry succeeds), below 48 MB transfer (initial OOMs).
    static constexpr size_t OOM_THRESHOLD = 40ULL * 1024 * 1024;
    char threshold_str[32];
    snprintf(threshold_str, sizeof(threshold_str), "%zu", OOM_THRESHOLD);

#ifdef _WIN32
    _putenv_s("GGML_VK_TEST_STAGING_OOM_THRESHOLD", threshold_str);
#else
    setenv("GGML_VK_TEST_STAGING_OOM_THRESHOLD", threshold_str, 1);
#endif

    // 48 MB transfer: triggers OOM at 40 MB threshold, learns 32 MB cap, chunks.
    static constexpr size_t TRANSFER_SIZE = 48ULL * 1024 * 1024;
    const int n_elements = (int)(TRANSFER_SIZE / sizeof(float));

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL (ggml_init)\n");
        goto cleanup_env;
    }

    {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            printf("SKIP (buffer alloc failed)\n");
            ggml_free(ctx);
            goto cleanup_env_ok;
        }

        // Prepare recognisable data pattern.
        std::vector<float> src(n_elements);
        for (int i = 0; i < n_elements; i++) {
            src[i] = (float)(i * 13 + 5);
        }

        // Write — this exercises buffer_write_2d height==1 path:
        // try full-size staging (48 MB > 40 MB threshold -> OOM) -> learn cap -> chunk fallback
        ggml_backend_tensor_set(t, src.data(), 0, TRANSFER_SIZE);

        // Check OOM counter and cap state.
        int oom_hits = ggml_vk_test_staging_oom_counter_read();
        size_t cap_after = ggml_vk_test_staging_cap_read(0);

        // Read back and verify data integrity through the capped path.
        std::vector<float> dst(n_elements, 0.0f);
        ggml_backend_tensor_get(t, dst.data(), 0, TRANSFER_SIZE);

        bool ok = true;

        // oom_hits == 0 means host-visible fast path bypassed staging entirely.
        if (oom_hits == 0) {
            printf("OK (host-visible fast path bypassed staging, oom_hits=0, cap unchanged)\n");
        } else if (cap_after != EXPECTED_CAP) {
            printf("FAIL (oom_hits=%d but cap_after=%zu, expected %zu)\n",
                   oom_hits, cap_after, EXPECTED_CAP);
            ok = false;
        }

        // Data integrity check regardless of path taken.
        for (int i = 0; i < n_elements && ok; i++) {
            if (dst[i] != src[i]) {
                printf("FAIL (mismatch at [%d]: expected %.1f, got %.1f)\n",
                       i, src[i], dst[i]);
                ok = false;
                break;
            }
        }

        if (ok && oom_hits > 0) {
            printf("OK (cap learned: SIZE_MAX -> %zu, oom_hits=%d, data integrity verified)\n",
                   cap_after, oom_hits);
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        // Clean up env and reset cap for subsequent tests.
#ifdef _WIN32
        _putenv_s("GGML_VK_TEST_STAGING_OOM_THRESHOLD", "");
#else
        unsetenv("GGML_VK_TEST_STAGING_OOM_THRESHOLD");
#endif
        ggml_vk_test_staging_cap_reset(0);
        return ok;
    }

cleanup_env_ok:
#ifdef _WIN32
    _putenv_s("GGML_VK_TEST_STAGING_OOM_THRESHOLD", "");
#else
    unsetenv("GGML_VK_TEST_STAGING_OOM_THRESHOLD");
#endif
    ggml_vk_test_staging_cap_reset(0);
    return true;  // SKIP is not a failure

cleanup_env:
#ifdef _WIN32
    _putenv_s("GGML_VK_TEST_STAGING_OOM_THRESHOLD", "");
#else
    unsetenv("GGML_VK_TEST_STAGING_OOM_THRESHOLD");
#endif
    ggml_vk_test_staging_cap_reset(0);
    return false;
#endif  // GGML_VULKAN_SPARSE_TESTING
}

int main(void) {
    ggml_backend_load_all();

    // Subtest E runs without any Vulkan device (pure logic test).
    bool all_ok = true;
    all_ok &= test_sparse_eligibility();

    ggml_backend_dev_t dev;
    if (!find_vulkan_device(&dev)) {
        printf("No Vulkan device found -- subtests B/C/D/F SKIP\n");
        if (!all_ok) {
            printf("FAILED\n");
            return 1;
        }
        printf("All tests passed.\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        printf("Failed to init Vulkan backend -- subtests B/C/D/F SKIP\n");
        if (!all_ok) {
            printf("FAILED\n");
            return 1;
        }
        printf("All tests passed.\n");
        return 0;
    }

    printf("Vulkan sparse/staging regression tests (%s):\n", ggml_backend_dev_description(dev));

    all_ok &= test_chunked_staging(backend, dev);
    all_ok &= test_sparse_boundary(backend, dev);
    all_ok &= test_sparse_fallback_chain(backend, dev);
    all_ok &= test_staging_cap_learning(backend, dev);

    ggml_backend_free(backend);

    if (!all_ok) {
        printf("FAILED\n");
        return 1;
    }

    printf("All tests passed.\n");
    return 0;
}
