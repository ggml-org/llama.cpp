// SYCL weight cache guard stress test.
// Exercises weight cache keying + host layout reorders and checks guard integrity.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-weight-cache-guard

#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

void * ggml_sycl_get_weight_layout_ptr(const ggml_tensor * tensor, int device, ggml_layout_mode target);

struct weight_case {
    ggml_type        type;
    int64_t          ncols;
    int64_t          nrows;
    ggml_layout_mode layout;
    const char *     name;
};

static void fill_pattern(std::vector<uint8_t> & data) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 131) ^ 0x5a);
    }
}

static bool run_case(const weight_case & tc,
                     int device_id,
                     ggml_backend_buffer_type_t buft,
                     ggml_context * ctx) {
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, tc.type, tc.ncols, tc.nrows);
    if (!weight) {
        fprintf(stderr, "FAIL: tensor allocation failed for %s\n", tc.name);
        return false;
    }
    ggml_set_name(weight, tc.name);

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        fprintf(stderr, "FAIL: buffer allocation failed for %s\n", tc.name);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));

    const size_t weight_bytes = ggml_nbytes(weight);
    std::vector<uint8_t> host_data(weight_bytes);
    fill_pattern(host_data);
    ggml_backend_tensor_set(weight, host_data.data(), 0, host_data.size());

    ggml_sycl_cache_id cache_key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
    if (!cache_key.valid) {
        fprintf(stderr, "FAIL: cache key not created for %s\n", tc.name);
        ggml_backend_buffer_free(weight_buffer);
        return false;
    }

    void * layout_ptr = ggml_sycl_get_weight_layout_ptr(weight, device_id, tc.layout);
    if (!layout_ptr) {
        fprintf(stderr, "FAIL: layout pointer not created for %s layout=%d\n", tc.name, (int) tc.layout);
        ggml_backend_buffer_free(weight_buffer);
        return false;
    }

    if (ggml_sycl::host_cache_guard_error_count() != 0) {
        fprintf(stderr, "FAIL: host cache guard reported corruption after %s\n", tc.name);
        ggml_backend_buffer_free(weight_buffer);
        return false;
    }

    ggml_sycl::host_cache * host_cache = ggml_sycl::get_host_cache_for_device(device_id);
    if (host_cache) {
        host_cache->remove(cache_key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, tc.layout);
    }

    ggml_backend_buffer_free(weight_buffer);
    return true;
}

int main() {
    setenv("GGML_SYCL_WEIGHT_CACHE_GUARD", "1", 1);
    setenv("GGML_SYCL_CACHE_ASSERT", "1", 1);
    setenv("GGML_SYCL_HOST_CACHE_GUARD", "1", 1);
    ggml_sycl::host_cache_guard_reset();

    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    try {
        sycl::queue q;
        printf("Using device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "FAIL: CPU backend unavailable\n");
        return 1;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);
    if (!buft) {
        fprintf(stderr, "FAIL: CPU buffer type unavailable\n");
        ggml_backend_free(cpu_backend);
        return 1;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "FAIL: ggml_init failed\n");
        ggml_backend_free(cpu_backend);
        return 1;
    }

    const weight_case cases[] = {
        { GGML_TYPE_Q4_0, QK4_0 * 32, 4, GGML_LAYOUT_COALESCED, "guard_q4_0_coalesced" },
        { GGML_TYPE_Q8_0, QK8_0 * 32, 4, GGML_LAYOUT_COALESCED, "guard_q8_0_coalesced" },
        { GGML_TYPE_Q6_K, QK_K * 56, 2, GGML_LAYOUT_COALESCED, "guard_q6k_coalesced" },
        { GGML_TYPE_Q4_K, QK_K * 8,  2, GGML_LAYOUT_SOA,       "guard_q4k_soa" },
    };

    bool ok = true;
    for (const auto & tc : cases) {
        if (!run_case(tc, /*device_id*/ 0, buft, ctx)) {
            ok = false;
            break;
        }
    }

    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    printf("\nWeight cache guard test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
