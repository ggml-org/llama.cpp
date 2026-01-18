// Host cache guard test (detect host layout overruns).
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-host-cache-guard

#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "ggml-quants.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sycl/sycl.hpp>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

void * ggml_sycl_get_weight_layout_ptr(const ggml_tensor * tensor, int device, ggml_layout_mode target);

static void fill_pattern(std::vector<uint8_t> & data) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 131) ^ 0x5a);
    }
}

static bool test_host_cache_guard(int device_id) {
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "Failed to init CPU backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    const int64_t ncols = QK4_0 * 4;
    const int64_t nrows = 4;
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, ncols, nrows);
    ggml_set_name(weight, "attn_q.weight");

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to allocate CPU weight buffer\n");
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
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to get cache key\n");
        return false;
    }

    void * ptr = ggml_sycl_get_weight_layout_ptr(weight, device_id, GGML_LAYOUT_COALESCED);
    if (!ptr) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to get COALESCED layout pointer\n");
        return false;
    }

    ggml_sycl::host_cache * host_cache = ggml_sycl::get_host_cache_for_device(device_id);
    if (!host_cache) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to get host cache\n");
        return false;
    }

    host_cache->remove(cache_key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_COALESCED);

    if (ggml_sycl::host_cache_guard_error_count() != 0) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Host cache guard reported corruption\n");
        return false;
    }

    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    return true;
}

int main() {
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

    bool ok = test_host_cache_guard(0);

    printf("\nHost cache guard test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
