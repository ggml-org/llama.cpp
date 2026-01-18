// Stress test ggml_sycl_get_weight_layout_ptr to exercise host_cache rehash paths.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-host-cache-layout-rehash

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

#include "ggml-quants.h"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/unified-cache.hpp"
#include <sycl/sycl.hpp>

struct cached_key {
    ggml_sycl_cache_id key;
    ggml_layout_mode layout;
};

static void fill_pattern(std::vector<uint8_t> & data, int seed) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 131 + seed) ^ 0x5a);
    }
}

static ggml_layout_mode pick_layout(const ggml_tensor * weight, int device_id) {
    tensor_usage usage = infer_tensor_usage(weight->name);
    ggml_layout_mode target = layout_policy::get_with_override(weight->type, usage, device_id);
    return ggml_sycl_adjust_layout_for_tensor(weight, target, device_id);
}

static bool stress_layout_rehash(int device_id) {
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "Failed to init CPU backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);

    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to init ggml context\n");
        return false;
    }

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(device_id);
    ggml_sycl::host_cache *    host_cache = ggml_sycl::get_host_cache_for_device(device_id);
    if (!cache || !host_cache) {
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to get caches\n");
        return false;
    }

    const int64_t tile_blocks = MMVQ_COALESCED_TILE_BLOCKS;
    const int64_t q4_cols = QK4_0 * tile_blocks;
    const int64_t q8_cols = QK8_0 * tile_blocks;

    struct weight_spec {
        ggml_type type;
        int64_t   ncols;
        int64_t   nrows;
        int       count;
    };

    const weight_spec specs[] = {
        { GGML_TYPE_Q4_0, q4_cols, 8, 4096 },
        { GGML_TYPE_Q8_0, q8_cols, 8, 2048 },
    };

    std::vector<ggml_backend_buffer_t> buffers;
    std::vector<cached_key>            keys;
    buffers.reserve(8192);
    keys.reserve(8192);

    bool ok = true;
    int seed = 3;

    for (const auto & spec : specs) {
        for (int i = 0; i < spec.count; ++i) {
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, spec.type, spec.ncols, spec.nrows);
            std::string   name = std::string("rehash_weight.") + std::to_string((int) spec.type) + "." +
                               std::to_string(i);
            ggml_set_name(weight, name.c_str());

            const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
            ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
            if (!weight_buffer) {
                fprintf(stderr, "Failed to allocate buffer type=%d idx=%d\n", (int) spec.type, i);
                ok = false;
                break;
            }
            buffers.push_back(weight_buffer);
            ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));

            std::vector<uint8_t> host_data(ggml_nbytes(weight));
            fill_pattern(host_data, seed++);
            ggml_backend_tensor_set(weight, host_data.data(), 0, host_data.size());

            ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
            if (!key.valid) {
                fprintf(stderr, "Failed to get cache key for type=%d idx=%d\n", (int) spec.type, i);
                ok = false;
                break;
            }

            const ggml_layout_mode target = pick_layout(weight, device_id);
            if (target == GGML_LAYOUT_AOS) {
                fprintf(stderr, "Unexpected AOS layout for type=%d idx=%d\n", (int) spec.type, i);
                ok = false;
                break;
            }

            void * cached = ggml_sycl_get_weight_layout_ptr(weight, device_id, target);
            if (!cached) {
                fprintf(stderr, "Failed to cache layout for type=%d idx=%d\n", (int) spec.type, i);
                ok = false;
                break;
            }

            keys.push_back({ key, target });
        }
        if (!ok) {
            break;
        }
    }

    if (ok && !cache->validate()) {
        fprintf(stderr, "Unified cache validation failed\n");
        ok = false;
    }

    for (const auto & entry : keys) {
        cache->remove(entry.key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, entry.layout);
        host_cache->remove(entry.key, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, entry.layout);
    }

    const int guard_errors = ggml_sycl::host_cache_guard_error_count();
    if (guard_errors != 0) {
        fprintf(stderr, "Host cache guard errors: %d\n", guard_errors);
        ok = false;
    }

    for (auto buf : buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    return ok;
}

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }
    setenv("GGML_SYCL_DISABLE_GRAPH", "1", 1);
    setenv("GGML_SYCL_HOST_CACHE_GUARD", "1", 1);
    setenv("GGML_SYCL_PINNED_CHUNK_MB", "256", 1);
    setenv("GGML_SYCL_WEIGHTS_EVICTABLE", "1", 1);

    ggml_sycl::set_unified_cache_host_budget_pct(1);
    ggml_sycl::host_cache_guard_reset();

    try {
        sycl::queue q;
        printf("Using device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }

    const bool ok = stress_layout_rehash(/*device_id=*/0);
    if (!ok) {
        fprintf(stderr, "Host cache layout rehash test: FAIL\n");
        return 1;
    }

    printf("Host cache layout rehash test: PASS\n");
    return 0;
}

#endif
