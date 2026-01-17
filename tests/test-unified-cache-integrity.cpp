// Stress test for unified cache map integrity during layout caching.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-unified-cache-integrity

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

static bool stress_layout_cache(ggml_context * ctx,
                                ggml_backend_buffer_type_t buft,
                                int device_id,
                                bool device_resident,
                                int weights_per_type) {
    bool ok = true;
    int  seed = device_resident ? 101 : 7;

    const int64_t ncols = 1024;  // Ensures coalesced tile alignment (32 blocks).
    const int64_t nrows = 4;

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(weights_per_type * 4);

    ggml_sycl::unified_cache * cache = nullptr;

    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4 };

    for (size_t type_idx = 0; type_idx < sizeof(types) / sizeof(types[0]); ++type_idx) {
        const ggml_type type = types[type_idx];
        for (int i = 0; i < weights_per_type; ++i) {
            ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, ncols, nrows);
            std::string   name =
                std::string(device_resident ? "attn_q.device." : "attn_q.host.") + std::to_string(type_idx) + "." +
                std::to_string(i);
            ggml_set_name(weight, name.c_str());

            const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
            ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
            if (!weight_buffer) {
                fprintf(stderr, "Failed to allocate buffer for type=%d idx=%d\n", (int) type, i);
                ok = false;
                break;
            }
            buffers.push_back(weight_buffer);
            ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_alloc(weight_buffer, weight, ggml_backend_buffer_get_base(weight_buffer));

            std::vector<uint8_t> host_data(ggml_nbytes(weight));
            fill_pattern(host_data, seed++);
            ggml_backend_tensor_set(weight, host_data.data(), 0, host_data.size());

            const ggml_layout_mode target = pick_layout(weight, device_id);
            void * cached = ggml_sycl_get_weight_layout_ptr(weight, device_id, target);
            if (!cached) {
                fprintf(stderr, "Failed to cache layout for type=%d idx=%d\n", (int) type, i);
                ok = false;
                break;
            }

            if (!cache) {
                cache = ggml_sycl::get_unified_cache_for_device(device_id);
                if (!cache) {
                    fprintf(stderr, "Failed to fetch unified cache\n");
                    ok = false;
                    break;
                }
            }

            if (((i + 1) % 32) == 0 && !cache->validate()) {
                fprintf(stderr, "Unified cache integrity check failed at type=%d idx=%d\n", (int) type, i);
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }
    }

    if (ok && cache && !cache->validate()) {
        fprintf(stderr, "Unified cache integrity check failed at end\n");
        ok = false;
    }

    for (auto buf : buffers) {
        ggml_backend_buffer_free(buf);
    }

    return ok;
}

int main() {
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
        fprintf(stderr, "Failed to init CPU backend\n");
        return 1;
    }

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to init SYCL backend\n");
        return 1;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 128 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(sycl_backend);
        ggml_backend_free(cpu_backend);
        fprintf(stderr, "Failed to init ggml context\n");
        return 1;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_get_default_buffer_type(cpu_backend);
    ggml_backend_buffer_type_t dev_buft  = ggml_backend_sycl_buffer_type(0);

    bool ok = stress_layout_cache(ctx, host_buft, 0, false, 128);
    if (ok && dev_buft) {
        ok = stress_layout_cache(ctx, dev_buft, 0, true, 64);
    } else if (!dev_buft) {
        fprintf(stderr, "Failed to get SYCL buffer type\n");
        ok = false;
    }

    ggml_free(ctx);
    ggml_backend_free(sycl_backend);
    ggml_backend_free(cpu_backend);
    printf("\nUnified cache integrity test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
