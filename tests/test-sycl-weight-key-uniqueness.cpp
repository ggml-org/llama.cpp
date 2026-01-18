// SYCL weight cache key uniqueness test.
// Ensures distinct weights map to distinct cache keys (no collisions) for GGUF and UUID paths.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

static ggml_backend_buffer_t alloc_weight_buffer(ggml_backend_buffer_type_t buft, ggml_tensor * weight) {
    const size_t buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, buf_size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(buffer, weight, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

static bool ensure_unique_keys(const std::vector<ggml_sycl_cache_id> & keys, const char * label) {
    std::unordered_set<ggml_sycl_cache_id,
                       ggml_sycl::detail::cache_id_hash,
                       ggml_sycl::detail::cache_id_equal_fn>
        uniq;
    for (const ggml_sycl_cache_id & key : keys) {
        if (!key.valid) {
            fprintf(stderr, "FAIL: %s contains null key\n", label);
            return false;
        }
        if (!uniq.emplace(key).second) {
            fprintf(stderr, "FAIL: %s key collision\n", label);
            return false;
        }
    }
    return true;
}

int main() {
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "FAIL: CPU backend unavailable\n");
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

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);
    if (!buft) {
        fprintf(stderr, "FAIL: CPU buffer type unavailable\n");
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    const int device_id = 0;
    const int count = 32;
    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(static_cast<size_t>(count) * 2);

    // Phase 1: GGUF identity path.
    std::vector<ggml_sycl_cache_id> gguf_keys;
    std::vector<ggml_sycl_cache_id> uuid_keys;
    gguf_keys.reserve(static_cast<size_t>(count));
    uuid_keys.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256 + i, 64);
        if (!weight) {
            fprintf(stderr, "FAIL: tensor allocation failed\n");
            goto fail;
        }
        char name_buf[64];
        std::snprintf(name_buf, sizeof(name_buf), "gguf_weight_%p_%d", (void *) weight, i);
        ggml_set_name(weight, name_buf);

        ggml_backend_buffer_t buffer = alloc_weight_buffer(buft, weight);
        if (!buffer) {
            fprintf(stderr, "FAIL: buffer allocation failed\n");
            goto fail;
        }
        buffers.push_back(buffer);

        const size_t nbytes = ggml_nbytes(weight);
        ggml_backend_sycl_register_weight_identity(weight, 0, static_cast<size_t>(i) * 4096, nbytes, 1);

        ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
        gguf_keys.push_back(key);
    }

    if (!ensure_unique_keys(gguf_keys, "gguf_keys")) {
        goto fail;
    }

    // Phase 2: UUID path (no GGUF identity registration, identical names).
    for (int i = 0; i < count; ++i) {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 128 + i, 32);
        if (!weight) {
            fprintf(stderr, "FAIL: tensor allocation failed\n");
            goto fail;
        }
        ggml_set_name(weight, "uuid_weight_shared_name");

        ggml_backend_buffer_t buffer = alloc_weight_buffer(buft, weight);
        if (!buffer) {
            fprintf(stderr, "FAIL: buffer allocation failed\n");
            goto fail;
        }
        buffers.push_back(buffer);

        ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, device_id);
        uuid_keys.push_back(key);
    }

    if (!ensure_unique_keys(uuid_keys, "uuid_keys")) {
        goto fail;
    }

    // Phase 3: tied weights should share identity (same GGUF offset).
    {
        ggml_tensor * tied_a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
        ggml_tensor * tied_b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
        if (!tied_a || !tied_b) {
            fprintf(stderr, "FAIL: tied tensor allocation failed\n");
            goto fail;
        }
        ggml_set_name(tied_a, "tied_weight_a");
        ggml_set_name(tied_b, "tied_weight_b");
        ggml_backend_buffer_t buf_a = alloc_weight_buffer(buft, tied_a);
        ggml_backend_buffer_t buf_b = alloc_weight_buffer(buft, tied_b);
        if (!buf_a || !buf_b) {
            fprintf(stderr, "FAIL: tied weight buffer allocation failed\n");
            goto fail;
        }
        buffers.push_back(buf_a);
        buffers.push_back(buf_b);

        const size_t tied_bytes = ggml_nbytes(tied_a);
        const uint64_t tied_model_id = 7;
        ggml_backend_sycl_register_weight_identity(tied_a, 0, 4096, tied_bytes, tied_model_id);
        ggml_backend_sycl_register_weight_identity(tied_b, 0, 4096, tied_bytes, tied_model_id);

        ggml_sycl_cache_id tied_key_a = ggml_backend_sycl_get_weight_cache_key(tied_a, device_id);
        ggml_sycl_cache_id tied_key_b = ggml_backend_sycl_get_weight_cache_key(tied_b, device_id);
        if (!tied_key_a.valid || !tied_key_b.valid) {
            fprintf(stderr, "FAIL: tied weight keys invalid\n");
            goto fail;
        }
        if (!ggml_sycl::detail::cache_id_equal(tied_key_a, tied_key_b)) {
            fprintf(stderr, "FAIL: tied weights did not share cache identity\n");
            goto fail;
        }
    }

    for (ggml_backend_buffer_t buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    printf("PASS: weight cache keys unique for GGUF/UUID paths and shared for tied weights\n");
    return 0;

fail:
    for (ggml_backend_buffer_t buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);
    return 1;
}

#endif
