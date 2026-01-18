// SYCL weight cache key stability test.
// Ensures weight cache keys remain stable even if GGUF identities are registered later.

#include <cstdio>
#include <cstdlib>

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

int main() {
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "FAIL: CPU backend unavailable\n");
        return 1;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "FAIL: ggml_init failed\n");
        ggml_backend_free(cpu_backend);
        return 1;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
    if (!weight) {
        fprintf(stderr, "FAIL: tensor allocation failed\n");
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    char name_buf[64];
    std::snprintf(name_buf, sizeof(name_buf), "layout_key_stability_%p", (void *) weight);
    ggml_set_name(weight, name_buf);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);
    size_t                     buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t      buffer = ggml_backend_buft_alloc_buffer(buft, buf_size);
    if (!buffer) {
        fprintf(stderr, "FAIL: buffer allocation failed\n");
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(buffer, weight, ggml_backend_buffer_get_base(buffer));

    const uint64_t model_id = 1;
    ggml_backend_sycl_register_weight_identity(weight, 0, 1234, ggml_nbytes(weight), model_id);

    ggml_sycl_cache_id key_before = ggml_backend_sycl_get_weight_cache_key(weight, 0);
    if (!key_before.valid) {
        fprintf(stderr, "FAIL: missing cache key after identity registration\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    if (!key_before.has_gguf || key_before.file_offs != 1234 || key_before.nbytes != ggml_nbytes(weight) ||
        key_before.model_id != model_id) {
        fprintf(stderr, "FAIL: cache key missing GGUF identity fields\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    ggml_sycl_cache_id key_after = ggml_backend_sycl_get_weight_cache_key(weight, 0);
    if (!key_after.valid) {
        fprintf(stderr, "FAIL: missing cache key on second lookup\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    if (!ggml_sycl::detail::cache_id_equal(key_before, key_after)) {
        fprintf(stderr, "FAIL: cache key changed between lookups\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu_backend);
        return 1;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    printf("PASS: weight cache key remained stable\n");
    return 0;
}

#endif
