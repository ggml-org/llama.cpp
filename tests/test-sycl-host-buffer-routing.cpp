// Validate SYCL host buffer routing (pinned vs CPU fallback).
// Run (opt-in):
//   GGML_SYCL_HOST_BUFFER_TEST=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
//   ./build/bin/test-sycl-host-buffer-routing

#include <cstdio>
#include <cstdlib>

#include <sycl/sycl.hpp>

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/dpct/helper.hpp"

size_t ggml_sycl_get_host_max_alloc_size();

static size_t parse_env_mb(const char * name, size_t def_mb) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return def_mb;
    }
    char * end = nullptr;
    long val = std::strtol(env, &end, 10);
    if (end == env || val <= 0) {
        return def_mb;
    }
    return static_cast<size_t>(val);
}

int main() {
    const char * enable = std::getenv("GGML_SYCL_HOST_BUFFER_TEST");
    if (!enable || std::atoi(enable) == 0) {
        std::printf("SKIP: set GGML_SYCL_HOST_BUFFER_TEST=1 to run\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::printf("SKIP: could not initialize SYCL backend\n");
        return 0;
    }

    const size_t size_mb = parse_env_mb("GGML_SYCL_HOST_BUFFER_SIZE_MB", 4);
    const size_t size_bytes = size_mb * 1024ULL * 1024ULL;
    const size_t host_max = ggml_sycl_get_host_max_alloc_size();
    if (host_max > 0 && size_bytes > host_max) {
        std::printf("SKIP: requested %.1f MB exceeds host max alloc %.1f MB\n",
                    size_mb * 1.0, host_max / (1024.0 * 1024.0));
        ggml_backend_free(backend);
        return 0;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(host_buft, size_bytes);
    if (!buffer) {
        std::fprintf(stderr, "FAIL: host buffer allocation failed\n");
        ggml_backend_free(backend);
        return 1;
    }

    void * base = ggml_backend_buffer_get_base(buffer);
    if (!base) {
        std::fprintf(stderr, "FAIL: host buffer base pointer is null\n");
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return 1;
    }

    sycl::context ctx = dpct::get_in_order_queue().get_context();
    const sycl::usm::alloc alloc = sycl::get_pointer_type(base, ctx);

    const bool expect_pinned = (std::getenv("GGML_SYCL_NO_PINNED") == nullptr);
    if (expect_pinned && alloc != sycl::usm::alloc::host) {
        std::fprintf(stderr,
                     "FAIL: expected pinned host allocation, got %d (set GGML_SYCL_NO_PINNED=1 to allow fallback)\n",
                     static_cast<int>(alloc));
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        return 1;
    }

    std::printf("OK: host buffer allocation type=%d size=%.1f MB\n", static_cast<int>(alloc), size_mb * 1.0);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return 0;
}
