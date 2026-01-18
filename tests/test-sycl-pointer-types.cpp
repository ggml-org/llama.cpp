// SYCL pointer type classification test.
// Validates sycl::get_pointer_type for device/host/shared/unknown pointers.

#include "ggml-sycl.h"

#include <cstdlib>
#include <cstdio>
#include <sycl/sycl.hpp>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

static bool expect_type(sycl::usm::alloc got, sycl::usm::alloc expected, const char * label) {
    if (got != expected) {
        fprintf(stderr, "FAIL: %s expected=%d got=%d\n", label, (int) expected, (int) got);
        return false;
    }
    return true;
}

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    sycl::queue q;
    const auto & ctx = q.get_context();

    void * dev = sycl::malloc_device(1024, q);
    void * host = sycl::malloc_host(1024, q);
    void * shared = sycl::malloc_shared(1024, q);
    void * heap = std::malloc(128);

    if (!dev || !host || !shared || !heap) {
        fprintf(stderr, "FAIL: allocation failed\n");
        if (dev) sycl::free(dev, q);
        if (host) sycl::free(host, q);
        if (shared) sycl::free(shared, q);
        if (heap) std::free(heap);
        return 1;
    }

    bool ok = true;
    ok = ok && expect_type(sycl::get_pointer_type(dev, ctx), sycl::usm::alloc::device, "device");
    ok = ok && expect_type(sycl::get_pointer_type(host, ctx), sycl::usm::alloc::host, "host");
    ok = ok && expect_type(sycl::get_pointer_type(shared, ctx), sycl::usm::alloc::shared, "shared");
    ok = ok && expect_type(sycl::get_pointer_type(heap, ctx), sycl::usm::alloc::unknown, "unknown");

    sycl::free(dev, q);
    sycl::free(host, q);
    sycl::free(shared, q);
    std::free(heap);

    printf("\nPointer type test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
