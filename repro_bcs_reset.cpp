#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

using namespace sycl;

// Owned-memory control, not a reproducer of the read-only mmap userptr failure.
// Build: icpx -fsycl repro_bcs_reset.cpp -o /tmp/repro-bcs-control
// Run: timeout --kill-after=5s 60s /tmp/repro-bcs-control
int main() try {
    size_t copy_size = 860672;
    int batch_size = 10;
    int iterations = 100;
    
    // Default queue is out-of-order
    queue q{[](exception_list errors) {
        for (const auto & error : errors) {
            try {
                std::rethrow_exception(error);
            } catch (const std::exception & e) {
                fprintf(stderr, "Asynchronous SYCL failure: %s\n", e.what());
            }
        }
        if (errors.size() != 0) {
            std::exit(EXIT_FAILURE);
        }
    }};
    
    void* host_ptr = aligned_alloc(4096, copy_size);
    void* dev_ptr = malloc_device(copy_size * batch_size, q);
    if (!host_ptr || !dev_ptr) {
        fprintf(stderr, "Failed to allocate control buffers\n");
        free(dev_ptr, q);
        std::free(host_ptr);
        return 1;
    }
    std::memset(host_ptr, 0, copy_size);
    
    printf("Owned-memory control: no read-only file mapping; success does not test the mmap defect.\n");
    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < batch_size; ++i) {
            event e = q.memcpy(static_cast<char *>(dev_ptr) + i * copy_size, host_ptr, copy_size);
            q.submit([&](handler& h) {
                h.depends_on(e);
                h.single_task([=]() { volatile int x = 0; (void) x; });
            });
        }
        q.wait_and_throw();
    }
    printf("Done.\n");
    free(dev_ptr, q);
    free(host_ptr);
    return 0;
} catch (const std::exception & e) {
    fprintf(stderr, "SYCL control failed: %s\n", e.what());
    return 1;
}
