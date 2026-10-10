#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

using namespace sycl;

// Owned-memory control, not a reproducer of the read-only mmap userptr failure.
// Build: icpx -fsycl bcs_owned_memory_control.cpp -o /tmp/bcs-owned-memory-control
// Required run: timeout --kill-after=5s 60s /tmp/bcs-owned-memory-control
int main() try {
    // Default queue (out-of-order)
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
    printf("Device: %s\n", q.get_device().get_info<info::device::name>().c_str());

    size_t copy_size = 860672;
    int num_experts = 32;

    void* host_ptr = aligned_alloc(4096, copy_size * num_experts);
    void* dev_ptr = malloc_device(copy_size * num_experts, q);
    if (!host_ptr || !dev_ptr) {
        fprintf(stderr, "Failed to allocate control buffers\n");
        free(dev_ptr, q);
        std::free(host_ptr);
        return 1;
    }
    std::memset(host_ptr, 0, copy_size * num_experts);

    printf("Owned-memory control: no read-only file mapping; success does not test the mmap defect.\n");
    printf("Starting stress loop...\n");
    for (int iter = 0; iter < 500; ++iter) {
        for (int i = 0; i < num_experts; ++i) {
            event e_copy = q.memcpy(static_cast<char*>(dev_ptr) + i * copy_size,
                                    static_cast<char*>(host_ptr) + i * copy_size,
                                    copy_size);
            q.submit([&](handler& h) {
                h.depends_on(e_copy);
                h.single_task([=](){ volatile int x = 0; (void) x; });
            });
        }
        q.wait_and_throw();
    }
    free(dev_ptr, q);
    free(host_ptr);
    printf("Done.\n");
    return 0;
} catch (const std::exception & e) {
    fprintf(stderr, "SYCL control failed: %s\n", e.what());
    return 1;
}
