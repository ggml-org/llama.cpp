#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

using namespace sycl;

// Owned-memory control, not a reproducer of the read-only mmap userptr failure.
// Build: icpx -fsycl repro_mem_pressure.cpp -o /tmp/bcs-pressure-control
// Required run: timeout --kill-after=5s 60s /tmp/bcs-pressure-control
int main() try {
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
    size_t copy_size = 860672;
    size_t gb = 1024 * 1024 * 1024;
    size_t vram_limit = 12LL * gb; // Target ~12GB pressure on 16GB A770

    // Allocate garbage to induce pressure
    printf("Allocating VRAM pressure pool...\n");
    void* pressure_pool = malloc_device(vram_limit, q);
    if (!pressure_pool) {
        fprintf(stderr, "Failed to allocate 12 GiB VRAM pressure pool\n");
        return 1;
    }

    // Actual test payload
    int num_experts = 32;
    void* host_ptr = aligned_alloc(4096, copy_size * num_experts);
    void* dev_ptr = malloc_device(copy_size * num_experts, q);
    if (!host_ptr || !dev_ptr) {
        fprintf(stderr, "Failed to allocate control buffers\n");
        free(dev_ptr, q);
        free(pressure_pool, q);
        std::free(host_ptr);
        return 1;
    }
    std::memset(host_ptr, 0, copy_size * num_experts);

    printf("Owned-memory control: no read-only file mapping; success does not test the mmap defect.\n");
    printf("Starting stress loop under pressure...\n");
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
    free(pressure_pool, q);
    free(host_ptr);
    printf("Done.\n");
    return 0;
} catch (const std::exception & e) {
    fprintf(stderr, "SYCL control failed: %s\n", e.what());
    return 1;
}
