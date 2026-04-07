// Minimal reproducer: does a GPU kernel hang when reading from a large
// sycl::malloc_device allocation on Intel Arc B580 (BMG)?
//
// Build:  icpx -fsycl -o test-large-alloc test-large-alloc-kernel.cpp
// Run:    ONEAPI_DEVICE_SELECTOR=level_zero:0 ./test-large-alloc
//
// Tests reading from offset 0 and from a high offset (>4 GB) within
// a single large device allocation.

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    auto dev = q.get_device();
    printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    size_t max_alloc = dev.get_info<sycl::info::device::max_mem_alloc_size>();
    printf("maxMemAllocSize: %.1f GB\n", max_alloc / (1024.0 * 1024.0 * 1024.0));

    // Test sizes: 1 GB, 4 GB, 8 GB
    size_t test_sizes[] = {
        1ULL * 1024 * 1024 * 1024,
        4ULL * 1024 * 1024 * 1024,
        8ULL * 1024 * 1024 * 1024,
    };
    const char * size_names[] = { "1 GB", "4 GB", "8 GB" };

    const size_t N = 1024;  // Elements to read/write
    const size_t elem_bytes = N * sizeof(float);

    float * host_buf = (float *)malloc(elem_bytes);
    float * result_buf = (float *)malloc(elem_bytes);

    for (int t = 0; t < 3; t++) {
        size_t alloc_size = test_sizes[t];
        if (alloc_size > max_alloc) {
            printf("\n=== SKIP %s (exceeds maxMemAllocSize) ===\n", size_names[t]);
            continue;
        }

        printf("\n=== TEST %s allocation ===\n", size_names[t]);

        // Allocate
        void * dev_ptr = nullptr;
        try {
            dev_ptr = sycl::malloc_device(alloc_size, q);
        } catch (const sycl::exception & e) {
            printf("  malloc_device(%s) FAILED: %s\n", size_names[t], e.what());
            continue;
        }
        if (!dev_ptr) {
            printf("  malloc_device(%s) returned nullptr\n", size_names[t]);
            continue;
        }
        printf("  Allocated at %p\n", dev_ptr);

        // Test 1: Write+read at offset 0
        for (size_t i = 0; i < N; i++) host_buf[i] = (float)(i + 1);
        printf("  [Offset 0] Writing %zu floats... ", N);
        q.memcpy(dev_ptr, host_buf, elem_bytes).wait();
        printf("OK. Reading... ");
        memset(result_buf, 0, elem_bytes);

        auto t1 = std::chrono::high_resolution_clock::now();
        // Simple kernel: read N floats from device, write to output
        float * src = (float *)dev_ptr;
        float * dst = sycl::malloc_device<float>(N, q);
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            dst[i] = src[i] * 2.0f;
        }).wait();
        auto t2 = std::chrono::high_resolution_clock::now();

        q.memcpy(result_buf, dst, elem_bytes).wait();
        sycl::free(dst, q);

        bool ok = true;
        for (size_t i = 0; i < N; i++) {
            if (result_buf[i] != (float)((i + 1) * 2)) {
                printf("MISMATCH at [%zu]: expected %.1f got %.1f\n", i, (float)((i+1)*2), result_buf[i]);
                ok = false;
                break;
            }
        }
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        printf("%s (%.2f ms)\n", ok ? "PASS" : "FAIL", ms);

        // Test 2: Write+read at high offset (near end of allocation)
        if (alloc_size > elem_bytes * 2) {
            size_t high_offset = alloc_size - elem_bytes;  // Last elem_bytes of the allocation
            float * high_src = (float *)((char *)dev_ptr + high_offset);

            for (size_t i = 0; i < N; i++) host_buf[i] = (float)(i + 1000);
            printf("  [Offset %.1f GB] Writing %zu floats... ",
                   high_offset / (1024.0 * 1024.0 * 1024.0), N);
            q.memcpy(high_src, host_buf, elem_bytes).wait();
            printf("OK. Reading via kernel... ");

            dst = sycl::malloc_device<float>(N, q);
            t1 = std::chrono::high_resolution_clock::now();
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst[i] = high_src[i] * 3.0f;
            }).wait();
            t2 = std::chrono::high_resolution_clock::now();

            q.memcpy(result_buf, dst, elem_bytes).wait();
            sycl::free(dst, q);

            ok = true;
            for (size_t i = 0; i < N; i++) {
                if (result_buf[i] != (float)((i + 1000) * 3)) {
                    printf("MISMATCH at [%zu]: expected %.1f got %.1f\n", i, (float)((i+1000)*3), result_buf[i]);
                    ok = false;
                    break;
                }
            }
            ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            printf("%s (%.2f ms)\n", ok ? "PASS" : "FAIL", ms);
        }

        sycl::free(dev_ptr, q);
        printf("  Freed.\n");
    }

    free(host_buf);
    free(result_buf);
    printf("\nDone.\n");
    return 0;
}
