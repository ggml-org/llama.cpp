#include "reference_kernels.hpp"

#include <chrono>

namespace sycl_bench {

bool run_memory_bandwidth(size_t bytes,
                          int warmup,
                          int iterations,
                          sycl::queue & queue,
                          ReferenceMetrics & out,
                          std::string & error) {
    if (bytes == 0) {
        error = "memory_bandwidth bytes must be > 0";
        return false;
    }

    const size_t elem_size = sizeof(uint32_t);
    size_t elem_count = bytes / elem_size;
    if (elem_count == 0) {
        elem_count = bytes;
    }

    size_t alloc_bytes = elem_count * elem_size;
    auto * src = static_cast<uint32_t *>(sycl::malloc_device(alloc_bytes, queue));
    auto * dst = static_cast<uint32_t *>(sycl::malloc_device(alloc_bytes, queue));

    if (!src || !dst) {
        if (src) sycl::free(src, queue);
        if (dst) sycl::free(dst, queue);
        error = "device allocation failed for memory_bandwidth";
        return false;
    }

    auto submit_copy = [&](uint32_t * d, const uint32_t * s) {
        return queue.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(elem_count), [=](sycl::id<1> idx) {
                d[idx] = s[idx];
            });
        });
    };

    for (int i = 0; i < warmup; ++i) {
        submit_copy(dst, src);
    }
    queue.wait_and_throw();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        submit_copy(dst, src);
    }
    queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us = (iterations > 0) ? total_us / iterations : 0.0;
    const double mean_s = mean_us * 1e-6;

    const double bytes_moved = static_cast<double>(alloc_bytes) * 2.0;
    out.total_us = mean_us;
    out.bandwidth_gbps = (mean_s > 0.0) ? (bytes_moved / mean_s) / 1.0e9 : 0.0;

    sycl::free(src, queue);
    sycl::free(dst, queue);
    return true;
}

}  // namespace sycl_bench
