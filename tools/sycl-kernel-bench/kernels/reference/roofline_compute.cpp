#include "reference_kernels.hpp"

#include <chrono>

namespace sycl_bench {

static bool get_event_timing(const sycl::event & event, uint64_t & start, uint64_t & end, std::string & error) {
    try {
        start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
    if (start == 0 || end == 0 || end <= start) {
        error = "profiling info unavailable (queue missing enable_profiling?)";
        return false;
    }
    return true;
}

bool run_roofline_compute(size_t elements,
                          int ops_per_element,
                          int warmup,
                          int iterations,
                          sycl::queue & queue,
                          ReferenceMetrics & out,
                          std::string & error) {
    if (elements == 0 || ops_per_element <= 0) {
        error = "roofline_compute requires elements > 0 and ops_per_element > 0";
        return false;
    }

    auto * data = static_cast<float *>(sycl::malloc_device(elements * sizeof(float), queue));
    if (!data) {
        error = "device allocation failed for roofline_compute";
        return false;
    }

    auto submit_kernel = [&]() {
        return queue.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> idx) {
                float acc = data[idx];
                for (int i = 0; i < ops_per_element; ++i) {
                    acc = sycl::fma(acc, 1.0001f, 0.0001f);
                }
                data[idx] = acc;
            });
        });
    };

    for (int i = 0; i < warmup; ++i) {
        submit_kernel();
    }
    queue.wait_and_throw();

    std::vector<sycl::event> events;
    events.reserve(static_cast<size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        events.push_back(submit_kernel());
    }
    sycl::event::wait(events);

    std::vector<double> iter_ns;
    iter_ns.reserve(events.size());
    for (const auto & evt : events) {
        uint64_t start = 0;
        uint64_t end = 0;
        if (!get_event_timing(evt, start, end, error)) {
            sycl::free(data, queue);
            return false;
        }
        iter_ns.push_back(static_cast<double>(end - start));
    }

    double sum = 0.0;
    for (double v : iter_ns) {
        sum += v;
    }
    const double mean_ns = (iter_ns.empty()) ? 0.0 : sum / static_cast<double>(iter_ns.size());
    const double mean_s = mean_ns * 1e-9;

    const double flops = static_cast<double>(elements) * static_cast<double>(ops_per_element) * 2.0;
    out.total_us = mean_ns * 1e-3;
    out.tflops = (mean_s > 0.0) ? (flops / mean_s) / 1.0e12 : 0.0;

    const double bytes = static_cast<double>(elements) * sizeof(float) * 2.0;
    out.arithmetic_intensity = (bytes > 0.0) ? flops / bytes : 0.0;

    sycl::free(data, queue);
    return true;
}

}  // namespace sycl_bench
