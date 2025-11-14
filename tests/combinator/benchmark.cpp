#include "tests.h"
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

// benchmark_test base class implementation
benchmark_test::benchmark_test(std::vector<std::unique_ptr<test_case>> cs): cases(std::move(cs)) {}

long long benchmark_test::run_benchmark(size_t which, int iterations) {
    if (which >= cases.size()) {
        throw std::runtime_error(std::string("Invalid index for benchmark test: ") + std::to_string(which));
    }
    std::chrono::microseconds duration(0);
    test_case& tc = *cases.at(which);
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        tc.run();
        auto end = std::chrono::high_resolution_clock::now();
        tc.reset();
        duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    return duration.count() / iterations;
}
