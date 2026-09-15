#include "ggml-threading.h"
#include <algorithm>
#include <mutex>
#include <thread>
#include <vector>

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {
    ggml_critical_section_mutex.unlock();
}

size_t ggml_quantize_chunk_mt(
        enum ggml_type   type,
           const float * src,
                  void * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           const float * imatrix,
                   int   n_threads) {
    if (n_threads <= 1 || nrows <= 1) {
        return ggml_quantize_chunk(type, src, dst, start, nrows, n_per_row, imatrix);
    }

    const int     n_t   = std::min((int64_t) n_threads, nrows);
    const int64_t chunk = (nrows + n_t - 1) / n_t;

    std::vector<size_t>      results(n_t, 0);
    std::vector<std::thread> threads;
    threads.reserve(n_t - 1);

    auto worker = [&](int t) {
        const int64_t r0      = (int64_t) t * chunk;
        const int64_t r1      = std::min(r0 + chunk, nrows);
        const int64_t nrows_t = r1 - r0;
        results[t] = ggml_quantize_chunk(type, src, dst, start + r0 * n_per_row, nrows_t, n_per_row, imatrix);
    };

    for (int t = 1; t < n_t; ++t) {
        threads.emplace_back(worker, t);
    }
    worker(0);
    for (auto & th : threads) {
        th.join();
    }

    size_t total = 0;
    for (int t = 0; t < n_t; ++t) {
        total += results[t];
    }
    return total;
}
