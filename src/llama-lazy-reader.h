#pragma once

// Serves rows of a lazy tensor with explicit pread()s instead of demand paging.
// This works because the row indices of a whole ubatch are known host-side
// before the graph runs. Hands out F32 rows, like ggml_get_rows does.

#include "ggml.h"
#include "llama-impl.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

struct llama_lazy_reader {
#ifdef _WIN32
    // pread()/open() are unavailable on Windows; --lazy-mode on-direct falls
    // back to the lazy mmap reads there (see llama_model_base::load_lazy_reader)
    const int64_t head_dim = 0;

    void gather(const int32_t *, int64_t, float *) const {
        GGML_ABORT("lazy direct reads are not supported on this platform");
    }
#else
    llama_lazy_reader(int fd, size_t base, size_t row_size, int64_t n_rows, int n_threads,
                      enum ggml_type type, int64_t head_dim)
        : fd(fd), base(base), row_size(row_size), n_rows(n_rows), n_threads(n_threads),
          head_dim(head_dim), to_float(type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(type)->to_float) {
        // F32 rows have no dequantizer; they are staged as-is, like ggml_get_rows
        GGML_ASSERT((type == GGML_TYPE_F32 || to_float != nullptr) && head_dim > 0);
    }

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    ~llama_lazy_reader() {
        if (fd >= 0) {
            ::close(fd);
        }
    }

    const int      fd;
    const size_t   base;       // file offset of row 0
    const size_t   row_size;   // bytes per quantized row
    const int64_t  n_rows;
    const int      n_threads;  // in-flight read workers
    const int64_t  head_dim;   // F32 elements per staged row
    ggml_to_float_t to_float;  // same dequantizer the ggml_get_rows CPU kernel uses

    // fill dst with the n gathered rows, dequantized to F32:
    // dst[slot * head_dim, ...) = to_float(table[rows[slot]])
    // thread-safe; never lets an exception escape a worker thread
    void gather(const int32_t * rows, int64_t n, float * dst) const {
        std::vector<std::pair<int32_t, int32_t>> pairs; // (row, dst slot)
        pairs.reserve(n);
        for (int64_t i = 0; i < n; ++i) {
            GGML_ASSERT(rows[i] >= 0 && (int64_t) rows[i] < n_rows);
            pairs.emplace_back(rows[i], (int32_t) i);
        }

        std::sort(pairs.begin(), pairs.end()); // equal rows adjacent, file order

        // small gathers are not worth a thread per row
        const int n_workers = (int) std::min<int64_t>(n_threads, std::max<int64_t>(1, n / 32));

        // worker w reads rows pairs[n*w/n_workers, n*(w+1)/n_workers)
        auto run_chunk = [&](int w, std::exception_ptr & err) {
            try {
                run_range(pairs, n * w / n_workers, n * (w + 1) / n_workers, dst);
            } catch (...) {
                err = std::current_exception();
            }
        };

        // an exception leaving a joinable std::thread, or destroying one,
        // terminates the process; keep worker creation failure-safe
        std::vector<std::exception_ptr> errs(n_workers);
        std::vector<std::thread> workers;
        try {
            for (int w = 1; w < n_workers; ++w) {
                workers.emplace_back([&run_chunk, &errs, w]() {
                    run_chunk(w, errs[w]);
                });
            }
        } catch (...) {
            for (auto & t : workers) {
                t.join();
            }
            throw;
        }

        run_chunk(0, errs[0]); // this thread takes the first chunk
        for (auto & t : workers) {
            t.join();
        }

        for (const auto & err : errs) {
            if (err) {
                std::rethrow_exception(err);
            }
        }
    }

private:
    void run_range(const std::vector<std::pair<int32_t, int32_t>> & pairs,
                   int64_t begin, int64_t end, float * dst) const {
        std::vector<uint8_t> bounce(row_size);
        for (int64_t i = begin; i < end; ) {
            int64_t j = i;
            while (j + 1 < end && pairs[j + 1].first == pairs[i].first) {
                ++j; // dedup: one read serves the whole run
            }
            const size_t off = base + (size_t) pairs[i].first * row_size;
            for (size_t done = 0; done < row_size; ) {
                const ssize_t n_read = ::pread(fd, bounce.data() + done, row_size - done, off + done);
                if (n_read < 0 && errno == EINTR) {
                    continue; // interrupted by a signal without SA_RESTART
                }
                if (n_read <= 0) {
                    throw std::runtime_error(format("lazy direct read of %zu bytes at file offset %zu failed: %s",
                            row_size, off, n_read == 0 ? "unexpected EOF" : strerror(errno)));
                }
                done += n_read;
            }
            float * first = dst + (size_t) pairs[i].second * head_dim;
            if (to_float) {
                to_float(bounce.data(), first, head_dim);
            } else {
                memcpy(first, bounce.data(), (size_t) head_dim * sizeof(float));
            }
            for (int64_t k = i + 1; k <= j; ++k) {
                memcpy(dst + (size_t) pairs[k].second * head_dim, first, (size_t) head_dim * sizeof(float));
            }
            i = j + 1;
        }
    }
#endif
};
