#include "llama-lazy-reader.h"

#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>

llama_lazy_reader::llama_lazy_reader(const llama_file & source, size_t offs, enum ggml_type type,
                                     int64_t row_elems, int64_t n_rows, int n_readers) :
    path(source.name()),
    type(type),
    offs(offs),
    rsize(ggml_row_size(type, row_elems)),
    relems(row_elems),
    nrows(n_rows) {
    if (type != GGML_TYPE_F32 && ggml_get_type_traits(type)->to_float == nullptr) {
        throw std::runtime_error(format("%s cannot be read row by row: %s has no F32 conversion",
                path.c_str(), ggml_type_name(type)));
    }

    GGML_ASSERT(row_elems > 0 && n_rows > 0 && n_readers > 0);

    files.reserve(n_readers);
    for (int i = 0; i < n_readers; ++i) {
        files.emplace_back(std::make_unique<llama_file>(path.c_str(), "rb", /*use_direct_io =*/ false));
        if (!source.same_file(*files.back())) {
            throw std::runtime_error(format("model file changed while opening lazy reader: %s", path.c_str()));
        }
    }
}

llama_lazy_reader::~llama_lazy_reader() = default;

std::unique_ptr<llama_lazy_reader> llama_lazy_reader::clone(int n_readers) const {
    return std::make_unique<llama_lazy_reader>(*files[0], offs, type, relems, nrows, n_readers);
}

void llama_lazy_reader::read_range(const std::pair<int32_t, int32_t> * pairs, int64_t begin, int64_t end,
                                   size_t fi, uint8_t * dst) const {
    std::vector<uint8_t> bounce(rsize);

    for (int64_t i = begin; i < end; ) {
        int64_t j = i;
        while (j + 1 < end && pairs[j + 1].first == pairs[i].first) {
            ++j;
        }

        files[fi]->read_at(offs + (size_t) pairs[i].first * rsize, bounce.data(), rsize);

        uint8_t * first = dst + (size_t) pairs[i].second * rsize;
        memcpy(first, bounce.data(), rsize);

        for (int64_t k = i + 1; k <= j; ++k) {
            memcpy(dst + (size_t) pairs[k].second * rsize, first, rsize);
        }

        i = j + 1;
    }
}

void llama_lazy_reader::gather(const int32_t * rows, int64_t n, uint8_t * dst) const {
    std::vector<std::pair<int32_t, int32_t>> pairs;
    pairs.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        GGML_ASSERT(rows[i] >= 0 && (int64_t) rows[i] < nrows);
        pairs.emplace_back(rows[i], (int32_t) i);
    }

    std::sort(pairs.begin(), pairs.end());

    const int n_workers = (int) std::min<int64_t>(files.size(), std::max<int64_t>(1, n / 32));

    auto run_chunk = [&](int w, std::exception_ptr & err) {
        try {
            read_range(pairs.data(), n * w / n_workers, n * (w + 1) / n_workers, w, dst);
        } catch (...) {
            err = std::current_exception();
        }
    };

    std::vector<std::exception_ptr> errs(n_workers);
    std::vector<std::thread> workers;
    try {
        workers.reserve(n_workers - 1);
        for (int w = 1; w < n_workers; ++w) {
            workers.emplace_back([&run_chunk, &errs, w]() { run_chunk(w, errs[w]); });
        }
    } catch (...) {
        for (auto & t : workers) {
            t.join();
        }
        throw;
    }

    run_chunk(0, errs[0]);

    for (auto & t : workers) {
        t.join();
    }

    for (const auto & err : errs) {
        if (err) {
            std::rethrow_exception(err);
        }
    }
}
