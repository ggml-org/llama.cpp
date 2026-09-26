#include "llama-lazy-reader.h"

#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <utility>

void llama_lazy_reader_factory::add(const ggml_tensor * tensor, const llama_file & source, size_t offs) {
    if (tensor->type != GGML_TYPE_F32 && ggml_get_type_traits(tensor->type)->to_float == nullptr) {
        throw std::runtime_error(format("%s cannot be read row by row: %s has no F32 conversion",
                source.name().c_str(), ggml_type_name(tensor->type)));
    }

    GGML_ASSERT(tensor->ne[0] > 0 && tensor->ne[1] > 0);

    size_t file = 0;
    while (file < sources.size() && !source.same_file(*sources[file])) {
        ++file;
    }
    if (file == sources.size()) {
        const auto path = std::filesystem::absolute(std::filesystem::u8path(source.name())).u8string();
        auto handle = std::make_unique<llama_file>(path.c_str(), "rb", /*use_direct_io =*/ false);
        if (!source.same_file(*handle)) {
            throw std::runtime_error(format("model file changed while opening lazy reader: %s", source.name().c_str()));
        }
        sources.emplace_back(std::move(handle));
    }

    auto result = tensors.emplace(tensor, tensor_info { file, offs, ggml_row_size(tensor->type, tensor->ne[0]), tensor->ne[1] });
    GGML_ASSERT(result.second);
}

bool llama_lazy_reader_factory::has(const ggml_tensor * tensor) const {
    return tensors.find(tensor) != tensors.end();
}

std::unique_ptr<llama_lazy_reader> llama_lazy_reader_factory::create(int n_readers) const {
    return std::make_unique<llama_lazy_reader>(*this, n_readers);
}

llama_lazy_reader::llama_lazy_reader(const llama_lazy_reader_factory & factory, int n_readers) : factory(factory) {
    GGML_ASSERT(n_readers > 0);

    files.resize(factory.sources.size());
    for (size_t i = 0; i < factory.sources.size(); ++i) {
        auto & workers = files[i];
        const auto & source = *factory.sources[i];
        workers.reserve(n_readers);
        // read_at is not thread-safe, so each worker needs its own handle
        for (int w = 0; w < n_readers; ++w) {
            auto handle = std::make_unique<llama_file>(source.name().c_str(), "rb", /*use_direct_io =*/ false);
            if (!source.same_file(*handle)) {
                throw std::runtime_error(format("model file changed while opening lazy reader: %s", source.name().c_str()));
            }
            workers.emplace_back(std::move(handle));
        }
    }
}

llama_lazy_reader::~llama_lazy_reader() = default;

void llama_lazy_reader::read_range(const llama_lazy_reader_factory::tensor_info & info,
                                   const std::pair<int32_t, int32_t> * pairs, int64_t begin, int64_t end,
                                   size_t fi, uint8_t * dst) const {
    std::vector<uint8_t> bounce(info.rsize);

    for (int64_t i = begin; i < end; ) {
        int64_t j = i;
        while (j + 1 < end && pairs[j + 1].first == pairs[i].first) {
            ++j;
        }

        files[info.file][fi]->read_at(info.offs + (size_t) pairs[i].first * info.rsize, bounce.data(), info.rsize);

        uint8_t * first = dst + (size_t) pairs[i].second * info.rsize;
        memcpy(first, bounce.data(), info.rsize);

        for (int64_t k = i + 1; k <= j; ++k) {
            memcpy(dst + (size_t) pairs[k].second * info.rsize, first, info.rsize);
        }

        i = j + 1;
    }
}

void llama_lazy_reader::gather(const ggml_tensor * tensor, const int32_t * rows, int64_t n, uint8_t * dst) const {
    const auto it = factory.tensors.find(tensor);
    GGML_ASSERT(it != factory.tensors.end());
    const auto & info = it->second;

    std::vector<std::pair<int32_t, int32_t>> pairs;
    pairs.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        GGML_ASSERT(rows[i] >= 0 && (int64_t) rows[i] < info.nrows);
        pairs.emplace_back(rows[i], (int32_t) i);
    }

    std::sort(pairs.begin(), pairs.end());

    const int n_workers = (int) std::min<int64_t>(files[info.file].size(), std::max<int64_t>(1, n / 32));

    auto run_chunk = [&](int w, std::exception_ptr & err) {
        try {
            read_range(info, pairs.data(), n * w / n_workers, n * (w + 1) / n_workers, w, dst);
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
