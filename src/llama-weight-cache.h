// SPDX-License-Identifier: MIT
//

#pragma once

#if defined(LLAMA_WEIGHT_CACHE)

#include "llama-mmap.h"

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>

struct ggml_context;
struct ggml_tensor;
struct llama_file;
struct llama_model_loader;

struct llama_weight_cache {
    struct impl;

    llama_weight_cache(bool check_tensors, bool from_file_ptr);
    ~llama_weight_cache();

    void add_source(uint16_t idx, const std::string & path, const llama_file * file);

    ggml_backend_buffer_t load(
            const llama_model_loader & loader,
            ggml_context * ctx,
            ggml_backend_buffer_type_t buft,
            bool use_mlock,
            llama_mlocks * mlocks);

    void save(
            const llama_model_loader & loader,
            ggml_context * ctx,
            ggml_backend_buffer_type_t buft);

    bool contains(const ggml_tensor * tensor) const;

private:
    std::unique_ptr<impl> pimpl;
};

#endif
