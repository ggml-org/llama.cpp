#pragma once

#include <future>
#include <memory>
#include <thread>

#include "buffer.hpp"
#include "ggml-qnn.h"
#include "tensor.hpp"
#include "utils.hpp"

namespace qnn {

// see also: ggml_backend_blas_context
struct qnn_convert_context_t {
    int                                          n_threads = std::thread::hardware_concurrency();
    std::vector<std::shared_ptr<qnn_mem_buffer>> buffers;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<void>> tasks;
#endif
};

std::vector<qnn::qnn_buffer_ptr> convert(std::shared_ptr<qnn_convert_context_t> convert_context,
                                         const ggml_tensor_array_t & tensors, ggml_type target_data_type);

}  // namespace qnn
