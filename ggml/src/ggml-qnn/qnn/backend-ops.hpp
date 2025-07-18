#pragma once

#ifndef NDEBUG
#    include <atomic>
#endif

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "convert.hpp"
#include "ggml-backend.h"
#include "ggml-qnn.h"
#include "ggml.h"
#include "graph.hpp"
#include "qnn-lib.hpp"

namespace qnn {

typedef std::unordered_map<std::string, std::unique_ptr<qnn::qnn_graph>> qnn_graph_cache_t;

struct ggml_backend_qnn_device_context {
    // initialize in constructor
    backend_index_type device;
    size_t             threads;
    std::string        name;
    std::string        description;

    // initialize in qnn init
    qnn::qcom_socinfo                   socinfo = {};
    size_t                              max_tensor_size_in_bytes;
    std::shared_ptr<qnn::qnn_instance>  instance;
    std::shared_ptr<qnn::qnn_interface> qnn_interface;

    qnn::qnn_graph_cache_t                      qnn_graph_cache;
    std::shared_ptr<qnn::qnn_convert_context_t> convert_context = std::make_shared<qnn::qnn_convert_context_t>();

#ifndef NDEBUG
    std::atomic_uint32_t supported_op_count   = 0;
    std::atomic_uint32_t unsupported_op_count = 0;
#endif

    bool     enable_cpu_dequantize = false;
    uint64_t supported_types;
    uint64_t cpu_preprocess_types;

    explicit ggml_backend_qnn_device_context(backend_index_type device, size_t threads, const char * name,
                                             uint64_t supported_types) :
        device(device),
        threads(threads),
        name(name),
        supported_types(supported_types) {}
};

bool device_supports_op(ggml_backend_qnn_device_context * ctx, const ggml_tensor * op);
bool device_compute_graph(ggml_backend_qnn_device_context * ctx, ggml_cgraph * cgraph);

}  // namespace qnn
